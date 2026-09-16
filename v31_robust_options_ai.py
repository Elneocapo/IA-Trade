"""IA-Trade V31 - IA robusta + motor/visualizacion de CALLs para NVDA 1H.

PAPER RESEARCH ONLY. NO BROKER. NO LIVE ORDERS.

V31 parte de la arquitectura V27/V28 y mantiene el test ciego.
Las opciones son SYNTHETICAS (Black-Scholes + IV proxy), no quotes historicos reales.
La capa de salida CALL selecciona sus reglas SOLO con validacion.
"""
from __future__ import annotations
import math
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from matplotlib.patches import Rectangle
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor

# Silence only the known sklearn/joblib warning, including spawned workers.
os.environ.setdefault("PYTHONWARNINGS", "ignore:.*sklearn.utils.parallel.delayed.*:UserWarning")
warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*", category=UserWarning)
warnings.filterwarnings("ignore")

ASSET="NVDA"; PERIOD="2y"; INTERVAL="1h"
LOOKBACK=65; TRAIN_WINDOW=1200; RETRAIN_EVERY=24; MAX_ITER=140
INITIAL_CASH=500.0; COST=0.001; MAX_HOLD=6
THRESHOLDS=(0.0,0.00025,0.0005,0.00075,0.001,0.00125,0.0015,0.00175,0.002,0.0025,0.003)
CONSENSUS=(0.50,0.67,1.00)

# Synthetic option layer: diagnostic only.
OPTION_DTE=14; OPTION_DELTA=0.60; OPTION_IV_MULT=1.00; OPTION_SPREAD=0.01; OPTION_BUDGET=0.20
CONTRACT_MULTIPLIER=100.0; RISK_FREE=0.0; TRADING_DAYS_PER_YEAR=252.0; HOURS_PER_TRADING_DAY=6.5
OPTION_MIN_HOLD_CANDIDATES=(4,6,8,10)
OPTION_EXIT_CONFIRM_CANDIDATES=(1,2,3)
OPTION_MAX_HOLD=36
OPTION_EXIT_SIGNAL=0.0


def load_data():
    df=yf.download(ASSET,period=PERIOD,interval=INTERVAL,auto_adjust=True,progress=False,prepost=False)
    if df.empty: raise RuntimeError(f"No data returned for {ASSET}")
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df=df[["Open","High","Low","Close","Volume"]].dropna().copy()
    if getattr(df.index,"tz",None) is not None: df.index=df.index.tz_convert("America/New_York")
    return df.between_time("09:30","16:00")


def make_features(df):
    c=df["Close"].astype(float); o=df["Open"].astype(float); h=df["High"].astype(float); l=df["Low"].astype(float); v=df["Volume"].astype(float)
    r=np.log(c).diff(); f=pd.DataFrame(index=df.index)
    f["r1"]=r; f["body"]=(c-o)/(o+1e-9); f["range"]=(h-l)/(c+1e-9); f["loc"]=(c-l)/(h-l+1e-9); f["upper"]=(h-np.maximum(o,c))/(c+1e-9); f["lower"]=(np.minimum(o,c)-l)/(c+1e-9); f["volchg"]=np.log1p(v).diff()
    for n in (2,3,6,12,24,48,65):
        f[f"mom{n}"]=c.pct_change(n); f[f"vol{n}"]=r.rolling(n).std(); f[f"ema{n}"]=c/c.ewm(span=n,adjust=False).mean()-1; f[f"range{n}"]=f["range"].rolling(n).mean(); f[f"vr{n}"]=v/(v.rolling(n).mean()+1e-9)
    f["ema6_24"]=c.ewm(span=6,adjust=False).mean()/c.ewm(span=24,adjust=False).mean()-1; f["ema12_48"]=c.ewm(span=12,adjust=False).mean()/c.ewm(span=48,adjust=False).mean()-1; f["vol_ratio"]=f["vol6"]/(f["vol48"]+1e-9); f["mom_ratio"]=f["mom6"]/(f["vol12"]+1e-9)
    for lag in (1,2,3,4,6,8,12,16,24,32,48):
        f[f"lag_r{lag}"]=r.shift(lag); f[f"lag_body{lag}"]=f["body"].shift(lag); f[f"lag_loc{lag}"]=f["loc"].shift(lag)
    mins=(df.index.hour*60+df.index.minute)-570; prog=np.clip(mins/390.0,0,1); first30=(mins<30).astype(float); first60=(mins<60).astype(float); first90=(mins<90).astype(float); mid=((mins>=120)&(mins<270)).astype(float); last60=(mins>=330).astype(float); last30=(mins>=360).astype(float)
    f["session_progress"]=prog; f["session_sin"]=np.sin(2*np.pi*prog); f["session_cos"]=np.cos(2*np.pi*prog); f["first30"]=first30; f["first60"]=first60; f["first90"]=first90; f["midday"]=mid; f["last60"]=last60; f["last30"]=last30; f["dow_sin"]=np.sin(2*np.pi*df.index.dayofweek/5); f["dow_cos"]=np.cos(2*np.pi*df.index.dayofweek/5)
    f["trend_agreement"]=np.sign(f["ema6_24"])*np.sign(f["ema12_48"]); f["trend_strength"]=0.6*f["ema6_24"]+0.4*f["ema12_48"]; f["range6_vs_48"]=f["range6"]/(f["range48"]+1e-9); f["vol6_vs_24"]=f["vol6"]/(f["vol24"]+1e-9); f["bar_pressure"]=(f["body"]+f["loc"]-0.5)*f["range"]; f["volume_pressure"]=f["bar_pressure"]*f["vr6"]
    f["session_x_mom6"]=prog*f["mom6"]; f["session_x_volratio"]=prog*f["vol_ratio"]; f["session_x_range"]=prog*f["range6_vs_48"]; f["early_strength"]=(first60+0.5*first30)*f["bar_pressure"]; f["late_strength"]=(last60+last30)*f["bar_pressure"]
    day_key=pd.Index(df.index.normalize()); daily_last=c.groupby(day_key).last(); prev_daily_last=daily_last.shift(1); prev_close=pd.Series(day_key,index=df.index).map(prev_daily_last); f["gap_prev_close"]=o/(prev_close+1e-9)-1
    f["vol_regime"]=(f["vol6"]/(f["vol24"]+1e-9)).clip(0,5); f["trend_x_mom"]=f["trend_strength"]*f["mom6"]; f["pressure_x_volume"]=f["bar_pressure"]*f["vol6_vs_24"]; f["mom_accel"]=f["mom3"]-f["mom12"]/4.0; f["range_accel"]=f["range6_vs_48"]-1.0
    return f.replace([np.inf,-np.inf],np.nan)


def fit_set(X,y1,y3,y6,c1,c3,c6):
    a=dict(max_iter=MAX_ITER,learning_rate=0.04,max_leaf_nodes=13,min_samples_leaf=20,l2_regularization=3.5,loss="absolute_error",random_state=42)
    b=dict(max_iter=MAX_ITER,learning_rate=0.032,max_leaf_nodes=21,min_samples_leaf=28,l2_regularization=6.5,loss="absolute_error",random_state=77)
    return {
        "r1a":HistGradientBoostingRegressor(**a).fit(X,y1),"r3a":HistGradientBoostingRegressor(**a).fit(X,y3),"r6a":HistGradientBoostingRegressor(**a).fit(X,y6),
        "r1b":HistGradientBoostingRegressor(**b).fit(X,y1),"r3b":HistGradientBoostingRegressor(**b).fit(X,y3),"r6b":HistGradientBoostingRegressor(**b).fit(X,y6),
        "c1":ExtraTreesClassifier(n_estimators=260,max_depth=9,min_samples_leaf=10,max_features=0.60,class_weight="balanced",n_jobs=-1,random_state=59).fit(X,c1),
        "c3":ExtraTreesClassifier(n_estimators=260,max_depth=8,min_samples_leaf=12,max_features=0.65,class_weight="balanced",n_jobs=-1,random_state=83).fit(X,c3),
        "c6":ExtraTreesClassifier(n_estimators=220,max_depth=8,min_samples_leaf=14,max_features=0.60,class_weight="balanced",n_jobs=-1,random_state=97).fit(X,c6),
    }


def predict_panel(df,start_i,end_i):
    f=make_features(df); c=df["Close"].astype(float); ret1=c.shift(-1)/c-1; ret3=c.shift(-3)/c-1; ret6=c.shift(-6)/c-1; scale=f["vol6"].clip(lower=0.0005)
    y1=ret1/scale; y3=ret3/scale; y6=ret6/scale; c1=(ret1>0.0015).astype(int); c3=(ret3>0.0015).astype(int); c6=(ret6>0.0020).astype(int); X=f.to_numpy(float); arrs=[q.to_numpy(float) for q in (y1,y3,y6,c1,c3,c6)]
    valid=[i for i in range(LOOKBACK-1,len(df)-6) if start_i<=i<end_i and np.isfinite(X[i]).all() and all(np.isfinite(q[i]) for q in arrs[:3])]
    models=None; last_fit=-10**9; out=[]; t=time.time()
    for n,i in enumerate(valid,1):
        if models is None or i-last_fit>=RETRAIN_EVERY:
            idx=np.arange(max(LOOKBACK-1,i-TRAIN_WINDOW),i); good=np.isfinite(X[idx]).all(axis=1)
            for q in arrs: good &= np.isfinite(q[idx])
            idx=idx[good]
            if len(idx)>=450: models=fit_set(X[idx],y1.to_numpy()[idx],y3.to_numpy()[idx],y6.to_numpy()[idx],c1.to_numpy()[idx].astype(int),c3.to_numpy()[idx].astype(int),c6.to_numpy()[idx].astype(int)); last_fit=i
        if models is None: continue
        row=X[i].reshape(1,-1); p1=.5*(models["r1a"].predict(row)[0]+models["r1b"].predict(row)[0]); p3=.5*(models["r3a"].predict(row)[0]+models["r3b"].predict(row)[0]); p6=.5*(models["r6a"].predict(row)[0]+models["r6b"].predict(row)[0]); prob1=float(models["c1"].predict_proba(row)[0,1]); prob3=float(models["c3"].predict_proba(row)[0,1]); prob6=float(models["c6"].predict_proba(row)[0,1]); prob=.50*prob1+.32*prob3+.18*prob6; s=max(float(scale.iloc[i]),.0005); r1=float(p1*s); r3=float(p3*s/3); r6=float(p6*s/6); base=.55*r1+.29*r3+.16*r6; conf=np.clip((prob-.5)*2,-1,1); trend=float(f["trend_agreement"].iloc[i]); boost=1.10 if trend>0 else (.90 if trend<0 else 1.0); dispersion=float(np.std([r1,r3,r6])); disagreement=np.clip(dispersion/(abs(base)+s*.15+1e-9),0,2); quality=np.clip(1-.20*disagreement,.55,1.0); vol_reg=float(f["vol_regime"].iloc[i]); vol_penalty=1.0 if .65<=vol_reg<=1.90 else .88; signal=base*(.68+.84*max(conf,0))*boost*quality*vol_penalty; consensus=float(np.mean([r1>0,r3>0,r6>0])); out.append((df.index[i],signal,prob,consensus,r1,r3,r6,quality,vol_reg,disagreement))
        if n==1 or n%200==0 or n==len(valid):
            elapsed=time.time()-t; rate=n/max(elapsed,1e-9); eta=(len(valid)-n)/max(rate,1e-9); print(f"prediction {n}/{len(valid)} | {rate:.1f} bars/s | ETA ~{eta/60:.1f} min",flush=True)
    return pd.DataFrame(out,columns=["date","signal","prob_consensus","consensus","pred1","pred3","pred6","signal_quality","vol_regime","model_disagreement"]).set_index("date")


def backtest(df,panel,start,end,threshold,minimum_consensus):
    pos={ts:i for i,ts in enumerate(df.index)}; cash=INITIAL_CASH; shares=0.; entry=None; outlay=None; trades=[]; curve=[]
    for ts,row in panel[(panel.index>=start)&(panel.index<end)].iterrows():
        i=pos.get(ts)
        if i is None or i+1>=len(df): continue
        ex=i+1; px=float(df["Open"].iloc[ex]);
        if not np.isfinite(px) or px<=0: continue
        want=float(row["signal"])>threshold and float(row["consensus"])>=minimum_consensus and float(row["prob_consensus"])>.50 and float(row["signal_quality"])>=.60; equity=cash+shares*px; target=equity if want else 0.; current=shares*px
        if shares>0 and entry is not None and ex-entry>=MAX_HOLD: target=0.
        if target<current*.98 and shares>0:
            sold=min(shares,(current-target)/px); cash+=sold*px*(1-COST); shares-=sold
            if shares<=1e-12: shares=0.; trades.append(cash/outlay-1) if outlay else None; entry=None; outlay=None
        elif target>current*1.02:
            desired=min(target-current,cash/(1+COST))
            if desired>max(.01,equity*.01):
                before=cash+shares*px; shares+=desired/(px*(1+COST)); cash-=desired
                if entry is None: entry=ex; outlay=before
        curve.append(cash+shares*px)
    if len(curve)<2:return {"final":INITIAL_CASH,"ret":0.,"dd":0.,"sharpe":0.,"trades":0,"wr":0.}
    a=np.asarray(curve,float); peak=np.maximum.accumulate(a); dd=float(np.min(a/np.maximum(peak,1e-9)-1)); rr=a[1:]/np.maximum(a[:-1],1e-9)-1; sh=float(np.mean(rr)/(np.std(rr)+1e-12)*math.sqrt(252*6.5)) if len(rr)>20 else 0.; final=float(a[-1]); wr=float(np.mean(np.asarray(trades)>0)) if trades else 0.; return {"final":final,"ret":final/INITIAL_CASH-1.,"dd":dd,"sharpe":sh,"trades":len(trades),"wr":wr}


def score(r): return r["ret"]-.38*abs(min(r["dd"],0))+.02*max(r["sharpe"],0)-.002*max(0,20-r["trades"])
def line(label,r): return f"{label}: inicio=€{INITIAL_CASH:.2f} | final=€{r['final']:.2f} | ganado={r['final']-INITIAL_CASH:+.2f}€ | retorno={r['ret']:+.2%} | trades={r['trades']} | DD={r['dd']:.2%} | Sharpe={r['sharpe']:.2f} | WR={r['wr']:.1%}"


def norm_cdf(x): return .5*(1+math.erf(float(x)/math.sqrt(2)))
def bs_call(S,K,T,sigma,r=RISK_FREE):
    if not np.isfinite(S) or S<=0 or K<=0:return np.nan
    if T<=0:return max(S-K,0.)
    sigma=max(float(sigma),1e-6); st=math.sqrt(T); d1=(math.log(S/K)+(r+.5*sigma*sigma)*T)/(sigma*st); d2=d1-sigma*st; return S*norm_cdf(d1)-K*math.exp(-r*T)*norm_cdf(d2)
def bs_delta(S,K,T,sigma,r=RISK_FREE):
    if T<=0:return 1. if S>K else 0.
    sigma=max(float(sigma),1e-6); st=math.sqrt(T); d1=(math.log(S/K)+(r+.5*sigma*sigma)*T)/(sigma*st); return norm_cdf(d1)
def strike_for_delta(S,T,sigma,target):
    lo=max(S*.20,.01); hi=S*2.50; target=float(np.clip(target,.05,.95))
    for _ in range(70):
        mid=(lo+hi)/2
        if bs_delta(S,mid,T,sigma)>target:lo=mid
        else:hi=mid
    return (lo+hi)/2
def iv_proxy(df,end_i):
    ret=np.log(df["Close"].astype(float)).diff(); w=ret.iloc[max(0,end_i-48):end_i+1].dropna(); base=float(w.std())*math.sqrt(TRADING_DAYS_PER_YEAR*HOURS_PER_TRADING_DAY) if len(w)>=12 else .50; return float(np.clip(base*OPTION_IV_MULT,.10,2.50))
def option_mark(S,K,expiry,now,sigma):
    T=max((expiry-now).total_seconds()/86400,0)/365; return max(float(bs_call(S,K,T,sigma)),0),float(bs_delta(S,K,T,sigma)),T


def option_backtest(df,panel,start,end,threshold,minimum_consensus,min_hold,confirm_bars):
    pos={ts:i for i,ts in enumerate(df.index)}; cash=INITIAL_CASH; contracts=0.; strike=np.nan; expiry=None; entry_i=None; entry_cost=0.; pending_bear=0; trades=[]; durations=[]; curve=[]; trace=[]; trade_no=0
    sigs=panel[(panel.index>=start)&(panel.index<end)]
    for signal_ts,row in sigs.iterrows():
        i=pos.get(signal_ts)
        if i is None or i+1>=len(df): continue
        ex=i+1; ts=df.index[ex]; S=float(df["Open"].iloc[ex]);
        if not np.isfinite(S) or S<=0: continue
        sigma=iv_proxy(df,ex); bullish=float(row["signal"])>threshold and float(row["consensus"])>=minimum_consensus and float(row["prob_consensus"])>.50 and float(row["signal_quality"])>=.60; event=""
        if contracts>0 and expiry is not None:
            mid,delta,T=option_mark(S,strike,expiry,ts,sigma); bid=mid*max(1-OPTION_SPREAD/2,.05); held=ex-entry_i; forced=held>=OPTION_MAX_HOLD or T<=0
            pending_bear = pending_bear+1 if (not bullish and float(row["signal"])<=OPTION_EXIT_SIGNAL) else 0
            confirmed_exit=held>=min_hold and pending_bear>=confirm_bars
            if confirmed_exit or forced:
                proceeds=contracts*bid*CONTRACT_MULTIPLIER*(1-COST); cash+=proceeds; tr=proceeds/max(entry_cost,1e-9)-1; trades.append(tr); durations.append(held); event=f"SELL CALL #{trade_no}" + (" [FORCED]" if forced and not confirmed_exit else ""); contracts=0.; strike=np.nan; expiry=None; entry_i=None; entry_cost=0.; pending_bear=0
        if contracts<=0 and bullish:
            expiry=ts.normalize()+pd.Timedelta(days=OPTION_DTE)+pd.Timedelta(hours=16); T=max((expiry-ts).total_seconds()/86400,1)/365; sigma=iv_proxy(df,ex); K=strike_for_delta(S,T,sigma,OPTION_DELTA); mid=bs_call(S,K,T,sigma); ask=mid*(1+OPTION_SPREAD/2); budget=cash*OPTION_BUDGET; contract_cost=ask*CONTRACT_MULTIPLIER*(1+COST); qty=budget/contract_cost if contract_cost>0 else 0
            if qty>0 and np.isfinite(qty):
                total=qty*contract_cost; cash-=total; contracts=qty; entry_i=ex; entry_cost=total; pending_bear=0; trade_no+=1; event=f"BUY CALL #{trade_no}"
        opt_mid=np.nan; delta=np.nan; T=np.nan; value=0.
        if contracts>0 and expiry is not None:
            opt_mid,delta,T=option_mark(float(df["Close"].iloc[ex]),strike,expiry,ts,iv_proxy(df,ex)); bid=opt_mid*max(1-OPTION_SPREAD/2,.05); value=contracts*bid*CONTRACT_MULTIPLIER
        equity=cash+value; trace.append((ts,float(df["Open"].iloc[ex]),float(df["High"].iloc[ex]),float(df["Low"].iloc[ex]),float(df["Close"].iloc[ex]),signal_ts,float(row["signal"]),float(row["prob_consensus"]),strike,expiry,contracts,opt_mid,delta,T,cash,value,equity,event))
        curve.append(equity)
    if contracts>0 and expiry is not None:
        last_candidates=df[(df.index>=start)&(df.index<end)]
        if not last_candidates.empty:
            last_ts=last_candidates.index[-1]; li=pos[last_ts]; S=float(df["Close"].iloc[li]); marked=option_mark(S,strike,expiry,last_ts,iv_proxy(df,li)); bid=marked[0]*max(1-OPTION_SPREAD/2,.05); proceeds=contracts*bid*CONTRACT_MULTIPLIER*(1-COST); cash+=proceeds; trades.append(proceeds/max(entry_cost,1e-9)-1); durations.append(max(li-entry_i,0)); contracts=0.; curve.append(cash)
    if len(curve)<2:return {"final":INITIAL_CASH,"ret":0.,"dd":0.,"sharpe":0.,"trades":0,"wr":0.,"avg_hold":0.,"trace":pd.DataFrame()}
    a=np.asarray(curve,float); peak=np.maximum.accumulate(a); dd=float(np.min(a/np.maximum(peak,1e-9)-1)); rr=a[1:]/np.maximum(a[:-1],1e-9)-1; sh=float(np.mean(rr)/(np.std(rr)+1e-12)*math.sqrt(252*6.5)) if len(rr)>20 else 0.; final=float(a[-1]); wr=float(np.mean(np.asarray(trades)>0)) if trades else 0.; avg=float(np.mean(durations)) if durations else 0.
    return {"final":final,"ret":final/INITIAL_CASH-1.,"dd":dd,"sharpe":sh,"trades":len(trades),"wr":wr,"avg_hold":avg,"trace":pd.DataFrame(trace,columns=["ts","open","high","low","close","signal_ts","signal","prob","strike","expiry","contracts","option_mid","delta","T","cash","option_value","equity","event"]).set_index("ts")}


def option_score(r): return r["ret"]-.60*abs(min(r["dd"],0))+.02*max(r["sharpe"],0)-.001*max(0,5-r["trades"])


def candlestick_axes(ax,x):
    if x.empty:return
    xs=mdates.date2num(pd.to_datetime(x.index).to_pydatetime()); step=float(np.median(np.diff(xs))) if len(xs)>1 else 1/24; width=step*.62
    for d,o,h,l,c in zip(xs,x["open"],x["high"],x["low"],x["close"]):
        up=c>=o; ax.vlines(d,l,h,linewidth=.9); bottom=min(o,c); height=max(abs(c-o),1e-8); ax.add_patch(Rectangle((d-width/2,bottom),width,height,fill=True,alpha=.65))
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); ax.grid(alpha=.2)


def plot_candles_and_calls(df,trace,title):
    x=trace.copy()
    fig,ax=plt.subplots(figsize=(15,8)); candlestick_axes(ax,x); ax.set_title(title); ax.set_ylabel("NVDA price")
    buys=x["event"].astype(str).str.startswith("BUY CALL"); sells=x["event"].astype(str).str.startswith("SELL CALL")
    ax.scatter(x.index[buys],x.loc[buys,"low"]*.995,marker="^",s=65,label="CALL entry"); ax.scatter(x.index[sells],x.loc[sells,"high"]*1.005,marker="v",s=65,label="CALL exit")
    active=x["strike"].notna(); ax.plot(x.index.where(active),x["strike"],linewidth=1.1,label="CALL strike")
    ax.legend(loc="upper left"); fig.tight_layout(); plt.show(block=False)


def plot_equity(trace):
    x=trace.copy(); fig,ax=plt.subplots(figsize=(15,6)); ax.plot(x.index,x["cash"],label="Cash"); ax.plot(x.index,x["option_value"],label="CALL value"); ax.plot(x.index,x["equity"],linewidth=2,label="Equity total"); ax.axhline(INITIAL_CASH,linestyle="--",label="Inicio €500"); ax.set_title("V31 — Equity de la estrategia de CALL"); ax.set_ylabel("EUR (€)"); ax.set_xlabel("Fecha"); ax.grid(alpha=.2); ax.legend(loc="upper left"); fig.tight_layout(); plt.show(block=True)


def main():
    t0=time.time(); df=load_data(); n=len(df); cut1=int(n*.60); cut2=int(n*.80); test_start=cut2+2
    print("=== V31 ROBUST AI | NVDA 1H | PAPER OPTIONS RESEARCH ==="); print(f"Capital inicial: €{INITIAL_CASH:.2f}"); print("Modo opciones: SYNTHETIC Black-Scholes + IV proxy (no quotes historicos reales)"); print("Generando predicciones...",flush=True)
    panel=predict_panel(df,cut1,n-1)
    if panel.empty:raise RuntimeError("No predictions generated")
    val=panel.index[panel.index<df.index[cut2]]; test=panel.index[panel.index>=df.index[test_start]]; vs,ve=val[0],val[-1]; ts,te=test[0],test[-1]; print(f"Validacion: {vs} -> {ve}"); print(f"Test ciego: {ts} -> {te}")
    best=None; print("\n=== SWEEP VALIDACION ONLY ===")
    for th in THRESHOLDS:
        for con in CONSENSUS:
            r=backtest(df,panel,vs,ve,th,con); r.update(threshold=th,consensus=con,score=score(r)); print(f"threshold={th:+.3%} | consensus>={con:.2f} | ret={r['ret']:+.2%} | trades={r['trades']} | DD={r['dd']:.2%} | Sharpe={r['sharpe']:.2f} | WR={r['wr']:.1%} | score={r['score']:+.4f}"); best=r if best is None or r["score"]>best["score"] else best
    print("\n=== ELEGIDO | VALIDACION ONLY ==="); print(f"threshold={best['threshold']:+.3%} | min_consensus={best['consensus']:.2f}"); print(line("VALIDACION",best))
    test_r=backtest(df,panel,ts,te,best["threshold"],best["consensus"]); p=df[(df.index>=ts)&(df.index<te)]["Open"].astype(float); bh=float(p.iloc[-1]/p.iloc[0]-1) if len(p)>1 else 0.; bh_final=INITIAL_CASH*(1+bh); print("\n=== TEST CIEGO ==="); print(line("TEST IA",test_r)); print(f"TEST B&H: inicio=€{INITIAL_CASH:.2f} | final=€{bh_final:.2f} | ganado={bh_final-INITIAL_CASH:+.2f}€ | retorno={bh:+.2%}"); print(f"Diferencia IA vs B&H: {(test_r['ret']-bh):+.2%}")
    print("\n=== OPCIONES | SALIDA SELECCIONADA SOLO CON VALIDACION ==="); opt_best=None
    for mh in OPTION_MIN_HOLD_CANDIDATES:
        for cb in OPTION_EXIT_CONFIRM_CANDIDATES:
            r=option_backtest(df,panel,vs,ve,best["threshold"],best["consensus"],mh,cb); r.update(min_hold=mh,confirm_bars=cb,score=option_score(r)); print(f"min_hold={mh} | confirmacion={cb} | ret={r['ret']:+.2%} | trades={r['trades']} | avg_hold={r['avg_hold']:.1f} barras | DD={r['dd']:.2%} | Sharpe={r['sharpe']:.2f} | WR={r['wr']:.1%} | score={r['score']:+.4f}"); opt_best=r if opt_best is None or r["score"]>opt_best["score"] else opt_best
    opt_test=option_backtest(df,panel,ts,te,best["threshold"],best["consensus"],opt_best["min_hold"],opt_best["confirm_bars"])
    print(f"\nCALL config: DTE={OPTION_DTE} | delta={OPTION_DELTA:.2f} | IVx={OPTION_IV_MULT:.2f} | spread={OPTION_SPREAD:.1%} | budget={OPTION_BUDGET:.0%} | min_hold={opt_best['min_hold']} | confirmacion={opt_best['confirm_bars']} | max_hold={OPTION_MAX_HOLD}")
    print(f"OPCIONES VALIDACION: inicio=€{INITIAL_CASH:.2f} | final=€{opt_best['final']:.2f} | ganado={opt_best['final']-INITIAL_CASH:+.2f}€ | retorno={opt_best['ret']:+.2%} | trades={opt_best['trades']} | avg_hold={opt_best['avg_hold']:.1f} | DD={opt_best['dd']:.2%} | Sharpe={opt_best['sharpe']:.2f} | WR={opt_best['wr']:.1%}")
    print(f"OPCIONES TEST CIEGO: inicio=€{INITIAL_CASH:.2f} | final=€{opt_test['final']:.2f} | ganado={opt_test['final']-INITIAL_CASH:+.2f}€ | retorno={opt_test['ret']:+.2%} | trades={opt_test['trades']} | avg_hold={opt_test['avg_hold']:.1f} | DD={opt_test['dd']:.2%} | Sharpe={opt_test['sharpe']:.2f} | WR={opt_test['wr']:.1%}")
    print("Limitacion critica: las primas son sinteticas; no equivalen a fills historicos reales de opciones.")
    panel.to_csv("v31_predictions.csv"); opt_test["trace"].to_csv("v31_options_trace.csv")
    print("Archivos: v31_predictions.csv | v31_options_trace.csv")
    plot_candles_and_calls(df,opt_test["trace"],"V31 — NVDA 1H + operaciones CALL (test ciego)")
    plot_equity(opt_test["trace"])
    print(f"runtime={(time.time()-t0)/60:.1f} min")

if __name__=="__main__":main()
