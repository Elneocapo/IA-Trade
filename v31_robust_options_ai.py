"""IA-Trade V31 - IA robusta + visualizacion de opciones para NVDA 1H.

PAPER RESEARCH ONLY. NO BROKER. NO LIVE ORDERS.
V30 permanece intacta; V31 conserva su output y añade:
- ensemble extra con clasificador de 6 barras;
- penalizacion por desacuerdo entre modelos/regimen de volatilidad;
- filtro de calidad de señal aprendido solo con validation;
- grafico de la estructura de opciones sinteticas: NVDA, CALL mid, equity,
  entradas/salidas y strike/expiry.

IMPORTANTE: las opciones son sinteticas (Black-Scholes + IV proxy), no una
reconstruccion historica de quotes reales. Sirve para investigacion/sensibilidad.
"""
from __future__ import annotations
import os
# Silence the specific sklearn/joblib warning in the parent AND child workers.
os.environ.setdefault("PYTHONWARNINGS", "ignore:.*sklearn.utils.parallel.delayed.*:UserWarning")
import math,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*", category=UserWarning)
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor
warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*", category=UserWarning)
warnings.filterwarnings("ignore")

ASSET="NVDA"; PERIOD="2y"; INTERVAL="1h"; LOOKBACK=65; TRAIN_WINDOW=1200; RETRAIN_EVERY=24; MAX_ITER=140
INITIAL_CASH=500.0; COST=0.001; MAX_HOLD=6
THRESHOLDS=(0.0,0.00025,0.0005,0.00075,0.001,0.00125,0.0015,0.00175,0.002,0.0025,0.003)
CONSENSUS=(0.50,0.67,1.00)

OPTION_DTE=14; OPTION_DELTA=0.60; OPTION_IV_MULT=1.00; OPTION_SPREAD=0.01; OPTION_BUDGET=0.20
CONTRACT_MULTIPLIER=100.0; RISK_FREE=0.0; TRADING_DAYS_PER_YEAR=252.0; HOURS_PER_TRADING_DAY=6.5

def load_data():
    df=yf.download(ASSET,period=PERIOD,interval=INTERVAL,auto_adjust=True,progress=False,prepost=False)
    if df.empty: raise RuntimeError("No data returned")
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df=df[["Open","High","Low","Close","Volume"]].dropna().copy()
    if getattr(df.index,"tz",None) is not None: df.index=df.index.tz_convert("America/New_York")
    return df.between_time("09:30","16:00")

def make_features(df):
    c=df["Close"].astype(float); o=df["Open"].astype(float); h=df["High"].astype(float); l=df["Low"].astype(float); v=df["Volume"].astype(float); r=np.log(c).diff(); f=pd.DataFrame(index=df.index)
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
    return {"r1a":HistGradientBoostingRegressor(**a).fit(X,y1),"r3a":HistGradientBoostingRegressor(**a).fit(X,y3),"r6a":HistGradientBoostingRegressor(**a).fit(X,y6),"r1b":HistGradientBoostingRegressor(**b).fit(X,y1),"r3b":HistGradientBoostingRegressor(**b).fit(X,y3),"r6b":HistGradientBoostingRegressor(**b).fit(X,y6),"c1":ExtraTreesClassifier(n_estimators=260,max_depth=9,min_samples_leaf=10,max_features=0.60,class_weight="balanced",n_jobs=-1,random_state=59).fit(X,c1),"c3":ExtraTreesClassifier(n_estimators=260,max_depth=8,min_samples_leaf=12,max_features=0.65,class_weight="balanced",n_jobs=-1,random_state=83).fit(X,c3),"c6":ExtraTreesClassifier(n_estimators=220,max_depth=8,min_samples_leaf=14,max_features=0.60,class_weight="balanced",n_jobs=-1,random_state=97).fit(X,c6)}

def predict_panel(df,start_i,end_i):
    f=make_features(df); c=df["Close"].astype(float); ret1=c.shift(-1)/c-1; ret3=c.shift(-3)/c-1; ret6=c.shift(-6)/c-1; scale=f["vol6"].clip(lower=0.0005); y1=ret1/scale; y3=ret3/scale; y6=ret6/scale; c1=(ret1>0.0015).astype(int); c3=(ret3>0.0015).astype(int); c6=(ret6>0.0020).astype(int); X=f.to_numpy(float); arrs=[q.to_numpy(float) for q in (y1,y3,y6,c1,c3,c6)]; valid=[i for i in range(LOOKBACK-1,len(df)-6) if start_i<=i<end_i and np.isfinite(X[i]).all() and all(np.isfinite(q[i]) for q in arrs[:3])]; models=None; last_fit=-10**9; out=[]; t=time.time()
    for n,i in enumerate(valid,1):
        if models is None or i-last_fit>=RETRAIN_EVERY:
            idx=np.arange(max(LOOKBACK-1,i-TRAIN_WINDOW),i); good=np.isfinite(X[idx]).all(axis=1)
            for q in arrs: good &= np.isfinite(q[idx])
            idx=idx[good]
            if len(idx)>=450: models=fit_set(X[idx],y1.to_numpy()[idx],y3.to_numpy()[idx],y6.to_numpy()[idx],c1.to_numpy()[idx].astype(int),c3.to_numpy()[idx].astype(int),c6.to_numpy()[idx].astype(int)); last_fit=i
        if models is None: continue
        row=X[i].reshape(1,-1); p1=0.5*(models["r1a"].predict(row)[0]+models["r1b"].predict(row)[0]); p3=0.5*(models["r3a"].predict(row)[0]+models["r3b"].predict(row)[0]); p6=0.5*(models["r6a"].predict(row)[0]+models["r6b"].predict(row)[0]); prob1=float(models["c1"].predict_proba(row)[0,1]); prob3=float(models["c3"].predict_proba(row)[0,1]); prob6=float(models["c6"].predict_proba(row)[0,1]); prob=0.50*prob1+0.32*prob3+0.18*prob6; s=max(float(scale.iloc[i]),0.0005); r1=float(p1*s); r3=float(p3*s/3); r6=float(p6*s/6); base=0.55*r1+0.29*r3+0.16*r6; conf=np.clip((prob-0.5)*2,-1,1); trend=float(f["trend_agreement"].iloc[i]); boost=1.10 if trend>0 else (0.90 if trend<0 else 1.0); dispersion=float(np.std([r1,r3,r6])); disagreement=np.clip(dispersion/(abs(base)+s*0.15+1e-9),0,2); quality=np.clip(1.0-0.20*disagreement,0.55,1.0); vol_reg=float(f["vol_regime"].iloc[i]); vol_penalty=1.0 if 0.65<=vol_reg<=1.90 else 0.88; signal=base*(0.68+0.84*max(conf,0))*boost*quality*vol_penalty; consensus=float(np.mean([r1>0,r3>0,r6>0])); out.append((df.index[i],signal,prob,consensus,r1,r3,r6,quality,vol_reg,disagreement))
        if n==1 or n%200==0 or n==len(valid): elapsed=time.time()-t; rate=n/max(elapsed,1e-9); eta=(len(valid)-n)/max(rate,1e-9); print(f"prediction {n}/{len(valid)} | {rate:.1f} bars/s | ETA ~{eta/60:.1f} min",flush=True)
    return pd.DataFrame(out,columns=["date","signal","prob_consensus","consensus","pred1","pred3","pred6","signal_quality","vol_regime","model_disagreement"]).set_index("date")

def backtest(df,panel,start,end,threshold,minimum_consensus):
    pos={ts:i for i,ts in enumerate(df.index)}; cash=INITIAL_CASH; shares=0.; entry=None; outlay=None; trades=[]; curve=[]
    for ts,row in panel[(panel.index>=start)&(panel.index<end)].iterrows():
        i=pos.get(ts)
        if i is None or i+1>=len(df): continue
        ex=i+1; px=float(df["Open"].iloc[ex])
        if not np.isfinite(px) or px<=0: continue
        want=float(row["signal"])>threshold and float(row["consensus"])>=minimum_consensus and float(row["prob_consensus"])>0.50 and float(row["signal_quality"])>=0.60; equity=cash+shares*px; target=equity if want else 0.; current=shares*px
        if shares>0 and entry is not None and ex-entry>=MAX_HOLD: target=0.
        if target<current*0.98 and shares>0:
            sold=min(shares,(current-target)/px); cash+=sold*px*(1-COST); shares-=sold
            if shares<=1e-12: shares=0.; trades.append(cash/outlay-1.0) if outlay else None; entry=None; outlay=None
        elif target>current*1.02:
            desired=min(target-current,cash/(1+COST))
            if desired>max(0.01,equity*0.01):
                before=cash+shares*px; shares+=desired/(px*(1+COST)); cash-=desired
                if entry is None: entry=ex; outlay=before
        curve.append(cash+shares*px)
    if len(curve)<2: return {"final":INITIAL_CASH,"ret":0.,"dd":0.,"sharpe":0.,"trades":0,"wr":0.}
    a=np.asarray(curve,float); peak=np.maximum.accumulate(a); dd=float(np.min(a/np.maximum(peak,1e-9)-1)); rr=a[1:]/np.maximum(a[:-1],1e-9)-1; sharpe=float(np.mean(rr)/(np.std(rr)+1e-12)*math.sqrt(252*6.5)) if len(rr)>20 else 0.; final=float(a[-1]); wr=float(np.mean(np.asarray(trades)>0)) if trades else 0.; return {"final":final,"ret":final/INITIAL_CASH-1.,"dd":dd,"sharpe":sharpe,"trades":len(trades),"wr":wr}

def score(r): return r["ret"]-0.38*abs(min(r["dd"],0))+0.02*max(r["sharpe"],0)-0.002*max(0,20-r["trades"])
def line(label,r): return f"{label}: inicio=€{INITIAL_CASH:.2f} | final=€{r['final']:.2f} | ganado={r['final']-INITIAL_CASH:+.2f}€ | retorno={r['ret']:+.2%} | trades={r['trades']} | DD={r['dd']:.2%} | Sharpe={r['sharpe']:.2f} | WR={r['wr']:.1%}"

def norm_cdf(x): return 0.5*(1.0+math.erf(float(x)/math.sqrt(2.0)))
def bs_call(S,K,T,sigma,r=RISK_FREE):
    if not np.isfinite(S) or S<=0 or K<=0: return np.nan
    if T<=0: return max(S-K,0.0)
    sigma=max(float(sigma),1e-6); st=math.sqrt(T); d1=(math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*st); d2=d1-sigma*st; return S*norm_cdf(d1)-K*math.exp(-r*T)*norm_cdf(d2)
def bs_delta(S,K,T,sigma,r=RISK_FREE):
    if T<=0: return 1.0 if S>K else 0.0
    sigma=max(float(sigma),1e-6); st=math.sqrt(T); d1=(math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*st); return norm_cdf(d1)
def strike_for_delta(S,T,sigma,target):
    lo=max(S*0.2,0.01); hi=S*2.5
    for _ in range(70):
        mid=(lo+hi)/2.0
        if bs_delta(S,mid,T,sigma)>target: lo=mid
        else: hi=mid
    return (lo+hi)/2.0
def iv_proxy(df,end_i):
    ret=np.log(df["Close"].astype(float)).diff(); w=ret.iloc[max(0,end_i-48):end_i+1].dropna(); base=(float(w.std())*math.sqrt(TRADING_DAYS_PER_YEAR*HOURS_PER_TRADING_DAY)) if len(w)>=12 else 0.50; return float(np.clip(base*OPTION_IV_MULT,0.10,2.50))
def option_mark(S,K,expiry,now,sigma):
    T=max((expiry-now).total_seconds()/86400.0,0.0)/365.0; mid=max(float(bs_call(S,K,T,sigma)),0.0); return mid,bs_delta(S,K,T,sigma),T

def option_trace(df,panel,start,end,threshold,minimum_consensus):
    positions={ts:i for i,ts in enumerate(df.index)}; rows=[]; cash=INITIAL_CASH; contracts=0.; strike=np.nan; expiry=None; entry_i=None; trade_no=0
    sigs=panel[(panel.index>=start)&(panel.index<end)]
    for signal_ts,row in sigs.iterrows():
        i=positions.get(signal_ts)
        if i is None or i+1>=len(df): continue
        ex=i+1; ts=df.index[ex]; S=float(df["Open"].iloc[ex])
        if not np.isfinite(S) or S<=0: continue
        sigma=iv_proxy(df,ex); bullish=float(row["signal"])>threshold and float(row["consensus"])>=minimum_consensus and float(row["prob_consensus"])>0.50 and float(row["signal_quality"])>=0.60; event=""
        if contracts>0 and expiry is not None:
            mid,delta,T=option_mark(S,strike,expiry,ts,sigma); bid=mid*max(1-OPTION_SPREAD/2,0.05); forced=ex-entry_i>=MAX_HOLD or T<=0
            if (not bullish) or forced:
                proceeds=contracts*bid*CONTRACT_MULTIPLIER*(1-COST); cash+=proceeds; event=f"SELL CALL #{trade_no}"; contracts=0.; strike=np.nan; expiry=None; entry_i=None
        if contracts<=0 and bullish:
            expiry=ts.normalize()+pd.Timedelta(days=OPTION_DTE)+pd.Timedelta(hours=16); T=max((expiry-ts).total_seconds()/86400.0,1.0)/365.0; sigma=iv_proxy(df,ex); K=strike_for_delta(S,T,sigma,OPTION_DELTA); mid=bs_call(S,K,T,sigma); ask=mid*(1+OPTION_SPREAD/2); budget=cash*OPTION_BUDGET; contract_cost=ask*CONTRACT_MULTIPLIER*(1+COST); qty=budget/contract_cost if contract_cost>0 else 0
            if qty>0 and np.isfinite(qty): cash-=qty*contract_cost; contracts=qty; strike=K; entry_i=ex; trade_no+=1; event=f"BUY CALL #{trade_no}"
        mark=0.; opt_mid=np.nan; delta=np.nan; T=np.nan
        if contracts>0 and expiry is not None:
            opt_mid,delta,T=option_mark(float(df["Close"].iloc[ex]),strike,expiry,ts,iv_proxy(df,ex)); bid=opt_mid*max(1-OPTION_SPREAD/2,0.05); mark=contracts*bid*CONTRACT_MULTIPLIER
        equity=cash+mark; rows.append((ts,float(df["Close"].iloc[ex]),signal_ts,row["signal"],row["prob_consensus"],strike,expiry,contracts,opt_mid,delta,T,cash,mark,equity,event))
    return pd.DataFrame(rows,columns=["ts","nvda","signal_ts","signal","prob","strike","expiry","contracts","option_mid","delta","T","cash","option_value","equity","event"]).set_index("ts")

def plot_options(df,trace,start,end):
    x=trace[(trace.index>=start)&(trace.index<end)].copy()
    if x.empty: print("Grafico opciones: sin operaciones en el intervalo seleccionado."); return
    fig,axes=plt.subplots(3,1,figsize=(15,11),sharex=True,gridspec_kw={"height_ratios":[2.2,2.0,1.4]})
    ax=axes[0]; ax.plot(x.index,x["nvda"],label="NVDA"); active=x["strike"].notna(); ax.plot(x.index[active],x.loc[active,"strike"],label="CALL strike",linewidth=1.2); ax.set_title("V31 Options View | NVDA + CALL strike"); ax.legend(loc="upper left"); ax.grid(alpha=0.2); buys=x["event"].astype(str).str.startswith("BUY CALL"); sells=x["event"].astype(str).str.startswith("SELL CALL"); ax.scatter(x.index[buys],x.loc[buys,"nvda"],marker="^",s=65,label="CALL entry"); ax.scatter(x.index[sells],x.loc[sells,"nvda"],marker="v",s=65,label="CALL exit"); ax.legend(loc="upper left")
    ax=axes[1]; ax.plot(x.index,x["option_mid"]*CONTRACT_MULTIPLIER,label="CALL mid x 100"); ax.plot(x.index,x["option_value"],label="Position value"); ax.set_title(f"Synthetic CALL | DTE={OPTION_DTE} | target delta={OPTION_DELTA:.2f} | IVx={OPTION_IV_MULT:.2f} | spread={OPTION_SPREAD:.1%}"); ax.legend(loc="upper left"); ax.grid(alpha=0.2)
    ax=axes[2]; ax.plot(x.index,x["equity"],label="Option equity"); ax.plot(x.index,np.full(len(x),INITIAL_CASH),label="Start €500",linestyle="--"); ax.set_title("Synthetic options portfolio equity"); ax.legend(loc="upper left"); ax.grid(alpha=0.2)
    fig.tight_layout(); plt.show(block=True)

def main():
    t0=time.time(); df=load_data(); n=len(df); cut1=int(n*0.60); cut2=int(n*0.80); test_start=cut2+2
    print("=== V31 ROBUST AI | NVDA 1H | paper research ==="); print(f"Capital inicial: €{INITIAL_CASH:.2f}"); print("Generando predicciones...",flush=True)
    panel=predict_panel(df,cut1,n-1)
    if panel.empty: raise RuntimeError("No predictions generated")
    val=panel.index[panel.index<df.index[cut2]]; test=panel.index[panel.index>=df.index[test_start]]; vs,ve=val[0],val[-1]; ts,te=test[0],test[-1]; best=None; print(f"Validacion: {vs} -> {ve}"); print(f"Test ciego: {ts} -> {te}"); print("\n=== SWEEP VALIDACION ONLY ===")
    for th in THRESHOLDS:
        for con in CONSENSUS:
            r=backtest(df,panel,vs,ve,th,con); r["threshold"]=th; r["consensus"]=con; r["score"]=score(r); print(f"threshold={th:+.3%} | consensus>={con:.2f} | ret={r['ret']:+.2%} | trades={r['trades']} | DD={r['dd']:.2%} | Sharpe={r['sharpe']:.2f} | WR={r['wr']:.1%} | score={r['score']:+.4f}"); best=r if best is None or r["score"]>best["score"] else best
    test_r=backtest(df,panel,ts,te,best["threshold"],best["consensus"]); p=df[(df.index>=ts)&(df.index<te)]["Open"].astype(float); bh=float(p.iloc[-1]/p.iloc[0]-1) if len(p)>1 else 0.; bh_final=INITIAL_CASH*(1+bh)
    print("\n=== ELEGIDO | VALIDACION ONLY ==="); print(f"threshold={best['threshold']:+.3%} | min_consensus={best['consensus']:.2f}"); print(line("VALIDACION",best)); print("\n=== TEST CIEGO ==="); print(line("TEST IA",test_r)); print(f"TEST B&H: inicio=€{INITIAL_CASH:.2f} | final=€{bh_final:.2f} | ganado={bh_final-INITIAL_CASH:+.2f}€ | retorno={bh:+.2%}"); print(f"Diferencia IA vs B&H: {(test_r['ret']-bh):+.2%}")
    panel.to_csv("v31_predictions.csv"); print("Archivo: v31_predictions.csv"); print("\n=== OPCIONES | VISUALIZACION SINTETICA ==="); print(f"CALL: DTE={OPTION_DTE} | delta objetivo={OPTION_DELTA:.2f} | IVx={OPTION_IV_MULT:.2f} | spread={OPTION_SPREAD:.1%} | presupuesto={OPTION_BUDGET:.0%}"); print("Las primas del grafico son SYNTHETIC, no quotes historicos reales.")
    trace=option_trace(df,panel,ts,te,best["threshold"],best["consensus"]); trace.to_csv("v31_options_trace.csv"); print(f"Archivo: v31_options_trace.csv | filas={len(trace)} | entradas={int(trace['event'].astype(str).str.startswith('BUY CALL').sum())} | salidas={int(trace['event'].astype(str).str.startswith('SELL CALL').sum())}"); plot_options(df,trace,ts,te); print(f"runtime={(time.time()-t0)/60:.1f} min")

if __name__=="__main__": main()
