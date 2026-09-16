"""IA-Trade V30 - señal robusta para NVDA 1H.

PAPER RESEARCH ONLY. NO BROKER. NO LIVE ORDERS.
"""
from __future__ import annotations
import math,time,warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
ASSET="NVDA"; PERIOD="2y"; INTERVAL="1h"; LOOKBACK=65; TRAIN_WINDOW=1200; RETRAIN_EVERY=24; MAX_ITER=120; INITIAL_CASH=500.0; COST=0.001; MAX_HOLD=6
THRESHOLDS=(0.0,0.00025,0.0005,0.00075,0.001,0.00125,0.0015,0.00175,0.002,0.0025,0.003); CONSENSUS=(0.50,0.67,1.00)

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
    f["trend_agreement"]=np.sign(f["ema6_24"])*np.sign(f["ema12_48"]); f["trend_strength"]=0.6*f["ema6_24"]+0.4*f["ema12_48"]; f["range6_vs_48"]=f["range6"]/(f["range48"]+1e-9); f["vol6_vs_24"]=f["vol6"]/(f["vol24"]+1e-9); f["bar_pressure"]=(f["body"]+f["loc"]-0.5)*f["range"]; f["volume_pressure"]=f["bar_pressure"]*f["vr6"]; f["session_x_mom6"]=prog*f["mom6"]; f["session_x_volratio"]=prog*f["vol_ratio"]
    # Previous-session close only: never uses the final close of the current session.
    day_key=pd.Index(df.index.normalize()); daily_last=c.groupby(day_key).last(); prev_daily_last=daily_last.shift(1); prev_close=pd.Series(day_key,index=df.index).map(prev_daily_last); f["gap_prev_close"]=o/(prev_close+1e-9)-1
    return f.replace([np.inf,-np.inf],np.nan)

def fit_set(X,y1,y3,y6,c1,c3):
    a=dict(max_iter=MAX_ITER,learning_rate=0.04,max_leaf_nodes=13,min_samples_leaf=20,l2_regularization=3.5,loss="absolute_error",random_state=42); b=dict(max_iter=MAX_ITER,learning_rate=0.035,max_leaf_nodes=19,min_samples_leaf=28,l2_regularization=6.5,loss="absolute_error",random_state=77)
    return {"r1a":HistGradientBoostingRegressor(**a).fit(X,y1),"r3a":HistGradientBoostingRegressor(**a).fit(X,y3),"r6a":HistGradientBoostingRegressor(**a).fit(X,y6),"r1b":HistGradientBoostingRegressor(**b).fit(X,y1),"r3b":HistGradientBoostingRegressor(**b).fit(X,y3),"r6b":HistGradientBoostingRegressor(**b).fit(X,y6),"c1":ExtraTreesClassifier(n_estimators=220,max_depth=9,min_samples_leaf=10,max_features=0.60,class_weight="balanced",n_jobs=-1,random_state=59).fit(X,c1),"c3":ExtraTreesClassifier(n_estimators=220,max_depth=8,min_samples_leaf=12,max_features=0.65,class_weight="balanced",n_jobs=-1,random_state=83).fit(X,c3)}

def predict_panel(df,start_i,end_i):
    f=make_features(df); c=df["Close"].astype(float); ret1=c.shift(-1)/c-1; ret3=c.shift(-3)/c-1; ret6=c.shift(-6)/c-1; scale=f["vol6"].clip(lower=0.0005); y1=ret1/scale; y3=ret3/scale; y6=ret6/scale; c1=(ret1>0.0015).astype(int); c3=(ret3>0.0015).astype(int); X=f.to_numpy(float); arrs=[q.to_numpy(float) for q in (y1,y3,y6,c1,c3)]; valid=[i for i in range(LOOKBACK-1,len(df)-6) if start_i<=i<end_i and np.isfinite(X[i]).all() and all(np.isfinite(q[i]) for q in arrs[:3])]; models=None; last_fit=-10**9; out=[]; t=time.time()
    for n,i in enumerate(valid,1):
        if models is None or i-last_fit>=RETRAIN_EVERY:
            idx=np.arange(max(LOOKBACK-1,i-TRAIN_WINDOW),i); good=np.isfinite(X[idx]).all(axis=1)
            for q in arrs: good &= np.isfinite(q[idx])
            idx=idx[good]
            if len(idx)>=450: models=fit_set(X[idx],y1.to_numpy()[idx],y3.to_numpy()[idx],y6.to_numpy()[idx],c1.to_numpy()[idx].astype(int),c3.to_numpy()[idx].astype(int)); last_fit=i
        if models is None: continue
        row=X[i].reshape(1,-1); p1=0.5*(models["r1a"].predict(row)[0]+models["r1b"].predict(row)[0]); p3=0.5*(models["r3a"].predict(row)[0]+models["r3b"].predict(row)[0]); p6=0.5*(models["r6a"].predict(row)[0]+models["r6b"].predict(row)[0]); prob1=float(models["c1"].predict_proba(row)[0,1]); prob3=float(models["c3"].predict_proba(row)[0,1]); prob=0.58*prob1+0.42*prob3; s=max(float(scale.iloc[i]),0.0005); r1=float(p1*s); r3=float(p3*s/3); r6=float(p6*s/6); base=0.56*r1+0.28*r3+0.16*r6; conf=np.clip((prob-0.5)*2,-1,1); trend=float(f["trend_agreement"].iloc[i]); boost=1.08 if trend>0 else (0.92 if trend<0 else 1.0); signal=base*(0.70+0.80*max(conf,0))*boost; consensus=float(np.mean([r1>0,r3>0,r6>0])); out.append((df.index[i],signal,prob,consensus,r1,r3,r6))
        if n==1 or n%200==0 or n==len(valid): elapsed=time.time()-t; rate=n/max(elapsed,1e-9); eta=(len(valid)-n)/max(rate,1e-9); print(f"prediction {n}/{len(valid)} | {rate:.1f} bars/s | ETA ~{eta/60:.1f} min",flush=True)
    return pd.DataFrame(out,columns=["date","signal","prob_consensus","consensus","pred1","pred3","pred6"]).set_index("date")

def backtest(df,panel,start,end,threshold,minimum_consensus):
    pos={ts:i for i,ts in enumerate(df.index)}; cash=INITIAL_CASH; shares=0.; entry=None; outlay=None; trades=[]; curve=[]
    for ts,row in panel[(panel.index>=start)&(panel.index<end)].iterrows():
        i=pos.get(ts)
        if i is None or i+1>=len(df): continue
        ex=i+1; px=float(df["Open"].iloc[ex]);
        if not np.isfinite(px) or px<=0: continue
        want=float(row["signal"])>threshold and float(row["consensus"])>=minimum_consensus and float(row["prob_consensus"])>0.50; equity=cash+shares*px; target=equity if want else 0.; current=shares*px
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

def main():
    t0=time.time(); df=load_data(); n=len(df); cut1=int(n*0.60); cut2=int(n*0.80); test_start=cut2+2; print("=== V30 ROBUST AI | NVDA 1H | paper research ==="); print(f"Capital inicial: €{INITIAL_CASH:.2f}"); print("Generando predicciones...",flush=True)
    panel=predict_panel(df,cut1,n-1)
    if panel.empty: raise RuntimeError("No predictions generated")
    val=panel.index[panel.index<df.index[cut2]]; test=panel.index[panel.index>=df.index[test_start]]; vs,ve=val[0],val[-1]; ts,te=test[0],test[-1]; print(f"Validacion: {vs} -> {ve}"); print(f"Test ciego: {ts} -> {te}"); best=None; print("\n=== SWEEP VALIDACION ONLY ===")
    for th in THRESHOLDS:
        for con in CONSENSUS:
            r=backtest(df,panel,vs,ve,th,con); r["threshold"]=th; r["consensus"]=con; r["score"]=score(r); print(f"threshold={th:+.3%} | consensus>={con:.2f} | ret={r['ret']:+.2%} | trades={r['trades']} | DD={r['dd']:.2%} | Sharpe={r['sharpe']:.2f} | WR={r['wr']:.1%} | score={r['score']:+.4f}"); best=r if best is None or r["score"]>best["score"] else best
    test_r=backtest(df,panel,ts,te,best["threshold"],best["consensus"]); p=df[(df.index>=ts)&(df.index<te)]["Open"].astype(float); bh=float(p.iloc[-1]/p.iloc[0]-1) if len(p)>1 else 0.; bh_final=INITIAL_CASH*(1+bh); print("\n=== ELEGIDO | VALIDACION ONLY ==="); print(f"threshold={best['threshold']:+.3%} | min_consensus={best['consensus']:.2f}"); print(line("VALIDACION",best)); print("\n=== TEST CIEGO ==="); print(line("TEST IA",test_r)); print(f"TEST B&H: inicio=€{INITIAL_CASH:.2f} | final=€{bh_final:.2f} | ganado={bh_final-INITIAL_CASH:+.2f}€ | retorno={bh:+.2%}"); print(f"Diferencia IA vs B&H: {(test_r['ret']-bh):+.2%}"); panel.to_csv("v30_predictions.csv"); print("Archivo: v30_predictions.csv"); print(f"runtime={(time.time()-t0)/60:.1f} min")

if __name__=="__main__": main()
