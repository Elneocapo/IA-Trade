"""IA-Trade V27 - evolucion directa de V23, session-aware reforzada.

Paper research only. No broker or live orders.

Base: V23. Mantiene la arquitectura walk-forward y la ejecucion next-open,
pero añade al entrenamiento interacciones explicitas entre estructura de precio,
volatilidad y fase de la sesion. El test final nunca selecciona parametros.
"""
from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesClassifier
warnings.filterwarnings("ignore")

ASSET="NVDA"; PERIOD="2y"; INTERVAL="1h"
LOOKBACK_BARS=65; TRAIN_WINDOW=1200; RETRAIN_EVERY=24; MODEL_MAX_ITER=140
INITIAL_CASH=50.0; COST=0.001; MAX_HOLD=6; WEIGHT=1.0
THRESHOLD_CANDIDATES=(0.0,0.00025,0.0005,0.00075,0.001,0.00125,0.0015,0.00175,0.002,0.0025,0.003,0.004)

def load_data():
    print(f"=== V27 | {ASSET} | {INTERVAL} | base V23 + session interactions ===",flush=True)
    df=yf.download(ASSET,period=PERIOD,interval=INTERVAL,auto_adjust=True,progress=False,prepost=False)
    if df.empty: raise RuntimeError(f"No data returned for {ASSET}")
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df=df[["Open","High","Low","Close","Volume"]].copy().dropna()
    if getattr(df.index,"tz",None) is not None: df.index=df.index.tz_convert("America/New_York")
    return df.between_time("09:30","16:00")

def features(df):
    c=df["Close"].astype(float); o=df["Open"].astype(float); h=df["High"].astype(float); l=df["Low"].astype(float); v=df["Volume"].astype(float)
    r=np.log(c).diff(); z=pd.DataFrame(index=df.index)
    z["r1"]=r; z["body"]=(c-o)/(o+1e-9); z["range"]=(h-l)/(c+1e-9); z["loc"]=(c-l)/(h-l+1e-9)
    z["upper"]=(h-np.maximum(o,c))/(c+1e-9); z["lower"]=(np.minimum(o,c)-l)/(c+1e-9); z["volchg"]=np.log1p(v).diff()
    for n in (2,3,6,12,24,48,65):
        z[f"mom{n}"]=c.pct_change(n); z[f"vol{n}"]=r.rolling(n).std(); z[f"ema{n}"]=c/c.ewm(span=n,adjust=False).mean()-1
        z[f"range{n}"]=z["range"].rolling(n).mean(); z[f"vr{n}"]=v/(v.rolling(n).mean()+1e-9)
    z["ema6_24"]=c.ewm(span=6,adjust=False).mean()/c.ewm(span=24,adjust=False).mean()-1
    z["ema12_48"]=c.ewm(span=12,adjust=False).mean()/c.ewm(span=48,adjust=False).mean()-1
    z["ema24_65"]=c.ewm(span=24,adjust=False).mean()/c.ewm(span=65,adjust=False).mean()-1
    z["vol_ratio"]=z["vol6"]/(z["vol48"]+1e-9); z["mom_ratio"]=z["mom6"]/(z["vol12"]+1e-9)
    for lag in (1,2,3,4,6,8,12,16,24,32,48):
        z[f"lag_r{lag}"]=r.shift(lag); z[f"lag_body{lag}"]=z["body"].shift(lag); z[f"lag_loc{lag}"]=z["loc"].shift(lag)

    minutes=(df.index.hour*60+df.index.minute)-(9*60+30); progress=np.clip(minutes/390.0,0.0,1.0)
    first30=(minutes<30).astype(float); first60=(minutes<60).astype(float); first90=(minutes<90).astype(float)
    midday=((minutes>=120)&(minutes<270)).astype(float); last60=(minutes>=330).astype(float); last30=(minutes>=360).astype(float)
    z["hour_sin"]=np.sin(2*np.pi*(df.index.hour+df.index.minute/60-9.5)/6.5); z["hour_cos"]=np.cos(2*np.pi*(df.index.hour+df.index.minute/60-9.5)/6.5)
    z["dow_sin"]=np.sin(2*np.pi*df.index.dayofweek/5); z["dow_cos"]=np.cos(2*np.pi*df.index.dayofweek/5)
    z["session_progress"]=progress; z["session_sin"]=np.sin(2*np.pi*progress); z["session_cos"]=np.cos(2*np.pi*progress)
    z["first_30m"]=first30; z["first_60m"]=first60; z["first_90m"]=first90; z["midday"]=midday; z["last_60m"]=last60; z["last_30m"]=last30
    z["open_distance"]=progress; z["open_distance_sq"]=progress**2; z["close_distance"]=1-progress
    z["is_monday"]=(df.index.dayofweek==0).astype(float); z["is_friday"]=(df.index.dayofweek==4).astype(float)

    # V27: interactions that let the model learn that identical price action can mean different things at different times.
    for name,flag in (("open30",first30),("open60",first60),("open90",first90),("mid",midday),("last60",last60),("last30",last30)):
        z[f"{name}_body"]=z["body"]*flag
        z[f"{name}_range"]=z["range"]*flag
        z[f"{name}_r1"]=z["r1"]*flag
        z[f"{name}_volratio"]=z["vol_ratio"]*flag
        z[f"{name}_mom6"]=z["mom6"]*flag
    z["session_x_mom6"]=progress*z["mom6"]
    z["session_x_volratio"]=progress*z["vol_ratio"]
    z["session_x_range"]=progress*z["range"]
    z["early_strength"]=first90*z["mom6"]
    z["late_strength"]=last90 if False else last60*z["mom6"]

    # Gap relativo al cierre de la sesion anterior, conocido antes de emitir la señal.
    prev_close=c.groupby(df.index.normalize()).transform("last").shift(1)
    z["gap_prev_close"]=(o/prev_close-1.0)

    return z.replace([np.inf,-np.inf],np.nan)

def make_supervised(df):
    f=features(df); c=df["Close"].astype(float)
    ret1=c.shift(-1)/c-1; ret3=c.shift(-3)/c-1; ret6=c.shift(-6)/c-1; scale=f["vol6"].clip(lower=0.0005)
    return f,ret1,ret1/scale,ret3/scale,ret6/scale,(ret1>0.0015).astype(int)

def fit_models(X,y1,y3,y6,yc):
    common=dict(max_iter=MODEL_MAX_ITER,learning_rate=0.04,max_leaf_nodes=13,min_samples_leaf=20,l2_regularization=3.5,loss="absolute_error",random_state=42)
    return (HistGradientBoostingRegressor(**common).fit(X,y1),HistGradientBoostingRegressor(**common).fit(X,y3),HistGradientBoostingRegressor(**common).fit(X,y6),ExtraTreesClassifier(n_estimators=220,max_depth=9,min_samples_leaf=10,max_features=0.60,class_weight="balanced",n_jobs=-1,random_state=59).fit(X,yc))

def sequential_predictions(df,split_start,split_end):
    f,ret1,y1,y3,y6,yc=make_supervised(df); X=f.to_numpy(float); a,b,c,d=[q.to_numpy(float) for q in (y1,y3,y6,yc)]
    valid=[i for i in range(LOOKBACK_BARS-1,len(f)-6) if split_start<=i<split_end and np.isfinite(a[i]) and np.isfinite(b[i]) and np.isfinite(c[i]) and np.isfinite(X[i]).all()]
    out=[]; models=None; last_fit=-10**9; started=time.time(); total=len(valid)
    for count,i in enumerate(valid,1):
        if models is None or i-last_fit>=RETRAIN_EVERY:
            end=i; start=max(LOOKBACK_BARS-1,end-TRAIN_WINDOW); idx=np.arange(start,end)
            good=np.isfinite(a[idx])&np.isfinite(b[idx])&np.isfinite(c[idx])&np.isfinite(d[idx])&np.isfinite(X[idx]).all(axis=1); idx=idx[good]
            if len(idx)>=400: models=fit_models(X[idx],a[idx],b[idx],c[idx],d[idx]); last_fit=i
        if models is None: continue
        m1,m3,m6,clf=models; row=X[i].reshape(1,-1)
        p1=float(m1.predict(row)[0]); p3=float(m3.predict(row)[0]); p6=float(m6.predict(row)[0]); prob=float(clf.predict_proba(row)[0,1])
        scale=max(float(f["vol6"].iloc[i]),0.0005); r1=p1*scale; r3=p3/3*scale; r6=p6/6*scale
        base=.62*r1+.25*r3+.13*r6; confidence=np.clip((prob-.5)*2,-1,1); signal=base*(.72+.85*max(confidence,0))
        out.append((df.index[i],float(df["Close"].iloc[i]),r1,r3,r6,prob,signal,float(f["ema12_48"].iloc[i]),float(f["vol_ratio"].iloc[i]),float(f["session_progress"].iloc[i])))
        if count==1 or count%200==0 or count==total:
            elapsed=time.time()-started; rate=count/max(elapsed,1e-9); eta=(total-count)/max(rate,1e-9); print(f"prediction {count}/{total} | {rate:.1f} bars/s | ETA ~{eta/60:.1f} min",flush=True)
    return pd.DataFrame(out,columns=["date","price","pred1","pred3","pred6","prob_up","signal","trend","vol_ratio","session_progress"]).set_index("date")

def backtest_next_open(df,panel,start,end,threshold):
    positions={ts:i for i,ts in enumerate(df.index)}; signals=panel[(panel.index>=start)&(panel.index<end)]
    cash=float(INITIAL_CASH); shares=0.; entry_i=None; entry_outlay=None; trade_returns=[]; curve=[]
    for signal_ts,row in signals.iterrows():
        i=positions.get(signal_ts)
        if i is None or i+1>=len(df): continue
        execution_i=i+1; execution_ts=df.index[execution_i]
        if execution_ts>=end: continue
        px=float(df["Open"].iloc[execution_i]);
        if not np.isfinite(px) or px<=0: continue
        equity=cash+shares*px; want=float(row["signal"])>threshold; target=equity*WEIGHT if want else 0.; current=shares*px
        if shares>0 and entry_i is not None and execution_i-entry_i>=MAX_HOLD: target=0.
        if target<current*.98 and shares>0:
            sell_value=min(current,current-target); sell_shares=min(shares,sell_value/px); cash+=sell_shares*px*(1-COST); shares-=sell_shares
            if shares<=1e-12:
                shares=0.
                if entry_outlay is not None: trade_returns.append(cash/entry_outlay-1.)
                entry_i=None; entry_outlay=None
        elif target>current*1.02:
            desired=min(target-current,cash/(1+COST))
            if desired>max(.01,equity*.01):
                before=cash+shares*px; shares+=desired/(px*(1+COST)); cash-=desired
                if entry_i is None: entry_i=execution_i; entry_outlay=before
        curve.append(cash+shares*px)
    if len(curve)<2:return {"return":0.,"final":INITIAL_CASH,"trades":0,"max_dd":0.,"sharpe":0.,"win_rate":0.}
    curve=np.asarray(curve); peak=np.maximum.accumulate(curve); dd=float(np.min(curve/peak-1)); rets=curve[1:]/curve[:-1]-1; sh=float(np.mean(rets)/(np.std(rets)+1e-12)*np.sqrt(252*6.5)) if len(rets)>20 else 0.; wr=float(np.mean(np.asarray(trade_returns)>0)) if trade_returns else 0.; final=float(curve[-1])
    return {"return":final/INITIAL_CASH-1.,"final":final,"trades":len(trade_returns),"max_dd":dd,"sharpe":sh,"win_rate":wr}

def score(r):
    s=r["return"]-0.30*abs(min(r["max_dd"],0.))+0.015*max(r["sharpe"],0.)
    if r["trades"]<15: s-=0.003*(15-r["trades"])
    return s

def buyhold(df,start,end):
    p=df[(df.index>=start)&(df.index<end)]["Open"].astype(float); return float(p.iloc[-1]/p.iloc[0]-1) if len(p)>1 else 0.

def main():
    t0=time.time(); df=load_data(); n=len(df); train_cut=int(n*.60); val_cut=int(n*.80); test_start=val_cut+2
    print("V27: generando predicciones basadas directamente en V23...",flush=True); panel=sequential_predictions(df,train_cut,n-1)
    if panel.empty: raise RuntimeError("No predictions")
    val=panel.index[panel.index<df.index[val_cut]]; test=panel.index[panel.index>=df.index[test_start]]; vs,ve=val[0],val[-1]; ts,te=test[0],test[-1]
    print(f"Validacion: {vs} -> {ve}"); print(f"Test ciego: {ts} -> {te}"); print("\n=== SWEEP UMBRAL | VALIDACION ONLY ===")
    results=[]
    for th in THRESHOLD_CANDIDATES:
        r=backtest_next_open(df,panel,vs,ve,th); r["threshold"]=th; r["score"]=score(r); results.append(r)
        print(f"threshold={th:+.3%} | ret={r['return']:+.2%} | trades={r['trades']} | DD={r['max_dd']:.2%} | Sharpe={r['sharpe']:.2f} | WR={r['win_rate']:.1%} | score={r['score']:+.4f}")
    best=max(results,key=lambda x:x["score"]); th=best["threshold"]; test_r=backtest_next_open(df,panel,ts,te,th); bh=buyhold(df,ts,te)
    print("\n=== CONFIGURACION ELEGIDA ==="); print(f"threshold={th:+.3%} (solo validacion)"); print(f"VALIDACION: retorno={best['return']:+.2%} | final=€{best['final']:.2f} | trades={best['trades']} | DD={best['max_dd']:.2%} | Sharpe={best['sharpe']:.2f} | WR={best['win_rate']:.1%}")
    print("\n=== TEST CIEGO ==="); print(f"TEST IA: retorno={test_r['return']:+.2%} | final=€{test_r['final']:.2f} | trades={test_r['trades']} | DD={test_r['max_dd']:.2%} | Sharpe={test_r['sharpe']:.2f} | WR={test_r['win_rate']:.1%}"); print(f"TEST B&H: {bh:+.2%} | final=€{INITIAL_CASH*(1+bh):.2f}"); print("Comparacion IA vs B&H:","POSITIVA" if test_r["return"]>bh else "NEGATIVA"); print(f"runtime={(time.time()-t0)/60:.1f} min"); print("NO reajustar parametros usando el test.")

if __name__=="__main__": main()
