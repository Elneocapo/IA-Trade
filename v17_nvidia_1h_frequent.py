"""IA-Trade V17 - NVDA only, sequential 1H higher-frequency paper research.

Goal: investigate whether the V16 edge is being suppressed by requiring a positive
signal only. Keeps the V16 model/features/walk-forward setup, but tests symmetric
LONG/SHORT entries and shorter holds. Final test remains blind.
"""
from __future__ import annotations
import time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesClassifier
warnings.filterwarnings("ignore")

ASSET="NVDA"; PERIOD="2y"; INTERVAL="1h"
LOOKBACK_BARS=65; TRAIN_WINDOW=1200; RETRAIN_EVERY=24; MODEL_MAX_ITER=140
COST=0.001; INITIAL_CASH=50.0
HOLD_CANDIDATES=(2,3,4,6,8,10)
THRESHOLD_CANDIDATES=(0.0,0.00015,0.00025,0.00035,0.0005,0.00075,0.001,0.00125,0.0015,0.002)


def load_data():
    df=yf.download(ASSET,period=PERIOD,interval=INTERVAL,auto_adjust=True,progress=False,prepost=False)
    if df.empty: raise RuntimeError(f"No data returned for {ASSET}")
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df=df[["Open","High","Low","Close","Volume"]].copy().dropna()
    if getattr(df.index,"tz",None) is not None: df.index=df.index.tz_convert("America/New_York")
    return df.between_time("09:30","16:00")


def features(df):
    c,o,h,l,v=[df[k].astype(float) for k in ("Close","Open","High","Low","Volume")]
    r=np.log(c).diff(); z=pd.DataFrame(index=df.index)
    z["r1"]=r; z["body"]=(c-o)/(o+1e-9); z["range"]=(h-l)/(c+1e-9)
    z["loc"]=(c-l)/(h-l+1e-9); z["upper"]=(h-np.maximum(o,c))/(c+1e-9); z["lower"]=(np.minimum(o,c)-l)/(c+1e-9)
    z["volchg"]=np.log1p(v).diff()
    for n in (2,3,6,12,24,48,65):
        z[f"mom{n}"]=c.pct_change(n); z[f"vol{n}"]=r.rolling(n).std(); z[f"ema{n}"]=c/c.ewm(span=n,adjust=False).mean()-1; z[f"range{n}"]=z["range"].rolling(n).mean(); z[f"vr{n}"]=v/(v.rolling(n).mean()+1e-9)
    z["ema6_24"]=c.ewm(span=6,adjust=False).mean()/c.ewm(span=24,adjust=False).mean()-1
    z["ema12_48"]=c.ewm(span=12,adjust=False).mean()/c.ewm(span=48,adjust=False).mean()-1
    z["ema24_65"]=c.ewm(span=24,adjust=False).mean()/c.ewm(span=65,adjust=False).mean()-1
    z["vol_ratio"]=z["vol6"]/(z["vol48"]+1e-9); z["mom_ratio"]=z["mom6"]/(z["vol12"]+1e-9)
    for lag in (1,2,3,4,6,8,12,16,24,32,48):
        z[f"lag_r{lag}"]=r.shift(lag); z[f"lag_body{lag}"]=z["body"].shift(lag); z[f"lag_loc{lag}"]=z["loc"].shift(lag)
    hour=df.index.hour+df.index.minute/60
    z["hour_sin"]=np.sin(2*np.pi*(hour-9.5)/6.5); z["hour_cos"]=np.cos(2*np.pi*(hour-9.5)/6.5)
    z["dow_sin"]=np.sin(2*np.pi*df.index.dayofweek/5); z["dow_cos"]=np.cos(2*np.pi*df.index.dayofweek/5)
    return z.replace([np.inf,-np.inf],np.nan)


def make_supervised(df):
    f=features(df); c=df["Close"].astype(float); ret1=c.shift(-1)/c-1; ret3=c.shift(-3)/c-1; ret6=c.shift(-6)/c-1
    scale=f["vol6"].clip(lower=0.0005)
    return f,ret1,ret1/scale,ret3/scale,ret6/scale,(ret1>0.0015).astype(int)


def fit_models(X,y1,y3,y6,yc):
    common=dict(max_iter=MODEL_MAX_ITER,learning_rate=0.04,max_leaf_nodes=13,min_samples_leaf=20,l2_regularization=3.5,loss="absolute_error",random_state=42)
    return (HistGradientBoostingRegressor(**common).fit(X,y1),HistGradientBoostingRegressor(**common).fit(X,y3),HistGradientBoostingRegressor(**common).fit(X,y6),ExtraTreesClassifier(n_estimators=220,max_depth=9,min_samples_leaf=10,max_features=0.60,class_weight="balanced",n_jobs=-1,random_state=59).fit(X,yc))


def sequential_predictions(df,split_start,split_end):
    f,ret1,y1,y3,y6,yc=make_supervised(df); X=f.to_numpy(float); a,b,c,d=[q.to_numpy(float) for q in (y1,y3,y6,yc)]
    valid=[i for i in range(LOOKBACK_BARS-1,len(f)-6) if split_start<=i<split_end and np.isfinite(a[i]) and np.isfinite(b[i]) and np.isfinite(c[i]) and np.isfinite(X[i]).all()]
    out=[]; models=None; last_fit=-10**9
    for count,i in enumerate(valid,1):
        if models is None or i-last_fit>=RETRAIN_EVERY:
            end=i; start=max(LOOKBACK_BARS-1,end-TRAIN_WINDOW); idx=np.arange(start,end)
            good=np.isfinite(a[idx])&np.isfinite(b[idx])&np.isfinite(c[idx])&np.isfinite(d[idx])&np.isfinite(X[idx]).all(axis=1); idx=idx[good]
            if len(idx)>=400: models=fit_models(X[idx],a[idx],b[idx],c[idx],d[idx]); last_fit=i
        if models is None: continue
        m1,m3,m6,clf=models; row=X[i].reshape(1,-1)
        p1=float(m1.predict(row)[0]); p3=float(m3.predict(row)[0]); p6=float(m6.predict(row)[0]); prob=float(clf.predict_proba(row)[0,1])
        scale=max(float(f["vol6"].iloc[i]),0.0005); r1=p1*scale; r3=p3/3*scale; r6=p6/6*scale
        base=.62*r1+.25*r3+.13*r6; confidence=np.clip((prob-.5)*2,-1,1)
        # Symmetric conviction: negative confidence strengthens bearish signals too.
        signal=base*(.72+.85*abs(confidence))
        out.append((df.index[i],float(df["Close"].iloc[i]),r1,r3,r6,prob,signal,float(f["ema12_48"].iloc[i]),float(f["vol_ratio"].iloc[i])))
    return pd.DataFrame(out,columns=["date","price","pred1","pred3","pred6","prob_up","signal","trend","vol_ratio"]).set_index("date")


def backtest(df,panel,start,end,threshold,max_weight,trend_filter,vol_filter,max_hold):
    dates=[d for d in panel.index if start<=d<end]; cash=INITIAL_CASH; shares=0.; entry_i=None; entry_value=None; curve=[]; trs=[]
    for i,d in enumerate(dates):
        row=panel.loc[d]; price=float(row.price); total=cash+shares*price
        sig=float(row.signal); direction=1 if sig>threshold else (-1 if sig<-threshold else 0)
        if trend_filter:
            tr=float(row.trend); direction=direction if direction*tr>0 else 0
        if vol_filter and float(row.vol_ratio)>1.9: direction=0
        target=direction*max_weight
        if shares!=0 and entry_i is not None and i-entry_i>=max_hold: target=0
        tv=target*total; cur=shares*price
        if abs(tv-cur)>total*.02:
            if tv>cur:
                buy=tv-cur; shares+=buy/(price*(1+COST)); cash-=buy*(1+COST)
            else:
                sell=cur-tv; cash+=sell*(1-COST); shares-=sell/price
            if abs(shares)<1e-12:
                shares=0.
                if entry_value is not None: trs.append(total/entry_value-1)
                entry_i=None; entry_value=None
            elif entry_i is None:
                entry_i=i; entry_value=total
        curve.append(cash+shares*price)
    if len(curve)<2:return 0,0,0,0,0,INITIAL_CASH
    curve=np.asarray(curve); peak=np.maximum.accumulate(curve); dd=float(np.min(curve/peak-1)); rets=curve[1:]/curve[:-1]-1
    sh=float(np.mean(rets)/(np.std(rets)+1e-12)*np.sqrt(252*6.5)) if len(rets)>20 else 0.; wr=float(np.mean(np.asarray(trs)>0)) if trs else 0.; final=float(curve[-1])
    return final/INITIAL_CASH-1,len(trs),dd,sh,wr,final


def buy_and_hold(df,start,end):
    p=df[(df.index>=start)&(df.index<end)]["Close"].astype(float); return float(p.iloc[-1]/p.iloc[0]-1) if len(p)>1 else 0.


def main():
    overall=time.time(); df=load_data(); n=len(df); train_cut=int(n*.60); val_cut=int(n*.80); test_start=val_cut+2
    print(f"=== V17 NVDA FREQUENT 1H === | bars={n}",flush=True)
    panel=sequential_predictions(df,train_cut,n-1)
    val=panel.index[panel.index<df.index[val_cut]]; test=panel.index[panel.index>=df.index[test_start]]; vs,ve=val[0],val[-1]; ts,te=test[0],test[-1]
    candidates=[]
    for threshold in THRESHOLD_CANDIDATES:
        for weight in (.25,.35,.50,.70,.90,1.):
            for hold in HOLD_CANDIDATES:
                for tf in (False,True):
                    for vf in (False,True):
                        r,tr,dd,sh,wr,_=backtest(df,panel,vs,ve,threshold,weight,tf,vf,hold)
                        # Still optimize economics first; only a mild reward for useful activity.
                        score=r-.32*abs(min(dd,0))+.018*max(sh,0)+min(tr,30)*0.00015
                        if tr<8: score-=.005*(8-tr)
                        candidates.append((score,r,threshold,weight,hold,tf,vf,tr,dd,sh,wr))
    _,vr,threshold,weight,hold,tf,vf,vt,vdd,vsh,vwr=max(candidates,key=lambda x:x[0])
    fr,ft,fdd,fsh,fwr,final_cash=backtest(df,panel,ts,te,threshold,weight,tf,vf,hold); bh=buy_and_hold(df,ts,te); bh_cash=INITIAL_CASH*(1+bh)
    print("\n=== V17 NVDA FREQUENT 1H SUMMARY ===")
    print(f"asset={ASSET} | candles={INTERVAL} | history={PERIOD}"); print(f"context={LOOKBACK_BARS} bars | prediction=NEXT 1H candle | LONG+SHORT"); print(f"max_hold selected={hold} bars (~{hold/6.5:.1f} trading days)"); print(f"data={df.index[0]} -> {df.index[-1]} | total_bars={len(df)}"); print(f"validation={vs} -> {ve} | test={ts} -> {te}"); print(f"selected threshold=±{threshold:.2%} | max_weight={weight:.0%} | trend_filter={tf} | vol_filter={vf}"); print(f"validation IA={vr:+.2%} | DD={vdd:.2%} | Sharpe={vsh:.2f} | trades={vt} | win_rate={vwr:.1%}"); print(f"FINAL IA={fr:+.2%} | {ASSET} B&H={bh:+.2%} | trades={ft} | maxDD={fdd:.2%} | Sharpe={fsh:.2f} | win_rate={fwr:.1%}"); print(f"€{INITIAL_CASH:.2f} -> IA €{final_cash:.2f} | profit/loss={final_cash-INITIAL_CASH:+.2f}€"); print(f"€{INITIAL_CASH:.2f} -> B&H €{bh_cash:.2f} | profit/loss={bh_cash-INITIAL_CASH:+.2f}€"); print(f"IA beats {ASSET} B&H: {'YES' if fr>bh else 'NO'}"); print(f"runtime={(time.time()-overall)/60:.1f} min")

if __name__=="__main__": main()
