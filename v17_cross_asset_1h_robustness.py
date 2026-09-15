"""IA-Trade V17 - cross-asset 1H robustness test.

Paper trading / research only. No broker or live orders.
Purpose: test whether the V16.7 NVDA signal survives on other liquid assets
without re-optimizing parameters for each asset.

IMPORTANT:
- Parameters are frozen from V16.7: threshold=0.18%, weight=100%, max_hold=6.
- No per-asset parameter selection.
- The final 20% is a blind test for each asset.
- This experiment is about robustness, not about maximizing any single result.
"""
from __future__ import annotations
import time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesClassifier
warnings.filterwarnings("ignore")

ASSETS=("NVDA","AAPL","MSFT","AMZN","META","SPY","QQQ")
PERIOD="2y"; INTERVAL="1h"
LOOKBACK_BARS=65; TRAIN_WINDOW=1200; RETRAIN_EVERY=24; MODEL_MAX_ITER=140
COST=0.001; INITIAL_CASH=50.0
THRESHOLD=0.0018; MAX_WEIGHT=1.0; MAX_HOLD=6


def load_data(asset):
    df=yf.download(asset,period=PERIOD,interval=INTERVAL,auto_adjust=True,progress=False,prepost=False)
    if df.empty: raise RuntimeError(f"No data returned for {asset}")
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
    f=features(df); c=df["Close"].astype(float); ret1=c.shift(-1); ret3=c.shift(-3); ret6=c.shift(-6)
    r1=ret1/c-1; r3=ret3/c-1; r6=ret6/c-1; scale=f["vol6"].clip(lower=0.0005)
    return f,r1,r1/scale,r3/scale,r6/scale,(r1>0.0015).astype(int)


def fit_models(X,y1,y3,y6,yc):
    common=dict(max_iter=MODEL_MAX_ITER,learning_rate=0.04,max_leaf_nodes=13,min_samples_leaf=20,l2_regularization=3.5,loss="absolute_error",random_state=42)
    return (HistGradientBoostingRegressor(**common).fit(X,y1),HistGradientBoostingRegressor(**common).fit(X,y3),HistGradientBoostingRegressor(**common).fit(X,y6),ExtraTreesClassifier(n_estimators=220,max_depth=9,min_samples_leaf=10,max_features=0.60,class_weight="balanced",n_jobs=-1,random_state=59).fit(X,yc))


def sequential_predictions(df):
    f,r1,y1,y3,y6,yc=make_supervised(df); X=f.to_numpy(float); a,b,c,d=[q.to_numpy(float) for q in (y1,y3,y6,yc)]
    n=len(df); train_cut=int(n*.60); valid=[i for i in range(LOOKBACK_BARS-1,n-6) if i>=train_cut and np.isfinite(a[i]) and np.isfinite(b[i]) and np.isfinite(c[i]) and np.isfinite(X[i]).all()]
    out=[]; models=None; last_fit=-10**9
    for i in valid:
        if models is None or i-last_fit>=RETRAIN_EVERY:
            end=i; start=max(LOOKBACK_BARS-1,end-TRAIN_WINDOW); idx=np.arange(start,end)
            good=np.isfinite(a[idx])&np.isfinite(b[idx])&np.isfinite(c[idx])&np.isfinite(d[idx])&np.isfinite(X[idx]).all(axis=1); idx=idx[good]
            if len(idx)>=400: models=fit_models(X[idx],a[idx],b[idx],c[idx],d[idx]); last_fit=i
        if models is None: continue
        m1,m3,m6,clf=models; row=X[i].reshape(1,-1)
        p1=float(m1.predict(row)[0]); p3=float(m3.predict(row)[0]); p6=float(m6.predict(row)[0]); prob=float(clf.predict_proba(row)[0,1])
        scale=max(float(f["vol6"].iloc[i]),0.0005); pr1=p1*scale; pr3=p3/3*scale; pr6=p6/6*scale
        base=.62*pr1+.25*pr3+.13*pr6; confidence=np.clip((prob-.5)*2,-1,1); signal=base*(.72+.85*max(confidence,0))
        out.append((df.index[i],float(df["Close"].iloc[i]),signal))
    return pd.DataFrame(out,columns=["date","price","signal"]).set_index("date")


def backtest(panel,start,end):
    dates=[d for d in panel.index if start<=d<end]; cash,shares=INITIAL_CASH,0.; entry=None; curve=[]; trs=[]; entry_value=None
    for i,d in enumerate(dates):
        row=panel.loc[d]; price=float(row.price); total=cash+shares*price; signal=float(row.signal)>THRESHOLD; target=MAX_WEIGHT if signal else 0.
        if shares>0 and entry is not None and i-entry>=MAX_HOLD: target=0.
        tv=target*total; cur=shares*price
        if tv<cur*.98 and cur>0:
            sell=cur-tv; cash+=sell*(1-COST); shares-=sell/price
            if shares<=1e-12:
                shares=0
                if entry_value is not None: trs.append(total/entry_value-1)
                entry=None; entry_value=None
        elif tv>cur*1.02:
            buy=min(tv-cur,cash/(1+COST))
            if buy>total*.01:
                before=total; shares+=buy/(price*(1+COST)); cash-=buy*(1+COST)
                if entry is None: entry=i; entry_value=before
        curve.append(cash+shares*price)
    if len(curve)<2:return 0,0,0,0,0,INITIAL_CASH
    curve=np.asarray(curve); peak=np.maximum.accumulate(curve); dd=float(np.min(curve/peak-1)); rets=curve[1:]/curve[:-1]-1
    sh=float(np.mean(rets)/(np.std(rets)+1e-12)*np.sqrt(252*6.5)) if len(rets)>20 else 0.; wr=float(np.mean(np.asarray(trs)>0)) if trs else 0.; final=float(curve[-1])
    return final/INITIAL_CASH-1,len(trs),dd,sh,wr,final


def main():
    overall=time.time(); rows=[]
    print("=== V17 CROSS-ASSET 1H ROBUSTNESS ===",flush=True)
    print(f"Frozen parameters: threshold={THRESHOLD:.2%}, max_hold={MAX_HOLD}, weight={MAX_WEIGHT:.0%}",flush=True)
    for asset in ASSETS:
        t=time.time()
        try:
            df=load_data(asset); n=len(df); test_start=int(n*.80)+2; panel=sequential_predictions(df)
            if panel.empty: raise RuntimeError("no predictions")
            ts=panel.index[panel.index>=df.index[test_start]][0]; te=panel.index[-1]
            fr,ft,fdd,fsh,fwr,final_cash=backtest(panel,ts,te)
            p=df[(df.index>=ts)&(df.index<te)]["Close"].astype(float); bh=float(p.iloc[-1]/p.iloc[0]-1) if len(p)>1 else 0.
            rows.append((asset,fr,bh,fr-bh,ft,fdd,fsh,fwr))
            print(f"{asset}: IA={fr:+.2%} | B&H={bh:+.2%} | excess={fr-bh:+.2%} | trades={ft} | DD={fdd:.2%} | Sharpe={fsh:.2f} | win={fwr:.1%} | {time.time()-t:.1f}s",flush=True)
        except Exception as e:
            print(f"{asset}: ERROR {e}",flush=True)
    print("\n=== V17 SUMMARY ===")
    if rows:
        r=pd.DataFrame(rows,columns=["asset","ia","bh","excess","trades","dd","sharpe","win_rate"])
        print(r.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
        print(f"assets tested={len(r)} | IA positive={int((r.ia>0).sum())}/{len(r)} | beats B&H={int((r.excess>0).sum())}/{len(r)}")
        print(f"median IA={r.ia.median():+.2%} | median excess={r.excess.median():+.2%} | total trades={int(r.trades.sum())}")
    print(f"runtime={(time.time()-overall)/60:.1f} min")

if __name__=="__main__": main()
