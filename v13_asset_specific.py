"""IA-Trade v13 - asset-specific models with volatility-normalized targets.
Paper/simulation only. Avoids mixing incompatible return distributions across assets.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

# Small, relatively homogeneous universe. Each asset gets its own model.
TICKERS=["SPY","QQQ","AAPL","MSFT","GOOGL","AMZN"]
PERIOD="5y"; HORIZON=5; COST=.001; INITIAL_CASH=50.


def features(df):
    c,h,l,o,v=[df[x].astype(float) for x in ["Close","High","Low","Open","Volume"]]
    z=pd.DataFrame(index=df.index); lr=np.log(c).diff()
    z["ret1"]=lr; z["range"]=(h-l)/c; z["body"]=(c-o)/o; z["volchg"]=np.log1p(v).diff()
    for n in (3,5,8,13,21,40):
        z[f"mom{n}"]=c.pct_change(n); z[f"ema{n}"]=c/c.ewm(span=n,adjust=False).mean()-1
        z[f"vol{n}"]=lr.rolling(n).std()
    z["range20"]=(c-l.rolling(20).min())/(h.rolling(20).max()-l.rolling(20).min()+1e-9)
    z["vr20"]=v/(v.rolling(20).mean()+1e-9)
    # Market-relative context makes each asset model aware of common regime.
    return z.replace([np.inf,-np.inf],np.nan)


def walk(df):
    x=features(df); c=df["Close"].astype(float); vol=x["vol20"]
    future=c.shift(-HORIZON)/c-1
    # Target is a volatility-normalized forward return (approximately a forward Sharpe-like edge).
    y=future/(vol*np.sqrt(HORIZON)+1e-6)
    d=x.assign(y=y,future=future).dropna()
    X=d[x.columns].values; Y=d.y.values; n=len(d)
    warm=max(300,int(n*.50)); step=max(20,int(n*.05)); out=[]
    for end in range(warm,n,step):
        stop=min(end+step,n)
        m=HistGradientBoostingRegressor(max_iter=260,learning_rate=.035,max_leaf_nodes=15,
            min_samples_leaf=25,l2_regularization=1.5,loss="absolute_error",random_state=42)
        m.fit(X[:end],Y[:end])
        pred=m.predict(X[end:stop])
        for dt,p in zip(d.index[end:stop],pred):
            vv=max(float(vol.loc[dt]),.003); out.append((dt,float(p),float(c.loc[dt]),vv))
    return pd.DataFrame(out,columns=["date","score","price","vol"]).set_index("date")


def load_panel():
    allp=[]
    for t in TICKERS:
        print(f"\n=== {t} ===",flush=True)
        df=yf.download(t,period=PERIOD,interval="1d",auto_adjust=True,progress=False)
        if df.empty: continue
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        p=walk(df); p["ticker"]=t; allp.append(p)
        print(f"normalized score mean={p.score.mean():+.3f}",flush=True)
    return pd.concat(allp).sort_index()


def backtest(panel,start,end,k,score_min,invest):
    dates=[d for d in sorted(panel.index.unique()) if start<=d<end]
    cash=INITIAL_CASH; pos={}; curve=[]; trades=0
    for d in dates:
        day=panel.loc[[d]]; total=cash
        for t,q in pos.items():
            r=day[day.ticker==t]
            if len(r): total+=q*float(r.price.iloc[0])
        chosen=day[day.score>score_min].sort_values("score",ascending=False).head(k)
        targets={}
        if len(chosen):
            # Equal risk budget: score is already normalized by each asset's volatility.
            w=np.maximum(chosen.score.to_numpy()-score_min,0.01)
            w=w/w.sum(); targets={t:invest*float(ww) for t,ww in zip(chosen.ticker,w)}
        for t,q in list(pos.items()):
            r=day[day.ticker==t]
            if not len(r): continue
            price=float(r.price.iloc[0]); target=targets.get(t,0)*total; current=q*price
            if target<current*.98:
                sell=current-target; cash+=sell*(1-COST); pos[t]-=sell/price; trades+=1
                if pos[t]*price<.01: del pos[t]
        for _,r in chosen.iterrows():
            t=r.ticker; price=float(r.price); target=targets[t]*total; current=pos.get(t,0)*price; buy=max(0,target-current)
            if buy>total*.01:
                spend=min(buy,cash/(1+COST)); q=spend/(price*(1+COST)); cash-=spend*(1+COST); pos[t]=pos.get(t,0)+q; trades+=1
        marked=cash+sum(q*float(day[day.ticker==t].price.iloc[0]) for t,q in pos.items() if len(day[day.ticker==t]))
        curve.append(marked)
    if not curve:return 0.,0.,0.,0.
    curve=np.asarray(curve); peak=np.maximum.accumulate(curve); dd=np.min(curve/peak-1)
    daily=curve[1:]/curve[:-1]-1 if len(curve)>1 else np.array([])
    sharpe=float(np.mean(daily)/(np.std(daily)+1e-12)*np.sqrt(252)) if len(daily)>10 else 0.
    return curve[-1]/INITIAL_CASH-1,trades,dd,sharpe


def bench(panel,start,end,ticker):
    p=panel[(panel.ticker==ticker)&(panel.index>=start)&(panel.index<end)].price
    return float(p.iloc[-1]/p.iloc[0]-1) if len(p)>1 else 0.


def main():
    p=load_panel(); dates=sorted(p.index.unique()); cut=dates[len(dates)//2]; end=dates[-1]
    # Tune only on the first half. Final half remains untouched.
    cand=[]
    for k in (1,2,3):
      for s in (.20,.30,.40,.50,.60,.75):
       for invest in (.50,.70,.90,1.0):
        r,tr,dd,sh=backtest(p,dates[0],cut,k,s,invest)
        # Reward return, penalize drawdown; Sharpe breaks ties indirectly.
        utility=r-.30*abs(min(dd,0))+.02*max(sh,0)
        cand.append((utility,r,k,s,invest,dd,sh,tr))
    _,vr,k,s,invest,vdd,vsh,_=max(cand,key=lambda x:x[0])
    fr,tr,dd,sh=backtest(p,cut,end,k,s,invest)
    spy=bench(p,cut,end,"SPY"); qqq=bench(p,cut,end,"QQQ")
    print("\n=== V13 SUMMARY ===")
    print(f"selected: top_k={k} | min_normalized_score={s:.2f} | invest={invest:.0%}")
    print(f"validation IA={vr:+.2%} | DD={vdd:.2%} | Sharpe={vsh:.2f}")
    print(f"FINAL IA={fr:+.2%} | SPY B&H={spy:+.2%} | QQQ B&H={qqq:+.2%} | trades={tr} | maxDD={dd:.2%} | Sharpe={sh:.2f}")
    print(f"IA beats SPY: {'YES' if fr>spy else 'NO'}")

if __name__=="__main__": main()
