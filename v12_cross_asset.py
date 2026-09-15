"""IA-Trade v12 - cross-asset ranking and volatility-aware paper backtest."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

TICKERS=["AAPL","MSFT","SPY","NVDA","AMZN","META","GOOGL","QQQ","AMD","JPM"]
PERIOD="5y"; HORIZON=5; COST=.001; INITIAL_CASH=50.; LOOKBACK=40

def features(df):
    c,h,l,o,v=[df[x].astype(float) for x in ["Close","High","Low","Open","Volume"]]
    z=pd.DataFrame(index=df.index); lr=np.log(c).diff()
    z["ret1"]=lr; z["range"]=(h-l)/c; z["body"]=(c-o)/o; z["volchg"]=np.log1p(v).diff()
    for n in (3,5,8,13,21,40):
        z[f"mom{n}"]=c.pct_change(n); z[f"ema{n}"]=c/c.ewm(span=n,adjust=False).mean()-1; z[f"vol{n}"]=lr.rolling(n).std()
    z["range20"]=(c-l.rolling(20).min())/(h.rolling(20).max()-l.rolling(20).min()+1e-9)
    z["vr20"]=v/(v.rolling(20).mean()+1e-9)
    return z.replace([np.inf,-np.inf],np.nan)

def walk(df):
    x=features(df); c=df["Close"].astype(float); y=c.shift(-HORIZON)/c-1
    d=x.assign(y=y).dropna(); X=d[x.columns].values; Y=d.y.values
    n=len(d); warm=max(260,int(n*.5)); step=max(20,int(n*.05)); out=[]
    for end in range(warm,n,step):
        stop=min(end+step,n)
        m=HistGradientBoostingRegressor(max_iter=220,learning_rate=.035,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=1.5,loss="huber",random_state=42)
        m.fit(X[:end],Y[:end])
        for dt,p in zip(d.index[end:stop],m.predict(X[end:stop])):
            out.append((dt,float(p),float(c.loc[dt]),float(x.loc[dt,"vol20"])))
    return pd.DataFrame(out,columns=["date","pred","price","vol"]).set_index("date")

def load_panel():
    allp=[]
    for t in TICKERS:
        print(f"\n=== {t} ===",flush=True)
        df=yf.download(t,period=PERIOD,interval="1d",auto_adjust=True,progress=False)
        if df.empty: continue
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        p=walk(df); p["ticker"]=t; allp.append(p)
        print(f"pred5 mean={p.pred.mean():+.3%}",flush=True)
    return pd.concat(allp).sort_index()

def backtest(panel,start,end,k,edge,invest):
    dates=[d for d in sorted(panel.index.unique()) if start<=d<end]
    cash=INITIAL_CASH; pos={}; curve=[]; trades=0
    for d in dates:
        day=panel.loc[[d]]; total=cash
        for t,q in pos.items():
            r=day[day.ticker==t]
            if len(r): total+=q*float(r.price.iloc[0])
        chosen=day[day.pred>edge].sort_values("pred",ascending=False).head(k)
        targets={}
        if len(chosen):
            raw=[]
            for _,r in chosen.iterrows(): raw.append((r.ticker,max(float(r.pred)-edge,.0001)/(max(float(r.vol),.005)**.5)))
            s=sum(v for _,v in raw); targets={t:invest*v/s for t,v in raw}
        for t,q in list(pos.items()):
            r=day[day.ticker==t]
            if not len(r): continue
            price=float(r.price.iloc[0]); target=targets.get(t,0)*total; current=q*price
            if target<current*.98:
                sell=current-target; cash+=sell*(1-COST); pos[t]-=sell/price; trades+=1
                if pos[t]*price<.01: del pos[t]
        for _,r in chosen.iterrows():
            t=r.ticker; price=float(r.price); target=targets.get(t,0)*total; current=pos.get(t,0)*price; buy=max(0,target-current)
            if buy>total*.01:
                spend=min(buy,cash/(1+COST)); q=spend/(price*(1+COST)); cash-=spend*(1+COST); pos[t]=pos.get(t,0)+q; trades+=1
        marked=cash+sum(q*float(day[day.ticker==t].price.iloc[0]) for t,q in pos.items() if len(day[day.ticker==t]))
        curve.append(marked)
    if not curve:return 0.,0,0.
    curve=np.asarray(curve); peak=np.maximum.accumulate(curve); dd=np.min(curve/peak-1)
    return curve[-1]/INITIAL_CASH-1,trades,dd

def bench(panel,start,end,ticker):
    p=panel[(panel.ticker==ticker)&(panel.index>=start)&(panel.index<end)].price
    return float(p.iloc[-1]/p.iloc[0]-1) if len(p)>1 else 0.

def main():
    p=load_panel(); dates=sorted(p.index.unique()); cut=dates[len(dates)//2]; end=dates[-1]
    # Select policy only on the first half; final half is untouched.
    cand=[]
    for k in (1,2,3):
      for edge in (.001,.003,.005,.008):
       for invest in (.6,.8,1.):
        r,tr,dd=backtest(p,dates[0],cut,k,edge,invest); cand.append((r-.35*abs(min(dd,0)),r,k,edge,invest,dd,tr))
    _,vr,k,edge,invest,vdd,_=max(cand,key=lambda x:x[0]); fr,tr,dd=backtest(p,cut,end,k,edge,invest)
    spy=bench(p,cut,end,"SPY")
    print("\n=== V12 SUMMARY ===")
    print(f"selected: top_k={k} | min_edge={edge:.3%} | invest={invest:.0%}")
    print(f"validation IA={vr:+.2%} | validation DD={vdd:.2%}")
    print(f"FINAL IA={fr:+.2%} | SPY B&H={spy:+.2%} | trades={tr} | maxDD={dd:.2%}")
    print(f"IA beats SPY: {'YES' if fr>spy else 'NO'}")

if __name__=="__main__": main()
