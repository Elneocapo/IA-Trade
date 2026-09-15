"""IA-Trade V19 - NVDA only, 1H entry refinement from V16.7.
Paper trading / research only. Final test is never used for parameter selection.

Core model/features/walk-forward/signal are kept from V16.7.
Only entry conversion is changed: fixed threshold vs adaptive rolling
signal-percentile entry, using past predictions only. Goal: test whether
more trades can be obtained without destroying the V16.7 edge.
"""
from __future__ import annotations
import time,warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesClassifier
warnings.filterwarnings("ignore")
ASSET="NVDA"; PERIOD="2y"; INTERVAL="1h"
LOOKBACK_BARS=65; TRAIN_WINDOW=1200; RETRAIN_EVERY=24; MODEL_MAX_ITER=140
COST=.001; INITIAL_CASH=50.; HOLD_CANDIDATES=(4,6,8,10,13)
THRESHOLD_CANDIDATES=(-.0015,-.00125,-.001,-.00075,-.0005,-.00025,0,.00025,.0005,.00075,.001,.00125,.0015,.00175,.002,.0025,.003,.004)
QUANTILE_CANDIDATES=(.60,.65,.70,.75,.80,.85)
WINDOW_CANDIDATES=(24,48,72,120)

def load_data():
 d=yf.download(ASSET,period=PERIOD,interval=INTERVAL,auto_adjust=True,progress=False,prepost=False)
 if d.empty: raise RuntimeError("No data")
 if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
 d=d[["Open","High","Low","Close","Volume"]].copy().dropna()
 if getattr(d.index,"tz",None) is not None:d.index=d.index.tz_convert("America/New_York")
 return d.between_time("09:30","16:00")

def features(df):
 c,o,h,l,v=[df[k].astype(float) for k in ("Close","Open","High","Low","Volume")]; r=np.log(c).diff(); z=pd.DataFrame(index=df.index)
 z["r1"]=r; z["body"]=(c-o)/(o+1e-9); z["range"]=(h-l)/(c+1e-9); z["loc"]=(c-l)/(h-l+1e-9); z["upper"]=(h-np.maximum(o,c))/(c+1e-9); z["lower"]=(np.minimum(o,c)-l)/(c+1e-9); z["volchg"]=np.log1p(v).diff()
 for n in (2,3,6,12,24,48,65):
  z[f"mom{n}"]=c.pct_change(n); z[f"vol{n}"]=r.rolling(n).std(); z[f"ema{n}"]=c/c.ewm(span=n,adjust=False).mean()-1; z[f"range{n}"]=z["range"].rolling(n).mean(); z[f"vr{n}"]=v/(v.rolling(n).mean()+1e-9)
 z["ema6_24"]=c.ewm(span=6,adjust=False).mean()/c.ewm(span=24,adjust=False).mean()-1; z["ema12_48"]=c.ewm(span=12,adjust=False).mean()/c.ewm(span=48,adjust=False).mean()-1; z["ema24_65"]=c.ewm(span=24,adjust=False).mean()/c.ewm(span=65,adjust=False).mean()-1; z["vol_ratio"]=z["vol6"]/(z["vol48"]+1e-9); z["mom_ratio"]=z["mom6"]/(z["vol12"]+1e-9)
 for lag in (1,2,3,4,6,8,12,16,24,32,48): z[f"lag_r{lag}"]=r.shift(lag); z[f"lag_body{lag}"]=z["body"].shift(lag); z[f"lag_loc{lag}"]=z["loc"].shift(lag)
 hour=df.index.hour+df.index.minute/60; z["hour_sin"]=np.sin(2*np.pi*(hour-9.5)/6.5); z["hour_cos"]=np.cos(2*np.pi*(hour-9.5)/6.5); z["dow_sin"]=np.sin(2*np.pi*df.index.dayofweek/5); z["dow_cos"]=np.cos(2*np.pi*df.index.dayofweek/5)
 return z.replace([np.inf,-np.inf],np.nan)

def make_supervised(df):
 f=features(df); c=df.Close.astype(float); ret1=c.shift(-1)/c-1; ret3=c.shift(-3)/c-1; ret6=c.shift(-6)/c-1; scale=f.vol6.clip(lower=.0005); return f,ret1,ret1/scale,ret3/scale,ret6/scale,(ret1>.0015).astype(int)

def fit(X,a,b,c,d):
 common=dict(max_iter=MODEL_MAX_ITER,learning_rate=.04,max_leaf_nodes=13,min_samples_leaf=20,l2_regularization=3.5,loss="absolute_error",random_state=42)
 return (HistGradientBoostingRegressor(**common).fit(X,a),HistGradientBoostingRegressor(**common).fit(X,b),HistGradientBoostingRegressor(**common).fit(X,c),ExtraTreesClassifier(n_estimators=220,max_depth=9,min_samples_leaf=10,max_features=.60,class_weight="balanced",n_jobs=-1,random_state=59).fit(X,d))

def predictions(df,start,end):
 f,ret1,a,b,c,d=make_supervised(df); X=f.to_numpy(float); A,B,C,D=[q.to_numpy(float) for q in (a,b,c,d)]; valid=[i for i in range(LOOKBACK_BARS-1,len(f)-6) if start<=i<end and np.isfinite(A[i]) and np.isfinite(B[i]) and np.isfinite(C[i]) and np.isfinite(X[i]).all()]; out=[]; models=None; last=-10**9
 for i in valid:
  if models is None or i-last>=RETRAIN_EVERY:
   e=i; s=max(LOOKBACK_BARS-1,e-TRAIN_WINDOW); idx=np.arange(s,e); good=np.isfinite(A[idx])&np.isfinite(B[idx])&np.isfinite(C[idx])&np.isfinite(D[idx])&np.isfinite(X[idx]).all(axis=1); idx=idx[good]
   if len(idx)>=400: models=fit(X[idx],A[idx],B[idx],C[idx],D[idx]); last=i
  if models is None: continue
  m1,m3,m6,clf=models; row=X[i].reshape(1,-1); p1=float(m1.predict(row)[0]); p3=float(m3.predict(row)[0]); p6=float(m6.predict(row)[0]); prob=float(clf.predict_proba(row)[0,1]); scale=max(float(f.vol6.iloc[i]),.0005); r1=p1*scale; r3=p3/3*scale; r6=p6/6*scale; base=.62*r1+.25*r3+.13*r6; conf=np.clip((prob-.5)*2,-1,1); sig=base*(.72+.85*max(conf,0)); out.append((df.index[i],float(df.Close.iloc[i]),sig,float(f.ema12_48.iloc[i]),float(f.vol_ratio.iloc[i])))
 return pd.DataFrame(out,columns=["date","price","signal","trend","vol_ratio"]).set_index("date")

def enter(row,panel,i,mode,param):
 s=float(row.signal)
 if mode=="fixed": return s>param
 hist=panel.signal.iloc[max(0,i-param[1]):i].dropna()
 if len(hist)<max(12,param[1]//2): return False
 return s>float(hist.quantile(param[0]))

def bt(panel,start,end,mode,param,weight,hold):
 dates=[d for d in panel.index if start<=d<end]; cash=INITIAL_CASH; shares=0.; entry=-1; curve=[]; trades=[]
 for i,d in enumerate(dates):
  row=panel.loc[d]; price=float(row.price); total=cash+shares*price; signal=enter(row,panel,panel.index.get_loc(d),mode,param)
  target=weight if signal else 0.
  if shares>0 and entry>=0 and i-entry>=hold: target=0.
  tv=target*total; cur=shares*price
  if tv<cur*.98 and cur>0:
   sell=cur-tv; cash+=sell*(1-COST); shares-=sell/price
   if shares<=1e-12: shares=0.; trades.append(total); entry=-1
  elif tv>cur*1.02:
   buy=min(tv-cur,cash/(1+COST))
   if buy>total*.01:
    shares+=buy/(price*(1+COST)); cash-=buy*(1+COST)
    if entry<0: entry=i
  curve.append(cash+shares*price)
 if len(curve)<2:return 0,0,0,0,0,INITIAL_CASH
 curve=np.asarray(curve); peak=np.maximum.accumulate(curve); dd=float(np.min(curve/peak-1)); rets=curve[1:]/curve[:-1]-1; sh=float(np.mean(rets)/(np.std(rets)+1e-12)*np.sqrt(252*6.5)) if len(rets)>20 else 0.; wr=0.
 if trades:
  # trade list stores exit equity, so win rate is not reconstructed here; omit as selector input.
  wr=np.nan
 return float(curve[-1]/INITIAL_CASH-1),len(trades),dd,sh,wr,float(curve[-1])

def bh(df,s,e):
 p=df[(df.index>=s)&(df.index<e)].Close.astype(float); return float(p.iloc[-1]/p.iloc[0]-1)

def main():
 t=time.time(); df=load_data(); n=len(df); cut=int(n*.60); vc=int(n*.80); test_start=vc+2; panel=predictions(df,cut,n-1); val=panel.index[panel.index<df.index[vc]]; test=panel.index[panel.index>=df.index[test_start]]; vs,ve=val[0],val[-1]; ts,te=test[0],test[-1]; cand=[]
 # Baseline plus adaptive percentile entry. All parameters are selected only on validation.
 for mode in ("fixed","quantile"):
  params=THRESHOLD_CANDIDATES if mode=="fixed" else [(q,w) for q in QUANTILE_CANDIDATES for w in WINDOW_CANDIDATES]
  for param in params:
   for weight in (.25,.35,.50,.70,.90,1.):
    for hold in HOLD_CANDIDATES:
     r,tr,dd,sh,wr,_=bt(panel,vs,ve,mode,param,weight,hold); score=r-.32*abs(min(dd,0))+.018*max(sh,0)
     if tr<10: score-=.006*(10-tr)
     cand.append((score,mode,param,weight,hold,r,tr,dd,sh))
 best=max(cand,key=lambda x:x[0]); _,mode,param,weight,hold,vr,vt,vdd,vsh=best; fr,ft,fdd,fsh,_,final=bt(panel,ts,te,mode,param,weight,hold); bhv=bh(df,ts,te); bfc=INITIAL_CASH*(1+bhv)
 print("=== V19 NVDA 1H ENTRY REFINEMENT ==="); print(f"mode={mode} | entry_param={param} | weight={weight:.0%} | hold={hold} bars"); print(f"data={df.index[0]} -> {df.index[-1]} | total_bars={len(df)}"); print(f"validation={vs} -> {ve} | test={ts} -> {te}"); print(f"validation IA={vr:+.2%} | DD={vdd:.2%} | Sharpe={vsh:.2f} | trades={vt}"); print(f"FINAL IA={fr:+.2%} | NVDA B&H={bhv:+.2%} | trades={ft} | maxDD={fdd:.2%} | Sharpe={fsh:.2f}"); print(f"€50 -> IA €{final:.2f} | B&H €{bfc:.2f}"); print(f"IA beats NVDA B&H: {'YES' if fr>bhv else 'NO'}"); print(f"runtime={(time.time()-t)/60:.1f} min")
if __name__=="__main__": main()
