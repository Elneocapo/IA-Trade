"""IA-Trade v15 - NVIDIA-only, short-horizon paper-trading research.

The model learns ONLY from ASSET and the strategy can ONLY trade that asset.
To create another asset-specific AI later, change ASSET (e.g. "AAPL").
Every trade has a hard maximum holding horizon of two trading days.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
ASSET="NVDA"; PERIOD="10y"; MAX_HOLD_DAYS=2; COST=0.001; INITIAL_CASH=50.0

def features(df):
 c,h,l,o,v=[df[k].astype(float) for k in ("Close","High","Low","Open","Volume")]; z=pd.DataFrame(index=df.index); lr=np.log(c).diff()
 z["ret1"]=lr; z["range"]=(h-l)/c; z["body"]=(c-o)/o; z["upper_wick"]=(h-np.maximum(o,c))/c; z["lower_wick"]=(np.minimum(o,c)-l)/c; z["close_location"]=(c-l)/(h-l+1e-9); z["volchg"]=np.log1p(v).diff()
 for n in (2,3,4,5,8,13,21,34,55): z[f"mom{n}"]=c.pct_change(n); z[f"ema{n}"]=c/c.ewm(span=n,adjust=False).mean()-1; z[f"vol{n}"]=lr.rolling(n).std()
 for n in (5,10,20,50): lo,hi=l.rolling(n).min(),h.rolling(n).max(); z[f"range_pos{n}"]=(c-lo)/(hi-lo+1e-9); z[f"volume_ratio{n}"]=v/(v.rolling(n).mean()+1e-9)
 z["vol_ratio_short_long"]=z["vol5"]/(z["vol34"]+1e-9); z["ema_fast_slow"]=c.ewm(span=13,adjust=False).mean()/c.ewm(span=55,adjust=False).mean()-1
 return z.replace([np.inf,-np.inf],np.nan)

def load_data():
 print(f"=== TRAINING ASSET: {ASSET} | MAX TRADE: {MAX_HOLD_DAYS} DAYS ===",flush=True); df=yf.download(ASSET,period=PERIOD,interval="1d",auto_adjust=True,progress=False)
 if df.empty: raise RuntimeError(f"No data returned for {ASSET}")
 if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
 return df

def walk_forward(df):
 x=features(df); c=df["Close"].astype(float); vol=x["vol5"]; f1=c.shift(-1)/c-1; f2=c.shift(-2)/c-1; y1=f1/(vol+1e-6); y2=f2/(vol*np.sqrt(2)+1e-6); d=x.assign(y1=y1,y2=y2).dropna(); X=d[x.columns].values; Y=d[["y1","y2"]].values; n=len(d); warm=max(500,int(n*.5)); step=max(20,int(n*.025)); out=[]
 for end in range(warm,n,step):
  stop=min(end+step,n); preds=[]
  for j in range(2):
   m=HistGradientBoostingRegressor(max_iter=800,learning_rate=.025,max_leaf_nodes=31,min_samples_leaf=12,l2_regularization=1.0,loss="absolute_error",random_state=42+j); m.fit(X[:end],Y[:end,j]); preds.append(m.predict(X[end:stop]))
  for dt,a,b in zip(d.index[end:stop],preds[0],preds[1]): out.append((dt,float(a),float(b),float(c.loc[dt]),max(float(vol.loc[dt]),.003)))
 return pd.DataFrame(out,columns=["date","score1","score2","price","vol"]).set_index("date")

def backtest(panel,start,end,threshold,invest):
 dates=[d for d in panel.index if start<=d<end]; cash=INITIAL_CASH; shares=0.; entry_i=None; curve=[]; trades=0
 for i,d in enumerate(dates):
  row=panel.loc[d]; price=float(row.price); total=cash+shares*price; score=.6*float(row.score1)+.4*float(row.score2); target_weight=invest if score>threshold else 0.
  if shares>0 and entry_i is not None and i-entry_i>=MAX_HOLD_DAYS: target_weight=0.
  target_value=target_weight*total; current_value=shares*price
  if target_value<current_value*.98 and current_value>0:
   sell_value=current_value-target_value; cash+=sell_value*(1-COST); shares-=sell_value/price; trades+=1
   if shares<=1e-12: shares=0.; entry_i=None
  if target_value>current_value*1.02:
   buy_value=min(target_value-current_value,cash/(1+COST))
   if buy_value>total*.01: shares+=buy_value/(price*(1+COST)); cash-=buy_value*(1+COST); trades+=1; entry_i=i if entry_i is None else entry_i
  curve.append(cash+shares*price)
 if not curve:return 0.,0,0.,0.
 curve=np.asarray(curve); peak=np.maximum.accumulate(curve); dd=float(np.min(curve/peak-1)); daily=curve[1:]/curve[:-1]-1; sharpe=float(np.mean(daily)/(np.std(daily)+1e-12)*np.sqrt(252)) if len(daily)>10 else 0.
 return float(curve[-1]/INITIAL_CASH-1),trades,dd,sharpe

def buy_and_hold(panel,start,end):
 p=panel[(panel.index>=start)&(panel.index<end)].price; return float(p.iloc[-1]/p.iloc[0]-1) if len(p)>1 else 0.

def main():
 df=load_data(); panel=walk_forward(df); dates=sorted(panel.index)
 if len(dates)<100: raise RuntimeError("Not enough out-of-sample observations")
 cut=dates[len(dates)//2]; end=dates[-1]; candidates=[]
 for threshold in (.1,.2,.3,.4,.5,.6,.75,1.):
  for invest in (.5,.7,.9,1.):
   r,t,dd,s=backtest(panel,dates[0],cut,threshold,invest); candidates.append((r-.3*abs(min(dd,0))+.02*max(s,0),r,threshold,invest,dd,s,t))
 _,vr,threshold,invest,vdd,vsh,_=max(candidates,key=lambda x:x[0]); fr,trades,fdd,fsh=backtest(panel,cut,end,threshold,invest); bh=buy_and_hold(panel,cut,end)
 print("\n=== V15 NVDA SHORT-HORIZON SUMMARY ==="); print(f"asset={ASSET} | data={PERIOD} | max_hold={MAX_HOLD_DAYS}d | targets=1d+2d"); print(f"selected threshold={threshold:.2f} | invest={invest:.0%}"); print(f"validation IA={vr:+.2%} | DD={vdd:.2%} | Sharpe={vsh:.2f}"); print(f"FINAL IA={fr:+.2%} | {ASSET} B&H={bh:+.2%} | trades={trades} | maxDD={fdd:.2%} | Sharpe={fsh:.2f}"); print(f"IA beats {ASSET} B&H: {'YES' if fr>bh else 'NO'}")

if __name__=="__main__": main()
