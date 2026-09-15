"""IA-Trade V16.3 - NVDA only, sequential 1H next-candle prediction.

Paper trading / research only. No broker or live orders.
At every hourly close the model uses only information already known and
predicts the NEXT 1H candle. The simulator acts only after that close.
"""
from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

ASSET = "NVDA"
PERIOD = "2y"
INTERVAL = "1h"
LOOKBACK_BARS = 65
MAX_HOLD_BARS = 13
TRAIN_WINDOW = 700
RETRAIN_EVERY = 48
MODEL_MAX_ITER = 100
COST = 0.001
INITIAL_CASH = 50.0


def load_data() -> pd.DataFrame:
    print(f"=== V16.3 | {ASSET} | {INTERVAL} | context={LOOKBACK_BARS} bars | max hold={MAX_HOLD_BARS} bars ===", flush=True)
    df = yf.download(ASSET, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False, prepost=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ASSET}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("America/New_York")
    return df.between_time("09:30", "16:00")


def features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o, v = [df[k].astype(float) for k in ("Close", "High", "Low", "Open", "Volume")]
    r = np.log(c).diff()
    z = pd.DataFrame(index=df.index)
    z["r1"] = r
    z["body"] = (c-o)/o
    z["range"] = (h-l)/c
    z["close_loc"] = (c-l)/(h-l+1e-9)
    z["upper"] = (h-np.maximum(o,c))/c
    z["lower"] = (np.minimum(o,c)-l)/c
    z["volchg"] = np.log1p(v).diff()
    for n in (3, 6, 12, 24, 48, 65):
        z[f"mom{n}"] = c.pct_change(n)
        z[f"vol{n}"] = r.rolling(n).std()
        z[f"ema{n}"] = c/c.ewm(span=n, adjust=False).mean()-1
        z[f"range{n}"] = z["range"].rolling(n).mean()
        z[f"vr{n}"] = v/(v.rolling(n).mean()+1e-9)
    z["ema12_48"] = c.ewm(span=12, adjust=False).mean()/c.ewm(span=48, adjust=False).mean()-1
    z["vol_ratio"] = z["vol6"]/(z["vol48"]+1e-9)
    hour = df.index.hour + df.index.minute/60
    z["hour_sin"] = np.sin(2*np.pi*(hour-9.5)/6.5)
    z["hour_cos"] = np.cos(2*np.pi*(hour-9.5)/6.5)
    z["dow_sin"] = np.sin(2*np.pi*df.index.dayofweek/5)
    z["dow_cos"] = np.cos(2*np.pi*df.index.dayofweek/5)
    return z.replace([np.inf,-np.inf],np.nan)


def make_supervised(df):
    f = features(df)
    c = df["Close"].astype(float)
    future = c.shift(-1)/c-1
    scale = f["vol6"].clip(lower=0.0005)
    y = future/scale
    # The model sees compact features computed from the full 65-bar context,
    # rather than 65 x N raw columns. This keeps the same information horizon
    # while greatly reducing dimensionality and overfitting.
    return f, future, y


def fit_model(X,y):
    return HistGradientBoostingRegressor(
        max_iter=MODEL_MAX_ITER,
        learning_rate=0.045,
        max_leaf_nodes=9,
        min_samples_leaf=18,
        l2_regularization=3.0,
        loss="absolute_error",
        random_state=42,
    ).fit(X,y)


def sequential_predictions(df, split_start, split_end):
    f, future_ret, y = make_supervised(df)
    valid = []
    for i in range(LOOKBACK_BARS-1, len(f)-1):
        if split_start <= i < split_end and np.isfinite(y.iloc[i]) and not f.iloc[i].isna().any():
            valid.append(i)
    if not valid:
        return pd.DataFrame()

    # Every training row is a single hourly decision. At prediction i,
    # only rows with known targets strictly before i are eligible.
    X = f.to_numpy(dtype=float)
    yy = y.to_numpy(dtype=float)
    outputs=[]; model=None; last_fit=-10**9
    total=len(valid); started=time.time()
    for count,i in enumerate(valid,1):
        if model is None or i-last_fit >= RETRAIN_EVERY:
            end=i
            start=max(LOOKBACK_BARS-1, end-TRAIN_WINDOW)
            idx=np.arange(start,end)
            good=np.isfinite(yy[idx]) & np.isfinite(X[idx]).all(axis=1)
            idx=idx[good]
            if len(idx)>=250:
                model=fit_model(X[idx],yy[idx]); last_fit=i
        if model is None:
            continue
        pred_norm=float(model.predict(X[i].reshape(1,-1))[0])
        pred_ret=pred_norm*max(float(f["vol6"].iloc[i]),0.0005)
        outputs.append((df.index[i],float(df["Close"].iloc[i]),pred_ret,pred_norm,float(f["ema12_48"].iloc[i])))
        if count==1 or count%200==0 or count==total:
            elapsed=time.time()-started; rate=count/max(elapsed,1e-9); eta=(total-count)/max(rate,1e-9)
            print(f"prediction {count}/{total} | {rate:.1f} bars/s | ETA ~{eta/60:.1f} min",flush=True)
    return pd.DataFrame(outputs,columns=["date","price","pred_return","pred_norm","trend"]).set_index("date")


def backtest(df,panel,start,end,threshold,max_weight,trend_filter):
    dates=[d for d in panel.index if start<=d<end]
    cash,shares=INITIAL_CASH,0.0; entry=None; curve=[]; trade_returns=[]; entry_value=None
    for i,d in enumerate(dates):
        row=panel.loc[d]; price=float(row.price); total=cash+shares*price
        signal=float(row.pred_return)>threshold
        if trend_filter and float(row.trend)<=0: signal=False
        target=max_weight if signal else 0.0
        if shares>0 and entry is not None and i-entry>=MAX_HOLD_BARS: target=0.0
        target_value=target*total; current=shares*price
        if target_value < current*0.98 and current>0:
            sell=current-target_value; cash += sell*(1-COST); shares -= sell/price
            if shares<=1e-12:
                shares=0.0
                if entry_value is not None: trade_returns.append(total/entry_value-1)
                entry=None; entry_value=None
        elif target_value > current*1.02:
            buy=min(target_value-current,cash/(1+COST))
            if buy>total*0.01:
                before=total; shares += buy/(price*(1+COST)); cash -= buy*(1+COST)
                if entry is None: entry=i; entry_value=before
        curve.append(cash+shares*price)
    if len(curve)<2: return 0,0,0,0,0,INITIAL_CASH
    curve=np.asarray(curve); peak=np.maximum.accumulate(curve); dd=float(np.min(curve/peak-1))
    rets=curve[1:]/curve[:-1]-1
    sh=float(np.mean(rets)/(np.std(rets)+1e-12)*np.sqrt(252*6.5)) if len(rets)>20 else 0
    wr=float(np.mean(np.asarray(trade_returns)>0)) if trade_returns else 0
    final=float(curve[-1])
    return final/INITIAL_CASH-1,len(trade_returns),dd,sh,wr,final


def buy_and_hold(df,start,end):
    p=df[(df.index>=start)&(df.index<end)]["Close"].astype(float)
    return float(p.iloc[-1]/p.iloc[0]-1) if len(p)>1 else 0


def duration_text(start,end):
    days=(pd.Timestamp(end)-pd.Timestamp(start)).total_seconds()/86400
    return f"{days:.0f} days (~{days/365.25:.2f} years)"


def main():
    overall=time.time(); df=load_data(); n=len(df)
    train_cut=int(n*0.60); val_cut=int(n*0.80); test_start=val_cut+2
    print(f"Preparing sequential walk-forward: {n:,} hourly bars...",flush=True)
    panel=sequential_predictions(df,train_cut,n-1)
    if panel.empty: raise RuntimeError("Could not generate sequential predictions")
    val=panel.index[panel.index<df.index[val_cut]]; test=panel.index[panel.index>=df.index[test_start]]
    if len(val)<100 or len(test)<100: raise RuntimeError("Not enough validation/test predictions")
    vs,ve=val[0],val[-1]; ts,te=test[0],test[-1]

    candidates=[]
    # Threshold is deliberately selected only on validation. Trend filtering is
    # also selected only there, then frozen for the untouched final test.
    for threshold in (0.001,0.0015,0.002,0.0025,0.003,0.004,0.005,0.006):
        for weight in (0.25,0.35,0.50,0.70):
            for tf in (False,True):
                r,tr,dd,sh,wr,_=backtest(df,panel,vs,ve,threshold,weight,tf)
                # Favor return, but penalize deep drawdowns and unstable Sharpe.
                score=r-0.50*abs(min(dd,0))+0.02*max(sh,0)
                candidates.append((score,r,threshold,weight,tf,tr,dd,sh,wr))
    best=max(candidates,key=lambda x:x[0])
    _,vr,threshold,weight,tf,vt,vdd,vsh,vwr=best
    fr,ft,fdd,fsh,fwr,final_cash=backtest(df,panel,ts,te,threshold,weight,tf)
    bh=buy_and_hold(df,ts,te); bh_cash=INITIAL_CASH*(1+bh); elapsed=time.time()-overall

    print("\n=== V16.3 NVDA SEQUENTIAL 1H SUMMARY ===")
    print(f"asset={ASSET} | candles={INTERVAL} | history={PERIOD}")
    print(f"context={LOOKBACK_BARS} bars (~10 trading days) | prediction=NEXT 1H candle | max_hold={MAX_HOLD_BARS} bars (~2 trading days)")
    print(f"data={df.index[0]} -> {df.index[-1]} | total_bars={len(df)}")
    print(f"validation={vs} -> {ve} | test={ts} -> {te}")
    print(f"test duration={duration_text(ts,te)}")
    print(f"selected threshold={threshold:.2%} | max_weight={weight:.0%} | trend_filter={tf}")
    print(f"validation IA={vr:+.2%} | DD={vdd:.2%} | Sharpe={vsh:.2f} | trades={vt} | win_rate={vwr:.1%}")
    print(f"FINAL IA={fr:+.2%} | {ASSET} B&H={bh:+.2%} | trades={ft} | maxDD={fdd:.2%} | Sharpe={fsh:.2f} | win_rate={fwr:.1%}")
    print(f"€{INITIAL_CASH:.2f} -> IA €{final_cash:.2f} | profit/loss={final_cash-INITIAL_CASH:+.2f}€")
    print(f"€{INITIAL_CASH:.2f} -> B&H €{bh_cash:.2f} | profit/loss={bh_cash-INITIAL_CASH:+.2f}€")
    print(f"IA beats {ASSET} B&H: {'YES' if fr>bh else 'NO'}")
    print(f"runtime={elapsed/60:.1f} min")

if __name__=="__main__": main()
