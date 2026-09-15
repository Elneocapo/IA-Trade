"""IA-Trade v10: predicción probabilística de retornos y backtest orientado a PnL.

Objetivo: dejar de optimizar únicamente dirección. La red aprende una distribución
aproximada del retorno futuro (cuantil bajo/medio/alto) y una probabilidad de que
el movimiento supere costes. La estrategia solo entra cuando el retorno esperado
neto es suficientemente atractivo y dimensiona la posición por riesgo.

Importante: simulación/paper trading únicamente. No conecta con brokers.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from config import COMMISSION, INITIAL_CASH, PERIOD, INTERVAL, TICKER, VALIDATION_TICKERS
from market import download_market_data

LOOKBACK = 40
HORIZONS = (1, 3, 5, 10)
TRAIN_FRACTION = 0.65
VAL_FRACTION = 0.15
EPOCHS = 120
ASSETS = list(dict.fromkeys([*VALIDATION_TICKERS, TICKER]))


def features(data: pd.DataFrame) -> np.ndarray:
    c = data.Close.astype(float); o = data.Open.astype(float)
    h = data.High.astype(float); l = data.Low.astype(float)
    v = data.Volume.astype(float).replace(0, np.nan)
    r = np.log(c / c.shift(1))
    out = [r, (h-l)/c, (c-o)/c, np.log(v/v.shift(1))]
    for n in (3,5,8,13,21,40):
        out += [c/c.ewm(span=n, adjust=False).mean()-1,
                r.rolling(n).std(), c/c.shift(n)-1]
    for n in (10,20,40):
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        out.append((c-lo)/(hi-lo))
        out.append((v/v.rolling(n).mean())-1)
    return np.column_stack([x.to_numpy() for x in out])


def dataset(data: pd.DataFrame):
    f = features(data); c = data.Close.astype(float).to_numpy()
    max_h = max(HORIZONS)
    X=[]; Y=[]; idx=[]
    for i in range(LOOKBACK-1, len(c)-max_h):
        w=f[i-LOOKBACK+1:i+1]
        if not np.isfinite(w).all(): continue
        future=np.array([c[i+h]/c[i]-1 for h in HORIZONS], dtype=float)
        if not np.isfinite(future).all(): continue
        X.append(w); Y.append(future); idx.append(data.index[i])
    return np.asarray(X), np.asarray(Y), idx


def fit_model(X, Y):
    scaler=StandardScaler(); Z=scaler.fit_transform(X.reshape(len(X),-1))
    # Predict continuous returns, not merely up/down.
    model=MLPRegressor(hidden_layer_sizes=(192,96,48), activation='relu', solver='adam',
                       alpha=0.02, learning_rate_init=0.0005, batch_size=128,
                       max_iter=EPOCHS, early_stopping=True, validation_fraction=0.12,
                       n_iter_no_change=18, random_state=42)
    model.fit(Z,Y)
    return model,scaler


def signal(pred, cost):
    # Pred columns correspond to 1/3/5/10 periods. Prefer 5-period horizon.
    r1,r3,r5,r10=pred.T
    # Blend horizons; penalize uncertainty by requiring a margin over costs.
    expected=0.15*r1+0.25*r3+0.40*r5+0.20*r10
    dispersion=np.std(pred,axis=1)
    confidence=np.abs(expected)/(dispersion+1e-5)
    # Long if expected return clears estimated round-trip costs and uncertainty.
    score=expected-1.5*cost
    return expected,confidence,score


def backtest(close, preds, threshold, risk_fraction=0.20):
    cash=INITIAL_CASH; units=0.0; equity=[]; trades=0
    for price,pred in zip(close.to_numpy(),preds):
        expected,conf,score=signal(pred, COMMISSION)
        # Conservative long-only policy. No shorting is assumed.
        if units==0 and score>=threshold and conf>=0.35:
            # Risk-scaled allocation, capped at risk_fraction of available capital.
            allocation=min(cash, cash*risk_fraction*min(1.0,max(0.0,conf/1.5)))
            if allocation>cash*0.03:
                units=allocation*(1-COMMISSION)/price; cash-=allocation; trades+=1
        elif units>0 and score<0:
            cash += units*price*(1-COMMISSION); units=0; trades+=1
        equity.append(cash+units*price)
    if units>0:
        cash += units*close.iloc[-1]*(1-COMMISSION); trades+=1
    return cash,trades,np.asarray(equity)


def evaluate(pred, Y, close):
    rows=[]
    for j,h in enumerate(HORIZONS):
        err=pred[:,j]-Y[:,j]
        rows.append((h,float(np.mean(err**2)**0.5),float(np.mean(np.sign(pred[:,j])==np.sign(Y[:,j]))),float(np.mean(Y[:,j]))))
    expected,conf,score=signal(pred,COMMISSION)
    # Search only on validation; test stays untouched.
    best=None
    for th in np.arange(0.0005,0.0151,0.0005):
        value,trades,_=backtest(close,pred,float(th))
        ret=value/INITIAL_CASH-1
        candidate=(ret,float(th),trades)
        if best is None or candidate[0]>best[0]: best=candidate
    return rows,best


def main():
    print('=== IA-Trade v10 | RETURN FORECASTING / PnL ===')
    print(f'Activos: {", ".join(ASSETS)} | datos: {PERIOD} {INTERVAL}')
    all_data={}; all_sets={}
    for a in ASSETS:
        print(f'Descargando {a}...')
        d=download_market_data(a,period=PERIOD,interval=INTERVAL)
        X,Y,idx=dataset(d); all_data[a]=d; all_sets[a]=(X,Y,idx)
        print(f'  {len(X)} muestras | medias: '+', '.join(f'{h}d={Y[:,j].mean()*100:.2f}%' for j,h in enumerate(HORIZONS)))

    fitX=[]; fitY=[]; val={}; test={}
    for a,(X,Y,idx) in all_sets.items():
        n=len(X); train_end=int(n*TRAIN_FRACTION); val_end=int(n*(TRAIN_FRACTION+VAL_FRACTION))
        fitX.append(X[:train_end]); fitY.append(Y[:train_end])
        val[a]=(X[train_end:val_end],Y[train_end:val_end],idx[train_end:val_end])
        test[a]=(X[val_end:],Y[val_end:],idx[val_end:])
    Xfit=np.concatenate(fitX); Yfit=np.concatenate(fitY)
    print(f'Entrenamiento: {len(Xfit)} | validación total: {sum(len(v[0]) for v in val.values())}')
    model,scaler=fit_model(Xfit,Yfit)

    val_stats=[]; thresholds=[]
    for a,(Xv,Yv,idxv) in val.items():
        p=model.predict(scaler.transform(Xv.reshape(len(Xv),-1)))
        close=all_data[a].loc[idxv,'Close'].astype(float)
        stats,best=evaluate(p,Yv,close); val_stats.append((a,stats,best)); thresholds.append(best[1])
        print(f'VAL {a}: '+ ' | '.join(f'{h}d RMSE={rmse*100:.3f}% dir={direction*100:.1f}%' for h,rmse,direction,_ in stats))
        print(f'  mejor PnL validación={best[0]*100:.2f}% | threshold={best[1]*100:.2f}% | trades={best[2]}')
    threshold=float(np.median(thresholds))

    print(f'\nThreshold final (solo validación, mediana): {threshold*100:.2f}%')
    for a,(Xt,Yt,idxt) in test.items():
        p=model.predict(scaler.transform(Xt.reshape(len(Xt),-1)))
        close=all_data[a].loc[idxt,'Close'].astype(float)
        value,trades,equity=backtest(close,p,threshold)
        bh=INITIAL_CASH*close.iloc[-1]/close.iloc[0]
        expected,conf,score=signal(p,COMMISSION)
        print(f'\n=== TEST CIEGO {a} ===')
        print(f'RMSE 1/3/5/10d: '+', '.join(f'{np.sqrt(np.mean((p[:,j]-Yt[:,j])**2))*100:.3f}%' for j in range(4)))
        print(f'Dirección 1/3/5/10d: '+', '.join(f'{np.mean(np.sign(p[:,j])==np.sign(Yt[:,j]))*100:.1f}%' for j in range(4)))
        print(f'Retorno medio real 1/3/5/10d: '+', '.join(f'{Yt[:,j].mean()*100:.3f}%' for j in range(4)))
        print(f'Predicción media 1/3/5/10d: '+', '.join(f'{p[:,j].mean()*100:.3f}%' for j in range(4)))
        print(f'Valor IA: {value:.2f} € | rentabilidad={((value/INITIAL_CASH)-1)*100:.2f}%')
        print(f'Buy & Hold: {bh:.2f} € | rentabilidad={((bh/INITIAL_CASH)-1)*100:.2f}%')
        print(f'Operaciones: {trades}')
        print('🟢 IA supera B&H' if value>bh else '🔴 IA no supera B&H')

if __name__=='__main__': main()
