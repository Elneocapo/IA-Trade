"""IA-Trade v4: entrenamiento largo y test final completamente separado."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import COMMISSION, INITIAL_CASH, INTRADAY_HORIZON_BARS, INTRADAY_INTERVAL, INTRADAY_PERIOD, TICKER
from market import download_market_data

LOOKBACK_BARS = 32
MIN_MOVE_RETURN = 0.003
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TRAIN_EPOCHS = 750

ENTRY_UP_PROBABILITY = 0.55
EXIT_DOWN_PROBABILITY = 0.50
MIN_PROBABILITY_EDGE = 0.08


def build_sequences(data):
    close = data["Close"].astype(float)
    open_price = data["Open"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float).replace(0, np.nan)
    log_return = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    features = [log_return, (high-low)/close, (close-open_price)/close,
                (high-np.maximum(open_price,close))/close,
                (np.minimum(open_price,close)-low)/close,
                np.log(volume/volume.shift(1)).replace([np.inf,-np.inf],np.nan)]
    ema8 = close.ewm(span=8,adjust=False).mean(); ema21 = close.ewm(span=21,adjust=False).mean()
    features += [close/ema8-1, close/ema21-1, ema8/ema21-1]
    features += [close/close.shift(4)-1, close/close.shift(8)-1]
    features += [log_return.rolling(n,min_periods=n).std() for n in (8,16,32)]
    features += [volume/volume.rolling(16,min_periods=8).mean()-1]
    for n in (16,32):
        rh=high.rolling(n,min_periods=n).max(); rl=low.rolling(n,min_periods=n).min()
        features.append((close-rl)/(rh-rl))
    frame=np.column_stack([f.to_numpy() for f in features])
    future_return=(close.shift(-INTRADAY_HORIZON_BARS)/close-1).to_numpy()
    volatility=features[13].to_numpy()
    sequences=[]; targets=[]; indices=[]
    for end in range(LOOKBACK_BARS-1,len(data)-INTRADAY_HORIZON_BARS):
        window=frame[end-LOOKBACK_BARS+1:end+1]; ret=future_return[end]; vol=volatility[end]
        if not np.isfinite(window).all() or not np.isfinite(ret) or not np.isfinite(vol): continue
        threshold=max(MIN_MOVE_RETURN,float(vol))
        label=2 if ret>=threshold else (0 if ret<=-threshold else 1)
        sequences.append(window); targets.append(label); indices.append(data.index[end])
    return np.asarray(sequences),np.asarray(targets),indices


def train_network(X_train,y_train):
    X_flat=X_train.reshape(len(X_train),-1)
    split=int(len(X_flat)*(1-VALIDATION_FRACTION))
    X_fit,X_val=X_flat[:split],X_flat[split:]; y_fit,y_val=y_train[:split],y_train[split:]
    scaler=StandardScaler(); X_fit_scaled=scaler.fit_transform(X_fit); X_val_scaled=scaler.transform(X_val)
    model=MLPClassifier(hidden_layer_sizes=(128,64),activation="relu",solver="adam",alpha=0.02,
                        batch_size=64,learning_rate_init=0.0005,max_iter=1,warm_start=True,
                        shuffle=True,random_state=42)
    losses=[]; val_scores=[]; best_score=-np.inf; best_coefs=None; best_intercepts=None; best_epoch=0
    print("Entrenando EXACTAMENTE 750 épocas. El TEST FINAL permanece completamente oculto.")
    for epoch in range(1,TRAIN_EPOCHS+1):
        model.fit(X_fit_scaled,y_fit)
        loss=model.loss_; pred_val=model.predict(X_val_scaled); score=balanced_accuracy_score(y_val,pred_val)
        losses.append(loss); val_scores.append(score)
        if score>best_score:
            best_score=score; best_coefs=[c.copy() for c in model.coefs_]; best_intercepts=[b.copy() for b in model.intercepts_]; best_epoch=epoch
        if epoch==1 or epoch%25==0:
            print(f"Época {epoch:03d} | pérdida={loss:.5f} | validación={score:.2%}")
    model.coefs_=best_coefs; model.intercepts_=best_intercepts; model.loss_curve_=losses; model.n_iter_=len(losses); model.loss_=losses[-1]
    print(f"Entrenamiento completo. Mejor validación: {best_score:.2%} en época {best_epoch}.")
    return model,scaler,losses,val_scores,best_score


def simulate_strategy(close,probabilities):
    cash=INITIAL_CASH; position=0.0; trades=0; equity=[]
    for price,probs in zip(close.to_numpy(),probabilities):
        p_down,p_neutral,p_up=probs; edge=p_up-p_down
        if position==0 and p_up>=ENTRY_UP_PROBABILITY and edge>=MIN_PROBABILITY_EDGE:
            position=cash*(1-COMMISSION)/price; cash=0.0; trades+=1
        elif position>0 and (p_down>=EXIT_DOWN_PROBABILITY or edge<=0):
            cash=position*price*(1-COMMISSION); position=0.0; trades+=1
        equity.append(cash if position==0 else position*price)
    if position>0: cash=position*close.iloc[-1]*(1-COMMISSION); trades+=1
    return cash,trades,equity


def main():
    print("=== IA-Trade | ENTRENAMIENTO LARGO -> EXAMEN FINAL ===")
    print(f"Activo: {TICKER}"); print(f"Datos: {INTRADAY_PERIOD} | {INTRADAY_INTERVAL}")
    print(f"Patrón: últimas {LOOKBACK_BARS} velas | horizonte: {INTRADAY_HORIZON_BARS} velas")
    print("Plan: 70% histórico para entrenar + 30% final totalmente oculto")
    print("Entrenamiento: 750 épocas completas (sin parada temprana)")
    print("El test final NO participa en el entrenamiento ni en la selección de época."); print("Modo: SIMULACIÓN / sin broker\n")
    data=download_market_data(TICKER,period=INTRADAY_PERIOD,interval=INTRADAY_INTERVAL)
    X,y,indices=build_sequences(data); split=int(len(X)*TRAIN_FRACTION)
    X_train,y_train=X[:split],y[:split]; X_test,y_test=X[split:],y[split:]; test_indices=indices[split:]
    print(f"Secuencias totales:       {len(X)}"); print(f"Entrenamiento:            {len(X_train)}"); print(f"TEST FINAL CIEGO:         {len(X_test)}")
    print(f"Clases train (↓/neutro/↑): {(y_train==0).sum()} / {(y_train==1).sum()} / {(y_train==2).sum()}")
    print(f"Clases test  (↓/neutro/↑): {(y_test==0).sum()} / {(y_test==1).sum()} / {(y_test==2).sum()}\n")
    model,scaler,losses,val_scores,best_val=train_network(X_train,y_train)
    X_test_scaled=scaler.transform(X_test.reshape(len(X_test),-1)); raw=model.predict_proba(X_test_scaled)
    probabilities=np.zeros((len(X_test),3))
    for col,cls in enumerate(model.classes_): probabilities[:,int(cls)]=raw[:,col]
    predictions=np.argmax(probabilities,axis=1); test_score=balanced_accuracy_score(y_test,predictions)*100
    close=data.loc[test_indices,"Close"].astype(float); final_value,trades,equity=simulate_strategy(close,probabilities)
    buy_hold=INITIAL_CASH*(close.iloc[-1]/close.iloc[0])
    print("\n=== EXAMEN FINAL (DATOS NUNCA VISTOS) ===")
    print(f"Validación durante entrenamiento: {best_val:.2%}"); print(f"Balanced accuracy TEST:            {test_score:.2f}%")
    print(f"Valor final IA:                    {final_value:.2f} €"); print(f"Rentabilidad IA:                   {(final_value/INITIAL_CASH-1)*100:.2f} %")
    print(f"Buy & Hold final:                  {buy_hold:.2f} €"); print(f"Operaciones:                       {trades}")
    print(f"Operaciones/día aprox.:            {trades/max(len(close)/26,1):.2f}")
    if final_value>buy_hold and test_score>50: print("🟢 Hay una señal interesante: ha generalizado en este periodo nunca visto.")
    elif test_score>50: print("🟡 La red detecta algo en el test, pero la estrategia todavía no supera Buy & Hold.")
    else: print("🔴 En el examen final no demuestra una ventaja estable.")
    print("\n--- ENTRENAMIENTO ---"); print(f"Épocas realizadas: {len(losses)}")
    plt.figure(figsize=(9,5)); plt.plot(losses,label="Pérdida entrenamiento"); plt.plot(val_scores,label="Validación temporal")
    plt.xlabel("Época"); plt.ylabel("Valor"); plt.title("Entrenamiento completo y validación temporal"); plt.legend(); plt.grid(True,alpha=0.25); plt.tight_layout(); plt.show()

if __name__=="__main__": main()
