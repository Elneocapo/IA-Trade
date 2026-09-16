import numpy as np
import pandas as pd

def plot_options(df,trace,start,end):
    x=trace[(trace.index>=start)&(trace.index<end)].copy()
    if x.empty: print("Grafico opciones: sin operaciones en el intervalo seleccionado."); return
    fig,axes=plt.subplots(3,1,figsize=(15,11),sharex=True,gridspec_kw={"height_ratios":[2.2,2.0,1.4]})
    ax=axes[0]; ax.plot(x.index,x["nvda"],label="NVDA"); active=x["strike"].notna(); ax.plot(x.index[active],x.loc[active,"strike"],label="CALL strike",linewidth=1.2); ax.set_title("V31 Options View | NVDA + CALL strike"); ax.legend(loc="upper left"); ax.grid(alpha=0.2); buys=x["event"].astype(str).str.startswith("BUY CALL"); sells=x["event"].astype(str).str.startswith("SELL CALL"); ax.scatter(x.index[buys],x.loc[buys,"nvda"],marker="^",s=65,label="CALL entry"); ax.scatter(x.index[sells],x.loc[sells,"nvda"],marker="v",s=65,label="CALL exit"); ax.legend(loc="upper left")
    ax=axes[1]; ax.plot(x.index,x["option_mid"]*CONTRACT_MULTIPLIER,label="CALL mid x 100"); ax.plot(x.index,x["option_value"],label="Position value"); ax.set_title(f"Synthetic CALL | DTE={OPTION_DTE} | target delta={OPTION_DELTA:.2f} | IVx={OPTION_IV_MULT:.2f} | spread={OPTION_SPREAD:.1%}"); ax.legend(loc="upper left"); ax.grid(alpha=0.2)
    ax=axes[2]; ax.plot(x.index,x["equity"],label="Option equity"); ax.plot(x.index,np.full(len(x),INITIAL_CASH),label="Start €500",linestyle="--"); ax.set_title("Synthetic options portfolio equity"); ax.legend(loc="upper left"); ax.grid(alpha=0.2)
    fig.tight_layout(); plt.show(block=True)
