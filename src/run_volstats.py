import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
import os as _os; R=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+"/"; sys.path.insert(0,R+"src")
from metrics import mcs
from scipy import stats as st
P=["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF"]; EPS=1e-14
OUT=np.load("volbake.npy",allow_pickle=True).item()
MODELS=list(OUT["EURUSD"]["F"].keys())
def qlike(f,y):
    f=np.maximum(f,EPS); y=np.maximum(y,EPS); return y/f-np.log(y/f)-1
def dm(d):
    n=len(d); mb=d.mean(); L=int(np.ceil(1.5*n**(1/3))); s=np.sum((d-mb)**2)/n
    for k in range(1,L+1): s+=2*(1-k/(L+1))*np.sum((d[k:]-mb)*(d[:-k]-mb))/n
    return mb/np.sqrt(max(s,1e-16)/n)
LOSS={}
for p in P:
    A=OUT[p]; msk=np.ones(A["n"],bool)
    for k in MODELS: msk&=np.isfinite(A["F"][k])
    LOSS[p]={k:qlike(A["F"][k][msk],A["rv"][msk]) for k in MODELS}

print("="*104); print("BẢNG 1 — DIEBOLD–MARIANO so với MA20-GK (mô hình đang nuôi panel)"); print("="*104)
print(f"{'mô hình':<22}"+"".join(f"{p:>12}" for p in P)+f"{'số cặp thắng':>14}")
print("-"*104)
for k in MODELS:
    if k=="MA20-GK": continue
    line=f"{k:<22}"; win=0
    for p in P:
        t=dm(LOSS[p][k]-LOSS[p]["MA20-GK"]); pv=2*(1-st.norm.cdf(abs(t)))
        sg="***" if pv<.01 else "**" if pv<.05 else "*" if pv<.1 else ""
        if t<0 and pv<.05: win+=1
        line+=f"{f'{t:+.2f}{sg}':>12}"
    print(line+f"{win:>11}/6")
print("-"*104); print("t âm = TỐT HƠN MA20-GK.  *** p<0,01  ** p<0,05  * p<0,1")

BEST="EN(STHARQ,HARQ,SHAR)"
print("\n"+"="*104); print(f"BẢNG 2 — DIEBOLD–MARIANO so với {BEST}"); print("="*104)
print(f"{'mô hình':<22}"+"".join(f"{p:>12}" for p in P)+f"{'số cặp thua':>14}")
print("-"*104)
for k in MODELS:
    if k==BEST: continue
    line=f"{k:<22}"; lose=0
    for p in P:
        t=dm(LOSS[p][k]-LOSS[p][BEST]); pv=2*(1-st.norm.cdf(abs(t)))
        sg="***" if pv<.01 else "**" if pv<.05 else "*" if pv<.1 else ""
        if t>0 and pv<.05: lose+=1
        line+=f"{f'{t:+.2f}{sg}':>12}"
    print(line+f"{lose:>11}/6")
print("-"*104); print("t dương = TỆ HƠN ứng viên vô địch")

print("\n"+"="*104); print("BẢNG 3 — MODEL CONFIDENCE SET (α=0,10, bootstrap khối)"); print("="*104)
print(f"{'cặp':<10}{'số mô hình trong MCS':>22}   thành viên")
print("-"*104)
keep={k:0 for k in MODELS}
for p in P:
    Lm=np.column_stack([LOSS[p][k] for k in MODELS])
    alive,_=mcs(Lm,alpha=0.10,B=500,block=20,seed=1)
    names=[MODELS[i] for i in alive]
    for nm in names:
        if nm in keep: keep[nm]+=1
    print(f"{p:<10}{len(names):>22}   {', '.join(names[:6])}{' ...' if len(names)>6 else ''}")
print("-"*104)
print("số lần có mặt trong MCS trên 6 cặp:")
for k,v in sorted(keep.items(),key=lambda x:-x[1]): print(f"  {k:<24}{v}/6")
