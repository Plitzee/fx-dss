import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sizing import f_kelly, f_ruin_cap
from sizing2 import simulate2, Fuzzy
P=["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF"]
PANEL=os.environ.get("FX_PANEL","panel2_6pairs.csv")
pan=pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data",PANEL))
print(f"panel: {PANEL}  ({len(pan):,} dòng)"); MU=0.0002
C={}
def build(p):
    if p not in C:
        d=pan[pan.pair==p].reset_index(drop=True); n=int(len(d)*0.70)
        C[p]=(d.iloc[n:].reset_index(drop=True), Fuzzy(d.sig.values[:n]))
    return C[p]
def K(st): return f_kelly(st["mu"],st["sig"])
def k_dd(dd): return np.clip(1.30-3.2*dd,0.5,1.30)
def k_vol(sig,F):
    lo,me,hi=F.mu_vol(sig); w=lo+me+hi
    return np.where(w>1e-9,(lo*1.30+me*1.10+hi*0.90)/np.maximum(w,1e-9),1.0)

M={
 "Trần trơn":            lambda st,F,b: np.minimum(K(st), f_ruin_cap(st["sig"],250,b,st["nu"])),
 "Trần × 1,08 (PPO)":    lambda st,F,b: np.minimum(K(st), 1.08*f_ruin_cap(st["sig"],250,b,st["nu"])),
 "Trần × tích 2 hệ số":  lambda st,F,b: np.minimum(K(st), k_dd(st["dd"])*k_vol(st["sig"],F)
                                                   *f_ruin_cap(st["sig"],250,b,st["nu"])),
 "Trần × fuzzy Mamdani": lambda st,F,b: np.minimum(K(st), F.k(st["sig"],st["dd"])
                                                   *f_ruin_cap(st["sig"],250,b,st["nu"])),
}
BUD=(0.003,0.01,0.03,0.08,0.20)
print("="*92)
print("PHEP THU QUYET DINH — fuzzy co hon mot TICH DON GIAN cua hai he so khong?")
print("Quet ngan sach rui ro, lay 3 seed. So sanh o cung muc pha san moi cong bang.")
print("="*92)
pts={}
for name,fn in M.items():
    pts[name]=[]
    print(f"\n{name}")
    print(f"  {'ngân sách':<12}{'tăng trưởng':>13}{'phá sản':>11}")
    for b in BUD:
        G=[];R=[]
        for sd in (1,2,3):
            g=[];r=[]
            for pair in P:
                te,F=build(pair)
                o=simulate2(te,lambda st,fn=fn,F=F,b=b: fn(st,F,b),n_paths=4000,
                            horizon=250,mu_true=MU,mu_believed=MU,seed=sd)
                g.append(o["mean_log_growth"]); r.append(o["p_ruin"])
            G.append(np.median(g)); R.append(max(r))
        pts[name].append((np.mean(R),np.mean(G)))
        print(f"  b={b:<10.3f}{np.mean(G):>12.2%}{np.mean(R):>10.2%}")

print("\n"+"="*92)
print("TANG TRUONG TAI CUNG MUC PHA SAN (nội suy trên biên)")
print("="*92)
T=[0.001,0.003,0.01,0.03]
print(f"{'Phương pháp':<24}"+"".join(f"{f'{t:.1%}':>14}" for t in T))
print("-"*92)
tab={}
for name,pl in pts.items():
    pl=sorted(pl); xs=[p[0] for p in pl]; ys=[p[1] for p in pl]
    row=[np.interp(t,xs,ys) if min(xs)-1e-9<=t<=max(xs)+1e-9 else np.nan for t in T]
    tab[name]=row
    print(f"{name:<24}"+"".join(f"{v:>13.2%}" if v==v else f"{'—':>14}" for v in row))
print("-"*92)
base=tab["Trần × tích 2 hệ số"]; fz=tab["Trần × fuzzy Mamdani"]
d=[f-b for f,b in zip(fz,base) if f==f and b==b]
print(f"\nFuzzy so voi tich don gian: {np.mean(d):+.2%} tang truong trung binh")
print("-> "+("FUZZY DANG DUNG" if np.mean(d)>0.005 else
      "FUZZY KHONG DANG DUNG — mot tich hai he so tuyen tinh cho ket qua tuong duong"))
