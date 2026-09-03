"""Them dac trung PHU THUOC DUONG DI (Guyon-Lekeufack) vao ho HAR."""
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import os as _os; R=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+"/"; sys.path.insert(0,R+"src")
import fxdata; fxdata.D=R+"data/prices"
from fxdata import load_daily
from vol import per_day_estimators
from volfc import merge_thin_days
P=["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF"]
EPS=1e-14; MINTR=500
ADV=pd.read_csv(R+"data/rv_adv.csv",parse_dates=["Date"])
def ols(X,y): return np.linalg.solve(X.T@X+1e-8*np.eye(X.shape[1]),X.T@y)
def qlike(f,y):
    f=np.maximum(f,EPS); y=np.maximum(y,EPS); return y/f-np.log(y/f)-1
def rollm(v,w): return pd.Series(v).rolling(w).mean().values
def ewma(v,lam):
    """Tong co trong so mu K(tau)=lam*exp(-lam*tau), tinh de quy."""
    a=np.exp(-lam); out=np.zeros(len(v)); s=0.0
    for i,x in enumerate(v):
        s=a*s+(1-a)*(0.0 if not np.isfinite(x) else x); out[i]=s
    return out

def build(pair):
    px=load_daily(pair)[["Date","open","high","low","close"]]
    a=ADV[ADV.pair==pair].drop(columns=["pair"])
    d=merge_thin_days(a.merge(px,on="Date",how="inner"))
    e=per_day_estimators(d)
    d=d.assign(r_cc=e.r_cc.values)
    gap=d.Date.diff().dt.days.values.astype(float).copy(); gap[0]=1
    d["cont"]=gap<=4
    return d

def feats(d):
    rv=np.maximum(d.rv5.values,EPS); n=len(rv); o=np.ones(n)
    lv=np.log(rv); lw=rollm(lv,5); lm=rollm(lv,22)
    lq=np.log(np.maximum(np.sqrt(np.maximum(d.rq5.values,EPS))/rv,EPS))
    lp=np.log(np.maximum(d.rsp.values,EPS)); ln_=np.log(np.maximum(d.rsn.values,EPS))
    mu=pd.Series(lv).rolling(250).mean().shift(1).values
    sd=pd.Series(lv).rolling(250).std().shift(1).values
    z=(lv-mu)/np.maximum(sd,1e-8); G=1/(1+np.exp(-1.5*z))
    r=np.nan_to_num(d.r_cc.values)
    # PDV: dac trung XU HUONG (loi suat co dau) va BIEN DONG, hai thang thoi gian
    R1a=ewma(r,1/3.0); R1b=ewma(r,1/25.0)          # nhanh / cham
    R2a=np.sqrt(np.maximum(ewma(r**2,1/5.0),EPS)); R2b=np.sqrt(np.maximum(ewma(r**2,1/50.0),EPS))
    # chuan hoa ve don vi "so lan do lech chuan" de he so on dinh
    s0=np.maximum(pd.Series(np.abs(r)).rolling(250).mean().shift(1).values,1e-8)
    T1a,T1b=R1a/s0,R1b/s0
    PD=np.column_stack([T1a,T1b,np.log(R2a/s0+EPS),np.log(R2b/s0+EPS)])
    H=np.column_stack([o,lv,lw,lm])
    X={}
    X["HAR"]=H
    X["STHARQ"]=np.column_stack([H,H*G[:,None],lq,lq*lv])
    X["HARQ"]=np.column_stack([H,lq,lq*lv])
    X["SHAR"]=np.column_stack([o,lp,ln_,lw,lm])
    X["HAR-PD"]=np.column_stack([H,PD])
    X["STHARQ-PD"]=np.column_stack([H,H*G[:,None],lq,lq*lv,PD])
    X["SHAR-PD"]=np.column_stack([o,lp,ln_,lw,lm,PD])
    return X,lv

REG=["HAR","HARQ","SHAR","STHARQ","HAR-PD","SHAR-PD","STHARQ-PD"]
ENS={"EN hiện tại":["STHARQ","HARQ","SHAR"],
     "EN + PDV":["STHARQ-PD","HARQ","SHAR-PD"],
     "EN toàn PDV":["STHARQ-PD","HAR-PD","SHAR-PD"]}
MODELS=REG+list(ENS)
t0=time.time(); OUT={}
for pair in P:
    d=build(pair); X,lv=feats(d); n=len(d); rv=d.rv5.values; cont=d.cont.values; y=lv[1:]
    F={k:np.full(n,np.nan) for k in MODELS}
    for t in range(MINTR,n-1):
        if not cont[t+1]: continue
        for k in REG:
            Xk=X[k]; Xd=Xk[:t]; yy=y[:t]
            ok=np.isfinite(Xd).all(1)&np.isfinite(yy)
            if ok.sum()<300: continue
            b=ols(Xd[ok],yy[ok]); xn=Xk[t]
            if not np.isfinite(xn).all(): continue
            F[k][t+1]=float(np.exp(np.clip(xn@b,-30,0)+0.5*(yy[ok]-Xd[ok]@b).var()))
        for nm,ks in ENS.items():
            v=[np.log(F[k][t+1]) for k in ks if np.isfinite(F[k][t+1])]
            if len(v)==len(ks): F[nm][t+1]=float(np.exp(np.mean(v)))
    OUT[pair]=dict(F=F,rv=rv,n=n)
    print(f"  {pair} [{time.time()-t0:.0f}s]",flush=True)
np.save("pdv.npy",OUT,allow_pickle=True)
print("\n"+"="*104); print("QLIKE — them dac trung phu thuoc duong di"); print("="*104)
print(f"{'mô hình':<16}"+"".join(f"{p:>12}" for p in P)+f"{'TB':>9}{'hạng':>7}")
print("-"*104)
sc={}; rk={k:[] for k in MODELS}
for p in P:
    A=OUT[p]; msk=np.ones(A["n"],bool)
    for k in MODELS: msk&=np.isfinite(A["F"][k])
    A["msk"]=msk
    s={k:float(qlike(A["F"][k][msk],A["rv"][msk]).mean()) for k in MODELS}
    sc[p]=s
    for i,k in enumerate(sorted(s,key=s.get)): rk[k].append(i+1)
for k,mu,r in sorted([(k,np.mean([sc[p][k] for p in P]),np.mean(rk[k])) for k in MODELS],key=lambda x:x[2]):
    line=f"{k:<16}"
    for p in P:
        v=sc[p][k]; b=min(sc[p],key=sc[p].get)
        line+=f"{f'{v:.4f}'+('*' if k==b else ' '):>12}"
    print(line+f"{mu:>9.4f}{r:>7.1f}")
