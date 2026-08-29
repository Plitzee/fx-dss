"""Vong 2 — them ho hoi quy chuyen che do (THAR/STHAR) va cay tang cuong."""
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import os as _os; R=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+"/"; sys.path.insert(0,R+"src")
import fxdata; fxdata.D=R+"data/prices"
from fxdata import load_daily
from vol import per_day_estimators
from sklearn.ensemble import HistGradientBoostingRegressor
P=["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF"]
EPS=1e-14; MINTR=500; GBM_REFIT=250
ADV=pd.read_csv(R+"data/rv_adv.csv",parse_dates=["Date"])
def ols(X,y): return np.linalg.solve(X.T@X+1e-8*np.eye(X.shape[1]),X.T@y)
def qlike(f,y):
    f=np.maximum(f,EPS); y=np.maximum(y,EPS); return y/f-np.log(y/f)-1
def roll(v,w): return pd.Series(v).rolling(w).mean().values
def build(pair):
    d=load_daily(pair); e=per_day_estimators(d)[["Date","gk","r_on","r_cc"]]
    a=ADV[ADV.pair==pair].drop(columns=["pair"])
    m=a.merge(e,on="Date",how="left").sort_values("Date").reset_index(drop=True)
    m=m[m.n5>=100].reset_index(drop=True)
    gap=m.Date.diff().dt.days.values.astype(float).copy(); gap[0]=1
    m["cont"]=gap<=4
    for c in ("rv5","bpv5","rq5","rsp","rsn","gk"): m[c]=m[c].clip(lower=EPS)
    return m
def features(m):
    rv=m.rv5.values; n=len(rv); o=np.ones(n)
    lv=np.log(rv); lw=roll(lv,5); lm=roll(lv,22)
    C=np.minimum(m.bpv5.values,rv); J=np.maximum(rv-m.bpv5.values,0.0)
    lc=np.log(np.maximum(C,EPS)); lj=np.log1p(J/np.maximum(C,EPS))
    lp=np.log(m.rsp.values); ln=np.log(m.rsn.values)
    lq=np.log(np.maximum(np.sqrt(m.rq5.values)/np.maximum(rv,EPS),EPS))
    lon=np.log(np.maximum(m.r_on.values**2,EPS))
    # bien chuyen che do: log RV ngay chuan hoa bang trung binh/do lech 250 phien TRUOC
    mu=pd.Series(lv).rolling(250).mean().shift(1).values
    sd=pd.Series(lv).rolling(250).std().shift(1).values
    z=(lv-mu)/np.maximum(sd,1e-8)
    G=1.0/(1.0+np.exp(-1.5*z))            # chuyen muot
    I=(z>0).astype(float)                  # nguong cung
    H=np.column_stack([o,lv,lw,lm])
    X={}
    X["HAR"]=H
    X["HARQ"]=np.column_stack([H,lq,lq*lv])
    X["HAR-CJ"]=np.column_stack([o,lc,roll(lc,5),roll(lc,22),lj,roll(lj,5)])
    X["SHAR"]=np.column_stack([o,lp,ln,lw,lm])
    X["THAR"]=np.column_stack([H,H*I[:,None]])
    X["STHAR"]=np.column_stack([H,H*G[:,None]])
    X["STHARQ"]=np.column_stack([H,H*G[:,None],lq,lq*lv])
    X["HAR-full"]=np.column_stack([o,lc,roll(lc,5),roll(lc,22),lj,lp,ln,lq,lq*lv,lon])
    gbm=np.column_stack([lv,lw,lm,lc,lj,lp,ln,lq,lon,z])
    return X,lv,gbm
REG=["HAR","HARQ","HAR-CJ","SHAR","THAR","STHAR","STHARQ","HAR-full"]
SIMPLE=["MA20-GK","MA5-RV5"]
EXTRA=["GBM","EN(HARQ,CJ,SHAR)","EN(STHARQ,HARQ,SHAR)","EN(tất cả HAR)"]
MODELS=SIMPLE+REG+EXTRA
t0=time.time(); OUT={}
for pair in P:
    m=build(pair); X,lv,GB=features(m); n=len(m)
    rv=m.rv5.values; gk=m.gk.values; cont=m.cont.values; y=lv[1:]
    F={k:np.full(n,np.nan) for k in MODELS}
    gbm=None; glast=-10**9; gs2=0.0
    for t in range(MINTR,n-1):
        if not cont[t+1]: continue
        F["MA20-GK"][t+1]=gk[t-19:t+1].mean(); F["MA5-RV5"][t+1]=rv[t-4:t+1].mean()
        for k in REG:
            Xk=X[k]; Xd=Xk[:t]; yy=y[:t]
            ok=np.isfinite(Xd).all(1)&np.isfinite(yy)
            if ok.sum()<300: continue
            b=ols(Xd[ok],yy[ok]); xn=Xk[t]
            if not np.isfinite(xn).all(): continue
            s2=(yy[ok]-Xd[ok]@b).var()
            F[k][t+1]=float(np.exp(np.clip(xn@b,-30,0)+0.5*s2))
        if t-glast>=GBM_REFIT:
            Xd=GB[:t]; yy=y[:t]; ok=np.isfinite(Xd).all(1)&np.isfinite(yy)
            if ok.sum()>=300:
                gbm=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,
                     max_depth=3,l2_regularization=1.0,random_state=0).fit(Xd[ok],yy[ok])
                gs2=float(((yy[ok]-gbm.predict(Xd[ok]))**2).mean()); glast=t
        if gbm is not None and np.isfinite(GB[t]).all():
            F["GBM"][t+1]=float(np.exp(np.clip(gbm.predict(GB[t:t+1])[0],-30,0)+0.5*gs2))
        for nm,ks in (("EN(HARQ,CJ,SHAR)",["HARQ","HAR-CJ","SHAR"]),
                      ("EN(STHARQ,HARQ,SHAR)",["STHARQ","HARQ","SHAR"]),
                      ("EN(tất cả HAR)",REG)):
            v=[np.log(F[k][t+1]) for k in ks if np.isfinite(F[k][t+1])]
            if len(v)==len(ks): F[nm][t+1]=float(np.exp(np.mean(v)))
    OUT[pair]=dict(F=F,rv=rv,dates=m.Date.values,n=n)
    print(f"  {pair}: {n:,} ngày  [{time.time()-t0:.0f}s]",flush=True)
np.save("volbake.npy",OUT,allow_pickle=True)

print("\n"+"="*112); print("QLIKE — ngày giao dịch đầy đủ, walk-forward"); print("="*112)
print(f"{'mô hình':<22}"+"".join(f"{p:>12}" for p in P)+f"{'TB':>9}{'hạng':>7}")
print("-"*112)
sc={}; rk={k:[] for k in MODELS}
for p in P:
    A=OUT[p]; msk=np.ones(A["n"],bool)
    for k in MODELS: msk&=np.isfinite(A["F"][k])
    A["msk"]=msk
    s={k:float(qlike(A["F"][k][msk],A["rv"][msk]).mean()) for k in MODELS}
    sc[p]=s
    for i,k in enumerate(sorted(s,key=s.get)): rk[k].append(i+1)
rows=sorted([(k,np.mean([sc[p][k] for p in P]),np.mean(rk[k])) for k in MODELS],key=lambda x:x[2])
for k,mu,r in rows:
    line=f"{k:<22}"
    for p in P:
        v=sc[p][k]; b=min(sc[p],key=sc[p].get)
        line+=f"{f'{v:.4f}'+('*' if k==b else ' '):>12}"
    print(line+f"{mu:>9.4f}{r:>7.1f}")
print(f"\nsố ngày chấm điểm/cặp: {[int(OUT[p]['msk'].sum()) for p in P]}")
print(f"[{time.time()-t0:.0f}s]")
