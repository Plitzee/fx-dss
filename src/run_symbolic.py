"""Them dac trung KY HIEU cua HuyH vao tang 2 — co cai thien QLIKE khong?"""
import os, sys, time
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import os as _os; R=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+"/"; sys.path.insert(0,R+"src")
import fxdata; fxdata.D=R+"data/prices"
from fxdata import load_daily
from vol import per_day_estimators
from volfc import merge_thin_days
from split import doan
from metrics import mcs
from scipy import stats as st
P=["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF"]
EPS=1e-14; MINTR=500
ADV=pd.read_csv(R+"data/rv_adv.csv",parse_dates=["Date"])
def ols(X,y): return np.linalg.solve(X.T@X+1e-8*np.eye(X.shape[1]),X.T@y)
def qlike(f,y):
    f=np.maximum(f,EPS); y=np.maximum(y,EPS); return y/f-np.log(y/f)-1
def rm(v,w): return pd.Series(v).rolling(w).mean().values
def build(pair):
    px=load_daily(pair)[["Date","open","high","low","close"]]
    a=ADV[ADV.pair==pair].drop(columns=["pair"])
    d=merge_thin_days(a.merge(px,on="Date",how="inner"))
    gap=d.Date.diff().dt.days.values.astype(float).copy(); gap[0]=1
    d["cont"]=gap<=4
    return d
def feats(d):
    rv=np.maximum(d.rv5.values,EPS); n=len(rv); o=np.ones(n)
    lv=np.log(rv); lw=rm(lv,5); lm=rm(lv,22)
    lq=np.log(np.maximum(np.sqrt(np.maximum(d.rq5.values,EPS))/rv,EPS))
    lp=np.log(np.maximum(d.rsp.values,EPS)); ln_=np.log(np.maximum(d.rsn.values,EPS))
    mu=pd.Series(lv).rolling(250).mean().shift(1).values
    sd=pd.Series(lv).rolling(250).std().shift(1).values
    z=(lv-mu)/np.maximum(sd,1e-8); G=1/(1+np.exp(-1.5*z))
    H=np.column_stack([o,lv,lw,lm])
    STHARQ=np.column_stack([H,H*G[:,None],lq,lq*lv])
    # ── dac trung KY HIEU: trang thai bien dong 3 muc, moc tu doan HUAN LUYEN
    g=doan(d.Date.values)
    q=np.quantile(rv[g==0],[1/3,2/3]); s=np.digitize(rv,q)      # 0=LOW 1=MED 2=HIGH
    def oh(x):
        M=np.zeros((n,3)); M[np.arange(n),x]=1; return M[:,1:]   # bo LOW lam nen
    S1=oh(s)
    S2=np.zeros((n,4)); S3=np.zeros((n,3))
    for t in range(2,n):
        S2[t,0]=(s[t-1]==2)&(s[t]==2)          # HIGH,HIGH
        S2[t,1]=(s[t-1]==0)&(s[t]==0)          # LOW,LOW
        S2[t,2]=(s[t-1]<s[t])                   # dang tang cap
        S2[t,3]=(s[t-1]>s[t])                   # dang giam cap
    for t in range(3,n):
        tri=(s[t-2],s[t-1],s[t])
        S3[t,0]=tri==(1,2,2)                    # MEDIUM,HIGH,HIGH
        S3[t,1]=tri==(0,1,0)                    # LOW,MEDIUM,LOW
        S3[t,2]=tri==(2,2,1)                    # HIGH,HIGH,MEDIUM
    return {"STHARQ":STHARQ,
            "STHARQ+trạng thái":np.column_stack([STHARQ,S1]),
            "STHARQ+cặp trạng thái":np.column_stack([STHARQ,S1,S2]),
            "STHARQ+3 mẫu HuyH":np.column_stack([STHARQ,S3]),
            "STHARQ+tất cả ký hiệu":np.column_stack([STHARQ,S1,S2,S3])}, lv, g
M=["STHARQ","STHARQ+trạng thái","STHARQ+cặp trạng thái","STHARQ+3 mẫu HuyH","STHARQ+tất cả ký hiệu"]
t0=time.time(); OUT={}
for pair in P:
    d=build(pair); X,lv,g=feats(d); n=len(d); rv=d.rv5.values; cont=d.cont.values; y=lv[1:]
    F={k:np.full(n,np.nan) for k in M}
    for t in range(MINTR,n-1):
        if not cont[t+1]: continue
        for k in M:
            Xk=X[k]; Xd=Xk[:t]; yy=y[:t]
            ok=np.isfinite(Xd).all(1)&np.isfinite(yy)
            if ok.sum()<300: continue
            b=ols(Xd[ok],yy[ok]); xn=Xk[t]
            if not np.isfinite(xn).all(): continue
            F[k][t+1]=float(np.exp(np.clip(xn@b,-30,0)+0.5*(yy[ok]-Xd[ok]@b).var()))
    OUT[pair]=dict(F=F,rv=rv,g=g,n=n)
    print(f"  {pair} [{time.time()-t0:.0f}s]",flush=True)

def diem(seg):
    sc={}
    for p in P:
        A=OUT[p]; m=(A["g"]==seg)
        for k in M: m=m&np.isfinite(A["F"][k])
        for k in M: sc.setdefault(k,[]).append(float(qlike(A["F"][k][m],A["rv"][m]).mean()))
    return {k:float(np.mean(v)) for k,v in sc.items()}
dv,dt=diem(1),diem(2)
print("\n"+"="*100)
print("DAC TRUNG KY HIEU CO CAI THIEN TANG 2 KHONG?")
print("="*100)
print(f"{'mô hình':<26}{'QLIKE kiểm định':>18}{'QLIKE kiểm tra':>18}{'so với gốc':>14}")
print("-"*100)
for k in M:
    print(f"{k:<26}{dv[k]:>18.4f}{dt[k]:>18.4f}{1-dt[k]/dt['STHARQ']:>13.2%}")
print("-"*100)
best=min(dv,key=dv.get); print(f"Kiểm định chọn: {best}")
def dm(x):
    n=len(x); mb=x.mean(); L=int(np.ceil(1.5*n**(1/3))); s=np.sum((x-mb)**2)/n
    for k in range(1,L+1): s+=2*(1-k/(L+1))*np.sum((x[k:]-mb)*(x[:-k]-mb))/n
    return mb/np.sqrt(max(s,1e-16)/n)
print("\nDiebold-Mariano trên đoạn kiểm tra, so với STHARQ gốc:")
print(f"{'mô hình':<26}"+"".join(f"{p:>12}" for p in P)+f"{'thắng':>8}")
LOSS={}
for p in P:
    A=OUT[p]; m=(A["g"]==2)
    for k in M: m=m&np.isfinite(A["F"][k])
    LOSS[p]={k:qlike(A["F"][k][m],A["rv"][m]) for k in M}
for k in M:
    if k=="STHARQ": continue
    line=f"{k:<26}"; w=0
    for p in P:
        t=dm(LOSS[p][k]-LOSS[p]["STHARQ"]); pv=2*(1-st.norm.cdf(abs(t)))
        sg="***" if pv<.01 else "**" if pv<.05 else "*" if pv<.1 else ""
        if t<0 and pv<.05: w+=1
        line+=f"{f'{t:+.2f}{sg}':>12}"
    print(line+f"{w:>5}/6")
print("t âm = tốt hơn STHARQ gốc")
keep={k:0 for k in M}
for p in P:
    n=min(len(LOSS[p][k]) for k in M)
    L=np.column_stack([LOSS[p][k][:n] for k in M])
    alive,_=mcs(L,alpha=0.10,B=400,block=20,seed=1)
    for i in alive: keep[M[i]]+=1
print("\nModel Confidence Set trên kiểm tra:")
for k,v in sorted(keep.items(),key=lambda x:-x[1]): print(f"  {k:<28}{v}/6")
