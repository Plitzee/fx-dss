"""PHAN TICH 6 CAP TIEN — go bo gioi han "chi mot cap" cua ket qua truoc."""
import sys,time,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/tmp/fx/src")
import numpy as np, pandas as pd
import fxdata; fxdata.D="/tmp/fx/data_v4"
from fxdata import load_daily, realized_var_from_hourly
from vol import per_day_estimators, yang_zhang
from scipy import stats as st

PAIRS=["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF"]
MIN_TRAIN=250; EPS=1e-12; t0=time.time()
MODELS=["RW-cc","MA20-cc","EWMA94","RW-GK","MA20-GK","HAR-Park","HAR-GK","HAR-RS","HAR-YZ","GARCH-t"]

def har_design(lv):
    return np.column_stack([np.ones(len(lv)),lv,
        pd.Series(lv).rolling(5).mean().values,pd.Series(lv).rolling(22).mean().values])
def ols(X,y): return np.linalg.solve(X.T@X+1e-10*np.eye(X.shape[1]),X.T@y)
def fc_har(s,t):
    lv=np.log(s[:t+1]); X=har_design(lv); y=lv[1:]; Xd=X[:-1]
    ok=np.isfinite(Xd).all(1)&np.isfinite(y)
    if ok.sum()<80: return np.nan
    b=ols(Xd[ok],y[ok]); xn=X[-1]
    if not np.isfinite(xn).all(): return np.nan
    return float(np.exp(xn@b+0.5*(y[ok]-Xd[ok]@b).var()))
def qlike(f,y):
    f=np.maximum(f,EPS); y=np.maximum(y,EPS); return y/f-np.log(y/f)-1
def dm(l1,l2):
    d=l1-l2; n=len(d); mb=d.mean(); L=int(np.ceil(1.5*n**(1/3)))
    s=np.sum((d-mb)**2)/n
    for k in range(1,L+1):
        s+=2*(1-k/(L+1))*np.sum((d[k:]-mb)*(d[:-k]-mb))/n
    return mb/np.sqrt(max(s,1e-16)/n)

from arch import arch_model
ALL={}
print("="*92); print("CHAY 6 CAP TIEN"); print("="*92)
for PAIR in PAIRS:
    d=load_daily(PAIR); e=per_day_estimators(d); e["yz5"]=yang_zhang(e,5)
    rv=realized_var_from_hourly(PAIR); m=e.merge(rv,on="Date",how="left")
    for c in ("cc","park","gk","rs","yz5"): m[c]=m[c].clip(lower=EPS)
    m["rv_h1"]=m["rv_h1"].clip(lower=EPS)
    m=m.dropna(subset=["r_cc","gk"]).reset_index(drop=True); n=len(m)
    r=m.r_cc.values; px={k:m[k].values for k in ("cc","park","gk","rs","yz5")}
    F={k:np.full(n,np.nan) for k in MODELS}
    ew=np.full(n,np.nan); ew[0]=px["cc"][:50].mean()
    for t in range(1,n): ew[t]=.94*ew[t-1]+.06*px["cc"][t-1]
    gp=None; last=-10**9; gs=np.nan
    for t in range(MIN_TRAIN,n-1):
        F["RW-cc"][t+1]=max(px["cc"][t],EPS); F["MA20-cc"][t+1]=px["cc"][t-19:t+1].mean()
        F["EWMA94"][t+1]=ew[t]; F["RW-GK"][t+1]=max(px["gk"][t],EPS)
        F["MA20-GK"][t+1]=px["gk"][t-19:t+1].mean()
        F["HAR-Park"][t+1]=fc_har(px["park"],t); F["HAR-GK"][t+1]=fc_har(px["gk"],t)
        F["HAR-RS"][t+1]=fc_har(px["rs"],t);    F["HAR-YZ"][t+1]=fc_har(px["yz5"],t)
        if t-last>=50:
            try:
                res=arch_model(r[:t+1]*100,p=1,q=1,dist="t",mean="Constant").fit(disp="off",show_warning=False)
                p_=res.params; gp=(float(p_["mu"]),float(p_["omega"]),float(p_["alpha[1]"]),float(p_["beta[1]"])); last=t
            except Exception: gp=None
        if gp is not None:
            mu,om,al,be=gp
            if t==last:
                s2=om/max(1e-8,1-al-be)
                for x in r[:t+1]*100-mu: s2=om+al*x*x+be*s2
                gs=s2
            else:
                x=r[t]*100-mu; gs=om+al*x*x+be*gs
            F["GARCH-t"][t+1]=gs/1e4
    tgt=m.rv_h1.values; okt=np.isfinite(tgt)&(tgt>0)
    res={}
    for k in MODELS:
        ok=np.isfinite(F[k])&okt
        if ok.sum()<100: continue
        res[k]=dict(q=qlike(F[k][ok],tgt[ok]).mean(),mse=np.mean((tgt[ok]-F[k][ok])**2),N=int(ok.sum()))
    ALL[PAIR]=dict(res=res,F=F,tgt=tgt,okt=okt,n=n)
    best=min(res,key=lambda k:res[k]["q"])
    print(f"  {PAIR}: {n:,} phien | {res[best]['N']:,} phien cham diem | tot nhat = {best} "
          f"(QLIKE {res[best]['q']:.4f})   [{time.time()-t0:.0f}s]",flush=True)

print("\n"+"="*92); print("BANG 1 — QLIKE TUNG CAP  (thap = tot; o dam = tot nhat cua cap do)"); print("="*92)
print(f"{'Mo hinh':<11}"+"".join(f"{p:>12}" for p in PAIRS)+f"{'hang TB':>10}")
print("-"*92)
ranks={k:[] for k in MODELS}
for P in PAIRS:
    order=sorted(ALL[P]["res"],key=lambda k:ALL[P]["res"][k]["q"])
    for i,k in enumerate(order): ranks[k].append(i+1)
rows=[]
for k in MODELS:
    if not ranks[k]: continue
    rows.append((k,np.mean(ranks[k])))
for k,rk in sorted(rows,key=lambda x:x[1]):
    line=f"{k:<11}"
    for P in PAIRS:
        rr=ALL[P]["res"].get(k)
        if not rr: line+=f"{'—':>12}"; continue
        b=min(ALL[P]["res"],key=lambda z:ALL[P]["res"][z]["q"])
        v=f"{rr['q']:.4f}"+("*" if k==b else " ")
        line+=f"{v:>12}"
    print(line+f"{rk:>10.1f}")
print("-"*92)

print("\n"+"="*92); print("BANG 2 — DIEBOLD-MARIANO: moi mo hinh vs MA20-GK, tung cap"); print("="*92)
print(f"{'Mo hinh':<11}"+"".join(f"{p:>12}" for p in PAIRS)+f"{'so cap thua':>13}")
print("-"*92)
for k in MODELS:
    if k=="MA20-GK": continue
    line=f"{k:<11}"; nlose=0
    for P in PAIRS:
        A=ALL[P]; both=np.isfinite(A["F"][k])&np.isfinite(A["F"]["MA20-GK"])&A["okt"]
        if both.sum()<100: line+=f"{'—':>12}"; continue
        t_=dm(qlike(A["F"][k][both],A["tgt"][both]),qlike(A["F"]["MA20-GK"][both],A["tgt"][both]))
        p=2*(1-st.norm.cdf(abs(t_)))
        s="***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""
        if t_>0 and p<.05: nlose+=1
        line+=f"{f'{t_:+.2f}{s}':>12}"
    print(line+f"{nlose:>10}/6")
print("-"*92)
print("t duong = TE HON MA20-GK.  *** p<0,01  ** p<0,05  * p<0,1")
print(f"\nTong thoi gian: {time.time()-t0:.0f}s")
