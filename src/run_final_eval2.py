"""CHAM DIEM CUOI — phan tang 4 va tang 6, dung luat huan luyen/kiem dinh/kiem tra."""
import os, sys
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
D=os.path.join(os.path.dirname(HERE),"data")
from split import doan
from scipy import stats as st
from decision_record import KhoangConformal, KhoangACI, TamHan, p_cham_stop
from position_sizing import PositionSizer, k_danh_muc
from sizing import f_ruin_cap
P=["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF"]
pan=pd.read_csv(os.path.join(D,"panel2_6pairs.csv"),parse_dates=["Date"])
LEV=0.90; A=1-LEV

def khoang_chay(d,g,method,seg):
    """Tra ve (z tren doan, nua be rong tuong ung, sig tren doan)."""
    s=d.sig.values; z=d.zT.values; tr=g==0
    idx=np.where(g==seg)[0]
    if method.startswith("ACI"):
        nb=2 if method.endswith("2") else 3
        kc=KhoangACI(z[tr],s[tr],n_bins=nb) if method!="ACI" else KhoangACI(z[tr],s[tr],n_bins=1)
        H=np.empty(len(idx))
        # nuoi bo nho bang moi ngay TRUOC doan cham diem, roi cham tien dan
        for i in np.where(~tr & (np.arange(len(d))<idx[0]))[0]:
            kc.quan_sat(z[i],s[i])
        for j,i in enumerate(idx):
            H[j]=kc.nua_be_rong(None,s[i]); kc.quan_sat(z[i],s[i])
        return z[idx],H,s[idx]
    nb={"tĩnh":1,"Mondrian 2":2,"Mondrian 3":3}[method]
    kc=KhoangConformal(z[tr],s[tr],n_bins=nb)
    H=np.array([kc.nua_be_rong(LEV,x) for x in s[idx]])
    return z[idx],H,s[idx]

METHODS=["tĩnh","Mondrian 2","Mondrian 3","ACI","ACI-tầng 2","ACI-tầng 3"]

def danh_gia(seg):
    out={}
    for m in METHODS:
        C=[];CV=[[],[]];SC=[];DD=[[],[]]
        for p in P:
            d=pan[pan.pair==p].reset_index(drop=True); g=doan(d.Date.values)
            z,H,s=khoang_chay(d,g,m,seg)
            ok=np.abs(z)<=H
            C.append(ok.mean())
            q=np.quantile(d.sig.values[g==0],[1/3,2/3]); gg=np.digitize(s,q)
            if (gg==0).sum()>20: CV[0].append(ok[gg==0].mean())
            if (gg==2).sum()>20: CV[1].append(ok[gg==2].mean())
            r=d.zT.values*d.sig.values
            cum=pd.Series(r).rolling(20).sum().values[g==seg]
            lo=cum<0
            if (~lo).sum()>20: DD[0].append(ok[~lo].mean())
            if lo.sum()>20: DD[1].append(ok[lo].mean())
            w=2*H*s; SC.append(np.mean(w+(2/A)*np.maximum(np.abs(z*s)-H*s,0)*2)*1e4)
        c=np.mean(C); v0=np.mean(CV[0]); v2=np.mean(CV[1])
        pk=np.mean(DD[0]); ll=np.mean(DD[1])
        out[m]=dict(chung=c,vol_thap=v0,vol_cao=v2,dinh=pk,lo=ll,
                    lech=max(abs(x-LEV) for x in (c,v0,v2,pk,ll)),diem=np.mean(SC))
    return out

dv,dt=danh_gia(1),danh_gia(2)
print("="*104)
print("BANG 4 — CHON CACH DUNG KHOANG TREN DOAN KIEM DINH")
print("="*104)
print(f"{'phương pháp':<14}{'phủ chung':>11}{'vol thấp':>10}{'vol cao':>9}{'ở đỉnh':>9}{'đang lỗ':>10}"
      f"{'|lệch| max':>12}{'điểm khoảng':>13}")
print("-"*104)
for m in sorted(dv,key=lambda k:dv[k]["lech"]):
    r=dv[m]
    print(f"{m:<14}{r['chung']:>11.1%}{r['vol_thap']:>10.1%}{r['vol_cao']:>9.1%}"
          f"{r['dinh']:>9.1%}{r['lo']:>10.1%}{r['lech']:>12.1%}{r['diem']:>13.1f}")
best=min(dv,key=lambda k:dv[k]["lech"])
print("-"*104); print(f"Chọn trên kiểm định: {best}")

print("\n"+"="*104)
print("BANG 5 — CHAM TREN DOAN KIEM TRA (chi cham, khong chon)")
print("="*104)
print(f"{'phương pháp':<14}{'phủ chung':>11}{'vol thấp':>10}{'vol cao':>9}{'ở đỉnh':>9}{'đang lỗ':>10}"
      f"{'|lệch| max':>12}{'điểm khoảng':>13}")
print("-"*104)
for m in sorted(dt,key=lambda k:dt[k]["lech"]):
    r=dt[m]; mark=" ←chọn" if m==best else ""
    print(f"{m:<14}{r['chung']:>11.1%}{r['vol_thap']:>10.1%}{r['vol_cao']:>9.1%}"
          f"{r['dinh']:>9.1%}{r['lo']:>10.1%}{r['lech']:>12.1%}{r['diem']:>13.1f}{mark}")
tb=sorted(dt,key=lambda k:dt[k]["lech"])[0]
print("-"*104)
print(f"  Cách được chọn trên kiểm tra: |lệch| max {dt[best]['lech']:.1%}")
print(f"  Tốt nhất có thể trên kiểm tra: {dt[tb]['lech']:.1%}  ({tb})")
print(f"  Giá phải trả cho việc chọn   : {dt[best]['lech']-dt[tb]['lech']:+.1%}")

print("\n"+"="*104)
print("BANG 6 — XAC SUAT CHAM STOP TREN DOAN KIEM TRA (tham so uoc luong tren huan luyen)")
print("="*104)
print(f"{'ngưỡng':<10}{'dự báo':>12}{'thực tế':>12}{'lệch':>10}{'n':>10}")
print("-"*104)
er=[]
for k in (0.5,1.0,1.5,2.0,2.5,3.0):
    pr=[];re=[];N=0
    for p in P:
        d=pan[pan.pair==p].reset_index(drop=True); g=doan(d.Date.values)
        pred=p_cham_stop(k,d.zT.values[g==0])
        hit=(d.zL.values[g==2]<=-k)
        pr.append(pred); re.append(hit.mean()); N+=hit.size
    e=np.mean(pr)-np.mean(re); er.append(abs(e))
    print(f"{k:<10.1f}{np.mean(pr):>11.1%}{np.mean(re):>12.1%}{e:>+10.1%}{N:>10,}")
print("-"*104); print(f"Lệch tuyệt đối trung bình trên kiểm tra: {np.mean(er):.2%}")

print("\n"+"="*104)
print("BANG 7 — BANG TAM HAN TREN DOAN KIEM TRA (he so uoc luong tren huan luyen)")
print("="*104)
print(f"{'tầm hạn':<10}{'dự báo (stop 2σ)':>20}{'thực tế':>12}{'lệch':>10}{'n':>10}")
print("-"*104)
import fxdata; fxdata.D=os.path.join(D,"prices")
from fxdata import load_daily
CACHE={}
for p in P:
    d=pan[pan.pair==p].reset_index(drop=True); g=doan(d.Date.values)
    px=load_daily(p)[["Date","low","close"]]; d2=d.merge(px,on="Date",how="left")
    r=d.zT.values*d.sig.values
    nu,_,sc=st.t.fit(d.zT.values[g==0],floc=0); nu=float(np.clip(nu,2.5,40))
    CACHE[p]=(d,g,d2.close.values,d2.low.values,d.sig.values,
              TamHan(d.sig.values[g==0],r[g==0]),nu,sc)
for h in (1,5,10,20):
    pr=[];re=[];N=0
    for p in P:
        d,g,c,lo,s,th,nu,sc=CACHE[p]
        idx=np.where(g==2)[0]; idx=idx[idx<len(d)-h]
        sh=np.array([th.sig_h(float(s[t]),h) for t in idx])
        pv=np.minimum(1.0,2.0*st.t.cdf(-2.0*s[idx]/np.maximum(sh,1e-12)/sc,nu))
        hit=np.empty(len(idx),bool)
        for j_,t in enumerate(idx):
            b=c[t]; pa=np.log(c[t+1:t+1+h]/b); pl=np.log(lo[t+1:t+1+h]/b)
            hit[j_]=np.minimum.accumulate(np.minimum(pa,pl))[-1]<=-2.0*s[t]
        pr.append(pv.mean()); re.append(hit.mean()); N+=len(idx)
    e=np.mean(pr)-np.mean(re)
    print(f"{h:<10}{np.mean(pr):>19.1%}{np.mean(re):>12.1%}{e:>+10.1%}{N:>10,}")

print("\n"+"="*104)
print("BANG 8 — HE SO DANH MUC KIEM CHUNG TREN DOAN KIEM TRA")
print("="*104)
SGN={"EURUSD":+1,"GBPUSD":+1,"AUDUSD":+1,"USDJPY":-1,"USDCAD":-1,"USDCHF":-1}
piv=pan.pivot_table(index="Date",columns="pair",values="zT")
sgp=pan.pivot_table(index="Date",columns="pair",values="sig")
r=(piv*sgp).dropna(); sgv=sgp.loc[r.index].values
gg=doan(r.index.values); al=r.values*np.array([SGN[p] for p in r.columns])
def run(k,hair,seg=2,seed=7,n_paths=6000,H=250):
    X=al[gg==seg]; S=sgv[gg==seg]; N=len(X)
    rng=np.random.default_rng(seed); blk=5; nb=int(np.ceil(H/blk))
    s0=rng.integers(0,N-blk,size=(n_paths,nb))
    idx=(s0[:,:,None]+np.arange(blk)[None,None,:]).reshape(n_paths,-1)[:,:H]
    eq=np.ones(n_paths); ruin=np.zeros(n_paths,bool)
    for t in range(H):
        tot=np.zeros(n_paths)
        for j in range(k):
            f=np.clip(hair*f_ruin_cap(S[idx[:,t],j],H,0.01,6.0),0,30.0)
            tot=tot+f*X[idx[:,t],j]
        eq=np.where(ruin,eq,eq*(1+tot)); eq=np.maximum(eq,1e-6)
        ruin|=(~ruin)&(eq<0.5)
    return float(ruin.mean())
print(f"{'số cặp':<10}{'k_danh_mục':>13}{'phá sản khi áp':>18}{'nếu không áp':>16}")
print("-"*104)
for k in (1,2,3,6):
    h=k_danh_muc(k)
    print(f"{k:<10}{h:>13.2f}{run(k,h):>18.2%}{run(k,1.0):>16.2%}")
print("-"*104); print("Ngân sách 1,00%.")
