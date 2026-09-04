"""TANG 6 — phieu chi hieu chuan cho MOT phien. Nguoi dung giu lenh nhieu ngay.
Cac con so rui ro con dung o tam han 5/10/20 phien khong?"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import os as _os; R=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+"/"; sys.path.insert(0,R+"src")
import fxdata; fxdata.D=R+"data/prices"
from fxdata import load_daily
from scipy import stats
P=["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF"]
pan=pd.read_csv(R+"data/panel2_6pairs.csv",parse_dates=["Date"])
HS=(1,5,10,20)

def paths(pair):
    d=pan[pan.pair==pair].reset_index(drop=True)
    px=load_daily(pair)[["Date","low","close"]]
    d=d.merge(px,on="Date",how="left")
    c=d.close.values; lo=d.low.values; s=d.sig.values
    n=len(d); out={}
    for h in HS:
        cum=np.full(n,np.nan); mn=np.full(n,np.nan)
        for t in range(n-h):
            base=c[t]
            path=np.log(c[t+1:t+1+h]/base)
            plow=np.log(lo[t+1:t+1+h]/base)
            cum[t]=path[-1]; mn[t]=np.minimum.accumulate(np.minimum(path,plow))[-1]
        out[h]=(cum,mn)
    return d,out

DATA={p:paths(p) for p in P}
print("="*100)
print("A — QUY TAC √h CO DUNG KHONG? (sigma_h duoc gia dinh = sigma_1 * căn h)")
print("="*100)
print(f"{'tầm hạn':<10}{'độ lệch chuẩn thực / (sig·√h)':>32}{'tỷ số ở vol thấp':>20}{'ở vol cao':>14}")
print("-"*100)
for h in HS:
    rr=[];lo_=[];hi_=[]
    for p in P:
        d,o=DATA[p]; cum,_=o[h]; s=d.sig.values
        m=np.isfinite(cum)
        z=cum[m]/(s[m]*np.sqrt(h))
        rr.append(np.std(z))
        q=np.quantile(s[m],[1/3,2/3]); g=np.digitize(s[m],q)
        lo_.append(np.std(z[g==0])); hi_.append(np.std(z[g==2]))
    print(f"{h:<10}{np.mean(rr):>32.3f}{np.mean(lo_):>20.3f}{np.mean(hi_):>14.3f}")
print("-"*100)
print("Bằng 1,00 nghĩa là √h đúng. >1 = hệ thống ĐÁNH GIÁ THẤP rủi ro ở tầm hạn đó.")

print("\n"+"="*100)
print("B — XAC SUAT CHAM STOP KHI GIU LENH NHIEU PHIEN (stop dat CO DINH tai k·sigma)")
print("="*100)
print("Day la cau hoi that cua nguoi dung: phieu ghi 'P(cham stop) 5,7% trong 1 phien'")
print("nhung neu toi giu 10 phien thi bao nhieu?\n")
print(f"{'tầm hạn':<10}{'ngưỡng':<10}{'dự báo':>12}{'thực tế':>12}{'lệch':>10}{'n':>10}")
print("-"*100)
tot={h:[] for h in HS}
for h in HS:
    for k in (1.0,2.0,3.0):
        pr=[];re=[];N=0
        for p in P:
            d,o=DATA[p]; cum,mn=o[h]; s=d.sig.values
            m=np.isfinite(mn); n=int(len(d)*.7)
            zt=(d.zT.values*d.sig.values/s)[:n]
            nu,_,sc=stats.t.fit(zt[np.isfinite(zt)],floc=0); nu=float(np.clip(nu,2.5,40))
            pred=min(1.0,2.0*stats.t.cdf(-k/(sc*np.sqrt(h)),nu))
            hit=(mn[m]<=-k*s[m])
            pr.append(pred); re.append(hit.mean()); N+=m.sum()
        d_=np.mean(pr)-np.mean(re); tot[h].append(abs(d_))
        print(f"{h:<10}{k:<10.1f}{np.mean(pr):>11.1%}{np.mean(re):>12.1%}{d_:>+10.1%}{N:>10,}")
print("-"*100)
print("Lệch tuyệt đối trung bình theo tầm hạn: "+"  ".join(f"h={h}: {np.mean(v):.1%}" for h,v in tot.items()))

print("\n"+"="*100)
print("C — DO PHU KHOANG DU BAO O TAM HAN DAI (conformal, muc 90%)")
print("="*100)
print(f"{'tầm hạn':<10}{'√h + conformal 1 phiên':>26}{'conformal trực tiếp tầm h':>28}{'bề rộng tỷ lệ':>16}")
print("-"*100)
for h in HS:
    a=[];b=[];w=[]
    for p in P:
        d,o=DATA[p]; cum,_=o[h]; s=d.sig.values; n=int(len(d)*.7)
        m=np.isfinite(cum)
        z1=(d.zT.values*d.sig.values/s)                  # chuan hoa 1 phien
        zh=cum/(s*np.sqrt(h))                            # chuan hoa h phien
        tr1=z1[:n][np.isfinite(z1[:n])]; trh=zh[:n][np.isfinite(zh[:n])]
        te=np.arange(len(d))>=n
        h1=np.quantile(np.abs(tr1),.90*(1+1/len(tr1)))
        hh=np.quantile(np.abs(trh),.90*(1+1/len(trh)))
        sel=te&m
        a.append(float(np.mean(np.abs(zh[sel])<=h1)))
        b.append(float(np.mean(np.abs(zh[sel])<=hh)))
        w.append(hh/h1)
    print(f"{h:<10}{np.mean(a):>26.1%}{np.mean(b):>28.1%}{np.mean(w):>16.2f}")
print("-"*100)
print("Cột 2 là cách hệ thống đang làm nếu người dùng nhân √h. Cột 3 là hiệu chuẩn")
print("trực tiếp trên lợi suất tích lũy h phiên.")
