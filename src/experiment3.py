"""
THI NGHIEM 3 — RUI RO DUONG DI vs RUI RO CUOI KY.
Y tuong then chot: voi chan troi mot ngay, ca hai deu QUAN SAT DUOC tu OHLC.
  - vi pham cuoi ky : Close <= rao   (cai ma VaR do)
  - cham rao        : Low   <= rao   (cai that su xay ra voi lenh cat lo)
Nen ta co the do truc tiep khoang cach giua hai thu, va kiem dinh nguyen ly phan xa.
"""
import numpy as np, pandas as pd, sys, json
sys.path.insert(0,"/tmp/fx/src")
from scipy import stats
from fxdata import load_daily, PAIRS

VOL = "HAR-RV"

def panel(pair):
    fc = pd.read_csv(f"/tmp/fx/fc_{pair}.csv", parse_dates=["Date"])
    d  = load_daily(pair)[["Date","open","high","low","close","is_patched"]]
    m  = fc.merge(d, on="Date").dropna(subset=[VOL]).reset_index(drop=True)
    m  = m[~m.is_patched.astype(bool)].copy()
    m["sig"] = np.sqrt(m[VOL].clip(lower=1e-12))     # do lech chuan du bao trong phien
    m["zT"]  = np.log(m.close/m.open)/m.sig          # loi suat cuoi ky chuan hoa
    m["zL"]  = np.log(m.low  /m.open)/m.sig          # cuc tieu trong phien chuan hoa
    m["zH"]  = np.log(m.high /m.open)/m.sig
    return m.dropna(subset=["zT","zL","zH"])

ALL = pd.concat([panel(p).assign(pair=p) for p in PAIRS], ignore_index=True)
n = len(ALL)
print("="*104)
print(f"A. TAN SUAT QUAN SAT DUOC — {n:,} phien, 12 cong cu, chan troi 1 ngay")
print("="*104)
print("Rao dat cach gia mo cua b lan do lech chuan DU BAO cho phien do.")
print(f"\n{'rao b (sigma)':>14}{'P(Close vuot)':>17}{'P(Low cham)':>15}{'ty le cham/cuoi':>19}"
      f"{'so lan cham':>14}{'ly thuyet BM':>15}")
print("-"*104)
rows=[]
for b in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
    pT = float((ALL.zT <= -b).mean())
    pL = float((ALL.zL <= -b).mean())
    ratio = pL/pT if pT>0 else np.nan
    rows.append(dict(b=b, pT=pT, pL=pL, ratio=ratio, nL=int((ALL.zL<=-b).sum())))
    print(f"{b:>14.1f}{pT:>17.4%}{pL:>15.4%}{ratio:>19.2f}{int((ALL.zL<=-b).sum()):>14}{2.00:>15.2f}")
print("-"*104)
print("Ly thuyet: voi chuyen dong Brown khong drift, nguyen ly phan xa cho")
print("P(min <= -b) = 2 P(X_T <= -b) — dung ty le 2, voi MOI b.")

print("\n" + "="*104)
print("B. NEU CHI DUNG VaR THI SAI BAO NHIEU?")
print("="*104)
print("Cau hoi thuc te: nha dau tu dat cat lo o muc ma VaR noi la 'chi 5% kha nang cham'.")
print("Thuc te bao nhieu phan tram bi cham?\n")
print(f"{'VaR noi':>12}{'rao tuong ung (sigma)':>24}{'thuc te bi cham':>19}{'sai so':>12}")
print("-"*104)
for q in (0.10, 0.05, 0.025, 0.01):
    b = -np.quantile(ALL.zT, q)                 # rao sao cho P(cuoi ky vuot) = q
    hit = float((ALL.zL <= -b).mean())
    print(f"{q:>11.1%}{b:>24.2f}{hit:>19.2%}{hit/q:>11.1f}x")
print("-"*104)

print("\n" + "="*104)
print("C. TY LE CHAM/CUOI KY THEO TUNG CAP (rao = 2 sigma)")
print("="*104)
print(f"{'Cap':<9}{'n':>7}{'P(cuoi ky)':>13}{'P(cham)':>11}{'ty le':>9}"
      f"{'  |  bien do TB / |loi suat| TB':>34}")
print("-"*104)
for p in PAIRS:
    s = ALL[ALL.pair==p]
    pT = (s.zT<=-2).mean(); pL = (s.zL<=-2).mean()
    rng = (s.zH-s.zL).mean(); ab = s.zT.abs().mean()
    print(f"{p:<9}{len(s):>7}{pT:>13.2%}{pL:>11.2%}{pL/pT if pT>0 else np.nan:>9.2f}{rng/ab:>34.2f}")
print("-"*104)

print("\n" + "="*104)
print("D. TAI SAO TY LE KHAC 2 — phan tich")
print("="*104)
print(f"Trung binh z_T          = {ALL.zT.mean():+.4f}  (ky vong 0 neu khong drift)")
print(f"Do lech chuan z_T       = {ALL.zT.std():.4f}  (ky vong 1 neu sigma du bao dung)")
print(f"Trung binh z_L          = {ALL.zL.mean():+.4f}")
print(f"Trung binh z_H          = {ALL.zH.mean():+.4f}")
print(f"Trung binh bien do z    = {(ALL.zH-ALL.zL).mean():.4f}"
      f"  (ky vong ~{np.sqrt(8/np.pi):.4f} cho BM chuan)")
print(f"Do lech (kurtosis) z_T  = {stats.kurtosis(ALL.zT):.3f}")
print("\nBa nguon lech so voi ly thuyet Brown:")
print("  1. Do lech chuan z_T khac 1 -> sigma du bao co chech he thong")
print("  2. Bien do quan sat < ly thuyet -> gia High/Low tu feed khong phai cuc tri that")
print("  3. Duoi day -> ty le cham/cuoi ky giam dan khi b tang (xem bang A)")

json.dump({"grid": rows,
           "sd_zT": float(ALL.zT.std()), "mean_zT": float(ALL.zT.mean()),
           "mean_range": float((ALL.zH-ALL.zL).mean()), "n": n},
          open("/tmp/fx/exp3_a.json","w"), indent=1)
ALL[["Date","pair","sig","zT","zL","zH"]].to_csv("/tmp/fx/exp3_panel.csv", index=False)
print("\nDa luu exp3_a.json va exp3_panel.csv")
