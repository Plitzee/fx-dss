"""Giai doan du lieu that: nhat quan d1<->h1, ty trong gap cuoi tuan, mua vu trong ngay."""
import numpy as np, pandas as pd, sys
sys.path.insert(0,"/tmp/fx/src")
from fxdata import load_daily, load_hourly, realized_var_from_hourly, PAIRS
from vol import per_day_estimators, yang_zhang

pd.set_option("display.width", 160)

print("="*96)
print("A. NHAT QUAN GIUA THANH NGAY (d1) VA THANH GIO (h1)")
print("="*96)
print(f"{'Cap':<9}{'ngay khop':>11}{'|dC| max':>11}{'H khop':>9}{'L khop':>9}{'H(h1)/H(d1)-1 TB':>19}")
print("-"*96)
for p in ["EURUSD","USDJPY","GBPUSD","EURCHF"]:
    d = load_daily(p, patch_eurchf=False); h = load_hourly(p)
    h["day"] = h.Date.dt.normalize()
    agg = h.groupby("day").agg(o=("open","first"), hi=("high","max"),
                               lo=("low","min"), c=("close","last")).reset_index()
    j = d.merge(agg, left_on="Date", right_on="day")
    dC = (j.close - j.c).abs().max()
    hok = np.isclose(j.high, j.hi, atol=1e-6).mean()
    lok = np.isclose(j.low,  j.lo, atol=1e-6).mean()
    rel = (j.hi/j.high - 1).mean()
    print(f"{p:<9}{len(j):>11}{dC:>11.2e}{hok:>9.1%}{lok:>9.1%}{rel:>19.2e}")
print("-"*96)
print("=> Thanh ngay va thanh gio cua cung mot feed khop nhau -> RV tu h1 va OHLC ngay cung mot phien.")

print("\n" + "="*96)
print("B. GAP QUA DEM CHIEM BAO NHIEU PHAN TRAM PHUONG SAI? (quyet dinh YZ co dang dung khong)")
print("="*96)
print(f"{'Cap':<9}{'Var(r_cc)':>12}{'Var(r_on)':>12}{'Var(r_oc)':>12}{'gap %':>9}"
      f"{'gap T2 %':>11}{'Var gap T2':>12}{'Var gap khac':>13}")
print("-"*96)
rows=[]
for p in PAIRS:
    d = load_daily(p); e = per_day_estimators(d)
    e["dow"] = pd.to_datetime(e.Date).dt.dayofweek
    v_cc, v_on, v_oc = e.r_cc.var(), e.r_on.var(), e.r_oc.var()
    mon = e[e.dow==0]; oth = e[e.dow!=0]
    print(f"{p:<9}{v_cc:>12.2e}{v_on:>12.2e}{v_oc:>12.2e}{100*v_on/v_cc:>8.1f}%"
          f"{100*mon.r_on.var()/v_cc:>10.1f}%{mon.r_on.var():>12.2e}{oth.r_on.var():>13.2e}")
    rows.append((p, 100*v_on/v_cc))
print("-"*96)
mean_share = np.mean([r[1] for r in rows])
print(f"Ty trong phuong sai gap trung binh: {mean_share:.1f}%  "
      f"(so sanh: co phieu thuong 20-60% vi nghi 17,5 tieng moi dem)")
print("=> FX chay 24x5 nen gap gan nhu chi ton tai o ranh gioi cuoi tuan.")

print("\n" + "="*96)
print("C. MUA VU TRONG NGAY — 'cai dong ho la mot bien du bao rui ro'")
print("="*96)
h = load_hourly("EURUSD")
h["r"] = np.log(h.close).diff()
h["day"] = h.Date.dt.normalize(); h["hour"] = h.Date.dt.hour
h.loc[h.day != h.day.shift(), "r"] = np.nan
prof = h.dropna(subset=["r"]).groupby("hour").r.agg(["std","count"])
prof["rel"] = prof["std"]/prof["std"].mean()
peak = prof.rel.idxmax(); trough = prof.rel.idxmin()
print("Do lech chuan loi suat theo gio server (chuan hoa ve trung binh = 1,00):")
for i in range(0, 24, 6):
    seg = prof.loc[i:i+5]
    print("   " + "  ".join(f"{hh:02d}h {r:.2f}" for hh, r in seg.rel.items()))
print(f"\nGio bien dong nhat: {peak:02d}h server ({prof.rel[peak]:.2f}x trung binh)")
print(f"Gio yen nhat      : {trough:02d}h server ({prof.rel[trough]:.2f}x trung binh)")
print(f"Ty le dinh/day    : {prof.rel[peak]/prof.rel[trough]:.1f} lan")
print("Server o UTC+2/+3 -> 00h server = 17:00 New York = ranh gioi phien FX chuan.")
print(f"=> {peak:02d}h server ~ {(peak-2)%24:02d}h UTC ~ vung chong lan London-New York.")

print("\n" + "="*96)
print("D. HIEU QUA THUC TE CUA CAC UOC LUONG TREN DU LIEU THAT")
print("="*96)
print("Thuoc do 'su that' = RV tu 24 thanh gio. Do bang tuong quan va R^2 voi log RV.")
print(f"\n{'Cap':<9}{'corr(cc,RV)':>13}{'corr(park,RV)':>15}{'corr(gk,RV)':>13}"
      f"{'corr(rs,RV)':>13}{'corr(yz5,RV)':>14}")
print("-"*96)
for p in PAIRS[:8]:
    d = load_daily(p); e = per_day_estimators(d)
    e["yz5"] = yang_zhang(e, 5)
    rv = realized_var_from_hourly(p)
    m = e.merge(rv, on="Date").query("n_bars >= 20").copy()
    lrv = np.log(m.rv_h1.clip(lower=1e-12))
    out=[]
    for k in ("cc","park","gk","rs","yz5"):
        x = np.log(m[k].clip(lower=1e-12))
        ok = np.isfinite(x) & np.isfinite(lrv)
        out.append(np.corrcoef(x[ok], lrv[ok])[0,1])
    print(f"{p:<9}" + "".join(f"{v:>13.3f}" if i<4 else f"{v:>14.3f}" for i,v in enumerate(out)))
print("-"*96)
print("cc = binh phuong loi suat ngay. Tuong quan thap cua no chinh la diem cua")
print("Andersen & Bollerslev 1998: thuoc do nhieu, khong phai mo hinh te.")
