"""Kiem dinh chat luong du lieu FX truoc khi lam bat cu dieu gi khac."""
import pandas as pd, numpy as np, glob, os, json

D = "/tmp/fx/data"
PAIRS = ["EURUSD","USDJPY","GBPUSD","USDCHF","AUDUSD","USDCAD",
         "EURCHF","EURGBP","EURJPY","GBPJPY","AUDJPY","XAUUSD"]

# Muc gia hop ly de suy ra thang chia (du lieu duoc luu nhan len)
PLAUSIBLE = {"EURUSD":1.1,"USDJPY":110,"GBPUSD":1.3,"USDCHF":0.98,"AUDUSD":0.75,
             "USDCAD":1.3,"EURCHF":1.1,"EURGBP":0.85,"EURJPY":125,"GBPJPY":145,
             "AUDJPY":80,"XAUUSD":1500}

def load(pair, tf):
    f = f"{D}/{pair}_{tf}.csv"
    df = pd.read_csv(f, parse_dates=["Date"])
    med = df["close"].median()
    target = PLAUSIBLE[pair]
    # thang chia phai la luy thua cua 10
    k = round(np.log10(med/target))
    scale = 10.0**k
    for c in ("open","high","low","close"):
        df[c] = df[c]/scale
    df = df.sort_values("Date").reset_index(drop=True)
    return df, scale

rows = []
print("="*104)
print("A. TONG QUAN DU LIEU NGAY (d1)")
print("="*104)
print(f"{'Cap':<8}{'n':>6}{'Tu':>13}{'Den':>13}{'Thang chia':>12}{'Gia cuoi':>11}"
      f"{'Trung vi':>10}{'OHLC loi':>10}{'Trung lap':>10}")
print("-"*104)
data = {}
for p in PAIRS:
    df, sc = load(p, "d1")
    bad = ((df.high < df[["open","close"]].max(axis=1)-1e-12) |
           (df.low  > df[["open","close"]].min(axis=1)+1e-12) |
           (df.high < df.low)).sum()
    dup = df.Date.duplicated().sum()
    data[p] = df
    print(f"{p:<8}{len(df):>6}{str(df.Date.iloc[0].date()):>13}{str(df.Date.iloc[-1].date()):>13}"
          f"{sc:>12.0f}{df.close.iloc[-1]:>11.4f}{df.close.median():>10.4f}{bad:>10}{dup:>10}")
    rows.append(dict(pair=p, n=len(df), start=str(df.Date.iloc[0].date()),
                     end=str(df.Date.iloc[-1].date()), scale=sc, ohlc_bad=int(bad), dup=int(dup)))
print("-"*104)

print("\n" + "="*104)
print("B. KHOANG TRONG LICH (so ngay giua hai thanh lien tiep)")
print("="*104)
print(f"{'Cap':<8}{'gap=1':>9}{'gap=2':>9}{'gap=3 (cuoi tuan)':>19}{'gap 4-7':>9}{'gap>7':>8}{'gap max':>9}")
print("-"*104)
for p in PAIRS:
    g = data[p].Date.diff().dt.days.dropna().astype(int)
    vc = g.value_counts()
    print(f"{p:<8}{vc.get(1,0):>9}{vc.get(2,0):>9}{vc.get(3,0):>19}"
          f"{g[(g>=4)&(g<=7)].size:>9}{g[g>7].size:>8}{g.max():>9}")
print("-"*104)
print("Ky vong: gap=1 trong tuan, gap=3 qua cuoi tuan (T6->T2). gap=2 va gap>3 la ngay nghi le.")

print("\n" + "="*104)
print("C. NGAY SU KIEN — kiem tra du lieu co bat duoc cu soc khong")
print("="*104)
EV = [("EURCHF","2015-01-15","SNB bo san 1,20"),
      ("USDCHF","2015-01-15","SNB bo san 1,20"),
      ("GBPUSD","2016-10-07","Flash crash bang Anh"),
      ("GBPUSD","2016-06-24","Trung cau Brexit"),
      ("EURUSD","2020-03-19","Dinh hoang loan COVID"),
      ("AUDJPY","2019-01-03","Flash crash yen")]
for p, d, name in EV:
    df = data[p]; m = df.Date == pd.Timestamp(d)
    if m.any():
        r = df[m].iloc[0]
        rng = np.log(r.high/r.low)*100
        prev = df[df.Date < pd.Timestamp(d)].close.iloc[-1]
        ret = np.log(r.close/prev)*100
        # bien do trung binh 250 ngay truoc
        hist = df[df.Date < pd.Timestamp(d)].tail(250)
        avg = (np.log(hist.high/hist.low)*100).mean()
        print(f"{p} {d} ({name}):")
        print(f"   O={r.open:.4f} H={r.high:.4f} L={r.low:.4f} C={r.close:.4f}"
              f" | bien do={rng:.2f}%  loi suat={ret:+.2f}%  |  bien do TB 250 ngay truoc={avg:.2f}%"
              f"  -> gap {rng/avg:.1f} lan")
    else:
        print(f"{p} {d} ({name}): KHONG CO trong du lieu")

print("\n" + "="*104)
print("D. DU LIEU GIO (h1) — quy uoc phien va ranh gioi thanh ngay")
print("="*104)
h, _ = load("EURUSD","h1")
h["dow"] = h.Date.dt.dayofweek; h["hour"] = h.Date.dt.hour
print(f"n = {len(h)},  tu {h.Date.iloc[0]} den {h.Date.iloc[-1]}")
cnt = h.groupby(h.Date.dt.date).size()
print(f"So thanh moi ngay: trung vi={cnt.median():.0f}  min={cnt.min()}  max={cnt.max()}")
print("\nPhan bo theo thu (0=T2 .. 6=CN):")
print("   " + "  ".join(f"{d}:{v}" for d, v in h.dow.value_counts().sort_index().items()))
print("\nGio xuat hien it nhat / nhieu nhat:")
hv = h.hour.value_counts().sort_index()
print("   " + " ".join(f"{i:02d}h:{v}" for i, v in hv.items()))
first_h = h.groupby(h.Date.dt.date).hour.min().mode()[0]
last_h  = h.groupby(h.Date.dt.date).hour.max().mode()[0]
print(f"\nGio dau tien pho bien nhat trong ngay: {first_h:02d}h ; gio cuoi: {last_h:02d}h")
print("=> Ranh gioi thanh ngay cua nguon nay la 00:00 theo gio server, KHONG phai 17:00 New York.")

json.dump(rows, open("/tmp/fx/data_audit.json","w"), indent=1)
print("\nDa luu /tmp/fx/data_audit.json")
