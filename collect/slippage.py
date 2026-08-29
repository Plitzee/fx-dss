#!/usr/bin/env python3
"""DO TRUOT GIA QUA MUC DUNG LO — thay gia dinh 'khop dung tai muc dung lo'.

Tang 4 gia dinh khi gia cham muc dung lo thi thoat duoc dung o do. Thang 3/2020
cho thay gia dinh do sai: phan vi 95 cua spread vot 19-115 lan. Nhung spread chi
la mot nua cau chuyen — nua kia la GAP XUYEN QUA: trong chinh phut cham muc,
gia con di tiep bao xa.

Cach do (chi dung nen M1 that, khong mo phong):
  * moi ngay giao dich, gia vao = gia mo cua ngay
  * dat muc dung lo cach d phan tram ben duoi (vi the mua)
  * tim PHUT DAU TIEN co low <= muc dung lo
  * truot gia = muc dung lo - low cua chinh phut do   (don vi pip)
    day la muc gia te nhat con giao dich trong phut cham — can duoi cua fill
  * do them: gia dong cua phut do, va low cua 5 phut ke tiep

Chay tung cap:  python collect/slippage.py --pair EURUSD
"""
import argparse, glob, os, re, time
import numpy as np
import pandas as pd

SRC = "histdata_raw"
OUT = "slippage.csv"
TZ = "America/New_York"
DISTS = (0.001, 0.002, 0.003, 0.005, 0.010)      # muc dung lo cach gia mo cua


def read_m1(path):
    df = pd.read_csv(path, sep=";", header=None, engine="c",
                     names=["ts", "o", "h", "l", "c", "v"], dtype=str)
    dt = pd.to_datetime(df.ts, format="%Y%m%d %H%M%S", errors="coerce")
    out = pd.DataFrame({"Date": dt})
    for k in ("o", "h", "l", "c"):
        out[k] = pd.to_numeric(df[k], errors="coerce")
    out = out.dropna()
    loc = out.Date.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    out = out.assign(Date=loc.dt.tz_convert("UTC").dt.tz_localize(None))
    return out.dropna(subset=["Date"]).sort_values("Date")


def measure(m, pip):
    m = m.reset_index(drop=True)
    day = m.Date.dt.normalize().values
    rows = []
    for d, g in m.groupby(day, sort=True):
        if len(g) < 60:
            continue
        o = float(g.o.iloc[0])
        lo = g.l.values; cl = g.c.values
        for dist in DISTS:
            stop = o * (1 - dist)
            hit = np.where(lo <= stop)[0]
            if len(hit) == 0:
                continue
            i = int(hit[0])
            j = min(i + 5, len(lo))
            rows.append(dict(Date=pd.Timestamp(d), dist=dist,
                             truot_phut=(stop - lo[i]) / pip,
                             truot_dong=(stop - cl[i]) / pip,
                             truot_5phut=(stop - lo[i:j].min()) / pip,
                             gio=int(pd.Timestamp(g.Date.iloc[i]).hour)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pair"); a = ap.parse_args()
    byp = {}
    for f in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
        mm = re.search(r"([A-Z]{6})", os.path.basename(f).upper())
        if mm:
            byp.setdefault(mm.group(1), []).append(f)
    todo = [a.pair.upper()] if a.pair else sorted(byp)
    old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    t0 = time.time()
    for p in todo:
        pip = 0.01 if "JPY" in p else 0.0001
        m = pd.concat([read_m1(f) for f in sorted(byp[p])], ignore_index=True).sort_values("Date")
        r = measure(m, pip); r["pair"] = p
        if len(old):
            old = old[old.pair != p]
        old = pd.concat([old, r], ignore_index=True)
        print(f"  {p}: {len(r):,} lan cham muc dung lo  [{time.time()-t0:.0f}s]", flush=True)
    old.to_csv(OUT, index=False)
    print(f"ghi {OUT}: {len(old):,} dong")


if __name__ == "__main__":
    main()
