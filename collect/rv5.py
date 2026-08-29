#!/usr/bin/env python3
"""TINH REALIZED VARIANCE O NHIEU TAN SUAT LAY MAU, TU NEN M1 GOC.

Vi sao: pipeline dang lay muc tieu RV tu 24 thanh GIO/ngay. Chuan trong tai lieu
(Andersen-Bollerslev) la lay mau 5 phut ~ 288 quan sat/ngay. M1 da co san.

Tinh cho moi ngay, moi cap:
    rv_m1  rv_m5  rv_m15  rv_h1   = tong binh phuong loi suat TRONG ngay
                                    (bo loi suat bac qua ranh gioi ngay)
Cho phep ve "volatility signature plot" — RV theo tan suat lay mau — de chon
tan suat co co so, thay vi chon bua.

Mui gio xu ly Y HET prep_fx.py: New York -> UTC, co doi gio mua he.
"""
import argparse, glob, os, re, sys, time
import numpy as np, pandas as pd

SRC = "histdata_raw"
OUT = "rv_multi.csv"
TZ  = "America/New_York"

def read_m1(path):
    df = pd.read_csv(path, sep=";", header=None, engine="c",
                     names=["ts", "o", "h", "l", "c", "v"], dtype=str)
    dt = pd.to_datetime(df.ts, format="%Y%m%d %H%M%S", errors="coerce")
    c  = pd.to_numeric(df.c, errors="coerce")
    m  = pd.DataFrame({"Date": dt, "close": c}).dropna()
    # New York (co DST) -> UTC, giong het prep_fx.py
    loc = m.Date.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    m = m.assign(Date=loc.dt.tz_convert("UTC").dt.tz_localize(None))
    return m.dropna(subset=["Date"]).sort_values("Date")

def rv_at(m, rule):
    """RV ngay o mot tan suat lay mau. Loi suat bac qua ranh gioi ngay bi bo."""
    g = m.assign(k=m.Date.dt.floor(rule)).groupby("k", sort=True).close.last()
    d = pd.DataFrame({"close": g})
    d["day"] = d.index.normalize()
    d["r"] = np.log(d.close).diff()
    d.loc[d.day != d.day.shift(), "r"] = np.nan
    x = d.dropna(subset=["r"]).groupby("day").r
    return x.apply(lambda s: float((s ** 2).sum())), x.size()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pair"); a=ap.parse_args()
    files = sorted(glob.glob(os.path.join(SRC, "*.csv")))
    byp = {}
    for f in files:
        mm = re.search(r"([A-Z]{6})", os.path.basename(f).upper())
        if mm:
            byp.setdefault(mm.group(1), []).append(f)
    print(f"{len(files)} file, {len(byp)} cap: {', '.join(sorted(byp))}", flush=True)

    t0 = time.time()
    out = []
    todo=[a.pair.upper()] if a.pair else sorted(byp)
    for p in todo:
        parts = []
        for f in sorted(byp[p]):
            try:
                parts.append(read_m1(f))
            except Exception as e:
                print(f"  ! {os.path.basename(f)}: {e}", flush=True)
        if not parts:
            continue
        m = pd.concat(parts, ignore_index=True).drop_duplicates("Date").sort_values("Date")
        rows = {}
        for name, rule in (("rv_m1", "1min"), ("rv_m5", "5min"),
                           ("rv_m15", "15min"), ("rv_h1", "1h")):
            rv, n = rv_at(m, rule)
            rows[name] = rv
            rows["n_" + name.split("_")[1]] = n
        df = pd.DataFrame(rows)
        df.index.name = "Date"
        df = df.reset_index()
        df.insert(1, "pair", p)
        out.append(df)
        print(f"  {p}: {len(m):,} nen M1 -> {len(df):,} ngay  "
              f"({m.Date.min().date()} -> {m.Date.max().date()})  {time.time()-t0:.0f}s",
              flush=True)
        del m, parts

    res = pd.concat(out, ignore_index=True)
    if a.pair:
        hdr = not os.path.exists(OUT)
        res.to_csv(OUT, mode="a", header=hdr, index=False)
        print(f"\nNoi them {len(res):,} dong vao {OUT} ({time.time()-t0:.0f}s)", flush=True)
        return
    res.to_csv(OUT, index=False)
    print(f"\nGhi {OUT}: {len(res):,} dong, {res.pair.nunique()} cap  "
          f"({time.time()-t0:.0f}s)", flush=True)

    print("\nVOLATILITY SIGNATURE — RV trung binh theo tan suat lay mau")
    print(f"{'Cap':<9}{'1 phut':>12}{'5 phut':>12}{'15 phut':>12}{'1 gio':>12}"
          f"{'M1/M5':>9}{'H1/M5':>9}")
    print("-" * 76)
    for p, g in res.groupby("pair"):
        a, b, c, d = [g[k].mean() * 1e6 for k in ("rv_m1", "rv_m5", "rv_m15", "rv_h1")]
        print(f"{p:<9}{a:>12.3f}{b:>12.3f}{c:>12.3f}{d:>12.3f}{a/b:>9.2f}{d/b:>9.2f}")
    print("\n(don vi 1e-6. M1/M5 > 1 nhieu = nhieu vi cau truc thi truong o tan so cao.")
    print(" H1/M5 lech khoi 1 = uoc luong theo gio thieu chinh xac.)")

if __name__ == "__main__":
    main()
