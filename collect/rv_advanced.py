#!/usr/bin/env python3
"""DO LUONG NOI NGAY NANG CAO — dau vao cho cac mo hinh HAR hien dai.

rv5.py chi cho realized variance. Cac mo hinh manh hon can them:

  rq5  realized quarticity  -> HARQ (Bollerslev-Patton-Quaedvlieg 2016):
                              he so cua HAR duoc cho phep thay doi theo do
                              CHINH XAC do luong cua ngay hom truoc
  bpv5 bipower variation    -> tach nhay (jump) khoi thanh phan lien tuc,
                              cho HAR-CJ (Andersen-Bollerslev-Diebold 2007)
  rsp/rsn semivariance      -> SHAR (Patton-Sheppard 2015): bien dong am du
                              bao tot hon bien dong duong

Mui gio xu ly Y HET prep_fx.py va rv5.py: New York -> UTC, co doi gio mua he.
Chay tung cap:  python collect/rv_advanced.py --pair EURUSD
"""
import argparse, glob, os, re, time
import numpy as np, pandas as pd

SRC = "histdata_raw"
OUT = "rv_adv.csv"
TZ  = "America/New_York"
RULE = "5min"

def read_m1(path):
    df = pd.read_csv(path, sep=";", header=None, engine="c",
                     names=["ts", "o", "h", "l", "c", "v"], dtype=str)
    dt = pd.to_datetime(df.ts, format="%Y%m%d %H%M%S", errors="coerce")
    c  = pd.to_numeric(df.c, errors="coerce")
    m  = pd.DataFrame({"Date": dt, "close": c}).dropna()
    loc = m.Date.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    m = m.assign(Date=loc.dt.tz_convert("UTC").dt.tz_localize(None))
    return m.dropna(subset=["Date"]).sort_values("Date")

def measures(m):
    g = m.assign(k=m.Date.dt.floor(RULE)).groupby("k", sort=True).close.last()
    d = pd.DataFrame({"close": g})
    d["day"] = d.index.normalize()
    d["r"] = np.log(d.close).diff()
    d.loc[d.day != d.day.shift(), "r"] = np.nan      # bo loi suat bac qua ngay
    rows = []
    MU1 = np.sqrt(2.0 / np.pi)
    for day, s in d.dropna(subset=["r"]).groupby("day"):
        r = s.r.values; n = len(r)
        if n < 2:
            continue
        a = np.abs(r)
        bpv = (MU1 ** -2) * float(np.sum(a[1:] * a[:-1])) * n / (n - 1)
        rows.append(dict(Date=day, rv5=float(np.sum(r ** 2)),
                         rq5=float(n / 3.0 * np.sum(r ** 4)),
                         bpv5=bpv,
                         rsp=float(np.sum(r[r > 0] ** 2)),
                         rsn=float(np.sum(r[r < 0] ** 2)),
                         n5=n))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair")
    a = ap.parse_args()
    byp = {}
    for f in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
        mm = re.search(r"([A-Z]{6})", os.path.basename(f).upper())
        if mm:
            byp.setdefault(mm.group(1), []).append(f)
    todo = [a.pair.upper()] if a.pair else sorted(byp)
    t0 = time.time()
    old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    for p in todo:
        parts = [read_m1(f) for f in sorted(byp[p])]
        m = pd.concat(parts, ignore_index=True).sort_values("Date")
        out = measures(m)
        out["pair"] = p
        if len(old):
            old = old[old.pair != p]
        old = pd.concat([old, out], ignore_index=True)
        print(f"  {p}: {len(out):,} ngay  [{time.time()-t0:.0f}s]", flush=True)
    old = old.sort_values(["pair", "Date"])
    old.to_csv(OUT, index=False)
    print(f"ghi {OUT}: {len(old):,} dong")

if __name__ == "__main__":
    main()
