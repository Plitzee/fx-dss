"""TANG 2 — XAY DAC TRUNG "TAP TRUNG RV TRONG NGAY" tu nen M1 goc.

BOI CANH. `docs/PHA2_KETQUA.md` muc 3b: bien |phan ung EURUSD| thang mo hinh
nen -2,7% ngoai mau, nhung HE SO AM va on dinh — nghia la no KHONG do "cu soc
lon -> bien dong cao hon", ma CHIET KHAU phan HAR ngoai suy thua. Cu nhay
trong cua so cong bo lam phong RV ngay hop; HAR mang sang ngay sau; cu nhay
do khong dai.

Neu doc do dung thi hieu ung phai TONG QUAT — khong rieng 135 ky FOMC. Day la
phep thu dung: do TRUC TIEP muc do tap trung cua RV trong ngay, cho MOI phien,
tu nen M1 cua chinh repo.

GIA THUYET — CHOT TRUOC, commit truoc khi cham:
  Cung mot muc RV ngay, RV TAP TRUNG trong mot cu no ngan thi IT DAI hon RV
  trai deu. Nen he so cua do tap trung trong hoi quy log_rv(t+1) phai AM.

VI SAO KHONG TRUNG voi bipower/jump da co. Bipower (da co trong `rv5`/`volfc2`
va trong dac trung ML `lbpv`/`ljump`) tach phan lien tuc khoi phan nhay nhung
BAT BIEN VOI THU TU THOI GIAN trong ngay: no khong phan biet mot cu no don
gon trong 15 phut voi cung ngan ay nang luong rai deu 24 gio. Do tap trung
bat dung chieu do.

DAC TRUNG (moi ngay, moi cap, tinh tu loi suat 5 phut trong ngay):
  hhi     = sum_i (rv_i / RV)^2          Herfindahl cua phan bo RV trong ngay
  top1    = max_i rv_i / RV              ti trong bin 5 phut manh nhat
  top12   = tong 12 bin manh nhat / RV   ti trong mot gio manh nhat (roi rac)
  n_bin   = so bin 5 phut co du lieu     (de loc ngay thieu du lieu)

KHONG RO RI: moi dac trung cua ngay t chi dung loi suat TRONG ngay t, va chi
duoc dung de du bao ngay t+1. Giong het cach `rv5` da lam.

MUI GIO: y het `rv5.py`/`prep_fx.py` — New York (co DST) -> UTC.

CHI DOC `dukas/histdata_raw` (bo phat trien, 6 cap, 2010-2025).
KHONG dong toi `dukas/histdata_seal`.

Chay:  python src/xay_tap_trung.py
Ghi:   data/tap_trung_rv.csv
"""
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(os.path.dirname(ROOT), "dukas", "histdata_raw")
RA = os.path.join(ROOT, "data", "tap_trung_rv.csv")
TZ = "America/New_York"
BIN = "5min"
TOI_THIEU = 100          # so bin toi thieu trong ngay moi tinh (288 la day du)


def doc_m1(duong):
    """Doc mot file nen M1 HistData, doi mui gio y het rv5.py."""
    df = pd.read_csv(duong, sep=";", header=None, engine="c",
                     names=["ts", "o", "h", "l", "c", "v"], dtype=str)
    dt = pd.to_datetime(df.ts, format="%Y%m%d %H%M%S", errors="coerce")
    c = pd.to_numeric(df.c, errors="coerce")
    m = pd.DataFrame({"Date": dt, "close": c}).dropna()
    loc = m.Date.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    m = m.assign(Date=loc.dt.tz_convert("UTC").dt.tz_localize(None))
    return m.dropna(subset=["Date"]).sort_values("Date")


def tap_trung_ngay(m):
    """Tra ve bang mot dong moi ngay: hhi, top1, top12, n_bin, rv_kiem."""
    g = m.assign(k=m.Date.dt.floor(BIN)).groupby("k", sort=True).close.last()
    d = pd.DataFrame({"close": g})
    d["ngay"] = d.index.normalize()
    d["r"] = np.log(d.close).diff()
    d.loc[d.ngay != d.ngay.shift(), "r"] = np.nan     # bo loi suat bac qua ngay
    d = d.dropna(subset=["r"])
    d["rv_bin"] = d.r.values ** 2

    hang = []
    for ngay, s in d.groupby("ngay", sort=True):
        v = s.rv_bin.values
        tong = v.sum()
        if len(v) < TOI_THIEU or not np.isfinite(tong) or tong <= 0:
            continue
        w = v / tong
        sap = np.sort(w)[::-1]
        hang.append(dict(Date=ngay, n_bin=len(v), rv_kiem=float(tong),
                         hhi=float((w ** 2).sum()),
                         top1=float(sap[0]),
                         top12=float(sap[:12].sum())))
    return pd.DataFrame(hang)


def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(SRC, "*.csv")))
    if not files:
        sys.exit(f"Khong thay nen M1 o {SRC}")
    theo_cap = {}
    for f in files:
        ten = os.path.basename(f)
        cap = re.match(r"([A-Z]{6})_", ten).group(1)
        theo_cap.setdefault(cap, []).append(f)

    print(f"nguon: {SRC}")
    print(f"{len(theo_cap)} cap · {len(files)} file nam\n")

    ra = []
    for cap, fs in sorted(theo_cap.items()):
        t1 = time.time()
        phan = [tap_trung_ngay(doc_m1(f)) for f in sorted(fs)]
        d = pd.concat(phan, ignore_index=True).sort_values("Date")
        d.insert(0, "pair", cap)
        ra.append(d)
        print(f"  {cap}  {len(d):>6,} ngay  "
              f"{d.Date.min().date()} -> {d.Date.max().date()}  "
              f"hhi trung vi {d.hhi.median():.4f}  "
              f"top1 trung vi {d.top1.median():.3f}  ({time.time()-t1:.0f}s)")

    out = pd.concat(ra, ignore_index=True)
    out.to_csv(RA, index=False)
    print(f"\n→ {RA}  ({len(out):,} dong, {time.time()-t0:.0f}s)")

    # kiem tra tinh tinh: hhi phai nam trong [1/n, 1]
    xau = out[(out.hhi < 1.0 / out.n_bin - 1e-9) | (out.hhi > 1.0 + 1e-9)]
    print(f"tu kiem: hhi ngoai khoang [1/n, 1] = {len(xau)} dong "
          f"({'DAT' if len(xau) == 0 else 'HONG'})")
    xau2 = out[(out.top1 > out.top12 + 1e-9) | (out.top12 > 1.0 + 1e-9)]
    print(f"tu kiem: top1 <= top12 <= 1        = {len(xau2)} vi pham "
          f"({'DAT' if len(xau2) == 0 else 'HONG'})")


if __name__ == "__main__":
    main()
