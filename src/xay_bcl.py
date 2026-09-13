"""BO LOC CHU KY NOI TUAN (Boudt-Croux-Laurent 2011) tu nen M1 goc.

Gia thuyet, cau hinh va tieu chi phu dinh DA CHOT TRUOC o `docs/BCL_TIEUCHI.md`.

VI SAO. `MAU_HINH_FX.md` muc A2 goi day la "cai tien tang 2 co co so tai lieu
manh nhat con chua lam":

  "Chuyen Chu nhat da sua (merge_thin_days) moi chi la phan tho nhat cua van de
  nay. Phan tinh hon — chu ky theo GIO TRONG NGAY va theo NGAY TRONG TUAN ben
  trong moi phien — van chua duoc khu trong rv_adv.csv."

DA XAC MINH no that su chua lam, va KHONG trung voi `deseason` da bi loai:
`volfc2.he_so_mua_vu` nhan `lv = log(rv5)`, tuc khu mua vu THEO THU tren chuoi
RV NGAY. BCL lam viec tren LOI SUAT NOI NGAY, TRUOC khi cong don thanh RV ngay.
Hai thu khac nhau.

NGUON: Boudt, Croux & Laurent (2011), *Robust Estimation of Intraweek
Periodicity in Volatility and Jump Detection*, Journal of Empirical Finance
18(2):353-367. Chu "intraweek" (khong chi intraday) la diem cot yeu: no xu ly
tuong tac NGAY-TRONG-TUAN x GIO-TRONG-NGAY, dung thu FX can vi tuan FX la
24x5 chu khong phai 24x7.
Ap cho FX tan suat cao: Yi (2023), *Finance Research Letters* 55:103821.

CACH DUNG DUNG DAN. BCL khong phai mot uoc luong RV thay the — no la bo loc
CHUAN HOA de nguong phat hien nhay ap dong deu qua moi o trong tuan. Mot loi
suat 3 sigma luc 14:00 New York (o co phuong sai cao du doan duoc) it kha nang
la nhay hon cung loi suat do luc 22:00. Nen san pham cua no la mot phep TACH
C/J tot hon, roi dua vao HAR duoi dang hai so hang — dung so do HAR-CJ
(Andersen, Bollerslev & Diebold 2007).

DOI CHUNG BAT BUOC: tinh CUNG phep tach nhung KHONG loc chu ky (nguong ap tren
loi suat tho). Neu doi chung cung thang thi cai an tien la "phat hien nhay",
khong phai "loc chu ky" — va BCL khong duoc ghi cong.

CHONG RO RI: he so chu ky s(thu, o) uoc CHI tren doan HUAN LUYEN, dong bang,
roi ap cho ca ba doan. Co tu kiem rieng.

CHI DOC `dukas/histdata_raw` (bo phat trien). KHONG dong `histdata_seal`.

Chay:  python src/xay_bcl.py
Ghi:   data/bcl_noituan.csv  ·  output/bcl_heso.json
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
SRC = os.path.join(os.path.dirname(ROOT), "dukas", "histdata_raw")
RA = os.path.join(ROOT, "data", "bcl_noituan.csv")
HS = os.path.join(ROOT, "output", "bcl_heso.json")
TZ = "America/New_York"
BIN = "5min"
O_NGAY = 288                 # so o 5 phut trong mot ngay
C_NGUONG = 4.0               # nguong phat hien nhay, chuan trong van lieu
TOI_THIEU_O = 100            # so o toi thieu trong ngay
TOI_THIEU_BIN = 60           # so quan sat toi thieu moi o tuan de uoc he so
HET_HUAN_LUYEN = "2021-10-13"   # bien doan huan luyen cua split.py


def doc_m1(duong):
    df = pd.read_csv(duong, sep=";", header=None, engine="c",
                     names=["ts", "o", "h", "l", "c", "v"], dtype=str)
    dt = pd.to_datetime(df.ts, format="%Y%m%d %H%M%S", errors="coerce")
    c = pd.to_numeric(df.c, errors="coerce")
    m = pd.DataFrame({"Date": dt, "close": c}).dropna()
    loc = m.Date.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    m = m.assign(Date=loc.dt.tz_convert("UTC").dt.tz_localize(None))
    return m.dropna(subset=["Date"]).sort_values("Date")


def loi_suat_5p(m):
    """Loi suat 5 phut trong ngay + nhan o tuan. Bo loi suat bac qua ranh gioi ngay."""
    g = m.assign(k=m.Date.dt.floor(BIN)).groupby("k", sort=True).close.last()
    d = pd.DataFrame({"close": g})
    d["ngay"] = d.index.normalize()
    d["r"] = np.log(d.close).diff()
    d.loc[d.ngay != d.ngay.shift(), "r"] = np.nan
    d = d.dropna(subset=["r"])
    thu = d.index.dayofweek.values
    o_ngay = (d.index.hour.values * 12 + d.index.minute.values // 5)
    d["o_tuan"] = thu * O_NGAY + o_ngay          # 0..1439
    return d


def _mad(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    return float(np.median(np.abs(x - np.median(x))) / 0.6745)


def he_so_chu_ky(d):
    """s(o tuan) uoc ROBUST (MAD) CHI tren doan huan luyen, chuan hoa mean(s^2)=1.

    Tra ve (s theo o_tuan do dai 1440, so o du mau).
    """
    tr = d[d.ngay < pd.Timestamp(HET_HUAN_LUYEN)]
    s = np.full(1440, np.nan)
    for o, grp in tr.groupby("o_tuan"):
        if len(grp) >= TOI_THIEU_BIN:
            s[o] = _mad(grp.r.values)
    du = int(np.isfinite(s).sum())
    # o thieu mau -> lay trung vi cua cac o con lai (khong de NaN lam mat du lieu)
    if du == 0:
        return None, 0
    s[~np.isfinite(s) | (s <= 0)] = np.nanmedian(s)
    s = s / np.sqrt(np.nanmean(s ** 2))          # chuan hoa BCL: mean(s^2) = 1
    return s, du


def tach_cj(d, s):
    """Tach C/J theo ngay, hai cach: CO loc chu ky (BCL) va KHONG loc (doi chung)."""
    hang = []
    for ngay, g in d.groupby("ngay", sort=True):
        r = g.r.values
        if len(r) < TOI_THIEU_O:
            continue
        rv = float(np.sum(r ** 2))
        if not np.isfinite(rv) or rv <= 0:
            continue
        sv = s[g.o_tuan.values]
        rt = r / sv                                   # da loc chu ky
        sd_bcl = _mad(rt)
        sd_tho = _mad(r)
        if not (np.isfinite(sd_bcl) and sd_bcl > 0
                and np.isfinite(sd_tho) and sd_tho > 0):
            continue
        nhay_bcl = np.abs(rt) > C_NGUONG * sd_bcl
        nhay_tho = np.abs(r) > C_NGUONG * sd_tho
        hang.append(dict(
            Date=ngay, rv_kiem=rv, n_o=len(r),
            j_bcl=float(np.sum(r[nhay_bcl] ** 2)),
            j_tho=float(np.sum(r[nhay_tho] ** 2)),
            n_nhay_bcl=int(nhay_bcl.sum()), n_nhay_tho=int(nhay_tho.sum()),
            # ti le phuong sai sau khi loc so truoc khi loc: "ngay nay chu ky co manh"
            ti_chuky=float(np.sum(rt ** 2) / rv)))
    return pd.DataFrame(hang)


def tu_kiem(s, d):
    """Ba phep tu kiem, in va tra ve dat/khong."""
    dat = True
    # 1. chuan hoa dung
    m2 = float(np.nanmean(s ** 2))
    ok = abs(m2 - 1.0) < 1e-9
    dat &= ok
    print(f"    chuẩn hoá mean(s²)=1: {m2:.10f}  {'ĐẠT' if ok else 'HỎNG'}")
    # 2. he so chi dung doan huan luyen — cat bo tuong lai khong doi he so
    d2 = d[d.ngay < pd.Timestamp("2023-01-01")]
    s2, _ = he_so_chu_ky(d2)
    lech = int(np.sum(~np.isclose(s, s2, equal_nan=True)))
    dat &= lech == 0
    print(f"    cắt dữ liệu sau 2023-01-01 → {lech}/1440 hệ số đổi  "
          f"{'ĐẠT' if lech == 0 else 'HỎNG'}")
    # 3. chu ky co that khong — bien thien giua cac o phai lon hon nhieu lay mau
    ok = np.nanmax(s) / np.nanmin(s) > 2.0
    dat &= ok
    print(f"    biên độ chu kỳ max/min = {np.nanmax(s)/np.nanmin(s):.2f}  "
          f"{'ĐẠT (chu kỳ có thật)' if ok else 'HỎNG'}")
    return dat


def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(SRC, "*.csv")))
    if not files:
        sys.exit(f"Khong thay nen M1 o {SRC}")
    theo_cap = {}
    for f in files:
        cap = re.match(r"([A-Z]{6})_", os.path.basename(f)).group(1)
        theo_cap.setdefault(cap, []).append(f)

    print("=" * 96)
    print("BỘ LỌC CHU KỲ NỘI TUẦN (Boudt–Croux–Laurent 2011) từ nến M1 gốc")
    print("tiêu chí chốt trước: docs/BCL_TIEUCHI.md")
    print("=" * 96)
    print(f"nguồn: {SRC}\nhệ số chu kỳ ước CHỈ trên huấn luyện (< {HET_HUAN_LUYEN})\n")

    ra, heso, dat_het = [], {}, True
    for cap, fs in sorted(theo_cap.items()):
        t1 = time.time()
        d = pd.concat([loi_suat_5p(doc_m1(f)) for f in sorted(fs)])
        d = d.sort_index()
        s, du = he_so_chu_ky(d)
        if s is None:
            print(f"  {cap}: không đủ mẫu, bỏ")
            continue
        print(f"  {cap}  {du}/1440 ô đủ mẫu · s: min {np.nanmin(s):.3f} "
              f"trung vị {np.nanmedian(s):.3f} max {np.nanmax(s):.3f}")
        dat_het &= tu_kiem(s, d)
        t = tach_cj(d, s)
        t.insert(0, "pair", cap)
        ra.append(t)
        heso[cap] = dict(s=[None if not np.isfinite(v) else round(float(v), 6)
                            for v in s], n_o_du=du)
        print(f"    {len(t):,} ngày · nhảy BCL {t.n_nhay_bcl.mean():.2f}/ngày · "
              f"thô {t.n_nhay_tho.mean():.2f}/ngày · "
              f"tỉ trọng J_bcl {(t.j_bcl/t.rv_kiem).mean():.4f} · "
              f"({time.time()-t1:.0f}s)")

    out = pd.concat(ra, ignore_index=True)
    out.to_csv(RA, index=False)
    os.makedirs(os.path.dirname(HS), exist_ok=True)
    json.dump(heso, open(HS, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n→ {RA}  ({len(out):,} dòng)")
    print(f"→ {HS}")
    print(f"TỰ KIỂM TỔNG: {'ĐẠT' if dat_het else 'HỎNG'} · {time.time()-t0:.0f}s")
    if not dat_het:
        sys.exit("tự kiểm HỎNG")


if __name__ == "__main__":
    main()
