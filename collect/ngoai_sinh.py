"""PHA 3B / WEEK 1 — tai bo bien NGOAI SINH kem NGU NGHIA DAU THOI GIAN.

Dac ta: `03_PHASE_3_CAUSALITY_AWARE.md` Week 1.
Bien, quy tac tre, va ba thu bi loai DA CHOT TRUOC o `docs/PHA3B_TIEUCHI.md`
muc 2-3 (commit e88ae66), truoc khi tai mot byte nao.

BA THU FILE NAY PHAI LAM, khong duoc bo bot:

  1. Tai 9 chuoi thi truong theo NGAY tu FRED + 1 chuoi lai suat thang da co.
  2. Ghi bang NGU NGHIA day du cho tung chuoi:
         observation_period · release_time · available_time
         · revision_time · value_as_released · revised_value · source
  3. Hai phep tu kiem BAT BUOC (muc 3d cua bien ban):
         (a) cat tuong lai  — dac trung truoc moc khong duoc doi
         (b) hieu dinh      — doi chieu ban ALFRED VINTAGE voi ban hien hanh,
                              KHONG duoc khang dinh suong "chuoi thi truong
                              khong hieu dinh"

Ve cot `available_time` — cot QUYET DINH:
  chuoi thi truong  ->  phien KE TIEP (gia tri ngay t chi dung tu t+1)
  lai suat thang    ->  tre 2 THANG (giu nguyen quy tac PHA3_TIEUCHI.md)
  lich cong bo      ->  biet truoc nhieu thang, khong ro ri

Chay:  python collect/ngoai_sinh.py
Ghi:   data/ngoai_sinh/chuoi_ngay.csv
       data/ngoai_sinh/ngu_nghia.csv
       data/ngoai_sinh/kiem_hieu_dinh.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RA = os.path.join(DATA, "ngoai_sinh")
API = "https://api.stlouisfed.org/fred"

# 9 chuoi thi truong theo NGAY — CHOT TRUOC, bien ban muc 2a
CHUOI = {
    "DGS2":         ("A_loi_suat", "lợi suất TPCP Mỹ 2 năm — kỳ vọng chính sách Fed"),
    "DGS10":        ("A_loi_suat", "lợi suất TPCP Mỹ 10 năm — định giá lại tài sản"),
    "T10Y2Y":       ("A_loi_suat", "độ dốc đường cong 10Y−2Y — chu kỳ chính sách"),
    "DFII10":       ("A_loi_suat", "lợi suất THỰC 10 năm (TIPS) — kênh lãi suất thực"),
    "VIXCLS":       ("B_an_y_bd", "ẩn ý biến động cổ phiếu Mỹ — khẩu vị rủi ro, tháo carry"),
    "GVZCLS":       ("B_an_y_bd", "ẩn ý biến động vàng — cầu trú ẩn, gắn CHF"),
    "OVXCLS":       ("B_an_y_bd", "ẩn ý biến động dầu — gắn CAD"),
    "DCOILBRENTEU": ("C_hang_hoa", "dầu Brent — điều kiện thương mại, gắn CAD"),
    "NIKKEI225":    ("D_co_phieu", "Nikkei 225 — đại diện rủi ro/carry cho JPY"),
}

# moc lay du lieu: som hon 2010 de co du dem cho cua so 20 phien
TU = "2009-01-01"
DEN = "2025-12-31"          # KHONG lay 2026 — toan bo 2026 nam trong tap khoa so

# muc 3d(b): doi chieu vintage tai mot moc qua khu
VINTAGE_MOC = "2024-06-28"
VINTAGE_CHUOI = ["DGS10", "VIXCLS", "DCOILBRENTEU"]     # >= 3 chuoi, chot truoc


def _khoa():
    k = os.environ.get("FRED_API_KEY") or os.environ.get("FRED_TOKEN")
    if not k:
        sys.exit("Thieu FRED_API_KEY. Dang ky mien phi: "
                 "https://fredaccount.stlouisfed.org/apikey")
    return k


def _goi(duong, mem=False, **tham):
    """mem=True: tra None thay vi dung, dung cho tra cuu KHONG rang buoc."""
    tham.update(api_key=_khoa(), file_type="json")
    for lan in range(4):
        try:
            r = requests.get(f"{API}/{duong}", params=tham, timeout=60)
            if r.status_code == 200:
                return r.json()
            time.sleep(1.5 * (lan + 1))
        except requests.RequestException:
            time.sleep(1.5 * (lan + 1))
    if mem:
        return None
    sys.exit(f"FRED khong tra loi: {duong} {tham.get('series_id','')}")


def quan_sat(ma, tu=TU, den=DEN, vintage=None):
    """Chuoi quan sat. vintage=YYYY-MM-DD -> ban ALFRED nhu thay o thoi diem do."""
    t = dict(series_id=ma, observation_start=tu, observation_end=den)
    if vintage:
        t.update(realtime_start=vintage, realtime_end=vintage)
    d = _goi("series/observations", **t)["observations"]
    x = pd.DataFrame(d)[["date", "value"]]
    x["date"] = pd.to_datetime(x.date)
    x["value"] = pd.to_numeric(x.value, errors="coerce")   # FRED dung "." cho khuyet
    return x.dropna().reset_index(drop=True)


def lich_cong_bo(ma):
    """Ngay cong bo that cua chuoi, tu FRED release dates."""
    r = _goi("series/release", mem=True, series_id=ma)
    if not r or not r.get("releases"):
        return None, pd.DatetimeIndex([])
    rel = r["releases"][0]
    ng = []
    for off in range(0, 40000, 10000):          # FRED gioi han limit = 10.000
        d = _goi("release/dates", mem=True, release_id=rel["id"], limit=10000,
                 offset=off, realtime_start=TU,
                 include_release_dates_with_no_data="false")
        lo = (d or {}).get("release_dates", [])
        ng += [x["date"] for x in lo]
        if len(lo) < 10000:
            break
    return rel["name"], pd.DatetimeIndex(sorted(pd.to_datetime(ng)))


def phien_giao_dich():
    """Lich phien cua chinh bo gia — available_time phai bam vao lich nay."""
    g = pd.read_csv(os.path.join(DATA, "prices", "EURUSD_d1.csv"),
                    parse_dates=["Date"])
    return pd.DatetimeIndex(sorted(g.Date.unique()))


def phien_ke_tiep(ngay, phien):
    """Phien giao dich dau tien NAM SAU `ngay` — dung cho available_time."""
    i = np.searchsorted(np.asarray(phien, "datetime64[ns]"),
                        np.asarray(ngay, "datetime64[ns]"), side="right")
    ra = np.full(len(ngay), np.datetime64("NaT"), "datetime64[ns]")
    co = i < len(phien)
    ra[co] = np.asarray(phien, "datetime64[ns]")[i[co]]
    return pd.DatetimeIndex(ra)


def kiem_hieu_dinh():
    """Muc 3d(b): ban vintage co khac ban hien hanh khong. PHAI do, khong doan."""
    ra = {}
    for ma in VINTAGE_CHUOI:
        nay = quan_sat(ma, den=VINTAGE_MOC)
        cu = quan_sat(ma, den=VINTAGE_MOC, vintage=VINTAGE_MOC)
        g = nay.merge(cu, on="date", suffixes=("_nay", "_vintage"))
        if not len(g):
            ra[ma] = dict(n=0, khac=None, ghi_chu="khong co quan sat chung")
            continue
        khac = int((~np.isclose(g.value_nay, g.value_vintage,
                                rtol=0, atol=5e-4)).sum())
        ra[ma] = dict(n=int(len(g)), khac=khac,
                      ty_le=round(100.0 * khac / len(g), 4),
                      dat=bool(khac / len(g) <= 0.005))
    return ra


def main():
    t0 = time.time()
    os.makedirs(RA, exist_ok=True)
    print("=" * 100)
    print("PHA 3B / WEEK 1 — BIẾN NGOẠI SINH + NGỮ NGHĨA DẤU THỜI GIAN")
    print("biến & quy tắc trễ chốt trước: docs/PHA3B_TIEUCHI.md mục 2–3 (e88ae66)")
    print("=" * 100)

    phien = phien_giao_dich()
    print(f"lịch phiên giao dịch: {len(phien):,} phiên · "
          f"{phien.min().date()} → {phien.max().date()}\n")

    khung, ngu_nghia = [], []
    print(f"{'chuỗi':<15}{'họ':<12}{'n':>7}{'từ':>13}{'đến':>13}  {'lịch công bố'}")
    print("-" * 100)
    for ma, (ho, ly_do) in CHUOI.items():
        x = quan_sat(ma)
        ten_rel, ngay_rel = lich_cong_bo(ma)
        x["ma"] = ma
        khung.append(x)

        # available_time = phien giao dich KE TIEP (quy tac chot truoc, muc 3b)
        av = phien_ke_tiep(x.date, phien)
        # release_time thuc te neu FRED cho biet: ngay cong bo dau tien >= ngay quan sat
        if len(ngay_rel):
            j = np.searchsorted(np.asarray(ngay_rel, "datetime64[ns]"),
                                np.asarray(x.date, "datetime64[ns]"), side="right")
            rt = np.where(j < len(ngay_rel),
                          np.asarray(ngay_rel, "datetime64[ns]")[np.minimum(
                              j, len(ngay_rel) - 1)],
                          np.datetime64("NaT"))
        else:
            rt = np.full(len(x), np.datetime64("NaT"), "datetime64[ns]")

        ngu_nghia.append(pd.DataFrame(dict(
            source=ma, ho=ho, ly_do_kinh_te=ly_do,
            observation_period=x.date,
            release_time=pd.DatetimeIndex(rt),
            available_time=av,
            revision_time=pd.NaT,          # chuoi thi truong: kiem o muc 3d(b)
            value_as_released=x.value,     # = revised_value neu khong hieu dinh
            revised_value=x.value)))
        print(f"{ma:<15}{ho:<12}{len(x):>7,}{str(x.date.min().date()):>13}"
              f"{str(x.date.max().date()):>13}  "
              f"{(ten_rel or '(không có)')[:40]}")

    ch = pd.concat(khung, ignore_index=True)
    nn = pd.concat(ngu_nghia, ignore_index=True)

    # ── dang rong theo ngay, chi so la NGAY QUAN SAT (chua ap tre)
    rong = ch.pivot_table(index="date", columns="ma", values="value",
                          aggfunc="last").sort_index()
    rong.to_csv(os.path.join(RA, "chuoi_ngay.csv"))
    nn.to_csv(os.path.join(RA, "ngu_nghia.csv"), index=False)
    print("-" * 100)
    print(f"bảng rộng: {rong.shape[0]:,} ngày × {rong.shape[1]} chuỗi "
          f"→ data/ngoai_sinh/chuoi_ngay.csv")
    print(f"ngữ nghĩa: {len(nn):,} dòng × 9 cột → data/ngoai_sinh/ngu_nghia.csv")

    # ── tu kiem (b): HIEU DINH
    print(f"\nTỰ KIỂM HIỆU ĐÍNH (ALFRED vintage @ {VINTAGE_MOC}) — mục 3d(b)")
    print("-" * 100)
    hd = kiem_hieu_dinh()
    dat_het = True
    for ma, r in hd.items():
        if r.get("n"):
            ok = r["dat"]
            dat_het &= ok
            print(f"  {ma:<15}{r['n']:>7,} quan sát chung · khác {r['khac']:>4} "
                  f"({r['ty_le']:.3f}%)   {'ĐẠT' if ok else 'HỎNG'}")
        else:
            print(f"  {ma:<15}{r['ghi_chu']}")
    print(f"  → {'ĐẠT' if dat_het else 'HỎNG'} "
          f"(ngưỡng chốt trước: ≤ 0,5% số quan sát)")
    json.dump(dict(moc=VINTAGE_MOC, ket_qua=hd, dat=bool(dat_het)),
              open(os.path.join(RA, "kiem_hieu_dinh.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if not dat_het:
        sys.exit("tự kiểm hiệu đính HỎNG — dừng, không báo cáo kết quả nào")

    # ── kiem tinh day du tren doan dung that
    print(f"\nĐỘ PHỦ trên đoạn dùng thật (2011-12 → 2025-12)")
    print("-" * 100)
    m = (rong.index >= "2011-12-01")
    for c in rong.columns:
        s = rong.loc[m, c]
        print(f"  {c:<15}{s.notna().sum():>7,} giá trị · khuyết "
              f"{s.isna().sum():>5,} ({100*s.isna().mean():>5.1f}%)")

    print(f"\n→ xong {time.time()-t0:.0f}s. "
          f"Bước sau: python src/pha3b_dacTrung.py (dựng đặc trưng + kiểm cắt tương lai)")


if __name__ == "__main__":
    main()
