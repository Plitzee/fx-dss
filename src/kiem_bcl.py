"""BCL — ABLATION: tach C/J da loc chu ky co thang moc HAR san xuat khong?

Nam cau hinh DA CHOT TRUOC o `docs/BCL_TIEUCHI.md` muc 5.
Khung so sanh giong HET `kiem_nen_ablation.py` va `run_tang2_taptrung.py`, de
ket qua dat canh |gap| (-0,67%, p=0,0445) va hhi (+0,76%) duoc.

DOI CHUNG THEN CHOT la P2: cung phep tach nguong nhung KHONG loc chu ky. Neu P2
cung thang thi thu an tien la "phat hien nhay bang nguong", khong phai BCL —
H11b chot truoc noi ro nhu vay.

KHOP huan luyen · CHON kiem dinh · CHAM MOT LAN kiem tra.

Chay:  python src/kiem_bcl.py
Ghi:   output/bcl_ablation.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
BCL = os.path.join(ROOT, "data", "bcl_noituan.csv")
EPS = 1e-12

import kiem_nen as N                                          # noqa: E402
from kiem_nen_ablation import qlike, khop_cham                 # noqa: E402
from run_final7 import dm_nw                                   # noqa: E402
from metrics import mcs                                        # noqa: E402

# 5 cau hinh CHOT TRUOC (muc 5) — khong them, khong bot
CAU_HINH = {
    "B0 mốc": [],
    "P1 BCL C/J": ["log_c_bcl", "ti_j_bcl"],
    "P2 ngưỡng thô (đối chứng)": ["log_c_tho", "ti_j_tho"],
    "P3 cả hai": ["log_c_bcl", "ti_j_bcl", "log_c_tho", "ti_j_tho"],
    "P4 chỉ tỉ lệ chu kỳ": ["ti_chuky"],
}


def dung_bang():
    """Bang cua kiem_nen (moc m_har + dich y_bien_do) noi voi dac trung BCL."""
    df = N.dung_bang()
    b = pd.read_csv(BCL, parse_dates=["Date"])
    b["c_bcl"] = np.maximum(b.rv_kiem - b.j_bcl, EPS)
    b["c_tho"] = np.maximum(b.rv_kiem - b.j_tho, EPS)
    b["log_c_bcl"] = np.log(b.c_bcl)
    b["log_c_tho"] = np.log(b.c_tho)
    b["ti_j_bcl"] = np.log1p(np.maximum(b.j_bcl, 0.0) / b.c_bcl)
    b["ti_j_tho"] = np.log1p(np.maximum(b.j_tho, 0.0) / b.c_tho)
    cot = ["pair", "Date", "log_c_bcl", "log_c_tho", "ti_j_bcl", "ti_j_tho",
           "ti_chuky"]
    ten_ngay = "ngay" if "ngay" in df.columns else "Date"
    df = df.merge(b[cot], left_on=["pair", ten_ngay],
                  right_on=["pair", "Date"], how="left",
                  suffixes=("", "_b"))
    return df.drop(columns=[c for c in df.columns if c.endswith("_b")])


def main():
    t0 = time.time()
    print("=" * 108)
    print("BCL — ABLATION trên trục BIÊN ĐỘ (5 cấu hình chốt trước, mục 5)")
    print("=" * 108)

    df = dung_bang()
    can = sorted({c for v in CAU_HINH.values() for c in v} | {"m_har", "y_bien_do"})
    df = df[df[can].notna().all(1)].reset_index(drop=True)
    g = df.doan.values
    tr, va, te = g == 0, g == 1, g == 2
    y_rv = np.maximum(np.exp(df.y_bien_do.values), EPS)
    print(f"bảng {len(df):,} hàng · huấn luyện {tr.sum():,} · "
          f"kiểm định {va.sum():,} · kiểm tra {te.sum():,}")
    print(f"tỉ lệ chu kỳ Σ(r/s)²/Σr²: trung vị {df.ti_chuky.median():.4f} · "
          f"p10 {df.ti_chuky.quantile(.1):.4f} · p90 {df.ti_chuky.quantile(.9):.4f}")

    H, HS = {}, {}
    for k, c in CAU_HINH.items():
        H[k], HS[k] = khop_cham(df, c, tr)
    ql = {k: qlike(y_rv, v) for k, v in H.items()}
    b0 = ql["B0 mốc"]

    print("\n" + "=" * 108)
    print(f"{'cấu hình':<28}{'#đt':>5}{'QLIKE kđ':>11}{'so B0':>9}"
          f"{'QLIKE kt':>11}{'so B0':>9}{'DM p (kt)':>11}")
    print("-" * 108)
    kq = {}
    for k in CAU_HINH:
        v, t_ = float(ql[k][va].mean()), float(ql[k][te].mean())
        dv = (v / b0[va].mean() - 1) * 100
        dt_ = (t_ / b0[te].mean() - 1) * 100
        p = np.nan if k == "B0 mốc" else dm_nw(ql[k][te] - b0[te])[1]
        kq[k] = dict(n_dt=len(CAU_HINH[k]), qlike_vd=v, qlike_kt=t_,
                     d_vd=float(dv), d_kt=float(dt_),
                     dm_p=float(p) if np.isfinite(p) else None,
                     he_so={a: float(b) for a, b in HS[k].items()})
        print(f"{k:<28}{len(CAU_HINH[k]):>5}{v:>11.4f}{dv:>8.2f}%{t_:>11.4f}"
              f"{dt_:>8.2f}%{p:>11.4f}")
    print("-" * 108)
    print("  (âm = TỐT HƠN mốc HAR sản xuất)")

    print("\nHỆ SỐ trên huấn luyện — H11 đòi hệ số log C > hệ số tỉ lệ J:")
    for k in CAU_HINH:
        if CAU_HINH[k]:
            print(f"  {k:<28}{ {a: round(b, 4) for a, b in HS[k].items()} }")

    # ── MCS
    try:
        ten = list(CAU_HINH)
        L = np.column_stack([ql[k][te] for k in ten])
        giu = mcs(L, alpha=0.10)
        giu_ten = [ten[i] for i in (giu if not isinstance(giu, dict)
                                    else giu.get("keep", []))]
        print(f"\nMCS (α=0,10) trên kiểm tra giữ: {giu_ten}")
    except Exception as e:                              # noqa: BLE001
        giu_ten = None
        print(f"\nMCS không chạy được: {e}")

    # ── chon tren KIEM DINH
    chon = min((k for k in CAU_HINH if k != "B0 mốc"),
               key=lambda k: kq[k]["qlike_vd"])
    r = kq[chon]
    print(f"\nCHỌN TRÊN KIỂM ĐỊNH: {chon}")
    print(f"  → trên kiểm tra: {r['d_kt']:+.2f}% · DM p thô {r['dm_p']:.4f}")

    # ── theo tung cap cho cau hinh duoc chon VA cho P1
    pr = df.pair.values
    theo_cap = {}
    for nhan in dict.fromkeys([chon, "P1 BCL C/J"]):
        print(f"\nTHEO TỪNG CẶP — {nhan} (đoạn kiểm tra):")
        print(f"  {'cặp':<9}{'QLIKE B0':>11}{'cấu hình':>11}{'chênh':>9}{'DM p':>9}")
        duong = 0
        tc = {}
        for p in sorted(set(pr)):
            m = te & (pr == p)
            a, bb = b0[m], ql[nhan][m]
            _, pp = dm_nw(bb - a)
            ch = (bb.mean() / a.mean() - 1) * 100
            duong += int(ch < 0)
            tc[p] = dict(chenh=float(ch), dm_p=float(pp))
            print(f"  {p:<9}{a.mean():>11.4f}{bb.mean():>11.4f}{ch:>8.2f}%{pp:>9.4f}")
        print(f"  → {duong}/{len(tc)} cặp cải thiện")
        theo_cap[nhan] = dict(cap=tc, duong=duong)

    # ── phan quyet theo muc 8
    p1, p2 = kq["P1 BCL C/J"], kq["P2 ngưỡng thô (đối chứng)"]
    dk1 = (r["d_kt"] < 0) and (r["dm_p"] is not None) and (r["dm_p"] < 0.05 / 4)
    dk2 = p1["d_kt"] < p2["d_kt"]
    dk3 = theo_cap[chon]["duong"] >= 5
    print("\n" + "=" * 108)
    print("PHÁN QUYẾT theo TIÊU CHÍ CHỐT TRƯỚC (docs/BCL_TIEUCHI.md mục 8)")
    print("-" * 108)
    print(f"  ĐK1 cấu hình được chọn thắng kiểm tra, p < 0,0125   "
          f"{'ĐẠT' if dk1 else 'TRƯỢT'}   ({r['d_kt']:+.2f}%, p={r['dm_p']:.4f})")
    print(f"  ĐK2 P1 (có lọc) tốt hơn P2 (không lọc) — H11b       "
          f"{'ĐẠT' if dk2 else 'TRƯỢT'}   (P1 {p1['d_kt']:+.2f}% · P2 {p2['d_kt']:+.2f}%)")
    print(f"  ĐK3 ≥ 5/6 cặp cải thiện                             "
          f"{'ĐẠT' if dk3 else 'TRƯỢT'}   ({theo_cap[chon]['duong']}/6)")
    print("-" * 108)
    xong = dk1 and dk2 and dk3
    print(f"  → {'DƯƠNG' if xong else 'KHÔNG đủ điều kiện dương — KHÔNG đổi sản xuất'}")
    if not dk2:
        print("  H11b: BCL KHÔNG được ghi công — lọc chu kỳ không hơn ngưỡng thô.")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(cau_hinh=kq, chon=chon, theo_cap=theo_cap, mcs=giu_ten,
                   dk=dict(dk1=bool(dk1), dk2=bool(dk2), dk3=bool(dk3)),
                   phan_quyet="duong" if xong else "am"),
              open(os.path.join(OUT, "bcl_ablation.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/bcl_ablation.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
