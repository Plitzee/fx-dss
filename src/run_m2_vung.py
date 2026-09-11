"""PHA 2 / WEEK 3 — KIEM TRA DO VUNG cho phat hien |phan ung EURUSD|.

`run_m2_batngo.py` tim ra bien duy nhat thang M1 tren truc bien dong:
do lon phan ung EURUSD trong cua so 100 phut quanh cong bo FOMC
(kiem tra -2,7%, DM p=0,0009, 6/6 cap cai thien).

Roadmap Week 3 (`02_PHASE_2_NEWS_AWARE.md`, muc 3) doi cac lat cat do vung
TRUOC khi ra quyet dinh stop/go:
    pair-by-pair          — DA LAM trong run_m2_batngo.py (6/6 cai thien)
    year-by-year          — file nay
    high-news vs low-news — file nay (tam phan vi do lon phan ung)
    walk-forward          — file nay (khop lai theo nam)

KY LUAT QUAN TRONG. Mo hinh da duoc CHON tren kiem dinh va da cham MOT LAN
tren kiem tra. Cac lat cat o day la CHAN DOAN cua ket qua da cham do — KHONG
duoc dung de doi mo hinh, doi dac trung, hay chon lai bien the. Neu mot lat
cat trong xau, no la thong tin phai bao cao, khong phai ly do de tinh chinh.

Chay:  python src/run_m2_vung.py
Ghi:   output/m2_vung.json
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
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import volfc2 as V2                                            # noqa: E402
from run_final7 import dm_nw                                    # noqa: E402
from run_h8_tintuc import nap_thong_cao, dac_trung_van_ban      # noqa: E402
from run_m2_bien_dong import dung_bang, qlike                    # noqa: E402
from run_m2_batngo import dac_trung_batngo                       # noqa: E402

DAC_TRUNG = ["abs_eurusd"]      # bien DA CHON, khong doi


CO_BAN = ["log_h_har", "buoc"]


def khop(d_tr, d_cham, cot):
    """Khop tren d_tr, cham QLIKE tren d_cham. Tra ve mang ton that.

    d_cham phai da duoc loc san (chung mot bo hang cho ca M1 va M2) de hai
    day ton that thang hang tung phan tu — DM test doi dieu do.
    """
    c = CO_BAN + list(cot)
    d_tr = d_tr[d_tr[c + ["y"]].notna().all(1)]
    if len(d_tr) < 100 or len(d_cham) < 20:
        return None
    X = np.column_stack([np.ones(len(d_tr))] + [d_tr[x].values for x in c])
    b, *_ = np.linalg.lstsq(X, d_tr.y.values, rcond=None)
    s2 = float(np.var(d_tr.y.values - X @ b))
    Xs = np.column_stack([np.ones(len(d_cham))] + [d_cham[x].values for x in c])
    h = np.exp(np.clip(Xs @ b, -30, 0) + 0.5 * s2)
    return qlike(d_cham.rv_that.values, h)


def loc_chung(d):
    """Giu cac hang co DU ca cot M1 lan cot M2 — de hai mo hinh cham cung hang."""
    c = CO_BAN + DAC_TRUNG + ["y", "rv_that"]
    return d[d[c].notna().all(1)]


def so_sanh(d_tr, d_cham, nhan):
    d_cham = loc_chung(d_cham)
    a = khop(d_tr, d_cham, [])
    b = khop(d_tr, d_cham, DAC_TRUNG)
    if a is None or b is None:
        return None
    _, pp = dm_nw(b - a)
    return dict(nhan=nhan, n=int(len(a)), m1=float(a.mean()), m2=float(b.mean()),
                chenh=float(b.mean() / a.mean() - 1) * 100, dm_p=float(pp))


def main():
    t0 = time.time()
    print("=" * 96)
    print("PHA 2 / WEEK 3 — ĐỘ VỮNG của |phản ứng EURUSD| (biến ĐÃ CHỌN, không đổi)")
    print("=" * 96)

    Fb = dac_trung_batngo()
    tc = nap_thong_cao()
    F4 = dac_trung_van_ban(tc)
    cb = ["abs_mp1", "mp1", "abs_ust2y", "abs_sp500", "abs_eurusd"]
    c4 = ["thay_doi_cau_chu", "doi_do_dai", "giong_dieu"]
    bang, chung = V2.nap_bang()
    df = dung_bang(bang, chung, [(Fb, cb), (F4, c4)])
    df["nam"] = pd.DatetimeIndex(df.ngay).year
    tr = df[df.doan == 0]
    ngoai = df[df.doan >= 1]            # kiểm định + kiểm tra
    ra = {}

    # ── 1. theo năm (ngoài mẫu huấn luyện)
    print(f"\n[1/3] THEO NĂM — khớp trên huấn luyện, chấm từng năm ngoài mẫu")
    print(f"  {'năm':<7}{'n':>6}{'QLIKE M1':>11}{'QLIKE M2':>11}{'chênh':>9}{'DM p':>9}")
    print("  " + "-" * 53)
    theo_nam, duong = [], 0
    for nam in sorted(ngoai.nam.unique()):
        r = so_sanh(tr, ngoai[ngoai.nam == nam], str(nam))
        if r is None:
            continue
        theo_nam.append(r)
        duong += int(r["chenh"] < 0)
        print(f"  {r['nhan']:<7}{r['n']:>6}{r['m1']:>11.4f}{r['m2']:>11.4f}"
              f"{r['chenh']:>8.1f}%{r['dm_p']:>9.4f}")
    print("  " + "-" * 53)
    print(f"  → {duong}/{len(theo_nam)} năm cải thiện")
    ra["theo_nam"] = theo_nam
    ra["nam_duong"] = f"{duong}/{len(theo_nam)}"

    # ── 2. tin manh vs tin yeu (tam phan vi do lon, nguong CHOT tren huan luyen)
    print(f"\n[2/3] TIN MẠNH vs TIN YẾU — tam phân vị |phản ứng EURUSD|, "
          f"ngưỡng chốt trên huấn luyện")
    q = np.nanquantile(tr.abs_eurusd.values, [1 / 3, 2 / 3])
    print(f"  ngưỡng: {q[0]:.3f}% · {q[1]:.3f}%")
    print(f"  {'nhóm':<12}{'n':>6}{'QLIKE M1':>11}{'QLIKE M2':>11}{'chênh':>9}{'DM p':>9}")
    print("  " + "-" * 58)
    theo_manh = []
    for i, ten in enumerate(("tin yếu", "vừa", "tin mạnh")):
        lo = -np.inf if i == 0 else q[i - 1]
        hi = np.inf if i == 2 else q[i]
        s = ngoai[(ngoai.abs_eurusd >= lo) & (ngoai.abs_eurusd < hi)]
        r = so_sanh(tr, s, ten)
        if r is None:
            continue
        theo_manh.append(r)
        print(f"  {ten:<12}{r['n']:>6}{r['m1']:>11.4f}{r['m2']:>11.4f}"
              f"{r['chenh']:>8.1f}%{r['dm_p']:>9.4f}")
    print("  " + "-" * 58)
    ra["theo_do_manh"] = theo_manh

    # ── 3. walk-forward: khop lai dau moi nam bang cua so mo rong
    print(f"\n[3/3] WALK-FORWARD — khớp lại đầu mỗi năm, cửa sổ mở rộng")
    ql_m1, ql_m2, nam_ok = [], [], 0
    nams = sorted(ngoai.nam.unique())
    for nam in nams:
        d_tr = df[df.nam < nam]
        d_te = loc_chung(ngoai[ngoai.nam == nam])
        a = khop(d_tr, d_te, [])
        b = khop(d_tr, d_te, DAC_TRUNG)
        if a is None or b is None:
            continue
        ql_m1.append(a); ql_m2.append(b)
        nam_ok += int(b.mean() < a.mean())
    if ql_m1:
        A = np.concatenate(ql_m1); B = np.concatenate(ql_m2)
        _, pp = dm_nw(B - A)
        ch = (B.mean() / A.mean() - 1) * 100
        print(f"  gộp {len(A):,} hàng · QLIKE M1 {A.mean():.4f} · M2 {B.mean():.4f}"
              f" · chênh {ch:+.1f}% · DM p {pp:.4f}")
        print(f"  → {nam_ok}/{len(nams)} năm cải thiện dưới walk-forward")
        ra["walk_forward"] = dict(n=int(len(A)), m1=float(A.mean()),
                                   m2=float(B.mean()), chenh=float(ch),
                                   dm_p=float(pp), nam_duong=f"{nam_ok}/{len(nams)}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(ra, open(os.path.join(OUT, "m2_vung.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/m2_vung.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
