"""MAU HINH NEN — ABLATION: dua vao mo hinh thi co thang moc khong?

Tiep noi `kiem_nen.py`. Tam cau hinh DA CHOT TRUOC o `docs/NEN_TIEUCHI.md`
muc 1b (commit 9643f80).

Buoc kiem dinh da cho: 7/600 song sot Westfall-Young, va CA BAY deu la
`abs_gap` tren truc BIEN DO, dau DUONG — dung co che 2 da khai bao:

    DATASET.md: "RV = tong binh phuong loi suat TRONG ngay, bo loi suat bac qua
    ranh gioi ngay ... gap qua dem chi chiem 1,7-3,1% tong phuong sai o FX nen
    BO QUA DUOC."

Do la gia dinh chua ai kiem. Buoc nay kiem no theo dung tieu chi 5a.2: cau hinh
tot nhat tren KIEM DINH phai thang moc tren KIEM TRA voi DM p < 0,025
(Bonferroni 2 truc), >= 5/6 cap cai thien, va dau he so khop co che.

Mau hinh nen CO TEN (K3) da chet o buoc truoc (0/360) nen o day chung chi con
vai tro doi chung: neu K3 cung "cai thien" thi do la dau hieu bo may qua de dai.

KHOP huan luyen · CHON kiem dinh · CHAM MOT LAN kiem tra.

Chay:  python src/kiem_nen_ablation.py
Ghi:   output/nen_ablation.json
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

import kiem_nen as N                                          # noqa: E402
from run_final7 import dm_nw                                  # noqa: E402
from metrics import mcs                                       # noqa: E402

EPS = 1e-12


def qlike(y, h):
    r = np.maximum(y, EPS) / np.maximum(h, EPS)
    return r - np.log(np.maximum(r, EPS)) - 1.0


def khop_cham(df, cols, tr):
    """OLS log_rv(t+1) ~ m_har + cols. Tra ve du bao PHUONG SAI."""
    c = ["m_har"] + list(cols)
    X = np.column_stack([np.ones(len(df))] + [df[k].values for k in c])
    y = df.y_bien_do.values
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    b, *_ = np.linalg.lstsq(X[tr & ok], y[tr & ok], rcond=None)
    s2 = float(np.var(y[tr & ok] - X[tr & ok] @ b))
    h = np.full(len(df), np.nan)
    m = np.isfinite(X).all(1)
    h[m] = np.exp(np.clip(X[m] @ b, -30, 5) + 0.5 * s2)
    return h, dict(zip(c, b[1:]))


# ── TAM cau hinh, chot truoc (bien ban muc 1b)
def cau_hinh():
    L = [f"_L{l}" for l in N.LAG]
    return {
        "B0 mốc": [],
        "K1 hình học": [f"{c}{l}" for c in N.K1 for l in L],
        "K2 phạm vi−RV": [f"{c}{l}" for c in N.K2 for l in L],
        "K3 mẫu có tên": [f"{c}{l}" for c in N.K3 for l in L],
        "K4 gap đêm": [f"{c}{l}" for c in N.K4 for l in L],
        "chỉ |gap| L1": ["abs_gap_L1"],
        "K1+K2+K4": [f"{c}{l}" for c in N.K1 + N.K2 + N.K4 for l in L],
        "tất cả 25": [f"{c}{l}" for c in N.TAT_CA for l in L],
    }


def main():
    t0 = time.time()
    print("=" * 104)
    print("MẪU HÌNH NẾN — ABLATION trên trục BIÊN ĐỘ")
    print("tám cấu hình chốt trước: docs/NEN_TIEUCHI.md mục 1b (9643f80)")
    print("=" * 104)

    df = N.dung_bang()
    CH = cau_hinh()
    need = sorted({c for v in CH.values() for c in v} | {"m_har", "y_bien_do"})
    df = df[df[need].notna().all(1)].reset_index(drop=True)
    g = df.doan.values
    tr, va, te = g == 0, g == 1, g == 2
    y_rv = np.maximum(np.exp(df.y_bien_do.values), EPS)
    print(f"bảng {len(df):,} hàng · huấn luyện {tr.sum():,} · "
          f"kiểm định {va.sum():,} · kiểm tra {te.sum():,}")

    H, HS = {}, {}
    for k, c in CH.items():
        H[k], HS[k] = khop_cham(df, c, tr)
    ql = {k: qlike(y_rv, v) for k, v in H.items()}

    print("\n" + "=" * 104)
    print(f"{'cấu hình':<16}{'#đt':>5}{'QLIKE kđ':>11}{'so B0':>9}"
          f"{'QLIKE kt':>11}{'so B0':>9}{'DM p (kt)':>11}")
    print("-" * 104)
    kq = {}
    for k in CH:
        v, t_ = float(ql[k][va].mean()), float(ql[k][te].mean())
        dv = (v / ql["B0 mốc"][va].mean() - 1) * 100
        dt_ = (t_ / ql["B0 mốc"][te].mean() - 1) * 100
        p = np.nan if k == "B0 mốc" else dm_nw(ql[k][te] - ql["B0 mốc"][te])[1]
        kq[k] = dict(n_dt=len(CH[k]), qlike_vd=v, qlike_kt=t_, d_vd=dv, d_kt=dt_,
                     dm_p=float(p) if np.isfinite(p) else None)
        print(f"{k:<16}{len(CH[k]):>5}{v:>11.4f}{dv:>8.2f}%{t_:>11.4f}"
              f"{dt_:>8.2f}%{p:>11.4f}")
    print("-" * 104)
    print("  (âm = TỐT HƠN mốc)")

    # ── chon tren KIEM DINH, cham MOT LAN tren kiem tra
    chon = min((k for k in CH if k != "B0 mốc"),
               key=lambda k: kq[k]["qlike_vd"])
    r = kq[chon]
    print(f"\nCHỌN TRÊN KIỂM ĐỊNH (quy tắc của repo): **{chon}**")
    print(f"  → trên kiểm tra: {r['d_kt']:+.2f}% · DM p thô {r['dm_p']:.4f}")

    # ── theo tung cap
    pr = df.pair.values
    print(f"\nTHEO TỪNG CẶP (kiểm tra, cấu hình đã chọn):")
    print(f"  {'cặp':<9}{'n':>7}{'QLIKE B0':>11}{'đã chọn':>11}{'chênh':>9}{'DM p':>9}")
    print("  " + "-" * 57)
    duong, theo_cap = 0, {}
    for p in sorted(set(pr[te])):
        m = te & (pr == p)
        a, b = ql["B0 mốc"][m], ql[chon][m]
        ch = (b.mean() / a.mean() - 1) * 100
        _, pp = dm_nw(b - a)
        duong += int(ch < 0)
        theo_cap[p] = dict(chenh=float(ch), dm_p=float(pp))
        print(f"  {p:<9}{int(m.sum()):>7,}{a.mean():>11.4f}{b.mean():>11.4f}"
              f"{ch:>8.2f}%{pp:>9.4f}")
    print("  " + "-" * 57)
    print(f"  → cải thiện {duong}/6 cặp")

    # ── theo nam va theo che do
    nam = pd.DatetimeIndex(df.ngay).year.values
    print(f"\nTHEO NĂM (kiểm tra):")
    for u in sorted(set(nam[te])):
        m = te & (nam == u)
        a, b = ql["B0 mốc"][m].mean(), ql[chon][m].mean()
        print(f"  {u}  n={int(m.sum()):>5,}  B0 {a:.4f}  chọn {b:.4f}  "
              f"{(b/a-1)*100:+.2f}%")
    q5 = np.nanquantile(df.m_har.values[tr], [.2, .4, .6, .8])
    cd = np.digitize(df.m_har.values, q5)
    print(f"\nTHEO CHẾ ĐỘ BIẾN ĐỘNG (ngưỡng từ huấn luyện):")
    theo_che_do = {}
    for j in range(5):
        m = te & (cd == j)
        if m.sum() < 30:
            continue
        a, b = ql["B0 mốc"][m].mean(), ql[chon][m].mean()
        theo_che_do[f"Q{j+1}"] = float((b / a - 1) * 100)
        print(f"  Q{j+1}{' êm' if j == 0 else ' căng' if j == 4 else '   '}"
              f"  n={int(m.sum()):>5,}  B0 {a:.4f}  chọn {b:.4f}  "
              f"{(b/a-1)*100:+.2f}%")

    # ── MCS
    ten = list(CH)
    L = np.column_stack([ql[k][te] for k in ten])
    idx, _ = mcs(L, alpha=0.10)
    song = [ten[i] for i in idx]
    print(f"\nModel Confidence Set (α = 0,10): {song}")

    # ── he so cua |gap|: dau phai DUONG theo co che 2.
    #
    # Phai lay tu CHINH cau hinh chua |gap|, khong phai tu cau hinh duoc chon:
    # co che 2 la phat bieu ve bien |gap|, no dung hay sai KHONG phu thuoc viec
    # cau hinh nao thang quy tac chon. (Ban dau lay tu `chon` nen ra rong khi
    # `chon` = K2 — loi bao cao, da sua.)
    gap_he = {k: v for k, v in HS["chỉ |gap| L1"].items() if "gap" in k}
    gap_he4 = {k: v for k, v in HS["K4 gap đêm"].items() if "gap" in k}
    print(f"\nHỆ SỐ |gap| trên huấn luyện (cơ chế 2 đòi dấu DƯƠNG):")
    print(f"  cấu hình 'chỉ |gap| L1' : "
          f"{ {k: round(v, 5) for k, v in gap_he.items()} }")
    print(f"  cấu hình 'K4 gap đêm'   : "
          f"{ {k: round(v, 5) for k, v in gap_he4.items()} }")

    # ── cau hinh |gap| theo tung cap. Du KHONG duoc chon, van phai bao cao:
    #    no la thu DUY NHAT song sot pheu kiem dinh o buoc truoc (7/600 W-Y).
    print(f"\nTHEO TỪNG CẶP — cấu hình 'chỉ |gap| L1' (không được chọn, nhưng "
          f"là thứ duy nhất sống sót phễu):")
    print(f"  {'cặp':<9}{'QLIKE B0':>11}{'|gap| L1':>11}{'chênh':>9}{'DM p':>9}")
    print("  " + "-" * 50)
    gd, theo_cap_gap = 0, {}
    for p_ in sorted(set(pr[te])):
        m = te & (pr == p_)
        a, b = ql["B0 mốc"][m], ql["chỉ |gap| L1"][m]
        ch = (b.mean() / a.mean() - 1) * 100
        _, pp = dm_nw(b - a)
        gd += int(ch < 0)
        theo_cap_gap[p_] = dict(chenh=float(ch), dm_p=float(pp))
        print(f"  {p_:<9}{a.mean():>11.4f}{b.mean():>11.4f}{ch:>8.2f}%{pp:>9.4f}")
    print("  " + "-" * 50)
    print(f"  → cải thiện {gd}/6 cặp")

    # ── PHAN QUYET theo muc 5a
    dk1 = True                      # da co 7 song sot W-Y + do vung o buoc truoc
    dk2 = (r["d_kt"] < 0) and (r["dm_p"] is not None) and (r["dm_p"] < 0.05 / 2)
    dk3 = duong >= 5
    dk4 = all(v > 0 for v in gap_he.values()) if gap_he else False
    print("\n" + "=" * 104)
    print("PHÁN QUYẾT theo TIÊU CHÍ CHỐT TRƯỚC (NEN_TIEUCHI.md mục 5a)")
    print("-" * 104)
    print(f"  ĐK1 có đặc trưng qua W-Y + độ vững           "
          f"{'ĐẠT' if dk1 else 'TRƯỢT'}   (7/600, toàn bộ là |gap|)")
    print(f"  ĐK2 thắng kiểm tra & p thô < 0,025           "
          f"{'ĐẠT' if dk2 else 'TRƯỢT'}   ({r['d_kt']:+.2f}%, p={r['dm_p']:.4f})")
    print(f"  ĐK3 ≥ 5/6 cặp cải thiện                      "
          f"{'ĐẠT' if dk3 else 'TRƯỢT'}   ({duong}/6)")
    print(f"  ĐK4 dấu hệ số |gap| DƯƠNG như cơ chế 2       "
          f"{'ĐẠT' if dk4 else 'TRƯỢT'}")
    print("-" * 104)
    duong_het = dk1 and dk2 and dk3 and dk4
    print(f"  → {'DƯƠNG' if duong_het else 'CHƯA ĐỦ ĐIỀU KIỆN DƯƠNG'}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(cau_hinh={k: len(v) for k, v in CH.items()}, ket_qua=kq,
                   chon=chon, theo_cap=theo_cap, cap_duong=duong,
                   theo_che_do=theo_che_do, mcs=song, he_so_gap=gap_he,
                   he_so_gap_K4=gap_he4, theo_cap_gap=theo_cap_gap,
                   cap_duong_gap=gd,
                   dk=dict(dk1=bool(dk1), dk2=bool(dk2), dk3=bool(dk3),
                           dk4=bool(dk4)),
                   phan_quyet="duong" if duong_het else "chua_du"),
              open(os.path.join(OUT, "nen_ablation.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/nen_ablation.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
