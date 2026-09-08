"""H2 — MOTIF (hinh dang lap lai), ho thu hai cua Giai doan 2 (REPLAN muc 3.1).

Khac voi khong gian dac trung cua run_quyluat.py (nguong tren TUNG DAC TRUNG
doc duoc), H2 hoi mot cau khac: co HINH DANG chuoi z gan day nao lap lai va
noi truoc duoc lop cua phien ke tiep khong, BAT KE hinh dang do co ten goi doc
duoc (RSI, MACD...) hay khong.

KHONG CO stumpy trong moi truong nay (matrix profile that). Thay bang MOT
CODEBOOK vector-quantization — ve mat thong ke day la "motif" theo dung nghia
(mot cum hinh dang hay gap), chi khac o cach tim: KMeans tren cua so z CHUAN
HOA (z-normalized), thay vi tinh khoang cach Euclid doi mot cho tung cap.

KHONG GIAN GIA THUYET, LIET KE TRUOC:
  3 do dai cua so (5, 10, 20 phien)  x  K=8 cum (codebook, hoc tren HUAN LUYEN,
  GOP CA 6 CAP)  x  3 lop dich  =  72 gia thuyet, mot con so BIET TRUOC.

Khong to hop cheo voi 12 dac trung cua run_quyluat.py de giu khong gian nho va
biet truoc — day la MOT HO RIENG, dung dung protocol WY/doi chung/LOPO nhu ho
dau (import truc tiep tu run_quyluat.py, khong viet lai).

Chay:  python src/run_h2_motif.py
Ghi:   output/h2_motif.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                             # noqa: E402
from split import doan                                        # noqa: E402
from run_quyluat import (                                     # noqa: E402
    nap_du_lieu, z_lift, westfall_young, doi_chung, la_vi_tu_nen,
    TEN_LOP, TEN_KIEM_SOAT, MIN_KHOP, LIFT_LOPO, MIN_CAP_DUONG, T_DIEU_KIEN,
    NPERM, KHOI, SEED, EPS,
)

DO_DAI = (5, 10, 20)
K = 8               # so cum codebook moi do dai — CHOT TRUOC, khong do roi chon
DAC_TRUNG_NEN_H2 = ()  # khong vi tu nao cua H2 la "chinh nen" theo nghia sigma^


def cua_so_chuan_hoa(z, L):
    """Cua so do dai L, KET THUC tai t (nhan qua), da z-normalize (tru trung
    binh, chia do lech chuan CUA CHINH CUA SO). NaN neu thieu du lieu."""
    n = len(z)
    S = np.full((n, L), np.nan)
    for t in range(L - 1, n):
        w = z[t - L + 1:t + 1]
        if np.isfinite(w).all():
            sd = w.std()
            S[t] = (w - w.mean()) / sd if sd > EPS else 0.0
    return S


def vi_tu_motif(zs, dts):
    """Tra ve (lit, ten, hop_le) — lit: (24, N) bool, hop_le: (N,) bool.

    CANH BAO DA BAT DUOC (08/09/2026): phien THIEU cua so hop le (gan bien
    ngay du lieu mong do merge_thin_days) co ty le lop "di ngang" toi 99,3%,
    trong khi ty le nen la 30,8% — tuc BAN THAN viec "co du lieu day 20 phien
    lien tuc" da la mot du bao gan hoan hao, hoan toan khong lien quan gi den
    HINH DANG cua so. Neu tinh z_lift tren toan bo `pha` (bao gom ca phien
    thieu cua so lam mau so), MOI cum deu "co tin hieu" nhu nhau — gia, vi tat
    ca cum deu ke thua dung mot dieu kien "co cua so" giong het nhau.

    SUA: chi so sanh trong dung tap phien CO DU CA BA do dai cua so (`hop_le`)
    — cung mau so cho moi gia thuyet, loai bo hoan toan phan sai lech do
    thieu du lieu."""
    from sklearn.cluster import KMeans
    lit_theo_L = {}
    hop_le_theo_L = {}
    for L in DO_DAI:
        subs = [cua_so_chuan_hoa(zs[i], L) for i in range(len(B.PAIRS))]
        ok_all = [np.isfinite(subs[i]).all(1) for i in range(len(B.PAIRS))]
        hop_le_theo_L[L] = np.concatenate(ok_all)
        tr_rows = []
        for i in range(len(B.PAIRS)):
            tri = doan(dts[i]) == 0
            tr_rows.append(subs[i][tri & ok_all[i]])
        Xtr = np.concatenate(tr_rows, axis=0)
        km = KMeans(n_clusters=K, n_init=5, random_state=0).fit(Xtr)
        lab_all = []
        for i in range(len(B.PAIRS)):
            lab = np.full(len(subs[i]), -1)
            lab[ok_all[i]] = km.predict(subs[i][ok_all[i]])
            lab_all.append(lab)
        lit_theo_L[L] = np.concatenate(lab_all)     # (N,) nhan cum, -1 = thieu
    hop_le = np.all([hop_le_theo_L[L] for L in DO_DAI], axis=0)
    lit, ten = [], []
    for L in DO_DAI:
        lab = lit_theo_L[L]
        for k in range(K):
            lit.append((lab == k) & hop_le)
            ten.append(f"motif L={L} cụm {k}")
    return np.array(lit), ten, hop_le


def main():
    t0 = time.time()
    print("=" * 112)
    print("H2 — MOTIF (VQ codebook trên z chuẩn hoá), họ thứ hai của Giai đoạn 2")
    print("=" * 112)
    du = nap_du_lieu()
    zs, dts = du["zs"], du["dts"]
    y, cap, dt = du["y"], du["cap"], du["dt"]
    kiem_soat, cum = du["kiem_soat"], du["cum"]
    tr, va, te, pha = du["tr"], du["va"], du["te"], du["pha"]

    print("Đang khớp codebook KMeans (K=8) cho 3 độ dài cửa sổ trên HUẤN LUYỆN…",
          flush=True)
    M, ten, hop_le = vi_tu_motif(zs, dts)
    print(f"KHÔNG GIAN GIẢ THUYẾT H2: {len(ten)} vị từ × 3 lớp = "
          f"{len(ten)*3} giả thuyết — liệt kê đầy đủ, biết trước")
    # SUA LOI RO RI: chi so sanh trong tap phien CO DU ca ba do dai cua so —
    # xem canh bao trong docstring cua vi_tu_motif(). Thu hep pha/te theo do.
    pha = pha & hop_le
    te = te & hop_le
    print(f"lọc còn phiên có đủ cả 3 cửa sổ: phát hiện {int(pha.sum()):,} hàng"
          f" · xác nhận {int(te.sum()):,} hàng\n")

    print(f"[1/4] Westfall–Young, {NPERM} hoán vị, null khối {KHOI} ngày…",
          flush=True)
    Z, L, nk, P, nguong = westfall_young(M, y, pha)
    du_khop = nk >= MIN_KHOP
    print(f"      {int(du_khop.sum()):,}/{len(ten)*3:,} giả thuyết đủ {MIN_KHOP} lần khớp")
    song = (P < 0.05) & np.isfinite(Z)
    tho = (np.abs(np.nan_to_num(Z)) > 1.96) & np.isfinite(Z)
    print(f"      sống sót W-Y p<0,05: {int(song.sum())} / thô p<0,05: {int(tho.sum())}"
          f" (nếu toàn nhiễu kỳ vọng {0.05*np.isfinite(Z).sum():.0f})")

    qua_dk, qua_lopo, xn = [], [], []
    if song.sum() > 0:
        print(f"\n[2/4] Đối chứng có điều kiện (|t| > {T_DIEU_KIEN})…", flush=True)
        ung = []
        for i, c in zip(*np.where(song)):
            b, t = doi_chung(M[i], y, c, kiem_soat, cum=cum)
            ung.append(dict(i=int(i), lop=int(c), ten=ten[i], n=int(nk[i]),
                            z=float(Z[i, c]), lift=float(L[i, c]), p_wy=float(P[i, c]),
                            b_dk=b, t_dk=t))
        qua_dk = [u for u in ung if np.isfinite(u["t_dk"]) and abs(u["t_dk"]) > T_DIEU_KIEN]
        print(f"      {len(qua_dk)}/{len(ung)} vị từ còn tin riêng sau điều kiện hoá")

        print(f"\n[3/4] Bỏ-một-cặp (lift ≥ {LIFT_LOPO}, ≥{MIN_CAP_DUONG}/6 cặp dương)…",
              flush=True)
        for u in qua_dk:
            mi, lifts = M[u["i"]], []
            for p in B.PAIRS:
                mp = (cap == p) & pha & (y >= 0)
                kh = mi & mp
                if kh.sum() < 20:
                    lifts.append(np.nan); continue
                pc = (y[mp] == u["lop"]).mean()
                lifts.append(float((y[kh] == u["lop"]).mean() / max(pc, EPS)))
            lifts = np.array(lifts)
            nd = int(np.nansum(lifts > 1.0))
            u["lift_cap"], u["so_cap_duong"] = lifts.tolist(), nd
            u["lift_min"] = float(np.nanmin(lifts))
            if nd >= MIN_CAP_DUONG and np.nanmin(lifts) >= LIFT_LOPO:
                qua_lopo.append(u)
        print(f"      {len(qua_lopo)}/{len(qua_dk)} chuyển giao được qua các cặp")

        print("\n[4/4] Xác nhận trên đoạn KIỂM TRA…", flush=True)
        Zte, Lte, nkte = z_lift(M, y, te)
        for u in qua_lopo:
            u["z_te"] = float(Zte[u["i"], u["lop"]])
            u["lift_te"] = float(Lte[u["i"], u["lop"]])
        xn = [u for u in qua_lopo if np.isfinite(u["z_te"]) and u["z_te"] > 1.96]
        print(f"      {len(xn)}/{len(qua_lopo)} tái lập trên kiểm tra")
    else:
        print("\n→ KHÔNG vị từ nào sống sót Westfall–Young.")

    print("\n" + "=" * 112)
    print(f"{'PHỄU H2':<46}{'còn lại':>10}")
    for nhan, v in (("không gian giả thuyết", len(ten) * 3),
                    ("đủ số lần khớp", int(du_khop.sum())),
                    ("thô p<0,05", int(tho.sum())),
                    ("sống sót Westfall–Young", int(song.sum())),
                    ("còn tin riêng sau đối chứng", len(qua_dk)),
                    ("chuyển giao được (LOPO)", len(qua_lopo)),
                    ("tái lập trên KIỂM TRA", len(xn))):
        print(f"{nhan:<46}{v:>10,}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(khong_gian=len(ten) * 3, du_khop=int(du_khop.sum()),
                    tho=int(tho.sum()), wy=int(song.sum()),
                    sau_dieu_kien=len(qua_dk), sau_lopo=len(qua_lopo),
                    xac_nhan=len(xn), quy_luat=xn, K=K, do_dai=list(DO_DAI)),
              open(os.path.join(OUT, "h2_motif.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/h2_motif.json — {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
