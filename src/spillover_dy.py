"""LAN TRUYEN BIEN DONG GIUA CAC CAP — Diebold-Yilmaz (2012), ap dung cho HAR.

VI SAO. Da nghien cuu van lieu 2024-2026 (xem doc/BAOCAO_SPILLOVER.md). Mot bai
2025 (Rubaszek, Szafranek & Uddin, J. Intl Money & Finance) dung DUNG BO CAP
USD/EUR,JPY,AUD,CAD,GBP, do duoc EUR TRUYEN cu soc bien dong, GBP NHAN. Day la
huong nhan qua GIUA CAC MA ma repo chua tung thu — khac han khai pha quy luat
KY THUAT (da can 0/1.890) va du bao HUONG (da can 0/18 su kien, AUC ~0,5).

QUAN TRONG: day la thu nghiem tren TRUC BIEN DONG, khong phai huong. Bien
dong la truc DUY NHAT da chung minh co tin hieu that (BSS +0,0152 ngoai mau,
h=1). Khong duoc dien dai sang du bao huong.

DA CO MOT PHIEN BAN THO trong volfc2.py (`crosspair=True`): trung binh KHONG
TRONG SO cua log-RV da khu mua vu cua CAC CAP KHAC tai ngay t, dung du bao cap p
tai t+1. Da do trong luoi 1.024 cau hinh (output/grid2_valid.csv):

    crosspair=0  QLIKE trung binh 0,120570  (280 cau hinh)
    crosspair=1  QLIKE trung binh 0,120763  (280 cau hinh)

GAN NHU KHONG DOI — trung binh deu khong huong dilute mat tin hieu neu chi mot
vai cap la "nguon truyen" thuc su (dung nhu van lieu noi EUR truyen, GBP nhan;
trung binh voi ca CAD/AUD/CHF/JPY se pha loang).

KHAC BIET CUA MODULE NAY: thay trung binh deu bang TRONG SO PHAN RA PHUONG SAI
DU BAO TONG QUAT (generalized FEVD) tu VAR — moi cap nhan dung ty le bien dong
CUA CHINH NO ma cac cap KHAC giai thich duoc, khong phai trung binh tho. Day la
cau truc DIEBOLD-YILMAZ (2012), khac ban tho o mot diem cot loi.

GIAO THUC NHAN QUA: he so VAR uoc tren CUA SO MO RONG, khop lai moi ~21 phien
(cung nhip voi balop.du_bao_cuon), chi dung du lieu < t0 de tinh trong so tai
t0. Tu kiem ep giong cac module khac trong repo.

Chay:  python src/spillover_dy.py
Ghi:   output/spillover_dy.json (ma tran TO/FROM binh quan) va so sanh QLIKE
       tren doan KIEM DINH so voi khong them cot ngoai sinh nao.
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

import volfc2 as V2                                          # noqa: E402
from split import doan, TEST_TU                               # noqa: E402
from volfc import merge_thin_days                             # noqa: E402

BUOC = 21           # khop lai VAR moi ~1 thang, cung nhip voi cua so mo rong
DAM = 500           # so quan sat toi thieu truoc khi duoc phep khop VAR
BAC = 1             # bac VAR — bien dong ngay co tu tuong quan ngan, VAR(1) du
H_FEVD = 10         # chan du bao cho phan ra phuong sai (chuan Diebold-Yilmaz)
EPS = 1e-12


def fevd_tong_quat(A, Sigma, h=H_FEVD):
    """Phan ra phuong sai du bao TONG QUAT (generalized FEVD), Diebold-Yilmaz.

    A     : he so VAR(1), ma tran k x k (x_t = A x_{t-1} + eps_t)
    Sigma : hiep phuong sai phan du, k x k
    Tra ve theta (k x k): theta[i,j] = ty le phuong sai du bao h-buoc cua bien i
    do CU SOC cua bien j giai thich, DA CHUAN HOA theo hang (tong = 1)."""
    k = A.shape[0]
    Psi = [np.eye(k)]
    for _ in range(h - 1):
        Psi.append(A @ Psi[-1])
    sd = np.sqrt(np.diag(Sigma))
    theta = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            tu = sum((Psi[l][i] @ Sigma[:, j]) ** 2 for l in range(h)) / max(Sigma[j, j], EPS)
            mau = sum(Psi[l][i] @ Sigma @ Psi[l][i].T for l in range(h))
            theta[i, j] = tu / max(mau, EPS)
    return theta / np.maximum(theta.sum(1, keepdims=True), EPS)


def khop_var1(X, tr_idx, dam=DAM):
    """VAR(1) OLS tren cac hang tr_idx. Tra ve (A, Sigma) hoac None neu thieu du."""
    if len(tr_idx) < dam:
        return None
    Y = X[tr_idx[1:]]
    Z = X[tr_idx[:-1]]
    ok = np.isfinite(Y).all(1) & np.isfinite(Z).all(1)
    Y, Z = Y[ok], Z[ok]
    if len(Y) < dam // 2:
        return None
    A, *_ = np.linalg.lstsq(Z, Y, rcond=None)     # Y = Z A  ->  A la (k,k), x_t = A^T x_{t-1}... dung dang nay cho gon
    resid = Y - Z @ A
    Sigma = (resid.T @ resid) / max(len(Y) - Z.shape[1], 1)
    return A.T, Sigma      # tra ve A dang x_t = A x_{t-1} + eps


def spillover_theo_thoi_gian(sig_pairs, dt_all, cap, buoc=BUOC, dam=DAM):
    """Tra ve dict[p] -> mang trong so NHAN duoc TU TUNG cap khac, theo thoi gian.

    sig_pairs: dict pair -> mang sigma^ (do dai n, da can theo dt_all chung).
    Khop lai VAR moi `buoc` don vi tren log(sigma^2), CUA SO MO RONG, chi dung
    du lieu < t0. `buoc`/`dam` truyen tay khi doi tan suat (D1 hay H1)."""
    pairs = list(sig_pairs)
    k = len(pairs)
    n = len(dt_all)
    X = np.column_stack([np.log(np.maximum(sig_pairs[p], EPS) ** 2) for p in pairs])
    W = np.full((n, k, k), np.nan)     # W[t, i, j] = trong so cap i nhan tu cap j
    mo_cuoi = None
    for t0 in range(dam, n, buoc):
        tr_idx = np.arange(0, t0)
        kq = khop_var1(X, tr_idx, dam=dam)
        if kq is None:
            continue
        A, Sigma = kq
        theta = fevd_tong_quat(A, Sigma)
        mo_cuoi = theta
        t1 = min(t0 + buoc, n)
        W[t0:t1] = theta[None, :, :]
    # dam dau chuoi: dung theta dau tien tinh duoc (khong co gi de noi suy hon)
    if mo_cuoi is not None:
        thieu = np.isnan(W[:, 0, 0])
        W[thieu] = mo_cuoi[None, :, :]
    return W, pairs


def spillover_ngoai_sinh(sig_pairs, dt_all, cap, buoc=BUOC, dam=DAM):
    """Cot ngoai sinh cho HAR: tai t, TONG bien dong cac cap KHAC (chuan hoa
    z-score theo lich su chinh no), TRONG SO boi ty le FEVD cap `cap` nhan tu
    cap do. Day la ban thay cho trung binh khong trong so cua `crosspair` cu.
    """
    pairs = list(sig_pairs)
    i = pairs.index(cap)
    W, _ = spillover_theo_thoi_gian(sig_pairs, dt_all, cap, buoc=buoc, dam=dam)
    n = len(dt_all)
    Z = np.zeros((n, len(pairs)))
    for j, p in enumerate(pairs):
        lv = np.log(np.maximum(sig_pairs[p], EPS) ** 2)
        mu = pd.Series(lv).expanding(min_periods=dam).mean().shift(1).values
        sd = pd.Series(lv).expanding(min_periods=dam).std().shift(1).values
        Z[:, j] = (lv - mu) / np.maximum(sd, EPS)
    trong_so_ngoai = W[:, i, :].copy()
    trong_so_ngoai[:, i] = 0.0                 # bo phan cap tu nhan tu chinh no
    tong = trong_so_ngoai.sum(1, keepdims=True)
    trong_so_ngoai = trong_so_ngoai / np.maximum(tong, EPS)
    return np.nan_to_num((trong_so_ngoai * Z).sum(1)), W


def main():
    t0 = time.time()
    from api.main import noi_chuoi, PAIRS

    print("=" * 100)
    print("LAN TRUYEN BIEN DONG (Diebold-Yilmaz) — thu lam bien ngoai sinh cho HAR")
    print("=" * 100)

    Ms, sigs, dts = {}, {}, {}
    for p in PAIRS:
        m = merge_thin_days(noi_chuoi(p))
        sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, p), 0.0))
        Ms[p] = m; sigs[p] = sig; dts[p] = m.Date.values

    # can theo NGAY CHUNG (giao cua ca 6 cap) de VAR dong bo duoc
    chung = None
    for p in PAIRS:
        s = pd.DatetimeIndex(dts[p])
        chung = s if chung is None else chung.intersection(s)
    chung = chung.sort_values()
    print(f"{len(chung):,} ngày chung cho cả 6 cặp\n")

    sig_can = {}
    for p in PAIRS:
        s = pd.Series(sigs[p], index=pd.DatetimeIndex(dts[p]))
        sig_can[p] = s.reindex(chung).values

    print(f"[1/2] Khớp VAR(1) cửa sổ mở rộng, khớp lại mỗi {BUOC} phiên…", flush=True)
    ma_tran_to_from = {}
    ngoai_sinh = {}
    for p in PAIRS:
        e, W = spillover_ngoai_sinh(sig_can, chung.values, p)
        ngoai_sinh[p] = e
        ma_tran_to_from[p] = W[-1, PAIRS.index(p), :].tolist()
    print("      xong\n")

    print("Ma trận TO/FROM tại lần khớp VAR gần nhất (hàng = cặp NHẬN, cột = cặp "
          "TRUYỀN — không phải trung bình theo thời gian, chỉ là ảnh chụp cuối kỳ):")
    print(f"{'':10}" + "".join(f"{p:>9}" for p in PAIRS))
    for p in PAIRS:
        print(f"{p:<10}" + "".join(f"{v:>9.3f}" for v in ma_tran_to_from[p]))

    # ── kiem tra bang HAR: them cot ngoai sinh, do QLIKE tren KIEM DINH ──
    print(f"\n[2/2] Đo QLIKE trên đoạn KIỂM ĐỊNH: không thêm cột ngoại sinh so với "
          f"thêm cột spillover Diebold-Yilmaz…", flush=True)
    print("      (Không đưa lịch NHTW vào cả hai bên để so sánh sạch — chỉ đo "
          "riêng tác dụng của cột spillover, không trộn với event=capday của "
          "sản xuất. Đây KHÔNG phải con số so với cấu hình sản xuất thật.)\n")

    ra = {"to_from": ma_tran_to_from, "pairs": PAIRS, "qlike_kiem_dinh": {}}
    for p in PAIRS:
        d = Ms[p]
        sig = sigs[p]
        lv = np.log(np.maximum(d.rv5.values, EPS))
        n = len(d)
        g_idx = doan(d.Date.values)
        tr = g_idx == 0
        va = g_idx == 1

        # ngoai sinh, can theo ngay CUA CAP p
        e_series = pd.Series(ngoai_sinh[p], index=chung)
        e_can = e_series.reindex(pd.DatetimeIndex(d.Date.values)).ffill().fillna(0.0).values

        for ten, extra in (("không thêm cột", None), ("spillover DY", [e_can])):
            X = V2.thiet_ke(d, lv, extra)
            y = np.empty(n); y[:-1] = lv[1:]; y[-1] = np.nan
            preds = {}
            for m in V2.MODELS:
                Xm = X[m]
                hople = np.isfinite(Xm).all(1) & np.isfinite(y)
                b, A, B, S, N = V2.he_so_cuon(Xm, y, hople, None)
                ssr = V2._ssr(b, A, B, S)
                s2 = np.where(N >= V2.MIN_FIT, ssr / np.maximum(N, 1), np.nan)
                fit = np.einsum("tk,tk->t", Xm, b)
                preds[m] = np.clip(fit, -30, 0) + 0.5 * np.maximum(s2, 0)
            g = np.stack([preds[m] for m in V2.MODELS]).mean(0)
            # deseason=none nen a_tg=0 — g da o dung thang RV, khong can cong lai gi
            f = np.where(np.isfinite(g), np.exp(g), np.nan)
            ql, nn = V2.qlike_tb(np.r_[np.nan, f[:-1]], d.rv5.values, va)
            ra["qlike_kiem_dinh"].setdefault(p, {})[ten] = {"qlike": ql, "n": nn}
        print(f"  {p:<8}" + "  ".join(
            f"{ten}={ra['qlike_kiem_dinh'][p][ten]['qlike']:.4f}"
            for ten in ("không thêm cột", "spillover DY")))

    tb = {ten: np.mean([ra["qlike_kiem_dinh"][p][ten]["qlike"] for p in PAIRS])
          for ten in ("không thêm cột", "spillover DY")}
    print(f"\n  TRUNG BÌNH 6 CẶP: không thêm cột {tb['không thêm cột']:.4f}  ·  "
          f"spillover DY {tb['spillover DY']:.4f}")
    ra["trung_binh"] = tb
    chenh = tb["spillover DY"] - tb["không thêm cột"]
    print(f"  → chênh lệch trung bình {chenh:+.6f} — " + (
        "quá nhỏ để gọi là cải thiện, không đủ để thay đổi cấu hình sản xuất"
        if abs(chenh) < 0.001 else
        ("SPILLOVER DY TỐT HƠN có thể xem xét" if chenh < 0 else "SPILLOVER DY TỆ HƠN")))

    with open(os.path.join(OUT, "spillover_dy.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/spillover_dy.json · {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


def main_h1():
    """Ban H1 — kiem tra gia thuyet 'lan truyen la hien tuong TRONG NGAY, da
    hoa tan khi gop ve D1' ma ban D1 tu ghi nhan la co the.

    Khong dung HAR vong 7 (danh cho D1, can rq5/rsp/rsn khong co o H1). Thay
    vao do dung chinh EWMA da khu mua vu cua `quyluat_h1.sigma_gio` lam du bao
    NEN, roi hoi quy log(r_t^2) ~ [1, log(sig_t^2)] (goc) so voi
    [1, log(sig_t^2), lan_truyen_t] (co them), CUA SO MO RONG (he_so_cuon —
    dung ham chung voi ban D1), cham QLIKE tren doan KIEM DINH.

    BUOC/DAM quy doi sang gio: ~1 thang (24*21) va ~2 thang (24*60) — VAR co
    k=6 bien, du du lieu on dinh voi vai tram quan sat, khong can dai nhu D1."""
    import quyluat_h1 as H1
    from api.main import PAIRS
    from split import VALID_TU, TEST_TU

    BUOC_H1 = 24 * 21
    DAM_H1 = 24 * 60
    t0 = time.time()
    print("=" * 100)
    print("LAN TRUYỀN BIẾN ĐỘNG (Diebold-Yilmaz) Ở H1 — kiểm gia thuyết 'hoà tan khi gộp D1'")
    print("=" * 100)

    D = H1.nap_h1(False)
    rs, sigs, gios, dts = {}, {}, {}, {}
    for p in PAIRS:
        d = D[p]
        c = d.close.values
        r = np.r_[np.nan, np.diff(np.log(np.maximum(c, EPS)))]
        gio = d.Date.dt.hour.values
        tr = (d.Date.values < np.datetime64(VALID_TU)) & np.isfinite(r)
        sig, _ = H1.sigma_gio(r, gio, tr)
        rs[p] = r; sigs[p] = sig; gios[p] = gio; dts[p] = d.Date.values

    chung = None
    for p in PAIRS:
        s = pd.DatetimeIndex(dts[p])
        chung = s if chung is None else chung.intersection(s)
    chung = chung.sort_values()
    print(f"{len(chung):,} giờ chung cho cả 6 cặp\n")

    sig_can = {p: pd.Series(sigs[p], index=pd.DatetimeIndex(dts[p])).reindex(chung).values
               for p in PAIRS}

    print(f"[1/2] Khớp VAR(1) cửa sổ mở rộng ở H1, khớp lại mỗi {BUOC_H1} giờ "
          f"(~1 tháng)…", flush=True)
    ngoai_sinh = {}
    for p in PAIRS:
        e, _ = spillover_ngoai_sinh(sig_can, chung.values, p, buoc=BUOC_H1, dam=DAM_H1)
        ngoai_sinh[p] = e
    print("      xong\n")

    print("[2/2] Đo QLIKE trên đoạn KIỂM ĐỊNH: EWMA gốc so với EWMA + lan truyền…\n",
          flush=True)
    ra = {"tam_han": "H1", "qlike_kiem_dinh": {}}
    for p in PAIRS:
        dt = dts[p]
        n = len(dt)
        r, sig = rs[p], sigs[p]
        y = np.log(np.maximum(r ** 2, EPS))
        va = (dt >= np.datetime64(VALID_TU)) & (dt < np.datetime64(TEST_TU))

        e_series = pd.Series(ngoai_sinh[p], index=chung)
        e_can = e_series.reindex(pd.DatetimeIndex(dt)).ffill().fillna(0.0).values

        for ten, extra_cols in (("EWMA gốc", []), ("EWMA + lan truyền", [e_can])):
            base = [np.ones(n), np.log(np.maximum(sig ** 2, EPS))]
            X = np.column_stack(base + extra_cols)
            hople = np.isfinite(X).all(1) & np.isfinite(y)
            b, A, B, S, N = V2.he_so_cuon(X, y, hople, None)
            ssr = V2._ssr(b, A, B, S)
            s2 = np.where(N >= 200, ssr / np.maximum(N, 1), np.nan)
            fit = np.einsum("tk,tk->t", X, b)
            f = np.where(np.isfinite(fit) & np.isfinite(s2),
                        np.exp(fit + 0.5 * np.maximum(s2, 0)), np.nan)
            ql, nn = V2.qlike_tb(f, r ** 2, va)
            ra["qlike_kiem_dinh"].setdefault(p, {})[ten] = {"qlike": ql, "n": nn}
        print(f"  {p:<8}" + "  ".join(
            f"{ten}={ra['qlike_kiem_dinh'][p][ten]['qlike']:.4f}"
            for ten in ("EWMA gốc", "EWMA + lan truyền")))

    tb = {ten: np.nanmean([ra["qlike_kiem_dinh"][p][ten]["qlike"] for p in PAIRS])
          for ten in ("EWMA gốc", "EWMA + lan truyền")}
    chenh = tb["EWMA + lan truyền"] - tb["EWMA gốc"]
    print(f"\n  TRUNG BÌNH 6 CẶP: gốc {tb['EWMA gốc']:.4f}  ·  "
          f"+ lan truyền {tb['EWMA + lan truyền']:.4f}  ·  chênh {chenh:+.6f}")
    ra["trung_binh"] = tb
    print("  → " + (
        "quá nhỏ để gọi là cải thiện" if abs(chenh) < 0.001 else
        ("CẢI THIỆN THẬT ở H1 — khác kết quả D1" if chenh < 0 else "TỆ HƠN, cùng chiều với D1")))

    with open(os.path.join(OUT, "spillover_dy_h1.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/spillover_dy_h1.json · {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main_h1() if "--h1" in sys.argv else main()
