"""qlikeHAR — KHOP HAR BANG CHINH HAM MAT DUNG DE CHAM DIEM.

NGUON: Puke & Schweikert (2026), "Coherent Forecasting of Realized Volatility",
Journal of Forecasting 45(1). SSRN 5233349. Ma tai lap:
github.com/marius-cp/qlikeHAR_replications (`00_functions/qlikeHAR.R`).

Y CHINH cua bai: van lieu danh gia du bao bien dong bang QLIKE, nhung lai UOC
LUONG HAR bang binh phuong nho nhat (MSE). Do la mot su KHONG KHOP. Khop lai
cho dung — uoc luong bang chinh QLIKE — cho "massive forecast performance
gains" tren du lieu cua ho.

VI SAO DANG THU TREN REPO NAY — khong phai vi bai bao moi, ma vi repo DA CO
BANG CHUNG RIENG cung chieu:

    "LightGBM voi ham muc tieu QLIKE tu viet dat 0,1593 so voi 0,1618 cua ban
     L2 thong thuong — tot hon 1,5% chi nho doi ham mat cho khop voi thuoc do
     danh gia."                                    (docs/ML_DL_VONG7.md, muc 4)

Tuc repo da chung minh dieu do CHO CAY TANG CUONG, roi khong bao gio ap cho
chinh HAR — mo hinh dang chay san xuat. Day la lo hong repo tu tao ra.

BON CAU HINH — CHOT TRUOC, khong them, khong bot:

  B0     moc san xuat: OLS tren log-RV, khop MOI PHIEN, hieu chinh log-chuan
         (`volfc2.du_bao_san_xuat`) — cau hinh da dong bang o KHOA_SO.md muc 4
  A0     Y HET B0 nhung khop lai DAU MOI NAM — de tach bach anh huong cua TAN
         SUAT KHOP ra khoi anh huong cua HAM MAT. Khong co dong nay thi moi
         chenh lech deu khong quy duoc ve dau
  Q_log  cung ma tran thiet ke, cung tan suat khop nhu A0, nhung he so chon de
         CUC TIEU QLIKE truc tiep voi h = exp(X beta)
  Q_lvl  ban SAT BAI BAO: tuyen tinh tren THANG MUC, h = X beta, cuc tieu QLIKE

QLIKE bo hang so:  S(h, y) = log(h) + y/h      (dung ham cua bai, qlikeHAR.R)

Q_log GIAI BANG NEWTON, khong phai Nelder-Mead. Ly do: voi h = exp(X beta),

    dS/dbeta  = X' (1 - y e^{-X beta})
    d2S/dbeta2 = X' diag(y e^{-X beta}) X          (nua xac dinh duong khi y>0)

nen bai toan LOI — Newton cho nghiem toan cuc trong vai vong lap. Day dung la
cap gradient/hessian ma repo DA tu dan cho LightGBM (`ML_DL_VONG7.md`), nay
dung lai cho HAR. Q_lvl thi khong loi (X beta phai duong) nen dung Nelder-Mead
tu nghiem OLS, y nhu ma R cua tac gia.

GIAO THUC: khop tren HUAN LUYEN · chon tren KIEM DINH · cham MOT LAN tren
KIEM TRA. Diebold-Mariano + Newey-West, va Model Confidence Set — cung bo may
da dung cho 14 mo hinh vong 7, de con so so sanh duoc truc tiep.

Chay:  python src/kiem_qlikehar.py
Ghi:   output/qlikehar.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import optimize

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import volfc2 as V2                                          # noqa: E402
from split import doan                                       # noqa: E402
from run_final7 import dm_nw                                 # noqa: E402
from metrics import mcs                                      # noqa: E402

EPS = 1e-12
MIN_FIT = 60          # so hang toi thieu de khop mot lan
NEWTON_LAP = 50
NEWTON_TOL = 1e-9


# ──────────────────────────────────────────────────────── ham mat
def qlike(y, h):
    """QLIKE bat bien thang do (Bregman): r - log r - 1, r = y/h."""
    r = np.maximum(y, EPS) / np.maximum(h, EPS)
    return r - np.log(np.maximum(r, EPS)) - 1.0


def S_qlike(m, y):
    """Ham mat cua bai (bo hang so), m = log h:  log(h) + y/h."""
    return m + y * np.exp(-m)


# ─────────────────────────────────────────── Q_log: Newton (bai toan LOI)
def khop_qlog(X, y, b0=None):
    """beta cuc tieu sum[ X beta + y e^{-X beta} ].  Loi -> Newton."""
    n, k = X.shape
    b = np.zeros(k) if b0 is None else b0.astype(float).copy()
    for _ in range(NEWTON_LAP):
        m = np.clip(X @ b, -50, 50)
        w = y * np.exp(-m)                       # = d2/dm2, luon >= 0
        g = X.T @ (1.0 - w)                      # gradient
        H = (X * w[:, None]).T @ X               # hessian, PSD
        try:
            step = np.linalg.solve(H + 1e-8 * np.eye(k), g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H + 1e-6 * np.eye(k), g, rcond=None)[0]
        # lui buoc: dam bao ham muc tieu khong tang
        f0 = S_qlike(m, y).sum()
        t = 1.0
        for _ in range(30):
            bn = b - t * step
            fn = S_qlike(np.clip(X @ bn, -50, 50), y).sum()
            if np.isfinite(fn) and fn <= f0:
                break
            t *= 0.5
        b = bn
        if np.max(np.abs(t * step)) < NEWTON_TOL:
            break
    return b


def _tu_kiem_newton():
    """Du lieu mo phong co nghiem biet truoc -> Newton phai tim lai duoc."""
    rng = np.random.default_rng(0)
    n = 4000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    b_that = np.array([-9.0, 0.55, -0.20])
    m = X @ b_that
    y = np.exp(m) * rng.gamma(shape=5.0, scale=1 / 5.0, size=n)   # E[y]=e^m
    b = khop_qlog(X, y)
    lech = float(np.max(np.abs(b - b_that)))
    # va: QLIKE cua nghiem Newton phai <= cua OLS tren log
    bo = np.linalg.lstsq(X, np.log(y), rcond=None)[0]
    q_n = S_qlike(X @ b, y).mean()
    q_o = S_qlike(X @ bo, y).mean()
    return lech, float(q_n), float(q_o)


# ──────────────────────────────────── Q_lvl: ban sat bai bao (thang muc)
def khop_qlvl(X, y, b0):
    """h = X beta tren THANG MUC. Khong loi -> Nelder-Mead tu OLS, nhu ma R."""
    def f(b):
        h = X @ b
        if not np.all(np.isfinite(h)) or np.any(h <= 0):
            return 1e12
        return float(np.sum(np.log(h) + y / h))
    r = optimize.minimize(f, b0, method="Nelder-Mead",
                          options=dict(maxiter=20000, fatol=1e-10, xatol=1e-10))
    return r.x


# ────────────────────────────────────────────────── dung bang mot cap
def bang_cap(d, pair):
    """Ma tran thiet ke + muc tieu, DUNG HET cua volfc2 — khong viet lai."""
    n = len(d)
    ngay = pd.DatetimeIndex(d.Date)
    lich = V2.nap_lich(ngay)
    lv = np.log(np.maximum(d.rv5.values, EPS))
    extra = V2._cot_su_kien(pair, lich, n, V2.CAUHINH_SANXUAT["event"])
    X = V2.thiet_ke(d, lv, extra or None)
    y_log = np.empty(n); y_log[:-1] = lv[1:]; y_log[-1] = np.nan
    y_rv = np.empty(n); y_rv[:-1] = np.maximum(d.rv5.values[1:], EPS); y_rv[-1] = np.nan
    gap = pd.Series(ngay).diff().dt.days.values.astype(float); gap[0] = 1
    lien = np.zeros(n, bool); lien[1:] = gap[1:] <= V2.MAX_GAP
    return X, y_log, y_rv, ngay, lien


def du_bao_theo_nam(X, y_log, y_rv, ngay, lien, kieu):
    """Khop lai DAU MOI NAM, cua so mo rong. kieu: 'ols' | 'qlog' | 'qlvl'.

    Tra ve mang du bao PHUONG SAI cho tung ngay (NaN neu chua du dam).
    """
    n = len(y_log)
    nam = ngay.year.values
    L = []
    for m in V2.MODELS:
        Xm = X[m]
        hople = np.isfinite(Xm).all(1) & np.isfinite(y_log) & np.isfinite(y_rv)
        fit = np.full(n, np.nan)
        for u in sorted(set(nam)):
            tr = (nam < u) & hople                 # cua so mo rong, NHAN QUA
            if tr.sum() < max(MIN_FIT, V2.MIN_TRAIN):
                continue
            ap = (nam == u) & np.isfinite(Xm).all(1)
            if not ap.any():
                continue
            Xtr, ytr_log, ytr_rv = Xm[tr], y_log[tr], y_rv[tr]
            b_ols = np.linalg.lstsq(Xtr, ytr_log, rcond=None)[0]
            if kieu == "ols":
                s2 = float(np.var(ytr_log - Xtr @ b_ols))
                fit[ap] = np.clip(Xm[ap] @ b_ols, -30, 0) + 0.5 * s2
            elif kieu == "qlog":
                b = khop_qlog(Xtr, ytr_rv, b_ols)
                fit[ap] = np.clip(Xm[ap] @ b, -30, 0)      # KHONG hieu chinh:
                #   ham mat da toi uu truc tiep cho chinh h = exp(X beta)
            elif kieu == "qlvl":
                h0 = np.exp(Xtr @ b_ols)
                bl0 = np.linalg.lstsq(Xtr, h0, rcond=None)[0]
                b = khop_qlvl(Xtr, ytr_rv, bl0)
                h = Xm[ap] @ b
                fit[ap] = np.log(np.maximum(h, EPS))
        L.append(fit)
    g = np.nanmean(np.stack(L), axis=0)            # trung binh hinh hoc, nhu B0
    ok = np.isfinite(g)
    out = np.full(n, np.nan)
    src = np.where(ok)[0]; tgt = src + 1
    v = tgt < n
    src, tgt = src[v], tgt[v]
    v2 = lien[tgt]
    out[tgt[v2]] = np.exp(np.clip(g[src[v2]], -30, 5))
    return out


def main():
    t0 = time.time()
    print("=" * 100)
    print("qlikeHAR — KHỚP HAR BẰNG CHÍNH HÀM MẤT DÙNG ĐỂ CHẤM ĐIỂM")
    print("nguồn: Puke & Schweikert (2026), J. Forecasting · SSRN 5233349")
    print("=" * 100)

    lech, q_n, q_o = _tu_kiem_newton()
    print(f"TỰ KIỂM Newton trên dữ liệu mô phỏng có nghiệm biết trước:")
    print(f"  |lệch hệ số| lớn nhất = {lech:.4f}   "
          f"{'ĐẠT' if lech < 0.05 else 'HỎNG'}")
    print(f"  QLIKE(Newton) = {q_n:.5f}  ≤  QLIKE(OLS trên log) = {q_o:.5f}   "
          f"{'ĐẠT' if q_n <= q_o else 'HỎNG'}")
    assert lech < 0.05 and q_n <= q_o, "Newton sai — dung"

    bang, _ = V2.nap_bang()
    hang = []
    print(f"\nchạy {len(V2.PAIRS)} cặp…")
    for p in V2.PAIRS:
        d = bang[p]
        X, y_log, y_rv, ngay, lien = bang_cap(d, p)
        hs = {"B0": V2.du_bao_san_xuat(d, p)}
        for ten, kieu in (("A0", "ols"), ("Q_log", "qlog"), ("Q_lvl", "qlvl")):
            hs[ten] = du_bao_theo_nam(X, y_log, y_rv, ngay, lien, kieu)
        g = doan(ngay.values)
        rv = np.maximum(d.rv5.values, EPS)
        for t in range(len(d)):
            if not np.isfinite(rv[t]):
                continue
            r = dict(pair=p, ngay=ngay[t], doan=int(g[t]), rv=float(rv[t]))
            for k, v in hs.items():
                r[k] = float(v[t]) if np.isfinite(v[t]) else np.nan
            hang.append(r)
        print(f"  {p} xong ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(hang)
    CH = ["B0", "A0", "Q_log", "Q_lvl"]
    df = df[df[CH].notna().all(1)]
    print(f"\nbảng chung {len(df):,} hàng (mọi cấu hình đều có dự báo)")
    for i, ten in enumerate(("huấn luyện", "kiểm định", "kiểm tra")):
        print(f"  {ten:<12}{int((df.doan == i).sum()):>7,}")

    ql = {k: qlike(df.rv.values, df[k].values) for k in CH}
    va, te = (df.doan == 1).values, (df.doan == 2).values

    print("\n" + "=" * 100)
    print(f"{'cấu hình':<8}{'mô tả':<34}{'QLIKE kđ':>11}{'so B0':>8}"
          f"{'QLIKE kt':>11}{'so B0':>8}{'DM p (kt)':>11}")
    print("-" * 100)
    MO_TA = {"B0": "mốc sản xuất (OLS log, khớp phiên)",
             "A0": "y hệt B0, khớp theo năm",
             "Q_log": "QLIKE trực tiếp, h = exp(Xβ)",
             "Q_lvl": "QLIKE trực tiếp, thang mức (bài báo)"}
    kq = {}
    for k in CH:
        v, t_ = float(ql[k][va].mean()), float(ql[k][te].mean())
        dv = (v / ql["B0"][va].mean() - 1) * 100
        dt_ = (t_ / ql["B0"][te].mean() - 1) * 100
        p = np.nan if k == "B0" else dm_nw(ql[k][te] - ql["B0"][te])[1]
        kq[k] = dict(qlike_vd=v, qlike_kt=t_, d_vd=dv, d_kt=dt_,
                     dm_p=float(p) if np.isfinite(p) else None)
        print(f"{k:<8}{MO_TA[k]:<34}{v:>11.4f}{dv:>7.2f}%{t_:>11.4f}"
              f"{dt_:>7.2f}%{p:>11.4f}")
    print("-" * 100)
    print("  (âm = TỐT HƠN mốc sản xuất)")

    # tach bach: ham mat dong gop bao nhieu, TAN SUAT KHOP bao nhieu
    d_tan_so = (ql["A0"][te].mean() / ql["B0"][te].mean() - 1) * 100
    d_ham_mat = (ql["Q_log"][te].mean() / ql["A0"][te].mean() - 1) * 100
    _, p_hm = dm_nw(ql["Q_log"][te] - ql["A0"][te])
    print(f"\nTÁCH BẠCH HAI NGUỒN CHÊNH LỆCH (đoạn kiểm tra):")
    print(f"  do TẦN SUẤT KHỚP  (A0 so B0)    {d_tan_so:+.2f}%")
    print(f"  do HÀM MẤT        (Q_log so A0) {d_ham_mat:+.2f}%  ·  DM p={p_hm:.4f}")

    print(f"\nTHEO TỪNG CẶP (đoạn kiểm tra, QLIKE):")
    print(f"  {'cặp':<9}{'B0':>10}{'A0':>10}{'Q_log':>10}{'Q_lvl':>10}"
          f"{'Q_log so B0':>13}{'DM p':>9}")
    print("  " + "-" * 71)
    duong, theo_cap = 0, {}
    for p in sorted(df.pair.unique()):
        m = te & (df.pair.values == p)
        a = ql["B0"][m].mean()
        r = {k: float(ql[k][m].mean()) for k in CH}
        ch = (r["Q_log"] / a - 1) * 100
        _, pp = dm_nw(ql["Q_log"][m] - ql["B0"][m])
        duong += int(ch < 0)
        theo_cap[p] = dict(chenh=float(ch), dm_p=float(pp))
        print(f"  {p:<9}{a:>10.4f}{r['A0']:>10.4f}{r['Q_log']:>10.4f}"
              f"{r['Q_lvl']:>10.4f}{ch:>12.2f}%{pp:>9.4f}")
    print("  " + "-" * 71)
    print(f"  → Q_log cải thiện {duong}/6 cặp")

    # Model Confidence Set — cung bo may da dung cho 14 mo hinh vong 7
    try:
        L = np.column_stack([ql[k][te] for k in CH])
        song = mcs(L, alpha=0.10)
        ten_song = [CH[i] for i in np.where(song)[0]] if song is not None else None
    except Exception as e:
        ten_song = f"(không chạy được: {type(e).__name__})"
    print(f"\nModel Confidence Set (α = 0,10) trên đoạn kiểm tra: {ten_song}")

    print(f"\nTHEO CHẾ ĐỘ BIẾN ĐỘNG (ngũ phân vị σ̂ mốc, ngưỡng từ huấn luyện):")
    tr = (df.doan == 0).values
    q5 = np.nanquantile(np.log(df.B0.values[tr]), [.2, .4, .6, .8])
    cd = np.digitize(np.log(df.B0.values), q5)
    print(f"  {'chế độ':<9}{'n':>7}{'B0':>10}{'Q_log':>10}{'chênh':>9}")
    print("  " + "-" * 45)
    theo_che_do = {}
    for j in range(5):
        m = te & (cd == j)
        if m.sum() < 30:
            continue
        a, b = ql["B0"][m].mean(), ql["Q_log"][m].mean()
        theo_che_do[f"Q{j+1}"] = float((b / a - 1) * 100)
        print(f"  {'Q'+str(j+1)+(' êm' if j == 0 else ' căng' if j == 4 else ''):<9}"
              f"{int(m.sum()):>7,}{a:>10.4f}{b:>10.4f}{(b/a-1)*100:>8.2f}%")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(ket_qua=kq, theo_cap=theo_cap, cap_duong=duong,
                   theo_che_do=theo_che_do, mcs=ten_song,
                   tach_bach=dict(tan_suat_khop=float(d_tan_so),
                                  ham_mat=float(d_ham_mat),
                                  ham_mat_dm_p=float(p_hm)),
                   tu_kiem=dict(lech_he_so=lech, qlike_newton=q_n,
                                qlike_ols=q_o),
                   nguon="Puke & Schweikert 2026, J. Forecasting, SSRN 5233349"),
              open(os.path.join(OUT, "qlikehar.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/qlikehar.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
