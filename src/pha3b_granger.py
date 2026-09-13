"""PHA 3B / WEEK 2 — PHAT HIEN QUAN HE DAN BAO THEO THOI GIAN (3 muc).

Dac ta: `03_PHASE_3_CAUSALITY_AWARE.md` Week 2.
Khong gian gia thuyet (594), do dai khoi (5), so hoan vi (1.000), va nguong
do vung DA CHOT TRUOC o `docs/PHA3B_TIEUCHI.md` muc 2f / 4 (commit e88ae66).

  Level 1  huu ich du bao   — hoi quy long nhau, dieu kien tren MOC
  Level 2  Granger          — 11 bien x 3 lag x 6 cap x 3 truc = 594 phep kiem
                              + Westfall-Young maxT tung buoc xuong, null KHOI
  Level 3  PCMCI RUT GON    — PC-stable + MCI tuyen tinh, tu viet (tigramite
                              khong co trong moi truong). KHAI BAO THANG day la
                              ban rut gon, khong phai PCMCI day du.

BA TRUC, moi truc mot dinh nghia "past Y / baseline state" rieng — dung theo
cau hoi Level 2 cua HuyH ("beyond past Y / baseline state"):

  bien do :  y = log rv5(t+1)   | moc = log h_HAR(t)
  huong   :  y = zT(t+1)        | moc = zT(t), log sig(t)
  rui ro  :  y = |zT(t+1)|      | moc = log h_HAR(t), |zT(t)|

THONG KE KIEM DINH: F cua mo hinh long nhau (OLS). Null la HOAN VI KHOI nen
phu thuoc chuoi ngan han da duoc bao toan boi chinh null — dung dung mot thong
ke cho ca quan sat lan hoan vi. Rieng cot "raw statistic" bao cao them p tiem
can cua Wald HAC (Newey-West) de doi chieu, KHONG dung lam cua.

PHAT HIEN tren HUAN LUYEN + KIEM DINH. Doan kiem tra KHONG cham o Week 2.

Chay:  python src/pha3b_granger.py
Ghi:   output/pha3b_granger.json · output/pha3b_quanhe.csv
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")

import pha3b_dactrung as DT                                  # noqa: E402
import volfc2 as V2                                          # noqa: E402
from split import doan                                       # noqa: E402

KHOI = 5            # do dai khoi hoan vi — chot truoc muc 4d
NHOAN_VI = 1000     # chot truoc muc 4d
SEED = 20260912
EPS = 1e-12
TRUC = ("bien_do", "huong", "rui_ro")


# ──────────────────────────────────────────────────── bang du lieu 3 truc
def dung_bang():
    """Mot hang = (cap, phien t). Kem muc tieu 3 truc va moc cua tung truc."""
    bang, _ = V2.nap_bang()
    pan = pd.read_csv(os.path.join(DATA, "panel2_6pairs.csv"),
                      parse_dates=["Date"])
    hang = []
    for p in V2.PAIRS:
        d = bang[p]
        ng = pd.DatetimeIndex(d.Date)
        rv = np.maximum(d.rv5.values, EPS)
        h = V2.du_bao_san_xuat(d, p)
        lrv = np.log(rv)
        z = (pan[pan.pair == p].set_index("Date")
             .reindex(ng)[["zT", "sig"]])
        zT, sig = z.zT.values, z.sig.values
        for t in range(1, len(d) - 1):
            if not (np.isfinite(h[t]) and h[t] > 0 and np.isfinite(lrv[t + 1])):
                continue
            hang.append(dict(
                pair=p, ngay=ng[t], doan=int(doan([ng[t]])[0]),
                y_bien_do=float(lrv[t + 1]),
                m_har=float(np.log(h[t])),
                y_huong=float(zT[t + 1]) if np.isfinite(zT[t + 1]) else np.nan,
                m_zt=float(zT[t]) if np.isfinite(zT[t]) else np.nan,
                m_lsig=float(np.log(max(sig[t], EPS))) if np.isfinite(sig[t]) else np.nan,
                y_rui_ro=abs(float(zT[t + 1])) if np.isfinite(zT[t + 1]) else np.nan,
                m_azt=abs(float(zT[t])) if np.isfinite(zT[t]) else np.nan))
    df = pd.DataFrame(hang).sort_values(["pair", "ngay"]).reset_index(drop=True)
    X = DT.dung(df.ngay.values, df.pair.values)
    return pd.concat([df, X], axis=1)


MOC_TRUC = {"bien_do": ["m_har"], "huong": ["m_zt", "m_lsig"],
            "rui_ro": ["m_har", "m_azt"]}
Y_TRUC = {"bien_do": "y_bien_do", "huong": "y_huong", "rui_ro": "y_rui_ro"}


# ──────────────────────────────────────────────────────── may kiem dinh
def _rss(Q, Y):
    """Tong binh phuong du cua Y (n,P) chieu len phan bu truc giao cua Q."""
    return np.einsum("ij,ij->j", Y - Q @ (Q.T @ Y), Y - Q @ (Q.T @ Y))


def _Q(X):
    q, _ = np.linalg.qr(X)
    return q


def f_long_nhau(Q0, Q1, Y, q, n, k1):
    """Thong ke F cho viec THEM q cot, tinh dong thoi cho moi cot cua Y."""
    r0, r1 = _rss(Q0, Y), _rss(Q1, Y)
    return ((r0 - r1) / q) / np.maximum(r1 / (n - k1), EPS)


def wald_hac(X0, X1, y, q):
    """p tiem can cua Wald HAC (Newey-West) — cot 'raw statistic' doi chieu."""
    n, k = X1.shape
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    e = y - X1 @ b
    L = int(np.ceil(1.5 * n ** (1 / 3)))
    S = (X1 * e[:, None]).T @ (X1 * e[:, None])
    for l in range(1, L + 1):
        A = (X1[l:] * e[l:, None]).T @ (X1[:-l] * e[:-l, None])
        S += (1 - l / (L + 1)) * (A + A.T)
    XtXi = np.linalg.pinv(X1.T @ X1)
    V = XtXi @ S @ XtXi
    j = np.arange(X0.shape[1], k)                 # cac cot MOI them
    Rb, RV = b[j], V[np.ix_(j, j)]
    w = float(Rb @ np.linalg.pinv(RV) @ Rb)
    return w, float(1 - stats.chi2.cdf(w, q))


def hoan_vi_khoi(n, rng, khoi=KHOI):
    """Chi so hoan vi theo KHOI lien tiep — giu phu thuoc chuoi ngan han."""
    nb = int(np.ceil(n / khoi))
    pad = nb * khoi
    idx = np.concatenate([np.arange(n), np.full(pad - n, -1)]).reshape(nb, khoi)
    idx = idx[rng.permutation(nb)].ravel()
    return idx[idx >= 0]


# ──────────────────────────────────────────────────────────────── Level 1
def level1(df):
    """Huu ich du bao — GOP moi cap, dieu kien tren moc cua tung truc."""
    ra = []
    d = df[df.doan <= 1]
    cap = pd.get_dummies(d.pair, drop_first=True).values.astype(float)
    for truc in TRUC:
        yc, mc = Y_TRUC[truc], MOC_TRUC[truc]
        for bien in DT.BIEN_GOC:
            for L in DT.LAG:
                cols = DT.cot_lag(bien, L)
                s = d[[yc] + mc + cols]
                ok = s.notna().all(1).values
                if ok.sum() < 500:
                    continue
                y = d[yc].values[ok]
                X0 = np.column_stack([np.ones(ok.sum()), cap[ok]]
                                     + [d[c].values[ok] for c in mc])
                B = np.column_stack([d[c].values[ok] for c in cols])
                X1 = np.column_stack([X0, B])
                n, q = ok.sum(), B.shape[1]
                F = float(f_long_nhau(_Q(X0), _Q(X1), y[:, None], q, n,
                                      X1.shape[1])[0])
                p = float(1 - stats.f.cdf(F, q, n - X1.shape[1]))
                r0 = float(_rss(_Q(X0), y[:, None])[0])
                r1 = float(_rss(_Q(X1), y[:, None])[0])
                ra.append(dict(truc=truc, bien=bien, lag=L, n=int(n), F=F,
                               p_tho=p, r2_rieng=(r0 - r1) / max(r0, EPS)))
    return pd.DataFrame(ra)


# ──────────────────────────────────────────────────────────────── Level 2
def level2(df, rng):
    """594 phep kiem Granger theo TUNG CAP + Westfall-Young maxT tung buoc."""
    d = df[df.doan <= 1]
    hang, F_qs, F_hv = [], [], []
    for truc in TRUC:
        yc, mc = Y_TRUC[truc], MOC_TRUC[truc]
        for p in sorted(d.pair.unique()):
            s = d[d.pair == p]
            for bien in DT.BIEN_GOC:
                for L in DT.LAG:
                    cols = DT.cot_lag(bien, L)
                    sub = s[[yc] + mc + cols]
                    ok = sub.notna().all(1).values
                    n = int(ok.sum())
                    if n < 300:
                        continue
                    y = s[yc].values[ok]
                    X0 = np.column_stack([np.ones(n)]
                                         + [s[c].values[ok] for c in mc])
                    B = np.column_stack([s[c].values[ok] for c in cols])
                    # bo cot hang so trong khoi (vd lich khong co su kien)
                    gi = B.std(0) > 1e-10
                    B = B[:, gi]
                    if B.shape[1] == 0:
                        continue
                    X1 = np.column_stack([X0, B])
                    q, k1 = B.shape[1], X1.shape[1]
                    Q0, Q1 = _Q(X0), _Q(X1)
                    F = float(f_long_nhau(Q0, Q1, y[:, None], q, n, k1)[0])
                    # hoan vi KHOI cua muc tieu — cung thong ke, cung thiet ke
                    Y = np.column_stack([y[hoan_vi_khoi(n, rng)]
                                         for _ in range(NHOAN_VI)])
                    Fh = f_long_nhau(Q0, Q1, Y, q, n, k1)
                    # he so + dau, cot manh nhat trong khoi
                    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
                    e = y - X1 @ b
                    se = np.sqrt(np.diag(np.linalg.pinv(X1.T @ X1))
                                 * (e @ e) / (n - k1))
                    tb = b[X0.shape[1]:] / np.maximum(se[X0.shape[1]:], EPS)
                    im = int(np.argmax(np.abs(tb)))
                    ten_c = [c for c, g in zip(cols, gi) if g][im]
                    _, p_hac = wald_hac(X0, X1, y, q)
                    hang.append(dict(truc=truc, pair=p, bien=bien, lag=L, n=n,
                                     F=F, q=q,
                                     p_tiem_can=float(1 - stats.f.cdf(F, q, n - k1)),
                                     p_hac=p_hac,
                                     cot_manh=ten_c, he_so=float(b[X0.shape[1] + im]),
                                     t_manh=float(tb[im]),
                                     dau=int(np.sign(b[X0.shape[1] + im]))))
                    F_qs.append(F)
                    F_hv.append(Fh)
    F_qs = np.asarray(F_qs)
    F_hv = np.asarray(F_hv)                       # (m, NHOAN_VI)
    R = pd.DataFrame(hang)
    R["p_hoan_vi"] = (1 + (F_hv >= F_qs[:, None]).sum(1)) / (NHOAN_VI + 1)
    R["p_wy"] = wy_tung_buoc(F_qs, F_hv)
    R["q_fdr"] = fdr_bh(R.p_hoan_vi.values)
    # PHAN PHOI NULL CUA THONG KE MAX — phan tich luc (MDES) phai dung dung
    # phan phoi nay, khong duoc xap xi bang Sidak tren p tung gia thuyet.
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "pha3b_null_max.npy"), F_hv.max(0))
    return R


def wy_tung_buoc(F_qs, F_hv):
    """Westfall-Young maxT TUNG BUOC XUONG (step-down), khong phai mot buoc.

    Ban mot buoc bi gia thuyet manh nhat chiem het thong ke max — bai hoc da
    ghi o REPLAN_2026.md muc 1.2.
    """
    m = len(F_qs)
    od = np.argsort(-F_qs)                       # manh -> yeu
    Fs, Hs = F_qs[od], F_hv[od]
    p = np.empty(m)
    con = np.ones(m, bool)
    truoc = 0.0
    for i in range(m):
        mx = Hs[i:].max(0)                       # max tren cac gia thuyet CON LAI
        pi = (1 + (mx >= Fs[i]).sum()) / (Hs.shape[1] + 1)
        truoc = max(truoc, pi)                   # buoc don dieu
        p[i] = truoc
    ra = np.empty(m)
    ra[od] = p
    return ra


def fdr_bh(p):
    m = len(p)
    o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    ra = np.empty(m)
    ra[o] = np.minimum(q, 1.0)
    return ra


# ────────────────────────────────────────────────── do vung (muc 4e)
def do_vung(df, R):
    """Dau nhat quan xuyen cap va xuyen nam — nguong chot truoc 5/6 va 4/5."""
    d = df[df.doan <= 1].copy()
    d["nam"] = pd.DatetimeIndex(d.ngay).year
    ra = []
    for (truc, bien, L), g in R.groupby(["truc", "bien", "lag"]):
        dau = g.dau.values
        cap_thuan = int(max((dau > 0).sum(), (dau < 0).sum()))
        # on dinh theo nam: dau cua he so tren cot manh nhat, gop moi cap
        ten_c = g.cot_manh.mode().iat[0]
        yc, mc = Y_TRUC[truc], MOC_TRUC[truc]
        nam_dau = []
        for nam, s in d.groupby("nam"):
            sub = s[[yc, ten_c] + mc]
            ok = sub.notna().all(1).values
            if ok.sum() < 200:
                continue
            y = s[yc].values[ok]
            X = np.column_stack([np.ones(ok.sum())]
                                + [s[c].values[ok] for c in mc]
                                + [s[ten_c].values[ok]])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            nam_dau.append(np.sign(b[-1]))
        nam_dau = np.asarray(nam_dau)
        nam_thuan = int(max((nam_dau > 0).sum(), (nam_dau < 0).sum())) if len(nam_dau) else 0
        ra.append(dict(truc=truc, bien=bien, lag=L, cot_manh=ten_c,
                       cap_cung_dau=cap_thuan, n_cap=int(len(dau)),
                       nam_cung_dau=nam_thuan, n_nam=int(len(nam_dau)),
                       dat_cap=bool(cap_thuan >= 5),
                       dat_nam=bool(len(nam_dau) and
                                    nam_thuan >= max(4, int(0.8 * len(nam_dau))))))
    return pd.DataFrame(ra)


# ──────────────────────────────────────────────────────────────── Level 3
def level3_pcmci(df, rng, alpha=0.05, max_cha=3):
    """PCMCI RUT GON — PC-stable + MCI, tuong quan RIENG PHAN tuyen tinh.

    KHAI BAO THANG: day KHONG phai PCMCI day du cua Runge et al. Thieu kiem
    dinh doc lap phi tuyen, thieu xu ly tre dong thoi, va chi chay tren muc
    tieu bien do. HuyH Week 2 ghi ro PCMCI khong bat buoc — muc nay la BO SUNG.
    """
    d = df[df.doan <= 1]
    ung_vien = [f"{c}_ad1_L1" for c in DT.CHUOI_TT] + \
               [f"{c}_lv_L1" for c in DT.CHUOI_TT]
    ra = []
    for p in sorted(d.pair.unique()):
        s = d[d.pair == p]
        cols = [c for c in ung_vien if c in s.columns]
        sub = s[["y_bien_do", "m_har"] + cols]
        ok = sub.notna().all(1).values
        if ok.sum() < 500:
            continue
        y = s.y_bien_do.values[ok]
        Z0 = s.m_har.values[ok]                      # LUON dieu kien tren moc
        Xs = {c: s[c].values[ok] for c in cols}

        # ── PC-stable: loai dan ung vien bang tuong quan rieng phan
        cha = list(cols)
        dk = []
        for buoc in range(max_cha + 1):
            bo = []
            for c in cha:
                dieu_kien = [Z0] + [Xs[z] for z in dk if z != c][:buoc]
                r, pv = _tuong_quan_rieng(Xs[c], y, dieu_kien)
                if pv > alpha:
                    bo.append(c)
            cha = [c for c in cha if c not in bo]
            if not cha:
                break
            dk = sorted(cha, key=lambda c: -abs(
                _tuong_quan_rieng(Xs[c], y, [Z0])[0]))[:max_cha]
        # ── MCI: dieu kien tren moc + tap cha da chon
        for c in cha:
            dieu_kien = [Z0] + [Xs[z] for z in cha if z != c][:max_cha]
            r, pv = _tuong_quan_rieng(Xs[c], y, dieu_kien)
            ra.append(dict(pair=p, bien=c, r_mci=float(r), p_mci=float(pv),
                           n=int(ok.sum())))
    return pd.DataFrame(ra)


def _tuong_quan_rieng(x, y, dieu_kien):
    """Tuong quan rieng phan cua x,y cho truoc tap dieu kien."""
    Z = np.column_stack([np.ones(len(x))] + list(dieu_kien)) if dieu_kien \
        else np.ones((len(x), 1))
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    ex, ey = x - Z @ bx, y - Z @ by
    n, k = len(x), Z.shape[1]
    r = float(np.corrcoef(ex, ey)[0, 1]) if ex.std() > 0 and ey.std() > 0 else 0.0
    dof = max(n - k - 1, 1)
    t = r * np.sqrt(dof / max(1 - r * r, EPS))
    return r, float(2 * (1 - stats.t.cdf(abs(t), dof)))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("=" * 104)
    print("PHA 3B / WEEK 2 — PHÁT HIỆN QUAN HỆ DẪN BÁO THEO THỜI GIAN")
    print("tiêu chí chốt trước: docs/PHA3B_TIEUCHI.md mục 2f/4 (commit e88ae66)")
    print("=" * 104)

    df = dung_bang()
    pt = df[df.doan <= 1]
    print(f"bảng {len(df):,} hàng · {df.pair.nunique()} cặp · "
          f"{df.ngay.min().date()} → {df.ngay.max().date()}")
    print(f"phát hiện trên HUẤN LUYỆN+KIỂM ĐỊNH: {len(pt):,} hàng "
          f"({pt.ngay.nunique():,} phiên) — đoạn kiểm tra KHÔNG chạm ở Week 2")

    # ── LEVEL 1
    print("\n" + "=" * 104)
    print("LEVEL 1 — HỮU ÍCH DỰ BÁO (hồi quy lồng nhau, gộp cặp, điều kiện trên mốc)")
    print("=" * 104)
    L1 = level1(df)
    print(f"{'trục':<10}{'biến':<22}{'lag':>4}{'n':>8}{'F':>9}{'p thô':>10}"
          f"{'R² riêng':>11}")
    print("-" * 104)
    for truc in TRUC:
        s = L1[L1.truc == truc].nsmallest(3, "p_tho")
        for _, r in s.iterrows():
            print(f"{r.truc:<10}{r.bien:<22}{int(r.lag):>4}{int(r.n):>8,}"
                  f"{r.F:>9.2f}{r.p_tho:>10.4f}{100*r.r2_rieng:>10.3f}%")
    print("-" * 104)
    print(f"  (3 dòng p thô nhỏ nhất mỗi trục, trong {len(L1)} tổ hợp — "
          f"CHƯA hiệu chỉnh bội)")

    # ── LEVEL 2
    print("\n" + "=" * 104)
    print(f"LEVEL 2 — GRANGER, {NHOAN_VI} hoán vị KHỐI {KHOI} phiên, "
          f"Westfall–Young maxT từng bước xuống")
    print("=" * 104)
    R = level2(df, rng)
    print(f"số phép kiểm thực hiện: {len(R)} "
          f"(khai báo trước: 11 biến × 3 lag × 6 cặp × 3 trục = 594)")
    ng = dict(p_tiem_can=0.05, p_hoan_vi=0.05, q_fdr=0.05, p_wy=0.05)
    print(f"\n{'cửa':<26}{'số sống sót':>14}{'ghi chú'}")
    print("-" * 104)
    print(f"{'p tiệm cận thô < 0,05':<26}{int((R.p_tiem_can < .05).sum()):>14}"
          f"      kỳ vọng dưới nhiễu thuần ≈ {0.05*len(R):.0f}")
    print(f"{'p hoán vị khối < 0,05':<26}{int((R.p_hoan_vi < .05).sum()):>14}"
          f"      chưa hiệu chỉnh bội")
    print(f"{'FDR-BH q < 0,05':<26}{int((R.q_fdr < .05).sum()):>14}"
          f"      tham chiếu, KHÔNG dùng làm cửa")
    n_wy = int((R.p_wy < .05).sum())
    print(f"{'Westfall–Young p < 0,05':<26}{n_wy:>14}      **CỬA QUYẾT ĐỊNH**")
    print("-" * 104)

    print(f"\n10 quan hệ mạnh nhất (theo F):")
    print(f"  {'trục':<9}{'cặp':<8}{'biến':<16}{'lag':>4}{'F':>8}"
          f"{'p hoán vị':>11}{'p W-Y':>9}{'q FDR':>8}  {'cột mạnh nhất':<22}{'dấu':>5}")
    print("  " + "-" * 100)
    for _, r in R.nlargest(10, "F").iterrows():
        print(f"  {r.truc:<9}{r.pair:<8}{r.bien:<16}{int(r.lag):>4}{r.F:>8.2f}"
              f"{r.p_hoan_vi:>11.4f}{r.p_wy:>9.4f}{r.q_fdr:>8.4f}  "
              f"{r.cot_manh:<22}{'+' if r.dau > 0 else '−':>5}")

    # ── DO VUNG
    print("\n" + "=" * 104)
    print("MÀN LỌC ĐỘ VỮNG (mục 4e) — ngưỡng chốt trước: ≥5/6 cặp cùng dấu, "
          "≥4/5 năm cùng dấu")
    print("=" * 104)
    V = do_vung(df, R)
    V = V.merge(R.groupby(["truc", "bien", "lag"]).p_wy.min().rename("p_wy_min"),
                on=["truc", "bien", "lag"])
    V["dat_het"] = V.dat_cap & V.dat_nam
    print(f"tổ hợp (trục × biến × lag): {len(V)} · "
          f"đạt độ vững cặp+năm: {int(V.dat_het.sum())}")
    print(f"\n  {'trục':<9}{'biến':<16}{'lag':>4}{'cặp cùng dấu':>14}"
          f"{'năm cùng dấu':>14}{'p W-Y nhỏ nhất':>16}  đạt")
    print("  " + "-" * 92)
    for _, r in V.nsmallest(8, "p_wy_min").iterrows():
        print(f"  {r.truc:<9}{r.bien:<16}{int(r.lag):>4}"
              f"{r.cap_cung_dau:>8}/{r.n_cap:<5}{r.nam_cung_dau:>8}/{r.n_nam:<5}"
              f"{r.p_wy_min:>16.4f}  {'✓' if r.dat_het else '—'}")

    # ── LEVEL 3
    print("\n" + "=" * 104)
    print("LEVEL 3 — PCMCI RÚT GỌN (PC-stable + MCI tuyến tính, tự viết)")
    print("  KHAI BÁO: bản rút gọn, KHÔNG phải PCMCI đầy đủ. HuyH Week 2: "
          "không bắt buộc.")
    print("=" * 104)
    P3 = level3_pcmci(df, rng)
    if len(P3):
        n_cha = P3.groupby("pair").size()
        print(f"số cha được giữ sau PC-stable + MCI, theo cặp: "
              f"{dict(n_cha)}")
        print(f"\n  {'cặp':<9}{'biến':<24}{'r MCI':>9}{'p MCI':>10}{'n':>8}")
        print("  " + "-" * 62)
        for _, r in P3.nsmallest(10, "p_mci").iterrows():
            print(f"  {r.pair:<9}{r.bien:<24}{r.r_mci:>+9.4f}{r.p_mci:>10.4f}"
                  f"{int(r.n):>8,}")
    else:
        print("  KHÔNG biến nào sống sót PC-stable — tập cha rỗng ở mọi cặp.")

    # ── UNG VIEN CHO E3
    song = R[(R.p_wy < 0.05)]
    vung = V[V.dat_het]
    ung = song.merge(vung[["truc", "bien", "lag"]], on=["truc", "bien", "lag"])
    print("\n" + "=" * 104)
    print("TẬP ỨNG VIÊN CHO E3 (qua W-Y **và** qua màn lọc độ vững)")
    print("=" * 104)
    print(f"  qua W-Y: {len(song)} · qua độ vững: {len(vung)} tổ hợp · "
          f"**giao: {len(ung)}**")
    e3 = sorted({c for _, r in ung.iterrows()
                 for c in DT.cot_lag(r.bien, int(r.lag))})
    if e3:
        print(f"  đặc trưng vào E3 ({len(e3)}): {', '.join(e3[:12])}"
              f"{' …' if len(e3) > 12 else ''}")
    else:
        print("  → E3 = TẬP RỖNG. Không quan hệ nào qua được cả hai cửa.")

    os.makedirs(OUT, exist_ok=True)
    R.to_csv(os.path.join(OUT, "pha3b_quanhe.csv"), index=False)
    json.dump(dict(
        n_kiem=int(len(R)), khai_bao=594, n_hoan_vi=NHOAN_VI, khoi=KHOI,
        song_sot={"p_tiem_can_005": int((R.p_tiem_can < .05).sum()),
                  "p_hoan_vi_005": int((R.p_hoan_vi < .05).sum()),
                  "fdr_005": int((R.q_fdr < .05).sum()),
                  "wy_005": int(n_wy)},
        do_vung_dat=int(V.dat_het.sum()),
        e3_dac_trung=e3,
        level1_top={t: L1[L1.truc == t].nsmallest(3, "p_tho")
                    .to_dict("records") for t in TRUC},
        level3_n=int(len(P3)),
        level3_top=P3.nsmallest(5, "p_mci").to_dict("records") if len(P3) else []),
        open(os.path.join(OUT, "pha3b_granger.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/pha3b_granger.json · output/pha3b_quanhe.csv · "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
