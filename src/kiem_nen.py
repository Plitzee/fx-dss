"""MAU HINH NEN — hinh hoc OHLC ngay co them gi ngoai tang 2 khong?

Dac trung, nguong, khong gian gia thuyet va tieu chi phan quyet DA CHOT TRUOC
o `docs/NEN_TIEUCHI.md` (commit 9643f80), truoc khi cham bat ky so lieu nao.

LO HONG DUOC NHAM TOI. Repo da khai pha 8.652 gia thuyet qua 12 nhanh nhung
MOI nhanh deu dung chuoi dong-dong hoac dai luong tu nen 5 phut. Ma tran thiet
ke san xuat (`volfc2.thiet_ke`) dung rv5/rq5/bpv5/rsp/rsn — KHONG cot OHLC nao.

BA CO CHE (muc 0b cua bien ban), co the sai, do duoc:
  1. rv5 hut cuc tri TRONG thanh 5 phut; high/low ngay thi khong
  2. gap qua dem bi loai theo thiet ke (DATASET.md: "chi 1,7-3,1% nen bo qua
     duoc" — mot gia dinh chua ai kiem)
  3. vi tri dong trong pham vi la thong tin THU TU — rv5/bpv5/rsp/rsn deu bat
     bien voi hoan vi thu tu loi suat trong ngay, (C-L)/(H-L) thi KHONG.
     Day la co che duy nhat khong bi lap luan "nhay tra hinh" cua
     PHA2_KETQUA muc 3d bac truoc.

Dung lai BO MAY DA KIEM cua Pha 3B (`pha3b_granger`): kiem dinh F long nhau,
hoan vi khoi, Westfall-Young maxT tung buoc xuong. Khong viet lai — de con so
xep chung duoc mot bang.

Chay:  python src/kiem_nen.py
Ghi:   output/nen.json · output/nen_quanhe.csv
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

import volfc2 as V2                                          # noqa: E402
import pha3b_granger as G                                    # noqa: E402
from split import doan                                       # noqa: E402
from run_final7 import dm_nw                                 # noqa: E402

EPS = 1e-12
LAG = [1, 2]                                                 # chot truoc
KHOI = G.KHOI
NHOAN_VI = G.NHOAN_VI
SEED = 20260912

K1 = ["than", "bong_tren", "bong_duoi", "vi_tri_dong", "huong"]
K2 = ["lr_park", "lr_gk", "lr_rs"]
K3 = ["doji", "bua", "sao_bang", "marubozu", "con_quay",
      "nhan_chim_tang", "nhan_chim_giam", "harami_tang", "harami_giam",
      "xuyen_tham", "may_den",
      "sao_mai", "sao_hom", "ba_linh_trang", "ba_qua_den"]
K4 = ["gap", "abs_gap"]
HO = {"K1 hình học": K1, "K2 phạm vi−RV": K2, "K3 mẫu có tên": K3,
      "K4 gap đêm": K4}
TAT_CA = K1 + K2 + K3 + K4                                   # 25, chot truoc


def dac_trung_nen(d):
    """25 dac trung tu OHLC ngay. Moi gia tri chi dung nen cua CHINH phien do;
    viec lui lag do `dung_bang` lam, nen o day khong co ro ri."""
    o = d.open.values.astype(float)
    h = d.high.values.astype(float)
    l = d.low.values.astype(float)
    c = d.close.values.astype(float)
    rv = np.maximum(d.rv5.values.astype(float), EPS)
    rng = np.maximum(h - l, EPS)
    than = np.abs(c - o)
    F = {}

    # ── K1 hinh hoc chuan hoa (vo thu nguyen)
    F["than"] = than / rng
    F["bong_tren"] = (h - np.maximum(o, c)) / rng
    F["bong_duoi"] = (np.minimum(o, c) - l) / rng
    F["vi_tri_dong"] = (c - l) / rng
    F["huong"] = np.sign(c - o)

    # ── K2 uoc luong theo pham vi so voi rv5 (co che 1)
    LN2 = np.log(2.0)
    park = (np.log(np.maximum(h / np.maximum(l, EPS), EPS)) ** 2) / (4 * LN2)
    gk = (0.5 * np.log(np.maximum(h / np.maximum(l, EPS), EPS)) ** 2
          - (2 * LN2 - 1) * np.log(np.maximum(c / np.maximum(o, EPS), EPS)) ** 2)
    rs = (np.log(np.maximum(h / np.maximum(c, EPS), EPS))
          * np.log(np.maximum(h / np.maximum(o, EPS), EPS))
          + np.log(np.maximum(l / np.maximum(c, EPS), EPS))
          * np.log(np.maximum(l / np.maximum(o, EPS), EPS)))
    for ten, x in (("lr_park", park), ("lr_gk", gk), ("lr_rs", rs)):
        F[ten] = np.log(np.maximum(x, EPS)) - np.log(rv)

    # ── K3 mau co ten — nguong CO DIEN, khong hieu chinh theo du lieu
    tt = than / rng                    # ti le than
    bt = F["bong_tren"]
    bd = F["bong_duoi"]
    tang = (c > o).astype(float)
    giam = (c < o).astype(float)
    F["doji"] = (tt < 0.05).astype(float)
    F["bua"] = ((tt < 0.35) & (bd > 2 * tt) & (bt < 0.15)).astype(float)
    F["sao_bang"] = ((tt < 0.35) & (bt > 2 * tt) & (bd < 0.15)).astype(float)
    F["marubozu"] = (tt > 0.90).astype(float)
    F["con_quay"] = ((tt < 0.35) & (bt > 0.25) & (bd > 0.25)).astype(float)

    def truoc(x, k=1):
        y = np.full(len(x), np.nan)
        y[k:] = x[:-k]
        return y

    o1, c1, h1, l1 = truoc(o), truoc(c), truoc(h), truoc(l)
    o2, c2 = truoc(o, 2), truoc(c, 2)
    tt1 = truoc(tt)
    F["nhan_chim_tang"] = ((c1 < o1) & (c > o) & (o <= c1) & (c >= o1)).astype(float)
    F["nhan_chim_giam"] = ((c1 > o1) & (c < o) & (o >= c1) & (c <= o1)).astype(float)
    F["harami_tang"] = ((c1 < o1) & (c > o) & (o >= c1) & (c <= o1)).astype(float)
    F["harami_giam"] = ((c1 > o1) & (c < o) & (o <= c1) & (c >= o1)).astype(float)
    F["xuyen_tham"] = ((c1 < o1) & (c > o) & (o < c1)
                       & (c > (o1 + c1) / 2) & (c < o1)).astype(float)
    F["may_den"] = ((c1 > o1) & (c < o) & (o > c1)
                    & (c < (o1 + c1) / 2) & (c > o1)).astype(float)
    F["sao_mai"] = ((c2 < o2) & (tt1 < 0.35) & (c > o)
                    & (c > (o2 + c2) / 2)).astype(float)
    F["sao_hom"] = ((c2 > o2) & (tt1 < 0.35) & (c < o)
                    & (c < (o2 + c2) / 2)).astype(float)
    F["ba_linh_trang"] = ((c2 > o2) & (c1 > o1) & (c > o)
                          & (c > c1) & (c1 > c2)).astype(float)
    F["ba_qua_den"] = ((c2 < o2) & (c1 < o1) & (c < o)
                       & (c < c1) & (c1 < c2)).astype(float)

    # ── K4 gap qua dem (co che 2)
    g = np.log(np.maximum(o, EPS)) - np.log(np.maximum(truoc(c), EPS))
    F["gap"] = g
    F["abs_gap"] = np.abs(g)
    return pd.DataFrame(F)


def dung_bang(cat_sau=None):
    """Mot hang = (cap, phien t). Muc tieu + moc hai truc + 25 dac trung x 2 lag."""
    bang, _ = V2.nap_bang()
    pan = pd.read_csv(os.path.join(DATA, "panel2_6pairs.csv"),
                      parse_dates=["Date"])
    ra = []
    for p in V2.PAIRS:
        d = bang[p]
        if cat_sau is not None:
            d = d[pd.DatetimeIndex(d.Date) <= pd.Timestamp(cat_sau)]
            d = d.reset_index(drop=True)
        ng = pd.DatetimeIndex(d.Date)
        n = len(d)
        rv = np.maximum(d.rv5.values, EPS)
        h = V2.du_bao_san_xuat(d, p)
        lrv = np.log(rv)
        F = dac_trung_nen(d)
        z = (pan[pan.pair == p].set_index("Date").reindex(ng)[["zT", "sig"]])
        zT, sig = z.zT.values, z.sig.values
        hang = dict(pair=p, ngay=ng, doan=doan(ng.values), rv_that=rv)
        base = pd.DataFrame(hang)
        base["y_bien_do"] = np.r_[lrv[1:], np.nan]
        base["m_har"] = np.log(np.maximum(h, EPS))
        base.loc[~np.isfinite(h) | (h <= 0), "m_har"] = np.nan
        base["y_huong"] = np.r_[zT[1:], np.nan]
        base["m_zt"] = zT
        base["m_lsig"] = np.log(np.maximum(sig, EPS))
        for L in LAG:
            sh = F.shift(L)
            for c in TAT_CA:
                base[f"{c}_L{L}"] = sh[c].values
        ra.append(base)
    df = pd.concat(ra, ignore_index=True)
    return df.sort_values(["pair", "ngay"]).reset_index(drop=True)


MOC = {"bien_do": ["m_har"], "huong": ["m_zt", "m_lsig"]}
Y = {"bien_do": "y_bien_do", "huong": "y_huong"}


def _tu_kiem_ro_ri(df):
    moc = "2023-01-01"
    dc = dung_bang(cat_sau=moc)
    k = ["pair", "ngay"]
    cols = [f"{c}_L{L}" for L in LAG for c in TAT_CA]
    a = df[df.ngay < pd.Timestamp(moc)].set_index(k)[cols].sort_index()
    b = dc[dc.ngay < pd.Timestamp(moc)].set_index(k)[cols].sort_index()
    ix = a.index.intersection(b.index)
    sai = int((~np.isclose(a.loc[ix].values, b.loc[ix].values,
                           equal_nan=True)).sum())
    return sai, int(a.loc[ix].size)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("=" * 104)
    print("MẪU HÌNH NẾN — hình học OHLC ngày có thêm gì ngoài tầng 2 không?")
    print("tiêu chí chốt trước: docs/NEN_TIEUCHI.md (commit 9643f80)")
    print("=" * 104)

    df = dung_bang()
    print(f"bảng {len(df):,} hàng · {df.pair.nunique()} cặp · "
          f"{df.ngay.min().date()} → {df.ngay.max().date()}")

    sai, n_o = _tu_kiem_ro_ri(df)
    print(f"\nTỰ KIỂM CẮT TƯƠNG LAI (cắt sau 2023-01-01): {sai}/{n_o:,} giá trị "
          f"đổi   {'ĐẠT' if sai == 0 else 'HỎNG'}")
    assert sai == 0, "ro ri nhin truoc"

    # ── so lan khop cua tung mau co ten (bien ban muc 6 doi phai in ra)
    print(f"\nSỐ LẦN KHỚP của 15 mẫu có tên (toàn bộ {len(df):,} hàng-phiên):")
    print(f"  {'mẫu':<18}{'n khớp':>9}{'tỷ lệ':>9}   {'mẫu':<18}{'n khớp':>9}{'tỷ lệ':>9}")
    khop = {}
    for i in range(0, len(K3), 2):
        line = "  "
        for c in K3[i:i + 2]:
            v = df[f"{c}_L1"].values
            k = int(np.nansum(v))
            khop[c] = k
            line += f"{c:<18}{k:>9,}{100*k/len(df):>8.2f}%   "
        print(line)

    pt = df.doan <= 1
    print(f"\nphát hiện trên HUẤN LUYỆN+KIỂM ĐỊNH: {int(pt.sum()):,} hàng")

    # ══════════════════ LEVEL 2 — 600 phep kiem + Westfall-Young
    print("\n" + "=" * 104)
    print(f"KIỂM ĐỊNH — 25 đặc trưng × {len(LAG)} lag × 6 cặp × 2 trục, "
          f"W-Y maxT, {NHOAN_VI} hoán vị khối {KHOI}")
    print("=" * 104)
    d = df[pt]
    hang, F_qs, F_hv = [], [], []
    for truc in ("bien_do", "huong"):
        yc, mc = Y[truc], MOC[truc]
        for p in sorted(d.pair.unique()):
            s = d[d.pair == p]
            for c in TAT_CA:
                for L in LAG:
                    col = f"{c}_L{L}"
                    sub = s[[yc, col] + mc]
                    ok = sub.notna().all(1).values
                    n = int(ok.sum())
                    if n < 300:
                        continue
                    y = s[yc].values[ok]
                    X0 = np.column_stack([np.ones(n)]
                                         + [s[m].values[ok] for m in mc])
                    b = s[col].values[ok]
                    if np.std(b) < 1e-12:
                        continue
                    X1 = np.column_stack([X0, b])
                    k1 = X1.shape[1]
                    Q0, Q1 = G._Q(X0), G._Q(X1)
                    F = float(G.f_long_nhau(Q0, Q1, y[:, None], 1, n, k1)[0])
                    Yp = np.column_stack([y[G.hoan_vi_khoi(n, rng)]
                                          for _ in range(NHOAN_VI)])
                    Fh = G.f_long_nhau(Q0, Q1, Yp, 1, n, k1)
                    bb, *_ = np.linalg.lstsq(X1, y, rcond=None)
                    hang.append(dict(truc=truc, pair=p, dac_trung=c, lag=L, n=n,
                                     n_khop=int(np.nansum(b)) if c in K3 else None,
                                     F=F, he_so=float(bb[-1]),
                                     dau=int(np.sign(bb[-1])),
                                     p_tiem_can=float(1 - stats.f.cdf(F, 1, n - k1))))
                    F_qs.append(F); F_hv.append(Fh)
    F_qs = np.asarray(F_qs); F_hv = np.asarray(F_hv)
    R = pd.DataFrame(hang)
    R["p_hoan_vi"] = (1 + (F_hv >= F_qs[:, None]).sum(1)) / (NHOAN_VI + 1)
    R["p_wy"] = G.wy_tung_buoc(F_qs, F_hv)
    R["q_fdr"] = G.fdr_bh(R.p_hoan_vi.values)
    np.save(os.path.join(OUT, "nen_null_max.npy"), F_hv.max(0))

    print(f"số phép kiểm thực hiện: {len(R)} (khai báo trước: 600)")
    print(f"\n{'cửa':<28}{'sống sót':>11}   ghi chú")
    print("-" * 104)
    print(f"{'p tiệm cận thô < 0,05':<28}{int((R.p_tiem_can < .05).sum()):>11}"
          f"   kỳ vọng dưới nhiễu ≈ {0.05*len(R):.0f}")
    print(f"{'p hoán vị khối < 0,05':<28}{int((R.p_hoan_vi < .05).sum()):>11}"
          f"   chưa hiệu chỉnh bội")
    print(f"{'FDR-BH q < 0,05':<28}{int((R.q_fdr < .05).sum()):>11}"
          f"   tham chiếu, KHÔNG dùng làm cửa")
    n_wy = int((R.p_wy < .05).sum())
    print(f"{'Westfall–Young p < 0,05':<28}{n_wy:>11}   **CỬA QUYẾT ĐỊNH**")
    print("-" * 104)

    print(f"\n12 quan hệ mạnh nhất (theo F):")
    print(f"  {'trục':<9}{'cặp':<8}{'đặc trưng':<16}{'lag':>4}{'n':>7}{'F':>8}"
          f"{'p hoán vị':>11}{'p W-Y':>9}{'dấu':>5}")
    print("  " + "-" * 88)
    for _, r in R.nlargest(12, "F").iterrows():
        print(f"  {r.truc:<9}{r.pair:<8}{r.dac_trung:<16}{int(r.lag):>4}"
              f"{int(r.n):>7,}{r.F:>8.2f}{r.p_hoan_vi:>11.4f}{r.p_wy:>9.4f}"
              f"{'+' if r.dau > 0 else '−':>5}")

    print(f"\nTHEO TRỤC:")
    for t in ("bien_do", "huong"):
        m = R.truc == t
        print(f"  {t:<10}{int(m.sum()):>5} phép kiểm · sống sót W-Y "
              f"{int((m & (R.p_wy < .05)).sum())} · p W-Y nhỏ nhất "
              f"{R[m].p_wy.min():.4f}")
    print(f"\nTHEO HỌ:")
    for ten, cols in HO.items():
        m = R.dac_trung.isin(cols)
        print(f"  {ten:<16}{int(m.sum()):>5} phép kiểm · sống sót W-Y "
              f"{int((m & (R.p_wy < .05)).sum())} · p W-Y nhỏ nhất "
              f"{R[m].p_wy.min():.4f} · F lớn nhất {R[m].F.max():.2f}")

    os.makedirs(OUT, exist_ok=True)
    R.to_csv(os.path.join(OUT, "nen_quanhe.csv"), index=False)
    json.dump(dict(
        n_kiem=int(len(R)), khai_bao=600, n_hoan_vi=NHOAN_VI, khoi=KHOI,
        song_sot=dict(tho=int((R.p_tiem_can < .05).sum()),
                      hoan_vi=int((R.p_hoan_vi < .05).sum()),
                      fdr=int((R.q_fdr < .05).sum()), wy=n_wy),
        theo_truc={t: dict(n=int((R.truc == t).sum()),
                           wy=int(((R.truc == t) & (R.p_wy < .05)).sum()),
                           p_min=float(R[R.truc == t].p_wy.min()))
                   for t in ("bien_do", "huong")},
        theo_ho={ten: dict(n=int(R.dac_trung.isin(c).sum()),
                           wy=int((R.dac_trung.isin(c) & (R.p_wy < .05)).sum()),
                           p_min=float(R[R.dac_trung.isin(c)].p_wy.min()),
                           F_max=float(R[R.dac_trung.isin(c)].F.max()))
                 for ten, c in HO.items()},
        n_khop=khop, tu_kiem_ro_ri=dict(sai=sai, n=n_o)),
        open(os.path.join(OUT, "nen.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/nen.json · output/nen_quanhe.csv · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
