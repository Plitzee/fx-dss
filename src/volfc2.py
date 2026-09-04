"""TANG 2 — ENGINE DU BAO BIEN DONG CO CAU HINH (vong 7).

Thay `volfc.py` bang mot ban co the bat/tat tung cai tien de BACKTEST, chu
khong phai mot quy tac cung. Ket qua cua tung cau hinh duoc chon tren doan
KIEM DINH; doan kiem tra chi cham diem mot lan.

Nam truc cau hinh, moi truc gan voi mot bai bao cu the:

  deseason  none / wd / wdcov
      Khu chu ky NOI TUAN cua RV. Boudt-Croux-Laurent (JEF 2011) uoc luong
      he so mua vu intraweek bang uoc luong ben; Dumitru-Hizmeri-Izzeldin
      (JBF 2025) cho thay chu ky lam nhieu chinh cac thanh phan RV dua vao
      HAR. Bien doi theo THU ap cho ca dau vao lan muc tieu (thu cua ngay
      t+1 biet truoc). Hieu chinh DO PHU theo n5 chi ap cho DAU VAO, vi n5
      cua ngay t+1 chua biet tai thoi diem du bao.

  shrink    lambda in [0,1]
      Co ngot he so ve trung binh 6 cap. Pesaran-Pick-Timmermann (QE 2026):
      uoc luong co ngot / Bayes thuc nghiem thang ca gop thuan lan rieng
      thuan. lambda=0 la rieng tung cap, lambda=1 la gop hoan toan.

  crosspair off / on
      Them MOT he so cho log RV trung binh cua cac cap khac tai ngay t.
      Rubaszek-Szafranek-Uddin (JIMF 2025) do lan toa bien dong noi ngay
      tren dung 5/6 cap nay; Jia et al. (JBF 2024) tim thay du bao cheo cap.

  event     off / on
      Bien lich cho ngay t+1: FOMC, ECB, NFP (thu Sau dau thang), cuoi
      thang. Lee & Wang (RAPS 2025) va Martins-Lopes (arXiv 2411.16244).
      Lich biet truoc nen khong ro ri.

  window    exp / r1000 / r1500 / r2000
      Cua so mo rong hay cuon chieu. Feng-Zhang-Wang (J. Forecasting 2024):
      cua so nao tot hon thay doi theo dieu kien thi truong, nen phai chon
      chu khong duoc mac dinh.

CACH TINH. Uoc luong lai moi phien bang OLS cua so mo rong. De chay duoc
lu'oi hang tram cau hinh, dung GRAM TICH LUY: A_t = sum_{i<t} x_i x_i' cap
nhat O(1) moi buoc thay vi dung lai ma tran thiet ke. Ket qua trung khop
voi volfc.py cu (co kiem chung o cuoi file).

Van giu ba dieu cua ban cu:
  1. Phien Chu nhat gop vao ngay ke tiep (merge_thin_days).
  2. Moi thu o khong gian log, hieu chinh log-chuan +0,5*var(phan du).
  3. Chi dung thong tin toi t — co kiem chung khong ro ri nhin truoc.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
D = os.path.join(os.path.dirname(HERE), "data")

from volfc import merge_thin_days, THIN_N5, EPS, GAMMA, WIN_Z, MIN_FIT  # noqa: E402

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]

# Ngan hang trung uong CUA DONG TIEN KHONG PHAI USD trong tung cap. Moi cap
# deu chiu them FOMC. Day la truc "mau hinh rieng tung dong tien" o dang re
# nhat va biet truoc hoan toan — lich hop cong bo tu nhieu nam.
NHTW = {"EURUSD": "ECB", "GBPUSD": "BOE", "USDJPY": "BOJ",
        "AUDUSD": "RBA", "USDCAD": "BOC", "USDCHF": "SNB"}
# Nam DAU TIEN co lich day du cho tung ngan hang (xem docs). Truoc moc nay
# bien gia bang 0 du thuc te co hop — sai so suy giam, chi roi vao doan
# huan luyen, khong gay ro ri.
LICH_TU = {"FOMC": 2010, "ECB": 2010, "BOC": 2010, "RBA": 2010, "SNB": 2010,
           "BOJ": 2015, "BOE": 2015}
MODELS = ("STHARQ", "HARQ", "SHAR")
MIN_TRAIN = 500
MAX_GAP = 4
RIDGE = 1e-8


# ────────────────────────────────────────────────────────────── nạp dữ liệu
def nap_bang(pairs=PAIRS):
    """Doc rv_adv + gia ngay, gop phien mong, cat ve luoi ngay CHUNG cho moi cap."""
    import fxdata
    fxdata.D = os.path.join(D, "prices")
    from fxdata import load_daily
    adv = pd.read_csv(os.path.join(D, "rv_adv.csv"), parse_dates=["Date"])
    raw = {}
    for p in pairs:
        px = load_daily(p)[["Date", "open", "high", "low", "close"]]
        d = adv[adv.pair == p].drop(columns=["pair"]).merge(px, on="Date", how="inner")
        raw[p] = merge_thin_days(d)
    chung = raw[pairs[0]].Date
    for p in pairs[1:]:
        chung = pd.Index(chung).intersection(pd.Index(raw[p].Date))
    chung = pd.DatetimeIndex(sorted(chung))
    return {p: raw[p][raw[p].Date.isin(chung)].reset_index(drop=True) for p in pairs}, chung


def nap_lich(dates):
    """Bien lich cho MOI ngay trong `dates`. Tat ca deu biet truoc."""
    d = pd.DatetimeIndex(dates)
    out = {}
    f = os.path.join(D, "cb_dates.csv")
    if os.path.exists(f):
        cb = pd.read_csv(f, parse_dates=["date"])
        for b in cb.bank.unique():
            s = set(cb.date[cb.bank == b])
            out[b.lower()] = np.array([x in s for x in d], float)
    for b in LICH_TU:
        out.setdefault(b.lower(), np.zeros(len(d)))
    # NFP: thu Sau dau tien cua thang
    fri = pd.Series(d.dayofweek == 4)
    out["nfp"] = ((d.day.values <= 7) & fri.values).astype(float)
    # cuoi thang: hai phien cuoi cung cua moi thang trong luoi
    ym = d.year * 12 + d.month
    rank_from_end = np.zeros(len(d))
    for k in np.unique(ym):
        i = np.where(ym == k)[0]
        rank_from_end[i] = np.arange(len(i))[::-1]
    out["cuoithang"] = (rank_from_end <= 1).astype(float)
    return out


# ─────────────────────────────────────────────────────────── khử mùa vụ
def he_so_mua_vu(lv, dates, n5, train, mode):
    """Tra ve (a_muctieu, a_dauvao): hai mang cong-vao-log can TRU khoi lv.

    a_muctieu chi chua thanh phan biet truoc (thu trong tuan).
    a_dauvao  them hieu chinh do phu theo n5 (chi dung cho hoi quy).
    Uoc luong bang TRUNG VI, chi tren `train` — ben voi duoi day (Boudt 2011).
    """
    n = len(lv)
    if mode == "none":
        z = np.zeros(n)
        return z, z.copy()
    d = pd.DatetimeIndex(dates)
    ok = train & np.isfinite(lv)
    base = np.median(lv[ok])
    wd = d.dayofweek.values
    a_t = np.zeros(n)
    for w in np.unique(wd):
        m = ok & (wd == w)
        if m.sum() >= 30:
            a_t[wd == w] = np.median(lv[m]) - base
    if mode == "wd":
        return a_t, a_t.copy()
    # do phu: phan du sau khi bo thu, chia theo thap phan vi cua n5
    r = lv - a_t
    q = np.quantile(n5[ok], np.linspace(0, 1, 11))
    q = np.unique(q)
    idx = np.clip(np.digitize(n5, q[1:-1]), 0, len(q) - 2)
    a_c = np.zeros(n)
    for b in np.unique(idx):
        m = ok & (idx == b)
        if m.sum() >= 30:
            a_c[idx == b] = np.median(r[m]) - base
    return a_t, a_t + a_c


# ─────────────────────────────────────────────────── ma trận thiết kế
def _roll(v, w):
    return pd.Series(v).rolling(w).mean().values


def thiet_ke(d, lv_in, extra=None):
    """Ma tran thiet ke ba mo hinh tu chuoi log-RV DA KHU MUA VU `lv_in`.

    `extra` la danh sach cot phu (da can le theo hang t, gia tri cua ngay t+1
    voi bien lich, hoac ngay t voi bien cheo cap) — noi vao ca ba mo hinh.
    """
    rv = np.maximum(d.rv5.values, EPS)
    n = len(rv); o = np.ones(n)
    lv = lv_in
    lw = _roll(lv, 5); lm = _roll(lv, 22)
    lq = np.log(np.maximum(np.sqrt(np.maximum(d.rq5.values, EPS)) / rv, EPS))
    # semivariance: khu cung mot he so (rsp + rsn = rv)
    shift = lv_in - np.log(rv)
    lp = np.log(np.maximum(d.rsp.values, EPS)) + shift
    ln_ = np.log(np.maximum(d.rsn.values, EPS)) + shift
    mu = pd.Series(lv).rolling(WIN_Z).mean().shift(1).values
    sd = pd.Series(lv).rolling(WIN_Z).std().shift(1).values
    z = (lv - mu) / np.maximum(sd, 1e-8)
    G = 1.0 / (1.0 + np.exp(-GAMMA * z))
    H = np.column_stack([o, lv, lw, lm])
    X = {"STHARQ": np.column_stack([H, H * G[:, None], lq, lq * lv]),
         "HARQ": np.column_stack([H, lq, lq * lv]),
         "SHAR": np.column_stack([o, lp, ln_, lw, lm])}
    if extra:
        E = np.column_stack(extra)
        X = {k: np.column_stack([v, E]) for k, v in X.items()}
    return X


# ──────────────────────────────────────────── OLS cửa sổ, Gram tích luỹ
def he_so_cuon(X, y, hople, window=None):
    """Tra ve (beta, A, B, Syy, cnt) voi beta[t] khop tren hang i<t hop le.

    window=None la cua so mo rong; so nguyen la cua so cuon chieu do dai do.
    """
    n, k = X.shape
    Z = np.where(hople[:, None], X, 0.0)
    yv = np.where(hople, y, 0.0)
    w = hople.astype(float)
    outer = Z[:, :, None] * Z[:, None, :]
    cA = np.cumsum(outer, 0); cB = np.cumsum(Z * yv[:, None], 0)
    cS = np.cumsum(w * yv ** 2); cN = np.cumsum(w)
    z3 = np.zeros((1, k, k)); z2 = np.zeros((1, k)); z1 = np.zeros(1)
    A = np.concatenate([z3, cA[:-1]]); B = np.concatenate([z2, cB[:-1]])
    S = np.concatenate([z1, cS[:-1]]); N = np.concatenate([z1, cN[:-1]])
    if window is not None:
        sh = window
        A[sh:] -= A[:-sh]; B[sh:] -= B[:-sh]; S[sh:] -= S[:-sh]; N[sh:] -= N[:-sh]
    I = np.eye(k)[None] * RIDGE
    try:
        beta = np.linalg.solve(A + I, B[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        beta = (np.linalg.pinv(A + I) @ B[:, :, None])[:, :, 0]
    return beta, A, B, S, N


def _ssr(beta, A, B, S):
    """Tong binh phuong phan du cua `beta` tren chinh Gram da tich luy."""
    return S - 2 * np.einsum("tk,tk->t", beta, B) + np.einsum("tk,tkl,tl->t", beta, A, beta)


# ──────────────────────────────────────────────────────── chạy một cấu hình
def _cot_su_kien(p, lich, n, mode):
    """Cot bien lich cho cap `p`, GIA TRI CUA NGAY t+1 (biet truoc, khong ro ri)."""
    if mode in (False, "off", None):
        return []
    if mode is True or mode == "chung":
        ten = ["fomc", "ecb", "nfp", "cuoithang"]
    elif mode == "cap":
        ten = ["fomc", NHTW[p].lower(), "nfp", "cuoithang"]
    elif mode == "cbonly":
        ten = ["fomc", NHTW[p].lower()]
    elif mode == "capday":
        ten = ["fomc", NHTW[p].lower(), "nfp", "cuoithang", "sau_fomc", "sau_nhtw"]
    else:
        raise ValueError(mode)
    cols = []
    for k in ten:
        if k == "sau_fomc":
            v0 = np.zeros(n); v0[1:] = lich["fomc"][:-1]
        elif k == "sau_nhtw":
            v0 = np.zeros(n); v0[1:] = lich[NHTW[p].lower()][:-1]
        else:
            v0 = lich[k]
        v = np.zeros(n); v[:-1] = v0[1:]
        cols.append(v)
    return cols


def _mz_recal(lf, lv_tg, hople, min_n=500):
    """Hieu chuan Mincer-Zarnowitz cua so MO RONG trong khong gian log.

    Tai moi t, khop lv[i+1] ~ a + b*lf[i+1] tren i<t roi ap cho t+1.
    Brini (arXiv 2607.05291): phan lon "thang" o chan troi ngan chi la
    du bao duoc CHIA TY LE tot hon, nen phai hieu chuan truoc khi ket luan.
    """
    n = len(lf)
    ok = hople & np.isfinite(lf) & np.isfinite(lv_tg)
    x = np.where(ok, lf, 0.0); y = np.where(ok, lv_tg, 0.0); w = ok.astype(float)
    def cum(v):
        c = np.cumsum(v); return np.concatenate([[0.0], c[:-1]])
    Sw, Sx, Sy = cum(w), cum(x), cum(y)
    Sxx, Sxy, Syy = cum(x * x), cum(x * y), cum(y * y)
    det = Sw * Sxx - Sx * Sx
    b = np.where(np.abs(det) > 1e-12, (Sw * Sxy - Sx * Sy) / np.where(det == 0, 1, det), 1.0)
    a = np.where(Sw > 0, (Sy - b * Sx) / np.maximum(Sw, 1), 0.0)
    ssr = Syy - 2 * (a * Sy + b * Sxy) + (a * a * Sw + 2 * a * b * Sx + b * b * Sxx)
    s2 = np.where(Sw >= min_n, ssr / np.maximum(Sw, 1), np.nan)
    du = Sw >= min_n
    out = np.where(du, a + b * lf + 0.5 * np.maximum(s2, 0), lf)
    return np.where(np.isfinite(out), out, lf)


def chay(bang, chung, deseason="none", crosspair=False, event=False,
         window=None, lams=(0.0,), train_mask=None, pairs=None, recal="off"):
    """Chay mot cau hinh cho MOI cap. Tra ve dict[lam][pair] = mang du bao.

    `train_mask` (bool theo `chung`) danh dau doan huan luyen — he so mua vu
    chi duoc uoc luong tren doan nay.
    """
    pairs = pairs or list(bang.keys())
    n = len(chung)
    if train_mask is None:
        train_mask = np.ones(n, bool)
    lich = nap_lich(chung)

    lv_adj_in, lv_adj_tg, a_tg = {}, {}, {}
    for p in pairs:
        d = bang[p]
        lv = np.log(np.maximum(d.rv5.values, EPS))
        at, ai = he_so_mua_vu(lv, chung, d.n5.values, train_mask, deseason)
        lv_adj_in[p] = lv - ai
        lv_adj_tg[p] = lv - at
        a_tg[p] = at

    # biến chéo cặp: trung bình log-RV (đã khử) của CÁC CẶP KHÁC tại ngày t
    cross = {}
    if crosspair:
        M = np.column_stack([lv_adj_in[p] for p in pairs])
        tong = np.nansum(M, 1); dem = np.isfinite(M).sum(1)
        for j, p in enumerate(pairs):
            cross[p] = (tong - np.nan_to_num(M[:, j])) / np.maximum(dem - 1, 1)

    ket = {lam: {} for lam in lams}
    BE = {m: {} for m in MODELS}
    XS = {m: {} for m in MODELS}
    S2 = {m: {} for m in MODELS}
    AA = {m: {} for m in MODELS}; BB = {m: {} for m in MODELS}
    SS = {m: {} for m in MODELS}; NN = {m: {} for m in MODELS}

    for p in pairs:
        d = bang[p]
        extra = []
        if crosspair:
            extra.append(cross[p])
        extra += _cot_su_kien(p, lich, n, event)
        X = thiet_ke(d, lv_adj_in[p], extra or None)
        y = np.empty(n); y[:-1] = lv_adj_tg[p][1:]; y[-1] = np.nan
        for m in MODELS:
            Xm = X[m]
            hople = np.isfinite(Xm).all(1) & np.isfinite(y)
            b, A, B, S, N = he_so_cuon(Xm, y, hople, window)
            BE[m][p] = b; XS[m][p] = Xm
            AA[m][p] = A; BB[m][p] = B; SS[m][p] = S; NN[m][p] = N
            S2[m][p] = N

    # trung bình panel của hệ số, theo từng thời điểm — chỉ dùng thông tin tới t
    for m in MODELS:
        BAR = np.nanmean(np.stack([BE[m][p] for p in pairs]), 0)
        for lam in lams:
            for p in pairs:
                b = (1 - lam) * BE[m][p] + lam * BAR
                ssr = _ssr(b, AA[m][p], BB[m][p], SS[m][p])
                cnt = NN[m][p]
                s2 = np.where(cnt >= MIN_FIT, ssr / np.maximum(cnt, 1), np.nan)
                fit = np.einsum("tk,tk->t", XS[m][p], b)
                lf = np.clip(fit, -30, 0) + 0.5 * np.maximum(s2, 0)
                ket[lam].setdefault(p, {})[m] = lf

    # gộp ba mô hình (trung bình hình học) và cộng lại mùa vụ của ngày t+1
    ra = {}
    for lam in lams:
        ra[lam] = {}
        for p in pairs:
            L = np.stack([ket[lam][p][m] for m in MODELS])
            g = L.mean(0)
            if recal == "mz":
                y_tg = np.empty(n); y_tg[:-1] = lv_adj_tg[p][1:]; y_tg[-1] = np.nan
                hp = np.isfinite(g) & np.isfinite(y_tg)
                g = _mz_recal(g, y_tg, hp)
            ok = np.isfinite(L).all(0)
            cnt_ok = np.stack([NN[m][p] for m in MODELS]).min(0) >= MIN_FIT
            f = np.full(n, np.nan)
            gap = pd.Series(chung).diff().dt.days.values.astype(float); gap[0] = 1
            lien = np.zeros(n, bool); lien[1:] = gap[1:] <= MAX_GAP
            i = np.arange(n)
            m_ok = ok & cnt_ok & (i >= MIN_TRAIN)
            src = np.where(m_ok)[0]
            tgt = src + 1
            v = tgt < n
            src, tgt = src[v], tgt[v]
            v2 = lien[tgt]
            f[tgt[v2]] = np.exp(g[src[v2]] + a_tg[p][tgt[v2]])
            ra[lam][p] = f
    return ra


def qlike_tb(f, y, m):
    """QLIKE trung binh tren mask m, bo cac phan tu khong huu han."""
    ok = m & np.isfinite(f) & np.isfinite(y) & (f > 0) & (y > 0)
    if ok.sum() == 0:
        return np.nan, 0
    r = y[ok] / f[ok]
    return float((r - np.log(r) - 1).mean()), int(ok.sum())


# ══════════════════════════════════════════════════════════════════════
# CAU HINH SAN XUAT — chot ngay 01/09/2026 sau khi backtest 1.024 cau hinh.
# Chon tren doan KIEM DINH bang Model Confidence Set roi lay cau hinh DON
# GIAN NHAT trong tap song sot. Chi tiet: docs/KETQUA_VONG7.md.
#
#   deseason  = none     khu mua vu theo thu KHONG con tac dung khi da co lich
#   crosspair = off      thong tin cheo cap da nam trong lich su cua chinh cap
#   event     = capday   lich NHTW RIENG tung cap + FOMC + ngay ke tiep + NFP
#   window    = exp      cua so mo rong
#   recal     = off      hieu chuan Mincer-Zarnowitz khong giup
#   lam       = 0.0      uoc luong rieng tung cap, khong co ngot ve panel
#
# HAI TINH CHAT QUAN TRONG CHO ONG DAN HANG NGAY:
#   * lam=0 va crosspair=off  =>  moi cap DOC LAP hoan toan. Cap nhat mot cap
#     khong can du lieu cua cap khac. Ham du_bao_san_xuat() duoi day khai thac
#     dung dieu do.
#   * bien lich la cua ngay t+1 va biet truoc nhieu nam  =>  chay duoc TRUOC
#     khi phien t+1 mo cua.
# ══════════════════════════════════════════════════════════════════════
CAUHINH_SANXUAT = dict(deseason="none", crosspair=False, event="capday",
                       window=None, recal="off", lam=0.0)


def du_bao_san_xuat(d, pair):
    """Du bao PHUONG SAI ngay t+1 cho MOT cap, dung cau hinh da chot.

    d : DataFrame mot cap DA qua merge_thin_days, co cot
        Date, open, high, low, close, rv5, rq5, bpv5, rsp, rsn, n5
    pair : ten cap, dung de tra ra ngan hang trung uong tuong ung

    Tra ve mang do dai len(d): phan tu i la du bao CHO ngay i (NaN neu chua
    du dam hoac buoc bac qua lo hong du lieu). Cung quy uoc voi volfc cu.

    Ham nay la DIEM VAO SAN XUAT: on dan hang ngay noi them mot hang vao `d`
    roi goi lai, khong can cap khac va khong can chay lai ca luoi.
    """
    n = len(d)
    ngay = pd.DatetimeIndex(d.Date)
    lich = nap_lich(ngay)
    lv = np.log(np.maximum(d.rv5.values, EPS))
    extra = _cot_su_kien(pair, lich, n, CAUHINH_SANXUAT["event"])
    X = thiet_ke(d, lv, extra or None)
    y = np.empty(n); y[:-1] = lv[1:]; y[-1] = np.nan

    gap = pd.Series(ngay).diff().dt.days.values.astype(float); gap[0] = 1
    lien = np.zeros(n, bool); lien[1:] = gap[1:] <= MAX_GAP

    L, cnt_ok = [], np.ones(n, bool)
    for m in MODELS:
        Xm = X[m]
        hople = np.isfinite(Xm).all(1) & np.isfinite(y)
        b, A, B, S, N = he_so_cuon(Xm, y, hople, CAUHINH_SANXUAT["window"])
        ssr = _ssr(b, A, B, S)
        s2 = np.where(N >= MIN_FIT, ssr / np.maximum(N, 1), np.nan)
        fit = np.einsum("tk,tk->t", Xm, b)
        L.append(np.clip(fit, -30, 0) + 0.5 * np.maximum(s2, 0))
        cnt_ok &= N >= MIN_FIT
    L = np.stack(L)
    g = L.mean(0)
    ok = np.isfinite(L).all(0) & cnt_ok & (np.arange(n) >= MIN_TRAIN)

    out = np.full(n, np.nan)
    src = np.where(ok)[0]; tgt = src + 1
    v = tgt < n
    src, tgt = src[v], tgt[v]
    v2 = lien[tgt]
    out[tgt[v2]] = np.exp(g[src[v2]])
    return out


if __name__ == "__main__":
    import warnings, time
    warnings.filterwarnings("ignore")
    import volfc as V
    print("TU KIEM volfc2")
    t0 = time.time()
    bang, chung = nap_bang()
    print(f"  nạp {len(bang)} cặp, {len(chung):,} phiên chung "
          f"{chung[0].date()} → {chung[-1].date()}  ({time.time()-t0:.1f}s)")

    # 1. trùng khớp với bản cũ trên cấu hình mặc định
    t0 = time.time()
    r = chay(bang, chung, lams=(0.0,))[0.0]
    print(f"  chạy cấu hình gốc: {time.time()-t0:.1f}s")
    p = "EURUSD"
    f_old = V.forecast_series(bang[p])
    f_new = r[p]
    m = np.isfinite(f_old) & np.isfinite(f_new)
    rel = np.abs(f_new[m] / f_old[m] - 1)
    print(f"  {p}: khớp bản cũ trên {m.sum():,} phiên, "
          f"lệch tương đối tối đa {rel.max():.2e}, trung vị {np.median(rel):.2e}")
    assert rel.max() < 1e-6, "engine moi phai trung khop volfc.py cu"

    y = bang[p].rv5.values
    q, nq = qlike_tb(f_new, y, np.ones(len(y), bool))
    print(f"  QLIKE ngoài mẫu {p}: {q:.4f} trên {nq:,} phiên")
    assert q < 0.30

    # 2. không rò rỉ nhìn trước
    b2 = {k: v.copy() for k, v in bang.items()}
    k = int(len(chung) * 0.8)
    for pp in b2:
        b2[pp].loc[k:, ["rv5", "bpv5", "rq5", "rsp", "rsn"]] *= 7.0
    tm = np.arange(len(chung)) < k
    R1 = chay(bang, chung, deseason="wdcov", crosspair=True, event=True, lams=(0.3,), train_mask=tm)[0.3]
    R2 = chay(b2, chung, deseason="wdcov", crosspair=True, event=True, lams=(0.3,), train_mask=tm)[0.3]
    for pp in PAIRS:
        f1 = R1[pp]; f2 = R2[pp]
        mm = np.isfinite(f1) & np.isfinite(f2) & (np.arange(len(chung)) <= k)
        assert np.allclose(f1[mm], f2[mm]), f"RO RI NHIN TRUOC o {pp}"
    print(f"  không rò rỉ nhìn trước: 6/6 cặp giữ nguyên dự báo quá khứ khi bóp méo tương lai")

    # 3. điểm vào sản xuất phải trùng khớp engine lưới
    R = chay(bang, chung, lams=(CAUHINH_SANXUAT["lam"],), train_mask=None,
             **{k: v for k, v in CAUHINH_SANXUAT.items() if k != "lam"})[CAUHINH_SANXUAT["lam"]]
    for pp in PAIRS:
        fa = R[pp]; fb = du_bao_san_xuat(bang[pp], pp)
        mm = np.isfinite(fa) & np.isfinite(fb)
        rel = np.abs(fb[mm] / fa[mm] - 1)
        assert mm.sum() > 3000 and rel.max() < 1e-9, \
            f"du_bao_san_xuat lech engine luoi o {pp}: max {rel.max():.2e}"
    print(f"  điểm vào sản xuất trùng engine lưới trên 6/6 cặp "
          f"({mm.sum():,} phiên, lệch tối đa {rel.max():.1e})")

    # 4. biến lịch có đúng số ngày
    lich = nap_lich(chung)
    print(f"  biến lịch trên lưới chung: FOMC {int(lich['fomc'].sum())}, "
          f"ECB {int(lich['ecb'].sum())}, NFP {int(lich['nfp'].sum())}, "
          f"cuối tháng {int(lich['cuoithang'].sum())}")
    assert 100 < lich["fomc"].sum() < 145 and 150 < lich["nfp"].sum() < 200
    print("  ĐẠT")
