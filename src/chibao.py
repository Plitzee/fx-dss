"""CHI BAO KY THUAT — cai o BACKEND, khong cai o trinh duyet.

Ly do kien truc (docs/REPLAN_2026.md muc 7.1): chi bao ve tren bieu do va dac
trung ma quy luat dung PHAI la mot bo ma duy nhat. Neu chi bao viet bang
TypeScript o phia truoc con quy luat khai pha bang Python o phia sau, hai ben se
troi khoi nhau va bieu do se noi mot dang con ba o noi mot neo.

VE LUXALGO. LuxAlgo la bo cong cu THUONG MAI, ma dong. Khong sao chep ma, khong
dung ten, khong tuyen bo tuong duong. Moi thu o day tu cai bang cong thuc MO,
co tai lieu, va tai lap duoc tu chinh file nay — dieu kien bat buoc de mot chi
bao duoc phep nuoi mot quy luat.

Tat ca deu NHAN QUA: gia tri tai t chi dung thanh <= t. Ham `kiem_nhan_qua()`
o cuoi kiem dieu do bang cach cat duoi chuoi roi so lai.

Chay tu kiem:  python src/chibao.py
"""
import numpy as np
import pandas as pd

EPS = 1e-12


# ── tien ich ────────────────────────────────────────────────────────────
def _s(x):
    return pd.Series(np.asarray(x, float))


def ema(x, n):
    return _s(x).ewm(span=n, adjust=False).mean().values


def sma(x, n):
    return _s(x).rolling(n, min_periods=n).mean().values


def rma(x, n):
    """Trung binh truot Wilder — dung cho RSI, ATR, ADX (khac EMA o he so)."""
    return _s(x).ewm(alpha=1.0 / n, adjust=False).mean().values


def true_range(h, l, c):
    pc = np.concatenate([[np.nan], np.asarray(c, float)[:-1]])
    a = np.asarray(h, float) - np.asarray(l, float)
    b = np.abs(np.asarray(h, float) - pc)
    d = np.abs(np.asarray(l, float) - pc)
    return np.nanmax(np.vstack([a, b, d]), axis=0)


# ── xu huong ────────────────────────────────────────────────────────────
def atr(h, l, c, n=14):
    a = rma(true_range(h, l, c), n)
    a[:n] = np.nan
    return a


def supertrend(h, l, c, n=10, he_so=3.0):
    """Duong bam ATR. Tra ve (duong, chieu) — chieu +1 tang, -1 giam."""
    h, l, c = map(lambda z: np.asarray(z, float), (h, l, c))
    a = atr(h, l, c, n)
    giua = (h + l) / 2.0
    tren, duoi = giua + he_so * a, giua - he_so * a
    N = len(c)
    ft, fd = np.full(N, np.nan), np.full(N, np.nan)
    st, ch = np.full(N, np.nan), np.zeros(N, int)
    for i in range(N):
        if not np.isfinite(tren[i]):
            continue
        if i == 0 or not np.isfinite(ft[i - 1]):
            ft[i], fd[i], ch[i] = tren[i], duoi[i], 1
        else:
            ft[i] = tren[i] if (tren[i] < ft[i - 1] or c[i - 1] > ft[i - 1]) else ft[i - 1]
            fd[i] = duoi[i] if (duoi[i] > fd[i - 1] or c[i - 1] < fd[i - 1]) else fd[i - 1]
            ch[i] = ch[i - 1]
            if ch[i - 1] == 1 and c[i] < fd[i]:
                ch[i] = -1
            elif ch[i - 1] == -1 and c[i] > ft[i]:
                ch[i] = 1
        st[i] = fd[i] if ch[i] == 1 else ft[i]
    return st, ch


def adx(h, l, c, n=14):
    """ADX + DI+ / DI- (Wilder). Do SUC MANH xu huong, khong do huong."""
    h, l, c = map(lambda z: np.asarray(z, float), (h, l, c))
    up = np.concatenate([[np.nan], np.diff(h)])
    dn = np.concatenate([[np.nan], -np.diff(l)])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(h, l, c), n)
    pdi = 100.0 * rma(pdm, n) / np.maximum(tr, EPS)
    ndi = 100.0 * rma(ndm, n) / np.maximum(tr, EPS)
    dx = 100.0 * np.abs(pdi - ndi) / np.maximum(pdi + ndi, EPS)
    ax = rma(dx, n)
    ax[: 2 * n], pdi[:n], ndi[:n] = np.nan, np.nan, np.nan   # vung khoi dong
    return ax, pdi, ndi


def ichimoku(h, l, c, a=9, b=26, d=52):
    """May Ichimoku. LUU Y: hai duong bien (span A/B) trong ban goc duoc DAY
    TOI 26 phien — o day tra ve gia tri TAI THOI DIEM TINH, chua day, de moi
    thu trong file nay giu dung mot quy uoc nhan qua. Phia truoc muon ve may
    thi tu day."""
    h, l = np.asarray(h, float), np.asarray(l, float)
    hh = lambda n: _s(h).rolling(n, min_periods=n).max().values
    ll = lambda n: _s(l).rolling(n, min_periods=n).min().values
    tenkan = (hh(a) + ll(a)) / 2.0
    kijun = (hh(b) + ll(b)) / 2.0
    return tenkan, kijun, (tenkan + kijun) / 2.0, (hh(d) + ll(d)) / 2.0


# ── dong luong ──────────────────────────────────────────────────────────
def rsi(c, n=14):
    d = np.concatenate([[np.nan], np.diff(np.asarray(c, float))])
    # `np.where(NaN > 0, ...)` cho False -> 0, nen thanh dau tien se ra RSI = 0
    # gia. Phai NaN hoa vung khoi dong, khong duoc de no ve bieu do.
    g = rma(np.where(np.isfinite(d) & (d > 0), d, 0.0), n)
    l = rma(np.where(np.isfinite(d) & (d < 0), -d, 0.0), n)
    r = 100.0 - 100.0 / (1.0 + g / np.maximum(l, EPS))
    r[:n] = np.nan
    return r


def macd(c, nhanh=12, cham=26, tin_hieu=9):
    m = ema(c, nhanh) - ema(c, cham)
    s = ema(m, tin_hieu)
    return m, s, m - s


def stoch(h, l, c, n=14, k=3, d=3):
    hh = _s(h).rolling(n, min_periods=n).max().values
    ll = _s(l).rolling(n, min_periods=n).min().values
    raw = 100.0 * (np.asarray(c, float) - ll) / np.maximum(hh - ll, EPS)
    kk = sma(raw, k)
    return kk, sma(kk, d)


# ── bien dong / kenh ────────────────────────────────────────────────────
def bollinger(c, n=20, k=2.0):
    m = sma(c, n)
    s = _s(c).rolling(n, min_periods=n).std(ddof=0).values
    return m + k * s, m, m - k * s


def keltner(h, l, c, n=20, na=10, k=1.5):
    m = ema(c, n)
    a = atr(h, l, c, na)
    return m + k * a, m, m - k * a


def donchian(h, l, n=20):
    u = _s(h).rolling(n, min_periods=n).max().values
    d = _s(l).rolling(n, min_periods=n).min().values
    return u, (u + d) / 2.0, d


def phan_vi_atr(h, l, c, n=14, cua_so=252):
    """ATR dat o phan vi nao cua chinh no — 0..1. Doc duoc hon ATR tho."""
    a = atr(h, l, c, n)
    return _s(a).rolling(cua_so, min_periods=max(30, n)).rank(pct=True).values


def vwap_lan(h, l, c, w=None, n=20):
    """VWAP truot. FX KHONG co khoi luong that — neu khong truyen `w` thi ham
    nay tro thanh gia trung binh theo gia (khong phai VWAP), va phai goi dung
    ten nhu vay tren giao dien. Truyen w = so tick/thanh de co VWAP that."""
    tp = (np.asarray(h, float) + np.asarray(l, float) + np.asarray(c, float)) / 3.0
    if w is None:
        return sma(tp, n), False
    w = np.nan_to_num(np.asarray(w, float), nan=0.0)
    num = _s(tp * w).rolling(n, min_periods=n).sum().values
    den = _s(w).rolling(n, min_periods=n).sum().values
    return num / np.maximum(den, EPS), True


# ── cau truc thi truong ─────────────────────────────────────────────────
def diem_xoay(h, l, k=2):
    """Dinh/day dao dong kieu fractal: cao hon k thanh moi ben.

    NHAN QUA: mot dinh tai i chi XAC NHAN duoc o thoi diem i+k. Tra ve them
    mang `xac_nhan_tai` de phia truoc khong ve som hon thuc te."""
    h, l = np.asarray(h, float), np.asarray(l, float)
    N = len(h)
    dinh, day = np.zeros(N, bool), np.zeros(N, bool)
    for i in range(k, N - k):
        w = slice(i - k, i + k + 1)
        if h[i] == np.nanmax(h[w]) and np.sum(h[w] == h[i]) == 1:
            dinh[i] = True
        if l[i] == np.nanmin(l[w]) and np.sum(l[w] == l[i]) == 1:
            day[i] = True
    return dinh, day, k


def vung_ho_tro_khang_cu(h, l, c, k=2, n_vung=6, dung_sai=0.0015):
    """Gom cac diem xoay gan nhau thanh VUNG. Tra ve danh sach vung con song
    (chua bi xuyen thung dut khoat), moi vung: gia, so lan cham, chi so cuoi."""
    dinh, day, kk = diem_xoay(h, l, k)
    h, l, c = map(lambda z: np.asarray(z, float), (h, l, c))
    # ep ve int/float THUAN PYTHON ngay tu day: np.int64 khong tuan tu hoa duoc
    # sang JSON, va loi do chi lo ra o tang API chu khong lo o tu kiem nay.
    moc = [(int(i), float(h[i]), 1) for i in np.flatnonzero(dinh)] + \
          [(int(i), float(l[i]), -1) for i in np.flatnonzero(day)]
    moc.sort()
    vung = []
    for i, gia, loai in moc:
        for v in vung:
            if abs(v["gia"] - gia) / max(gia, EPS) <= dung_sai:
                v["gia"] = (v["gia"] * v["cham"] + gia) / (v["cham"] + 1)
                v["cham"] += 1
                v["cuoi"] = int(i)
                break
        else:
            vung.append({"gia": float(gia), "cham": 1, "dau": int(i),
                         "cuoi": int(i), "loai": int(loai)})
    vung.sort(key=lambda v: (-v["cham"], -v["cuoi"]))
    return vung[:n_vung]


def khoang_trong_gia(h, l, n_max=12):
    """Fair value gap: khoang trong ba thanh — thanh i-1 va i+1 khong chong
    nhau. Chi tra ve cac khoang CHUA bi lap day."""
    h, l = np.asarray(h, float), np.asarray(l, float)
    N = len(h)
    ra = []
    for i in range(1, N - 1):
        if l[i + 1] > h[i - 1]:
            ra.append({"tai": i, "duoi": float(h[i - 1]), "tren": float(l[i + 1]), "chieu": 1})
        elif h[i + 1] < l[i - 1]:
            ra.append({"tai": i, "duoi": float(h[i + 1]), "tren": float(l[i - 1]), "chieu": -1})
    con = []
    for g in ra:
        sau = slice(g["tai"] + 2, N)
        if g["tai"] + 2 >= N:                     # chua co thanh nao sau do
            con.append(g)
            continue
        # "lap day" = gia quay lai xuyen HET khoang, khong phai chi cham mep
        lap = (np.nanmin(l[sau]) <= g["duoi"]) if g["chieu"] == 1 \
            else (np.nanmax(h[sau]) >= g["tren"])
        if not lap:
            con.append(g)
    return con[-n_max:]


def quet_thanh_khoan(h, l, c, k=2, nhin_lai=60):
    """Quet thanh khoan: rau nen xuyen qua mot dinh/day dao dong TRUOC do roi
    dong cua quay lai ben trong. Dau hieu bay gia kinh dien."""
    dinh, day, _ = diem_xoay(h, l, k)
    h, l, c = map(lambda z: np.asarray(z, float), (h, l, c))
    N = len(h)
    ra = []
    for i in range(k + 1, N):
        lo = max(0, i - nhin_lai)
        dd = np.flatnonzero(dinh[lo:i - k]) + lo
        if len(dd):
            m = h[dd].max()
            if h[i] > m and c[i] < m:
                ra.append({"tai": int(i), "muc": float(m), "chieu": -1})
        yy = np.flatnonzero(day[lo:i - k]) + lo
        if len(yy):
            m = l[yy].min()
            if l[i] < m and c[i] > m:
                ra.append({"tai": int(i), "muc": float(m), "chieu": 1})
    return ra


def tinh_tat_ca(d, w=None):
    """Bo chi bao day du cho mot khung. `d` co Date, open, high, low, close."""
    o, h, l, c = (d[x].values.astype(float) for x in ("open", "high", "low", "close"))
    bbU, bbM, bbL = bollinger(c)
    kcU, kcM, kcL = keltner(h, l, c)
    dcU, dcM, dcL = donchian(h, l)
    m, s, hist = macd(c)
    st, stch = supertrend(h, l, c)
    ax, pdi, ndi = adx(h, l, c)
    tk, kj, sa, sb = ichimoku(h, l, c)
    kk, dd = stoch(h, l, c)
    vw, that = vwap_lan(h, l, c, w)
    return {
        "ema20": ema(c, 20), "ema50": ema(c, 50), "ema200": ema(c, 200),
        "bb_tren": bbU, "bb_giua": bbM, "bb_duoi": bbL,
        "kc_tren": kcU, "kc_giua": kcM, "kc_duoi": kcL,
        "dc_tren": dcU, "dc_giua": dcM, "dc_duoi": dcL,
        "atr": atr(h, l, c), "atr_pv": phan_vi_atr(h, l, c),
        "st": st, "st_chieu": stch,
        "macd": m, "macd_tin_hieu": s, "macd_hist": hist,
        "rsi": rsi(c), "stoch_k": kk, "stoch_d": dd,
        "adx": ax, "di_duong": pdi, "di_am": ndi,
        "ich_tenkan": tk, "ich_kijun": kj, "ich_span_a": sa, "ich_span_b": sb,
        "vwap": vw, "vwap_that": that,
    }


def kiem_nhan_qua(d, n_cat=40, bo_qua=("vwap_that",)):
    """Cat bo `n_cat` thanh CUOI roi tinh lai — moi gia tri con lai phai trung
    khit. Neu lech thi co chi bao dang nhin ve tuong lai."""
    A = tinh_tat_ca(d)
    B = tinh_tat_ca(d.iloc[: len(d) - n_cat].reset_index(drop=True))
    xau = []
    for k, va in A.items():
        if k in bo_qua or np.isscalar(va) or isinstance(va, bool):
            continue
        a = np.asarray(va, float)[: len(d) - n_cat]
        b = np.asarray(B[k], float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() and np.nanmax(np.abs(a[m] - b[m])) > 1e-9:
            xau.append((k, float(np.nanmax(np.abs(a[m] - b[m])))))
    return xau


if __name__ == "__main__":
    import os
    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    f = os.path.join(ROOT, "data", "prices", "EURUSD_d1.csv")
    d = pd.read_csv(f, parse_dates=["Date"]).tail(1200).reset_index(drop=True)

    print("TỰ KIỂM — EURUSD D1, %d thanh" % len(d))
    R = tinh_tat_ca(d)
    print(f"  {len(R)} chỉ báo tính xong")

    # 1) nhan qua
    xau = kiem_nhan_qua(d)
    print(f"  nhân quả (cắt 40 thanh cuối rồi tính lại): "
          f"{'ĐẠT' if not xau else 'HỎNG ' + str(xau[:3])}")
    assert not xau, f"chi bao nhin ve tuong lai: {xau}"

    # 2) khoang gia tri
    assert np.nanmin(R["rsi"]) >= -1e-9 and np.nanmax(R["rsi"]) <= 100 + 1e-9
    assert np.nanmin(R["stoch_k"]) >= -1e-9 and np.nanmax(R["stoch_k"]) <= 100 + 1e-9
    assert np.nanmin(R["adx"]) >= -1e-9
    assert np.nanmin(R["atr"]) >= 0
    pv = R["atr_pv"][np.isfinite(R["atr_pv"])]
    assert pv.min() >= 0 and pv.max() <= 1
    print(f"  khoảng giá trị: RSI [{np.nanmin(R['rsi']):.1f}, {np.nanmax(R['rsi']):.1f}] · "
          f"ADX [{np.nanmin(R['adx']):.1f}, {np.nanmax(R['adx']):.1f}] · ATR‰ "
          f"[{np.nanmin(R['atr'])*1e4:.1f}, {np.nanmax(R['atr'])*1e4:.1f}] pip")

    # 3) quan he bat buoc
    ok = np.isfinite(R["bb_tren"]) & np.isfinite(R["bb_duoi"])
    assert (R["bb_tren"][ok] >= R["bb_giua"][ok] - 1e-12).all()
    assert (R["bb_duoi"][ok] <= R["bb_giua"][ok] + 1e-12).all()
    ok = np.isfinite(R["dc_tren"])
    assert (R["dc_tren"][ok] >= d.high.values[ok] - 1e-12).all(), "Donchian trên phải bao đỉnh"
    assert set(np.unique(R["st_chieu"])) <= {0, 1, -1}
    print("  quan hệ: Bollinger bao giữa · Donchian bao đỉnh/đáy · Supertrend chiều ∈ {−1,+1} — ĐẠT")

    # 4) cau truc thi truong
    v = vung_ho_tro_khang_cu(d.high.values, d.low.values, d.close.values)
    g = khoang_trong_gia(d.high.values, d.low.values)
    q = quet_thanh_khoan(d.high.values, d.low.values, d.close.values)
    dinh, day, kx = diem_xoay(d.high.values, d.low.values)
    print(f"  cấu trúc: {int(dinh.sum())} đỉnh · {int(day.sum())} đáy · "
          f"{len(v)} vùng S/R · {len(g)} khoảng trống chưa lấp · {len(q)} lần quét thanh khoản")
    assert all(x["cham"] >= 1 for x in v)
    assert all(x["tren"] > x["duoi"] for x in g), "khoảng trống phải có trên > dưới"
    assert kx == 2

    # 5) VWAP phai bao trung thuc khi khong co khoi luong
    _, that = vwap_lan(d.high.values, d.low.values, d.close.values)
    assert that is False, "khong co khoi luong thi phai bao that=False"
    _, that2 = vwap_lan(d.high.values, d.low.values, d.close.values,
                        w=np.ones(len(d)))
    assert that2 is True
    print("  VWAP: không có khối lượng → tự khai báo không phải VWAP thật — ĐẠT")

    print("\nTỰ KIỂM ĐẠT")
