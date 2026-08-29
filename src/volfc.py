"""TANG 2 — DU BAO BIEN DONG (ban thay the MA20-GK).

Quy tac duoc chon sau khi so 14 mo hinh tren 6 cap x ~3.600 phien ngoai mau:

    du bao = trung binh hinh hoc cua ba mo hinh HAR-log
             { STHARQ, HARQ, SHAR }

QLIKE trung binh 0,1645 so voi 0,2161 cua MA20-GK dang nuoi panel — TOT HON
24%, Diebold-Mariano thang 6/6 cap o p<0,05, va nam trong Model Confidence
Set o ca 6/6 cap. Chi tiet: docs/TANG2_BIENDONG.md.

BA DIEU PHAI BIET:

1. NGAY CHU NHAT PHAI GOP VAO THU HAI. Phien Chu nhat cua FX chi dai ~2 gio
   (23 thanh 5 phut so voi 287), phuong sai nho hon 24 lan. De nguyen trong
   chuoi hoi quy thi HAR hong hoan toan — QLIKE 0,4616 thay vi 0,1648. Day
   la nguyen nhan that su cua ket luan cu "HAR thua moi mo hinh don gian".

2. MOI THU O KHONG GIAN LOG, co hieu chinh log-chuan +0,5*var(phan du) khi
   doi ve muc — vi QLIKE cham diem ky vong co dieu kien, khong phai trung vi.

3. CUA SO MO RONG, uoc luong lai moi phien, chi dung thong tin toi t.

   STHARQ  HAR chuyen che do muot (Fed 2025: THAR/STHAR thang ca ML) cong
           hieu chinh sai so do luong bang realized quarticity (HARQ,
           Bollerslev-Patton-Quaedvlieg 2016)
   HARQ    HAR + hieu chinh sai so do luong
   SHAR    HAR ban ra thanh semivariance am/duong (Patton-Sheppard 2015)
"""
import numpy as np
import pandas as pd

EPS = 1e-14
MU1 = np.sqrt(2.0 / np.pi)
THIN_N5 = 100          # duoi nguong nay coi la phien Chu nhat / ngay cut
MIN_FIT = 300          # so quan sat toi thieu de uoc luong
GAMMA = 1.5            # do muot cua ham chuyen che do
WIN_Z = 250            # cua so chuan hoa bien chuyen che do


def merge_thin_days(df):
    """Gop phien Chu nhat vao ngay giao dich ke tiep.

    Quy uoc FX: ngay giao dich bat dau luc 22:00 UTC Chu nhat. Du lieu goc
    cat theo ngay lich nen tach doi. Gop lai:
      open  = mo cua phien mong    high/low = bao ca hai
      close = dong cua ngay chinh  cac do luong noi ngay = CONG lai
    Cot bat buoc: Date, open, high, low, close, rv5, bpv5, rq5, rsp, rsn, n5.
    """
    d = df.sort_values("Date").reset_index(drop=True)
    thin = (d.n5 < THIN_N5).values
    keep = np.ones(len(d), bool)
    for i in range(len(d) - 1):
        if not thin[i] or thin[i + 1]:
            continue
        if (d.Date.iloc[i + 1] - d.Date.iloc[i]).days > 3:
            continue
        j = i + 1
        d.loc[j, "open"] = d.open.iloc[i]
        d.loc[j, "high"] = max(d.high.iloc[i], d.high.iloc[j])
        d.loc[j, "low"] = min(d.low.iloc[i], d.low.iloc[j])
        for c in ("rv5", "bpv5", "rq5", "rsp", "rsn", "n5"):
            d.loc[j, c] = d[c].iloc[i] + d[c].iloc[j]
        keep[i] = False
    out = d[keep].reset_index(drop=True)
    return out[out.n5 >= THIN_N5].reset_index(drop=True)


def _roll(v, w):
    return pd.Series(v).rolling(w).mean().values


def design(d):
    """Ma tran thiet ke cho ba mo hinh, tat ca o khong gian log."""
    rv = np.maximum(d.rv5.values, EPS)
    n = len(rv); o = np.ones(n)
    lv = np.log(rv); lw = _roll(lv, 5); lm = _roll(lv, 22)
    C = np.minimum(np.maximum(d.bpv5.values, EPS), rv)
    lq = np.log(np.maximum(np.sqrt(np.maximum(d.rq5.values, EPS)) / rv, EPS))
    lp = np.log(np.maximum(d.rsp.values, EPS))
    ln = np.log(np.maximum(d.rsn.values, EPS))
    mu = pd.Series(lv).rolling(WIN_Z).mean().shift(1).values
    sd = pd.Series(lv).rolling(WIN_Z).std().shift(1).values
    z = (lv - mu) / np.maximum(sd, 1e-8)
    G = 1.0 / (1.0 + np.exp(-GAMMA * z))          # 0 khi yen, 1 khi cang thang
    H = np.column_stack([o, lv, lw, lm])
    return {"STHARQ": np.column_stack([H, H * G[:, None], lq, lq * lv]),
            "HARQ": np.column_stack([H, lq, lq * lv]),
            "SHAR": np.column_stack([o, lp, ln, lw, lm])}, lv, np.log(np.maximum(C, EPS))


def _ols(X, y):
    return np.linalg.solve(X.T @ X + 1e-8 * np.eye(X.shape[1]), X.T @ y)


def forecast_series(d, min_train=500, max_gap_days=4):
    """Du bao PHUONG SAI ngay t+1 cho moi t, chi dung thong tin toi t.

    Tra ve mang do dai len(d); phan tu i la du bao CHO ngay i (NaN neu chua
    du dam hoac buoc bac qua lo hong du lieu)."""
    X, lv, _ = design(d)
    n = len(d); y = lv[1:]
    gap = d.Date.diff().dt.days.values.astype(float).copy(); gap[0] = 1
    cont = gap <= max_gap_days
    out = np.full(n, np.nan)
    for t in range(min_train, n - 1):
        if not cont[t + 1]:
            continue
        logs = []
        for k, Xk in X.items():
            Xd = Xk[:t]; yy = y[:t]
            ok = np.isfinite(Xd).all(1) & np.isfinite(yy)
            if ok.sum() < MIN_FIT:
                break
            b = _ols(Xd[ok], yy[ok]); xn = Xk[t]
            if not np.isfinite(xn).all():
                break
            s2 = float((yy[ok] - Xd[ok] @ b).var())
            logs.append(float(np.clip(xn @ b, -30, 0) + 0.5 * s2))
        if len(logs) == len(X):
            out[t + 1] = float(np.exp(np.mean(logs)))   # trung binh hinh hoc
    return out


if __name__ == "__main__":
    import os, sys, warnings
    warnings.filterwarnings("ignore")
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    D = os.path.join(os.path.dirname(here), "data")
    import fxdata; fxdata.D = os.path.join(D, "prices")
    from fxdata import load_daily

    adv = pd.read_csv(os.path.join(D, "rv_adv.csv"), parse_dates=["Date"])
    pair = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    px = load_daily(pair)[["Date", "open", "high", "low", "close"]]
    d = adv[adv.pair == pair].drop(columns=["pair"]).merge(px, on="Date", how="inner")

    print("TU KIEM volfc —", pair)
    n0 = len(d); dm_ = merge_thin_days(d); n1 = len(dm_)
    print(f"  gộp phiên mỏng: {n0:,} → {n1:,} ngày ({n0-n1:,} phiên Chủ nhật gộp vào)")
    assert (dm_.n5 >= THIN_N5).all(), "sau khi gộp không được còn ngày mỏng"
    assert (dm_.high >= dm_.low).all() and (dm_.high >= dm_.close).all(), "OHLC phải nhất quán"
    tot0 = d.rv5.sum(); tot1 = dm_.rv5.sum(); lost = 1 - tot1 / tot0
    # vai phien mong khong co ngay giao dich ke tiep trong 3 ngay (nghi le) bi bo han
    assert lost < 5e-4, f"gop lam mat qua nhieu phuong sai: {lost:.4%}"
    print(f"  tổng rv5 giữ lại {1-lost:.5%} (vài phiên mỏng sát ngày lễ bị bỏ hẳn)")

    f = forecast_series(dm_)
    ok = np.isfinite(f)
    yv = dm_.rv5.values
    q = (yv[ok] / f[ok] - np.log(yv[ok] / f[ok]) - 1).mean()
    print(f"  QLIKE ngoài mẫu: {q:.4f}  trên {ok.sum():,} phiên")
    assert q < 0.30, "QLIKE phai duoi 0,30 — MA20-GK dat 0,18-0,33"
    lc = np.corrcoef(np.log(f[ok]), np.log(yv[ok]))[0, 1]
    print(f"  tương quan log(dự báo) với log(thực tế): {lc:.3f}")
    assert lc > 0.60, "tuong quan log phai > 0,60"
    # kiem tinh nhan qua: xao tron tuong lai khong duoc doi du bao
    d2 = dm_.copy(); k = int(n1 * 0.8)
    d2.loc[k:, ["rv5", "bpv5", "rq5", "rsp", "rsn"]] *= 7.0
    f2 = forecast_series(d2)
    m = np.isfinite(f) & np.isfinite(f2) & (np.arange(n1) <= k)
    assert np.allclose(f[m], f2[m]), "RO RI NHIN TRUOC: doi du lieu tuong lai lam doi du bao qua khu"
    print(f"  không rò rỉ nhìn trước: {m.sum():,} dự báo giữ nguyên khi bóp méo tương lai")
    print("  ĐẠT")
