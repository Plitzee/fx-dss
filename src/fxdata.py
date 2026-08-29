"""Lop du lieu: doc, chuan hoa thang gia, kiem tra, va vá lo hong EURCHF bang chao."""
import pandas as pd, numpy as np

D = "/tmp/fx/data"
PLAUSIBLE = {"EURUSD":1.1,"USDJPY":110,"GBPUSD":1.3,"USDCHF":0.98,"AUDUSD":0.75,
             "USDCAD":1.3,"EURCHF":1.1,"EURGBP":0.85,"EURJPY":125,"GBPJPY":145,
             "AUDJPY":80,"XAUUSD":1500}
PAIRS = list(PLAUSIBLE)
OHLC = ["open","high","low","close"]

def _load_raw(pair, tf="d1"):
    df = pd.read_csv(f"{D}/{pair}_{tf}.csv", parse_dates=["Date"])
    scale = 10.0 ** round(np.log10(df["close"].median()/PLAUSIBLE[pair]))
    df[OHLC] = df[OHLC]/scale
    return df.sort_values("Date").reset_index(drop=True)

def load_daily(pair, patch_eurchf=True):
    df = _load_raw(pair, "d1")
    if pair == "EURCHF" and patch_eurchf:
        df = _patch_eurchf(df)
    df["is_patched"] = df.get("is_patched", False)
    return df

def _patch_eurchf(df):
    """Feed EURCHF thieu han 31/12/2014 -> 19/01/2015 (19 ngay), tuc bo mat chinh
    ngay SNB bo san. Suy lai bang chao EURUSD x USDCHF, danh dau is_patched."""
    eu, uc = _load_raw("EURUSD"), _load_raw("USDCHF")
    j = eu.merge(uc, on="Date", suffixes=("_e","_u"))
    cross = pd.DataFrame({"Date": j.Date})
    for c in OHLC:                       # xap xi: nhan tung thanh phan
        cross[c] = j[f"{c}_e"] * j[f"{c}_u"]
    # H/L cua chao khong bang tich H/L; sua lai cho nhat quan
    cross["high"] = cross[["open","high","low","close"]].max(axis=1)
    cross["low"]  = cross[["open","high","low","close"]].min(axis=1)
    cross["n_bars"] = np.nan
    have = set(df.Date)
    miss = cross[(~cross.Date.isin(have)) &
                 (cross.Date > df.Date.min()) & (cross.Date < df.Date.max())].copy()
    miss["is_patched"] = True
    df = df.copy(); df["is_patched"] = False
    out = pd.concat([df, miss], ignore_index=True).sort_values("Date").reset_index(drop=True)
    return out

def load_hourly(pair):
    return _load_raw(pair, "h1")

def realized_var_from_hourly(pair):
    """RV ngay = tong binh phuong loi suat gio TRONG ngay (khong gom gap qua dem)."""
    h = load_hourly(pair)
    h["day"] = h.Date.dt.normalize()
    h["r"] = np.log(h.close).diff()
    h.loc[h.day != h.day.shift(), "r"] = np.nan      # bo loi suat bac qua ranh gioi ngay
    g = h.dropna(subset=["r"]).groupby("day")["r"]
    rv = g.apply(lambda x: np.sum(x**2)).rename("rv_h1")
    nbar = g.size().rename("n_bars")
    return pd.concat([rv, nbar], axis=1).reset_index().rename(columns={"day":"Date"})


# ─────────────────────────────────────────────────────────────────────
# MUC TIEU CHUAN — realized variance lay mau 5 phut (Andersen-Bollerslev).
# Thay cho realized_var_from_hourly (23 quan sat/ngay) tu 28/08/2026.
# Do duoc: khung gio hut 8-13% so voi khung 5 phut; nhieu vi cau truc o
# khung 1 phut chi 5%. Sinh boi rv5.py tu 34,9tr nen M1.
# ─────────────────────────────────────────────────────────────────────
RV_FILE = "/tmp/fx/rv_multi.csv"
RV_DEFAULT = "m5"

def realized_var(pair, freq=RV_DEFAULT, path=None):
    """RV ngay o tan suat lay mau chon truoc. freq: m1 | m5 | m15 | h1."""
    import os
    f = path or RV_FILE
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"Khong thay {f}. Chay rv5.py tren may co du lieu M1 de sinh ra no.")
    col = f"rv_{freq}"
    df = pd.read_csv(f, parse_dates=["Date"])
    df = df[df.pair == pair]
    if col not in df.columns:
        raise ValueError(f"Khong co cot {col}; co: {[c for c in df.columns if c.startswith('rv_')]}")
    out = df[["Date", col, f"n_{freq}"]].rename(
        columns={col: "rv", f"n_{freq}": "n_bars"})
    return out.reset_index(drop=True)
