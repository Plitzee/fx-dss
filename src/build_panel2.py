"""Dung lai panel rui ro bang du bao bien dong moi (volfc) thay cho MA20-GK.

Xuat data/panel2_6pairs.csv voi cung lugc do cot nhu panel_6pairs.csv:
    Date, pair, sig, zT, zL, zH
cong them cot chan doan:
    rv5   phuong sai thuc do ngay do (de cham diem du bao)
    sig_old  du bao cu MA20-GK, de so sanh truc tiep tren cung hang

Ngay o day la NGAY GIAO DICH FX (phien Chu nhat da gop vao ngay ke tiep),
nen so hang it hon panel cu khoang 17%.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
D = os.path.join(os.path.dirname(HERE), "data")
import fxdata; fxdata.D = os.path.join(D, "prices")
from fxdata import load_daily
from vol import per_day_estimators
from volfc import merge_thin_days, forecast_series

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]


def build(pair):
    adv = pd.read_csv(os.path.join(D, "rv_adv.csv"), parse_dates=["Date"])
    px = load_daily(pair)[["Date", "open", "high", "low", "close"]]
    d = adv[adv.pair == pair].drop(columns=["pair"]).merge(px, on="Date", how="inner")
    d = merge_thin_days(d)
    f = forecast_series(d)                      # du bao PHUONG SAI cho ngay t
    e = per_day_estimators(d)                   # r_cc tren ngay giao dich da gop
    sig = np.sqrt(f)
    # du bao cu de doi chieu: MA20 cua Garman-Klass, da dich mot phien
    gk = e.gk.clip(lower=1e-14).values
    sig_old = np.sqrt(pd.Series(gk).rolling(20).mean().shift(1).values)
    c = d.close.values; cp = np.r_[np.nan, c[:-1]]
    out = pd.DataFrame({
        "Date": d.Date, "pair": pair, "sig": sig, "sig_old": sig_old,
        "zT": np.log(c / cp) / sig,
        "zL": np.log(d.low.values / cp) / sig,
        "zH": np.log(d.high.values / cp) / sig,
        "rv5": d.rv5.values})
    return out.dropna(subset=["sig", "zT"]).reset_index(drop=True)


if __name__ == "__main__":
    parts = []
    for p in PAIRS:
        o = build(p); parts.append(o)
        print(f"  {p}: {len(o):,} phiên  {o.Date.min().date()} → {o.Date.max().date()}", flush=True)
    allp = pd.concat(parts, ignore_index=True)
    f = os.path.join(D, "panel2_6pairs.csv")
    allp.to_csv(f, index=False)
    print(f"ghi {f}: {len(allp):,} dòng")
