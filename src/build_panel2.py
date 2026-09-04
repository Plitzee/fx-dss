"""Dung lai panel rui ro bang du bao bien dong SAN XUAT (volfc2, vong 7).

Xuat data/panel2_6pairs.csv voi cung lugc do cot nhu panel_6pairs.csv:
    Date, pair, sig, zT, zL, zH
cong them cot chan doan:
    rv5   phuong sai thuc do ngay do (de cham diem du bao)
    sig_old  du bao cu MA20-GK, de so sanh truc tiep tren cung hang

Ngay o day la NGAY GIAO DICH FX (phien Chu nhat da gop vao ngay ke tiep),
nen so hang it hon panel cu khoang 17%.

VONG 7: cot `sig` gio sinh ra tu volfc2.du_bao_san_xuat() voi cau hinh da
chot (lich ngan hang trung uong RIENG tung cap). Ban panel truoc do duoc giu
lai o data/panel2_v6_6pairs.csv de doi chieu. Cot `sig_old` (MA20-GK) khong
doi — no la nen cu, khong phu thuoc cau hinh HAR.
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
from volfc import merge_thin_days
from volfc2 import du_bao_san_xuat, CAUHINH_SANXUAT

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]


def build(pair):
    adv = pd.read_csv(os.path.join(D, "rv_adv.csv"), parse_dates=["Date"])
    px = load_daily(pair)[["Date", "open", "high", "low", "close"]]
    d = adv[adv.pair == pair].drop(columns=["pair"]).merge(px, on="Date", how="inner")
    d = merge_thin_days(d)
    f = du_bao_san_xuat(d, pair)                # du bao PHUONG SAI cho ngay t
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
    print(f"cấu hình: {CAUHINH_SANXUAT}")
    old = os.path.join(D, "panel2_v6_6pairs.csv")
    if os.path.exists(old):
        o = pd.read_csv(old, parse_dates=["Date"])
        m = allp.merge(o[["Date", "pair", "sig"]], on=["Date", "pair"],
                       how="inner", suffixes=("", "_cu"))
        r = (m.sig / m.sig_cu - 1).abs()
        print(f"so với panel vòng 6 trên {len(m):,} hàng chung: "
              f"|Δsig|/sig trung vị {r.median():.3%}, p95 {r.quantile(.95):.3%}, "
              f"tối đa {r.max():.3%}")
