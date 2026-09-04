"""Ham serverless Vercel — nen noi ngay + chi bao, tinh luc co request.

VI SAO TACH RIENG KHOI api/main.py. Ban FastAPI day du nap `volfc2`, `balop`,
scipy va ~36 MB CSV lich su de tinh sigma — qua nang cho gioi han cua Vercel
(goi ham toi da 250 MB da giai nen; rieng pandas+scipy+numpy da hon 120 MB).
Nen o day tach lam hai:

  * DU BAO (sigma + ba xac suat) tinh SAN o may minh, commit thanh
    public/ui_data.json. Hop ly vi mo hinh la mo hinh NGAY — no chi doi mot
    lan moi ngay, tinh lai moi request la lang phi.
  * NEN NOI NGAY + CHI BAO tinh o day, luc co request, chi can requests +
    pandas + numpy. Khong dung scipy, khong doc CSV lich su.

Duong dan:  /api/intraday?pair=EURUSD&tf=H1
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(os.path.dirname(HERE), "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

import chibao as CB  # noqa: E402

PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF")
UA = {"User-Agent": "Mozilla/5.0 (compatible; fx-dss-thesis/1.0)"}
# Gioi han la CUA YAHOO, khong phai lua chon thiet ke.
KHUNG = {"M1": ("7d", "1m", "chỉ 7 ngày — giới hạn cứng của Yahoo"),
         "M15": ("60d", "15m", "chỉ 60 ngày — giới hạn nhà cung cấp"),
         "H1": ("730d", "1h", "730 ngày từ nhà cung cấp"),
         "D1": ("730d", "1h", "gộp từ thanh giờ — nến ngày của Yahoo không hợp lệ")}
PIP = {"USDJPY": 0.01}


def yahoo(pair, rng, iv):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{pair}=X",
                     params={"range": rng, "interval": iv}, headers=UA, timeout=20)
    r.raise_for_status()
    d = r.json()["chart"]["result"][0]
    q = d["indicators"]["quote"][0]
    f = pd.DataFrame({"ts": pd.to_datetime(d["timestamp"], unit="s", utc=True),
                      "open": q["open"], "high": q["high"],
                      "low": q["low"], "close": q["close"]}).dropna()
    return f.reset_index(drop=True)


def _py(o):
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def lam(pair, tf, n):
    rng, iv, ghi_chu = KHUNG[tf]
    f = yahoo(pair, rng, iv)
    if tf == "D1":
        # KHONG dung interval=1d cua Yahoo: 37,1% so ngay co close==open va
        # high<open. Gop tu thanh gio giong het collect/live_fx.py.
        f["Date"] = f.ts.dt.normalize().dt.tz_localize(None)
        d = f.groupby("Date").agg(open=("open", "first"), high=("high", "max"),
                                  low=("low", "min"), close=("close", "last"),
                                  n=("close", "size")).reset_index()
        d = d[d.n >= 18].rename(columns={"Date": "ts"}).reset_index(drop=True)
    else:
        d = f.copy()
        d["ts"] = d.ts.dt.tz_convert("UTC").dt.tz_localize(None)
    d = d.tail(n).reset_index(drop=True)

    R = CB.tinh_tat_ca(d)
    # `t` la moc thoi gian CHO BIEU DO. Lightweight Charts chi chap nhan chuoi
    # 'YYYY-MM-DD' hoac SO GIAY UNIX — chuoi kieu '2026-09-04 03:00' bi parse
    # ra rac va bieu do hien trang. Da gap that tren ban da trien khai.
    # `ngay` luon la 'YYYY-MM-DD' de giao dien do duoc ve du bao NGAY.
    ngay = [str(pd.Timestamp(x).date()) for x in d.ts.values]
    t = (ngay if tf == "D1"
         else [int(pd.Timestamp(x).timestamp()) for x in d.ts.values])
    tem = ngay
    h, l, c = d.high.values, d.low.values, d.close.values
    dinh, day, k = CB.diem_xoay(h, l)
    return _py({
        "pair": pair, "tf": tf, "ghi_chu": ghi_chu, "nguon": "yahoo",
        "pip": PIP.get(pair, 0.0001), "ngay": tem, "t": t,
        "o": [round(float(v), 6) for v in d.open.values],
        "h": [round(float(v), 6) for v in h],
        "l": [round(float(v), 6) for v in l],
        "c": [round(float(v), 6) for v in c],
        "duong": {kk: [None if not np.isfinite(v) else round(float(v), 6)
                       for v in np.asarray(vv, float)]
                  for kk, vv in R.items() if kk not in ("st_chieu", "vwap_that")},
        "st_chieu": [int(v) for v in R["st_chieu"]],
        "vwap_that": bool(R["vwap_that"]),
        "cau_truc": {"xoay_k": k,
                     "dinh": [int(i) for i in np.flatnonzero(dinh)][-60:],
                     "day": [int(i) for i in np.flatnonzero(day)][-60:],
                     "vung": CB.vung_ho_tro_khang_cu(h, l, c),
                     "khoang_trong": CB.khoang_trong_gia(h, l),
                     "quet": CB.quet_thanh_khoan(h, l, c)[-20:]}})


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        pair = (q.get("pair", ["EURUSD"])[0] or "EURUSD").upper()
        tf = (q.get("tf", ["H1"])[0] or "H1").upper()
        try:
            n = min(int(q.get("n", ["900"])[0]), 1500)
        except ValueError:
            n = 900
        try:
            if pair not in PAIRS:
                raise ValueError(f"không có cặp {pair}")
            if tf not in KHUNG:
                raise ValueError(f"khung phải thuộc {list(KHUNG)}")
            body, ma = lam(pair, tf, n), 200
        except Exception as e:
            body, ma = {"loi": str(e)[:200]}, 400 if "không" in str(e) else 502
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(ma)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        # nen ngay/gio doi cham -> cho phep dem 5 phut, giam so lan goi Yahoo
        self.send_header("Cache-Control", "public, s-maxage=300, stale-while-revalidate=600")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
