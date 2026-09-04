"""API SAN XUAT — FastAPI, phuc vu giao dien tang 7.

Noi chuoi LICH SU (HistData, den 2025-12-31) voi chuoi HIEN HANH (Yahoo, tu
2026-01-01, do collect/live_fx.py tai), roi chay DUNG diem vao san xuat cua
tang 2 — `volfc2.du_bao_san_xuat` — de ra sigma cho ngay ke tiep, va cac nen
cua giai doan 1 (`balop`) de ra ba xac suat.

KHONG co so bia o day. Cai gi khong tinh duoc thi tra ve null va noi ro vi sao.

MOI NOI HAI NHA CUNG CAP la co that va duoc CONG BO qua /health va /meta:
lech trung vi 0,20-3,40 pip tuy cap (do tren doan chong lan). `docs/KHOA_SO.md`
tung tu choi va du lieu bang nguon thu hai vi ly do nay; o day buoc phai noi
nen phai do va noi ra.

CANH BAO NIEM PHONG: toan bo 2026 nam trong tap khoa so cua docs/KHOA_SO.md.
Phuc vu du lieu 2026 qua API nay TIEU mot phan niem phong do. Chi lam khi da
chot cau hinh va ghi bien ban — xem /meta truong `canh_bao`.

Chay:  python -m uvicorn api.main:app --port 8899
"""
import datetime as dt
import functools
import os
import sys
import threading

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
D = os.path.join(ROOT, "data")
LIVE = os.path.join(D, "live")
WEB = os.path.join(ROOT, "web")

import balop as B                                      # noqa: E402
import volfc2 as V2                                    # noqa: E402
from volfc import merge_thin_days                      # noqa: E402
from split import doan, VALID_TU, TEST_TU              # noqa: E402

PAIRS = B.PAIRS
MOC_NOI = pd.Timestamp("2026-01-01")
PIP = {"USDJPY": 0.01}
HS = (1, 5, 20)
NEN_THEO_H = {1: "chỉ σ̂", 5: "σ̂ + chế độ", 20: "σ̂ + chế độ"}

app = FastAPI(title="FX-DSS API", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_khoa = threading.Lock()
_bo_nho = {}


def pip_size(p):
    return PIP.get(p, 0.0001)


def _nap_lich_su(p):
    """Chuoi HistData: gia ngay + do luong noi ngay tu rv_adv.csv."""
    g = pd.read_csv(os.path.join(D, "prices", f"{p}_d1.csv"), parse_dates=["Date"])
    a = pd.read_csv(os.path.join(D, "rv_adv.csv"), parse_dates=["Date"])
    a = a[a.pair == p].drop(columns=["pair"])
    d = pd.merge(g[["Date", "open", "high", "low", "close"]], a, on="Date", how="inner")
    d["nguon"] = "histdata"
    return d


def _nap_hien_hanh(p):
    f = os.path.join(LIVE, f"{p}_d1.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, parse_dates=["Date"])
    d = d.rename(columns={"rsp5": "rsp", "rsn5": "rsn"})
    d["nguon"] = "yahoo"
    return d[["Date", "open", "high", "low", "close", "rv5", "rq5", "bpv5",
              "rsp", "rsn", "n5", "rv_uoc", "nguon"]]


def noi_chuoi(p):
    """Lich su den 2025-12-31 + hien hanh tu 2026-01-01."""
    ls = _nap_lich_su(p)
    ls = ls[ls.Date < MOC_NOI]
    hh = _nap_hien_hanh(p)
    if hh is not None:
        hh = hh[hh.Date >= MOC_NOI]
        d = pd.concat([ls, hh], ignore_index=True)
    else:
        d = ls
    if "rv_uoc" not in d:
        d["rv_uoc"] = 0
    d["rv_uoc"] = d.rv_uoc.fillna(0).astype(int)
    return d.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def tinh(p):
    """Tinh sigma + ba xac suat cho MOT cap. Ket qua duoc nho lai."""
    d = noi_chuoi(p)
    m = merge_thin_days(d)
    sig2 = V2.du_bao_san_xuat(m, p)                     # phuong sai du bao
    sig = np.sqrt(np.maximum(sig2, 0.0))

    pan = pd.DataFrame({"Date": m.Date.values, "sig": sig})
    zt = np.empty(len(m)); zt[:] = np.nan
    c = m.close.values
    zt[1:] = np.log(c[1:] / np.maximum(c[:-1], 1e-12)) / np.maximum(sig[1:], 1e-12)
    pan["zT"] = zt
    ok = np.isfinite(pan.sig.values) & (pan.sig.values > 0)
    pan = pan[ok].reset_index(drop=True)
    tr = doan(pan.Date.values) == 0

    xs = {}
    for h in HS:
        T = B.dung_muc_tieu(pan, h, tr)
        ns = B.ChiSigma().khop(T["z"][tr])
        cd = B.SigmaCheDo().khop(T["z"][tr], pan.sig.values[tr])
        mo = ns if NEN_THEO_H[h] == "chỉ σ̂" else cd
        P = mo.du_bao(len(pan), canh=T["canh_P"], sigma_h=T["sigma_h"],
                      sig=pan.sig.values)
        xs[h] = dict(P=P, b=T["b"], sigma_h=T["sigma_h"], kP=T["kP"], c_h=T["c_h"])

    nguong = np.quantile(pan.sig.values[tr], [1 / 3, 2 / 3])
    return dict(m=m, pan=pan, sig=pan.sig.values, xs=xs,
                che_do=np.digitize(pan.sig.values, nguong), nguong=nguong,
                tinh_luc=dt.datetime.utcnow())


def lay(p, moi=False):
    if p not in PAIRS:
        raise HTTPException(404, f"không có cặp {p}")
    with _khoa:
        if moi or p not in _bo_nho:
            _bo_nho[p] = tinh(p)
        return _bo_nho[p]


def _idx(K, ngay):
    d = K["pan"].Date.values
    if ngay is None:
        return len(d) - 1
    t = np.datetime64(pd.Timestamp(ngay))
    i = int(np.searchsorted(d, t))
    if i >= len(d) or d[i] != t:
        i = min(max(i - 1, 0), len(d) - 1)
    return i


# ── endpoint ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    ra = {"ok": True, "gio_utc": dt.datetime.utcnow().isoformat() + "Z", "cap": {}}
    for p in PAIRS:
        f = os.path.join(LIVE, f"{p}_d1.csv")
        if os.path.exists(f):
            d = pd.read_csv(f, parse_dates=["Date"])
            ra["cap"][p] = {"du_lieu_den": str(d.Date.max().date()),
                            "rv5_that_ngay": int((d.rv_uoc == 0).sum()),
                            "tre_ngay": (dt.datetime.utcnow().date() - d.Date.max().date()).days}
        else:
            ra["cap"][p] = {"loi": "chưa tải dữ liệu hiện hành"}
            ra["ok"] = False
    return ra


@app.get("/meta")
def meta():
    mn = os.path.join(LIVE, "moi_noi.json")
    import json
    return {
        "cap": list(PAIRS), "tam_han": list(HS),
        "nen_theo_h": {str(k): v for k, v in NEN_THEO_H.items()},
        "moc_noi_nguon": str(MOC_NOI.date()),
        "valid_tu": str(VALID_TU.date()), "test_tu": str(TEST_TU.date()),
        "moi_noi": json.load(open(mn, encoding="utf-8")) if os.path.exists(mn) else None,
        "canh_bao": [
            "Chuỗi trước 2026-01-01 là HistData (tick); từ 2026 là Yahoo (báo giá chỉ dẫn). "
            "Lệch trung vị 0,20–3,40 pip tuỳ cặp — xem trường moi_noi.",
            "Toàn bộ 2026 nằm trong tập khoá sổ của docs/KHOA_SO.md. Phục vụ dữ liệu 2026 "
            "qua API này tiêu một phần niêm phong đó.",
            "Yahoo là endpoint KHÔNG chính thức, không có cam kết dịch vụ.",
            "Chưa có sổ dự báo, nên chỉ số hiệu chuẩn là số tĩnh đo trên đoạn kiểm định.",
        ]}


@app.get("/series")
def series(pair: str = Query(...), tu: str = Query(None), n: int = Query(1500)):
    K = lay(pair)
    m, ps = K["m"], pip_size(pair)
    d = m if tu is None else m[m.Date >= pd.Timestamp(tu)]
    d = d.tail(n)
    i0 = len(m) - len(d)
    return {"pair": pair, "pip": ps,
            "ngay": [str(x)[:10] for x in d.Date.values],
            "o": [round(float(v), 6) for v in d.open.values],
            "h": [round(float(v), 6) for v in d.high.values],
            "l": [round(float(v), 6) for v in d.low.values],
            "c": [round(float(v), 6) for v in d.close.values],
            "nguon": list(d.nguon.values) if "nguon" in d else None,
            "rv_uoc": [int(v) for v in d.rv_uoc.values] if "rv_uoc" in d else None,
            "i0": i0}


@app.get("/forecast")
def forecast(pair: str = Query(...), h: int = Query(1), ngay: str = Query(None)):
    if h not in HS:
        raise HTTPException(400, f"tầm hạn phải thuộc {HS}")
    K = lay(pair)
    i = _idx(K, ngay)
    X = K["xs"][h]
    ps = pip_size(pair)
    P = X["P"][i]
    pan = K["pan"]
    nen12 = float(pd.Series(X["P"][:, 1]).rolling(252, min_periods=60).mean().iloc[i])
    return {
        "pair": pair, "h": h, "ngay": str(pan.Date.values[i])[:10],
        "p_giam": round(float(P[0]), 4), "p_ngang": round(float(P[1]), 4),
        "p_tang": round(float(P[2]), 4),
        "dai_pip": round(float(X["b"][i]) / ps, 2),
        "sigma_pip": round(float(X["sigma_h"][i]) / ps, 2),
        "sigma_1_pip": round(float(K["sig"][i]) / ps, 2),
        "che_do": int(K["che_do"][i]),
        "che_do_ten": ["bình tĩnh", "vừa", "căng thẳng"][int(K["che_do"][i])],
        "nen_12thang_ngang": round(nen12, 4) if np.isfinite(nen12) else None,
        "kP": round(float(X["kP"]), 4), "c_h": round(float(X["c_h"]), 4),
        "mo_hinh": NEN_THEO_H[h],
        "ky_nang_huong": "không phân biệt được (AUC phủ 0,50 ở 24/24 ô — xem /calibration)",
        "tinh_luc": K["tinh_luc"].isoformat() + "Z"}


@app.get("/forecast_series")
def forecast_series(pair: str = Query(...), n: int = Query(1500)):
    """Ca chuoi ba xac suat + dai + sigma cho MOI tam han — de giao dien re
    chuot tren bieu do ma khong phai goi lai tung ngay."""
    K = lay(pair)
    ps = pip_size(pair)
    pan = K["pan"]
    n = min(n, len(pan))
    sl = slice(len(pan) - n, len(pan))
    ra = {"pair": pair, "pip": ps,
          "ngay": [str(x)[:10] for x in pan.Date.values[sl]],
          "che_do": [int(v) for v in K["che_do"][sl]],
          "sig_pip": [round(float(v) / ps, 2) for v in K["sig"][sl]],
          "tam": {}, "nen12": {}}
    for h in HS:
        X = K["xs"][h]
        ra["tam"][str(h)] = {
            "p": [[round(float(v), 4) for v in row] for row in X["P"][sl]],
            "b_pip": [round(float(v) / ps, 2) for v in X["b"][sl]],
            "sig_pip": [round(float(v) / ps, 2) for v in X["sigma_h"][sl]],
            "kP": round(float(X["kP"]), 4), "c_h": round(float(X["c_h"]), 4),
            "nen": NEN_THEO_H[h]}
        s = pd.Series(X["P"][:, 1]).rolling(252, min_periods=60).mean().iloc[-1]
        ra["nen12"][str(h)] = round(float(s), 4) if np.isfinite(s) else 0.33
    return ra


@app.get("/cost")
def cost(pair: str = Query(...)):
    f = os.path.join(D, "cost_table.csv")
    if not os.path.exists(f):
        raise HTTPException(503, "chưa có cost_table.csv")
    c = pd.read_csv(f)
    c = c[(c.pair == pair) & (c.regime == "post2015")].sort_values("hour")
    return {"pair": pair,
            "med": [round(float(v), 3) for v in c.spread_med.values],
            "p95": [round(float(v), 3) for v in c.spread_p95.values]}


@app.get("/calibration")
def calibration():
    import json
    f = os.path.join(ROOT, "output", "nen3.json")
    if not os.path.exists(f):
        raise HTTPException(503, "chưa chạy src/run_balop.py")
    return {"doan": "kiểm định", "valid_tu": str(VALID_TU.date()),
            "test_tu": str(TEST_TU.date()), "bang": json.load(open(f, encoding="utf-8"))}


@app.get("/events")
def events(tu: str = Query("2024-01-01")):
    f = os.path.join(D, "cb_dates.csv")
    if not os.path.exists(f):
        return {"su_kien": []}
    c = pd.read_csv(f, parse_dates=["date"])
    c = c[c.date >= pd.Timestamp(tu)]
    c["ngay"] = c.date.astype(str).str[:10]
    return {"su_kien": [{"ngay": k, "nhan": sorted(set(v))}
                        for k, v in c.groupby("ngay").bank.apply(list).items()]}


@app.post("/refresh")
def refresh(pair: str = Query(None)):
    """Tinh lai tu du lieu tren dia. KHONG tai lai tu mang — viec do do
    jobs/cap_nhat.py lam, de mot request cua nguoi dung khong the goi ra ngoai."""
    ds = [pair] if pair else list(PAIRS)
    for p in ds:
        lay(p, moi=True)
    return {"da_tinh_lai": ds, "luc": dt.datetime.utcnow().isoformat() + "Z"}


@app.get("/")
def trang():
    f = os.path.join(WEB, "ui_live.html")
    if os.path.exists(f):
        return FileResponse(f)
    return {"ok": True, "xem": "/docs"}
