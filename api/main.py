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
import chibao as CB                                    # noqa: E402
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


def sang_pip(v, gia, pair):
    """Doi mot dai luong TUONG DOI (do lech chuan cua loi suat log) sang pip.

    Phai nhan voi MUC GIA roi moi chia co pip. Chia thang co pip la sai: voi
    EURUSD gia ~1,16 thi gan dung nen loi khong lo ra, nhung voi USDJPY gia
    ~156 thi lech 156 lan — man hinh tung hien sigma 0,4 pip thay vi ~65."""
    return np.asarray(v, float) * np.asarray(gia, float) / pip_size(pair)


def _py(o):
    """Ep kieu numpy ve kieu Python thuan — lop chan cuoi truoc khi ra JSON.
    np.int64/np.float64 khong tuan tu hoa duoc, va NaN/Inf thi khong hop le
    trong JSON nen doi thanh null."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    return o


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
        xs[h] = dict(P=P, b=T["b"], sigma_h=T["sigma_h"], kP=T["kP"], c_h=T["c_h"],
                     mo=mo)      # giu mo hinh DA KHOP de dung lai cho phien ke tiep

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


KHUNG = ("D1", "H1", "M15", "M1")


def nap_khung(pair, tf):
    """Nen cho MOT khung thoi gian.

      D1  lich su day (HistData 2010 -> 2025-12) + Yahoo tu 2026
      H1  lich su day (repo prices/{P}_h1.csv tu 2010) + Yahoo 730 ngay
      M15 chi Yahoo, 60 ngay      — gioi han cua nha cung cap
      M1  chi Yahoo, 7 NGAY       — gioi han cung cua Yahoo (range=30d bao loi)

    Do sau khac nhau la RANG BUOC CUA NGUON, khong phai lua chon thiet ke; ham
    tra ve `ghi_chu` de giao dien noi ro cho nguoi dung."""
    if tf == "D1":
        d = lay(pair)["m"][["Date", "open", "high", "low", "close", "nguon", "rv_uoc"]]
        return d.rename(columns={"Date": "ts"}), "lịch sử đầy đủ từ 2010"
    if tf == "H1":
        ph = []
        f = os.path.join(D, "prices", f"{pair}_h1.csv")
        if os.path.exists(f):
            a = pd.read_csv(f, parse_dates=["Date"]).rename(columns={"Date": "ts"})
            a = a[a.ts < MOC_NOI]
            a["nguon"] = "histdata"
            ph.append(a[["ts", "open", "high", "low", "close", "nguon"]])
        g = os.path.join(LIVE, f"{pair}_H1.csv")
        if os.path.exists(g):
            b = pd.read_csv(g, parse_dates=["ts"])
            b = b[b.ts >= MOC_NOI]
            b["nguon"] = "yahoo"
            ph.append(b[["ts", "open", "high", "low", "close", "nguon"]])
        if not ph:
            raise HTTPException(503, "chưa có dữ liệu H1")
        d = pd.concat(ph, ignore_index=True).sort_values("ts")
        return d.drop_duplicates("ts").reset_index(drop=True), "lịch sử đầy đủ từ 2010"
    g = os.path.join(LIVE, f"{pair}_{tf}.csv")
    if not os.path.exists(g):
        raise HTTPException(503, f"chưa tải khung {tf} — chạy collect/live_fx.py")
    d = pd.read_csv(g, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    d["nguon"] = "yahoo"
    han = {"M15": "chỉ 60 ngày — giới hạn nhà cung cấp",
           "M1": "chỉ 7 ngày — giới hạn cứng của Yahoo"}
    return d, han.get(tf, "")


def _tem(d, tf):
    """Tra ve (ngay, t).

    `ngay` luon la 'YYYY-MM-DD' — de giao dien do duoc ve du bao NGAY.
    `t` la moc cho BIEU DO: Lightweight Charts chi nhan chuoi 'YYYY-MM-DD'
    hoac SO GIAY UNIX. Chuoi kieu '2026-09-04 03:00' bi parse ra rac va bieu
    do hien trang — da gap that tren ban da trien khai."""
    ngay = [str(pd.Timestamp(x).date()) for x in d.ts.values]
    if tf == "D1":
        return ngay, ngay
    return ngay, [int(pd.Timestamp(x).timestamp()) for x in d.ts.values]


@app.get("/series")
def series(pair: str = Query(...), tf: str = Query("D1"),
           tu: str = Query(None), n: int = Query(1500)):
    if tf not in KHUNG:
        raise HTTPException(400, f"khung phải thuộc {KHUNG}")
    lay(pair)
    d, ghi_chu = nap_khung(pair, tf)
    if tu:
        d = d[d.ts >= pd.Timestamp(tu)]
    d = d.tail(n)
    ngay, t = _tem(d, tf)
    ra = {"pair": pair, "tf": tf, "pip": pip_size(pair), "ghi_chu": ghi_chu,
          "ngay": ngay, "t": t,
          "o": [round(float(v), 6) for v in d.open.values],
          "h": [round(float(v), 6) for v in d.high.values],
          "l": [round(float(v), 6) for v in d.low.values],
          "c": [round(float(v), 6) for v in d.close.values],
          "nguon": list(d.nguon.values) if "nguon" in d else None}
    if "rv_uoc" in d:
        ra["rv_uoc"] = [int(v) for v in d.rv_uoc.fillna(0).values]
    return ra


@app.get("/indicators")
def indicators(pair: str = Query(...), tf: str = Query("D1"), n: int = Query(1500)):
    """Chi bao tinh o BACKEND (src/chibao.py) — cung bo ma ma quy luat se dung.

    Xem docs/REPLAN_2026.md muc 7.1: neu chi bao ve bang TypeScript o phia truoc
    con quy luat khai pha bang Python o phia sau thi hai ben se troi khoi nhau."""
    if tf not in KHUNG:
        raise HTTPException(400, f"khung phải thuộc {KHUNG}")
    lay(pair)
    d, _ = nap_khung(pair, tf)
    d = d.tail(n).reset_index(drop=True)
    R = CB.tinh_tat_ca(d)
    lam = lambda a: [None if not np.isfinite(v) else round(float(v), 6)
                     for v in np.asarray(a, float)]
    ngay, t = _tem(d, tf)
    ra = {"pair": pair, "tf": tf, "ngay": ngay, "t": t,
          "duong": {k: lam(v) for k, v in R.items()
                    if k not in ("st_chieu", "vwap_that")},
          "st_chieu": [int(v) for v in R["st_chieu"]],
          "vwap_that": bool(R["vwap_that"])}
    h, l, c = d.high.values, d.low.values, d.close.values
    dinh, day, k = CB.diem_xoay(h, l)
    ra["cau_truc"] = {
        "xoay_k": k,
        "dinh": [int(i) for i in np.flatnonzero(dinh)][-60:],
        "day": [int(i) for i in np.flatnonzero(day)][-60:],
        "vung": CB.vung_ho_tro_khang_cu(h, l, c),
        "khoang_trong": CB.khoang_trong_gia(h, l),
        "quet": CB.quet_thanh_khoan(h, l, c)[-20:]}
    return _py(ra)


@app.get("/forecast")
def forecast(pair: str = Query(...), h: int = Query(1), ngay: str = Query(None)):
    if h not in HS:
        raise HTTPException(400, f"tầm hạn phải thuộc {HS}")
    K = lay(pair)
    i = _idx(K, ngay)
    X = K["xs"][h]
    pan = K["pan"]
    gia_i = float(K["m"].close.values[min(i, len(K["m"]) - 1)])
    P = X["P"][i]
    nen12 = float(pd.Series(X["P"][:, 1]).rolling(252, min_periods=60).mean().iloc[i])
    return {
        "pair": pair, "h": h, "ngay": str(pan.Date.values[i])[:10],
        "p_giam": round(float(P[0]), 4), "p_ngang": round(float(P[1]), 4),
        "p_tang": round(float(P[2]), 4),
        "dai_pip": round(float(sang_pip(X["b"][i], gia_i, pair)), 2),
        "sigma_pip": round(float(sang_pip(X["sigma_h"][i], gia_i, pair)), 2),
        "sigma_1_pip": round(float(sang_pip(K["sig"][i], gia_i, pair)), 2),
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
    _gia = K["m"].close.values[: len(pan)][sl]
    ra = {"pair": pair, "pip": ps,
          "ngay": [str(x)[:10] for x in pan.Date.values[sl]],
          "che_do": [int(v) for v in K["che_do"][sl]],
          "sig_pip": [round(float(v), 2) for v in sang_pip(K["sig"][sl], _gia, pair)],
          "tam": {}, "nen12": {}}
    for h in HS:
        X = K["xs"][h]
        ra["tam"][str(h)] = {
            "p": [[round(float(v), 4) for v in row] for row in X["P"][sl]],
            "b_pip": [round(float(v), 2) for v in sang_pip(X["b"][sl], _gia, pair)],
            "sig_pip": [round(float(v), 2) for v in sang_pip(X["sigma_h"][sl], _gia, pair)],
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


def _phien_ke_tiep(d):
    t = pd.Timestamp(d) + pd.Timedelta(days=1)
    while t.weekday() >= 5:                 # FX nghi thu 7 va Chu nhat
        t += pd.Timedelta(days=1)
    return t


@app.get("/forecast_next")
def forecast_next(pair: str = Query(...)):
    """Du bao cho phien CHUA MO CUA — thu duy nhat duoc phep ghi vao so.

    Cach lam theo dung docs/DONGBO_SANXUAT.md muc 1: bien lich la cua ngay t+1
    va biet truoc nhieu nam, nen chay duoc TRUOC khi phien t+1 mo cua. Ta noi
    them mot hang rong cho phien ke tiep roi goi lai diem vao san xuat.

    LUU Y da do: noi them hang lam du bao CAC NGAY CU doi nhe (lech toi 6,4e-07
    tren phuong sai) vi mo hinh khop lai theo cua so mo rong. Do la ly do so du
    bao phai CHI GHI THEM — no giu dung con so da hien luc do, khong phai con
    so tinh lai hom nay."""
    K = lay(pair)
    m = K["m"]
    kt = _phien_ke_tiep(m.Date.iloc[-1])
    hang = {c: np.nan for c in m.columns}
    hang["Date"] = kt
    hang["close"] = m.close.iloc[-1]
    m2 = pd.concat([m, pd.DataFrame([hang])], ignore_index=True)
    sig2 = V2.du_bao_san_xuat(m2, pair)
    sg = float(np.sqrt(max(sig2[-1], 0.0)))
    if not np.isfinite(sg) or sg <= 0:
        raise HTTPException(503, "chưa dựng được σ̂ cho phiên kế tiếp")

    gia_kt = float(m.close.iloc[-1])
    nguong = K["nguong"]
    ra = {"pair": pair, "ngay": str(kt.date()),
          "sigma_pip": round(float(sang_pip(sg, gia_kt, pair)), 2),
          "che_do": int(np.digitize([sg], nguong)[0]),
          "du_lieu_den": str(m.Date.iloc[-1].date()), "tam": {}}
    for h in HS:
        X = K["xs"][h]
        b = float(X["b"][-1])                       # dai doi cham, dung ban cuoi
        sh = sg * np.sqrt(h) * float(X["c_h"])
        # dung lai CHINH mo hinh da khop trong tinh() — khong khop lai
        P = X["mo"].du_bao(1, canh=np.array([b]), sigma_h=np.array([sh]),
                           sig=np.array([sg]))[0]
        ra["tam"][str(h)] = {
            "p_giam": round(float(P[0]), 4), "p_ngang": round(float(P[1]), 4),
            "p_tang": round(float(P[2]), 4),
            "dai_pip": round(float(sang_pip(b, gia_kt, pair)), 2),
            "sigma_pip": round(float(sang_pip(sh, gia_kt, pair)), 2),
            "mo_hinh": NEN_THEO_H[h], "kP": round(float(X["kP"]), 4),
            "c_h": round(float(X["c_h"]), 4)}
    return _py(ra)


@app.get("/journal")
def journal(h: int = Query(1), n_toi_thieu: int = Query(30)):
    """So du bao — hieu chuan TRUOT, do tren chinh cai he thong da noi ra.

    Khac han /calibration: /calibration la so do tren doan KIEM DINH 2021-2023,
    con day la so do tren nhung du bao he thong DA THUC SU dua ra. Khi chua du
    mau thi tra `du_mau: false` va KHONG tra chi so — khong doan, khong muon
    tam so cua doan kiem dinh."""
    import so_dubao as SD
    t = SD.thong_ke(h=h, n_toi_thieu=n_toi_thieu)
    db = SD.doc_dubao()
    t["tong_du_bao"] = int(len(db))
    t["cho_ket_cuc"] = int(len(db) - len(SD.ghep())) if len(db) else 0
    t["n_toi_thieu"] = int(n_toi_thieu)
    return _py(t)


@app.get("/calibration")
def calibration():
    import json
    f = os.path.join(ROOT, "output", "nen3.json")
    if not os.path.exists(f):
        raise HTTPException(503, "chưa chạy src/run_balop.py")
    return {"doan": "kiểm định", "valid_tu": str(VALID_TU.date()),
            "test_tu": str(TEST_TU.date()), "bang": json.load(open(f, encoding="utf-8"))}


@app.get("/risk")
def risk(pair: str = Query(...), dd: float = Query(0.0),
         so_vi_the: int = Query(1), stop_sigma: float = Query(2.0)):
    """PHIEU RUI RO — bay ra tang 4 va 6b da kiem dinh ma giao dien chua he hien.

    Khong co gi moi o day: PositionSizer va p_cham_stop deu da co va da duoc
    kiem dinh; viec cua endpoint nay chi la truy vet tung thanh phan de nguoi
    dung thay don bay khuyen nghi den TU DAU, va rang buoc nao dang buoc."""
    import sys as _s
    if SRC not in _s.path:
        _s.path.insert(0, SRC)
    from position_sizing import PositionSizer, k_danh_muc
    from decision_record import p_cham_stop
    from scipy import stats as _st

    K = lay(pair)
    pan, m = K["pan"], K["m"]
    tr = doan(pan.Date.values) == 0
    sizer = PositionSizer(pan.sig.values[tr])
    z_tr = pan.zT.values[tr]
    z_tr = z_tr[np.isfinite(z_tr)]
    nu = float(np.clip(_st.t.fit(z_tr, floc=0)[0], 2.5, 40))

    sg = float(pan.sig.values[-1])
    gia = float(m.close.values[-1])
    ps = pip_size(pair)
    # loi the ky vong = carry ngay (dau theo carry), giong run_e2e
    import optimal_stop as O
    cr = float(np.median(O.carry_ngay(pair, pan.Date.values[-260:])))
    ex = sizer.explain(sg, abs(cr), nu, dd=dd, so_vi_the=so_vi_the)

    # P(cham stop) theo tam han — bang ma docs/TANG6_TAMHAN.md canh bao
    tam = []
    for h in (1, 5, 10, 20):
        sh = sg * np.sqrt(h)
        tam.append({"h": h,
                    "p_cham": round(float(p_cham_stop(stop_sigma * sg / sh, z_tr)), 4)})

    # do nhay theo sut giam
    nhay = []
    for d_ in (0.0, 0.05, 0.10, 0.20, 0.30):
        e = sizer.explain(sg, abs(cr), nu, dd=d_, so_vi_the=so_vi_the)
        nhay.append({"dd": d_, "f": round(e["f"], 3), "k_dd": round(e["k_dd"], 3)})

    return _py({
        "pair": pair, "ngay": str(pan.Date.values[-1])[:10],
        "gia": gia, "sigma_pip": round(float(sang_pip(sg, gia, pair)), 2),
        "che_do": ["bình tĩnh", "vừa", "căng thẳng"][int(K["che_do"][-1])],
        "carry_ngay": cr, "nu": round(nu, 2),
        "sut_giam": dd, "so_vi_the": so_vi_the, "stop_sigma": stop_sigma,
        "stop_pip": round(float(sang_pip(stop_sigma * sg, gia, pair)), 1),
        "thanh_phan": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in ex.items()},
        "tam_han": tam, "theo_sut_giam": nhay,
        "he_so_danh_muc": [{"k": k, "he_so": round(float(k_danh_muc(k)), 4)}
                           for k in range(1, 7)],
        "canh_bao": [
            "Đòn bẩy khuyến nghị là TRẦN, không phải lệnh mua. Hệ thống không "
            "dự báo hướng — xem AUC ở tab Mô hình.",
            "Bảng tầm hạn: đọc P(chạm stop) ở h=1 rồi giữ 10 phiên là sai. "
            "Xem docs/TANG6_TAMHAN.md.",
            "Conformal phủ thiếu ~1 điểm phần trăm khi tài khoản đang lỗ "
            "(90,3% ở đỉnh vốn → 89,3% khi lỗ) — đo được, chưa vá.",
        ]})


@app.get("/models")
def models():
    """Chi so cua CAC TANG MO HINH — de giao dien hien duoc, khong phai van xuoi.

      chi_so_sigma : QLIKE, MAE, RMSE, CRPS, PIT, do phu  (src/chiso_mohinh.py)
      ba_lop       : ML/DL/hoc truc tuyen tren ba lop      (src/run_ml3.py)
      quy_luat     : pheu khai pha quy luat                (src/run_quyluat.py)
      bien_dong_14 : 14 mo hinh du bao phuong sai          (vong 7)
    """
    import json
    ra = {}
    for khoa, ten in (("chi_so_sigma", "chiso_mohinh.json"),
                      ("ba_lop", "ml3.json"),
                      ("quy_luat", "quyluat.json"),
                      ("bien_dong_14", "ketqua_ml_dl.json"),
                      ("tin_cay", "tincay.json"),
                      ("su_kien", "sukien_profile.json")):
        f = os.path.join(ROOT, "output", ten)
        ra[khoa] = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None
    return ra


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


from fastapi.staticfiles import StaticFiles  # noqa: E402

_DATA_DIR = os.path.join(WEB, "data")
if os.path.isdir(_DATA_DIR):
    # ban truc tiep va ban Vercel cung doc /data/meta.json va /data/{PAIR}.json
    app.mount("/data", StaticFiles(directory=_DATA_DIR), name="data")


@app.get("/")
def trang():
    f = os.path.join(WEB, "ui_live.html")
    if os.path.exists(f):
        return FileResponse(f)
    return {"ok": True, "xem": "/docs"}
