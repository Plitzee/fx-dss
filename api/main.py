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
# Chon tren doan KIEM DINH (output/nen3.json), khong phai tren kiem tra:
#   h=1   chi sigma +0,0105 · to hop +0,0102 · sigma+che do +0,0098   (hoa)
#   h=5   sigma+che do +0,0137 · to hop +0,0134                        (hoa)
#   h=20  TO HOP +0,0200 [+0,0065] · sigma+che do +0,0137 [+0,0005]    (thang ro)
# To hop khong thua o dau, thang ro o tam han dang hong nhat, va ECE tot nhat o
# h=1 va h=5. Them vao do no co CHAN HOI TIEC va tu ha trong so chuyen gia hong
# khi che do troi — thu ma nen co dinh khong lam duoc. Nen dung no ca ba tam han.
NEN_THEO_H = {1: "tổ hợp trực tuyến", 5: "tổ hợp trực tuyến",
              20: "tổ hợp trực tuyến"}

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
        n = len(pan)
        yt = B.lop_truoc(T["yP"], h)
        ns = B.ChiSigma().khop(T["z"][tr])
        cd = B.SigmaCheDo().khop(T["z"][tr], pan.sig.values[tr])
        kh = B.KhiHauHoc().khop(T["yP"][tr])
        qt = B.QuanTinh().khop(T["yP"][tr], yt[tr])
        kw = dict(canh=T["canh_P"], sigma_h=T["sigma_h"], sig=pan.sig.values,
                  y_truoc=yt)
        if NEN_THEO_H[h] == "tổ hợp trực tuyến":
            # Hoc truc tuyen: trong so cap nhat tu ket cuc DA BIET, tre dung h
            # phien. Du bao cho phien moi nhat dung trong so hoc tu toan bo qua
            # khu truoc no — dung nghia "hom qua sai thi hom nay chinh".
            mo = B.ToHopTrucTuyen([("khí hậu học", kh), ("quán tính", qt),
                                   ("chỉ σ̂", ns), ("σ̂ + chế độ", cd)], tre=h)
            P = mo.du_bao(n, y_that=T["yP"], **kw)
        else:
            mo = ns if NEN_THEO_H[h] == "chỉ σ̂" else cd
            P = mo.du_bao(n, **kw)
        xs[h] = dict(P=P, b=T["b"], sigma_h=T["sigma_h"], kP=T["kP"], c_h=T["c_h"],
                     mo=mo, trong_so=getattr(mo, "trong_so", None))

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
        kw = dict(canh=np.array([b]), sigma_h=np.array([sh]), sig=np.array([sg]))
        # Tang to hop phai dung TRONG SO DA HOC — goi du_bao(1,...) se khoi dong
        # lai trong so tu deu nhau, tuc vut bo dung cai phan da hoc.
        P = (X["mo"].du_bao_ke_tiep(**kw)[0]
             if hasattr(X["mo"], "du_bao_ke_tiep") else X["mo"].du_bao(1, **kw)[0])
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
        "xuat_xu": xuat_xu_rui_ro(pair, z_tr, nu, cr, sizer),
        "var_es": var_es(pair, K, z_tr, gia, don_bay=float(ex["f"])),
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


MUC_VAR = (0.05, 0.01)


def var_es(pair, K, z_tr, gia, von=10000.0, don_bay=1.0):
    """TANG RUI RO DUOI — VaR va ES, kem BACKTEST cua chinh no.

    VaR muc alpha = phan vi alpha cua loi suat phien toi. ES = ky vong loi suat
    KHI DA roi vao duoi VaR — tuc "neu ngay xau xay ra thi lo trung binh bao
    nhieu". VaR mot minh khong du: no noi nguong, khong noi do sau.

    Uoc bang PHAN VI THUC NGHIEM cua z (khong gia dinh chuan) nhan sigma^ hom
    nay. Day dung la cach `src/evaluate2.py` da lam va da dat het backtest.

    Kem backtest chay TAI CHO tren doan KIEM TRA:
      Kupiec        ty le vi pham co dung bang alpha khong
      Christoffersen  cac lan vi pham co dinh cum khong
      DQ            manh hon ca hai (Engle & Manganelli 2004)
    p >= 0,05 = khong bac bo duoc. Voi ~720 phien moi cap thi luc kiem dinh
    RAT THAP — phai noi ra, khong duoc doc "dat" thanh "da chung minh".
    """
    import sys as _s
    if SRC not in _s.path:
        _s.path.insert(0, SRC)
    from metrics import kupiec, christoffersen_ind, dq_test

    pan = K["pan"]
    sig = pan.sig.values
    sg = float(sig[-1])
    ps = pip_size(pair)
    g = doan(pan.Date.values)
    z_all = pan.zT.values
    te = (g == 2) & np.isfinite(z_all) & np.isfinite(sig) & (sig > 0)

    ra = {"von_mau": von, "muc": []}
    for a in MUC_VAR:
        qz = float(np.quantile(z_tr, a))               # phan vi z tren HUAN LUYEN
        ez = float(np.mean(z_tr[z_tr <= qz])) if (z_tr <= qz).any() else qz
        v_r, e_r = qz * sg, ez * sg                    # loi suat (am)
        # backtest tren doan kiem tra: nguong di dong theo sigma^ tung phien
        vt = qz * sig[te]
        y = z_all[te] * sig[te]
        hits = (y <= vt).astype(int)
        _, pk, ph = kupiec(hits, a)
        _, pi_ = christoffersen_ind(hits)
        _, pdq = dq_test(hits, vt, a)
        # ES co du sau khong: trong CHINH nhung phien vi pham, lo thuc te TB
        # co bang ES da du bao khong? < 1 la mo hinh danh gia THAP muc lo.
        m = hits.astype(bool)
        es_du = float(np.mean(ez * sig[te][m])) if m.any() else float("nan")
        es_th = float(np.mean(y[m])) if m.any() else float("nan")
        r4 = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 4)
        ra["muc"].append({
            "alpha": a,
            "var_pip": round(float(sang_pip(abs(v_r), gia, pair)), 1),
            "es_pip": round(float(sang_pip(abs(e_r), gia, pair)), 1),
            "var_usd": round(von * don_bay * abs(v_r), 0),
            "es_usd": round(von * don_bay * abs(e_r), 0),
            "n_kiem_tra": int(te.sum()), "n_vi_pham": int(hits.sum()),
            "ty_le_vi_pham": r4(ph), "ky_vong": a,
            "kupiec_p": r4(pk), "chris_p": r4(pi_), "dq_p": r4(pdq),
            "es_du_bao_pip": round(float(sang_pip(abs(es_du), gia, pair)), 1) if np.isfinite(es_du) else None,
            "es_thuc_te_pip": round(float(sang_pip(abs(es_th), gia, pair)), 1) if np.isfinite(es_th) else None,
            "es_ty_le": r4(es_du / es_th) if np.isfinite(es_du) and np.isfinite(es_th) and es_th != 0 else None,
            "dat": bool((pk is None or not np.isfinite(pk) or pk >= 0.05)
                        and (pi_ is None or not np.isfinite(pi_) or pi_ >= 0.05)
                        and (pdq is None or not np.isfinite(pdq) or pdq >= 0.05)),
        })
    ra["canh_bao_luc"] = (
        f"Ở mức 1% chỉ kỳ vọng ~{round(0.01*int(te.sum()))} lần vi phạm trên "
        f"{int(te.sum())} phiên — lực kiểm định THẤP. \"Không bác bỏ được\" "
        f"không có nghĩa là \"đã chứng minh đúng\".")
    return ra


def xuat_xu_rui_ro(pair, z_tr, nu, cr, sizer):
    """XUAT XU TUNG CON SO tren phieu rui ro.

    Nha dau tu chi tin duoc neu thay: con so nay ra tu CONG THUC nao, uoc tren
    BAO NHIEU mau, va cai gi CHUNG MINH no dung. Cot cuoi la chi so da do —
    QLIKE/MAE/CRPS/PIT cho tang sigma, ty le phu thuc te cho tang khoang, so
    lan cham stop cho tang truot gia. Khong dong nao la tham so dat tay.
    """
    import json
    f = os.path.join(ROOT, "output", "chiso_mohinh.json")
    C = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    q = ((C.get("cap") or {}).get(pair) or {}).get("kiểm tra") or {}
    g = (C.get("gop") or {}).get("kiểm tra") or {}
    r2 = lambda v, n=2: None if v is None else round(float(v), n)

    return {
        "doan_do": "kiểm tra (2023-11-20 → nay), chưa từng dùng để khớp",
        "n_sigma": q.get("n"), "n_z": int(len(z_tr)),
        "muc": [
            {"ten": "σ̂ — biên độ dao động dự kiến",
             "cong_thuc": "HAR vòng 7: log RV = f(ngày, tuần, tháng) + hiệu chỉnh "
                          "realized quarticity + semivariance ± + bipower/jump + "
                          "lịch NHTW riêng từng cặp",
             "de_hieu": "Dự đoán hôm nay giá sẽ dao động bao nhiêu pip, dựa trên mức "
                        "dao động của hôm qua, tuần qua, tháng qua, cộng thêm ngày họp "
                        "ngân hàng trung ương đã biết trước.",
             "uoc_tren": f"{q.get('n', 0):,} phiên đoạn kiểm tra",
             "chi_so": [("QLIKE", r2(q.get("qlike"), 4), "0 là hoàn hảo; gộp 6 cặp "
                         f"{r2(g.get('qlike'), 4)}"),
                        ("MAE", r2(q.get("mae_sigma_pip")), "pip — sai số tuyệt đối "
                         "trung bình của chính σ̂"),
                        ("RMSE", r2(q.get("rmse_sigma_pip")), "pip — phạt nặng lần "
                         "trượt lớn"),
                        ("CRPS", r2(q.get("crps_pip")), "pip — chấm CẢ PHÂN PHỐI, "
                         "không chỉ điểm giữa"),
                        ("PIT (KS p)", r2(q.get("pit_ks_p"), 4),
                         "p < 0,05 nghĩa là hình dạng phân phối bị BÁC BỎ"),
                        ("độ phủ 90%", r2(q.get("do_phu_90"), 4),
                         "phải gần 0,90; thấp hơn = σ̂ hụt, khoảng quá hẹp")]},
            {"ten": "Dừng lỗ = 2 σ̂",
             "cong_thuc": "quét 1,0–4,0 σ̂ trên lưới, chọn theo tiền cuối kỳ có "
                          "trừ trượt giá thật",
             "de_hieu": "Thử mọi khoảng dừng lỗ từ hẹp tới rộng trên toàn bộ lịch sử, "
                        "chọn khoảng cho nhiều tiền nhất SAU KHI đã trừ phí và trượt giá.",
             "uoc_tren": "docs/TANG6B_DUNGTOIUU.md — 60.617 lần chạm stop đã đo trượt giá",
             "chi_so": [("hệ số cắt", 0.92, "trượt giá thực làm mất 8% so với giả "
                         "định khớp đúng giá stop")]},
            {"ten": "P(chạm dừng lỗ) theo tầm hạn",
             "cong_thuc": "mô phỏng chạm rào trên PHÂN PHỐI z THỰC NGHIỆM (không "
                          "giả định chuẩn), ngưỡng 2σ̂/√h",
             "de_hieu": "Đếm trên dữ liệu thật xem giá đã chạm mức dừng lỗ đó bao nhiêu "
                        "lần, chứ không giả định giá đi theo đường cong lý thuyết.",
             "uoc_tren": f"{len(z_tr):,} phiên huấn luyện của chính cặp này",
             "chi_so": [("bậc tự do t", r2(nu), "đuôi càng dày ν càng nhỏ; ν<10 là "
                         "đuôi rất dày")]},
            {"ten": "Kelly",
             "cong_thuc": "f* = lợi thế / σ̂² — lợi thế lấy từ CARRY đo được, "
                          "KHÔNG dùng dự báo hướng",
             "de_hieu": "Cỡ lệnh tối ưu về dài hạn. Lợi thế duy nhất hệ thống dùng là "
                        "chênh lệch lãi suất giữa hai đồng tiền — KHÔNG dùng dự đoán hướng.",
             "uoc_tren": "carry trung vị 260 phiên gần nhất của cặp này",
             "chi_so": [("carry", r2(1e4 * cr), "bp/ngày")]},
            {"ten": "Trần rủi ro phá sản",
             "cong_thuc": "ngân sách phá sản 1% trên chuỗi tổn thất đuôi t(ν), "
                          "rồi nhân hệ số trượt giá 0,92",
             "de_hieu": "Trần cứng: cỡ lệnh lớn nhất mà xác suất cháy tài khoản vẫn "
                        "dưới 1%, đã tính cả những phiên giá nhảy bất thường.",
             "uoc_tren": "toàn mạch ~26 năm, mọi cấu hình đều không cháy tài khoản",
             "chi_so": [("vốn cuối", "1,004–1,037", "lần — lợi thế NHỎ, đây là sự thật")]},
            {"ten": "Hệ số danh mục 1/√(k+k(k−1)ρ)",
             "cong_thuc": "ρ hiệu dụng đo theo chế độ; căng thẳng thì ρ nhảy lên",
             "de_hieu": "Sáu cặp đều có USD nên chúng cùng thắng cùng thua. Mở nhiều "
                        "lệnh thì phải thu nhỏ từng lệnh, nếu không rủi ro cộng dồn.",
             "uoc_tren": "docs/TANG4_DANHMUC.md",
             "chi_so": [("không cắt", "73,6%", "xác suất phá sản khi mở 6 lệnh cùng "
                         "chiều USD mà không thu nhỏ — thay vì 1%")]},
        ]}


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

    # TRONG SO SONG cua tang to hop truc tuyen — bang chung nhin thay duoc rang
    # he thong dang hoc: chuyen gia nao vua sai nhieu thi trong so tut xuong.
    ts = {}
    for p_ in PAIRS:
        try:
            K = lay(p_)
        except Exception:
            continue
        ts[p_] = {str(h): K["xs"][h].get("trong_so") for h in HS}
    ra["trong_so_truc_tuyen"] = {"nen_theo_h": {str(k): v for k, v in NEN_THEO_H.items()},
                                 "theo_cap": ts}
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
