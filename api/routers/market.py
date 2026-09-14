"""/series, /indicators, /cost, /events — du lieu nen va chi bao ky thuat."""
import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.config import CB, D, KHUNG
from api.cache import _tem, lay, nap_khung
from api.schemas import CostResponse, EventsResponse
from api.utils import _py, pip_size

router = APIRouter()


@router.get("/series")
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
    if "n5" in d:
        # so lan lai suat 5 phut da quan sat trong ngay (toi da ~287) — dai
        # dien cho "ngay giao dich co day du du lieu khong", dung de bao cho
        # nguoi dung khi ngay le/ngay mong lam rv5 kem tin cay hon.
        ra["n5"] = [None if pd.isna(v) else int(v) for v in d.n5.values]
    return ra


@router.get("/indicators")
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


@router.get("/cost", response_model=CostResponse)
def cost(pair: str = Query(...)):
    f = os.path.join(D, "cost_table.csv")
    if not os.path.exists(f):
        raise HTTPException(503, "chưa có cost_table.csv")
    c = pd.read_csv(f)
    c = c[(c.pair == pair) & (c.regime == "post2015")].sort_values("hour")
    return {"pair": pair,
            "med": [round(float(v), 3) for v in c.spread_med.values],
            "p95": [round(float(v), 3) for v in c.spread_p95.values]}


@router.get("/events", response_model=EventsResponse)
def events(tu: str = Query("2024-01-01")):
    # uu tien lich da mo rong (collect/lich_su_kien.py), roi ve lich NHTW
    f = os.path.join(D, "su_kien.csv")
    if not os.path.exists(f):
        f = os.path.join(D, "cb_dates.csv")
    if not os.path.exists(f):
        return {"su_kien": []}
    c = pd.read_csv(f, parse_dates=["date"])
    if "ma" in c.columns and "bank" not in c.columns:
        c = c.rename(columns={"ma": "bank"})      # su_kien.csv dung cot `ma`
    c = c[c.date >= pd.Timestamp(tu)]
    c["ngay"] = c.date.astype(str).str[:10]
    return {"su_kien": [{"ngay": k, "nhan": sorted(set(v))}
                        for k, v in c.groupby("ngay").bank.apply(list).items()]}
