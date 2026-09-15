"""/forecast, /forecast_series, /forecast_next, /journal, /calibration,
/models — cac endpoint doc du bao san xuat."""
import json
import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.config import HS, KY_NANG_THEO_H, NEN_THEO_H, PAIRS, ROOT, TEST_TU, V2, VALID_TU
from api.cache import _idx, _phien_ke_tiep, gia_theo_ngay, lay
from api.schemas import ForecastResponse
from api.utils import _py, _tap_conformal, pip_size, sang_pip

router = APIRouter()


@router.get("/forecast", response_model=ForecastResponse)
def forecast(pair: str = Query(...), h: int = Query(1), ngay: str = Query(None)):
    if h not in HS:
        raise HTTPException(400, f"tầm hạn phải thuộc {HS}")
    K = lay(pair)
    i = _idx(K, ngay)
    X = K["xs"][h]
    pan = K["pan"]
    gia_i = float(gia_theo_ngay(K, [pan.Date.values[i]])[0])
    P = X["P"][i]
    nen12 = float(pd.Series(X["P"][:, 1]).rolling(252, min_periods=60).mean().iloc[i])
    return {
        "pair": pair, "h": h, "ngay": str(pan.Date.values[i])[:10],
        "gia": gia_i,
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
        **_tap_conformal(X, i),
        "ky_nang_do_duoc": KY_NANG_THEO_H[h],
        "tinh_luc": K["tinh_luc"].isoformat() + "Z",
        "xuat_xu_nhan_qua": {
            "bien_song_sot": ["VIXCLS (VIX trễ 1 ngày)", "Lịch công bố vĩ mô (ngoài NHTW)"],
            "phuong_phap_tam_giac": ["Double ML", "PCMCI", "Granger (Westfall-Young)", "Causal Forest", "CausalImpact"],
            "vai_tro_tham_dinh": "Dùng để lọc tập đặc trưng đóng băng, chống trôi phân phối so với chọn đơn biến",
            "vai_tro_du_bao_truc_tiep": "Chuyển sang Deferred / Future Work do QLIKE cải thiện không có ý nghĩa thống kê sau hiệu chỉnh Holm (q=1.000)",
            "vai_tro_rui_ro_meta": "Tích hợp vào mô hình meta-labeling tầng rủi ro để nhận diện biến động bất thường và điều chỉnh đòn bẩy / conformal",
        }}


@router.get("/forecast_series")
def forecast_series(pair: str = Query(...), n: int = Query(1500)):
    """Ca chuoi ba xac suat + dai + sigma cho MOI tam han — de giao dien re
    chuot tren bieu do ma khong phai goi lai tung ngay."""
    K = lay(pair)
    ps = pip_size(pair)
    pan = K["pan"]
    n = min(n, len(pan))
    sl = slice(len(pan) - n, len(pan))
    _gia = gia_theo_ngay(K, pan.Date.values[sl])
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
            "nen": NEN_THEO_H[h],
            # tap conformal — giao dien doc de noi "loai tru duoc gi"
            "tap": ([[bool(v) for v in row] for row in X["tap"][sl]]
                    if X.get("tap") is not None else None)}
        s = pd.Series(X["P"][:, 1]).rolling(252, min_periods=60).mean().iloc[-1]
        ra["nen12"][str(h)] = round(float(s), 4) if np.isfinite(s) else 0.33
    return ra


@router.get("/forecast_next")
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


@router.get("/journal")
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


@router.get("/calibration")
def calibration():
    f = os.path.join(ROOT, "output", "nen3.json")
    if not os.path.exists(f):
        raise HTTPException(503, "chưa chạy src/run_balop.py")
    return {"doan": "kiểm định", "valid_tu": str(VALID_TU.date()),
            "test_tu": str(TEST_TU.date()), "bang": json.load(open(f, encoding="utf-8"))}


@router.get("/models")
def models():
    """Chi so cua CAC TANG MO HINH — de giao dien hien duoc, khong phai van xuoi.

      chi_so_sigma : QLIKE, MAE, RMSE, CRPS, PIT, do phu  (src/chiso_mohinh.py)
      ba_lop       : ML/DL/hoc truc tuyen tren ba lop      (src/run_ml3.py)
      quy_luat     : pheu khai pha quy luat                (src/run_quyluat.py)
      bien_dong_14 : 14 mo hinh du bao phuong sai          (vong 7)
      nhan_qua     : pheu ngoai sinh / dan bao thoi gian   (Pha 3B)
    """
    ra = {}
    for khoa, ten in (("chi_so_sigma", "chiso_mohinh.json"),
                      ("ba_lop", "ml3.json"),
                      ("quy_luat", "quyluat.json"),
                      ("nhan_qua", "pha3b_ui.json"),
                      ("nen", "nen_ui.json"),
                      ("bien_dong_14", "ketqua_ml_dl.json"),
                      ("tin_cay", "tincay.json"),
                      ("ngoai_mau", "ngoai_mau.json"),
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
