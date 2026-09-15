"""/health va /meta — trang thai du lieu va cong bo giao thuc danh gia."""
import datetime as dt
import json
import os

import pandas as pd
from fastapi import APIRouter

from api.cache import DEM_CANH_BAO, HAR_TRE, kiem_rv_that, noi_chuoi
from api.config import (HS, KY_NANG_THEO_H, LIVE, MOC_NOI, NEN_THEO_H, PAIRS, TEST_TU,
                        VALID_TU, merge_thin_days)
from api.schemas import HealthResponse, MetaResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
def health():
    ra = {"ok": True, "gio_utc": dt.datetime.utcnow().isoformat() + "Z", "cap": {},
          "rv_chan": False, "rv_canh_bao": False}
    for p in PAIRS:
        f = os.path.join(LIVE, f"{p}_d1.csv")
        if os.path.exists(f):
            d = pd.read_csv(f, parse_dates=["Date"])
            o = {"du_lieu_den": str(d.Date.max().date()),
                 "rv5_that_ngay": int((d.rv_uoc == 0).sum()),
                 "tre_ngay": (dt.datetime.utcnow().date() - d.Date.max().date()).days}
            # rao chan phai do tren chuoi DA NOI, khong phai rieng tep live:
            # cua so HAR 22 phien cua san xuat bac qua moi noi 2026-01-01
            try:
                rv = kiem_rv_that(merge_thin_days(noi_chuoi(p)))
                o.update(rv_that_lien_tiep=rv["n_that_lien_tiep"], rv_dem=rv["dem"],
                         rv_chan=rv["chan"], rv_canh_bao=rv["canh_bao"])
                ra["rv_chan"] = ra["rv_chan"] or rv["chan"]
                ra["rv_canh_bao"] = ra["rv_canh_bao"] or rv["canh_bao"]
                if rv["chan"]:
                    ra["ok"] = False
            except Exception as e:                       # noqa: BLE001
                o["loi"] = f"không kiểm được RV5 thật: {e}"
                ra["ok"] = False
            ra["cap"][p] = o
        else:
            ra["cap"][p] = {"loi": "chưa tải dữ liệu hiện hành"}
            ra["ok"] = False
    return ra


def _trang_thai_rv():
    """Gop trang thai rao chan RV5 that cho ca sau cap, de giao dien hien mot dong."""
    xau = {}
    for p in PAIRS:
        try:
            rv = kiem_rv_that(merge_thin_days(noi_chuoi(p)))
        except Exception:                                # noqa: BLE001
            continue
        if rv["chan"] or rv["canh_bao"]:
            xau[p] = {"lien_tiep": rv["n_that_lien_tiep"], "dem": rv["dem"],
                      "chan": rv["chan"]}
    return {"har_tre": HAR_TRE, "dem_canh_bao": DEM_CANH_BAO, "cap_xau": xau}


@router.get("/meta", response_model=MetaResponse)
def meta():
    mn = os.path.join(LIVE, "moi_noi.json")
    return {
        "cap": list(PAIRS), "tam_han": list(HS),
        "nen_theo_h": {str(k): v for k, v in NEN_THEO_H.items()},
        "ky_nang_theo_h": {str(k): v for k, v in KY_NANG_THEO_H.items()},
        "moc_noi_nguon": str(MOC_NOI.date()),
        "valid_tu": str(VALID_TU.date()), "test_tu": str(TEST_TU.date()),
        "moi_noi": json.load(open(mn, encoding="utf-8")) if os.path.exists(mn) else None,
        "rv_rao_chan": _trang_thai_rv(),
        "canh_bao": [
            "Chuỗi trước 2026-01-01 là HistData (tick); từ 2026 là Yahoo (báo giá chỉ dẫn). "
            "Lệch trung vị 0,20–3,40 pip tuỳ cặp — xem trường moi_noi.",
            "Toàn bộ 2026 nằm trong tập khoá sổ của docs/KHOA_SO.md. Phục vụ dữ liệu 2026 "
            "qua API này tiêu một phần niêm phong đó.",
            "Yahoo là endpoint KHÔNG chính thức, không có cam kết dịch vụ.",
            "Chưa có sổ dự báo, nên chỉ số hiệu chuẩn là số tĩnh đo trên đoạn kiểm định.",
        ]}
