"""/health va /meta — trang thai du lieu va cong bo giao thuc danh gia."""
import datetime as dt
import json
import os

import pandas as pd
from fastapi import APIRouter

from api.config import HS, KY_NANG_THEO_H, LIVE, MOC_NOI, NEN_THEO_H, PAIRS, TEST_TU, VALID_TU
from api.schemas import HealthResponse, MetaResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
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
        "canh_bao": [
            "Chuỗi trước 2026-01-01 là HistData (tick); từ 2026 là Yahoo (báo giá chỉ dẫn). "
            "Lệch trung vị 0,20–3,40 pip tuỳ cặp — xem trường moi_noi.",
            "Toàn bộ 2026 nằm trong tập khoá sổ của docs/KHOA_SO.md. Phục vụ dữ liệu 2026 "
            "qua API này tiêu một phần niêm phong đó.",
            "Yahoo là endpoint KHÔNG chính thức, không có cam kết dịch vụ.",
            "Chưa có sổ dự báo, nên chỉ số hiệu chuẩn là số tĩnh đo trên đoạn kiểm định.",
        ]}
