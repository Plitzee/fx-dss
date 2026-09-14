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

── CAU TRUC ────────────────────────────────────────────────────────────────
File nay chi con la diem lap rap: tao app, gan router, mount static. Logic
that nam o:
  api/config.py        duong dan, thu vien ben ngoai (balop/chibao/...), hang so
  api/cache.py          noi chuoi, dung san xuat (tinh/lay), nap nen theo khung
  api/utils.py          doi pip, ep kieu JSON, dinh dang tap conformal
  api/risk_logic.py      VaR/ES + xuat xu rui ro
  api/routers/*.py       tung nhom endpoint (meta, market, forecast, risk, admin)

`noi_chuoi`, `PAIRS`, `NEN_THEO_H` duoc TAI XUAT o day vi hang chuc script
trong src/ va jobs/cap_nhat.py van `from api.main import ...` chung — xoa di
se lam gay toan bo pipeline nghien cuu, khong chi API.
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.config import HS, NEN_THEO_H, PAIRS, WEB                    # noqa: F401  (tai xuat)
from api.cache import _idx, _phien_ke_tiep, _tem, lay, nap_khung, noi_chuoi, tinh  # noqa: F401,E501  (tai xuat)
from api.utils import _py, _tap_conformal, pip_size, sang_pip         # noqa: F401  (tai xuat)
from api.routers import admin, forecast, market, meta, risk

app = FastAPI(title="FX-DSS API", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

app.include_router(meta.router)
app.include_router(market.router)
app.include_router(forecast.router)
app.include_router(risk.router)
app.include_router(admin.router)

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
