"""Pydantic response model — chi khai bao cho endpoint co HINH DANG ON DINH,
da doc het code de xac nhan DU truong (khong thieu, khong bia them).

CO Y THUC KHONG khai bao cho /journal, /calibration, /risk, /models,
/forecast_series, /forecast_next, /series, /indicators: cac endpoint nay tra
ve JSON doc thang tu file nghien cuu (chiso_mohinh.json, ml3.json, ...) hoac
co nhanh re lam so truong thay doi theo du lieu. `response_model` cua FastAPI
LOC bo moi truong khong khai bao — khai thieu o day se ROI DUNG vao triet ly
cua du an ("moi con so phai truy duoc, khong giau"). Thay vi khai bao rong
(`Dict[str, Any]`, khong kiem tra duoc gi), de nguyen dang dict thuan va dung
OpenAPI tu sinh tu chu ky ham + docstring.
"""
from typing import Any, Optional

from pydantic import BaseModel


class CapHealth(BaseModel):
    du_lieu_den: Optional[str] = None
    rv5_that_ngay: Optional[int] = None
    tre_ngay: Optional[int] = None
    loi: Optional[str] = None


class HealthResponse(BaseModel):
    ok: bool
    gio_utc: str
    cap: dict[str, CapHealth]


class KyNangMuc(BaseModel):
    muc: str
    chi_tiet: str


class MetaResponse(BaseModel):
    cap: list[str]
    tam_han: list[int]
    nen_theo_h: dict[str, str]
    ky_nang_theo_h: dict[str, KyNangMuc]
    moc_noi_nguon: str
    valid_tu: str
    test_tu: str
    # noi tu moi_noi.json, hinh dang khong do api nay kiem soat
    moi_noi: Optional[Any] = None
    canh_bao: list[str]


class CostResponse(BaseModel):
    pair: str
    med: list[float]
    p95: list[float]


class SuKienNgay(BaseModel):
    ngay: str
    nhan: list[str]


class EventsResponse(BaseModel):
    su_kien: list[SuKienNgay]


class ForecastResponse(BaseModel):
    pair: str
    h: int
    ngay: str
    p_giam: float
    p_ngang: float
    p_tang: float
    dai_pip: float
    sigma_pip: float
    sigma_1_pip: float
    che_do: int
    che_do_ten: str
    nen_12thang_ngang: Optional[float] = None
    kP: float
    c_h: float
    mo_hinh: str
    ky_nang_huong: str
    tap_du_bao: Optional[list[str]] = None
    tap_kich_thuoc: Optional[int] = None
    conformal_alpha: Optional[float] = None
    tap_ghi_chu: str
    ky_nang_do_duoc: KyNangMuc
    tinh_luc: str
