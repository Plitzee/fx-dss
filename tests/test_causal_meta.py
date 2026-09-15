"""Test cho các nhiệm vụ Tuần 9-10-11:
- Xuất xứ nhân quả tại endpoint /forecast
- Đánh giá độ tin cậy từ Meta-Labeling ngoại sinh tại endpoint /risk
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_forecast_xuat_xu_nhan_qua(client):
    r = client.get("/forecast", params={"pair": "USDJPY", "h": 1})
    assert r.status_code == 200
    b = r.json()
    assert "xuat_xu_nhan_qua" in b and b["xuat_xu_nhan_qua"] is not None
    xx = b["xuat_xu_nhan_qua"]
    assert "VIXCLS" in str(xx["bien_song_sot"])
    assert "Double ML" in xx["phuong_phap_tam_giac"]
    assert "vai_tro_tham_dinh" in xx
    assert "vai_tro_du_bao_truc_tiep" in xx
    assert "vai_tro_rui_ro_meta" in xx


def test_risk_meta_label_usdjpy(client):
    r = client.get("/risk", params={"pair": "USDJPY"})
    assert r.status_code == 200
    b = r.json()
    assert "meta_label_tin_cay" in b
    m = b["meta_label_tin_cay"]
    assert m["ap_dung"] is True
    assert m["bss"] is not None and m["bss"] > 0
    assert m["auc"] is not None and m["auc"] > 0.70
    assert "VIX" in m["bien_ngoai_sinh"]


def test_risk_meta_label_eurusd(client):
    r = client.get("/risk", params={"pair": "EURUSD"})
    assert r.status_code == 200
    b = r.json()
    assert "meta_label_tin_cay" in b
    assert b["meta_label_tin_cay"]["ap_dung"] is True
