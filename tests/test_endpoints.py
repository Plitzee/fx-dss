"""Smoke test cho API san xuat (api/main.py), chay tren du lieu THAT tren dia
(khong goi mang, khong mock) — dung de khoa hanh vi hien tai truoc khi tach
api/main.py thanh package, va de phat hien hoi quy sau nay.

Chi warm-cache cho MOT cap (EURUSD) o muc module: tinh() cho 6 cap x 3 tam han
+ ACI ton ~20s/cap, nen cac test thuong chay tren cap da nam san trong _bo_nho.
Test nao can quet ca 6 cap duoc danh dau `slow`.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import HS, PAIRS, app

PAIR = "EURUSD"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module", autouse=True)
def warm_cache(client):
    """Chay mot lan de nap _bo_nho[PAIR] truoc khi cac test khac dung no."""
    r = client.get("/forecast", params={"pair": PAIR, "h": 1})
    assert r.status_code == 200
    return r


def test_health_liet_ke_du_ca_sau_cap(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert set(r.json()["cap"].keys()) == set(PAIRS)


def test_meta_cong_bo_giao_thuc_danh_gia(client):
    r = client.get("/meta")
    assert r.status_code == 200
    b = r.json()
    assert set(b["cap"]) == set(PAIRS)
    assert b["tam_han"] == list(HS)
    assert "valid_tu" in b and "test_tu" in b
    assert isinstance(b["canh_bao"], list) and len(b["canh_bao"]) > 0


def test_forecast_ba_xac_suat_cong_lai_bang_1(client):
    r = client.get("/forecast", params={"pair": PAIR, "h": 1})
    assert r.status_code == 200
    b = r.json()
    tong = b["p_giam"] + b["p_ngang"] + b["p_tang"]
    assert tong == pytest.approx(1.0, abs=1e-6)
    assert b["che_do"] in (0, 1, 2)
    assert b["sigma_pip"] > 0


def test_forecast_tu_choi_cap_khong_ton_tai(client):
    r = client.get("/forecast", params={"pair": "XXXYYY", "h": 1})
    assert r.status_code == 404


def test_forecast_tu_choi_tam_han_khong_ho_tro(client):
    r = client.get("/forecast", params={"pair": PAIR, "h": 3})
    assert r.status_code == 400


def test_series_cac_mang_khop_do_dai(client):
    r = client.get("/series", params={"pair": PAIR, "tf": "D1", "n": 200})
    assert r.status_code == 200
    b = r.json()
    n = len(b["ngay"])
    assert n > 0
    for k in ("t", "o", "h", "l", "c"):
        assert len(b[k]) == n


def test_forecast_series_co_du_moi_tam_han(client):
    r = client.get("/forecast_series", params={"pair": PAIR, "n": 100})
    assert r.status_code == 200
    b = r.json()
    assert set(b["tam"].keys()) == {str(h) for h in HS}


def test_calibration_tra_ve_bang_hoac_503_neu_chua_chay(client):
    r = client.get("/calibration")
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert "bang" in r.json()


def test_risk_tra_ve_var_es_kem_backtest(client):
    r = client.get("/risk", params={"pair": PAIR})
    assert r.status_code == 200
    b = r.json()
    assert "var_es" in b and "muc" in b["var_es"]
    alphas = {m["alpha"] for m in b["var_es"]["muc"]}
    assert {0.05, 0.01} <= alphas
    for m in b["var_es"]["muc"]:
        assert m["var_pip"] >= 0 and m["es_pip"] >= m["var_pip"]


def test_events_tra_ve_danh_sach(client):
    r = client.get("/events", params={"tu": "2024-01-01"})
    assert r.status_code == 200
    assert "su_kien" in r.json()


def test_journal_bao_cao_co_mau(client):
    r = client.get("/journal", params={"h": 1})
    assert r.status_code == 200
    assert "tong_du_bao" in r.json()


@pytest.mark.slow
def test_models_quet_du_ca_sau_cap(client):
    r = client.get("/models")
    assert r.status_code == 200
    b = r.json()
    assert set(b["trong_so_truc_tuyen"]["theo_cap"].keys()) == set(PAIRS)
