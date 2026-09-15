"""Kiểm thử tầng quản trị rủi ro đuôi (Tail-Risk Recalibration) theo đề cương:
- Loại trừ cửa sổ điểm gãy SNB 2015 cho USDCHF
- Adaptive Conformal Inference (ACI) và Conformal PID Control cho USDJPY
- Kiểm tra tính hợp lệ của endpoint /risk cho các cặp trọng tâm
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.cache import lay
from api.main import app
from split import doan
import src.conformal_risk as CR


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_conformal_risk_aci_step():
    # Khi có vi phạm (err = 1), alpha_t phải giảm để mở rộng phân vị
    a_init = 0.01
    a_sau_vipham = CR.cap_nhat_aci(a_init, a_target=0.01, err=1.0, gamma=0.01)
    assert a_sau_vipham < a_init
    assert a_sau_vipham == pytest.approx(0.01 + 0.01 * (0.01 - 1.0))

    # Khi không có vi phạm (err = 0), alpha_t tăng nhẹ để siết lại
    a_sau_an_toan = CR.cap_nhat_aci(a_init, a_target=0.01, err=0.0, gamma=0.01)
    assert a_sau_an_toan > a_init


def test_conformal_risk_pid_step():
    a_init = 0.01
    a_sau, integ = CR.cap_nhat_pid(a_init, a_target=0.01, err=1.0, kp=0.01, ki=0.001, kd=0.002)
    assert a_sau < a_init
    assert integ < 0  # Tích luỹ sai số âm khi vi phạm


def test_usdchf_snb_exclusion():
    K = lay("USDCHF")
    pan = K["pan"]
    dat = pan.Date.values
    tr = doan(dat) == 0

    m_snb = (dat >= np.datetime64("2015-01-14")) & (dat <= np.datetime64("2015-04-08"))
    assert m_snb.sum() > 50  # 61 phiên

    z_full = pan.zT.values[tr]
    z_clean = pan.zT.values[tr & ~m_snb]
    z_full = z_full[np.isfinite(z_full)]
    z_clean = z_clean[np.isfinite(z_clean)]

    from scipy import stats as _st
    nu_full = float(_st.t.fit(z_full, floc=0)[0])
    nu_clean = float(_st.t.fit(z_clean, floc=0)[0])

    # Khử SNB làm tăng bậc tự do (đuôi bớt bị phồng nhân tạo)
    assert nu_clean > nu_full
    assert nu_clean >= 9.0


def test_risk_endpoint_usdjpy_aci(client):
    r = client.get("/risk", params={"pair": "USDJPY"})
    assert r.status_code == 200
    b = r.json()
    assert "var_es" in b
    m1 = [m for m in b["var_es"]["muc"] if m["alpha"] == 0.01][0]
    assert "ACI" in m1["phuong_phap"]
    assert "alpha_hieu_chinh" in m1
    assert m1["alpha_hieu_chinh"] > 0
    assert m1["kupiec_p"] >= 0.05  # ACI giúp Kupiec không bị bác bỏ


def test_risk_endpoint_usdchf_snb(client):
    r = client.get("/risk", params={"pair": "USDCHF"})
    assert r.status_code == 200
    b = r.json()
    assert "var_es" in b
    m1 = [m for m in b["var_es"]["muc"] if m["alpha"] == 0.01][0]
    assert "SNB" in m1["phuong_phap"]
