"""Unit test cho cac ham thuan (khong I/O) trong api/main.py.

Day la lop bao ve re nhat: cac ham nay bi goi hang tram lan trong tung
endpoint, nen mot loi don vi (vi du quen nhan gia truoc khi doi sang pip)
se lam sai MOI con so hien tren giao dien ma khong bao gio bao loi ro rang.
"""
import numpy as np
import pandas as pd
import pytest

from api.main import _phien_ke_tiep, _py, pip_size, sang_pip


def test_pip_size_mac_dinh_la_0_0001():
    assert pip_size("EURUSD") == 0.0001
    assert pip_size("GBPUSD") == 0.0001


def test_pip_size_usdjpy_la_0_01():
    assert pip_size("USDJPY") == 0.01


def test_sang_pip_phai_nhan_gia_truoc_khi_chia_pip():
    # 0,001 (do lech chuan loi suat log) tai gia 1,10 -> 0,001*1,10/0,0001 = 11 pip
    assert sang_pip(0.001, 1.10, "EURUSD") == pytest.approx(11.0)


def test_sang_pip_usdjpy_dung_pip_size_rieng():
    # Day la loi da tung gap: chia thang cho 0,0001 (pip_size mac dinh) se ra
    # sigma sai gap 100 lan (ty le 0,01/0,0001) so voi dung pip_size cua JPY.
    v_dung = sang_pip(0.001, 156.0, "USDJPY")
    v_neu_dung_pip_mac_dinh = 0.001 * 156.0 / 0.0001
    assert v_dung == pytest.approx(0.001 * 156.0 / 0.01)
    assert v_dung == pytest.approx(v_neu_dung_pip_mac_dinh / 100)


def test_sang_pip_nhan_mang():
    v = sang_pip(np.array([0.001, 0.002]), np.array([1.1, 1.2]), "EURUSD")
    assert v == pytest.approx([11.0, 24.0])


def test_py_ep_kieu_numpy_ve_python_thuan():
    out = _py({"a": np.int64(3), "b": np.float64(2.5), "c": np.bool_(True)})
    assert out == {"a": 3, "b": 2.5, "c": True}
    assert isinstance(out["a"], int)
    assert isinstance(out["b"], float)
    assert isinstance(out["c"], bool)


def test_py_doi_nan_va_inf_thanh_none():
    # JSON khong co NaN/Inf hop le — phai doi ve null truoc khi tra ve.
    out = _py({"a": float("nan"), "b": float("inf"), "c": np.float64("nan")})
    assert out == {"a": None, "b": None, "c": None}


def test_py_de_quy_vao_list_va_tuple():
    out = _py([np.int64(1), (np.float64(2.0), float("nan"))])
    assert out == [1, [2.0, None]]


def test_phien_ke_tiep_khong_bao_gio_roi_vao_cuoi_tuan():
    for d in pd.date_range("2026-01-01", periods=14, freq="D"):
        nxt = _phien_ke_tiep(d)
        assert nxt.weekday() < 5
        assert nxt > pd.Timestamp(d)


def test_phien_ke_tiep_tu_thu_sau_nhay_qua_thu_hai():
    thu_sau = pd.Timestamp("2026-01-02")
    assert thu_sau.weekday() == 4, "gia dinh cua test: 2026-01-02 phai la thu Sau"
    assert _phien_ke_tiep(thu_sau) == pd.Timestamp("2026-01-05")
