"""Test cho logic chon nguon hien hanh (`api/cache.py::_nap_hien_hanh`).

Tu 15/09/2026 chuoi hien hanh UU TIEN Dukascopy, roi ve Yahoo cho nhung ngay
Dukascopy khong co. Day la duong du lieu SAN XUAT — moi con so sigma^, ba xac
suat, VaR/ES deu di qua no — nen phai co test, va test phai kiem ca truong hop
roi ve lan truong hop nhan `nguon` tung dong.
"""
import os

import pandas as pd
import pytest

import api.cache as C

COT_DUKAS = ["Date", "open", "high", "low", "close", "rv5", "rq5", "bpv5",
             "rsp", "rsn", "n5", "rv_uoc", "n_h1"]


def _ghi(thu_muc, ten, ngays, close=1.1, rv5=1e-6, rv_uoc=0):
    d = pd.DataFrame({
        "Date": pd.to_datetime(ngays), "open": close, "high": close + 1e-3,
        "low": close - 1e-3, "close": close, "rv5": rv5, "rq5": 1e-12,
        "bpv5": rv5 * 0.9, "rsp": rv5 / 2, "rsn": rv5 / 2, "n5": 287,
        "rv_uoc": rv_uoc, "n_h1": 23})
    d.to_csv(os.path.join(thu_muc, ten), index=False)


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "LIVE", str(tmp_path))
    return str(tmp_path)


def test_uu_tien_dukascopy_khi_co_ca_hai(live):
    ngays = ["2026-01-02", "2026-01-05"]
    _ghi(live, "EURUSD_d1_dukas.csv", ngays, close=1.11, rv5=2e-6)
    _ghi(live, "EURUSD_d1.csv", ngays, close=1.22, rv5=9e-6, rv_uoc=1)
    d = C._nap_hien_hanh("EURUSD")
    assert list(d.nguon.unique()) == ["dukascopy"]
    assert d.close.tolist() == [1.11, 1.11], "phải lấy giá của Dukascopy"
    assert d.rv_uoc.tolist() == [0, 0]


def test_yahoo_bu_dung_nhung_ngay_dukascopy_thieu(live):
    _ghi(live, "EURUSD_d1_dukas.csv", ["2026-01-02"], close=1.11)
    _ghi(live, "EURUSD_d1.csv", ["2026-01-02", "2026-01-05"], close=1.22, rv_uoc=1)
    d = C._nap_hien_hanh("EURUSD").sort_values("Date").reset_index(drop=True)
    assert len(d) == 2
    assert d.nguon.tolist() == ["dukascopy", "yahoo"], "nhãn phải đúng TỪNG DÒNG"
    assert d.close.tolist() == [1.11, 1.22]


def test_khong_co_dukascopy_thi_ve_nguyen_yahoo(live):
    _ghi(live, "EURUSD_d1.csv", ["2026-01-02"], close=1.22, rv_uoc=1)
    d = C._nap_hien_hanh("EURUSD")
    assert list(d.nguon.unique()) == ["yahoo"]
    assert d.close.tolist() == [1.22]


def test_xoa_file_dukascopy_la_dao_nguoc_duoc(live):
    """Cách rollback đã hứa trong docstring — phải thật sự hoạt động."""
    _ghi(live, "EURUSD_d1_dukas.csv", ["2026-01-02"], close=1.11)
    _ghi(live, "EURUSD_d1.csv", ["2026-01-02"], close=1.22, rv_uoc=1)
    assert C._nap_hien_hanh("EURUSD").close.iloc[0] == 1.11
    os.remove(os.path.join(live, "EURUSD_d1_dukas.csv"))
    assert C._nap_hien_hanh("EURUSD").close.iloc[0] == 1.22


def test_chi_co_dukascopy_van_chay(live):
    _ghi(live, "EURUSD_d1_dukas.csv", ["2026-01-02"], close=1.11)
    d = C._nap_hien_hanh("EURUSD")
    assert list(d.nguon.unique()) == ["dukascopy"]


def test_khong_co_file_nao_tra_None(live):
    assert C._nap_hien_hanh("EURUSD") is None


def test_file_thieu_cot_bi_bo_qua_chu_khong_lam_hong(live):
    """File Dukascopy hỏng/cụt không được kéo sập cả chuỗi — phải rơi về Yahoo."""
    pd.DataFrame({"Date": pd.to_datetime(["2026-01-02"]), "close": [1.11]}).to_csv(
        os.path.join(live, "EURUSD_d1_dukas.csv"), index=False)
    _ghi(live, "EURUSD_d1.csv", ["2026-01-02"], close=1.22, rv_uoc=1)
    d = C._nap_hien_hanh("EURUSD")
    assert d is not None and list(d.nguon.unique()) == ["yahoo"]


def test_tra_ve_du_cot_hop_dong(live):
    _ghi(live, "EURUSD_d1_dukas.csv", ["2026-01-02"])
    d = C._nap_hien_hanh("EURUSD")
    assert list(d.columns) == C.COT_HH
