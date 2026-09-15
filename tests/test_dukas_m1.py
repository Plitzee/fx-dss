"""Test cho bo tai nen M1 Dukascopy (`collect/dukas_m1.py`).

TRONG TAM la phep giu DONG BO cong thuc. `dukas_m1.do_luong` chep lai nam do
luong noi ngay cua `live_fx.do_luong_noi_ngay` thay vi import — de mot script
thu thap khong phu thuoc script thu thap khac. Cai gia phai tra la hai ban co
the troi khoi nhau, va neu troi thi hai nguon du lieu se cho hai con so khac
nhau cho cung mot ngay MA KHONG CO GI BAO. Test nay la thu giu chung khop.
"""
import datetime as dt
import lzma
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLLECT = os.path.join(ROOT, "collect")
if COLLECT not in sys.path:
    sys.path.insert(0, COLLECT)

import dukas_m1 as DM                                      # noqa: E402
import live_fx as LF                                       # noqa: E402


@pytest.fixture
def chuoi():
    return [1.0, 1.01, 1.005, 1.02, 1.0195, 1.03]


def test_cong_thuc_khop_tung_so_voi_live_fx(chuoi):
    """Neu test nay hong: mot trong hai ban da doi. Sua ca hai, dung sua test."""
    m5 = pd.DataFrame({"close": chuoi})
    ta = DM.do_luong(m5)
    f = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=len(chuoi), freq="5min"),
                      "close": chuoi})
    ref = LF.do_luong_noi_ngay(f, "5").iloc[0]
    for a, b in (("rv5", "rv5"), ("rq5", "rq5"), ("bpv5", "bpv5"),
                 ("rsp", "rsp5"), ("rsn", "rsn5"), ("n5", "n5")):
        assert ta[a] == pytest.approx(ref[b], abs=1e-18), f"lệch ở {a}"


def test_rsp_cong_rsn_bang_rv(chuoi):
    """`volfc2.thiet_ke` dua vao dang thuc nay (chu thich 'khu cung mot he so')."""
    d = DM.do_luong(pd.DataFrame({"close": chuoi}))
    assert d["rsp"] + d["rsn"] == pytest.approx(d["rv5"], abs=1e-18)


def test_rv5_cua_chuoi_khong_doi_bang_khong():
    d = DM.do_luong(pd.DataFrame({"close": [1.1] * 10}))
    assert d["rv5"] == pytest.approx(0.0, abs=1e-18)
    assert d["n5"] == 9


def test_qua_it_nen_thi_tra_None():
    assert DM.do_luong(pd.DataFrame({"close": [1.0, 1.01]})) is None


def test_point_jpy_khac_cac_cap_khac():
    assert DM.diem("USDJPY") == 1e-3
    assert DM.diem("EURJPY") == 1e-3
    assert DM.diem("EURUSD") == 1e-5


def _bi5(n=5, vol_dau=1.0):
    buf = b"".join(DM.REC.pack(t * 60, 110_000 + t, 110_000 + t, 110_000 + t,
                               110_000 + t, vol_dau if t == 0 else 1.0)
                   for t in range(n))
    return lzma.compress(buf, format=lzma.FORMAT_ALONE)


def test_giai_bo_nen_volume_0():
    g = DM.giai(_bi5(5, vol_dau=0.0), "EURUSD", dt.date(2024, 4, 8))
    assert len(g) == 4, "nến volume=0 phải bị bỏ"
    assert g.close.iloc[0] == pytest.approx(1.10001)


def test_giai_ap_dung_point_theo_cap():
    d = dt.date(2024, 4, 8)
    assert DM.giai(_bi5(), "EURUSD", d).close.iloc[0] == pytest.approx(1.10000)
    assert DM.giai(_bi5(), "USDJPY", d).close.iloc[0] == pytest.approx(110.000)


def test_giai_raw_rong_tra_None():
    assert DM.giai(b"", "EURUSD", dt.date(2024, 4, 8)) is None


def test_nen_ngay_lay_dung_ohlc():
    m1 = pd.DataFrame({"ts": pd.date_range("2024-04-08", periods=3, freq="1min"),
                       "open": [1.1, 1.2, 1.15], "high": [1.25, 1.25, 1.2],
                       "low": [1.05, 1.1, 1.1], "close": [1.2, 1.15, 1.18]})
    d, xau = DM.nen_ngay(m1)
    assert (d["open"], d["high"], d["low"], d["close"]) == (1.1, 1.25, 1.05, 1.18)
    assert xau == 0


def test_nen_ngay_bat_duoc_nen_khong_hop_le():
    """high < max(open, close) la nen vo nghia — phai dem, dung im lang."""
    m1 = pd.DataFrame({"ts": pd.date_range("2024-04-08", periods=2, freq="1min"),
                       "open": [1.30, 1.30], "high": [1.10, 1.10],
                       "low": [1.05, 1.05], "close": [1.06, 1.06]})
    _, xau = DM.nen_ngay(m1)
    assert xau == 1


def test_gop_m5_lay_gia_dong_cua_khoang():
    ts = pd.date_range("2024-04-08 00:00", periods=10, freq="1min")
    m1 = pd.DataFrame({"ts": ts, "open": np.arange(10) + 1.0,
                       "high": np.arange(10) + 1.0, "low": np.arange(10) + 1.0,
                       "close": np.arange(10) + 1.0})
    g = DM.gop_m5(m1)
    assert len(g) == 2
    assert g.close.iloc[0] == 5.0      # nen 00:00-00:04 -> dong o phut thu 5
    assert g.close.iloc[1] == 10.0


def test_khoang_niem_phong_dung_nhu_khoa_so():
    """KHOA_SO muc 2 danh rieng 2026-01 -> 2026-08 cho bo niem phong."""
    assert DM.NIEM_PHONG == (dt.date(2026, 1, 1), dt.date(2026, 8, 31))
