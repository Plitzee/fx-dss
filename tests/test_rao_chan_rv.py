"""Test cho rao chan RV5 that (`api/cache.py::kiem_rv_that`).

VI SAO CAN TEST NAY. Rao chan bao ve mot lỗi AM THAM: neu nguon 5 phut ngung
phuc vu, `live_fx.py` van chay binh thuong, van ghi du lieu, van ra so — chi
khac la rv5 duoc UOC tu thanh gio thay vi do that. Khong co gi hong, khong co
gi bao. Cua so HAR 22 phien vao THANG ma tran thiet ke nen du bao se lech ma
khong ai biet.

Mot rao chan khong bao gio kich hoat thi vo dung, nen test o day kiem CA HAI
chieu: khong bao dong gia khi du lieu lanh, VA phai chan khi du lieu ban.
"""
import numpy as np
import pandas as pd
import pytest

from api.cache import DEM_CANH_BAO, HAR_TRE, kiem_rv_that


def chuoi(n_that_cuoi, n=400):
    """Chuoi gia lap: `n_that_cuoi` phien cuoi la RV5 THAT, phan con lai la uoc."""
    u = np.ones(n, int)
    u[n - n_that_cuoi:] = 0
    return pd.DataFrame({"rv_uoc": u})


@pytest.mark.parametrize("k", [0, 1, 5, HAR_TRE - 1])
def test_chan_khi_cua_so_har_bi_nhiem(k):
    r = kiem_rv_that(chuoi(k))
    assert r["chan"] is True, f"{k} phien that < {HAR_TRE} thi PHAI chan"
    assert r["uoc_trong_cua_so_har"] > 0


@pytest.mark.parametrize("k", [HAR_TRE, HAR_TRE + 5, HAR_TRE + DEM_CANH_BAO - 1])
def test_canh_bao_trong_vung_dem(k):
    r = kiem_rv_that(chuoi(k))
    assert r["chan"] is False, "cua so HAR con sach thi khong duoc chan"
    assert r["canh_bao"] is True, "nhung phai canh bao vi dem sap het"


@pytest.mark.parametrize("k", [HAR_TRE + DEM_CANH_BAO, 59, 120])
def test_khong_bao_dong_gia_khi_du_lieu_lanh(k):
    r = kiem_rv_that(chuoi(k))
    assert r["chan"] is False
    assert r["canh_bao"] is False, "du dem thi khong duoc lam phien"


def test_dem_dung_bang_khoang_cach_toi_nguong_chan():
    assert kiem_rv_that(chuoi(59))["dem"] == 59 - HAR_TRE


def test_mot_khoang_uoc_CHEN_GIUA_van_cat_cua_so():
    """Dai luong quyet dinh la so phien that LIEN TIEP tu cuoi, khong phai tong.

    Day la ca de lot nhat: tong so phien that rat lon (378) nhung co mot khoang
    uoc chen vao sat cuoi, nen cua so HAR 22 phien van ban.
    """
    u = np.zeros(400, int)
    u[-5:] = 1                      # 5 phien CUOI la uoc
    d = pd.DataFrame({"rv_uoc": u})
    r = kiem_rv_that(d)
    assert int((d.rv_uoc == 0).sum()) == 395, "tong so phien that van rat lon"
    assert r["n_that_lien_tiep"] == 0
    assert r["chan"] is True, "nhung phai chan, vi cua so HAR bi nhiem"


def test_thieu_cot_rv_uoc_thi_khong_chan():
    """Chuoi lich su thuan (HistData) khong co cot nay — khong duoc chan oan."""
    r = kiem_rv_that(pd.DataFrame({"close": [1.0, 1.1]}))
    assert r["chan"] is False
    assert r["thieu_cot"] is True


def test_nguong_hai_noi_phai_khop_nhau():
    """`collect/live_fx.py` khong import duoc `api/` nen phai lap lai hang so.

    Lap lai thi de troi. Test nay giu hai ban dong bo — neu ai sua mot ben ma
    quen ben kia, rao chan se bao mot dang o collector va mot dang o API.
    """
    import re
    import os
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "collect", "live_fx.py")
    src = open(f, encoding="utf-8").read()
    lay = lambda ten: int(re.search(rf"^{ten}\s*=\s*(\d+)", src, re.M).group(1))
    assert lay("HAR_TRE") == HAR_TRE
    assert lay("DEM_CANH_BAO") == DEM_CANH_BAO


def test_du_lieu_that_hien_tai_dang_lanh():
    """Chot trang thai do duoc 15/09/2026: 59 phien that lien tiep, dem 37.

    Neu test nay hong thi hoac nguon du lieu da doi, hoac job cap nhat chua
    chay du lau — ca hai deu la thu can biet, khong phai test gion.
    """
    from api.cache import merge_thin_days, noi_chuoi
    r = kiem_rv_that(merge_thin_days(noi_chuoi("EURUSD")))
    assert r["chan"] is False, "du lieu that dang bi chan — kiem collect/live_fx.py"
