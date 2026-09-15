"""Test cho job cap nhat hang ngay (`jobs/cap_nhat.py`).

Trong tam: hai hang so lap lai o noi khac, va lop chan do moi du lieu phai
nhin duoc CA hai nguon hien hanh. Tu 15/09/2026 Dukascopy la nguon chinh; neu
lop chan chi soi tep Yahoo thi no se mu doi voi dung cai nguon dang nuoi du bao.
"""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = os.path.join(ROOT, "jobs")
if JOBS not in sys.path:
    sys.path.insert(0, JOBS)

import cap_nhat as J                                        # noqa: E402
from api.config import MOC_NOI                              # noqa: E402


def test_moc_noi_khop_api_config():
    """Hai ban hang so nay phai bang nhau, neu khong job se tai sai khoang."""
    assert pd.Timestamp(J.MOC_NOI) == MOC_NOI


def _ghi(thu_muc, ten, ngays):
    pd.DataFrame({"Date": pd.to_datetime(ngays), "close": 1.1}).to_csv(
        os.path.join(thu_muc, ten), index=False)


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "LIVE", str(tmp_path))
    return str(tmp_path)


def test_tep_hien_hanh_gom_ca_hai_nguon(live):
    _ghi(live, "EURUSD_d1_dukas.csv", ["2026-01-02"])
    _ghi(live, "EURUSD_d1.csv", ["2026-01-02"])
    nhan = dict(J._tep_hien_hanh())
    assert "EURUSD (dukascopy)" in nhan
    assert "EURUSD (yahoo)" in nhan


def test_glob_yahoo_khong_nuot_tep_dukascopy(live):
    """`*_d1.csv` không được khớp `*_d1_dukas.csv` — nếu khớp thì một tệp bị
    đếm hai lần dưới hai nhãn và cảnh báo sẽ vô nghĩa."""
    _ghi(live, "EURUSD_d1_dukas.csv", ["2026-01-02"])
    duong = [f for _, f in J._tep_hien_hanh()]
    assert len(duong) == 1, "chỉ có một tệp trên đĩa thì chỉ được liệt kê một lần"


NGAY8 = pd.bdate_range("2026-01-02", periods=8).strftime("%Y-%m-%d").tolist()


def test_ngay_moi_nhat_dung_cung_nhan_voi_kiem_tra(live):
    """Hai hàm phải sinh CÙNG dạng khoá, nếu không phép so luôn trượt và lớp
    chặn im lặng mất tác dụng."""
    _ghi(live, "EURUSD_d1_dukas.csv", NGAY8)
    truoc = J._ngay_moi_nhat_hien_co()
    assert set(truoc) == {"EURUSD (dukascopy)"}
    J.kiem_tra_du_lieu_moi(truoc)          # không được ném: dữ liệu không lùi


def test_chan_khi_du_lieu_lui_ngay(live):
    _ghi(live, "EURUSD_d1_dukas.csv", NGAY8)
    truoc = J._ngay_moi_nhat_hien_co()
    _ghi(live, "EURUSD_d1_dukas.csv", NGAY8[:-1])          # lùi một ngày
    with pytest.raises(SystemExit, match="LÙI"):
        J.kiem_tra_du_lieu_moi(truoc)


def test_chan_khi_tep_qua_it_dong(live):
    """Lớp chặn 'chỉ N dòng' — đã bắt được đúng một fixture sai khi viết test này."""
    _ghi(live, "EURUSD_d1_dukas.csv", ["2026-01-02", "2026-01-05"])
    with pytest.raises(SystemExit, match="nghi ngờ dữ liệu rỗng"):
        J.kiem_tra_du_lieu_moi({})


def test_chan_khi_live_rong(live):
    with pytest.raises(SystemExit, match="rỗng"):
        J.kiem_tra_du_lieu_moi({})
