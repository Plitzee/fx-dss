"""CỜ BÁO BIẾN ĐỘNG BẤT THƯỜNG — tính năng MINH BẠCH cho giao diện, KHÔNG
phải sửa mô hình rủi ro (xem thảo luận 14/09/2026: đây là hai bài toán khác
nhau — "giải thích sau khi đã xảy ra" khác "sửa mô hình dự báo trước").

THIẾT KẾ CÓ CHỦ ĐÍCH THẬN TRỌNG: KHÔNG tự động gán nguyên nhân cụ thể cho
một ngày biến động bằng cách crawl/đọc tin tức tự do rồi suy diễn bằng NLP —
rủi ro gán sai nguyên nhân cho người dùng cao hơn lợi ích (nhiều tin xảy ra
cùng lúc, thuật toán không biết cái nào thật sự gây ra biến động). Thay vào
đó, làm đúng MỘT việc đáng tin: đối chiếu ngày biến động bất thường với HAI
nguồn lịch ĐÃ CÓ SẴN, ĐÃ ĐƯỢC CON NGƯỜI XÁC NHẬN — không suy đoán:

  1. Lịch công bố ĐỊNH KỲ (`data/su_kien.csv`, `data/cb_dates.csv`) — họp
     NHTW, công bố vĩ mô đã biết trước cả năm.
  2. Lịch can thiệp KHÔNG định kỳ ĐÃ GHI NHẬN (`LICH_CAN_THIEP` — cùng danh
     sách đã dùng ở `va_duoi_canthiep.py`, lấy từ sự kiện đã công khai).

Nếu một ngày biến động bất thường KHÔNG khớp với cả hai nguồn trên, cờ báo
ghi rõ "CHƯA CÓ GIẢI THÍCH TỰ ĐỘNG" — trung thực về giới hạn, không đoán mò.

Chạy độc lập:  python src/bat_thuong.py [PAIR]
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)          # de import duoc `api.main`
DATA = os.path.join(ROOT, "data")

# Cùng danh sách đã dùng ở va_duoi_canthiep.py — MỘT nguồn sự thật duy nhất,
# không định nghĩa lại ở hai nơi.
LICH_CAN_THIEP = {
    "USDJPY": ["2022-09-22", "2022-10-21", "2022-10-24", "2022-12-20"],
    "USDCHF": ["2015-01-15"],
}
NGUONG_MAC_DINH = 3.0     # |z| > 3 do lech chuan — nguong "dang chu y"
CUA_SO_LICH_DINH_KY = 1   # +-1 phien quanh ngay cong bo dinh ky duoc tinh la "khop"


def _lich_dinh_ky():
    f = os.path.join(DATA, "su_kien.csv")
    if not os.path.exists(f):
        f = os.path.join(DATA, "cb_dates.csv")
    if not os.path.exists(f):
        return pd.DataFrame(columns=["date", "ten"])
    c = pd.read_csv(f, parse_dates=["date"])
    if "ten" not in c.columns:
        c["ten"] = c.get("ma", c.get("bank", "sự kiện"))
    return c[["date", "ten"]]


def _khop_dinh_ky(ngay, lich, cua_so=CUA_SO_LICH_DINH_KY):
    m = (lich.date >= ngay - pd.Timedelta(days=cua_so)) & \
        (lich.date <= ngay + pd.Timedelta(days=cua_so))
    ten = sorted(set(lich.loc[m, "ten"].tolist()))
    return ten


def _khop_can_thiep(ngay, pair, cua_so_ngay=10):
    ra = []
    for ns in LICH_CAN_THIEP.get(pair, []):
        t0 = pd.Timestamp(ns)
        if t0 <= ngay <= t0 + pd.Timedelta(days=cua_so_ngay):
            ra.append(ns)
    return ra


def phat_hien(pair, nguong=NGUONG_MAC_DINH, chi_doan=None):
    """Trả về DataFrame các ngày |z| > nguong, kèm cờ giải thích.

    chi_doan: None (mặc định, toàn chuỗi) hoặc mảng bool cùng độ dài để giới
    hạn (vd chỉ xét đoạn kiểm tra khi CHẨN ĐOÁN — không dùng để CHỌN mô hình).
    """
    from api.main import noi_chuoi
    import volfc2 as V2
    from volfc import merge_thin_days

    m = merge_thin_days(noi_chuoi(pair))
    sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, pair), 0.0))
    c = m.close.values
    z = np.full(len(m), np.nan)
    z[1:] = np.log(c[1:] / np.maximum(c[:-1], 1e-12)) / np.maximum(sig[1:], 1e-12)

    mask = np.abs(z) > nguong
    if chi_doan is not None:
        mask &= np.asarray(chi_doan, bool)

    lich = _lich_dinh_ky()
    hang = []
    for i in np.flatnonzero(mask):
        ngay = pd.Timestamp(m.Date.values[i])
        dinh_ky = _khop_dinh_ky(ngay, lich)
        can_thiep = _khop_can_thiep(ngay, pair)
        if can_thiep:
            giai_thich = f"can thiệp đã biết: {', '.join(can_thiep)}"
        elif dinh_ky:
            giai_thich = f"lịch định kỳ: {', '.join(dinh_ky[:3])}"
        else:
            giai_thich = "CHƯA CÓ GIẢI THÍCH TỰ ĐỘNG — đáng chú ý"
        hang.append(dict(ngay=str(ngay.date()), z=round(float(z[i]), 3),
                         co_giai_thich=bool(dinh_ky or can_thiep),
                         giai_thich=giai_thich))
    return pd.DataFrame(hang)


if __name__ == "__main__":
    pair = sys.argv[1] if len(sys.argv) > 1 else "USDJPY"
    df = phat_hien(pair)
    n_chua_giai_thich = int((~df.co_giai_thich).sum()) if len(df) else 0
    print(f"{pair}: {len(df)} ngày bất thường (|z|>{NGUONG_MAC_DINH}), "
         f"{n_chua_giai_thich} CHƯA có giải thích tự động")
    if len(df):
        print(df.to_string(index=False))
