"""CHIA DU LIEU CHINH THUC — huan luyen / kiem dinh / kiem tra (70/15/15).

Vi sao can file nay. Cho toi truoc vong nay, ca du an chi co 70/30. Nghia la
MOI lua chon — chon mo hinh bien dong, chon so tang cua conformal, chon
K_SLIP, chon dang cua he so danh muc — deu duoc cham diem tren dung cai tap
30% ma sau do lai dung de bao cao ket qua. Do la ro ri lua chon (selection
leakage): con so bao cao se lac quan hon su that, va khong ai biet lac quan
bao nhieu.

CHU Y VE DINH NGHIA. Tang 2 uoc luong lai tham so MOI PHIEN bang cua so mo
rong. Nen "huan luyen" o day KHONG phai mot tap khop co dinh, ma la doan thoi
gian ma minh DUOC PHEP NHIN vao du bao. Ba doan chia theo NGAY (khong theo
chi so) de moi cap cung mot moc:

    huan luyen  ... < 2021-10-13    xay dung, go loi, nhin thoai mai
    kiem dinh   2021-10-13 .. 2023-11-20   CHON mo hinh, chon sieu tham so
    kiem tra    >= 2023-11-20       CHAM DIEM MOT LAN, khong duoc quay lai

Ty le tinh tren 3.649 phien CO DU BAO (chung cho ca 6 cap, tu 2011-12-05 sau
khi tru 500 phien dam cua cua so mo rong): 2.554 / 547 / 548 = 70 / 15 / 15.

GHI CHU TRUNG THUC. Moc kiem tra cu la 2023-03-24; doan kiem tra moi
(2023-11-20 tro di) NAM TRONG doan kiem tra cu, va doan cu da tung duoc cham
diem mot lan o vong truoc. Nen tap nay khong con hoan toan trinh nguyen.
Lop bao ve that su van la TAP KHOA SO (6 cap cheo + toan bo 2026), chua he
duoc mo. Moi con so bao cao tren doan kiem tra phai kem ghi chu nay.

LUAT: bat cu con so nao dung de CHON thi phai lay tu doan kiem dinh. Doan
kiem tra chi duoc cham diem sau khi cau hinh da chot va ghi vao bien ban.
"""
import numpy as np
import pandas as pd

VALID_TU = pd.Timestamp("2021-10-13")
TEST_TU = pd.Timestamp("2023-11-20")
DUBAO_TU = pd.Timestamp("2011-12-05")     # phien dau tien co du bao o moi cap
TEN = ("huấn luyện", "kiểm định", "kiểm tra")
TY_LE = (0.70, 0.15, 0.15)


def doan(dates):
    """Tra ve mang nhan 0/1/2 cho tung ngay."""
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    return np.where(d < VALID_TU, 0, np.where(d < TEST_TU, 1, 2))


def mask(dates, ten):
    """mask(dates, 'kiểm định') -> mang bool."""
    i = TEN.index(ten) if ten in TEN else {"train": 0, "valid": 1, "test": 2}[ten]
    return doan(dates) == i


def tach(df, cot="Date"):
    """Tra ve (huan_luyen, kiem_dinh, kiem_tra) cua mot DataFrame."""
    g = doan(df[cot].values)
    return (df[g == 0].reset_index(drop=True),
            df[g == 1].reset_index(drop=True),
            df[g == 2].reset_index(drop=True))


if __name__ == "__main__":
    import os
    D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    pan = pd.read_csv(os.path.join(D, "panel2_6pairs.csv"), parse_dates=["Date"])
    print("CHIA DU LIEU CHINH THUC — 70/15/15")
    print("-" * 78)
    print(f"{'đoạn':<14}{'từ':>13}{'đến':>13}{'số phiên/cặp':>16}{'tổng':>10}")
    print("-" * 78)
    for i, ten in enumerate(TEN):
        m = doan(pan.Date.values) == i
        s = pan[m]
        print(f"{ten:<14}{str(s.Date.min().date()):>13}{str(s.Date.max().date()):>13}"
              f"{len(s)//6:>16,}{len(s):>10,}")
    print("-" * 78)
    g = doan(pan.Date.values)
    assert set(np.unique(g)) == {0, 1, 2}, "phai co du ba doan"
    for i in range(2):
        assert pan.Date[g == i].max() < pan.Date[g == i + 1].min(), "cac doan phai roi nhau va dung thu tu"
    per = pd.Series(g).groupby([pan.pair.values, g]).size().unstack()
    assert per.notna().all().all(), "moi cap phai co du ba doan"
    assert per.std().max() < 3, "cac cap phai co so phien gan bang nhau moi doan"
    assert VALID_TU < TEST_TU, "moc phai dung thu tu"
    print("  ba đoạn rời nhau, đúng thứ tự thời gian, mọi cặp đều đủ: ĐẠT")
    print(f"  số phiên mỗi cặp theo đoạn:\n{per.to_string()}")
