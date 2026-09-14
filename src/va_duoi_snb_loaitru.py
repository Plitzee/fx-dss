"""LOAI TRU CUA SO SNB cho USDCHF — huong vach rieng cho USDCHF, khac han
USDJPY (xem va_duoi_evt_trongso.py). Co che da chan doan (va_duoi_diemgay.py):
USDCHF chi co 5 diem gay trong 12+ nam, va cu soc lon nhat (+200,3%/-71,0%
o sd(z) cuon 60 phien) khop CHINH XAC voi SNB bo san EUR/CHF 15/01/2015 —
mot su kien LICH SU, MOT LAN, nam trong doan HUAN LUYEN. Khong co diem gay
dang ke nao khac gan moc kiem tra. Day KHAC han USDJPY (troi dan lien tuc):
USDCHF la MOT diem o nhiem tinh, nen huong vas dung la LOAI TRU/HA TRONG SO
cua so do khi uoc luong duoi, KHONG PHAI conformal co trong so theo dich
chuyen (khong co dich chuyen lien tuc de sua).

LY DO KINH TE DA BIET TRUOC (khong phai chon theo do tren kiem dinh):
SNB bo san la mot QUYET DINH CHINH SACH cong bo, ngay thang xac dinh, da
duoc chinh PELT tim ra doc lap — khong phai "thu nhieu cua so roi chon cai
dep nhat". Day la dung "loi ra (a)" da thong nhat: chon theo ly do kinh te,
khong theo hieu nang do duoc.

Chay:  python src/va_duoi_snb_loaitru.py
Ghi:   output/va_duoi_snb_loaitru.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import optimize as _opt
from scipy import stats as _st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "output")
PANEL = os.path.join(ROOT, "data", "panel2_6pairs.csv")

TRAIN_TU = pd.Timestamp("2012-02-14")     # dau panel2 (V0 khop tren doan huan luyen)
VALID_TU = pd.Timestamp("2021-10-13")
TEST_TU = pd.Timestamp("2023-11-20")
PAIR = "USDCHF"

# Cua so SNB — DUNG TU DIEM GAY DA DO boi PELT o va_duoi_diemgay.py, khong
# dat tay: diem gay dau (2015-01-14) den diem gay ke tiep (2015-04-08), noi
# sd(z) cuon 60 phien tro lai muc binh thuong.
SNB_TU = pd.Timestamp("2015-01-14")
SNB_DEN = pd.Timestamp("2015-04-08")
MUC_DUOI = (0.90, 0.95, 0.99)


def main():
    t0 = time.time()
    print("=" * 96)
    print(f"LOẠI TRỪ CỬA SỔ SNB (2015-01-14 → 2015-04-08) — {PAIR}")
    print("=" * 96)

    pan_all = pd.read_csv(PANEL, parse_dates=["Date"])
    pan = pan_all[pan_all.pair == PAIR].sort_values("Date")
    dat = pan.Date
    m_tr = (dat >= TRAIN_TU) & (dat < VALID_TU)
    z_tr_full = pan[m_tr].zT.values
    m_snb = (dat >= SNB_TU) & (dat <= SNB_DEN) & m_tr
    n_snb = int(m_snb.sum())
    print(f"n huấn luyện = {m_tr.sum()} · trong đó {n_snb} phiên nằm trong cửa sổ SNB "
          f"({100*n_snb/m_tr.sum():.2f}%)")

    z_tr_loaitru = pan[m_tr & ~m_snb].zT.values

    def studentt_va_phanvi(z):
        nu, _, sc = _st.t.fit(z, floc=0)
        nu = float(np.clip(nu, 2.5, 40))
        ra = {}
        for muc in MUC_DUOI:
            ra[str(muc)] = float(_st.t.ppf((1 + muc) / 2, nu) * sc)
        return nu, sc, ra

    def thucnghiem_phanvi(z):
        az = np.abs(z)
        return {str(muc): float(np.quantile(az, muc)) for muc in MUC_DUOI}

    nu0, sc0, tt0 = studentt_va_phanvi(z_tr_full)
    nu1, sc1, tt1 = studentt_va_phanvi(z_tr_loaitru)
    tn0 = thucnghiem_phanvi(z_tr_full)
    tn1 = thucnghiem_phanvi(z_tr_loaitru)

    print(f"\nStudent-t ĐẦY ĐỦ (có SNB):   ν={nu0:.2f}, scale={sc0:.4f}")
    print(f"Student-t LOẠI TRỪ (không SNB): ν={nu1:.2f}, scale={sc1:.4f}")

    print(f"\n{'mức':>6}{'TN có SNB':>12}{'TN loại trừ':>13}{'t có SNB':>11}"
          f"{'t loại trừ':>12}{'đổi (t)':>10}")
    print("-" * 66)
    ra = {"pair": PAIR, "n_huanluyen": int(m_tr.sum()), "n_snb": n_snb,
         "studentt": {"co_snb": {"nu": round(nu0, 3), "scale": round(sc0, 5)},
                     "loai_tru": {"nu": round(nu1, 3), "scale": round(sc1, 5)}},
         "muc": {}}
    for muc in MUC_DUOI:
        doi = 100 * (tt1[str(muc)] / tt0[str(muc)] - 1)
        ra["muc"][str(muc)] = {
            "thucnghiem_co_snb": round(tn0[str(muc)], 4),
            "thucnghiem_loai_tru": round(tn1[str(muc)], 4),
            "studentt_co_snb": round(tt0[str(muc)], 4),
            "studentt_loai_tru": round(tt1[str(muc)], 4),
            "doi_studentt_%": round(doi, 1)}
        print(f"{muc:>6.2f}{tn0[str(muc)]:>12.4f}{tn1[str(muc)]:>13.4f}"
              f"{tt0[str(muc)]:>11.4f}{tt1[str(muc)]:>12.4f}{doi:>+9.1f}%")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "va_duoi_snb_loaitru.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)

    print("\n" + "-" * 96)
    print("→ Diễn giải: nếu Student-t LOẠI TRỪ cho ES/VaR NHỎ HƠN đáng kể ở mức 99%,")
    print("  khớp với chẩn đoán 'ES thừa 15%' (mô hình hiện quá bảo thủ vì cú sốc SNB")
    print("  làm phồng đuôi đã khớp) — loại trừ sẽ đưa mô hình về gần thực tế hơn.")
    print(f"\n→ output/va_duoi_snb_loaitru.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
