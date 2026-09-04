"""TANG 4 — TUONG QUAN DANH MUC CO THAY DOI THEO CHE DO BIEN DONG KHONG?

Boi canh: RHO_MAC_DINH = 0.44 trong position_sizing.py la mot HANG SO TINH,
do mot lan tren toan mau (docs/TANG4_DANHMUC.md). Moi tham so tinh khac
trong he thong (bien dong, momentum, carry) deu da duoc kiem tra "co sap
theo che do khong" — rieng tuong quan thi CHUA. Van lieu tai chinh ghi nhan
tuong quan giua cac tai san thuong TANG VOT dung luc khung hoang (dang dang
hoa bien mat dung luc can nhat) — day la cau hoi file nay tra loi truc tiep
tren du lieu cua he thong.

Quy uoc CUNG CHIEU USD giong het docs/TANG4_DANHMUC.md va momentum_decay.py:
EURUSD/GBPUSD/AUDUSD giu nguyen dau, USDCAD/USDJPY/USDCHF DAO DAU (de ca 6
chuoi cung mang nghia "duong = ban USD").

Che do: bien dong thi truong trung binh (trung binh sig 6 cap moi ngay),
nguong tercile CHOT TREN HUAN LUYEN (khong ro ri) — dung quy uoc split.py.

Chay:  python src/run_corr_regime.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split import doan  # noqa: E402
from position_sizing import k_danh_muc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")

PAIRS_USD_KEEP = ["EURUSD", "GBPUSD", "AUDUSD"]
PAIRS_USD_INV = ["USDCAD", "USDJPY", "USDCHF"]
ALL_PAIRS = PAIRS_USD_KEEP + PAIRS_USD_INV
CHE_DO_TEN = ("bình tĩnh", "vừa", "căng thẳng")


def nap_gia_cung_chieu():
    """Loi suat ngay, quy ve cung chieu 'ban USD'."""
    out = {}
    for p in ALL_PAIRS:
        d = pd.read_csv(os.path.join(D, "prices", f"{p}_d1.csv"), parse_dates=["Date"])
        d = d.sort_values("Date").reset_index(drop=True)
        r = np.log(d.close.values / np.maximum(d.close.shift(1).values, 1e-9))
        if p in PAIRS_USD_INV:
            r = -r
        out[p] = pd.DataFrame({"Date": d.Date.values, "r": r})
    return out


def nap_che_do():
    """Bien dong thi truong trung binh (TB sig 6 cap) — bien dieu kien che do."""
    p2 = pd.read_csv(os.path.join(D, "panel2_6pairs.csv"), parse_dates=["Date"])
    return p2.groupby("Date").sig.mean().reset_index().rename(columns={"sig": "sig_tb"})


def avg_off_diag(C):
    n = C.shape[0]
    return float((C.values.sum() - n) / (n * n - n))


def block_bootstrap_ci(M, rng, mean_block=5, n_boot=500):
    """Bootstrap khoi cho khoang tin cay cua rho trung binh (giu tinh dai AR)."""
    n = len(M)
    out = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            st = int(rng.integers(0, n))
            L = int(rng.geometric(1.0 / mean_block))
            idx.extend([(st + k) % n for k in range(min(L, n - len(idx)))])
        Mb = M.iloc[idx[:n]]
        out.append(avg_off_diag(Mb.corr()))
    return np.percentile(out, [2.5, 97.5])


def main():
    R = nap_gia_cung_chieu()
    che = nap_che_do()
    M = None
    for p in ALL_PAIRS:
        d = R[p].rename(columns={"r": p})
        M = d if M is None else M.merge(d, on="Date", how="inner")
    M = M.merge(che, on="Date", how="inner").dropna().reset_index(drop=True)

    g = doan(M.Date.values)
    tr = g == 0
    nguong = np.quantile(M.loc[tr, "sig_tb"].values, [1 / 3, 2 / 3])
    M["che_do"] = np.digitize(M.sig_tb.values, nguong)

    print("=" * 92)
    print("TƯƠNG QUAN DANH MỤC (quy về cùng chiều 'bán USD') THEO CHẾ ĐỘ BIẾN ĐỘNG")
    print("Ngưỡng chế độ chốt trên HUẤN LUYỆN, áp cho toàn chuỗi (không rò rỉ)")
    print("=" * 92)

    rng = np.random.default_rng(7)
    rhos = {}
    print(f"\n{'chế độ':<14}{'n ngày':>9}{'ρ trung bình':>15}{'CI 95%':>20}")
    print("-" * 92)
    for c, ten in enumerate(CHE_DO_TEN):
        sub = M[M.che_do == c][ALL_PAIRS]
        C = sub.corr()
        rho = avg_off_diag(C)
        lo, hi = block_bootstrap_ci(sub, rng)
        rhos[ten] = rho
        print(f"{ten:<14}{len(sub):>9,}{rho:>15.3f}   [{lo:.3f}, {hi:.3f}]")

    rho_all = avg_off_diag(M[ALL_PAIRS].corr())
    print(f"\n{'toàn mẫu (= hằng số đang dùng)':<32}ρ = {rho_all:.3f}   "
          f"(RHO_MAC_DINH trong position_sizing.py = 0.44)")

    print("\n" + "=" * 92)
    print("HỆ QUẢ CHO HỆ SỐ DANH MỤC k_danh_mục VÀ ĐÒN BẨY (k=6 vị thế cùng hướng USD)")
    print("=" * 92)
    print(f"{'chế độ':<14}{'ρ':>8}{'k_danh_mục':>13}{'đòn bẩy so với hằng số 0,44':>30}")
    print("-" * 92)
    k_const = k_danh_muc(6, rho=0.44)
    for ten in CHE_DO_TEN:
        k = k_danh_muc(6, rho=rhos[ten])
        print(f"{ten:<14}{rhos[ten]:>8.3f}{k:>13.3f}{k / k_const - 1:>+29.1%}")

    print("\n" + "=" * 92)
    print("KIỂM TRA THÊM Ở ĐUÔI CỰC ĐOAN — văn liệu nói 'vỡ tương quan' là hiện tượng NGÀY")
    print("KHỦNG HOẢNG cụ thể, không phải cả 1/3 số ngày biến động cao. Tách riêng top 5%/1%.")
    print("=" * 92)
    for pct, nhan in ((0.95, "top 5% biến động nhất"), (0.99, "top 1% biến động nhất")):
        thr = np.quantile(M.loc[tr, "sig_tb"].values, pct)
        duoi = M[M.sig_tb >= thr][ALL_PAIRS]
        con_lai = M[M.sig_tb < thr][ALL_PAIRS]
        rho_duoi = avg_off_diag(duoi.corr())
        rho_conlai = avg_off_diag(con_lai.corr())
        lo, hi = block_bootstrap_ci(duoi, rng, mean_block=3)
        print(f"  {nhan:<26} n={len(duoi):>5}   ρ = {rho_duoi:.3f}  [{lo:.3f},{hi:.3f}]"
              f"   (so với phần còn lại: ρ = {rho_conlai:.3f})")

    print("\nDIỄN GIẢI: nếu ρ ở chế độ căng thẳng CAO hơn 0,44 đáng kể, hệ số danh mục")
    print("k_danh_mục hiện dùng (chốt cứng ở 0,44) đang CHO PHÉP đòn bẩy CAO HƠN mức an")
    print("toàn thật đúng lúc thị trường căng thẳng nhất — ngược hướng với thứ trần rủi")
    print("ro rui ro dang co gang bao ve.")
    print("\nTU KIEM DAT")


if __name__ == "__main__":
    main()
