"""VONG 7 — TabPFN v2 (mo hinh NEN cho du lieu bang, Nature 2025) cho HOI QUY
log-RV, DUNG giao thuc voi HAR/ML/DL (khong phai ban phan loai cua run_ml3.py).

`kiem_tabpfn.py` da thu TabPFN cho bai toan PHAN LOAI huong (P/R, muc tieu
khac). File nay la lan dau thu TabPFN cho DUNG bai toan hoi quy bien dong
(log RV) cua vong 7 — cung dac trung (`ml_data.xay`), cung khop lai dau moi
nam, cung cach doi log -> phuong sai.

VI PHAM THIET KE DA BIET (ghi ro, khong lo di):
  * TabPFN v2 thiet ke cho <= 10.000 hang ngu canh. O day KHONG dung
    `ignore_pretraining_limits` de vi pham nhu kiem_tabpfn.py da lam (se
    cho ket qua khong dai dien cho cach TabPFN duoc thiet ke dung) — thay
    vao do GIOI HAN ngu canh o 8.000 hang GAN NHAT (trong vung thiet ke),
    thay vi toan bo cua so mo rong. Day la khac biet CO Y so voi HAR/ML/DL
    khac (dung TOAN BO lich su) — ghi ro trong ket qua.
  * Gia dinh hoan vi duoc (i.i.d.) cua TabPFN van bi vi pham boi chuoi thoi
    gian tai chinh co doi che do — nhu kiem_tabpfn.py da neu.
  * n_estimators=1 (khong GPU, du lieu 52 dac trung x 8.000 hang moi lan
    khop la rat nang cho CPU).
  * TabPFN mac dinh CHAN chay tren CPU voi >1000 hang (canh bao toc do,
    KHONG phai gioi han thiet ke ve chat luong). Phai dat bien moi truong
    `TABPFN_ALLOW_CPU_LARGE_DATASET=1` de vuot qua — day CHI la canh bao
    hieu nang, khac voi vi pham gioi han <=10k hang o tren.

Chay:  python src/run_tabpfn_vol.py
Ghi:   output/_tabpfn_vol_pred.npz
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")

from split import VALID_TU, TEST_TU                     # noqa: E402

KHOP_TU, KHOP_DEN = 2015, 2026
NGU_CANH_TOI_DA = 8000        # trong vung thiet ke cua TabPFN (<=10k)
N_EST = 4             # tăng từ 1 (bản chạy CPU) lên 4 nhờ GPU đủ nhanh — vẫn dưới
                       # mặc định 8 của TabPFN để giữ thời gian hợp lý
EPS = 1e-12


def nap():
    z = np.load(os.path.join(OUT, "_ml_feat.npz"))
    return z["X"], z["y"], pd.DatetimeIndex(z["dts"])


def _tu_kiem_khong_ro_ri(X, y, dts, ok_row):
    """Tu kiem: du bao cho phien t CHI duoc dung du lieu t' < moc(nam cua t).
    Kiem bang cach VO HIEU HOA (khong xoa hang — giu nguyen vi tri toan cuc,
    vi cac cap duoc noi thanh khoi, khong sap xep theo ngay toan cuc) du
    lieu SAU mot moc gia dinh, roi xem ngu canh 8k hang gan nhat co doi
    khong (phai giu nguyen neu chi vo hieu hoa phan SAU diem cuoi ngu canh)."""
    moc = pd.Timestamp("2020-01-01")
    idx_tr1 = np.where(ok_row & (dts < moc))[0]
    ngu_canh1 = idx_tr1[-NGU_CANH_TOI_DA:]
    ok_row2 = ok_row & (dts < pd.Timestamp("2021-06-01"))
    idx_tr2 = np.where(ok_row2 & (dts < moc))[0]
    ngu_canh2 = idx_tr2[-NGU_CANH_TOI_DA:]
    ok = np.array_equal(ngu_canh1, ngu_canh2)
    print(f"  tự kiểm không rò rỉ: cắt dữ liệu SAU điểm huấn luyện không đổi "
          f"ngữ cảnh — {'ĐẠT' if ok else 'THẤT BẠI'}")
    return ok


def main():
    t0 = time.time()
    print("=" * 104)
    print("TabPFN v2 — HỒI QUY log-RV, đúng giao thức vòng 7 (khớp lại đầu mỗi năm)")
    print("=" * 104)
    from tabpfn import TabPFNRegressor
    import torch
    thiet_bi = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"thiết bị: {thiet_bi}"
          + (f" ({torch.cuda.get_device_name(0)})" if thiet_bi == "cuda" else ""))

    X, y, dts = nap()
    rv = np.exp(y)
    ok_row = np.isfinite(X).all(1) & np.isfinite(y)
    assert _tu_kiem_khong_ro_ri(X, y, dts, ok_row), "tự kiểm rò rỉ thất bại — dừng"
    print(f"ngữ cảnh giới hạn {NGU_CANH_TOI_DA:,} hàng gần nhất (trong vùng "
          f"thiết kế TabPFN ≤10.000), n_estimators={N_EST}\n")

    n = len(y)
    mu = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    for yr in range(KHOP_TU, KHOP_DEN + 1):
        moc = pd.Timestamp(f"{yr}-01-01")
        het = pd.Timestamp(f"{yr+1}-01-01")
        idx_tr_full = np.where(ok_row & (dts < moc))[0]
        idx_te = np.where(ok_row & (dts >= moc) & (dts < het))[0]
        if len(idx_tr_full) < 500 or len(idx_te) == 0:
            continue
        idx_tr = idx_tr_full[-NGU_CANH_TOI_DA:]
        Xtr, ytr = X[idx_tr], y[idx_tr]
        Xte = X[idx_te]
        tf = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = TabPFNRegressor(n_estimators=N_EST, random_state=0, device=thiet_bi)
            m.fit(Xtr, ytr)
            mu_te = m.predict(Xte)
            mu_tr = m.predict(Xtr)
        resid = ytr - mu_tr
        mu[idx_te] = mu_te
        s2[idx_te] = float(resid.var())
        print(f"  [{yr}] ngữ cảnh {len(idx_tr):,}  dự báo {len(idx_te):,} phiên"
              f"  ({time.time()-tf:.0f}s, tổng {time.time()-t0:.0f}s)", flush=True)

    f = np.exp(np.clip(mu, -30, 0) + 0.5 * np.nan_to_num(s2))
    va = (dts >= VALID_TU) & (dts < TEST_TU)
    okv = va & np.isfinite(f) & (f > 0)
    r = rv[okv] / f[okv]
    qv = float((r - np.log(r) - 1).mean())
    print(f"\nQLIKE kiểm định: {qv:.4f}")

    np.savez_compressed(os.path.join(OUT, "_tabpfn_vol_pred.npz"),
                        f0=f, ten=np.array(["TabPFN v2 (ngữ cảnh 8k, khớp năm)"]),
                        hp=np.array([f"ctx={NGU_CANH_TOI_DA},n_est={N_EST}"]),
                        qv=np.array([qv]))
    print(f"→ output/_tabpfn_vol_pred.npz · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
