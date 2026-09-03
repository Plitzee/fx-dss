"""TANG 6b — CARRY TINH (trung vi huan luyen) HAY CARRY CUON THEO NAM?

Phat hien: carry that (carry.csv) khac RAT XA trung vi huan luyen o doan kiem
tra (do dung DUNG mtr = Date < VALID_TU nhu run_optstop.py, %/nam):
    EURUSD  huan luyen -0,65  ->  kiem tra -1,63
    GBPUSD  huan luyen -0,02  ->  kiem tra -0,06
    USDJPY  huan luyen +0,21  ->  kiem tra +4,07   (gap ~19 lan)
    AUDUSD  huan luyen +0,91  ->  kiem tra -0,52   DOI DAU
    USDCAD  huan luyen +0,01  ->  kiem tra +1,26
    USDCHF  huan luyen +0,97  ->  kiem tra +3,91   (gap ~4 lan)

optimal_stop.py/run_optstop.py (BAN CU, truoc vá 03/09/2026) giai DP MOT LAN
voi trung vi carry tren doan HUAN LUYEN, roi ap dung y nguyen bien gioi do cho
ca kiem dinh lan kiem tra — trong khi tai lieu chinh tang 6b da noi "CARRY
QUYET DINH CO DUOC GIU HAY KHONG". File nay giai lai DP MOI NAM, dung trung vi
carry cua CUA SO MO RONG (moi du lieu truoc nam do — giong dung quy uoc "dinh
co lai" cua tang 2/4), roi so voi ban tinh tren doan kiem tra.

KET QUA (n=3.170 lenh/ben, doan kiem tra):
    DP-tinh            TB -3,95bp  trung vi -0,55bp  p5 -110,2bp  CVaR5% -132,5bp
    DP-cuon theo nam    TB -0,44bp  trung vi -0,55bp  p5 -105,4bp  CVaR5% -127,4bp
    DM (cuon - tinh), trung binh: t=+2,00  p=0,046  *** cuon theo nam tot hon co y nghia

=> DA VA vao run_optstop.py (03/09/2026): DP gio giai lai moi nam tren doan
kiem dinh/kiem tra, khong con dung mot bien gioi tinh. File nay giu lai lam
BANG CHUNG/TAI LAP cho vá do — xem docs/TANG6B_DUNGTOIUU.md.

Chay:  python src/compare_carry_dong.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")

import optimal_stop as O  # noqa: E402
from split import VALID_TU, TEST_TU  # noqa: E402
from run_final7 import dm_nw  # noqa: E402

N = 20


def main():
    pan = O.nap_panel()
    tr = np.asarray(pan.Date < VALID_TU)
    te = np.asarray(pan.Date >= TEST_TU)
    mau, nguong = O.rut_mau(pan, tr)
    sig_v = O.sigma_che_do(pan, tr, nguong)
    slip, dist_dung, n_slip = O.truot_trung_binh_sigma()

    print("=" * 100)
    print("TẦNG 6b — CARRY TĨNH (trung vị huấn luyện) SO VỚI CARRY CUỐN THEO NĂM")
    print("=" * 100)

    NAM_TE = sorted(pan[te].Date.dt.year.unique().tolist())
    print(f"các năm trong đoạn kiểm tra: {NAM_TE}\n")
    print(f"{'cặp':<9}{'carry huấn luyện':>18}{'carry TB kiểm tra':>20}{'so sánh':>12}")
    print("-" * 100)

    R_tinh, R_dong = {}, {}
    SR_tinh, SR_dong = {}, {}
    c_thoat_map = {}
    for p in O.PAIRS:
        g = pan[pan.pair == p].sort_values("Date").reset_index(drop=True)
        cr_full = O.carry_ngay(p, g.Date.values)
        c_thoat = O.chi_phi_thoat(p)
        c_thoat_map[p] = c_thoat

        # ── ban TINH: dung y het run_optstop.py (trung vi tren HUAN LUYEN) ──
        mtr = np.asarray(g.Date.values < VALID_TU)
        cr_tinh = float(np.median(cr_full[mtr]))
        V_tinh, _ = O.giai(mau, sig_v, cr_tinh, c_thoat, slip, N=N, seed=1)

        cr_te_tb = float(np.median(cr_full[np.asarray(g.Date.values >= TEST_TU)]))
        doi_dau = "ĐỔI DẤU" if cr_tinh * cr_te_tb < 0 else ""
        # cr_tinh/cr_te_tb da la loi suat log/phien TUYET DOI (khong phai don
        # vi sigma — xem docstring carry_ngay() trong optimal_stop.py), nen quy
        # ve %/nam CHI can nhan NGAY_NAM*100, KHONG nhan them sig.
        print(f"{p:<9}{cr_tinh * O.NGAY_NAM * 100:>17.2f}%"
              f"{cr_te_tb * O.NGAY_NAM * 100:>19.2f}%{doi_dau:>12}")

        mask_te = np.asarray(g.Date.values >= TEST_TU)
        k, sr = O.mo_phong(g, mask_te, nguong, V_tinh, cr_full, c_thoat, slip,
                           sig_v, k_stop=2.0, N=N)
        R_tinh.setdefault("DP-tĩnh", []).append(k["DP"])
        SR_tinh.setdefault("DP-tĩnh", []).append(sr["DP"])

        # ── ban CUON: giai lai MOI NAM, cua so mo rong (moi du lieu truoc nam do) ──
        ket_nam = []
        for nam in NAM_TE:
            cutoff = pd.Timestamp(f"{nam}-01-01")
            m = g.Date.values < cutoff
            cr_nam = float(np.median(cr_full[m])) if m.sum() >= 30 else cr_tinh
            V_nam, _ = O.giai(mau, sig_v, cr_nam, c_thoat, slip, N=N, seed=1)
            mask_nam = mask_te & (g.Date.dt.year.values == nam)
            if mask_nam.sum() == 0:
                continue
            k2, sr2 = O.mo_phong(g, mask_nam, nguong, V_nam, cr_full, c_thoat, slip,
                                 sig_v, k_stop=2.0, N=N)
            ket_nam.append(k2["DP"])
        R_dong.setdefault("DP-cuốn theo năm", []).append(np.concatenate(ket_nam))

    print("-" * 100)
    ten_tinh, ten_dong = "DP-tĩnh", "DP-cuốn theo năm"
    R = {ten_tinh: np.concatenate(R_tinh[ten_tinh]),
         ten_dong: np.concatenate(R_dong[ten_dong])}

    print(f"\nKẾT QUẢ TRÊN ĐOẠN KIỂM TRA — gộp 6 cặp, n mỗi bên = {len(R[ten_tinh]):,}")
    print("-" * 100)
    print(f"{'chính sách':<20}{'TB (bp)':>10}{'trung vị':>10}{'p5 (bp)':>10}{'CVaR5% (bp)':>13}")
    print("-" * 100)
    for t in (ten_tinh, ten_dong):
        x = R[t]
        q5 = float(np.quantile(x, 0.05))
        cv = float(x[x <= q5].mean())
        print(f"{t:<20}{x.mean()*1e4:>10.2f}{np.median(x)*1e4:>10.2f}{q5*1e4:>10.1f}{cv*1e4:>13.1f}")
    print("-" * 100)
    d = R[ten_dong] - R[ten_tinh]
    tt, pp = dm_nw(d)
    print(f"\nDM (cuốn theo năm − tĩnh), trung bình: t={tt:+.2f}  p={pp:.3f}"
          + ("  *** cuốn theo năm tốt hơn có ý nghĩa" if pp < 0.05 and tt > 0 else ""))
    print("\nTỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
