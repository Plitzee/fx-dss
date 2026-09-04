"""TANG 6b — KHOP NOI DON BAY (TANG 4) VOI QUYET DINH GIU/DONG (TANG 6b).

BOI CANH. optimal_stop.py toi da hoa E[loi suat log KHONG don bay]. Muc tieu
THAT cua he thong la E[ln(1 + f*R)] — ham LOM, phat duoi trai nang hon —
voi f la don bay ma TANG 4 (PositionSizer) khuyen nghi. Voi E[z]~0 (khong co
loi the huong, tang 1) thi hai ham gan bang nhau NEN DP cu "tinh co" da giam
duoi nhu mot he qua phu (docs/TANG6B_DUNGTOIUU.md, "Gioi han"). Cau hoi o day
la: giai TRUC TIEP cho ham LOM co lam tot hon khong, hay ap don bay SAU LEN
mot bien gioi da chon tu truoc (khong biet se co don bay) la du?

DON BAY DUNG O DAY LA GI. Vi the trong bai toan nay khong co loi the HUONG
(tang 1 da bac bo — mu=0 thi Kelly=0, khong co gi de toi uu). Loi the DUY
NHAT o day la CARRY (tai lieu, va chinh tang 6b da chung minh: "carry la
thu duy nhat tra cong cho viec GIU"). Nen dung dung khung Kelly cua tang 4
nhung voi mu = carry (khung Kelly cho MOT giao dich carry, khong phai giao
dich huong): f_kelly = carry / sigma^2. Tran rui ro (f_ruin_cap*k_vol) cua
tang 4 van ap nhu binh thuong — f = min(Kelly-carry, tran*k_vol). Carry am
thi Kelly am, PositionSizer cat ve 0 — dung y "carry khong bu noi thi dung
giu" ma bang bien gioi DP da tim thay tu truoc, kiem tra cheo duoc.

BA CACH DUNG DE SO SANH TREN CUNG MOT THUOC DO E[ln(1+f*R)]:
    (A) DP-log, khong don bay        — chinh sach cu (mo_phong khong f_v)
    (B) DP-log, don bay AP SAU       — quyet dinh theo bien gioi (A), nhung
                                        CHAM DIEM co don bay (f_v_diem)
    (C) DP-lom, giai TRUC TIEP       — giai() nhan f_v ngay tu dau, quyet
                                        dinh VA cham diem deu co don bay

Neu (C) khong hon (B) co y nghia, nghia la giai truc tiep ham lom KHONG can
thiet — ap don bay sau la du, vi voi don bay carry-Kelly qua nho (~1-2 lan,
xem duoi) do cong lom gan nhu tuyen tinh. Bao cao trung thuc ca hai chieu.

Chay:  python src/compare_leverage_dp.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")

import optimal_stop as O
from split import VALID_TU, TEST_TU
from run_final7 import dm_nw
from position_sizing import PositionSizer, f_kelly

N = 20
SO_VI_THE = 1          # kiem tra tung cap DUNG MOT MINH — khong ap k_danh_muc


def main():
    pan = O.nap_panel()
    tr = np.asarray(pan.Date < VALID_TU)
    te = np.asarray(pan.Date >= TEST_TU)
    mau, nguong = O.rut_mau(pan, tr)
    sig_v = O.sigma_che_do(pan, tr, nguong)
    slip, dist_dung, n_slip = O.truot_trung_binh_sigma()

    # nu (bac tu do Student-t) khop TREN HUAN LUYEN — dung y het decision_record.py
    nu_fit, _, _ = stats.t.fit(pan.zT.values[tr], floc=0)
    nu = float(np.clip(nu_fit, 2.5, 40))
    sizer = PositionSizer(pan.sig.values[tr])

    print("=" * 100)
    print("TẦNG 6b — KHỚP NỐI ĐÒN BẨY TẦNG 4 VỚI QUYẾT ĐỊNH GIỮ/ĐÓNG")
    print("=" * 100)
    print(f"nu (Student-t, huấn luyện) = {nu:.2f}  |  sigma điển hình mỗi chế độ: {sig_v.round(5)}")
    print("đòn bẩy dùng ở đây = khung Kelly TẦNG 4 với mu = carry (không phải hướng — tầng 1 vô hiệu),")
    print("cắt bởi trần rủi ro tầng 4 (f_ruin_cap × k_vol), 1 vị thế đơn lẻ (không có k_danh_mục)")

    NAM_TE = sorted(pan[te].Date.dt.year.unique().tolist())

    print(f"\n{'cặp':<9}" + "".join(f"{'f Kelly-carry '+str(nam):>18}" for nam in NAM_TE))
    print("-" * 100)

    ACC = {"A: DP-log, không đòn bẩy": [], "B: DP-log, đòn bẩy áp sau": [],
           "C: DP-lõm, giải trực tiếp": []}

    for p in O.PAIRS:
        g = pan[pan.pair == p].sort_values("Date").reset_index(drop=True)
        idx = g.index.values
        cr_full = O.carry_ngay(p, g.Date.values)
        c_thoat = O.chi_phi_thoat(p)
        cr_du_phong = float(np.median(cr_full[np.asarray(tr)[idx]]))

        f_hang = []
        for nam in NAM_TE:
            cutoff = pd.Timestamp(f"{nam}-01-01")
            m = g.Date.values < cutoff
            cr_nam = float(np.median(cr_full[m])) if m.sum() >= 30 else cr_du_phong

            # f Kelly-carry moi che do — carry LA "mu" (dung khung Kelly cho
            # giao dich carry, khong phai huong), tran boi truoc tang 4
            f_v = sizer.size(sig_v, cr_nam, nu, dd=0.0, so_vi_the=SO_VI_THE)
            f_hang.append(f_v)

            mask_nam = np.asarray(te)[idx] & (g.Date.dt.year.values == nam)
            if not mask_nam.any():
                continue

            # (A) bien gioi KHONG don bay — dung lai item 3 (cuon theo nam)
            V_log, _ = O.giai(mau, sig_v, cr_nam, c_thoat, slip, N=N, seed=1)
            # (C) bien gioi CO don bay — giai truc tiep ham lom
            V_lev, _ = O.giai(mau, sig_v, cr_nam, c_thoat, slip, N=N, seed=1, f_v=f_v)

            kA, _ = O.mo_phong(g, mask_nam, nguong, V_log, cr_full, c_thoat, slip,
                               sig_v, k_stop=2.0, N=N)  # f_v=None ca hai -> khong don bay
            kB, _ = O.mo_phong(g, mask_nam, nguong, V_log, cr_full, c_thoat, slip,
                               sig_v, k_stop=2.0, N=N, f_v_diem=f_v)  # quyet dinh cu, cham co don bay
            kC, _ = O.mo_phong(g, mask_nam, nguong, V_lev, cr_full, c_thoat, slip,
                               sig_v, k_stop=2.0, N=N, f_v=f_v)  # quyet dinh VA cham deu co don bay

            ACC["A: DP-log, không đòn bẩy"].append(kA["DP"])
            ACC["B: DP-log, đòn bẩy áp sau"].append(kB["DP"])
            ACC["C: DP-lõm, giải trực tiếp"].append(kC["DP"])

        row = f"{p:<9}"
        for fv in f_hang:
            row += f"{np.array2string(fv, precision=2, floatmode='fixed'):>18}"
        print(row)

    print("-" * 100)
    print("  f theo [chế độ 0 (êm), chế độ 1, chế độ 2 (căng)] — carry âm thì Kelly âm, cắt về 0")

    R = {k: np.concatenate(v) for k, v in ACC.items()}
    n_lenh = len(R["A: DP-log, không đòn bẩy"])

    print(f"\nKẾT QUẢ TRÊN ĐOẠN KIỂM TRA — gộp 6 cặp, n mỗi bên = {n_lenh:,}")
    print("thước đo: loi suat log CO DON BAY (ln(1+f*R) cộng dồn từng phiên) — đây là mục tiêu THẬT")
    print("-" * 100)
    print(f"{'chính sách':<28}{'TB (bp)':>10}{'trung vị':>10}{'p5 (bp)':>10}{'CVaR5% (bp)':>13}")
    print("-" * 100)
    for t in R:
        x = R[t]
        q5 = float(np.quantile(x, 0.05))
        cv = float(x[x <= q5].mean())
        print(f"{t:<28}{x.mean()*1e4:>10.2f}{np.median(x)*1e4:>10.2f}{q5*1e4:>10.1f}{cv*1e4:>13.1f}")
    print("-" * 100)

    dAB = R["B: DP-log, đòn bẩy áp sau"] - R["A: DP-log, không đòn bẩy"]
    dCB = R["C: DP-lõm, giải trực tiếp"] - R["B: DP-log, đòn bẩy áp sau"]
    tAB, pAB = dm_nw(dAB)
    tCB, pCB = dm_nw(dCB)
    print(f"\nDM (B − A), trung bình: t={tAB:+.2f}  p={pAB:.3f}"
          + ("  *** có đòn bẩy tốt hơn không có, có ý nghĩa" if pAB < 0.05 and tAB > 0 else
             "  (đòn bẩy carry-Kelly quá nhỏ để đổi kết luận mục tiêu KHÔNG đòn bẩy)"))
    print(f"DM (C − B), trung bình: t={tCB:+.2f}  p={pCB:.3f}"
          + ("  *** giải trực tiếp ham lõm tốt hơn áp đòn bẩy sau, có ý nghĩa" if pCB < 0.05 and tCB > 0
             else "  (giải trực tiếp KHÔNG hơn áp đòn bẩy sau có ý nghĩa — bằng chứng ủng hộ đơn giản hơn)"))
    print("\nDIỄN GIẢI: (B−A) đo đòn bẩy tầng 4 có đáng dùng không (so với không vay gì)."
          "\n           (C−B) đo có ĐÁNG giải lại DP cho đúng hàm lõm, hay áp đòn bẩy sau lên"
          "\n           biên giới log-return cũ là đủ — đây là câu trả lời trực tiếp cho việc")
    print("           khớp nối tầng 4 và tầng 6b.")
    print("\nTỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
