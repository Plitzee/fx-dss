"""TANG 6b — CHAY DUNG TOI UU: giai tren HUAN LUYEN, chon tren KIEM DINH,
cham diem MOT LAN tren KIEM TRA.

So bon chinh sach:
    DP                 bien gioi tu quy hoach dong
    giu het tam han    giu toi khi cham stop hoac het N phien
    dong ngay          dong lap tuc (chi tra chi phi thoat)
    dong sau 5 / 10 phien
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

import optimal_stop as O
from split import VALID_TU, TEST_TU
from run_final7 import dm_nw

N = 20


def main():
    pan = O.nap_panel()
    tr = np.asarray(pan.Date < VALID_TU)
    va = np.asarray((pan.Date >= VALID_TU) & (pan.Date < TEST_TU))
    te = np.asarray(pan.Date >= TEST_TU)
    mau, nguong = O.rut_mau(pan, tr)
    sig_v = O.sigma_che_do(pan, tr, nguong)
    slip, dist_dung, n_slip = O.truot_trung_binh_sigma()

    print("=" * 100)
    print("TANG 6b — DUNG TOI UU CHO QUYET DINH GIU / DONG")
    print("=" * 100)
    print(f"trạng thái: khoảng cách tới stop (đơn vị sigma) × {O.N_CHE_DO} chế độ biến động")
    print(f"lưới {len(O.LUOI_S)} điểm × {O.N_CHE_DO} chế độ × {N} bước — giải CHÍNH XÁC, không xấp xỉ")
    print(f"ngưỡng chế độ (sigma, từ huấn luyện): {nguong.round(5)}")
    print(f"sigma điển hình mỗi chế độ: {sig_v.round(5)}")
    print(f"mẫu chuyển trạng thái: " +
          ", ".join(f"chế độ {v}: {len(mau[v]['zT']):,}" for v in range(O.N_CHE_DO)))
    print(f"trượt giá kỳ vọng: {slip:.4f} sigma "
          f"(từ {n_slip:,} lần chạm ở khoảng cách {dist_dung:.1%})")

    carry_map, carry_tb, c_thoat_map = {}, {}, {}
    for p in O.PAIRS:
        g = pan[pan.pair == p].sort_values("Date")
        cr = O.carry_ngay(p, g.Date.values)          # lợi suất log/phiên
        full = np.zeros(len(pan)); full[g.index.values] = cr
        carry_map[p] = full
        mtr = np.asarray(tr)[g.index.values]
        carry_tb[p] = float(np.median(cr[mtr] / g.sig.values[mtr]))   # đơn vị sigma
        c_thoat_map[p] = O.chi_phi_thoat(p)

    print(f"\n{'cặp':<9}{'carry/phiên (sigma)':>22}{'carry %/năm':>14}"
          f"{'chi phí thoát (sigma)':>24}")
    print("-" * 100)
    for p in O.PAIRS:
        g = pan[pan.pair == p]
        ann = carry_tb[p] * float(g.sig.median()) * O.NGAY_NAM * 100
        print(f"{p:<9}{carry_tb[p]:>22.5f}{ann:>13.2f}%{c_thoat_map[p]:>24.4f}")
    print("-" * 100)

    # ── giải DP riêng từng cặp, trên mẫu HUẤN LUYỆN (biên giới KHỞI ĐỘNG, để
    #    minh hoạ — bản dùng để CHẤM ĐIỂM là bản cuốn theo năm ở dưới)
    Vs = {}
    print(f"\nBIÊN GIỚI ĐÓNG LỆNH s* KHỞI ĐỘNG (đóng khi khoảng cách tới stop DƯỚI mức này)")
    print("  đọc: s* = inf nghĩa là đóng ở MỌI trạng thái — carry không bù nổi rủi ro trượt giá")
    print("  (đây là bản GIẢI MỘT LẦN trên carry trung vị huấn luyện, chỉ để minh hoạ cơ chế —")
    print("   xem mục 'cuốn theo năm' bên dưới để biết bản THẬT SỰ dùng khi chấm điểm)")
    print("-" * 100)
    print(f"{'cặp':<9}" + "".join(f"{'chế độ '+str(v):>16}" for v in range(O.N_CHE_DO))
          + f"{'giữ ở n=20?':>16}")
    print("-" * 100)
    for p in O.PAIRS:
        g = pan[pan.pair == p].sort_values("Date")
        mtr = np.asarray(tr)[g.index.values]
        cr_abs = float(np.median(carry_map[p][g.index.values][mtr]))
        V, bien = O.giai(mau, sig_v, cr_abs, c_thoat_map[p], slip, N=N, seed=1)
        Vs[p] = V
        giu20 = [O.nen_giu(V, 2.0, v, N, c_thoat_map[p], sig_v) for v in range(O.N_CHE_DO)]
        row = f"{p:<9}"
        for v in range(O.N_CHE_DO):
            b = bien[N, v]
            row += f"{('đóng hết' if not np.isfinite(b) else f'{b:.2f}σ'):>16}"
        row += f"{('/'.join('G' if x else 'Đ' for x in giu20)):>16}"
        print(row)
    print("-" * 100)

    # ── VÁ 03/09/2026 — carry KHÔNG đứng yên: 3/6 cặp đổi dấu carry giữa huấn
    #    luyện và kiểm tra (docs/TANG6B_DUNGTOIUU.md, src/compare_carry_dong.py).
    #    Giải một lần trên trung vị huấn luyện rồi áp y nguyên cho cả kiểm định
    #    lẫn kiểm tra là dùng biên giới CŨ cho một thế giới carry đã khác — cùng
    #    lỗi phạm vi áp dụng như tầng 4 (xem docs/TANG4_DANHMUC.md mục 2), chữa
    #    bằng đúng quy ước "cửa sổ mở rộng" đã dùng ở tầng 2/4: giải LẠI DP mỗi
    #    năm, carry lấy trung vị của MỌI dữ liệu TRƯỚC năm đó (nhân quả, không
    #    rò rỉ). compare_carry_dong.py đã kiểm định: DM t=+2,00 p=0,046 — cuốn
    #    theo năm tốt hơn có ý nghĩa thống kê trên đoạn kiểm tra.
    NAM_TAT_CA = sorted(set(pan.loc[np.asarray(va) | np.asarray(te), "Date"]
                             .dt.year.tolist()))
    Vs_nam = {p: {} for p in O.PAIRS}
    for p in O.PAIRS:
        g = pan[pan.pair == p].sort_values("Date")
        idx = g.index.values
        cr_full = carry_map[p][idx]
        dates_p = g.Date.values
        cr_du_phong = float(np.median(cr_full[np.asarray(tr)[idx]]))  # nếu năm quá ít dữ liệu
        for nam in NAM_TAT_CA:
            cutoff = pd.Timestamp(f"{nam}-01-01")
            m = dates_p < cutoff
            cr_nam = float(np.median(cr_full[m])) if m.sum() >= 30 else cr_du_phong
            V_nam, _ = O.giai(mau, sig_v, cr_nam, c_thoat_map[p], slip, N=N, seed=1)
            Vs_nam[p][nam] = V_nam

    print(f"\nBIÊN GIỚI CUỐN THEO NĂM — carry trung vị của MỌI dữ liệu TRƯỚC mỗi năm, giải lại DP")
    print("  (bản THẬT SỰ dùng để chấm điểm kiểm định/kiểm tra bên dưới)")
    print("-" * 100)
    print(f"{'cặp':<9}" + "".join(f"{nam:>12}" for nam in NAM_TAT_CA))
    print("-" * 100)
    for p in O.PAIRS:
        row = f"{p:<9}"
        for nam in NAM_TAT_CA:
            giu20 = [O.nen_giu(Vs_nam[p][nam], 2.0, v, N, c_thoat_map[p], sig_v)
                     for v in range(O.N_CHE_DO)]
            row += f"{('/'.join('G' if x else 'Đ' for x in giu20)):>12}"
        print(row)
    print("-" * 100)
    print("  G/Đ theo từng chế độ biến động (bình tĩnh/vừa/căng thẳng) ở trạng thái s=2σ, n=20")

    # ── mô phỏng — dùng biên giới CUỐN THEO NĂM (V khác nhau theo từng năm)
    def chay(mask, ten_doan):
        acc, srate = {}, {}
        for p in O.PAIRS:
            g = pan[pan.pair == p].sort_values("Date").reset_index(drop=True)
            idx = pan[pan.pair == p].sort_values("Date").index.values
            mask_p = np.asarray(mask)[idx]
            nam_p = g.Date.dt.year.values
            for nam in NAM_TAT_CA:
                mask_nam = mask_p & (nam_p == nam)
                if not mask_nam.any():
                    continue
                k, sr = O.mo_phong(g, mask_nam, nguong, Vs_nam[p][nam],
                                   carry_map[p][idx], c_thoat_map[p], slip,
                                   sig_v, k_stop=2.0, N=N)
                for t in k:
                    acc.setdefault(t, []).append(k[t])
                    n_t = len(k[t])
                    if n_t:
                        srate.setdefault(t, []).append((sr[t], n_t))
        # ty le bi stop can QUYEN SO theo so lenh (khong phai trung binh
        # thuong cua trung binh) vi moi (cap, nam) co so lenh khac nhau
        return ({t: np.concatenate(v) for t, v in acc.items()},
                {t: float(np.average([m for m, _ in v], weights=[n for _, n in v]))
                 for t, v in srate.items()})

    for mask, ten in ((va, "KIỂM ĐỊNH"), (te, "KIỂM TRA")):
        R, SR = chay(mask, ten)
        print(f"\n\nKẾT QUẢ TRÊN ĐOẠN {ten} — mở một lệnh mua mỗi phiên, stop 2σ, tầm hạn {N}")
        print("-" * 100)
        print(f"{'chính sách':<22}{'TB (bp)':>10}{'trung vị':>10}{'p5 (bp)':>10}"
              f"{'CVaR5% (bp)':>13}{'bị stop':>10}{'DM vs DP':>11}{'p':>8}")
        print("-" * 100)
        base = R["DP"]
        for t in sorted(R, key=lambda x: -R[x].mean()):
            d = R[t] - base
            tt, pp = (0.0, 1.0) if t == "DP" else dm_nw(d)
            q5 = float(np.quantile(R[t], 0.05))
            cv = float(R[t][R[t] <= q5].mean())
            print(f"{t:<22}{R[t].mean()*1e4:>10.2f}{np.median(R[t])*1e4:>10.2f}"
                  f"{q5*1e4:>10.1f}{cv*1e4:>13.1f}{SR[t]:>10.1%}{tt:>11.2f}{pp:>8.3f}")
        print("-" * 100)
        print(f"  n = {len(base):,} lệnh mô phỏng.  DM DƯƠNG = chính sách đó tốt hơn DP.")
        print(f"  p5 = phân vị 5 của lợi suất mỗi lệnh; CVaR5% = trung bình của 5% tệ nhất.")
        if ten == "KIỂM TRA":
            json.dump({t: dict(mean_bp=float(R[t].mean()*1e4),
                               median_bp=float(np.median(R[t])*1e4),
                               p5_bp=float(np.quantile(R[t], .05)*1e4),
                               cvar5_bp=float(R[t][R[t] <= np.quantile(R[t], .05)].mean()*1e4),
                               stop_rate=SR[t], n=len(R[t])) for t in R},
                      open(os.path.join(OUT, "ketqua_optstop.json"), "w"),
                      indent=1, ensure_ascii=False)
    print("\nđã ghi output/ketqua_optstop.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
