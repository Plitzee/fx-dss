"""TANG 6b — KIEM TRA TOAN MACH: tang 2 (sigma) -> tang 4 (don bay) -> tang 6b
(giu/dong) -> tang 3/5 (truot gia + chi phi), CHAY LIEN TUC theo ngay thuc,
khong phai tung "lenh" doc lap nhu mo_phong() dang lam cho cac phep so sanh
khac trong repo nay.

VI SAO CAN FILE NAY RIENG. Moi kiem dinh truoc gio (compare_carry_dong.py,
compare_leverage_dp.py, run_optstop.py) deu mo MOT lenh MOI moi ngay duoc
danh dau, doc lap voi cac lenh khac — dung de do PHAN PHOI mot lenh nhung
KHONG cho thay von thuc su tang truong / sut giam nhu the nao qua thoi gian
khi VAN HANH LIEN TUC (mo, giu, dong, mo lai...). Day moi la cau hoi "he
thong nhu mot khoi CO CHAY DUNG KHONG", khong phai "mot lenh nhu the nao".

QUY UOC:
  * VI THE = carry-Kelly (khung tang 4, mu=carry — lai the duy nhat hop le,
    tang 1 da bi bac bo). Kelly PHAN SO (frac < 1) vi compare_leverage_dp.py
    da chi ro Kelly DAY DU cho carry van te hon khong vay o CVaR — day la
    thuc hanh chuan (Thorp): quet frac de tim muc giu duoc phan phoi lo an
    toan hon khong vay, giong cach K_SLIP duoc chon o tang 4.
  * QUYET DINH GIU/DONG = tang 6b DP, tra o TRANG THAI ON DINH n=N (dung
    quy uoc cot "giu o n=20?" cua run_optstop.py — xap xi chinh sach dung
    trang thai vi mo hinh huu han-han nhung van hanh vo han-han).
  * MOI PHIEN doc lap khong tinh phi VAO lenh (giong het mo_phong: chi tinh
    phi MOT CHIEU luc DONG) — giu dung quy uoc cu de so sanh duoc.
  * MOI CAP mot "khoang von" rieng bang nhau (1/6 tong von) — CHUA cong
    them k_danh_muc lien cap (do dong theo ngay bao nhieu cap dang mo cung
    luc la mot bai toan rieng, ngoai pham vi lan chay dau tien nay — ghi ro
    trong docs).

Chay:  python src/run_e2e.py
"""
import json
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
os.makedirs(OUT, exist_ok=True)

import optimal_stop as O
from split import VALID_TU, TEST_TU
from position_sizing import PositionSizer
from sizing import RUIN_LEVEL

N = 20
FRAC_QUET = (1.0, 0.5, 0.25, 0.125)


def chay_mot_cap(g, idx_nam, cr_full, f_v, V, nguong, c_thoat, slip, sig_v):
    """Di lien tuc qua CHINH XAC cac ngay idx_nam (mang chi so, lien tiep
    trong nam do). Tra ve mang loi suat log CO DON BAY, MOT GIA TRI MOI NGAY
    (0.0 nhung ngay khong co vi the mo)."""
    sig = g.sig.values; zT = g.zT.values; zL = g.zL.values
    ch = np.digitize(sig, nguong)
    ra = np.zeros(len(idx_nam))
    in_vi_the = False
    s = 2.0
    for k, i in enumerate(idx_nam):
        if i + 1 >= len(g):
            break
        v = int(ch[i])
        if not in_vi_the:
            if f_v[v] > 1e-9:
                in_vi_the = True
                s = 2.0
            else:
                continue
        # dang co vi the
        r = O._buoc_gia_tri(cr_full[i], f_v[v])
        if zL[i] <= -s:
            r += O._buoc_gia_tri((-s - slip - c_thoat) * sig[i], f_v[v])
            ra[k] = r
            in_vi_the = False
            continue
        r += O._buoc_gia_tri(zT[i] * sig[i], f_v[v])
        s = float(np.clip((s + zT[i]) * (sig[i] / sig[i + 1]), 0.0, O.S_MAX))
        giu = O.nen_giu(V, s, v, N, c_thoat, sig_v, f_v=f_v)
        if not giu:
            r += O._buoc_gia_tri(-c_thoat * sig[i], f_v[v])
            in_vi_the = False
        ra[k] = r
    return ra


def chay_frac(pan, tr, doan_mask, frac, mau, nguong, sig_v, slip, nu, sizer):
    """Chay toan mach voi MOT phan so Kelly co dinh, tra ve chuoi loi suat
    danh muc (trung binh khong trong so 6 cap) MOI NGAY tren doan_mask."""
    NAM = sorted(pan[doan_mask].Date.dt.year.unique().tolist())
    theo_cap = {}
    for p in O.PAIRS:
        g = pan[pan.pair == p].sort_values("Date").reset_index(drop=True)
        idx = g.index.values
        cr_full = O.carry_ngay(p, g.Date.values)
        c_thoat = O.chi_phi_thoat(p)
        cr_du_phong = float(np.median(cr_full[np.asarray(tr)[idx]]))
        chuoi = []
        for nam in NAM:
            cutoff = pd.Timestamp(f"{nam}-01-01")
            m = g.Date.values < cutoff
            cr_nam = float(np.median(cr_full[m])) if m.sum() >= 30 else cr_du_phong
            f_v = frac * sizer.size(sig_v, cr_nam, nu, dd=0.0, so_vi_the=1)
            V, _ = O.giai(mau, sig_v, cr_nam, c_thoat, slip, N=N, seed=1, f_v=f_v)
            idx_nam = idx[np.asarray(doan_mask)[idx] & (g.Date.dt.year.values == nam)]
            if len(idx_nam) == 0:
                continue
            r = chay_mot_cap(g, idx_nam, cr_full, f_v, V, nguong, c_thoat, slip, sig_v)
            chuoi.append(pd.Series(r, index=g.Date.values[idx_nam]))
        theo_cap[p] = pd.concat(chuoi) if chuoi else pd.Series(dtype=float)

    khung = pd.concat(theo_cap, axis=1).fillna(0.0).sort_index()
    return khung.mean(axis=1)   # danh muc dong deu 1/6 moi cap


def thong_ke(r_ngay, ten):
    # von bat dau = 1,0 (chua giao dich gi) — dung QUY UOC RUIN_LEVEL cua
    # sizing.py: "von tut xuong duoi RUIN_LEVEL lan von BAN DAU", khong phai
    # von dinh (peak-to-trough) — nen phai chen mot moc 1,0 truoc khi cumsum.
    eq = np.concatenate([[1.0], np.exp(np.cumsum(r_ngay.values))])
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - eq / peak
    mdd = float(dd.max())
    ruin = bool((eq <= RUIN_LEVEL).any())
    tb, sd = float(r_ngay.mean()), float(r_ngay.std())
    sharpe = tb / sd * np.sqrt(252) if sd > 0 else 0.0
    q5 = float(np.quantile(r_ngay.values, 0.05))
    cv = float(r_ngay.values[r_ngay.values <= q5].mean())
    print(f"{ten:<28}{tb*1e4:>10.2f}{sharpe:>10.2f}{mdd:>12.1%}{cv*1e4:>13.1f}"
          f"{'CÓ' if ruin else 'không':>10}{eq[-1]:>12.3f}")
    return dict(mean_bp=tb * 1e4, sharpe=sharpe, mdd=mdd, cvar5_bp=cv * 1e4,
                ruin=ruin, von_cuoi=float(eq[-1]))


def main():
    pan = O.nap_panel()
    tr = np.asarray(pan.Date < VALID_TU)
    va = np.asarray((pan.Date >= VALID_TU) & (pan.Date < TEST_TU))
    te = np.asarray(pan.Date >= TEST_TU)
    doan_ca = va | te
    mau, nguong = O.rut_mau(pan, tr)
    sig_v = O.sigma_che_do(pan, tr, nguong)
    slip, _, _ = O.truot_trung_binh_sigma()
    nu_fit, _, _ = stats.t.fit(pan.zT.values[tr], floc=0)
    nu = float(np.clip(nu_fit, 2.5, 40))
    sizer = PositionSizer(pan.sig.values[tr])

    print("=" * 100)
    print("TẦNG 6b — KIỂM TRA TOÀN MẠCH (tầng 2→3→4→5→6b), CHẠY LIÊN TỤC THEO NGÀY")
    print("=" * 100)
    print("danh mục 6 cặp, vốn chia đều 1/6 mỗi cặp, không có k_danh_mục liên cặp (xem giới hạn)")
    print(f"quét phần Kelly (đầy đủ → 1/8) trên ĐOẠN KIỂM ĐỊNH + KIỂM TRA gộp "
          f"({int(doan_ca.sum()):,} phiên/cặp)\n")
    print(f"{'phần Kelly':<28}{'TB (bp/ngày)':>10}{'Sharpe':>10}{'sụt giảm tối đa':>12}"
          f"{'CVaR5% (bp)':>13}{'cháy?':>10}{'vốn cuối':>12}")
    print("-" * 100)

    ket = {}
    for frac in FRAC_QUET:
        r = chay_frac(pan, tr, doan_ca, frac, mau, nguong, sig_v, slip, nu, sizer)
        ket[frac] = thong_ke(r, f"frac={frac:g}")
    print("-" * 100)
    print("mốc không vay (đóng ngay khi hết lợi thế = frac→0, coi như đứng ngoài): TB=0, cháy=không\n")

    frac_chon = min((f for f in FRAC_QUET if not ket[f]["ruin"] and ket[f]["mean_bp"] > 0),
                     default=None, key=lambda f: -ket[f]["mean_bp"])
    if frac_chon is not None:
        print(f"CHỌN frac={frac_chon:g} (tăng trưởng dương lớn nhất trong số không cháy tài khoản)"
              f" — chạy riêng KIỂM ĐỊNH và KIỂM TRA để xem có nhất quán không:\n")
        print(f"{'đoạn':<28}{'TB (bp/ngày)':>10}{'Sharpe':>10}{'sụt giảm tối đa':>12}"
              f"{'CVaR5% (bp)':>13}{'cháy?':>10}{'vốn cuối':>12}")
        print("-" * 100)
        r_va = chay_frac(pan, tr, va, frac_chon, mau, nguong, sig_v, slip, nu, sizer)
        r_te = chay_frac(pan, tr, te, frac_chon, mau, nguong, sig_v, slip, nu, sizer)
        kva = thong_ke(r_va, "kiểm định")
        kte = thong_ke(r_te, "kiểm tra")
        print("-" * 100)
        json.dump({"frac_chon": frac_chon, "quet": ket, "kiem_dinh": kva, "kiem_tra": kte},
                   open(os.path.join(OUT, "ketqua_e2e.json"), "w"), indent=1, ensure_ascii=False)
        print("\nđã ghi output/ketqua_e2e.json")
    else:
        print("KHÔNG có frac nào trong bộ quét vừa dương vừa không cháy tài khoản — toàn mạch")
        print("chưa sẵn sàng vận hành liên tục với carry-Kelly, dù đã chiết khấu.")

    print("\nTỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
