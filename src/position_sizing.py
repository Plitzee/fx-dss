"""TANG DINH CO VI THE — quy tac duoc chon sau khi so sanh 9 phuong phap.

    f = min( Kelly,  k_bien_dong × k_sut_giam × tran_rui_ro )

Vi sao la quy tac nay chu khong phai fuzzy hay hoc tang cuong: xem
docs/SIZING_COMPARISON.md. Tom tat ba dong:

  * DIEU KIEN HOA THEO TRANG THAI moi la thu quan trong. Cho he so co gian
    theo sut giam va bien dong cho them ~3,3 diem phan tram tang truong o
    cung muc pha san 0,1%, so voi tran tron.
  * FUZZY MAMDANI khong hon mot tich hai he so tuyen tinh (+0,08%, trong
    nhieu). Chin luat de tai tao mot phep nhan.
  * HOC TANG CUONG khong tim ra dieu kien hoa nay. PPO va CVaR-PPO deu co
    sut giam trong vector trang thai nhung hoc ra he so gan nhu HANG SO
    (bien do 0,018 va 0,030 so voi 0,800 cua quy tac nay). Huan luyen lau
    hon lam bien do NHO DI, khong lon len.

TRUNG THUC VE HE SO: cac gia tri trong K_VOL va K_DD den tu phan doan
chuyen mon, khong phai toi uu tu du lieu. Nhieu ±30% cho tang truong
14,5%–18,5% va pha san toi 1,82% — tuc CO phu thuoc he so. Thu ben vung
la HUONG dieu kien hoa (bien dong cao hoac lo sau thi giam co), khong
phai con so cu the. Luan van phai co bang do nhay va noi ro dieu nay.
"""
import numpy as np

from sizing import f_kelly, f_ruin_cap

# he so theo bien dong: thap / vua / cao  (noi suy muot qua tam phan vi)
K_VOL_HI, K_VOL_LO = 1.30, 0.90
# he so theo sut giam: 1,30 khi o dinh -> 0,50 khi sut giam 25%
K_DD_HI, K_DD_SLOPE, K_DD_FLOOR = 1.30, 3.2, 0.50

# HE SO TRUOT GIA — day KHONG phai phan doan chuyen mon, no la mot SO DO.
# Tran rui ro gia dinh khi cham muc dung lo thi thoat duoc dung o do. Sai:
# 60.617 lan cham muc dung lo do tu nen M1 cho thay truot gia p95 bang 35%
# khoang cach dung lo (src/slippage_model.py). Dua phan phoi truot that vao
# mo phong thi xac suat pha san tang 2,5 lan (0,25% -> 0,61%). Quet he so
# cat cho thay 0,92 dua pha san ve dung muc cu.
K_SLIP = 0.92

# HE SO DANH MUC — cung la SO DO, khong phai phan doan.
# Tran rui ro tra ve don bay cho MOT vi the dung mot minh. Nguoi dung mo nhieu
# cap thi ngan sach pha san bi tieu nhieu lan. Do tren 6 cap: neu quy ve cung
# mot huong USD thi tuong quan trung binh la +0,44 (khong phai 0), nen phan tan
# chi bang mot nua muc "doc lap". Mo 6 lenh cung huong USD, moi lenh o tran day
# du, cho pha san 73,6% trong khi ngan sach la 1%.
#   k_danh_muc = 1 / sqrt( k + k(k-1)*rho )
# Do lai sau khi ap: 0,44%-0,76% cho moi k tu 1 den 6. Xem
# docs/TANG4_DANHMUC.md.
RHO_MAC_DINH = 0.44

# TUONG QUAN VUNG CANG THANG — cung la SO DO, do rieng (src/run_corr_regime.py).
# ρ=0,44 la trung binh TOAN MAU; chia tercile bien dong thi ρ gan nhu phang
# (0,43-0,46) nhung tach dung DUOI CUC DOAN thi khac han: top 5% ngay bien
# dong nhat ρ=0,544 [0,461-0,625]; top 1% ρ=0,594 [0,497-0,684] — khoang tin
# cay KHONG cham 0,44, tuc tang that, dung luc rui ro can duoc kiem soat chat
# nhat. RHO_CANG_THANG lam tron GIUA hai muc do (an toan hon la lay muc thap).
RHO_CANG_THANG = 0.55
NGUONG_CANG_THANG_PCTL = 0.95   # tercile qua tho de thay hieu ung nay — phai dung dung do phan giai da do


def k_danh_muc(so_vi_the, rho=RHO_MAC_DINH):
    """He so phai nhan them khi mo nhieu vi the cung luc.

    so_vi_the=1 tra ve 1,0. rho la tuong quan trung binh giua cac vi the SAU
    khi quy ve cung huong (mua EURUSD va ban USDJPY deu la 'ban USD')."""
    k = int(so_vi_the)
    if k <= 1:
        return 1.0
    rho = float(np.clip(rho, -1.0 / (k - 1) + 1e-9, 1.0))
    return float(1.0 / np.sqrt(k + k * (k - 1) * rho))


class PositionSizer:
    """Khop tam phan vi bien dong tren TAP HUAN LUYEN, roi dinh co tren tap moi."""

    def __init__(self, sig_train, budget=0.01, horizon=250):
        # Vá 03/09/2026: mặc định cũ 0,03 (3%) MÂU THUẪN với ngân sách 1% ghi
        # trong docs/TANG4_DANHMUC.md và trong chính chữ "ngân sách 1%" mà
        # PhieuQuyetDinh.in_ra() in ra — và KHÔNG nơi nào trong repo truyền
        # budget= để ghi đè, nên mọi lệnh gọi thực tế đều âm thầm chạy ở 3%.
        # Xem docs/TANG6_HIEU_CHUAN.md, mục "Vá 03/09/2026".
        sig_train = np.asarray(sig_train, float)
        q = np.quantile(sig_train, [1 / 3, 2 / 3])
        self.s1, self.s2 = float(q[0]), float(q[1])
        self.s_stress = float(np.quantile(sig_train, NGUONG_CANG_THANG_PCTL))
        self.budget, self.horizon = float(budget), int(horizon)

    def k_vol(self, sig):
        """1,30 o bien dong thap, 0,90 o bien dong cao, tuyen tinh o giua."""
        t = np.clip((np.asarray(sig, float) - self.s1) / max(self.s2 - self.s1, 1e-12), 0, 1)
        return K_VOL_HI + (K_VOL_LO - K_VOL_HI) * t

    def k_dd(self, dd):
        """Giam co khi dang lo. Chan duoi 0,50 de khong tat han vi the."""
        return np.clip(K_DD_HI - K_DD_SLOPE * np.asarray(dd, float), K_DD_FLOOR, K_DD_HI)

    def cap(self, sig, nu):
        """Tran rui ro, DA cat bot theo truot gia do duoc."""
        return K_SLIP * f_ruin_cap(sig, self.horizon, self.budget, nu)

    def _rho_hieu_dung(self, sig, rho):
        """Tang rho len RHO_CANG_THANG khi sig cham nguong cuc doan (top 5%
        cua huan luyen) — tuong quan danh muc do duoc TANG THAT o vung nay,
        khong phai gia dinh (src/run_corr_regime.py)."""
        return np.where(np.asarray(sig, float) >= self.s_stress, RHO_CANG_THANG, rho)

    def _k_dm_vec(self, so_vi_the, rho_eff):
        """Ban vector hoa cua k_danh_muc — rho co the la mang (khac nhau theo ngay)."""
        k = int(so_vi_the)
        if k <= 1:
            return np.ones_like(np.asarray(rho_eff, float))
        r = np.clip(np.asarray(rho_eff, float), -1.0 / (k - 1) + 1e-9, 1.0)
        return 1.0 / np.sqrt(k + k * (k - 1) * r)

    def size(self, sig, mu, nu, dd=0.0, lev_max=30.0, so_vi_the=1, rho=RHO_MAC_DINH):
        """Don bay khuyen nghi. sig/dd co the la vector.

        so_vi_the: tong so vi the se mo cung luc (ke ca vi the nay). Bat buoc
        khai bao dung — de mac dinh 1 khi thuc te mo 6 lenh la sai 70 lan."""
        rho_eff = self._rho_hieu_dung(sig, rho)
        k = self.k_vol(sig) * self.k_dd(dd) * self._k_dm_vec(so_vi_the, rho_eff)
        return np.clip(np.minimum(f_kelly(mu, sig), k * self.cap(sig, nu)), 0.0, lev_max)

    def explain(self, sig, mu, nu, dd=0.0, so_vi_the=1, rho=RHO_MAC_DINH, lev_max=30.0):
        """Truy vet tung thanh phan — dau vao cua tang giai thich o tang 6.

        Vá 03/09/2026: `f` giờ CHẶN Ở [0, lev_max] giống hệt `size()` — trước
        đây `explain()` trả `min(kelly, trần)` KHÔNG chặn, nên khi mu (carry)
        âm (điều chỉ lộ ra khi tầng 6 hết dùng mu bịa 0,0002 luôn dương), phiếu
        in ra đòn bẩy ÂM (khuyến nghị bán khống trong một khung "mua"), và vì
        Kelly âm luôn nhỏ hơn trần dương nên trần rủi ro/k_dd mất tác dụng
        hoàn toàn — sụt giảm sâu hơn không còn giảm cỡ được nữa. Xem
        docs/TANG6_HIEU_CHUAN.md, mục "Vá 03/09/2026"."""
        kv, kd = float(self.k_vol(sig)), float(self.k_dd(dd))
        rho_eff = float(self._rho_hieu_dung(sig, rho))
        kdm = float(self._k_dm_vec(so_vi_the, rho_eff))
        cap, kel = float(self.cap(sig, nu)), float(f_kelly(mu, sig))
        eff = kv * kd * kdm * cap
        f = float(np.clip(min(kel, eff), 0.0, lev_max))
        if kel <= 0.0:
            rang_buoc = "không có lợi thế (Kelly ≤ 0)"
        elif f >= lev_max - 1e-9:
            rang_buoc = "trần đòn bẩy"
        else:
            rang_buoc = "Kelly" if kel < eff else "trần rủi ro"
        return dict(kelly=kel, ruin_cap=cap, k_vol=kv, k_dd=kd, k_dm=kdm,
                    rho_hieu_dung=rho_eff, che_do_cang_thang=bool(sig >= self.s_stress),
                    so_vi_the=int(so_vi_the),
                    cap_hieu_chinh=eff, f=f,
                    rang_buoc=rang_buoc,
                    muc_bien_dong=("thấp" if sig < self.s1 else
                                   "cao" if sig > self.s2 else "vừa"))


if __name__ == "__main__":
    import pandas as pd, os
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    p = pd.read_csv(os.path.join(d, os.environ.get("FX_PANEL", "panel2_6pairs.csv")))
    p = p[p.pair == "EURUSD"].reset_index(drop=True)
    n = int(len(p) * 0.70)
    ps = PositionSizer(p.sig.values[:n])
    print("TU KIEM")
    lo, hi = ps.s1 * 0.8, ps.s2 * 1.2
    print(f"  ngưỡng biến động: {ps.s1:.5f} / {ps.s2:.5f}")
    a = float(ps.k_vol(lo)); b = float(ps.k_vol(hi))
    print(f"  k_vol: {a:.3f} (vol thấp) -> {b:.3f} (vol cao)")
    c = float(ps.k_dd(0.0)); e = float(ps.k_dd(0.30))
    print(f"  k_dd : {c:.3f} (ở đỉnh)   -> {e:.3f} (sụt giảm 30%)")
    assert a > b, "biến động cao phải giảm cỡ"
    assert c > e, "sụt giảm sâu phải giảm cỡ"
    assert e >= K_DD_FLOOR - 1e-9, "phải có chặn dưới"
    f0 = ps.size(lo, 0.0002, 6.0, 0.0); f1 = ps.size(lo, 0.0002, 6.0, 0.30)
    print(f"  đòn bẩy: {f0:.2f} (ở đỉnh) -> {f1:.2f} (sụt giảm 30%)")
    assert f1 < f0, "phải giảm đòn bẩy khi sụt giảm"
    ex = ps.explain(float(p.sig.iloc[n]), 0.0002, 6.0, 0.12)
    print(f"  truy vết: ràng buộc={ex['rang_buoc']}, biến động={ex['muc_bien_dong']}, "
          f"k_vol={ex['k_vol']:.2f}, k_dd={ex['k_dd']:.2f}, f={ex['f']:.2f}")
    # bien do dieu kien hoa — con so doi chieu voi RL trong bang so sanh
    amp = float(ps.k_dd(0.0) - ps.k_dd(0.25))
    print(f"  biên độ k theo sụt giảm: {amp:.3f}  (PPO đạt 0,018 · CVaR-PPO 0,030)")
    from sizing import f_ruin_cap as _frc
    assert abs(ps.cap(lo, 6.0) / _frc(lo, 250, ps.budget, 6.0) - K_SLIP) < 1e-9, "phai co cat truot gia"
    print(f"  hệ số trượt giá đang áp: {K_SLIP:.2f} (đo từ 60.617 lần chạm stop)")
    print(f"  ngân sách phá sản: {ps.budget:.0%} (P(vốn tụt dưới 50% trong "
          f"{ps.horizon} phiên) ≤ {ps.budget:.0%} — khớp docs/TANG4_DANHMUC.md)")
    print("  hệ số danh mục:", "  ".join(f"{k}→{k_danh_muc(k):.2f}" for k in (1, 2, 3, 4, 6)))
    assert k_danh_muc(1) == 1.0, "mot vi the thi khong cat"
    assert k_danh_muc(6) < k_danh_muc(3) < k_danh_muc(2) < 1.0, "cang nhieu vi the cang phai cat"
    assert k_danh_muc(6) < 1 / np.sqrt(6), "phai cat manh hon 1/can k vi cac cap co tuong quan"
    f1 = ps.size(lo, 0.0002, 6.0, 0.0, so_vi_the=1)
    f6 = ps.size(lo, 0.0002, 6.0, 0.0, so_vi_the=6)
    print(f"  đòn bẩy mỗi lệnh: {f1:.2f} (mở 1 lệnh) → {f6:.2f} (mở 6 lệnh)")
    assert f6 < f1 / 3, "mo 6 lenh phai cat manh"
    assert amp > 0.7, "bien do phai lon — day chinh la thu RL khong hoc duoc"
    # he so danh muc vung cang thang — rho phai tang, k_danh_muc phai giam
    s_hi = ps.s_stress * 1.5  # chac chan tren nguong cang thang
    ex_binhthuong = ps.explain(hi * 0.5, 0.0002, 6.0, 0.0, so_vi_the=6)
    ex_cangthang = ps.explain(s_hi, 0.0002, 6.0, 0.0, so_vi_the=6)
    print(f"  hệ số danh mục (k=6): bình thường {ex_binhthuong['k_dm']:.3f} "
          f"(ρ={ex_binhthuong['rho_hieu_dung']:.2f}) → vùng căng thẳng "
          f"{ex_cangthang['k_dm']:.3f} (ρ={ex_cangthang['rho_hieu_dung']:.2f})")
    assert ex_cangthang["che_do_cang_thang"] and not ex_binhthuong["che_do_cang_thang"]
    assert ex_cangthang["rho_hieu_dung"] == RHO_CANG_THANG
    assert ex_cangthang["k_dm"] < ex_binhthuong["k_dm"], "vung cang thang phai cat chat hon"
    print("  DAT")
