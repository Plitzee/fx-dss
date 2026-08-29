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


class PositionSizer:
    """Khop tam phan vi bien dong tren TAP HUAN LUYEN, roi dinh co tren tap moi."""

    def __init__(self, sig_train, budget=0.03, horizon=250):
        q = np.quantile(np.asarray(sig_train, float), [1 / 3, 2 / 3])
        self.s1, self.s2 = float(q[0]), float(q[1])
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

    def size(self, sig, mu, nu, dd=0.0, lev_max=30.0):
        """Don bay khuyen nghi. sig/dd co the la vector."""
        k = self.k_vol(sig) * self.k_dd(dd)
        return np.clip(np.minimum(f_kelly(mu, sig), k * self.cap(sig, nu)), 0.0, lev_max)

    def explain(self, sig, mu, nu, dd=0.0):
        """Truy vet tung thanh phan — dau vao cua tang giai thich o tang 6."""
        kv, kd = float(self.k_vol(sig)), float(self.k_dd(dd))
        cap, kel = float(self.cap(sig, nu)), float(f_kelly(mu, sig))
        eff = kv * kd * cap
        return dict(kelly=kel, ruin_cap=cap, k_vol=kv, k_dd=kd,
                    cap_hieu_chinh=eff, f=min(kel, eff),
                    rang_buoc="Kelly" if kel < eff else "trần rủi ro",
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
    assert abs(ps.cap(lo, 6.0) / _frc(lo, 250, 0.03, 6.0) - K_SLIP) < 1e-9, "phai co cat truot gia"
    print(f"  hệ số trượt giá đang áp: {K_SLIP:.2f} (đo từ 60.617 lần chạm stop)")
    assert amp > 0.7, "bien do phai lon — day chinh la thu RL khong hoc duoc"
    print("  DAT")
