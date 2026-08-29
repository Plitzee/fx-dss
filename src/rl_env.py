"""
MOI TRUONG RL CHO BAI TOAN DINH CO VI THE.

Diem khac biet so voi da so cong trinh "DRL for trading":
  * Agent KHONG chon huong. Huong do nguoi dung dua vao (nhu Kelly). Agent chon KHOI LUONG.
  * Trang thai la cac dac trung RUI RO DA KIEM DINH HIEU CHINH o giai doan 2-3
    (sigma du bao, xac suat cham stop theo nguyen ly phan xa, sut giam hien tai),
    KHONG phai gia tho hay chi bao ky thuat.
  * Chuan so sanh la quy tac tinh min(Kelly, tran rui ro) cua giai doan 4,
    khong phai buy-and-hold.

Toan bo vector hoa tren N duong song song bang numpy — khong can GPU.
"""
import numpy as np
from scipy import stats

LEV_MAX = 30.0
RUIN_LEVEL = 0.50
MAINT = 0.5 / LEV_MAX          # nguong margin close-out theo ty le lo trong phien
COST = 1.0e-4 / 1.1            # 1 pip khu hoi
N_STATE = 6
RUIN_PEN = 1.0                 # phat khi pha san, cung bac do lon voi phan thuong


class SizingEnv:
    def __init__(self, panel, horizon=250, block=5, rebalance=5,
                 sharpe_true=0.5, belief_noise=0.0, seed=0):
        z = panel.zT.values.astype(float).copy()
        zl = panel.zL.values.astype(float).copy()
        m0 = z.mean()
        self.zT, self.zL = z - m0, zl - m0          # khu drift noi sinh
        self.sig = panel.sig.values.astype(float)
        self.sig_bar = float(np.median(self.sig))
        self.horizon, self.block, self.rebalance = horizon, block, rebalance
        # sharpe_true co the la mot so, hoac (lo, hi) -> rut ngau nhien theo tung duong.
        # PHAI ngau nhien khi huan luyen, neu khong dac trung 'loi the tin' se HOAN TOAN
        # cong tuyen voi 1/sigma va agent khong the phan biet duoc hai thu.
        self.sharpe_true = sharpe_true
        self.belief_noise = belief_noise            # do lech chuan log cua sai so niem tin
        self.rng = np.random.default_rng(seed)
        self.nu = float(np.clip(stats.t.fit(self.zT, floc=0)[0], 2.5, 40))
        self.scale_t = stats.t.fit(self.zT, floc=0)[2]

    # ── dac trung rui ro: xac suat cham stop 1%, theo nguyen ly phan xa (GD 3) ──
    def p_touch(self, sig, stop=0.01):
        b = stop / np.maximum(sig, 1e-9) / self.scale_t
        return np.minimum(1.0, 2.0 * stats.t.cdf(-b, self.nu))

    def reset(self, n_paths):
        r = self.rng
        nblk = int(np.ceil(self.horizon / self.block))
        st = r.integers(0, len(self.zT) - self.block, size=(n_paths, nblk))
        self.idx = (st[:, :, None] + np.arange(self.block)[None, None, :]
                    ).reshape(n_paths, -1)[:, :self.horizon]
        self.n = n_paths
        self.t = 0
        self.eq = np.ones(n_paths)
        self.peak = np.ones(n_paths)
        self.f = np.zeros(n_paths)
        self.dead = np.zeros(n_paths, bool)
        # drift THAT (chung cho moi duong) va drift NHA DAU TU TIN (co the lech)
        if isinstance(self.sharpe_true, (tuple, list)):
            sr = r.uniform(self.sharpe_true[0], self.sharpe_true[1], n_paths)
        else:
            sr = np.full(n_paths, float(self.sharpe_true))
        self.sr_true = sr
        self.mu_true = sr * self.sig_bar * np.sqrt(250) / 250
        if self.belief_noise > 0:
            mult = np.exp(r.normal(0, self.belief_noise, n_paths))
        else:
            mult = np.ones(n_paths)
        self.mu_bel = self.mu_true * mult
        # bo dem cho differential Sharpe
        # khoi tao bo dem differential Sharpe bang mot tien nghiem hop ly thay vi 0:
        # neu B ~ 0 thi mau so (B - A^2)^1.5 no ra va phan thuong nhung buoc dau vo nghia
        self.A = np.zeros(n_paths)
        self.B = np.full(n_paths, 4e-4)
        return self._obs()[0]

    def _obs(self):
        s = self.sig[self.idx[:, min(self.t, self.horizon - 1)]]
        dd = 1 - self.eq / self.peak
        return np.column_stack([
            np.log(s / self.sig_bar),                      # bien dong tuong doi
            self.p_touch(s) * 2 - 1,                       # xac suat cham stop 1%
            dd * 5,                                        # sut giam hien tai
            self.f / LEV_MAX * 2 - 1,                      # vi the hien tai
            self.mu_bel / s * np.sqrt(250),                # loi the TIN, don vi Sharpe
            np.log(np.maximum(self.eq, 1e-6)),             # von tich luy
        ]), s

    def step(self, f_target, reward_kind="dsr", cvar_pen=0.0):
        """f_target: don bay muc tieu (n,). Tra ve (obs, reward, done)."""
        s = self.sig[self.idx[:, self.t]]
        f_new = np.clip(f_target, 0.0, LEV_MAX)
        f_new = np.where(self.dead, 0.0, f_new)
        cost = np.abs(f_new - self.f) * COST
        self.f = f_new
        rc = self.mu_true + self.zT[self.idx[:, self.t]] * s
        rl = self.mu_true + self.zL[self.idx[:, self.t]] * s
        hit = (~self.dead) & (self.f > 0) & (rl <= -MAINT)
        r_eff = np.where(hit, -MAINT, rc)
        pnl = self.f * r_eff - cost
        eq_new = np.where(self.dead, self.eq, self.eq * (1 + pnl))
        eq_new = np.maximum(eq_new, 1e-6)
        step_ret = np.log(eq_new / np.maximum(self.eq, 1e-9))
        self.eq = eq_new
        self.f = np.where(hit, 0.0, self.f)
        newly_dead = (~self.dead) & (self.eq < RUIN_LEVEL)
        self.dead |= newly_dead
        self.peak = np.maximum(self.peak, self.eq)
        dd = 1 - self.eq / self.peak

        # ── bon dang phan thuong. Diem then chot: BAC THUAN NHAT theo f
        #    quyet dinh viec KHOI LUONG co hoc duoc hay khong.
        if reward_kind == "pnl_arith":
            rew = (self.f * r_eff - cost) * 20.0      # TUYEN TINH theo f -> khong co cuc dai
        elif reward_kind == "pnl_log":
            rew = step_ret * 20.0                     # LOM theo f -> cuc dai tai Kelly
        else:
            eta = 0.02
            dA = step_ret - self.A
            dB = step_ret ** 2 - self.B
            den = np.maximum(self.B - self.A ** 2, 1e-10) ** 1.5
            # BAT BIEN VOI THANG do cua f -> ban than no khong noi gi ve khoi luong
            rew = np.clip((self.B * dA - 0.5 * self.A * dB) / den, -10.0, 10.0)
            self.A += eta * dA
            self.B += eta * dB
        rew = np.where(newly_dead, rew - RUIN_PEN, rew)            # phat pha san
        if cvar_pen > 0:
            rew = rew - cvar_pen * np.maximum(0.0, dd - 0.20) ** 2
        if self.t < 10:                      # bo qua giai doan bo dem chua on dinh
            rew = np.zeros_like(rew)
        rew = np.where(self.dead & ~newly_dead, 0.0, rew)
        self.t += 1
        done = self.t >= self.horizon
        obs = self._obs()[0] if not done else None
        return obs, rew, done

    def stats(self):
        g = np.log(np.maximum(self.eq, 1e-9))
        return dict(growth=float(np.mean(g) * 250 / self.horizon),
                    median_eq=float(np.median(self.eq)),
                    p_ruin=float(self.dead.mean()),
                    p_loss=float((self.eq < 1).mean()),
                    maxdd=float(np.mean(1 - self.eq / self.peak)))
