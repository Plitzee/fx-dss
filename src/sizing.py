"""
TANG QUYET DINH — sau quy tac dinh co vi the, va bo mo phong danh gia chung.

Nguyen tac trung thuc: chung ta KHONG tuyen bo du bao duoc huong. Cau hoi o day la:
  "CHO TRUOC mot muc loi the ma nha dau tu tin minh co, quy tac dinh co nao song sot?"
Duong gia lay tu bootstrap khoi du lieu FX THAT (giu duoi day va cum bien dong),
roi bom vao mot drift co kiem soat.

Hai co che ket thuc duoc mo hinh rieng:
  * PHA SAN  : von tut xuong duoi RUIN_LEVEL lan von ban dau (mac dinh 50%)
  * THANH LY : lo trong mot phien vuot muc ky quy duy tri cua san (margin close-out)
"""
import numpy as np
from scipy import stats

RUIN_LEVEL = 0.50          # coi la "chay tai khoan" khi von con duoi 50% ban dau


# ─────────────────────── cac quy tac dinh co ───────────────────────
def f_fixed_risk(sig, risk_frac=0.02, stop_sigma=1.0):
    """Quy tac '2% moi lenh' cua retail: dat stop cach 1 sigma, rui ro 2% von."""
    return risk_frac / (stop_sigma * sig)


def f_kelly(mu, sig, frac=1.0):
    """Kelly (xap xi Gauss): f* = mu / sigma^2. frac<1 la Kelly phan so."""
    return frac * mu / sig ** 2


def f_cvar(sig, es_z, budget=0.02):
    """Rang buoc CVaR: f sao cho ES_97,5% cua f*r bang 'budget' phan von."""
    return budget / (abs(es_z) * sig)


def f_ruin_cap(sig, horizon_days=250, budget=0.01, nu=6.0, ruin_level=RUIN_LEVEL):
    """TRAN don bay sao cho P(von tut duoi ruin_level trong 'horizon_days') <= budget.
    Dung nguyen ly phan xa: P(min <= -b) = 2 P(X_T <= -b) — chinh la ket qua da do
    duoc o giai doan 3 (ty le 1,95-2,01)."""
    s_h = sig * np.sqrt(horizon_days)
    z = stats.t.ppf(budget / 2.0, nu) / np.sqrt(nu / (nu - 2))   # phan vi duoi
    b = -z * s_h                                                  # bien dong can thiet
    loss_allowed = -np.log(ruin_level)                            # ~0,693 voi 50%
    return loss_allowed / np.maximum(b, 1e-9)


# ─────────────────────── bo mo phong ───────────────────────
def simulate(panel, rule, n_paths=20000, horizon=250, mu_true=0.0, mu_believed=None,
             cost_pips=1.0, price=1.1, lev_cap=30.0, maint_margin=0.5,
             seed=0, block=5, rebalance=5):
    """
    panel : DataFrame co cot sig (du bao do lech chuan ngay), zT, zL (chuan hoa)
    rule  : ham (sig, mu, es_z, nu) -> DON BAY f
    """
    if mu_believed is None:
        mu_believed = mu_true
    rng = np.random.default_rng(seed)
    zT, zL, sg = panel.zT.values.copy(), panel.zL.values.copy(), panel.sig.values
    # khu trung binh: chi de mu_true la nguon drift duy nhat, neu khong phan du lich su
    # tu no da mang mot loi the nho (+0,023 sigma) va lam sai kich ban "khong co loi the"
    m0 = zT.mean(); zT -= m0; zL -= m0
    n = len(zT)
    nblk = int(np.ceil(horizon / block))
    starts = rng.integers(0, n - block, size=(n_paths, nblk))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :horizon]

    es_z = zT[zT <= np.quantile(zT, 0.025)].mean()
    nu = float(np.clip(stats.t.fit(zT, floc=0)[0], 2.5, 40))

    eq = np.ones(n_paths)
    peak = np.ones(n_paths)
    dd = np.zeros(n_paths)
    dd2sum = np.zeros(n_paths)
    under = np.zeros(n_paths)
    ruined = np.zeros(n_paths, bool)
    liquidated = np.zeros(n_paths, bool)
    f_cur = np.zeros(n_paths)
    lev_log = []

    for t in range(horizon):
        s = sg[idx[:, t]]
        if t % rebalance == 0:
            f_new = np.clip(np.asarray(rule(s, mu_believed, es_z, nu), float), 0.0, lev_cap)
            f_new = np.where(ruined, 0.0, f_new)
            eq = eq - np.abs(f_new - f_cur) * (cost_pips * 1e-4 / price)
            f_cur = f_new
            lev_log.append(float(np.mean(f_new[~ruined])) if (~ruined).any() else 0.0)
        r_close = mu_true + zT[idx[:, t]] * s
        r_low = mu_true + zL[idx[:, t]] * s
        # margin close-out: lo trong phien an het (1 - maint_margin) phan ky quy ban dau
        # ky quy ban dau = f/lev_cap phan von  ->  nguong lo = maint_margin*f/lev_cap / f
        barrier = -np.full_like(f_cur, maint_margin / lev_cap)
        hit = (~ruined) & (f_cur > 0) & (r_low <= barrier)
        r_eff = np.where(hit, barrier, r_close)
        eq = np.where(ruined, eq, eq * (1 + f_cur * r_eff))
        eq = np.maximum(eq, 1e-6)
        liquidated |= hit
        f_cur = np.where(hit, 0.0, f_cur)
        newly = (~ruined) & (eq < RUIN_LEVEL)
        ruined |= newly
        peak = np.maximum(peak, eq)
        d = 1 - eq / peak
        dd = np.maximum(dd, d)
        dd2sum += d ** 2
        under += (d > 1e-9).astype(float)

    g = np.log(np.maximum(eq, 1e-9)) / horizon
    return dict(
        median_eq=float(np.median(eq)),
        mean_log_growth=float(np.mean(g) * 250),
        p10=float(np.quantile(eq, 0.10)),
        p90=float(np.quantile(eq, 0.90)),
        p_ruin=float(ruined.mean()),
        p_liquidated=float(liquidated.mean()),
        p_loss=float((eq < 1).mean()),
        maxdd=float(np.mean(dd)),
        maxdd_p95=float(np.quantile(dd, 0.95)),
        ulcer=float(np.mean(np.sqrt(dd2sum / horizon))),
        time_under=float(np.mean(under) / horizon),
        avg_lev=float(np.mean(lev_log)),
    )


RULES = {
    "Cố định 2%":        lambda sig, mu, es_z, nu: f_fixed_risk(sig, 0.02),
    "Kelly đầy đủ":      lambda sig, mu, es_z, nu: f_kelly(mu, sig, 1.0),
    "Kelly 1/2":         lambda sig, mu, es_z, nu: f_kelly(mu, sig, 0.5),
    "Kelly 1/4":         lambda sig, mu, es_z, nu: f_kelly(mu, sig, 0.25),
    "Ràng buộc CVaR":    lambda sig, mu, es_z, nu: f_cvar(sig, es_z, 0.02),
    "Kelly + trần rủi ro": lambda sig, mu, es_z, nu: np.minimum(
        f_kelly(mu, sig, 1.0), f_ruin_cap(sig, 250, 0.01, nu)),
}
