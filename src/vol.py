"""Nam uoc luong bien dong tu OHLC + kiem chung Monte Carlo."""
import numpy as np, pandas as pd
LN2 = np.log(2.0)

def per_day_estimators(df):
    """Tra ve DataFrame cac uoc luong PHUONG SAI theo tung ngay (don vi: (log-return)^2)."""
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    cp = np.r_[np.nan, c[:-1]]
    out = pd.DataFrame({"Date": df.Date.values})
    out["r_cc"]  = np.log(c/cp)                       # loi suat dong->dong
    out["cc"]    = out.r_cc**2                        # binh phuong loi suat ngay
    out["park"]  = (np.log(h/l)**2)/(4*LN2)           # Parkinson 1980
    out["gk"]    = 0.5*np.log(h/l)**2 - (2*LN2-1)*np.log(c/o)**2   # Garman-Klass 1980
    out["rs"]    = np.log(h/c)*np.log(h/o) + np.log(l/c)*np.log(l/o)  # Rogers-Satchell 1991
    out["r_on"]  = np.log(o/cp)                       # gap qua dem / cuoi tuan
    out["r_oc"]  = np.log(c/o)                        # trong phien
    return out

def yang_zhang(est, n):
    """YZ cua so n ngay. est: output cua per_day_estimators. Tra ve Series phuong sai."""
    k = 0.34/(1.34 + (n+1)/(n-1))
    v_o = est.r_on.rolling(n).var(ddof=1)
    v_c = est.r_oc.rolling(n).var(ddof=1)
    v_rs = est.rs.rolling(n).mean()
    return v_o + k*v_c + (1-k)*v_rs

# ───────────────────────── kiem chung Monte Carlo ─────────────────────────
def simulate_ohlc(n_days, steps, sigma_d, mu_d=0.0, seed=0):
    """GBM voi sigma ngay biet truoc; OHLC lay tu 'steps' quan sat trong ngay."""
    rng = np.random.default_rng(seed)
    dt = 1.0/steps
    z = rng.normal(0, 1, size=(n_days, steps))
    inc = mu_d*dt + sigma_d*np.sqrt(dt)*z
    logp = np.cumsum(inc, axis=1)
    logp = logp + np.r_[0, np.cumsum(logp[:-1, -1])][:, None]   # noi lien cac ngay
    O = np.exp(np.r_[0, logp[:-1, -1]])
    P = np.exp(logp)
    return pd.DataFrame({"Date": pd.date_range("2000-01-03", periods=n_days, freq="B"),
                         "open": O, "high": P.max(1), "low": P.min(1), "close": P[:, -1]})

def mc_efficiency(n_days=60, reps=4000, sigma_d=0.01, steps=20000, seed=1):
    """Hieu qua = Var(uoc luong close-to-close) / Var(uoc luong khac), cung cua so n_days.
    Ly thuyet (quan sat lien tuc): Parkinson ~5,2x  GK ~7,4x  RS ~6x  YZ ~14x (voi drift)."""
    res = {k: [] for k in ("cc","park","gk","rs","yz")}
    for r in range(reps):
        df = simulate_ohlc(n_days, steps, sigma_d, 0.0, seed=seed+r)
        e = per_day_estimators(df)
        res["cc"].append(np.nanvar(e.r_cc.values, ddof=1))
        for k in ("park","gk","rs"):
            res[k].append(np.nanmean(e[k].values))
        res["yz"].append(yang_zhang(e, n_days).iloc[-1])
    true_v = sigma_d**2
    rows = []
    base = np.nanvar(np.array(res["cc"]), ddof=1)
    for k in ("cc","park","gk","rs","yz"):
        a = np.array(res[k], dtype=float)
        rows.append(dict(est=k, mean=np.nanmean(a), bias_pct=100*(np.nanmean(a)/true_v-1),
                         var=np.nanvar(a, ddof=1), eff=base/np.nanvar(a, ddof=1)))
    return pd.DataFrame(rows), true_v
