"""
DUONG ONG CO SO — thi nghiem 1: mo hinh nao du bao bien dong tot nhat?
Backtest rolling-origin, cua so mo rong, du bao 1 ngay.
MUC TIEU: realized variance lay mau 5 PHUT (doi tu khung gio ngay 28/08/2026).
"""
import numpy as np, pandas as pd, sys, json, warnings, time
sys.path.insert(0,"/tmp/fx/src")
warnings.filterwarnings("ignore")
from fxdata import load_daily, realized_var, PAIRS
from contig import run_length
from vol import per_day_estimators, yang_zhang
from metrics import qlike, mse_var, mcs

MIN_TRAIN = 250        # ha tu 750: bo loc "nen ma" cu vut 93% du lieu
GARCH_REFIT = 50
EPS = 1e-12

def build_panel(pair):
    d = load_daily(pair)
    e = per_day_estimators(d)
    e["yz5"] = yang_zhang(e, 5)
    rv = realized_var(pair)                          # mac dinh: lay mau 5 phut
    m = e.merge(rv, on="Date", how="left")
    m = m[m.n_bars.fillna(0) >= 200].copy()          # ngay du thanh 5 phut (~287 khi day du)
    for c in ("cc","park","gk","rs","yz5","rv"):
        m[c] = m[c].clip(lower=EPS)
    m = m.dropna(subset=["r_cc","gk","rv"]).reset_index(drop=True)
    m["run"] = run_length(m.Date.values)              # rao chan lien mach
    return m

def har_design(logv):
    """Tra ve X (const, d, w, m) va chi so hang hop le. logv la mang log phuong sai."""
    n = len(logv)
    d = logv
    w = pd.Series(logv).rolling(5).mean().values
    mo = pd.Series(logv).rolling(22).mean().values
    X = np.column_stack([np.ones(n), d, w, mo])
    return X

def ols(X, y):
    return np.linalg.solve(X.T@X + 1e-10*np.eye(X.shape[1]), X.T@y)

def forecast_har(series, t):
    """Du bao log phuong sai ngay t+1 dung du lieu den het ngay t (chi so 0..t)."""
    lv = np.log(series[:t+1])
    X = har_design(lv)
    y = lv[1:]; Xd = X[:-1]
    ok = np.isfinite(Xd).all(1) & np.isfinite(y)
    if ok.sum() < 100: return np.nan
    b = ols(Xd[ok], y[ok])
    xn = X[-1]
    if not np.isfinite(xn).all(): return np.nan
    resid = y[ok] - Xd[ok]@b
    # hieu chinh Jensen khi doi tu log ve muc
    return float(np.exp(xn@b + 0.5*resid.var()))

def run_pair(pair, verbose=False):
    m = build_panel(pair)
    n = len(m)
    r = m.r_cc.values
    proxies = {k: m[k].values for k in ("cc","park","gk","rs","yz5","rv")}
    target = m.rv.values

    models = ["RW-cc","MA20-cc","EWMA94-cc","RW-GK","MA20-GK",
              "HAR-Park","HAR-GK","HAR-RS","HAR-RV","GARCH-t"]
    F = {k: np.full(n, np.nan) for k in models}

    # EWMA (RiskMetrics) chay truot
    lam = 0.94
    ew = np.full(n, np.nan); ew[0] = proxies["cc"][:50].mean()
    for t in range(1, n):
        ew[t] = lam*ew[t-1] + (1-lam)*proxies["cc"][t-1]

    from arch import arch_model
    garch_params = None; last_fit = -10**9; NU_HIST=[]; g_state=np.nan

    for t in range(MIN_TRAIN, n-1):
        F["RW-cc"][t+1]   = max(proxies["cc"][t], EPS)
        F["MA20-cc"][t+1] = proxies["cc"][t-19:t+1].mean()
        F["EWMA94-cc"][t+1] = ew[t]
        F["RW-GK"][t+1]   = max(proxies["gk"][t], EPS)
        F["MA20-GK"][t+1] = proxies["gk"][t-19:t+1].mean()
        F["HAR-Park"][t+1] = forecast_har(proxies["park"], t)
        F["HAR-GK"][t+1]   = forecast_har(proxies["gk"], t)
        F["HAR-RS"][t+1]   = forecast_har(proxies["rs"], t)
        F["HAR-RV"][t+1]   = forecast_har(proxies["rv"], t)
        # GARCH(1,1)-t: uoc luong lai dinh ky, giua cac lan thi GIU tham so
        # va chay loc phuong sai tien len bang loi suat that (chuan thuc hanh).
        if t - last_fit >= GARCH_REFIT:
            try:
                res = arch_model(r[:t+1]*100, p=1, q=1, dist="t",
                                 mean="Constant").fit(disp="off", show_warning=False)
                pr = res.params
                garch_params = (float(pr["mu"]), float(pr["omega"]),
                                float(pr["alpha[1]"]), float(pr["beta[1]"]), float(pr["nu"]))
                last_fit = t
            except Exception:
                garch_params = None
        if garch_params is not None:
            mu, om, al, be, nu = garch_params
            if t == last_fit:                    # vua uoc luong lai -> loc lai tu dau
                eps = r[:t+1]*100 - mu
                s2 = om/max(1e-8, 1-al-be)
                for x in eps:
                    s2 = om + al*x*x + be*s2
                g_state = s2
            else:                                # cap nhat tang dan bang loi suat moi nhat
                x = r[t]*100 - mu
                g_state = om + al*x*x + be*g_state
            F["GARCH-t"][t+1] = g_state/1e4
            NU_HIST.append(nu)
    out = pd.DataFrame({"Date": m.Date.values, "r": r, "rv": target})
    for k in models: out[k] = F[k]
    out["gk"] = proxies["gk"]; out["park"] = proxies["park"]
    out.attrs["nu_med"] = float(np.median(NU_HIST)) if NU_HIST else np.nan
    return out, models

if __name__ == "__main__":
    t0 = time.time()
    allres = {}
    for p in PAIRS:
        o, models = run_pair(p)
        o.to_csv(f"/tmp/fx/fc_{p}.csv", index=False)
        allres[p] = len(o.dropna(subset=models))
        print(f"{p}: {allres[p]} ngay ngoai mau, {time.time()-t0:.0f}s", flush=True)
    json.dump(allres, open("/tmp/fx/fc_counts.json","w"), indent=1)
    print("XONG", time.time()-t0)
