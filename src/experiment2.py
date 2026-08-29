"""
THI NGHIEM 2 — cho truoc du bao bien dong, phan phoi nhieu nao cho mat do tot nhat?
Danh gia: CRPS, PIT, VaR/ES 97,5% & 99%, FZ0, Kupiec, Christoffersen, DQ.
Toan bo phan phoi nhieu duoc uoc luong CHI tu phan du ngoai mau da qua -> khong ro ri.
"""
import numpy as np, pandas as pd, sys, json, warnings
sys.path.insert(0,"/tmp/fx/src")
warnings.filterwarnings("ignore")
from scipy import stats
from fxdata import PAIRS
from metrics import (crps_from_quantiles, TAU_GRID, pinball, fz0,
                     kupiec, christoffersen_ind, dq_test, mcs)

VOL_MODEL = "HAR-RV"
BURN = 250          # so ngay ngoai mau dau tien dung de khoi tao phan phoi nhieu
ALPHAS = [0.025, 0.01]
DISTS = ["Chuan", "Student-t", "FHS", "FHS+EVT"]

def gpd_tail(z_neg, p, u_q=0.90):
    """z_neg = -z (ton that duong). Tra ve (VaR, ES) o xac suat duoi p cua z."""
    u = np.quantile(z_neg, u_q)
    exc = z_neg[z_neg > u] - u
    if len(exc) < 30: return None
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    n, Nu = len(z_neg), len(exc)
    if xi <= 0: xi = 1e-6
    xp = u + (beta/xi)*(((p*n/Nu)**(-xi)) - 1)
    es = (xp + beta - xi*u)/(1-xi) if xi < 1 else np.nan
    return -xp, -es          # doi ve phia trai (am)

def build(pair):
    o = pd.read_csv(f"/tmp/fx/fc_{pair}.csv", parse_dates=["Date"])
    o = o.dropna(subset=[VOL_MODEL,"r"]).reset_index(drop=True)
    o["sig"] = np.sqrt(o[VOL_MODEL].clip(lower=1e-12))
    o["z"] = o.r/o.sig
    return o

def run_pair(pair):
    o = build(pair); n = len(o)
    rec = {d: {"crps": [], "pit": [], "var": {a: [] for a in ALPHAS},
               "es": {a: [] for a in ALPHAS}} for d in DISTS}
    idx = []
    REFIT = 5          # uoc luong lai tham so t va GPD moi 5 ngay (tham so bien doi cham)
    cache = {}
    for t in range(BURN, n):
        zh = o.z.values[:t]                       # chi dung phan du DA QUAN SAT
        s = o.sig.values[t]; y = o.r.values[t]
        idx.append(t)
        if t % REFIT == 0 or not cache:
            nu, _, sc = stats.t.fit(zh, floc=0)
            nu = float(np.clip(nu, 2.1, 60))
            sd_z = float(zh.std(ddof=1))          # <- SUA LOI: phan phoi chuan cung phai
            qf = np.quantile(zh, TAU_GRID)        #    uoc luong thang do, khong gia dinh 1
            gp = {a: gpd_tail(-zh, a) for a in ALPHAS}
            emp = {a: (np.quantile(zh, a), zh[zh <= np.quantile(zh, a)]) for a in ALPHAS}
            cache = dict(nu=nu, sc=sc, sd_z=sd_z, qf=qf, gp=gp, emp=emp)
        nu, sc, sd_z, qf = cache["nu"], cache["sc"], cache["sd_z"], cache["qf"]
        Qn = s*sd_z*stats.norm.ppf(TAU_GRID)
        Qt = s*stats.t.ppf(TAU_GRID, nu)*sc
        Qf = s*qf
        Qe = Qf.copy()
        for dname, Q in (("Chuan",Qn),("Student-t",Qt),("FHS",Qf),("FHS+EVT",Qe)):
            rec[dname]["crps"].append(float(crps_from_quantiles(np.array([y]), Q[None,:])[0]))
            rec[dname]["pit"].append(float(np.interp(y, Q, TAU_GRID, left=0.001, right=0.999)))
        for a in ALPHAS:
            zq = stats.norm.ppf(a)
            rec["Chuan"]["var"][a].append(s*sd_z*zq)
            rec["Chuan"]["es"][a].append(-s*sd_z*stats.norm.pdf(zq)/a)
            tq = stats.t.ppf(a, nu)
            es_t = -stats.t.pdf(tq, nu)*(nu+tq**2)/((nu-1)*a)
            rec["Student-t"]["var"][a].append(s*tq*sc)
            rec["Student-t"]["es"][a].append(s*es_t*sc)
            vq, tail = cache["emp"][a]
            rec["FHS"]["var"][a].append(s*vq)
            rec["FHS"]["es"][a].append(s*(tail.mean() if len(tail) else vq))
            g = cache["gp"][a]
            v_e, e_e = (vq, (tail.mean() if len(tail) else vq)) if g is None else g
            rec["FHS+EVT"]["var"][a].append(s*v_e)
            rec["FHS+EVT"]["es"][a].append(s*e_e)
    y = o.r.values[BURN:]
    return o, y, rec, idx

if __name__ == "__main__":
    out = {}
    for p in PAIRS:
        o, y, rec, idx = run_pair(p)
        out[p] = dict(y=y.tolist(),
                      rec={d: {"crps": rec[d]["crps"], "pit": rec[d]["pit"],
                               "var": {str(a): rec[d]["var"][a] for a in ALPHAS},
                               "es":  {str(a): rec[d]["es"][a]  for a in ALPHAS}}
                           for d in DISTS})
        print(f"{p}: {len(y)} ngay danh gia", flush=True)
    json.dump(out, open("/tmp/fx/exp2.json","w"))
    print("XONG")
