"""Kiem chung cai dat 5 uoc luong bang mo phong GBM co sigma biet truoc.
Vector hoa hoan toan: mo phong (reps x n_days x steps) theo tung khoi."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/tmp/fx/src")
from vol import per_day_estimators, yang_zhang
LN2 = np.log(2.0)

def run(n_days=60, reps=2000, steps=1000, sigma_d=0.01, mu_d=0.0, seed=1, chunk=100):
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ("cc","park","gk","rs","yz")}
    k_yz = 0.34/(1.34 + (n_days+1)/(n_days-1))
    dt = 1.0/steps
    done = 0
    while done < reps:
        b = min(chunk, reps-done); done += b
        z = rng.normal(0, 1, size=(b, n_days, steps))
        inc = mu_d*dt + sigma_d*np.sqrt(dt)*z
        lp = np.cumsum(inc, axis=2)                      # log-gia trong ngay (bat dau tu 0)
        dayend = lp[:, :, -1]
        base = np.concatenate([np.zeros((b,1)), np.cumsum(dayend, axis=1)[:, :-1]], axis=1)
        lp = lp + base[:, :, None]
        O = np.concatenate([np.zeros((b,1)), np.cumsum(dayend, axis=1)[:, :-1]], axis=1)
        H = lp.max(axis=2); L = lp.min(axis=2); C = lp[:, :, -1]
        hl = H-L; co = C-O
        park = hl**2/(4*LN2)
        gk   = 0.5*hl**2 - (2*LN2-1)*co**2
        rs   = (H-C)*(H-O) + (L-C)*(L-O)
        Cprev = np.concatenate([np.zeros((b,1)), C[:, :-1]], axis=1)
        r_cc = C - Cprev
        r_on = O - Cprev
        r_oc = co
        acc["cc"].append(r_cc.var(axis=1, ddof=1))
        acc["park"].append(park.mean(axis=1))
        acc["gk"].append(gk.mean(axis=1))
        acc["rs"].append(rs.mean(axis=1))
        acc["yz"].append(r_on.var(axis=1, ddof=1) + k_yz*r_oc.var(axis=1, ddof=1)
                         + (1-k_yz)*rs.mean(axis=1))
    true_v = sigma_d**2
    rows = []
    cc = np.concatenate(acc["cc"]); base_var = cc.var(ddof=1)
    for k in ("cc","park","gk","rs","yz"):
        a = np.concatenate(acc[k])
        rows.append(dict(est=k, mean=a.mean(), bias_pct=100*(a.mean()/true_v-1),
                         sd=a.std(ddof=1), eff=base_var/a.var(ddof=1)))
    return pd.DataFrame(rows), true_v

NAMES = {"cc":"Close-to-close","park":"Parkinson 1980","gk":"Garman-Klass 1980",
         "rs":"Rogers-Satchell 1991","yz":"Yang-Zhang 2000"}
THEORY = {"cc":1.0,"park":5.2,"gk":7.4,"rs":6.0,"yz":14.0}

def show(title, df, tv):
    print("\n" + title)
    print("-"*88)
    print(f"{'Uoc luong':<24}{'TB uoc luong':>15}{'Lech (%)':>11}{'Hieu qua do':>14}{'Ly thuyet':>12}")
    print("-"*88)
    for _, r in df.iterrows():
        print(f"{NAMES[r.est]:<24}{r['mean']:>15.3e}{r.bias_pct:>11.2f}{r.eff:>14.2f}{THEORY[r.est]:>12.1f}")
    print("-"*88)
    print(f"phuong sai that = {tv:.3e}")

if __name__ == "__main__":
    d1, tv = run(n_days=60, reps=2000, steps=1000, mu_d=0.0, seed=1)
    show("A. Khong co drift, 1 000 buoc/ngay (xap xi quan sat lien tuc)", d1, tv)
    d2, _ = run(n_days=60, reps=2000, steps=1000, mu_d=0.02, seed=7)
    show("B. CO drift 2%/ngay — kiem tra tinh ben voi drift cua RS va YZ", d2, tv)
    d3, _ = run(n_days=60, reps=2000, steps=24, mu_d=0.0, seed=11)
    show("C. Chi 24 quan sat/ngay (dung nhu du lieu h1 that) — lech do roi rac hoa", d3, tv)
    d4, _ = run(n_days=60, reps=2000, steps=288, mu_d=0.0, seed=13)
    show("D. 288 quan sat/ngay (tuong duong thanh 5 phut)", d4, tv)
