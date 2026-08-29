"""Bo sung: (E) co gap qua dem that -> YZ moi phat huy; (F) drift thuc te."""
import numpy as np, pandas as pd, sys
sys.path.insert(0,"/tmp/fx/src")
LN2 = np.log(2.0)

def run(n_days=60, reps=2000, steps=1000, sigma_d=0.01, mu_d=0.0,
        sigma_on=0.0, seed=1, chunk=100):
    """sigma_on: do lech chuan cua gap qua dem (nhay rieng giua dong hom truoc va mo hom nay)."""
    rng = np.random.default_rng(seed); dt = 1.0/steps
    k_yz = 0.34/(1.34 + (n_days+1)/(n_days-1))
    acc = {k: [] for k in ("cc","park","gk","rs","yz")}
    done = 0
    while done < reps:
        b = min(chunk, reps-done); done += b
        z = rng.normal(0,1,size=(b,n_days,steps))
        lp = np.cumsum(mu_d*dt + sigma_d*np.sqrt(dt)*z, axis=2)
        gap = rng.normal(0, sigma_on, size=(b, n_days)) if sigma_on > 0 else np.zeros((b,n_days))
        dayend = lp[:,:,-1]
        # gia mo cua ngay i = dong cua ngay i-1 + gap_i
        O = np.zeros((b,n_days))
        prevC = np.zeros(b)
        for i in range(n_days):
            O[:,i] = prevC + gap[:,i]
            prevC = O[:,i] + dayend[:,i]
        lp = lp + O[:,:,None]
        H = lp.max(axis=2); L = lp.min(axis=2); C = lp[:,:,-1]
        hl = H-L; co = C-O
        park = hl**2/(4*LN2)
        gk   = 0.5*hl**2 - (2*LN2-1)*co**2
        rs   = (H-C)*(H-O) + (L-C)*(L-O)
        Cprev = np.concatenate([np.zeros((b,1)), C[:,:-1]], axis=1)
        r_cc = C-Cprev; r_on = O-Cprev
        acc["cc"].append(r_cc.var(axis=1,ddof=1))
        acc["park"].append(park.mean(axis=1)); acc["gk"].append(gk.mean(axis=1))
        acc["rs"].append(rs.mean(axis=1))
        acc["yz"].append(r_on.var(axis=1,ddof=1) + k_yz*co.var(axis=1,ddof=1)
                         + (1-k_yz)*rs.mean(axis=1))
    true_v = sigma_d**2 + sigma_on**2      # tong phuong sai dong->dong
    cc = np.concatenate(acc["cc"]); base = cc.var(ddof=1)
    rows=[]
    for k in ("cc","park","gk","rs","yz"):
        a = np.concatenate(acc[k])
        rows.append(dict(est=k, mean=a.mean(), bias=100*(a.mean()/true_v-1), eff=base/a.var(ddof=1)))
    return pd.DataFrame(rows), true_v

NAMES={"cc":"Close-to-close","park":"Parkinson","gk":"Garman-Klass","rs":"Rogers-Satchell","yz":"Yang-Zhang"}
def show(t, df, tv):
    print("\n"+t); print("-"*78)
    print(f"{'Uoc luong':<20}{'TB':>13}{'Lech vs that (%)':>19}{'Hieu qua':>12}")
    print("-"*78)
    for _,r in df.iterrows():
        print(f"{NAMES[r.est]:<20}{r['mean']:>13.3e}{r.bias:>19.2f}{r.eff:>12.2f}")
    print("-"*78); print(f"phuong sai dong->dong that = {tv:.3e}")

if __name__=="__main__":
    e,tv = run(sigma_on=0.005, mu_d=0.0, seed=21)
    show("E. CO gap qua dem (sigma_on = 0,5%/ngay, tuc 20% tong phuong sai)", e, tv)
    print("   -> Parkinson/GK/RS chi do phan TRONG phien nen thieu han phan gap;")
    print("      chi YZ (va close-to-close) tinh duoc tong phuong sai dong->dong.")
    f,tv2 = run(sigma_on=0.0, mu_d=0.0005, seed=23)
    show("F. Drift thuc te 0,05%/ngay (~13%/nam)", f, tv2)
    g,tv3 = run(sigma_on=0.0, mu_d=0.005, seed=25)
    show("G. Drift manh 0,5%/ngay (giai doan xu huong manh)", g, tv3)
