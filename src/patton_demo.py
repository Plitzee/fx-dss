"""KIEM CHUNG PATTON (2011) DUNG CACH — bang mo phong, noi ta BIET su that.
Patton chung minh: chi MSE va QLIKE giu nguyen thu hang mo hinh khi thuoc do bi nhieu
(voi thuoc do khong chech co dieu kien). MAE, MSE-tren-do-lech-chuan, MAPE thi khong.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

def sim_garch(T, omega=2e-6, a=0.08, b=0.90, seed=0, m=48):
    """Sinh chuoi co sigma^2 THAT biet truoc; tra ve (sigma2, r, rv_m)."""
    rng = np.random.default_rng(seed)
    s2 = np.empty(T); s2[0] = omega/(1-a-b)
    z = rng.normal(0,1,T); r = np.empty(T)
    for t in range(T):
        if t: s2[t] = omega + a*r[t-1]**2 + b*s2[t-1]
        r[t] = np.sqrt(s2[t])*z[t]
    # RV voi m quan sat trong ngay: khong chech, it nhieu hon r^2
    zz = rng.normal(0,1,(T,m))
    rv = (s2[:,None]/m * zz**2).sum(axis=1)
    return s2, r, rv

L = {
 "MSE      (ben)":      lambda p,h: (p-h)**2,
 "QLIKE    (ben)":      lambda p,h: np.log(h)+p/h,
 "MAE      (khong ben)":lambda p,h: np.abs(p-h),
 "MSE-sd   (khong ben)":lambda p,h: (np.sqrt(p)-np.sqrt(h))**2,
 "MAPE     (khong ben)":lambda p,h: np.abs(p-h)/p,
}

def make_forecasts(s2, rng):
    """Bon du bao co CHAT LUONG THAT xep hang biet truoc (do bang MSE voi sigma^2 that)."""
    T = len(s2)
    F = {}
    F["A: hoan hao"]      = s2.copy()
    F["B: lech +12%"]     = s2*1.12
    F["C: lam muot 5"]    = pd.Series(s2).rolling(5,min_periods=1).mean().values
    F["D: nhieu nhan"]    = s2*np.exp(rng.normal(0,0.35,T)-0.35**2/2)
    return F

def one_run(T=2000, seed=0, m=48):
    rng = np.random.default_rng(seed+9999)
    s2, r, rv = sim_garch(T, seed=seed, m=m)
    F = make_forecasts(s2, rng)
    names = list(F)
    proxies = {"sigma^2 THAT": s2, "RV (48 quan sat)": rv, "r^2 ngay": r**2}
    out = {}
    for lname, lf in L.items():
        rk = {}
        for pname, px in proxies.items():
            px = np.clip(px, 1e-14, None)
            vals = np.array([lf(px, np.clip(F[n],1e-14,None)).mean() for n in names])
            rk[pname] = vals
        out[lname] = rk
    return names, out

if __name__ == "__main__":
    NR = 300
    names, _ = one_run(seed=0)
    print("="*104)
    print("Bon du bao voi chat luong THAT da biet truoc (A tot nhat ... D te nhat).")
    print("Cau hoi: khi thay sigma^2 that bang mot THUOC DO NHIEU, ham mat mat nao van xep dung?")
    print("="*104)
    agree_rv = {k:0 for k in L}; agree_cc = {k:0 for k in L}
    top_rv = {k:0 for k in L}; top_cc = {k:0 for k in L}
    rho_rv = {k:[] for k in L}; rho_cc = {k:[] for k in L}
    for s in range(NR):
        _, out = one_run(T=2000, seed=s)
        for k in L:
            v_true = out[k]["sigma^2 THAT"]; v_rv = out[k]["RV (48 quan sat)"]; v_cc = out[k]["r^2 ngay"]
            r_true = pd.Series(v_true).rank().values
            for tag, v, agree, top, rho in (("rv", v_rv, agree_rv, top_rv, rho_rv),
                                            ("cc", v_cc, agree_cc, top_cc, rho_cc)):
                rr = pd.Series(v).rank().values
                agree[k] += int((rr == r_true).all())
                top[k]   += int(np.argmin(v) == np.argmin(v_true))
                rho[k].append(spearmanr(rr, r_true).statistic)
    print(f"\n{NR} lan lap doc lap, moi lan T = 2 000 ngay.\n")
    print(f"{'Ham mat mat':<24}{'Thuoc do RV (48)':>34}{'Thuoc do r^2 ngay':>34}")
    print(f"{'':<24}{'xep dung ca 4':>16}{'chon dung #1':>18}{'xep dung ca 4':>16}{'chon dung #1':>18}")
    print("-"*104)
    for k in L:
        print(f"{k:<24}{agree_rv[k]/NR:>15.1%}{top_rv[k]/NR:>18.1%}"
              f"{agree_cc[k]/NR:>16.1%}{top_cc[k]/NR:>18.1%}")
    print("-"*104)
    print(f"{'':<24}{'tuong quan hang TB':>34}{'tuong quan hang TB':>34}")
    for k in L:
        print(f"{k:<24}{np.mean(rho_rv[k]):>34.3f}{np.mean(rho_cc[k]):>34.3f}")
    print("-"*104)
    print("\nDoc ket qua: cot 'xep dung ca 4' la ty le lan ma thu hang duoi thuoc do nhieu")
    print("TRUNG KHOP hoan toan voi thu hang duoi sigma^2 that. MSE va QLIKE phai cao;")
    print("MAE / MSE-sd / MAPE phai thap hon ro ret — do la dinh ly cua Patton (2011).")

# ─────────────────────────────────────────────────────────────────────────────
def population_test(NR=400, T=2000, m=48):
    """Kiem dinh DUNG cua Patton: so sanh thu hang o muc KY VONG (gop het cac lan lap),
    chu khong phai o muc mot mau. Robust <=> thu hang duoi thuoc do nhieu TRUNG voi
    thu hang duoi sigma^2 that khi so quan sat -> vo cung."""
    names = None
    tot = {k: {p: None for p in ("true","rv","cc")} for k in L}
    for s in range(NR):
        rng = np.random.default_rng(s+9999)
        s2, r, rv = sim_garch(T, seed=s, m=m)
        F = make_forecasts(s2, rng); names = list(F)
        px = {"true": s2, "rv": rv, "cc": r**2}
        for k, lf in L.items():
            for pn, pv in px.items():
                pv = np.clip(pv, 1e-14, None)
                v = np.array([lf(pv, np.clip(F[n],1e-14,None)).mean() for n in names])
                tot[k][pn] = v if tot[k][pn] is None else tot[k][pn]+v
    print("\n" + "="*104)
    print(f"KIEM DINH O MUC KY VONG — gop {NR*T:,} quan sat mo phong cho moi o")
    print("="*104)
    print(f"{'Ham mat mat':<24}{'Thu hang duoi sigma^2 THAT':>28}{'duoi RV(48)':>20}{'duoi r^2 ngay':>22}")
    print("-"*104)
    for k in L:
        rt = pd.Series(tot[k]["true"]/NR, index=names).rank().astype(int)
        rr = pd.Series(tot[k]["rv"]/NR,   index=names).rank().astype(int)
        rc = pd.Series(tot[k]["cc"]/NR,   index=names).rank().astype(int)
        f = lambda s: "".join(str(s[n]) for n in names)
        ok_rv = "OK " if (rr.values==rt.values).all() else "DOI"
        ok_cc = "OK " if (rc.values==rt.values).all() else "DOI"
        print(f"{k:<24}{f(rt):>28}{f(rr)+'  '+ok_rv:>20}{f(rc)+'  '+ok_cc:>22}")
    print("-"*104)
    print("Chuoi 4 chu so = hang cua (A hoan hao, B lech +12%, C lam muot, D nhieu nhan).")
    print("'DOI' nghia la thuoc do nhieu lam DAO thu hang so voi su that — do la loi suy dien,")
    print("khong phai xui xeo lay mau: no khong bien mat du them bao nhieu du lieu.")

if __name__ == "__main__":
    population_test()
