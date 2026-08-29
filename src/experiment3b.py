"""
THI NGHIEM 3b — DU BAO XAC SUAT CHAM STOP, va hieu chinh cua no.
Bai toan thuc te: nha dau tu dat cat lo cach gia vao lenh d phan tram.
Xac suat bi cham TRONG PHIEN hom nay la bao nhieu?
Rao co dinh theo GIA nen b_t = d/sigma_t thay doi moi ngay -> day la bai toan du bao that.
"""
import numpy as np, pandas as pd, sys, json
sys.path.insert(0,"/tmp/fx/src")
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from metrics import mcs

P = pd.read_csv("/tmp/fx/exp3_panel.csv", parse_dates=["Date"]).sort_values(["Date","pair"])
STOPS = [0.0025, 0.005, 0.01, 0.0167]     # 0,25% · 0,5% · 1% · 1,67% (= stop-out don bay 30:1)
BURN = 300

def simulate_touch_prob(b, nu, m=24, nsim=20000, rng=None):
    """P(min cua duong di m buoc <= -b) voi gia so Student-t chuan hoa."""
    rng = rng or np.random.default_rng(0)
    z = rng.standard_t(nu, size=(nsim, m))/np.sqrt(nu/(nu-2))
    path = np.cumsum(z, axis=1)/np.sqrt(m)
    return float((path.min(axis=1) <= -b).mean())

# bang tra cuu mo phong: P_touch theo b, cho vai gia tri nu
GRID_B = np.concatenate([np.arange(0.05, 4.0, 0.05), np.arange(4.0, 12.0, 0.25)])
rng = np.random.default_rng(11)
SIM = {}
for nu in (4, 6, 8, 12, 30):
    z = rng.standard_t(nu, size=(60000, 24))/np.sqrt(nu/(nu-2))
    path = np.cumsum(z, axis=1)/np.sqrt(24)
    mn = path.min(axis=1)
    SIM[nu] = np.array([ (mn <= -b).mean() for b in GRID_B ])
def sim_lookup(b, nu):
    nus = np.array(list(SIM)); k = nus[np.argmin(np.abs(nus-nu))]
    return float(np.interp(b, GRID_B, SIM[k], left=1.0, right=0.0))

rows = []
for stop in STOPS:
    sub = P.copy()
    sub["b"] = stop/sub.sig                       # rao tinh theo don vi sigma du bao
    sub["hit"] = (sub.zL <= -sub.b).astype(int)
    sub = sub.sort_values("Date").reset_index(drop=True)
    n = len(sub)
    fc = {k: np.full(n, np.nan) for k in
          ["Cuoi ky (VaR)","Phan xa x2","Mo phong t","Hieu chinh"]}
    iso_hist = []
    for i in range(BURN, n):
        past = sub.iloc[:i]
        nu, _, sc = stats.t.fit(past.zT.values, floc=0)
        nu = float(np.clip(nu, 2.5, 40))
        b = sub.b.iloc[i]
        pT = float(stats.t.cdf(-b/sc, nu))                     # P(cuoi ky vuot)
        fc["Cuoi ky (VaR)"][i] = pT
        fc["Phan xa x2"][i]    = min(1.0, 2*pT)
        fc["Mo phong t"][i]    = sim_lookup(b/sc, nu)
        # hieu chinh dang tang: hoi quy dang dieu P(cham) theo P_phan xa tren du lieu DA QUA
        if i % 25 == 0 or not iso_hist:
            x = np.minimum(1.0, 2*stats.t.cdf(-past.b.values/sc, nu))
            y = past.hit.values
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, y)
            iso_hist = [ir]
        fc["Hieu chinh"][i] = float(iso_hist[0].predict([min(1.0, 2*pT)])[0])
    ev = sub.iloc[BURN:].copy()
    for k in fc: ev[k] = fc[k][BURN:]
    ev = ev.dropna()
    rows.append((stop, ev))

MODELS = ["Cuoi ky (VaR)","Phan xa x2","Mo phong t","Hieu chinh"]
print("="*112)
print("A. DU BAO XAC SUAT CHAM STOP — Brier score (thap hon = tot hon) va do lech tan suat")
print("="*112)
print(f"{'Cat lo':>9}{'n':>7}{'thuc te cham':>15}" +
      "".join(f"{m:>21}" for m in MODELS))
print("-"*112)
BR = {m: [] for m in MODELS}
for stop, ev in rows:
    real = ev.hit.mean()
    cells=[]
    for m in MODELS:
        br = np.mean((ev[m]-ev.hit)**2)
        BR[m].append(np.mean((ev[m]-ev.hit)**2 * np.ones(len(ev))) if False else (ev[m]-ev.hit)**2)
        cells.append(f"{br:.5f} ({ev[m].mean():.1%})")
    print(f"{stop:>8.2%}{len(ev):>7}{real:>15.2%}" + "".join(f"{c:>21}" for c in cells))
print("-"*112)
print("Trong ngoac la xac suat TRUNG BINH mo hinh du bao — so voi cot 'thuc te cham'.")

print("\n" + "="*112)
print("B. GOP TAT CA MUC CAT LO — Brier, log score, va do lech hieu chinh")
print("="*112)
allev = pd.concat([ev.assign(stop=s) for s, ev in rows], ignore_index=True)
print(f"{'Mo hinh':<20}{'Brier':>11}{'Log score':>13}{'TB du bao':>13}{'Thuc te':>11}"
      f"{'do lech':>11}{'  he so hieu chinh (hoi quy)':>30}")
print("-"*112)
L = {}
for m in MODELS:
    p = np.clip(allev[m].values, 1e-6, 1-1e-6); y = allev.hit.values
    br = np.mean((p-y)**2); ls = -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
    # hoi quy hieu chinh: y ~ a + c*p ; c=1, a=0 la hoan hao
    X = np.column_stack([np.ones(len(p)), p]); coef = np.linalg.lstsq(X, y, rcond=None)[0]
    L[m] = (p-y)**2
    print(f"{m:<20}{br:>11.5f}{ls:>13.5f}{p.mean():>13.2%}{y.mean():>11.2%}"
          f"{p.mean()-y.mean():>+11.2%}{f'a={coef[0]:+.3f}  c={coef[1]:.3f}':>30}")
print("-"*112)
# MCS: lay TRUNG BINH mat mat tren 4 muc cat lo cua CUNG mot ngay
# (bon muc dung chung ngay nen khong doc lap — gop thang se pha vo gia dinh cua bootstrap)
piv = {m: allev.pivot_table(index=["Date","pair"], columns="stop",
                            values=None, aggfunc="mean") for m in []}
key = allev.groupby(["Date","pair"]).ngroup()
Lm = np.column_stack([pd.Series(L[m]).groupby(key).mean().values for m in MODELS])
print(f"\nMCS chay tren {Lm.shape[0]:,} ngay-cap (trung binh 4 muc cat lo)")
keep, elim = mcs(Lm, alpha=0.10, B=800, block=20, seed=9)
print(f"Model Confidence Set 90% tren Brier: {[MODELS[i] for i in keep]}")
for i,(k,pv) in enumerate(elim,1): print(f"   loai {i}. {MODELS[k]:<18} p = {pv:.3f}")

print("\n" + "="*112)
print("C. BIEU DO TIN CAY (CORP) — mo hinh noi 20% thi co dung 20% xay ra khong?")
print("="*112)
BINS = [(0,.02),(.02,.05),(.05,.10),(.10,.20),(.20,.35),(.35,.55),(.55,1.01)]
print(f"{'Khoang du bao':>16}{'n':>8}" + "".join(f"{m:>22}" for m in MODELS))
print("-"*112)
for lo, hi in BINS:
    cells=[]; ns=[]
    for m in MODELS:
        s = allev[(allev[m]>=lo)&(allev[m]<hi)]
        ns.append(len(s))
        cells.append(f"{s[m].mean():.1%} -> {s.hit.mean():.1%}" if len(s)>=40 else "  (it quan sat)")
    print(f"{f'{lo:.0%}-{hi:.0%}':>16}{max(ns):>8}" + "".join(f"{c:>22}" for c in cells))
print("-"*112)
print("Doc: 'du bao -> thuc te'. Hai so cang sat nhau cang tot.")

json.dump({"models": MODELS,
           "brier": {m: float(np.mean(L[m])) for m in MODELS},
           "by_stop": [{"stop": s, "n": int(len(e)), "real": float(e.hit.mean()),
                        **{m: float(e[m].mean()) for m in MODELS}} for s, e in rows]},
          open("/tmp/fx/exp3b.json","w"), indent=1)
allev.to_csv("/tmp/fx/exp3b_panel.csv", index=False)
print("\nDa luu exp3b.json")
