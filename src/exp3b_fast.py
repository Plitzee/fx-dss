"""
THI NGHIEM 3b — DU BAO XAC SUAT CHAM STOP.
Rao co dinh theo GIA (nha dau tu dat cat lo cach gia vao lenh d%), nen b_t = d/sigma_t
thay doi moi ngay theo du bao bien dong -> day la mot bai toan du bao xac suat that su.
Bon phuong phap, danh gia bang Brier score, log score, bieu do tin cay va MCS.
"""
import numpy as np, pandas as pd, sys, json, time
sys.path.insert(0, "/tmp/fx/src")
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from metrics import mcs

P = (pd.read_csv("/tmp/fx/exp3_panel.csv", parse_dates=["Date"])
       .sort_values(["Date", "pair"]).reset_index(drop=True))
STOPS = [0.0025, 0.005, 0.01, 0.0167]      # 0,25% · 0,5% · 1% · 1,67% (stop-out 30:1)
BURN, REFIT, WIN = 400, 250, 5000
MODELS = ["Cuoi ky (VaR)", "Phan xa x2", "Mo phong t", "Hieu chinh"]

# ── bang tra cuu mo phong duong di: P(min cua duong 24 buoc <= -b) theo nu ──
GRID_B = np.concatenate([np.arange(0.02, 5.0, 0.02), np.arange(5.0, 20.0, 0.2)])
_rng = np.random.default_rng(11)
SIM = {}
for _nu in (3, 4, 5, 6, 8, 12, 20, 40):
    _z = _rng.standard_t(_nu, size=(120000, 24)) / np.sqrt(_nu / (_nu - 2))
    _mn = (np.cumsum(_z, axis=1) / np.sqrt(24)).min(axis=1)
    SIM[_nu] = np.array([(_mn <= -b).mean() for b in GRID_B])
NUS = np.array(sorted(SIM))
print(f"bang mo phong xong ({len(NUS)} gia tri nu)", flush=True)


def sim_lookup(bs, nu):
    k = NUS[np.argmin(np.abs(NUS - nu))]
    return np.interp(bs, GRID_B, SIM[k], left=1.0, right=0.0)


t0 = time.time()
rows = []
for stop in STOPS:
    s = P.copy()
    s["b"] = stop / s.sig
    s["hit"] = (s.zL <= -s.b).astype(int)
    n = len(s)
    b_all, zT_all, hit_all = s.b.values, s.zT.values, s.hit.values
    F = {k: np.full(n, np.nan) for k in MODELS}
    for start in range(BURN, n, REFIT):
        end = min(start + REFIT, n)
        past = slice(max(0, start - WIN), start)
        nu_, _, sc = stats.t.fit(zT_all[past], floc=0)
        nu = float(np.clip(nu_, 2.5, 40))
        bb = b_all[start:end] / sc
        pT = stats.t.cdf(-bb, nu)
        F["Cuoi ky (VaR)"][start:end] = pT
        F["Phan xa x2"][start:end] = np.minimum(1.0, 2 * pT)
        F["Mo phong t"][start:end] = sim_lookup(bb, nu)
        x = np.minimum(1.0, 2 * stats.t.cdf(-b_all[past] / sc, nu))
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, hit_all[past])
        F["Hieu chinh"][start:end] = iso.predict(np.minimum(1.0, 2 * pT))
    ev = s.iloc[BURN:].copy()
    for k in MODELS:
        ev[k] = F[k][BURN:]
    rows.append((stop, ev.dropna()))
    print(f"  cat lo {stop:.2%}: {len(rows[-1][1])} quan sat, {time.time()-t0:.0f}s", flush=True)

print("\n" + "=" * 112)
print("A. THEO TUNG MUC CAT LO — Brier score (thap hon = tot hon)")
print("=" * 112)
print(f"{'Cat lo':>9}{'n':>7}{'thuc te cham':>15}" + "".join(f"{m:>22}" for m in MODELS))
print("-" * 112)
for stop, ev in rows:
    cells = [f"{np.mean((ev[m]-ev.hit)**2):.5f} ({ev[m].mean():.1%})" for m in MODELS]
    print(f"{stop:>8.2%}{len(ev):>7}{ev.hit.mean():>15.2%}" + "".join(f"{c:>22}" for c in cells))
print("-" * 112)
print("Trong ngoac: xac suat TRUNG BINH mo hinh du bao, so voi cot 'thuc te cham'.")

allev = pd.concat([e.assign(stop=st) for st, e in rows], ignore_index=True)
print("\n" + "=" * 112)
print("B. GOP TAT CA — Brier, log score, hieu chinh")
print("=" * 112)
print(f"{'Mo hinh':<18}{'Brier':>11}{'Log score':>12}{'TB du bao':>12}{'Thuc te':>10}"
      f"{'do lech':>11}{'hoi quy hieu chinh':>26}")
print("-" * 112)
L = {}
for m in MODELS:
    p = np.clip(allev[m].values, 1e-6, 1 - 1e-6)
    y = allev.hit.values
    br = np.mean((p - y) ** 2)
    ls = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    X = np.column_stack([np.ones(len(p)), p])
    a, c = np.linalg.lstsq(X, y, rcond=None)[0]
    L[m] = (p - y) ** 2
    print(f"{m:<18}{br:>11.5f}{ls:>12.5f}{p.mean():>12.2%}{y.mean():>10.2%}"
          f"{p.mean()-y.mean():>+11.2%}{f'a={a:+.4f}  c={c:.3f}':>26}")
print("-" * 112)
print("Hieu chinh hoan hao: a = 0 va c = 1.")

# MCS tren trung binh mat mat cua 4 muc cat lo trong CUNG mot ngay-cap
key = allev.groupby(["Date", "pair"]).ngroup().values
Lm = np.column_stack([pd.Series(L[m]).groupby(key).mean().values for m in MODELS])
print(f"\nMCS chay tren {Lm.shape[0]:,} ngay-cap (trung binh 4 muc cat lo)")
keep, elim = mcs(Lm, alpha=0.10, B=600, block=20, seed=9)
print(f"Model Confidence Set 90% tren Brier: {[MODELS[i] for i in keep]}")
for i, (k, pv) in enumerate(elim, 1):
    print(f"   loai {i}. {MODELS[k]:<18} p = {pv:.3f}")

print("\n" + "=" * 112)
print("C. BIEU DO TIN CAY — mo hinh noi 20% thi co dung 20% xay ra khong?")
print("=" * 112)
BINS = [(0, .02), (.02, .05), (.05, .10), (.10, .20), (.20, .35), (.35, .60), (.60, 1.01)]
print(f"{'Khoang':>13}" + "".join(f"{m:>24}" for m in MODELS))
print("-" * 112)
for lo, hi in BINS:
    cells = []
    for m in MODELS:
        q = allev[(allev[m] >= lo) & (allev[m] < hi)]
        cells.append(f"{q[m].mean():.1%} -> {q.hit.mean():.1%}  n={len(q)}" if len(q) >= 100 else "  —")
    print(f"{f'{lo:.0%}-{hi:.0%}':>13}" + "".join(f"{c:>24}" for c in cells))
print("-" * 112)
print("Doc: 'du bao -> thuc te'. Hai so cang sat nhau cang tot.")

json.dump({"models": MODELS, "brier": {m: float(np.mean(L[m])) for m in MODELS},
           "by_stop": [{"stop": st, "n": int(len(e)), "real": float(e.hit.mean()),
                        **{m: float(e[m].mean()) for m in MODELS}} for st, e in rows]},
          open("/tmp/fx/exp3b.json", "w"), indent=1)
allev.to_csv("/tmp/fx/exp3b_panel.csv", index=False)
print("\nDa luu exp3b.json")
