"""THI NGHIEM 1 — mo hinh nao du bao bien dong tot nhat, va Patton (2011) co dung khong?"""
import numpy as np, pandas as pd, sys, json
sys.path.insert(0,"/tmp/fx/src")
from fxdata import PAIRS
from metrics import qlike, mse_var, mcs
pd.set_option("display.width", 200)

MODELS = ["RW-cc","MA20-cc","EWMA94-cc","RW-GK","MA20-GK",
          "HAR-Park","HAR-GK","HAR-RS","HAR-RV","GARCH-t"]

def load(p):
    o = pd.read_csv(f"/tmp/fx/fc_{p}.csv", parse_dates=["Date"])
    return o.dropna(subset=MODELS+["rv","gk","r"]).reset_index(drop=True)

# ── ham mat mat: 2 ben (Patton), 2 khong ben
def L_qlike(prox,h): return np.log(h)+prox/h
def L_mse(prox,h):   return (prox-h)**2
def L_mae(prox,h):   return np.abs(prox-h)                 # KHONG ben
def L_mse_sd(prox,h):return (np.sqrt(prox)-np.sqrt(h))**2  # KHONG ben
LOSSES = {"QLIKE":L_qlike, "MSE":L_mse, "MAE (khong ben)":L_mae, "MSE-sd (khong ben)":L_mse_sd}

print("="*118)
print("A. QLIKE TRUNG BINH THEO TUNG CAP  (thuoc do = RV tu 24 thanh gio; thap hon = tot hon)")
print("="*118)
Q = {}
hdr = f"{'Cap':<9}" + "".join(f"{m:>11}" for m in MODELS)
print(hdr); print("-"*118)
allL = {}
for p in PAIRS:
    o = load(p); prox = o.rv.values
    row = []
    L = np.column_stack([L_qlike(prox, o[m].values) for m in MODELS])
    allL[p] = L
    q = L.mean(axis=0); Q[p] = q
    best = int(np.argmin(q))
    print(f"{p:<9}" + "".join((f"{v:>10.4f}*" if i==best else f"{v:>11.4f}") for i,v in enumerate(q)))
print("-"*118)
QM = pd.DataFrame(Q, index=MODELS).T
print(f"{'TRUNG BINH':<9}" + "".join(f"{v:>11.4f}" for v in QM.mean()))
rank = QM.rank(axis=1).mean()
print(f"{'HANG TB':<9}" + "".join(f"{v:>11.2f}" for v in rank))
print(f"{'So lan #1':<9}" + "".join(f"{int((QM.idxmin(axis=1)==m).sum()):>11}" for m in MODELS))
print("-"*118)
print("* = tot nhat trong hang. QLIKE co the am; chi so tuyet doi khong co y nghia, chi so SANH moi co.")

print("\n" + "="*118)
print("B. MODEL CONFIDENCE SET (Hansen-Lunde-Nason), alpha = 10%, bootstrap khoi 1000 lan")
print("="*118)
Lpool = np.vstack([allL[p] for p in PAIRS])
keep, elim = mcs(Lpool, alpha=0.10, B=1000, block=20, seed=7)
print(f"Gop toan bo {Lpool.shape[0]} quan sat ngoai mau cua 12 cap:")
print(f"   TAP TIN CAY 90%: {[MODELS[i] for i in keep]}")
print(f"   Bi loai (theo thu tu):")
for i,(m,pv) in enumerate(elim, 1):
    print(f"      {i:>2}. {MODELS[m]:<12} loai o p = {pv:.3f}")
print("\nTung cap rieng:")
print(f"{'Cap':<9}{'Kich thuoc MCS':>16}   Cac mo hinh song sot")
print("-"*118)
for p in PAIRS:
    k,_ = mcs(allL[p], alpha=0.10, B=500, block=20, seed=11)
    print(f"{p:<9}{len(k):>16}   {', '.join(MODELS[i] for i in k)}")
print("-"*118)

print("\n" + "="*118)
print("C. KIEM CHUNG PATTON (2011) TREN DU LIEU THAT")
print("="*118)
print("Neu ham mat mat BEN voi thuoc do nhieu thi doi thuoc do KHONG duoc lam doi thu hang.")
print("Ba thuoc do: RV tu h1 (it nhieu nhat) | Garman-Klass | binh phuong loi suat ngay (nhieu nhat).\n")
PROXIES = {"RV_h1":"rv", "Garman-Klass":"gk", "r^2 ngay":"cc"}
for lname, lf in LOSSES.items():
    ranks = {}
    for pxname, col in PROXIES.items():
        agg = np.zeros(len(MODELS))
        for p in PAIRS:
            o = load(p)
            prox = (o.r.values**2) if col=="cc" else o[col].values
            prox = np.clip(prox, 1e-12, None)
            agg += np.array([lf(prox, o[m].values).mean() for m in MODELS])
        ranks[pxname] = pd.Series(agg/len(PAIRS), index=MODELS).rank()
    R = pd.DataFrame(ranks)
    from scipy.stats import spearmanr
    rho_gk = spearmanr(R["RV_h1"], R["Garman-Klass"]).statistic
    rho_cc = spearmanr(R["RV_h1"], R["r^2 ngay"]).statistic
    top = {k: R[k].idxmin() for k in R}
    print(f"{lname}")
    print(f"   tuong quan hang RV vs GK = {rho_gk:+.3f} | RV vs r^2 = {rho_cc:+.3f}"
          f"   |  mo hinh #1 theo tung thuoc do: {top['RV_h1']} / {top['Garman-Klass']} / {top['r^2 ngay']}")
    same = (top['RV_h1']==top['Garman-Klass']==top['r^2 ngay'])
    print(f"   -> {'GIU nguyen mo hinh tot nhat' if same else 'DOI mo hinh tot nhat khi doi thuoc do'}")
print("-"*118)

json.dump({m: float(QM.mean()[m]) for m in MODELS}, open("/tmp/fx/exp1_qlike.json","w"), indent=1)
QM.to_csv("/tmp/fx/exp1_qlike_bypair.csv")
print("\nDa luu exp1_qlike.json, exp1_qlike_bypair.csv")
