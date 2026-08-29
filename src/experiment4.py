"""
THI NGHIEM 4 — DON BAY -> XAC SUAT BI THANH LY.
Rao stop-out: theo ESMA, tai khoan bi dong khi von con 50% ky quy ban dau.
Voi don bay L, ky quy ban dau = 1/L gia tri danh nghia, nen rao gia = 0,5/L.
Mo phong nhieu ngay bang bootstrap khoi cac bo ba chuan hoa (z_T, z_L) tu lich su
-> giu nguyen ca duoi day LAN cau truc duong di trong phien.
"""
import numpy as np, pandas as pd, sys, json
sys.path.insert(0,"/tmp/fx/src")
from fxdata import PAIRS

P = pd.read_csv("/tmp/fx/exp3_panel.csv", parse_dates=["Date"])
LEV = [(30,"ESMA toi da (cap chinh)"), (20,"ESMA cap phu"), (50,"CFTC My (cap chinh)"),
       (10,"thuc te thuc dung"), (5,"than trong"), (2,"rat than trong")]
HOR = [1, 5, 20, 60]
NSIM = 40000

def survival(pair, lev, horizon, nsim=NSIM, seed=0):
    """Tra ve P(cham rao stop-out trong vong 'horizon' phien)."""
    s = P[P.pair == pair]
    zT = s.zT.values; zL = s.zL.values; sig = s.sig.values
    rng = np.random.default_rng(seed)
    barrier = -0.5/lev                                  # log-khoang cach ~ -0,5/L
    n = len(zT)
    idx = rng.integers(0, n, size=(nsim, horizon))      # bootstrap ngay (giu cap zT,zL)
    sg  = sig[idx]                                       # sigma cua chinh ngay do
    cum = np.zeros(nsim); hit = np.zeros(nsim, bool)
    for h in range(horizon):
        low_h  = cum + zL[idx[:, h]]*sg[:, h]           # cuc tieu trong phien h
        hit |= (low_h <= barrier)
        cum   = cum + zT[idx[:, h]]*sg[:, h]            # dong cua phien h
    return float(hit.mean())

print("="*104)
print("A. XAC SUAT BI THANH LY THEO DON BAY VA CHAN TROI  (EURUSD)")
print("="*104)
print("Rao stop-out = 0,5/L (dong tai khoan khi von con 50% ky quy ban dau, chuan ESMA).")
print(f"\n{'Don bay':>9}{'rao gia':>10}{'rao (sigma ngay)':>19}" +
      "".join(f"{f'{h} phien':>12}" for h in HOR) + "   Ghi chu")
print("-"*104)
sig_med = P[P.pair=="EURUSD"].sig.median()
for lev, note in LEV:
    b = 0.5/lev
    cells = [survival("EURUSD", lev, h, seed=100+lev) for h in HOR]
    print(f"{f'{lev}:1':>9}{b:>9.2%}{b/sig_med:>19.1f}" +
          "".join(f"{c:>11.1%}" for c in cells) + f"   {note}")
print("-"*104)

print("\n" + "="*104)
print("B. CUNG DON BAY 30:1 — TAT CA CAC CAP")
print("="*104)
print(f"{'Cap':<9}{'sigma ngay TB':>15}{'rao (sigma)':>13}" +
      "".join(f"{f'{h} phien':>12}" for h in HOR))
print("-"*104)
res_b = []
for p in PAIRS:
    sm = P[P.pair==p].sig.median()
    b = 0.5/30
    cells = [survival(p, 30, h, seed=7) for h in HOR]
    res_b.append(dict(pair=p, sig=float(sm), b_sig=float(b/sm),
                      **{f"h{h}": c for h, c in zip(HOR, cells)}))
    print(f"{p:<9}{sm:>14.3%}{b/sm:>13.1f}" + "".join(f"{c:>11.1%}" for c in cells))
print("-"*104)

print("\n" + "="*104)
print("C. RUI RO DUONG DI vs RUI RO CUOI KY O MUC DANH MUC")
print("="*104)
print("Cung mot vi the, cung chan troi: xac suat LO qua rao o CUOI KY so voi")
print("xac suat CHAM rao bat ky luc nao tren duong di.")
print(f"\n{'Don bay':>9}{'chan troi':>11}{'P(cuoi ky duoi rao)':>22}{'P(cham rao)':>14}{'ty le':>9}")
print("-"*104)
def terminal_only(pair, lev, horizon, nsim=NSIM, seed=0):
    s = P[P.pair==pair]; zT=s.zT.values; sig=s.sig.values
    rng = np.random.default_rng(seed); n=len(zT)
    idx = rng.integers(0,n,size=(nsim,horizon))
    cum = (zT[idx]*sig[idx]).sum(axis=1)
    return float((cum <= -0.5/lev).mean())
for lev in (30, 10, 5):
    for h in (1, 5, 20):
        pt = terminal_only("EURUSD", lev, h, seed=3)
        pp = survival("EURUSD", lev, h, seed=3)
        r = pp/pt if pt>0 else np.nan
        print(f"{f'{lev}:1':>9}{h:>11}{pt:>22.2%}{pp:>14.2%}{r:>9.2f}")
print("-"*104)
print("Ty le nay chinh la con so ma mot he thong chi bao cao VaR se bo sot.")

print("\n" + "="*104)
print("D. KHOI LUONG VI THE TOI DA DE XAC SUAT THANH LY <= 1% TRONG 20 PHIEN")
print("="*104)
print(f"{'Cap':<9}{'don bay toi da':>17}{'rao tuong ung':>16}{'P(thanh ly) 20 phien':>23}")
print("-"*104)
res_d=[]
for p in PAIRS:
    best=None
    for lev in np.arange(1.0, 60.5, 0.5):
        if survival(p, lev, 20, nsim=20000, seed=5) > 0.01:
            break
        best = lev
    if best is None: best = 1.0
    pr = survival(p, best, 20, nsim=20000, seed=5)
    res_d.append(dict(pair=p, lev=float(best), p=float(pr)))
    print(f"{p:<9}{f'{best:.1f}:1':>17}{0.5/best:>16.2%}{pr:>23.2%}")
print("-"*104)
print("So sanh: tran phap ly cua ESMA la 30:1 va cua CFTC la 50:1.")

json.dump({"lev_all": res_b, "max_lev_1pct": res_d}, open("/tmp/fx/exp4.json","w"), indent=1)
print("\nDa luu exp4.json")
