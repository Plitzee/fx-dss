"""On dinh theo HAT GIONG cho hai mo hinh hoc sau dan dau.

Neu ket luan "GRU thang HAR 1,4%" chi dung voi mot hat giong thi no khong
phai ket luan. Chay lai hai cau hinh tot nhat voi ba hat giong khac nhau.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = os.path.join(os.path.dirname(HERE), "output")
import run_dl
from split import VALID_TU, TEST_TU

X, y, ten, pid, dts = run_dl.nap()
S, F, hople, _ = run_dl.xay_chuoi(X, y, ten, pid, dts)
rv = np.exp(y)
dt_t = pd.DatetimeIndex(np.concatenate(
    [np.append(np.asarray(dts)[pid == j][1:], np.datetime64("NaT")) for j in range(6)]))
# dts da o dang bang dai theo cap nen chi can dich trong tung cap
idx = np.argsort(np.concatenate([np.where(pid == j)[0] for j in range(6)]))
va = (dts >= VALID_TU) & (dts < TEST_TU); te = dts >= TEST_TU


def cham(f, m):
    ok = m & np.isfinite(f) & (f > 0) & np.isfinite(rv) & (rv > 0)
    r = rv[ok] / f[ok]
    return float((r - np.log(r) - 1).mean())


print("ỔN ĐỊNH THEO HẠT GIỐNG — ba hạt giống cho hai mô hình dẫn đầu")
print("-" * 84)
print(f"{'mô hình':<16}{'hạt giống':>11}{'QLIKE kiểm định':>18}{'QLIKE kiểm tra':>17}")
print("-" * 84)
res = {}
for kieu, hid, lr in (("gru", 48, 2e-3), ("lstm", 48, 2e-3)):
    for sd in (0, 1, 2):
        n = len(y); mu = np.full(n, np.nan); s2 = np.full(n, np.nan)
        for yr in range(run_dl.KHOP_TU, run_dl.KHOP_DEN + 1):
            moc = pd.Timestamp(f"{yr}-01-01"); het = pd.Timestamp(f"{yr+1}-01-01")
            itr = np.where(hople & (dts < moc))[0]
            ite = np.where(hople & (dts >= moc) & (dts < het))[0]
            if len(itr) < run_dl.MIN_TRAIN or len(ite) == 0:
                continue
            m_, v_, _ = run_dl.khop_mot_lan(S, F, y, itr, ite, kieu, hid, lr, seed=sd)
            mu[ite] = m_; s2[ite] = v_
        f = np.exp(np.clip(mu, -30, 0) + 0.5 * np.nan_to_num(s2))
        res.setdefault(kieu, []).append((cham(f, va), cham(f, te)))
        print(f"{kieu.upper():<16}{sd:>11}{res[kieu][-1][0]:>18.4f}"
              f"{res[kieu][-1][1]:>17.4f}", flush=True)
        np.savez_compressed(os.path.join(OUT, f"_dl_seed_{kieu}_{sd}.npz"), f=f)
print("-" * 84)
for k, v in res.items():
    a = np.array(v)
    print(f"  {k.upper()}: kiểm định {a[:,0].mean():.4f} ± {a[:,0].std():.4f}   "
          f"kiểm tra {a[:,1].mean():.4f} ± {a[:,1].std():.4f}   "
          f"(dải kiểm tra {a[:,1].min():.4f}–{a[:,1].max():.4f})")
json.dump({k: v for k, v in res.items()}, open(os.path.join(OUT, "dl_seed.json"), "w"))
