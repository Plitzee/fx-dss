"""VONG 7 — LUOI CAU HINH TANG 2, CHON TREN DOAN KIEM DINH.

Chay moi to hop cua nam truc trong volfc2, cham diem QLIKE tren doan KIEM
DINH, xep hang. Doan kiem tra KHONG duoc dung o day.

    deseason  none / wd / wdcov          (Boudt 2011; Dumitru 2025)
    crosspair off / on                   (Rubaszek 2025; Jia 2024)
    event     off / on                   (Lee-Wang 2025; Martins-Lopes 2024)
    window    exp / r1000 / r1500 / r2000 (Feng-Zhang-Wang 2024)
    shrink    lambda 0 .. 1              (Pesaran-Pick-Timmermann 2026)

Xuat: output/grid_valid.csv (moi dong mot cau hinh) va bang tom tat ra man hinh.
"""
import os
import sys
import time
import pickle
import warnings
import itertools
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

import volfc2 as V2
from split import VALID_TU, TEST_TU

CACHE = os.path.join(OUT, "_panel_cache.pkl")
LAMS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
DESEASON = ("none", "wd", "wdcov")
WINDOWS = (("exp", None), ("r1000", 1000), ("r1500", 1500), ("r2000", 2000))


def bang_cache():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    b = V2.nap_bang()
    with open(CACHE, "wb") as f:
        pickle.dump(b, f)
    return b


def main():
    bang, chung = bang_cache()
    n = len(chung)
    tr = np.asarray(chung < VALID_TU)
    va = np.asarray((chung >= VALID_TU) & (chung < TEST_TU))
    Y = {p: bang[p].rv5.values for p in V2.PAIRS}
    print(f"lưới cấu hình tầng 2 — {n:,} phiên chung, "
          f"huấn luyện {tr.sum():,} / kiểm định {va.sum():,}")
    print(f"chọn theo QLIKE trung bình 6 cặp trên đoạn KIỂM ĐỊNH "
          f"({chung[va][0].date()} → {chung[va][-1].date()})\n")

    rows = []
    t0 = time.time()
    combos = list(itertools.product(DESEASON, (False, True), (False, True), WINDOWS))
    for i, (ds, cp, ev, (wn, wv)) in enumerate(combos):
        r = V2.chay(bang, chung, deseason=ds, crosspair=cp, event=ev,
                    window=wv, lams=LAMS, train_mask=tr)
        for lam in LAMS:
            rec = {"deseason": ds, "crosspair": int(cp), "event": int(ev),
                   "window": wn, "lam": lam}
            qv, qt, ns = [], [], []
            for p in V2.PAIRS:
                f = r[lam][p]
                a, na = V2.qlike_tb(f, Y[p], va)
                b, _ = V2.qlike_tb(f, Y[p], tr)
                rec[f"v_{p}"] = a
                qv.append(a); qt.append(b); ns.append(na)
            rec["qlike_valid"] = float(np.mean(qv))
            rec["qlike_train"] = float(np.mean(qt))
            rec["n_valid"] = int(np.min(ns))
            rows.append(rec)
        if (i + 1) % 8 == 0:
            print(f"  {i+1:>2}/{len(combos)} tổ hợp  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows).sort_values("qlike_valid").reset_index(drop=True)
    df.to_csv(os.path.join(OUT, "grid_valid.csv"), index=False)

    base = df[(df.deseason == "none") & (df.crosspair == 0) & (df.event == 0)
              & (df.window == "exp") & (df.lam == 0.0)].iloc[0]
    print(f"\nCẤU HÌNH GỐC (volfc.py hiện tại): QLIKE kiểm định {base.qlike_valid:.4f}")
    print(f"\n15 cấu hình tốt nhất trên KIỂM ĐỊNH  ({len(df)} cấu hình đã thử)")
    print("-" * 92)
    print(f"{'#':>3} {'deseason':<9}{'chéo':>5}{'sự kiện':>9}{'cửa sổ':>8}{'lambda':>8}"
          f"{'QLIKE kđ':>11}{'so gốc':>9}{'QLIKE hl':>11}")
    print("-" * 92)
    for i, r in df.head(15).iterrows():
        d = (r.qlike_valid / base.qlike_valid - 1) * 100
        print(f"{i+1:>3} {r.deseason:<9}{r.crosspair:>5}{r.event:>9}{r.window:>8}"
              f"{r.lam:>8.2f}{r.qlike_valid:>11.4f}{d:>8.1f}%{r.qlike_train:>11.4f}")
    print("-" * 92)

    print("\nẢNH HƯỞNG RIÊNG TỪNG TRỤC (QLIKE kiểm định trung bình trên mọi cấu hình khác)")
    for col in ("deseason", "crosspair", "event", "window", "lam"):
        g = df.groupby(col).qlike_valid.agg(["mean", "min"]).sort_values("mean")
        print(f"\n  {col}:")
        for k, v in g.iterrows():
            print(f"    {str(k):<8} trung bình {v['mean']:.4f}   tốt nhất {v['min']:.4f}")

    b = df.iloc[0]
    print("\n" + "=" * 92)
    print("CẤU HÌNH THẮNG TRÊN KIỂM ĐỊNH:")
    print(f"  deseason={b.deseason}  crosspair={b.crosspair}  event={b.event}  "
          f"window={b.window}  lambda={b.lam}")
    print(f"  QLIKE kiểm định {b.qlike_valid:.4f} so với gốc {base.qlike_valid:.4f} "
          f"({(b.qlike_valid/base.qlike_valid-1)*100:+.1f}%)")
    print("\n  QLIKE kiểm định theo từng cặp (thắng so với gốc):")
    for p in V2.PAIRS:
        print(f"    {p}  {b['v_'+p]:.4f}  vs gốc {base['v_'+p]:.4f}  "
              f"({(b['v_'+p]/base['v_'+p]-1)*100:+.1f}%)")
    print("=" * 92)


if __name__ == "__main__":
    main()
