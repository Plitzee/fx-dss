"""VONG 7 — LUOI CAU HINH DAY DU + CHON BANG MODEL CONFIDENCE SET.

Van de cua run_grid.py: chon cau hinh TOT NHAT trong hang tram cau hinh
tren doan kiem dinh chinh la mot bai toan kiem dinh boi. Chenh lech giua
cau hinh thu 1 va thu 15 chi 0,0009 QLIKE — nho hon nhieu so voi sai so
lay mau. Chon cai nho nhat la overfit doan kiem dinh.

Giao thuc dung o day:
  1. Chay toan bo luoi, cham QLIKE tren doan KIEM DINH.
  2. Lay 40 cau hinh dau, chay Model Confidence Set (Hansen-Lunde-Nason 2011)
     tren chuoi ton that KIEM DINH.
  3. Trong tap MCS, chon cau hinh DON GIAN NHAT (it truc bat nhat), khong
     phai cau hinh co QLIKE nho nhat.

Sau buoc 3 la CHOT. Doan kiem tra chi duoc mo o run_final7.py.
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
from metrics import mcs
from split import VALID_TU, TEST_TU
from run_grid import bang_cache

LAMS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
DESEASON = ("none", "wd")
EVENTS = ("off", "chung", "cap", "cbonly", "capday")
WINDOWS = (("exp", None), ("r2000", 2000))
RECAL = ("off", "mz")

# "do phuc tap" — de chon cau hinh don gian nhat trong tap MCS
CHIPHI = {"deseason": {"none": 0, "wd": 1},
          "event": {"off": 0, "cbonly": 1, "cap": 2, "chung": 2, "capday": 3},
          "window": {"exp": 0, "r2000": 1},
          "recal": {"off": 0, "mz": 1}}


def do_phuc_tap(r):
    return (CHIPHI["deseason"][r.deseason] + CHIPHI["event"][r.event]
            + CHIPHI["window"][r.window] + CHIPHI["recal"][r.recal]
            + int(r.crosspair) + (0 if r.lam == 0.0 else 1))


def main():
    bang, chung = bang_cache()
    n = len(chung)
    tr = np.asarray(chung < VALID_TU)
    va = np.asarray((chung >= VALID_TU) & (chung < TEST_TU))
    Y = {p: bang[p].rv5.values for p in V2.PAIRS}
    iva = np.where(va)[0]

    def loss_ngay(f, p):
        """QLIKE tung ngay tren doan kiem dinh (NaN neu thieu)."""
        y = Y[p][iva]; g = f[iva]
        ok = np.isfinite(g) & np.isfinite(y) & (g > 0) & (y > 0)
        r = np.full(len(iva), np.nan)
        rr = y[ok] / g[ok]
        r[ok] = rr - np.log(rr) - 1
        return r

    combos = list(itertools.product(DESEASON, (False, True), EVENTS, WINDOWS, RECAL))
    print(f"lưới đầy đủ: {len(combos)} tổ hợp × {len(LAMS)} lambda = "
          f"{len(combos)*len(LAMS)} cấu hình")
    print(f"chấm trên KIỂM ĐỊNH {chung[va][0].date()} → {chung[va][-1].date()} "
          f"({va.sum()} phiên)\n")

    rows, Ls = [], []
    t0 = time.time()
    for i, (ds, cp, ev, (wn, wv), rc) in enumerate(combos):
        r = V2.chay(bang, chung, deseason=ds, crosspair=cp, event=ev,
                    window=wv, lams=LAMS, train_mask=tr, recal=rc)
        for lam in LAMS:
            M = np.column_stack([loss_ngay(r[lam][p], p) for p in V2.PAIRS])
            Ls.append(np.nanmean(M, 1))
            rec = {"deseason": ds, "crosspair": int(cp), "event": ev,
                   "window": wn, "recal": rc, "lam": lam}
            for j, p in enumerate(V2.PAIRS):
                rec["v_" + p] = float(np.nanmean(M[:, j]))
            rec["qlike_valid"] = float(np.nanmean(np.nanmean(M, 1)))
            rec["qlike_valid_bo_jpy"] = float(np.nanmean(
                [rec["v_" + p] for p in V2.PAIRS if p != "USDJPY"]))
            rows.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  {i+1:>3}/{len(combos)} tổ hợp ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    L = np.column_stack(Ls)
    ok = np.isfinite(L).all(1)
    L = L[ok]
    print(f"\n  chuỗi tổn thất kiểm định: {L.shape[0]} phiên × {L.shape[1]} cấu hình")

    df["rank"] = df.qlike_valid.rank().astype(int)
    ord_ = np.argsort(df.qlike_valid.values)
    df.to_csv(os.path.join(OUT, "grid2_valid.csv"), index=False)

    base_i = df.index[(df.deseason == "none") & (df.crosspair == 0) & (df.event == "off")
                      & (df.window == "exp") & (df.recal == "off") & (df.lam == 0.0)][0]
    base = df.loc[base_i]
    print(f"\nCẤU HÌNH GỐC (volfc.py hiện tại): QLIKE kiểm định {base.qlike_valid:.4f}")

    print(f"\n20 cấu hình tốt nhất trên KIỂM ĐỊNH")
    print("-" * 104)
    print(f"{'#':>3} {'deseason':<9}{'chéo':>5}{'sự kiện':>9}{'cửa sổ':>7}{'hiệu chuẩn':>12}"
          f"{'lambda':>8}{'QLIKE kđ':>11}{'so gốc':>9}{'phức tạp':>10}")
    print("-" * 104)
    for k, j in enumerate(ord_[:20]):
        r = df.loc[j]
        d = (r.qlike_valid / base.qlike_valid - 1) * 100
        print(f"{k+1:>3} {r.deseason:<9}{r.crosspair:>5}{r.event:>9}{r.window:>7}"
              f"{r.recal:>12}{r.lam:>8.2f}{r.qlike_valid:>11.4f}{d:>8.1f}%"
              f"{do_phuc_tap(r):>10}")
    print("-" * 104)

    print("\nẢNH HƯỞNG RIÊNG TỪNG TRỤC (QLIKE kiểm định)")
    for col in ("deseason", "crosspair", "event", "window", "recal", "lam"):
        g = df.groupby(col).qlike_valid.agg(["mean", "min"]).sort_values("mean")
        print(f"\n  {col}:")
        for k, v in g.iterrows():
            print(f"    {str(k):<8} trung bình {v['mean']:.4f}   tốt nhất {v['min']:.4f}")

    # ── Model Confidence Set trên 40 cấu hình đầu
    top = ord_[:40]
    print(f"\n\nMODEL CONFIDENCE SET trên {len(top)} cấu hình đầu (alpha=0,10, bootstrap khối 20)")
    alive, elim = mcs(L[:, top], alpha=0.10, B=2000, block=20, seed=7)
    keep = [top[a] for a in alive]
    print(f"  {len(keep)}/{len(top)} cấu hình SỐNG SÓT — không phân biệt được về mặt thống kê")
    sub = df.loc[keep].copy()
    sub["phuctap"] = [do_phuc_tap(r) for _, r in sub.iterrows()]
    sub = sub.sort_values(["phuctap", "qlike_valid"])
    print(f"\n  Trong tập MCS, xếp theo ĐỘ PHỨC TẠP rồi mới tới QLIKE:")
    print("  " + "-" * 96)
    print(f"  {'deseason':<9}{'chéo':>5}{'sự kiện':>9}{'cửa sổ':>7}{'hiệu chuẩn':>12}"
          f"{'lambda':>8}{'phức tạp':>10}{'QLIKE kđ':>11}")
    print("  " + "-" * 96)
    for _, r in sub.head(12).iterrows():
        print(f"  {r.deseason:<9}{r.crosspair:>5}{r.event:>9}{r.window:>7}{r.recal:>12}"
              f"{r.lam:>8.2f}{r.phuctap:>10}{r.qlike_valid:>11.4f}")
    print("  " + "-" * 96)

    ch = sub.iloc[0]
    print("\n" + "=" * 104)
    print("CẤU HÌNH CHỐT (đơn giản nhất trong tập MCS trên đoạn kiểm định):")
    print(f"  deseason={ch.deseason}  crosspair={ch.crosspair}  event={ch.event}  "
          f"window={ch.window}  recal={ch.recal}  lambda={ch.lam}")
    print(f"  QLIKE kiểm định {ch.qlike_valid:.4f}  so với gốc {base.qlike_valid:.4f}  "
          f"({(ch.qlike_valid/base.qlike_valid-1)*100:+.1f}%)")
    print(f"  (cấu hình QLIKE nhỏ nhất là {df.loc[ord_[0]].qlike_valid:.4f} — "
          f"chênh {abs(ch.qlike_valid-df.loc[ord_[0]].qlike_valid):.4f}, nằm trong sai số)")
    print("\n  QLIKE kiểm định theo từng cặp:")
    for p in V2.PAIRS:
        print(f"    {p}  {ch['v_'+p]:.4f}   gốc {base['v_'+p]:.4f}   "
              f"({(ch['v_'+p]/base['v_'+p]-1)*100:+.1f}%)")
    print(f"\n  trung bình 6 cặp     {ch.qlike_valid:.4f}")
    print(f"  trung bình 5 cặp (bỏ USDJPY) {ch.qlike_valid_bo_jpy:.4f}")
    print("=" * 104)

    with open(os.path.join(OUT, "cauhinh_chot.pkl"), "wb") as f:
        pickle.dump({k: ch[k] for k in ("deseason", "crosspair", "event",
                                        "window", "recal", "lam")}, f)
    print(f"\nđã ghi output/cauhinh_chot.pkl — run_final7.py sẽ đọc file này")


if __name__ == "__main__":
    main()
