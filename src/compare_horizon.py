"""TANG 4 — KHUNG THOI GIAN CUA TRAN RUI RO: 250 phien (hien dung) hay 20 phien?

Cau hoi tu nghien cuu thi truong: nha dau tu FX thuc te khong nghi theo nam
(retail trung vi 16 PHUT — NBER w22146; swap lien ngan hang 42% khoi luong
la duoi 7 NGAY — BIS Triennial 2025; ngay ca carry-trade hoc thuat cham nhat
cung tai can bang HANG THANG). Tang 6b da dung N=20 phien (~1 thang) cho
dung tam. Nhung `position_sizing.py` van dung horizon=250 (~1 nam) cho
CONG THUC TRAN RUI RO PHA SAN — tuc gia dinh don bay giu NGUYEN trong ca
nam khi tinh xac suat pha san, du thuc te DUOC TAI CAN BANG MOI 5 PHIEN.

Thiet ke phep thu: GIU NGUYEN do dai mo phong that (simulate2 horizon=250,
rebalance=5 — dung 1 nam thuc, tai can bang moi 5 ngay, giong het
compare_sizing.py) — CHI DOI tham so horizon BEN TRONG cong thuc f_ruin_cap
(cai quyet dinh don bay o moi lan tai can bang). Day la phep thu cong bang:
neu horizon=20 van giu duoc xac suat pha san THUC DO trong mo phong <=
ngan sach da cam ket, ma tang truong cao hon, thi horizon=250 la qua
than trong mot cach khong can thiet.

Chay:  python src/compare_horizon.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sizing import f_kelly, f_ruin_cap  # noqa: E402
from sizing2 import simulate2, Fuzzy  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
PANEL = os.environ.get("FX_PANEL", "panel2_6pairs.csv")
pan = pd.read_csv(os.path.join(ROOT, "data", PANEL))
print(f"panel: {PANEL}  ({len(pan):,} dòng)")
MU = 0.0002
C = {}


def build(p):
    if p not in C:
        d = pan[pan.pair == p].reset_index(drop=True)
        n = int(len(d) * 0.70)
        C[p] = (d.iloc[n:].reset_index(drop=True), Fuzzy(d.sig.values[:n]))
    return C[p]


def K(st):
    return f_kelly(st["mu"], st["sig"])


def k_dd(dd):
    return np.clip(1.30 - 3.2 * dd, 0.5, 1.30)


def k_vol(sig, F):
    lo, me, hi = F.mu_vol(sig)
    w = lo + me + hi
    return np.where(w > 1e-9, (lo * 1.30 + me * 1.10 + hi * 0.90) / np.maximum(w, 1e-9), 1.0)


def rule_h(h):
    """Quy tac san xuat (tich hai he so), khung rui ro `h` phien ben trong f_ruin_cap."""
    return lambda st, F, b: np.minimum(
        K(st), k_dd(st["dd"]) * k_vol(st["sig"], F) * f_ruin_cap(st["sig"], h, b, st["nu"]))


M = {
    "khung 250 phiên (hiện dùng)": rule_h(250),
    "khung 20 phiên (đề xuất)": rule_h(20),
}
BUD = (0.003, 0.01, 0.03, 0.08, 0.20)

print("=" * 96)
print("TRAN RUI RO PHA SAN: KHUNG 250 PHIEN (~1 nam) HAY 20 PHIEN (~1 thang)?")
print("Mo phong THAT giu nguyen 250 phien, tai can bang moi 5 phien (giong compare_sizing.py).")
print("Chi doi tham so horizon BEN TRONG cong thuc tran — do ung xu that khi chay ca nam.")
print("=" * 96)
pts = {}
for name, fn in M.items():
    pts[name] = []
    print(f"\n{name}")
    print(f"  {'ngân sách':<12}{'tăng trưởng':>13}{'phá sản THỰC ĐO':>18}{'đòn bẩy TB':>13}")
    for b in BUD:
        G, R, L = [], [], []
        for sd in (1, 2, 3):
            g, r, lv = [], [], []
            for pair in P:
                te, F = build(pair)
                o = simulate2(te, lambda st, fn=fn, F=F, b=b: fn(st, F, b), n_paths=4000,
                              horizon=250, mu_true=MU, mu_believed=MU, seed=sd)
                g.append(o["mean_log_growth"]); r.append(o["p_ruin"]); lv.append(o["avg_lev"])
            G.append(np.median(g)); R.append(max(r)); L.append(np.mean(lv))
        pts[name].append((np.mean(R), np.mean(G)))
        print(f"  b={b:<10.3f}{np.mean(G):>12.2%}{np.mean(R):>17.2%}{np.mean(L):>13.2f}")

print("\n" + "=" * 96)
print("TANG TRUONG TAI CUNG MUC PHA SAN THUC DO (nội suy trên biên)")
print("=" * 96)
T = [0.001, 0.003, 0.01, 0.03]
print(f"{'Khung':<32}" + "".join(f"{f'{t:.1%}':>14}" for t in T))
print("-" * 96)
tab = {}
for name, pl in pts.items():
    pl = sorted(pl); xs = [p[0] for p in pl]; ys = [p[1] for p in pl]
    row = [np.interp(t, xs, ys) if min(xs) - 1e-9 <= t <= max(xs) + 1e-9 else np.nan for t in T]
    tab[name] = row
    print(f"{name:<32}" + "".join(f"{v:>13.2%}" if v == v else f"{'—':>14}" for v in row))
print("-" * 96)

b250, b20 = tab["khung 250 phiên (hiện dùng)"], tab["khung 20 phiên (đề xuất)"]
d = [a - c for a, c in zip(b20, b250) if a == a and c == c]
print(f"\nKhung 20 phiên so với 250 phiên: {np.mean(d):+.2%} tăng trưởng trung bình, "
      "TẠI CÙNG mức phá sản thực đo (không phải phá sản cam kết).")
print("\nTU KIEM DAT")
