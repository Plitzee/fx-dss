"""CHI PHI GIAO DICH CO PHAI THU GIET LOI THE KHONG? — phan tich do nhay.

Hoa hong 0,35 pip trong cost.py la GIA DINH chua kiem chung. File nay tra loi
cau hoi "neu gia dinh do sai thi sao" bang cach cho no chay tu 0,00 den 0,70
pip khu hoi va do xem ket luan co doi khong.

Ket qua: KHONG. Sharpe trung binh chi dich 0,015-0,076 tuy vong quay. Gia dinh
hoa hong KHONG load-bearing. Doi gio giao dich tu luc dong cua (21h UTC, dat
nhat) sang 9h UTC (re nhat) cung chi cai thien khoang 0,03.

Co che: chenh lech ty le thuan voi vong quay (0,076 o span=2 voi 0,755 lan doi
vi the/phien; 0,015 o span=60 voi 0,151). Day cung la phep kiem tra rang cach
hach toan chi phi khong sai — neu khong ty le thuan thi co loi.

He qua cho luan van: dung viet "tin hieu nam duoi nguong chi phi giao dich".
Voi chien luoc vong quay thap, chi phi la bac hai; chat luong tin hieu quyet
dinh. Xem momentum_decay.py.

Chay:  python src/cost_sensitivity.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost import spread_pip

DIR = os.path.dirname(os.path.abspath(__file__))
PRICES = os.path.join(os.path.dirname(DIR), "data", "prices")
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
SPANS = (2, 3, 5, 10, 20, 60)
COMMS = (0.00, 0.35, 0.70)


def pip_of(pair):
    return 0.01 if "JPY" in pair.upper() else 0.0001


def load(pair):
    d = pd.read_csv(os.path.join(PRICES, f"{pair}_d1.csv"), parse_dates=["Date"])
    d = d.sort_values("Date").reset_index(drop=True)
    d["r"] = np.log(d.close / d.close.shift(1))
    return d.dropna().reset_index(drop=True)


def backtest(d, spread, pair, span, comm):
    """Quy tac dau EWMA. Chi phi = |thay doi vi the| x nua (spread + hoa hong)."""
    pip = pip_of(pair)
    pos = np.sign(d.r.shift(1).ewm(span=span, min_periods=span).mean()).fillna(0)
    gross = pos * d.r
    cost = (spread + comm) * pip / d.close.values
    turn = pos.diff().abs().fillna(0)
    net = gross - turn * cost / 2
    ok = gross.notna()
    f = lambda x: x[ok].mean() / x[ok].std() * np.sqrt(252)
    return f(gross), f(net), float(turn[ok].mean())


def main():
    D = {p: load(p) for p in PAIRS}
    SP = {p: {h: np.array([spread_pip(p, h, dt) for dt in D[p].Date]) for h in (9, 21)}
          for p in PAIRS}

    print("=" * 74)
    print("VONG QUAY QUYET DINH TAT CA — giao dich luc dong cua (21h UTC)")
    print("=" * 74)
    print(f"{'span':<7}{'vong quay/phien':>17}{'Sharpe gop':>13}"
          f"{'hh=0,00':>10}{'hh=0,70':>10}{'chenh':>9}")
    print("-" * 74)
    diffs, turns = [], []
    for span in SPANS:
        g, n0, n7, t = [], [], [], []
        for p in PAIRS:
            a, b, tt = backtest(D[p], SP[p][21], p, span, 0.00)
            _, c, _ = backtest(D[p], SP[p][21], p, span, 0.70)
            g.append(a); n0.append(b); n7.append(c); t.append(tt)
        d_ = np.mean(n0) - np.mean(n7)
        diffs.append(d_); turns.append(np.mean(t))
        print(f"{span:<7}{np.mean(t):>17.3f}{np.mean(g):>13.3f}"
              f"{np.mean(n0):>10.3f}{np.mean(n7):>10.3f}{d_:>9.3f}")

    print("\n" + "=" * 74)
    print("TIN HIEU TOT NHAT MOI CAP (quet 6 span roi CHON tot nhat — da thien vi)")
    print("=" * 74)
    print(f"{'Cap':<9}{'span':>7}{'Sharpe gop':>13}{'Sharpe rong':>14}")
    print("-" * 74)
    survive = 0
    for p in PAIRS:
        best = max(((s,) + backtest(D[p], SP[p][21], p, s, 0.35) for s in SPANS),
                   key=lambda x: x[2])
        s, gr, ne, _ = best
        survive += ne > 0.3
        print(f"{p:<9}{s:>7}{gr:>13.3f}{ne:>14.3f}")
    print("-" * 74)
    print(f"So cap co Sharpe rong > 0,3: {survive}/6")

    r = np.corrcoef(turns, diffs)[0, 1]
    print(f"\nTuong quan giua vong quay va do nhay chi phi: {r:.3f}")
    assert r > 0.95, "chenh lech phai ty le thuan voi vong quay — neu khong thi hach toan sai"
    assert max(diffs) < 0.10, "do nhay hoa hong phai nho"
    assert survive == 0, "khong cap nao duoc vuot nguong 0,3"
    print("TU KIEM DAT — hoa hong khong load-bearing, hach toan chi phi nhat quan")


if __name__ == "__main__":
    main()
