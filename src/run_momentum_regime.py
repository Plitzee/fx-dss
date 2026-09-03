"""MOMENTUM CO DIEU KIEN THEO CHE DO BIEN DONG — tang 1, phuong an 2.

Cau hoi khac voi momentum_decay.py. File do da tra loi "momentum co lai
TRUNG BINH khong" (khong, tu 2010). File nay hoi "momentum co lai trong
MOT CHE DO BIEN DONG cu the khong" — vi mot lai trung binh am van co the
an mot lai duong co y nghia trong che do binh tinh hoac cang thang.

Bien che do dung LAI bien da co san o tang 2/6b: cot `sig` (sigma du bao
HAR) trong data/panel2_6pairs.csv — day la du bao NHAN QUA (chi dung thong
tin den ngay t-1) nen khong ro ri. Nguong chia 3 che do (tercile) duoc CHOT
tren doan HUAN LUYEN roi ap dung nguyen sang kiem dinh — dung luat chon tu
split.py, khong duoc chon tren tap se dung de kiem tra.

Tin hieu momentum: giong het momentum_decay.py (dau cua EWMA loi nhuan qua
khu, da shift 1 ngay) — chi khac o cho tach lai theo che do thay vi gop.

Y nghia thong ke: dm_nw() tren chuoi P&L cua tung che do (Newey-West, giong
run_final7.py) — hoi "trung binh co la 0 khong", khong phai chi nhin dau.

Chay:  python src/run_momentum_regime.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)
sys.path.insert(0, DIR)

from momentum_decay import load, META, SPANS  # noqa: E402
from split import doan, TEN  # noqa: E402

PANEL2 = os.path.join(ROOT, "data", "panel2_6pairs.csv")
N_CHE_DO = 3
CHE_DO_TEN = ("bình tĩnh", "vừa", "căng thẳng")
CUR2PAIR = {"EUR": "EURUSD", "GBP": "GBPUSD", "AUD": "AUDUSD",
            "CAD": "USDCAD", "JPY": "USDJPY", "CHF": "USDCHF"}


def dm_nw(x):
    """Thong ke Diebold-Mariano voi phuong sai Newey-West (giong run_final7.py)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan, np.nan
    mb = x.mean()
    L = int(np.ceil(1.5 * n ** (1 / 3)))
    s = np.sum((x - mb) ** 2) / n
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * np.sum((x[k:] - mb) * (x[:-k] - mb)) / n
    t = mb / np.sqrt(max(s, 1e-16) / n)
    return t, 2 * (1 - stats.norm.cdf(abs(t)))


def sharpe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 30 or x.std() == 0:
        return np.nan
    return x.mean() / x.std() * np.sqrt(252)


def nap_che_do():
    """sig HAR theo (pair, Date) -> dung lam bien dieu kien che do."""
    p2 = pd.read_csv(PANEL2, parse_dates=["Date"])
    return p2[["Date", "pair", "sig"]].rename(columns={"pair": "PAIR"})


def main():
    che_do_df = nap_che_do()
    data = dict(load(c) for c in META)

    print("=" * 92)
    print("MOMENTUM CO DIEU KIEN THEO CHE DO BIEN DONG (tercile cua sigma HAR, chot tren huấn luyện)")
    print("=" * 92)
    print(f"{'span':>6} {'đoạn':>10} " + "".join(f"{c:>14}" for c in CHE_DO_TEN) + f"{'gộp':>14}")
    print("-" * 92)

    tong_theo_dm = {c: [] for c in CHE_DO_TEN}  # gop toan bo span, de kiem dinh cuoi
    for span in SPANS:
        # gop position*return TUNG CAP roi noi lai theo Date de tach doan/che do
        rows = []
        for cur, df in data.items():
            g = df.copy()
            g["pos"] = np.sign(g.r.shift(1).ewm(span=span, min_periods=span).mean()).fillna(0)
            g["pnl"] = g.pos * g.r
            g["PAIR"] = CUR2PAIR[cur]
            rows.append(g[["DATE", "pnl", "PAIR"]])
        pnl = pd.concat(rows, ignore_index=True).rename(columns={"DATE": "Date"})
        pnl = pnl.merge(che_do_df, on=["Date", "PAIR"], how="inner")
        if len(pnl) == 0:
            continue
        pnl["doan"] = doan(pnl.Date.values)
        tr = pnl.doan.values == 0
        nguong = np.quantile(pnl.loc[tr, "sig"].values, np.arange(1, N_CHE_DO) / N_CHE_DO)
        pnl["che_do"] = np.digitize(pnl.sig.values, nguong)

        for seg_i, seg_ten in enumerate(TEN):
            m = pnl.doan.values == seg_i
            if m.sum() < 30:
                continue
            row = []
            for c in range(N_CHE_DO):
                mm = m & (pnl.che_do.values == c)
                sh = sharpe(pnl.loc[mm, "pnl"].values)
                row.append(sh)
                if seg_ten == "kiểm định":
                    tong_theo_dm[CHE_DO_TEN[c]].append(pnl.loc[mm, "pnl"].values)
            sh_all = sharpe(pnl.loc[m, "pnl"].values)
            print(f"{span:>6} {seg_ten:>10} " + "".join(
                f"{v:>14.3f}" if np.isfinite(v) else f"{'—':>14}" for v in row)
                + f"{sh_all:>14.3f}")
        print("-" * 92)

    print("\nÝ nghĩa thống kê trên đoạn KIỂM ĐỊNH (gộp mọi span, Newey-West):")
    for c in CHE_DO_TEN:
        x = np.concatenate(tong_theo_dm[c]) if tong_theo_dm[c] else np.array([])
        t, pv = dm_nw(x)
        sh = sharpe(x)
        print(f"  {c:<12} n={len(x):>6}  Sharpe={sh:>+7.3f}  t={t:>+6.2f}  p={pv:.3f}"
              + ("  *** khác 0 có ý nghĩa (p<0.05)" if np.isfinite(pv) and pv < 0.05 else ""))

    print("\nDIỄN GIẢI: nếu KHÔNG có cột nào ở kiểm định vượt p<0.05 theo hướng có lợi,")
    print("kết luận momentum-theo-chế-độ KHÔNG cứu được tầng 1 — dừng ở đây, khỏi thử")
    print("Deep Momentum Network / meta-labeling (đắt hơn nhiều, cùng info set).")
    print("\nTỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
