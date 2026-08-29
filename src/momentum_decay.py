"""MOMENTUM NGOAI HOI CO SUY GIAM KHONG? — do tren 55 nam.

Vi sao co file nay: ban truoc cua luan van viet "Sharpe gop 0,27 -> rong 0,05
sau chi phi", ham y CHI PHI la thu giet loi the. Sai. Con so 0,27 mo ta mau
FRED 55 nam gop lai; tren 2010-2025 — dung khoang thoi gian he thong van hanh
— momentum am NGAY TRUOC khi tru chi phi.

Ket qua (Sharpe GOP, trung binh 4 span x 6 cap):
    1971-1985   +1,05
    1986-2000   +0,50
    2001-2009   -0,08
    2010-2025   -0,16
Suy giam don dieu qua ca bon giai doan va ca bon span, khong ngoai le.

Day la phan ra alpha — hien tuong duoc ghi nhan rong rai trong tai lieu tai
chinh. Ket luan "khong co loi the huong di" vi the manh hon: no co CO CHE,
khong chi la mot phep kiem dinh null.

Chay:  python src/momentum_decay.py
"""
import os
import numpy as np
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
FRED = os.path.join(os.path.dirname(DIR), "data", "fred")

# quy uoc yet gia: USD tren 1 don vi ngoai te (giong Notebook 01 cua HuyH)
META = {"DEXUSEU": ("EUR", "KEEP"), "DEXUSUK": ("GBP", "KEEP"),
        "DEXUSAL": ("AUD", "KEEP"), "DEXCAUS": ("CAD", "INVERT"),
        "DEXJPUS": ("JPY", "INVERT"), "DEXSZUS": ("CHF", "INVERT")}
PERIODS = [(1971, 1985), (1986, 2000), (2001, 2009), (2010, 2025)]
SPANS = (5, 10, 20, 60)


def load(code):
    d = pd.read_csv(os.path.join(FRED, f"{code}.csv"))
    d = d.rename(columns={"observation_date": "DATE"})
    d["DATE"] = pd.to_datetime(d.DATE, errors="coerce")
    d[code] = pd.to_numeric(d[code], errors="coerce")
    d = d.dropna().sort_values("DATE")
    cur, act = META[code]
    px = d[code] if act == "KEEP" else 1.0 / d[code]
    o = pd.DataFrame({"DATE": d.DATE.values, "px": px.values})
    o["r"] = np.log(o.px / o.px.shift(1))
    return cur, o.dropna().reset_index(drop=True)


def sharpe_gross(df, span):
    """Quy tac dau cua EWMA. Chi dung thong tin den t-1 (da shift)."""
    pos = np.sign(df.r.shift(1).ewm(span=span, min_periods=span).mean()).fillna(0)
    x = (pos * df.r).dropna()
    return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan


def main():
    data = dict(load(c) for c in META)
    print("=" * 66)
    print("SHARPE GOP (chua tru chi phi), trung binh 6 cap")
    print("=" * 66)
    print(f"{'Giai doan':<14}" + "".join(f"{'span '+str(s):>9}" for s in SPANS) + f"{'TB':>9}")
    print("-" * 66)
    means = []
    for lo, hi in PERIODS:
        row = []
        for span in SPANS:
            v = [sharpe_gross(df[(df.DATE.dt.year >= lo) & (df.DATE.dt.year <= hi)]
                              .reset_index(drop=True), span)
                 for df in data.values()
                 if len(df[(df.DATE.dt.year >= lo) & (df.DATE.dt.year <= hi)]) >= 300]
            row.append(np.mean(v) if v else np.nan)
        means.append(np.nanmean(row))
        print(f"{lo}-{hi:<9}" + "".join(f"{v:>9.3f}" for v in row) + f"{means[-1]:>9.3f}")
    print("-" * 66)
    print(f"\n{PERIODS[0][0]}-{PERIODS[0][1]}: {means[0]:+.3f}   ->   "
          f"{PERIODS[-1][0]}-{PERIODS[-1][1]}: {means[-1]:+.3f}   "
          f"(chenh {means[0]-means[-1]:+.3f})")

    mono = all(means[i] > means[i + 1] for i in range(len(means) - 1))
    print(f"Suy giam don dieu qua moi giai doan: {'CO' if mono else 'KHONG'}")
    assert mono, "ky vong suy giam don dieu — neu that bai thi da co gi doi"
    assert means[0] > 0.8 and means[-1] < 0, "bien do ky vong da doi"
    print("\nTU KIEM DAT")


if __name__ == "__main__":
    main()
