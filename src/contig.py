"""RAO CHAN LIEN MACH — cua so truot khong duoc bac qua lo hong du lieu.

Vi sao can: cua so truot lam viec theo CHI SO, khong theo NGAY. Neu chuoi
thieu 11 thang, "trung binh 20 phien gan nhat" se noi thang qua lo hong nhu
the chung lien nhau. Nhom da dinh dung loi nay o HAR-RV (751/1251 ngay co du
lieu, doan lien mach dai nhat 155 ngay, trong khi HAR can 22 ngay lien tiep).

Cach dung:
    run = run_length(dates, max_gap_days=4)
    if run[t] >= 20:  # du 20 phien lien mach tinh ca phien t
        ma20 = x[t-19:t+1].mean()
"""
import numpy as np
import pandas as pd


def run_length(dates, max_gap_days=4):
    """So phien lien mach tinh den t (ke ca t).

    Hai phien lien tiep duoc coi la lien mach neu cach nhau <= max_gap_days
    ngay lich. Mac dinh 4: cuoi tuan la 3 ngay, cuoi tuan co ngay le la 4.
    """
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    gap = d.diff().dt.days.values
    n = len(d)
    run = np.ones(n, dtype=int)
    for t in range(1, n):
        run[t] = run[t - 1] + 1 if gap[t] <= max_gap_days else 1
    return run


def guard(run, window):
    """Mat na boolean: True o nhung t co du `window` phien lien mach."""
    return run >= window


def report(dates, max_gap_days=4, label=""):
    """In chan doan lo hong. Tra ve (so lo hong, lo hong lon nhat)."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    gap = d.diff().dt.days
    big = gap[gap > max_gap_days]
    if len(big) == 0:
        print(f"  {label:<10} lien mach — khong co lo hong > {max_gap_days} ngay "
              f"(cach xa nhat: {int(gap.max()) if len(gap.dropna()) else 0} ngay)")
        return 0, 0
    i = big.idxmax()
    print(f"  {label:<10} {len(big)} lo hong > {max_gap_days} ngay | lon nhat "
          f"{int(big.max())} ngay tai {d[i].date()}")
    return len(big), int(big.max())
