import pandas as pd, numpy as np
df = pd.read_csv("/tmp/fx/data/EURCHF_d1.csv", parse_dates=["Date"])
for c in ("open","high","low","close"): df[c] = df[c]/1e5
m = (df.Date >= "2014-12-20") & (df.Date <= "2015-02-20")
print("EURCHF quanh 15/01/2015 (nguon: mirror GitHub cua feed MT5):")
print(df[m].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print()
g = df.Date.diff().dt.days
big = df[g > 7]
print("Cac khoang trong > 7 ngay:")
for i in big.index:
    print(f"   tu {df.Date[i-1].date()} den {df.Date[i].date()}  = {int(g[i])} ngay")
print()
# so sanh: USDCHF co day du khong?
u = pd.read_csv("/tmp/fx/data/USDCHF_d1.csv", parse_dates=["Date"])
for c in ("open","high","low","close"): u[c] = u[c]/1e5
mu = (u.Date >= "2015-01-12") & (u.Date <= "2015-01-22")
print("USDCHF cung giai doan — CO du lieu:")
print(u[mu].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print()
# suy ra EURCHF tu EURUSD * USDCHF de kiem chung
e = pd.read_csv("/tmp/fx/data/EURUSD_d1.csv", parse_dates=["Date"])
for c in ("open","high","low","close"): e[c] = e[c]/1e5
j = e.merge(u, on="Date", suffixes=("_eu","_uc"))
j["cross_close"] = j.close_eu * j.close_uc
mj = (j.Date >= "2015-01-12") & (j.Date <= "2015-01-22")
print("EURCHF suy ra = EURUSD x USDCHF (kiem chung chao):")
print(j[mj][["Date","cross_close"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
