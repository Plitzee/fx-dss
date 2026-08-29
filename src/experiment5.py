"""
THI NGHIEM 5 — TANG QUYET DINH.
Cau hoi 1: cho truoc mot muc loi the, quy tac dinh co nao cho ket qua tot nhat?
Cau hoi 2 (quan trong hon): dieu gi xay ra khi nha dau tu UOC LUONG SAI loi the cua minh?
"""
import numpy as np, pandas as pd, sys, json
sys.path.insert(0, "/tmp/fx/src")
from sizing import simulate, RULES

P = pd.read_csv("/tmp/fx/exp3_panel.csv", parse_dates=["Date"])
EU = P[P.pair == "EURUSD"].reset_index(drop=True)
sig_med = EU.sig.median()
SR_TARGET = {"khong co loi the": 0.0, "Sharpe 0,5/nam": 0.5, "Sharpe 1,0/nam": 1.0}


def mu_from_sharpe(sr, sig_daily):
    """Doi tu Sharpe nam sang drift ngay."""
    return sr * sig_daily * np.sqrt(250) / 250


print("=" * 116)
print("A. SAU QUY TAC DINH CO — khi nha dau tu uoc luong DUNG loi the cua minh")
print("=" * 116)
print(f"EURUSD, sigma ngay trung vi = {sig_med:.3%}. Chan troi 250 phien. 20 000 duong mo phong.")
print("Chi phi 1 pip khu hoi moi lan tai can bang (5 phien/lan). Tran don bay 30:1. Stop-out o 50% ky quy.")

rows = []
for lab, sr in SR_TARGET.items():
    mu = mu_from_sharpe(sr, sig_med)
    print(f"\n  ── {lab}  (drift = {mu:.5%}/phien) " + "─" * 60)
    print(f"  {'Quy tac':<22}{'don bay TB':>10}{'tang truong/nam':>18}{'von trung vi':>14}"
          f"{'P(pha san)':>12}{'P(lo)':>9}{'MaxDD TB':>11}{'Ulcer':>9}{'duoi nuoc':>11}")
    print("  " + "-" * 110)
    for name, rule in RULES.items():
        r = simulate(EU, rule, n_paths=20000, horizon=250, mu_true=mu, seed=42)
        rows.append(dict(scenario=lab, rule=name, **r))
        print(f"  {name:<22}{r['avg_lev']:>10.1f}{r['mean_log_growth']:>17.1%}"
              f"{r['median_eq']:>14.3f}{r['p_ruin']:>11.1%}{r['p_loss']:>9.1%}"
              f"{r['maxdd']:>11.1%}{r['ulcer']:>9.3f}{r['time_under']:>11.0%}")
    print("  " + "-" * 110)

print("\n\n" + "=" * 116)
print("B. PHEP THU QUAN TRONG NHAT — nha dau tu TIN minh gioi hon thuc te")
print("=" * 116)
print("Loi the THAT co dinh o Sharpe 0,5/nam. Thay doi muc nha dau tu TIN minh co.")
print("Kelly nhay cam voi sai so uoc luong mu; cac quy tac dua tren RUI RO thi khong.")
print(f"\n{'Ty le tin/that':>16}" + "".join(f"{n:>19}" for n in RULES))
print("-" * 116)
mu_true = mu_from_sharpe(0.5, sig_med)
sens = []
for ratio in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
    cells = []
    for name, rule in RULES.items():
        r = simulate(EU, rule, n_paths=20000, horizon=250,
                     mu_true=mu_true, mu_believed=mu_true * ratio, seed=7)
        sens.append(dict(ratio=ratio, rule=name, **r))
        cells.append(f"{r['mean_log_growth']:+.0%} / {r['p_ruin']:.0%}")
    print(f"{ratio:>15.1f}x" + "".join(f"{c:>19}" for c in cells))
print("-" * 116)
print("Moi o la: tang truong log/nam  /  xac suat pha san trong 250 phien.")

print("\n\n" + "=" * 116)
print("C. CUNG QUY TAC, KHAC CAP TIEN  (loi the that = Sharpe 0,5/nam)")
print("=" * 116)
print(f"{'Cap':<9}{'sigma ngay':>12}" + "".join(f"{n:>19}" for n in
      ["Cố định 2%", "Kelly 1/2", "Ràng buộc CVaR", "Kelly + trần rủi ro"]))
print("-" * 116)
pair_rows = []
for p in ["EURUSD", "USDJPY", "GBPUSD", "AUDJPY", "XAUUSD"]:
    sub = P[P.pair == p].reset_index(drop=True)
    sm = sub.sig.median()
    mu = mu_from_sharpe(0.5, sm)
    cells = []
    for name in ["Cố định 2%", "Kelly 1/2", "Ràng buộc CVaR", "Kelly + trần rủi ro"]:
        r = simulate(sub, RULES[name], n_paths=12000, horizon=250, mu_true=mu, seed=11)
        pair_rows.append(dict(pair=p, rule=name, **r))
        cells.append(f"{r['mean_log_growth']:+.0%} / {r['p_ruin']:.0%}")
    print(f"{p:<9}{sm:>11.3%}" + "".join(f"{c:>19}" for c in cells))
print("-" * 116)
print("Moi o: tang truong log/nam / xac suat pha san. Chu y quy tac 'Co dinh 2%'")
print("cho ket qua RAT KHAC nhau giua cac cap vi no khong tinh den duoi phan phoi.")

json.dump({"main": rows, "sensitivity": sens, "by_pair": pair_rows},
          open("/tmp/fx/exp5.json", "w"), indent=1)
print("\nDa luu exp5.json")
