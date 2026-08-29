"""Danh gia thi nghiem 2: chat luong mat do du bao va tang rui ro duoi."""
import numpy as np, pandas as pd, json, sys
sys.path.insert(0,"/tmp/fx/src")
from scipy import stats
from metrics import fz0, kupiec, christoffersen_ind, dq_test, mcs, pinball
from fxdata import PAIRS
DISTS = ["Chuan","Student-t","FHS","FHS+EVT"]
d = json.load(open("/tmp/fx/exp2.json"))

print("="*110)
print("A. CRPS TRUNG BINH (x10^4, don vi pip tren loi suat log; thap hon = tot hon)")
print("="*110)
print(f"{'Cap':<9}{'n':>7}" + "".join(f"{x:>14}" for x in DISTS) + f"{'  tot nhat':>14}")
print("-"*110)
Lc = {k: [] for k in DISTS}
for p in PAIRS:
    y = np.array(d[p]["y"]); row=[]
    for k in DISTS:
        c = np.array(d[p]["rec"][k]["crps"]); Lc[k].append(c); row.append(c.mean()*1e4)
    b = DISTS[int(np.argmin(row))]
    print(f"{p:<9}{len(y):>7}" + "".join(f"{v:>14.4f}" for v in row) + f"{b:>14}")
print("-"*110)
CR = {k: np.concatenate(Lc[k]) for k in DISTS}
print(f"{'GOP':<9}{len(CR[DISTS[0]]):>7}" + "".join(f"{CR[k].mean()*1e4:>14.4f}" for k in DISTS))
base = CR["Chuan"].mean()
print(f"{'vs Chuan':<9}{'':>7}" + "".join(f"{100*(CR[k].mean()/base-1):>13.2f}%" for k in DISTS))
M = np.column_stack([CR[k] for k in DISTS])
keep, elim = mcs(M, alpha=0.10, B=1000, block=20, seed=3)
print(f"\nModel Confidence Set 90% tren CRPS gop: {[DISTS[i] for i in keep]}")
for i,(m,pv) in enumerate(elim,1): print(f"   loai {i}. {DISTS[m]:<12} p = {pv:.3f}")

print("\n" + "="*110)
print("B. HIEU CHINH — kiem dinh PIT (neu mat do dung thi PIT ~ U(0,1) doc lap)")
print("="*110)
print(f"{'Phan phoi':<14}{'KS stat':>10}{'KS p':>9}{'do phu 50%':>13}{'do phu 90%':>13}"
      f"{'do phu 98%':>13}{'|lech| TB':>12}")
print("-"*110)
for k in DISTS:
    pit = np.concatenate([np.array(d[p]["rec"][k]["pit"]) for p in PAIRS])
    ks = stats.kstest(pit, "uniform")
    c50 = np.mean((pit>0.25)&(pit<0.75)); c90 = np.mean((pit>0.05)&(pit<0.95))
    c98 = np.mean((pit>0.01)&(pit<0.99))
    dev = np.mean([abs(c50-0.50), abs(c90-0.90), abs(c98-0.98)])
    print(f"{k:<14}{ks.statistic:>10.4f}{ks.pvalue:>9.3f}{c50:>12.1%}{c90:>13.1%}{c98:>13.1%}{dev:>11.2%}")
print("-"*110)
print("Do phu danh nghia: 50% / 90% / 98%. Lech duong = phan phoi qua RONG, am = qua HEP.")

print("\n" + "="*110)
print("C. TANG RUI RO DUOI — VaR va ES, backtest day du")
print("="*110)
for a in (0.025, 0.01):
    print(f"\n  ── Muc {1-a:.1%} (alpha = {a}) " + "─"*70)
    print(f"  {'Phan phoi':<12}{'ty le vi pham':>15}{'ky vong':>9}{'Kupiec p':>11}"
          f"{'Chris.ind p':>13}{'DQ p':>9}{'FZ0':>11}{'ket luan':>22}")
    print("  " + "-"*104)
    fz_all = {}
    for k in DISTS:
        y  = np.concatenate([np.array(d[p]["y"]) for p in PAIRS])
        v  = np.concatenate([np.array(d[p]["rec"][k]["var"][str(a)]) for p in PAIRS])
        e  = np.concatenate([np.array(d[p]["rec"][k]["es"][str(a)])  for p in PAIRS])
        hits = (y <= v).astype(int)
        _, pk, ph = kupiec(hits, a)
        _, pi_ = christoffersen_ind(hits)
        _, pdq = dq_test(hits, v, a)
        f = fz0(y, v, np.minimum(e, v-1e-9), a); fz_all[k] = f
        verdict = []
        if pk is not np.nan and pk < 0.05: verdict.append("do phu SAI")
        if pi_ is not np.nan and pi_ < 0.05: verdict.append("vi pham TU CUM")
        if pdq is not np.nan and pdq < 0.05: verdict.append("DQ truot")
        vtxt = ", ".join(verdict) if verdict else "dat het"
        print(f"  {k:<12}{ph:>14.2%}{a:>9.1%}{pk:>11.3f}{pi_:>13.3f}{pdq:>9.3f}"
              f"{np.mean(f):>11.4f}{vtxt:>22}")
    print("  " + "-"*104)
    Mf = np.column_stack([fz_all[k] for k in DISTS])
    kp, _ = mcs(Mf, alpha=0.10, B=800, block=20, seed=5)
    print(f"  MCS 90% tren FZ0: {[DISTS[i] for i in kp]}")

print("\n" + "="*110)
print("D. PINBALL LOSS O HAI MUC DUOI + HANH VI TRONG NHUNG NGAY TE NHAT")
print("="*110)
y  = np.concatenate([np.array(d[p]["y"]) for p in PAIRS])
print(f"{'Phan phoi':<14}{'pinball 1%':>13}{'pinball 2,5%':>15}{'ES du bao TB':>15}"
      f"{'ES thuc te TB':>16}{'ty le ES/thuc':>15}")
print("-"*110)
for k in DISTS:
    v1 = np.concatenate([np.array(d[p]["rec"][k]["var"]["0.01"]) for p in PAIRS])
    v2 = np.concatenate([np.array(d[p]["rec"][k]["var"]["0.025"]) for p in PAIRS])
    e1 = np.concatenate([np.array(d[p]["rec"][k]["es"]["0.01"]) for p in PAIRS])
    pb1 = np.mean(pinball(y, v1, 0.01))*1e4
    pb2 = np.mean(pinball(y, v2, 0.025))*1e4
    hit = y <= v1
    es_pred = e1[hit].mean()*1e4          # ES du bao trong nhung ngay co vi pham
    es_real = y[hit].mean()*1e4           # ton that thuc te trung binh trong nhung ngay do
    print(f"{k:<14}{pb1:>13.4f}{pb2:>15.4f}{es_pred:>15.2f}{es_real:>16.2f}{es_pred/es_real:>15.3f}")
print("-"*110)
print("Ba cot cuoi la phep kiem ES quan trong nhat: trong CHINH nhung ngay VaR 1% bi vuot,")
print("ton that trung binh THUC TE co bang muc ES da du bao khong? Ty le = 1,00 la hoan hao;")
print("< 1,00 nghia la mo hinh DANH GIA THAP muc lo khi bien co duoi xay ra (nguy hiem).")

print("\n" + "="*110)
print("E. TOM TAT")
print("="*110)
print("Cot moc so sanh: neu bo MAE/RMSE va thay bang CRPS + PIT + FZ0 + DQ thi ta biet")
print("them dieu gi? Bang A cho biet mat do NAO sac hon; bang B cho biet no co TRUNG THUC")
print("khong; bang C cho biet duoi co dung khong VA cac vi pham co tu cum khong.")
print("MAE/RMSE tren loi suat khong tra loi duoc bat ky cau nao trong ba cau tren.")
