"""TabPFN v2 — mo hinh NEN cho du lieu bang, do tren dung giao thuc cua repo.

VI SAO THU. `run_ml3.py` do LightGBM (2017) va GRU (2014) — ca hai deu thua nen
"chi sigma^". Cau hoi hop ly: hai mo hinh do da cu, SOTA 2026 co khac khong?

Tren TabArena 2026, LightGBM khong con dan dau o bat ky che do nao (hang trung
binh 5,16 — mo hinh CO DIEN tot nhat, khong phai mo hinh tot nhat). Thu thay
the no la TabPFN v2 (Hollmann et al., Nature 2025): mot transformer da huan
luyen truoc tren hang trieu bo du lieu bang TONG HOP, suy dien IN-CONTEXT —
dua ca tap huan luyen vao lam ngu canh, doc ra du bao trong MOT luot tien.

HAI LY DO NO DANG THU O DUNG DU AN NAY:

  1. KHONG CO SIEU THAM SO DE DO. Moi lua chon trong repo deu phai chot tren
     kiem dinh, va moi lan do tham so la mot lan tang nguy co overfit lua chon.
     TabPFN khong co gi de do — no chay zero-shot.
  2. VUNG DU LIEU NHO la cho no manh nhat, va bang nay (15.024 hang huan luyen)
     nam dung vung do.

BA DIEU KIEN BI VI PHAM — ghi ro thay vi lo di:

  a. GIOI HAN 10.000 HANG. TabPFN v2 thiet ke cho <= 10k hang; o day la 15.024,
     phai bat `ignore_pretraining_limits`. Chay NGOAI vung thiet ke.
  b. GIA DINH HOAN VI DUOC (i.i.d.). TabPFN coi cac hang la hoan vi duoc; chuoi
     thoi gian tai chinh co doi che do thi khong. Day la vi pham nghiem trong
     hon (a). Neu TabPFN THANG, phai soi lai xem co ro ri thoi gian khong; neu
     THUA, vi pham nay la mot loi giai thich hop ly.
  c. n_estimators = 1. May nay khong co GPU va 52 dac trung x 15.024 hang rat
     nang; ensemble 4 thanh vien se mat nhieu gio. Dung 1 la cho TabPFN dieu
     kien KEM HON mac dinh — neu no thang thi ket luan cang manh, con neu thua
     thi phai ghi nhan day la mot han che cua phep do, khong phai ket luan chac
     chan ve mo hinh.

PHAM VI: muc tieu P (thu hien tren giao dien), h = 1, cham tren KIEM DINH —
dung cho ma giao thuc quyet dinh. Chi khi no VUOT nguong moi chay tiep phan
kiem tra; khong mo kiem tra cho mot mo hinh da thua tren kiem dinh.

Chay:  python src/kiem_tabpfn.py
Ghi:   output/tabpfn.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                             # noqa: E402
import diem3 as D                                             # noqa: E402
from split import VALID_TU, TEST_TU                           # noqa: E402
from run_ml3 import nap, khi_hau_hoc, _chuan, SEED            # noqa: E402

N_EST = 1                  # xem ghi chu (c) o docstring
NGUONG_VUOT = 0.0105       # BSS cua nen "chi sigma^" tren kiem dinh, muc tieu P


def main():
    t0 = time.time()
    print("=" * 104)
    print("TabPFN v2 — mô hình NỀN cho dữ liệu bảng (Nature 2025)")
    print("=" * 104)

    X, yR, yP, phu, ten, pid, dts, hop_le = nap()
    tr = (dts < VALID_TU) & hop_le
    va = (dts >= VALID_TU) & (dts < TEST_TU) & hop_le
    y = yP
    ok = y >= 0
    Xtr, ytr = X[tr & ok], y[tr & ok]
    Xva, yva = X[va], y[va]
    cap_va = phu.pair.values[va]

    print(f"huấn luyện {len(ytr):,} hàng × {X.shape[1]} đặc trưng "
          f"· kiểm định {int(va.sum()):,}")
    print(f"NGOÀI vùng thiết kế của TabPFN (≤ 10.000 hàng) — bật "
          f"ignore_pretraining_limits, n_estimators={N_EST}")
    print(f"ngưỡng phải vượt: BSS > +{NGUONG_VUOT:.4f} (nền \"chỉ σ̂\")\n")

    from tabpfn import TabPFNClassifier
    print("đang khớp TabPFN… (không có GPU, dự kiến lâu)", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = TabPFNClassifier(n_estimators=N_EST, ignore_pretraining_limits=True,
                             random_state=SEED)
        m.fit(Xtr, ytr)
        P = _chuan(np.asarray(m.predict_proba(Xva), float))
    giay = time.time() - t0
    print(f"xong sau {giay:.0f}s\n")

    Pkh = khi_hau_hoc(ytr, int(va.sum()))
    r = D.bang(P, yva, Pkh, nhom=cap_va)
    lo, hi = D.bss_ktc(P, yva, Pkh, nhom=cap_va, nboot=300, khoi=20, seed=7)
    sd_p = float(np.std(P, axis=0).mean())

    print(f"  {'mô hình':<20}{'log':>9}{'BSS':>9}{'KTC 95% của BSS':>22}"
          f"{'ECE':>8}{'AUC':>8}{'sd(P)':>9}")
    print(f"  {'TabPFN v2':<20}{r['log']:>9.4f}{r['bss']:>+9.4f}"
          f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{r['ece']:>8.4f}{r['auc']:>8.4f}"
          f"{sd_p:>9.4f}")

    # doi chieu voi cac mo hinh da do truoc do, CUNG giao thuc
    cu = {}
    tep = os.path.join(OUT, "ml3.json")
    if os.path.exists(tep):
        with open(tep, encoding="utf-8") as f:
            cu = json.load(f).get("P", {}).get("kiem_dinh", {})
    for k, v in cu.items():
        if isinstance(v, dict):
            print(f"  {k:<20}{v.get('log',0):>9.4f}{v.get('bss',0):>+9.4f}"
                  f"{'':>22}{v.get('ece',0):>8.4f}{v.get('auc',0):>8.4f}"
                  f"{v.get('sd_p', float('nan')):>9.4f}")

    vuot = bool(r["bss"] > NGUONG_VUOT and np.isfinite(lo) and lo > 0)
    suy_bien = sd_p < 1e-3
    print()
    if suy_bien:
        print("  ⚠ CHỐT SUY BIẾN: sd(P) ≈ 0 — mô hình xuất gần như hằng số.")
    print(f"  → {'VƯỢT ngưỡng' if vuot else 'KHÔNG vượt ngưỡng'} "
          f"(BSS {r['bss']:+.4f} so với +{NGUONG_VUOT:.4f})")
    if not vuot:
        print("    Không mở đoạn kiểm tra cho một mô hình đã thua trên kiểm định.")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(n_estimators=N_EST, n_train=int(len(ytr)),
                   n_dac_trung=int(X.shape[1]), giay=giay, muc_tieu="P", h=1,
                   kiem_dinh=dict(**r, bss_lo=lo, bss_hi=hi, sd_p=sd_p),
                   nguong_vuot=NGUONG_VUOT, vuot=vuot),
              open(os.path.join(OUT, "tabpfn.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/tabpfn.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
