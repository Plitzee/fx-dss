"""B1 — Hansen SPA, ap cho ho mo hinh Giai doan 1 vs nen "chi sigma^".

BOI CANH. `docs/REPLAN_2026.md` muc 10.4 dat tieu chi dung cho Giai doan 2:
tien de khai pha quy luat bi coi la khong dung vung neu CA BA dieu sau dung,
dieu dau tien la "ca nam ho deu khong bac bo duoc Hansen SPA so nen chi
sigma^". `src/metrics.py` truoc day chi co MCS (Hansen-Lunde-Nason 2011),
khong co SPA (Hansen 2005) — nen tieu chi nay CHUA DONG duoc.

PHAM VI TRUNG THUC CUA FILE NAY. Kien truc "nam ho" H1-H5 mo ta trong
REPLAN_2026.md muc 3.1 (SAX, motif, rule-list, tran GBM, che do) phan lon
CHUA TON TAI thanh code doc lap — thu muc `rules/mining/` khong co trong repo;
H2 (motif) va H3 (rule-list) chua duoc viet; H5 (che do) moi lam mot nua qua
`run_corr_regime.py`/`run_momentum_regime.py`. Viet du ca nam ho la mot khoi
luong cong viec khac han "cai them mot phep kiem" — khong lam gia trong file
nay.

Thay vao do, file nay DONG DUOC mot phan cua B1: cai dat `spa_test()` (xem
metrics.py) va AP DUNG NGAY cho du lieu da co san — ho mo hinh bien dong cua
CHINH Giai doan 1 (`run_balop.py`), tat ca deu la ung vien THAT su da khop/du
bao dung giao thuc, khong phai gia lap. Cau hoi: co ung vien nao trong day
THANG duoc nen "chi sigma^" (chinh nen ma REPLAN dinh nghia la muc phai vuot)
mot cach co y nghia, dong thoi kiem soat cho viec thu NHIEU ung vien cung luc?

Day KHONG PHAI cau tra loi day du cho tieu chi 10.4 (thieu H2, H3, H5 day
du) — ghi ro trong output va trong CHISO_DANHGIA.md rang B1 moi dong MOT
phan, khong phai toan bo.

Chay:  python src/run_spa.py
Ghi:   output/spa.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                            # noqa: E402
from run_balop import chuan_bi, KHOI                          # noqa: E402
from metrics import spa_test                                  # noqa: E402

NEN = "chỉ σ̂"          # nen ma REPLAN_2026.md dinh nghia la muc phai vuot
EPS = 1e-12


def loss_log(P, y):
    """Ton that log tung phien: -log P[i, y_i]. Dung lam input cho SPA."""
    P = np.asarray(P, float)
    y = np.asarray(y, int)
    return -np.log(np.maximum(P[np.arange(len(y)), y], EPS))


def main():
    print("=" * 100)
    print("B1 (một phần) — Hansen SPA: họ mô hình Giai đoạn 1 so nền \"chỉ σ̂\"")
    print("=" * 100)
    print("PHẠM VI: chỉ họ mô hình biến động của run_balop.py — KHÔNG bao phủ")
    print("H2 (motif), H3 (rule-list), H5 đầy đủ (chưa có code). Xem docstring.")

    ket = {}
    for h in B.HS:
        dat, _ = chuan_bi(h)
        print(f"\n{'─'*100}\nTẦM HẠN h = {h} phiên")
        for mt in ("P", "R"):
            y = dat[mt]["y"]
            cap = dat[mt]["cap"]
            ten_cac = [t for t in dat[mt]["P"] if t != NEN]
            L_nen = loss_log(dat[mt]["P"][NEN], y)
            L_cac = np.column_stack([loss_log(dat[mt]["P"][t], y) for t in ten_cac])
            p, T_SPA = spa_test(L_nen, L_cac, B=1000, block=KHOI[h], seed=7, nhom=cap)
            print(f"  mục tiêu {mt}: n={len(y):,}  T_SPA={T_SPA:.4f}  p={p:.4f}"
                  f"  {'BÁC BỎ H0 (có ứng viên thắng nền)' if p < 0.05 else 'KHÔNG bác bỏ H0'}")
            # dong gop tung ung vien (dbar > 0 la huong dung, khong hieu chinh boi)
            dbar = {t: float(np.mean(L_nen - loss_log(dat[mt]["P"][t], y)))
                    for t in ten_cac}
            tot = max(dbar, key=dbar.get)
            print(f"    ứng viên tốt nhất (thô, chưa hiệu chỉnh bội): "
                  f"{tot}  Δlog={dbar[tot]:+.5f}")
            ket[f"h{h}_{mt}"] = dict(nen=NEN, ung_vien=ten_cac, p=p, T_SPA=T_SPA,
                                     bac_bo=bool(p < 0.05), dbar=dbar)

    print("\n" + "=" * 100)
    print("TỔNG KẾT — bác bỏ H0 (⇔ có ứng viên thắng nền, kiểm soát thử nhiều lần)")
    print(f"  {'h/mục tiêu':<14}{'p-value':>10}{'bác bỏ ở α=0,05':>18}")
    for k, v in ket.items():
        print(f"  {k:<14}{v['p']:>10.4f}{'CÓ' if v['bac_bo'] else 'không':>18}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "spa.json"), "w", encoding="utf-8") as f:
        json.dump(ket, f, ensure_ascii=False, indent=1, default=float)
    print("\n→ output/spa.json")
    print("\nNHẮC LẠI PHẠM VI: kết quả này chỉ nói về họ mô hình run_balop.py.")
    print("Tiêu chí dừng đầy đủ ở REPLAN_2026.md mục 10.4 (\"cả năm họ\") CHƯA")
    print("đóng — H2/H3/H5 chưa có code để sinh chuỗi tổn thất tương ứng.")


if __name__ == "__main__":
    main()
