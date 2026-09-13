"""RL (REINFORCE) DE RA XAC SUAT HUONG — thu that, khong chi ly thuyet.

CAU HOI NGUOI DUNG DAT RA: "RL se hoc va dua ket qua chinh xac dua tren
nhung lan thuong va phat khi train" — ap dung cho viec RA BA XAC SUAT
(giam/di ngang/tang), khong phai dinh co lenh.

RL da thu TRONG REPO NAY chi cho DINH CO LENH (REINFORCE, PPO, CVaR-PPO —
xem docs/KHOA_SO.md muc 5, docs/SIZING_COMPARISON.md). Chua bao gio thu cho
RA XAC SUAT. File nay lam dung thu do — MOT PHEP KIEM CO KIEM SOAT, khong
phai suy luan.

──────────────────────────────────────────────────────────────────────────
VI SAO RL PHU HOP CHO DINH CO LENH NHUNG KHONG PHU HOP O DAY
──────────────────────────────────────────────────────────────────────────
Dinh co lenh la bai toan NHIEU BUOC NOI TIEP: hom nay dat don bay, ket cuc
anh huong VON, hanh dong ngay mai phu thuoc von con lai. Do la cau truc RL
that (Markov Decision Process voi trang thai lien tuc qua thoi gian).

Ra xac suat cho MOT phien la bai toan MOT BUOC, DOC LAP: dac trung hom nay ->
xac suat hom nay -> ket cuc hom nay BIET NGAY, khong anh huong gi den phien
sau. Khi do, gradient chinh xac cua ham mat mat (cross-entropy = log score)
DA TINH DUOC TRUC TIEP tu chinh nhan that (do la dieu logistic/LightGBM/GRU
da lam). REINFORCE thay gradient chinh xac do bang uoc luong qua LAY MAU
ngau nhien tu chinh sach — ve mat ly thuyet (dinh ly gradient chinh sach),
ky vong cua uoc luong REINFORCE HOI TU VE DUNG gradient sau nay, nhung
PHUONG SAI cao hon han vi khong dung nhan that truc tiep trong cong thuc dao
ham ma dung MOT hanh dong lay mau ngau nhien.

Noi cach khac: neu chinh sach la softmax tuyen tinh (giong het logistic hoi
quy da thuc), thi REINFORCE va supervised learning toi uu CUNG MOT ham muc
tieu, chi khac CACH uoc luong dao ham — mot ben chinh xac, mot ben nhieu.
Day la PHEP KIEM SO SANH CO KIEM SOAT: CUNG dang mo hinh (softmax tuyen
tinh), CUNG dac trung (52 cot cua run_ml3.py), CHI khac quy trinh huan
luyen.

DU DOAN CO THE KIEM CHUNG: REINFORCE hoi tu CHAM HON, dao dong NHIEU HON,
va KHONG VUOT duoc "logistic da thuc" da do (BSS +0,0077, log 1,0905,
AUC 0,5183) — vi ca hai cung toi uu mot ham, nhung logistic dung dao ham
chinh xac con REINFORCE chi uoc luong no.

Chay:  python src/kiem_rl_xacsuat.py
Ghi:   output/rl_xacsuat.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import diem3 as D                                             # noqa: E402
from run_ml3 import nap                                        # noqa: E402
from split import VALID_TU                                     # noqa: E402

SEED = 0
SO_EPOCH = 400
LR = 0.05
CO_LR = 0.9995          # giam dan toc do hoc — chuan RL, giup hoi tu on dinh
BATCH = 512


def chuan_hoa(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-8
    return (X - mu) / sd, mu, sd


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def logistic_giam_dan(Xtr, ytr, Xva, yva, rng, so_epoch=SO_EPOCH, lr=LR):
    """Doi chieu SUPERVISED — CUNG dang mo hinh (softmax tuyen tinh), CUNG
    dac trung, huan luyen bang GRADIENT CHINH XAC (khong lay mau). Day la
    thu REINFORCE se ky vong HOI TU VE, khong phai vuot qua."""
    n, f = Xtr.shape
    Xb = np.column_stack([np.ones(n), Xtr])
    theta = np.zeros((f + 1, 3))
    Y = np.zeros((n, 3)); Y[np.arange(n), ytr] = 1.0
    duong_cong = []
    for ep in range(so_epoch):
        idx = rng.permutation(n)
        for s in range(0, n, BATCH):
            b = idx[s:s + BATCH]
            P = softmax(Xb[b] @ theta)
            grad = Xb[b].T @ (Y[b] - P) / len(b)     # gradient CHINH XAC cua log-likelihood
            theta += lr * grad
        if ep % 20 == 0 or ep == so_epoch - 1:
            Pva = softmax(np.column_stack([np.ones(len(Xva)), Xva]) @ theta)
            duong_cong.append((ep, D.bss(Pva, yva, _khi_hau_hoc(yva, len(yva)))))
    Pva = softmax(np.column_stack([np.ones(len(Xva)), Xva]) @ theta)
    return theta, Pva, duong_cong


def reinforce(Xtr, ytr, Xva, yva, rng, so_epoch=SO_EPOCH, lr=LR):
    """CHINH SACH softmax tuyen tinh — CUNG dang mo hinh nhu logistic o tren.
    Huan luyen bang REINFORCE: LAY MAU hanh dong tu chinh sach, thuong 1 neu
    dung/0 neu sai, gradient uoc luong qua dinh ly gradient chinh sach voi
    duong nen (baseline) la trung binh truot cua thuong — dung ky thuat giam
    phuong sai chuan, khong phai REINFORCE tho nhat co the lam."""
    n, f = Xtr.shape
    Xb = np.column_stack([np.ones(n), Xtr])
    theta = np.zeros((f + 1, 3))
    duong_nen = 1.0 / 3.0            # khoi tao = ty le thuong ky vong ngau nhien
    duong_cong = []
    lr_hien = lr
    for ep in range(so_epoch):
        idx = rng.permutation(n)
        for s in range(0, n, BATCH):
            b = idx[s:s + BATCH]
            P = softmax(Xb[b] @ theta)
            # LAY MAU hanh dong tu CHINH SACH — day la buoc "thu" cua RL
            u = rng.random(len(b))
            cdf = np.cumsum(P, axis=1)
            a = (u[:, None] > cdf[:, :-1]).sum(1)     # hanh dong lay mau ~ P
            thuong = (a == ytr[b]).astype(float)      # 1 neu dung, 0 neu sai — THUONG/PHAT
            loi_the = thuong - duong_nen               # tru duong nen (giam phuong sai)
            duong_nen = 0.99 * duong_nen + 0.01 * thuong.mean()
            A = np.zeros((len(b), 3)); A[np.arange(len(b)), a] = 1.0
            # dinh ly gradient chinh sach: E[(thuong-nen) * d/dtheta log pi(a|s)]
            grad = Xb[b].T @ ((A - P) * loi_the[:, None]) / len(b)
            theta += lr_hien * grad
        lr_hien *= CO_LR
        if ep % 20 == 0 or ep == so_epoch - 1:
            Pva = softmax(np.column_stack([np.ones(len(Xva)), Xva]) @ theta)
            duong_cong.append((ep, D.bss(Pva, yva, _khi_hau_hoc(yva, len(yva)))))
    Pva = softmax(np.column_stack([np.ones(len(Xva)), Xva]) @ theta)
    return theta, Pva, duong_cong


def _khi_hau_hoc(y, n):
    c = np.bincount(y[y >= 0], minlength=3).astype(float)
    P = np.tile(c / c.sum(), (n, 1))
    return np.clip(P, 1e-9, None) / np.clip(P, 1e-9, None).sum(1, keepdims=True)


def main():
    t0 = time.time()
    print("=" * 104)
    print("RL (REINFORCE) ĐỂ RA XÁC SUẤT HƯỚNG — phép kiểm có kiểm soát")
    print("=" * 104)
    print("So sánh CÙNG dạng mô hình (softmax tuyến tính), CÙNG đặc trưng (52 cột),")
    print("CHỈ khác quy trình huấn luyện: gradient CHÍNH XÁC (supervised) so với")
    print("gradient ƯỚC LƯỢNG qua lấy mẫu thưởng/phạt (REINFORCE).\n")

    X, yR, yP, phu, ten, pid, dts, hop_le = nap()
    tr = (dts < VALID_TU) & hop_le
    from split import TEST_TU
    va = (dts >= VALID_TU) & (dts < TEST_TU) & hop_le
    y = yP
    ok = y >= 0
    Xtr_raw, ytr = X[tr & ok], y[tr & ok]
    Xva_raw, yva = X[va & (y >= 0)], y[va & (y >= 0)]
    Xtr, mu, sd = chuan_hoa(Xtr_raw)
    Xva, _, _ = chuan_hoa(Xva_raw, mu, sd)
    print(f"huấn luyện {len(ytr):,} hàng × {X.shape[1]} đặc trưng · kiểm định {len(yva):,}\n")

    rng1 = np.random.default_rng(SEED)
    print(f"[1/2] Huấn luyện SUPERVISED (gradient chính xác), {SO_EPOCH} epoch…", flush=True)
    _, P_sup, cc_sup = logistic_giam_dan(Xtr, ytr, Xva, yva, rng1)

    rng2 = np.random.default_rng(SEED)
    print(f"[2/2] Huấn luyện REINFORCE (thưởng/phạt qua lấy mẫu), {SO_EPOCH} epoch…", flush=True)
    _, P_rl, cc_rl = reinforce(Xtr, ytr, Xva, yva, rng2)

    Pkh = _khi_hau_hoc(yva, len(yva))
    r_sup = D.bang(P_sup, yva, Pkh)
    r_rl = D.bang(P_rl, yva, Pkh)
    lo_s, hi_s = D.bss_ktc(P_sup, yva, Pkh, nboot=300, khoi=20, seed=7)
    lo_r, hi_r = D.bss_ktc(P_rl, yva, Pkh, nboot=300, khoi=20, seed=7)

    print("\n" + "─" * 104)
    print(f"{'quy trình huấn luyện':<26}{'log':>9}{'BSS':>9}{'KTC 95% BSS':>20}"
          f"{'AUC':>8}{'sd(P)':>9}")
    print(f"{'SUPERVISED (logistic)':<26}{r_sup['log']:>9.4f}{r_sup['bss']:>+9.4f}"
          f"{f'[{lo_s:+.4f},{hi_s:+.4f}]':>20}{r_sup['auc']:>8.4f}"
          f"{np.std(P_sup,0).mean():>9.4f}")
    print(f"{'RL/REINFORCE':<26}{r_rl['log']:>9.4f}{r_rl['bss']:>+9.4f}"
          f"{f'[{lo_r:+.4f},{hi_r:+.4f}]':>20}{r_rl['auc']:>8.4f}"
          f"{np.std(P_rl,0).mean():>9.4f}")

    print("\nĐường cong hội tụ trên kiểm định (BSS mỗi 20 epoch):")
    print(f"  {'epoch':>7}{'BSS supervised':>18}{'BSS REINFORCE':>18}")
    for (e1, b1), (e2, b2) in zip(cc_sup, cc_rl):
        assert e1 == e2
        print(f"  {e1:>7}{b1:>+18.4f}{b2:>+18.4f}")

    bien_thien_sup = float(np.std([b for _, b in cc_sup[-5:]]))
    bien_thien_rl = float(np.std([b for _, b in cc_rl[-5:]]))
    print(f"\nbiến thiên BSS ở 5 điểm đo cuối (đo độ ồn của hội tụ):")
    print(f"  supervised {bien_thien_sup:.5f}  ·  REINFORCE {bien_thien_rl:.5f}"
          f"  → REINFORCE ồn hơn {bien_thien_rl/max(bien_thien_sup,1e-9):.1f}×")

    ket = dict(
        supervised=dict(**r_sup, bss_lo=lo_s, bss_hi=hi_s, duong_cong=cc_sup),
        reinforce=dict(**r_rl, bss_lo=lo_r, bss_hi=hi_r, duong_cong=cc_rl),
        bien_thien_sup=bien_thien_sup, bien_thien_rl=bien_thien_rl,
        so_epoch=SO_EPOCH, lr=LR)
    os.makedirs(OUT, exist_ok=True)
    json.dump(ket, open(os.path.join(OUT, "rl_xacsuat.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)

    print("\n" + "=" * 104)
    vuot = r_rl["bss"] > r_sup["bss"]
    print(f"→ REINFORCE {'VƯỢT' if vuot else 'KHÔNG vượt'} supervised cùng dạng mô hình"
          f" ({r_rl['bss']:+.4f} so với {r_sup['bss']:+.4f})")
    print(f"→ output/rl_xacsuat.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
