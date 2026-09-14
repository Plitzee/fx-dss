"""PHA 3B — DYNOTEARS (Pamfil et al. 2020, "Structure Learning from
Time-Series Data") de tim DAG nhan qua DONG THOI + TRE 1 phien giua 6 cap
tien te (bien dong log-RV) va 3 ung vien ngoai sinh manh nhat cua Level 2/3
(VIXCLS, GVZCLS, DFII10).

VI SAO TU VIET (khong dung `causalnex`, thu vien tham chieu cua DYNOTEARS):
`causalnex` moi nhat (0.12.2) gioi han Requires-Python <3.11; moi truong nay
chay Python 3.11.9 nen KHONG the `pip install causalnex` (kiem bang `pip
install causalnex --dry-run` truoc khi viet file nay — bao "Ignored ...
Requires-Python ... <3.11" cho MOI ban phat hanh). Ham `_hoc_dynotears`
duoi day cai dat LAI dung cong thuc cua bai bao goc: NOTEARS (Zheng 2018)
tren khong gian [X_t | X_{t-1}], rang buoc phi chu trinh (acyclicity,
tr(exp(W∘W))=d) CHI ap len khoi DONG THOI W — khoi TRE A1 khong can rang
buoc vi thu tu thoi gian tu no da dam bao phi chu trinh.

DIEU MA PCMCI(+) O DAY KHONG LAM DUOC: `pha3b_pcmci_doclap.py` (tau_min=1)
va `pha3b_pcmci_plus.py` (tau_min=0) deu kiem TUNG CAP tien te RIENG LE
(bien do 1 cap tai 1 thoi diem, dieu kien tren m_har + 3 ngoai sinh). DAG o
day dung CHUNG ca 6 cap trong MOT mo hinh, nen co the phat hien LAN TRUYEN
bien dong DONG THOI GIUA CAC CAP (vd AUDUSD <-> USDCAD, hai dong hang hoa)
ma khung PCMCI don-cap khong nhin thay.

KHONG mo lai quyet dinh da chot cua Pha3B — day la MOT KIEM TRA THEM, ket
qua khong anh huong toi tap E3 hay quyet dinh khong tich hop nhan qua vao du
bao (xem docs/TONG_QUAN_CHO_AI_KHAC.md muc 4.5). Trong so W/A1 la he so
tuyen tinh tren du lieu DA CHUAN HOA (tu thong ke doan huan luyen), nen
lambda/nguong o day KHONG so sanh truc tiep duoc voi r_MCI cua PCMCI.

Chay:  python src/dynotears_tuviet.py
Ghi:   output/dynotears_tuviet.json
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import pha3b_dactrung as DT                                    # noqa: E402
import volfc2 as V2                                             # noqa: E402
from split import doan                                          # noqa: E402

EPS = 1e-12
BIEN_NGOAI_SINH = ["VIXCLS_lv", "GVZCLS_lv", "DFII10_lv"]
LAMBDA_W = 0.05      # phat L1 tren canh dong thoi (khoi W)
LAMBDA_A = 0.05      # phat L1 tren canh tre 1 phien (khoi A1)
W_NGUONG = 0.05      # nguong cat sau toi uu, tren he so cua du lieu da chuan hoa


# ─────────────────────────────────────────────────── du lieu (9 bien x t)
def dung_bang():
    """Bang rong: hang = phien chung 6 cap, cot = log-RV moi cap + 3 ngoai sinh.

    Chi dung doan huan luyen + kiem dinh (giong cac ban PCMCI); chuan hoa
    tung cot bang trung binh/do lech cua RIENG doan huan luyen.
    """
    bang, chung = V2.nap_bang()
    lrv = {}
    for p in V2.PAIRS:
        d = bang[p].set_index("Date").reindex(chung)
        lrv[p] = np.log(np.maximum(d.rv5.values, EPS))
    tt = DT.bien_doi_thi_truong().reindex(chung)

    cot_cap = [f"{p}_lrv" for p in V2.PAIRS]
    du_lieu = {f"{p}_lrv": lrv[p] for p in V2.PAIRS}
    du_lieu.update({c: tt[c].values for c in BIEN_NGOAI_SINH})
    df = pd.DataFrame(du_lieu, index=chung)

    g = doan(chung.values)
    df = df[g < 2]                                     # bo doan kiem tra
    tr = doan(df.index.values) == 0
    mu, sd = df[tr].mean(), df[tr].std().replace(0, np.nan)
    df = ((df - mu) / sd).dropna()
    return df, cot_cap + BIEN_NGOAI_SINH


# ───────────────────────────────────────────────────── loi NOTEARS/DYNOTEARS
def _h_va_dao_ham(W):
    """h(W) = tr(exp(W∘W)) - d — 0 khi va chi khi W phi chu trinh (Zheng 2018)."""
    E = expm(W * W)
    h = np.trace(E) - W.shape[0]
    G_h = E.T * W * 2
    return h, G_h


def _hoc_dynotears(X, Y1, lambda_w, lambda_a, h_tol=1e-8, rho_max=1e16, max_vong=100):
    """X: (n,d) hien tai. Y1: (n,d) tre 1 phien. Tra ve (W, A1), moi ma (d,d).

    Tham so hoa W=Wp-Wn, A1=Ap-An voi Wp,Wn,Ap,An>=0 (chuan NOTEARS goc) de
    phat L1 tro thanh mot ham tuyen tinh muot — L-BFGS-B toi uu truc tiep,
    khong can subgradient. Duong cheo cua W bi khoa = 0 (khong tu-canh).
    Vong ngoai la dual ascent tren rho/alpha, giong het NOTEARS goc.
    """
    n, d = X.shape

    def goi(w):
        return (w[0:d * d].reshape(d, d), w[d * d:2 * d * d].reshape(d, d),
                w[2 * d * d:3 * d * d].reshape(d, d), w[3 * d * d:4 * d * d].reshape(d, d))

    def muctieu(w, rho, alpha):
        Wp, Wn, Ap, An = goi(w)
        W, A = Wp - Wn, Ap - An
        R = X - X @ W - Y1 @ A
        loss = 0.5 / n * (R ** 2).sum()
        G_loss_W = -(1.0 / n) * X.T @ R
        G_loss_A = -(1.0 / n) * Y1.T @ R
        h, G_h = _h_va_dao_ham(W)
        obj = (loss + 0.5 * rho * h * h + alpha * h
               + lambda_w * (Wp + Wn).sum() + lambda_a * (Ap + An).sum())
        G_W = G_loss_W + (rho * h + alpha) * G_h
        g = np.concatenate([
            (G_W + lambda_w).flatten(), (-G_W + lambda_w).flatten(),
            (G_loss_A + lambda_a).flatten(), (-G_loss_A + lambda_a).flatten()])
        return obj, g

    bounds = []
    for bloc in range(4):
        for i in range(d):
            for j in range(d):
                bounds.append((0, 0) if (bloc < 2 and i == j) else (0, None))

    w = np.zeros(4 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_vong):
        while rho < rho_max:
            sol = minimize(muctieu, w, args=(rho, alpha), method="L-BFGS-B",
                           jac=True, bounds=bounds)
            w_moi = sol.x
            Wp, Wn, _, _ = goi(w_moi)
            h_moi, _ = _h_va_dao_ham(Wp - Wn)
            if h_moi > 0.25 * h:
                rho *= 10
            else:
                break
        w, h = w_moi, h_moi
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    Wp, Wn, Ap, An = goi(w)
    return Wp - Wn, Ap - An


# ────────────────────────────────────────────────────────────────── chay
def chay():
    print("Đang dựng bảng 9 biến (6 cặp log-RV + 3 ngoại sinh)...")
    df, ten_bien = dung_bang()
    d = len(ten_bien)
    X = df[ten_bien].values[1:]
    Y1 = df[ten_bien].values[:-1]
    print(f"n={len(X)} phiên chung (huấn luyện+kiểm định) · d={d} biến: "
          f"{', '.join(ten_bien)}")

    W, A1 = _hoc_dynotears(X, Y1, LAMBDA_W, LAMBDA_A)
    h_cuoi, _ = _h_va_dao_ham(W)
    print(f"h(W) cuối (phải ~0 nếu phi chu trình đạt) = {h_cuoi:.2e}")

    canh_dong_thoi, canh_tre = [], []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if abs(W[i, j]) >= W_NGUONG:
                canh_dong_thoi.append(dict(tu=ten_bien[i], den=ten_bien[j],
                                           trong_so=round(float(W[i, j]), 4)))
            if abs(A1[i, j]) >= W_NGUONG:
                canh_tre.append(dict(tu=ten_bien[i], den=ten_bien[j],
                                     trong_so=round(float(A1[i, j]), 4)))

    canh_dong_thoi.sort(key=lambda x: -abs(x["trong_so"]))
    canh_tre.sort(key=lambda x: -abs(x["trong_so"]))

    print(f"\nCạnh ĐỒNG THỜI sống sót (|W|>={W_NGUONG}): {len(canh_dong_thoi)}")
    for c in canh_dong_thoi:
        print(f"  {c['tu']:>12s} -> {c['den']:<12s}  w={c['trong_so']:+.4f}")
    print(f"\nCạnh TRỄ 1 phiên sống sót (|A1|>={W_NGUONG}): {len(canh_tre)}")
    for c in canh_tre:
        print(f"  {c['tu']:>12s} -> {c['den']:<12s}  w={c['trong_so']:+.4f}")

    ra = dict(n=len(X), bien=ten_bien, lambda_w=LAMBDA_W, lambda_a=LAMBDA_A,
             nguong=W_NGUONG, h_cuoi=float(h_cuoi),
             canh_dong_thoi=canh_dong_thoi, canh_tre_1phien=canh_tre)
    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "dynotears_tuviet.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("\nĐã ghi", outp)
    return ra


if __name__ == "__main__":
    chay()
