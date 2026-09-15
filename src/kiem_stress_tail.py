"""KIỂM THỬ ĐỘ VỮNG RỦI RO ĐUÔI DƯỚI DỊCH CHUYỂN PHÂN PHỐI
(Stress-Perturbed Validation Suite for Tail-Risk)

Mục đích: Giải quyết bế tắc phương pháp luận ở TONG_QUAN_CHO_AI_KHAC.md mục 5.1:
  "Làm sao kiểm chứng một phương pháp vá đuôi mà KHÔNG nhìn trước tập kiểm tra,
   khi lỗi chỉ bộc lộ ngoài mẫu?"

Cách tiếp cận:
  Tạo ra một bộ kiểm chuẩn nghịch đảo (Adversarial Stress Suite) NGAY TRÊN
  ĐOẠN KIỂM ĐỊNH (Validation, g==1):
  1. Stress 1: Scale drift nhân tạo (+12% phương sai của z, mô phỏng đúng mức
     trôi đo được giữa Train và Test của USDJPY).
  2. Stress 2: Tiêm sốc can thiệp tỷ giá kiểu BOJ (Poisson Jump Injection với
     |z| >= 3.0 sigma).
  3. Đo lường: Kupiec, Christoffersen, DQ cho V0 (phân vị tĩnh) vs ACI (gamma=0.01)
     vs Conformal PID.

Chạy: py -3.11 src/kiem_stress_tail.py
"""
import os
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import balop as B
from metrics import kupiec, christoffersen_ind, dq_test
from va_duoi import nap, cham
import conformal_risk as CR

ALPHA_MUC = 0.01
SEED = 42


def danh_gia_chuoi(z, sig, qz, ez, mask, a=0.01):
    """Tính các chỉ số Kupiec, Christoffersen, DQ trên vùng mask."""
    y = z[mask] * sig[mask]
    v = qz[mask] * sig[mask]
    e = ez[mask] * sig[mask]
    hits = (y <= v).astype(int)
    ph = float(np.mean(hits))
    _, pk, _ = kupiec(hits, a)
    _, pi_ = christoffersen_ind(hits)
    _, pdq = dq_test(hits, v, a)
    h = hits.astype(bool)
    tyle_es = float(np.mean(e[h]) / np.mean(y[h])) if h.any() else np.nan
    dat = ((pk is not None and np.isfinite(pk) and pk >= 0.05)
           and (pi_ is not None and np.isfinite(pi_) and pi_ >= 0.05)
           and (pdq is not None and np.isfinite(pdq) and pdq >= 0.05))
    return {
        "n": int(mask.sum()),
        "vi_pham": round(ph, 4),
        "kupiec_p": round(float(pk), 4) if pk is not None and np.isfinite(pk) else None,
        "chris_p": round(float(pi_), 4) if pi_ is not None and np.isfinite(pi_) else None,
        "dq_p": round(float(pdq), 4) if pdq is not None and np.isfinite(pdq) else None,
        "es_ratio": round(float(tyle_es), 4) if np.isfinite(tyle_es) else None,
        "dat": bool(dat)
    }


def chay_stress_suite():
    np.random.seed(SEED)
    print("=" * 95)
    print("STRESS-PERTURBED VALIDATION SUITE — ĐÁNH GIÁ ĐỘ BỀN VỮNG TRÊN KIỂM ĐỊNH (g==1)")
    print("=" * 95)

    D = nap()
    pairs = ["USDJPY", "USDCHF", "EURUSD"]  # 2 cặp trọng tâm + 1 cặp đối chứng

    for pair in pairs:
        print(f"\n>>> Cặp {pair} (Mức VaR 99%, alpha = 0.01) <<<")
        z_orig = D[pair]["z"].copy()
        sig = D[pair]["sig"]
        g = D[pair]["g"]
        mask_val = (g == 1) & np.isfinite(z_orig) & np.isfinite(sig)

        # 3 kịch bản:
        # Kịch bản 0: Chuỗi kiểm định nguyên bản
        # Kịch bản 1: Scale drift (+12% biên độ ở nửa sau kiểm định)
        # Kịch bản 2: Tiêm sốc can thiệp (4 phiên có |z| ~ 3.5 phân bố ngẫu nhiên)
        kich_ban = {"Gốc (Validation)": z_orig.copy()}

        z_drift = z_orig.copy()
        idx_val = np.where(mask_val)[0]
        n_half = len(idx_val) // 2
        z_drift[idx_val[n_half:]] *= 1.12  # +12% scale drift
        kich_ban["Scale Drift (+12%)"] = z_drift

        z_shock = z_drift.copy()
        shock_points = np.random.choice(idx_val[n_half:], size=4, replace=False)
        z_shock[shock_points] = -np.random.uniform(3.2, 4.2, size=4)
        kich_ban["Drift + Sốc can thiệp (BOJ Jump)"] = z_shock

        for ten_kb, z_cur in kich_ban.items():
            print(f"\n  --- Kịch bản: {ten_kb} ---")
            print(f"  {'Phương pháp':<22}{'Vi phạm':>9}{'Kupiec':>9}{'Chris':>8}{'DQ':>8}{'Tỷ lệ ES':>10}{'Đạt?':>6}")
            print("  " + "-" * 72)

            # 1. V0 tĩnh (ước trên huấn luyện)
            z_tr = z_cur[(g == 0) & np.isfinite(z_cur)]
            q_v0 = float(np.quantile(z_tr, ALPHA_MUC))
            e_v0 = float(np.mean(z_tr[z_tr <= q_v0])) if (z_tr <= q_v0).any() else q_v0
            qz_v0 = np.full(len(z_cur), q_v0)
            ez_v0 = np.full(len(z_cur), e_v0)
            res_v0 = danh_gia_chuoi(z_cur, sig, qz_v0, ez_v0, mask_val, ALPHA_MUC)
            print(f"  {'V0 (Tĩnh)':<22}{res_v0['vi_pham']:>9.4f}{str(res_v0['kupiec_p']):>9}"
                  f"{str(res_v0['chris_p']):>8}{str(res_v0['dq_p']):>8}"
                  f"{str(res_v0['es_ratio']):>10}{'YES' if res_v0['dat'] else 'NO':>6}")

            # 2. ACI (gamma = 0.01)
            qz_aci, ez_aci, _ = CR.chay_aci_duoi(z_cur, a_target=ALPHA_MUC, gamma=0.01, dam=750, cuon=500)
            res_aci = danh_gia_chuoi(z_cur, sig, qz_aci, ez_aci, mask_val, ALPHA_MUC)
            print(f"  {'ACI (gamma=0.01)':<22}{res_aci['vi_pham']:>9.4f}{str(res_aci['kupiec_p']):>9}"
                  f"{str(res_aci['chris_p']):>8}{str(res_aci['dq_p']):>8}"
                  f"{str(res_aci['es_ratio']):>10}{'YES' if res_aci['dat'] else 'NO':>6}")

            # 3. Conformal PID Control
            qz_pid, ez_pid, _ = CR.chay_pid_duoi(z_cur, a_target=ALPHA_MUC, kp=0.01, ki=0.001, kd=0.002, dam=750, cuon=500)
            res_pid = danh_gia_chuoi(z_cur, sig, qz_pid, ez_pid, mask_val, ALPHA_MUC)
            print(f"  {'Conformal PID':<22}{res_pid['vi_pham']:>9.4f}{str(res_pid['kupiec_p']):>9}"
                  f"{str(res_pid['chris_p']):>8}{str(res_pid['dq_p']):>8}"
                  f"{str(res_pid['es_ratio']):>10}{'YES' if res_pid['dat'] else 'NO':>6}")


if __name__ == "__main__":
    chay_stress_suite()
