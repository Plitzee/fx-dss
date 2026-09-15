"""CONFORMAL RISK — Adaptive Conformal Inference (ACI) và Conformal PID Control
áp dụng cho tầng rủi ro đuôi (VaR và Expected Shortfall).

Dựa trên:
- Gibbs & Candès (2021), "Adaptive Conformal Inference Under Distribution Shift", NeurIPS 35.
- Angelopoulos, Candès, Tibshirani (2023), "Conformal PID Control for Time Series Prediction", NeurIPS 36.
- Zaffran et al. (2022), "Adaptive Conformal Predictions for Time Series", ICML 162 (AgACI).
"""
import numpy as np

A_SAN = 1e-4
CUON_MAC_DINH = 500
DAM_MAC_DINH = 750
GAMMA_MAC_DINH = 0.01


def cap_nhat_aci(a_t: float, a_target: float, err: float, gamma: float = GAMMA_MAC_DINH, a_san: float = A_SAN) -> float:
    """Cập nhật mức alpha theo ACI chuẩn (Gibbs & Candès 2021).

    alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
    với err_t = 1{z_t <= q_t}.
    """
    a_moi = a_t + gamma * (a_target - err)
    return float(np.clip(a_moi, a_san, 1.0 - a_san))


def cap_nhat_pid(a_t: float, a_target: float, err: float,
                 err_prev: float = 0.0,
                 err_integral: float = 0.0,
                 kp: float = 0.01,
                 ki: float = 0.001,
                 kd: float = 0.002,
                 decay_i: float = 0.98,
                 a_san: float = A_SAN):
    """Cập nhật mức alpha theo Conformal PID Control (Angelopoulos et al. NeurIPS 2023).

    - Khâu P: K_P * (a_target - err_t) phản ứng tức thời với sai số hiện tại.
    - Khâu I: K_I * Integral tích luỹ sai số quá khứ (có hệ số phân rã decay_i).
    - Khâu D: K_D * Derivative giảm chấn dao động vi phạm giữa 2 phiên liên tiếp.

    Trả về: (alpha_{t+1}, err_integral_{t+1})
    """
    delta_err = a_target - err
    moi_i = decay_i * err_integral + delta_err
    delta_d = err_prev - err  # dương nếu phiên trước vi phạm mà phiên này không vi phạm

    u = kp * delta_err + ki * moi_i + kd * delta_d
    a_moi = float(np.clip(a_t + u, a_san, 1.0 - a_san))
    return a_moi, moi_i


def tinh_quantile_es(cua_so: np.ndarray, a_t: float, a_san: float = A_SAN):
    """Tính phân vị mức a_t và Expected Shortfall (ES) trên một cửa sổ quan sát z.

    - q_t = np.quantile(cua_so, a_t)
    - ES_t = trung bình của các phần tử cua_so <= q_t (hoặc q_t nếu rỗng).
    """
    c = np.asarray(cua_so, dtype=float)
    c = c[np.isfinite(c)]
    if len(c) < 20:
        return np.nan, np.nan
    q_level = float(np.clip(a_t, a_san, 1.0 - a_san))
    q = float(np.quantile(c, q_level))
    duoi = c[c <= q]
    es = float(duoi.mean()) if len(duoi) > 0 else q
    return q, es


def chay_aci_duoi(z: np.ndarray, a_target: float = 0.01, gamma: float = GAMMA_MAC_DINH,
                   dam: int = DAM_MAC_DINH, cuon: int = CUON_MAC_DINH):
    """Chạy bộ lọc ACI hoàn chỉnh trên chuỗi z từ đầu đến cuối một cách nhân quả.

    Trả về:
      qz: mảng phân vị dự báo tại mỗi phiên t (chỉ dùng z[<t])
      ez: mảng ES dự báo tại mỗi phiên t
      alphas: mức alpha_t thực tế tại mỗi phiên t
    """
    n = len(z)
    qz = np.full(n, np.nan)
    ez = np.full(n, np.nan)
    alphas = np.full(n, np.nan)

    a_t = float(a_target)
    for t in range(dam, n):
        cua = z[max(0, t - cuon):t]
        q, e = tinh_quantile_es(cua, a_t)
        qz[t], ez[t], alphas[t] = q, e, a_t

        if not np.isfinite(z[t]) or not np.isfinite(q):
            continue

        err = 1.0 if z[t] <= q else 0.0
        a_t = cap_nhat_aci(a_t, a_target, err, gamma=gamma)

    return qz, ez, alphas


def chay_pid_duoi(z: np.ndarray, a_target: float = 0.01,
                   kp: float = 0.01, ki: float = 0.001, kd: float = 0.002,
                   dam: int = DAM_MAC_DINH, cuon: int = CUON_MAC_DINH):
    """Chạy bộ lọc Conformal PID Control trên chuỗi z từ đầu đến cuối một cách nhân quả.

    Trả về: (qz, ez, alphas)
    """
    n = len(z)
    qz = np.full(n, np.nan)
    ez = np.full(n, np.nan)
    alphas = np.full(n, np.nan)

    a_t = float(a_target)
    err_prev = 0.0
    err_i = 0.0

    for t in range(dam, n):
        cua = z[max(0, t - cuon):t]
        q, e = tinh_quantile_es(cua, a_t)
        qz[t], ez[t], alphas[t] = q, e, a_t

        if not np.isfinite(z[t]) or not np.isfinite(q):
            continue

        err = 1.0 if z[t] <= q else 0.0
        a_t, err_i = cap_nhat_pid(a_t, a_target, err, err_prev=err_prev, err_integral=err_i,
                                  kp=kp, ki=ki, kd=kd)
        err_prev = err

    return qz, ez, alphas
