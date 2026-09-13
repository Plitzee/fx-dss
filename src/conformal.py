"""CONFORMAL PREDICTION — cac ham LOI, khong phu thuoc gi ngoai numpy.

Tach rieng khoi `run_conformal.py` (script do dac) vi `api/main.py` can dung
cac ham nay o san xuat, ma `run_conformal.py` lai import `run_walkforward`,
con `run_walkforward` import `api.main` — vong tron. Module nay khong import gi
cua repo nen ca hai ben dung duoc.

Ly thuyet va so do dac: docs/CHISO_DANHGIA.md muc 16.
"""
import numpy as np

EPS = 1e-12


def diem_lac(P, y=None):
    """LAC (Sadinle et al. 2019): s(x,y) = 1 - p_y.

    Cho tap NHO NHAT voi cung do phu bien duyen. Neu y=None tra ve ma tran
    (n,3) diem so cho MOI nhan."""
    P = np.asarray(P, float)
    return 1.0 - P if y is None else 1.0 - P[np.arange(len(y)), y]


def diem_aps(P, y=None):
    """APS (Romano et al. 2020): tong xac suat da sap giam den khi cham y.

    Tap LON hon LAC nhung do phu CO DIEU KIEN theo lop tot hon."""
    P = np.asarray(P, float)
    thu = np.argsort(-P, axis=1)
    cong = np.cumsum(np.take_along_axis(P, thu, axis=1), axis=1)
    S = np.empty_like(P)
    np.put_along_axis(S, thu, cong, axis=1)
    return S if y is None else S[np.arange(len(y)), y]


DIEM = {"LAC": diem_lac, "APS": diem_aps}


def nguong(diem_hc, alpha):
    """Phan vi conformal, co hieu chinh huu han mau (n+1)(1-alpha)/n."""
    d = np.asarray(diem_hc, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return np.inf
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(d)[k - 1])


def chay_tinh(P_hc, y_hc, P_dg, ham, alpha=0.10):
    """Split conformal TINH — q^ chot mot lan tren tap hieu chuan.

    Bao dam chi dung khi du lieu HOAN VI DUOC. Chuoi thoi gian tai chinh thi
    khong, va do la ly do ham nay HONG tren du lieu that (do o muc 16)."""
    q = nguong(ham(P_hc, y_hc), alpha)
    return ham(P_dg) <= q, np.full(len(P_dg), alpha)


def chay_aci(P_hc, y_hc, P_dg, y_dg, ham, alpha=0.10, gamma=0.01, cua_so=500):
    """ACI (Gibbs & Candes 2021) — alpha cap nhat TRUC TUYEN.

        alpha_{t+1} = alpha_t + gamma * (alpha_muc_tieu - err_t)

    Bao dam do phu DAI HAN duoi troi phan phoi tuy y — thu ma du lieu nay co
    that (xem muc 16.2).

    NHAN QUA: tai phien t, nguong lay tu cac diem so DA THUC HIEN (hieu chuan
    ban dau + moi ket cuc da biet den t-1). Ket cuc cua chinh phien t chi duoc
    dung SAU khi da phat tap du bao cho no — dung nhu khi chay that, noi phien
    moi nhat chua co ket cuc.

    Tra ve (tap (n,3) bool, alpha theo tung phien)."""
    d_hc = list(ham(P_hc, y_hc))
    a_t = float(alpha)
    n = len(P_dg)
    S_dg = ham(P_dg)
    tap = np.zeros((n, 3), bool)
    alphas = np.empty(n)
    for t in range(n):
        alphas[t] = a_t
        q = nguong(d_hc[-cua_so:], min(max(a_t, 1e-4), 0.999))
        tap[t] = S_dg[t] <= q
        err = 0.0 if tap[t, y_dg[t]] else 1.0
        a_t = float(np.clip(a_t + gamma * (alpha - err), 1e-4, 0.999))
        d_hc.append(float(S_dg[t, y_dg[t]]))
    return tap, alphas


def cham(tap, y):
    """(do phu thuc te, kich thuoc tap trung binh, ty le tap rong)."""
    phu = float(tap[np.arange(len(y)), y].mean())
    kt = tap.sum(1)
    return phu, float(kt.mean()), float((kt == 0).mean())
