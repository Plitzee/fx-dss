"""Ban RUT GON src/optimal_stop.py — chi phan CAN cho mot request cua API.

Vi sao rut gon rieng thay vi import optimal_stop.py thang: file goc "import
pandas" o dau va co cac ham doc CSV bang pandas (nap_panel, carry_ngay,
chi_phi_thoat qua module cost.py) — nhung thu do da duoc TINH SAN va xuat ra
JSON boi src/export_ui_state.py, request-time khong can nua. Bo pandas ra
khoi ham Vercel Python giup bundle nhe va cold start nhanh hon.

_buoc_gia_tri / giai / nen_giu sao chep Y HET logic o optimal_stop.py (dung
nguon that, khong viet lai tu dau) — chi bo phan doc file. Neu sua cong thuc
o optimal_stop.py thi PHAI sua o day cho khop (co test doi chieu, xem
scripts/check_dp_core.py convention trong README cua thu muc web/).
"""
import numpy as np

N_CHE_DO = 3
S_MAX, S_STEP = 8.0, 0.05
LUOI_S = np.arange(0.0, S_MAX + 1e-9, S_STEP)


def _buoc_gia_tri(r, f):
    if f is None:
        return r
    if f <= 1e-12:
        return np.zeros_like(np.asarray(r, float)) if hasattr(r, "shape") else 0.0
    rc = np.clip(np.asarray(r, float), -50.0, 50.0)
    arg = f * np.expm1(rc)
    return np.where(arg > -1.0 + 1e-12, np.log1p(np.maximum(arg, -1.0 + 1e-12)), -50.0)


def giai(mau, sig_v, carry_ngay_abs, c_thoat_sigma, slip_sigma, N=20,
         M=4000, seed=0, f_v=None):
    rng = np.random.default_rng(seed)
    ns, nv = len(LUOI_S), N_CHE_DO
    DR = {}
    for v in range(nv):
        n = len(mau[v]["zT"])
        j = rng.integers(0, n, M) if n > M else np.arange(n)
        DR[v] = {k: mau[v][k][j] for k in ("zT", "zL", "ty", "v2")}

    fv = None if f_v is None else np.asarray(f_v, float)

    V = np.zeros((N + 1, ns, nv))
    for v in range(nv):
        f = None if fv is None else float(fv[v])
        V[0, :, v] = _buoc_gia_tri(-c_thoat_sigma * sig_v[v], f)
    bien = np.full((N + 1, nv), np.nan)
    for n in range(1, N + 1):
        for v in range(nv):
            f = None if fv is None else float(fv[v])
            zT, zL, ty, v2 = (DR[v][k] for k in ("zT", "zL", "ty", "v2"))
            sv = sig_v[v]
            cham = zL[None, :] <= -LUOI_S[:, None]
            s2 = np.clip((LUOI_S[:, None] + zT[None, :]) * ty[None, :], 0.0, S_MAX)
            Vn = np.empty_like(s2)
            for vv in range(nv):
                sel = v2 == vv
                if sel.any():
                    Vn[:, sel] = np.interp(s2[:, sel].ravel(), LUOI_S,
                                           V[n - 1, :, vv]).reshape(s2[:, sel].shape)
            pay_cham = _buoc_gia_tri(
                (-LUOI_S[:, None] - slip_sigma - c_thoat_sigma) * sv, f)
            pay_song = _buoc_gia_tri(zT[None, :] * sv, f) + Vn
            carry_buoc = _buoc_gia_tri(carry_ngay_abs, f)
            giu = carry_buoc + np.where(cham, pay_cham, pay_song).mean(1)
            dong = np.full(ns, _buoc_gia_tri(-c_thoat_sigma * sv, f))
            V[n, :, v] = np.maximum(giu, dong)
            gi = giu > dong
            bien[n, v] = LUOI_S[np.argmax(gi)] if gi.any() else np.inf
    return V, bien


def nen_giu(V, s, v, n, c_thoat_sigma, sig_v, f_v=None):
    if n <= 0:
        return False
    i = int(np.clip(round(s / S_STEP), 0, len(LUOI_S) - 1))
    n = int(np.clip(n, 0, V.shape[0] - 1))
    v = int(v)
    f = None if f_v is None else float(np.asarray(f_v, float)[v])
    nguong = _buoc_gia_tri(-c_thoat_sigma * sig_v[v], f)
    return bool(V[n, i, v] > nguong + 1e-14)
