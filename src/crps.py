"""CRPS — Continuous Ranked Probability Score, dung chung cho ca du an.

VI SAO CAN FILE NAY. Bang ket qua o `xuat_bang_baocao.py` co cot CRPS TRUNG
KHIT cot MAE cho moi mo hinh tru Chronos. Do khong phai loi sao chep: CRPS
cua mot du bao DIEM bang dung |y - f| (phan phoi Dirac), la dang thuc chinh
xac (Gneiting & Raftery 2007). Nhung no lam cot CRPS vo nghia — no khong noi
them gi so voi MAE.

CRPS chi co gia tri rieng khi mo hinh xuat mot PHAN PHOI du bao day du. File
nay cung cap hai cach tinh CRPS cho hai dang phan phoi ma repo nay thuc su
co:

  1. `crps_mau_chung` — phan phoi MAU (empirical / ensemble). Dung cho phan
     phoi loi suat cua he san xuat: r(t+1) ~ sigma^(t) x {z_1..z_m}, trong do
     {z_i} la phan phoi thuc nghiem cua loi suat chuan hoa uoc tren HUAN
     LUYEN (dung bo phan vi z ma `va_duoi.py` V0 da dong bang).

  2. `crps_lognormal` — dang DONG cho phan phoi log-chuan. Dung cho cac mo
     hinh bien dong: moi mo hinh du bao log-RV roi quy doi
     h = exp(mu + s^2/2); nghia la no NGAM dinh mot phan phoi log-chuan
     LN(mu, s^2) cho RV, chu khong phai mot diem. Chi can tach mu va s^2 ra
     la tinh duoc CRPS that.

CONG THUC.

  Mau (Gneiting & Raftery 2007, ct. 20-21):
      CRPS(F, y) = E|X - y| - 0,5 E|X - X'|
  voi X, X' doc lap tu F. Voi mau {x_1..x_m}:
      CRPS = (1/m) sum_i |x_i - y| - (1/(2m^2)) sum_i sum_j |x_i - x_j|
  So hang thu hai KHONG phu thuoc y — tinh mot lan (hieu trung binh Gini).

  Log-chuan (Baran & Lerch 2015):
      CRPS(LN(mu,s), y) = y(2*Phi(w) - 1)
                          - 2*exp(mu + s^2/2)*(Phi(w - s) + Phi(s/sqrt2) - 1)
  voi w = (ln y - mu)/s.

CA HAI DEU CO TU KIEM Monte Carlo o `_tu_kiem()` — doi chieu voi dinh nghia
tich phan goc tren du lieu mo phong co tham so BIET TRUOC.
"""
import numpy as np
from scipy import stats

EPS = 1e-300


def _gini_mau(mau):
    """(1/m^2) sum_i sum_j |x_i - x_j| — tinh O(m log m) thay vi O(m^2).

    Voi mau da sap tang x_(1)..x_(m):
        sum_i sum_j |x_i - x_j| = 2 * sum_i (2i - m - 1) * x_(i)   (i tu 1)
    """
    x = np.sort(np.asarray(mau, float))
    m = len(x)
    i = np.arange(1, m + 1)
    return float(2.0 * np.sum((2 * i - m - 1) * x) / (m * m))


def crps_mau_chung(mau, y):
    """CRPS cua MOT phan phoi mau co dinh `mau` so voi tung quan sat trong `y`.

    Tra ve mang cung do dai `y`. Dung `np.searchsorted` nen O((m+n) log m)
    thay vi O(m*n) — quan trong vi m ~ vai nghin va n ~ vai nghin.
    """
    x = np.sort(np.asarray(mau, float))
    m = len(x)
    y = np.asarray(y, float)
    S = np.concatenate([[0.0], np.cumsum(x)])          # tong tich luy
    k = np.searchsorted(x, y, side="right")             # so phan tu <= y
    # sum_i |x_i - y| = (k*y - S_k) + ((S_m - S_k) - (m-k)*y)
    tong_tuyet_doi = (k * y - S[k]) + ((S[m] - S[k]) - (m - k) * y)
    return tong_tuyet_doi / m - 0.5 * _gini_mau(x)


def crps_lognormal(mu, sigma, y):
    """CRPS dang dong cho phan phoi log-chuan LN(mu, sigma^2) so voi y > 0."""
    mu = np.asarray(mu, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-12)
    y = np.asarray(y, float)
    w = (np.log(np.maximum(y, EPS)) - mu) / sigma
    Phi = stats.norm.cdf
    return (y * (2 * Phi(w) - 1)
            - 2 * np.exp(mu + sigma ** 2 / 2)
            * (Phi(w - sigma) + Phi(sigma / np.sqrt(2)) - 1))


def crps_chuan(mu, sigma, y):
    """CRPS dang dong cho phan phoi chuan N(mu, sigma^2).

        CRPS = sigma * [ w(2*Phi(w) - 1) + 2*phi(w) - 1/sqrt(pi) ],  w=(y-mu)/sigma

    Dung de doi chieu: neu he thong dung sigma^ DUNG nhung gia dinh hinh dang
    phan phoi SAI (chuan thay vi duoi day thuc nghiem), CRPS se cho thay."""
    mu = np.asarray(mu, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-300)
    w = (np.asarray(y, float) - mu) / sigma
    return sigma * (w * (2 * stats.norm.cdf(w) - 1)
                    + 2 * stats.norm.pdf(w) - 1.0 / np.sqrt(np.pi))


def _crps_tich_phan(mau_lon, y):
    """Dinh nghia GOC bang mo phong: CRPS = E|X-y| - 0,5 E|X-X'|, uoc bang
    mau rat lon. Chi dung de TU KIEM hai ham tren, khong dung trong san xuat."""
    X = np.asarray(mau_lon, float)
    rng = np.random.default_rng(0)
    Xp = rng.permutation(X)
    return float(np.mean(np.abs(X - y)) - 0.5 * np.mean(np.abs(X - Xp)))


def _tu_kiem(im_lang=False):
    """Doi chieu ca hai ham voi dinh nghia goc tren du lieu mo phong."""
    rng = np.random.default_rng(7)
    ok = True

    # (1) Gini O(m log m) phai bang brute force O(m^2)
    x = rng.normal(0, 1, 400)
    tho = float(np.abs(x[:, None] - x[None, :]).mean())
    nhanh = _gini_mau(x)
    d1 = abs(tho - nhanh)
    ok &= d1 < 1e-9
    if not im_lang:
        print(f"  [1] Gini nhanh vs brute force: lệch {d1:.2e}  "
              f"{'ĐẠT' if d1 < 1e-9 else 'THẤT BẠI'}")

    # (2) crps_mau_chung phai bang dinh nghia goc (mau lon)
    mau = rng.normal(0.0, 1.0, 20000)
    y_thu = np.array([-2.0, -0.5, 0.0, 0.7, 3.0])
    a = crps_mau_chung(mau, y_thu)
    b = np.array([_crps_tich_phan(mau, yy) for yy in y_thu])
    d2 = float(np.max(np.abs(a - b)))
    ok &= d2 < 5e-3
    if not im_lang:
        print(f"  [2] CRPS mẫu vs định nghĩa gốc: lệch tối đa {d2:.2e}  "
              f"{'ĐẠT' if d2 < 5e-3 else 'THẤT BẠI'}")

    # (3) crps_lognormal (dang dong) phai bang CRPS cua mau tu chinh LN do
    mu0, s0 = -10.5, 0.55
    mau_ln = rng.lognormal(mu0, s0, 400000)
    y_ln = np.exp(mu0 + s0 * np.array([-1.5, -0.3, 0.4, 1.8]))
    c_dong = crps_lognormal(mu0, s0, y_ln)
    c_mau = crps_mau_chung(mau_ln, y_ln)
    d3 = float(np.max(np.abs(c_dong - c_mau) / np.maximum(np.abs(c_mau), EPS)))
    ok &= d3 < 0.02
    if not im_lang:
        print(f"  [3] CRPS log-chuẩn dạng đóng vs mẫu: lệch tương đối tối đa "
              f"{d3:.2%}  {'ĐẠT' if d3 < 0.02 else 'THẤT BẠI'}")

    # (4) Truong hop suy bien: phan phoi co ve MOT DIEM -> CRPS = MAE
    diem = np.full(5000, 2.0)
    y_d = np.array([1.0, 2.0, 3.5])
    c_diem = crps_mau_chung(diem, y_d)
    mae = np.abs(y_d - 2.0)
    d4 = float(np.max(np.abs(c_diem - mae)))
    ok &= d4 < 1e-9
    if not im_lang:
        print(f"  [4] Phân phối Dirac → CRPS = MAE: lệch {d4:.2e}  "
              f"{'ĐẠT' if d4 < 1e-9 else 'THẤT BẠI'}")

    # (4b) crps_chuan dang dong phai bang CRPS cua mau tu chinh N do
    mu_n, s_n = 0.002, 0.0071
    mau_n = rng.normal(mu_n, s_n, 400000)
    y_n = mu_n + s_n * np.array([-2.0, -0.4, 0.9, 2.3])
    d4b = float(np.max(np.abs(crps_chuan(mu_n, s_n, y_n)
                              - crps_mau_chung(mau_n, y_n))
                       / np.maximum(np.abs(crps_mau_chung(mau_n, y_n)), EPS)))
    ok &= d4b < 0.02
    if not im_lang:
        print(f"  [4b] CRPS chuẩn dạng đóng vs mẫu: lệch tương đối tối đa "
              f"{d4b:.2%}  {'ĐẠT' if d4b < 0.02 else 'THẤT BẠI'}")

    # (5) Tinh CHINH DANG (proper): phan phoi DUNG phai an diem tot hon
    #     phan phoi qua rong VA phan phoi lech tam.
    that = rng.normal(0.0, 1.0, 30000)
    y_that = rng.normal(0.0, 1.0, 3000)
    c_dung = crps_mau_chung(that, y_that).mean()
    c_rong = crps_mau_chung(rng.normal(0.0, 3.0, 30000), y_that).mean()
    c_lech = crps_mau_chung(rng.normal(1.5, 1.0, 30000), y_that).mean()
    tot = (c_dung < c_rong) and (c_dung < c_lech)
    ok &= tot
    if not im_lang:
        print(f"  [5] Chính đáng: đúng {c_dung:.4f} < rộng {c_rong:.4f} và "
              f"< lệch {c_lech:.4f}  {'ĐẠT' if tot else 'THẤT BẠI'}")
    return bool(ok)


if __name__ == "__main__":
    print("=" * 80)
    print("TỰ KIỂM MODULE CRPS")
    print("=" * 80)
    print("ĐẠT TOÀN BỘ" if _tu_kiem() else "CÓ PHÉP KIỂM THẤT BẠI")
