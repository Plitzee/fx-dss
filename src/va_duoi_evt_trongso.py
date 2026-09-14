"""EVT/POT CO TRONG SO cho USDJPY — buoc thu 3 sau khi thuc nghiem co trong so
(src/va_duoi_conformal_trongso.py) va Student-t co trong so deu KHONG cai
thien ro rang o dung muc 99% (VaR 1% dang bi vi pham that).

VI SAO THU EVT. Du an DA THU EVT/POT khong trong so truoc do (McNeil-Frey
2000, 4 bien the nguong 90%/95%) — "thang theo thuoc do lien tuc nhung truot
tieu chi nhi phan da chot" (xem TONG_QUAN_CHO_AI_KHAC.md). EVT duoc THIET KE
RIENG cho uoc luong duoi cuc doan (khac Student-t la phan phoi cho CA THAN),
nen ket hop voi trong so dich chuyen (Tibshirani) la mot to hop CHUA THU —
EVT sua van de "hinh dang duoi", trong so sua van de "dich chuyen theo thoi
gian", hai van de khac nhau da chan doan o hai buoc truoc.

PHUONG PHAP — Generalized Pareto co trong so tren phan vuot nguong:
  P(|Z| > x) = zeta_u * [1 + xi*(x-u)/beta]^(-1/xi),  x > u
  zeta_u = P(|Z| > u) uoc CO TRONG SO (khong phai ty le dem don gian)
  (xi, beta) khop MLE CO TRONG SO tren cac diem VUOT NGUONG

Dung LAI trong so w(x) = p(x)/(1-p(x)) tu logistic da huan luyen trong
va_duoi_conformal_trongso.py — khong tinh lai bang cach khac.

KHONG PHAM LUAT GIAO THUC — giong hai script truoc, day la THIET KE, khong
cham diem tren kiem tra.

Chay:  python src/va_duoi_evt_trongso.py
Ghi:   output/va_duoi_evt_trongso.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import optimize as _opt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "output")
PANEL = os.path.join(ROOT, "data", "panel2_6pairs.csv")

VALID_TU = pd.Timestamp("2021-10-13")
TEST_TU = pd.Timestamp("2023-11-20")
PAIR = "USDJPY"
SEED = 20260914
MUC_NGUONG = (0.90, 0.95)     # hai bien the nguong da dung o EVT khong trong so truoc do
MUC_DUOI = (0.99,)            # muc can sua nhat — VaR 1%


def dac_trung(pan):
    d = pan.sort_values("Date").reset_index(drop=True).copy()
    z = pd.Series(d.zT.values)
    az = z.abs()
    f = pd.DataFrame({"Date": d.Date, "zT": d.zT.values})
    f["z"] = z.values
    f["az"] = az.values
    for w in (20, 60):
        f[f"sd_z_{w}"] = z.rolling(w, min_periods=w // 2).std().values
        f[f"kurt_z_{w}"] = z.rolling(w, min_periods=w).kurt().values
        for k in (1.5, 2.0):
            f[f"tyle_vuot_{k}_{w}"] = (az > k).rolling(w, min_periods=w // 2).mean().values
    f["dow"] = pd.to_datetime(d.Date).dt.dayofweek.values
    return f.dropna().reset_index(drop=True)


def gpd_co_trong_so(vuot, w):
    """MLE co trong so cho Generalized Pareto tren phan VUOT NGUONG (>0).
    Tra ve (xi, beta)."""
    vuot = np.asarray(vuot, float)
    w = np.asarray(w, float)
    w = w / w.sum()

    def am_loglik(tham):
        xi, beta = tham
        if beta <= 0:
            return 1e12
        z = 1.0 + xi * vuot / beta
        if np.any(z <= 0):
            return 1e12
        if abs(xi) < 1e-8:
            logpdf = -np.log(beta) - vuot / beta
        else:
            logpdf = -np.log(beta) - (1.0 / xi + 1.0) * np.log(z)
        return -float(np.sum(w * logpdf))

    kq = _opt.minimize(am_loglik, x0=[0.1, np.std(vuot)], method="Nelder-Mead")
    xi, beta = kq.x
    return float(xi), float(beta)


def phan_vi_evt(u, xi, beta, zeta_u, muc):
    """Cong thuc phan vi POT chuan: x = u + (beta/xi)*[((1-muc)/zeta_u)^(-xi) - 1]."""
    if abs(xi) < 1e-8:
        return u - beta * np.log((1 - muc) / zeta_u)
    return u + (beta / xi) * (((1 - muc) / zeta_u) ** (-xi) - 1.0)


def main():
    t0 = time.time()
    print("=" * 96)
    print(f"EVT/POT CÓ TRỌNG SỐ (Tibshirani weighting + Generalized Pareto) — {PAIR}")
    print("=" * 96)

    pan_all = pd.read_csv(PANEL, parse_dates=["Date"])
    pan = pan_all[pan_all.pair == PAIR]
    f = dac_trung(pan)
    dat = pd.to_datetime(f.Date)
    mkd = (dat >= VALID_TU) & (dat < TEST_TU)
    mkt = dat >= TEST_TU
    fkd, fkt = f[mkd].reset_index(drop=True), f[mkt].reset_index(drop=True)

    cot = [c for c in f.columns if c not in ("Date", "zT")]
    X = pd.concat([fkd[cot], fkt[cot]], ignore_index=True).values
    y = np.r_[np.zeros(len(fkd)), np.ones(len(fkt))]
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(C=0.3, max_iter=2000, random_state=SEED)
    p_oof = cross_val_predict(clf, Xs, y, cv=cv, method="predict_proba")[:, 1]
    p_kd = np.clip(p_oof[: len(fkd)], 0.02, 0.98)
    w = p_kd / (1.0 - p_kd)
    w_deu = np.ones_like(w)

    az = np.abs(fkd.zT.values)
    print(f"n kiểm định={len(fkd)}")

    ra = {"pair": PAIR, "n_kiemdinh": int(len(fkd)), "nguong": {}}
    for muc_nguong in MUC_NGUONG:
        u = float(np.quantile(az, muc_nguong))
        m_vuot = az > u
        vuot = az[m_vuot] - u
        print(f"\n── Ngưỡng {muc_nguong:.0%} (u={u:.4f}) — {m_vuot.sum()} điểm vượt ngưỡng ──")

        ra_nguong = {"u": round(u, 4), "n_vuot": int(m_vuot.sum())}
        for ten, ww in (("khong_trongso", w_deu), ("co_trongso", w)):
            xi, beta = gpd_co_trong_so(vuot, ww[m_vuot])
            zeta_u = float(ww[m_vuot].sum() / ww.sum())
            print(f"  {ten:<14}ξ={xi:+.4f}  β={beta:.4f}  ζ_u={zeta_u:.4f}")
            muc_ra = {}
            for muc in MUC_DUOI:
                q = phan_vi_evt(u, xi, beta, zeta_u, muc)
                muc_ra[str(muc)] = round(float(q), 4)
                print(f"    mức {muc:.2f}: phân vị = {q:.4f}")
            ra_nguong[ten] = {"xi": round(xi, 4), "beta": round(beta, 4),
                              "zeta_u": round(zeta_u, 4), "phan_vi": muc_ra}
        doi = 100 * (ra_nguong["co_trongso"]["phan_vi"]["0.99"]
                     / ra_nguong["khong_trongso"]["phan_vi"]["0.99"] - 1)
        ra_nguong["doi_99_%"] = round(doi, 1)
        print(f"  → đổi ở mức 99% (có TS so không TS): {doi:+.1f}%")
        ra["nguong"][str(muc_nguong)] = ra_nguong

    # doi chieu voi moc thuc nghiem tho (khong EVT, khong trong so) de biet
    # EVT+trong-so co "an toan hon" hien trang hay khong
    moc_tho = float(np.quantile(az, 0.99))
    ra["moc_thucnghiem_tho_99"] = round(moc_tho, 4)
    print(f"\nMốc thực nghiệm thô (không EVT, không trọng số) ở 99%: {moc_tho:.4f}")
    for muc_nguong in MUC_NGUONG:
        q_ts = ra["nguong"][str(muc_nguong)]["co_trongso"]["phan_vi"]["0.99"]
        doi = 100 * (q_ts / moc_tho - 1)
        print(f"  EVT+TS ngưỡng {muc_nguong:.0%} so mốc thô: {doi:+.1f}%")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "va_duoi_evt_trongso.json"), "w", encoding="utf-8") as f_:
        json.dump(ra, f_, ensure_ascii=False, indent=1)

    print(f"\n→ output/va_duoi_evt_trongso.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
