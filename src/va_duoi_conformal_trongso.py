"""CONFORMAL CO TRONG SO (Tibshirani et al. 2019, "Conformal Prediction Under
Covariate Shift") cho be tac tail-risk USDJPY — buoc tiep theo sau khi
src/va_duoi_dichchuyen.py XAC NHAN co dich chuyen phan phoi that giua doan
kiem dinh va kiem tra (AUC 0,99, p<0,0001) va src/va_duoi_diemgay.py cho
thay USDJPY la TROI DAN (khong phai mot cu gay), nen hop voi huong sua
bang trong so hon la loai bo mot cua so nhu USDCHF.

Y TUONG. Phan vi duoi thuc nghiem hien dung (V0, xem decision_record.py) coi
moi quan sat trong doan hieu chuan NANG NHU NHAU. Nhung neu ta DA BIET (qua
bo phan loai adversarial validation) rang mot so quan sat "giong dieu kien
gan day" hon nhung quan sat khac, thi cho chung trong so LON HON khi uoc
luong phan vi la hop ly hon — dung DUNG khung Tibshirani: trong so = ty le
kha nang w(x) = p(x)/(1-p(x)), voi p(x) uoc tu bo phan loai phan biet
"giong kiem dinh" hay "giong kiem tra".

KHONG PHAM LUAT GIAO THUC: day la THIET KE dua tren MOT PHAT HIEN DA BIET
TRUOC (dich chuyen da do o buoc truoc), khong phai chon bang cach do hieu
nang tren kiem tra. Script nay KHONG cham diem VaR/ES tren kiem tra — chi
BAO CAO phan vi thay doi bao nhieu khi co trong so, de CHUAN BI mot phuong
an dong bang truoc khi mo tap khoa so — dung theo dung ke hoach da thong
nhat (xem hoi thoai).

Chay:  python src/va_duoi_conformal_trongso.py
Ghi:   output/va_duoi_conformal_trongso.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import optimize as _opt
from scipy import stats as _st
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
MUC_DUOI = (0.90, 0.95, 0.99)          # cac muc phan vi duoi hay dung cho VaR/khoang
EPS = 1e-9


def dac_trung(pan):
    """Y HET va_duoi_dichchuyen.py — de bo phan loai va trong so nhat quan
    voi phan da chan doan truoc do, khong tinh lai theo cong thuc khac."""
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


def student_t_co_trong_so(z, w):
    """Khop Student-t (floc=0, dung quy uoc decision_record.py) bang MLE CO
    TRONG SO — toi thieu hoa log-hop-ly AM co trong so truc tiep bang
    scipy.optimize, KHONG dung resample xap xi (chinh xac hon, khong nhieu
    do lay mau). Tra ve (nu, scale)."""
    z = np.asarray(z, float)
    w = np.asarray(w, float)
    w = w / w.sum()

    def am_loglik(tham):
        nu, scale = tham
        if nu <= 1.0 or scale <= 0:
            return 1e12
        logpdf = _st.t.logpdf(z / scale, nu) - np.log(scale)
        return -float(np.sum(w * logpdf))

    nu0, _, sc0 = _st.t.fit(z, floc=0)
    kq = _opt.minimize(am_loglik, x0=[np.clip(nu0, 2.5, 40), sc0],
                       method="Nelder-Mead",
                       bounds=[(2.5, 40), (1e-6, None)])
    nu, scale = kq.x
    return float(np.clip(nu, 2.5, 40)), float(scale)


def phan_vi_co_trong_so(z, w, muc):
    """Phan vi thuc nghiem CO TRONG SO — cong thuc chuan cua weighted
    conformal (Tibshirani et al. 2019 muc 2): sap xep |z| tang dan, tim
    nguong nho nhat ma tong trong so CHUAN HOA (w_i / sum w) da vuot muc."""
    az = np.abs(np.asarray(z, float))
    w = np.asarray(w, float)
    w = w / w.sum()
    idx = np.argsort(az)
    az_sap, w_sap = az[idx], w[idx]
    tich_luy = np.cumsum(w_sap)
    j = np.searchsorted(tich_luy, muc)
    j = min(j, len(az_sap) - 1)
    return float(az_sap[j])


def main():
    t0 = time.time()
    print("=" * 96)
    print(f"CONFORMAL CÓ TRỌNG SỐ (Tibshirani 2019) — {PAIR}")
    print("=" * 96)

    pan_all = pd.read_csv(PANEL, parse_dates=["Date"])
    pan = pan_all[pan_all.pair == PAIR]
    f = dac_trung(pan)
    dat = pd.to_datetime(f.Date)
    mkd = (dat >= VALID_TU) & (dat < TEST_TU)
    mkt = dat >= TEST_TU
    fkd, fkt = f[mkd].reset_index(drop=True), f[mkt].reset_index(drop=True)
    print(f"n kiểm định={len(fkd)} · n kiểm tra={len(fkt)}")

    cot = [c for c in f.columns if c not in ("Date", "zT")]
    X = pd.concat([fkd[cot], fkt[cot]], ignore_index=True).values
    y = np.r_[np.zeros(len(fkd)), np.ones(len(fkt))]

    # DUNG logistic regression (khong dung GradientBoosting nhu
    # va_duoi_dichchuyen.py): GBM phan biet QUA SAC (AUC~0,99) khien trong so
    # w=p/(1-p) don vao rat it hang — chay thu lan dau cho co mau hieu dung
    # chi 26,9/547, khong dung duoc. Logistic tuyen tinh + chuan hoa cho xac
    # suat MUOT hon, dung DUNG khuyen nghi chuan cua van liệu covariate-shift
    # (uoc luong ty le kha nang bang mo hinh CO KIEM SOAT do phuc tap, khong
    # phai mo hinh linh hoat nhat co the).
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(C=0.3, max_iter=2000, random_state=SEED)
    p_oof = cross_val_predict(clf, Xs, y, cv=cv, method="predict_proba")[:, 1]
    p_kd = p_oof[: len(fkd)]          # xac suat "giong kiem tra" cho TUNG hang kiem dinh

    # trong so Tibshirani: w(x) = p(x) / (1 - p(x)) — hang nao cang "giong
    # dieu kien gan day (kiem tra)" thi cang duoc trong so lon hon khi uoc
    # luong phan vi tu doan kiem dinh. CAT NGUONG p o [0,02; 0,98] — thuc
    # hanh chuan cua importance weighting de tranh mot vai hang co p gan 1
    # chiem het trong so (Cole & Hernan 2008 goi la "stabilized/truncated
    # weights", ky thuat pho bien trong ca kinh te luong lan ML nhan qua).
    p_kd_cat = np.clip(p_kd, 0.02, 0.98)
    w = p_kd_cat / (1.0 - p_kd_cat)

    z_kd = fkd.zT.values

    # Bon phuong an: {thuc nghiem, Student-t} x {khong trong so, co trong so}
    nu0, sc0 = student_t_co_trong_so(z_kd, np.ones_like(w))
    nu1, sc1 = student_t_co_trong_so(z_kd, w)
    print(f"\nStudent-t KHÔNG trọng số: ν={nu0:.2f}, scale={sc0:.4f}")
    print(f"Student-t CÓ trọng số:    ν={nu1:.2f}, scale={sc1:.4f}")

    ra = {"pair": PAIR, "n_kiemdinh": int(len(fkd)),
         "trong_so_hieu_dung": round(float((w.sum() ** 2) / (w ** 2).sum()), 1),
         "student_t": {"khong_trongso": {"nu": round(nu0, 3), "scale": round(sc0, 5)},
                       "co_trongso": {"nu": round(nu1, 3), "scale": round(sc1, 5)}},
         "muc": {}}
    print(f"\nCỡ mẫu hiệu dụng sau khi có trọng số: {ra['trong_so_hieu_dung']:.1f} "
          f"(so với {len(fkd)} không trọng số — trọng số càng lệch thì số này càng nhỏ)")
    print(f"\n{'mức':>6}{'TN không TS':>13}{'TN có TS':>11}{'t không TS':>13}{'t có TS':>10}"
          f"{'đổi tốt nhất':>14}")
    print("-" * 78)
    for muc in MUC_DUOI:
        q_tn0 = phan_vi_co_trong_so(z_kd, np.ones_like(w), muc)
        q_tn1 = phan_vi_co_trong_so(z_kd, w, muc)
        q_t0 = float(_st.t.ppf((1 + muc) / 2, nu0) * sc0)     # phan vi doi xung |z|
        q_t1 = float(_st.t.ppf((1 + muc) / 2, nu1) * sc1)
        doi = 100 * (q_t1 / q_tn0 - 1)
        ra["muc"][str(muc)] = {
            "thucnghiem_khong_trongso": round(q_tn0, 4),
            "thucnghiem_co_trongso": round(q_tn1, 4),
            "studentt_khong_trongso": round(q_t0, 4),
            "studentt_co_trongso": round(q_t1, 4)}
        print(f"{muc:>6.2f}{q_tn0:>13.4f}{q_tn1:>11.4f}{q_t0:>13.4f}{q_t1:>10.4f}"
              f"{doi:>+13.1f}%")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "va_duoi_conformal_trongso.json"), "w", encoding="utf-8") as f_:
        json.dump(ra, f_, ensure_ascii=False, indent=1)

    print("\n" + "-" * 96)
    print("→ ĐÂY LÀ THIẾT KẾ, CHƯA PHẢI QUYẾT ĐỊNH CUỐI — không chấm trên kiểm tra ở đây.")
    print("  Nếu đồng ý hướng này, đóng băng cách tính trọng số NÀY (không đổi thêm), rồi")
    print("  chấm điểm CUỐI CÙNG cùng lúc với lần mở tập khoá sổ của luận văn.")
    print(f"\n→ output/va_duoi_conformal_trongso.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
