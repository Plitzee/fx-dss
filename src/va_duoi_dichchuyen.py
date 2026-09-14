"""ADVERSARIAL VALIDATION cho be tac tail-risk USDJPY/USDCHF (TONG_QUAN_CHO_AI_KHAC.md
muc 5.1). Cau hoi: doan KIEM DINH va doan KIEM TRA co THAT SU khac cau truc
khong — do bang so, khong phai nhin bang mat "sd(z) tang dan".

VI SAO KHONG PHAM LUAT "chon tren kiem dinh, cham kiem tra mot lan". Day KHONG
PHAI mot cau hinh mo hinh rui ro moi duoc cham diem VaR/ES tren kiem tra — no
la mot PHEP DO CHAN DOAN tren DAC TRUNG (khong dung nhan/ket qua VaR). Theo
src/split.py: doan "kiem tra" (>= 2023-11-20) DA duoc cham diem VaR/ES it nhat
mot lan roi (do la cach phat hien ra loi nay tu dau) va tu nhan la "khong con
hoan toan trinh nguyen" — lop bao ve that su la TAP KHOA SO (chua he mo). Dung
GIA/dac trung cong khai da biet (nhu da dung lich can thiep BOJ/SNB o lan thu
truoc) khong phai ro ri lua chon.

CACH LAM: gop hang cua doan KIEM DINH (nhan=0) va doan KIEM TRA (nhan=1), roi
huan luyen mot bo phan loai chi tren DAC TRUNG (khong dung VaR/ES) de doan
hang do thuoc doan nao. AUC ~ 0,50 -> khong phan biet duoc -> hai doan giong
nhau. AUC cao co y nghia (p < 0,05 qua Mann-Whitney U) -> hai doan THAT SU khac phan
phoi, giai thich duoc vi sao moi cau hinh deu "dat" tren kiem dinh nhung "truot"
tren kiem tra.

DOI CHUNG AM: chay them EURUSD (khong co van de tail-risk da biet) — neu EURUSD
CUNG cho AUC cao thi ket qua vo nghia (chi la troi thoi gian chung, khong rieng
gi USDJPY/USDCHF).

Chay:  python src/va_duoi_dichchuyen.py
Ghi:   output/va_duoi_dichchuyen.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as _st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "output")
PANEL = os.path.join(ROOT, "data", "panel2_6pairs.csv")

VALID_TU = pd.Timestamp("2021-10-13")
TEST_TU = pd.Timestamp("2023-11-20")

CAP_CO_VAN_DE = ("USDJPY", "USDCHF")
CAP_DOI_CHUNG = "EURUSD"
SEED = 20260914

# Ngay BOJ can thiep ty gia — TONG_QUAN_CHO_AI_KHAC.md muc 5.1. Lan can thiep
# dau tien ke tu 1998. Dung de kiem tra rieng: dac trung "gan ngay can thiep"
# co quan trong CHO USDJPY ma vo nghia cho doi chung EURUSD khong.
NGAY_BOJ = pd.to_datetime(["2022-09-22", "2022-10-21", "2022-10-24", "2022-12-20"])


def dac_trung(pan):
    """Dac trung TAP TRUNG VAO DUOI — KHONG dung nhan/ket qua VaR, va KHONG
    dung dac trung "dinh danh thoi gian" de dang ro ri (thang, muc sigma^
    TUYET DOI) vi che do lai suat toan cau doi ca giai doan 2022-2023 se lam
    MOI cap deu phan biet duoc — do la dung y cua ban chay dau tien (AUC cao
    o ca doi chung EURUSD, xem lich su chinh sua file nay). O day chi giu
    dac trung ve HINH DANG DUOI cua z (chuan hoa, khong con thang do gia
    tuyet doi), de tim tin hieu RIENG cho USDJPY/USDCHF thay vi trung thoi
    gian chung."""
    d = pan.sort_values("Date").reset_index(drop=True).copy()
    z = pd.Series(d.zT.values)
    az = z.abs()
    f = pd.DataFrame({"Date": d.Date})
    f["z"] = z.values
    f["az"] = az.values
    for w in (20, 60):
        f[f"sd_z_{w}"] = z.rolling(w, min_periods=w // 2).std().values
        f[f"kurt_z_{w}"] = z.rolling(w, min_periods=w).kurt().values
        for k in (1.5, 2.0):
            f[f"tyle_vuot_{k}_{w}"] = (az > k).rolling(w, min_periods=w // 2).mean().values
    f["dow"] = pd.to_datetime(d.Date).dt.dayofweek.values
    return f.dropna().reset_index(drop=True)


def them_dac_trung_boj(f):
    """Them 'so ngay giao dich toi ngay BOJ can thiep gan nhat' (ca truoc va
    sau). Ap dung THU cho MOI cap ke ca EURUSD lam gia ve (placebo) — neu dac
    trung nay quan trong cho ca EURUSD (khong lien quan gi BOJ) thi no chi la
    bien ngay-thang doi lot, khong phai tin hieu that."""
    f = f.copy()
    ngay = pd.to_datetime(f.Date).values.astype("datetime64[D]")
    boj = NGAY_BOJ.values.astype("datetime64[D]")
    kc = np.min(np.abs(ngay[:, None] - boj[None, :]).astype("timedelta64[D]").astype(int), axis=1)
    f["ngay_toi_boj_gan_nhat"] = kc
    return f


def chay_mot_cap(pair, pan_all, them_boj=False):
    pan = pan_all[pan_all.pair == pair]
    f = dac_trung(pan)
    if them_boj:
        f = them_dac_trung_boj(f)
    dat = pd.to_datetime(f.Date)
    mkd = (dat >= VALID_TU) & (dat < TEST_TU)
    mkt = dat >= TEST_TU
    fkd, fkt = f[mkd], f[mkt]
    if len(fkd) < 50 or len(fkt) < 50:
        return {"loi": f"quá ít dữ liệu: kiểm định={len(fkd)}, kiểm tra={len(fkt)}"}

    cot = [c for c in f.columns if c not in ("Date",)]
    X = pd.concat([fkd[cot], fkt[cot]], ignore_index=True).values
    y = np.r_[np.zeros(len(fkd)), np.ones(len(fkt))]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = GradientBoostingClassifier(random_state=SEED, max_depth=3, n_estimators=200)
    # cross_val_predict: du doan NGOAI-FOLD (khong bao gio du doan mot hang bang
    # mo hinh da hoc chinh no) — AUC tinh tu day khong bi lac quan do qua khop.
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, p))

    # p-value CHINH XAC cho "AUC khac 0,5 co y nghia" bang kiem dinh Mann-Whitney
    # U — tuong duong dai so voi AUC (U = AUC * n1 * n2), khong can hoan vi tron
    # lai toan bo pipeline (dat neu lam voi GradientBoosting x 1000 lan).
    p0, p1 = p[y == 0], p[y == 1]
    _, p_value = _st.mannwhitneyu(p1, p0, alternative="greater")
    p_value = float(p_value)

    clf_full = GradientBoostingClassifier(random_state=SEED, max_depth=3, n_estimators=200)
    clf_full.fit(X, y)
    dt_quan_trong = sorted(zip(cot, clf_full.feature_importances_),
                           key=lambda x: -x[1])

    return {
        "n_kiemdinh": int(len(fkd)), "n_kiemtra": int(len(fkt)),
        "auc": round(auc, 4), "p_value": round(p_value, 4),
        "co_y_nghia_005": bool(p_value < 0.05),
        "dac_trung_quan_trong_nhat": [(c, round(float(v), 4)) for c, v in dt_quan_trong[:5]],
        "moi_dac_trung": [(c, round(float(v), 4)) for c, v in dt_quan_trong],
    }


def main():
    t0 = time.time()
    print("=" * 96)
    print("ADVERSARIAL VALIDATION — kiểm định vs kiểm tra có khác phân phối thật không")
    print("PHÉP ĐO CHẨN ĐOÁN trên đặc trưng, KHÔNG dùng nhãn/kết quả VaR — không phạm")
    print("luật 'chọn trên kiểm định, chấm kiểm tra một lần' (xem docstring).")
    print("=" * 96)

    pan_all = pd.read_csv(PANEL, parse_dates=["Date"])
    ra = {}
    for p in (*CAP_CO_VAN_DE, CAP_DOI_CHUNG):
        print(f"\n[{p}]" + ("  <- đối chứng âm, KHÔNG có vấn đề tail-risk đã biết"
                            if p == CAP_DOI_CHUNG else "  <- có vấn đề tail-risk đã biết"))
        kq = chay_mot_cap(p, pan_all)
        ra[p] = kq
        if "loi" in kq:
            print("  lỗi:", kq["loi"])
            continue
        print(f"  n kiểm định={kq['n_kiemdinh']} · n kiểm tra={kq['n_kiemtra']}")
        print(f"  AUC = {kq['auc']:.4f}  (0,50 = không phân biệt được)")
        print(f"  p-value (Mann-Whitney U) = {kq['p_value']:.4f}"
              f"  -> {'CÓ Ý NGHĨA — hai đoạn khác phân phối thật' if kq['co_y_nghia_005'] else 'KHÔNG có ý nghĩa'}")
        print("  đặc trưng phân biệt mạnh nhất:")
        for c, v in kq["dac_trung_quan_trong_nhat"]:
            print(f"    {c:<14}{v:.4f}")

    # Vong 2 — them dac trung "gan ngay BOJ can thiep", ap cho CA USDJPY (that)
    # LAN EURUSD (gia ve/placebo). Neu dac trung nay quan trong o ca EURUSD thi
    # no chi la bien ngay-thang doi lot ("gan quy 4/2022"), khong phai tin hieu
    # rieng cho USDJPY.
    print("\n" + "=" * 96)
    print("VÒNG 2 — thêm đặc trưng 'gần ngày BOJ can thiệp' (USDJPY thật, EURUSD giả vờ/placebo)")
    print("=" * 96)
    ra_boj = {}
    for p in ("USDJPY", CAP_DOI_CHUNG):
        kq = chay_mot_cap(p, pan_all, them_boj=True)
        ra_boj[p] = kq
        vi_tri = None
        for i, (c, v) in enumerate(kq.get("moi_dac_trung", [])):
            if c == "ngay_toi_boj_gan_nhat":
                vi_tri = (i + 1, v)
        print(f"\n[{p}] AUC={kq.get('auc')} · 'ngày tới BOJ gần nhất' "
              + (f"hạng #{vi_tri[0]}/{len(kq.get('moi_dac_trung', []))}, "
                 f"độ quan trọng={vi_tri[1]:.4f}" if vi_tri else "không tìm thấy"))

    ra["vong2_boj"] = ra_boj

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "va_duoi_dichchuyen.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)

    print("\n" + "-" * 96)
    soc = [p for p in CAP_CO_VAN_DE if ra.get(p, {}).get("co_y_nghia_005")]
    doi_chung_soc = ra.get(CAP_DOI_CHUNG, {}).get("co_y_nghia_005")
    if soc and not doi_chung_soc:
        print(f"→ XÁC NHẬN: {', '.join(soc)} có dịch chuyển phân phối thật (kiểm định≠kiểm tra),")
        print(f"  còn {CAP_DOI_CHUNG} (đối chứng) thì không — không phải trôi thời gian chung.")
    elif soc and doi_chung_soc:
        print("→ CẢNH BÁO: đối chứng CŨNG có ý nghĩa — có thể là trôi thời gian CHUNG cho")
        print("  mọi cặp (ví dụ chế độ lãi suất đổi toàn cầu), không phải đặc thù USDJPY/USDCHF.")
    else:
        print("→ KHÔNG xác nhận được dịch chuyển phân phối qua đặc trưng đã chọn.")
    print(f"\n→ output/va_duoi_dichchuyen.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
