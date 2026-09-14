"""META-LABELING cho tang bien do — mo hinh THU CAP du doan "hom nay du bao
sigma^ co kha nang SAI NHIEU khong", KHONG du doan lai bien dong. Dung dung
khung meta-labeling (Lopez de Prado 2018): mo hinh SO CAP (HAR vong 7) giu
nguyen; mo hinh THU CAP chi quyet dinh CO NEN TIN nhieu hay it vao phien nay.

TAI SAO NHAM VAO BIEN DO, KHONG PHAI HUONG. Truc huong da xac nhan 198/198
gia thuyet khong song sot qua kiem soat boi — thu meta-label cho "du bao
huong co dung khong" gan nhu chac chan that bai vi chinh cai no dua tren
(huong) da duoc chung minh khong doan duoc. Truc bien do THI CO ky nang that
(BSS+, QLIKE thang MCS) — nen "phien nao du bao bien do kem tin cay hon" la
cau hoi co co so de hoi, khac han "phien nao se tang/giam".

GIA THUYET CHOT TRUOC (H_META): mot mo hinh phan loai tren dac trung da co
(do lech chuan cuon cua z, do nhon duoi, ty le vuot nguong, che do) co the
phan biet phien "QLIKE tot" khoi phien "QLIKE kem" tot hon mot mo hinh hang
so (khi hau hoc), do bang BSS > 0 tren tap GIU RIENG (khong dung de chon).

DICH: y=1 neu QLIKE phien do nam trong NHOM 20% TE NHAT (do tren tap huan
luyen, dong bang), y=0 con lai. Chon nguong tu HUAN LUYEN, ap sang kiem dinh.

Chay:  python src/metalabel_qlike.py
Ghi:   output/metalabel_qlike.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import brier_score_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "output")
PANEL = os.path.join(ROOT, "data", "panel2_6pairs.csv")
NGOAI_SINH = os.path.join(ROOT, "data", "ngoai_sinh", "chuoi_ngay.csv")

VALID_TU = pd.Timestamp("2021-10-13")
TEST_TU = pd.Timestamp("2023-11-20")
PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF")
SEED = 20260914
NHOM_TE = 0.20       # 20% phien QLIKE te nhat -> nhan 1


def nap_vix():
    """VIX tre 1 ngay — bien NGOAI SINH DUY NHAT da qua tam giac hoa nhan qua
    (5 phuong phap, Proposed Method cua de cuong) va con song sot trong tap
    dac trung E3. Dung dung bien nay, khong dua ca 9 chuoi FRED vao (se lap
    lai dung sai lam validation-selected da do la KEM hon causally-filtered
    o PHA3B_KETQUA.md)."""
    v = pd.read_csv(NGOAI_SINH, parse_dates=["date"])[["date", "VIXCLS"]]
    v = v.rename(columns={"date": "Date"}).sort_values("Date")
    v["vix_tre1"] = v.VIXCLS.shift(1)          # tre 1 ngay, dung quy uoc chung
    return v[["Date", "vix_tre1"]]


def dac_trung_va_qlike(pan, vix=None):
    d = pan.sort_values("Date").reset_index(drop=True).copy()
    sig2 = np.maximum(d.sig.values, 1e-12) ** 2
    rv5 = np.maximum(d.rv5.values, 1e-12)
    qlike = np.log(sig2) + rv5 / sig2

    z = pd.Series(d.zT.values)
    az = z.abs()
    # QUAN TRONG — da tung SAI o ban dau: rolling() MAC DINH tinh CA hang hien
    # tai. Muc tieu la du bao TRUOC khi biet ket qua phien do, nen moi dac
    # trung phai la HAM CUA z_{t-1}, z_{t-2}, ... KHONG bao gio z_t. Dung
    # shift(1) TRUOC roi moi rolling — da kiem chung bang vi du don gian
    # (xem hoi thoai) rang khong shift se dua chinh gia tri hom nay vao dac
    # trung "du bao truoc" cua no.
    z_tre = z.shift(1)
    az_tre = az.shift(1)
    f = pd.DataFrame({"Date": d.Date, "qlike": qlike})
    for w in (20, 60):
        f[f"sd_z_{w}"] = z_tre.rolling(w, min_periods=w // 2).std().values
        f[f"kurt_z_{w}"] = z_tre.rolling(w, min_periods=w).kurt().values
        f[f"tyle_vuot_1.5_{w}"] = (az_tre > 1.5).rolling(w, min_periods=w // 2).mean().values
    # KHONG dua "sig" (du bao HAR hom nay) vao dac trung — DA DO: corr(log(sig^2),
    # qlike) = 0,64, vi qlike = log(sig^2) + rv5/sig^2 CHUA CHINH sig trong cong
    # thuc. Dua sig vao se cho BSS +0,37..+0,55 GIA (ro ri vong tron), da xac
    # minh: bo sig di thi BSS am (-0,11). Day la loai loi H2-mau-so/lech-pha
    # dac-trung-muc-tieu ma du an da tung mac va sua truoc do.
    f["dow"] = pd.to_datetime(d.Date).dt.dayofweek.values
    if vix is not None:
        f = pd.merge(f, vix, on="Date", how="left")
        # NGAY CUOI TUAN/le khong co VIX (san chung khoan My dong): dien
        # tien nhiem gan nhat — hop ly vi VIX la muc, khong phai bien co.
        f["vix_tre1"] = f["vix_tre1"].ffill()
    return f.dropna().reset_index(drop=True)


def chay_mot_cap(pair, pan_all, vix=None):
    pan = pan_all[pan_all.pair == pair]
    f = dac_trung_va_qlike(pan, vix=vix)
    dat = pd.to_datetime(f.Date)
    m_tr = dat < VALID_TU
    m_kd = (dat >= VALID_TU) & (dat < TEST_TU)
    f_tr, f_kd = f[m_tr], f[m_kd]
    if len(f_tr) < 100 or len(f_kd) < 50:
        return {"loi": f"quá ít dữ liệu: huấn luyện={len(f_tr)}, kiểm định={len(f_kd)}"}

    # nguong QLIKE "te" CHOT TREN HUAN LUYEN, ap sang kiem dinh — khong nhin
    # kiem dinh de chon nguong.
    nguong = float(np.quantile(f_tr.qlike.values, 1 - NHOM_TE))
    y_tr = (f_tr.qlike.values > nguong).astype(int)
    y_kd = (f_kd.qlike.values > nguong).astype(int)

    cot = [c for c in f.columns if c not in ("Date", "qlike")]
    X_tr, X_kd = f_tr[cot].values, f_kd[cot].values

    clf = GradientBoostingClassifier(random_state=SEED, max_depth=2, n_estimators=150)
    clf.fit(X_tr, y_tr)
    p_kd = clf.predict_proba(X_kd)[:, 1]

    p_khihauhoc = np.full(len(y_kd), y_tr.mean())    # mo hinh hang so tu huan luyen
    brier_mh = brier_score_loss(y_kd, p_kd)
    brier_kh = brier_score_loss(y_kd, p_khihauhoc)
    bss = 1 - brier_mh / brier_kh
    auc = roc_auc_score(y_kd, p_kd) if len(set(y_kd)) > 1 else float("nan")

    dt = sorted(zip(cot, clf.feature_importances_), key=lambda x: -x[1])[:5]

    return {
        "n_huanluyen": int(len(f_tr)), "n_kiemdinh": int(len(f_kd)),
        "nguong_qlike_te": round(nguong, 4),
        "ty_le_te_kiemdinh": round(float(y_kd.mean()), 4),
        "brier_mohinh": round(float(brier_mh), 5), "brier_khihauhoc": round(float(brier_kh), 5),
        "bss": round(float(bss), 4), "auc": round(float(auc), 4),
        "dat_H_META": bool(bss > 0),
        "dac_trung_quan_trong_nhat": [(c, round(float(v), 4)) for c, v in dt],
    }


def main():
    t0 = time.time()
    print("=" * 96)
    print("META-LABELING — dự đoán phiên nào QLIKE (σ̂) sẽ TỆ, không dự đoán lại biến động")
    print("Giả thuyết chốt trước H_META: BSS > 0 trên kiểm định (chọn ngưỡng từ huấn luyện)")
    print("=" * 96)

    pan_all = pd.read_csv(PANEL, parse_dates=["Date"])
    vix = nap_vix()
    ra = {}
    dat_may_cap = 0
    for p in PAIRS:
        print(f"\n[{p}]")
        kq0 = chay_mot_cap(p, pan_all)                # khong VIX — moc doi chieu
        kq = chay_mot_cap(p, pan_all, vix=vix)         # co VIX
        kq["bss_khong_vix"] = kq0.get("bss")
        ra[p] = kq
        if "loi" in kq:
            print("  lỗi:", kq["loi"]); continue
        print(f"  BSS không VIX = {kq0.get('bss'):+.4f}  →  BSS có VIX = {kq['bss']:+.4f}")
        print(f"  n huấn luyện={kq['n_huanluyen']} · n kiểm định={kq['n_kiemdinh']} "
              f"· tỉ lệ tệ thật ở kiểm định={100*kq['ty_le_te_kiemdinh']:.1f}%")
        print(f"  Brier mô hình={kq['brier_mohinh']} · Brier khí hậu học={kq['brier_khihauhoc']}")
        print(f"  BSS = {kq['bss']:+.4f}  ·  AUC = {kq['auc']:.4f}  "
              f"->  {'ĐẠT H_META' if kq['dat_H_META'] else 'KHÔNG ĐẠT'}")
        print("  đặc trưng quan trọng nhất:", ", ".join(f"{c}={v}" for c, v in kq["dac_trung_quan_trong_nhat"][:3]))
        dat_may_cap += int(kq["dat_H_META"])

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "metalabel_qlike.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)

    print("\n" + "-" * 96)
    print(f"→ ĐẠT H_META (BSS>0) ở {dat_may_cap}/{len(PAIRS)} cặp trên đoạn kiểm định.")
    print(f"→ output/metalabel_qlike.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
