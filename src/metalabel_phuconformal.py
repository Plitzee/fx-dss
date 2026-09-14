"""META-LABELING cho LO HONG DA BIET, CHUA SUA: "Conformal phu thieu khi
dang lo" (decision_record.py::KhoangACI — da thu NAM cach, khong cach nao
xoa duoc lech 0,6-0,8 diem phan tram, xem docstring lop do).

Ke tiep dung khung metalabel_qlike.py (da sua 2 loi ro ri, con lai BSS+ that
o 3/6 cap) — ap dung y het ky luat do cho MOT muc tieu khac: KHONG du bao
lai bien dong/huong, chi du bao "phien nay khoang conformal co kha nang
KHONG PHU (vi pham) hay khong", dung dac trung DA TRE (shift(1) truoc roi
moi rolling — sua dung 2 loi da bat duoc o metalabel_qlike.py, ap dung tu
dau o day, khong lap lai loi cu).

MUC TIEU: nguong nua-be-rong 90% CHOT TREN HUAN LUYEN (dung cong thuc giong
KhoangConformal.nua_be_rong: quantile(|z_train|, 0.90)). y=1 neu |z_t| VUOT
nguong do (vi pham do phu) tai phien t, y=0 con lai. Neu meta-model bat duoc
mau hinh CO DIEU KIEN (vi du: sut giam/nhieu bien dong gan day -> vi pham de
xay ra hon), day la co so DE THUC SU sua duoc lo hong da biet, thay vi chi
bao "van con lech" nhu 5 lan thu truoc.

Chay:  python src/metalabel_phuconformal.py
Ghi:   output/metalabel_phuconformal.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "output")
PANEL = os.path.join(ROOT, "data", "panel2_6pairs.csv")

VALID_TU = pd.Timestamp("2021-10-13")
TEST_TU = pd.Timestamp("2023-11-20")
PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF")
SEED = 20260914
MUC_PHU = 0.90


def dac_trung_va_vipham(pan):
    d = pan.sort_values("Date").reset_index(drop=True).copy()
    z = pd.Series(d.zT.values)
    az = z.abs()

    # DAC TRUNG: chi dung thong tin QUA THU {t-1, t-2, ...} — shift(1) TRUOC
    # roi moi rolling, dung nhu bai hoc rut ra tu metalabel_qlike.py.
    z_tre, az_tre = z.shift(1), az.shift(1)
    f = pd.DataFrame({"Date": d.Date, "az": az.values})
    for w in (20, 60):
        f[f"sd_z_{w}"] = z_tre.rolling(w, min_periods=w // 2).std().values
        f[f"kurt_z_{w}"] = z_tre.rolling(w, min_periods=w).kurt().values
        f[f"tyle_vuot_1.5_{w}"] = (az_tre > 1.5).rolling(w, min_periods=w // 2).mean().values
        # trang thai "sut giam" tho: bao nhieu phien gan day co z am lien tiep
        f[f"soi_giam_{w}"] = z_tre.rolling(w, min_periods=w // 2).apply(
            lambda x: float(np.mean(x < 0)), raw=True).values
    f["dow"] = pd.to_datetime(d.Date).dt.dayofweek.values
    return f.dropna().reset_index(drop=True)


def chay_mot_cap(pair, pan_all):
    pan = pan_all[pan_all.pair == pair]
    f = dac_trung_va_vipham(pan)
    dat = pd.to_datetime(f.Date)
    m_tr = dat < VALID_TU
    m_kd = (dat >= VALID_TU) & (dat < TEST_TU)
    f_tr, f_kd = f[m_tr], f[m_kd]
    if len(f_tr) < 100 or len(f_kd) < 50:
        return {"loi": f"quá ít dữ liệu: huấn luyện={len(f_tr)}, kiểm định={len(f_kd)}"}

    # nguong nua-be-rong CHOT TREN HUAN LUYEN — dung cong thuc KhoangConformal
    nguong = float(np.quantile(f_tr.az.values, MUC_PHU))
    y_tr = (f_tr.az.values > nguong).astype(int)
    y_kd = (f_kd.az.values > nguong).astype(int)

    cot = [c for c in f.columns if c not in ("Date", "az")]
    clf = GradientBoostingClassifier(random_state=SEED, max_depth=2, n_estimators=150)
    clf.fit(f_tr[cot].values, y_tr)
    p_kd = clf.predict_proba(f_kd[cot].values)[:, 1]

    p_khihauhoc = np.full(len(y_kd), y_tr.mean())
    brier_mh = brier_score_loss(y_kd, p_kd)
    brier_kh = brier_score_loss(y_kd, p_khihauhoc)
    bss = 1 - brier_mh / brier_kh
    auc = roc_auc_score(y_kd, p_kd) if len(set(y_kd)) > 1 else float("nan")
    dt = sorted(zip(cot, clf.feature_importances_), key=lambda x: -x[1])[:5]

    return {
        "n_huanluyen": int(len(f_tr)), "n_kiemdinh": int(len(f_kd)),
        "nguong_az": round(nguong, 4),
        "ty_le_vipham_huanluyen": round(float(y_tr.mean()), 4),
        "ty_le_vipham_kiemdinh": round(float(y_kd.mean()), 4),
        "brier_mohinh": round(float(brier_mh), 5), "brier_khihauhoc": round(float(brier_kh), 5),
        "bss": round(float(bss), 4), "auc": round(float(auc), 4),
        "dat_H_META": bool(bss > 0),
        "dac_trung_quan_trong_nhat": [(c, round(float(v), 4)) for c, v in dt],
    }


def main():
    t0 = time.time()
    print("=" * 96)
    print(f"META-LABELING — dự đoán phiên nào KHOẢNG CONFORMAL {MUC_PHU:.0%} SẼ VI PHẠM")
    print("Nhắm vào lỗ hổng ĐÃ BIẾT: 'phủ thiếu khi đang lỗ' — 5 lần thử trước đều")
    print("không sửa được vì chỉ nới BỀ RỘNG; ở đây thử dự đoán ĐIỀU KIỆN vi phạm.")
    print("=" * 96)

    pan_all = pd.read_csv(PANEL, parse_dates=["Date"])
    ra = {}
    dat = 0
    for p in PAIRS:
        print(f"\n[{p}]")
        kq = chay_mot_cap(p, pan_all)
        ra[p] = kq
        if "loi" in kq:
            print("  lỗi:", kq["loi"]); continue
        print(f"  tỉ lệ vi phạm: huấn luyện={100*kq['ty_le_vipham_huanluyen']:.1f}% "
              f"· kiểm định={100*kq['ty_le_vipham_kiemdinh']:.1f}%")
        print(f"  BSS = {kq['bss']:+.4f}  ·  AUC = {kq['auc']:.4f}  "
              f"->  {'ĐẠT H_META' if kq['dat_H_META'] else 'KHÔNG ĐẠT'}")
        print("  đặc trưng quan trọng nhất:",
              ", ".join(f"{c}={v}" for c, v in kq["dac_trung_quan_trong_nhat"][:3]))
        dat += int(kq["dat_H_META"])

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "metalabel_phuconformal.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)

    print("\n" + "-" * 96)
    print(f"→ ĐẠT H_META (BSS>0) ở {dat}/{len(PAIRS)} cặp trên đoạn kiểm định.")
    print(f"→ output/metalabel_phuconformal.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
