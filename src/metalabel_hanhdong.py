"""KIEM CHUNG HANH DONG cua meta-label QLIKE — buoc con thieu ghi trong
METALABEL_KETQUA.md muc 3.3: biet truoc "hom nay kho tin sigma^" khong tu
dong co ich neu khong hanh dong dung cach. O day thu HANH DONG DON GIAN
NHAT co the: khi meta-label bao xac suat "ngay te" cao, TU DONG NOI khoang
conformal len — roi DO xem do phu thuc te co gan 90% danh nghia hon khong,
so voi khoang TINH (khong doi).

THIET KE:
  - Khoang goc: nua-be-rong = quantile(|z_train|, 0,90) — dung cong thuc
    KhoangConformal.nua_be_rong().
  - Meta-label (tu metalabel_qlike.py, CO VIX) cho p_te = P(QLIKE nam
    trong nhom 20% te nhat) cho tung phien kiem dinh.
  - Hanh dong: neu p_te > NGUONG_HANHDONG (median cua p_te tren kiem
    dinh, chon KHONG nhin do phu — chi la mot quy tac chia doi mau, khong
    phai chon theo ket qua), NOI nua-be-rong len HE_SO_NOI lan.
  - So sanh: do phu thuc te (ty le |z_t| <= nua_be_rong_t) cua ban TINH
    vs ban DONG, tren KIEM DINH. Ban dong TOT HON neu do phu gan 90% hon
    VA khong phai vi no rong hon MOI NGAY (kiem tra rieng: do phu CHI
    trong nhung ngay da nong len, so voi do phu chung cua ban tinh).

Chay:  python src/metalabel_hanhdong.py
Ghi:   output/metalabel_hanhdong.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

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
MUC_PHU = 0.90
NHOM_TE = 0.20
HE_SO_NOI = 1.15          # nong khoang len 15% khi meta-label bao te


def nap_vix():
    v = pd.read_csv(NGOAI_SINH, parse_dates=["date"])[["date", "VIXCLS"]]
    v = v.rename(columns={"date": "Date"}).sort_values("Date")
    v["vix_tre1"] = v.VIXCLS.shift(1)
    return v[["Date", "vix_tre1"]]


def dac_trung_day_du(pan, vix):
    d = pan.sort_values("Date").reset_index(drop=True).copy()
    sig2 = np.maximum(d.sig.values, 1e-12) ** 2
    rv5 = np.maximum(d.rv5.values, 1e-12)
    qlike = np.log(sig2) + rv5 / sig2
    z = pd.Series(d.zT.values)
    az = z.abs()
    z_tre, az_tre = z.shift(1), az.shift(1)
    f = pd.DataFrame({"Date": d.Date, "qlike": qlike, "z": z.values})
    for w in (20, 60):
        f[f"sd_z_{w}"] = z_tre.rolling(w, min_periods=w // 2).std().values
        f[f"kurt_z_{w}"] = z_tre.rolling(w, min_periods=w).kurt().values
        f[f"tyle_vuot_1.5_{w}"] = (az_tre > 1.5).rolling(w, min_periods=w // 2).mean().values
    f["dow"] = pd.to_datetime(d.Date).dt.dayofweek.values
    f = pd.merge(f, vix, on="Date", how="left")
    f["vix_tre1"] = f["vix_tre1"].ffill()
    return f.dropna().reset_index(drop=True)


def chay_mot_cap(pair, pan_all, vix):
    pan = pan_all[pan_all.pair == pair]
    f = dac_trung_day_du(pan, vix)
    dat = pd.to_datetime(f.Date)
    m_tr = dat < VALID_TU
    m_kd = (dat >= VALID_TU) & (dat < TEST_TU)
    f_tr, f_kd = f[m_tr].reset_index(drop=True), f[m_kd].reset_index(drop=True)
    if len(f_tr) < 100 or len(f_kd) < 50:
        return {"loi": "quá ít dữ liệu"}

    # 1) meta-label — CHOT tren huan luyen
    nguong_qlike = float(np.quantile(f_tr.qlike.values, 1 - NHOM_TE))
    y_tr = (f_tr.qlike.values > nguong_qlike).astype(int)
    cot = [c for c in f.columns if c not in ("Date", "qlike", "z")]
    clf = GradientBoostingClassifier(random_state=SEED, max_depth=2, n_estimators=150)
    clf.fit(f_tr[cot].values, y_tr)
    p_te_kd = clf.predict_proba(f_kd[cot].values)[:, 1]

    # 2) khoang conformal goc — CHOT tren huan luyen
    nua_be_rong = float(np.quantile(np.abs(f_tr.z.values), MUC_PHU))

    # 3) hanh dong — nguong chia doi p_te_kd (khong nhin do phu de chon)
    nguong_hanhdong = float(np.median(p_te_kd))
    nong = p_te_kd > nguong_hanhdong

    az_kd = np.abs(f_kd.z.values)
    phu_tinh = az_kd <= nua_be_rong
    be_rong_dong = np.where(nong, nua_be_rong * HE_SO_NOI, nua_be_rong)
    phu_dong = az_kd <= be_rong_dong

    do_phu_tinh = float(phu_tinh.mean())
    do_phu_dong = float(phu_dong.mean())
    be_rong_tb_dong = float(be_rong_dong.mean())

    # do phu RIENG trong nhung ngay da nong len — de kiem tra hanh dong co
    # DUNG CHO thoi diem can hay khong (khong phai chi rong hon lung tung)
    if nong.sum() > 5:
        do_phu_tinh_khinong = float(phu_tinh[nong].mean())
        do_phu_dong_khinong = float(phu_dong[nong].mean())
    else:
        do_phu_tinh_khinong = do_phu_dong_khinong = float("nan")

    return {
        "n_kiemdinh": int(len(f_kd)), "nua_be_rong_goc": round(nua_be_rong, 4),
        "ty_le_nong": round(float(nong.mean()), 3),
        "do_phu_tinh": round(do_phu_tinh, 4), "do_phu_dong": round(do_phu_dong, 4),
        "be_rong_tb_tinh": round(nua_be_rong, 4), "be_rong_tb_dong": round(be_rong_tb_dong, 4),
        "do_phu_tinh_khinong": round(do_phu_tinh_khinong, 4) if do_phu_tinh_khinong == do_phu_tinh_khinong else None,
        "do_phu_dong_khinong": round(do_phu_dong_khinong, 4) if do_phu_dong_khinong == do_phu_dong_khinong else None,
        "gan_muc_danh_nghia_hon": bool(abs(do_phu_dong - MUC_PHU) < abs(do_phu_tinh - MUC_PHU)),
    }


def main():
    t0 = time.time()
    print("=" * 96)
    print(f"KIỂM CHỨNG HÀNH ĐỘNG — nới khoảng conformal {HE_SO_NOI}× khi meta-label báo 'tệ'")
    print("=" * 96)

    pan_all = pd.read_csv(PANEL, parse_dates=["Date"])
    vix = nap_vix()
    ra = {}
    tot_hon = 0
    for p in PAIRS:
        print(f"\n[{p}]")
        kq = chay_mot_cap(p, pan_all, vix)
        ra[p] = kq
        if "loi" in kq:
            print("  lỗi:", kq["loi"]); continue
        print(f"  độ phủ TĨNH  = {100*kq['do_phu_tinh']:.1f}%  (bề rộng cố định {kq['be_rong_tb_tinh']})")
        print(f"  độ phủ ĐỘNG  = {100*kq['do_phu_dong']:.1f}%  (bề rộng TB {kq['be_rong_tb_dong']}, "
              f"nới {100*kq['ty_le_nong']:.0f}% số phiên)")
        if kq["do_phu_tinh_khinong"] is not None:
            print(f"  riêng những ngày đã 'nóng': tĩnh={100*kq['do_phu_tinh_khinong']:.1f}% "
                  f"→ động={100*kq['do_phu_dong_khinong']:.1f}%")
        print(f"  → {'ĐỘNG gần 90% danh nghĩa hơn' if kq['gan_muc_danh_nghia_hon'] else 'KHÔNG cải thiện'}")
        tot_hon += int(kq["gan_muc_danh_nghia_hon"])

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "metalabel_hanhdong.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)

    print("\n" + "-" * 96)
    print(f"→ Hành động ĐỘNG tốt hơn TĨNH ở {tot_hon}/{len(PAIRS)} cặp.")
    print(f"→ output/metalabel_hanhdong.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
