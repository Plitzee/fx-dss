"""XUAT BANG THI NGHIEM — results/experiment_summary.csv + ba bang bao cao.

Ke hoach Pha 1 cua HuyH (muc 11 "Experiment tracking" va muc "Phase 1 research
outputs") doi mot bang thi nghiem duy nhat theo doi moi lan chay, va ba bang
bao cao:

    Bang A — cac nen (naive, momentum, logistic, XGBoost/LightGBM)
    Bang B — ablation tri tue lich su (co ban / + HMM / + Matrix Profile / + mau)
    Bang C — do ben (walk-forward, cheo cap, hieu chuan)

Repo da co day du so lieu nhung nam rai rac trong output/*.json. File nay GOM
lai, khong tinh lai gi — nen no chay trong vai giay va luon dong bo voi lan
chay gan nhat cua tung script.

LUU Y VE do CHINH XAC va MACRO F1. Ke hoach doi hai chi so nay nen chung co
trong bang. Nhung quyet dinh cua repo van dua tren log score / Brier / BSS, vi
ly do da ghi o docs/CHISO_DANHGIA.md muc 1: hai chi so tren NHAN CUNG vut bo
toan bo phan phoi, ma san pham giao cho nguoi dung lai la BA con so xac suat.

Chay:  python src/xuat_experiment.py
Ghi:   results/experiment_summary.csv
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "output")
RES = os.path.join(ROOT, "results")

from split import VALID_TU, TEST_TU                           # noqa: E402

CAP = "EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF"
KHOANG = dict(train=f"2012-02-14..{str(VALID_TU)[:10]}",
              valid=f"{str(VALID_TU)[:10]}..{str(TEST_TU)[:10]}",
              test=f"{str(TEST_TU)[:10]}..2026-09")


def _doc(ten):
    d = os.path.join(OUT, ten)
    if not os.path.exists(d):
        return None
    with open(d, encoding="utf-8") as f:
        return json.load(f)


def _hang(**kw):
    r = dict(experiment_id="", phase="1", model="", feature_set="",
             pairs=CAP, train_range=KHOANG["train"],
             valid_range=KHOANG["valid"], test_range=KHOANG["test"],
             eval_segment="", horizon="", hyperparameters="", n="",
             accuracy="", macro_f1="", log_loss="", brier_score="",
             bss="", calibration_error="", auc_huong="", notes="")
    r.update(kw)
    return r


def bang_A(hang):
    """Cac nen Giai doan 1 + ML/DL — output/nen3.json, output/ml3.json."""
    nen = _doc("nen3.json")
    if nen:
        for k, v in nen.items():
            if "chedo" in k:
                continue
            bo = k.split("_")
            h, mt, mo = bo[0][1:], bo[1], "_".join(bo[2:])
            hang.append(_hang(
                experiment_id=f"G1_{k}", model=mo, feature_set="σ̂ (HAR vòng 7)",
                eval_segment="kiểm định", horizon=h,
                n=v.get("n", ""), accuracy=v.get("chinh_xac", ""),
                macro_f1=v.get("f1_vi_mo", ""), log_loss=v.get("log", ""),
                brier_score=v.get("brier", ""), bss=v.get("bss", ""),
                calibration_error=v.get("ece", ""), auc_huong=v.get("auc", ""),
                notes=f"mục tiêu {mt}; KTC BSS [{v.get('bss_lo','')}, "
                      f"{v.get('bss_hi','')}]"))
    ml = _doc("ml3.json")
    if ml:
        for mt, khoi in ml.items():
            if not isinstance(khoi, dict):
                continue
            for doan_ten, models in khoi.items():
                if doan_ten not in ("kiem_dinh", "kiem_tra") or not isinstance(models, dict):
                    continue
                for mo, v in models.items():
                    if not isinstance(v, dict):
                        continue
                    hang.append(_hang(
                        experiment_id=f"ML_{mt}_{doan_ten}_{mo}", model=mo,
                        feature_set="12 đặc trưng đọc được + σ̂",
                        eval_segment="kiểm định" if doan_ten == "kiem_dinh" else "kiểm tra",
                        horizon="1", n=v.get("n", ""),
                        accuracy=v.get("chinh_xac", ""), macro_f1=v.get("f1_vi_mo", ""),
                        log_loss=v.get("log", ""), brier_score=v.get("brier", ""),
                        bss=v.get("bss", ""), calibration_error=v.get("ece", ""),
                        auc_huong=v.get("auc", ""), notes=f"mục tiêu {mt}"))


def bang_B(hang):
    """Ablation tri tue lich su — bay ho khai pha quy luat."""
    ho = [("H1_ngưỡng đặc trưng", "quyluat.json", "630 vị từ 1-2 mệnh đề"),
          ("H2_motif (KMeans VQ)", "h2_motif.json", "24 cụm hình dạng"),
          ("H3_rule-list (CART)", "h3_rulelist.json", "8 lá CART sâu 3"),
          ("H5_chế độ tự tương quan", "h5_chedo.json", "630 vị từ × 3 chế độ"),
          ("H6_HMM", "h6_hmm.json", "trạng thái ẩn K=2,3,4"),
          ("H7_Matrix Profile", "h7_matrixprofile.json", "analog K=20, L=5/10/20")]
    spa = _doc("spa_ho2.json") or {}
    khop_spa = {"H2": "H2_motif", "H3": "H3_rulelist", "H5": "H5_chedo",
                "H6": "H6_hmm", "H7": "H7_matrixprofile"}
    for ten, tep, dt in ho:
        d = _doc(tep)
        if not d:
            continue
        ma = ten.split("_")[0]
        s = spa.get(khop_spa.get(ma, ""), {})
        hang.append(_hang(
            experiment_id=f"G2_{ma}", model=ten.split("_", 1)[1],
            feature_set=dt, eval_segment="huấn luyện+kiểm định (phát hiện)",
            horizon="1", n=d.get("khong_gian", ""),
            notes=(f"phễu: {d.get('khong_gian','?')} giả thuyết → "
                   f"{d.get('tho','?')} thô p<0,05 → {d.get('wy','?')} sống sót W-Y → "
                   f"{d.get('sau_dieu_kien','?')} sau đối chứng → "
                   f"{d.get('xac_nhan','?')} tái lập kiểm tra"
                   + (f"; SPA p={s.get('p'):.3f}" if s.get("p") is not None else ""))))


def bang_C(hang):
    """Do ben — walk-forward, luc pheu, hieu chuan."""
    wf = _doc("walkforward.json")
    if wf:
        for k, v in wf.items():
            for nam, r in v.get("theo_nam", {}).items():
                hang.append(_hang(
                    experiment_id=f"WF_{k}_{nam}", model=v.get("mo_hinh", ""),
                    feature_set="σ̂ (HAR vòng 7)", eval_segment=r.get("doan", ""),
                    horizon=k[1:], n=r.get("n", ""),
                    accuracy=r.get("chinh_xac", ""), macro_f1=r.get("f1_vi_mo", ""),
                    log_loss=r.get("log", ""), brier_score=r.get("brier", ""),
                    bss=r.get("bss", ""), calibration_error=r.get("ece", ""),
                    auc_huong=r.get("auc", ""), notes=f"walk-forward năm {nam}"))
    kp = _doc("kiem_pheu.json")
    if kp:
        mdes = kp.get("lift_nho_nhat_luc80", "?")
        am = kp.get("doi_chung_am", [])
        gia = (sum(x.get("qua_dieu_kien", 0) for x in am) / len(am)) if am else "?"
        hang.append(_hang(
            experiment_id="ROBUST_luc_pheu", model="phễu Giai đoạn 2",
            feature_set="đối chứng âm + dương",
            eval_segment="huấn luyện+kiểm định", horizon="1",
            notes=f"MDES lực 80% = lift {mdes}; "
                  f"dương tính giả trên nhiễu thuần: {gia}"))


def main():
    hang = []
    bang_A(hang); bang_B(hang); bang_C(hang)
    df = pd.DataFrame(hang)
    os.makedirs(RES, exist_ok=True)
    duong = os.path.join(RES, "experiment_summary.csv")
    df.to_csv(duong, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print(f"ĐÃ XUẤT results/experiment_summary.csv — {len(df)} dòng")
    print("=" * 100)

    wf = _doc("walkforward.json") or {}
    spa = _doc("spa_ho2.json") or {}

    print("\nBẢNG A — các nền (kiểm định, mục tiêu P, h=1)")
    print(f"  {'mô hình':<24}{'log':>9}{'BSS':>9}{'chính xác':>11}"
          f"{'macro F1':>10}{'ECE':>8}{'AUC':>8}")
    nen = _doc("nen3.json") or {}
    for k, v in nen.items():
        if not k.startswith("h1_P_") or "chedo" in k:
            continue
        print(f"  {k[5:]:<24}{v.get('log',0):>9.4f}{v.get('bss',0):>+9.4f}"
              f"{v.get('chinh_xac',0):>11.4f}{v.get('f1_vi_mo',0):>10.4f}"
              f"{v.get('ece',0):>8.4f}{v.get('auc',0):>8.4f}")

    print("\nBẢNG B — ablation trí tuệ lịch sử (phễu khai phá quy luật)")
    print(f"  {'họ':<26}{'giả thuyết':>12}{'thô':>7}{'W-Y':>6}"
          f"{'sau đ/c':>9}{'kiểm tra':>10}{'SPA p':>8}")
    khop = {"quyluat.json": ("H1 ngưỡng đặc trưng", None),
            "h2_motif.json": ("H2 motif", "H2_motif"),
            "h3_rulelist.json": ("H3 rule-list", "H3_rulelist"),
            "h5_chedo.json": ("H5 chế độ tự t.quan", "H5_chedo"),
            "h6_hmm.json": ("H6 HMM", "H6_hmm"),
            "h7_matrixprofile.json": ("H7 Matrix Profile", "H7_matrixprofile")}
    for tep, (ten, kspa) in khop.items():
        d = _doc(tep)
        if not d:
            continue
        p = spa.get(kspa, {}).get("p") if kspa else None
        print(f"  {ten:<26}{d.get('khong_gian',0):>12,}{d.get('tho',0):>7}"
              f"{d.get('wy',0):>6}{d.get('sau_dieu_kien',0):>9}"
              f"{d.get('xac_nhan',0):>10}"
              f"{(f'{p:.3f}' if p is not None else '—'):>8}")

    print("\nBẢNG C — độ bền")
    print(f"  {'tầm hạn':<10}{'năm BSS dương':>16}{'BSS trung vị':>15}{'khoảng BSS':>26}")
    for k, v in wf.items():
        ty = f"{v.get('so_nam_duong','?')}/{v.get('so_nam','?')}"
        kho = f"[{v.get('bss_min',0):+.4f}; {v.get('bss_max',0):+.4f}]"
        print(f"  h={k[1:]:<8}{ty:>16}{v.get('bss_trung_vi',0):>+15.4f}{kho:>26}")
    kp = _doc("kiem_pheu.json")
    if kp:
        am = kp.get("doi_chung_am", [])
        gia = (sum(x.get("qua_dieu_kien", 0) for x in am) / len(am)) if am else "?"
        print(f"  lực phễu: MDES (80%) = lift {kp.get('lift_nho_nhat_luc80','?')} · "
              f"dương tính giả trên nhiễu thuần {gia}")
    print(f"\n→ {duong}")


if __name__ == "__main__":
    main()
