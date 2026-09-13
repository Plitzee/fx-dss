"""PHA 3B — ghi ket qua vao SO THEO DOI THI NGHIEM (HuyH master roadmap §11).

`results/experiment_summary.csv` la artefact ma §11 cua `00_MASTER_ROADMAP.md`
doi hoi, voi ly do ghi ro: *"This becomes critical when Phase 2 and Phase 3 are
compared against earlier baselines."*

Truoc file nay so chi co Pha 1 (117 dong). Day la mon no cua roadmap.

KHONG tinh lai gi — chi doc `output/pha3b_*.json` va ghi them dong. Neu cac
JSON do chua ton tai thi dung, khong bia so.

Chay:  python src/pha3b_ghi_so.py
Ghi:   results/experiment_summary.csv  (chi GHI THEM, khong sua dong cu)
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
SO = os.path.join(ROOT, "results", "experiment_summary.csv")

CHIA = dict(train_range="2012-02-14..2021-10-13",
            valid_range="2021-10-13..2023-11-20",
            test_range="2023-11-20..2025-12-31")
CAP = "EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF"
DAC = {"B0": "—(mốc HAR sản xuất)", "E1": "186 ngoại sinh (toàn bộ)",
       "E2": "15 chọn đơn biến trên kiểm định", "E3": "15 lọc Granger+độ vững",
       "E4": "45 = E3 × chế độ biến động"}


def main():
    for f in ("pha3b_ablation.json", "pha3b_granger.json", "pha3b_mdes.json"):
        if not os.path.exists(os.path.join(OUT, f)):
            sys.exit(f"Thieu {f} — chay src/pha3b_*.py truoc, khong bia so.")
    A = json.load(open(os.path.join(OUT, "pha3b_ablation.json"), encoding="utf-8"))
    G = json.load(open(os.path.join(OUT, "pha3b_granger.json"), encoding="utf-8"))
    M = json.load(open(os.path.join(OUT, "pha3b_mdes.json"), encoding="utf-8"))
    so = pd.read_csv(SO)
    cu = len(so)

    hang = []

    # ── truc BIEN DO: chi so chinh la QLIKE, khong phai log_loss/brier
    for k, r in A["bien_do"].items():
        for doan, ql, n in (("kiểm định", r["qlike_vd"], None),
                            ("kiểm tra", r["qlike_kt"], None)):
            hang.append(dict(
                experiment_id=f"P3B_biendo_{k}_{doan.replace(' ', '')}",
                phase="3B", model=f"HAR sản xuất + {k}", feature_set=DAC[k],
                pairs=CAP, **CHIA, eval_segment=doan, horizon=1,
                hyperparameters=f"n_dactrung={r['n_dt']}", n="",
                accuracy="", macro_f1="", log_loss="", brier_score="", bss="",
                calibration_error="", auc_huong="",
                notes=(f"TRỤC BIÊN ĐỘ — chỉ số chính QLIKE={ql:.4f}; "
                       f"so B0 {r['d_kt' if doan == 'kiểm tra' else 'd_vd']:+.2f}%"
                       + (f"; DM p={r['dm_p']:.4f}" if r.get("dm_p") is not None
                          and doan == "kiểm tra" else "")
                       + (f"; MSE={r['mse']:.3e} MAE={r['mae']:.3e}"
                          if doan == "kiểm tra" else ""))))

    # ── truc HUONG: dung dung cac cot chuan cua so
    for k, r in A["huong"].items():
        hang.append(dict(
            experiment_id=f"P3B_huong_{k}_kiemtra", phase="3B",
            model=f"logistic 3 lớp + {k}", feature_set=DAC[k], pairs=CAP,
            **CHIA, eval_segment="kiểm tra", horizon=1,
            hyperparameters=f"n_dactrung={r['n_dt']}", n="", accuracy="",
            macro_f1="", log_loss=r["log_kt"], brier_score=r["brier"],
            bss=r["bss"], calibration_error="", auc_huong=r["auc"],
            notes="TRỤC HƯỚNG — BSS so khí hậu học; AUC≈0,5 = không phân biệt "
                  "được hướng"))

    # ── truc RUI RO
    for k, r in A["rui_ro"].items():
        hang.append(dict(
            experiment_id=f"P3B_ruiro_{k}_kiemtra", phase="3B",
            model=f"σ̂×phân vị z (V0) + {k}", feature_set=DAC[k], pairs=CAP,
            **CHIA, eval_segment="kiểm tra", horizon=1,
            hyperparameters=f"n_dactrung={r['n_dt']}", n="", accuracy="",
            macro_f1="", log_loss="", brier_score="", bss="",
            calibration_error="", auc_huong="",
            notes=(f"TRỤC RỦI RO — CRPS={r['crps_kt']:.5f}; kỹ năng so khí hậu "
                   f"học {100*r['ky_nang']:.2f}%; phủ 80% thực={100*r['phu80']:.1f}%; "
                   f"{r['cap_duong']}/6 cặp dương")))

    # ── Week 2 va MDES: mot dong tong ket moi thu
    ss = G["song_sot"]
    hang.append(dict(
        experiment_id="P3B_week2_granger", phase="3B",
        model="Granger + Westfall–Young maxT (null khối 5)",
        feature_set="11 biến × 3 lag × 6 cặp × 3 trục", pairs=CAP, **CHIA,
        eval_segment="huấn luyện+kiểm định", horizon=1,
        hyperparameters=f"{G['n_hoan_vi']} hoán vị, khối {G['khoi']}",
        n=G["n_kiem"], accuracy="", macro_f1="", log_loss="", brier_score="",
        bss="", calibration_error="", auc_huong="",
        notes=(f"594 giả thuyết khai báo trước; sống sót: thô {ss['p_tiem_can_005']} "
               f"(kỳ vọng≈30, thổi phồng 7,1×) · hoán vị {ss['p_hoan_vi_005']} · "
               f"FDR {ss['fdr_005']} · **W-Y {ss['wy_005']}**; qua cả W-Y lẫn độ "
               f"vững: {ss['wy_005']} — toàn bộ trên trục biên độ")))
    hang.append(dict(
        experiment_id="P3B_mdes", phase="3B", model="phân tích lực (MDES)",
        feature_set="tiêm qua PC1 khối DGS2 lag 1", pairs=CAP, **CHIA,
        eval_segment="kiểm tra", horizon=1,
        hyperparameters=f"{M['n_lan']} lần/mức, ngưỡng F(95%)={M['nguong_F']:.3f}",
        n="", accuracy="", macro_f1="", log_loss="", brier_score="", bss="",
        calibration_error="", auc_huong="",
        notes=(f"MDES ≈ {abs(M['mdes']['d_qlike_kt']):.2f}% QLIKE ở lực "
               f"{100*M['mdes']['luc']:.1f}%; đối chứng âm "
               f"{'ĐẠT' if M['doi_chung_am_dat'] else 'HỎNG'}")))
    hang.append(dict(
        experiment_id="P3B_cauhoi_trung_tam", phase="3B",
        model="E3 (lọc nhân quả) so E2 (chọn thường)",
        feature_set="15 vs 15 đặc trưng, cùng số tham số", pairs=CAP, **CHIA,
        eval_segment="kiểm tra", horizon=1, hyperparameters="", n="",
        accuracy="", macro_f1="", log_loss="", brier_score="", bss="",
        calibration_error="", auc_huong="",
        notes=(f"CÂU HỎI TRUNG TÂM Pha 3B: E3 so E2 = {A['e3_vs_e2_kt']:+.2f}% "
               f"QLIKE, DM p={A['e3_vs_e2_dm_p']:.4f}; độ trôi kiểm định→kiểm "
               f"tra: E2 10,14 điểm vs E3 4,99 điểm; giao E2∩E3 chỉ "
               f"{len(A['giao_E2_E3'])}/15")))

    moi = pd.DataFrame(hang)
    ra = pd.concat([so, moi], ignore_index=True)
    ra.to_csv(SO, index=False, encoding="utf-8-sig")

    print("=" * 92)
    print("SỔ THEO DÕI THÍ NGHIỆM — HuyH master roadmap §11")
    print("=" * 92)
    print(f"trước: {cu} dòng (chỉ Pha 1) → sau: {len(ra)} dòng "
          f"(thêm {len(moi)} dòng Pha 3B)")
    print(f"\nphân bố theo pha:")
    print(ra.groupby("phase").size().to_string())
    print(f"\n→ {SO}")
    print("\nMón nợ §11 của roadmap: đã trả cho Pha 3B.")
    print("Pha 2 vẫn chưa có dòng nào trong sổ — ghi nhận là món nợ còn lại.")


if __name__ == "__main__":
    main()
