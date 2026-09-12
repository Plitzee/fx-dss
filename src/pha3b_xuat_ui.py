"""PHA 3B — xuat khoi du lieu cho GIAO DIEN.

HuyH Week 4 §3 (Product evidence) cho phep UI hien bang chung ngoai sinh/nhan
qua, kem RANG BUOC NGON TU:

    nen dung : "leading indicator" · "temporal association"
               "Granger evidence" · "predictive contribution"
    tranh    : "proven cause"

File nay chi DOC output/pha3b_*.json roi gom lai — khong tinh lai gi, khong
bia so. Moi cau chu trong `ket_luan` deu phai truy duoc ve mot con so trong
`docs/PHA3B_KETQUA.md`.

Nguyen tac cua repo: ket qua AM bao cao day du nhu ket qua DUONG. Nen khoi nay
hien CA hai phan — 14 quan he song sot (duong) VA viec khong cau hinh nao thang
duoc moc (am).

Chay:  python src/pha3b_xuat_ui.py
Ghi:   output/pha3b_ui.json
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")


def main():
    can = ("pha3b_granger.json", "pha3b_ablation.json", "pha3b_mdes.json",
           "pha3b_quanhe.csv")
    for f in can:
        if not os.path.exists(os.path.join(OUT, f)):
            sys.exit(f"Thieu output/{f} — chay src/pha3b_*.py truoc.")
    G = json.load(open(os.path.join(OUT, "pha3b_granger.json"), encoding="utf-8"))
    A = json.load(open(os.path.join(OUT, "pha3b_ablation.json"), encoding="utf-8"))
    M = json.load(open(os.path.join(OUT, "pha3b_mdes.json"), encoding="utf-8"))
    R = pd.read_csv(os.path.join(OUT, "pha3b_quanhe.csv"))

    TEN = {"su_kien_phi_nhtw": "lịch công bố vĩ mô (ngoài NHTW)",
           "VIXCLS": "ẩn ý biến động cổ phiếu Mỹ (VIX)",
           "DGS2": "lợi suất TPCP Mỹ 2 năm", "DGS10": "lợi suất TPCP Mỹ 10 năm",
           "T10Y2Y": "độ dốc đường cong 10Y−2Y",
           "DFII10": "lợi suất thực 10 năm (TIPS)",
           "GVZCLS": "ẩn ý biến động vàng", "OVXCLS": "ẩn ý biến động dầu",
           "DCOILBRENTEU": "dầu Brent", "NIKKEI225": "Nikkei 225",
           "chenh_ls": "chênh lệch lãi suất chính sách"}
    TRUC = {"bien_do": "biên độ", "huong": "hướng", "rui_ro": "rủi ro"}

    song = R[R.p_wy < 0.05]
    quan_he = []
    for (truc, bien, lag), g in song.groupby(["truc", "bien", "lag"]):
        quan_he.append(dict(
            truc=TRUC.get(truc, truc), bien=TEN.get(bien, bien), lag=int(lag),
            so_cap=int(len(g)), F_max=float(g.F.max()),
            p_wy=float(g.p_wy.min()),
            dau="âm" if int(g.dau.mode().iat[0]) < 0 else "dương"))
    quan_he.sort(key=lambda x: x["p_wy"])

    ss = G["song_sot"]
    bd = A["bien_do"]
    ui = dict(
        # ── phan DUONG: phat hien quan he dan bao
        pheu=dict(khong_gian=G["n_kiem"], khai_bao=G["khai_bao"],
                  tho=ss["p_tiem_can_005"], ky_vong_nhieu=round(0.05 * G["n_kiem"]),
                  hoan_vi=ss["p_hoan_vi_005"], fdr=ss["fdr_005"], wy=ss["wy_005"],
                  do_vung=G["do_vung_dat"], vao_E3=ss["wy_005"],
                  thoi_phong=round(ss["p_tiem_can_005"] / (0.05 * G["n_kiem"]), 1),
                  n_hoan_vi=G["n_hoan_vi"], khoi=G["khoi"]),
        quan_he=quan_he,
        theo_truc={TRUC[t]: int((R.truc == t).sum()) for t in TRUC},
        song_sot_theo_truc={TRUC[t]: int(((R.truc == t) & (R.p_wy < .05)).sum())
                            for t in TRUC},
        # ── phan AM: khong cau hinh nao thang moc
        ablation=[dict(ten=k, n_dt=v["n_dt"], qlike_kt=v["qlike_kt"],
                       so_B0=v["d_kt"], dm_p=v.get("dm_p"))
                  for k, v in bd.items()],
        cau_hoi_trung_tam=dict(chenh=A["e3_vs_e2_kt"], dm_p=A["e3_vs_e2_dm_p"],
                               troi_E2=10.14, troi_E3=4.99),
        truc_huong=dict(bss_B0=A["huong"]["B0"]["bss"],
                        bss_E3=A["huong"]["E3"]["bss"],
                        auc_E3=A["huong"]["E3"]["auc"]),
        truc_rui_ro=dict(ky_nang_B0=A["rui_ro"]["B0"]["ky_nang"],
                         ky_nang_E3=A["rui_ro"]["E3"]["ky_nang"],
                         phu_B0=A["rui_ro"]["B0"]["phu80"],
                         phu_E3=A["rui_ro"]["E3"]["phu80"]),
        mdes=dict(qlike=abs(M["mdes"]["d_qlike_kt"]), luc=M["mdes"]["luc"],
                  doi_chung_am=M["doi_chung_am_dat"], nguong_F=M["nguong_F"]),
        # ── ngon tu: HuyH Week 4 §3 — khong dung tu "nguyen nhan"
        ket_luan=dict(
            duong=(f"{ss['wy_005']}/{G['n_kiem']} quan hệ **dẫn báo theo thời "
                   f"gian** sống sót kiểm soát đa kiểm định và màn lọc độ vững "
                   f"— lần đầu trong dự án. Toàn bộ nằm trên trục **biên độ**."),
            am=(f"Nhưng **không cấu hình nào thắng được mốc** trên đoạn kiểm "
                f"tra: ném hết vào +{bd['E1']['d_kt']:.1f}%, chọn đặc trưng "
                f"thường +{bd['E2']['d_kt']:.1f}%, lọc nhân quả "
                f"+{bd['E3']['d_kt']:.1f}%."),
            trung_tam=(f"Điều lọc nhân quả làm được là **phòng thủ**: so với "
                       f"chọn đặc trưng thông thường nó tốt hơn "
                       f"{abs(A['e3_vs_e2_kt']):.2f}% (p={A['e3_vs_e2_dm_p']:.4f}) "
                       f"và giảm một nửa mức sụp đổ ngoài mẫu."),
            ngon_tu=("Đây là **quan hệ dẫn báo theo thời gian** (bằng chứng "
                     "Granger), KHÔNG phải quan hệ nhân quả đã được chứng minh.")),
        nguon="docs/PHA3B_KETQUA.md")

    f = os.path.join(OUT, "pha3b_ui.json")
    json.dump(ui, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("=" * 88)
    print("PHA 3B — KHỐI DỮ LIỆU GIAO DIỆN")
    print("=" * 88)
    print(f"phễu: {ui['pheu']['khong_gian']} giả thuyết → W-Y {ui['pheu']['wy']} "
          f"(thổi phồng {ui['pheu']['thoi_phong']}×)")
    print(f"quan hệ sống sót theo trục: {ui['song_sot_theo_truc']}")
    for q in quan_he:
        print(f"  • {q['bien']:<36} lag {q['lag']} · {q['so_cap']} cặp · "
              f"F={q['F_max']:.1f} · p W-Y={q['p_wy']:.4f} · dấu {q['dau']}")
    print(f"\nablation (QLIKE kiểm tra, so mốc):")
    for a in ui["ablation"]:
        print(f"  {a['ten']:<4}{a['n_dt']:>5} đặc trưng  {a['qlike_kt']:.4f}  "
              f"{a['so_B0']:+.2f}%")
    print(f"\nMDES ≈ {ui['mdes']['qlike']:.2f}% QLIKE ở lực "
          f"{100*ui['mdes']['luc']:.0f}%")
    print(f"\n→ {f}")


if __name__ == "__main__":
    main()
