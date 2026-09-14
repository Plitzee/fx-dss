"""PHA 3B — TAM GIAC HOA DOC LAP bang PCMCI THAT (thu vien `tigramite` cua
chinh Jakob Runge, tac gia duoc trich [18] trong de cuong), doi chieu voi
Level 2 (Granger nested-regression + Westfall-Young) va Level 3 (PCMCI RUT
GON tu viet) cua `pha3b_granger.py`.

KHONG mo lai quyet dinh da dong bang cua Pha3B. File nay la MOT KIEM TRA
ROBUSTNESS o BEN NGOAI protocol: dung LAI dung cac ham dung du lieu da co
(`pha3b_dactrung.py`, `pha3b_granger.dung_bang`) de dam bao apples-to-apples,
KHONG dua ra hypothesis moi, KHONG anh huong toi tap E3 (14 dac trung) da
chot. Muc dich la tra loi truc tiep cau hoi "lam sao biet day la nhan qua
that chu khong phai tuong quan ngau nhien":

  1. PCMCI dieu kien tren TOAN BO tap ung vien DONG THOI (khong phai tung
     cap doi-mot nhu Granger/nested-regression), nen mot bien chi "an theo"
     bien khac (confounding) se BI LOAI ngay trong buoc PC (phat hien tap
     cha) hoac trong buoc MCI (kiem dinh dieu kien tren tap cha da tim).
  2. tau_min=1 — CHI tim quan he TRE, giu dung gioi han da khai bao o Level
     2/3 (khong xu ly contemporaneous causality) de so sanh cong bang.
  3. FDR-BH tren TOAN BO lien ket da kiem (ham `get_corrected_pvalues` cua
     chinh tigramite) — kiem soat da kiem dinh, dung tinh than voi
     Westfall-Young cua Level 2.

Chay:  python src/pha3b_pcmci_doclap.py
Ghi:   output/pha3b_pcmci_doclap.json
Can:   pip install tigramite
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import pha3b_dactrung as DT                                  # noqa: E402
import pha3b_granger as G                                    # noqa: E402

from tigramite import data_processing as pp                  # noqa: E402
from tigramite.pcmci import PCMCI                             # noqa: E402
from tigramite.independence_tests.parcorr import ParCorr      # noqa: E402

TAU_MAX = 5
# AUDUSD: cap manh nhat theo Level 3 rut gon (level3_top). EURUSD, USDJPY:
# doi chieu — hai cap KHONG noi bat trong level3_top, kiem xem PCMCI that
# co "phat hien bua" (tuc mat kiem soat sai so loai I) hay khong.
CAP_KIEM = ["AUDUSD", "EURUSD", "USDJPY"]
# Ba ung vien manh nhat trong 14 dac trung song sot cua Level 2 + top cua
# Level 3 rut gon (VIXCLS, GVZCLS deu la chi so bien dong; DFII10 la loi
# suat thuc — ba loai khac nhau, hop ly de kiem tra confounding lan nhau).
BIEN_NGOAI_SINH = ["VIXCLS_lv", "GVZCLS_lv", "DFII10_lv"]


def _chuoi_mot_cap(bang, tt, pair):
    d = bang[bang.pair == pair].sort_values("ngay").copy()
    d = d[d.doan.isin([0, 1])]              # CHI huan luyen+kiem dinh, giong Level 2
    d = d.set_index("ngay")
    ex = tt.reindex(d.index)[BIEN_NGOAI_SINH]
    return pd.concat([d[["y_bien_do", "m_har"]], ex], axis=1).dropna()


def _chay_pcmci(df, var_names):
    arr = df[var_names].values.astype(float)
    dataframe = pp.DataFrame(arr, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"),
                  verbosity=0)
    res = pcmci.run_pcmci(tau_min=1, tau_max=TAU_MAX, pc_alpha=0.05)
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=res["p_matrix"], tau_min=1,
                                            tau_max=TAU_MAX, fdr_method="fdr_bh")
    return res["p_matrix"], res["val_matrix"], q_matrix


def chay():
    print("Đang dựng bảng mục tiêu (dùng lại pha3b_granger.dung_bang)...")
    bang = G.dung_bang()
    tt = DT.bien_doi_thi_truong()

    ra = {}
    for pair in CAP_KIEM:
        df = _chuoi_mot_cap(bang, tt, pair)
        var_names = ["y_bien_do", "m_har"] + BIEN_NGOAI_SINH
        p_matrix, val_matrix, q_matrix = _chay_pcmci(df, var_names)

        j = var_names.index("y_bien_do")
        song_sot = []
        for i, ten in enumerate(var_names):
            if i == j:
                continue
            for tau in range(1, TAU_MAX + 1):
                q = float(q_matrix[i, j, tau])
                if q < 0.05:
                    song_sot.append(dict(bien=ten, tau=tau,
                                         r_mci=round(float(val_matrix[i, j, tau]), 4),
                                         p=float(p_matrix[i, j, tau]), q_fdr=q))

        # ── kiem tra confounding: GVZCLS mot minh so voi GVZCLS + VIXCLS ──
        # (chi chay cho cap co du lieu manh nhat, AUDUSD, de khong nhan ban
        # tinh toan; day la minh hoa co che, khong phai kiem dinh chinh)
        confound = None
        if pair == "AUDUSD":
            p1, v1, q1 = _chay_pcmci(df, ["y_bien_do", "m_har", "GVZCLS_lv"])
            p2, v2, q2 = _chay_pcmci(df, ["y_bien_do", "m_har", "VIXCLS_lv", "GVZCLS_lv"])
            j1 = ["y_bien_do", "m_har", "GVZCLS_lv"].index("y_bien_do")
            j2 = ["y_bien_do", "m_har", "VIXCLS_lv", "GVZCLS_lv"].index("y_bien_do")
            i1 = ["y_bien_do", "m_har", "GVZCLS_lv"].index("GVZCLS_lv")
            i2 = ["y_bien_do", "m_har", "VIXCLS_lv", "GVZCLS_lv"].index("GVZCLS_lv")
            q_don = float(min(q1[i1, j1, tau] for tau in range(1, TAU_MAX + 1)))
            q_chung = float(min(q2[i2, j2, tau] for tau in range(1, TAU_MAX + 1)))
            confound = dict(
                gvzcls_mot_minh_q_min=round(q_don, 6),
                gvzcls_dieu_kien_tren_vixcls_q_min=round(q_chung, 6),
                ket_luan=("GVZCLS có ý nghĩa khi đứng một mình nhưng MẤT ý nghĩa khi "
                          "điều kiện thêm trên VIXCLS — dấu hiệu confounding qua chỉ số "
                          "biến động chung, không phải quan hệ nhân quả riêng của vàng.")
                if q_don < 0.05 <= q_chung else
                "không quan sát thấy khác biệt rõ giữa hai cấu hình")

        ra[pair] = dict(n=len(df), lien_ket_song_sot_fdr05=song_sot, kiem_tra_confounding=confound)
        print(f"\n=== {pair} (n={len(df)}) ===")
        for s in song_sot:
            print(f"  {s['bien']:>12s} -> y_bien_do @ tau={s['tau']}  "
                  f"r_MCI={s['r_mci']:+.4f}  q(FDR)={s['q_fdr']:.2e}")
        if confound:
            print("  [confounding]", confound["ket_luan"] if isinstance(confound, dict)
                  else confound)

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "pha3b_pcmci_doclap.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("\nĐã ghi", outp)
    return ra


if __name__ == "__main__":
    chay()
