"""PHA 3B — mo rong `pha3b_pcmci_doclap.py` sang PCMCI+ (tigramite that,
`run_pcmciplus`) de tra loi cau hoi ma ban PCMCI cu CHU DINH bo qua:

    "Co quan he DONG THOI (tau=0) giua cac ung vien ngoai sinh va bien do
    hay khong, va neu co thi PCMCI+ co dinh huong duoc canh do khong?"

`pha3b_pcmci_doclap.py` dat tau_min=1 CO CHU DICH de so sanh cong bang voi
Level 2/3 (hai muc nay khong xu ly dong thoi). File nay KHONG thay the ket
qua da chot — chi la MOT KIEM TRA THEM (tau_min=0), dung LAI y het du lieu
va tap ung vien cua ban tau_min=1 de doi chieu truc tiep tau=0 vs tau>=1.

Chay:  python src/pha3b_pcmci_plus.py
Ghi:   output/pha3b_pcmci_plus.json
Can:   pip install tigramite (da co san, v5.2.10.1)
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import pha3b_granger as G                                     # noqa: E402
import pha3b_pcmci_doclap as P0                                # noqa: E402
import pha3b_dactrung as DT                                    # noqa: E402

from tigramite import data_processing as pp                   # noqa: E402
from tigramite.pcmci import PCMCI                              # noqa: E402
from tigramite.independence_tests.parcorr import ParCorr       # noqa: E402

TAU_MAX = P0.TAU_MAX
CAP_KIEM = P0.CAP_KIEM
BIEN_NGOAI_SINH = P0.BIEN_NGOAI_SINH


def _chay_pcmciplus(df, var_names):
    arr = df[var_names].values.astype(float)
    dataframe = pp.DataFrame(arr, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"),
                  verbosity=0)
    res = pcmci.run_pcmciplus(tau_min=0, tau_max=TAU_MAX, pc_alpha=0.05,
                               fdr_method="fdr_bh")
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=res["p_matrix"], tau_min=0,
                                            tau_max=TAU_MAX, fdr_method="fdr_bh")
    return res["graph"], res["p_matrix"], res["val_matrix"], q_matrix


def chay():
    print("Đang dựng bảng mục tiêu (dùng lại pha3b_granger.dung_bang)...")
    bang = G.dung_bang()
    tt = DT.bien_doi_thi_truong()

    ra = {}
    for pair in CAP_KIEM:
        df = P0._chuoi_mot_cap(bang, tt, pair)
        var_names = ["y_bien_do", "m_har"] + BIEN_NGOAI_SINH
        graph, p_matrix, val_matrix, q_matrix = _chay_pcmciplus(df, var_names)
        j = var_names.index("y_bien_do")

        # ── tau=0: canh dong thoi vao y_bien_do, theo huong graph tim ra ──
        dong_thoi = []
        for i, ten in enumerate(var_names):
            if i == j:
                continue
            mac = graph[i, j, 0]
            if mac == "":
                continue
            dong_thoi.append(dict(
                bien=ten, mac_pcmciplus=mac,
                r_mci=round(float(val_matrix[i, j, 0]), 4),
                p=float(p_matrix[i, j, 0]), q_fdr=float(q_matrix[i, j, 0]),
                dinh_huong_vao_y=(mac == "-->")))

        # ── tau>=1: canh tre vao y_bien_do, doi chieu voi ban tau_min=1 ──
        tre = []
        for i, ten in enumerate(var_names):
            if i == j:
                continue
            for tau in range(1, TAU_MAX + 1):
                mac = graph[i, j, tau]
                if mac == "":
                    continue
                tre.append(dict(bien=ten, tau=tau, mac_pcmciplus=mac,
                                 r_mci=round(float(val_matrix[i, j, tau]), 4),
                                 p=float(p_matrix[i, j, tau]), q_fdr=float(q_matrix[i, j, tau])))

        ra[pair] = dict(n=len(df), canh_dong_thoi_vao_y=dong_thoi, canh_tre_vao_y=tre)
        print(f"\n=== {pair} (n={len(df)}) ===")
        print("  [tau=0, dong thoi]")
        for s in dong_thoi:
            huong = "->" if s["dinh_huong_vao_y"] else s["mac_pcmciplus"]
            print(f"    {s['bien']:>12s} {huong:>4s} y_bien_do  "
                  f"r_MCI={s['r_mci']:+.4f}  q(FDR)={s['q_fdr']:.2e}")
        print("  [tau>=1, tre — doi chieu voi ban tau_min=1]")
        for s in tre:
            print(f"    {s['bien']:>12s} -> y_bien_do @ tau={s['tau']}  "
                  f"r_MCI={s['r_mci']:+.4f}  q(FDR)={s['q_fdr']:.2e}")
        if not dong_thoi:
            print("    (không có cạnh đồng thời nào sống sót)")
        if not tre:
            print("    (không có cạnh trễ nào sống sót — khác với bản tau_min=1?)")

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "pha3b_pcmci_plus.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("\nĐã ghi", outp)
    return ra


if __name__ == "__main__":
    chay()
