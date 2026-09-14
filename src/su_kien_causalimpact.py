"""THI DIEM CausalImpact (Brodersen et al. 2015, Bayesian structural time
series — ban `pycausalimpact`) tren su kien ECB — phuong phap nhan qua thu
BA (sau Granger va PCMCI), o CAP DO TUNG SU KIEN thay vi cap do dac trung.

CAU HOI KHAC voi Pha3B: Pha3B hoi "bien ngoai sinh X co du bao duoc bien dong
KHONG" (dac trung, gop toan chuoi). O day hoi "PHIEN HOP CU THE nay co day
bien dong THUC SU vuot qua muc phan-nen (counterfactual) hay khong" — dung
dung cho tung su kien rieng le, khop voi hang muc "su kien theo tung cap voi
phan hoi lich su do duoc" da hua trong de cuong (System Implementation).

THIET KE:
  phan ung   : log RV(EURUSD)
  hiep bien  : log RV(USDCAD) — dai dien che do bien dong CHUNG nhung it
               chiu tac dong TRUC TIEP tu tin ECB hon (dong CAD gan voi
               dau/BoC/Fed hon la ECB). Neu ECB thuc su la nguyen nhan RIENG
               cho EURUSD, hieu ung phai vuot qua phan mo hinh da giai thich
               duoc tu chinh dong bien dong chung nay.
  pre_period : 30 phien truoc phien hop (khong tinh ngay hop)
  post_period: phien hop + 2 phien sau (bien dong sau tin thuong tat nhanh)

HAN CHE TU KHAI BAO: ECB va USDCAD khong tach biet hoan toan (ca hai deu
chiu anh huong tu che do rui-ro-toan-cau), nen hiep bien nay KHONG loai het
confounding — day la THACH THUC CO BAN cua CausalImpact voi su kien vi mo
duoc cong bo rong (moi tai san deu phan ung it nhieu). Phu hop hon voi su
kien DAC THU mot dong tien (vd can thiep rieng cua BOJ/SNB) hon la FOMC/ECB.
Ket qua duoi day la THI DIEM phuong phap, khong phai ket luan chinh thuc cua
luan van — CHUA dua vao quyet dinh nao cua Pha3B.

Chay:  python src/su_kien_causalimpact.py
Ghi:   output/su_kien_causalimpact_ecb.json
Can:   pip install pycausalimpact statsmodels
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from causalimpact import CausalImpact
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

PRE_N, POST_N = 30, 3
N_SU_KIEN = 20          # 20 phien hop gan moc kiem tra nhat, van trong huan luyen+kiem dinh


def _chuoi_rv():
    rv = pd.read_csv(os.path.join(DATA, "rv_adv.csv"), parse_dates=["Date"])
    piv = rv.pivot_table(index="Date", columns="pair", values="rv5")
    lrv = np.log(np.maximum(piv, 1e-12))
    return lrv.asfreq("B").ffill(limit=2)   # luoi ngay lam viec, bu le <=2 phien


def _ngay_hop(ma="ECB", truoc=None):
    cb = pd.read_csv(os.path.join(DATA, "cb_dates.csv"), parse_dates=["date"])
    d = sorted(cb[cb.bank == ma].date)
    if truoc is not None:
        d = [x for x in d if x < truoc]
    return d


def chay(ma="ECB", phan_ung="EURUSD", hiep_bien="USDCAD"):
    from split import TEST_TU        # tranh vong lap import khi goi truc tiep
    lrv = _chuoi_rv()
    ngay = _ngay_hop(ma, truoc=TEST_TU)[-N_SU_KIEN:]

    ket = []
    idx = lrv.index
    for d0 in ngay:
        pos = idx.searchsorted(d0)
        if pos >= len(idx) or pos < PRE_N or pos + POST_N >= len(idx):
            continue
        df = lrv.loc[idx[pos - PRE_N]: idx[pos + POST_N - 1],
                     [phan_ung, hiep_bien]].dropna().reset_index(drop=True)
        if len(df) < PRE_N + POST_N - 3:
            continue
        try:
            ci = CausalImpact(df, [0, PRE_N - 1], [PRE_N, PRE_N + POST_N - 1],
                              model_args={"nseasons": 5})
            s = ci.summary_data
            ket.append(dict(ngay=str(d0.date()),
                            hieu_ung_trung_binh=round(float(s.loc["abs_effect", "average"]), 4),
                            p_gia_tri=round(float(ci.p_value), 4)))
        except Exception as e:
            ket.append(dict(ngay=str(d0.date()), loi=str(e)[:150]))

    hop_le = [k for k in ket if "loi" not in k]
    if hop_le:
        rej, q, _, _ = multipletests([k["p_gia_tri"] for k in hop_le],
                                     alpha=0.05, method="fdr_bh")
        for k, r, qq in zip(hop_le, rej, q):
            k["q_fdr"] = round(float(qq), 4)
            k["co_y_nghia_fdr"] = bool(r)

    n_sig = sum(k.get("co_y_nghia_fdr", False) for k in hop_le)
    n_pos = sum(1 for k in hop_le if k.get("co_y_nghia_fdr") and k["hieu_ung_trung_binh"] > 0)
    print(f"{ma}: {len(hop_le)}/{len(ngay)} phiên hợp lệ · "
          f"{n_sig} có ý nghĩa sau FDR-BH ({n_pos} cùng chiều dương)")
    for k in ket:
        print(" ", k)

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, f"su_kien_causalimpact_{ma.lower()}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(dict(ma=ma, phan_ung=phan_ung, hiep_bien=hiep_bien,
                       pre_n=PRE_N, post_n=POST_N, ket_qua=ket,
                       tom_tat=dict(n=len(hop_le), n_y_nghia_fdr=int(n_sig),
                                   n_cung_chieu_duong=int(n_pos))),
                 f, ensure_ascii=False, indent=1)
    print("Đã ghi", outp)
    return ket


if __name__ == "__main__":
    chay()
