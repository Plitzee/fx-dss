"""PHA 3B — HIỆU ỨNG NHÂN QUẢ BẰNG DOUBLE MACHINE LEARNING (Chernozhukov,
Chetverikov, Demirer, Duflo, Hansen, Newey & Robins, 2018, "Double/debiased
machine learning for treatment and structural parameters," The Econometrics
Journal 21(1)) — thư viện gốc `doubleml` của chính nhóm tác giả.

VÌ SAO CẦN THÊM DML SAU Granger + PCMCI:
  Granger (Level 2) và PCMCI thật (`pha3b_pcmci_doclap.py`) đều dùng hồi quy
  TUYẾN TÍNH (OLS / ParCorr) cho phần "kiểm soát nhiễu nền" — nếu quan hệ
  thật giữa VIX và biến động là PHI TUYẾN, hai phương pháp đó có thể bỏ sót
  hoặc ước lượng sai độ lớn hiệu ứng. DML dùng mô hình học máy LINH HOẠT
  (gradient boosting) cho các hàm phiền toái (nuisance functions) — hồi quy
  outcome theo biến kiểm soát, và hồi quy treatment theo biến kiểm soát —
  rồi "gỡ nhiễu kép" (cross-fitting + Neyman orthogonality) để hệ số nhân
  quả CÒN LẠI có suy diễn hợp lệ (sai số chuẩn, khoảng tin cậy) NGAY CẢ KHI
  hai hàm phiền toái đó phi tuyến/phức tạp. Đây là khác biệt cốt lõi so với
  ParCorr của PCMCI: PCMCI kiểm TỒN TẠI QUAN HỆ, DML ước lượng ĐỘ LỚN hiệu
  ứng một cách vững (robust) với mô hình phiền toái sai dạng hàm.

CÂU HỎI: tăng 1 độ lệch chuẩn VIXCLS (đã chuẩn hoá) tại t làm log RV(t+1)
tăng/giảm BAO NHIÊU, sau khi đã "gỡ" phần biến động do mốc HAR, GVZCLS,
DFII10 giải thích được — đây là ĐỘ LỚN hiệu ứng nhân quả, không chỉ "có hay
không" như kiểm định giả thuyết thuần tuý.

KHÔNG mở lại quyết định đã đóng băng của Pha3B — đây là tam giác hoá bằng
phương pháp thứ BA (sau Granger, PCMCI thật), dùng CÙNG một câu hỏi, cùng dữ
liệu (huấn luyện+kiểm định, không chạm kiểm tra/niêm phong).

Chạy:  python src/pha3b_dml_causal.py
Ghi:   output/pha3b_dml_causal.json
Cần:   pip install doubleml
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import pha3b_dactrung as DT                                  # noqa: E402
import pha3b_granger as G                                    # noqa: E402

import doubleml as dml                                        # noqa: E402

SEED = 20260914
CAP_KIEM = ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCAD", "USDCHF"]
DIEU_TRI = "VIXCLS_lv"
KIEM_SOAT = ["m_har", "GVZCLS_lv", "DFII10_lv"]


def _chuoi_mot_cap(bang, tt, pair):
    d = bang[bang.pair == pair].sort_values("ngay").copy()
    d = d[d.doan.isin([0, 1])]              # CHI huan luyen+kiem dinh, giong Level 2/PCMCI
    d = d.set_index("ngay")
    ex = tt.reindex(d.index)[[DIEU_TRI] + KIEM_SOAT[1:]]
    return pd.concat([d[["y_bien_do", "m_har"]], ex], axis=1).dropna()


def _dml_mot_cap(df, pair, seed):
    data = dml.DoubleMLData(df, y_col="y_bien_do", d_cols=DIEU_TRI, x_cols=KIEM_SOAT)
    ml_l = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=seed)
    ml_m = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=seed)
    np.random.seed(seed)
    plr = dml.DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=5, n_rep=3, score="partialling out")
    plr.fit()
    return dict(pair=pair, n=len(df),
               he_so=round(float(plr.coef[0]), 5),
               se=round(float(plr.se[0]), 5),
               t=round(float(plr.t_stat[0]), 4),
               p=float(plr.pval[0]),
               ci_95=[round(float(x), 5) for x in plr.confint().values[0]])


def chay():
    print("Đang dựng bảng mục tiêu (dùng lại pha3b_granger.dung_bang)...")
    bang = G.dung_bang()
    tt = DT.bien_doi_thi_truong()
    # chuan hoa VIX + kiem soat theo z-score TREN HUAN LUYEN cua chinh chuoi,
    # de he so "1 don vi" co y nghia so sanh duoc giua cac cap
    tr_mask = G.doan(tt.index.values) == 0
    mu, sd = tt[tr_mask].mean(), tt[tr_mask].std()
    tt_z = (tt - mu) / sd

    ra = {}
    for pair in CAP_KIEM:
        df = _chuoi_mot_cap(bang, tt_z, pair)
        r = _dml_mot_cap(df, pair, SEED)
        ra[pair] = r
        print(f"{pair}: n={r['n']}  hệ số(θ)={r['he_so']:+.5f}  "
              f"SE={r['se']:.5f}  p={r['p']:.4f}  CI95%={r['ci_95']}")

    ps = [ra[p]["p"] for p in CAP_KIEM]
    rej_holm, q_holm, _, _ = multipletests(ps, alpha=0.05, method="holm")
    n_song_sot = int(sum(rej_holm))
    for p, r, rh, qh in zip(CAP_KIEM, ps, rej_holm, q_holm):
        ra[p]["q_holm"] = round(float(qh), 5)
        ra[p]["song_sot_holm"] = bool(rh)

    tong = dict(dieu_tri=DIEU_TRI, kiem_soat=KIEM_SOAT,
               n_song_sot_holm=n_song_sot, n_cap=len(CAP_KIEM),
               ket_luan=(f"{n_song_sot}/{len(CAP_KIEM)} cặp có hiệu ứng nhân quả "
                        f"VIX→biến_động sống sót sau Holm — "
                        + ("ĐẠT ngưỡng độ vững ≥5/6 của Pha3B"
                           if n_song_sot >= 5 else "CHƯA đạt ngưỡng độ vững ≥5/6 của Pha3B")))
    ra["_tong_ket"] = tong
    print("\n" + json.dumps(tong, ensure_ascii=False, indent=1))

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "pha3b_dml_causal.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("Đã ghi", outp)
    return ra


if __name__ == "__main__":
    chay()
