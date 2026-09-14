"""PHA 3B — TÍCH HỢP hiệu ứng nhân quả DML vào DỰ BÁO SẢN XUẤT (không chỉ
kiểm định tồn tại quan hệ — đây là bước biến kết quả nhân quả thành một
SỐ CÓ THẬT trên dự báo, để trả lời "phần nhân quả có hiệu quả và giữ vai
trò chủ chốt trong hệ thống không" bằng một phép đo, không phải khẳng định
suông).

CÔNG THỨC — lớp phủ nhân quả (causal overlay) trên mốc HAR:

    σ̂²_có_lớp_phủ(t) = σ̂²_HAR(t) · exp(θ_cặp · VIX_chuẩn_hoá(t))

θ_cặp là hệ số nhân quả ước lượng bằng Double ML (`pha3b_dml_causal.py`),
ƯỚC LƯỢNG DUY NHẤT TRÊN ĐOẠN HUẤN LUYỆN (không dùng đoạn kiểm định) — rồi
ĐÓNG BĂNG θ đó và áp dụng lên đoạn kiểm định để đo QLIKE thật. Điều này
tránh đúng lỗi rò rỉ đã gặp và sửa ở `kiem_cnn_nen.py`: ước lượng và đánh
giá không bao giờ dùng chung một đoạn dữ liệu.

VIX_chuẩn_hoá(t) dùng thống kê chuẩn hoá ước trên CHÍNH đoạn huấn luyện của
chuỗi VIX (nhất quán với `pha3b_dactrung.py`) — không rò rỉ tương lai.

Chạy:  python src/pha3b_dml_tichhop.py
Ghi:   output/pha3b_dml_tichhop.json
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

import pha3b_dactrung as DT                                   # noqa: E402
import pha3b_granger as G                                     # noqa: E402
from metrics import qlike                                     # noqa: E402
from split import doan                                        # noqa: E402

import doubleml as dml                                         # noqa: E402

SEED = 20260914
CAP_KIEM = ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCAD", "USDCHF"]
DIEU_TRI = "VIXCLS_lv"
KIEM_SOAT = ["m_har", "GVZCLS_lv", "DFII10_lv"]


def dm_nw(x):
    """Diebold-Mariano voi phuong sai Newey-West — dung lai nguyen ham cua
    src/run_final7.py de nhat quan trong toan repo."""
    from scipy import stats
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); mb = x.mean(); L = int(np.ceil(1.5 * n ** (1 / 3)))
    s = np.sum((x - mb) ** 2) / n
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * np.sum((x[k:] - mb) * (x[:-k] - mb)) / n
    t = mb / np.sqrt(max(s, 1e-16) / n)
    return t, 2 * (1 - stats.norm.cdf(abs(t)))


def chay(pair):
    bang = G.dung_bang()
    tt = DT.bien_doi_thi_truong()
    tr_mask_tt = doan(tt.index.values) == 0
    mu, sd = tt[tr_mask_tt].mean(), tt[tr_mask_tt].std()
    tt_z = (tt - mu) / sd

    d = bang[bang.pair == pair].sort_values("ngay").copy()
    d = d[d.doan.isin([0, 1])].set_index("ngay")
    ex = tt_z.reindex(d.index)[[DIEU_TRI] + KIEM_SOAT[1:]]
    df = pd.concat([d[["y_bien_do", "m_har", "doan"]], ex], axis=1).dropna()

    tr = df.doan == 0
    vl = df.doan == 1

    # ── B1: uoc theta CHI tren doan huan luyen ──────────────────────────
    data_tr = dml.DoubleMLData(df[tr].drop(columns=["doan"]),
                               y_col="y_bien_do", d_cols=DIEU_TRI, x_cols=KIEM_SOAT)
    np.random.seed(SEED)
    ml_l = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=SEED)
    ml_m = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=SEED)
    plr = dml.DoubleMLPLR(data_tr, ml_l=ml_l, ml_m=ml_m, n_folds=5, n_rep=3,
                          score="partialling out")
    plr.fit()
    theta = float(plr.coef[0])

    # ── B2: dong bang theta, ap len lop phu tren doan KIEM DINH (chua nhin thay) ──
    y_rv1_vl = np.exp(df.loc[vl, "y_bien_do"].values)         # y_bien_do = log RV(t+1)
    h_har_vl = np.exp(df.loc[vl, "m_har"].values)              # m_har = log h_HAR(t)
    vix_vl = df.loc[vl, DIEU_TRI].values
    h_lop_phu = h_har_vl * np.exp(theta * vix_vl)

    ql_har = qlike(y_rv1_vl, h_har_vl)
    ql_lp = qlike(y_rv1_vl, h_lop_phu)
    t_dm, p_dm = dm_nw(ql_har - ql_lp)          # >0 nghia la lop phu loi hon

    chenh = 100 * (ql_lp.mean() - ql_har.mean()) / abs(ql_har.mean())
    ra = dict(pair=pair, theta_huan_luyen=round(theta, 5),
             n_train=int(tr.sum()), n_valid=int(vl.sum()),
             qlike_har=round(float(ql_har.mean()), 6),
             qlike_lop_phu=round(float(ql_lp.mean()), 6),
             chenh_phan_tram=round(float(chenh), 4),
             dm_t=round(float(t_dm), 4), dm_p=round(float(p_dm), 4))
    print(f"{pair}: θ(huấn luyện)={theta:+.5f}  QLIKE HAR={ql_har.mean():.4f} → "
          f"+lớp phủ={ql_lp.mean():.4f}  chênh={chenh:+.3f}%  DM p={p_dm:.4f}")
    return ra


def main():
    ra = {}
    for pair in CAP_KIEM:
        ra[pair] = chay(pair)

    ps = [ra[p]["dm_p"] for p in CAP_KIEM]
    rej_holm, q_holm, _, _ = multipletests(ps, alpha=0.05, method="holm")
    n_cai_thien = sum(1 for p in CAP_KIEM if ra[p]["chenh_phan_tram"] < 0)
    n_song_sot = int(sum(r and ra[p]["chenh_phan_tram"] < 0
                         for p, r in zip(CAP_KIEM, rej_holm)))
    for p, rh, qh in zip(CAP_KIEM, rej_holm, q_holm):
        ra[p]["q_holm"] = round(float(qh), 5)
        ra[p]["song_sot_holm"] = bool(rh)

    tong = dict(n_cai_thien_huong=int(n_cai_thien), n_cap=len(CAP_KIEM),
               n_song_sot_holm=n_song_sot,
               ket_luan=(f"{n_cai_thien}/{len(CAP_KIEM)} cặp cải thiện QLIKE về HƯỚNG; "
                        f"{n_song_sot}/{len(CAP_KIEM)} cải thiện CÓ Ý NGHĨA sau Holm"))
    ra["_tong_ket"] = tong
    print("\n" + json.dumps(tong, ensure_ascii=False, indent=1))

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "pha3b_dml_tichhop.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("Đã ghi", outp)
    return ra


if __name__ == "__main__":
    main()
