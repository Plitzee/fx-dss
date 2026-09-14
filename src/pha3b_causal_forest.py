"""PHA 3B — HIỆU ỨNG NHÂN QUẢ KHÔNG ĐỒNG NHẤT THEO CHẾ ĐỘ, bằng Causal
Forest (Wager & Athey 2018, JASA; Athey, Tibshirani & Wager 2019,
"Generalized Random Forests," Annals of Statistics — thư viện `econml`
của Microsoft Research, cài đặt chính thức của nhóm tác giả).

CÂU HỎI TIẾP THEO sau khi lớp phủ nhân quả TUYẾN TÍNH (θ cố định, mọi chế
độ) ở `pha3b_dml_tichhop.py` KHÔNG cải thiện QLIKE có ý nghĩa (0/6 cặp):
liệu hiệu ứng nhân quả của VIX có THAY ĐỔI theo chế độ biến động (bình
tĩnh/vừa/căng thẳng) không? Nếu hiệu ứng tập trung ở chế độ CĂNG THẲNG mà
lớp phủ tuyến tính lại rải đều hiệu chỉnh nhỏ ra CẢ BA chế độ, phần hiệu
chỉnh ở chế độ bình tĩnh chỉ thêm nhiễu — đây là lý do khả dĩ cho kết quả
âm ở bước trước, và Causal Forest kiểm tra được giả thuyết này trực tiếp
bằng CATE (Conditional Average Treatment Effect) theo biến điều biến X
(effect modifier) = chế độ.

Chạy:  python src/pha3b_causal_forest.py
Ghi:   output/pha3b_causal_forest.json
Cần:   pip install econml
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import pha3b_dactrung as DT                                   # noqa: E402
import pha3b_granger as G                                     # noqa: E402
from split import doan                                        # noqa: E402

from econml.dml import CausalForestDML                         # noqa: E402

SEED = 20260914
CAP_KIEM = ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCAD", "USDCHF"]
DIEU_TRI = "VIXCLS_lv"
KIEM_SOAT = ["m_har", "GVZCLS_lv", "DFII10_lv"]
TEN_CHE_DO = ["bình tĩnh", "vừa", "căng thẳng"]


def _chuoi_mot_cap(bang, tt_z, pair):
    d = bang[bang.pair == pair].sort_values("ngay").copy()
    d = d[d.doan.isin([0, 1])].set_index("ngay")
    ex = tt_z.reindex(d.index)[[DIEU_TRI] + KIEM_SOAT[1:]]
    df = pd.concat([d[["y_bien_do", "m_har", "doan"]], ex], axis=1).dropna()
    # che do bien dong — tam phan vi cua chinh sigma (m_har) TREN HUAN LUYEN,
    # nhat quan voi cach api/cache.py chia che do (tam phan vi cua sigma).
    tr = df.doan == 0
    nguong = np.quantile(df.loc[tr, "m_har"], [1 / 3, 2 / 3])
    df["che_do"] = np.digitize(df["m_har"], nguong)
    return df


def chay(pair):
    bang = G.dung_bang()
    tt = DT.bien_doi_thi_truong()
    tr_mask_tt = doan(tt.index.values) == 0
    mu, sd = tt[tr_mask_tt].mean(), tt[tr_mask_tt].std()
    tt_z = (tt - mu) / sd

    df = _chuoi_mot_cap(bang, tt_z, pair)
    Y = df["y_bien_do"].values
    Tt = df[DIEU_TRI].values
    X = df[["che_do"]].values.astype(float)     # bien dieu bien (effect modifier)
    W = df[KIEM_SOAT].values                    # bien kiem soat con lai (m_har, GVZCLS, DFII10)

    np.random.seed(SEED)
    est = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=SEED),
        model_t=GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=SEED),
        n_estimators=500, min_samples_leaf=20, max_depth=None,
        random_state=SEED, cv=5)
    est.fit(Y, Tt, X=X, W=W)

    cate = {}
    for r in (0, 1, 2):
        xr = np.array([[float(r)]])
        eff = est.effect(xr)[0]
        lo, hi = est.effect_interval(xr, alpha=0.05)
        cate[TEN_CHE_DO[r]] = dict(cate=round(float(eff), 5),
                                   ci_95=[round(float(lo[0]), 5), round(float(hi[0]), 5)],
                                   n=int((df.che_do == r).sum()))
    print(f"\n=== {pair} — CATE(VIX→biến_động) theo chế độ ===")
    for ten, v in cate.items():
        print(f"  {ten:>10s}: θ={v['cate']:+.5f}  CI95%={v['ci_95']}  n={v['n']:,}")
    return dict(pair=pair, cate_theo_che_do=cate)


def main():
    ra = {}
    for pair in CAP_KIEM:
        ra[pair] = chay(pair)

    # tong hop: chieu tang theo che do co nhat quan qua CA 6 cap khong?
    tang_dan = sum(1 for p in CAP_KIEM
                   if (ra[p]["cate_theo_che_do"]["căng thẳng"]["cate"] >
                       ra[p]["cate_theo_che_do"]["bình tĩnh"]["cate"]))
    tong = dict(n_cap_tang_dan_theo_che_do=int(tang_dan), n_cap=len(CAP_KIEM),
               ket_luan=(f"{tang_dan}/{len(CAP_KIEM)} cặp có CATE(căng thẳng) > "
                        f"CATE(bình tĩnh) — {'nhất quán' if tang_dan >= 5 else 'KHÔNG nhất quán'} "
                        f"với giả thuyết hiệu ứng tập trung ở chế độ căng thẳng"))
    ra["_tong_ket"] = tong
    print("\n" + json.dumps(tong, ensure_ascii=False, indent=1))

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "pha3b_causal_forest.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("Đã ghi", outp)
    return ra


if __name__ == "__main__":
    main()
