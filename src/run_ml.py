"""VONG 7 — HOC MAY CO DICH THUC SU THANG HAR KHONG?

Ba bai gan day deu tra loi KHONG, tren du lieu cua ho:
  * Branco, Rubesam & Zevallos (J. Empirical Finance 2024) — 10 chi so, khong
    co bang chung ML phi tuyen vuot mo hinh tuyen tinh.
  * Kilic (Fed FEDS 2025-061) — THAR/STHAR thang XGBoost, DNN, LSTM, GRU.
  * Brini (arXiv 2607.05291) — foundation model khong thang Log-HAR.
File nay kiem lai tren DU LIEU CUA MINH, dung dung giao thuc 70/15/15.

BA DIEU LAM CHO SO SANH CONG BANG:

1. CUNG TAP THONG TIN, thuc ra ML con duoc NHIEU HON: toan bo dac trung HAR,
   cong lich NHTW rieng tung cap, cong 22 do tre tho, thu trong tuan, ma cap.
   HAR khong duoc 22 do tre va khong duoc thu trong tuan.

2. CUNG TAN SUAT KHOP LAI. HAR san xuat khop lai moi phien; mang no-ron thi
   khong the. Nen o day MOI mo hinh — ke ca OLS — deu khop lai MOI NAM bang
   cua so mo rong. Co san mot dong "OLS HAR (khop nam)" de tach bach: chenh
   lech con lai la do LOP HAM, khong phai do tan suat khop.

3. CUNG CACH DOI VE PHUONG SAI. Moi mo hinh du bao log RV roi doi bang hieu
   chinh log-chuan +0,5*var(phan du) uoc luong tren chinh doan huan luyen cua
   lan khop do. Neu khong lam the thi QLIKE phat oan ML.

Sieu tham so chon tren doan KIEM DINH, cham diem mot lan tren doan KIEM TRA.
"""
import os
import sys
import json
import time
import warnings
import itertools
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")

from split import VALID_TU, TEST_TU
import volfc2 as V2

MIN_TRAIN = 3000                      # hàng gộp 6 cặp
KHOP_TU, KHOP_DEN = 2015, 2026        # khớp lại đầu mỗi năm


def nap():
    z = np.load(os.path.join(OUT, "_ml_feat.npz"))
    ten = json.load(open(os.path.join(OUT, "_ml_cols.json")))
    X = z["X"]; y = z["y"]; pid = z["pid"]
    dts = pd.DatetimeIndex(z["dts"])
    return X, y, ten, pid, dts


# ─────────────────────────────────────────────────────── các họ mô hình
def _fit_tuyen_tinh(Xtr, ytr, alpha):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    return lambda X: m.predict(sc.transform(X))


def _fit_rf(Xtr, ytr, hp):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=300, max_depth=hp[0],
                              min_samples_leaf=hp[1], n_jobs=2,
                              random_state=0).fit(Xtr, ytr)
    return m.predict


def _fit_gbm(Xtr, ytr, hp):
    import lightgbm as lgb
    nl, lr, ne = hp
    m = lgb.LGBMRegressor(num_leaves=nl, learning_rate=lr, n_estimators=ne,
                          min_child_samples=50, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, n_jobs=2, verbose=-1,
                          random_state=0).fit(Xtr, ytr)
    return m.predict


def _fit_gbm_qlike(Xtr, rv_tr, hp):
    """LightGBM toi uu TRUC TIEP QLIKE. Du bao m = log f.

    QLIKE = m + y*exp(-m);  grad = 1 - y*exp(-m);  hess = y*exp(-m).
    """
    import lightgbm as lgb
    nl, lr, ne = hp
    init = float(np.mean(np.log(np.maximum(rv_tr, 1e-16))))

    def obj(y_pred, ds):
        # LightGBM >= 4 goi objective(preds, Dataset); preds DA gom init_score
        y_true = ds.get_label()
        e = np.exp(-np.clip(np.asarray(y_pred, float), -30, 5)) * y_true
        return (1.0 - e), np.maximum(e, 1e-6)

    ds = lgb.Dataset(Xtr, label=rv_tr, init_score=np.full(len(rv_tr), init))
    bst = lgb.train(dict(objective=obj, num_leaves=nl, learning_rate=lr,
                         min_child_samples=50, feature_fraction=0.8,
                         bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                         seed=0), ds, num_boost_round=ne)
    return lambda X: bst.predict(X) + init


def _fit_mlp(Xtr, ytr, hp):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = MLPRegressor(hidden_layer_sizes=hp[0], alpha=hp[1], max_iter=400,
                     early_stopping=True, n_iter_no_change=15,
                     validation_fraction=0.12, learning_rate_init=1e-3,
                     random_state=0).fit(sc.transform(Xtr), ytr)
    return lambda X: m.predict(sc.transform(X))


HO = {
    "OLS HAR (khớp năm)": (lambda Xtr, ytr, rv, hp, cot: _fit_tuyen_tinh(
        Xtr[:, cot["har"]], ytr, 1e-8), [None], "har"),
    "Ridge (toàn bộ đặc trưng)": (lambda Xtr, ytr, rv, hp, cot:
                                  _fit_tuyen_tinh(Xtr, ytr, hp),
                                  [0.1, 1.0, 10.0, 100.0], "all"),
    "Random Forest": (lambda Xtr, ytr, rv, hp, cot: _fit_rf(Xtr, ytr, hp),
                      [(8, 20), (16, 20), (None, 50)], "all"),
    "LightGBM (L2 trên log)": (lambda Xtr, ytr, rv, hp, cot: _fit_gbm(Xtr, ytr, hp),
                               [(15, 0.03, 400), (31, 0.05, 400),
                                (63, 0.05, 800)], "all"),
    "LightGBM (QLIKE trực tiếp)": (lambda Xtr, ytr, rv, hp, cot:
                                   _fit_gbm_qlike(Xtr, rv, hp),
                                   [(15, 0.03, 400), (31, 0.05, 400)], "all"),
    "MLP": (lambda Xtr, ytr, rv, hp, cot: _fit_mlp(Xtr, ytr, hp),
            [((64, 32), 1e-4), ((128, 64), 1e-3)], "all"),
}


def chay_walkforward(X, y, dts, cot, fit, hp, ho_kieu):
    """Khop lai dau moi nam bang cua so mo rong; tra ve (mu_log, s2) toan bang."""
    n = len(y)
    mu = np.full(n, np.nan); s2 = np.full(n, np.nan)
    ok_row = np.isfinite(X).all(1) & np.isfinite(y)
    rv = np.exp(y)
    for yr in range(KHOP_TU, KHOP_DEN + 1):
        moc = pd.Timestamp(f"{yr}-01-01")
        het = pd.Timestamp(f"{yr+1}-01-01")
        tr = ok_row & (dts < moc)
        te = ok_row & (dts >= moc) & (dts < het)
        if tr.sum() < MIN_TRAIN or te.sum() == 0:
            continue
        Xtr = X[tr]; ytr = y[tr]
        if ho_kieu == "qlike":
            pred = fit(Xtr, ytr, rv[tr], hp, cot)
        else:
            pred = fit(Xtr, ytr, rv[tr], hp, cot)
        sub = cot["har"] if hp is None and ho_kieu == "har" else slice(None)
        Xa = Xtr[:, cot["har"]] if ho_kieu == "har" else Xtr
        Xb = X[te][:, cot["har"]] if ho_kieu == "har" else X[te]
        r = ytr - pred(Xa)
        mu[te] = pred(Xb)
        s2[te] = float(r.var())
    return mu, s2


def main():
    X, y, ten, pid, dts = nap()
    har_cot = [ten.index(c) for c in
               ("lrv_d", "lrv_w", "lrv_m", "lq", "lq_x_lrv", "lrsp", "lrsn", "G")
               ] + [ten.index(c) for c in ten if c.startswith("ev_")] \
              + [ten.index(f"pair{j}") for j in range(6)]
    cot = {"har": np.array(sorted(set(har_cot))), "all": np.arange(X.shape[1])}
    rv = np.exp(y)
    va = (dts >= VALID_TU) & (dts < TEST_TU)
    te = dts >= TEST_TU

    print("=" * 100)
    print("HỌC MÁY / HỌC SÂU CÓ THẮNG HAR KHÔNG?")
    print("=" * 100)
    print(f"bảng dài {X.shape[0]:,} hàng × {X.shape[1]} đặc trưng, 6 cặp gộp chung")
    print(f"khớp lại đầu mỗi năm {KHOP_TU}–{KHOP_DEN}, cửa sổ mở rộng")
    print(f"kiểm định {int(va.sum()):,} hàng, kiểm tra {int(te.sum()):,} hàng\n")

    import hashlib
    CACHE = os.path.join(OUT, "_ml_cache")
    os.makedirs(CACHE, exist_ok=True)

    def cache_path(a, b):
        h = hashlib.md5(f"{a}|{b}".encode()).hexdigest()[:16]
        return os.path.join(CACHE, f"{h}.npz")

    kq = []
    t0 = time.time()
    for ten_ho, (fit, grid, kieu_cot) in HO.items():
        ho_kieu = ("har" if ten_ho.startswith("OLS")
                   else "qlike" if "QLIKE" in ten_ho else "std")
        best = None
        for hp in grid:
            cp = cache_path(ten_ho, hp)
            if os.path.exists(cp):
                z2 = np.load(cp); mu, s2 = z2["mu"], z2["s2"]
                dau = "  [đã lưu]"
            else:
                mu, s2 = chay_walkforward(X, y, dts, cot, fit, hp, ho_kieu)
                np.savez_compressed(cp, mu=mu, s2=s2)
                dau = ""
            f = np.exp(np.clip(mu, -30, 0) + 0.5 * np.nan_to_num(s2))
            okv = va & np.isfinite(f) & (f > 0)
            r = rv[okv] / f[okv]
            qv = float((r - np.log(r) - 1).mean())
            if best is None or qv < best[0]:
                best = (qv, hp, mu, s2)
            print(f"  {ten_ho:<28} hp={str(hp):<18} QLIKE kiểm định {qv:.4f}"
                  f"   ({time.time()-t0:.0f}s){dau}", flush=True)
        qv, hp, mu, s2 = best
        f = np.exp(np.clip(mu, -30, 0) + 0.5 * np.nan_to_num(s2))
        kq.append(dict(ten=ten_ho, hp=str(hp), qlike_valid=qv, f=f))
        print(f"  → chọn hp={hp}\n")

    np.savez_compressed(os.path.join(OUT, "_ml_pred.npz"),
                        **{f"f{i}": k["f"] for i, k in enumerate(kq)},
                        ten=np.array([k["ten"] for k in kq]),
                        hp=np.array([k["hp"] for k in kq]),
                        qv=np.array([k["qlike_valid"] for k in kq]))
    print("\nXẾP HẠNG TRÊN ĐOẠN KIỂM ĐỊNH (chưa mở kiểm tra)")
    print("-" * 100)
    for k in sorted(kq, key=lambda z: z["qlike_valid"]):
        print(f"  {k['ten']:<30}{k['qlike_valid']:>10.4f}   hp={k['hp']}")
    print("-" * 100)
    print("đã ghi output/_ml_pred.npz — run_ml_final.py sẽ chấm trên kiểm tra")


if __name__ == "__main__":
    main()
