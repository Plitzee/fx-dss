"""VONG 7 — MO RONG HO MO HINH: XGBoost + CatBoost (GBM hien dai hon LightGBM).

Nguoi dung muon nghien cuu sau hon cac mo hinh ML hien dai TRUOC KHI chi
dung to hop don gian. `run_ml.py` da co Ridge/RF/LightGBM(x2)/MLP; file nay
them HAI ho GBM khac — XGBoost (histogram, ho tro objective tuy bien) va
CatBoost (ordered boosting, thuong on dinh hon tren du lieu nho/nhieu) —
CUNG giao thuc: cung tap dac trung (`ml_data.xay`), cung khop lai dau moi
nam (cua so mo rong), cung cach doi log RV -> phuong sai (+0,5*var(du)),
sieu tham so chon tren KIEM DINH, cham diem MOT LAN tren KIEM TRA sau nay
(o kiem_tohop3.py).

Chay:  python src/run_ml2.py
Ghi:   output/_ml2_pred.npz
"""
import os
import sys
import time
import hashlib
import warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")

from run_ml import nap, chay_walkforward, VALID_TU, TEST_TU  # noqa: E402

CACHE = os.path.join(OUT, "_ml_cache")   # DUNG CHUNG cache voi run_ml.py
os.makedirs(CACHE, exist_ok=True)


def _fit_xgb(Xtr, ytr, rv, hp, cot):
    import xgboost as xgb
    md, lr, ne = hp
    m = xgb.XGBRegressor(max_depth=md, learning_rate=lr, n_estimators=ne,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_lambda=1.0, n_jobs=2, random_state=0,
                        tree_method="hist").fit(Xtr, ytr)
    return lambda X: m.predict(X)


def _fit_catboost(Xtr, ytr, rv, hp, cot):
    from catboost import CatBoostRegressor
    depth, lr, ne = hp
    m = CatBoostRegressor(depth=depth, learning_rate=lr, iterations=ne,
                          loss_function="RMSE", thread_count=2, verbose=False,
                          random_seed=0).fit(Xtr, ytr)
    return lambda X: m.predict(X)


HO = {
    "XGBoost": (_fit_xgb, [(4, 0.05, 300), (6, 0.03, 400), (4, 0.03, 600)], "all"),
    "CatBoost": (_fit_catboost, [(6, 0.05, 300), (8, 0.03, 400)], "all"),
}


def main():
    X, y, ten, pid, dts = nap()
    har_cot = [ten.index(c) for c in
              ("lrv_d", "lrv_w", "lrv_m", "lq", "lq_x_lrv", "lrsp", "lrsn", "G")
              ] + [ten.index(c) for c in ten if c.startswith("ev_")] \
             + [ten.index(f"pair{j}") for j in range(6)]
    cot = {"har": np.array(sorted(set(har_cot))), "all": np.arange(X.shape[1])}
    rv = np.exp(y)
    va = (dts >= VALID_TU) & (dts < TEST_TU)

    print("=" * 100)
    print("MỞ RỘNG HỌ MÔ HÌNH — XGBoost + CatBoost (GBM hiện đại hơn)")
    print("=" * 100)

    def cache_path(a, b):
        h = hashlib.md5(f"ml2|{a}|{b}".encode()).hexdigest()[:16]
        return os.path.join(CACHE, f"{h}.npz")

    kq = []
    t0 = time.time()
    for ten_ho, (fit, grid, kieu_cot) in HO.items():
        best = None
        for hp in grid:
            cp = cache_path(ten_ho, hp)
            if os.path.exists(cp):
                z2 = np.load(cp); mu, s2 = z2["mu"], z2["s2"]
                dau = "  [đã lưu]"
            else:
                mu, s2 = chay_walkforward(X, y, dts, cot, fit, hp, "std")
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

    np.savez_compressed(os.path.join(OUT, "_ml2_pred.npz"),
                        **{f"f{i}": k["f"] for i, k in enumerate(kq)},
                        ten=np.array([k["ten"] for k in kq]),
                        hp=np.array([k["hp"] for k in kq]),
                        qv=np.array([k["qlike_valid"] for k in kq]))
    print("XẾP HẠNG TRÊN ĐOẠN KIỂM ĐỊNH (chưa mở kiểm tra)")
    print("-" * 100)
    for k in sorted(kq, key=lambda z: z["qlike_valid"]):
        print(f"  {k['ten']:<30}{k['qlike_valid']:>10.4f}   hp={k['hp']}")
    print("-" * 100)
    print("đã ghi output/_ml2_pred.npz")


if __name__ == "__main__":
    main()
