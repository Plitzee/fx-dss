"""VONG 7+ — TO HOP DU BAO, MO RONG voi cac ho GBM/nen HIEN DAI HON (XGBoost,
CatBoost, TabPFN v2) VA mot bo to hop PHI TUYEN (stacking) — truoc khi ket
luan chi can trung binh deu tay don gian.

Nguoi dung yeu cau: dung dung lai o to hop don gian (HAR+GRU deu tay) ma
phai nghien cuu sau hon cac mo hinh HIEN DAI, thu ket hop chung truoc.
`kiem_tohop2.py` da thu {HAR, LightGBM, GRU, LSTM, Ridge} x {deu tay, trong
so 1/QLIKE, hoi quy Granger-Ramanathan}. File nay THEM:

  * XGBoost, CatBoost — hai ho GBM hien dai hon LightGBM (`run_ml2.py`).
  * TabPFN v2 (Hollmann et al., Nature 2025) cho DUNG bai toan hoi quy
    bien dong — mo hinh NEN cho du lieu bang, khac han GBM/RNN ve co che
    (in-context, khong gradient) — them TINH DA DANG thuc su cho to hop
    (`run_tabpfn_vol.py`).
  * (4) STACKING PHI TUYEN: mot LightGBM RAT NONG (num_leaves=3, it vong
    lap, regularize manh) lam META-LEARNER, khop tren cac du bao (thang
    log) cua toan bo mo hinh goc, muc tieu la log_rv_that, CHI tren doan
    KIEM DINH (co tach 15% cuoi kiem dinh de dung som, tranh overfit ngay
    tren kiem dinh). Day la ky thuat "stacked generalization" (Wolpert
    1992) hien dai hon hoi quy tuyen tinh GR — cho phep quan he PHI TUYEN
    giua cac du bao goc, dung ky thuat cac bai stacking-ensemble 2025 da
    dung (XGBoost/GBM lam meta-learner tren HAR+ML+DL).

Van CUNG giao thuc QLIKE bat bien thang do, cung phan doan, trong so/he so
CHI khop tren KIEM DINH.

Chay:  python src/kiem_tohop3.py
Ghi:   output/ketqua_tohop3.json
"""
import os
import sys
import json
import time
import pickle
import itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")

import volfc2 as V2                                     # noqa: E402
import metrics as M                                       # noqa: E402
from split import VALID_TU, TEST_TU                      # noqa: E402
from run_grid import bang_cache                          # noqa: E402
from run_final7 import dm_nw                              # noqa: E402
from kiem_tohop2 import (nap_du_bao, tb_hinh_hoc, tb_trong_so_nghich_dao,   # noqa: E402
                         gr_hoi_quy, qlike_arr, EPS)

P = V2.PAIRS


def nap_them(F):
    """Nap them du bao XGBoost/CatBoost (_ml2_pred.npz) va TabPFN
    (_tabpfn_vol_pred.npz) vao F — CUNG chi so hang voi F da co (dua tren
    _ml_feat.npz, cung thu tu nap_du_bao() dung)."""
    for f_ in ("_ml2_pred.npz", "_tabpfn_vol_pred.npz"):
        pth = os.path.join(OUT, f_)
        if not os.path.exists(pth):
            print(f"  (chưa có {f_} — bỏ qua)")
            continue
        d = np.load(pth, allow_pickle=True)
        for i, t in enumerate(d["ten"]):
            F[str(t)] = d[f"f{i}"]
    return F


def _tu_kiem_stack():
    """Tu kiem stacking: cho 1 du bao hoan hao + 1 du bao toan nhieu, meta-
    learner phai hoc duoc QLIKE gan HOAN HAO (dua chu yeu vao du bao tot)."""
    rng = np.random.default_rng(1)
    n = 6000
    log_rv = rng.normal(-10.0, 0.5, n)
    f_perfect = np.exp(log_rv)
    f_nhieu = np.exp(log_rv + rng.normal(0, 2.5, n))
    F = {"perfect": f_perfect, "nhieu": f_nhieu}
    va_mask = np.ones(n, bool)
    du_bao, _ = stack_hoiquy(F, ["perfect", "nhieu"], log_rv, va_mask)
    h = du_bao(va_mask)
    rv = np.exp(log_rv)
    ql = qlike_arr(rv, h).mean()
    ok = ql < 0.02   # gan hoan hao (QLIKE ~0 neu doan dung f_perfect)
    print(f"  tự kiểm stacking: QLIKE trên dữ liệu mô phỏng = {ql:.4f} "
          f"(kỳ vọng gần 0)  {'ĐẠT' if ok else 'THẤT BẠI'}")
    return ok


def stack_hoiquy(F, ten_, log_rv, va_mask, seed=0):
    """Meta-learner LightGBM RAT NONG tren thang log cac du bao goc.
    Khop CHI tren va_mask (kiem dinh), tach 15% cuoi lam dung som."""
    import lightgbm as lgb
    L_ = np.column_stack([np.log(np.maximum(F[t], EPS)) for t in ten_])
    m = va_mask.copy()
    for t in ten_:
        m &= np.isfinite(F[t]) & (F[t] > 0)
    m &= np.isfinite(log_rv)
    idx = np.where(m)[0]
    k = int(len(idx) * 0.85)
    i_fit, i_es = idx[:k], idx[k:]
    ds_fit = lgb.Dataset(L_[i_fit], label=log_rv[i_fit])
    ds_es = lgb.Dataset(L_[i_es], label=log_rv[i_es], reference=ds_fit)
    bst = lgb.train(dict(objective="regression", num_leaves=3, max_depth=2,
                        learning_rate=0.03, min_child_samples=100,
                        lambda_l2=5.0, feature_fraction=0.8, verbose=-1,
                        seed=seed),
                    ds_fit, num_boost_round=300, valid_sets=[ds_es],
                    callbacks=[lgb.early_stopping(20, verbose=False)])
    pred_fit = bst.predict(L_[idx])
    resid = log_rv[idx] - pred_fit
    hc = 0.5 * float(np.var(resid))

    def du_bao(mask):
        mm = mask.copy()
        for t in ten_:
            mm &= np.isfinite(F[t]) & (F[t] > 0)
        out = np.full(len(mm), np.nan)
        Lp = np.column_stack([np.log(np.maximum(F[t][mm], EPS)) for t in ten_])
        out[mm] = np.exp(bst.predict(Lp) + hc)
        return out

    return du_bao, dict(best_iter=bst.best_iteration, hc=hc)


def main():
    t0 = time.time()
    print("=" * 108)
    print("TỔ HỢP DỰ BÁO MỞ RỘNG — thêm XGBoost/CatBoost/TabPFN v2 + stacking phi tuyến")
    print("=" * 108)
    assert _tu_kiem_stack(), "tự kiểm stacking thất bại — dừng"

    F, rv, dts = nap_du_bao()
    F = nap_them(F)
    va = (dts >= VALID_TU) & (dts < TEST_TU)
    te = dts >= TEST_TU
    log_rv = np.log(np.maximum(rv, EPS))

    goc = "HAR vòng 7 (khớp mỗi phiên)"
    ten_map = {
        "HAR": goc,
        "LGBM": "LightGBM (QLIKE trực tiếp)",
        "GRU": "GRU (khớp năm)",
        "LSTM": "LSTM (khớp năm)",
        "Ridge": "Ridge (toàn bộ đặc trưng)",
        "XGB": "XGBoost",
        "CatB": "CatBoost",
        "TabPFN": "TabPFN v2 (ngữ cảnh 8k, khớp năm)",
    }
    ten_map = {k: v for k, v in ten_map.items() if v in F}
    print(f"\nứng viên có sẵn ({len(ten_map)}): {list(ten_map)}\n")

    KETQUA = {}
    H = {}

    def cham(nhan, h):
        H[nhan] = h
        m = te & np.isfinite(h) & (h > 0) & np.isfinite(rv) & (rv > 0)
        ql_te = float(qlike_arr(rv[m], h[m]).mean())
        mv = va & np.isfinite(h) & (h > 0) & np.isfinite(rv) & (rv > 0)
        ql_va = float(qlike_arr(rv[mv], h[mv]).mean()) if mv.sum() else float("nan")
        return ql_va, ql_te, int(m.sum())

    qv0, qt0, _ = cham(goc, F[goc])
    KETQUA[goc] = dict(qlike_valid=qv0, qlike_test=qt0)
    print(f"{'mô hình gốc HAR':<52}{'kiểm định':>12}{'kiểm tra':>12}")
    print(f"{goc:<52}{qv0:>12.4f}{qt0:>12.4f}")

    # từng mô hình ĐƠN LẺ trước (để biết XGB/CatBoost/TabPFN có cạnh tranh không)
    print(f"\n{'--- mô hình đơn lẻ (mới thêm) ---':<52}")
    for k, t in ten_map.items():
        if t == goc:
            continue
        qv, qt, _ = cham(t, F[t])
        print(f"{t:<52}{qv:>12.4f}{qt:>12.4f}  ({(qt/qt0-1)*100:+.1f}% so HAR)")

    khoa = [k for k in ten_map if k != "HAR"]
    tap_con = [("HAR",) + c for r in range(1, len(khoa) + 1)
              for c in itertools.combinations(khoa, r)]
    print(f"\n{len(tap_con)} tập con (luôn gồm HAR) × 4 cách tổ hợp\n")

    for ci, c in enumerate(tap_con):
        if ci % 20 == 0:
            print(f"  ... {ci}/{len(tap_con)} tập con ({time.time()-t0:.0f}s)", flush=True)
        ten_full = [ten_map[k] for k in c]
        nhan = "TB đều · " + "+".join(c)
        qv, qt, _ = cham(nhan, tb_hinh_hoc(F, ten_full))
        KETQUA[nhan] = dict(qlike_valid=qv, qlike_test=qt)

        w0 = np.array([1.0 / cham(t, F[t])[0] for t in ten_full]); w0 /= w0.sum()
        nhan2 = "TB 1/QLIKE · " + "+".join(c)
        qv, qt, _ = cham(nhan2, tb_trong_so_nghich_dao(F, ten_full, w0))
        KETQUA[nhan2] = dict(qlike_valid=qv, qlike_test=qt)

        if len(c) >= 2:
            du_bao, coef = gr_hoi_quy(F, ten_full, log_rv, va)
            h_full = np.where(np.isnan(du_bao(te)), du_bao(va), du_bao(te))
            nhan3 = "GR hồi quy · " + "+".join(c)
            qv, qt, _ = cham(nhan3, h_full)
            KETQUA[nhan3] = dict(qlike_valid=qv, qlike_test=qt, he_so=coef)

            du_bao_s, coef_s = stack_hoiquy(F, ten_full, log_rv, va)
            h_full_s = np.where(np.isnan(du_bao_s(te)), du_bao_s(va), du_bao_s(te))
            nhan4 = "Stack LGBM · " + "+".join(c)
            qv, qt, _ = cham(nhan4, h_full_s)
            KETQUA[nhan4] = dict(qlike_valid=qv, qlike_test=qt, meta=coef_s)

    print("\n" + "=" * 108)
    xep = sorted(KETQUA, key=lambda t: KETQUA[t]["qlike_test"])
    print(f"XẾP HẠNG TOÀN BỘ ({len(xep)} mục) — DM so với HAR v7 đơn, top 25")
    print("-" * 108)
    h_goc = F[goc]
    m_te = te & np.isfinite(h_goc) & (h_goc > 0) & np.isfinite(rv) & (rv > 0)
    print(f"{'#':>4} {'tổ hợp':<52}{'kiểm định':>11}{'kiểm tra':>11}{'so HAR':>9}"
          f"{'DM t':>8}{'DM p':>8}")
    print("-" * 108)
    for i, t in enumerate(xep[:25] + ([goc] if goc not in xep[:25] else [])):
        qv, qt = KETQUA[t]["qlike_valid"], KETQUA[t]["qlike_test"]
        chenh = (qt / qt0 - 1) * 100
        if t == goc:
            tt, pp = 0.0, 1.0
        else:
            h_t = H[t]
            m_ = te & np.isfinite(h_t) & (h_t > 0) & m_te
            d = qlike_arr(rv[m_], h_t[m_]) - qlike_arr(rv[m_], h_goc[m_])
            tt, pp = dm_nw(d)
        KETQUA[t]["dm_t"] = tt; KETQUA[t]["dm_p"] = pp
        print(f"{i+1:>4} {t:<52}{qv:>11.4f}{qt:>11.4f}{chenh:>8.1f}%{tt:>8.2f}{pp:>8.4f}")
    print("-" * 108)
    print("  (DM âm = tổ hợp TỐT HƠN HAR v7 có ý nghĩa; p<0,05 là chênh lệch có ý nghĩa)")

    top15 = xep[:15] + ([goc] if goc not in xep[:15] else [])
    ngay = dts[m_te]
    Ldf = {}
    for t in top15:
        h_t = H[t]
        Ldf[t] = qlike_arr(rv[m_te], np.where(np.isfinite(h_t[m_te]) & (h_t[m_te] > 0),
                                              h_t[m_te], np.nan))
    df = pd.DataFrame(Ldf)
    df["ngay"] = ngay.values
    G = df.groupby("ngay").mean()
    alive, elim = M.mcs(G[top15].values, alpha=0.10, B=3000, block=20, seed=17)
    print(f"\nMODEL CONFIDENCE SET (α=0,10) trên {len(top15)} tổ hợp đầu + HAR gốc, {G.shape[0]} phiên:")
    song = [top15[a] for a in alive]
    for t in song:
        print(f"    ★ {t}")
    print(f"  {len(song)}/{len(top15)} sống sót; bị loại sớm nhất: "
          + ", ".join(top15[i] for i, _ in elim[:3]))
    KETQUA["_mcs_top15"] = song
    KETQUA["_ung_vien"] = list(ten_map)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ketqua_tohop3.json"), "w", encoding="utf-8") as f:
        json.dump(KETQUA, f, indent=1, ensure_ascii=False, default=float)
    print("\n→ output/ketqua_tohop3.json")


if __name__ == "__main__":
    main()
