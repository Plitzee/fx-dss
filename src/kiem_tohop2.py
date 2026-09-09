"""VONG 7+ — TO HOP DU BAO (forecast combination) giua HAR/ML/DL da co san.

`run_ml_final.py` da thu MOT kieu to hop: trung binh hinh hoc DEU TAY (equal-
weight log-average) giua HAR v7 + GRU/LSTM — va no da THANG tat ca mo hinh
don le (QLIKE kiem tra 0,1550-0,1556 so voi HAR 0,1585, xem docs/ML_DL_VONG7.md).

File nay THU SAU HON theo dung tinh than nguoi dung yeu cau — "hieu ro cac
mo hinh, tim cach ket hop cai thien ket qua":

  1. Trung binh hinh hoc DEU TAY tren NHIEU tap con hon (HAR+LightGBM,
     HAR+GRU+LightGBM, HAR+GRU+LSTM+LightGBM, ca 4+Ridge...) — mo rong phep
     thu da co, khong chi gioi han o GRU/LSTM.
  2. To hop hoi quy Granger-Ramanathan (1984), dang khong rang buoc, tren
     THANG LOG: log(rv_that) = a + sum_k b_k*log(f_k) + e, khop OLS tren
     doan KIEM DINH (khong dung doan kiem tra de tranh ro ri), roi
     h = exp(a + sum b_k*log(f_k) + 0,5*var(du)). Day la ky thuat combination
     kinh dien manh hon trung binh deu tay vi cho phep TRONG SO KHAC NHAU
     moi mo hinh thay vi ep bang nhau — xem Granger & Ramanathan (1984,
     J. Forecasting) va tong quan Wang et al. "Forecast combinations: an
     over 50-year review" (arXiv 2205.04216).
  3. To hop trong so nghich dao QLIKE kiem dinh (dang shrinkage don gian
     hon GR, khong can khop hoi quy — trong so w_k ~ 1/qlike_valid_k,
     chuan hoa tong=1) — de doi chieu xem hoi quy day du co that su can
     thiet hay trong so tho da du.

Tat ca DUNG CHINH XAC giao thuc QLIKE bat bien thang do (r-log(r)-1) va
phan doan kiem dinh/kiem tra nhu run_ml_final.py va toan bo vong 7.
Trong so/he so CHI khop tren doan KIEM DINH — doan KIEM TRA chi dung de
cham diem cuoi cung, dung mot lan (khong dung de chon mo hinh).

Chay:  python src/kiem_tohop2.py
Ghi:   output/ketqua_tohop2.json
"""
import os
import sys
import json
import pickle
import itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
D = os.path.join(ROOT, "data")

import volfc2 as V2                                   # noqa: E402
import metrics as M                                     # noqa: E402
from split import VALID_TU, TEST_TU                    # noqa: E402
from run_grid import bang_cache                        # noqa: E402
from run_final7 import dm_nw                             # noqa: E402

P = V2.PAIRS
EPS = 1e-300


def qlike_arr(rv, h):
    r = rv / h
    return r - np.log(r) - 1


def nap_du_bao():
    """Sao chep dung logic tai du bao cua run_ml_final.py -> tra ve F, rv, dts."""
    z = np.load(os.path.join(OUT, "_ml_feat.npz"))
    y = z["y"]; dts = pd.DatetimeIndex(z["dts"])
    rv = np.exp(y)

    F = {}
    for f_, nhan in ((("_ml_pred.npz"), "ml"), (("_dl_pred.npz"), "dl")):
        pth = os.path.join(OUT, f_)
        if not os.path.exists(pth):
            continue
        d = np.load(pth, allow_pickle=True)
        qv_ = d["qv"]
        tot = {}
        for i, t in enumerate(d["ten"]):
            ho = str(t).split(" h=")[0]
            if ho not in tot or qv_[i] < qv_[tot[ho]]:
                tot[ho] = i
        for ho, i in tot.items():
            F[f"{ho} (khớp năm)" if nhan == "dl" else str(d["ten"][i])] = d[f"f{i}"]

    bang, chung = bang_cache()
    tr = np.asarray(chung < VALID_TU)
    with open(os.path.join(OUT, "cauhinh_chot.pkl"), "rb") as f:
        CH = pickle.load(f)
    H = V2.chay(bang, chung, deseason=CH["deseason"], crosspair=bool(CH["crosspair"]),
                event=CH["event"], window=None, lams=(CH["lam"],),
                train_mask=tr, recal=CH["recal"])[CH["lam"]]
    n1 = len(chung)

    def ve_luoi_ml(arr):
        v = np.full(n1, np.nan); v[:-1] = np.asarray(arr)[1:]
        return v

    F["HAR vòng 7 (khớp mỗi phiên)"] = np.concatenate([ve_luoi_ml(H[p]) for p in P])

    dt_tgt = np.concatenate([np.append(np.asarray(chung)[1:], np.datetime64("NaT"))
                             for _ in P])
    dt_tgt = pd.DatetimeIndex(dt_tgt)
    return F, rv, dt_tgt


def tb_hinh_hoc(F, ten_):
    """Trung binh hinh hoc DEU TAY tren thang log (nhu run_ml_final.gop)."""
    L_ = [np.log(np.maximum(F[t], EPS)) for t in ten_]
    m = np.ones(len(L_[0]), bool)
    for t in ten_:
        m &= np.isfinite(F[t]) & (F[t] > 0)
    out = np.full(len(m), np.nan)
    out[m] = np.exp(np.mean([l[m] for l in L_], 0))
    return out


def tb_trong_so_nghich_dao(F, ten_, w):
    """Trung binh hinh hoc CO TRONG SO (w_k, tong=1) tren thang log."""
    L_ = [np.log(np.maximum(F[t], EPS)) for t in ten_]
    m = np.ones(len(L_[0]), bool)
    for t in ten_:
        m &= np.isfinite(F[t]) & (F[t] > 0)
    out = np.full(len(m), np.nan)
    stacked = np.stack([l[m] for l in L_], 0)
    out[m] = np.exp(np.average(stacked, axis=0, weights=w))
    return out


def gr_hoi_quy(F, ten_, log_rv, va_mask):
    """Granger-Ramanathan khong rang buoc, thang log:
    log_rv = a + sum_k b_k*log(f_k) + e, khop OLS tren va_mask.
    Tra ve ham du_bao(mask) -> h va he so (a, b, hc)."""
    L_ = [np.log(np.maximum(F[t], EPS)) for t in ten_]
    m = va_mask.copy()
    for t in ten_:
        m &= np.isfinite(F[t]) & (F[t] > 0)
    m &= np.isfinite(log_rv)
    X = np.column_stack([np.ones(m.sum())] + [l[m] for l in L_])
    yv = log_rv[m]
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    a = beta[0]; b = beta[1:]
    resid = yv - X @ beta
    hc = 0.5 * float(np.var(resid))

    def du_bao(mask):
        mm = mask.copy()
        for t in ten_:
            mm &= np.isfinite(F[t]) & (F[t] > 0)
        out = np.full(len(mm), np.nan)
        Xp = np.column_stack([np.ones(mm.sum())] +
                              [np.log(np.maximum(F[t][mm], EPS)) for t in ten_])
        out[mm] = np.exp(Xp @ beta + hc)
        return out

    return du_bao, dict(a=float(a), b=[float(x) for x in b], hc=hc)


def _tu_kiem_gr():
    """Tu kiem: neu mot mo hinh CHINH XAC bang RV that (khong nhieu), GR
    phai hoc duoc b~1, a~0, hc~0 cho no va bo qua cac mo hinh nhieu khac."""
    rng = np.random.default_rng(0)
    n = 4000
    log_rv = rng.normal(-10.0, 0.5, n)
    f_perfect = np.exp(log_rv)                       # du bao hoan hao
    f_nhieu = np.exp(log_rv + rng.normal(0, 2.0, n))  # du bao rat nhieu
    F = {"perfect": f_perfect, "nhieu": f_nhieu}
    va_mask = np.ones(n, bool)
    du_bao, coef = gr_hoi_quy(F, ["perfect", "nhieu"], log_rv, va_mask)
    ok = abs(coef["b"][0] - 1.0) < 0.05 and abs(coef["b"][1]) < 0.05 and abs(coef["a"]) < 0.05
    print(f"  tự kiểm GR: b_perfect={coef['b'][0]:.3f} (kỳ vọng ~1), "
          f"b_nhiễu={coef['b'][1]:.3f} (kỳ vọng ~0), a={coef['a']:.3f}  "
          f"{'ĐẠT' if ok else 'THẤT BẠI'}")
    return ok


def main():
    print("=" * 108)
    print("TỔ HỢP DỰ BÁO (forecast combination) — HAR/ML/DL đã có, mở rộng ngoài equal-weight")
    print("=" * 108)
    assert _tu_kiem_gr(), "tự kiểm GR thất bại — dừng"

    F, rv, dts = nap_du_bao()
    va = (dts >= VALID_TU) & (dts < TEST_TU)
    te = dts >= TEST_TU
    log_rv = np.log(np.maximum(rv, EPS))

    goc = "HAR vòng 7 (khớp mỗi phiên)"
    ung_vien = {
        "HAR": goc,
        "LGBM": "LightGBM (QLIKE trực tiếp)",
        "GRU": "GRU (khớp năm)",
        "LSTM": "LSTM (khớp năm)",
        "Ridge": "Ridge (toàn bộ đặc trưng)",
    }
    ung_vien = {k: v for k, v in ung_vien.items() if v in F}
    print(f"\nứng viên có sẵn: {list(ung_vien)}\n")

    KETQUA = {}
    H = {}  # ten -> h_full (mảng dự báo, để tính DM sau khi xếp hạng)

    def cham(ten_hienthi, h):
        H[ten_hienthi] = h
        m = te & np.isfinite(h) & (h > 0) & np.isfinite(rv) & (rv > 0)
        ql_te = float(qlike_arr(rv[m], h[m]).mean())
        mv = va & np.isfinite(h) & (h > 0) & np.isfinite(rv) & (rv > 0)
        ql_va = float(qlike_arr(rv[mv], h[mv]).mean()) if mv.sum() else float("nan")
        return ql_va, ql_te, int(m.sum())

    # mốc HAR đơn
    qv0, qt0, _ = cham(goc, F[goc])
    KETQUA[goc] = dict(qlike_valid=qv0, qlike_test=qt0)
    print(f"{'mô hình gốc HAR':<48}{'kiểm định':>12}{'kiểm tra':>12}")
    print(f"{goc:<48}{qv0:>12.4f}{qt0:>12.4f}")

    print(f"\n{'--- (1) trung bình hình học ĐỀU TAY, các tập con ---':<48}")
    tap_con = []
    khoa = [k for k in ung_vien if k != "HAR"]
    for r in range(1, len(khoa) + 1):
        for c in itertools.combinations(khoa, r):
            tap_con.append(("HAR",) + c)
    for c in tap_con:
        ten_full = [ung_vien[k] for k in c]
        h = tb_hinh_hoc(F, ten_full)
        nhan = "TB đều · " + "+".join(c)
        qv, qt, n = cham(nhan, h)
        KETQUA[nhan] = dict(qlike_valid=qv, qlike_test=qt)
        print(f"{nhan:<48}{qv:>12.4f}{qt:>12.4f}")

    print(f"\n{'--- (2) trung bình trọng số nghịch đảo QLIKE(kiểm định) ---':<48}")
    for c in tap_con:
        ten_full = [ung_vien[k] for k in c]
        w0 = np.array([1.0 / cham(t, F[t])[0] for t in ten_full])
        w0 = w0 / w0.sum()
        h = tb_trong_so_nghich_dao(F, ten_full, w0)
        nhan = "TB 1/QLIKE · " + "+".join(c)
        qv, qt, n = cham(nhan, h)
        KETQUA[nhan] = dict(qlike_valid=qv, qlike_test=qt,
                             trong_so={k: float(x) for k, x in zip(c, w0)})
        print(f"{nhan:<48}{qv:>12.4f}{qt:>12.4f}  trọng số={dict(zip(c, np.round(w0,3)))}")

    print(f"\n{'--- (3) hồi quy Granger-Ramanathan (khớp trên KIỂM ĐỊNH) ---':<48}")
    for c in tap_con:
        if len(c) < 2:
            continue
        ten_full = [ung_vien[k] for k in c]
        du_bao, coef = gr_hoi_quy(F, ten_full, log_rv, va)
        h_te = du_bao(te)
        h_va = du_bao(va)  # trong-mẫu, chỉ để tham khảo — KHÔNG dùng chọn mô hình
        h_full = np.where(np.isnan(h_te), h_va, h_te)
        nhan = "GR hồi quy · " + "+".join(c)
        qv, qt, n = cham(nhan, h_full)
        b_str = ", ".join(f"{k}:{b:+.2f}" for k, b in zip(c, coef["b"]))
        KETQUA[nhan] = dict(qlike_valid=qv, qlike_test=qt, he_so=coef)
        print(f"{nhan:<48}{qv:>12.4f}{qt:>12.4f}  a={coef['a']:+.2f} [{b_str}]")

    print("\n" + "=" * 108)
    xep = sorted(KETQUA, key=lambda t: KETQUA[t]["qlike_test"])
    print("XẾP HẠNG TOÀN BỘ (theo QLIKE kiểm tra) — DM so với HAR v7 đơn")
    print("-" * 108)
    h_goc = F[goc]
    m_te = te & np.isfinite(h_goc) & (h_goc > 0) & np.isfinite(rv) & (rv > 0)
    L_goc = qlike_arr(rv[m_te], h_goc[m_te])
    print(f"{'#':>3} {'tổ hợp':<48}{'kiểm định':>11}{'kiểm tra':>11}{'so HAR':>9}"
          f"{'DM t':>9}{'DM p':>9}")
    print("-" * 108)
    for i, t in enumerate(xep):
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
        print(f"{i+1:>3} {t:<48}{qv:>11.4f}{qt:>11.4f}{chenh:>8.1f}%{tt:>9.2f}{pp:>9.4f}")
    print("-" * 108)
    print("  (DM âm = tổ hợp TỐT HƠN HAR v7 có ý nghĩa; p<0,05 là chênh lệch có ý nghĩa)")
    print("  (45 phép so sánh cùng lúc — DM đơn lẻ dễ dương tính giả; xem MCS bên dưới)")

    # MCS trên 15 tổ hợp đầu + HAR gốc, đúng như run_ml_final.py làm cho bảng 14 mô hình
    top15 = xep[:15] + [goc]
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

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ketqua_tohop2.json"), "w", encoding="utf-8") as f:
        json.dump(KETQUA, f, indent=1, ensure_ascii=False, default=float)
    print("\n→ output/ketqua_tohop2.json")


if __name__ == "__main__":
    main()
