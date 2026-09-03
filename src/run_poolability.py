"""VIEC 3 — GOP HAY TACH 6 CAP? KIEM DINH POOLABILITY + KET QUA CO NGOT.

Pesaran, Pick & Timmermann (Quantitative Economics 2026) dua ra hai thu:
mot KIEM DINH tinh gop duoc cua du bao, va ket luan rang uoc luong CO NGOT
thuong thang ca gop thuan lan rieng thuan. File nay chay ca hai tren du lieu
cua minh, tren doan HUAN LUYEN, roi doi chieu voi ket qua ngoai mau cua luoi
cau hinh (run_grid2.py).
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")

import volfc2 as V2
from run_grid import bang_cache
from split import VALID_TU

EPS = V2.EPS


def main():
    bang, chung = bang_cache()
    tr = np.asarray(chung < VALID_TU)
    print("=" * 100)
    print("VIỆC 3 — GỘP HAY TÁCH 6 CẶP?")
    print("=" * 100)
    print("Ước lượng trên đoạn HUẤN LUYỆN, mô hình HARQ (log-space).\n")

    Xs, ys = {}, {}
    for p in V2.PAIRS:
        d = bang[p]
        lv = np.log(np.maximum(d.rv5.values, EPS))
        X = V2.thiet_ke(d, lv)["HARQ"]
        y = np.empty(len(lv)); y[:-1] = lv[1:]; y[-1] = np.nan
        m = tr & np.isfinite(X).all(1) & np.isfinite(y)
        Xs[p] = X[m]; ys[p] = y[m]

    k = Xs[V2.PAIRS[0]].shape[1]
    P = len(V2.PAIRS)
    Xa = np.vstack([Xs[p] for p in V2.PAIRS]); ya = np.concatenate([ys[p] for p in V2.PAIRS])
    bp = np.linalg.lstsq(Xa, ya, rcond=None)[0]
    ssr_r = float(((ya - Xa @ bp) ** 2).sum())
    ssr_u = 0.0; B = {}
    for p in V2.PAIRS:
        b = np.linalg.lstsq(Xs[p], ys[p], rcond=None)[0]
        B[p] = b
        ssr_u += float(((ys[p] - Xs[p] @ b) ** 2).sum())
    N = len(ya); q = (P - 1) * k; dfd = N - P * k
    F = ((ssr_r - ssr_u) / q) / (ssr_u / dfd)
    pv = 1 - stats.f.cdf(F, q, dfd)
    print(f"KIỂM ĐỊNH POOLABILITY (Chow / Roy-Zellner), H0: hệ số bằng nhau ở 6 cặp")
    print("-" * 100)
    print(f"  N = {N:,}   k = {k}   ràng buộc q = {q}")
    print(f"  SSR gộp {ssr_r:,.1f}   SSR riêng {ssr_u:,.1f}   "
          f"F({q}, {dfd:,}) = {F:.2f}   p = {pv:.3e}")
    print(f"  → {'BÁC BỎ tính gộp được' if pv < 0.01 else 'không bác bỏ'}: "
          f"hệ số HAR KHÁC NHAU giữa các cặp.")
    print("  Lưu ý: F chuẩn giả định phần dư độc lập giữa các cặp. Ở FX thì KHÔNG —")
    print("  các cặp cùng chịu nhân tố dollar, nên F này lạc quan. Đọc nó như dấu hiệu")
    print("  chứ đừng đọc như bằng chứng chốt.\n")

    print("ĐỘ TÁN CỦA HỆ SỐ GIỮA 6 CẶP (HARQ)")
    print("-" * 100)
    ten = ["hằng số", "log RV(d)", "log RV(w)", "log RV(m)", "log Q", "log Q × log RV"]
    Bm = np.vstack([B[p] for p in V2.PAIRS])
    print(f"{'hệ số':<18}" + "".join(f"{p:>11}" for p in V2.PAIRS)
          + f"{'gộp':>10}{'độ tán/|TB|':>13}")
    print("-" * 100)
    for j in range(k):
        cv = Bm[:, j].std() / max(abs(Bm[:, j].mean()), 1e-9)
        print(f"{ten[j]:<18}" + "".join(f"{B[p][j]:>11.3f}" for p in V2.PAIRS)
              + f"{bp[j]:>10.3f}{cv:>13.2f}")
    print("-" * 100)

    print("\nNHƯNG TÍNH KHÁC NHAU TRONG MẪU KHÔNG CÓ NGHĨA LÀ TÁCH THÌ DỰ BÁO TỐT HƠN.")
    print("Kết quả ngoài mẫu từ lưới cấu hình (run_grid2.py, QLIKE trên KIỂM ĐỊNH):")
    print("-" * 100)
    g = pd.read_csv(os.path.join(OUT, "grid2_valid.csv"))
    best = g.sort_values("qlike_valid").iloc[0]
    fix = {c: best[c] for c in ("deseason", "crosspair", "event", "window", "recal")}
    sub = g
    for c, v in fix.items():
        sub = sub[sub[c] == v]
    sub = sub.sort_values("lam")
    print(f"  (giữ nguyên cấu hình thắng: {fix})")
    print(f"  {'lambda co ngót':<20}{'ý nghĩa':<34}{'QLIKE kiểm định':>18}")
    for _, r in sub.iterrows():
        y = ("riêng hoàn toàn từng cặp" if r.lam == 0 else
             "gộp hoàn toàn 6 cặp" if r.lam == 1 else f"co ngót {r.lam:.0%} về trung bình")
        print(f"  {r.lam:<20.2f}{y:<34}{r.qlike_valid:>18.4f}")
    lo, hi = sub.qlike_valid.min(), sub.qlike_valid.max()
    print("-" * 100)
    print(f"  Toàn dải lambda chỉ chênh {hi-lo:.4f} QLIKE ({(hi/lo-1)*100:.1f}%) — nhỏ hơn")
    print(f"  nhiều so với lợi ích của lịch sự kiện ({(g[g.event=='off'].qlike_valid.min()/g.qlike_valid.min()-1)*100:.1f}%).")
    print("  KẾT LUẬN: hệ số quả thật khác nhau giữa các cặp (bác bỏ poolability), nhưng với")
    print("  ~2.500 quan sát mỗi cặp thì sai số ước lượng đã đủ nhỏ để KHÔNG cần mượn thông")
    print("  tin từ cặp khác. Đây đúng là đánh đổi thiên lệch–phương sai mà Pesaran, Pick &")
    print("  Timmermann mô tả, và ở phía T lớn thì ước lượng riêng thắng. Co ngót là kết quả")
    print("  ÂM ở đây, và đó là một kết quả đáng báo cáo chứ không phải một thất bại.")
    print("=" * 100)


if __name__ == "__main__":
    main()
