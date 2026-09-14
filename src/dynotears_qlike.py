"""AP DUNG THU canh CHEO CAP cua DYNOTEARS (`dynotears_tuviet.py`) lam cot
ngoai sinh cho HAR, do QLIKE tren doan KIEM DINH — cung cau hoi va cung khung
danh gia voi `spillover_dy.py` (Diebold-Yilmaz), de tra loi truc tiep: "DAG
DYNOTEARS tim duoc co giup du bao THAT khong, hay chi la cau truc dong thoi
khong dung duoc cho du bao?"

QUAN TRONG — `spillover_dy.py` DA THU dung dung cau hoi nay TRUOC, voi mot
phuong phap MANH HON (Diebold-Yilmaz generalized FEVD tren VAR khop lai moi
21 phien, cua so mo rong, khong ro ri) va THAT BAI o CA hai tan suat:

    D1 (commit f20a661): "KHONG cai thien o tan suat ngay"
    H1 (commit 1ff94d4): "TE HON o CA 6/6 cap"

File nay dung LAI chinh khung danh gia do (V2.thiet_ke + V2.he_so_cuon cua
so mo rong + V2.qlike_tb tren doan kiem dinh) nhung thay trong so lan truyen
bang CANH DONG THOI cua DYNOTEARS (khoi 6 cap, khong tinh 3 ngoai sinh —
kenh do da co san trong he thong san xuat qua duong khac). DE TRANH RO RI:
DAG duoc khop LAI TU DAU CHI tren doan HUAN LUYEN (khac voi ban chay thu
dau tien trong `dynotears_tuviet.py`, ban do fit tren ca huan luyen+kiem
dinh vi muc dich lan la CHAN DOAN cau truc, khong phai danh gia du bao).

Chay:  python src/dynotears_qlike.py
Ghi:   output/dynotears_qlike.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

warnings_ = None
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import dynotears_tuviet as DY                                  # noqa: E402
import volfc2 as V2                                             # noqa: E402
from split import doan                                          # noqa: E402

EPS = DY.EPS


def hoc_trong_so_cheo_cap():
    """Khop LAI DYNOTEARS chi tren khoi 6 cap (bo 3 ngoai sinh), CHI tren
    doan HUAN LUYEN, de dung lam trong so ngoai suy SACH sang kiem dinh."""
    bang, chung = V2.nap_bang()
    lrv = {p: np.log(np.maximum(bang[p].set_index("Date").reindex(chung).rv5.values, EPS))
           for p in V2.PAIRS}
    df = pd.DataFrame(lrv, index=chung)
    g = doan(chung.values)
    tr = g == 0
    mu, sd = df[tr].mean(), df[tr].std().replace(0, np.nan)
    dfz = (df - mu) / sd                      # ap cho CA CHUOI, thong ke chi tu huan luyen
    dfz_tr = dfz[tr].dropna()
    X, Y1 = dfz_tr.values[1:], dfz_tr.values[:-1]
    W, _ = DY._hoc_dynotears(X, Y1, DY.LAMBDA_W, DY.LAMBDA_A)
    return W, dfz, chung, bang


def chay():
    t0 = time.time()
    print("=" * 100)
    print("DYNOTEARS -> CỘT NGOẠI SINH CHO HAR — đo QLIKE trên đoạn KIỂM ĐỊNH")
    print("(cùng khung đánh giá với spillover_dy.py, đã kiểm D1 lẫn H1 trước đó)")
    print("=" * 100)

    W, dfz, chung, bang = hoc_trong_so_cheo_cap()
    print("\nTrọng số cạnh chéo cặp (khớp CHỈ trên huấn luyện):")
    print(f"{'':10}" + "".join(f"{p:>9}" for p in V2.PAIRS))
    for i, p in enumerate(V2.PAIRS):
        print(f"{p:<10}" + "".join(f"{W[i, j]:>9.3f}" for j in range(len(V2.PAIRS))))

    feat = dfz.values @ W          # feat[:, j] = tổng có trọng số các cặp KHÁC -> cặp j

    ra = {"w_cheo_cap": W.tolist(), "pairs": V2.PAIRS, "qlike_kiem_dinh": {}}
    for j, p in enumerate(V2.PAIRS):
        d = bang[p]
        lv = np.log(np.maximum(d.rv5.values, EPS))
        n = len(d)
        va = doan(d.Date.values) == 1

        e_series = pd.Series(feat[:, j], index=chung)
        e_can = e_series.reindex(pd.DatetimeIndex(d.Date.values)).ffill().fillna(0.0).values
        co_canh_vao = bool((np.abs(W[:, j]) >= DY.W_NGUONG).any())

        for ten, extra in (("không thêm cột", None), ("spillover DYNOTEARS", [e_can])):
            X = V2.thiet_ke(d, lv, extra)
            y = np.empty(n)
            y[:-1] = lv[1:]
            y[-1] = np.nan
            preds = {}
            for m in V2.MODELS:
                Xm = X[m]
                hople = np.isfinite(Xm).all(1) & np.isfinite(y)
                b, A, B, S, N = V2.he_so_cuon(Xm, y, hople, None)
                ssr = V2._ssr(b, A, B, S)
                s2 = np.where(N >= V2.MIN_FIT, ssr / np.maximum(N, 1), np.nan)
                fit = np.einsum("tk,tk->t", Xm, b)
                preds[m] = np.clip(fit, -30, 0) + 0.5 * np.maximum(s2, 0)
            g = np.stack([preds[m] for m in V2.MODELS]).mean(0)
            f = np.where(np.isfinite(g), np.exp(g), np.nan)
            ql, nn = V2.qlike_tb(np.r_[np.nan, f[:-1]], d.rv5.values, va)
            ra["qlike_kiem_dinh"].setdefault(p, {})[ten] = {"qlike": ql, "n": nn}
        print(f"\n  {p:<8} (có cạnh chéo cặp vào: {co_canh_vao})  " + "  ".join(
            f"{ten}={ra['qlike_kiem_dinh'][p][ten]['qlike']:.4f}"
            for ten in ("không thêm cột", "spillover DYNOTEARS")))

    tb = {ten: float(np.mean([ra["qlike_kiem_dinh"][p][ten]["qlike"] for p in V2.PAIRS]))
          for ten in ("không thêm cột", "spillover DYNOTEARS")}
    chenh = tb["spillover DYNOTEARS"] - tb["không thêm cột"]
    ra["trung_binh"] = tb
    print(f"\n  TRUNG BÌNH 6 CẶP: không thêm cột {tb['không thêm cột']:.4f}  ·  "
          f"spillover DYNOTEARS {tb['spillover DYNOTEARS']:.4f}")
    print(f"  → chênh lệch trung bình {chenh:+.6f} — " + (
        "quá nhỏ để gọi là cải thiện, không đủ để thay đổi cấu hình sản xuất"
        if abs(chenh) < 0.001 else
        ("DYNOTEARS TỐT HƠN có thể xem xét" if chenh < 0 else "DYNOTEARS TỆ HƠN")))

    with open(os.path.join(OUT, "dynotears_qlike.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/dynotears_qlike.json · {time.time()-t0:.0f}s")
    return ra


if __name__ == "__main__":
    chay()
