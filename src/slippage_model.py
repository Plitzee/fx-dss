"""TANG 3 — MO HINH TRUOT GIA QUA MUC DUNG LO, do tu nen M1 that.

Thay gia dinh "khop dung tai muc dung lo" bang so do. Nguon: data/slippage.csv
sinh boi collect/slippage.py — 60.617 lan cham muc dung lo tren 6 cap x 16 nam.

Chi phi thoat that = spread luc thoat (cost.py) + truot gia (file nay).
"""
import os
import numpy as np
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(DIR), "data")
REGIME_SPLIT = 2015


def load():
    d = pd.read_csv(os.path.join(DATA, "slippage.csv"))
    d["Date"] = pd.to_datetime(d.Date, format="ISO8601")
    d["che_do"] = np.where(d.Date.dt.year < REGIME_SPLIT, "pre2015", "post2015")
    return d


def bang(d):
    g = d.groupby(["che_do", "dist"]).truot_phut
    return pd.DataFrame({"trung vị": g.median(), "p95": g.quantile(.95),
                         "p99": g.quantile(.99), "n": g.size()}).reset_index()


def he_so(d):
    """Truot gia quy ve BOI so cua khoang cach dung lo — de cam vao tran rui ro."""
    out = {}
    for che, gg in d.groupby("che_do"):
        r = []
        for dist, g in gg.groupby("dist"):
            pip_dist = dist * (1 / 1e-4)          # khoang cach stop tinh theo pip (xap xi, gia~1)
            r.append((dist, float(g.truot_phut.quantile(.95)) / pip_dist))
        out[che] = float(np.median([x[1] for x in r]))
    return out


if __name__ == "__main__":
    d = load()
    print("=" * 92)
    print("TRUOT GIA QUA MUC DUNG LO — do tu nen M1, 6 cap x 16 nam")
    print("=" * 92)
    print(f"{len(d):,} lần chạm mức dừng lỗ\n")
    print(f"{'chế độ':<11}{'stop cách':>11}{'trượt trung vị':>17}{'p95':>9}{'p99':>9}{'n':>9}")
    print("-" * 92)
    for _, r in bang(d).iterrows():
        print(f"{r['che_do']:<11}{r['dist']:>10.1%}{r['trung vị']:>17.2f}"
              f"{r['p95']:>9.2f}{r['p99']:>9.2f}{int(r['n']):>9,}")
    print("-" * 92)
    print("Đơn vị: pip. 'trượt' = mức dừng lỗ − giá thấp nhất của chính phút chạm.")

    print("\n" + "=" * 92)
    print("TRƯỢT GIÁ THEO GIỜ UTC (stop cách 0,3%, sau 2015)")
    print("=" * 92)
    x = d[(d.dist == 0.003) & (d.che_do == "post2015")]
    g = x.groupby("gio").truot_phut
    t = pd.DataFrame({"trung vị": g.median(), "p95": g.quantile(.95), "n": g.size()})
    worst = t.sort_values("p95", ascending=False).head(5)
    best = t.sort_values("p95").head(5)
    print(f"{'giờ tệ nhất':<14}{'p95 (pip)':>11}   |   {'giờ tốt nhất':<14}{'p95 (pip)':>11}")
    print("-" * 92)
    for (h1, r1), (h2, r2) in zip(worst.iterrows(), best.iterrows()):
        print(f"{h1:>6}:00{'':<6}{r1['p95']:>11.2f}   |   {h2:>6}:00{'':<6}{r2['p95']:>11.2f}")

    print("\n" + "=" * 92)
    print("SO SÁNH VỚI GIẢ ĐỊNH CŨ")
    print("=" * 92)
    hs = he_so(d)
    for che, v in hs.items():
        print(f"  {che}: trượt p95 ≈ {v:.1%} khoảng cách dừng lỗ")
    p15 = float(d[d.che_do == "post2015"].truot_phut.median())
    p99 = float(d[d.che_do == "post2015"].truot_phut.quantile(.99))
    print(f"\n  Giả định cũ: trượt = 0 pip")
    print(f"  Thực tế sau 2015: trung vị {p15:.2f} pip, p99 {p99:.2f} pip")
    print(f"  → chi phí thoát buộc phải cộng thêm; ở đuôi nó lớn hơn cả spread p95")

    # tu kiem
    assert (d.truot_phut >= -1e-9).all(), "truot khong duoc am"
    assert d.groupby("pair").size().min() > 5000, "moi cap phai co du mau"
    med_pre = d[d.che_do == "pre2015"].truot_phut.median()
    med_post = d[d.che_do == "post2015"].truot_phut.median()
    print(f"\n  TU KIEM: trượt trung vị trước 2015 {med_pre:.2f} pip, sau 2015 {med_post:.2f} pip")
    print("  ĐẠT")
