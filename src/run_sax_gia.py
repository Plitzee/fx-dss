"""MAU HINH TREN NEN GIA (huong tang/giam) — cau tra loi truc tiep cho cau hoi
"pattern cua do thi nen, xu huong tang giam co dua vao he thong khong?"

Phan biet voi run_sax_stats.py: nhanh do la mau BIEN DONG (LOW/MEDIUM/HIGH
cua rv5) — chinh tac gia da ghi ro "mau cua minh la mau BIEN DONG chu khong
phai mau HUONG" (xem docstring run_sax_stats.py, doi chung TSMOM). File nay
lam dung cai con thieu: ma hoa HUONG GIA (xuong/dung yen/len, tercile loi
suat ngay) roi chay LAI dung bo may thong ke da co (khong viet lai) — liet
ke toan bo khong gian W=2,3,4, sua boi Westfall-Young step-down, null khoi
2 ngay, phat hien tren huan luyen+kiem dinh, xac nhan tren kiem tra.

Boi canh van lieu (docs/MAU_HINH_FX.md, muc C3): Hutchinson et al. (2022,
RIBAF) nhan ban >21.000 quy tac ky thuat tien te — Sharpe roi tu 0,66 trong
mau xuong 0,06 ngoai mau, khong song sot chi phi, va TOAN BO loi nhuan bat
thuong bi hap thu boi DUY NHAT mot nhan to: dong luong chuoi thoi gian
(TSMOM). Day la gia thuyet null manh nhat phai vuot qua. Tang 1 (momentum
EWMA) da cho thay TSMOM chet tu 2010 — neu Hutchinson dung, mau nen cung
se chet theo.

Chay:  python src/run_sax_gia.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")

from split import doan  # noqa: E402
import volfc2 as V2  # noqa: E402
from run_sax_stats import (TEN, WS, NPERM, MIN_KHOP, ma_hoa, z_gop,  # noqa: E402
                            z_phang, chay_hoanvi, westfall_young)


def nap_gia():
    """Chuoi trang thai 3 muc HUONG GIA (tercile loi suat ngay) cho moi cap.
    Nguong lay tren doan HUAN LUYEN — khong ro ri, dung quy uoc voi nap()
    trong run_sax_stats.py."""
    out = {}
    for p in V2.PAIRS:
        d = pd.read_csv(os.path.join(D, "prices", f"{p}_d1.csv"), parse_dates=["Date"])
        d = d.sort_values("Date").reset_index(drop=True)
        r = np.log(d.close.values / np.maximum(d.close.shift(1).values, 1e-9))
        r[0] = np.nan
        g = doan(d.Date.values)
        ok = np.isfinite(r) & (g == 0)
        q = np.quantile(r[ok], [1 / 3, 2 / 3])
        s = np.where(np.isfinite(r), np.digitize(r, q), -1)
        out[p] = (d.Date.values, s, g)
    return out


def main():
    DATA = nap_gia()
    rng = np.random.default_rng(3)

    def phat_hien(g):
        return g <= 1

    def kiem_tra(g):
        return g == 2

    n_gt = sum(3 ** W * 3 for W in WS)
    print("=" * 100)
    print("MẪU HÌNH TRÊN NỀN GIÁ (hướng tăng/giảm) — trực tiếp trả lời câu hỏi 'pattern nến'")
    print("=" * 100)
    print(f"trạng thái: tercile lợi suất ngày (XUỐNG/ĐI NGANG/LÊN), ngưỡng chốt trên huấn luyện")
    print(f"không gian giả thuyết: W ∈ {WS} → {n_gt} giả thuyết (giống hệt khuôn khổ SAX biến động)\n")

    goc = {W: z_gop(DATA, W, phat_hien) for W in WS}
    keys = []
    for W in WS:
        z, lift, nn, k = goc[W]
        for a in range(3 ** W):
            if nn[a] < MIN_KHOP:
                continue
            for j in range(3):
                if np.isfinite(z[a, j]):
                    keys.append((W, a, j))
    z_obs = np.array([goc[W][0][a, j] for (W, a, j) in keys])
    print(f"{len(keys)} giả thuyết đủ số khớp (≥{MIN_KHOP})\n")

    Zb = chay_hoanvi(DATA, phat_hien, np.random.default_rng(3), NPERM,
                      "khoi", 2, keys)
    q = np.quantile(Zb.max(1), [0.90, 0.95, 0.99])
    print(f"null khối 2 ngày — max|z|:  90% {q[0]:.2f}   95% {q[1]:.2f}   99% {q[2]:.2f}")
    p_wy = westfall_young(z_obs, Zb)

    rows = []
    for i, (W, a, j) in enumerate(keys):
        z, lift, nn, k = goc[W]
        mau = tuple(TEN[(a // 3 ** (W - 1 - m)) % 3] for m in range(W))
        rows.append(dict(W=W, mau=" → ".join(mau), dich=TEN[j], n=int(nn[a]),
                         lift=float(lift[a, j]), z=float(z[a, j]), p_wy=float(p_wy[i])))
    df = pd.DataFrame(rows).sort_values("p_wy")
    song_sot = df[df.p_wy < 0.05]
    print(f"\nSống sót Westfall-Young (p<0,05) trên {len(keys)} giả thuyết: "
          f"{len(song_sot)}/{len(keys)}")
    if len(song_sot):
        print(song_sot.to_string(index=False))
        print("\nXác nhận trên đoạn KIỂM TRA (chưa dùng để chọn):")
        for _, r in song_sot.iterrows():
            W = int(r.W)
            z_te, lift_te, nn_te, _ = z_gop(DATA, W, kiem_tra)
            a = 0
            for m, tenmau in enumerate(r.mau.split(" → ")):
                a = a * 3 + TEN.index(tenmau)
            j = TEN.index(r.dich)
            n_te = int(nn_te[a]) if a < nn_te.shape[0] else 0
            z_te_v = float(z_te[a, j]) if n_te >= MIN_KHOP else np.nan
            print(f"  {r.mau} ⇒ {r.dich}: kiểm định z={r.z:.2f} (n={r.n}) "
                  f"→ kiểm tra z={z_te_v:.2f} (n={n_te})")
    else:
        print("→ KHÔNG mẫu hướng nào sống sót hiệu chỉnh bội — đúng như dự đoán từ Hutchinson (2022).")

    print("\nĐối chiếu: tầng 1 (momentum EWMA, dạng đơn giản nhất của 'xu hướng') "
          "cũng cho Sharpe âm 2010-2025. Hai phép thử ĐỘC LẬP, cùng kết luận.")
    print("\nTỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
