"""THỬ NGHIỆM — STACKING (hồi quy logistic đa thức trên xác suất của 4 nền)
làm phương pháp thay thế cho HỌC TRỰC TUYẾN (Hedge) trong tầng tổ hợp ba xác
suất — trả lời câu hỏi tối ưu hoá: tầng này trước giờ chỉ đối chiếu 2 cách
tổ hợp trực tuyến (Hedge, Fixed-Share — cả hai đều "trực tuyến", chưa thử
STACKING kiểu OFFLINE/có giám sát).

BỐN NỀN — giống hệt production `api/cache.py:tinh()` — h=1, mục tiêu P:
  khí hậu học, quán tính, chỉ σ̂, σ̂ + chế độ

HAI CÁCH TỔ HỢP:
  Hedge (đang sản xuất)  — trọng số mũ cập nhật trực tuyến, không cần nhãn
                           lúc huấn luyện, tự thích nghi theo thời gian.
  Stacking (thử ở đây)   — hồi quy logistic đa thức (multinomial) khớp MỘT
                           LẦN trên HUẤN LUYỆN, dùng 12 đặc trưng = xác suất
                           của 4 nền × 3 lớp, áp dụng tĩnh cho phần còn lại.

ĐƠN GIẢN HOÁ CÓ CHỦ ĐÍCH: cả hai cách tổ hợp đều dùng bốn nền ở dạng KHỚP
TĨNH một lần trên huấn luyện (không khớp lại cuộn như sản xuất thật) — đây
là so sánh CÔNG BẰNG về CÁCH TỔ HỢP, giữ nguyên cách khớp nền, không phải
tái tạo bit-for-bit toàn bộ pipeline cuộn của sản xuất.

GIAO THỨC: khớp nền + stacking trên HUẤN LUYỆN (đoạn 0), báo cáo BSS/DM trên
KIỂM ĐỊNH (đoạn 1, chưa từng dùng để khớp bất kỳ tham số nào). Gộp 6 cặp,
đúng quy ước `run_ml3.py`.

Chạy:  python src/kiem_stacking_bahop.py
Ghi:   output/kiem_stacking_bahop.json
"""
import json
import os
import sys
import warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                             # noqa: E402
import volfc2 as V2                                           # noqa: E402
from split import doan                                         # noqa: E402
from volfc import merge_thin_days                              # noqa: E402

EPS = 1e-12
SEED = 0
TEN_LOP = 3


def dm_nw(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); mb = x.mean(); L = int(np.ceil(1.5 * n ** (1 / 3)))
    s = np.sum((x - mb) ** 2) / n
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * np.sum((x[k:] - mb) * (x[:-k] - mb)) / n
    t = mb / np.sqrt(max(s, 1e-16) / n)
    return t, 2 * (1 - stats.norm.cdf(abs(t)))


def brier_moi_hang(P, y):
    Y = np.eye(TEN_LOP)[y]
    return np.sum((P - Y) ** 2, axis=1)


def dung_mot_cap(pair):
    from api.main import noi_chuoi
    m = merge_thin_days(noi_chuoi(pair))
    sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, pair), 0.0))
    d = __import__("pandas").DataFrame({"Date": m.Date.values, "sig": sig})
    c = m.close.values
    zt = np.full(len(m), np.nan)
    zt[1:] = np.log(c[1:] / np.maximum(c[:-1], EPS)) / np.maximum(sig[1:], EPS)
    d["zT"] = zt
    tr_mask = doan(d.Date.values) == 0
    T = B.dung_muc_tieu(d, 1, tr_mask)
    yt = B.lop_truoc(T["yP"], 1)

    kh = B.KhiHauHoc().khop(T["yP"][tr_mask])
    qt = B.QuanTinh().khop(T["yP"][tr_mask], yt[tr_mask])
    ns = B.ChiSigma().khop(T["z"][tr_mask])
    cd = B.SigmaCheDo().khop(T["z"][tr_mask], d.sig.values[tr_mask])

    n = len(d)
    kw = dict(canh=T["canh_P"], sigma_h=T["sigma_h"], sig=d.sig.values, y_truoc=yt)
    P_kh = kh.du_bao(n, **kw)
    P_qt = qt.du_bao(n, **kw)
    P_ns = ns.du_bao(n, **kw)
    P_cd = cd.du_bao(n, **kw)
    return dict(y=T["yP"], doan=doan(d.Date.values),
               P_kh=P_kh, P_qt=P_qt, P_ns=P_ns, P_cd=P_cd)


def main():
    du = {p: dung_mot_cap(p) for p in B.PAIRS}
    y = np.concatenate([du[p]["y"] for p in B.PAIRS])
    g = np.concatenate([du[p]["doan"] for p in B.PAIRS])
    X4 = {k: np.concatenate([du[p][f"P_{k}"] for p in B.PAIRS], axis=0)
         for k in ("kh", "qt", "ns", "cd")}
    ok = (y >= 0) & np.all([np.isfinite(X4[k]).all(1) for k in X4], axis=0)
    tr, va = (g == 0) & ok, (g == 1) & ok
    print(f"gộp 6 cặp: huấn luyện {tr.sum():,} · kiểm định (báo cáo) {va.sum():,}")

    # ── Hedge (san xuat) — dung lai NGUYEN VAN B.ToHopTrucTuyen ──────────
    class _CoSan:
        def __init__(self, P):
            self.P = P
        def du_bao(self, n, **kw):
            return self.P

    cg = [(k, _CoSan(X4[k])) for k in ("kh", "qt", "ns", "cd")]
    mo_hedge = B.ToHopTrucTuyen(cg, eta=0.5, tre=1)
    P_hedge_full = mo_hedge.du_bao(len(y), y_that=np.where(ok, y, -1))
    P_hedge = P_hedge_full[va]

    # ── Stacking — hoi quy logistic da thuc, khop MOT LAN tren huan luyen ──
    feat = np.column_stack([X4[k] for k in ("kh", "qt", "ns", "cd")])   # (N,12)
    meta = LogisticRegression(max_iter=2000, multi_class="multinomial",
                              C=1.0, random_state=SEED)
    meta.fit(feat[tr], y[tr])
    P_stack = meta.predict_proba(feat[va])
    # dam bao dung thu tu lop 0/1/2 (LogisticRegression sap xep theo classes_)
    thu_tu = np.argsort(meta.classes_)
    P_stack = P_stack[:, thu_tu]

    tan_suat = np.bincount(y[tr], minlength=3) / tr.sum()
    P_kh_va = np.tile(tan_suat, (va.sum(), 1))

    b_kh = brier_moi_hang(P_kh_va, y[va])
    b_hedge = brier_moi_hang(P_hedge, y[va])
    b_stack = brier_moi_hang(P_stack, y[va])

    bss_hedge = 1 - b_hedge.mean() / b_kh.mean()
    bss_stack = 1 - b_stack.mean() / b_kh.mean()
    t_h, p_h = dm_nw(b_kh - b_hedge)
    t_s, p_s = dm_nw(b_kh - b_stack)
    t_sh, p_sh = dm_nw(b_hedge - b_stack)     # >0 nghia la stacking loi hon Hedge

    ra = dict(
        n_train=int(tr.sum()), n_bao_cao=int(va.sum()),
        hedge=dict(bss=round(float(bss_hedge), 5), dm_vs_khihauhoc_p=round(float(p_h), 5)),
        stacking=dict(bss=round(float(bss_stack), 5), dm_vs_khihauhoc_p=round(float(p_s), 5)),
        stacking_vs_hedge=dict(dm_t=round(float(t_sh), 4), dm_p=round(float(p_sh), 5),
                               chenh_bss=round(float(bss_stack - bss_hedge), 5)),
        ket_luan=("Stacking thắng Hedge có ý nghĩa" if p_sh < 0.05 and bss_stack > bss_hedge
                  else "Hedge thắng Stacking có ý nghĩa" if p_sh < 0.05 and bss_hedge > bss_stack
                  else "không khác biệt có ý nghĩa giữa hai cách tổ hợp"))
    print(json.dumps(ra, ensure_ascii=False, indent=1))

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "kiem_stacking_bahop.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("Đã ghi", outp)
    return ra


if __name__ == "__main__":
    main()
