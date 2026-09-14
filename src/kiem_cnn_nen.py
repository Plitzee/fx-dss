"""THỬ NGHIỆM — mã hoá cửa sổ nến thành ẢNH rồi học bằng CNN, theo đúng ý
tưởng của Chen & Tsai (2020), "Encoding candlesticks as images for pattern
classification using convolutional neural networks," Financial Innovation,
6:26 — bài báo hiện đại (2020), có phản biện, ĐÚNG chủ đề "mẫu hình nến".

KHÁC 12 nhánh khai phá quy luật đã đóng băng (SAX/motif/matrix-profile/
rule-list) và nhánh K1-K4 (hình học + mẫu có tên): những nhánh đó dùng ĐẶC
TRƯNG SỐ (thống kê tóm tắt của nến). Ở đây CNN học TRỰC TIẾP trên ẢNH — hình
dạng KHÔNG GIAN của thân/bấc nến qua nhiều phiên liên tiếp — đúng cơ chế mà
paper gốc và trực giác "đọc biểu đồ nến" của nhà giao dịch dựa vào.

CÂU HỎI: hình ảnh W nến gần nhất có chứa thông tin dự báo biến động NGOÀI
mốc HAR sản xuất hay không? Học phần DƯ (residual) so với log HAR, không học
lại từ đầu — tránh CNN chỉ học lại đúng cái HAR đã biết.

GIAO THỨC: khớp trên ĐOẠN HUẤN LUYỆN, dừng sớm + chọn ngưỡng trên ĐOẠN KIỂM
ĐỊNH, KHÔNG chạm đoạn kiểm tra — giống mọi nhánh khác trong Pha3B/nhánh nến.
Đây là THỬ NGHIỆM PHƯƠNG PHÁP, chưa phải quyết định sản xuất.

Chạy:  python src/kiem_cnn_nen.py [PAIR]
Ghi:   output/kiem_cnn_nen_{PAIR}.json
Cần:   pip install torch
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

import volfc2 as V2                                          # noqa: E402
from split import doan                                        # noqa: E402
from metrics import qlike                                     # noqa: E402

import torch                                                   # noqa: E402
import torch.nn as nn                                          # noqa: E402

W = 10          # so nen trong mot cua so (nhu Chen & Tsai dung cua so ngan)
H_IMG = 32      # chieu cao anh (pixel gia)
BAR_W = 4       # be rong moi nen tren truc ngang (pixel), gom than + khe ho
SEED = 20260914
EPS = 1e-12


def dm_nw(x):
    """Thong ke Diebold-Mariano voi phuong sai Newey-West — DUNG LAI nguyen
    ham da dung o src/run_final7.py de nhat quan trong toan repo."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); mb = x.mean(); L = int(np.ceil(1.5 * n ** (1 / 3)))
    s = np.sum((x - mb) ** 2) / n
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * np.sum((x[k:] - mb) * (x[:-k] - mb)) / n
    t = mb / np.sqrt(max(s, 1e-16) / n)
    return t, 2 * (1 - stats.norm.cdf(abs(t)))


def _ve_anh(o, h, l, c):
    """Ve MOT cua so W nen thanh anh (2, H_IMG, W*BAR_W) — kenh 0 = bac
    (wick), kenh 1 = than (+1 tang / -1 giam), chuan hoa theo bien do GIA
    CUA CHINH cua so (khong dung thong tin ngoai cua so — khong ro ri)."""
    lo, hi = float(np.min(l)), float(np.max(h))
    rng = max(hi - lo, 1e-9)

    def hang(gia):
        return np.clip(((gia - lo) / rng) * (H_IMG - 1), 0, H_IMG - 1)

    img = np.zeros((2, H_IMG, W * BAR_W), dtype=np.float32)
    for i in range(W):
        x0 = i * BAR_W + 1
        r_hi, r_lo = hang(h[i]), hang(l[i])
        r_top = int(round(H_IMG - 1 - max(r_hi, r_lo)))
        r_bot = int(round(H_IMG - 1 - min(r_hi, r_lo)))
        img[0, r_top:r_bot + 1, x0 + 1] = 1.0                  # bac (1 pixel rong)

        r_o, r_c = hang(o[i]), hang(c[i])
        r_top2 = int(round(H_IMG - 1 - max(r_o, r_c)))
        r_bot2 = int(round(H_IMG - 1 - min(r_o, r_c)))
        dau = 1.0 if c[i] >= o[i] else -1.0
        img[1, r_top2:r_bot2 + 1, x0:x0 + BAR_W - 1] = dau     # than (rong hon)
    return img


def dung_du_lieu(pair):
    g = pd.read_csv(os.path.join(DATA, "prices", f"{pair}_d1.csv"), parse_dates=["Date"])
    g = g.sort_values("Date").reset_index(drop=True)
    rv = pd.read_csv(os.path.join(DATA, "rv_adv.csv"), parse_dates=["Date"])
    rv = rv[rv.pair == pair].drop(columns=["pair"])
    d = pd.merge(g, rv, on="Date", how="inner").reset_index(drop=True)
    h_har = V2.du_bao_san_xuat(d, pair)                        # phuong sai du bao HAR

    o, hi, lo, c = d.open.values, d.high.values, d.low.values, d.close.values
    y_rv = np.maximum(d.rv5.values, EPS)
    g_doan = doan(d.Date.values)

    X, Y_resid, Y_rv1, H1, G, ok_idx = [], [], [], [], [], []
    for t in range(W - 1, len(d) - 1):                          # du bao cho t+1
        if not (np.isfinite(h_har[t]) and h_har[t] > 0 and np.isfinite(y_rv[t + 1])):
            continue
        sl = slice(t - W + 1, t + 1)
        X.append(_ve_anh(o[sl], hi[sl], lo[sl], c[sl]))
        Y_rv1.append(float(y_rv[t + 1]))
        H1.append(float(h_har[t]))
        Y_resid.append(float(np.log(y_rv[t + 1]) - np.log(h_har[t])))
        G.append(int(g_doan[t]))
        ok_idx.append(t)
    return (np.stack(X), np.array(Y_resid, np.float32), np.array(Y_rv1),
           np.array(H1), np.array(G), d.Date.values[np.array(ok_idx) + 1])


class CNNNen(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def chay(pair="EURUSD"):
    """QUAN TRONG — tach doan KIEM DINH lam HAI nua theo THOI GIAN:

      nua dau  (`sel`)  — CHI dung de chon epoch dung lai som (early stopping)
      nua sau  (`ev`)   — CHUA TUNG duoc mo hinh nhin thay o BAT KY buoc nao,
                          dung DUY NHAT de bao cao QLIKE/DM

    Neu dung CHUNG mot doan kiem dinh vua de chon epoch vua de bao cao ket
    qua (loi ban dau cua ban nay), con so se lac quan gia tao vi da "nhin
    truoc" chinh doan minh dinh cham diem — dung loi ma toan bo triet ly
    Westfall-Young/sealed-set cua du an ton tai de tranh."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, y_resid, y_rv1, h_har, g, ngay = dung_du_lieu(pair)
    tr = g == 0
    vl_idx = np.flatnonzero(g == 1)
    nua = len(vl_idx) // 2
    sel, ev = vl_idx[:nua], vl_idx[nua:]           # chon epoch // bao cao — TACH BACH
    print(f"{pair}: {len(X):,} cửa sổ · huấn luyện {tr.sum():,} · "
          f"chọn epoch {len(sel):,} · báo cáo (chưa từng thấy) {len(ev):,}")

    mu_y, sd_y = y_resid[tr].mean(), y_resid[tr].std() + EPS   # chuan hoa CHI tu huan luyen

    Xt = torch.tensor(X[tr], device=dev)
    yt = torch.tensor((y_resid[tr] - mu_y) / sd_y, device=dev)
    Xs = torch.tensor(X[sel], device=dev)

    model = CNNNen().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    n = len(Xt)
    best_qlike, best_state, patience, bad = np.inf, None, 8, 0

    for epoch in range(60):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = lossf(pred, yt[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_s = model(Xs).cpu().numpy() * sd_y + mu_y     # ve lai thang do log-du
        h_ket_hop = h_har[sel] * np.exp(pred_s)                # HAR + hieu chinh CNN
        ql_kh = qlike(y_rv1[sel], h_ket_hop).mean()
        if ql_kh < best_qlike - 1e-6:
            best_qlike, best_state, bad = ql_kh, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    Xe = torch.tensor(X[ev], device=dev)
    with torch.no_grad():
        pred_e = model(Xe).cpu().numpy() * sd_y + mu_y
    h_ket_hop = h_har[ev] * np.exp(pred_e)

    ql_har = qlike(y_rv1[ev], h_har[ev])
    ql_cnn = qlike(y_rv1[ev], h_ket_hop)
    t_dm, p_dm = dm_nw(ql_har - ql_cnn)          # >0 nghia la CNN loi hon (QLIKE thap hon)

    # QLIKE o day AM (phuong sai lai suat ngay rat nho) — chia truc tiep cho
    # ql_har se DOI DAU (am/am = duong) va doc nham thanh "toi hon". Dung
    # abs(ql_har) lam mau de dau cua % KHOP voi dau thuc te: am = cai thien.
    chenh = 100 * (ql_cnn.mean() - ql_har.mean()) / abs(ql_har.mean())
    ra = dict(pair=pair, n_train=int(tr.sum()), n_chon_epoch=len(sel), n_bao_cao=len(ev),
             qlike_har=round(float(ql_har.mean()), 6),
             qlike_har_cnn=round(float(ql_cnn.mean()), 6),
             chenh_phan_tram=round(float(chenh), 4),
             dm_t=round(float(t_dm), 4), dm_p=round(float(p_dm), 4),
             ket_luan=("CNN cải thiện có ý nghĩa (p<0,05)" if p_dm < 0.05 and ql_cnn.mean() < ql_har.mean()
                       else "không cải thiện có ý nghĩa so với mốc HAR"))
    print(json.dumps(ra, ensure_ascii=False, indent=1))

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, f"kiem_cnn_nen_{pair}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("Đã ghi", outp)
    return ra


if __name__ == "__main__":
    chay(sys.argv[1] if len(sys.argv) > 1 else "EURUSD")
