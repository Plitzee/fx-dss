"""VONG 7 — HOC SAU: LSTM, GRU, Transformer (PatchTST rut gon).

Cung giao thuc voi run_ml.py: cung tap thong tin, khop lai dau moi nam bang
cua so mo rong, cung cach doi log -> phuong sai, sieu tham so chon tren doan
KIEM DINH, cham diem mot lan tren KIEM TRA.

Dau vao mang tuan tu: 22 phien gan nhat cua nam kenh
    [log rv5, log rsp, log rsn, log(sqrt(rq5)/rv5), z che do]
chuan hoa bang trung binh/do lech cua RIENG doan huan luyen moi lan khop.
Dac trung TINH (lich NHTW cua ngay t+1, thu trong tuan, ma cap) noi vao dau
ra cua bo ma hoa truoc lop tuyen tinh cuoi.

Muc tieu: log rv5[t+1]. Ham mat mat MSE tren log; doi ve phuong sai bang
hieu chinh log-chuan giong het HAR.

Quy mo mo hinh co y de NHO (hidden 48-64, 1-2 lop). Ly do khong phai luoi:
voi ~15.000 mau huan luyen va 5 kenh, mo hinh lon hon chi overfit nhanh hon.
Do la chinh ket luan cua Branco et al. (2024) va Kilic (2025).
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")

from split import VALID_TU, TEST_TU

SEQ = 22
KHOP_TU, KHOP_DEN = 2015, 2026
MIN_TRAIN = 3000
EPOCHS = 60
PATIENCE = 8
BATCH = 256


def nap():
    z = np.load(os.path.join(OUT, "_ml_feat.npz"))
    ten = json.load(open(os.path.join(OUT, "_ml_cols.json")))
    return z["X"], z["y"], ten, z["pid"], pd.DatetimeIndex(z["dts"])


def xay_chuoi(X, y, ten, pid, dts):
    """Tra ve (S, F, y, dts, pid, hople): S la (n, SEQ, 5), F la đặc trưng tĩnh."""
    kenh = ["lrv_d", "lrsp", "lrsn", "lq", "z"]
    ik = [ten.index(c) for c in kenh]
    lag = [ten.index(f"lrv_lag{k}") for k in range(1, SEQ)]
    tinh = [ten.index(c) for c in ten
            if c.startswith(("ev_", "dow", "pair"))] + [ten.index("lrv_m"),
                                                        ten.index("lrv_q"),
                                                        ten.index("G")]
    n = len(y)
    S = np.full((n, SEQ, len(ik)), np.nan, np.float32)
    # kênh 0 (log rv) có đủ độ trễ; các kênh khác chỉ có giá trị tại t nên
    # dựng lại bằng cách dịch trong từng cặp
    for j, c in enumerate(ik):
        v = X[:, c]
        for p in np.unique(pid):
            m = np.where(pid == p)[0]
            vv = v[m]
            for k in range(SEQ):
                col = np.full(len(m), np.nan)
                if k == 0:
                    col = vv
                else:
                    col[k:] = vv[:-k]
                S[m, SEQ - 1 - k, j] = col
    F = X[:, tinh].astype(np.float32)
    hople = np.isfinite(S).all((1, 2)) & np.isfinite(F).all(1) & np.isfinite(y)
    return S, F, hople, [ten[i] for i in tinh]


def _mo_hinh(kieu, nk, nf, hid, torch, nn):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kieu = kieu
            if kieu == "lstm":
                self.enc = nn.LSTM(nk, hid, num_layers=1, batch_first=True)
            elif kieu == "gru":
                self.enc = nn.GRU(nk, hid, num_layers=1, batch_first=True)
            else:                                   # transformer rút gọn
                self.inp = nn.Linear(nk, hid)
                self.pos = nn.Parameter(torch.zeros(1, SEQ, hid))
                lay = nn.TransformerEncoderLayer(hid, nhead=4,
                                                 dim_feedforward=2 * hid,
                                                 dropout=0.1, batch_first=True,
                                                 norm_first=True)
                self.enc = nn.TransformerEncoder(lay, num_layers=2)
            self.head = nn.Sequential(nn.Linear(hid + nf, 64), nn.GELU(),
                                      nn.Dropout(0.1), nn.Linear(64, 1))

        def forward(self, s, f):
            if self.kieu in ("lstm", "gru"):
                o, _ = self.enc(s)
                h = o[:, -1]
            else:
                h = self.enc(self.inp(s) + self.pos)[:, -1]
            return self.head(torch.cat([h, f], 1)).squeeze(-1)
    return Net()


def khop_mot_lan(S, F, y, itr, ite, kieu, hid, lr, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(2)
    # tách 12% cuối của đoạn huấn luyện làm tập dừng sớm (theo THỜI GIAN)
    k = int(len(itr) * 0.88)
    i_fit, i_es = itr[:k], itr[k:]
    mS = S[i_fit].reshape(-1, S.shape[2]).mean(0)
    sS = S[i_fit].reshape(-1, S.shape[2]).std(0) + 1e-8
    mF = F[i_fit].mean(0); sF = F[i_fit].std(0) + 1e-8
    my = y[i_fit].mean(); sy = y[i_fit].std() + 1e-8

    def T(idx):
        return (torch.tensor((S[idx] - mS) / sS, dtype=torch.float32),
                torch.tensor((F[idx] - mF) / sF, dtype=torch.float32),
                torch.tensor((y[idx] - my) / sy, dtype=torch.float32))
    Str, Ftr, ytr = T(i_fit)
    Ses, Fes, yes = T(i_es)
    net = _mo_hinh(kieu, S.shape[2], F.shape[1], hid, torch, nn)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best = (1e18, None, 0)
    nb = len(ytr)
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(nb)
        for i in range(0, nb, BATCH):
            j = perm[i:i + BATCH]
            opt.zero_grad()
            l = lossf(net(Str[j], Ftr[j]), ytr[j])
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(Ses, Fes), yes))
        if v < best[0] - 1e-5:
            best = (v, {k2: p.detach().clone() for k2, p in net.state_dict().items()}, ep)
        elif ep - best[2] >= PATIENCE:
            break
    net.load_state_dict(best[1])
    net.eval()
    with torch.no_grad():
        Sa, Fa, _ = T(itr)
        r = y[itr] - (net(Sa, Fa).numpy() * sy + my)
        Sb, Fb, _ = T(ite)
        mu = net(Sb, Fb).numpy() * sy + my
    return mu, float(r.var()), best[2] + 1


def walkforward(S, F, y, dts, hople, kieu, hid, lr):
    n = len(y)
    mu = np.full(n, np.nan); s2 = np.full(n, np.nan); eps = []
    for yr in range(KHOP_TU, KHOP_DEN + 1):
        moc = pd.Timestamp(f"{yr}-01-01"); het = pd.Timestamp(f"{yr+1}-01-01")
        itr = np.where(hople & (dts < moc))[0]
        ite = np.where(hople & (dts >= moc) & (dts < het))[0]
        if len(itr) < MIN_TRAIN or len(ite) == 0:
            continue
        m, v, e = khop_mot_lan(S, F, y, itr, ite, kieu, hid, lr)
        mu[ite] = m; s2[ite] = v; eps.append(e)
    return mu, s2, eps


def main():
    X, y, ten, pid, dts = nap()
    S, F, hople, ten_tinh = xay_chuoi(X, y, ten, pid, dts)
    rv = np.exp(y)
    va = (dts >= VALID_TU) & (dts < TEST_TU)
    print("=" * 100)
    print("HỌC SÂU — LSTM / GRU / Transformer")
    print("=" * 100)
    print(f"chuỗi {S.shape[0]:,} × {SEQ} phiên × {S.shape[2]} kênh, "
          f"{F.shape[1]} đặc trưng tĩnh, {int(hople.sum()):,} hàng đủ dữ liệu")
    print(f"khớp lại đầu mỗi năm {KHOP_TU}–{KHOP_DEN}, dừng sớm trên 12% cuối "
          f"của đoạn huấn luyện\n")

    GRID = [("lstm", 48, 2e-3), ("lstm", 96, 1e-3),
            ("gru", 48, 2e-3), ("gru", 96, 1e-3),
            ("tst", 64, 1e-3)]
    TENHO = {"lstm": "LSTM", "gru": "GRU", "tst": "Transformer (PatchTST rút gọn)"}
    kq = []
    t0 = time.time()
    for kieu, hid, lr in GRID:
        mu, s2, eps = walkforward(S, F, y, dts, hople, kieu, hid, lr)
        f = np.exp(np.clip(mu, -30, 0) + 0.5 * np.nan_to_num(s2))
        ok = va & np.isfinite(f) & (f > 0)
        r = rv[ok] / f[ok]
        q = float((r - np.log(r) - 1).mean())
        kq.append(dict(ten=f"{TENHO[kieu]} h={hid}", f=f, qlike_valid=q,
                       hp=f"hid={hid}, lr={lr}"))
        print(f"  {TENHO[kieu]:<32} hid={hid:<4} lr={lr:<7} "
              f"QLIKE kiểm định {q:.4f}   epoch TB {np.mean(eps):.0f}"
              f"   ({time.time()-t0:.0f}s)")

    np.savez_compressed(os.path.join(OUT, "_dl_pred.npz"),
                        **{f"f{i}": k["f"] for i, k in enumerate(kq)},
                        ten=np.array([k["ten"] for k in kq]),
                        hp=np.array([k["hp"] for k in kq]),
                        qv=np.array([k["qlike_valid"] for k in kq]))
    print("\nXẾP HẠNG TRÊN ĐOẠN KIỂM ĐỊNH")
    print("-" * 100)
    for k in sorted(kq, key=lambda z: z["qlike_valid"]):
        print(f"  {k['ten']:<34}{k['qlike_valid']:>10.4f}   {k['hp']}")
    print("-" * 100)
    print("đã ghi output/_dl_pred.npz")


if __name__ == "__main__":
    main()
