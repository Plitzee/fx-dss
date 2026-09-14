"""THỬ NGHIỆM — Vision Transformer (ViT) trên CÙNG ảnh nến đã dùng cho CNN
(`kiem_cnn_nen.py`), để so sánh KIẾN TRÚC một cách công bằng (cùng dữ liệu,
cùng mã hoá ảnh, cùng giao thức huấn luyện/báo cáo — chỉ đổi kiến trúc học).

VÌ SAO THỬ VIT SAU CNN: khảo sát văn liệu 2024-2026 (Stanford CS231n 2025
"Learning Predictive Candlestick Patterns: Vision Transformers for Technical
Analysis"; arXiv 2605.00875 "Visual Chart Representations for Cryptocurrency
Regime Prediction: A Systematic Deep Learning Study") báo cáo Vision
Transformer thường vượt CNN trên đúng bài toán "học ảnh nến" — cơ chế tự chú
ý (self-attention) giữa các nến trong cửa sổ, thay vì tích chập cục bộ, có
thể bắt được quan hệ xa giữa các nến (vd nến 1 và nến 10) mà CNN 2 lớp khó
bắt trực tiếp. Nhưng văn liệu KHÔNG đồng thuận: nhiều bài khác cũng thấy
"candlestick images không thêm giá trị ngoài dữ liệu số", và độ chính xác
phân loại đỉnh ~70% dù kiến trúc nào. Đây là phép thử THỰC NGHIỆM, không
phải tin trước kết quả.

KIẾN TRÚC: patch embedding (patch 4×4, không chồng lấp) + token [CLS] +
vị trí học được + 2 lớp Transformer Encoder (4 đầu chú ý) + đầu hồi quy
tuyến tính trên CLS — kiến trúc ViT tối giản, có chủ đích: dữ liệu huấn
luyện chỉ ~3.000 cửa sổ/cặp, một ViT cỡ lớn (kiểu ViT-B pretrain ImageNet)
sẽ overfit nặng và ảnh nến (2 kênh, trừu tượng) không phải ảnh tự nhiên nên
đặc trưng pretrain trên ImageNet không chắc chuyển giao được — khớp đúng
nhận định của arXiv 2605.00875 rằng so sánh CNN/Transformer cho ảnh nến còn
CHƯA có nghiên cứu hệ thống, nên ở đây so sánh CÙNG QUY MÔ tham số với CNN.

GIAO THỨC — Y HỆT kiem_cnn_nen.py: khớp trên huấn luyện, dừng sớm trên NỬA
ĐẦU kiểm định, báo cáo QLIKE/DM trên NỬA SAU (chưa từng thấy). Học phần dư
so với log HAR, không học lại từ đầu.

Chạy:  python src/kiem_vit_nen.py [PAIR]
Ghi:   output/kiem_vit_nen_{PAIR}.json
Cần:   pip install torch (đã có, dùng lại dung_du_lieu/dm_nw từ kiem_cnn_nen.py)
"""
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")

from metrics import qlike                                     # noqa: E402
from kiem_cnn_nen import H_IMG, SEED, W, BAR_W, dm_nw, dung_du_lieu  # noqa: E402

PATCH = 4                       # kich thuoc mieng anh, khong chong lan
EMBED_DIM = 32
N_HEAD = 4
N_LAYER = 2
MLP_DIM = 64
EPS = 1e-12


class ViTNen(nn.Module):
    """ViT toi gian: patch embedding tuyen tinh + CLS + vi tri hoc duoc +
    N lop Transformer Encoder + dau hoi quy tuyen tinh tren CLS."""

    def __init__(self, img_h, img_w, in_ch=2, patch=PATCH, dim=EMBED_DIM,
                nhead=N_HEAD, depth=N_LAYER, mlp_dim=MLP_DIM):
        super().__init__()
        assert img_h % patch == 0 and img_w % patch == 0
        n_patch = (img_h // patch) * (img_w // patch)
        patch_dim = in_ch * patch * patch
        self.patch = patch
        self.embed = nn.Linear(patch_dim, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patch + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        lop = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=mlp_dim,
                                         dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(lop, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def _patchify(self, x):
        # x: (B, C, H, W) -> (B, n_patch, C*patch*patch)
        B, C, H, Wd = x.shape
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)          # (B,C,H/p,W/p,p,p)
        x = x.contiguous().view(B, C, -1, p, p)         # (B,C,n_patch,p,p)
        x = x.permute(0, 2, 1, 3, 4).contiguous()       # (B,n_patch,C,p,p)
        return x.view(B, x.shape[1], -1)                # (B,n_patch,C*p*p)

    def forward(self, x):
        B = x.shape[0]
        tok = self.embed(self._patchify(x))             # (B,n_patch,dim)
        cls = self.cls.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1) + self.pos
        out = self.encoder(tok)
        return self.head(self.norm(out[:, 0])).squeeze(-1)


def chay(pair="EURUSD"):
    """Giao thuc GIONG HET kiem_cnn_nen.chay() — chi doi kien truc mo hinh,
    de so sanh CNN vs ViT tren CUNG mot du lieu/giao thuc."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, y_resid, y_rv1, h_har, g, ngay = dung_du_lieu(pair)
    tr = g == 0
    vl_idx = np.flatnonzero(g == 1)
    nua = len(vl_idx) // 2
    sel, ev = vl_idx[:nua], vl_idx[nua:]
    print(f"{pair}: {len(X):,} cửa sổ · huấn luyện {tr.sum():,} · "
          f"chọn epoch {len(sel):,} · báo cáo (chưa từng thấy) {len(ev):,}")

    mu_y, sd_y = y_resid[tr].mean(), y_resid[tr].std() + EPS

    Xt = torch.tensor(X[tr], device=dev)
    yt = torch.tensor((y_resid[tr] - mu_y) / sd_y, device=dev)
    Xs = torch.tensor(X[sel], device=dev)

    img_h, img_w = X.shape[2], X.shape[3]
    model = ViTNen(img_h, img_w).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
            pred_s = model(Xs).cpu().numpy() * sd_y + mu_y
        h_ket_hop = h_har[sel] * np.exp(pred_s)
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
    ql_vit = qlike(y_rv1[ev], h_ket_hop)
    t_dm, p_dm = dm_nw(ql_har - ql_vit)

    chenh = 100 * (ql_vit.mean() - ql_har.mean()) / abs(ql_har.mean())
    ra = dict(pair=pair, n_train=int(tr.sum()), n_chon_epoch=len(sel), n_bao_cao=len(ev),
             n_tham_so=sum(p.numel() for p in model.parameters()),
             qlike_har=round(float(ql_har.mean()), 6),
             qlike_har_vit=round(float(ql_vit.mean()), 6),
             chenh_phan_tram=round(float(chenh), 4),
             dm_t=round(float(t_dm), 4), dm_p=round(float(p_dm), 4),
             ket_luan=("ViT cải thiện có ý nghĩa (p<0,05)" if p_dm < 0.05 and ql_vit.mean() < ql_har.mean()
                       else "không cải thiện có ý nghĩa so với mốc HAR"))
    print(json.dumps(ra, ensure_ascii=False, indent=1))

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, f"kiem_vit_nen_{pair}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("Đã ghi", outp)
    return ra


if __name__ == "__main__":
    chay(sys.argv[1] if len(sys.argv) > 1 else "EURUSD")
