"""ML / DL / HOC TRUC TUYEN CO THANG DUOC BA O KHONG?

Cau hoi nay CHUA AI TRA LOI. Bang 14 mo hinh o docs/ML_DL_VONG7.md do QLIKE cua
PHUONG SAI — bai toan hoi quy. Day la bai toan PHAN LOAI ba lop, ham mat khac,
muc tieu khac. Ket luan cua bang kia khong ap sang day duoc.

────────────────────────────────────────────────────────────────────────────
"RL" O DAY LA HOC TRUC TUYEN, KHONG PHAI PPO
────────────────────────────────────────────────────────────────────────────
docs/SIZING_COMPARISON.md da loai PPO — nhung do la RL hoc CHINH SACH DINH CO
VI THE. Thu khac han voi "hom nay du bao sai thi mai sua". Cai sau la hoc truc
tuyen co phan hoi, va ho thuat toan dung cho no la TRONG SO MU (Hedge /
exponential weights): moi phien, chuyen gia nao vua sai thi bi ha trong so theo
ham mu cua ton that. No co CHAN HOI TIEC (regret bound), re, va khong can mo
phong moi truong. Van lieu 2025-2026 goi ho nay la online ensembling duoi troi
khai niem (OneNet, SAOCP, AdaWeather).

────────────────────────────────────────────────────────────────────────────
CAI BAY PHAI CHAN — da thay that o nhanh TSF
────────────────────────────────────────────────────────────────────────────
Mot bo phan loai 3 lop khong co tin hieu se hoi tu ve TAN SUAT NEN, tuc dung
bang khi hau hoc, ma loss van giam dep. O nhanh TSF dieu nay da xay ra o dang
nang hon: CAIFormer xuat `mu` = 0 tuyet doi tren ca 419 diem, ba mo hinh khac
xuat hang so (do lech chuan 1e-11). Nen o day co CHOT SUY BIEN: do do lech
chuan cua P theo ngay, gan 0 thi bao ngay.

Giao thuc: khop tren HUAN LUYEN, chon tren KIEM DINH, cham KIEM TRA dung mot
lan. Nguong phai vuot lay tu docs/GIAIDOAN1_NEN.md.

Chay:  python src/run_ml3.py
Ghi:   output/ml3.json, output/log_ml3.txt
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

import balop as B                                          # noqa: E402
import diem3 as D                                          # noqa: E402
import ml_data as MD                                       # noqa: E402
import volfc2 as V2                                        # noqa: E402
from split import VALID_TU, TEST_TU, doan                  # noqa: E402
from volfc import merge_thin_days                          # noqa: E402

H = 1
SEED = 0
EPS = 1e-12


# ── du lieu ─────────────────────────────────────────────────────────────
def nap():
    """Tra ve X (dac trung), cac dich 3 lop, va nen san xuat hien hanh."""
    from api.main import noi_chuoi
    bang, pan = {}, {}
    for p in B.PAIRS:
        m = merge_thin_days(noi_chuoi(p))
        sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, p), 0.0))
        c = m.close.values
        zt = np.full(len(m), np.nan)
        zt[1:] = np.log(c[1:] / np.maximum(c[:-1], EPS)) / np.maximum(sig[1:], EPS)
        bang[p] = m
        pan[p] = pd.DataFrame({"Date": m.Date.values, "sig": sig, "zT": zt})

    chung = pan[B.PAIRS[0]].Date.values
    for p in B.PAIRS[1:]:
        chung = np.intersect1d(chung, pan[p].Date.values)
    chung = pd.DatetimeIndex(chung)

    # `ml_data.xay` gia dinh MOI bang da cat ve dung tap ngay chung — no dung
    # n = len(chung) de dung mang roi gan thang tu d.rv5, nen lech do dai la vo
    # ngay lap tuc. Cat truoc khi truyen.
    bang = {p: b[b.Date.isin(chung)].sort_values("Date").reset_index(drop=True)
            for p, b in bang.items()}
    for p, b in bang.items():
        assert len(b) == len(chung), f"{p}: {len(b)} != {len(chung)}"

    X, _, ten, pid, dts = MD.xay(bang, chung)

    # ── CAN CHINH — cho da tra gia mot lan ───────────────────────────────
    # `ml_data.xay` dat y[t] = log rv5[t+1]: hang t la dac trung DE DU BAO
    # NGAY t+1. Nhung `balop.dung_muc_tieu` tra ve lop CUA CHINH ngay t.
    # Ghep thang hai cai do la RO RI TRUC TIEP: X[t] chua lrv_d = log rv5[t]
    # (phuong sai thuc hien cua chinh ngay t) va lrsp/lrsn (ban phuong sai
    # duong/am cua ngay t — tuc cho thang DAU cua phien do). Lan chay dau cho
    # AUC 0,967 va BSS +0,35 tren muc tieu R — con so khong the that, va do
    # dung la mo hinh doc dap an trong dac trung.
    # Nen phai DICH dich len mot phien: hang t <-> lop cua ngay t+1.
    yR, yP, phu = [], [], []
    for ip, p in enumerate(B.PAIRS):
        d = pan[p][pan[p].Date.isin(chung)].reset_index(drop=True)
        tr = doan(d.Date.values) == 0
        T = B.dung_muc_tieu(d, H, tr)

        def dich(a, thieu=-1):
            b = np.full(len(a), thieu, dtype=a.dtype)
            b[:-1] = a[1:]
            return b

        yR.append(dich(T["yR"]))
        yP.append(dich(T["yP"]))
        phu.append(pd.DataFrame({
            "pair": p,
            "Date": d.Date.values,                      # ngay cua DAC TRUNG
            "ngay_dich": np.r_[d.Date.values[1:], np.datetime64("NaT")],
            "sig": dich(d.sig.values, np.nan),
            "canh_R": dich(T["canh_R"], np.nan),
            "canh_P": dich(T["canh_P"], np.nan),
            "sigma_h": dich(T["sigma_h"], np.nan),
            "z": dich(T["z"], np.nan)}))
    phu = pd.concat(phu, ignore_index=True)

    # CHOT: ngay cua dich phai NAM SAU ngay cua dac trung, moi hang.
    m = phu.ngay_dich.notna()
    assert (phu.ngay_dich[m].values > phu.Date[m].values).all(), \
        "dich khong nam sau dac trung — con ro ri"

    # hop le = dac trung du, dich du, va dau vao cua nen du
    hop_le = (np.isfinite(X).all(1) & phu.ngay_dich.notna().values
              & np.isfinite(phu.sigma_h.values) & np.isfinite(phu.canh_P.values)
              & np.isfinite(phu.z.values))
    return (X, np.concatenate(yR), np.concatenate(yP), phu, ten, pid, dts,
            hop_le)


# ── mo hinh ─────────────────────────────────────────────────────────────
def _chuan(P):
    P = np.clip(np.asarray(P, float), 1e-9, None)
    return P / P.sum(1, keepdims=True)


def khi_hau_hoc(ytr, n):
    c = np.bincount(ytr[ytr >= 0], minlength=3).astype(float)
    return _chuan(np.tile(c / c.sum(), (n, 1)))


def logistic_da_thuc(Xtr, ytr, Xte, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, C=C)   # sklearn moi mac dinh la da thuc
    m.fit(s.transform(Xtr), ytr)
    return _chuan(m.predict_proba(s.transform(Xte))), m


def lightgbm_3lop(Xtr, ytr, Xte, la=31, nl=300, lr=0.05):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=nl,
                           learning_rate=lr, num_leaves=la, min_child_samples=60,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                           reg_lambda=1.0, random_state=SEED, verbose=-1)
    m.fit(Xtr, ytr)
    return _chuan(m.predict_proba(Xte)), m


def gru_3lop(Xtr, ytr, Xte, ep=30, an=48, lr=1e-3, L=20):
    """GRU tren cua so L phien cua chinh vector dac trung."""
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8

    def cua_so(X):
        Z = (X - mu) / sd
        n, k = Z.shape
        pad = np.repeat(Z[:1], L - 1, axis=0)
        Zp = np.vstack([pad, Z])
        return np.stack([Zp[i:i + L] for i in range(n)]).astype(np.float32)

    class M(nn.Module):
        def __init__(s, k):
            super().__init__()
            s.g = nn.GRU(k, an, batch_first=True)
            s.o = nn.Linear(an, 3)

        def forward(s, x):
            h, _ = s.g(x)
            return s.o(h[:, -1])

    m = M(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lf = nn.CrossEntropyLoss()
    Wtr = torch.tensor(cua_so(Xtr))
    Ytr = torch.tensor(ytr.astype(np.int64))
    n = len(Wtr)
    for _ in range(ep):
        idx = torch.randperm(n)
        for i in range(0, n, 256):
            j = idx[i:i + 256]
            opt.zero_grad()
            lf(m(Wtr[j]), Ytr[j]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        P = torch.softmax(m(torch.tensor(cua_so(Xte))), 1).numpy()
    return _chuan(P), m


def trong_so_mu(Ps, y, eta=0.5, nhom=None):
    """HOC TRUC TUYEN — Hedge / trong so mu.

    Ps  : dict ten -> mang (n,3) du bao cua tung chuyen gia
    y   : lop thuc, dung de CAP NHAT SAU KHI da du bao (khong ro ri)
    eta : toc do hoc

    Moi phien: du bao bang to hop co trong so hien tai, RỒI moi nhin ket cuc va
    ha trong so cua chuyen gia vua sai theo ham mu cua ton that log. Day la
    dung nghia "hom nay sai thi mai sua", va no co chan hoi tiec
    O(sqrt(T log N)) so voi chuyen gia tot nhat nhin lai."""
    ten = list(Ps)
    A = np.stack([Ps[t] for t in ten])                 # (N, n, 3)
    N, n, _ = A.shape
    w = np.ones(N) / N
    ra = np.zeros((n, 3))
    lich_su = np.zeros((n, N))
    g = np.zeros(n, int) if nhom is None else pd.factorize(np.asarray(nhom))[0]
    ws = {v: np.ones(N) / N for v in np.unique(g)}      # trong so RIENG tung cap
    for t in range(n):
        wt = ws[g[t]]
        ra[t] = _chuan((wt[:, None] * A[:, t, :]).sum(0)[None, :])[0]
        lich_su[t] = wt
        if y[t] < 0:
            continue
        l = -np.log(np.maximum(A[:, t, y[t]], 1e-9))    # ton that log tung chuyen gia
        wt = wt * np.exp(-eta * (l - l.min()))
        ws[g[t]] = wt / max(wt.sum(), EPS)
    return _chuan(ra), ten, lich_su


# ── chan suy bien ───────────────────────────────────────────────────────
def chot_suy_bien(P, ten):
    """Bat dung cai bay da thay o nhanh TSF: mo hinh xuat HANG SO."""
    sd = P.std(0)
    n_kh = len(np.unique(np.round(P, 6), axis=0))
    xau = sd.max() < 1e-6
    return dict(sd_max=float(sd.max()), so_du_bao_khac_nhau=int(n_kh),
                suy_bien=bool(xau))


def bang(P, y, Pkh, nhom, ten):
    m = y >= 0
    P, y, Pkh, nhom = P[m], y[m], Pkh[m], np.asarray(nhom)[m]
    r = D.bang(P, y, Pkh, nhom=nhom)
    lo, hi = D.bss_ktc(P, y, Pkh, nhom=nhom, nboot=300, khoi=20, seed=7)
    alo, ahi = D.auc_ktc(P, y, nhom=nhom, nboot=300, khoi=20, seed=7)
    r.update(bss_lo=lo, bss_hi=hi, auc_lo=alo, auc_hi=ahi,
             **chot_suy_bien(P, ten))
    return r


def in_bang(tieu_de, hang):
    print(f"\n  {tieu_de}")
    print(f"  {'mô hình':<24}{'log':>8}{'BSS':>9}{'KTC 95% của BSS':>22}"
          f"{'ECE':>8}{'AUC':>8}{'KTC 95%':>17}{'sd(P)':>9}")
    for t, r in hang:
        ktc_bss = "[{:+.4f}; {:+.4f}]".format(r["bss_lo"], r["bss_hi"])
        ktc_auc = "[{:.3f}; {:.3f}]".format(r["auc_lo"], r["auc_hi"])
        sao = " *" if r["bss_lo"] > 0 else ""
        sb = "  ⚠ HẰNG SỐ" if r["suy_bien"] else ""
        print(f"  {t:<24}{r['log']:>8.4f}{r['bss']:>+9.4f}{ktc_bss:>22}"
              f"{r['ece']:>8.4f}{r['auc']:>8.3f}{ktc_auc:>17}"
              f"{r['sd_max']:>9.4f}{sao}{sb}")


def main():
    t0 = time.time()
    print("=" * 118)
    print("ML / DL / HỌC TRỰC TUYẾN CÓ THẮNG ĐƯỢC BA Ô KHÔNG?")
    print("=" * 118)
    X, yR, yP, phu, ten, pid, dts, hop_le = nap()
    tr = (dts < VALID_TU) & hop_le
    va = (dts >= VALID_TU) & (dts < TEST_TU) & hop_le
    te = (dts >= TEST_TU) & hop_le
    print(f"loại {int((~hop_le).sum()):,} hàng thiếu đặc trưng hoặc thiếu đích")
    i0 = int(np.flatnonzero(hop_le)[0])
    print(f"căn chỉnh: đặc trưng ngày {str(phu.Date.values[i0])[:10]} "
          f"→ đích ngày {str(phu.ngay_dich.values[i0])[:10]}  (phải LỆCH một phiên)")
    print(f"bảng dài {X.shape[0]:,} hàng × {X.shape[1]} đặc trưng, 6 cặp gộp")
    print(f"huấn luyện {int(tr.sum()):,} · kiểm định {int(va.sum()):,} · "
          f"kiểm tra {int(te.sum()):,}")
    print("ngưỡng phải vượt (docs/GIAIDOAN1_NEN.md): mục tiêu P h=1 BSS > +0,0105")
    print("                                          mục tiêu R h=1 BSS > 0\n")

    ket = {}
    for ten_mt, y in (("R", yR), ("P", yP)):
        print("─" * 118)
        print(f"MỤC TIÊU {ten_mt}" + ("  (σ̂ đã chia ra — mọi kỹ năng ở đây là kỹ năng VƯỢT TRÊN tầng 2)"
                                      if ten_mt == "R" else "  (dải cố định — thứ hiển thị trên giao diện)"))
        ok = y >= 0
        Xtr, ytr = X[tr & ok], y[tr & ok]
        Xva, Xte = X[va], X[te]

        Ps_va, Ps_te = {}, {}
        Ps_va["khí hậu học"] = khi_hau_hoc(ytr, int(va.sum()))
        Ps_te["khí hậu học"] = khi_hau_hoc(ytr, int(te.sum()))

        # nen san xuat hien hanh
        for nen_ten, lop in (("chỉ σ̂", B.ChiSigma), ("σ̂ + chế độ", B.SigmaCheDo)):
            Pv = np.zeros((len(X), 3))
            for p in B.PAIRS:
                mp = phu.pair.values == p
                z = phu.z.values[mp]
                trp = tr[mp]
                mo = lop()
                mo.khop(z[trp]) if lop is B.ChiSigma else mo.khop(z[trp], phu.sig.values[mp][trp])
                canh = phu["canh_R" if ten_mt == "R" else "canh_P"].values[mp]
                Pv[mp] = mo.du_bao(int(mp.sum()), canh=canh,
                                   sigma_h=phu.sigma_h.values[mp],
                                   sig=phu.sig.values[mp])
            Pv = np.where(np.isfinite(Pv), Pv, 1.0 / 3.0)
            Ps_va[nen_ten], Ps_te[nen_ten] = _chuan(Pv[va]), _chuan(Pv[te])

        print("  đang khớp… ", end="", flush=True)
        for nhan, ham in (("logistic đa thức", logistic_da_thuc),
                          ("LightGBM 3 lớp", lightgbm_3lop),
                          ("GRU 3 lớp", gru_3lop)):
            try:
                Pv, _ = ham(Xtr, ytr, Xva)
                Pt, _ = ham(np.vstack([Xtr, Xva[y[va] >= 0]]),
                            np.concatenate([ytr, y[va][y[va] >= 0]]), Xte)
                Ps_va[nhan], Ps_te[nhan] = Pv, Pt
                print(f"{nhan} ✓ ", end="", flush=True)
            except Exception as e:
                print(f"{nhan} ✗({str(e)[:40]}) ", end="", flush=True)
        print()

        # hoc truc tuyen tren KIEM TRA — dung nghia "hom nay sai thi mai sua"
        chuyen_gia = {k: v for k, v in Ps_te.items() if k != "khí hậu học"}
        Pon, ten_cg, ls = trong_so_mu(chuyen_gia, y[te],
                                      nhom=phu.pair.values[te])
        Ps_te["học trực tuyến (Hedge)"] = Pon

        hv, ht = [], []
        for k in Ps_va:
            hv.append((k, bang(Ps_va[k], y[va], Ps_va["khí hậu học"],
                               phu.pair.values[va], k)))
        for k in Ps_te:
            ht.append((k, bang(Ps_te[k], y[te], Ps_te["khí hậu học"],
                               phu.pair.values[te], k)))
        in_bang("KIỂM ĐỊNH (dùng để chọn)", hv)
        in_bang("KIỂM TRA (chấm một lần)", ht)
        print(f"\n  trọng số cuối của học trực tuyến: "
              + " · ".join(f"{t}={w:.2f}" for t, w in zip(ten_cg, ls[-1])))
        ket[ten_mt] = {"kiem_dinh": {k: v for k, v in hv},
                       "kiem_tra": {k: v for k, v in ht},
                       "trong_so_cuoi": dict(zip(ten_cg, ls[-1].tolist()))}

    with open(os.path.join(OUT, "ml3.json"), "w", encoding="utf-8") as f:
        json.dump(ket, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n{'=' * 118}\nđã ghi output/ml3.json — {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
