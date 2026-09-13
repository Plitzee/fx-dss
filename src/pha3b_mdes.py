"""PHA 3B / WEEK 3 — MDES: phieu nay bat duoc hieu ung NHO DEN DAU?

Bat buoc theo `docs/PHA3B_TIEUCHI.md` muc 5c va HuyH Week 3 ("Power/MDES ...
Nếu không thấy gain: MDES mandatory").

Khong co muc nay thi ket luan am KHONG phat bieu duoc thanh gi — khong phan
biet duoc "khong co gi" voi "co nhung ta khong du suc thay". Day dung la lo
hong 2.2 ma `NHANXET_ROADMAP.md` chi ra trong roadmap goc.

HAI QUYET DINH THIET KE — ca hai deu de tranh do luc GIA:

  1. CUA phai la CUA THAT. Luc do bang phan phoi null cua THONG KE MAX lay
     tu chinh lan chay Week 2 (`output/pha3b_null_max.npy`, 1.000 hoan vi khoi
     tren 594 gia thuyet). KHONG xap xi bang Sidak tren p tung gia thuyet —
     voi so hoan vi huu han, xap xi do khong bao gio xuong duoi 0,05 va se bao
     "luc 0%" cho moi co hieu ung, mot ket qua GIA.

  2. VAT MANG hieu ung phai la DAC TRUNG THAT. Tiem qua thanh phan chinh thu
     nhat cua mot KHOI dac trung that (DGS2 lag 1 — bien KHONG song sot Week 2),
     nen cau truc cong tuyen trong khoi va phu thuoc chuoi deu giu nguyen nhu
     du lieu that. Tiem mot bien Gauss doc lap se cho luc LAC QUAN.

Lap lai co ngau nhien bang BOOTSTRAP KHOI phan du cua mo hinh moc — giu
phuong sai va phu thuoc chuoi ngan han cua chuoi that.

DOI CHUNG AM bat buoc: beta = 0 phai cho ty le phat hien <= alpha.

Chay:  python src/pha3b_mdes.py
Ghi:   output/pha3b_mdes.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import pha3b_dactrung as DT                                   # noqa: E402
import pha3b_granger as G                                     # noqa: E402
from pha3b_ablation import khop_bien_do, qlike                # noqa: E402

SEED = 20260912
N_LAN = 300
ALPHA = 0.05
BETA = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24]
VAT_MANG = "DGS2"          # khoi dac trung that lam vat mang — KHONG song sot Week 2
LAG_MANG = 1
EPS = 1e-12


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("=" * 100)
    print("PHA 3B — MDES: PHỄU NÀY BẮT ĐƯỢC HIỆU ỨNG NHỎ ĐẾN ĐÂU?")
    print("bắt buộc theo PHA3B_TIEUCHI.md mục 5c · HuyH Week 3")
    print("=" * 100)

    f_null = np.load(os.path.join(OUT, "pha3b_null_max.npy"))
    nguong = float(np.quantile(f_null, 1 - ALPHA))
    print(f"CỬA THẬT — phân phối null của thống kê max, lấy từ lần chạy Week 2:")
    print(f"  {len(f_null):,} hoán vị khối trên 594 giả thuyết · "
          f"ngưỡng F(95%) = {nguong:.3f}")
    print(f"  (đối chiếu: F quan sát lớn nhất ở Week 2 = 67,52 — su_kien lag 1)")

    df = G.dung_bang()
    g = df.doan.values
    tr, te = g == 0, g == 2
    pt = g <= 1
    cols = DT.cot_lag(VAT_MANG, LAG_MANG)
    print(f"\nVẬT MANG: khối {VAT_MANG} lag {LAG_MANG} ({len(cols)} cột) — "
          f"biến KHÔNG sống sót Week 2")

    # ── chuan bi theo tung cap: moc, phan du, vat mang (PC1 cua khoi)
    goi = []
    for p in sorted(df.pair.unique()):
        m = (df.pair.values == p) & pt
        sub = df.loc[m, ["y_bien_do", "m_har"] + cols]
        ok = sub.notna().all(1).values
        idx = np.where(m)[0][ok]
        y = df.y_bien_do.values[idx]
        X0 = np.column_stack([np.ones(len(idx)), df.m_har.values[idx]])
        B = np.column_stack([df[c].values[idx] for c in cols])
        Bc = (B - B.mean(0)) / np.maximum(B.std(0), EPS)
        u, s, vt = np.linalg.svd(Bc - Bc.mean(0), full_matrices=False)
        pc1 = u[:, 0] * s[0]
        pc1 = (pc1 - pc1.mean()) / pc1.std()          # vat mang chuan hoa
        b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
        goi.append(dict(idx=idx, y=y, X0=X0, B=B, pc1=pc1,
                        fit=X0 @ b0, du=y - X0 @ b0,
                        Q0=np.linalg.qr(X0)[0],
                        Q1=np.linalg.qr(np.column_stack([X0, B]))[0],
                        n=len(idx), q=B.shape[1], k1=2 + B.shape[1]))
    print(f"  {len(goi)} cặp · {sum(x['n'] for x in goi):,} hàng phát hiện")

    # ── goi TOAN BO doan (ke ca kiem tra) — chi dung de QUY DOI do lon hieu ung
    toan_bo = []
    for p in sorted(df.pair.unique()):
        m = df.pair.values == p
        sub = df.loc[m, ["y_bien_do", "m_har"] + cols]
        ok = sub.notna().all(1).values
        idx = np.where(m)[0][ok]
        y = df.y_bien_do.values[idx]
        X0 = np.column_stack([np.ones(len(idx)), df.m_har.values[idx]])
        B = np.column_stack([df[c].values[idx] for c in cols])
        Bc = (B - B.mean(0)) / np.maximum(B.std(0), EPS)
        u, s, _ = np.linalg.svd(Bc - Bc.mean(0), full_matrices=False)
        pc1 = u[:, 0] * s[0]
        pc1 = (pc1 - pc1.mean()) / pc1.std()
        b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
        toan_bo.append(dict(idx=idx, fit=X0 @ b0, du=y - X0 @ b0, pc1=pc1))
    print(f"  quy đổi độ lớn hiệu ứng trên toàn bộ "
          f"{sum(len(x['idx']) for x in toan_bo):,} hàng (gồm cả kiểm tra)")

    print(f"\n{N_LAN} lần lặp/mức · bootstrap KHỐI {G.KHOI} phiên trên phần dư\n")
    print(f"{'β':>6}{'ΔQLIKE kiểm tra':>18}{'lực (cửa W-Y thật)':>21}"
          f"{'F trung vị':>13}  {'kết luận'}")
    print("-" * 100)
    ket = []
    for beta in BETA:
        bat, Fs, dq = 0, [], []
        for lan in range(N_LAN):
            F_lan = []
            for x in goi:
                idx = G.hoan_vi_khoi(x["n"], rng)          # bootstrap khoi phan du
                ys = x["fit"] + beta * x["pc1"] + x["du"][idx]
                r0 = float(np.sum((ys - x["Q0"] @ (x["Q0"].T @ ys)) ** 2))
                r1 = float(np.sum((ys - x["Q1"] @ (x["Q1"].T @ ys)) ** 2))
                F_lan.append(((r0 - r1) / x["q"]) / max(r1 / (x["n"] - x["k1"]),
                                                        EPS))
            F = float(np.max(F_lan))       # cap manh nhat — dung cach W-Y hoat dong
            Fs.append(F)
            bat += int(F > nguong)
        # do lon hieu ung theo don vi DOC DUOC: % QLIKE kiem tra
        #
        # PHAI tiem tren MOI doan (ke ca kiem tra). Neu chi tiem vao doan phat
        # hien roi do QLIKE tren doan kiem tra CHUA tiem thi con so vo nghia —
        # mo hinh hoc mot hieu ung khong ton tai o noi duoc cham diem.
        dfx = df.copy()
        ys_full = df.y_bien_do.values.copy()
        for x in toan_bo:
            ys_full[x["idx"]] = x["fit"] + beta * x["pc1"] + x["du"]
        dfx["y_bien_do"] = ys_full
        yr = np.maximum(np.exp(ys_full), EPS)
        hb = khop_bien_do(dfx, [], tr, None, None)
        hx = khop_bien_do(dfx, cols, tr, None, None)
        d = 100 * (qlike(yr, hx)[te].mean() / qlike(yr, hb)[te].mean() - 1)
        luc = bat / N_LAN
        kl = ("đối chứng âm" if beta == 0 else
              "← MDES (lực ≥ 80%)" if luc >= 0.8 else "")
        print(f"{beta:>6.2f}{d:>17.2f}%{100*luc:>20.1f}%"
              f"{float(np.median(Fs)):>13.2f}  {kl}")
        ket.append(dict(beta=beta, d_qlike_kt=float(d), luc=float(luc),
                        F_trung_vi=float(np.median(Fs))))
    print("-" * 100)

    am = ket[0]
    dat = am["luc"] <= ALPHA
    print(f"\nĐỐI CHỨNG ÂM (β = 0): lực = {100*am['luc']:.1f}% "
          f"(ngưỡng ≤ {100*ALPHA:.0f}%)   {'ĐẠT' if dat else 'HỎNG'}")

    qua = [k for k in ket if k["luc"] >= 0.8 and k["beta"] > 0]
    mdes = min(qua, key=lambda k: k["beta"]) if qua else None
    if mdes:
        print(f"\nMDES (lực 80% qua cửa Westfall–Young thật):")
        print(f"  β = {mdes['beta']:.2f}  ⟺  cải thiện QLIKE kiểm tra "
              f"{abs(mdes['d_qlike_kt']):.2f}%")
        print(f"\n  PHÁT BIỂU DÙNG ĐƯỢC CHO LUẬN VĂN:")
        print(f"  \"Phễu Pha 3B phát hiện được, với xác suất 80%, một quan hệ")
        print(f"   ngoại sinh đủ mạnh để cải thiện QLIKE khoảng "
              f"{abs(mdes['d_qlike_kt']):.1f}%.")
        print(f"   Mọi hiệu ứng mạnh hơn thế đã bị loại trừ trên dữ liệu này.\"")
    else:
        print("\n  Không mức β nào đạt lực 80% — phải mở rộng lưới.")

    json.dump(dict(nguong_F=nguong, n_null=int(len(f_null)), n_lan=N_LAN,
                   vat_mang=f"{VAT_MANG}_L{LAG_MANG}", alpha=ALPHA,
                   ket_qua=ket, doi_chung_am_dat=bool(dat), mdes=mdes),
              open(os.path.join(OUT, "pha3b_mdes.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/pha3b_mdes.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
