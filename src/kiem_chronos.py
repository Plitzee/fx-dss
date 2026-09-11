"""CHRONOS-BOLT (mo hinh nen chuoi thoi gian, Amazon) — thu THAT, khong chi trich dan.

BOI CANH. `docs/ML_DL_VONG7.md` dong 189 ghi "khong thu foundation model — khong
tai duoc trong so trong moi truong nay", roi dan Brini (arXiv 2607.05291) ho
do giup. Nguoi dung hoi thang: ly do la gi, va noi san sang bo thoi gian de
thu that. Kiem tra lai thi day KHONG PHAI gioi han moi truong — la xung dot
phien ban `transformers`/`huggingface_hub` (ham `is_offline_mode` bi doi cho).
Sua xong (`pip install -U "transformers<4.50" "huggingface_hub<0.28"`), tai
duoc trong so `amazon/chronos-bolt-small` trong 46s. File nay chay THAT tren
dung du lieu va giao thuc cua `docs/ML_DL_VONG7.md`.

GIAO THUC — GIONG HET bang 14 mo hinh da co, de so sanh cong bang:
  - Cung du lieu: `volfc2.nap_bang()`, cung luoi ngay chung 6 cap
  - Cung phan doan: `split.doan()` — huan luyen / kiem dinh / kiem tra
  - Cung chi so: QLIKE tren THANG PHUONG SAI (khong phai log-RV)
  - Cung ky luat mo doan kiem tra: chi cham SAU KHI da chon tren kiem dinh

CHRONOS LA ZERO-SHOT — KHONG "TRAIN" THEO NGHIA THONG THUONG. Day la diem
nguoi dung can hieu: mo hinh nen chuoi thoi gian da huan luyen truoc tren
hang ty diem du lieu TONG HOP (khong phai FX), va o day chi dung de SUY DIEN
truc tiep tren chuoi log-RV, KHONG khop lai tham so nao rieng cho tung cap —
dung dinh nghia "zero-shot forecasting" trong tai lieu (Ansari et al. 2024).
Day CHINH LA cach cac bai bao so sanh TSFM voi HAR dang lam (kem ca Brini
2026) — khong phai lam qua loa.

DOI TU LOG-RV SANG PHUONG SAI DUNG CACH — diem de sai nhat. HAR san xuat du
bao log-RV roi cong +0,5*var(phan du) truoc khi exp() (hieu chinh log-chuan),
neu khong QLIKE se phat oan mo hinh (da ghi trong ML_DL_VONG7.md). Chronos
KHONG can hieu chinh rieng: no da tra ve CA CHIN PHAN VI cua phan phoi du bao
(khong chi trung vi), nen du bao PHUONG SAI dung la:

    E[RV] = E[exp(log RV)] ~ trung binh cong cua exp(phan_vi_i), i=1..9

Day la xap xi Monte Carlo tren luoi 9 diem cua chinh phan phoi Chronos dua ra
— KHONG dung trung vi roi exp() suong (se lech xuong duoi vi ham exp loi), va
KHONG can uoc mot he so hieu chinh rieng nhu HAR.

Chay:  python src/kiem_chronos.py
Ghi:   output/chronos.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import volfc2 as V2                                            # noqa: E402
from split import doan                                         # noqa: E402


def qlike(proxy, h):
    """QLIKE dang BAT BIEN THANG DO — DUNG cong thuc thuc su tao ra bang 14
    mo hinh o docs/ML_DL_VONG7.md (`src/run_ml_final.py` dong 118-122), KHONG
    PHAI `metrics.qlike()` (log(h)+proxy/h — dang sach giao khoa).

    LOI DA BAT: chay lan dau dung metrics.qlike() cho QLIKE ~ -9,5, trong khi
    moc HAR v7 la +0,12 — khong the so sanh. Truy ra: hai cong thuc CHI khac
    nhau mot hang so CONG THEM log(proxy)+1 (proxy la RV THUC, khong doi theo
    mo hinh), nen thu tu XEP HANG giua cac mo hinh khong doi — nhung GIA TRI
    TUYET DOI thi lech han, va voi rv5 o thang ~1e-5 (FX ngay), hang so do
    ~-9,7, dung bang do lech quan sat duoc.

    Cong thuc dung: r = proxy/h; QLIKE = r - log(r) - 1. Luon >= 0 (bat dang
    thuc AM-GM), = 0 dung khi h = proxy (du bao hoan hao). Day la dang
    "Bregman divergence" cua QLIKE, khong phu thuoc thang do tuyet doi cua
    proxy — chinh vi vay moi la dang repo nay dung de bao cao."""
    proxy = np.asarray(proxy, float)
    h = np.asarray(h, float)
    r = proxy / np.maximum(h, EPS)
    return r - np.log(np.maximum(r, EPS)) - 1.0

MO_HINH = "amazon/chronos-bolt-small"
NGU_CANH_TOI_DA = 2048         # gioi han cua chinh mo hinh
DAM = 250                       # so phien toi thieu truoc khi du bao
CO_BATCH = 32                   # so chuoi moi lan goi mo hinh — toc do
EPS = 1e-12


def du_bao_pair(pipe, lv, dam, chi_tai=None, batch=CO_BATCH,
                 ngu_canh_toi_da=NGU_CANH_TOI_DA, thu_phan_vi=None):
    """Du bao 1-buoc cho tap phien `chi_tai` (mac dinh: MOI phien tu `dam`
    tro di), NHAN QUA (ngu canh chi den het phien truoc). Tra ve mang
    h_forecast (thang PHUONG SAI, da doi tu phan vi log-RV).

    `chi_tai`: gioi han tinh toan CHI cho kiem dinh+kiem tra thay vi ca
    doan huan luyen — lan chay dau tinh du ca doan huan luyen (khong dung
    den) lam ton ~4 lan cong suc khong can thiet, da bi giet giua chung vi
    tuong nham la treo. Sua lai de chi tinh dung phan can."""
    n = len(lv)
    ra = np.full(n, np.nan)
    ung_vien = chi_tai if chi_tai is not None else range(dam, n)
    idx = [t for t in ung_vien if np.isfinite(lv[:t]).sum() >= dam]
    for s in range(0, len(idx), batch):
        lo = idx[s:s + batch]
        ctx = []
        for t in lo:
            v = lv[:t]
            v = v[np.isfinite(v)][-ngu_canh_toi_da:]
            ctx.append(torch.tensor(v, dtype=torch.float32))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q, _ = pipe.predict_quantiles(ctx, prediction_length=1)
        # q: (batch, 1, 9) — phan vi cua log-RV. Doi ve PHUONG SAI dung cach:
        # trung binh cong cua exp(phan_vi), XAP XI E[RV] = E[exp(log RV)].
        Q = q.numpy()[:, 0, :]                          # (batch, 9)
        h = np.exp(Q).mean(1)                            # (batch,)
        for t, v in zip(lo, h):
            ra[t] = v
        if thu_phan_vi is not None:
            for t, qv in zip(lo, np.exp(Q)):             # phân vị đã đổi sang thang RV
                thu_phan_vi[t] = qv
    return ra


def _tu_kiem(pipe):
    """Doi chieu cong thuc doi phan vi: tren du lieu MO PHONG voi phan phoi
    BIET TRUOC (log-RV ~ N(0, 0,25)), trung binh cong cua exp(phan vi) phai
    GAN E[X] ly thuyet cua log-normal (= exp(mu + sigma^2/2)) hon la
    exp(trung vi) — chung minh buoc doi don vi khong lam meo."""
    rng = np.random.default_rng(0)
    mu, sigma = 0.0, 0.5
    lv = rng.normal(mu, sigma, 900)
    ctx = [torch.tensor(lv[:800], dtype=torch.float32)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q, _ = pipe.predict_quantiles(ctx, prediction_length=1)
    Q = q.numpy()[0, 0, :]
    h_tb_mu = float(np.exp(Q).mean())            # cach dung: TB cua exp(phan vi)
    h_exp_trungvi = float(np.exp(Q[4]))            # cach sai: exp(trung vi)
    ly_thuyet = float(np.exp(mu + sigma ** 2 / 2))  # E[X] that cua log-normal
    return h_tb_mu, h_exp_trungvi, ly_thuyet


def main():
    t0 = time.time()
    print("=" * 108)
    print(f"CHRONOS-BOLT ({MO_HINH}) — mô hình nền chuỗi thời gian, đo THẬT")
    print("cùng giao thức QLIKE với bảng 14 mô hình ở docs/ML_DL_VONG7.md")
    print("=" * 108)

    print("Đang tải trọng số…", flush=True)
    from chronos import BaseChronosPipeline
    tt = time.time()
    pipe = BaseChronosPipeline.from_pretrained(MO_HINH, device_map="cpu")
    print(f"  tải xong sau {time.time()-tt:.0f}s — ZERO-SHOT, không khớp lại "
          f"tham số riêng cho từng cặp")

    print("\ntự kiểm cách đổi phân vị log-RV sang phương sai…", flush=True)
    h_tb, h_ev, ly_thuyet = _tu_kiem(pipe)
    print(f"  TB(exp(phân vị))      = {h_tb:.4f}  ← cách dùng trong file này")
    print(f"  exp(trung vị)          = {h_ev:.4f}  ← cách SAI, lệch xuống")
    print(f"  E[X] lý thuyết (log-normal) = {ly_thuyet:.4f}")
    lech_tb = abs(h_tb - ly_thuyet) / ly_thuyet
    lech_ev = abs(h_ev - ly_thuyet) / ly_thuyet
    print(f"  lệch: TB(exp) {lech_tb:.1%}  vs  exp(trung vị) {lech_ev:.1%}"
          f"  {'ĐẠT' if lech_tb < lech_ev else '← BẤT THƯỜNG'}")
    assert lech_tb < lech_ev, "cách đổi đơn vị phải gần E[X] hơn cách dùng trung vị"

    bang, chung = V2.nap_bang()
    ket = {}
    RAW_PAIR, RAW_DATE, RAW_H, RAW_RV, RAW_Q = [], [], [], [], []
    for p in bang:
        d = bang[p]
        g = doan(d.Date.values)
        rv = np.maximum(d.rv5.values, EPS)
        lv = np.log(rv)
        chi_tai = np.flatnonzero(g >= 1)      # CHI kiem dinh+kiem tra — khong tinh thua
        print(f"\n[{p}] đang dự báo {len(chi_tai):,} phiên (kiểm định+kiểm tra), "
              f"context tối đa {NGU_CANH_TOI_DA}…", flush=True)
        tp = time.time()
        thu_qv = {}
        h = du_bao_pair(pipe, lv, DAM, chi_tai=chi_tai, thu_phan_vi=thu_qv)
        print(f"  xong {time.time()-tp:.0f}s", flush=True)

        dat = d.Date.values
        for t in chi_tai:
            if not np.isfinite(h[t]):
                continue
            RAW_PAIR.append(p); RAW_DATE.append(dat[t])
            RAW_H.append(float(h[t])); RAW_RV.append(float(rv[t]))
            RAW_Q.append(thu_qv.get(t, np.full(9, np.nan)))

        for ten_doan, gid in (("kiem_dinh", 1), ("kiem_tra", 2)):
            m = (g == gid) & np.isfinite(h)
            if m.sum() < 50:
                continue
            ql = qlike(rv[m], h[m])
            ket.setdefault(ten_doan, {})[p] = dict(n=int(m.sum()), qlike=float(np.mean(ql)))
            print(f"  {ten_doan:<12} n={int(m.sum()):>4}  QLIKE {np.mean(ql):.4f}", flush=True)

    print("\n" + "=" * 108)
    print("TỔNG KẾT — gộp 6 cặp, đối chiếu bảng 14 mô hình đã có")
    print(f"  {'đoạn':<14}{'QLIKE Chronos-bolt':>20}{'QLIKE HAR v7 (đã ghi)':>24}{'chênh':>10}")
    moc = {"kiem_dinh": 0.1162, "kiem_tra": 0.1585}   # ML_DL_VONG7.md, hàng #5
    for ten_doan in ("kiem_dinh", "kiem_tra"):
        if ten_doan not in ket:
            continue
        vals = [v["qlike"] for v in ket[ten_doan].values()]
        ns = [v["n"] for v in ket[ten_doan].values()]
        gop = float(np.average(vals, weights=ns))
        chenh = (gop - moc[ten_doan]) / moc[ten_doan]
        print(f"  {ten_doan:<14}{gop:>20.4f}{moc[ten_doan]:>24.4f}{chenh:>+10.1%}")
        ket[f"gop_{ten_doan}"] = gop

    os.makedirs(OUT, exist_ok=True)
    ket["tu_kiem"] = dict(tb_exp_phanvi=h_tb, exp_trungvi=h_ev, ly_thuyet=ly_thuyet)
    ket["mo_hinh"] = MO_HINH
    ket["moc_har_v7"] = moc
    json.dump(ket, open(os.path.join(OUT, "chronos.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, "chronos_raw.npz"),
                        pair=np.array(RAW_PAIR), date=np.array(RAW_DATE),
                        h=np.array(RAW_H), rv=np.array(RAW_RV),
                        q=np.array(RAW_Q))
    print("→ output/chronos_raw.npz (dự báo + 9 phân vị thô, cho bảng MSE/MAE/CRPS)")
    print(f"\n→ output/chronos.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
