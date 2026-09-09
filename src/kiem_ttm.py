"""TTM (Tiny Time Mixers, IBM/Granite) — mo hinh nen chuoi thoi gian DUY NHAT
trong tai lieu thang duoc Log-HAR co do (Brini, arXiv 2607.05291). Thu that.

BOI CANH. `docs/ML_DL_VONG7.md` da ghi Chronos-bolt-small THUA HAR vong 7
(+17-19% QLIKE) — khop voi Brini: "chi TTM hon Log-HAR 1,3-1,8%, cac mo hinh
khac khong thang". File nay thu chinh cai model do, tren cung du lieu/giao
thuc.

KHAC VOI CHRONOS-BOLT O HAI DIEM KY THUAT QUAN TRONG:

  1. NGU CANH CO DINH 512, KHONG PHAI CO GIAN. TTM khong nhan chuoi do dai
     tuy y nhu Chronos — `context_length=512` la HANG SO cua kien truc. Moi
     du bao dung DUNG 512 diem gan nhat.
  2. DIEM DU BAO, KHONG PHAI PHAN VI. Chronos-bolt tra ve 9 phan vi nen doi
     sang phuong sai bang trung binh cong cua exp(phan vi) — KHONG can uoc
     rieng he so hieu chinh. TTM (ban goc, khong bat dau ra xac suat) chi tra
     ve MOT diem du bao log-RV, dung y het HAR truoc khi doi don vi. Nen phai
     ap DUNG HEU CHINH LOG-CHUAN ma docs/ML_DL_VONG7.md da dat ra cho HAR/ML:

         h = exp(du_bao_log_rv + 0,5 * var(phan du huan luyen))

     He so hieu chinh uoc tren DOAN HUAN LUYEN (mau cach quang de nhanh, xem
     UOC_MOI), KHONG dung doan kiem dinh/kiem tra — dung nguyen tac "khong ro
     ri" ma moi ho khac trong repo nay tuan theo.

GIAO THUC — GIONG HET Chronos va bang 14 mo hinh: cung du lieu
(`volfc2.nap_bang()`), cung phan doan (`split.doan()`), cung cong thuc QLIKE
bat bien thang do `r - log(r) - 1` (r = proxy/h) — DA HOC TU LOI O
`kiem_chronos.py` (lan dau dung nham metrics.qlike() sach giao khoa, cho QLIKE
am sau vo nghia vi phu thuoc thang do tuyet doi cua RV).

CANH BAO MOI TRUONG. Cai `transformers>=4.57.6` de chay TTM da NANG CAP tu
ban `<4.50` dung cho Chronos — hai thu vien nay xung dot yeu cau. Neu chay lai
`kiem_chronos.py` sau file nay, phai HA cap lai
(`pip install "transformers<4.50" "huggingface_hub<0.28"`) truoc.

Chay:  python src/kiem_ttm.py
Ghi:   output/ttm.json
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

MO_HINH = "ibm-granite/granite-timeseries-ttm-r2"
CO_BATCH = 64
DAM = 250            # so phien toi thieu truoc khi cho du bao (giong Chronos)
UOC_MOI = 5           # mau cach quang tren huan luyen de uoc he so hieu chinh
EPS = 1e-12


def qlike(proxy, h):
    """QLIKE bat bien thang do — DUNG cong thuc thuc su tao bang 14 mo hinh
    (`src/run_ml_final.py`), khong phai `metrics.qlike()` sach giao khoa. Xem
    giai thich day du trong docstring cua `kiem_chronos.qlike()`."""
    proxy = np.asarray(proxy, float)
    h = np.asarray(h, float)
    r = proxy / np.maximum(h, EPS)
    return r - np.log(np.maximum(r, EPS)) - 1.0


def du_bao_diem(model, lv, idx, ctx_len, batch=CO_BATCH):
    """Du bao 1-buoc (log-RV) cho tap chi so `idx`, NHAN QUA: ngu canh la
    DUNG `ctx_len` diem lien truoc t (khong bao gom t). Cac diem thieu du
    lich su bi bo qua. Tra ve dict {t: du_bao_log_rv}."""
    ra = {}
    du = [t for t in idx if t >= ctx_len and np.isfinite(lv[t - ctx_len:t]).all()]
    for s in range(0, len(du), batch):
        lo = du[s:s + batch]
        X = np.stack([lv[t - ctx_len:t] for t in lo]).astype(np.float32)
        x = torch.tensor(X).unsqueeze(-1)                # (batch, ctx_len, 1)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model(past_values=x)
        buoc1 = out.prediction_outputs[:, 0, 0].numpy()   # h=1: buoc dau
        for t, v in zip(lo, buoc1):
            ra[t] = float(v)
    return ra


def hoi_quy_tai_hieu_chuan(du_bao_tr, log_rv_tr):
    """Mincer-Zarnowitz — DUNG KY THUAT Brini (arXiv 2607.05291) da dung de
    "cuu" TSFM: hoi quy log_rv_that ~ a + b*du_bao_diem TREN HUAN LUYEN, roi
    dung (a, b) do de tai hieu chuan du bao truoc khi doi sang phuong sai.

    Brini: "much of the short-horizon advantage reflects better-scaled
    forecasts rather than better prediction of volatility dynamics" — tuc
    TSFM thang duoc KHONG PHAI vi hieu dong luc bien dong tot hon, ma vi sau
    khi CAN CHINH LAI THANG DO (a, b) thi no moi khop. Phep cong hang so
    +0,5*var don gian (dung truoc do cho Chronos) la mot dang tai hieu chuan
    YEU HON — no chi dich CHUNG, khong xoay (b != 1) hay doi diem goc (a != 0).
    Day la phep thu CONG BANG NHAT voi chinh phuong phap da cho TTM thang HAR
    trong tai lieu.

    Tra ve (a, b, he_so_hieu_chinh_du) — he_so_hieu_chinh_du la +0,5*var cua
    PHAN DU SAU KHI DA hoi quy (buoc thu hai, van can vi hoi quy tuyen tinh
    khong tu dong sua sai so log-chuan)."""
    X = np.column_stack([np.ones(len(du_bao_tr)), du_bao_tr])
    beta, *_ = np.linalg.lstsq(X, log_rv_tr, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    resid = log_rv_tr - (a + b * du_bao_tr)
    hc = 0.5 * float(np.var(resid))
    return a, b, hc


def _tu_kiem(model, ctx_len):
    """Doi chieu he so hieu chinh log-chuan: mo phong log-RV ~ N(mu, sigma^2)
    KHONG PHU THUOC context (nhieu trang), TTM se du bao gan trung binh cua
    context (xap xi mu). Sau hieu chinh +0,5*sigma^2, exp() phai gan E[X] ly
    thuyet CUA log-normal hon la khong hieu chinh."""
    rng = np.random.default_rng(0)
    mu, sigma = -10.0, 0.5           # THANG THAT cua RV FX (~exp(-10) ~ 4,5e-5)
    lv = rng.normal(mu, sigma, ctx_len + 300)
    idx = list(range(ctx_len, len(lv)))
    du = du_bao_diem(model, lv, idx, ctx_len)
    resid = np.array([lv[t] - du[t] for t in idx])
    hc = 0.5 * float(np.var(resid))
    h_khong_hc = float(np.mean(np.exp([du[t] for t in idx])))
    h_co_hc = float(np.mean(np.exp(np.array([du[t] for t in idx]) + hc)))
    ly_thuyet = float(np.exp(mu + sigma ** 2 / 2))
    return h_khong_hc, h_co_hc, ly_thuyet, hc


def main():
    t0 = time.time()
    print("=" * 108)
    print(f"TTM ({MO_HINH}) — mô hình nền chuỗi thời gian, DUY NHẤT thắng Log-HAR trong Brini 2026")
    print("cùng giao thức QLIKE với Chronos-bolt và bảng 14 mô hình ở docs/ML_DL_VONG7.md")
    print("=" * 108)

    print("Đang tải trọng số…", flush=True)
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    tt = time.time()
    model = TinyTimeMixerForPrediction.from_pretrained(MO_HINH, num_input_channels=1)
    model.eval()
    ctx_len = model.config.context_length
    print(f"  tải xong sau {time.time()-tt:.0f}s — context_length={ctx_len}, "
          f"prediction_length={model.config.prediction_length}")
    print("  ZERO-SHOT, không khớp lại tham số riêng cho từng cặp")

    print("\ntự kiểm hệ số hiệu chỉnh log-chuẩn…", flush=True)
    h0, h1, ly_thuyet, hc = _tu_kiem(model, ctx_len)
    print(f"  KHÔNG hiệu chỉnh   = {h0:.3e}")
    print(f"  CÓ hiệu chỉnh      = {h1:.3e}  (hệ số +{hc:.4f})")
    print(f"  E[X] lý thuyết     = {ly_thuyet:.3e}")
    lech0 = abs(h0 - ly_thuyet) / ly_thuyet
    lech1 = abs(h1 - ly_thuyet) / ly_thuyet
    print(f"  lệch: không hc {lech0:.1%}  vs  có hc {lech1:.1%}"
          f"  {'ĐẠT' if lech1 < lech0 else '← BẤT THƯỜNG'}")
    assert lech1 < lech0, "hiệu chỉnh log-chuẩn phải kéo gần E[X] hơn"

    bang, chung = V2.nap_bang()
    ket = {}
    for p in bang:
        d = bang[p]
        g = doan(d.Date.values)
        rv = np.maximum(d.rv5.values, EPS)
        lv = np.log(rv)

        # HE SO HIEU CHINH — uoc TREN HUAN LUYEN, mau cach quang de nhanh
        idx_tr = np.flatnonzero(g == 0)
        idx_tr = idx_tr[idx_tr >= ctx_len][::UOC_MOI]
        print(f"\n[{p}] ước hệ số hiệu chỉnh trên {len(idx_tr):,} điểm mẫu huấn luyện…",
              flush=True)
        du_tr = du_bao_diem(model, lv, idx_tr, ctx_len)
        resid = np.array([lv[t] - du_tr[t] for t in du_tr])
        hc_p = 0.5 * float(np.var(resid))
        print(f"  hệ số hiệu chỉnh = +{hc_p:.4f}  (từ {len(resid):,} điểm)")

        chi_tai = np.flatnonzero(g >= 1)
        print(f"  đang dự báo {len(chi_tai):,} phiên (kiểm định+kiểm tra)…", flush=True)
        tp = time.time()
        du = du_bao_diem(model, lv, chi_tai, ctx_len)
        print(f"  xong {time.time()-tp:.0f}s", flush=True)

        for ten_doan, gid in (("kiem_dinh", 1), ("kiem_tra", 2)):
            m = [t for t in np.flatnonzero(g == gid) if t in du]
            if len(m) < 50:
                continue
            h = np.array([np.exp(du[t] + hc_p) for t in m])
            ql = qlike(rv[m], h)
            ket.setdefault(ten_doan, {})[p] = dict(n=len(m), qlike=float(np.mean(ql)))
            print(f"  {ten_doan:<12} n={len(m):>4}  QLIKE {np.mean(ql):.4f}", flush=True)

    print("\n" + "=" * 108)
    print("TỔNG KẾT — gộp 6 cặp, đối chiếu Chronos-bolt và HAR v7 đã đo")
    print(f"  {'đoạn':<14}{'QLIKE TTM':>14}{'QLIKE Chronos':>16}{'QLIKE HAR v7':>15}"
          f"{'TTM so HAR':>13}")
    moc_har = {"kiem_dinh": 0.1162, "kiem_tra": 0.1585}
    moc_chronos = {"kiem_dinh": 0.1383, "kiem_tra": 0.1851}
    for ten_doan in ("kiem_dinh", "kiem_tra"):
        if ten_doan not in ket:
            continue
        vals = [v["qlike"] for v in ket[ten_doan].values()]
        ns = [v["n"] for v in ket[ten_doan].values()]
        gop = float(np.average(vals, weights=ns))
        chenh = (gop - moc_har[ten_doan]) / moc_har[ten_doan]
        print(f"  {ten_doan:<14}{gop:>14.4f}{moc_chronos[ten_doan]:>16.4f}"
              f"{moc_har[ten_doan]:>15.4f}{chenh:>+13.1%}")
        ket[f"gop_{ten_doan}"] = gop

    os.makedirs(OUT, exist_ok=True)
    ket["tu_kiem"] = dict(khong_hc=h0, co_hc=h1, ly_thuyet=ly_thuyet, he_so=hc)
    ket["mo_hinh"] = MO_HINH
    ket["moc_har_v7"] = moc_har
    ket["moc_chronos_bolt"] = moc_chronos
    json.dump(ket, open(os.path.join(OUT, "ttm.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/ttm.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
