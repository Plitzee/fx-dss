"""TTM VOI TAI HIEU CHUAN MINCER-ZARNOWITZ — phep thu cong bang nhat voi Brini.

`kiem_ttm.py` dung hieu chinh log-chuan DON GIAN (+0,5*var, dich CHUNG) va cho
TTM THUA HAR +17-19% — khong khop voi Brini (arXiv 2607.05291): "chi TTM hon
Log-HAR". Brini tu noi ro ly do: phan lon loi the cua TSFM o chan troi ngan
la do TAI HIEU CHUAN (better-scaled), khong phai hieu dong luc tot hon.

File nay dung DUNG ky thuat do — hoi quy Mincer-Zarnowitz
`log_rv_that ~ a + b*du_bao_diem`, khop tren HUAN LUYEN — thay vi chi dich
mot hang so. Day la phep thu CONG BANG NHAT: neu TTM van thua sau khi da tai
hieu chuan dung cach Brini lam, thi ket luan "thua HAR" dung vung; neu no
thang, thi chinh xac dieu Brini mo ta da xay ra o day.

Chay:  python src/kiem_ttm_mz.py
Ghi:   output/ttm_mz.json
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
from kiem_ttm import du_bao_diem, hoi_quy_tai_hieu_chuan, qlike, EPS  # noqa: E402

MO_HINH = "ibm-granite/granite-timeseries-ttm-r2"


def main():
    t0 = time.time()
    print("=" * 108)
    print("TTM + TÁI HIỆU CHUẨN MINCER-ZARNOWITZ (đúng kỹ thuật Brini 2026)")
    print("=" * 108)

    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    model = TinyTimeMixerForPrediction.from_pretrained(MO_HINH, num_input_channels=1)
    model.eval()
    ctx_len = model.config.context_length
    print(f"model tải xong, context_length={ctx_len}\n")

    bang, chung = V2.nap_bang()
    ket = {}
    for p in bang:
        d = bang[p]
        g = doan(d.Date.values)
        rv = np.maximum(d.rv5.values, EPS)
        lv = np.log(rv)

        idx_tr = np.flatnonzero(g == 0)
        idx_tr = idx_tr[idx_tr >= ctx_len]
        du_tr = du_bao_diem(model, lv, idx_tr, ctx_len)
        t_tr = sorted(du_tr)
        x_tr = np.array([du_tr[t] for t in t_tr])
        y_tr = np.array([lv[t] for t in t_tr])
        a, b, hc = hoi_quy_tai_hieu_chuan(x_tr, y_tr)
        print(f"[{p}] hồi quy MZ: log_rv = {a:+.3f} + {b:.3f}·dự_báo  "
              f"(n={len(t_tr):,}, hệ số dư +{hc:.4f})")

        chi_tai = np.flatnonzero(g >= 1)
        du = du_bao_diem(model, lv, chi_tai, ctx_len)

        for ten_doan, gid in (("kiem_dinh", 1), ("kiem_tra", 2)):
            m = [t for t in np.flatnonzero(g == gid) if t in du]
            if len(m) < 50:
                continue
            h = np.array([np.exp(a + b * du[t] + hc) for t in m])
            ql = qlike(rv[m], h)
            ket.setdefault(ten_doan, {})[p] = dict(n=len(m), qlike=float(np.mean(ql)))
            print(f"  {ten_doan:<12} n={len(m):>4}  QLIKE {np.mean(ql):.4f}")

    print("\n" + "=" * 108)
    print("TỔNG KẾT — TTM+MZ so với TTM đơn giản, Chronos, HAR v7")
    moc = dict(har=dict(kiem_dinh=0.1162, kiem_tra=0.1585),
              chronos=dict(kiem_dinh=0.1383, kiem_tra=0.1851),
              ttm_don_gian=dict(kiem_dinh=0.1377, kiem_tra=0.1855))
    print(f"  {'đoạn':<12}{'TTM+MZ':>10}{'TTM đơn giản':>14}{'Chronos':>10}"
          f"{'HAR v7':>9}{'TTM+MZ so HAR':>16}")
    for ten_doan in ("kiem_dinh", "kiem_tra"):
        if ten_doan not in ket:
            continue
        vals = [v["qlike"] for v in ket[ten_doan].values()]
        ns = [v["n"] for v in ket[ten_doan].values()]
        gop = float(np.average(vals, weights=ns))
        chenh = (gop - moc["har"][ten_doan]) / moc["har"][ten_doan]
        print(f"  {ten_doan:<12}{gop:>10.4f}{moc['ttm_don_gian'][ten_doan]:>14.4f}"
              f"{moc['chronos'][ten_doan]:>10.4f}{moc['har'][ten_doan]:>9.4f}"
              f"{chenh:>+16.1%}")
        ket[f"gop_{ten_doan}"] = gop

    os.makedirs(OUT, exist_ok=True)
    ket["moc"] = moc
    json.dump(ket, open(os.path.join(OUT, "ttm_mz.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/ttm_mz.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
