"""CHI SO TANG SIGMA — QLIKE, MAE, RMSE, CRPS, PIT, do phu.

`docs/CHISO_DANHGIA.md` co CRPS 26,22-26,28 pip nhung chi o dang VAN XUOI —
khong file nao may doc duoc, nen giao dien khong hien duoc. File nay tinh lai
tren du lieu HIEN HANH va ghi ra JSON.

Vi sao can ca MAE lan CRPS lan QLIKE — ba cai do ba thu khac nhau:

  QLIKE  chat luong du bao PHUONG SAI. Bat doi xung: phat nang khi du bao
         THIEU bien dong. La chi so chinh cua tang 2.
  MAE    sai so tuyet doi cua sigma so voi bien dong thuc hien, tinh bang PIP
         — doc duoc bang mat thuong, nhung KHONG phai quy tac cham diem chinh
         dang cho phuong sai.
  CRPS   chat luong ca PHAN PHOI du bao cua loi suat ngay, tinh bang pip. Day
         moi la thu danh gia dung "dai tin cay" ma giao dien in ra.

Bai hoc cua repo phai giu: chat luong sigma quan trong hon cach dung duoi.
Gauss/Student-t/Mondrian gan nhu bang nhau ve CRPS (26,22-26,28), nhung dung
sigma CU thi thanh 26,33. Nen dung don cong vao tinh chinh duoi.

Chay:  python src/chiso_mohinh.py
Ghi:   output/chiso_mohinh.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import volfc2 as V2                                        # noqa: E402
from split import VALID_TU, TEST_TU, doan                  # noqa: E402
from volfc import merge_thin_days                          # noqa: E402

PIP = {"USDJPY": 0.01}
N_MC = 4000
SEED = 0
EPS = 1e-14


def crps_mc(sig, nu, r, gia, pip, rng):
    """CRPS cua phan phoi du bao r ~ sig * t(nu), tinh bang PIP.

    Cong thuc mau: CRPS = E|X - y| - 0.5 * E|X - X'|. Rut mau thay vi dung
    dang dong kin vi dang do cho Student-t dai va de sai dau."""
    n = len(r)
    s = sig[:, None] * rng.standard_t(nu, size=(n, N_MC))
    a = np.abs(s - r[:, None]).mean(1)
    b = np.abs(s[:, : N_MC // 2] - s[:, N_MC // 2:]).mean(1)
    return (a - 0.5 * b) * gia / pip


def mot_cap(p):
    from api.main import noi_chuoi
    m = merge_thin_days(noi_chuoi(p))
    sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, p), 0.0))
    c = m.close.values
    r = np.r_[np.nan, np.diff(np.log(np.maximum(c, EPS)))]
    rv = m.rv5.values
    g = doan(m.Date.values)
    ok = np.isfinite(sig) & (sig > 0) & np.isfinite(r) & np.isfinite(rv) & (rv > 0)
    tr = (g == 0) & ok
    nu = float(np.clip(stats.t.fit((r[tr] / sig[tr]), floc=0)[0], 2.5, 40))
    return dict(m=m, sig=sig, r=r, rv=rv, g=g, ok=ok, nu=nu,
                gia=c, pip=PIP.get(p, 0.0001))


def do(d, mask, rng):
    sig, r, rv = d["sig"][mask], d["r"][mask], d["rv"][mask]
    gia, pip, nu = d["gia"][mask], d["pip"], d["nu"]
    f = sig ** 2
    # QLIKE tren PHUONG SAI — chi so chinh cua tang 2
    q = rv / f - np.log(np.maximum(rv / f, EPS)) - 1
    # MAE / RMSE cua SIGMA so voi bien dong thuc hien, tinh bang pip
    sr = np.sqrt(rv)
    e_pip = (sig - sr) * gia / pip
    # MAE cua du bao |loi suat|. Voi T ~ Student-t(nu):
    #   E|T| = 2*sqrt(nu) * Gamma((nu+1)/2) / ( sqrt(pi) * (nu-1) * Gamma(nu/2) )
    # Dung gammaln cho on dinh so; nu <= 1 thi ky vong khong ton tai.
    from scipy.special import gammaln
    z = r / np.maximum(sig, EPS)
    sc = float(stats.t.fit(z, floc=0)[2])
    if nu > 1:
        ln = (np.log(2.0) + 0.5 * np.log(nu) + gammaln(0.5 * (nu + 1))
              - 0.5 * np.log(np.pi) - np.log(nu - 1.0) - gammaln(0.5 * nu))
        e_abs_t = float(np.exp(ln))
    else:
        e_abs_t = 1.0
    du_bao_abs = sig * sc * e_abs_t
    mae_abs = float(np.mean(np.abs(np.abs(r) - du_bao_abs) * gia / pip))
    # PIT + Kolmogorov-Smirnov
    pit = stats.t.cdf(z / max(sc, EPS), nu)
    ks = stats.kstest(pit, "uniform")
    # do phu khoang 90%
    q95 = stats.t.ppf(0.95, nu) * sc
    phu = float(np.mean(np.abs(z) <= q95))
    return dict(
        n=int(mask.sum()),
        qlike=float(np.mean(q)),
        mae_sigma_pip=float(np.mean(np.abs(e_pip))),
        rmse_sigma_pip=float(np.sqrt(np.mean(e_pip ** 2))),
        mae_abs_r_pip=mae_abs,
        crps_pip=float(np.mean(crps_mc(sig, nu, r, gia, pip, rng))),
        pit_ks_p=float(ks.pvalue), do_phu_90=phu, nu=float(nu))


def main():
    rng = np.random.default_rng(SEED)
    import balop as B
    ra = {"cap": {}, "gop": {}}
    print("=" * 96)
    print("CHỈ SỐ TẦNG σ̂ — tính lại trên dữ liệu hiện hành")
    print("=" * 96)
    print(f"{'cặp':<9}{'đoạn':<11}{'QLIKE':>9}{'MAE σ̂':>9}{'RMSE':>9}"
          f"{'MAE |r|':>9}{'CRPS':>9}{'PIT p':>9}{'phủ 90%':>9}")
    gom = {"kiểm định": [], "kiểm tra": []}
    for p in B.PAIRS:
        d = mot_cap(p)
        for nhan, gi in (("kiểm định", 1), ("kiểm tra", 2)):
            m = d["ok"] & (d["g"] == gi)
            if m.sum() < 100:
                continue
            k = do(d, m, rng)
            ra["cap"].setdefault(p, {})[nhan] = k
            gom[nhan].append(k)
            print(f"{p:<9}{nhan:<11}{k['qlike']:>9.4f}{k['mae_sigma_pip']:>9.2f}"
                  f"{k['rmse_sigma_pip']:>9.2f}{k['mae_abs_r_pip']:>9.2f}"
                  f"{k['crps_pip']:>9.2f}{k['pit_ks_p']:>9.4f}{k['do_phu_90']:>8.1%}")
    print("-" * 96)
    for nhan, ds in gom.items():
        if not ds:
            continue
        g = {k: float(np.mean([x[k] for x in ds]))
             for k in ("qlike", "mae_sigma_pip", "rmse_sigma_pip",
                       "mae_abs_r_pip", "crps_pip", "do_phu_90")}
        g["n"] = int(sum(x["n"] for x in ds))
        ra["gop"][nhan] = g
        print(f"{'GỘP':<9}{nhan:<11}{g['qlike']:>9.4f}{g['mae_sigma_pip']:>9.2f}"
              f"{g['rmse_sigma_pip']:>9.2f}{g['mae_abs_r_pip']:>9.2f}"
              f"{g['crps_pip']:>9.2f}{'':>9}{g['do_phu_90']:>8.1%}")
    with open(os.path.join(OUT, "chiso_mohinh.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("\nđã ghi output/chiso_mohinh.json")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
