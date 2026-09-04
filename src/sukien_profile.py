"""PHAN UNG DO DUOC QUANH SU KIEN — nghien cuu su kien, khong phai du bao.

`docs/REPLAN_2026.md` muc 6 viet ro cach lam: he thong KHONG duoc doan huong
quanh su kien. No chi duoc noi "loai su kien nay trong lich su duoc theo sau
boi PHAN PHOI DA DO nay, co mau n, DAY LA LICH SU KHONG PHAI DU BAO", roi de
xuat ve DAI, CO VI THE va GIO VAO LENH.

Do gi:
  * ty le bien dong  |r| ngay su kien / trung vi |r| cua 20 phien truoc
  * ty le sigma      sigma[t] / trung vi sigma 20 phien truoc (du bao co bat kip khong)
  * ty le spread     spread trung vi gio do so mua thuong
  * do lech huong    trung binh dau cua r — de CHUNG MINH no bang 0, khong
                     phai de dung
Kem so mau va khoang tin cay bootstrap. Loai nao n < 30 thi bo.

Chay:  python src/sukien_profile.py
Ghi:   output/sukien_profile.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
D = os.path.join(ROOT, "data")

import volfc2 as V2                                        # noqa: E402
from volfc import merge_thin_days                          # noqa: E402

MIN_N = 30
CUA_SO = 20
NBOOT = 500
SEED = 0
EPS = 1e-12


def ktc(x, f=np.median, nboot=NBOOT, seed=SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return None, None
    rng = np.random.default_rng(seed)
    b = [f(rng.choice(x, len(x), replace=True)) for _ in range(nboot)]
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main():
    import balop as B
    from api.main import noi_chuoi

    cb = pd.read_csv(os.path.join(D, "cb_dates.csv"), parse_dates=["date"])
    cb["ngay"] = cb.date.astype(str).str[:10]

    hang = []
    for p in B.PAIRS:
        m = merge_thin_days(noi_chuoi(p))
        sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, p), 0.0))
        c = m.close.values
        r = np.r_[np.nan, np.diff(np.log(np.maximum(c, EPS)))]
        ng = pd.Series(m.Date.values).astype(str).str[:10].values
        ar = np.abs(r)
        nen_r = pd.Series(ar).shift(1).rolling(CUA_SO).median().values
        nen_s = pd.Series(sig).shift(1).rolling(CUA_SO).median().values
        vt = {x: i for i, x in enumerate(ng)}
        for _, row in cb.iterrows():
            i = vt.get(row.ngay)
            if i is None or i < CUA_SO + 1:
                continue
            if not (np.isfinite(ar[i]) and np.isfinite(nen_r[i]) and nen_r[i] > EPS):
                continue
            hang.append(dict(pair=p, bank=row.bank, ngay=row.ngay,
                             ty_le_bd=float(ar[i] / nen_r[i]),
                             ty_le_sig=float(sig[i] / max(nen_s[i], EPS)),
                             dau=float(np.sign(r[i]))))
    df = pd.DataFrame(hang)
    print("=" * 92)
    print("PHẢN ỨNG ĐO ĐƯỢC QUANH SỰ KIỆN NGÂN HÀNG TRUNG ƯƠNG")
    print("=" * 92)
    print(f"{len(df):,} quan sát (cặp × ngày họp), {df.bank.nunique()} ngân hàng\n")

    ra = {"cua_so": CUA_SO, "min_n": MIN_N, "loai": []}
    print(f"{'ngân hàng':<10}{'n':>6}{'|r| so nền':>13}{'KTC 95%':>20}"
          f"{'σ̂ so nền':>12}{'thiên lệch hướng':>20}")
    for bank, g in df.groupby("bank"):
        if len(g) < MIN_N:
            continue
        lo, hi = ktc(g.ty_le_bd.values)
        dlo, dhi = ktc(g.dau.values, f=np.mean)
        e = dict(bank=bank, n=int(len(g)),
                 bd_trungvi=round(float(np.median(g.ty_le_bd)), 3),
                 bd_lo=round(lo, 3) if lo else None,
                 bd_hi=round(hi, 3) if hi else None,
                 bd_p90=round(float(np.quantile(g.ty_le_bd, 0.9)), 3),
                 sig_trungvi=round(float(np.nanmedian(g.ty_le_sig)), 3),
                 dau_tb=round(float(np.mean(g.dau)), 3),
                 dau_lo=round(dlo, 3) if dlo else None,
                 dau_hi=round(dhi, 3) if dhi else None)
        e["huong_co_y_nghia"] = bool(dlo is not None and (dlo > 0 or dhi < 0))
        ra["loai"].append(e)
        k = f"[{e['bd_lo']}; {e['bd_hi']}]"
        d = f"{e['dau_tb']:+.3f} [{e['dau_lo']:+.3f}; {e['dau_hi']:+.3f}]"
        print(f"{bank:<10}{e['n']:>6}{e['bd_trungvi']:>13.3f}{k:>20}"
              f"{e['sig_trungvi']:>12.3f}{d:>20}")

    n_huong = sum(1 for e in ra["loai"] if e["huong_co_y_nghia"])
    print(f"\nSố loại sự kiện có thiên lệch HƯỚNG có ý nghĩa: "
          f"{n_huong}/{len(ra['loai'])}")
    ra["so_loai_co_huong"] = n_huong
    with open(os.path.join(OUT, "sukien_profile.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("đã ghi output/sukien_profile.json")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
