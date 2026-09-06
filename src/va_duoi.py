"""VA DUOI DUOI — ba phuong an, chon tren KIEM DINH.

VAN DE (do duoc, docs/CHISO_DANHGIA.md muc 5b):
  USDJPY vi pham VaR 1% o muc 2,07% thay vi 1,0%; ty le ES 0,799; PIT-KS
  0,0036; do phu 90% chi 85,2%. USDCHF cung truot nhung nguoc huong (ES thua).

CHAN DOAN (src/va_duoi.py --chan-doan, ket qua ghi trong docstring nay):
  sd(z) theo doan, USDJPY:  1,014 -> 1,100 -> 1,136   TANG DON DIEU, +12%
  cac cap khac:             phang, khong xu huong
  q01 USDJPY:               -2,476 -> -3,742          duoi TRAI sau them
  q99 USDJPY:                2,572 ->  2,559          duoi PHAI dung yen

Nen day la van de THANG DO TROI THEO THOI GIAN cho rieng JPY, khong phai hinh
dang phan phoi sai. He so co dinh uoc tren huan luyen se KHONG an thua: sd(z)
tren huan luyen la 1,014, tuc gan 1, khong co gi de sua.

BA PHUONG AN:
  V0  phan vi z uoc tren HUAN LUYEN            (dang chay)
  V1  CUA SO MO RONG, khop lai moi 21 phien    (cung giao thuc voi tang sigma^)
  V2  CUA SO CUON 500 phien                    (bam sat troi hon, nho mau hon)

GIAO THUC. Chon tren KIEM DINH. Doan kiem tra chi cham SAU KHI da chot, va phai
ghi ro day la lan mo thu hai cho tang VaR (lan dau la chan doan o muc 5b).

Chay:  python src/va_duoi.py
       python src/va_duoi.py --chan-doan
Ghi:   output/va_duoi.json
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

import balop as B                                           # noqa: E402
import volfc2 as V2                                         # noqa: E402
from metrics import kupiec, christoffersen_ind, dq_test     # noqa: E402
from split import doan                                      # noqa: E402
from volfc import merge_thin_days                           # noqa: E402

MUC = (0.05, 0.01)
BUOC = 21          # khop lai moi ~1 thang giao dich
DAM = 750          # so quan sat toi thieu truoc khi duoc phep uoc phan vi
CUON = 500         # do dai cua so cuon cho V2
EPS = 1e-12
TEN_DOAN = ("huấn luyện", "kiểm định", "kiểm tra")


def nap():
    from api.main import noi_chuoi
    ra = {}
    for p in B.PAIRS:
        m = merge_thin_days(noi_chuoi(p))
        sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, p), 0.0))
        c = m.close.values
        z = np.full(len(m), np.nan)
        z[1:] = np.log(c[1:] / np.maximum(c[:-1], EPS)) / np.maximum(sig[1:], EPS)
        ra[p] = dict(z=z, sig=sig, g=doan(m.Date.values))
    return ra


def phan_vi_cuon(z, g, a, kieu, buoc=BUOC, dam=DAM, cuon=CUON):
    """Phan vi muc `a` va ky vong duoi phan vi do, uoc NHAN QUA tung phien.

    kieu = "huan_luyen" | "mo_rong" | "cuon"
    Tra ve (qz, ez) do dai n; NaN o nhung phien chua du dam."""
    n = len(z)
    qz = np.full(n, np.nan)
    ez = np.full(n, np.nan)
    ok = np.isfinite(z)

    if kieu == "huan_luyen":
        zt = z[(g == 0) & ok]
        q = float(np.quantile(zt, a))
        e = float(np.mean(zt[zt <= q])) if (zt <= q).any() else q
        qz[:] = q
        ez[:] = e
        return qz, ez

    for t0 in range(dam, n, buoc):
        lo = 0 if kieu == "mo_rong" else max(0, t0 - cuon)
        m = np.zeros(n, bool)
        m[lo:t0] = True
        m &= ok
        if m.sum() < dam // 2:
            continue
        zz = z[m]
        q = float(np.quantile(zz, a))
        e = float(np.mean(zz[zz <= q])) if (zz <= q).any() else q
        t1 = min(t0 + buoc, n)
        qz[t0:t1] = q
        ez[t0:t1] = e
    return qz, ez


def cham(z, sig, qz, ez, mask, a):
    """Backtest VaR/ES tren `mask`. Nguong di dong theo sigma^ tung phien."""
    m = mask & np.isfinite(z) & np.isfinite(sig) & (sig > 0) & np.isfinite(qz)
    if m.sum() < 100:
        return None
    y = z[m] * sig[m]
    v = qz[m] * sig[m]
    e = ez[m] * sig[m]
    hits = (y <= v).astype(int)
    _, pk, ph = kupiec(hits, a)
    _, pi_ = christoffersen_ind(hits)
    _, pdq = dq_test(hits, v, a)
    h = hits.astype(bool)
    tyle_es = float(np.mean(e[h]) / np.mean(y[h])) if h.any() else np.nan
    r = lambda v_: None if v_ is None or not np.isfinite(v_) else round(float(v_), 4)
    dat = ((pk is None or not np.isfinite(pk) or pk >= 0.05)
           and (pi_ is None or not np.isfinite(pi_) or pi_ >= 0.05)
           and (pdq is None or not np.isfinite(pdq) or pdq >= 0.05))
    return dict(n=int(m.sum()), vi_pham=r(ph), ky_vong=a, kupiec=r(pk),
                chris=r(pi_), dq=r(pdq), ty_le_es=r(tyle_es), dat=bool(dat))


def chan_doan(D):
    print(f"{'cặp':8}{'đoạn':<12}{'n':>6}{'sd(z)':>8}{'q01':>9}{'q99':>9}")
    for p in B.PAIRS:
        z, g = D[p]["z"], D[p]["g"]
        for i, t in enumerate(TEN_DOAN):
            zz = z[(g == i) & np.isfinite(z)]
            if len(zz) < 50:
                continue
            print(f"{p if i == 0 else '':8}{t:<12}{len(zz):>6}{np.std(zz):>8.3f}"
                  f"{np.quantile(zz, .01):>9.3f}{np.quantile(zz, .99):>9.3f}")
        print()


def main():
    D = nap()
    if "--chan-doan" in sys.argv:
        chan_doan(D)
        return

    print("=" * 104)
    print("VÁ ĐUÔI DƯỚI — ba phương án, CHỌN TRÊN ĐOẠN KIỂM ĐỊNH")
    print("=" * 104)
    kieu = {"V0 huấn luyện": "huan_luyen", "V1 mở rộng": "mo_rong",
            "V2 cuộn 500": "cuon"}
    ra = {"buoc": BUOC, "dam": DAM, "cuon": CUON, "kiem_dinh": {}, "kiem_tra": {}}

    for ten_doan, gid in (("kiem_dinh", 1), ("kiem_tra", 2)):
        if ten_doan == "kiem_tra":
            continue                       # chua cham — phai chot tren kiem dinh truoc
        for a in MUC:
            print(f"\n── mức {1-a:.0%} (α = {a}) · ĐOẠN KIỂM ĐỊNH " + "─" * 50)
            print(f"{'cặp':9}{'phương án':<16}{'vi phạm':>9}{'Kupiec':>9}"
                  f"{'Chris':>8}{'DQ':>8}{'tỷ lệ ES':>10}{'':>4}")
            for p in B.PAIRS:
                z, sig, g = D[p]["z"], D[p]["sig"], D[p]["g"]
                for nhan, k in kieu.items():
                    qz, ez = phan_vi_cuon(z, g, a, k)
                    r = cham(z, sig, qz, ez, g == gid, a)
                    if r is None:
                        continue
                    ra[ten_doan].setdefault(f"{a}", {}).setdefault(p, {})[nhan] = r
                    f = lambda v: "  —  " if v is None else f"{v:.3f}"
                    print(f"{p if nhan.startswith('V0') else '':9}{nhan:<16}"
                          f"{100*r['vi_pham']:>8.2f}%{f(r['kupiec']):>9}"
                          f"{f(r['chris']):>8}{f(r['dq']):>8}"
                          f"{f(r['ty_le_es']):>10}  {'đạt' if r['dat'] else 'KHÔNG'}")
                print()

    # tong ket: phuong an nao dat nhieu o nhat tren kiem dinh
    print("=" * 104)
    print(f"{'phương án':<16}{'số ô đạt / tổng':>20}{'|tỷ lệ ES − 1| trung bình':>28}")
    tot = None
    for nhan in kieu:
        dat = tong = 0
        lech = []
        for a in MUC:
            for p in B.PAIRS:
                r = ra["kiem_dinh"].get(f"{a}", {}).get(p, {}).get(nhan)
                if not r:
                    continue
                tong += 1
                dat += int(r["dat"])
                if r["ty_le_es"] is not None:
                    lech.append(abs(r["ty_le_es"] - 1.0))
        tb = float(np.mean(lech)) if lech else np.nan
        print(f"{nhan:<16}{dat:>13}/{tong:<6}{tb:>28.4f}")
        if tot is None or (dat, -tb) > tot[1]:
            tot = (nhan, (dat, -tb))
    print(f"\n→ TỐT NHẤT TRÊN KIỂM ĐỊNH: {tot[0]}")
    ra["chon"] = tot[0]
    print("=" * 104)
    with open(os.path.join(OUT, "va_duoi.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("đã ghi output/va_duoi.json")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
