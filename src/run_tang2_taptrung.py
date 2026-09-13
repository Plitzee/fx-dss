"""TANG 2 — PHEP THU: do tap trung RV trong ngay co cai thien du bao khong?

Gia thuyet DA CHOT TRUOC o commit 07bbe1b (truoc khi cham bat ky so lieu nao):
  Cung mot muc RV ngay, RV TAP TRUNG trong mot cu no ngan thi IT DAI hon RV
  trai deu -> he so cua do tap trung phai AM.

Day la ban TONG QUAT cua phat hien S4 trong `PHA2_KETQUA.md` muc 3b, chay tren
MOI phien cua ca 6 cap (24.902 hang) thay vi chi 135 ky FOMC (3.330 hang).

GIAO THUC: y het nhanh tin tuc — khop tren HUAN LUYEN, chon tren KIEM DINH,
cham MOT LAN tren KIEM TRA. QLIKE bat bien thang do, DM + Newey-West.

MOC SO SANH la du bao HAR SAN XUAT (`V2.du_bao_san_xuat`) — mo hinh that cua
he thong, DA CO thanh phan bipower/jump. Neu do tap trung them duoc gi thi do
la thong tin bipower KHONG bat duoc.

DOI CHUNG QUAN TRONG: mot bien the dung TI TRONG NHAY (rv - bpv)/rv thay cho
do tap trung. Neu doi chung nay cung thang thi do tap trung chi la nhay tra
hinh; neu no thua ma do tap trung thang thi thong tin nam o THU TU THOI GIAN
trong ngay, dung nhu ly le cua gia thuyet.

Chay:  python src/run_tang2_taptrung.py
Ghi:   output/tang2_taptrung.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
TT = os.path.join(ROOT, "data", "tap_trung_rv.csv")
EPS = 1e-12

import volfc2 as V2                                # noqa: E402
from split import doan                              # noqa: E402
from run_final7 import dm_nw                        # noqa: E402
from run_m2_bien_dong import qlike                  # noqa: E402

# danh sach bien the CHOT TRUOC — dem vao KHOA_SO
BIEN_THE = [
    ("B1 mốc (HAR sản xuất)", []),
    ("B2·a + hhi", ["hhi"]),
    ("B2·b + top1", ["top1"]),
    ("B2·c + top12", ["top12"]),
    ("B2·d + log hhi", ["log_hhi"]),
    ("B2·e + tỉ trọng nhảy (đối chứng)", ["ti_nhay"]),
    ("B2·f + hhi + tỉ trọng nhảy", ["hhi", "ti_nhay"]),
]


def dung_bang():
    """Bang dai: moi hang mot (cap, phien) co du HAR, dich t+1 va do tap trung."""
    tt = pd.read_csv(TT, parse_dates=["Date"])
    bang, _ = V2.nap_bang()
    hang = []
    for p in V2.PAIRS:
        d = bang[p]
        ng = pd.DatetimeIndex(d.Date)
        g = doan(d.Date.values)
        rv = np.maximum(d.rv5.values, EPS)
        bpv = np.minimum(np.maximum(d.bpv5.values, EPS), rv)
        h = V2.du_bao_san_xuat(d, p)
        lrv = np.log(rv)
        ti_nhay = (rv - bpv) / rv                      # trong [0, 1)
        s = tt[tt.pair == p].set_index("Date")
        co = s.index
        vt = pd.Index(ng).get_indexer(co)              # ngay tap trung -> hang bang
        tra = {c: np.full(len(d), np.nan) for c in ("hhi", "top1", "top12")}
        ok = vt >= 0
        for c in tra:
            tra[c][vt[ok]] = s[c].values[ok]
        for t in range(len(d) - 1):
            if not (np.isfinite(h[t]) and h[t] > 0 and np.isfinite(lrv[t + 1])):
                continue
            if not np.isfinite(tra["hhi"][t]):
                continue
            hang.append(dict(pair=p, ngay=ng[t], doan=int(g[t]),
                             log_h_har=float(np.log(h[t])),
                             hhi=float(tra["hhi"][t]),
                             log_hhi=float(np.log(tra["hhi"][t])),
                             top1=float(tra["top1"][t]),
                             top12=float(tra["top12"][t]),
                             ti_nhay=float(ti_nhay[t]),
                             y=float(lrv[t + 1]), rv_that=float(rv[t + 1])))
    return pd.DataFrame(hang)


def khop_cham(df, them, ten):
    cot = ["log_h_har"] + list(them)
    d = df[df[cot + ["y"]].notna().all(1)]
    tr = d[d.doan == 0]
    X = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cot])
    beta, *_ = np.linalg.lstsq(X, tr.y.values, rcond=None)
    s2 = float(np.var(tr.y.values - X @ beta))
    ra = {"ten": ten, "he_so": dict(zip(["hằng số"] + cot, beta.round(4).tolist())),
          "s2": s2}
    for nhan, gid in (("kiem_dinh", 1), ("kiem_tra", 2)):
        s = d[d.doan == gid]
        Xs = np.column_stack([np.ones(len(s))] + [s[c].values for c in cot])
        h = np.exp(np.clip(Xs @ beta, -30, 0) + 0.5 * s2)
        ql = qlike(s.rv_that.values, h)
        ra[nhan] = dict(n=int(len(s)), qlike=float(ql.mean()))
        ra[f"_ql_{nhan}"] = ql
        ra[f"_pair_{nhan}"] = s.pair.values
    return ra


def main():
    t0 = time.time()
    print("=" * 100)
    print("TẦNG 2 — ĐỘ TẬP TRUNG RV TRONG NGÀY (giả thuyết chốt trước, commit 07bbe1b)")
    print("=" * 100)

    df = dung_bang()
    print(f"bảng {len(df):,} hàng · {df.pair.nunique()} cặp · "
          f"{df.ngay.min().date()} → {df.ngay.max().date()}")
    for nhan, gid in (("huấn luyện", 0), ("kiểm định", 1), ("kiểm tra", 2)):
        print(f"  {nhan:<12}{int((df.doan == gid).sum()):>8,} hàng")
    print(f"\nhhi: trung vị {df.hhi.median():.4f} · "
          f"p10 {df.hhi.quantile(.1):.4f} · p90 {df.hhi.quantile(.9):.4f}")
    print(f"tương quan hhi ~ log_rv(t+1) (thô): "
          f"{np.corrcoef(df.hhi, df.y)[0,1]:+.4f}")
    print(f"tương quan hhi ~ tỉ trọng nhảy:     "
          f"{np.corrcoef(df.hhi, df.ti_nhay)[0,1]:+.4f}")

    kq = {t: khop_cham(df, c, t) for t, c in BIEN_THE}
    m1 = kq[BIEN_THE[0][0]]

    print("\n" + "=" * 100)
    print(f"{'biến thể':<36}{'QLIKE kđ':>11}{'so B1':>8}"
          f"{'QLIKE kt':>11}{'so B1':>8}{'DM p (kt)':>11}")
    print("-" * 100)
    for ten, r in kq.items():
        vd, kt = r["kiem_dinh"]["qlike"], r["kiem_tra"]["qlike"]
        d_vd = (vd / m1["kiem_dinh"]["qlike"] - 1) * 100
        d_kt = (kt / m1["kiem_tra"]["qlike"] - 1) * 100
        pp = (np.nan if ten == m1["ten"]
              else dm_nw(r["_ql_kiem_tra"] - m1["_ql_kiem_tra"])[1])
        print(f"{ten:<36}{vd:>11.4f}{d_vd:>7.2f}%{kt:>11.4f}{d_kt:>7.2f}%{pp:>11.4f}")
    print("-" * 100)
    print("  (âm = TỐT HƠN mốc HAR sản xuất)")

    print("\nHỆ SỐ trên huấn luyện — kiểm dấu so với giả thuyết chốt trước (phải ÂM):")
    for ten, r in kq.items():
        if ten == m1["ten"]:
            continue
        print(f"  {ten:<36}{r['he_so']}")

    tot = min((t for t in kq if t != m1["ten"]),
              key=lambda t: kq[t]["kiem_dinh"]["qlike"])
    r = kq[tot]
    d_kt = (r["kiem_tra"]["qlike"] / m1["kiem_tra"]["qlike"] - 1) * 100
    print(f"\nTỐT NHẤT TRÊN KIỂM ĐỊNH (quy tắc chọn): {tot}")
    print(f"  → trên kiểm tra: {d_kt:+.2f}% so với B1 "
          f"({'TỐT HƠN' if d_kt < 0 else 'tệ hơn'})")

    # theo tung cap
    print(f"\nTHEO TỪNG CẶP (biến thể đã chọn, đoạn kiểm tra):")
    print(f"  {'cặp':<9}{'n':>6}{'QLIKE B1':>11}{'đã chọn':>11}{'chênh':>9}{'DM p':>9}")
    print("  " + "-" * 55)
    pr = r["_pair_kiem_tra"]; duong = 0
    theo_cap = {}
    for p in sorted(set(pr)):
        m = pr == p
        a, b = m1["_ql_kiem_tra"][m], r["_ql_kiem_tra"][m]
        _, pp = dm_nw(b - a)
        ch = (b.mean() / a.mean() - 1) * 100
        duong += int(ch < 0)
        theo_cap[p] = dict(n=int(m.sum()), b1=float(a.mean()), chon=float(b.mean()),
                           chenh=float(ch), dm_p=float(pp))
        print(f"  {p:<9}{int(m.sum()):>6}{a.mean():>11.4f}{b.mean():>11.4f}"
              f"{ch:>8.2f}%{pp:>9.4f}")
    print("  " + "-" * 55)
    print(f"  → {duong}/{len(theo_cap)} cặp cải thiện")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"bien_the": {t: {k: v for k, v in x.items() if not k.startswith("_")}
                            for t, x in kq.items()},
               "chon": tot, "chenh_kiem_tra": float(d_kt),
               "theo_cap": theo_cap, "cap_duong": f"{duong}/{len(theo_cap)}"},
              open(os.path.join(OUT, "tang2_taptrung.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/tang2_taptrung.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
