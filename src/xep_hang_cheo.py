"""XEP HANG CHEO SAU DONG TIEN — doi cau hoi, khong doi mo hinh.

Gia thuyet, tin hieu, cua chi phi va tieu chi phu dinh DA CHOT TRUOC o
`docs/XEPHANG_TIEUCHI.md`.

VI SAO. `KEHOACH_2026Q4.md` muc 1.3: thay cau hoi "EURUSD se len hay xuong"
bang "trong 6 cap, cap nao manh nhat". Da kiem (grep toan src/ va docs/): huong
nay KHONG co mot dong ma nao trong repo — chi xuat hien trong chinh file ke
hoach. Day la lo hong that.

Ly le: truc huong da can kiet qua 8.652 gia thuyet va 13 nhanh doc lap, NHUNG
tat ca deu do huong TUYET DOI cua tung cap. Nhan to do-la chung (rho = 0,443)
chiem gan nua bien thien va lam nhieu moi phep do do. Xep hang CHEO triet tieu
nhan to do bang cach KHU TRUNG BINH ngang qua 6 dong tien tai moi phien — nen
no la mot bai toan KHAC, khong phai lan thu 14 cua cung bai toan.

Van lieu: currency momentum (Menkhoff, Sarno, Schmeling & Schrimpf 2012,
*Currency Momentum Strategies*, J. Financial Economics 106(3):660-684) va
cross-sectional carry (Lustig, Roussanov & Verdelhan 2011). Ca hai bao cao bang
chung manh hon han du bao huong tung cap.

QUY DOI VE CUNG GOC: loi suat cua dong X so USD.
    cap niem yet XXXUSD (EUR, GBP, AUD):  r_X = +log(P_t/P_{t-1})
    cap niem yet USDXXX (JPY, CAD, CHF):  r_X = -log(P_t/P_{t-1})

Chay:  python src/xep_hang_cheo.py
Ghi:   output/xep_hang_cheo.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
LS = os.path.join(ROOT, "data", "fred_rates.csv")
EPS = 1e-12

import volfc2 as V2                                # noqa: E402
from split import doan                              # noqa: E402
from run_final7 import dm_nw                        # noqa: E402

# dong tien <- cap, kem dau quy doi
DONG = {"EUR": ("EURUSD", +1), "GBP": ("GBPUSD", +1), "AUD": ("AUDUSD", +1),
        "JPY": ("USDJPY", -1), "CAD": ("USDCAD", -1), "CHF": ("USDCHF", -1)}
TRE_THANG = 2          # y het Pha 3 — chuoi FRED thang cong bo tre

# 6 tin hieu CHOT TRUOC
TIN_HIEU = ["mom_1m", "mom_3m", "mom_12m", "dao_1w", "carry", "gop"]


def bang_loi_suat():
    """Ma tran loi suat ngay cua 6 dong so USD + nhan doan, theo ngay chung."""
    bang, chung = V2.nap_bang()
    idx = pd.DatetimeIndex(chung)
    R = pd.DataFrame(index=idx, dtype=float)
    for dt, (cap, dau) in DONG.items():
        d = bang[cap]
        s = pd.Series(np.log(np.maximum(d.close.values, EPS)),
                      index=pd.DatetimeIndex(d.Date)).reindex(idx)
        R[dt] = dau * s.diff()
    g = pd.Series(doan(idx.values), index=idx)
    return R, g


def bang_carry():
    """Chenh lech lai suat dong X tru USD, theo thang, TRE 2 THANG."""
    r = pd.read_csv(LS, parse_dates=["DATE"])
    w = r.pivot_table(index="DATE", columns="cur", values="rate", aggfunc="last")
    w = w.sort_index().resample("MS").last().ffill(limit=3)
    c = pd.DataFrame(index=w.index)
    for dt in DONG:
        if dt in w.columns:
            c[dt] = w[dt] - w["USD"]
    c.index = c.index + pd.DateOffset(months=TRE_THANG)
    return c


def dung_tin_hieu(R, C):
    """Dict ten -> DataFrame tin hieu (da tre, khong ro ri)."""
    # moi tin hieu tai ngay t chi dung du lieu DEN HET ngay t, du bao ngay t+1
    T = {}
    T["mom_1m"] = R.rolling(21, min_periods=15).sum()
    T["mom_3m"] = R.rolling(63, min_periods=45).sum()
    T["mom_12m"] = R.rolling(252, min_periods=180).sum()
    T["dao_1w"] = -R.rolling(5, min_periods=4).sum()      # dao chieu: dau AM
    ca = C.reindex(R.index, method="ffill")
    T["carry"] = ca[[c for c in R.columns if c in ca.columns]]
    # gop = trung binh cua hang (z-score ngang) cua 4 tin hieu co dau chot truoc
    z = []
    for k in ("mom_1m", "mom_3m", "mom_12m", "carry"):
        x = T[k].reindex(columns=R.columns)
        z.append(x.sub(x.mean(1), axis=0).div(x.std(1).replace(0, np.nan), axis=0))
    T["gop"] = sum(z) / len(z)
    return {k: v.reindex(columns=R.columns) for k, v in T.items()}


def ic_ngay(S, R):
    """Spearman IC tung phien: xep hang tin hieu(t) vs loi suat KHU TRUNG BINH (t+1)."""
    y = R.shift(-1)
    y = y.sub(y.mean(1), axis=0)            # khu nhan to do-la chung
    ic, ngay = [], []
    for t in R.index:
        s, r = S.loc[t], y.loc[t]
        m = s.notna() & r.notna()
        if m.sum() < 4:
            continue
        rho = stats.spearmanr(s[m].values, r[m].values).correlation
        if np.isfinite(rho):
            ic.append(float(rho)); ngay.append(t)
    return pd.Series(ic, index=pd.DatetimeIndex(ngay))


def danh_muc(S, R, spread_pip=None):
    """Loi suat ngay cua danh muc mua dinh / ban day (1 dong moi ben), da khu do-la.

    Chi phi: vao/ra moi khi thanh phan doi -> tru 2 lan spread mot chieu.
    """
    y = R.shift(-1)
    y = y.sub(y.mean(1), axis=0)
    lo, ra, truoc = [], [], (None, None)
    for t in R.index:
        s, r = S.loc[t], y.loc[t]
        m = s.notna() & r.notna()
        if m.sum() < 4:
            continue
        ss = s[m].sort_values()
        d_, t_ = ss.index[0], ss.index[-1]
        pl = float(r[t_] - r[d_]) / 2.0
        if spread_pip is not None and (d_, t_) != truoc:
            n_doi = 2 - len({d_, t_} & set(truoc if truoc[0] else ()))
            pl -= spread_pip * n_doi
        truoc = (d_, t_)
        lo.append(pl); ra.append(t)
    return pd.Series(lo, index=pd.DatetimeIndex(ra))


def tom_tat(x):
    if len(x) < 30:
        return dict(n=len(x), tb=np.nan, t_nw=np.nan, p=np.nan)
    _, p = dm_nw(x.values - 0.0)
    t_nw = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if x.std() > 0 else np.nan
    return dict(n=int(len(x)), tb=float(x.mean()), t_nw=t_nw, p=float(p))


def main():
    t0 = time.time()
    print("=" * 104)
    print("XẾP HẠNG CHÉO SÁU ĐỒNG TIỀN — hướng repo chưa thử lần nào")
    print("tiêu chí chốt trước: docs/XEPHANG_TIEUCHI.md")
    print("=" * 104)

    R, g = bang_loi_suat()
    C = bang_carry()
    T = dung_tin_hieu(R, C)
    print(f"{len(R):,} phiên · {list(R.columns)}")
    for nhan, gid in (("huấn luyện", 0), ("kiểm định", 1), ("kiểm tra", 2)):
        print(f"  {nhan:<12}{int((g == gid).sum()):>7,} phiên")

    # tu kiem ro ri: cat bo tuong lai khong doi tin hieu o qua khu
    moc = pd.Timestamp("2023-01-01")
    T2 = dung_tin_hieu(R[R.index < moc], C)
    sai = 0
    for k in T:
        a = T[k][T[k].index < moc]
        b = T2[k].reindex(index=a.index, columns=a.columns)
        sai += int((~np.isclose(a.values, b.values, equal_nan=True)).sum())
    print(f"\ntự kiểm rò rỉ: cắt dữ liệu sau {moc.date()} → {sai} giá trị đổi "
          f"({'ĐẠT' if sai == 0 else 'HỎNG'})")
    if sai:
        sys.exit("tự kiểm rò rỉ HỎNG — dừng")

    spread = float(np.median([1.0 / 10000.0]))        # ~1 pip, xem mục 6 của biên bản
    ra = {}
    for nhan_doan, gid in (("kiem_dinh", 1), ("kiem_tra", 2)):
        print(f"\n── ĐOẠN {nhan_doan.upper()} " + "─" * 70)
        print(f"{'tín hiệu':<12}{'n':>6}{'IC TB':>10}{'t':>8}{'p':>9}"
              f"{'lãi ròng/phiên':>16}{'t ròng':>9}")
        m = (g == gid)
        ra[nhan_doan] = {}
        for k in TIN_HIEU:
            S = T[k][m.values]
            Rm = R[m.values]
            ic = ic_ngay(S, Rm)
            pl = danh_muc(S, Rm, spread_pip=spread)
            a, b = tom_tat(ic), tom_tat(pl)
            ra[nhan_doan][k] = dict(ic=a, danh_muc=b)
            print(f"{k:<12}{a['n']:>6}{a['tb']:>10.4f}{a['t_nw']:>8.2f}"
                  f"{a['p']:>9.4f}{b['tb']*1e4:>14.2f}bp{b['t_nw']:>9.2f}")

    # chon tren kiem dinh theo IC, roi doc ket qua kiem tra cua dung tin hieu do
    tot = max(TIN_HIEU, key=lambda k: ra["kiem_dinh"][k]["ic"]["tb"])
    kt = ra["kiem_tra"][tot]
    print("\n" + "=" * 104)
    print(f"CHỌN TRÊN KIỂM ĐỊNH (IC cao nhất): {tot}")
    print(f"  → kiểm tra: IC {kt['ic']['tb']:+.4f} (t={kt['ic']['t_nw']:.2f}, "
          f"p={kt['ic']['p']:.4f}) · lãi ròng {kt['danh_muc']['tb']*1e4:+.2f}bp/phiên "
          f"(t={kt['danh_muc']['t_nw']:.2f})")

    dk1 = kt["ic"]["t_nw"] > 1.96
    dk2 = kt["danh_muc"]["tb"] > 0 and kt["danh_muc"]["t_nw"] > 1.96
    print("-" * 104)
    print(f"  ĐK1 IC > 0 có ý nghĩa trên kiểm tra (t > 1,96)  "
          f"{'ĐẠT' if dk1 else 'TRƯỢT'}   (t={kt['ic']['t_nw']:.2f})")
    print(f"  ĐK2 lãi RÒNG sau chi phí > 0 có ý nghĩa          "
          f"{'ĐẠT' if dk2 else 'TRƯỢT'}   ({kt['danh_muc']['tb']*1e4:+.2f}bp, "
          f"t={kt['danh_muc']['t_nw']:.2f})")
    print("-" * 104)
    xong = dk1 and dk2
    print(f"  → {'DƯƠNG' if xong else 'ÂM — xếp hạng chéo không mang lại giá trị đo được'}")

    ra.update(chon=tot, dk=dict(dk1=bool(dk1), dk2=bool(dk2)),
              phan_quyet="duong" if xong else "am", spread_mot_chieu=spread)
    os.makedirs(OUT, exist_ok=True)
    json.dump(ra, open(os.path.join(OUT, "xep_hang_cheo.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1,
              default=float)
    print(f"\n→ output/xep_hang_cheo.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
