"""PHA 3 — TANG VI MO (M3) tren truc bien do.

Gia thuyet, dac trung, tieu chi phu dinh va khai bao luc DA CHOT TRUOC o
`docs/PHA3_TIEUCHI.md` (commit 5651be0), truoc khi cham bat ky so lieu nao.

  H9  Do lon thay doi chenh lech lai suat (phan ky lap truong chinh sach)
      du bao BIEN DONG CAO HON. DAU DU KIEN: DUONG.
      Co che: thao vi the carry (Brunnermeier, Nagel & Pedersen 2009).

MOC B1 = HAR san xuat (`V2.du_bao_san_xuat`), hoi quy lai bang OLS.
Khop tren HUAN LUYEN, chon tren KIEM DINH, cham MOT LAN tren KIEM TRA.

CHONG RO RI (chot truoc, muc 4): chuoi FRED thang tre 2 THANG — gia tri cua
thang m chi dung cho cac phien tu thang m+2. Kem tu kiem "cat tuong lai khong
doi gia tri dac trung cua bat ky phien nao".

Chay:  python src/run_pha3_vimo.py
Ghi:   output/pha3_vimo.json
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
LS = os.path.join(ROOT, "data", "fred_rates.csv")
EPS = 1e-12
TRE_THANG = 2          # chot truoc — muc 4 cua PHA3_TIEUCHI.md
CUA_SO_D = 3           # thay doi 3 thang — chot truoc

import volfc2 as V2                                # noqa: E402
from split import doan                              # noqa: E402
from run_final7 import dm_nw                        # noqa: E402
from run_m2_bien_dong import qlike                  # noqa: E402

# 7 bien the CHOT TRUOC o muc 3a — khong them, khong bot
BIEN_THE = [
    ("B1 mốc (HAR sản xuất)", []),
    ("M3·1 chênh lãi suất (mức)", ["chenh_ls"]),
    ("M3·2 Δ3th chênh lãi suất (dấu)", ["d_chenh_ls"]),
    ("M3·3 |Δ3th chênh lãi suất| ← H9", ["abs_d_chenh_ls"]),
    ("M3·4 lãi suất USD (mức)", ["ls_usd"]),
    ("M3·5 |Δ3th lãi suất USD|", ["abs_d_ls_usd"]),
    ("M3·6 gộp |Δ| cặp + |Δ| USD", ["abs_d_chenh_ls", "abs_d_ls_usd"]),
]


def bang_lai_suat():
    """Bang lai suat thang theo dong tien, da TRE 2 THANG (chot truoc)."""
    r = pd.read_csv(LS, parse_dates=["DATE"])
    w = r.pivot_table(index="DATE", columns="cur", values="rate", aggfunc="last")
    w = w.sort_index().resample("MS").last().ffill(limit=3)
    return w


def dac_trung_thang(w):
    """Moi dong: mot (cap, thang ap dung). Thang ap dung = thang du lieu + tre."""
    dt = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
          "AUDUSD": ("AUD", "USD"), "USDJPY": ("USD", "JPY"),
          "USDCAD": ("USD", "CAD"), "USDCHF": ("USD", "CHF")}
    hang = []
    for p, (a, b) in dt.items():
        if a not in w.columns or b not in w.columns:
            continue
        ch = (w[a] - w[b]).dropna()
        d = ch.diff(CUA_SO_D)
        usd = w["USD"].dropna()
        d_usd = usd.diff(CUA_SO_D)
        for thang in ch.index:
            if not np.isfinite(ch.get(thang, np.nan)):
                continue
            hang.append(dict(
                pair=p,
                # thang AP DUNG = thang du lieu + tre 2 thang
                thang_ap=(thang + pd.DateOffset(months=TRE_THANG)).to_period("M"),
                chenh_ls=float(ch[thang]),
                d_chenh_ls=float(d.get(thang, np.nan)),
                abs_d_chenh_ls=abs(float(d.get(thang, np.nan))),
                ls_usd=float(usd.get(thang, np.nan)),
                abs_d_ls_usd=abs(float(d_usd.get(thang, np.nan)))))
    return pd.DataFrame(hang)


def dung_bang(F):
    bang, _ = V2.nap_bang()
    hang = []
    for p in V2.PAIRS:
        d = bang[p]
        ng = pd.DatetimeIndex(d.Date)
        g = doan(d.Date.values)
        rv = np.maximum(d.rv5.values, EPS)
        h = V2.du_bao_san_xuat(d, p)
        lrv = np.log(rv)
        for t in range(len(d) - 1):
            if not (np.isfinite(h[t]) and h[t] > 0 and np.isfinite(lrv[t + 1])):
                continue
            hang.append(dict(pair=p, ngay=ng[t], thang_ap=ng[t].to_period("M"),
                             doan=int(g[t]), log_h_har=float(np.log(h[t])),
                             y=float(lrv[t + 1]), rv_that=float(rv[t + 1])))
    df = pd.DataFrame(hang)
    return df.merge(F, on=["pair", "thang_ap"], how="left")


def tu_kiem_ro_ri(w):
    """Cat bo toan bo du lieu sau mot moc -> dac trung truoc moc do phai KHONG DOI."""
    moc = pd.Timestamp("2023-01-01")
    day = dac_trung_thang(w)
    cut = dac_trung_thang(w[w.index < moc])
    k = ["pair", "thang_ap"]
    c = [x for x in day.columns if x not in k]
    a = day[day.thang_ap < moc.to_period("M")].set_index(k)[c].sort_index()
    b = cut[cut.thang_ap < moc.to_period("M")].set_index(k)[c].sort_index()
    chung = a.index.intersection(b.index)
    if len(chung) == 0:
        return None, 0
    sai = int((~np.isclose(a.loc[chung].values, b.loc[chung].values,
                           equal_nan=True)).sum())
    return sai, len(chung)


def khop_cham(df, them, ten):
    cot = ["log_h_har"] + list(them)
    d = df[df[cot + ["y"]].notna().all(1)]
    tr = d[d.doan == 0]
    X = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cot])
    beta, *_ = np.linalg.lstsq(X, tr.y.values, rcond=None)
    s2 = float(np.var(tr.y.values - X @ beta))
    ra = {"ten": ten,
          "he_so": dict(zip(["hằng số"] + cot, beta.round(5).tolist()))}
    for nhan, gid in (("kiem_dinh", 1), ("kiem_tra", 2)):
        s = d[d.doan == gid]
        Xs = np.column_stack([np.ones(len(s))] + [s[c].values for c in cot])
        hh = np.exp(np.clip(Xs @ beta, -30, 0) + 0.5 * s2)
        ql = qlike(s.rv_that.values, hh)
        ra[nhan] = dict(n=int(len(s)), qlike=float(ql.mean()))
        ra[f"_ql_{nhan}"] = ql
        ra[f"_pair_{nhan}"] = s.pair.values
    return ra


def main():
    t0 = time.time()
    print("=" * 100)
    print("PHA 3 — TẦNG VĨ MÔ (M3), trục biên độ")
    print("tiêu chí chốt trước: docs/PHA3_TIEUCHI.md (commit 5651be0)")
    print("=" * 100)

    w = bang_lai_suat()
    print(f"lãi suất: {list(w.columns)} · {w.index.min().date()} → "
          f"{w.index.max().date()}")
    sai, n = tu_kiem_ro_ri(w)
    print(f"tự kiểm rò rỉ: cắt dữ liệu sau 2023-01-01 → {sai}/{n*5} giá trị đổi "
          f"({'ĐẠT' if sai == 0 else 'HỎNG'})")
    if sai:
        sys.exit("tự kiểm rò rỉ HỎNG — dừng")

    F = dac_trung_thang(w)
    df = dung_bang(F)
    cot5 = ["chenh_ls", "d_chenh_ls", "abs_d_chenh_ls", "ls_usd", "abs_d_ls_usd"]
    co = df[cot5].notna().all(1)
    df = df[co]
    print(f"\nbảng {len(df):,} hàng · {df.pair.nunique()} cặp · "
          f"{df.ngay.min().date()} → {df.ngay.max().date()}")
    for nhan, gid in (("huấn luyện", 0), ("kiểm định", 1), ("kiểm tra", 2)):
        s = df[df.doan == gid]
        print(f"  {nhan:<12}{len(s):>7,} phiên · "
              f"{s.thang_ap.nunique():>4} tháng độc lập")
    print(f"\ntương quan |Δ3th chênh lãi suất| ~ log_rv(t+1) (thô): "
          f"{np.corrcoef(df.abs_d_chenh_ls, df.y)[0,1]:+.4f}")

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

    print(f"\nHỆ SỐ trên huấn luyện — H9 đòi dấu của |Δ3th chênh lãi suất| là DƯƠNG:")
    for ten, r in kq.items():
        if ten != m1["ten"]:
            print(f"  {ten:<36}{r['he_so']}")

    tot = min((t for t in kq if t != m1["ten"]),
              key=lambda t: kq[t]["kiem_dinh"]["qlike"])
    r = kq[tot]
    d_kt = (r["kiem_tra"]["qlike"] / m1["kiem_tra"]["qlike"] - 1) * 100
    _, p_kt = dm_nw(r["_ql_kiem_tra"] - m1["_ql_kiem_tra"])
    print(f"\nTỐT NHẤT TRÊN KIỂM ĐỊNH (quy tắc chọn): {tot}")
    print(f"  → trên kiểm tra: {d_kt:+.2f}% · DM p thô {p_kt:.4f}")

    pr = r["_pair_kiem_tra"]; duong = 0; theo_cap = {}
    print(f"\nTHEO TỪNG CẶP (biến thể đã chọn, đoạn kiểm tra):")
    print(f"  {'cặp':<9}{'n':>6}{'QLIKE B1':>11}{'đã chọn':>11}{'chênh':>9}{'DM p':>9}")
    print("  " + "-" * 55)
    for p in sorted(set(pr)):
        m = pr == p
        a, b = m1["_ql_kiem_tra"][m], r["_ql_kiem_tra"][m]
        _, pp = dm_nw(b - a)
        ch = (b.mean() / a.mean() - 1) * 100
        duong += int(ch < 0)
        theo_cap[p] = dict(n=int(m.sum()), chenh=float(ch), dm_p=float(pp))
        print(f"  {p:<9}{int(m.sum()):>6}{a.mean():>11.4f}{b.mean():>11.4f}"
              f"{ch:>8.2f}%{pp:>9.4f}")
    print("  " + "-" * 55)
    print(f"  → {duong}/{len(theo_cap)} cặp cải thiện")

    # ── phan quyet theo muc 6, KHONG dien giai lai
    he_so_h9 = kq["M3·3 |Δ3th chênh lãi suất| ← H9"]["he_so"]["abs_d_chenh_ls"]
    dk1 = (d_kt < 0) and (p_kt < 0.05 / 6)       # Bonferroni 6 bien the
    dk2 = duong >= 5
    dk3 = he_so_h9 > 0
    print("\n" + "=" * 100)
    print("PHÁN QUYẾT theo TIÊU CHÍ CHỐT TRƯỚC (PHA3_TIEUCHI.md mục 6)")
    print("-" * 100)
    print(f"  ĐK1 thắng kiểm tra & p thô < 0,0083 (Bonferroni)   "
          f"{'ĐẠT' if dk1 else 'TRƯỢT'}   ({d_kt:+.2f}%, p={p_kt:.4f})")
    print(f"  ĐK2 ≥ 5/6 cặp cải thiện                            "
          f"{'ĐẠT' if dk2 else 'TRƯỢT'}   ({duong}/6)")
    print(f"  ĐK3 hệ số |Δ chênh lãi suất| DƯƠNG như H9          "
          f"{'ĐẠT' if dk3 else 'TRƯỢT'}   ({he_so_h9:+.5f})")
    print("-" * 100)
    duong_het = dk1 and dk2 and dk3
    print(f"  → {'DƯƠNG' if duong_het else 'ÂM — tầng vĩ mô không mang lại giá trị đo được'}")
    if not duong_het:
        print("  Nhắc lại khai báo lực (mục 7): chỉ ~26 tháng độc lập ở kiểm tra.")
        print("  Kết luận âm này phát biểu được là 'không phát hiện được hiệu ứng")
        print("  với dữ liệu tháng trên 26 tháng', KHÔNG phải 'tầng vĩ mô rỗng'.")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"bien_the": {t: {k: v for k, v in x.items() if not k.startswith("_")}
                            for t, x in kq.items()},
               "chon": tot, "chenh_kiem_tra": float(d_kt), "dm_p_tho": float(p_kt),
               "theo_cap": theo_cap, "cap_duong": f"{duong}/{len(theo_cap)}",
               "he_so_h9": float(he_so_h9),
               "dk": {"dk1": bool(dk1), "dk2": bool(dk2), "dk3": bool(dk3)},
               "phan_quyet": "duong" if duong_het else "am"},
              open(os.path.join(OUT, "pha3_vimo.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/pha3_vimo.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
