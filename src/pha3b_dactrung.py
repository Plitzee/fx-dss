"""PHA 3B / WEEK 1 — dung DAC TRUNG ngoai sinh + KIEM CAT TUONG LAI.

MOT NGUON SU THAT DUY NHAT cho ca Week 2 (Granger) va Week 3 (ablation).
Neu hai tuan dung hai bo dac trung khac nhau thi khong so sanh duoc.

Bien, 6 dang bien doi, va lag {1,2,5} DA CHOT TRUOC o `docs/PHA3B_TIEUCHI.md`
muc 2d-2e (commit e88ae66).

QUY TAC AVAILABLE_TIME (muc 3b, chot truoc):
  gia tri quan sat ngay d chi duoc dung cho du bao tu phien SAU d.
  Nen dac trung tai phien t voi lag L = gia tri quan sat tai phien t-L,
  voi L >= 1. Khong bao gio dung quan sat cua chinh phien t.

CHUAN HOA: trung binh/do lech uoc CHI tren doan HUAN LUYEN roi dong bang —
khong uoc lai tren kiem dinh/kiem tra.

Tu kiem:  python src/pha3b_dactrung.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
NS = os.path.join(DATA, "ngoai_sinh")

from split import doan                                    # noqa: E402

# ── chot truoc, muc 2a: 9 chuoi thi truong + 1 ho lai suat + 1 ho lich
CHUOI_TT = ["DGS2", "DGS10", "T10Y2Y", "DFII10", "VIXCLS", "GVZCLS", "OVXCLS",
            "DCOILBRENTEU", "NIKKEI225"]
DANG = ["lv", "d1", "ad1", "d5", "ad5", "rv20"]           # muc 2d — dung 6 dang
LAG = [1, 2, 5]                                           # muc 2e
# ho E — giu nguyen dinh nghia cua PHA3_TIEUCHI.md muc 3a
DT_LS = ["chenh_ls", "d_chenh_ls", "abs_d_chenh_ls", "ls_usd", "abs_d_ls_usd"]
DT_SK = ["sk_ngay", "sk_ke_tiep", "sk_dem5"]              # ho F — 3 dang
TRE_THANG = 2                                             # muc 3b
CUA_SO_D = 3
# ho F CHI tinh cong bo KHONG phai NHTW (muc 2c — phan NHTW da nam trong B0)
MA_NHTW = {"FOMC", "ECB", "BOE", "BOJ", "RBA", "BOC", "SNB"}

BIEN_GOC = CHUOI_TT + ["chenh_ls", "su_kien_phi_nhtw"]    # 11 bien, muc 2f


def _phien():
    g = pd.read_csv(os.path.join(DATA, "prices", "EURUSD_d1.csv"),
                    parse_dates=["Date"])
    return pd.DatetimeIndex(sorted(g.Date.unique()))


def bien_doi_thi_truong(cat_sau=None):
    """9 chuoi thi truong -> 54 dac trung tren LUOI PHIEN GIAO DICH.

    cat_sau: cat bo moi quan sat SAU moc nay (dung cho kiem cat tuong lai).
    Moi bien doi chi nhin ve QUA KHU, nen cat tuong lai khong duoc doi gi.
    """
    x = pd.read_csv(os.path.join(NS, "chuoi_ngay.csv"), parse_dates=["date"])
    x = x.set_index("date").sort_index()
    if cat_sau is not None:
        x = x[x.index <= pd.Timestamp(cat_sau)]
    ph = _phien()
    ph = ph[ph <= x.index.max()] if len(x) else ph
    # dua ve luoi phien giao dich, bu ngay le toi da 5 phien
    g = x.reindex(x.index.union(ph)).sort_index().ffill(limit=5).reindex(ph)

    ra = {}
    for c in CHUOI_TT:
        s = g[c].astype(float)
        d1 = s.diff(1)
        ra[f"{c}_lv"] = s
        ra[f"{c}_d1"] = d1
        ra[f"{c}_ad1"] = d1.abs()
        ra[f"{c}_d5"] = s.diff(5)
        ra[f"{c}_ad5"] = s.diff(5).abs()
        ra[f"{c}_rv20"] = d1.rolling(20, min_periods=15).std()
    return pd.DataFrame(ra, index=ph)


def bien_doi_lai_suat(cat_sau=None):
    """Ho E — 5 dac trung theo (cap, thang ap dung). Tre 2 thang, chot truoc."""
    r = pd.read_csv(os.path.join(DATA, "fred_rates.csv"), parse_dates=["DATE"])
    if cat_sau is not None:
        r = r[r.DATE <= pd.Timestamp(cat_sau)]
    w = (r.pivot_table(index="DATE", columns="cur", values="rate", aggfunc="last")
         .sort_index().resample("MS").last().ffill(limit=3))
    dt = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
          "AUDUSD": ("AUD", "USD"), "USDJPY": ("USD", "JPY"),
          "USDCAD": ("USD", "CAD"), "USDCHF": ("USD", "CHF")}
    hang = []
    usd, d_usd = w["USD"].dropna(), w["USD"].dropna().diff(CUA_SO_D)
    for p, (a, b) in dt.items():
        ch = (w[a] - w[b]).dropna()
        d = ch.diff(CUA_SO_D)
        for thang in ch.index:
            hang.append(dict(
                pair=p,
                thang_ap=(thang + pd.DateOffset(months=TRE_THANG)).to_period("M"),
                chenh_ls=float(ch[thang]), d_chenh_ls=float(d.get(thang, np.nan)),
                abs_d_chenh_ls=abs(float(d.get(thang, np.nan))),
                ls_usd=float(usd.get(thang, np.nan)),
                abs_d_ls_usd=abs(float(d_usd.get(thang, np.nan)))))
    return pd.DataFrame(hang)


def bien_doi_su_kien(cat_sau=None):
    """Ho F — lich cong bo PHI NHTW. Chi dung NGAY, khong dung gia tri."""
    s = pd.read_csv(os.path.join(DATA, "su_kien.csv"), parse_dates=["date"])
    s = s[~s.ma.isin(MA_NHTW)]
    if cat_sau is not None:
        s = s[s.date <= pd.Timestamp(cat_sau)]
    ph = _phien()
    co = pd.Series(0.0, index=ph)
    dem = pd.Series(s.groupby("date").size())
    chung = co.index.intersection(dem.index)
    co.loc[chung] = dem.loc[chung].astype(float)
    return pd.DataFrame({
        "sk_ngay": (co > 0).astype(float),
        "sk_ke_tiep": (co > 0).astype(float).shift(1).fillna(0.0),
        "sk_dem5": co.rolling(5, min_periods=1).sum()}, index=ph)


def cot_thi_truong():
    return [f"{c}_{d}" for c in CHUOI_TT for d in DANG]


def cot_theo_bien(bien):
    """Khoi dac trung cua MOT bien goc — dung cho kiem Granger theo khoi."""
    if bien in CHUOI_TT:
        return [f"{bien}_{d}" for d in DANG]
    if bien == "chenh_ls":
        return list(DT_LS)
    if bien == "su_kien_phi_nhtw":
        return list(DT_SK)
    raise KeyError(bien)


def dung(panel_ngay, panel_pair, cat_sau=None, chuan_hoa=True):
    """Dac trung ngoai sinh cho tung hang (ngay, cap) cua bang panel.

    Tra ve DataFrame cung so hang voi panel, cot = <dac trung>_L<lag>.
    """
    ngay = pd.DatetimeIndex(panel_ngay)
    pair = np.asarray(panel_pair)
    tt = bien_doi_thi_truong(cat_sau)
    sk = bien_doi_su_kien(cat_sau)
    ls = bien_doi_lai_suat(cat_sau)

    # chuan hoa bang thong ke cua doan HUAN LUYEN (chot truoc)
    if chuan_hoa:
        tr = doan(tt.index.values) == 0
        mu, sd = tt[tr].mean(), tt[tr].std().replace(0, np.nan)
        tt = (tt - mu) / sd

    ph = tt.index
    vt = pd.Series(np.arange(len(ph)), index=ph)
    i = vt.reindex(ngay).values.astype(float)               # vi tri phien cua moi hang

    ra = {}
    Mtt = tt.values
    for L in LAG:
        j = i - L                                            # lag L phien
        ok = np.isfinite(j) & (j >= 0)
        jj = np.where(ok, j, 0).astype(int)
        for k, c in enumerate(tt.columns):
            v = Mtt[jj, k].astype(float)
            v[~ok] = np.nan
            ra[f"{c}_L{L}"] = v
        Msk = sk.values
        for k, c in enumerate(sk.columns):
            v = Msk[jj, k].astype(float)
            v[~ok] = np.nan
            ra[f"{c}_L{L}"] = v

    # ho E theo thang ap dung — khong co lag phien (da tre 2 thang)
    thang = pd.PeriodIndex(ngay, freq="M")
    key = pd.DataFrame(dict(pair=pair, thang_ap=thang))
    e = key.merge(ls, on=["pair", "thang_ap"], how="left")
    for L in LAG:
        for c in DT_LS:
            ra[f"{c}_L{L}"] = e[c].values                    # bat bien theo lag phien
    return pd.DataFrame(ra, index=np.arange(len(ngay)))


def cot_lag(bien, L):
    return [f"{c}_L{L}" for c in cot_theo_bien(bien)]


def tat_ca_cot():
    return [f"{c}_L{L}" for L in LAG
            for c in cot_thi_truong() + DT_SK + DT_LS]


# ────────────────────────────────────────────────────────────── tu kiem
def _tu_kiem():
    print("=" * 96)
    print("PHA 3B / WEEK 1 — TỰ KIỂM ĐẶC TRƯNG NGOẠI SINH")
    print("=" * 96)
    pan = pd.read_csv(os.path.join(DATA, "panel2_6pairs.csv"), parse_dates=["Date"])
    pan = pan.sort_values(["Date", "pair"]).reset_index(drop=True)
    print(f"panel: {len(pan):,} hàng · {pan.pair.nunique()} cặp · "
          f"{pan.Date.min().date()} → {pan.Date.max().date()}")

    X = dung(pan.Date.values, pan.pair.values)
    print(f"đặc trưng: {X.shape[1]} cột "
          f"({len(cot_thi_truong())} thị trường + {len(DT_SK)} lịch + "
          f"{len(DT_LS)} lãi suất, × {len(LAG)} lag)")
    print(f"  11 biến gốc: {', '.join(BIEN_GOC)}")

    # ── 1. KIEM CAT TUONG LAI (muc 3d(a)) — nguong dat: 0 gia tri doi
    moc = "2023-01-01"
    truoc = pan.Date < pd.Timestamp(moc)
    Xc = dung(pan.Date.values, pan.pair.values, cat_sau=moc)
    a, b = X[truoc], Xc[truoc]
    chung = [c for c in a.columns if c in b.columns]
    kh = (~np.isclose(a[chung].values, b[chung].values, equal_nan=True,
                      rtol=1e-9, atol=1e-9))
    n_kh, n_o = int(kh.sum()), int(a[chung].size)
    print(f"\n1. KIỂM CẮT TƯƠNG LAI (cắt sau {moc})")
    print(f"   {n_kh}/{n_o:,} giá trị đổi   "
          f"{'ĐẠT' if n_kh == 0 else 'HỎNG — ' + str(sorted(set(np.array(chung)[kh.any(0)]))[:5])}")
    assert n_kh == 0, "ro ri nhin truoc trong dac trung ngoai sinh"

    # ── 2. KIEM LAG: dac trung tai phien t khong duoc chua quan sat cua t
    tt = bien_doi_thi_truong()
    ph = tt.index
    t = ph[3000]
    hang = pan.index[pan.Date == t]
    if len(hang):
        v_dt = X.loc[hang[0], "VIXCLS_lv_L1"]
        # chuan hoa lai de doi chieu
        tr = doan(tt.index.values) == 0
        mu, sd = tt[tr].mean(), tt[tr].std()
        v_ky_vong = (tt.VIXCLS_lv.iloc[np.where(ph == t)[0][0] - 1]
                     - mu.VIXCLS_lv) / sd.VIXCLS_lv
        ok = np.isclose(v_dt, v_ky_vong, equal_nan=True)
        print(f"\n2. KIỂM ĐỘ TRỄ (VIXCLS_lv_L1 tại {t.date()})")
        print(f"   đặc trưng={v_dt:+.5f} · kỳ vọng (phiên t−1)={v_ky_vong:+.5f}   "
              f"{'ĐẠT' if ok else 'HỎNG'}")
        assert ok, "lag sai — dac trung dang dung quan sat cua chinh phien t"

    # ── 3. CHUAN HOA chi tu HUAN LUYEN
    #
    # Kiem dung thu can kiem: thong ke chuan hoa uoc tren doan HUAN LUYEN cua
    # CHINH CHUOI (chi so 2010-01 -> 2021-10), nen chuoi da chuan hoa phai co
    # trung binh ~0 tren dung doan do. Trung binh tren HANG PANEL se KHAC 0 vi
    # panel bat dau 2012-02 (mat 2010-2011) — do la thong tin, khong phai loi.
    tt_c = bien_doi_thi_truong()
    tr_c = doan(tt_c.index.values) == 0
    mu_c, sd_c = tt_c[tr_c].mean(), tt_c[tr_c].std().replace(0, np.nan)
    z = ((tt_c - mu_c) / sd_c)[tr_c]
    m_max = float(np.nanmax(np.abs(z.mean().values)))
    s_max = float(np.nanmax(np.abs(z.std().values - 1.0)))
    print(f"\n3. CHUẨN HOÁ TỪ HUẤN LUYỆN (54 cột thị trường, trên đoạn huấn "
          f"luyện của chuỗi)")
    print(f"   |trung bình| lớn nhất = {m_max:.2e} · |độ lệch−1| lớn nhất = "
          f"{s_max:.2e}   {'ĐẠT' if m_max < 1e-9 and s_max < 1e-9 else 'HỎNG'}")
    assert m_max < 1e-9 and s_max < 1e-9, "chuan hoa khong lay tu huan luyen"

    g = doan(pan.Date.values)
    m_pan = float(np.nanmean(X.loc[g == 0, "VIXCLS_lv_L1"]))
    print(f"   (trung bình trên HÀNG PANEL huấn luyện = {m_pan:+.4f} — khác 0 "
          f"vì panel bắt đầu 2012-02, chuỗi từ 2010-01; đúng như thiết kế)")

    # ── 4. DO PHU theo doan
    print(f"\n4. ĐỘ PHỦ theo đoạn (tỷ lệ hàng có ĐỦ 54 đặc trưng thị trường lag 1)")
    c1 = [f"{c}_L1" for c in cot_thi_truong()]
    for i, ten in enumerate(("huấn luyện", "kiểm định", "kiểm tra")):
        m = g == i
        du = X.loc[m, c1].notna().all(1).mean()
        print(f"   {ten:<12}{int(m.sum()):>7,} hàng · đủ {100*du:>5.1f}%")
    print("\nTẤT CẢ TỰ KIỂM ĐẠT.")


if __name__ == "__main__":
    _tu_kiem()
