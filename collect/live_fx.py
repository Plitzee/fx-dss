"""TAI DU LIEU FX HIEN HANH — lap day tu 2026-01-01 den hom nay.

Boi canh. Du lieu goc cua repo la HistData M1 -> H1 -> D1, dung o 2025-12-31.
`docs/DONGBO_SANXUAT.md` da ghi cho tac nghen: HistData phat hanh THEO NAM nen
khong dung cap nhat hang ngay duoc, con bo tai Dukascopy `.bi5` thi chua ai
viet. Da do tu may nay:

  * Dukascopy  KHONG truy cap duoc (timeout 15-21s, moi lan thu)
  * Stooq      tra ve trang kiem tra bot (proof-of-work) — khong vuot
  * FRED       duoc, chinh thuc, nhung tre ~1 tuan va la gia trua NY, khong OHLC
  * Yahoo      duoc, co OHLC + thanh 5 phut. KHONG CHINH THUC, khong cam ket
               dich vu. Du cho luan van va ban trinh dien; he thong chay that
               nen mua nha cung cap co hop dong.

BAI HOC DA TRA GIA TRONG CHINH FILE NAY. Ban dau lay thang `interval=1d` cua
Yahoo. Sai: 37,1% so ngay co `close == open` va `high < open` — thanh nen KHONG
HOP LE, lech trung binh 16,6 pip so HistData. Cach dung la lam DUNG NHU repo
van lam: lay thanh GIO roi tu gop len ngay. Sau khi doi: 0 thanh khong hop le,
lech trung vi 3,40 pip, tuong quan 0,9999855.

RV5 THAT CHO DU BAO HOM NAY. HAR can phuong sai thuc hien 5 phut. Thanh 5 phut
cua Yahoo phu 60 ngay, ma do tre dai nhat cua HAR la 22 phien — nen du bao cho
hom nay dung RV5 THAT hoan toan. Doan 2026-01 -> truoc cua so 60 ngay chi dung
de VE BIEU DO va duoc uoc tu thanh gio (danh dau `rv_uoc=1`), khong nuoi du bao.

VI SAO PHAI DO MOI NOI. `docs/KHOA_SO.md` tung tu choi va du lieu bang nguon
thu hai vi "va se cay mot moi noi giua hai nha cung cap vao giua chuoi". O day
buoc phai noi, nen phai DO va CONG BO thay vi giau.

Chay:  python collect/live_fx.py
Ghi:   data/live/{PAIR}_d1.csv, {PAIR}_rv.csv, moi_noi.json
"""
import datetime as dt
import io
import json
import os
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")
OUT = os.path.join(D, "live")
os.makedirs(OUT, exist_ok=True)

PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF")
FRED_ID = {"EURUSD": "DEXUSEU", "GBPUSD": "DEXUSUK", "USDJPY": "DEXJPUS",
           "AUDUSD": "DEXUSAL", "USDCAD": "DEXCAUS", "USDCHF": "DEXSZUS"}
FRED_DAO = {"EURUSD": False, "GBPUSD": False, "AUDUSD": False,
            "USDJPY": False, "USDCAD": False, "USDCHF": False}
UA = {"User-Agent": "Mozilla/5.0 (compatible; fx-dss-thesis/1.0)"}
PIP = {"USDJPY": 0.01}
MIN_GIO_NGAY = 18      # ngay du thanh gio moi coi la phien day du
MIN_M5_NGAY = 100      # ngay du thanh 5 phut moi tinh rv5 that
EPS = 1e-14


def pip_size(p):
    return PIP.get(p, 0.0001)


def yahoo(sym, rng, iv, thu=3):
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    for k in range(thu):
        try:
            r = requests.get(u, params={"range": rng, "interval": iv},
                             headers=UA, timeout=30)
            r.raise_for_status()
            d = r.json()["chart"]["result"][0]
            q = d["indicators"]["quote"][0]
            f = pd.DataFrame({
                "ts": pd.to_datetime(d["timestamp"], unit="s", utc=True),
                "open": q["open"], "high": q["high"],
                "low": q["low"], "close": q["close"]})
            return f.dropna().reset_index(drop=True)
        except Exception:
            if k == thu - 1:
                raise
            time.sleep(1.5 * (k + 1))


def _ngay(f):
    return f.ts.dt.normalize().dt.tz_localize(None)


def gop_ngay(h1):
    """Gop thanh gio -> nen ngay UTC, dung quy uoc cua collect/prep_fx.py."""
    h1 = h1.copy()
    h1["Date"] = _ngay(h1)
    g = h1.groupby("Date").agg(open=("open", "first"), high=("high", "max"),
                               low=("low", "min"), close=("close", "last"),
                               n_h1=("close", "size")).reset_index()
    xau = ((g.high < g[["open", "close"]].max(1) - 1e-9) |
           (g.low > g[["open", "close"]].min(1) + 1e-9)).sum()
    return g, int(xau)


def do_luong_noi_ngay(f, hau_to):
    """Bo NAM do luong noi ngay ma `volfc2.du_bao_san_xuat` doi hoi, tinh tu
    loi suat log TRONG NGAY (bo gap qua dem — quy uoc collect/rv5.py va
    collect/rv_advanced.py; gap qua dem chi 1,7-3,1% tong phuong sai o FX):

      rv   = sum r^2                     phuong sai thuc hien
      rq   = n/3 * sum r^4               realized quarticity (sai so do luong)
      bpv  = pi/2 * sum |r_i||r_{i-1}|   bipower variation (bo phan nhay)
      rsp  = sum r^2 * 1{r>0}            ban phuong sai duong
      rsn  = sum r^2 * 1{r<0}            ban phuong sai am   (rsp + rsn = rv)
      n    = so loi suat dung de tinh
    """
    f = f.copy()
    f["Date"] = _ngay(f)
    ra = []
    for d, g in f.groupby("Date"):
        c = g.close.values
        if len(c) < 3:
            continue
        r = np.diff(np.log(np.maximum(c, EPS)))
        n = len(r)
        a = np.abs(r)
        ra.append({
            "Date": d,
            f"rv{hau_to}": float(np.sum(r ** 2)),
            f"rq{hau_to}": float(n / 3.0 * np.sum(r ** 4)),
            f"bpv{hau_to}": float(np.pi / 2.0 * np.sum(a[1:] * a[:-1])),
            f"rsp{hau_to}": float(np.sum(r[r > 0] ** 2)),
            f"rsn{hau_to}": float(np.sum(r[r < 0] ** 2)),
            f"n{hau_to}": n})
    return pd.DataFrame(ra)


def do_moi_noi(p, d1):
    f = os.path.join(D, "prices", f"{p}_d1.csv")
    if not os.path.exists(f):
        return None
    cu = pd.read_csv(f, parse_dates=["Date"])[["Date", "close"]]
    j = pd.merge(cu.rename(columns={"close": "hd"}),
                 d1[["Date", "close", "n_h1"]], on="Date", how="inner")
    j = j[j.n_h1 >= MIN_GIO_NGAY]
    if len(j) < 30:
        return None
    e = (j.close - j.hd).abs() / pip_size(p)
    return dict(n=int(len(j)), tu=str(j.Date.min().date()), den=str(j.Date.max().date()),
                trungvi_pip=round(float(e.median()), 3),
                tb_pip=round(float(e.mean()), 3),
                p95_pip=round(float(e.quantile(0.95)), 3),
                max_pip=round(float(e.max()), 2),
                tuong_quan=round(float(np.corrcoef(j.close, j.hd)[0, 1]), 7))


def doi_chung_fred(p, d1):
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": FRED_ID[p]}, headers=UA, timeout=30)
        r.raise_for_status()
        f = pd.read_csv(io.StringIO(r.text))
        f.columns = ["Date", "v"]
        f["Date"] = pd.to_datetime(f.Date)
        f["v"] = pd.to_numeric(f.v, errors="coerce")
        f = f.dropna()
        j = pd.merge(f, d1[["Date", "close", "n_h1"]], on="Date", how="inner")
        j = j[j.n_h1 >= MIN_GIO_NGAY]
        if len(j) < 30:
            return dict(chuoi=FRED_ID[p], ghi_chu="quá ít mẫu chồng lấn")
        e = (j.close - j.v).abs() / pip_size(p)
        return dict(chuoi=FRED_ID[p], n=int(len(j)), fred_den=str(f.Date.max().date()),
                    trungvi_pip=round(float(e.median()), 2))
    except Exception as e:
        return dict(chuoi=FRED_ID[p], loi=str(e)[:100])


def mot_cap(p):
    h1 = yahoo(f"{p}=X", "730d", "1h")
    m5 = yahoo(f"{p}=X", "60d", "5m")

    d1, xau = gop_ngay(h1)
    a1 = do_luong_noi_ngay(h1, "_h")          # tu thanh GIO  (phu 2 nam)
    a5 = do_luong_noi_ngay(m5, "_m")          # tu thanh 5 PHUT (phu 60 ngay)
    a5 = a5[a5.n_m >= MIN_M5_NGAY]

    a = pd.merge(a1, a5, on="Date", how="left")
    # He so quy doi tung do luong tu khung GIO sang khung 5 PHUT, DO tren chinh
    # doan chong lan. DATASET.md ghi khung gio hut 8-13% so khung 5 phut; do lai
    # thay vi tin so chep tay, va do RIENG tung do luong vi chung khong cung
    # bac theo tan suat lay mau (rq ~ n, bpv ~ rv, ...).
    hs, m_ol = {}, (a.rv_m.notna() & (a.rv_h > EPS))
    for c in ("rv", "rq", "bpv"):
        m = m_ol & (a[c + "_h"] > EPS) & a[c + "_m"].notna()
        hs[c] = float(np.median(a[c + "_m"][m] / a[c + "_h"][m])) if m.sum() >= 20 else 1.0
        a[c + "5"] = np.where(a[c + "_m"].notna(), a[c + "_m"], a[c + "_h"] * hs[c])

    # Ban phuong sai: KHONG co gian doc lap. `volfc2.thiet_ke` dua vao dang
    # thuc rsp + rsn = rv (xem chu thich "khu cung mot he so" trong ham do), ma
    # co gian rsp/rsn bang hai he so rieng lam vo dang thuc ~7%. Cach dung: lay
    # rv5 da co roi PHAN BO theo ty le duong/am do tu khung gio.
    ty = a.rsp_h / np.maximum(a.rsp_h + a.rsn_h, EPS)
    ty = ty.clip(0.05, 0.95).fillna(0.5)
    a["rsp5"] = np.where(a.rsp_m.notna(), a.rsp_m, a["rv5"] * ty)
    a["rsn5"] = np.where(a.rsn_m.notna(), a.rsn_m, a["rv5"] * (1.0 - ty))
    hs["rsp"], hs["rsn"] = float("nan"), float("nan")

    a["n5"] = np.where(a.n_m.notna(), a.n_m, a.n_h * 12.0)
    a["rv_uoc"] = (~a.rv_m.notna()).astype(int)

    cot = ["Date", "rv5", "rq5", "bpv5", "rsp5", "rsn5", "n5", "rv_uoc"]
    d1 = pd.merge(d1, a[cot], on="Date", how="left")
    d1 = d1.rename(columns={"rsp5": "rsp", "rsn5": "rsn"})
    d1 = d1[(d1.n_h1 >= MIN_GIO_NGAY) & d1.rv5.notna()].reset_index(drop=True)
    return d1, xau, hs["rv"], int(m_ol.sum())


def main():
    hom_nay = dt.datetime.utcnow()
    print("=" * 100)
    print(f"TẢI DỮ LIỆU HIỆN HÀNH — {hom_nay:%Y-%m-%d %H:%M} UTC")
    print("=" * 100)
    print("nguồn: Yahoo (thanh GIỜ tự gộp lên ngày — KHÔNG dùng nến ngày của họ,")
    print("       37,1% số ngày ở đó có close==open và high<open) · đối chứng: FRED\n")
    bc = {"chay_luc": f"{hom_nay:%Y-%m-%dT%H:%M:%SZ}", "nguon": "yahoo 1h→D1 + 5m→rv5",
          "cap": {}}

    print(f"{'cặp':<9}{'ngày':>7}{'đến':>12}{'nến xấu':>9}{'rv5 thật':>10}"
          f"{'h1→m5':>8}{'lệch vs HistData (pip)':>30}")
    print("-" * 100)
    for p in PAIRS:
        d1, xau, he_so, n_ol = mot_cap(p)
        d1.to_csv(os.path.join(OUT, f"{p}_d1.csv"), index=False)
        mn = do_moi_noi(p, d1)
        fr = doi_chung_fred(p, d1)
        n_that = int((d1.rv_uoc == 0).sum())
        bc["cap"][p] = dict(n_ngay=int(len(d1)), den=str(d1.Date.max().date()),
                            nen_khong_hop_le=xau, n_rv5_that=n_that,
                            he_so_h1_sang_m5=round(he_so, 4), n_chong_lan=n_ol,
                            moi_noi=mn, fred=fr)
        s = (f"trung vị {mn['trungvi_pip']} · p95 {mn['p95_pip']} · r={mn['tuong_quan']}"
             if mn else "—")
        print(f"{p:<9}{len(d1):>7}{str(d1.Date.max().date()):>12}{xau:>9}"
              f"{n_that:>10}{he_so:>8.3f}{s:>30}")
        time.sleep(0.4)

    with open(os.path.join(OUT, "moi_noi.json"), "w", encoding="utf-8") as f:
        json.dump(bc, f, ensure_ascii=False, indent=1)
    print("-" * 100)
    tv = [v["moi_noi"]["trungvi_pip"] for v in bc["cap"].values() if v["moi_noi"]]
    print(f"\nMỐI NỐI HAI NHÀ CUNG CẤP: lệch trung vị {min(tv):.2f}–{max(tv):.2f} pip.")
    print("Phải công bố trong luận văn. Đối chiếu: kiểm tra chéo tick-vs-tick cũ của")
    print("repo cho 0,350 pip — con số ở đây lớn hơn vì Yahoo là báo giá chỉ dẫn,")
    print("không phải dữ liệu tick. Chuỗi < 2026-01-01 là HistData, từ đó là Yahoo.")
    print(f"\nđã ghi {OUT}")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
