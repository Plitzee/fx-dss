"""TAI NEN M1 DUKASCOPY -> nen NGAY + nam do luong noi ngay, RV5 THAT moi phien.

VAN DE NO GIAI. Nguon hien hanh (Yahoo, `collect/live_fx.py`) chi phuc vu thanh
5 phut trong ~60 ngay. Ngoai cua so do rv5 phai UOC tu thanh gio (`rv_uoc=1`),
va `api/cache.py::kiem_rv_that` phai dung rao chan de he thong khong am tham an
so uoc. Dukascopy khong co vach do: nen M1 co tu 2003, MOT file cho MOT ngay.

DA DO 15/09/2026 (EURUSD 2026-09-08, 84 nen M5 chung, 08-15 UTC):
    RV5 tu tick Dukascopy    1,6651e-06   (moc — su that)
    RV5 tu nen M1 Dukascopy  1,6651e-06   lech  0,0%
    RV5 tu EODHD             1,8374e-06   lech +10,3%
Nen M1 khop TUYET DOI voi tick vi no duoc dung tu chinh tick do. Tai lap:
`python collect/probe_nguon.py`.

CHI PHI. Mot request cho mot ngay mot cap (1.440 nen), so voi 24 request neu
lay tick. Van hanh hang ngay: 6 request. Dap lai lich su thi dat hon nhieu —
xem "HAN MUC" duoi day — nhung chi phai lam MOT LAN vi co bo nho dem.

HAN MUC. Dukascopy cho khoang 15-20 request roi siet (quan sat ghi trong
../dukas/dukas_v3.py; xac nhan 15/09/2026: ban 20 lan lien tuc -> 9/20, tre
trung vi 14,2s). Nen o day ban TUNG DOT roi NGHI, va tu noi dot nghi khi ty le
thanh cong tut. Dung bo phan nghi di de chay cho nhanh — se cham hon, khong
phai nhanh hon.

BO NHO DEM. Moi file .bi5 tai ve duoc luu nguyen ven o `data/dukas_cache/`.
Chay lai KHONG goi mang cho nhung ngay da co. Day la thu lam cho viec dap lich
su kha thi: dut mang giua chung thi chay lai, khong mat gi.

KHONG GHI DE SAN XUAT. Ket qua ra `data/live/{PAIR}_d1_dukas.csv`, KHONG phai
`{PAIR}_d1.csv`. Doi nguon du lieu san xuat la viec phai DO truoc roi quyet —
dung `--doi-chung` de so voi nguon dang chay.

Chay:  python collect/dukas_m1.py --tu-kiem
       python collect/dukas_m1.py --tu 2026-06-01 --den 2026-09-11
       python collect/dukas_m1.py --tu 2026-06-01 --den 2026-09-11 --doi-chung
Ghi:   data/live/{PAIR}_d1_dukas.csv · data/dukas_cache/{PAIR}/*.bi5
"""
import argparse
import datetime as dt
import lzma
import os
import struct
import sys
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")
LIVE = os.path.join(D, "live")
CACHE = os.path.join(D, "dukas_cache")

UA = {"User-Agent": "Mozilla/5.0 (compatible; fx-dss-thesis/1.0)"}
BASE = "https://datafeed.dukascopy.com/datafeed"
REC = struct.Struct(">IIIIIf")        # t(giay tu 00:00 UTC), o, c, l, h, volume
PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF")
EPS = 1e-14

DOT = 8            # so request moi dot truoc khi nghi
NGHI = 60          # giay nghi giua hai dot (tu noi khi bi siet)
NGHI_MAX = 600
KHE = 1.5          # giay giua hai request trong cung mot dot
LAN = 4            # so lan thu lai mot URL truoc khi bo qua
MIN_M1_NGAY = 600  # duoi nguong nay coi la phien cut (nua phien / ngay le)

# KHOA_SO muc 2 danh rieng 2026-01..08 cua CA 12 cap cho bo niem phong.
NIEM_PHONG = (dt.date(2026, 1, 1), dt.date(2026, 8, 31))


def diem(pair):
    return 1e-3 if "JPY" in pair.upper() else 1e-5


def _duong_cache(pair, ngay):
    return os.path.join(CACHE, pair, f"{ngay:%Y-%m-%d}.bi5")


def lay_ngay(pair, ngay, phien=None):
    """Tra ve raw .bi5 cho mot ngay. Uu tien bo nho dem; b'' = khong co du lieu
    (cuoi tuan/ngay nghi); None = khong lay duoc."""
    f = _duong_cache(pair, ngay)
    if os.path.exists(f):
        with open(f, "rb") as fh:
            return fh.read(), True
    u = (f"{BASE}/{pair}/{ngay.year}/{ngay.month-1:02d}/{ngay.day:02d}"
         f"/BID_candles_min_1.bi5")
    s = phien or requests
    for _ in range(LAN):
        try:
            r = s.get(u, headers=UA, timeout=30)
            if r.status_code in (200, 404):
                b = r.content if r.status_code == 200 else b""
                os.makedirs(os.path.dirname(f), exist_ok=True)
                with open(f, "wb") as fh:
                    fh.write(b)
                return b, False
        except requests.RequestException:
            pass
        time.sleep(4.0)
    return None, False


def giai(raw, pair, ngay):
    """raw .bi5 -> DataFrame nen M1 (ts, open, high, low, close). Bo nen khong
    co giao dich (volume = 0) — do la phut thi truong dong hoac khong co quote."""
    if not raw:
        return None
    buf = lzma.LZMADecompressor().decompress(raw)
    P = diem(pair)
    n = len(buf) // REC.size
    ra = []
    goc = dt.datetime.combine(ngay, dt.time(0))
    for i in range(n):
        t, o, c, l, h, v = REC.unpack_from(buf, i * REC.size)
        if v <= 0:
            continue
        ra.append((goc + dt.timedelta(seconds=int(t)), o * P, h * P, l * P, c * P))
    if not ra:
        return None
    return pd.DataFrame(ra, columns=["ts", "open", "high", "low", "close"])


def gop_m5(m1):
    """M1 -> M5, dung nhan 5 phut lam moc (giong khung 5m cua live_fx)."""
    g = m1.set_index("ts").resample("5min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"))
    return g.dropna().reset_index()


def do_luong(m5):
    """NAM do luong noi ngay ma `volfc2.du_bao_san_xuat` doi hoi.

    Cong thuc GIU NGUYEN cua `live_fx.do_luong_noi_ngay` — chep sang day thay vi
    import de file nay khong phu thuoc mot script thu thap khac, nhung bat ky
    thay doi nao o mot ben deu phai doi ben kia (co test giu dong bo).

      rv  = sum r^2 · rq = n/3 sum r^4 · bpv = pi/2 sum|r_i||r_{i-1}|
      rsp = sum r^2 1{r>0} · rsn = sum r^2 1{r<0}   (rsp + rsn = rv)
    Loi suat tinh TRONG NGAY, bo gap qua dem — dung quy uoc collect/rv5.py.
    """
    c = m5.close.values
    if len(c) < 3:
        return None
    r = np.diff(np.log(np.maximum(c, EPS)))
    n = len(r)
    a = np.abs(r)
    return dict(rv5=float(np.sum(r ** 2)),
                rq5=float(n / 3.0 * np.sum(r ** 4)),
                bpv5=float(np.pi / 2.0 * np.sum(a[1:] * a[:-1])),
                rsp=float(np.sum(r[r > 0] ** 2)),
                rsn=float(np.sum(r[r < 0] ** 2)),
                n5=int(n))


def nen_ngay(m1):
    """Nen NGAY tu M1. Tra ve (dict, so_nen_khong_hop_le)."""
    o, h = float(m1.open.iloc[0]), float(m1.high.max())
    l, c = float(m1.low.min()), float(m1.close.iloc[-1])
    xau = int(h < max(o, c) - 1e-9 or l > min(o, c) + 1e-9)
    return dict(open=o, high=h, low=l, close=c, n_h1=int(len(m1) // 60)), xau


def mot_ngay(pair, ngay, phien):
    raw, tu_dem = lay_ngay(pair, ngay, phien)
    if raw is None:
        return None, "khong tai duoc", tu_dem
    m1 = giai(raw, pair, ngay)
    if m1 is None or len(m1) < MIN_M1_NGAY:
        return None, ("cuoi tuan/ngay nghi" if m1 is None else
                      f"phien cut ({len(m1)} nen M1)"), tu_dem
    m5 = gop_m5(m1)
    dl = do_luong(m5)
    if dl is None:
        return None, "khong du nen M5", tu_dem
    d1, xau = nen_ngay(m1)
    return dict(Date=pd.Timestamp(ngay), **d1, **dl, rv_uoc=0, nen_xau=xau), None, tu_dem


def tu_kiem():
    """Kiem cac phep BIEN DOI, khong cham mang."""
    print("TỰ KIỂM (không gọi mạng)")
    dat = True
    # do_luong khop nghiem giai tich tren chuoi biet truoc
    m5 = pd.DataFrame({"close": [1.0, 1.01, 1.0, 1.02]})
    r = np.diff(np.log(m5.close.values))
    d = do_luong(m5)
    ok = (abs(d["rv5"] - float(np.sum(r ** 2))) < 1e-18
          and abs(d["rsp"] + d["rsn"] - d["rv5"]) < 1e-18)
    dat &= ok
    print(f"  rv5 đúng công thức và rsp+rsn=rv5   {'ĐẠT' if ok else 'HỎNG'}")
    # cong thuc phai TRUNG live_fx — nguon su that duy nhat cho 5 do luong nay
    sys.path.insert(0, HERE)
    import live_fx as LF                                  # noqa: E402
    f = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=4, freq="5min"),
                      "close": [1.0, 1.01, 1.0, 1.02]})
    ref = LF.do_luong_noi_ngay(f, "5").iloc[0]
    ok = (abs(ref["rv5"] - d["rv5"]) < 1e-18
          and abs(ref["rq5"] - d["rq5"]) < 1e-18
          and abs(ref["bpv5"] - d["bpv5"]) < 1e-18
          and abs(ref["rsp5"] - d["rsp"]) < 1e-18
          and abs(ref["rsn5"] - d["rsn"]) < 1e-18)
    dat &= ok
    print(f"  khớp từng số với live_fx            {'ĐẠT' if ok else 'HỎNG'}")
    # giai() bo nen volume 0 va doi point dung cho JPY
    goc = dt.date(2024, 4, 8)
    buf = b"".join(REC.pack(t * 60, 110_000 + t, 110_000 + t, 110_000 + t,
                            110_000 + t, 0.0 if t == 0 else 1.0) for t in range(3))
    raw = lzma.compress(buf, format=lzma.FORMAT_ALONE)
    g = giai(raw, "EURUSD", goc)
    ok = g is not None and len(g) == 2 and abs(g.close.iloc[0] - 1.10001) < 1e-9
    dat &= ok
    print(f"  giải nén: bỏ volume=0, point đúng   {'ĐẠT' if ok else 'HỎNG'}")
    gj = giai(raw, "USDJPY", goc)
    ok = gj is not None and abs(gj.close.iloc[0] - 110.001) < 1e-6
    dat &= ok
    print(f"  point JPY = 1e-3                    {'ĐẠT' if ok else 'HỎNG'}")
    # nen ngay: high/low phai bao duoc open/close
    m1 = pd.DataFrame({"ts": pd.date_range("2024-04-08", periods=3, freq="1min"),
                       "open": [1.1, 1.2, 1.15], "high": [1.25, 1.25, 1.2],
                       "low": [1.05, 1.1, 1.1], "close": [1.2, 1.15, 1.18]})
    nd, xau = nen_ngay(m1)
    ok = (nd["open"] == 1.1 and nd["close"] == 1.18
          and nd["high"] == 1.25 and nd["low"] == 1.05 and xau == 0)
    dat &= ok
    print(f"  nến ngày lấy đúng O/H/L/C           {'ĐẠT' if ok else 'HỎNG'}")
    print(f"→ TỰ KIỂM {'ĐẠT' if dat else 'HỎNG'}")
    return dat


def doi_chung(pair, moi):
    """So nen ngay moi voi nguon dang chay (`{PAIR}_d1.csv`, Yahoo)."""
    f = os.path.join(LIVE, f"{pair}_d1.csv")
    if not os.path.exists(f) or moi.empty:
        return None
    cu = pd.read_csv(f, parse_dates=["Date"])
    m = pd.merge(cu[["Date", "close", "rv5", "rv_uoc"]],
                 moi[["Date", "close", "rv5"]], on="Date",
                 suffixes=("_yahoo", "_dukas"))
    if len(m) < 10:
        return None
    pip = 1e-2 if "JPY" in pair else 1e-4
    lech = (m.close_dukas - m.close_yahoo).abs() / pip
    that = m[m.rv_uoc == 0]
    ty = (that.rv5_dukas / that.rv5_yahoo.replace(0, np.nan)).dropna()
    ty_u = None
    uoc = m[m.rv_uoc == 1]
    if len(uoc) >= 10:
        t2 = (uoc.rv5_dukas / uoc.rv5_yahoo.replace(0, np.nan)).dropna()
        ty_u = float(t2.median()) if len(t2) else None
    return dict(n=len(m), lech_trungvi_pip=float(lech.median()),
                lech_p95_pip=float(lech.quantile(.95)),
                lech_max_pip=float(lech.max()),
                rv5_ty_le_khi_yahoo_that=(float(ty.median()) if len(ty) else None),
                n_yahoo_that=int(len(that)),
                rv5_ty_le_khi_yahoo_uoc=ty_u, n_yahoo_uoc=int(len(uoc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default=",".join(PAIRS))
    ap.add_argument("--tu", default=None, help="YYYY-MM-DD (mặc định: 120 ngày trước)")
    ap.add_argument("--den", default=None, help="YYYY-MM-DD (mặc định: hôm qua)")
    ap.add_argument("--dot", type=int, default=DOT)
    ap.add_argument("--nghi", type=int, default=NGHI)
    ap.add_argument("--doi-chung", action="store_true")
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()

    if a.tu_kiem:
        sys.exit(0 if tu_kiem() else 1)
    if not tu_kiem():
        sys.exit("tự kiểm HỎNG — dừng, không tải dữ liệu")

    den = dt.date.fromisoformat(a.den) if a.den else dt.date.today() - dt.timedelta(days=1)
    tu = dt.date.fromisoformat(a.tu) if a.tu else den - dt.timedelta(days=120)
    caps = [x.strip().upper() for x in a.cap.split(",") if x.strip()]
    ngays = [tu + dt.timedelta(days=i) for i in range((den - tu).days + 1)]
    ngays = [d for d in ngays if d.weekday() < 5]

    print("\n" + "=" * 92)
    print("TẢI NẾN M1 DUKASCOPY → nến ngày + RV5 THẬT")
    print(f"{tu} → {den} · {len(ngays)} ngày trong tuần × {len(caps)} cặp "
          f"= {len(ngays)*len(caps)} ngày-cặp")
    print(f"đợt {a.dot} request rồi nghỉ {a.nghi}s · bộ nhớ đệm: data/dukas_cache/")
    print("=" * 92)

    trong_np = [d for d in ngays if NIEM_PHONG[0] <= d <= NIEM_PHONG[1]]
    if trong_np:
        print(f"\n  ⚠ {len(trong_np)} ngày nằm trong khoảng KHOA_SO mục 2 dành cho bộ")
        print(f"    niêm phong ({NIEM_PHONG[0]} → {NIEM_PHONG[1]}). Tải về để PHỤC VỤ")
        print("    sản xuất là việc đã làm (nguồn hiện hành cũng phủ khoảng này), nhưng")
        print("    KHÔNG được dùng chúng để chọn mô hình. Xem api/routers/meta.py::canh_bao.\n")

    phien = requests.Session()
    tong = {}
    for p in caps:
        hang, bo, loi = [], 0, 0
        nghi, dem_dot, ok_dot, goi_that = a.nghi, 0, 0, 0
        t0 = time.time()
        for i, ng in enumerate(ngays):
            r, vi_sao, tu_dem = mot_ngay(p, ng, phien)
            if r is not None:
                hang.append(r)
            elif vi_sao == "khong tai duoc":
                loi += 1
            else:
                bo += 1
            if tu_dem:
                continue                      # doc tu dem: khong ton han muc
            goi_that += 1
            ok_dot += int(vi_sao != "khong tai duoc")
            dem_dot += 1
            time.sleep(KHE)
            if dem_dot >= a.dot and i < len(ngays) - 1:
                ty = ok_dot / max(dem_dot, 1)
                nghi = (min(NGHI_MAX, int(nghi * 1.6)) if ty < 0.5
                        else max(20, int(nghi * 0.8)) if ty > 0.9 else nghi)
                con = len(ngays) - i - 1
                print(f"  {p}: {i+1}/{len(ngays)} ngày · {len(hang)} phiên · "
                      f"đợt {100*ty:.0f}% · nghỉ {nghi}s · còn ~{con}")
                time.sleep(nghi)
                dem_dot = ok_dot = 0
        if not hang:
            print(f"  {p}: KHÔNG lấy được phiên nào")
            continue
        d = pd.DataFrame(hang).sort_values("Date").reset_index(drop=True)
        xau = int(d.nen_xau.sum())
        d = d.drop(columns=["nen_xau"])
        cot = ["Date", "open", "high", "low", "close", "rv5", "rq5", "bpv5",
               "rsp", "rsn", "n5", "rv_uoc", "n_h1"]
        d = d[[c for c in cot if c in d]]
        os.makedirs(LIVE, exist_ok=True)
        f = os.path.join(LIVE, f"{p}_d1_dukas.csv")
        d.to_csv(f, index=False)
        tong[p] = dict(n=len(d), bo=bo, loi=loi, nen_xau=xau,
                       den=str(d.Date.max().date()), goi_that=goi_that,
                       giay=round(time.time() - t0, 1))
        print(f"  {p}: {len(d)} phiên · bỏ {bo} · lỗi {loi} · nến xấu {xau} "
              f"· {goi_that} request thật · {time.time()-t0:.0f}s → {os.path.basename(f)}")

    print("\n" + "=" * 92)
    print(f"{'cặp':<9}{'phiên':>7}{'nến xấu':>9}{'rv_uoc=1':>10}{'đến':>13}")
    for p, v in tong.items():
        print(f"{p:<9}{v['n']:>7}{v['nen_xau']:>9}{0:>10}{v['den']:>13}")
    print("\nMọi phiên đều rv_uoc=0 — RV5 dựng từ nến M1 thật, không ước từ thanh giờ.")

    if a.doi_chung:
        print("\n" + "=" * 92)
        print("ĐỐI CHỨNG với nguồn đang chạy (Yahoo, {PAIR}_d1.csv)")
        print(f"{'cặp':<9}{'n':>6}{'|lệch| giá đóng (pip)':>26}{'rv5 Dukas/Yahoo':>20}")
        print(f"{'':<9}{'':>6}{'trung vị / p95 / max':>26}{'Yahoo thật / Yahoo ước':>24}")
        for p in caps:
            f = os.path.join(LIVE, f"{p}_d1_dukas.csv")
            if not os.path.exists(f):
                continue
            dc = doi_chung(p, pd.read_csv(f, parse_dates=["Date"]))
            if not dc:
                print(f"{p:<9} (không đủ ngày chung để so)")
                continue
            tu_ = f"{dc['rv5_ty_le_khi_yahoo_that']:.3f}" if dc["rv5_ty_le_khi_yahoo_that"] else "—"
            tu2 = f"{dc['rv5_ty_le_khi_yahoo_uoc']:.3f}" if dc["rv5_ty_le_khi_yahoo_uoc"] else "—"
            print(f"{p:<9}{dc['n']:>6}{dc['lech_trungvi_pip']:>10.2f}"
                  f"{dc['lech_p95_pip']:>8.2f}{dc['lech_max_pip']:>8.2f}"
                  f"{tu_:>13} ({dc['n_yahoo_that']}){tu2:>8} ({dc['n_yahoo_uoc']})")
        print("\n  Cột cuối là tỷ lệ rv5 Dukascopy/Yahoo, tách theo việc Yahoo dùng RV5")
        print("  THẬT hay RV ƯỚC ở ngày đó. Tỷ lệ ở nhóm 'Yahoo ước' lệch xa 1 hơn là")
        print("  điều cần kiểm — đó chính là phần rào chắn rv_uoc đang bảo vệ.")


if __name__ == "__main__":
    main()
