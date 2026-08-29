#!/usr/bin/env python3
"""HOAN THIEN DATASET — ba manh con thieu.

  PHA 1  LAI SUAT (FRED)      -> fred_rates.csv, carry.csv
         Chenh lech lai suat = tin hieu FX duy nhat co bang chung ben vung
         trong tai lieu. Tang 2b dang de trong muc nay.

  PHA 2  KHOI LUONG (Dukascopy D1 theo nam)  -> dukas_volume.csv
         HistData ghi volume = 0 nen hien khong co bien dai dien thanh khoan.
         96 request, file nho.

  PHA 3  SPREAD THAT (Dukascopy H1, BID va ASK rieng)  -> spread_hourly.csv
         Tang 5 dang dung hang so 0,91 pip cho MOI gio. Ket luan quan trong
         nhat cua tang 2b treo vao con so do.
         Lay 2 nam (2019 va 2024) x 6 cap = 288 request. Du de dung duong
         cong spread theo gio; khong can 16 nam.

Chay:
    py finish_dataset.py              # ca ba pha
    py finish_dataset.py --phase 1    # rieng tung pha
    py finish_dataset.py --phase 3 --years 2024
"""
import argparse, io, lzma, os, struct, sys, time, urllib.request
from datetime import datetime, timedelta, timezone

UA = {"User-Agent": "Mozilla/5.0 (research data download)"}
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
DUKAS = "https://datafeed.dukascopy.com/datafeed"
REC = struct.Struct(">IIIIIf")
POINT = {"JPY": 1e-3, "DEFAULT": 1e-5}

# FRED: nhieu ma du phong cho moi dong tien (OECD da ngung mot so chuoi)
FRED = {
    "USD": ["IR3TIB01USM156N", "DGS3MO", "IRSTCI01USM156N"],
    "EUR": ["IR3TIB01EZM156N", "IRSTCI01EZM156N", "ECBDFR"],
    "GBP": ["IR3TIB01GBM156N", "IRSTCI01GBM156N"],
    "JPY": ["IR3TIB01JPM156N", "IRSTCI01JPM156N"],
    "AUD": ["IR3TIB01AUM156N", "IRSTCI01AUM156N"],
    "CAD": ["IR3TIB01CAM156N", "IRSTCI01CAM156N"],
    "CHF": ["IR3TIB01CHM156N", "IRSTCI01CHM156N"],
    "NZD": ["IR3TIB01NZM156N", "IRSTCI01NZM156N"],
}
# cap tien -> (dong yet gia co so, dong dinh gia).  EURUSD: 1 EUR = x USD
QUOTE = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "AUDUSD": ("AUD", "USD"),
         "USDJPY": ("USD", "JPY"), "USDCAD": ("USD", "CAD"), "USDCHF": ("USD", "CHF"),
         "NZDUSD": ("NZD", "USD")}


def point_of(sym):
    return POINT["JPY"] if "JPY" in sym.upper() else POINT["DEFAULT"]


def fetch(url, attempts=6, timeout=45):
    """Lui bac thang: 3s, 9s, 27s, 60s, 60s. Dukascopy siet nhip thi cho lau hon."""
    last = None
    for i in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            if i < attempts - 1:
                time.sleep(min(60, 3.0 * (3 ** i)))
    raise last


def decode(raw):
    if not raw:
        return b""
    try:
        return lzma.decompress(raw)
    except lzma.LZMAError:
        return lzma.LZMADecompressor(lzma.FORMAT_AUTO).decompress(raw)


def parse(raw, sym):
    buf = decode(raw); pt = point_of(sym); out = {}
    for i in range(len(buf) // REC.size):
        t, o, c, l, h, v = REC.unpack_from(buf, i * REC.size)
        out[t] = (o * pt, c * pt, l * pt, h * pt, v)
    return out


# ───────────────────────── PHA 1 — LAI SUAT ─────────────────────────
def phase1():
    import pandas as pd
    print("=" * 78); print("PHA 1 — LAI SUAT NGAN HAN TU FRED"); print("=" * 78)
    got = {}
    for cur, codes in FRED.items():
        for code in codes:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
            try:
                raw = fetch(url, attempts=2, timeout=30)
                df = pd.read_csv(io.BytesIO(raw))
                df.columns = ["DATE", "rate"]
                df["DATE"] = pd.to_datetime(df.DATE, errors="coerce")
                df["rate"] = pd.to_numeric(df.rate, errors="coerce")
                df = df.dropna()
                if len(df) < 100:
                    print(f"  {cur} {code:<18} qua ngan ({len(df)} dong), thu ma khac")
                    continue
                got[cur] = df.assign(cur=cur, code=code)
                print(f"  {cur} {code:<18} OK  {len(df):,} dong  "
                      f"{df.DATE.min().date()} -> {df.DATE.max().date()}")
                break
            except Exception as e:
                print(f"  {cur} {code:<18} -- {str(e)[:50]}")
    if not got:
        print("\n  Khong lay duoc chuoi nao."); return
    allr = pd.concat(got.values(), ignore_index=True)
    allr.to_csv("fred_rates.csv", index=False)
    print(f"\n  Ghi fred_rates.csv ({len(allr):,} dong, {len(got)} dong tien)")

    # carry = lai suat dong CO SO tru lai suat dong DINH GIA
    print("\n  CHENH LECH LAI SUAT (carry) theo cap:")
    wide = allr.pivot_table(index="DATE", columns="cur", values="rate")
    wide = wide.sort_index().ffill()
    rows = []
    for pair, (base, quote) in QUOTE.items():
        if base in wide.columns and quote in wide.columns:
            d = (wide[base] - wide[quote]).dropna()
            rows.append(pd.DataFrame({"DATE": d.index, "pair": pair, "carry": d.values}))
            print(f"    {pair}: {len(d):,} quan sat | trung binh {d.mean():+.2f}%/nam "
                  f"| khoang [{d.min():+.2f}, {d.max():+.2f}]")
    if rows:
        c = pd.concat(rows, ignore_index=True)
        c.to_csv("carry.csv", index=False)
        print(f"\n  Ghi carry.csv ({len(c):,} dong)")


# ───────────────────── PHA 2 — KHOI LUONG NGAY ─────────────────────
CACHE = "dukas_cache"

def phase2(y0=2010, y1=2025):
    """GHI NGAY TUNG NAM vao cache. Ngat giua chung khong mat gi; chay lai
    chi tai phan con thieu."""
    import csv, glob
    os.makedirs(CACHE, exist_ok=True)
    print("\n" + "=" * 78)
    print("PHA 2 — KHOI LUONG NGAY (Dukascopy D1 theo nam) — CO CACHE, CHAY LAI DUOC")
    print("=" * 78)
    jobs = [(s_, y) for s_ in PAIRS for y in range(y0, y1 + 1)]
    have = {os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/*.csv")}
    todo = [(s_, y) for s_, y in jobs if f"{s_}_{y}" not in have]
    print(f"  {len(jobs)} nam-cap | da co {len(jobs)-len(todo)} | can tai {len(todo)}")
    nfail = 0
    t0 = time.time()
    for i, (sym, year) in enumerate(todo, 1):
        try:
            rec = parse(fetch(f"{DUKAS}/{sym}/{year}/BID_candles_day_1.bi5"), sym)
        except Exception as e:
            print(f"  [{i:3}/{len(todo)}] {sym} {year}  LOI {str(e)[:40]}", flush=True)
            nfail += 1
            continue
        base = datetime(year, 1, 1, tzinfo=timezone.utc)
        rows = [dict(Date=(base + timedelta(seconds=t)).strftime("%Y-%m-%d"),
                     pair=sym, n_ticks=round(v, 1))
                for t, (o, c, l, h, v) in sorted(rec.items()) if v > 0]
        with open(f"{CACHE}/{sym}_{year}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Date", "pair", "n_ticks"])
            w.writeheader(); w.writerows(rows)
        print(f"  [{i:3}/{len(todo)}] {sym} {year}  {len(rows):>4} ngay  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        time.sleep(0.5)

    files = sorted(glob.glob(f"{CACHE}/*.csv"))
    out = []
    for f in files:
        with open(f) as fh:
            r = list(csv.DictReader(fh))
            out += r
    if out:
        out.sort(key=lambda r: (r["pair"], r["Date"]))
        with open("dukas_volume.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Date", "pair", "n_ticks"])
            w.writeheader(); w.writerows(out)
        print(f"\n  Ghi dukas_volume.csv ({len(out):,} dong tu {len(files)} nam-cap)")
    if nfail:
        print(f"  {nfail} nam loi — chay lai lenh nay, no chi tai phan con thieu")


# ───────────────────── PHA 3 — SPREAD THEO GIO ─────────────────────
def phase3(years=(2019, 2024)):
    import csv
    print("\n" + "=" * 78)
    print(f"PHA 3 — SPREAD THAT (Dukascopy H1 BID va ASK), nam {', '.join(map(str,years))}")
    print("=" * 78)
    out, nfail = [], 0
    t0 = time.time()
    for sym in PAIRS:
        pt = point_of(sym); n = 0
        for year in years:
            for month in range(1, 13):
                u = f"{DUKAS}/{sym}/{year}/{month-1:02d}"
                try:
                    bid = parse(fetch(f"{u}/BID_candles_hour_1.bi5"), sym)
                    ask = parse(fetch(f"{u}/ASK_candles_hour_1.bi5"), sym)
                except Exception as e:
                    print(f"  ! {sym} {year}-{month:02d}: {str(e)[:40]}", flush=True)
                    nfail += 1; continue
                base = datetime(year, month, 1, tzinfo=timezone.utc)
                for t in sorted(set(bid) & set(ask)):
                    bo, bc, bl, bh, bv = bid[t]
                    ao, ac, al, ah, av = ask[t]
                    if bv <= 0 and av <= 0:
                        continue
                    ts = base + timedelta(seconds=t)
                    out.append(dict(Date=ts.strftime("%Y-%m-%d"), hour=ts.hour, pair=sym,
                                    spread_pip=round((ac - bc) / pt / 10, 3),
                                    volume=round((bv + av) / 2, 1)))
                    n += 1
                time.sleep(0.3)
            print(f"  {sym} {year}: {n:,} thanh gio  [{time.time()-t0:.0f}s]", flush=True)
    if not out:
        print("\n  Khong lay duoc gi."); return
    with open("spread_hourly.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Date", "hour", "pair", "spread_pip", "volume"])
        w.writeheader(); w.writerows(out)
    print(f"\n  Ghi spread_hourly.csv ({len(out):,} dong, {nfail} thang loi)")

    try:
        import pandas as pd
        d = pd.DataFrame(out)
        print("\n  SPREAD TRUNG VI THEO GIO UTC (pip) — thay cho hang so 0,91")
        pv = d.pivot_table(index="hour", columns="pair", values="spread_pip", aggfunc="median")
        print(pv.round(2).to_string())
        print(f"\n  Trung vi chung: {d.spread_pip.median():.2f} pip | "
              f"p95 {d.spread_pip.quantile(.95):.2f} | p99 {d.spread_pip.quantile(.99):.2f}")
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="123")
    ap.add_argument("--years", default="2019,2024")
    a = ap.parse_args()
    ys = tuple(int(x) for x in a.years.split(","))
    if "1" in a.phase: phase1()
    if "2" in a.phase: phase2()
    if "3" in a.phase: phase3(ys)
    print("\n" + "=" * 78)
    print("Gui nguyen man hinh nay cho Claude.")
