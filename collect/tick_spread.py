#!/usr/bin/env python3
"""SPREAD THAT TU TICK HISTDATA — thay hang so 0,91 pip.

Nguon: HistData tick quotes. Da xac nhan co BID va ASK rieng:
    20240602 170007220,1.084620,1.085410,0
    ^ngay gio (co mili giay)  ^bid      ^ask   ^volume (luon 0)

Cung nha cung cap voi nen M1 da dung, cung mui gio (New York, co doi gio he),
va vua chay 192/192 file khong loi. Khong dung Dukascopy cho phan nay.

Lam ba viec:
  1. Tai tick theo thang (co resume — file da co thi bo qua)
  2. Doc theo dong, dung HISTOGRAM 0,01 pip nen khong ngon bo nho
  3. Gop thanh spread theo (ngay, gio UTC) cho tung cap -> spread_hourly.csv

Chay:
    py tick_spread.py --probe                 # xem se tai bao nhieu file
    py tick_spread.py                         # 6 cap x 2024, 72 file ~600MB
    py tick_spread.py --years 2019,2024       # them mot nam de so che do
    py tick_spread.py --months 2,6,10         # chi 3 thang/nam cho nhe
    py tick_spread.py --skip-download         # chi xu ly file da tai
"""
import argparse, csv, io, os, re, sys, time, urllib.parse, urllib.request, zipfile
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research"}
PAGE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/{p}/{y}/{m}"
POST = "https://www.histdata.com/get.php"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
RAW = "histdata_tick"
CACHE = "tick_cache"
OUT = "spread_hourly.csv"
NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

NBIN = 5000          # 0,00 -> 50,00 pip, buoc 0,01 pip
STEP = 0.01


def pip_mult(sym):
    return 100.0 if "JPY" in sym.upper() else 10000.0


def get(url, ref=None, data=None, timeout=180):
    h = dict(UA)
    if ref:
        h["Referer"] = ref
    body = None
    if data is not None:
        body = urllib.parse.urlencode(data).encode()
        h["Content-Type"] = "application/x-www-form-urlencoded"
        h["Origin"] = "https://www.histdata.com"
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=h),
                                timeout=timeout) as r:
        return r.read()


def download(pair, year, month):
    path = os.path.join(RAW, f"{pair}_{year}-{month:02d}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path, "co san"
    ref = PAGE.format(p=pair.lower(), y=year, m=month)
    html = get(ref).decode("utf-8", "ignore")
    m = re.search(r'id="tk"\s+value="([^"]+)"', html) or \
        re.search(r'name="tk"\s+value="([^"]+)"', html)
    if not m:
        raise RuntimeError("khong thay token tk")
    body = {"tk": m.group(1), "date": str(year), "datemonth": f"{year}{month:02d}",
            "platform": "ASCII", "timeframe": "T", "fxpair": pair.upper()}
    raw = get(POST, ref=ref, data=body)
    if raw[:2] != b"PK":
        raise RuntimeError(f"khong phai zip ({len(raw)}B)")
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    with open(path, "wb") as f:
        f.write(z.read(name))
    return path, f"{os.path.getsize(path)/1e6:.0f}MB"


def aggregate(path, pair):
    """Doc theo dong. Gom histogram spread theo (ngay-gio dia phuong)."""
    mult = pip_mult(pair)
    hist = {}                      # (YYYYMMDD, HH) -> [dem theo bin]
    bad = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            try:
                ts, bid, ask, _ = line.split(",", 3)
                b = float(bid); a = float(ask)
            except Exception:
                bad += 1
                continue
            sp = (a - b) * mult
            if sp < 0 or sp != sp:
                bad += 1
                continue
            k = (ts[:8], ts[9:11])
            h = hist.get(k)
            if h is None:
                h = hist[k] = [0] * (NBIN + 1)
            i = int(sp / STEP)
            h[i if i < NBIN else NBIN] += 1
    return hist, bad


def quantile(h, q):
    n = sum(h)
    if n == 0:
        return None
    tgt = q * n
    c = 0
    for i, v in enumerate(h):
        c += v
        if c >= tgt:
            return round(i * STEP + STEP / 2, 3)
    return round(NBIN * STEP, 3)


def local_hour_to_utc(daystr, hourstr):
    dt = datetime(int(daystr[:4]), int(daystr[4:6]), int(daystr[6:8]), int(hourstr))
    try:
        return dt.replace(tzinfo=NY).astimezone(UTC)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--years", default="2024")
    ap.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()

    pairs = [p.strip().upper() for p in a.pairs.split(",")]
    years = [int(y) for y in a.years.split(",")]
    months = [int(m) for m in a.months.split(",")]
    jobs = [(p, y, m) for p in pairs for y in years for m in months]

    os.makedirs(RAW, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    print("=" * 78)
    print(f"SPREAD THAT TU TICK — {len(jobs)} file "
          f"({len(pairs)} cap x {len(years)} nam x {len(months)} thang)")
    print(f"Uoc luong ~8MB/file -> ~{len(jobs)*8/1000:.1f} GB")
    print("=" * 78)
    if a.probe:
        for p, y, m in jobs[:5]:
            print(f"  {p} {y}-{m:02d}")
        print(f"  ... tong {len(jobs)} file"); return

    t0 = time.time(); nfail = 0
    rows = []
    for i, (pair, year, month) in enumerate(jobs, 1):
        tag = f"{pair} {year}-{month:02d}"
        cache = os.path.join(CACHE, f"{pair}_{year}-{month:02d}.csv")
        if os.path.exists(cache):
            with open(cache) as f:
                rows += list(csv.DictReader(f))
            print(f"  [{i:3}/{len(jobs)}] {tag}  cache", flush=True)
            continue
        try:
            if not a.skip_download:
                path, msg = download(pair, year, month)
            else:
                path = os.path.join(RAW, f"{pair}_{year}-{month:02d}.csv")
                msg = "co san"
                if not os.path.exists(path):
                    raise FileNotFoundError("chua tai")
            hist, bad = aggregate(path, pair)
        except Exception as e:
            print(f"  [{i:3}/{len(jobs)}] {tag}  LOI {str(e)[:50]}", flush=True)
            nfail += 1
            continue

        out = []
        for (d, hh), h in sorted(hist.items()):
            u = local_hour_to_utc(d, hh)
            if u is None:
                continue
            out.append(dict(Date=u.strftime("%Y-%m-%d"), hour=u.hour, pair=pair,
                            n_ticks=sum(h),
                            spread_med=quantile(h, .5),
                            spread_p95=quantile(h, .95)))
        with open(cache, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Date", "hour", "pair", "n_ticks",
                                              "spread_med", "spread_p95"])
            w.writeheader(); w.writerows(out)
        rows += out
        nt = sum(r["n_ticks"] for r in out)
        print(f"  [{i:3}/{len(jobs)}] {tag}  {msg:>7}  {nt:>10,} tick  "
              f"{len(out):>4} gio  [{time.time()-t0:.0f}s]", flush=True)

    if not rows:
        print("\n  Khong co du lieu."); return
    rows.sort(key=lambda r: (r["pair"], r["Date"], int(r["hour"])))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Date", "hour", "pair", "n_ticks",
                                          "spread_med", "spread_p95"])
        w.writeheader(); w.writerows(rows)
    print(f"\n  Ghi {OUT} ({len(rows):,} dong, {nfail} file loi)")

    # bang tong ket
    try:
        import statistics as stt
        by = {}
        for r in rows:
            by.setdefault((r["pair"], int(r["hour"])), []).append(float(r["spread_med"]))
        print("\n  SPREAD TRUNG VI THEO GIO UTC (pip) — thay cho hang so 0,91")
        print("  gio " + "".join(f"{p:>9}" for p in pairs))
        for hh in range(24):
            line = f"  {hh:>3} "
            for p in pairs:
                v = by.get((p, hh))
                line += f"{stt.median(v):>9.2f}" if v else f"{'—':>9}"
            print(line)
        allv = [float(r["spread_med"]) for r in rows]
        print(f"\n  Trung vi chung {stt.median(allv):.2f} pip | "
              f"gio re nhat {min(range(24), key=lambda h: stt.median(by.get((pairs[0],h),[99])))} UTC | "
              f"gio dat nhat {max(range(24), key=lambda h: stt.median(by.get((pairs[0],h),[0])))} UTC")
    except Exception as e:
        print(f"  (khong dung duoc bang tong ket: {e})")
    print("\n  Gui nguyen man hinh nay cho Claude.")


if __name__ == "__main__":
    main()
