#!/usr/bin/env python3
"""DOT TAI CUOI — bo sung 6 cap CHEO (2010-2025) + 2026 YTD cho toan bo 12 cap.

Vi sao can:
  1. Sau cap dang co DEU co USD mot ve -> khong doc lap voi nhau. Ket luan chi
     phat bieu duoc cho "cac cap chinh doi USD". Sau cap cheo go bo han che nay.
  2. 2026 la du lieu NAM NGOAI moi thu da chay -> tap khoa so that su.

Hai pha:
  PHA A  cap cheo, tai theo NAM   (2010-2025)   -> {PAIR}_{YYYY}.csv
  PHA B  toan bo cap, 2026 theo THANG           -> {PAIR}_{YYYY}-{MM}.csv
         (HistData chi cho tai theo thang doi voi nam hien hanh)

Co resume: file da co thi bo qua. Ngat giua chung roi chay lai duoc.

Chay:
    py histdata_dl2.py --probe          # chi do, khong tai
    py histdata_dl2.py                  # tai ca hai pha
    py histdata_dl2.py --phase a        # chi cap cheo
    py histdata_dl2.py --phase b        # chi 2026
"""
import argparse, io, os, re, sys, time, zipfile
import urllib.request, urllib.parse

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research"}
PAGE_Y = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{p}/{y}"
PAGE_M = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{p}/{y}/{m}"
POST   = "https://www.histdata.com/get.php"

CROSS = ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCHF", "NZDUSD"]
USDP  = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
# QUAN TRONG: thu muc RIENG, KHONG phai histdata_raw.
# prep_fx.py tu quet histdata_raw/ -> neu de chung, du lieu khoa so se chay
# thang vao pipeline ngay hom nay va bien ban khoa so mat tac dung.
OUT   = "histdata_seal"


def get(url, ref=None, data=None, timeout=90):
    h = dict(UA)
    if ref:
        h["Referer"] = ref
    body = None
    if data is not None:
        body = urllib.parse.urlencode(data).encode()
        h["Content-Type"] = "application/x-www-form-urlencoded"
        h["Origin"] = "https://www.histdata.com"
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def token_of(html):
    m = re.search(r'id="tk"\s+value="([^"]+)"', html)
    if not m:
        m = re.search(r'name="tk"\s+value="([^"]+)"', html)
    if not m:
        raise RuntimeError("khong thay token tk tren trang")
    return m.group(1)


def unzip_one(raw):
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise RuntimeError("zip khong co file csv")
    return z.read(names[0])


def fetch(pair, year, month=None, verbose=False):
    """month=None -> tai ca nam.  month=1..12 -> tai 1 thang."""
    if month is None:
        ref = PAGE_Y.format(p=pair.lower(), y=year)
        variants = [{"date": str(year), "datemonth": str(year)}]
    else:
        ref = PAGE_M.format(p=pair.lower(), y=year, m=month)
        # HistData khong nhat quan giua cac ban; thu ca hai kieu roi kiem chung
        variants = [{"date": str(year), "datemonth": f"{year}{month:02d}"},
                    {"date": str(year), "datemonth": f"{year}{month}"}]

    html = get(ref).decode("utf-8", "ignore")
    tk = token_of(html)

    last = None
    for v in variants:
        body = dict(v, tk=tk, platform="ASCII", timeframe="M1", fxpair=pair.upper())
        try:
            raw = get(POST, ref=ref, data=body)
        except Exception as e:
            last = e
            continue
        if raw[:2] == b"PK":
            if verbose:
                print(f"      datemonth={body['datemonth']} -> OK {len(raw):,}B")
            return unzip_one(raw)
        last = RuntimeError(f"khong phai zip (datemonth={body['datemonth']}, "
                            f"{len(raw)}B, dau={raw[:60]!r})")
    raise last


def probe():
    print("=" * 78)
    print("DO TRUOC KHI TAI")
    print("=" * 78)
    ok_a = ok_b = False

    print("\n  PHA A — cap cheo tai theo nam:  EURGBP 2020")
    try:
        csv = fetch("EURGBP", 2020, verbose=True)
        n = csv.count(b"\n")
        print(f"    OK  {len(csv):,} byte, {n:,} dong")
        print(f"    Dong dau: {csv.split(chr(10).encode())[0][:60].decode('ascii','ignore')}")
        ok_a = True
    except Exception as e:
        print(f"    THAT BAI: {str(e)[:160]}")

    print("\n  PHA B — nam hien hanh tai theo thang:  EURUSD 2026-01")
    try:
        csv = fetch("EURUSD", 2026, 1, verbose=True)
        n = csv.count(b"\n")
        print(f"    OK  {len(csv):,} byte, {n:,} dong")
        print(f"    Dong dau: {csv.split(chr(10).encode())[0][:60].decode('ascii','ignore')}")
        ok_b = True
    except Exception as e:
        print(f"    THAT BAI: {str(e)[:160]}")

    print("\n" + "=" * 78)
    if ok_a and ok_b:
        print("  CA HAI PHA CHAY DUOC — bo --probe di de tai that.")
    elif ok_a:
        print("  Chi PHA A chay duoc. 2026 co the chua mo. Chay: py histdata_dl2.py --phase a")
    elif ok_b:
        print("  Chi PHA B chay duoc. Gui ket qua nay cho Claude.")
    else:
        print("  KHONG PHA NAO CHAY. Gui nguyen man hinh nay cho Claude.")
    return ok_a, ok_b


def save(pair, year, month, verbose, force=False):
    tag = f"{year}" if month is None else f"{year}-{month:02d}"
    path = os.path.join(OUT, f"{pair}_{tag}.csv")
    if not force and os.path.exists(path) and os.path.getsize(path) > 50_000:
        return "co san", 0
    t0 = time.time()
    csv = fetch(pair, year, month, verbose)
    with open(path, "wb") as f:
        f.write(csv)
    return f"{len(csv)/1e6:.1f}MB", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--phase", default="ab", choices=["a", "b", "ab"])
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--months", type=int, default=8, help="so thang cua nam hien hanh")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--pairs", help="chi tai cac cap nay, vi du AUDJPY")
    ap.add_argument("--years", help="chi tai cac nam nay, vi du 2012 hoac 2012,2013")
    ap.add_argument("--force", action="store_true", help="tai lai ke ca khi file da co")
    ap.add_argument("--monthly", action="store_true",
                    help="tai theo THANG cho cac nam trong --years (dung khi file nam bi thieu)")
    a = ap.parse_args()

    if a.probe:
        probe()
        return

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    nok = nskip = nfail = 0
    fails = []

    jobs = []
    only_p = [x.strip().upper() for x in a.pairs.split(",")] if a.pairs else None
    only_y = [int(x) for x in a.years.split(",")] if a.years else None

    if only_p and only_y:
        for p in only_p:
            for y in only_y:
                if a.monthly:
                    jobs += [(p, y, m) for m in range(1, 13)]
                else:
                    jobs.append((p, y, None))
        print("=" * 78)
        print(f"TAI LAI {len(jobs)} file  -> {os.path.abspath(OUT)}")
        print("=" * 78)
        run(jobs, a)
        return

    if "a" in a.phase:
        for p in CROSS:
            for y in range(2010, 2026):
                jobs.append((p, y, None))
    if "b" in a.phase:
        for p in USDP + CROSS:
            for m in range(1, a.months + 1):
                jobs.append((p, a.year, m))

    print("=" * 78)
    print(f"TAI {len(jobs)} file  (pha {a.phase.upper()})  -> {os.path.abspath(OUT)}")
    print("=" * 78)
    run(jobs, a)


def run(jobs, a):
    t0 = time.time()
    nok = nskip = nfail = 0
    fails = []
    for i, (p, y, m) in enumerate(jobs, 1):
        tag = f"{y}" if m is None else f"{y}-{m:02d}"
        try:
            msg, dt = save(p, y, m, a.verbose, a.force)
            if msg == "co san":
                nskip += 1
            else:
                nok += 1
            print(f"  [{i:3}/{len(jobs)}] {p} {tag:<8} {msg:>8}  {dt:4.1f}s", flush=True)
        except Exception as e:
            nfail += 1
            fails.append((p, tag, str(e)[:90]))
            print(f"  [{i:3}/{len(jobs)}] {p} {tag:<8}   LOI  {str(e)[:70]}", flush=True)
        time.sleep(0.4)

    print("\n" + "=" * 78)
    print(f"XONG: {nok} tai moi | {nskip} da co san | {nfail} loi | "
          f"{time.time()-t0:.0f}s")
    if fails:
        print("\n  Cac file loi (chay lai lenh nay de tai bu, no tu bo qua file da co):")
        for p, tag, e in fails[:20]:
            print(f"    {p} {tag}: {e}")
    print(f"\n  Tong so file trong {OUT}: {len(os.listdir(OUT))}")
    print("\n  " + "!"*70)
    print("  DAY LA TAP KHOA SO. KHONG chay prep_fx.py tren thu muc nay bay gio.")
    print("  Xem KHOA_SO.md. Chi mo ra o lan chay cuoi, bang lenh:")
    print("      py prep_fx.py --src histdata_seal --out fx_seal")
    print("  " + "!"*70)
    print("\n  Gui nguyen man hinh nay cho Claude.")


if __name__ == "__main__":
    main()
