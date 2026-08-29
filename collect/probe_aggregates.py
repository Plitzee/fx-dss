#!/usr/bin/env python3
"""DO CAC MUC DO GOP CUA DUKASCOPY — CHAY TREN MAY BAN (PowerShell).

Muc dich: tim xem Dukascopy co cho tai nen GIO (1 file/thang) va nen NGAY
(1 file/nam) khong. Neu co, thoi gian tai giam tu ~65 gio xuong ~3 gio.

Chay:  py probe_aggregates.py
"""
import urllib.request, lzma, struct, time

BASE = "https://datafeed.dukascopy.com/datafeed"
UA = {"User-Agent": "Mozilla/5.0 (research data download)"}
REC = struct.Struct(">IIIIIf")   # cung dinh dang voi nen M1 da giai ma duoc
SYM = "EURUSD"
PT = 1e-5

# (nhan, url, so ban ghi KY VONG, don vi thoi gian moi ban ghi)
# Thang danh so tu 0 -> thang 5/2013 la "04"
CANDIDATES = [
    ("M1  1 file/NGAY   (dang dung)", f"{BASE}/{SYM}/2013/04/15/BID_candles_min_1.bi5",   1440, 60),
    ("M5  1 file/NGAY",               f"{BASE}/{SYM}/2013/04/15/BID_candles_min_5.bi5",    288, 300),
    ("M15 1 file/NGAY",               f"{BASE}/{SYM}/2013/04/15/BID_candles_min_15.bi5",    96, 900),
    ("H1  1 file/THANG  <== quan trong", f"{BASE}/{SYM}/2013/04/BID_candles_hour_1.bi5",   744, 3600),
    ("H1  1 file/THANG (bien the)",   f"{BASE}/{SYM}/2013/04/BID_candles_hour_1.bi5?",     744, 3600),
    ("H4  1 file/THANG",              f"{BASE}/{SYM}/2013/04/BID_candles_hour_4.bi5",      186, 14400),
    ("D1  1 file/NAM    <== quan trong", f"{BASE}/{SYM}/2013/BID_candles_day_1.bi5",       365, 86400),
    ("W1  1 file/NAM",                f"{BASE}/{SYM}/2013/BID_candles_week_1.bi5",          52, 604800),
]


def fetch(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def decode(raw):
    if not raw:
        return b""
    try:
        return lzma.decompress(raw)
    except lzma.LZMAError:
        return lzma.LZMADecompressor(lzma.FORMAT_AUTO).decompress(raw)


print("=" * 86)
print(f"DO CAC MUC DO GOP — {SYM}, thang 5/2013")
print("=" * 86)

working = []
for label, url, expect, unit_s in CANDIDATES:
    t0 = time.time()
    try:
        raw = fetch(url)
        buf = decode(raw)
        n = len(buf) // REC.size
        dt = time.time() - t0
        if n == 0:
            print(f"  {label:<34} TRONG (0 ban ghi)")
            continue
        print(f"  {label:<34} OK  {n:>5} ban ghi  "
              f"(ky vong ~{expect})  {len(raw):>7,}B nen  {dt:4.1f}s")
        working.append((label, url, n, unit_s))
    except Exception as e:
        msg = str(e)[:40]
        print(f"  {label:<34} -- {msg}")

if not working:
    print("\nKhong tai duoc muc gop nao. Van phai dung M1 nhu hien tai.")
    raise SystemExit(0)

print("\n" + "=" * 86)
print("KIEM CHUNG NOI DUNG (gia phai hop ly, thoi gian phai tang deu)")
print("=" * 86)
for label, url, n, unit_s in working:
    try:
        buf = decode(fetch(url))
        print(f"\n  {label}  ({n} ban ghi):")
        offs = []
        for i in (0, 1, 2, n - 1):
            if i >= n:
                continue
            t, o, c, l, h, v = REC.unpack_from(buf, i * REC.size)
            offs.append(t)
            print(f"     #{i:<4} t={t:>9}s  O={o*PT:.5f} H={h*PT:.5f} "
                  f"L={l*PT:.5f} C={c*PT:.5f} vol={v:>10,.0f}")
        if len(offs) >= 2:
            step = offs[1] - offs[0]
            ok = "DUNG" if abs(step - unit_s) < unit_s * 0.5 else f"LA (ky vong {unit_s}s)"
            print(f"     -> buoc thoi gian = {step}s  [{ok}]")
    except Exception as e:
        print(f"  {label}: loi khi giai ma ({e})")

print("\n" + "=" * 86)
print("KET LUAN")
print("=" * 86)
has_h1 = any("H1" in w[0] for w in working)
has_d1 = any("D1" in w[0] for w in working)
if has_h1:
    print("  H1 theo THANG hoat dong -> 12 cap x 15 nam chi con ~4.320 request")
    print("     (thay vi 93.600). Du dung cho rv_h1 vi code dang dung thanh GIO.")
if has_d1:
    print("  D1 theo NAM hoat dong  -> lay OHLC ngay cho Parkinson/GK/RS")
    print("     chi mat ~360 request cho toan bo 12 cap x 15 nam.")
if has_h1 and has_d1:
    print("\n  => PHUONG AN TOI UU: D1(nam) cho OHLC ngay + H1(thang) cho muc tieu rv.")
    print("     Tong ~4.680 request ~ 3 gio, thay vi ~65 gio nhu hien tai.")
if not (has_h1 or has_d1):
    print("  Chi co cac muc theo NGAY. Van nen doi M1 -> M5 hoac M15:")
    print("  cung so request nhung file nho hon nhieu, tai nhanh hon.")
print("\nGui ket qua nay cho Claude de viet lenh tai moi.")
