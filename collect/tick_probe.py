#!/usr/bin/env python3
"""DO XEM HISTDATA CO PHAT HANH TICK KEM BID/ASK KHONG.

Vi sao quan trong: nen M1 cua HistData KHONG co bid/ask (cot volume luon 0).
Tang 5 dang dung hang so 0,91 pip cho moi gio trong ngay, trong khi ket luan
quan trong nhat cua tang 2b ("tin hieu nam duoi nguong chi phi") treo hoan
toan vao con so do.

Neu HistData co tick kem bid/ask thi khong can quay lai Dukascopy.
Chi can 1-2 nam x 2-3 cap la du dung duong cong spread theo gio.

Chay:  py tick_probe.py
"""
import io, re, urllib.parse, urllib.request, zipfile

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research"}
PAGE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/{p}/{y}/{m}"
POST = "https://www.histdata.com/get.php"


def get(url, ref=None, data=None, timeout=90):
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


def probe(pair, year, month):
    ref = PAGE.format(p=pair.lower(), y=year, m=month)
    print(f"\n  {pair} {year}-{month:02d}")
    print(f"    trang: {ref}")
    try:
        html = get(ref).decode("utf-8", "ignore")
    except Exception as e:
        print(f"    KHONG MO DUOC TRANG: {str(e)[:90]}")
        return False
    m = re.search(r'id="tk"\s+value="([^"]+)"', html) or \
        re.search(r'name="tk"\s+value="([^"]+)"', html)
    if not m:
        print("    Trang khong co token tk -> HistData khong phat hanh tick cho muc nay")
        return False
    tk = m.group(1)
    print(f"    token OK")

    for tf in ("T", "tick", "T_ASCII"):
        body = {"tk": tk, "date": str(year), "datemonth": f"{year}{month:02d}",
                "platform": "ASCII", "timeframe": tf, "fxpair": pair.upper()}
        try:
            raw = get(POST, ref=ref, data=body)
        except Exception as e:
            print(f"    timeframe={tf:<8} loi: {str(e)[:60]}")
            continue
        if raw[:2] != b"PK":
            print(f"    timeframe={tf:<8} khong phai zip ({len(raw)}B)")
            continue
        z = zipfile.ZipFile(io.BytesIO(raw))
        name = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))][0]
        head = z.read(name)[:400].decode("ascii", "ignore").splitlines()[:4]
        print(f"    timeframe={tf:<8} OK  zip {len(raw):,}B  -> {name}")
        for ln in head:
            print(f"        {ln}")
        ncol = len(head[0].split(",")) if "," in head[0] else len(head[0].split(";"))
        print(f"    -> {ncol} cot")
        if ncol >= 3:
            print("    ==> CO BID/ASK. Khong can quay lai Dukascopy.")
        else:
            print("    ==> Chi co 1 chuoi gia, KHONG co bid/ask.")
        return True
    print("    Khong bien the timeframe nao tra ve zip.")
    return False


print("=" * 78)
print("DO TICK DATA HISTDATA — tim spread that")
print("=" * 78)
ok = probe("EURUSD", 2024, 6)
if not ok:
    probe("EURUSD", 2020, 6)
print("\n" + "=" * 78)
print("Gui nguyen man hinh nay cho Claude.")
