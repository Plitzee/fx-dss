"""LICH SU KIEN VI MO — mo rong tu 7 ngan hang trung uong sang ca cong bo vi mo.

VAN DE DO DUOC. `src/sukien_profile.py` do: ngay hop NHTW gia dong manh hon nen
1,08-1,45 lan, ma sigma^ chi du bao duoc 1,02-1,23 lan — HUT 11-20%. Va he thong
hien chi biet 7 NHTW, 901 ngay. Nhung cong bo tac dong cao nhat theo moi lich
kinh te — NFP, CPI, GDP, ban le, PCE — thi KHONG BIET GI CA.

NFP mot minh la ~190 ngay tac dong cao moi 16 nam. Noi vao se va dung cho
sigma^ dang hut, tuc cai thien TRUC BIEN DONG — truc duy nhat da chung minh co
thong tin (BSS +0,0152 ngoai mau o h=1).

NGUON. FRED `releases/dates` cua Fed St. Louis: chinh thuc, mien phi, co lich su
day du tu nhung nam 1950. Day la NGAY CONG BO THAT, khong phai suy ra tu quy tac
"thu Sau dau thang" — quy tac do sai vai lan moi nam va ngay sai se LAM LOANG
phan ung do duoc, khien mo hinh te di chu khong tot len.

CAN MOT KHOA. Dang ky mien phi 30 giay tai
    https://fredaccount.stlouisfed.org/apikey
roi dat vao bien moi truong (KHONG dat vao ma nguon, khong commit):
    Windows :  setx FRED_API_KEY "khoa_cua_ban"
    bash    :  export FRED_API_KEY=khoa_cua_ban
    Actions :  Settings -> Secrets -> FRED_API_KEY

KHONG LAY DUOC GI TU FRED: consensus forecast. FRED chi co ngay cong bo va gia
tri thuc. Muon do "bat ngo" (|thuc - du bao|) thi phai co nguon khac co tra phi.
Nhung ngay cong bo da du de va phan sigma^ hut, va do la muc tieu cua pha nay.

Chay:  python collect/lich_su_kien.py
       python collect/lich_su_kien.py --liet-ke     (xem cac release co san)
Ghi:   data/su_kien.csv   (date, ma, ten, tien_te, nguon)
"""
import os
import sys
import time

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
API = "https://api.stlouisfed.org/fred"
KHOA = os.environ.get("FRED_API_KEY", "").strip()
TU_NGAY = "2009-06-01"          # som hon panel mot chut de co dam

# Cac cong bo TAC DONG CAO. Khop theo TEN vi so hieu release cua FRED co the doi;
# khop ten thi on dinh hon va tu kiem duoc bang --liet-ke.
MUC_TIEU = [
    ("NFP",     "Employment Situation",                 "USD"),
    ("CPI",     "Consumer Price Index",                 "USD"),
    ("GDP",     "Gross Domestic Product",               "USD"),
    ("BANLE",   "Advance Monthly Sales for Retail",     "USD"),
    ("PCE",     "Personal Income and Outlays",          "USD"),
    ("PPI",     "Producer Price Index",                 "USD"),
    ("JOLTS",   "Job Openings and Labor Turnover",      "USD"),
    ("ISM",     "ISM Manufacturing",                    "USD"),
]


def goi(duong, **tham):
    if not KHOA:
        raise SystemExit(
            "\nTHIEU FRED_API_KEY.\n"
            "  1. Lay khoa mien phi: https://fredaccount.stlouisfed.org/apikey\n"
            '  2. Windows:  setx FRED_API_KEY "khoa_cua_ban"  (mo lai terminal)\n'
            "     bash   :  export FRED_API_KEY=khoa_cua_ban\n"
            "  3. Chay lai lenh nay.\n"
            "KHONG dat khoa vao ma nguon va khong commit no.")
    tham.update(api_key=KHOA, file_type="json")
    for lan in range(4):
        try:
            r = requests.get(f"{API}/{duong}", params=tham, timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 + 3 * lan)
                continue
            raise SystemExit(f"FRED tra ve {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            if lan == 3:
                raise SystemExit(f"khong goi duoc FRED: {e}")
            time.sleep(2 + 3 * lan)


def cac_release():
    ra, off = [], 0
    while True:
        j = goi("releases", limit=1000, offset=off)
        r = j.get("releases", [])
        ra += r
        if len(r) < 1000:
            break
        off += 1000
    return ra


def main():
    if "--liet-ke" in sys.argv:
        for r in sorted(cac_release(), key=lambda x: x["name"]):
            print(f"  {r['id']:>6}  {r['name']}")
        return

    print("=" * 84)
    print("LICH SU KIEN VI MO — FRED releases/dates")
    print("=" * 84)
    rs = cac_release()
    print(f"{len(rs):,} release tren FRED\n")

    hang = []
    print(f"{'ma':<8}{'release khop':<46}{'so ngay':>9}{'tu':>12}")
    for ma, mau, tien in MUC_TIEU:
        khop = [r for r in rs if mau.lower() in r["name"].lower()]
        if not khop:
            print(f"{ma:<8}{'(khong tim thay: ' + mau + ')':<46}{'—':>9}")
            continue
        r = min(khop, key=lambda x: len(x["name"]))     # ten ngan nhat = ban chinh
        j = goi("release/dates", release_id=r["id"], limit=10000,
                include_release_dates_with_no_data="false")
        ngay = [d["date"] for d in j.get("release_dates", []) if d["date"] >= TU_NGAY]
        for n in ngay:
            hang.append(dict(date=n, ma=ma, ten=r["name"], tien_te=tien, nguon="FRED"))
        print(f"{ma:<8}{r['name'][:44]:<46}{len(ngay):>9,}{(min(ngay) if ngay else '—'):>12}")

    # gop voi lich NHTW da co — giu nguyen, chi noi them
    cb = pd.read_csv(os.path.join(D, "cb_dates.csv"))
    cb["date"] = cb.date.astype(str).str[:10]
    TIEN_CB = {"FOMC": "USD", "ECB": "EUR", "BOE": "GBP", "BOJ": "JPY",
               "BOC": "CAD", "RBA": "AUD", "SNB": "CHF"}
    cb_h = [dict(date=r.date, ma=r.bank, ten=f"{r.bank} rate decision",
                 tien_te=TIEN_CB.get(r.bank, ""), nguon="cb_dates")
            for r in cb.itertuples()]

    df = pd.DataFrame(cb_h + hang).drop_duplicates(["date", "ma"]).sort_values("date")
    f = os.path.join(D, "su_kien.csv")
    df.to_csv(f, index=False)
    print(f"\n{len(df):,} dong -> data/su_kien.csv  "
          f"({df.ma.nunique()} loai, {df.date.min()} -> {df.date.max()})")
    print("\nBUOC KE TIEP — BAT BUOC:")
    print("  python src/sukien_profile.py      do phan ung THAT cua tung loai moi")
    print("  Khong duoc gan nhan 'tac dong cao' cho NFP/CPI truoc khi do. Giao dien")
    print("  nay xep muc do bang TY LE DA DO, khong bang quy uoc.")
    print("TU KIEM DAT")


if __name__ == "__main__":
    main()
