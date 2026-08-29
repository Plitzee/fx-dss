"""MO HINH CHI PHI GIAO DICH — thay hang so 0,91 pip.

Do tu 103.504 gio-ngay tick that (HistData, bid/ask rieng), 8 thoi ky mau
2010-2025, 6 cap tien. Ba tang cau truc, theo dung thu tu quan trong:

  1. CHE DO  — spread FX nen lai mot lan giua 2014 va 2016 roi phang.
               Trung binh 6 cap: 3,50 -> 2,45 -> 0,70 pip. Khong phai xu huong
               tuyen tinh, nen dung 2 che do thay vi noi suy 16 nam.
               Luu y: so truoc 2015 gan nhu chac chan la spread CO DINH do nha
               moi gioi niem yet (toan so tron 3,00 / 4,00; R^2 voi bien dong
               chi 0,01-0,10), khong phai spread thi truong. Voi mot he thong
               danh cho nha dau tu ca nhan thi do van la chi phi phai tra that.

  2. GIO     — gio 21 UTC (chot ngay giao dich) dat gap 2-4 lan gio thuong,
               va so tick tut 2-6 lan. Luat cho tang 6: khong mo vi the luc do.

  3. BIEN DONG — spread gian theo bien dong, nhung YEU: do co gian 0,03-0,43,
               R^2 0,00-0,33. Dung de hieu chinh, khong dung lam tru cot.

QUAN TRONG NHAT — DUNG TRUNG VI CHO CHI PHI THONG THUONG, DUNG DUOI CHO THOAT
BUOC. Thang 3/2020: trung vi chi tang 2,1-3,3 lan, nhung p95 tang 19-115 lan
(EUR/USD ngay 09/03: trung vi 0,51 pip, p95 33,91 pip — gap 67 lan).
Khi lenh dung lo bi kich hoat giua khung hoang, thu ban tra la DUOI, khong phai
trung vi. Mo hinh chi dung trung vi se noi "COVID lam chi phi tang gap doi" —
vua sai vua nguy hiem vi nghe rat yen tam.

Dung:
    from cost import spread_pip, roundtrip_pip
    spread_pip("EURUSD", hour=9, date="2024-06-03")            -> 0.20
    spread_pip("EURUSD", hour=21, date="2024-06-03")           -> 0.90
    spread_pip("EURUSD", hour=9, date="2020-03-09", q="p95")   -> chi phi thoat buoc
    roundtrip_pip("EURUSD", hour=9, date="2024-06-03", commission_pip=0.35)
"""
import json, os
import numpy as np
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)


def _find(name):
    """Tim trong data/ truoc (bo cuc repo), roi thu thu muc cha (bo cuc sandbox)."""
    for c in (os.path.join(ROOT, "data", name), os.path.join(ROOT, name),
              os.path.join(DIR, name)):
        if os.path.exists(c):
            return c
    return os.path.join(ROOT, "data", name)      # de thong bao loi chi dung cho


TABLE = _find("cost_table.csv")
ELAST = _find("cost_elasticity.json")

# hoa hong khu hoi mac dinh — GIA DINH cho tai khoan ECN, CHUA kiem chung.
# Phai chay phan tich do nhay 0,0 / 0,35 / 0,70 truoc khi dua vao ket luan.
COMMISSION_PIP = 0.35
REGIME_SPLIT = 2015

_tab = _el = None


def _load():
    global _tab, _el
    if _tab is None:
        _tab = pd.read_csv(TABLE).set_index(["regime", "pair", "hour"]).sort_index()
        _el = json.load(open(ELAST))
    return _tab, _el


def regime_of(date):
    return "pre2015" if pd.Timestamp(date).year < REGIME_SPLIT else "post2015"


def spread_pip(pair, hour, date, q="med", sigma=None):
    """Spread ky vong, don vi pip.

    q      : "med" chi phi thong thuong | "p95" thoat buoc | "p99" kich ban xau
    sigma  : do lech chuan ngay du bao (don vi nhu sqrt(rv_m5)*1e4). Neu co,
             hieu chinh theo do co gian da uoc luong. Bo qua = dung muc nen.
    """
    tab, el = _load()
    key = (regime_of(date), pair.upper(), int(hour))
    if key not in tab.index:
        raise KeyError(f"khong co {key} trong bang chi phi")
    row = tab.loc[key]
    base = float(row[{"med": "spread_med", "p95": "spread_p95",
                      "p99": "spread_p99"}[q]])
    if sigma is None or pair.upper() not in el:
        return base
    e = el[pair.upper()]
    beta = e["beta_p95"] if q in ("p95", "p99") else e["beta_med"]
    return base * (max(sigma, 1e-9) / e["sig_ref"]) ** beta


def roundtrip_pip(pair, hour, date, q="med", sigma=None,
                  commission_pip=COMMISSION_PIP):
    """Tong chi phi khu hoi = spread + hoa hong. Day moi la con so de so voi
    nguong hoa von cua tin hieu."""
    return spread_pip(pair, hour, date, q, sigma) + commission_pip


def hour_profile(pair, date, q="med"):
    """Duong cong 24 gio — dung cho phieu quyet dinh o tang 6."""
    return pd.Series({h: spread_pip(pair, h, date, q) for h in range(24)},
                     name=f"{pair} {regime_of(date)} {q}")


def cheapest_hours(pair, date, k=6):
    """k gio re nhat trong ngay, theo gio UTC."""
    return hour_profile(pair, date).nsmallest(k).index.tolist()


if __name__ == "__main__":
    print("TU KIEM")
    a = spread_pip("EURUSD", 9, "2024-06-03")
    b = spread_pip("EURUSD", 21, "2024-06-03")
    c = spread_pip("EURUSD", 9, "2012-06-04")
    d = spread_pip("EURUSD", 9, "2020-03-09", q="p95")
    print(f"  EURUSD 09h 2024      {a:.2f} pip   (ky vong ~0,2-0,3)")
    print(f"  EURUSD 21h 2024      {b:.2f} pip   (phu phi chot ngay, phai > 09h)")
    print(f"  EURUSD 09h 2012      {c:.2f} pip   (che do cu, phai >> 2024)")
    print(f"  EURUSD 09h p95       {d:.2f} pip   (duoi, phai > trung vi)")
    assert b > a, "gio 21 phai dat hon gio 9"
    assert c > a * 3, "che do truoc 2015 phai dat hon nhieu"
    assert d > a, "duoi phai rong hon trung vi"
    print(f"  6 gio re nhat 2024:  {cheapest_hours('EURUSD','2024-06-03')}")
    print(f"  Khu hoi co hoa hong: {roundtrip_pip('EURUSD',9,'2024-06-03'):.2f} pip")
    print("  DAT")
