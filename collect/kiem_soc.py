"""KIEM SOC GIUA PHIEN — phat hien khi gia thuc te da di xa hon gia thiet cua
du bao HOM NAY, de tai tinh SOM HON lich cron thuong (hien 4 lan/ngay, ~6h/lan).

QUAN TRONG VE PHAM VI. Day KHONG PHAI mot mo hinh du bao moi trong phien —
toan bo du an da chung minh ky nang chi ton tai o tam NGAY (ha pheu xuong H1
van 0 quy luat song sot). Script nay chi lam MOT viec: so sanh gia HIEN TAI
voi gia da dung de tinh du bao hom nay, quy ra don vi sigma^ (z = log(p/p0)/sig,
dung dung quy uoc da dung khap noi trong repo — xem rui_ro_gap.py, api/risk_logic.py),
va neu |z| vuot nguong thi kich hoat lai DUNG pipeline san xuat hien co
(jobs/cap_nhat.py) SOM HON, khong doi cron 6 tieng.

NGUON GIA HIEN TAI: TrueFX (webrates.truefx.com) — bao gia bid/ask THOI GIAN
THUC, moc mili-giay, KHONG can dang ky/khoa. Da do truc tiep 14/09/2026: goi
duoc CA 6 cap trong MOT lan HTTP, 5 lan goi lien tiep deu HTTP 200 khong bi
chan, khong thay gioi han ro rang (khac Twelve Data — 8 credit/phut, 800/ngay,
tung buoc ep phai kiem thua moi 15 phut). KHONG dung Yahoo interval=1m — da do
98,7% thanh Yahoo M1 la anh chup gia, khong phai OHLC that, xem collect/live_fx.py.

Dinh dang CSV TrueFX (9 cot, vi du that da do):
    EUR/USD,1789395502988,1.15,361,1.15,363,1.15229,1.16041,1.15991
    ma,timestamp_ms,bid_phan_nguyen,bid_pip,ask_phan_nguyen,ask_pip,high,low,open
Gia = NOI CHUOI truc tiep phan_nguyen + pip (vi du "1.15"+"361"="1.15361") —
cach nay dung DONG NHAT ca cap thuong (5 chu so) lan cap JPY (vi du
"154."+"750"="154.750", da kiem chung).

CAN API DANG CHAY (giong jobs/cap_nhat.py) de lay gia_moc + sigma^ da tinh
san — KHONG goi lai collect/live_fx.py o day (do la buoc nang, dung Yahoo,
danh cho pipeline day du).

Chay:   python collect/kiem_soc.py
Doc:    FXDSS_API (mac dinh http://127.0.0.1:8899)
Ghi:    output/kiem_soc_trangthai.json
Thoat:  0 = khong soc (hoac bo qua vi loi tam thoi cua TrueFX/API)
        42 = CO SOC — workflow goi nen tai tinh toan bo ngay
"""
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
API = os.environ.get("FXDSS_API", "http://127.0.0.1:8899")
UA = {"User-Agent": "Mozilla/5.0 (compatible; fx-dss-thesis/1.0)"}

PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF")
TF_MA = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
         "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "USDCHF": "USD/CHF"}
NGUONG_SOC = 1.5          # cung don vi k*sigma^ da dung cho stop-loss khap noi
EPS = 1e-14


def gia_va_sigma(pair):
    """Goi API DANG CHAY de lay gia THAM CHIEU va sigma^ da dung cho du bao
    hom nay — dung DUNG so API dang phuc vu, khong tinh lai bang cong thuc
    khac (nguyen tac "mot nguon su that duy nhat").

    Dung /forecast (~0,05s SAU KHI cache am), KHONG dung /risk (~20-60s/cap —
    chay het backtest VaR Kupiec/Christoffersen/DQ moi lan goi, qua nang cho
    mot phep kiem chay moi 15 phut x 6 cap). Nhung lan goi DAU TIEN cho mot
    cap tren mot tien trinh API MOI (nhu workflow nay se luon la, moi lan
    chay mot container moi) van phai am cache lay(pair) — da do ~60s cho lan
    dau — nen timeout phai du rong, khong duoc dat theo toc do luc da am."""
    r = requests.get(f"{API}/forecast", params={"pair": pair, "h": 1}, timeout=90)
    r.raise_for_status()
    d = r.json()
    ps = {"USDJPY": 0.01}.get(pair, 0.0001)
    return float(d["gia"]), float(d["sigma_1_pip"]) * ps, str(d["ngay"])


def gia_hien_tai_truefx(thu=3):
    """Goi MOT LAN duy nhat lay ca 6 cap tu TrueFX — tra ve dict
    {pair: (gia_giua, luc_truefx)}. Gia giua = (bid+ask)/2."""
    u = "https://webrates.truefx.com/rates/connect.html"
    ma_nguoc = {v: k for k, v in TF_MA.items()}
    for k in range(thu):
        try:
            r = requests.get(u, params={"f": "csv", "c": ",".join(TF_MA.values())},
                             headers=UA, timeout=12)
            r.raise_for_status()
            ra = {}
            for dong in r.text.strip().splitlines():
                c = dong.strip().split(",")
                if len(c) != 9 or c[0] not in ma_nguoc:
                    continue
                bid = float(c[2] + c[3])
                ask = float(c[4] + c[5])
                luc_ms = int(c[1])
                luc = dt.datetime.utcfromtimestamp(luc_ms / 1000.0)
                ra[ma_nguoc[c[0]]] = ((bid + ask) / 2.0, luc.isoformat() + "Z")
            thieu = set(PAIRS) - set(ra)
            if thieu:
                raise RuntimeError(f"TrueFX thiếu cặp: {sorted(thieu)}")
            return ra
        except (requests.RequestException, ValueError, RuntimeError) as e:
            if k == thu - 1:
                raise
            time.sleep(2)


def main():
    luc = dt.datetime.utcnow()
    print("=" * 96)
    print(f"KIỂM SỐC GIỮA PHIÊN — {luc:%Y-%m-%d %H:%M} UTC")
    print("=" * 96)

    try:
        requests.get(f"{API}/health", timeout=8).raise_for_status()
    except Exception as e:
        print(f"API không phản hồi tại {API}: {e} — bỏ qua, không báo sốc.")
        sys.exit(0)

    try:
        gia_hien_tai = gia_hien_tai_truefx()
    except Exception as e:
        print(f"TrueFX không phản hồi được: {e} — bỏ qua, không báo sốc.")
        sys.exit(0)

    ket_qua, co_soc, loi = {}, False, []
    print(f"{'cặp':<9}{'giá mốc':>12}{'giá hiện tại':>14}{'z=Δ/σ̂':>10}{'ngưỡng':>8}{'sốc?':>7}")
    print("-" * 62)
    for p in PAIRS:
        try:
            gia_moc, sig, ngay_dubao = gia_va_sigma(p)
            gia_hnay, luc_tf = gia_hien_tai[p]
            z = float(np.log(max(gia_hnay, EPS) / max(gia_moc, EPS)) / max(sig, EPS))
            soc = abs(z) > NGUONG_SOC
            co_soc = co_soc or soc
            ket_qua[p] = dict(gia_moc=gia_moc, ngay_dubao=ngay_dubao,
                              gia_hien_tai=gia_hnay, luc_truefx=luc_tf,
                              sigma=sig, z=round(z, 3), soc=soc)
            print(f"{p:<9}{gia_moc:>12.5f}{gia_hnay:>14.5f}{z:>10.3f}"
                  f"{NGUONG_SOC:>8.2f}{'CÓ' if soc else '—':>7}")
        except Exception as e:
            loi.append(f"{p}: {e}")
            print(f"{p:<9}  lỗi: {e}")

    os.makedirs(OUT, exist_ok=True)
    trang_thai = dict(luc=f"{luc:%Y-%m-%dT%H:%M:%SZ}", nguong=NGUONG_SOC,
                      co_soc=co_soc, ket_qua=ket_qua, loi=loi)
    with open(os.path.join(OUT, "kiem_soc_trangthai.json"), "w", encoding="utf-8") as f:
        json.dump(trang_thai, f, ensure_ascii=False, indent=1)

    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"soc={'true' if co_soc else 'false'}\n")

    print("-" * 62)
    if co_soc:
        print("→ CÓ SỐC — cần tái tính toàn bộ ngay, không chờ lịch thường.")
        sys.exit(42)
    print("→ không sốc — mọi cặp trong ngưỡng, giữ nguyên dự báo hiện tại.")
    sys.exit(0)


if __name__ == "__main__":
    main()
