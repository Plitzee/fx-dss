"""KIEM SOC GIUA PHIEN — phat hien khi gia thuc te da di xa hon gia thiet cua
du bao HOM NAY, de tai tinh SOM HON lich cron thuong (hien 4 lan/ngay, ~6h/lan).

QUAN TRONG VE PHAM VI. Day KHONG PHAI mot mo hinh du bao moi trong phien —
toan bo du an da chung minh ky nang chi ton tai o tam NGAY (ha pheu xuong H1
van 0 quy luat song sot). Script nay chi lam MOT viec: so sanh gia HIEN TAI
voi gia da dung de tinh du bao hom nay, quy ra don vi sigma^ (z = log(p/p0)/sig,
dung dung quy uoc da dung khap noi trong repo — xem rui_ro_gap.py, api/risk_logic.py),
va neu |z| vuot nguong thi kich hoat lai DUNG pipeline san xuat hien co
(jobs/cap_nhat.py) SOM HON, khong doi cron 6 tieng.

NGUON GIA HIEN TAI: Twelve Data (KHONG dung Yahoo interval=1m — da do 98,7%
thanh Yahoo M1 la anh chup gia, khong phai OHLC that, xem collect/live_fx.py).
Twelve Data can khoa mien phi: https://twelvedata.com/pricing
    Windows :  setx TWELVEDATA_API_KEY "khoa_cua_ban"
    bash    :  export TWELVEDATA_API_KEY=khoa_cua_ban
    Actions :  Settings -> Secrets -> TWELVEDATA_TOKEN
Khong co khoa thi script thoat sach, khong bao soc (an toan, giong quy uoc
FRED_API_KEY).

NGAN SACH CREDIT — DO THAT tren tai khoan free "basic" ngay 14/09/2026:
  8 credit/phut, 800 credit/ngay. Moi ma = 1 credit, GOP NHIEU MA TRONG MOT
  LAN GOI KHONG RE HON (da thu: 6 ma = 6 credit, khong giam). 6 cap kiem moi
  15 phut = 6 x 96 = 576 credit/ngay — vua, con du ~220. KHONG duoc ha xuong
  duoi ~12 phut/lan (6 x 120 = 720, sat tran; 10 phut = 864, VUOT tran).

CAN API DANG CHAY (giong jobs/cap_nhat.py) de lay gia_moc + sigma^ da tinh
san — KHONG goi lai collect/live_fx.py o day (do la buoc nang, dung Yahoo,
danh cho pipeline day du).

Chay:   python collect/kiem_soc.py
Doc:    FXDSS_API (mac dinh http://127.0.0.1:8899), TWELVEDATA_API_KEY
Ghi:    output/kiem_soc_trangthai.json
Thoat:  0 = khong soc (hoac bo qua vi thieu khoa/loi tam thoi)
        42 = CO SOC — workflow goi nen tai tinh toan bo ngay
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
API = os.environ.get("FXDSS_API", "http://127.0.0.1:8899")
TD_KEY = os.environ.get("TWELVEDATA_API_KEY", "").strip()
UA = {"User-Agent": "Mozilla/5.0 (compatible; fx-dss-thesis/1.0)"}

PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF")
TD_MA = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
         "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "USDCHF": "USD/CHF"}
NGUONG_SOC = 1.5          # cung don vi k*sigma^ da dung cho stop-loss khap noi
EPS = 1e-14


def gia_va_sigma(pair):
    """Goi API DANG CHAY (khong ton credit Twelve Data) de lay gia THAM CHIEU
    va sigma^ da dung cho du bao hom nay — dung DUNG so API dang phuc vu, khong
    tinh lai bang cong thuc khac (nguyen tac "mot nguon su that duy nhat").

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


def gia_hien_tai_td(pair, thu=2):
    """Gia dong cua nen M1 gan nhat tu Twelve Data — DA DO: Yahoo M1 la anh
    chup gia (98,7% o=h=l=c), Twelve Data M1 la nen that (xem doc string)."""
    u = "https://api.twelvedata.com/time_series"
    for k in range(thu):
        try:
            r = requests.get(u, params={"symbol": TD_MA[pair], "interval": "1min",
                                        "outputsize": 1, "apikey": TD_KEY},
                             headers=UA, timeout=15)
            d = r.json()
            if d.get("status") == "error":
                if "credit" in str(d.get("message", "")).lower() or d.get("code") == 429:
                    raise RuntimeError(f"hết credit phút này: {d.get('message')}")
                raise RuntimeError(str(d.get("message")))
            return float(d["values"][0]["close"]), str(d["values"][0]["datetime"])
        except (requests.RequestException, KeyError, RuntimeError) as e:
            if k == thu - 1:
                raise
            import time as _t
            _t.sleep(2)


def main():
    luc = dt.datetime.utcnow()
    print("=" * 96)
    print(f"KIỂM SỐC GIỮA PHIÊN — {luc:%Y-%m-%d %H:%M} UTC")
    print("=" * 96)

    if not TD_KEY:
        print("Thiếu TWELVEDATA_API_KEY — bỏ qua, không báo sốc (an toàn).")
        sys.exit(0)

    try:
        requests.get(f"{API}/health", timeout=8).raise_for_status()
    except Exception as e:
        print(f"API không phản hồi tại {API}: {e} — bỏ qua, không báo sốc.")
        sys.exit(0)

    ket_qua, co_soc, loi = {}, False, []
    print(f"{'cặp':<9}{'giá mốc':>12}{'giá hiện tại':>14}{'z=Δ/σ̂':>10}{'ngưỡng':>8}{'sốc?':>7}")
    print("-" * 62)
    for p in PAIRS:
        try:
            gia_moc, sig, ngay_dubao = gia_va_sigma(p)
            gia_hnay, luc_td = gia_hien_tai_td(p)
            z = float(np.log(max(gia_hnay, EPS) / max(gia_moc, EPS)) / max(sig, EPS))
            soc = abs(z) > NGUONG_SOC
            co_soc = co_soc or soc
            ket_qua[p] = dict(gia_moc=gia_moc, ngay_dubao=ngay_dubao,
                              gia_hien_tai=gia_hnay, luc_twelvedata=luc_td,
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
