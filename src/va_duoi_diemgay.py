"""DIEM GAY CAU TRUC (kieu Bai-Perron) cho be tac tail-risk USDJPY/USDCHF.

Tiep theo src/va_duoi_dichchuyen.py: adversarial validation da XAC NHAN co
dich chuyen phan phoi giua kiem dinh/kiem tra nhung KHONG rieng cho USDJPY/
USDCHF (EURUSD doi chung cung bi, AUC con cao hon). Cau hoi con lai: liem
gay CU THE nam o DAU, va co PHAI TRUNG dung vao moc kiem tra (2023-11-20)
cho JPY/CHF ma khong trung cho doi chung khong.

PHUONG PHAP: PELT (Killick et al. 2012, thu vien `ruptures`) — thuat toan
thuc hanh pho bien nhat cho da-diem-gay, cung ho thuoc dong Bai-Perron
(tim K diem gay toi uu hoa ham mat co phat theo so diem gay, khong can
biet truoc K). Chay tren CHUOI DA CHUAN HOA (khong phai VaR/ket qua) nen
KHONG pham luat "khong nhin ket qua kiem tra" — day la dac trung dau vao,
giong adversarial validation.

Chay tren TOAN BO chuoi (huan luyen+kiem dinh+kiem tra) VI MUC DICH la
CHAN DOAN/HIEU CO CHE cho ke hoach sua sau nay — KHONG PHAI mot quyet dinh
chon mo hinh dang bi rang buoc "chi duoc nhin kiem dinh" nhu backtest VaR.

Chay:  python src/va_duoi_diemgay.py
Ghi:   output/va_duoi_diemgay.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import ruptures as rpt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "output")
PANEL = os.path.join(ROOT, "data", "panel2_6pairs.csv")

VALID_TU = pd.Timestamp("2021-10-13")
TEST_TU = pd.Timestamp("2023-11-20")
CAP = ("USDJPY", "USDCHF", "EURUSD")
CUA_SO = 60          # do lech chuan cuon z, cung khung voi va_duoi_dichchuyen.py
PHAT = "bic"         # phat theo BIC — chuan, khong dat tay so diem gay


def chuoi_do_luong(pan):
    """Chuoi dau vao cho do gay: do lech chuan cuon 60 phien cua z — dung
    dai luong nay vi no la dai luong da xac nhan quan trong nhat o
    va_duoi_dichchuyen.py (sd_z_60 luon trong top dac trung phan biet)."""
    d = pan.sort_values("Date").reset_index(drop=True)
    z = pd.Series(d.zT.values)
    sd = z.rolling(CUA_SO, min_periods=CUA_SO).std()
    ra = pd.DataFrame({"Date": d.Date, "sd_z_60": sd})
    return ra.dropna().reset_index(drop=True)


def chay_mot_cap(pair, pan_all):
    pan = pan_all[pan_all.pair == pair]
    d = chuoi_do_luong(pan)
    x = d.sd_z_60.values.reshape(-1, 1)

    algo = rpt.Pelt(model="l2", min_size=20).fit(x)
    diem_gay = algo.predict(pen=np.log(len(x)) * np.var(x) * 2)  # phat BIC-xap xi
    diem_gay = [i for i in diem_gay if i < len(x)]  # bo diem cuoi (het chuoi)

    ngay_gay = [str(d.Date.iloc[i])[:10] for i in diem_gay]
    ket = []
    truoc = 0
    for i, ngay in zip(diem_gay, ngay_gay):
        muc_truoc = float(np.mean(x[truoc:i])) if i > truoc else float("nan")
        muc_sau_n = min(i + CUA_SO, len(x))
        muc_sau = float(np.mean(x[i:muc_sau_n])) if muc_sau_n > i else float("nan")
        ket.append({"ngay": ngay, "sd_z_60_truoc": round(muc_truoc, 4),
                    "sd_z_60_sau_60phien": round(muc_sau, 4),
                    "doi_%": round(100 * (muc_sau / muc_truoc - 1), 1)
                    if muc_truoc == muc_truoc and muc_truoc != 0 else None})
        truoc = i

    ngay_ts = pd.to_datetime(d.Date)
    gan_test = [g for g in ket
               if abs((pd.Timestamp(g["ngay"]) - TEST_TU).days) <= 90]

    return {"n": len(x), "so_diem_gay": len(diem_gay), "diem_gay": ket,
           "diem_gay_gan_moc_kiemtra_90ngay": gan_test}


def main():
    t0 = time.time()
    print("=" * 96)
    print("ĐIỂM GÃY CẤU TRÚC (PELT, kiểu Bai–Perron) trên σ̂(z) cuộn 60 phiên")
    print("=" * 96)

    pan_all = pd.read_csv(PANEL, parse_dates=["Date"])
    ra = {}
    for p in CAP:
        print(f"\n[{p}]")
        kq = chay_mot_cap(p, pan_all)
        ra[p] = kq
        print(f"  {kq['n']} phiên · {kq['so_diem_gay']} điểm gãy phát hiện được")
        for g in kq["diem_gay"]:
            gan = " ← GẦN MỐC KIỂM TRA (2023-11-20)" if g in kq["diem_gay_gan_moc_kiemtra_90ngay"] else ""
            print(f"    {g['ngay']}: {g['sd_z_60_truoc']} → {g['sd_z_60_sau_60phien']}"
                  f"  ({g['doi_%']:+.1f}%){gan}" if g['doi_%'] is not None
                  else f"    {g['ngay']}: (đầu chuỗi){gan}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "va_duoi_diemgay.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)

    print("\n" + "-" * 96)
    for p in CAP:
        n = len(ra[p]["diem_gay_gan_moc_kiemtra_90ngay"])
        print(f"  {p}: {n} điểm gãy trong vòng 90 ngày quanh mốc kiểm tra 2023-11-20")
    print(f"\n→ output/va_duoi_diemgay.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
