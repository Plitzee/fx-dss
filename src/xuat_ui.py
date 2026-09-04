"""XUAT DU LIEU CHO GIAO DIEN MVP — mot file JSON tu chua.

Moi con so trong file nay deu do tu repo, khong co so bia:
  * nen ngay OHLC that (data/prices/{PAIR}_d1.csv)
  * sigma du bao cua tang 2 (data/panel2_6pairs.csv, cot `sig`)
  * ba xac suat lop tu NEN da cham o giai doan 1 (src/balop.py), MUC TIEU P
  * lich ngan hang trung uong that (data/cb_dates.csv)
  * chi phi giao dich theo gio that (data/cost_table.csv)
  * chi so hieu chuan that (output/nen3.json)

Tham so nen (k, kP, c_h, nu, nguong che do) uoc luong CHI tren doan huan luyen
roi dong bang — dung luat split.py. Doan kiem tra KHONG duoc dung de chon gi.

Chay:  python src/xuat_ui.py
Ghi:   web/ui_data.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")

import balop as B                                    # noqa: E402
from split import doan, VALID_TU, TEST_TU            # noqa: E402

TU_NGAY = "2018-01-01"          # cat bot lich su cho trang nhe, van con 8 nam
PIP = {"USDJPY": 0.01}          # con lai 0.0001
NEN_THEO_H = {1: "chỉ σ̂", 5: "σ̂ + chế độ", 20: "σ̂ + chế độ"}


def pip_size(p):
    return PIP.get(p, 0.0001)


def nap_gia(p):
    d = pd.read_csv(os.path.join(D, "prices", f"{p}_d1.csv"), parse_dates=["Date"])
    return d.sort_values("Date").reset_index(drop=True)


def main():
    Bd = B.nap()
    cb = pd.read_csv(os.path.join(D, "cb_dates.csv"), parse_dates=["date"]) \
        if os.path.exists(os.path.join(D, "cb_dates.csv")) else None
    cost = pd.read_csv(os.path.join(D, "cost_table.csv"))

    ra = {"cap": {}, "meta": {}}
    for p in B.PAIRS:
        d = Bd[p]                                     # Date, sig, zT
        g = doan(d.Date.values)
        tr = g == 0
        gia = nap_gia(p)
        m = pd.merge(d[["Date", "sig", "zT"]], gia, on="Date", how="inner")
        m = m.sort_values("Date").reset_index(drop=True)

        # khop nen tren HUAN LUYEN, du bao cho toan chuoi
        d2 = m[["Date", "sig", "zT"]].copy()
        tr2 = doan(d2.Date.values) == 0
        nguong = np.quantile(d2.sig.values[tr2], [1 / 3, 2 / 3])
        che_do = np.digitize(d2.sig.values, nguong)

        xs = {}
        for h in (1, 5, 20):
            T = B.dung_muc_tieu(d2, h, tr2)
            ns = B.ChiSigma().khop(T["z"][tr2])
            cd = B.SigmaCheDo().khop(T["z"][tr2], d2.sig.values[tr2])
            mo = ns if NEN_THEO_H[h] == "chỉ σ̂" else cd
            P = mo.du_bao(len(d2), canh=T["canh_P"], sigma_h=T["sigma_h"],
                          sig=d2.sig.values)
            xs[str(h)] = dict(
                p=[[round(float(v), 4) for v in row] for row in P],
                b_pip=[round(float(v) / pip_size(p) * 1e0, 1) for v in T["b"]],
                sig_pip=[round(float(v) / pip_size(p), 1) for v in T["sigma_h"]],
                kP=round(float(T["kP"]), 4), c_h=round(float(T["c_h"]), 4),
                nen=NEN_THEO_H[h],
            )

        # nen 12 thang de so sanh "hom nay so voi binh thuong"
        nen12 = {h: float(pd.Series([r[1] for r in xs[h]["p"]])
                          .rolling(252, min_periods=60).mean().iloc[-1]) for h in xs}

        keep = m.Date >= pd.Timestamp(TU_NGAY)
        idx = np.flatnonzero(keep.values)
        ra["cap"][p] = dict(
            ngay=[str(x)[:10] for x in m.Date.values[idx]],
            o=[round(float(v), 5) for v in m.open.values[idx]],
            h=[round(float(v), 5) for v in m.high.values[idx]],
            l=[round(float(v), 5) for v in m.low.values[idx]],
            c=[round(float(v), 5) for v in m.close.values[idx]],
            sig_pip=[round(float(v) / pip_size(p), 1) for v in m.sig.values[idx]],
            che_do=[int(v) for v in che_do[idx]],
            pip=pip_size(p),
            nen12={k: round(v, 4) for k, v in nen12.items()},
            tam={k: dict(p=[v["p"][i] for i in idx],
                         b_pip=[v["b_pip"][i] for i in idx],
                         sig_pip=[v["sig_pip"][i] for i in idx],
                         kP=v["kP"], c_h=v["c_h"], nen=v["nen"])
                 for k, v in xs.items()},
        )

        # chi phi theo gio (che do hien hanh)
        c = cost[(cost.pair == p) & (cost.regime == "post2015")].sort_values("hour")
        ra["cap"][p]["chi_phi_gio"] = dict(
            med=[round(float(v), 3) for v in c.spread_med.values],
            p95=[round(float(v), 3) for v in c.spread_p95.values])

    # lich ngan hang trung uong — cot la `date, bank`, gop cac ngan hang hop
    # cung mot ngay thanh mot dong
    if cb is not None:
        c2 = cb[cb.date >= pd.Timestamp(TU_NGAY)].copy()
        c2["ngay"] = c2.date.astype(str).str[:10]
        ra["su_kien"] = [dict(ngay=k, nhan=sorted(set(v)))
                         for k, v in c2.groupby("ngay").bank.apply(list).items()]

    # chi so hieu chuan that tu giai doan 1
    nen3 = os.path.join(ROOT, "output", "nen3.json")
    if os.path.exists(nen3):
        ra["hieu_chuan"] = json.load(open(nen3, encoding="utf-8"))

    ra["meta"] = dict(
        cap=list(B.PAIRS), tu=TU_NGAY,
        valid_tu=str(VALID_TU.date()), test_tu=str(TEST_TU.date()),
        nen_theo_h={str(k): v for k, v in NEN_THEO_H.items()},
        ghi_chu="Mọi con số đo từ repo. Đoạn kiểm tra chưa mở.")

    out = os.path.join(ROOT, "web", "ui_data.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, separators=(",", ":"))
    kb = os.path.getsize(out) / 1024
    n = sum(len(v["ngay"]) for v in ra["cap"].values())
    print(f"đã ghi {out}  ({kb:,.0f} KB, {n:,} nến, {len(ra['cap'])} cặp)")
    print(f"sự kiện NHTW: {len(ra.get('su_kien', []))}")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
