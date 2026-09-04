"""BIEU DO TIN CAY va BAN THEO CHE DO — cho ba o.

Hai thu nay sinh ra tu cung mot bai hoc cua repo:

  1. ECE la MOT SO. No noi "trung binh lech bao nhieu" nhung khong noi lech o
     DAU. Mot mo hinh co the co ECE dep ma van sai be ben o vung xac suat cao —
     dung cho giao dien in ra con so manh nhat. Bieu do tin cay cho thay cho.
  2. TRUNG BINH GOP GIAU DUNG CHO QUAN TRONG NHAT. docs/KETQUA_VONG7.md do
     duoc QLIKE Q5/Q1 = 1,82 o nen cu so 1,01 o HAR v7 — khoang cach giua hai
     mo hinh rong GAP 13 LAN o che do cang nhat. Nen moi chi so deu phai co
     ban theo che do bien dong, khong chi ban gop.

Cham tren doan KIEM TRA, mo hinh dang chay that (theo NEN_THEO_H cua API).

Chay:  python src/tincay.py
Ghi:   output/tincay.json
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
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                          # noqa: E402
import diem3 as D                                          # noqa: E402
import volfc2 as V2                                        # noqa: E402
from split import doan                                     # noqa: E402
from volfc import merge_thin_days                          # noqa: E402

HS = (1, 5, 20)
NEN = {1: "chỉ σ̂", 5: "σ̂ + chế độ", 20: "σ̂ + chế độ"}
NBIN = 10
TEN_LOP = ("giảm", "đi ngang", "tăng")
CHE_DO = ("bình tĩnh", "vừa", "căng thẳng")
EPS = 1e-12


def nap():
    from api.main import noi_chuoi
    P, Y, CD, CAP = {h: [] for h in HS}, {h: [] for h in HS}, [], []
    for p in B.PAIRS:
        m = merge_thin_days(noi_chuoi(p))
        sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, p), 0.0))
        c = m.close.values
        zt = np.full(len(m), np.nan)
        zt[1:] = np.log(c[1:] / np.maximum(c[:-1], EPS)) / np.maximum(sig[1:], EPS)
        d = pd.DataFrame({"Date": m.Date.values, "sig": sig, "zT": zt})
        ok = np.isfinite(sig) & (sig > 0)
        d = d[ok].reset_index(drop=True)
        g = doan(d.Date.values)
        tr, te = g == 0, g == 2
        nguong = np.quantile(d.sig.values[tr], [1 / 3, 2 / 3])
        cd = np.digitize(d.sig.values, nguong)
        for h in HS:
            T = B.dung_muc_tieu(d, h, tr)
            mo = (B.ChiSigma().khop(T["z"][tr]) if NEN[h] == "chỉ σ̂"
                  else B.SigmaCheDo().khop(T["z"][tr], d.sig.values[tr]))
            Pv = mo.du_bao(len(d), canh=T["canh_P"], sigma_h=T["sigma_h"],
                           sig=d.sig.values)
            m2 = te & (T["yP"] >= 0)
            P[h].append(Pv[m2])
            Y[h].append(T["yP"][m2])
            if h == HS[0]:
                CD.append(cd[m2])
                CAP.append(np.full(int(m2.sum()), p))
    return ({h: np.vstack(P[h]) for h in HS}, {h: np.concatenate(Y[h]) for h in HS},
            np.concatenate(CD), np.concatenate(CAP))


def main():
    Pm, Ym, cd, cap = nap()
    ra = {"nbin": NBIN, "ten_lop": list(TEN_LOP), "che_do_ten": list(CHE_DO),
          "tam": {}}
    print("=" * 92)
    print("BIỂU ĐỒ TIN CẬY + BẢN THEO CHẾ ĐỘ — mục tiêu P, đoạn kiểm tra")
    print("=" * 92)
    for h in HS:
        P, y = Pm[h], Ym[h]
        n = min(len(P), len(cd))
        P, y, c_, k_ = P[:n], y[:n], cd[:n], cap[:n]
        Pkh = np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))

        # ── bieu do tin cay theo tung lop ──
        tc = {}
        for lop in range(3):
            b = D.do_tin_cay(P, y, lop, NBIN)
            tc[TEN_LOP[lop]] = [{"p": None if not np.isfinite(x[0]) else round(x[0], 4),
                                 "that": None if not np.isfinite(x[1]) else round(x[1], 4),
                                 "n": x[2]} for x in b]

        # ── ban theo CHE DO ──
        theo_cd = []
        for v in range(3):
            m = c_ == v
            if m.sum() < 100:
                continue
            r = D.bang(P[m], y[m], Pkh[m], nhom=k_[m])
            lo, hi = D.bss_ktc(P[m], y[m], Pkh[m], nhom=k_[m], nboot=300, seed=7)
            tl = np.bincount(y[m], minlength=3) / m.sum()
            theo_cd.append({"che_do": CHE_DO[v], "n": int(m.sum()),
                            "log": round(r["log"], 4), "ece": round(r["ece"], 4),
                            "mce": round(r["mce"], 4), "bss": round(r["bss"], 4),
                            "bss_lo": round(lo, 4), "bss_hi": round(hi, 4),
                            "auc": round(float(r["auc"]), 3),
                            "ty_le": [round(float(x), 4) for x in tl],
                            "p_tb": [round(float(x), 4) for x in P[m].mean(0)]})
        rg = D.bang(P, y, Pkh, nhom=k_)
        ra["tam"][str(h)] = {"nen": NEN[h], "n": int(len(y)), "tin_cay": tc,
                             "theo_che_do": theo_cd,
                             "gop": {"log": round(rg["log"], 4),
                                     "ece": round(rg["ece"], 4),
                                     "mce": round(rg["mce"], 4)}}

        print(f"\nh = {h} ({NEN[h]}), n = {len(y):,}")
        print(f"  {'chế độ':<12}{'n':>7}{'log':>9}{'ECE':>8}{'MCE':>8}{'BSS':>9}"
              f"{'KTC 95%':>22}{'AUC':>7}")
        for t in theo_cd:
            k = f"[{t['bss_lo']:+.4f}; {t['bss_hi']:+.4f}]"
            print(f"  {t['che_do']:<12}{t['n']:>7}{t['log']:>9.4f}{t['ece']:>8.4f}"
                  f"{t['mce']:>8.4f}{t['bss']:>+9.4f}{k:>22}{t['auc']:>7.3f}")
        print(f"  {'GỘP':<12}{len(y):>7}{rg['log']:>9.4f}{rg['ece']:>8.4f}"
              f"{rg['mce']:>8.4f}")

    with open(os.path.join(OUT, "tincay.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("\nđã ghi output/tincay.json")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
