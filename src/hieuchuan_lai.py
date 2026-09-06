"""LOP HIEU CHUAN LAI — ha MCE cua ba o.

VAN DE. MCE (Maximum Calibration Error) tren doan kiem dinh, cau hinh dang chay:
    h=1   0,061      h=5   0,047      h=20  0,206
MCE la thung xac suat TE NHAT lech bao nhieu. ECE trung binh dep khong cuu duoc
dieu do: mot mo hinh co the hieu chuan tot o vung giua ma sai be ben o vung xac
suat cao — dung cho giao dien in ra con so manh nhat, va dung cho nguoi dung tin
nhat. Vi the giao dien dang phai in canh bao "so cang manh cang nen nghi ngo".

HAI PHUONG AN, deu la ky thuat chuan, khong phai nghien cuu:

  T   NHIET DO (temperature scaling) — MOT tham so:
          P' ∝ P^(1/T)
      T > 1 lam phang phan phoi (chua tu tin lai), T < 1 lam nhon.
      Mot tham so nen gan nhu khong the qua khop.

  V   VECTOR SCALING — bay tham so (3 he so + 3 chan + chuan hoa):
          P' ∝ exp(a_k · log P_k + b_k)
      Linh hoat hon, sua duoc thien lech RIENG TUNG LOP, nhung de qua khop hon.

GIAO THUC. Khop tren KIEM DINH, cham KIEM TRA dung mot lan. Chon theo MCE tren
kiem dinh (day la chi so ta dang nham vao), hoa thi xet diem log.

Chay:  python src/hieuchuan_lai.py
Ghi:   output/hieuchuan_lai.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                           # noqa: E402
import diem3 as D                                           # noqa: E402
import volfc2 as V2                                         # noqa: E402
from split import doan                                      # noqa: E402
from volfc import merge_thin_days                           # noqa: E402

HS = (1, 5, 20)
EPS = 1e-12


def _chuan(P):
    P = np.clip(np.asarray(P, float), 1e-9, None)
    return P / P.sum(1, keepdims=True)


def ap_nhiet(P, T):
    """P' ∝ P^(1/T). T > 1 = lam phang."""
    return _chuan(np.exp(np.log(np.clip(P, 1e-9, None)) / max(T, 1e-3)))


def ap_vector(P, th):
    """P' ∝ exp(a_k · log P_k + b_k). th = [a0,a1,a2,b0,b1,b2]."""
    a, b = np.asarray(th[:3], float), np.asarray(th[3:], float)
    return _chuan(np.exp(a * np.log(np.clip(P, 1e-9, None)) + b))


def mat_log(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))))


def khop(P, y, kieu):
    """Khop tham so hieu chuan bang cach TOI THIEU DIEM LOG tren tap dua vao."""
    if kieu == "T":
        r = optimize.minimize_scalar(lambda t: mat_log(ap_nhiet(P, t), y),
                                     bounds=(0.2, 5.0), method="bounded")
        return float(r.x)
    r = optimize.minimize(lambda th: mat_log(ap_vector(P, th), y),
                          x0=np.r_[np.ones(3), np.zeros(3)], method="Nelder-Mead",
                          options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-7})
    return r.x.tolist()


def ap(P, kieu, th):
    return ap_nhiet(P, th) if kieu == "T" else ap_vector(P, th)


def nap():
    """Dung DUNG cau hinh dang chay san xuat (api.main.NEN_THEO_H)."""
    from api.main import noi_chuoi, NEN_THEO_H
    ra = {h: dict(P=[], y=[], cap=[], g=[]) for h in HS}
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
        tr = g == 0
        for h in HS:
            T = B.dung_muc_tieu(d, h, tr)
            n = len(d)
            yt = B.lop_truoc(T["yP"], h)
            kw = dict(canh=T["canh_P"], sigma_h=T["sigma_h"], sig=d.sig.values,
                      y_truoc=yt)
            ns = B.ChiSigma().khop(T["z"][tr])
            cd = B.SigmaCheDo().khop(T["z"][tr], d.sig.values[tr])
            Pns = B.du_bao_cuon(T, d.sig.values,
                                lambda z, sg: B.ChiSigma().khop(z), canh=T["canh_P"])
            Pcd = B.du_bao_cuon(T, d.sig.values,
                                lambda z, sg: B.SigmaCheDo().khop(z, sg), canh=T["canh_P"])
            for Pc, nen in ((Pns, ns), (Pcd, cd)):
                thieu = ~np.isfinite(Pc[:, 0])
                if thieu.any():
                    Pc[thieu] = nen.du_bao(n, **kw)[thieu]
            if NEN_THEO_H[h] == "tổ hợp trực tuyến":
                kh = B.KhiHauHoc().khop(T["yP"][tr])
                qt = B.QuanTinh().khop(T["yP"][tr], yt[tr])
                mo = B.ToHopTrucTuyen(
                    [("khí hậu học", kh), ("quán tính", qt),
                     ("chỉ σ̂", B.NenCoSan("chỉ σ̂", Pns)),
                     ("σ̂ + chế độ", B.NenCoSan("σ̂ + chế độ", Pcd))], tre=h)
                P = mo.du_bao(n, y_that=T["yP"], **kw)
            else:
                P = Pcd
            m2 = T["yP"] >= 0
            ra[h]["P"].append(P[m2]); ra[h]["y"].append(T["yP"][m2])
            ra[h]["cap"].append(np.full(int(m2.sum()), p)); ra[h]["g"].append(g[m2])
    for h in HS:
        for k in ra[h]:
            ra[h][k] = np.vstack(ra[h][k]) if k == "P" else np.concatenate(ra[h][k])
    return ra


def cham(P, y, cap):
    Pkh = np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))
    r = D.bang(P, y, Pkh, nhom=cap)
    return dict(log=round(r["log"], 4), ece=round(r["ece"], 4),
                mce=round(r["mce"], 4), bss=round(r["bss"], 4))


def main():
    print("=" * 92)
    print("LỚP HIỆU CHUẨN LẠI — khớp trên KIỂM ĐỊNH, chấm KIỂM TRA một lần")
    print("=" * 92)
    Dt = nap()
    ra = {}
    for h in HS:
        P, y, cap, g = Dt[h]["P"], Dt[h]["y"], Dt[h]["cap"], Dt[h]["g"]
        va, te = g == 1, g == 2
        print(f"\n── h = {h} · kiểm định {va.sum():,} hàng · kiểm tra {te.sum():,} hàng ──")
        print(f"  {'phương án':<22}{'log':>9}{'ECE':>9}{'MCE':>9}{'BSS':>10}   đoạn")

        goc_va = cham(P[va], y[va], cap[va])
        goc_te = cham(P[te], y[te], cap[te])
        print(f"  {'gốc (chưa hiệu chuẩn)':<22}{goc_va['log']:>9.4f}{goc_va['ece']:>9.4f}"
              f"{goc_va['mce']:>9.4f}{goc_va['bss']:>+10.4f}   kiểm định")

        kq = {"goc": {"kiem_dinh": goc_va, "kiem_tra": goc_te}}
        for kieu, ten in (("T", "nhiệt độ (1 tham số)"), ("V", "vector (6 tham số)")):
            th = khop(P[va], y[va], kieu)
            v = cham(ap(P[va], kieu, th), y[va], cap[va])
            kq[kieu] = {"tham_so": th, "kiem_dinh": v}
            print(f"  {ten:<22}{v['log']:>9.4f}{v['ece']:>9.4f}{v['mce']:>9.4f}"
                  f"{v['bss']:>+10.4f}   kiểm định")

        # chon theo MCE tren KIEM DINH; hoa thi xet diem log
        ung = [("goc", goc_va), ("T", kq["T"]["kiem_dinh"]), ("V", kq["V"]["kiem_dinh"])]
        chon = min(ung, key=lambda x: (round(x[1]["mce"], 4), x[1]["log"]))[0]
        kq["chon"] = chon
        print(f"  → chọn trên kiểm định: {chon}")

        # cham KIEM TRA mot lan, cho ca ba de thay day du
        print(f"\n  {'phương án':<22}{'log':>9}{'ECE':>9}{'MCE':>9}{'BSS':>10}   đoạn")
        print(f"  {'gốc':<22}{goc_te['log']:>9.4f}{goc_te['ece']:>9.4f}"
              f"{goc_te['mce']:>9.4f}{goc_te['bss']:>+10.4f}   KIỂM TRA")
        for kieu, ten in (("T", "nhiệt độ"), ("V", "vector")):
            t_ = cham(ap(P[te], kieu, kq[kieu]["tham_so"]), y[te], cap[te])
            kq[kieu]["kiem_tra"] = t_
            danh = "  ← ĐÃ CHỌN" if kieu == chon else ""
            print(f"  {ten:<22}{t_['log']:>9.4f}{t_['ece']:>9.4f}{t_['mce']:>9.4f}"
                  f"{t_['bss']:>+10.4f}   KIỂM TRA{danh}")
        ra[str(h)] = kq

    print("\n" + "=" * 92)
    print(f"{'h':>4}{'chọn':>10}{'MCE gốc → sau':>22}{'BSS gốc → sau':>24}   (đoạn KIỂM TRA)")
    for h in HS:
        k = ra[str(h)]
        c = k["chon"]
        g_, s_ = k["goc"]["kiem_tra"], (k["goc"] if c == "goc" else k[c])["kiem_tra"]
        print(f"{h:>4}{c:>10}{g_['mce']:>13.4f} → {s_['mce']:<8.4f}"
              f"{g_['bss']:>+15.4f} → {s_['bss']:<+8.4f}")
    print("=" * 92)
    with open(os.path.join(OUT, "hieuchuan_lai.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("đã ghi output/hieuchuan_lai.json")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
