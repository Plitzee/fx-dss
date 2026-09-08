"""KIEM TRA CHEO MOT QUY LUAT UNG VIEN — bo khung bon phep kiem.

KET CUC: quy luat H3 ma file nay dung de soi HOA RA KHONG CO THAT. Bon phep
kiem duoi day da bac no (chet o muc tieu R, chet o h=5/h=20, dong gop BSS co
KTC chua 0), va viec truy nguyen nhan dan thang toi mot LOI DU LIEU lam hong
ca Giai doan 2 — 3.306 hang bi bia loi suat bang 0, xem CHISO_DANHGIA.md muc
13. Sau khi sua loi, khong la CART nao con song sot Westfall-Young, nen khong
con quy luat nao de kiem.

Giu file lai vi BO KHUNG con dung duoc: bat ky quy luat ung vien nao sau nay
cung phai qua dung bon phep kiem A/B/C/D o duoi truoc khi duoc tin. Chay lai
file nay bay gio la chay tren la CART #0 cua cay DA KHOP LAI tren du lieu da
sua — no chet o ca sau o (h x muc tieu), dung nhu mong doi khi khong con tin
hieu nao.

────────────────────────────────────────────────────────────────────────────

BOI CANH GOC (giu nguyen, vi trinh tu suy luan la phan dang doc):

BOI CANH. `run_h3_rulelist.py` (08/09/2026) tim duoc dung MOT quy luat qua het
bon cua cua Giai doan 2:

    sigma^ rat thap (<= decile-1)  VA  ATR phan vi rat thap (<= ~p23)
        -> lop "di ngang",  lift 1,67,  t|dieu kien = 4,34,
           LOPO 6/6 cap duong,  tai lap tren KIEM TRA z = 4,31

Day la quy luat DAU TIEN cua ca Giai doan 2 lam duoc dieu do. Chinh vi the no
phai bi soi ky nhat, khong phai duoc an mung nhanh nhat.

DIEM DANG NGO, phai tra loi truoc khi dua vao luan van:

  CA HAI VE CUA QUY LUAT DEU LA THUOC DO BIEN DONG. `la_vi_tu_nen()` trong
  run_quyluat.py loai vi tu "chi noi ve chinh nen", nhung danh sach
  DAC_TRUNG_NEN hien chi co ("sigma^",) — ATR khong nam trong do. Chinh
  `CHISO_DANHGIA.md` muc 9.1 da ghi viec con no: "la_vi_tu_nen hien chi loai
  3/630 vi tu. Dung ra phai loai moi vi tu chi noi ve bien dong."

  Neu the: quy luat nay co the chi la "hai thuoc do bien dong cung noi bien
  dong dang rat thap" — tuc mot UOC LUONG BIEN DONG TOT HON, chu khong phai
  mot quy luat noi them dieu gi ngoai bien dong. Voi muc tieu P (dai pip CO
  DINH) thi bien dong thap -> bien do nho -> nam trong dai -> "di ngang" la
  quan he CO HOC, khong phai kham pha.

BA PHEP KIEM, tu sac nhat den nhe nhat:

  A. DOI CHUNG MANH HON. Them bien gia phan vi cua ATR (rieng tung cap, 10 o,
     dung y het kiem_soat_sigma) vao bo kiem soat. Neu quy luat song sot ca
     khi DA khu ca sigma^ LAN ATR mot cach mem deo, thi phan con lai la
     TUONG TAC (ca hai cung cuc thap) — mot thu that su moi. Neu chet, no la
     "hai thuoc do bien dong dong thuan" va phai bao cao dung nhu the.

  B. MUC TIEU R. Muc tieu R chia bien do cho sigma^ (dai = k * sigma_h), nen
     quan he co hoc "bien dong thap -> trong dai" BI KHU. Quy luat con song o
     R nghia la no noi duoc dieu gi ma tang bien dong chua noi.

  C. TAM HAN KHAC (h = 5, 20). Quy luat duoc phat hien o h=1; neu no la cau
     truc that thi phai con dau vet o tam han dai hon, du yeu hon.

GIAO THUC. Vi tu (la CART) duoc DINH NGHIA LAI y het cach run_h3_rulelist.py
da lam — cay khop tren HUAN LUYEN cua h=1/muc tieu P, KHONG khop lai theo tung
muc tieu. Quy luat gio la GIA THUYET DA CHOT, nen cac phep kiem duoi day la
kiem CHINH THUC mot gia thuyet duy nhat, khong can hieu chinh boi. Doan KIEM
TRA chi duoc dung o dung cho da dung roi (bao cao lai), khong mo them.

Chay:  python src/kiem_h3.py
Ghi:   output/kiem_h3.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                              # noqa: E402
import chibao as CB                                            # noqa: E402
import diem3 as D                                              # noqa: E402
from split import doan                                         # noqa: E402
from run_quyluat import (                                      # noqa: E402
    nap_du_lieu, doi_chung, kiem_soat_sigma, T_DIEU_KIEN, MIN_KHOP, EPS,
)
from run_h3_rulelist import xay_cay, MAX_DEPTH, MIN_LA          # noqa: E402

TEN_LOP = ("giảm", "đi ngang", "tăng")
LA_QUY_LUAT = 0            # "la CART #0" — la da song sot, chot tu output/h3_rulelist.json


def kiem_soat_atr(Ms, cap, tr, npv=10):
    """Bien gia phan vi cua ATR, RIENG TUNG CAP — y het kiem_soat_sigma nhung
    cho ATR. Day la null bien dong THU HAI, doc lap voi sigma^ ve cach tinh:
    ATR la bien do thuc te DA XAY RA (high-low-close), con sigma^ la DU BAO
    cua HAR. Neu quy luat chi la "hai thuoc do cung noi bien dong thap" thi
    khu ca hai se lam no bien mat."""
    atr = []
    for m in Ms:
        R = CB.tinh_tat_ca(m)
        atr.append(np.asarray(R["atr_pv"], float))
    atr = np.concatenate(atr)
    cap = np.asarray(cap)
    ten_cap = list(dict.fromkeys(cap.tolist()))
    X = np.zeros((len(atr), len(ten_cap) * (npv - 1)), np.float32)
    for j, p in enumerate(ten_cap):
        mp = cap == p
        v = atr[mp & tr & np.isfinite(atr)]
        if len(v) < 100:
            continue
        q = np.quantile(v, np.linspace(0, 1, npv + 1)[1:-1])
        b = np.digitize(atr, q)
        for k in range(1, npv):
            X[mp & (b == k), j * (npv - 1) + k - 1] = 1.0
    return X


def muc_tieu(Ms, sigs, dts, h, ten_mt):
    """y DA DICH 1 phien cho (tam han h, muc tieu P hoac R), gop 6 cap —
    dung het cach nap_du_lieu() dung cho h=1/P."""
    ys = []
    for i in range(len(B.PAIRS)):
        d = pd.DataFrame({"Date": Ms[i].Date.values, "sig": sigs[i]})
        c = Ms[i].close.values
        zt = np.full(len(Ms[i]), np.nan)
        zt[1:] = np.log(c[1:] / np.maximum(c[:-1], EPS)) / np.maximum(sigs[i][1:], EPS)
        d["zT"] = zt
        tri = doan(d.Date.values) == 0
        T = B.dung_muc_tieu(d, h, tri)
        yy = T["yP"] if ten_mt == "P" else T["yR"]
        yv = np.full(len(d), -1)
        yv[:-1] = yy[1:]
        ys.append(yv)
    return np.concatenate(ys)


def z_lift_1(mkhop, y, mask):
    """z va lift cho MOT vi tu, tung lop. Cung cong thuc z_lift nhung 1 hang."""
    m = mkhop & mask
    nk = float(m.sum())
    ra = {}
    for c in range(3):
        yc = (y[mask] == c).astype(float)
        p = yc.mean()
        k = float((y[m] == c).sum())
        sd = np.sqrt(max(nk * p * (1 - p), EPS))
        ra[c] = dict(z=(k - nk * p) / sd if nk >= MIN_KHOP else np.nan,
                     lift=k / max(nk * p, EPS) if nk >= MIN_KHOP else np.nan,
                     n=int(nk))
    return ra


def main():
    t0 = time.time()
    print("=" * 108)
    print("KIỂM TRA CHÉO QUY LUẬT H3 — \"σ̂ rất thấp VÀ ATR rất thấp → đi ngang\"")
    print("=" * 108)
    du = nap_du_lieu()
    Ms, sigs, zs, dts = du["Ms"], du["sigs"], du["zs"], du["dts"]
    cap, cum = du["cap"], du["cum"]
    tr, va, te, pha = du["tr"], du["va"], du["te"], du["pha"]
    ks_goc = du["kiem_soat"]
    y1P = du["y"]

    # ── dung lai DUNG vi tu la CART, khop tren huan luyen h=1/P ─────────
    from sklearn.tree import DecisionTreeClassifier
    X, ten_dt, ok = xay_cay(zs, dts, Ms, sigs, tr)
    cay = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LA,
                                  random_state=0)
    cay.fit(X[tr & ok], y1P[tr & ok])
    la = cay.apply(np.where(np.isfinite(X), X, 0.0))
    la_id = sorted(set(la[tr & ok].tolist()))
    vt = (la == la_id[LA_QUY_LUAT]) & ok
    print(f"vị từ: lá CART #{LA_QUY_LUAT} · n = {int(vt.sum()):,} phiên khớp "
          f"({int((vt & pha).sum()):,} trên phát hiện)")

    ket = {}

    # ── A. ĐỐI CHỨNG MẠNH HƠN: thêm biến giả phân vị ATR ────────────────
    print("\n" + "─" * 108)
    print("A. ĐỐI CHỨNG CÓ ĐIỀU KIỆN MẠNH HƠN — khử cả σ̂ LẪN ATR (mềm dẻo, riêng từng cặp)")
    print("─" * 108)
    ks_atr = kiem_soat_atr(Ms, cap, tr)
    ks_manh = np.column_stack([ks_goc, ks_atr])
    print(f"  bộ kiểm soát gốc: {ks_goc.shape[1]} cột  →  thêm ATR: {ks_manh.shape[1]} cột")
    for ten_ks, KS in (("gốc (σ̂ + TSMOM + đô-la + carry)", ks_goc),
                       ("MẠNH (+ phân vị ATR riêng từng cặp)", ks_manh)):
        b, t = doi_chung(vt, y1P, 1, KS, cum=cum)
        dat = np.isfinite(t) and abs(t) > T_DIEU_KIEN
        print(f"  {ten_ks:<40} b = {b:+.4f}   t = {t:+.2f}   "
              f"{'CÒN tin riêng' if dat else 'MẤT tin riêng (≤ 3,0)'}")
        ket[f"A_{ten_ks[:4]}"] = dict(b=float(b) if np.isfinite(b) else None,
                                       t=float(t) if np.isfinite(t) else None,
                                       dat=bool(dat))

    # ── B & C. MỤC TIÊU R và TẦM HẠN KHÁC ───────────────────────────────
    print("\n" + "─" * 108)
    print("B+C. TÁI LẬP TRÊN MỤC TIÊU R (đã chia σ̂ — khử quan hệ cơ học) VÀ TẦM HẠN KHÁC")
    print("─" * 108)
    print("  mục tiêu P = |r_h| < dải pip CỐ ĐỊNH   ·   mục tiêu R = |z_h| < k (dải THEO σ̂)")
    print(f"\n  {'tầm hạn':<9}{'mục tiêu':<10}{'n khớp':>9}{'lift':>8}{'z':>8}"
          f"{'t|đk gốc':>11}{'t|đk MẠNH':>12}   kết luận")
    for h in (1, 5, 20):
        for mt in ("P", "R"):
            y = muc_tieu(Ms, sigs, dts, h, mt)
            m_pha = pha & (y >= 0)
            r = z_lift_1(vt, y, m_pha)[1]           # lop 1 = "di ngang"
            b1, t1 = doi_chung(vt, y, 1, ks_goc, cum=cum)
            b2, t2 = doi_chung(vt, y, 1, ks_manh, cum=cum)
            ok_goc = np.isfinite(t1) and abs(t1) > T_DIEU_KIEN
            ok_manh = np.isfinite(t2) and abs(t2) > T_DIEU_KIEN
            kl = ("SỐNG cả hai" if ok_goc and ok_manh else
                  "sống đối chứng gốc, CHẾT khi khử ATR" if ok_goc else
                  "CHẾT")
            print(f"  h={h:<7}{mt:<10}{r['n']:>9,}{r['lift']:>8.3f}{r['z']:>8.2f}"
                  f"{t1:>11.2f}{t2:>12.2f}   {kl}")
            ket[f"h{h}_{mt}"] = dict(n=r["n"], lift=float(r["lift"]),
                                      z=float(r["z"]), t_goc=float(t1),
                                      t_manh=float(t2), song_goc=bool(ok_goc),
                                      song_manh=bool(ok_manh))

    # ── D. ĐÓNG GÓP KINH TẾ: thêm quy luật vào nền có cải thiện BSS không ──
    print("\n" + "─" * 108)
    print("D. ĐÓNG GÓP THẬT — thêm quy luật vào nền \"chỉ σ̂\" có cải thiện BSS không (kiểm định)")
    print("─" * 108)
    from run_spa_ho2 import nen_chi_sigma
    P_nen = nen_chi_sigma(Ms, sigs, dts)
    y = y1P
    m_va = va & (y >= 0) & np.isfinite(P_nen).all(1)
    kh_tr = np.bincount(y[tr & (y >= 0)], minlength=3).astype(float)
    kh_tr /= kh_tr.sum()
    act_tr = vt & tr & (y >= 0)
    f_rule = np.bincount(y[act_tr], minlength=3).astype(float)
    f_rule /= f_rule.sum()
    print(f"  tần suất lớp khi quy luật bật (huấn luyện): "
          f"giảm {f_rule[0]:.3f} · đi ngang {f_rule[1]:.3f} · tăng {f_rule[2]:.3f}")
    print(f"  tần suất nền (khí hậu học huấn luyện):      "
          f"giảm {kh_tr[0]:.3f} · đi ngang {kh_tr[1]:.3f} · tăng {kh_tr[2]:.3f}")
    Pa = P_nen[m_va].copy()                       # nen chi sigma^
    Pb = Pa.copy()
    act_va = vt[m_va]
    Pb[act_va] = f_rule                           # nen + quy luat de len
    yv = y[m_va]
    Pkh = np.tile(kh_tr, (len(yv), 1))
    r_a = D.bang(Pa, yv, Pkh, nhom=cap[m_va])
    r_b = D.bang(Pb, yv, Pkh, nhom=cap[m_va])
    dlo, dhi = D.delta_bss_ktc(Pb, Pa, yv, Pkh, nhom=cap[m_va], nboot=400,
                                khoi=20, seed=7)
    print(f"  n(kiểm định) = {len(yv):,} · số phiên quy luật bật = {int(act_va.sum()):,}"
          f" ({act_va.mean():.1%})")
    print(f"  BSS nền \"chỉ σ̂\"          : {r_a['bss']:+.4f}   (log {r_a['log']:.4f})")
    print(f"  BSS nền + quy luật H3     : {r_b['bss']:+.4f}   (log {r_b['log']:.4f})")
    print(f"  Δ BSS, KTC 95% ghép cặp   : [{dlo:+.5f}, {dhi:+.5f}]   "
          f"{'CÓ cải thiện có ý nghĩa' if np.isfinite(dlo) and dlo > 0 else 'không tách khỏi 0'}")
    ket["D_dong_gop"] = dict(bss_nen=r_a["bss"], bss_them=r_b["bss"],
                              d_lo=dlo, d_hi=dhi, n_va=int(len(yv)),
                              ty_le_bat=float(act_va.mean()))

    os.makedirs(OUT, exist_ok=True)
    json.dump(ket, open(os.path.join(OUT, "kiem_h3.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/kiem_h3.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
