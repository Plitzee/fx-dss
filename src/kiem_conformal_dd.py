"""CONFORMAL PHAN TANG THEO TRANG THAI SUT GIAM — mon no cua TANG6_HIEU_CHUAN.

VAN DE. `docs/TANG6_HIEU_CHUAN.md` muc 5 do duoc: MOI cach dung khoang deu
PHU THIEU khi tai khoan dang lo.

    phuong phap        o dinh von   dang lo
    Gauss                  89,7%     88,6%
    Student-t              88,8%     87,8%
    Conformal              90,3%     89,3%

"Day dung la luc nguoi dung can con so chinh xac nhat." Va tai lieu do da ghi
san huong va: *"them trang thai sut giam vao bien phan tang Mondrian"*.

Huong do CHUA BAO GIO DUOC LAM. `decision_record.py` ghi da thu NAM cach
(tinh, Mondrian 3, cua so truot, ACI chung, DtACI, ACI theo tang) va khong
cach nao xoa duoc khoang chenh — nhung CA NAM deu phan tang theo BIEN DONG.
Khong cach nao phan tang theo chinh cai bien ma do phu dang lech tren no.

Day dung la cach chua ma van lieu conformal goi la Mondrian co dieu kien: neu
do phu lech theo mot bien, thi dua CHINH bien do vao lam bien phan tang.
(Vovk et al.; va cac ban 2025-2026 ve "risk-conditional coverage disparity",
vd arXiv 2512.11779 "Conditional Coverage Diagnostics for Conformal
Prediction".)

HAI CAU HINH MOI — CHOT TRUOC, khong them:

  ACI-dd 2      phan tang CHI theo trang thai sut giam (2 tang)
                -> tra loi: mot minh bien sut giam co du khong?
  ACI-2D 2x2    phan tang theo (bien dong 2 tang) x (sut giam 2 trang thai)
                -> dung han bai va cua TANG6_HIEU_CHUAN muc 5

CHONG RO RI — rang buoc quan trong nhat. Trang thai sut giam tai phien t phai
tinh tu loi suat DEN t-1. Bang danh gia cu (`run_final_eval2.py`) dung
`rolling(20).sum()` KHONG dich — dung de BAO CAO (chi la cach nhom) nhung SAI
neu dung lam bien phan tang cua mo hinh, vi no chua loi suat cua chinh phien
t. O day: bao cao giu nguyen dinh nghia cu de so sanh duoc, con mo hinh dung
ban DA DICH MOT PHIEN. Co tu kiem ep.

GIAO THUC: chon tren KIEM DINH theo |lech| max, cham MOT LAN tren KIEM TRA —
y het `run_final_eval2.py` de con so xep chung duoc mot bang.

Chay:  python src/kiem_conformal_dd.py
Ghi:   output/conformal_dd.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

from split import doan                                        # noqa: E402
from decision_record import KhoangConformal, KhoangACI        # noqa: E402

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
LEV = 0.90
A = 1 - LEV
CUA_SO_DD = 20          # y het dinh nghia cua run_final_eval2.py
MIN_N = 60


def trang_thai_dd(r, dich=True):
    """1 = dang lo (tong loi suat 20 phien gan nhat am), 0 = o dinh.

    dich=True: gia tri tai t chi dung loi suat DEN t-1 — bat buoc khi dung
    lam bien phan tang. dich=False: dinh nghia bao cao cu, giu de so sanh.
    """
    s = pd.Series(r).rolling(CUA_SO_DD).sum()
    if dich:
        s = s.shift(1)
    return (s.values < 0).astype(int)


class KhoangACI2D:
    """ACI + Mondrian tren HAI bien phan tang: bien dong x trang thai sut giam.

    Cung co che ACI cua `KhoangACI` (alpha cap nhat truc tuyen cho TUNG o),
    chi khac chi so o la tich hai bien thay vi mot.
    """

    def __init__(self, z_tr, sig_tr, dd_tr, n_vol=2, n_dd=2, muc=0.90,
                 gamma=0.01, cua_so=750):
        z = np.asarray(z_tr, float)
        s = np.asarray(sig_tr, float)
        d = np.asarray(dd_tr, int)
        self.n_vol, self.n_dd = int(n_vol), int(n_dd)
        self.muc, self.gamma, self.cua_so = float(muc), float(gamma), int(cua_so)
        self.edges = (np.quantile(s, np.arange(1, self.n_vol) / self.n_vol)
                      if self.n_vol > 1 else np.array([]))
        self.n_o = self.n_vol * self.n_dd
        self.alpha = np.full(self.n_o, 1.0 - self.muc)
        self.z_hist = list(z[-self.cua_so:])
        self.s_hist = list(s[-self.cua_so:])
        self.d_hist = list(d[-self.cua_so:])

    def _o(self, sig, dd):
        iv = int(np.digitize([float(sig)], self.edges)[0]) if self.n_vol > 1 else 0
        return iv * self.n_dd + int(dd)

    def _cal(self, i):
        z = np.asarray(self.z_hist)
        s = np.asarray(self.s_hist)
        d = np.asarray(self.d_hist, int)
        iv = (np.digitize(s, self.edges) if self.n_vol > 1
              else np.zeros(len(s), int))
        sel = z[iv * self.n_dd + d == i]
        if len(sel) >= MIN_N:
            return sel
        # lui ve: thu gop theo bien dong truoc, roi moi den bo chung
        sel2 = z[iv == (i // self.n_dd)]
        return sel2 if len(sel2) >= MIN_N else z

    def nua_be_rong(self, muc=None, sig=None, dd=0):
        i = self._o(sig, dd) if sig is not None else 0
        lev = (1.0 - self.alpha[i]) if muc is None else float(muc)
        c = self._cal(i)
        return float(np.quantile(np.abs(c),
                                 min(max(lev, 0.0) * (1 + 1 / len(c)), 0.9999)))

    def quan_sat(self, z, sig, dd):
        i = self._o(sig, dd)
        h = self.nua_be_rong(None, sig, dd)
        err = 1.0 if abs(float(z)) > h else 0.0
        self.alpha[i] = float(np.clip(
            self.alpha[i] + self.gamma * ((1 - self.muc) - err), 1e-4, 0.5))
        self.z_hist.append(float(z)); self.s_hist.append(float(sig))
        self.d_hist.append(int(dd))
        if len(self.z_hist) > self.cua_so:
            self.z_hist.pop(0); self.s_hist.pop(0); self.d_hist.pop(0)


CU = ["tĩnh", "Mondrian 2", "Mondrian 3", "ACI", "ACI-tầng 2", "ACI-tầng 3"]
MOI = ["ACI-dd 2", "ACI-2D 2x2"]


def khoang_chay(d, g, method, seg):
    s = d.sig.values
    z = d.zT.values
    r = d.zT.values * d.sig.values
    dd_mo_hinh = trang_thai_dd(r, dich=True)       # DA DICH — dung cho mo hinh
    tr = g == 0
    idx = np.where(g == seg)[0]
    truoc = np.where(~tr & (np.arange(len(d)) < idx[0]))[0]

    if method in MOI:
        n_vol = 1 if method == "ACI-dd 2" else 2
        kc = KhoangACI2D(z[tr], s[tr], dd_mo_hinh[tr], n_vol=n_vol, n_dd=2)
        H = np.empty(len(idx))
        for i in truoc:
            kc.quan_sat(z[i], s[i], dd_mo_hinh[i])
        for j, i in enumerate(idx):
            H[j] = kc.nua_be_rong(None, s[i], dd_mo_hinh[i])
            kc.quan_sat(z[i], s[i], dd_mo_hinh[i])
        return z[idx], H, s[idx]

    if method.startswith("ACI"):
        nb = 1 if method == "ACI" else (2 if method.endswith("2") else 3)
        kc = KhoangACI(z[tr], s[tr], n_bins=nb)
        H = np.empty(len(idx))
        for i in truoc:
            kc.quan_sat(z[i], s[i])
        for j, i in enumerate(idx):
            H[j] = kc.nua_be_rong(None, s[i])
            kc.quan_sat(z[i], s[i])
        return z[idx], H, s[idx]

    nb = {"tĩnh": 1, "Mondrian 2": 2, "Mondrian 3": 3}[method]
    kc = KhoangConformal(z[tr], s[tr], n_bins=nb)
    H = np.array([kc.nua_be_rong(LEV, x) for x in s[idx]])
    return z[idx], H, s[idx]


def danh_gia(pan, seg, methods):
    out = {}
    for m in methods:
        C, CV, SC, DD = [], [[], []], [], [[], []]
        for p in PAIRS:
            d = pan[pan.pair == p].reset_index(drop=True)
            g = doan(d.Date.values)
            z, H, s = khoang_chay(d, g, m, seg)
            ok = np.abs(z) <= H
            C.append(ok.mean())
            q = np.quantile(d.sig.values[g == 0], [1 / 3, 2 / 3])
            gg = np.digitize(s, q)
            if (gg == 0).sum() > 20:
                CV[0].append(ok[gg == 0].mean())
            if (gg == 2).sum() > 20:
                CV[1].append(ok[gg == 2].mean())
            # BAO CAO: giu dung dinh nghia cu (khong dich) de so sanh duoc
            r = d.zT.values * d.sig.values
            lo = trang_thai_dd(r, dich=False)[g == seg].astype(bool)
            if (~lo).sum() > 20:
                DD[0].append(ok[~lo].mean())
            if lo.sum() > 20:
                DD[1].append(ok[lo].mean())
            w = 2 * H * s
            SC.append(np.mean(w + (2 / A) * np.maximum(np.abs(z * s) - H * s, 0)
                              * 2) * 1e4)
        c, v0, v2 = np.mean(C), np.mean(CV[0]), np.mean(CV[1])
        pk, ll = np.mean(DD[0]), np.mean(DD[1])
        out[m] = dict(chung=c, vol_thap=v0, vol_cao=v2, dinh=pk, lo=ll,
                      khe_dd=pk - ll,
                      lech=max(abs(x - LEV) for x in (c, v0, v2, pk, ll)),
                      diem=np.mean(SC))
    return out


def _tu_kiem_ro_ri(pan):
    """Bien phan tang cua MO HINH phai khong doi khi cat bo tuong lai."""
    d = pan[pan.pair == "EURUSD"].reset_index(drop=True)
    r = (d.zT.values * d.sig.values)
    moc = 3000
    day = trang_thai_dd(r, dich=True)[:moc]
    cat = trang_thai_dd(r[:moc], dich=True)
    n = min(len(day), len(cat))
    sai = int((day[:n] != cat[:n]).sum())
    # va: ban KHONG dich phai CHUA thong tin cua chinh phien t (doi chung duong)
    kd = trang_thai_dd(r, dich=False)[:moc]
    khac = int((kd[:n] != day[:n]).sum())
    return sai, n, khac


def main():
    t0 = time.time()
    print("=" * 104)
    print("CONFORMAL PHÂN TẦNG THEO TRẠNG THÁI SỤT GIẢM")
    print("món nợ của docs/TANG6_HIEU_CHUAN.md mục 5 — hướng vá đã ghi, "
          "chưa bao giờ làm")
    print("=" * 104)

    pan = pd.read_csv(os.path.join(DATA, "panel2_6pairs.csv"),
                      parse_dates=["Date"])
    sai, n, khac = _tu_kiem_ro_ri(pan)
    print(f"TỰ KIỂM CHỐNG RÒ RỈ (biến phân tầng của mô hình)")
    print(f"  cắt bỏ tương lai → {sai}/{n:,} giá trị đổi   "
          f"{'ĐẠT' if sai == 0 else 'HỎNG'}")
    print(f"  đối chứng dương: bản KHÔNG dịch khác bản đã dịch ở {khac:,}/{n:,} "
          f"phiên   {'ĐẠT' if khac > 0 else 'HỎNG — hai bản giống nhau?'}")
    assert sai == 0 and khac > 0, "tu kiem ro ri HONG"

    tat = CU + MOI
    dv = danh_gia(pan, 1, tat)
    dt = danh_gia(pan, 2, tat)

    def bang(d, ten, chon=None):
        print("\n" + "=" * 104)
        print(ten)
        print("=" * 104)
        print(f"{'phương pháp':<14}{'phủ chung':>11}{'vol thấp':>10}{'vol cao':>9}"
              f"{'ở đỉnh':>9}{'đang lỗ':>10}{'khe đỉnh−lỗ':>14}"
              f"{'|lệch| max':>12}{'điểm khoảng':>13}")
        print("-" * 104)
        for m in sorted(d, key=lambda k: d[k]["lech"]):
            r = d[m]
            mk = ""
            if m in MOI:
                mk = "  ← MỚI"
            if chon and m == chon:
                mk += "  ←chọn"
            print(f"{m:<14}{r['chung']:>11.1%}{r['vol_thap']:>10.1%}"
                  f"{r['vol_cao']:>9.1%}{r['dinh']:>9.1%}{r['lo']:>10.1%}"
                  f"{r['khe_dd']:>13.1%}{r['lech']:>12.1%}{r['diem']:>13.1f}{mk}")

    bang(dv, "BẢNG A — CHỌN TRÊN ĐOẠN KIỂM ĐỊNH (|lệch| max nhỏ nhất)")
    best = min(dv, key=lambda k: dv[k]["lech"])
    print("-" * 104)
    print(f"Chọn trên kiểm định: {best}")

    bang(dt, "BẢNG B — CHẤM MỘT LẦN TRÊN ĐOẠN KIỂM TRA", chon=best)
    tb = min(dt, key=lambda k: dt[k]["lech"])
    print("-" * 104)
    print(f"  cách được chọn, trên kiểm tra : |lệch| max {dt[best]['lech']:.1%}")
    print(f"  tốt nhất có thể trên kiểm tra : {dt[tb]['lech']:.1%}  ({tb})")
    print(f"  giá phải trả cho việc chọn    : "
          f"{dt[best]['lech']-dt[tb]['lech']:+.1%}")

    print("\n" + "=" * 104)
    print("CÂU HỎI TRUNG TÂM — KHE ĐỈNH−LỖ CÓ HẸP LẠI KHÔNG?")
    print("=" * 104)
    cu_tot = min(CU, key=lambda k: abs(dt[k]["khe_dd"]))
    print(f"{'':<16}{'khe kiểm định':>16}{'khe kiểm tra':>15}")
    print("-" * 104)
    for m in CU + MOI:
        mk = "  ← MỚI" if m in MOI else ""
        print(f"{m:<16}{dv[m]['khe_dd']:>15.2%}{dt[m]['khe_dd']:>15.2%}{mk}")
    print("-" * 104)
    print(f"  khe nhỏ nhất trong 6 cách CŨ (kiểm tra): {dt[cu_tot]['khe_dd']:.2%}"
          f"  ({cu_tot})")
    for m in MOI:
        print(f"  {m:<14} khe kiểm tra {dt[m]['khe_dd']:.2%}  → "
              f"{'HẸP HƠN' if abs(dt[m]['khe_dd']) < abs(dt[cu_tot]['khe_dd']) else 'KHÔNG hẹp hơn'}"
              f" cách cũ tốt nhất")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(kiem_dinh=dv, kiem_tra=dt, chon=best,
                   tot_nhat_kiem_tra=tb, cu_tot_khe=cu_tot,
                   tu_kiem=dict(ro_ri_sai=sai, n=n, doi_chung_duong=khac)),
              open(os.path.join(OUT, "conformal_dd.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/conformal_dd.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
