"""PHA 2 / H8c — CHU DE CHINH cua thong cao FOMC co noi them gi khong?

Ho D (embedding, H8b) da thu va THUA dac trung thu cong (H8). Con lai HO C
cua tuan 2 roadmap ("event representation — structured event types") chua
thu: phan loai MOI thong cao vao mot CHU DE CHINH roi rac, thay vi do MUC DO
(giong dieu lien tuc nhu H8) hay embedding (H8b).

BON CHU DE — CHOT TRUOC, tu dien liet ke day du, khong sua sau khi nhin ket
qua (cung tinh than HAWK/DOVE cua H8):

  lam_phat        inflation, inflationary, price stability, prices
  viec_lam        employment, labor market, unemployment, jobs, payroll
  tang_truong     economic activity, growth, expansion, output, gdp
  on_dinh_taichinh financial conditions, financial stability, credit,
                   banking, market functioning

CHU DE CHINH cua mot thong cao = tu dien nao co MAT DO tu (dem/tong so tu)
CAO NHAT — hoa thi giu thu tu liet ke o tren. Day la vi tu RIENG RE (khong
phai tam phan vi lien tuc nhu H8): "chu de chinh la lam_phat" v.v.

KHONG GIAN GIA THUYET: 4 chu de x 2 cua so (1, 5 phien) = 8 vi tu x 3 lop
= 24 gia thuyet — doc lap voi H8 (54) va H8b (54).

Cung giao thuc phau bon cua, cung bo kiem soat (4 bien goc + chi bao cua so
sau hop) nhu H8/H8b.

Chay:  python src/run_h8c_chude.py
Ghi:   output/h8c_chude.json
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

import balop as B                                             # noqa: E402
from run_h8_tintuc import nap_thong_cao                        # noqa: E402
from run_quyluat import (                                     # noqa: E402
    nap_du_lieu, z_lift, westfall_young, doi_chung, TEN_LOP,
    MIN_KHOP, LIFT_LOPO, MIN_CAP_DUONG, T_DIEU_KIEN, NPERM, KHOI, EPS,
)

CUA_SO = (1, 5)
MAX_PHU = 2.0 / 3.0

TU_DIEN = {
    "lạm phát": ("inflation", "inflationary", "price stability", "prices"),
    "việc làm": ("employment", "labor market", "unemployment", "jobs", "payroll"),
    "tăng trưởng": ("economic activity", "growth", "expansion", "output", "gdp"),
    "ổn định tài chính": ("financial conditions", "financial stability", "credit",
                          "banking", "market functioning"),
}


def chu_de_chinh(van_ban):
    """Chu de co MAT DO tu cao nhat. Hoa -> thu tu liet ke trong TU_DIEN."""
    w = van_ban.lower()
    n = max(len(w.split()), 1)
    mat_do = {}
    for cd, tu_khoa in TU_DIEN.items():
        dem = sum(w.count(t) for t in tu_khoa)
        mat_do[cd] = dem / n
    best = max(mat_do.values())
    if best == 0:
        return None
    for cd in TU_DIEN:                       # thu tu chot truoc, cho hoa
        if mat_do[cd] == best:
            return cd


def dac_trung_chu_de(tc):
    cd = [chu_de_chinh(t) for t in tc.van_ban]
    return pd.DataFrame(dict(ngay=tc.ngay, chu_de=cd))


def gan_vao_phien(F, dts_cap, cua_so):
    ng = pd.DatetimeIndex(dts_cap)
    ra = np.full(len(ng), None, dtype=object)
    for _, r in F.iterrows():
        if r.chu_de is None:
            continue
        sau = np.flatnonzero(ng > r.ngay)
        if len(sau) == 0:
            continue
        ra[sau[:cua_so]] = r.chu_de
    return ra


def _tu_kiem(F, dts_cap):
    ng = pd.DatetimeIndex(dts_cap)
    A = gan_vao_phien(F, dts_cap, 5)
    xau = 0
    for _, r in F.iterrows():
        if r.chu_de is None:
            continue
        i = np.flatnonzero(ng <= r.ngay)
        if len(i) and A[i[-1]] is not None and A[i[-1]] == r.chu_de:
            xau += 1
    return xau


def vi_tu_chude(F, dts, tr, pha):
    lit, ten = [], []
    for cs in CUA_SO:
        A = np.concatenate([gan_vao_phien(F, dts[i], cs)
                            for i in range(len(B.PAIRS))], axis=0)
        for cd in TU_DIEN:
            v = (A == cd)
            vt = v[tr]
            if vt.sum() < 100:
                continue
            lit.append(v)
            ten.append(f"chủ đề {cd} [{cs} phiên sau]")
    M = np.array(lit)
    phu = (M & pha[None, :]).sum(1) / max(int(pha.sum()), 1)
    rong = phu > MAX_PHU
    if rong.any():
        M, ten = M[~rong], [t for t, b in zip(ten, rong) if not b]
    return M, ten


def main():
    t0 = time.time()
    print("=" * 112)
    print("PHA 2 / H8c — CHỦ ĐỀ CHÍNH thông cáo FOMC (Historical + News)")
    print("=" * 112)

    tc = nap_thong_cao()
    if len(tc) < 30:
        print(f"CHỈ có {len(tc)} thông cáo — chạy `python collect/tin_tuc_nhtw.py` trước.")
        return
    print(f"{len(tc)} thông cáo FOMC · {tc.ngay.min().date()} → {tc.ngay.max().date()}")
    F = dac_trung_chu_de(tc)
    print("\nphân bố chủ đề chính:")
    print(F.chu_de.value_counts(dropna=False).to_string())

    d = nap_du_lieu()
    Ms, dts = d["Ms"], d["dts"]
    y, cap = d["y"], d["cap"]
    kiem_soat, cum = d["kiem_soat"], d["cum"]
    tr, va, te, pha = d["tr"], d["va"], d["te"], d["pha"]

    print("\ntự kiểm rò rỉ — chủ đề KHÔNG được xuất hiện ở chính phiên họp…",
          flush=True)
    xau = _tu_kiem(F, dts[0])
    print(f"  {xau} vi phạm  {'ĐẠT' if xau == 0 else '← RÒ RỈ'}")
    assert xau == 0, "đặc trưng rò rỉ vào phiên họp"

    M, ten = vi_tu_chude(F, dts, tr, pha)
    print()
    print(f"KHÔNG GIAN GIẢ THUYẾT H8c: {len(ten)} vị từ × 3 lớp = "
          f"{len(ten)*3} giả thuyết — liệt kê đầy đủ, biết trước")

    trong_cs = np.zeros(len(y), bool)
    off = np.concatenate([[0], np.cumsum([len(x) for x in dts])])
    for i in range(len(B.PAIRS)):
        ng = pd.DatetimeIndex(dts[i])
        for ngay_tc in F.ngay:
            sau = np.flatnonzero(ng > ngay_tc)[:max(CUA_SO)]
            trong_cs[off[i] + sau] = True
    ks2 = np.column_stack([kiem_soat, trong_cs.astype(float)])
    print(f"bộ kiểm soát: 4 biến gốc + chỉ báo lịch họp "
          f"({int(trong_cs.sum()):,} phiên trong cửa sổ sau họp)")
    print(f"phát hiện {int(pha.sum()):,} hàng · xác nhận {int(te.sum()):,} hàng\n")

    print(f"[1/4] Westfall–Young, {NPERM} hoán vị, null khối {KHOI} ngày…", flush=True)
    Z, L, nk, P, nguong = westfall_young(M, y, pha)
    print(f"      ngưỡng max|z| null khối: 90% {nguong[0]:.2f} · 95% {nguong[1]:.2f}")
    du_khop = nk >= MIN_KHOP
    song = (P < 0.05) & np.isfinite(Z)
    tho = (np.abs(np.nan_to_num(Z)) > 1.96) & np.isfinite(Z)
    print(f"      {int(du_khop.sum())}/{len(ten)} vị từ đủ {MIN_KHOP} lần khớp")
    print(f"      sống sót W-Y p<0,05: {int(song.sum())} / thô p<0,05: {int(tho.sum())}"
          f" (nếu toàn nhiễu kỳ vọng {0.05*np.isfinite(Z).sum():.0f})")

    qua_dk, qua_lopo, xn = [], [], []
    if song.sum() > 0:
        print(f"\n[2/4] Đối chứng có điều kiện (|t| > {T_DIEU_KIEN}, ĐÃ khử cả "
              f"lịch họp)…", flush=True)
        ung = []
        for i, c in zip(*np.where(song)):
            b, t = doi_chung(M[i], y, c, ks2, cum=cum)
            ung.append(dict(i=int(i), lop=int(c), ten=ten[i], n=int(nk[i]),
                            z=float(Z[i, c]), lift=float(L[i, c]),
                            p_wy=float(P[i, c]), b_dk=b, t_dk=t))
        qua_dk = [u for u in ung
                  if np.isfinite(u["t_dk"]) and abs(u["t_dk"]) > T_DIEU_KIEN]
        print(f"      {len(qua_dk)}/{len(ung)} vị từ còn tin riêng")
        print(f"\n      {'vị từ':<28}{'lớp':<10}{'n':>7}{'lift':>7}{'z':>8}{'t|đk':>8}")
        for u in sorted(ung, key=lambda x: -abs(x["z"]))[:10]:
            print(f"      {u['ten'][:26]:<28}{TEN_LOP[u['lop']]:<10}{u['n']:>7}"
                  f"{u['lift']:>7.3f}{u['z']:>8.2f}{u['t_dk']:>8.2f}")

        print(f"\n[3/4] Bỏ-một-cặp…", flush=True)
        for u in qua_dk:
            mi, lifts = M[u["i"]], []
            for p in B.PAIRS:
                mp = (cap == p) & pha & (y >= 0)
                kh = mi & mp
                if kh.sum() < 20:
                    lifts.append(np.nan); continue
                pc = (y[mp] == u["lop"]).mean()
                lifts.append(float((y[kh] == u["lop"]).mean() / max(pc, EPS)))
            lifts = np.array(lifts)
            nd = int(np.nansum(lifts > 1.0))
            u["lift_cap"], u["so_cap_duong"] = lifts.tolist(), nd
            u["lift_min"] = float(np.nanmin(lifts))
            if nd >= MIN_CAP_DUONG and np.nanmin(lifts) >= LIFT_LOPO:
                qua_lopo.append(u)
        print(f"      {len(qua_lopo)}/{len(qua_dk)} chuyển giao được")

        print("\n[4/4] Xác nhận trên KIỂM TRA…", flush=True)
        Zte, Lte, _ = z_lift(M, y, te)
        for u in qua_lopo:
            u["z_te"] = float(Zte[u["i"], u["lop"]])
            u["lift_te"] = float(Lte[u["i"], u["lop"]])
        xn = [u for u in qua_lopo if np.isfinite(u["z_te"]) and u["z_te"] > 1.96]
        print(f"      {len(xn)}/{len(qua_lopo)} tái lập")
    else:
        print("\n→ KHÔNG vị từ nào sống sót Westfall–Young.")

    print("\n" + "=" * 112)
    print(f"{'PHỄU H8c (chủ đề thông cáo FOMC)':<46}{'còn lại':>10}")
    for nhan, v in (("không gian giả thuyết", len(ten) * 3),
                    ("đủ số lần khớp", int(du_khop.sum())),
                    ("thô p<0,05", int(tho.sum())),
                    ("sống sót Westfall–Young", int(song.sum())),
                    ("còn tin riêng sau đối chứng (đã khử lịch)", len(qua_dk)),
                    ("chuyển giao được (LOPO)", len(qua_lopo)),
                    ("tái lập trên KIỂM TRA", len(xn))):
        print(f"{nhan:<46}{v:>10,}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(khong_gian=len(ten) * 3, du_khop=int(du_khop.sum()),
                    tho=int(tho.sum()), wy=int(song.sum()),
                    sau_dieu_kien=len(qua_dk), sau_lopo=len(qua_lopo),
                    xac_nhan=len(xn), quy_luat=xn, n_thong_cao=len(tc),
                    cua_so=list(CUA_SO),
                    phan_bo_chu_de=F.chu_de.value_counts(dropna=False).to_dict()),
              open(os.path.join(OUT, "h8c_chude.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/h8c_chude.json — {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
