"""H5 — CHE DO LAM LOP DIEU KIEN, ho thu nam cua Giai doan 2.

REPLAN_2026.md muc 3.1 mo ta H5 la "che do lam lop dieu kien" (HMM/diem ngat).
Khong co hmmlearn trong moi truong nay; va quan trong hon, NEU dung che do la
tam phan vi cua chinh sigma^ (nhu SigmaCheDo/A3) thi H5 se TRUNG LAP voi chinh
khong gian 630 vi tu cua run_quyluat.py — sigma^ DA LA mot trong 12 dac trung,
va vet_can() DA sinh san moi to hop "dac_trung X & sigma^ o o Y" roi.

Nen H5 o day dung MOT TRUC CHE DO KHAC HAN, khong phai bien dong: CHE DO TU
TUONG QUAN cua loi suat — tuong quan bac 1 (lag-1) tren cua so cuon 20 phien,
NHAN QUA (chi dung den het phien t). Tam phan vi CHOT tren huan luyen, RIENG
TUNG CAP:
  che do 0: tu tuong quan AM (co xu huong dao chieu ngan han)
  che do 1: trung tinh
  che do 2: tu tuong quan DUONG (co xu huong noi tiep — "trending")

KHONG GIAN GIA THUYET: dung LAI dung 630 vi tu (1-2 menh de) cua run_quyluat.py
— nhung MOI vi tu duoc GIAO them voi MOI che do trong 3 che do o tren, thanh
630 x 3 = 1.890 vi tu moi x 3 lop = 5.670 gia thuyet. Cau hoi: co vi tu nao chi
lo ra KHI da tach theo che do tu tuong quan, ma bi che khuat khi gop chung
khong?

Chay:  python src/run_h5_chedo.py
Ghi:   output/h5_chedo.json
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
from split import doan                                        # noqa: E402
from run_quyluat import (                                     # noqa: E402
    nap_du_lieu, dac_trung, roi_rac, vet_can, z_lift, westfall_young,
    doi_chung, la_vi_tu_nen, TEN_LOP, TEN_KIEM_SOAT, MIN_KHOP, LIFT_LOPO,
    MIN_CAP_DUONG, T_DIEU_KIEN, NPERM, KHOI, SEED, EPS,
)

N_CD = 3
CD_TEN = ("tự-tương-quan âm", "trung tính", "tự-tương-quan dương")


def acf1_cuon(r, cua_so=20):
    """Tu tuong quan lag-1, cua so cuon `cua_so` phien, KET THUC tai t (nhan
    qua). NaN o dau chuoi hoac thieu du lieu."""
    n = len(r)
    ra = np.full(n, np.nan)
    for t in range(cua_so, n):
        w = r[t - cua_so + 1:t + 1]
        if not np.isfinite(w).all() or w.std() < EPS:
            continue
        a, b = w[:-1], w[1:]
        if a.std() < EPS or b.std() < EPS:
            continue
        ra[t] = float(np.corrcoef(a, b)[0, 1])
    return ra


def che_do_tu_tuong_quan(Ms, dts):
    """Tra ve (N,) nhan che do 0/1/2, nguong CHOT tren huan luyen RIENG TUNG CAP."""
    ra = []
    for i, p in enumerate(B.PAIRS):
        c = Ms[i].close.values
        r = np.r_[np.nan, np.diff(np.log(np.maximum(c, EPS)))]
        acf = acf1_cuon(r)
        tri = doan(dts[i]) == 0
        v = acf[tri & np.isfinite(acf)]
        if len(v) < 200:
            ra.append(np.full(len(c), -1)); continue
        nguong = np.quantile(v, [1 / 3, 2 / 3])
        lab = np.where(np.isfinite(acf), np.digitize(acf, nguong), -1)
        ra.append(lab)
    return np.concatenate(ra)


def la_vi_tu_nen_h5(ten_vt):
    """Nhu la_vi_tu_nen() nhung bo qua menh de che-do-tu-tuong-quan da them —
    chi xet phan GOC (truoc dau '['), vi che do tu tuong quan KHONG PHAI sigma^
    nen tu no khong lam vi tu "chinh nen"."""
    goc = str(ten_vt).split(" [")[0]
    return la_vi_tu_nen(goc)


def main():
    t0 = time.time()
    print("=" * 112)
    print("H5 — CHẾ ĐỘ TỰ TƯƠNG QUAN LÀM LỚP ĐIỀU KIỆN, họ thứ năm của Giai đoạn 2")
    print("=" * 112)
    du = nap_du_lieu()
    Ms, sigs, zs, dts = du["Ms"], du["sigs"], du["zs"], du["dts"]
    y, cap, dt = du["y"], du["cap"], du["dt"]
    kiem_soat, cum = du["kiem_soat"], du["cum"]
    tr, va, te, pha = du["tr"], du["va"], du["te"], du["pha"]

    lit_all, ten_lit = [], None
    for i, p in enumerate(B.PAIRS):
        F = dac_trung(Ms[i], sigs[i], zs[i])
        tri = doan(dts[i]) == 0
        L, tn = roi_rac(F, tri)
        lit_all.append(L)
        ten_lit = tn
    lit = np.concatenate(lit_all, axis=1)
    M0, ten0 = vet_can(lit, ten_lit)
    print(f"không gian gốc (dùng lại run_quyluat.py): {len(ten0)} vị từ")

    print("Đang tính chế độ tự-tương-quan (cửa sổ cuộn 20 phiên, chốt trên "
          "huấn luyện)…", flush=True)
    che_do = che_do_tu_tuong_quan(Ms, dts)
    for v in range(N_CD):
        print(f"  {CD_TEN[v]:<22} n={int((che_do==v).sum()):,}")

    M = np.concatenate([M0 & (che_do == v)[None, :] for v in range(N_CD)], axis=0)
    ten = [f"{t} [{CD_TEN[v]}]" for v in range(N_CD) for t in ten0]
    print(f"\nKHÔNG GIAN GIẢ THUYẾT H5: {len(ten0)} vị từ × {N_CD} chế độ = "
          f"{len(ten):,} vị từ × 3 lớp = {len(ten)*3:,} giả thuyết")
    print(f"phát hiện {int(pha.sum()):,} hàng · xác nhận {int(te.sum()):,} hàng\n")

    print(f"[1/4] Westfall–Young, {NPERM} hoán vị, null khối {KHOI} ngày…",
          flush=True)
    Z, L, nk, P, nguong = westfall_young(M, y, pha)
    du_khop = nk >= MIN_KHOP
    song = (P < 0.05) & np.isfinite(Z)
    tho = (np.abs(np.nan_to_num(Z)) > 1.96) & np.isfinite(Z)
    print(f"      {int(du_khop.sum()):,}/{len(ten):,} vị từ đủ {MIN_KHOP} lần khớp")
    print(f"      sống sót W-Y p<0,05: {int(song.sum())} / thô p<0,05: {int(tho.sum())}"
          f" (nếu toàn nhiễu kỳ vọng {0.05*np.isfinite(Z).sum():.0f})")

    qua_dk, qua_lopo, xn = [], [], []
    if song.sum() > 0:
        print(f"\n[2/4] Đối chứng có điều kiện (|t| > {T_DIEU_KIEN})…", flush=True)
        ung = []
        for i, c in zip(*np.where(song)):
            b, t = doi_chung(M[i], y, c, kiem_soat, cum=cum)
            ung.append(dict(i=int(i), lop=int(c), ten=ten[i], n=int(nk[i]),
                            z=float(Z[i, c]), lift=float(L[i, c]), p_wy=float(P[i, c]),
                            b_dk=b, t_dk=t))
        n_nen = sum(1 for u in ung if la_vi_tu_nen_h5(u["ten"]))
        qua_dk = [u for u in ung if not la_vi_tu_nen_h5(u["ten"])
                  and np.isfinite(u["t_dk"]) and abs(u["t_dk"]) > T_DIEU_KIEN]
        print(f"      {n_nen}/{len(ung)} vị từ là chính σ̂ (loại) — "
              f"{len(qua_dk)}/{len(ung)-n_nen} còn tin riêng sau điều kiện hoá")

        print(f"\n[3/4] Bỏ-một-cặp (lift ≥ {LIFT_LOPO}, ≥{MIN_CAP_DUONG}/6 cặp dương)…",
              flush=True)
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
        print(f"      {len(qua_lopo)}/{len(qua_dk)} chuyển giao được qua các cặp")

        print("\n[4/4] Xác nhận trên đoạn KIỂM TRA…", flush=True)
        Zte, Lte, nkte = z_lift(M, y, te)
        for u in qua_lopo:
            u["z_te"] = float(Zte[u["i"], u["lop"]])
            u["lift_te"] = float(Lte[u["i"], u["lop"]])
        xn = [u for u in qua_lopo if np.isfinite(u["z_te"]) and u["z_te"] > 1.96]
        print(f"      {len(xn)}/{len(qua_lopo)} tái lập trên kiểm tra")
    else:
        print("\n→ KHÔNG vị từ nào sống sót Westfall–Young.")

    print("\n" + "=" * 112)
    print(f"{'PHỄU H5':<46}{'còn lại':>10}")
    for nhan, v in (("không gian giả thuyết", len(ten) * 3),
                    ("đủ số lần khớp", int(du_khop.sum())),
                    ("thô p<0,05", int(tho.sum())),
                    ("sống sót Westfall–Young", int(song.sum())),
                    ("còn tin riêng sau đối chứng", len(qua_dk)),
                    ("chuyển giao được (LOPO)", len(qua_lopo)),
                    ("tái lập trên KIỂM TRA", len(xn))):
        print(f"{nhan:<46}{v:>10,}")

    if xn:
        print("\nTHƯ VIỆN QUY LUẬT H5")
        for u in sorted(xn, key=lambda x: -x["z_te"])[:15]:
            print(f"  {u['ten'][:60]:<62}{TEN_LOP[u['lop']]:<10}n={u['n']:<6}"
                  f"lift={u['lift']:.3f}  t|đk={u['t_dk']:.2f}  z_kt={u['z_te']:.2f}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(khong_gian=len(ten) * 3, du_khop=int(du_khop.sum()),
                    tho=int(tho.sum()), wy=int(song.sum()),
                    sau_dieu_kien=len(qua_dk), sau_lopo=len(qua_lopo),
                    xac_nhan=len(xn), quy_luat=xn),
              open(os.path.join(OUT, "h5_chedo.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/h5_chedo.json — {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
