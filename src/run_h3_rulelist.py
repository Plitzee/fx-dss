"""H3 — RULE LIST (CART nong, doc duoc), ho thu ba cua Giai doan 2.

Khac voi khong gian nguong-tren-tung-dac-trung cua run_quyluat.py (menh de
DOC LAP tung dac trung, toi da hai menh de), H3 hoi: mot CAY QUYET DINH nong
(may hoc TU DONG chon dac trung va nguong bang toi uu hoa, khong phai con
nguoi chon truoc) co tim duoc TO HOP dac trung nao tot hon khong gian thu
cong o tren khong?

KHONG GIAN GIA THUYET, LIET KE TRUOC (CHOT truoc khi nhin ket qua):
  max_depth = 3 (toi da 8 la), min_samples_leaf = 200 — MOT cau hinh duy nhat,
  khong do nhieu depth roi chon dep nhat (do se la "chay CART roi lay la" ma
  chinh run_quyluat.py canh bao o dau file — pha vo nguyen tac liet ke truoc).
  So gia thuyet = so la thuc te sinh ra (<= 8) x 3 lop, in ra khi chay.

Dung LAI dung 12 dac trung doc duoc cua run_quyluat.py (dac_trung()), nhung
CHUYEN VE HANG PHAN VI (rank tren HUAN LUYEN, RIENG TUNG CAP) truoc khi dua
vao cay — vi thang do goc khac nhau giua cac cap (vd sigma^ JPY vs EUR lech
bac do lon, xem docstring kiem_soat_sigma() trong run_quyluat.py), gop truc
tiep se lam cay chi hoc duoc su khac biet GIUA CAC CAP thay vi trong cap.

Chay:  python src/run_h3_rulelist.py
Ghi:   output/h3_rulelist.json
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
    nap_du_lieu, dac_trung, z_lift, westfall_young, doi_chung, la_vi_tu_nen,
    TEN_LOP, TEN_KIEM_SOAT, MIN_KHOP, LIFT_LOPO, MIN_CAP_DUONG, T_DIEU_KIEN,
    NPERM, KHOI, SEED, EPS,
)

MAX_DEPTH = 3            # CHOT TRUOC — khong do nhieu do sau roi chon
MIN_LA = 200


def hang_phan_vi(F, tr_per_pair, Ms):
    """Chuyen moi dac trung ve HANG PHAN VI trong [0,1], nguong tu phan vi
    thuc nghiem chot tren HUAN LUYEN CUA TUNG CAP, ap dung nguyen sang toan
    chuoi (khong ro ri — giong het tinh than roi_rac() nhung lien tuc thay vi
    ba o)."""
    ra = {}
    for k, v in F.items():
        v = np.asarray(v, float)
        vt = v[tr_per_pair & np.isfinite(v)]
        if len(vt) < 200:
            continue
        vt_sort = np.sort(vt)
        r = np.searchsorted(vt_sort, v, side="right") / max(len(vt_sort), 1)
        r = np.clip(r, 0.0, 1.0)
        r[~np.isfinite(v)] = np.nan
        ra[k] = r
    return ra


def xay_cay(zs, dts, Ms, sigs, tr_mask):
    """Tra ve (X (N,F) hang phan vi gop 6 cap, ten dac trung, mask ok)."""
    ten_dt = None
    X_all = []
    for i, p in enumerate(B.PAIRS):
        F = dac_trung(Ms[i], sigs[i], zs[i])
        tri = doan(dts[i]) == 0
        R = hang_phan_vi(F, tri, Ms[i])
        ten_dt = list(R.keys())
        X_all.append(np.column_stack([R[k] for k in ten_dt]))
    X = np.concatenate(X_all, axis=0)
    ok = np.isfinite(X).all(1)
    return X, ten_dt, ok


def main():
    t0 = time.time()
    print("=" * 112)
    print("H3 — RULE LIST (CART nông, đọc được), họ thứ ba của Giai đoạn 2")
    print("=" * 112)
    du = nap_du_lieu()
    zs, dts, Ms, sigs = du["zs"], du["dts"], du["Ms"], du["sigs"]
    y, cap, dt = du["y"], du["cap"], du["dt"]
    kiem_soat, cum = du["kiem_soat"], du["cum"]
    tr, va, te, pha = du["tr"], du["va"], du["te"], du["pha"]

    print("Đang tính đặc trưng + chuyển hạng phân vị (chốt trên huấn luyện)…",
          flush=True)
    X, ten_dt, ok = xay_cay(zs, dts, Ms, sigs, tr)
    m_hop = pha & ok
    print(f"đặc trưng: {', '.join(ten_dt)}")
    print(f"{int(m_hop.sum()):,}/{len(y):,} hàng đủ toàn bộ {len(ten_dt)} đặc trưng")

    from sklearn.tree import DecisionTreeClassifier
    cay = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LA,
                                  random_state=0)
    cay.fit(X[tr & ok], y[tr & ok])
    X_dam = np.where(np.isfinite(X), X, 0.0)   # apply() khong chiu duoc NaN;
    la = cay.apply(X_dam)                      # hang NaN se bi loai bang `ok` sau
    la_id = sorted(set(la[tr & ok].tolist()))
    print(f"cây (max_depth={MAX_DEPTH}, min_samples_leaf={MIN_LA}) sinh "
          f"{len(la_id)} lá")

    lit = np.array([(la == lid) & ok for lid in la_id])
    ten = [f"lá CART #{i}" for i in range(len(la_id))]
    print(f"KHÔNG GIAN GIẢ THUYẾT H3: {len(ten)} vị từ × 3 lớp = "
          f"{len(ten)*3} giả thuyết — liệt kê đầy đủ, biết trước "
          f"(cấu hình max_depth/min_samples_leaf CHỐT TRƯỚC khi chạy)")
    print(f"phát hiện {int(m_hop.sum()):,} hàng · xác nhận "
          f"{int((te&ok).sum()):,} hàng\n")

    print(f"[1/4] Westfall–Young, {NPERM} hoán vị, null khối {KHOI} ngày…",
          flush=True)
    Z, L, nk, P, nguong = westfall_young(lit, y, m_hop)
    du_khop = nk >= MIN_KHOP
    song = (P < 0.05) & np.isfinite(Z)
    tho = (np.abs(np.nan_to_num(Z)) > 1.96) & np.isfinite(Z)
    print(f"      {int(du_khop.sum())}/{len(ten)} vị từ đủ {MIN_KHOP} lần khớp")
    print(f"      sống sót W-Y p<0,05: {int(song.sum())} / thô p<0,05: {int(tho.sum())}"
          f" (nếu toàn nhiễu kỳ vọng {0.05*np.isfinite(Z).sum():.0f})")

    qua_dk, qua_lopo, xn = [], [], []
    if song.sum() > 0:
        print(f"\n[2/4] Đối chứng có điều kiện (|t| > {T_DIEU_KIEN})…", flush=True)
        ung = []
        for i, c in zip(*np.where(song)):
            b, t = doi_chung(lit[i], y, c, kiem_soat, cum=cum)
            ung.append(dict(i=int(i), lop=int(c), ten=ten[i], n=int(nk[i]),
                            z=float(Z[i, c]), lift=float(L[i, c]), p_wy=float(P[i, c]),
                            b_dk=b, t_dk=t))
        qua_dk = [u for u in ung if np.isfinite(u["t_dk"]) and abs(u["t_dk"]) > T_DIEU_KIEN]
        print(f"      {len(qua_dk)}/{len(ung)} vị từ còn tin riêng sau điều kiện hoá")

        print(f"\n[3/4] Bỏ-một-cặp (lift ≥ {LIFT_LOPO}, ≥{MIN_CAP_DUONG}/6 cặp dương)…",
              flush=True)
        for u in qua_dk:
            mi, lifts = lit[u["i"]], []
            for p in B.PAIRS:
                mp = (cap == p) & m_hop
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
        m_te = te & ok
        Zte, Lte, nkte = z_lift(lit, y, m_te)
        for u in qua_lopo:
            u["z_te"] = float(Zte[u["i"], u["lop"]])
            u["lift_te"] = float(Lte[u["i"], u["lop"]])
        xn = [u for u in qua_lopo if np.isfinite(u["z_te"]) and u["z_te"] > 1.96]
        print(f"      {len(xn)}/{len(qua_lopo)} tái lập trên kiểm tra")
    else:
        print("\n→ KHÔNG vị từ nào sống sót Westfall–Young.")

    print("\n" + "=" * 112)
    print(f"{'PHỄU H3':<46}{'còn lại':>10}")
    for nhan, v in (("không gian giả thuyết", len(ten) * 3),
                    ("đủ số lần khớp", int(du_khop.sum())),
                    ("thô p<0,05", int(tho.sum())),
                    ("sống sót Westfall–Young", int(song.sum())),
                    ("còn tin riêng sau đối chứng", len(qua_dk)),
                    ("chuyển giao được (LOPO)", len(qua_lopo)),
                    ("tái lập trên KIỂM TRA", len(xn))):
        print(f"{nhan:<46}{v:>10,}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(khong_gian=len(ten) * 3, du_khop=int(du_khop.sum()),
                    tho=int(tho.sum()), wy=int(song.sum()),
                    sau_dieu_kien=len(qua_dk), sau_lopo=len(qua_lopo),
                    xac_nhan=len(xn), quy_luat=xn, so_la=len(la_id),
                    max_depth=MAX_DEPTH, min_samples_leaf=MIN_LA,
                    dac_trung=ten_dt),
              open(os.path.join(OUT, "h3_rulelist.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/h3_rulelist.json — {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
