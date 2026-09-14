"""HAR-J — them thanh phan lien tuc/nhay TACH BANG BIPOWER vao to hop san xuat.

Gia thuyet, cau hinh va tieu chi phu dinh DA CHOT TRUOC o `docs/HARJ_TIEUCHI.md`.

KHAC BCL DEM QUA: BCL tach nhay bang NGUONG DEM (threshold-crossing) tren
khung hoi quy don gian hoa. O day dung dung BIPOWER VARIATION (`bpv5`, da co
san trong `rv_adv.csv`, Barndorff-Nielsen & Shephard 2004) va TAI TAO DUNG
co che to hop that dang chay san xuat: moi mo hinh con khop OLS CUA SO MO
RONG rieng biet (Gram tich luy nhu `volfc2.he_so_cuon`), roi TRUNG BINH CONG
log-du bao qua cac mo hinh — dung cong thuc `g = L.mean(0)` cua
`volfc2.du_bao_san_xuat`.

Chay:  python src/kiem_harj.py
Ghi:   output/harj.json
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
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
EPS = 1e-12

import volfc2 as V2                                            # noqa: E402
from volfc import EPS as EPS_V, GAMMA, WIN_Z, MIN_FIT           # noqa: E402
from split import doan                                          # noqa: E402
from run_final7 import dm_nw                                    # noqa: E402


def qlike(rv, h):
    r = np.maximum(rv, EPS) / np.maximum(h, EPS)
    return r - np.log(np.maximum(r, EPS)) - 1.0


def _roll(v, w):
    return pd.Series(v).rolling(w).mean().values


def them_harj(X, d):
    """Them mo hinh HARJ vao dict thiet ke X (dung dinh dang cua thiet_ke)."""
    rv = np.maximum(d.rv5.values, EPS)
    bpv = np.maximum(d.bpv5.values, EPS)
    C = np.minimum(bpv, rv)
    J = np.maximum(0.0, rv - bpv)
    lc = np.log(np.maximum(C, EPS))
    lj = np.log1p(J / C)
    o = np.ones(len(d))
    lv = np.log(rv)
    lw = _roll(lv, 5)
    lm = _roll(lv, 22)
    X["HARJ"] = np.column_stack([o, lc, lj, lw, lm])
    return X, C, J


def tu_kiem(d):
    rv = np.maximum(d.rv5.values, EPS)
    bpv = np.maximum(d.bpv5.values, EPS)
    C = np.minimum(bpv, rv)
    J = np.maximum(0.0, rv - bpv)
    ok1 = bool((C <= rv + 1e-9).all())
    ok2 = bool((J >= -1e-9).all())
    print(f"  tự kiểm C_t ≤ RV_t mọi phiên: {'ĐẠT' if ok1 else 'HỎNG'}")
    print(f"  tự kiểm J_t ≥ 0 mọi phiên:    {'ĐẠT' if ok2 else 'HỎNG'}")
    return ok1 and ok2


def du_bao_to_hop(d, pair, ten_models):
    """Ban sao trung thuc cua `volfc2.du_bao_san_xuat`, nhung to hop la
    `ten_models` (co the khac MODELS goc), va co them mo hinh HARJ."""
    n = len(d)
    ngay = pd.DatetimeIndex(d.Date)
    lich = V2.nap_lich(ngay)
    lv = np.log(np.maximum(d.rv5.values, EPS))
    extra = V2._cot_su_kien(pair, lich, n, V2.CAUHINH_SANXUAT["event"])
    X = V2.thiet_ke(d, lv, extra or None)
    X, _, _ = them_harj(X, d)
    if extra:
        E = np.column_stack(extra)
        X["HARJ"] = np.column_stack([X["HARJ"], E])
    y = np.empty(n); y[:-1] = lv[1:]; y[-1] = np.nan

    gap = pd.Series(ngay).diff().dt.days.values.astype(float); gap[0] = 1
    lien = np.zeros(n, bool); lien[1:] = gap[1:] <= V2.MAX_GAP

    L, cnt_ok = [], np.ones(n, bool)
    he_so_cuoi = {}
    for m in ten_models:
        Xm = X[m]
        hople = np.isfinite(Xm).all(1) & np.isfinite(y)
        b, A, B, S, N = V2.he_so_cuon(Xm, y, hople, V2.CAUHINH_SANXUAT["window"])
        ssr = V2._ssr(b, A, B, S)
        s2 = np.where(N >= MIN_FIT, ssr / np.maximum(N, 1), np.nan)
        fit = np.einsum("tk,tk->t", Xm, b)
        L.append(np.clip(fit, -30, 0) + 0.5 * np.maximum(s2, 0))
        cnt_ok &= N >= MIN_FIT
        he_so_cuoi[m] = b[-1].copy() if len(b) else None
    L = np.stack(L)
    g = L.mean(0)
    ok = np.isfinite(L).all(0) & cnt_ok & (np.arange(n) >= V2.MIN_TRAIN)

    out = np.full(n, np.nan)
    src = np.where(ok)[0]; tgt = src + 1
    v = tgt < n
    out[tgt[v]] = np.exp(g[src[v]])
    return out, he_so_cuoi


# 4 cau hinh CHOT TRUOC (muc 4) — B0 khong tinh la moi
CAU_HINH = {
    "B0 mốc": ("STHARQ", "HARQ", "SHAR"),
    "B0+HARJ": ("STHARQ", "HARQ", "SHAR", "HARJ"),
    "HARJ thay SHAR": ("STHARQ", "HARQ", "HARJ"),
    "HARJ riêng": ("HARJ",),
}


def main():
    t0 = time.time()
    print("=" * 104)
    print("HAR-J — tách liên tục/nhảy bằng BIPOWER, tái tạo đúng cơ chế tổ hợp sản xuất")
    print("tiêu chí chốt trước: docs/HARJ_TIEUCHI.md")
    print("=" * 104)

    bang, chung = V2.nap_bang()
    print("\nTự kiểm C_t ≤ RV_t và J_t ≥ 0 (mỗi cặp):")
    dat_het = True
    for p in V2.PAIRS:
        ok = tu_kiem(bang[p])
        dat_het &= ok
        print(f"  {p}: {'ĐẠT' if ok else 'HỎNG'}")
    if not dat_het:
        sys.exit("tự kiểm HỎNG — dừng")

    ket_qua = {p: {} for p in V2.PAIRS}
    for p in V2.PAIRS:
        d = bang[p]
        g = doan(d.Date.values)
        rv_that = np.maximum(d.rv5.values, EPS)
        for ten, models in CAU_HINH.items():
            h, he_so = du_bao_to_hop(d, p, models)
            ql = qlike(rv_that, h)
            ket_qua[p][ten] = dict(ql=ql, g=g, he_so=he_so)

    print("\n" + "=" * 104)
    print(f"{'cấu hình':<20}{'QLIKE kđ':>11}{'so B0':>9}{'QLIKE kt':>11}"
          f"{'so B0':>9}{'DM p (kt)':>11}{'≥5/6 cặp':>10}")
    print("-" * 104)
    tong = {}
    for ten in CAU_HINH:
        vd_all, kt_all, b0_kt_all = [], [], []
        duong_cap = 0
        for p in V2.PAIRS:
            r = ket_qua[p][ten]; b0 = ket_qua[p]["B0 mốc"]
            g = r["g"]
            m_vd = (g == 1) & np.isfinite(r["ql"])
            m_kt = (g == 2) & np.isfinite(r["ql"]) & np.isfinite(b0["ql"])
            vd_all.append(r["ql"][m_vd])
            kt_all.append(r["ql"][m_kt])
            b0_kt_all.append(b0["ql"][m_kt])
            if m_kt.sum() > 30:
                duong_cap += int(r["ql"][m_kt].mean() < b0["ql"][m_kt].mean())
        vd = np.concatenate(vd_all); kt = np.concatenate(kt_all)
        b0kt = np.concatenate(b0_kt_all)
        d_vd = np.nan  # so B0 tren kiem dinh can moc rieng, in duoi
        p_dm = np.nan if ten == "B0 mốc" else dm_nw(kt - b0kt)[1]
        tong[ten] = dict(qlike_vd=float(vd.mean()), qlike_kt=float(kt.mean()),
                         dm_p=float(p_dm) if np.isfinite(p_dm) else None,
                         cap_duong=duong_cap)
        b0vd = tong["B0 mốc"]["qlike_vd"] if "B0 mốc" in tong else vd.mean()
        d_vd_pct = (vd.mean() / b0vd - 1) * 100
        d_kt_pct = (kt.mean() / tong["B0 mốc"]["qlike_kt"] - 1) * 100 if "B0 mốc" in tong else 0
        print(f"{ten:<20}{vd.mean():>11.4f}{d_vd_pct:>8.2f}%{kt.mean():>11.4f}"
              f"{d_kt_pct:>8.2f}%{(p_dm if np.isfinite(p_dm) else float('nan')):>11.4f}"
              f"{duong_cap:>7}/6")
    print("-" * 104)
    print("  (âm = TỐT HƠN mốc)")

    print("\nHỆ SỐ HARJ trên huấn luyện (phiên cuối) — H14 đòi log(C) > log(1+J/C):")
    for p in V2.PAIRS:
        b = ket_qua[p]["B0+HARJ"]["he_so"].get("HARJ")
        if b is not None:
            print(f"  {p:<9}hằng số {b[0]:+.4f}  log(C) {b[1]:+.4f}  "
                  f"log(1+J/C) {b[2]:+.4f}  tuần {b[3]:+.4f}  tháng {b[4]:+.4f}")

    # ── phan quyet theo muc 7
    tot = min((t for t in CAU_HINH if t != "B0 mốc"),
              key=lambda t: tong[t]["qlike_vd"])
    r = tong[tot]
    print(f"\nTỐT NHẤT TRÊN KIỂM ĐỊNH: {tot}")
    d_kt = (r["qlike_kt"] / tong["B0 mốc"]["qlike_kt"] - 1) * 100
    dk1 = (d_kt < 0) and (r["dm_p"] is not None and r["dm_p"] < 0.05 / 3)
    dk2 = r["cap_duong"] >= 5
    lc_gt_lj = [ket_qua[p]["B0+HARJ"]["he_so"]["HARJ"][1]
               > ket_qua[p]["B0+HARJ"]["he_so"]["HARJ"][2] for p in V2.PAIRS]
    dk3 = sum(lc_gt_lj) >= 4
    print("\n" + "=" * 104)
    print("PHÁN QUYẾT theo TIÊU CHÍ CHỐT TRƯỚC (docs/HARJ_TIEUCHI.md mục 7)")
    print("-" * 104)
    print(f"  ĐK1 thắng kiểm tra & p < 0,0167 (Bonferroni 3)  "
          f"{'ĐẠT' if dk1 else 'TRƯỢT'}   ({d_kt:+.2f}%, p={r['dm_p']})")
    print(f"  ĐK2 ≥ 5/6 cặp cải thiện                         "
          f"{'ĐẠT' if dk2 else 'TRƯỢT'}   ({r['cap_duong']}/6)")
    print(f"  ĐK3 log(C) > log(1+J/C) ở ≥ 4/6 cặp             "
          f"{'ĐẠT' if dk3 else 'TRƯỢT'}   ({sum(lc_gt_lj)}/6)")
    print("-" * 104)
    xong = dk1 and dk2 and dk3
    print(f"  → {'DƯƠNG' if xong else 'KHÔNG đủ điều kiện dương — KHÔNG đổi sản xuất'}")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"tong": tong, "chon_kiem_dinh": tot,
               "dk": dict(dk1=bool(dk1), dk2=bool(dk2), dk3=bool(dk3)),
               "phan_quyet": "duong" if xong else "am"},
              open(os.path.join(OUT, "harj.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/harj.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
