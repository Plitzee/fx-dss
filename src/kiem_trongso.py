"""TRONG SO NGAY trong to hop san xuat — them LV va/hoac doi sang Hedge.

Gia thuyet, cau hinh va tieu chi phu dinh DA CHOT TRUOC o
`docs/TOHOP_TRONGSO_TIEUCHI.md`.

BANG CHUNG LOAI TRU: BCL va HAR-J deu thu tach thanh phan NHAY de giai thich
khoang cach ~7-8% QLIKE (them log rv5(t) tho cho -7,78%) — CA HAI THAT BAI.
O day thu giai thich khac: TO HOP dang dinh trong so ngay duoi toi uu (ba mo
hinh con dinh trong so ngay ba cach khac nhau, roi trung binh cong don gian).

Tai tao dung co che `du_bao_san_xuat()` (nhu `kiem_harj.py`), them:
  - mo hinh LV: HAR co dien thuan Corsi (2009), [1, lv, lw, lm]
  - trong so Hedge: mu tren ton that QLIKE qua khu, eta=0,5 (dung nhu
    balop.ToHopTrucTuyen da dung cho tang xac suat)

Chay:  python src/kiem_trongso.py
Ghi:   output/trongso.json
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
ETA = 0.5          # chot truoc — dung eta da dung cho balop.ToHopTrucTuyen


def qlike(rv, h):
    r = np.maximum(rv, EPS) / np.maximum(h, EPS)
    return r - np.log(np.maximum(r, EPS)) - 1.0


import volfc2 as V2                                            # noqa: E402
from volfc import MIN_FIT                                      # noqa: E402
from split import doan                                          # noqa: E402
from run_final7 import dm_nw                                    # noqa: E402


def them_lv(X, d):
    lv = np.log(np.maximum(d.rv5.values, EPS))
    lw = pd.Series(lv).rolling(5).mean().values
    lm = pd.Series(lv).rolling(22).mean().values
    o = np.ones(len(d))
    X["LV"] = np.column_stack([o, lv, lw, lm])
    return X


def cac_du_bao_thanh_vien(d, pair, ten_models):
    """Tra ve (L, rv_that) — L[i] = mang log-du bao cua mo hinh thu i cho MOI
    phien (chua tong hop), dung dinh nghia giong het `du_bao_san_xuat`."""
    n = len(d)
    ngay = pd.DatetimeIndex(d.Date)
    lich = V2.nap_lich(ngay)
    lv = np.log(np.maximum(d.rv5.values, EPS))
    extra = V2._cot_su_kien(pair, lich, n, V2.CAUHINH_SANXUAT["event"])
    X = V2.thiet_ke(d, lv, extra or None)
    X = them_lv(X, d)
    if extra:
        E = np.column_stack(extra)
        X["LV"] = np.column_stack([X["LV"], E])
    y = np.empty(n); y[:-1] = lv[1:]; y[-1] = np.nan

    L, cnt_ok = [], np.ones(n, bool)
    for m in ten_models:
        Xm = X[m]
        hople = np.isfinite(Xm).all(1) & np.isfinite(y)
        b, A, B, S, N = V2.he_so_cuon(Xm, y, hople, V2.CAUHINH_SANXUAT["window"])
        ssr = V2._ssr(b, A, B, S)
        s2 = np.where(N >= MIN_FIT, ssr / np.maximum(N, 1), np.nan)
        fit = np.einsum("tk,tk->t", Xm, b)
        L.append(np.clip(fit, -30, 0) + 0.5 * np.maximum(s2, 0))
        cnt_ok &= N >= MIN_FIT
    L = np.stack(L)               # (k, n) — L[j, t] = log-du bao cho NGAY t+1, biet tai t
    ok = np.isfinite(L).all(0) & cnt_ok & (np.arange(n) >= V2.MIN_TRAIN)
    rv_that = np.maximum(d.rv5.values, EPS)
    return L, ok, rv_that


def to_hop_deu(L, ok):
    g = np.where(ok, L.mean(0), np.nan)
    return _dich_chuyen(g, ok)


def to_hop_hedge(L, ok, rv_that, eta=ETA):
    """Hedge: trong so mu tren ton that QLIKE cua CHINH phien vua biet ket cuc.

    Tai moi t hop le: du bao cho t+1 dung trong so w(t) (da chuan hoa). Sau khi
    biet RV that cua NGAY t (da co tai buoc t, vi la qua khu), cap nhat w cho
    buoc t+1 bang ton that QLIKE cua tung mo hinh cho CHINH ngay t — khong
    nhin tuong lai: w dung de tao du bao cho t+1 chi phu thuoc ton that tai
    cac ngay <= t.
    """
    k, n = L.shape
    w = np.full(k, 1.0 / k)
    g = np.full(n, np.nan)
    for t in range(n):
        if not ok[t]:
            continue
        g[t] = float(np.dot(w, L[:, t]))
        # cap nhat w bang ton that cua CHINH mo hinh tai t, dung khi da co ket
        # cuc that cua ngay t (du bao L[:,t] la cho ngay t+1, nhung du bao DO
        # DUOC LAM tai buoc t-1 cho ngay t — tuc de cap nhat dung nhan qua,
        # phai dung ton that cua du bao DA LAM cho ngay t, khong phai t+1)
        if t > 0 and ok[t - 1] and np.isfinite(rv_that[t]):
            h_t = np.exp(np.clip(L[:, t - 1], -30, 5))   # du bao cua tung mo
            # hinh cho NGAY t, lam tai t-1
            ql_t = qlike(rv_that[t], h_t)
            w = w * np.exp(-eta * ql_t)
            w = w / w.sum()
    return _dich_chuyen(g, ok)


def _dich_chuyen(g, ok):
    n = len(g)
    out = np.full(n, np.nan)
    src = np.where(ok)[0]; tgt = src + 1
    v = tgt < n
    out[tgt[v]] = np.exp(g[src[v]])
    return out


CAU_HINH = {
    "B0 mốc": (("STHARQ", "HARQ", "SHAR"), "đều"),
    "B0+LV": (("STHARQ", "HARQ", "SHAR", "LV"), "đều"),
    "B0 Hedge": (("STHARQ", "HARQ", "SHAR"), "hedge"),
    "B0+LV Hedge": (("STHARQ", "HARQ", "SHAR", "LV"), "hedge"),
}


def main():
    t0 = time.time()
    print("=" * 104)
    print("TRỌNG SỐ NGÀY trong tổ hợp sản xuất — thêm LV và/hoặc Hedge")
    print("tiêu chí chốt trước: docs/TOHOP_TRONGSO_TIEUCHI.md")
    print("=" * 104)

    bang, chung = V2.nap_bang()
    ket_qua = {p: {} for p in V2.PAIRS}
    for p in V2.PAIRS:
        d = bang[p]
        g_doan = doan(d.Date.values)
        for ten, (models, kieu) in CAU_HINH.items():
            L, ok, rv_that = cac_du_bao_thanh_vien(d, p, models)
            h = to_hop_deu(L, ok) if kieu == "đều" else to_hop_hedge(L, ok, rv_that)
            ql = qlike(rv_that, h)
            ket_qua[p][ten] = dict(ql=ql, g=g_doan)

    print("\n" + "=" * 104)
    print(f"{'cấu hình':<16}{'QLIKE kđ':>11}{'so B0':>9}{'QLIKE kt':>11}"
          f"{'so B0':>9}{'DM p (kt)':>11}{'≥5/6 cặp':>10}")
    print("-" * 104)
    tong = {}
    for ten in CAU_HINH:
        vd_all, kt_all, b0_kt_all = [], [], []
        duong_cap = 0
        for p in V2.PAIRS:
            r = ket_qua[p][ten]; b0 = ket_qua[p]["B0 mốc"]
            gg = r["g"]
            m_vd = (gg == 1) & np.isfinite(r["ql"])
            m_kt = (gg == 2) & np.isfinite(r["ql"]) & np.isfinite(b0["ql"])
            vd_all.append(r["ql"][m_vd]); kt_all.append(r["ql"][m_kt])
            b0_kt_all.append(b0["ql"][m_kt])
            if m_kt.sum() > 30:
                duong_cap += int(r["ql"][m_kt].mean() < b0["ql"][m_kt].mean())
        vd = np.concatenate(vd_all); kt = np.concatenate(kt_all)
        b0kt = np.concatenate(b0_kt_all)
        p_dm = np.nan if ten == "B0 mốc" else dm_nw(kt - b0kt)[1]
        tong[ten] = dict(qlike_vd=float(vd.mean()), qlike_kt=float(kt.mean()),
                         dm_p=float(p_dm) if np.isfinite(p_dm) else None,
                         cap_duong=duong_cap)
        b0vd = tong["B0 mốc"]["qlike_vd"]
        b0ktv = tong["B0 mốc"]["qlike_kt"]
        d_vd = (vd.mean() / b0vd - 1) * 100
        d_kt = (kt.mean() / b0ktv - 1) * 100
        print(f"{ten:<16}{vd.mean():>11.4f}{d_vd:>8.2f}%{kt.mean():>11.4f}"
              f"{d_kt:>8.2f}%{(p_dm if np.isfinite(p_dm) else float('nan')):>11.4f}"
              f"{duong_cap:>7}/6")
    print("-" * 104)
    print("  (âm = TỐT HƠN mốc)")

    tot = min((t for t in CAU_HINH if t != "B0 mốc"),
              key=lambda t: tong[t]["qlike_vd"])
    r = tong[tot]
    d_kt = (r["qlike_kt"] / tong["B0 mốc"]["qlike_kt"] - 1) * 100
    dk1 = (d_kt < 0) and (r["dm_p"] is not None and r["dm_p"] < 0.05 / 3)
    dk2 = r["cap_duong"] >= 5
    print(f"\nTỐT NHẤT TRÊN KIỂM ĐỊNH: {tot}")
    print("\n" + "=" * 104)
    print("PHÁN QUYẾT theo TIÊU CHÍ CHỐT TRƯỚC (docs/TOHOP_TRONGSO_TIEUCHI.md mục 6)")
    print("-" * 104)
    print(f"  ĐK1 thắng kiểm tra & p < 0,0167 (Bonferroni 3)  "
          f"{'ĐẠT' if dk1 else 'TRƯỢT'}   ({d_kt:+.2f}%, p={r['dm_p']})")
    print(f"  ĐK2 ≥ 5/6 cặp cải thiện                         "
          f"{'ĐẠT' if dk2 else 'TRƯỢT'}   ({r['cap_duong']}/6)")
    print("-" * 104)
    xong = dk1 and dk2
    print(f"  → {'DƯƠNG' if xong else 'KHÔNG đủ điều kiện dương — KHÔNG đổi sản xuất'}")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"tong": tong, "chon_kiem_dinh": tot,
               "dk": dict(dk1=bool(dk1), dk2=bool(dk2)),
               "phan_quyet": "duong" if xong else "am"},
              open(os.path.join(OUT, "trongso.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/trongso.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
