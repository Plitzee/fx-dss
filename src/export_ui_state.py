"""TANG 7 — XUAT TRANG THAI CHO UI (Next.js + Vercel Python function).

Tinh MOT LAN, ngoai luong, moi thu ma api/decision.py can de dung lai
mot phieu quyet dinh MA KHONG can pandas/scipy.stats.t.fit o request-time
(chi con mot cuoc goi scipy.stats.t.cdf nhe trong p_cham_stop, va mot lan
giai() DP moi request — do la phan BAT BUOC phai tinh lai vi carry phu
thuoc NGAY nguoi dung chon).

DUNG DUNG PHAN CHIA CHINH THUC (split.py, VALID_TU) de huan luyen tham so
— khong dung lai bo 70% tuy tien cua tu kiem decision_record.py. Day la
mot cai thien nho an theo: tu kiem chi la vi du minh hoa, con UI la mat
tien nguoi dung that su dung, nen phai dung dung quy uoc chinh thuc.

Xuat ra web/api/_data/:
  meta.json     — tham so tung cap (s1/s2/s_stress, nu, c_thoat_sigma,
                  bang khoang conformal, bang tam han, luat_ky_hieu q,
                  ngay bat dau duoc chon (valid_tu)) + slip_sigma toan cuc
  regime.json   — mau (pooled 6 cap, 3 che do), nguong, sig_v — dung truc
                  tiep cho optimal_stop.giai()/nen_giu() moi request
  series_{PAIR}.json — chuoi day du theo ngay: sig, zT, zL, rv5, close,
                  carry (da noi suy) — dung de doc du lieu "hom nay" va
                  tinh carry cua so mo rong tai request-time (numpy thuan,
                  so sanh chuoi ISO date, khong can parse ngay)

Chay:  python src/export_ui_state.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "web", "api", "_data")
os.makedirs(OUT, exist_ok=True)

import optimal_stop as O
from split import VALID_TU
import position_sizing as PS
from position_sizing import PositionSizer
from decision_record import KhoangConformal, TamHan, LuatKyHieu


def main():
    pan = O.nap_panel()
    tr = np.asarray(pan.Date < VALID_TU)
    print(f"huấn luyện: {int(tr.sum()):,} phiên (< {VALID_TU.date()}), "
          f"toàn panel: {len(pan):,} phiên, {len(O.PAIRS)} cặp")

    # ── tầng 6b: mau/nguong/sig_v pooled, dùng CHUNG cho mọi cặp ──
    mau, nguong = O.rut_mau(pan, tr)
    sig_v = O.sigma_che_do(pan, tr, nguong)
    slip_sigma, dist_dung, n_slip = O.truot_trung_binh_sigma()
    print(f"trượt giá kỳ vọng (quy sigma): {slip_sigma:.4f}  (đo từ {n_slip:,} lần chạm, "
          f"khoảng cách gần {dist_dung:.4f})")

    regime = {
        "nguong": nguong.tolist(),
        "sig_v": sig_v.tolist(),
        "mau": {
            str(v): {k: mau[v][k].tolist() for k in ("zT", "zL", "ty", "v2")}
            for v in range(O.N_CHE_DO)
        },
    }
    n_mau_total = sum(len(mau[v]["zT"]) for v in range(O.N_CHE_DO))
    print(f"mẫu tầng 6b (pooled 6 cặp): {n_mau_total:,} dòng qua {O.N_CHE_DO} chế độ "
          f"({[len(mau[v]['zT']) for v in range(O.N_CHE_DO)]})")
    with open(os.path.join(OUT, "regime.json"), "w") as f:
        json.dump(regime, f)

    # Hang so tu position_sizing.py/sizing.py — xuat truc tiep tu module (khong
    # chep tay) de neu file goc doi thi lan export sau tu dong khop lai.
    meta = {"valid_tu": str(VALID_TU.date()), "slip_sigma": float(slip_sigma),
            "n_che_do": O.N_CHE_DO,
            "const": dict(
                k_vol_hi=PS.K_VOL_HI, k_vol_lo=PS.K_VOL_LO,
                k_dd_hi=PS.K_DD_HI, k_dd_slope=PS.K_DD_SLOPE, k_dd_floor=PS.K_DD_FLOOR,
                k_slip=PS.K_SLIP, rho_mac_dinh=PS.RHO_MAC_DINH,
                rho_cang_thang=PS.RHO_CANG_THANG, budget=0.01, horizon=250, lev_max=30.0,
            ),
            "pairs": {}}

    MUC_KHOANG = (0.80, 0.90, 0.95)

    for p in O.PAIRS:
        g = pan[pan.pair == p].sort_values("Date").reset_index(drop=True)
        g_tr = g[g.Date < VALID_TU]

        sizer = PositionSizer(g_tr.sig.values)
        nu_fit, _, sc_fit = stats.t.fit(g_tr.zT.values, floc=0)
        nu = float(np.clip(nu_fit, 2.5, 40))
        t_scale = float(sc_fit)          # dung cho p_cham_stop, KHONG bi clip nhu nu
        c_thoat_sigma = O.chi_phi_thoat(p)

        kc = KhoangConformal(g_tr.zT.values, g_tr.sig.values)
        # khoa dang "0.80"/"0.90"/"0.95" (f"{m:.2f}") — CO DINH, khong dung str(m)
        # (str(0.8) -> "0.8", de nham lan/sai khoa o phia API doc lai)
        khoang_hw = {"chung": {f"{m:.2f}": float(kc.nua_be_rong(m, None)) for m in MUC_KHOANG}}
        for reg in range(kc.n_bins):
            # sig dai dien cua che do reg de tra ve dung tang trong nua_be_rong
            sig_dai_dien = float(kc.edges[0]) * (0.5 if reg == 0 else 1.5)
            khoang_hw[str(reg)] = {f"{m:.2f}": float(kc.nua_be_rong(m, sig_dai_dien)) for m in MUC_KHOANG}

        th = TamHan(g_tr.sig.values, (g_tr.zT.values * g_tr.sig.values))
        tamhan = {"tam_han": list(th.tam_han), "edges": th.edges.tolist(),
                  "c": {str(h): th.c[h] for h in th.tam_han}}

        lk = LuatKyHieu(g_tr.rv5.values)

        meta["pairs"][p] = dict(
            s1=sizer.s1, s2=sizer.s2, s_stress=sizer.s_stress,
            nu=nu, t_scale=t_scale, c_thoat_sigma=c_thoat_sigma,
            khoang_edges=kc.edges.tolist(), khoang_n_bins=kc.n_bins,
            khoang_halfwidth=khoang_hw, khoang_n_theo_che_do=kc.n_theo_che_do,
            tamhan=tamhan, luat_q=lk.q.tolist(),
        )

        # ── chuoi day du theo ngay: sig/zT/zL/rv5 + gia that + carry ──
        prices = pd.read_csv(os.path.join(D, "prices", f"{p}_d1.csv"), parse_dates=["Date"])
        prices = prices[["Date", "close"]]
        gm = g.merge(prices, on="Date", how="left")
        gm["close"] = gm["close"].ffill()
        n_missing = int(gm["close"].isna().sum())
        if n_missing:
            gm = gm.dropna(subset=["close"]).reset_index(drop=True)
        carry_full = O.carry_ngay(p, gm.Date.values)

        series = {
            "dates": gm.Date.dt.strftime("%Y-%m-%d").tolist(),
            "sig": gm.sig.round(6).tolist(),
            "zT": gm.zT.round(4).tolist(),
            "zL": gm.zL.round(4).tolist(),
            "rv5": gm.rv5.round(8).tolist(),
            "close": gm.close.round(5).tolist(),
            "carry": np.round(carry_full, 8).tolist(),
        }
        with open(os.path.join(OUT, f"series_{p}.json"), "w") as f:
            json.dump(series, f)

        print(f"  {p:<8} s1={sizer.s1:.5f} s2={sizer.s2:.5f} nu={nu:.2f} sc={t_scale:.5f} "
              f"c_thoat_sigma={c_thoat_sigma:.4f}  chuỗi {len(series['dates']):,} phiên"
              + (f"  (thiếu giá {n_missing} phiên, đã bỏ)" if n_missing else ""))

    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)

    # ── kích thước, để biết có hợp lý cho một Vercel function bundle không ──
    total = 0
    for fn in os.listdir(OUT):
        s = os.path.getsize(os.path.join(OUT, fn))
        total += s
        print(f"  {fn:<24}{s/1024:>10.1f} KB")
    print(f"TỔNG: {total/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
