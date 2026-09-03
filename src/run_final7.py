"""VONG 7 — CHAM DIEM MOT LAN TREN DOAN KIEM TRA.

Cau hinh da CHOT o run_grid2.py (chon tren doan kiem dinh, don gian nhat
trong tap Model Confidence Set). File nay mo doan kiem tra dung mot lan.

Bao gom:
  A. Kiem chung khong ro ri cho dung cau hinh da chot.
  B. QLIKE tren doan kiem tra: MA20-GK (san xuat cu) / HAR goc / HAR vong 7.
  C. Diebold-Mariano (Newey-West) va Model Confidence Set.
  D. VIEC 1 — CHAM DIEM PHAN TANG THEO CHE DO (nguu phan vi bien dong
     du bao). Chagas et al. arXiv:2608.01599; Rossi JEL 2021.
  E. Kiem dinh forecast breakdown kieu Giacomini-Rossi (RES 2009).
  F. Bo chi so phan phoi: CRPS, pinball, log score, PIT+KS, Kupiec,
     Christoffersen, DQ, FZ0 — tren doan kiem tra.
  G. Bang co va khong co USDJPY.
"""
import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
D = os.path.join(ROOT, "data")

import volfc2 as V2
import metrics as M
from split import VALID_TU, TEST_TU
from run_grid import bang_cache

ALPHA = 0.025
P = V2.PAIRS


def dm_nw(x):
    """Thong ke Diebold-Mariano voi phuong sai Newey-West."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); mb = x.mean(); L = int(np.ceil(1.5 * n ** (1 / 3)))
    s = np.sum((x - mb) ** 2) / n
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * np.sum((x[k:] - mb) * (x[:-k] - mb)) / n
    t = mb / np.sqrt(max(s, 1e-16) / n)
    return t, 2 * (1 - stats.norm.cdf(abs(t)))


def main():
    with open(os.path.join(OUT, "cauhinh_chot.pkl"), "rb") as f:
        CH = pickle.load(f)
    CH["crosspair"] = bool(CH["crosspair"])
    print("=" * 100)
    print("VONG 7 — CHAM DIEM MOT LAN TREN DOAN KIEM TRA")
    print("=" * 100)
    print(f"cấu hình đã chốt trên kiểm định: {CH}")

    bang, chung = bang_cache()
    n = len(chung)
    tr = np.asarray(chung < VALID_TU)
    va = np.asarray((chung >= VALID_TU) & (chung < TEST_TU))
    te = np.asarray(chung >= TEST_TU)
    Y = {p: bang[p].rv5.values for p in P}
    wv = {"exp": None, "r1000": 1000, "r1500": 1500, "r2000": 2000}[CH["window"]]

    # ── A. kiểm chứng không rò rỉ cho ĐÚNG cấu hình đã chốt
    b2 = {k: v.copy() for k, v in bang.items()}
    k0 = int(n * 0.75)
    for pp in b2:
        b2[pp].loc[k0:, ["rv5", "bpv5", "rq5", "rsp", "rsn"]] *= 9.0
    A1 = V2.chay(bang, chung, window=wv, lams=(CH["lam"],), train_mask=tr,
                 deseason=CH["deseason"], crosspair=CH["crosspair"],
                 event=CH["event"], recal=CH["recal"])[CH["lam"]]
    A2 = V2.chay(b2, chung, window=wv, lams=(CH["lam"],), train_mask=tr,
                 deseason=CH["deseason"], crosspair=CH["crosspair"],
                 event=CH["event"], recal=CH["recal"])[CH["lam"]]
    for pp in P:
        m = np.isfinite(A1[pp]) & np.isfinite(A2[pp]) & (np.arange(n) <= k0)
        assert np.allclose(A1[pp][m], A2[pp][m]), f"RO RI NHIN TRUOC o {pp}"
    print(f"\nA. không rò rỉ nhìn trước cho cấu hình đã chốt: 6/6 cặp ĐẠT "
          f"({m.sum():,} dự báo quá khứ giữ nguyên khi bóp méo tương lai)")

    # ── B. ba mô hình trên đoạn kiểm tra
    F = {"HAR vòng 7": A1}
    F["HAR gốc"] = V2.chay(bang, chung, lams=(0.0,), train_mask=tr)[0.0]
    pan = pd.read_csv(os.path.join(D, "panel2_6pairs.csv"), parse_dates=["Date"])
    ma = {}
    for p in P:
        s = pan[pan.pair == p].set_index("Date").sig_old.reindex(chung)
        ma[p] = (s.values ** 2)
    F["MA20-GK"] = ma
    TEN = ["MA20-GK", "HAR gốc", "HAR vòng 7"]

    ite = np.where(te)[0]

    def loss(fdict, p, idx):
        y = Y[p][idx]; g = np.asarray(fdict[p])[idx]
        ok = np.isfinite(g) & np.isfinite(y) & (g > 0) & (y > 0)
        r = np.full(len(idx), np.nan); rr = y[ok] / g[ok]
        r[ok] = rr - np.log(rr) - 1
        return r

    Ltest = {t: np.column_stack([loss(F[t], p, ite) for p in P]) for t in TEN}
    ok_all = np.ones(len(ite), bool)
    for t in TEN:
        ok_all &= np.isfinite(Ltest[t]).all(1)
    print(f"\nB. QLIKE TRÊN ĐOẠN KIỂM TRA  {chung[te][0].date()} → {chung[te][-1].date()}"
          f"  ({ok_all.sum()} phiên × 6 cặp, cả ba mô hình cùng có dự báo)")
    print("-" * 100)
    hdr = f"{'cặp':<10}" + "".join(f"{t:>16}" for t in TEN) + f"{'vòng7 vs gốc':>16}"
    print(hdr); print("-" * 100)
    tong = {t: [] for t in TEN}
    for j, p in enumerate(P):
        v = {t: float(np.mean(Ltest[t][ok_all, j])) for t in TEN}
        for t in TEN:
            tong[t].append(v[t])
        d = (v["HAR vòng 7"] / v["HAR gốc"] - 1) * 100
        print(f"{p:<10}" + "".join(f"{v[t]:>16.4f}" for t in TEN) + f"{d:>15.1f}%")
    print("-" * 100)
    tb = {t: float(np.mean(tong[t])) for t in TEN}
    print(f"{'trung bình':<10}" + "".join(f"{tb[t]:>16.4f}" for t in TEN)
          + f"{(tb['HAR vòng 7']/tb['HAR gốc']-1)*100:>15.1f}%")
    tb5 = {t: float(np.mean([tong[t][i] for i, p in enumerate(P) if p != "USDJPY"]))
           for t in TEN}
    print(f"{'bỏ USDJPY':<10}" + "".join(f"{tb5[t]:>16.4f}" for t in TEN)
          + f"{(tb5['HAR vòng 7']/tb5['HAR gốc']-1)*100:>15.1f}%")
    print(f"{'trung vị':<10}" + "".join(
        f"{float(np.median(np.concatenate([Ltest[t][ok_all,j] for j in range(6)]))):>16.4f}"
        for t in TEN))
    print("-" * 100)

    # ── C. DM và MCS
    print("\nC. DIEBOLD-MARIANO (Newey-West) — HAR vòng 7 so với từng đối thủ")
    print("-" * 100)
    print(f"{'cặp':<10}{'vs HAR gốc':>26}{'vs MA20-GK':>26}")
    print(f"{'':<10}{'t':>13}{'p':>13}{'t':>13}{'p':>13}")
    print("-" * 100)
    win = {"HAR gốc": 0, "MA20-GK": 0}
    for j, p in enumerate(P):
        line = f"{p:<10}"
        for rival in ("HAR gốc", "MA20-GK"):
            d = Ltest["HAR vòng 7"][ok_all, j] - Ltest[rival][ok_all, j]
            t, pv = dm_nw(d)
            if t < 0 and pv < 0.05:
                win[rival] += 1
            line += f"{t:>13.2f}{pv:>13.4f}"
        print(line)
    print("-" * 100)
    print(f"  (t âm = HAR vòng 7 tốt hơn)  thắng có ý nghĩa p<0,05: "
          f"{win['HAR gốc']}/6 so với HAR gốc, {win['MA20-GK']}/6 so với MA20-GK")

    Lm = np.column_stack([Ltest[t][ok_all].mean(1) for t in TEN])
    alive, elim = M.mcs(Lm, alpha=0.10, B=3000, block=20, seed=11)
    print(f"\n  Model Confidence Set (alpha=0,10): "
          f"{{{', '.join(TEN[a] for a in alive)}}}")
    for i, pv in elim:
        print(f"    loại {TEN[i]} ở p={pv:.4f}")

    # ── D. VIỆC 1 — phân tầng theo chế độ
    print("\n\nD. VIỆC 1 — CHẤM ĐIỂM PHÂN TẦNG THEO CHẾ ĐỘ (ngũ phân vị biến động DỰ BÁO)")
    print("   Ngưỡng ngũ phân vị lấy từ đoạn HUẤN LUYỆN của từng cặp — biết trước, không rò rỉ.")
    print("-" * 100)
    print(f"{'chế độ':<24}{'n':>7}{'MA20-GK':>12}{'HAR gốc':>12}{'HAR v7':>12}"
          f"{'thiên lệch log':>16}{'tỷ lệ dự báo thiếu':>20}")
    print("-" * 100)
    NG = ["Q1 êm nhất", "Q2", "Q3", "Q4", "Q5 căng nhất"]
    strat = {}
    for q in range(5):
        acc = {t: [] for t in TEN}; bias = []; und = []; nn = 0
        for j, p in enumerate(P):
            f = np.asarray(F["HAR vòng 7"][p])
            cut = np.nanquantile(f[tr & np.isfinite(f)], np.linspace(0, 1, 6))[1:-1]
            b = np.digitize(f[ite], cut)
            m = (b == q) & ok_all
            if m.sum() == 0:
                continue
            nn += int(m.sum())
            for t in TEN:
                acc[t].append(Ltest[t][m, j])
            lr = np.log(Y[p][ite][m] / f[ite][m])
            bias.append(lr); und.append((Y[p][ite][m] > f[ite][m]).astype(float))
        bb = np.concatenate(bias); uu = np.concatenate(und)
        row = {t: float(np.mean(np.concatenate(acc[t]))) for t in TEN}
        strat[NG[q]] = dict(row, n=nn, bias=float(bb.mean()), under=float(uu.mean()))
        print(f"{NG[q]:<24}{nn:>7}" + "".join(f"{row[t]:>12.4f}" for t in TEN)
              + f"{bb.mean():>16.3f}{uu.mean():>19.1%}")
    print("-" * 100)
    for t in TEN:
        r5 = strat["Q5 căng nhất"][t] / strat["Q1 êm nhất"][t]
        print(f"  {t:<12} Q5/Q1 = {r5:.2f}")
    print("  Đọc bảng: QLIKE là thước đo TƯƠNG ĐỐI nên không tự tăng theo mức biến động.")
    print("  Điều đáng đọc là ĐỘ BỀN THEO CHẾ ĐỘ: MA20-GK xấu đi mạnh khi thị trường căng")
    print("  (Q1→Q5), còn HAR vòng 7 gần như phẳng. Trung bình gộp che giấu đúng chỗ này —")
    print("  ở Q5 khoảng cách giữa hai mô hình rộng gấp nhiều lần so với ở Q1.")
    print(f"  Khoảng cách MA20-GK trừ HAR v7:  Q1 {strat['Q1 êm nhất']['MA20-GK']-strat['Q1 êm nhất']['HAR vòng 7']:+.4f}"
          f"   Q5 {strat['Q5 căng nhất']['MA20-GK']-strat['Q5 căng nhất']['HAR vòng 7']:+.4f}")
    print("  Thiên lệch log âm ở mọi chế độ là ĐÚNG chứ không phải lỗi: QLIKE chấm KỲ VỌNG")
    print("  có điều kiện, mà phân phối RV lệch phải, nên trung vị tỷ số thực/dự báo < 1.")

    # tập trung tổn thất ở đuôi
    print("\n   TẬP TRUNG TỔN THẤT Ở ĐUÔI (vì sao trung bình gộp dễ gây hiểu nhầm)")
    print("   " + "-" * 88)
    print(f"   {'cặp':<10}{'QLIKE trung bình':>18}{'trung vị':>12}{'1% ngày tệ nhất':>20}{'5% tệ nhất':>14}")
    print("   " + "-" * 88)
    for j, p in enumerate(P):
        L = np.sort(Ltest["HAR vòng 7"][ok_all, j])[::-1]
        k1 = max(1, int(0.01 * len(L))); k5 = max(1, int(0.05 * len(L)))
        print(f"   {p:<10}{L.mean():>18.4f}{np.median(L):>12.4f}"
              f"{L[:k1].sum()/L.sum():>19.1%}{L[:k5].sum()/L.sum():>13.1%}")
    print("   " + "-" * 88)

    # ── E. forecast breakdown kiểu Giacomini-Rossi
    print("\n\nE. KIỂM ĐỊNH FORECAST BREAKDOWN (Giacomini-Rossi 2009, bản rút gọn)")
    print("   H0: tổn thất trên đoạn kiểm tra KHÔNG tệ hơn tổn thất trên đoạn huấn luyện.")
    print("   Bản rút gọn: chưa có số hạng hiệu chỉnh sai số ước lượng tham số, nên")
    print("   thống kê hơi rộng rãi; dùng để so sánh giữa các cặp là chính.")
    print("-" * 100)
    print(f"{'cặp':<10}{'L̄ huấn luyện':>16}{'L̄ kiểm tra':>16}{'chênh':>12}{'t':>10}{'p một phía':>12}{'kết luận':>18}")
    print("-" * 100)
    itr = np.where(tr)[0]
    for j, p in enumerate(P):
        li = loss(F["HAR vòng 7"], p, itr); li = li[np.isfinite(li)]
        lo = Ltest["HAR vòng 7"][ok_all, j]
        sl = lo - li.mean()
        t, _ = dm_nw(sl)
        pv = float(1 - stats.norm.cdf(t))          # MOT PHIA: gay = ton that TANG
        kl = "GÃY" if pv < 0.05 else "không bác bỏ"
        print(f"{p:<10}{li.mean():>16.4f}{lo.mean():>16.4f}{lo.mean()-li.mean():>12.4f}"
              f"{t:>10.2f}{pv:>12.4f}{kl:>18}")
    print("-" * 100)

    # ── F. bộ chỉ số phân phối
    print("\n\nF. BỘ CHỈ SỐ PHÂN PHỐI TRÊN ĐOẠN KIỂM TRA")
    print("   Phân phối dự báo: lợi suất / sig ~ t Student, tham số ước lượng trên HUẤN LUYỆN.")
    print("-" * 100)
    print(f"{'cặp':<9}{'CRPS×1e4':>11}{'pinball5%×1e5':>15}{'log score':>11}"
          f"{'PIT-KS p':>10}{'vi phạm':>9}{'Kupiec p':>10}{'Chris p':>9}{'DQ p':>8}{'FZ0':>9}")
    print("-" * 100)
    RES = {}
    for p in P:
        px = bang[p]
        ret = np.concatenate([[np.nan], np.diff(np.log(px.close.values))])
        f = np.asarray(F["HAR vòng 7"][p]); sig = np.sqrt(f)
        good = np.isfinite(ret) & np.isfinite(sig) & (sig > 0)
        z_tr = (ret / sig)[good & tr]
        nu, _, sc = stats.t.fit(z_tr, floc=0)
        nu = float(np.clip(nu, 2.5, 40)); sc = float(sc)
        m = good & te
        y = ret[m]; s = sig[m]
        Q = stats.t.ppf(M.TAU_GRID, nu)[None, :] * sc * s[:, None]
        crps = M.crps_from_quantiles(y, Q).mean()
        q05 = stats.t.ppf(0.05, nu) * sc * s
        pb = M.pinball(y, q05, 0.05).mean()
        ls = float(-(stats.t.logpdf(y / s / sc, nu) - np.log(sc) - np.log(s)).mean())
        u = stats.t.cdf(y / s / sc, nu)
        ks = stats.kstest(u, "uniform").pvalue
        v = stats.t.ppf(ALPHA, nu) * sc * s
        hits = (y <= v).astype(int)
        _, pk, ph = M.kupiec(hits, ALPHA)
        _, pc = M.christoffersen_ind(hits)
        _, pd_ = M.dq_test(hits, v, ALPHA)
        x = stats.t.ppf(ALPHA, nu)
        es = -stats.t.pdf(x, nu) * (nu + x ** 2) / ((nu - 1) * ALPHA) * sc * s
        fz = M.fz0(y, v, es, ALPHA).mean()
        RES[p] = dict(crps=float(crps), pinball=float(pb), logscore=ls, ks=float(ks),
                      viol=float(ph), kupiec=float(pk), chris=float(pc),
                      dq=float(pd_), fz0=float(fz), nu=nu, n=int(m.sum()))
        print(f"{p:<9}{crps*1e4:>11.3f}{pb*1e5:>15.3f}{ls:>11.3f}{ks:>10.3f}"
              f"{ph:>9.3%}{pk:>10.3f}{pc:>9.3f}{pd_:>8.3f}{fz:>9.3f}")
    print("-" * 100)
    print(f"  mức VaR danh nghĩa {ALPHA:.1%}; p<0,05 là BÁC BỎ hiệu chuẩn.")
    nb = sum(1 for p in P if RES[p]["kupiec"] > 0.05)
    print(f"  Kupiec không bác bỏ ở {nb}/6 cặp, "
          f"Christoffersen {sum(1 for p in P if RES[p]['chris']>0.05)}/6, "
          f"DQ {sum(1 for p in P if RES[p]['dq']>0.05)}/6, "
          f"PIT-KS {sum(1 for p in P if RES[p]['ks']>0.05)}/6")

    js = {"cauhinh": {k: (str(v) if not isinstance(v, (int, float)) else v)
                      for k, v in CH.items()},
          "qlike_test": {t: tb[t] for t in TEN},
          "qlike_test_bo_jpy": {t: tb5[t] for t in TEN},
          "qlike_test_theo_cap": {p: {t: tong[t][i] for t in TEN}
                                  for i, p in enumerate(P)},
          "phan_tang": strat, "phan_phoi": RES,
          "mcs_song_sot": [TEN[a] for a in alive]}
    with open(os.path.join(OUT, "ketqua_vong7.json"), "w") as f:
        json.dump(js, f, indent=1, ensure_ascii=False)
    print(f"\nđã ghi output/ketqua_vong7.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
