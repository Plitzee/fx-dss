"""VONG 7 — CHAM DIEM MOT LAN: HAR so voi TAT CA mo hinh ML/DL.

Doc du bao da luu tu run_ml.py va run_dl.py, cong du bao HAR vong 7, roi:
  * QLIKE tren doan kiem tra, toan bo va tung cap
  * Diebold-Mariano tung mo hinh so voi HAR vong 7
  * Model Confidence Set tren toan bo tap mo hinh
  * bang phan tang theo che do cho ba mo hinh dan dau
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
from run_final7 import dm_nw

P = V2.PAIRS


def main():
    z = np.load(os.path.join(OUT, "_ml_feat.npz"))
    y = z["y"]; pid = z["pid"]; dts = pd.DatetimeIndex(z["dts"])
    rv = np.exp(y)
    va = (dts >= VALID_TU) & (dts < TEST_TU)
    te = dts >= TEST_TU

    F = {}
    for f_, nhan in ((("_ml_pred.npz"), "ml"), (("_dl_pred.npz"), "dl")):
        pth = os.path.join(OUT, f_)
        if not os.path.exists(pth):
            print(f"  (chưa có {f_})")
            continue
        d = np.load(pth, allow_pickle=True)
        qv_ = d["qv"]
        # chi giu BAN TOT NHAT TREN KIEM DINH cua moi HO mo hinh — neu bao cao
        # ca 5 bien the DL tren doan kiem tra thi lai la kiem dinh boi
        tot = {}
        for i, t in enumerate(d["ten"]):
            ho = str(t).split(" h=")[0]
            if ho not in tot or qv_[i] < qv_[tot[ho]]:
                tot[ho] = i
        for ho, i in tot.items():
            F[f"{ho} (khớp năm)" if nhan == "dl" else str(d["ten"][i])] = d[f"f{i}"]

    # HAR vòng 7 — trải lại theo đúng thứ tự bảng dài
    bang, chung = bang_cache()
    tr = np.asarray(chung < VALID_TU)
    with open(os.path.join(OUT, "cauhinh_chot.pkl"), "rb") as f:
        CH = pickle.load(f)
    H = V2.chay(bang, chung, deseason=CH["deseason"], crosspair=bool(CH["crosspair"]),
                event=CH["event"], window=None, lams=(CH["lam"],),
                train_mask=tr, recal=CH["recal"])[CH["lam"]]
    Hb = V2.chay(bang, chung, lams=(0.0,), train_mask=tr)[0.0]
    n1 = len(chung)

    def ve_luoi_ml(arr):
        """volfc2 luu du bao CHO ngay j tai chi so j; bang ML dat muc tieu
        rv5[i+1] tai hang i. Phai DICH MOT NGAY, neu khong la lech pha."""
        v = np.full(n1, np.nan); v[:-1] = np.asarray(arr)[1:]
        return v

    F["HAR vòng 7 (khớp mỗi phiên)"] = np.concatenate([ve_luoi_ml(H[p]) for p in P])
    F["HAR gốc (khớp mỗi phiên)"] = np.concatenate([ve_luoi_ml(Hb[p]) for p in P])
    pan = pd.read_csv(os.path.join(D, "panel2_6pairs.csv"), parse_dates=["Date"])
    F["MA20-GK (nền cũ)"] = np.concatenate(
        [ve_luoi_ml(pan[pan.pair == p].set_index("Date").sig_old.reindex(chung).values ** 2)
         for p in P])

    # ngay CUA MUC TIEU (hang i du bao ngay i+1) — dung de chia doan
    dt_tgt = np.concatenate([np.append(np.asarray(chung)[1:], np.datetime64("NaT"))
                             for _ in P])
    dt_tgt = pd.DatetimeIndex(dt_tgt)
    va = (dt_tgt >= VALID_TU) & (dt_tgt < TEST_TU)
    te = dt_tgt >= TEST_TU
    dts = dt_tgt

    # TO HOP DEU TAY — Brini (arXiv 2607.05291) thay trung binh deu tay giua
    # foundation model va Log-HAR nam trong MCS 98-100%. Thu lai o day.
    def gop(*ten_):
        L_ = [np.log(np.maximum(F[t], 1e-300)) for t in ten_]
        m = np.ones(len(L_[0]), bool)
        for t in ten_:
            m &= np.isfinite(F[t]) & (F[t] > 0)
        out = np.full(len(m), np.nan)
        out[m] = np.exp(np.mean([l[m] for l in L_], 0))
        return out

    dl_ten = [t for t in F if t.startswith(("GRU", "LSTM"))]
    if dl_ten:
        F["Tổ hợp HAR v7 + GRU"] = gop("HAR vòng 7 (khớp mỗi phiên)", dl_ten[0])
        F["Tổ hợp HAR v7 + GRU + LSTM"] = gop("HAR vòng 7 (khớp mỗi phiên)", *dl_ten)

    TEN = list(F)
    ok = te.copy()
    for t in TEN:
        ok &= np.isfinite(F[t]) & (F[t] > 0)
    ok &= np.isfinite(rv) & (rv > 0)
    print("=" * 104)
    print("CHẤM ĐIỂM MỘT LẦN TRÊN ĐOẠN KIỂM TRA — HAR so với toàn bộ ML/DL")
    print("=" * 104)
    print(f"{int(ok.sum()):,} quan sát (6 cặp × {int(ok.sum())//6} phiên), "
          f"{dts[te].min().date()} → {dts[te].max().date()}")
    print(f"{len(TEN)} mô hình, tất cả đều có dự báo trên mọi quan sát\n")

    L = {}
    for t in TEN:
        r = rv[ok] / F[t][ok]
        L[t] = r - np.log(r) - 1
    qv = {}
    for t in TEN:
        m = va & np.isfinite(F[t]) & (F[t] > 0) & np.isfinite(rv)
        r = rv[m] / F[t][m]
        qv[t] = float((r - np.log(r) - 1).mean())

    xep = sorted(TEN, key=lambda t: L[t].mean())
    goc = "HAR vòng 7 (khớp mỗi phiên)"
    print(f"{'#':>3} {'mô hình':<34}{'QLIKE kiểm định':>18}{'QLIKE KIỂM TRA':>17}"
          f"{'so HAR v7':>12}{'DM t':>9}{'DM p':>9}")
    print("-" * 104)
    for i, t in enumerate(xep):
        d = L[t] - L[goc]
        if t == goc:
            tt, pp = 0.0, 1.0
        else:
            tt, pp = dm_nw(d)
        print(f"{i+1:>3} {t:<34}{qv[t]:>18.4f}{L[t].mean():>17.4f}"
              f"{(L[t].mean()/L[goc].mean()-1)*100:>11.1f}%{tt:>9.2f}{pp:>9.4f}")
    print("-" * 104)
    print("  (DM dương = mô hình đó TỆ HƠN HAR vòng 7; p<0,05 là chênh lệch có ý nghĩa)")

    # MCS trên chuỗi tổn thất trung bình theo ngày
    ngay = dts[ok]
    df = pd.DataFrame({t: L[t] for t in TEN})
    df["ngay"] = ngay.values
    G = df.groupby("ngay").mean()
    alive, elim = M.mcs(G[TEN].values, alpha=0.10, B=3000, block=20, seed=17)
    print(f"\nMODEL CONFIDENCE SET (α=0,10) trên {G.shape[0]} phiên:")
    song = [TEN[a] for a in alive]
    for t in song:
        print(f"    ★ {t}")
    print(f"  {len(song)}/{len(TEN)} mô hình sống sót; bị loại sớm nhất: "
          + ", ".join(TEN[i] for i, _ in elim[:3]))

    print(f"\nQLIKE KIỂM TRA THEO TỪNG CẶP — 6 mô hình dẫn đầu")
    print("-" * 104)
    top = xep[:6]
    print(f"{'cặp':<10}" + "".join(f"{t[:15]:>16}" for t in top))
    print("-" * 104)
    for j, p in enumerate(P):
        mp = ok & (pid == j)
        line = f"{p:<10}"
        for t in top:
            r = rv[mp] / F[t][mp]
            line += f"{float((r-np.log(r)-1).mean()):>16.4f}"
        print(line)
    print("-" * 104)

    # ── phân tầng theo chế độ cho các mô hình dẫn đầu
    print("\nPHÂN TẦNG THEO CHẾ ĐỘ (ngũ phân vị biến động dự báo của HAR vòng 7,")
    print("ngưỡng lấy từ đoạn huấn luyện) — QLIKE kiểm tra")
    print("-" * 104)
    fh = F["HAR vòng 7 (khớp mỗi phiên)"]
    trm = dts < VALID_TU
    b = np.full(len(fh), -1)
    for j in range(6):
        mp = pid == j
        c = np.nanquantile(fh[mp & trm & np.isfinite(fh)], np.linspace(0, 1, 6))[1:-1]
        b[mp] = np.digitize(fh[mp], c)
    show = xep[:5] + ["MA20-GK (nền cũ)"]
    print(f"{'chế độ':<16}{'n':>7}" + "".join(f"{t[:14]:>15}" for t in show))
    print("-" * 104)
    for q in range(5):
        m = ok & (b == q)
        line = f"{['Q1 êm','Q2','Q3','Q4','Q5 căng'][q]:<16}{int(m.sum()):>7}"
        for t in show:
            r = rv[m] / F[t][m]
            line += f"{float((r-np.log(r)-1).mean()):>15.4f}"
        print(line)
    print("-" * 104)

    js = {t: {"qlike_valid": qv[t], "qlike_test": float(L[t].mean())} for t in TEN}
    js["_mcs"] = song
    with open(os.path.join(OUT, "ketqua_ml_dl.json"), "w") as f:
        json.dump(js, f, indent=1, ensure_ascii=False)
    print("\nđã ghi output/ketqua_ml_dl.json")
    print("=" * 104)


if __name__ == "__main__":
    main()
