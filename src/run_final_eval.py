"""CHAM DIEM CUOI — huan luyen / kiem dinh / kiem tra, dung luat khong quay lai.

Luat ap dung o day:
  * MOI lua chon chi duoc nhin doan KIEM DINH
  * doan KIEM TRA chi duoc cham diem, khong duoc dung de chon
  * moi du bao deu walk-forward, cua so mo rong, chi dung thong tin toi t

Muc dich khong phai lam con so dep hon ma la biet CON SO THAT LA BAO NHIEU
sau khi tru phan loi the do chinh minh da chon mo hinh tren tap cham diem.
"""
import os, sys, time
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
D = os.path.join(os.path.dirname(HERE), "data")
import fxdata; fxdata.D = os.path.join(D, "prices")
from fxdata import load_daily
from vol import per_day_estimators
from volfc import merge_thin_days
from split import doan, TEN
from metrics import mcs
from scipy import stats as st

P = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
EPS = 1e-14; MINTR = 500
ADV = pd.read_csv(os.path.join(D, "rv_adv.csv"), parse_dates=["Date"])


def ols(X, y):
    return np.linalg.solve(X.T @ X + 1e-8 * np.eye(X.shape[1]), X.T @ y)


def qlike(f, y):
    f = np.maximum(f, EPS); y = np.maximum(y, EPS)
    return y / f - np.log(y / f) - 1


def rm(v, w):
    return pd.Series(v).rolling(w).mean().values


def build(pair):
    px = load_daily(pair)[["Date", "open", "high", "low", "close"]]
    a = ADV[ADV.pair == pair].drop(columns=["pair"])
    d = merge_thin_days(a.merge(px, on="Date", how="inner"))
    e = per_day_estimators(d)
    d = d.assign(gk=e.gk.clip(lower=EPS).values, r_cc=e.r_cc.values)
    gap = d.Date.diff().dt.days.values.astype(float).copy(); gap[0] = 1
    d["cont"] = gap <= 4
    return d


def feats(d):
    rv = np.maximum(d.rv5.values, EPS); n = len(rv); o = np.ones(n)
    lv = np.log(rv); lw = rm(lv, 5); lm = rm(lv, 22)
    C = np.minimum(np.maximum(d.bpv5.values, EPS), rv)
    lc = np.log(C); lj = np.log1p(np.maximum(rv - d.bpv5.values, 0) / C)
    lp = np.log(np.maximum(d.rsp.values, EPS)); ln_ = np.log(np.maximum(d.rsn.values, EPS))
    lq = np.log(np.maximum(np.sqrt(np.maximum(d.rq5.values, EPS)) / rv, EPS))
    mu = pd.Series(lv).rolling(250).mean().shift(1).values
    sd = pd.Series(lv).rolling(250).std().shift(1).values
    z = (lv - mu) / np.maximum(sd, 1e-8); G = 1 / (1 + np.exp(-1.5 * z)); I = (z > 0).astype(float)
    H = np.column_stack([o, lv, lw, lm])
    return {"HAR": H,
            "HARQ": np.column_stack([H, lq, lq * lv]),
            "HAR-CJ": np.column_stack([o, lc, rm(lc, 5), rm(lc, 22), lj, rm(lj, 5)]),
            "SHAR": np.column_stack([o, lp, ln_, lw, lm]),
            "THAR": np.column_stack([H, H * I[:, None]]),
            "STHAR": np.column_stack([H, H * G[:, None]]),
            "STHARQ": np.column_stack([H, H * G[:, None], lq, lq * lv])}, lv


REG = ["HAR", "HARQ", "HAR-CJ", "SHAR", "THAR", "STHAR", "STHARQ"]
ENS = {"EN(STHARQ,HARQ,SHAR)": ["STHARQ", "HARQ", "SHAR"],
       "EN(tất cả)": REG}
MODELS = ["MA20-GK", "MA5-RV5"] + REG + list(ENS)


def main():
    t0 = time.time(); OUT = {}
    for pair in P:
        d = build(pair); X, lv = feats(d); n = len(d)
        rv = d.rv5.values; gk = d.gk.values; cont = d.cont.values; y = lv[1:]
        F = {k: np.full(n, np.nan) for k in MODELS}
        for t in range(MINTR, n - 1):
            if not cont[t + 1]:
                continue
            F["MA20-GK"][t + 1] = gk[t - 19:t + 1].mean()
            F["MA5-RV5"][t + 1] = rv[t - 4:t + 1].mean()
            for k in REG:
                Xk = X[k]; Xd = Xk[:t]; yy = y[:t]
                ok = np.isfinite(Xd).all(1) & np.isfinite(yy)
                if ok.sum() < 300:
                    continue
                b = ols(Xd[ok], yy[ok]); xn = Xk[t]
                if not np.isfinite(xn).all():
                    continue
                F[k][t + 1] = float(np.exp(np.clip(xn @ b, -30, 0) + 0.5 * (yy[ok] - Xd[ok] @ b).var()))
            for nm, ks in ENS.items():
                v = [np.log(F[k][t + 1]) for k in ks if np.isfinite(F[k][t + 1])]
                if len(v) == len(ks):
                    F[nm][t + 1] = float(np.exp(np.mean(v)))
        OUT[pair] = dict(F=F, rv=rv, g=doan(d.Date.values), n=n)
        print(f"  {pair} [{time.time()-t0:.0f}s]", flush=True)

    def diem(seg):
        sc = {}
        for p in P:
            A = OUT[p]; m = (A["g"] == seg)
            for k in MODELS:
                m = m & np.isfinite(A["F"][k])
            for k in MODELS:
                sc.setdefault(k, []).append(float(qlike(A["F"][k][m], A["rv"][m]).mean()))
        return {k: float(np.mean(v)) for k, v in sc.items()}

    dv, dt = diem(1), diem(2)
    print("\n" + "=" * 96)
    print("BANG 1 — CHON MO HINH BIEN DONG TREN DOAN KIEM DINH, CHAM TREN DOAN KIEM TRA")
    print("=" * 96)
    print(f"{'mô hình':<22}{'QLIKE kiểm định':>18}{'hạng':>7}{'QLIKE kiểm tra':>18}{'hạng':>7}")
    print("-" * 96)
    rv_ = sorted(dv, key=dv.get); rt_ = sorted(dt, key=dt.get)
    for k in rv_:
        print(f"{k:<22}{dv[k]:>18.4f}{rv_.index(k)+1:>7}{dt[k]:>18.4f}{rt_.index(k)+1:>7}")
    best_v = rv_[0]
    print("-" * 96)
    print(f"Chọn trên kiểm định: {best_v}")
    print(f"  QLIKE kiểm tra của mô hình được chọn : {dt[best_v]:.4f}  (hạng {rt_.index(best_v)+1}/{len(MODELS)})")
    print(f"  QLIKE kiểm tra tốt nhất có thể       : {dt[rt_[0]]:.4f}  ({rt_[0]})")
    print(f"  Giá phải trả cho việc chọn mô hình   : {dt[best_v]-dt[rt_[0]]:+.4f}"
          f"  ({(dt[best_v]/dt[rt_[0]]-1):+.1%})")
    print(f"  Nền so sánh MA20-GK trên kiểm tra    : {dt['MA20-GK']:.4f}"
          f"  → cải thiện {(1-dt[best_v]/dt['MA20-GK']):.1%}")

    print("\n" + "=" * 96)
    print("BANG 2 — DIEBOLD-MARIANO TREN DOAN KIEM TRA, so voi MA20-GK")
    print("=" * 96)
    def dm(x):
        n = len(x); mb = x.mean(); L = int(np.ceil(1.5 * n ** (1 / 3))); s = np.sum((x - mb) ** 2) / n
        for k in range(1, L + 1):
            s += 2 * (1 - k / (L + 1)) * np.sum((x[k:] - mb) * (x[:-k] - mb)) / n
        return mb / np.sqrt(max(s, 1e-16) / n)
    print(f"{'mô hình':<22}" + "".join(f"{p:>12}" for p in P) + f"{'thắng':>9}")
    print("-" * 96)
    LOSS = {}
    for p in P:
        A = OUT[p]; m = (A["g"] == 2)
        for k in MODELS:
            m = m & np.isfinite(A["F"][k])
        LOSS[p] = {k: qlike(A["F"][k][m], A["rv"][m]) for k in MODELS}
    for k in [best_v, "EN(STHARQ,HARQ,SHAR)", "HARQ", "HAR"]:
        line = f"{k:<22}"; w = 0
        for p in P:
            t = dm(LOSS[p][k] - LOSS[p]["MA20-GK"]); pv = 2 * (1 - st.norm.cdf(abs(t)))
            sg = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .1 else ""
            if t < 0 and pv < .05:
                w += 1
            line += f"{f'{t:+.2f}{sg}':>12}"
        print(line + f"{w:>6}/6")
    print("-" * 96)
    print("t âm = tốt hơn MA20-GK.  *** p<0,01  ** p<0,05  * p<0,1")

    print("\n" + "=" * 96)
    print("BANG 3 — MODEL CONFIDENCE SET TREN DOAN KIEM TRA")
    print("=" * 96)
    keep = {k: 0 for k in MODELS}
    for p in P:
        Lm = np.column_stack([LOSS[p][k] for k in MODELS])
        alive, _ = mcs(Lm, alpha=0.10, B=400, block=20, seed=1)
        for i in alive:
            keep[MODELS[i]] += 1
    for k, v in sorted(keep.items(), key=lambda x: -x[1]):
        print(f"  {k:<24}{v}/6")
    np.save(os.path.join(D, "..", "final_eval.npy"), OUT, allow_pickle=True)
    print(f"\n[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
