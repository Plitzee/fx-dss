"""BO DAC TRUNG DUNG CHUNG CHO ML/DL — cung TAP THONG TIN voi HAR vong 7.

Nguyen tac: mo hinh hoc may phai duoc cho DUNG NHUNG GI HAR duoc cho, khong
hon khong kem, de so sanh la so sanh LOP HAM chu khong phai so sanh du lieu.
Vi vay bo dac trung nay chua:

  * ba thanh phan HAR o khong gian log (ngay / tuan / thang)
  * hieu chinh sai so do luong (realized quarticity) va tuong tac
  * semivariance am/duong (SHAR) va bipower variation
  * bien chuyen che do muot G (dung cong thuc cua STHARQ)
  * lich ngan hang trung uong RIENG TUNG CAP cho ngay t+1 (dung cot da chot
    o vong 7) + NFP + cuoi thang
  * PLUS: 22 do tre log RV, thu trong tuan, ma cap — nhung thu HAR khong co
    va ML co the tu tim ra. Cho ML LOI THE nay co chu dich.

Muc tieu: log rv5[t+1]. Doi ve phuong sai bang hieu chinh log-chuan
+0,5*var(phan du) uoc luong tren chinh doan huan luyen cua tung lan khop —
giong het cach HAR lam, neu khong thi QLIKE se phat oan ML.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import volfc2 as V2

EPS = V2.EPS
NLAG = 22


def _roll(v, w):
    return pd.Series(v).rolling(w).mean().values


def xay(bang, chung):
    """Tra ve (X, y, ten_cot, pair_id, ngay) dang bang dai (pair × ngay)."""
    lich = V2.nap_lich(chung)
    n = len(chung)
    d_ = pd.DatetimeIndex(chung)
    dow = d_.dayofweek.values
    Xs, ys, pid, dts = [], [], [], []
    ten = None
    for ip, p in enumerate(V2.PAIRS):
        d = bang[p]
        rv = np.maximum(d.rv5.values, EPS)
        lv = np.log(rv)
        cols, nm = [], []

        def add(v, t):
            cols.append(np.asarray(v, float)); nm.append(t)

        add(lv, "lrv_d"); add(_roll(lv, 5), "lrv_w"); add(_roll(lv, 22), "lrv_m")
        add(_roll(lv, 66), "lrv_q")
        lq = np.log(np.maximum(np.sqrt(np.maximum(d.rq5.values, EPS)) / rv, EPS))
        add(lq, "lq"); add(lq * lv, "lq_x_lrv")
        add(np.log(np.maximum(d.rsp.values, EPS)), "lrsp")
        add(np.log(np.maximum(d.rsn.values, EPS)), "lrsn")
        add(np.log(np.maximum(np.minimum(d.bpv5.values, rv), EPS)), "lbpv")
        add(np.log(np.maximum(rv - np.minimum(d.bpv5.values, rv), EPS)), "ljump")
        mu = pd.Series(lv).rolling(V2.WIN_Z).mean().shift(1).values
        sd = pd.Series(lv).rolling(V2.WIN_Z).std().shift(1).values
        z = (lv - mu) / np.maximum(sd, 1e-8)
        add(z, "z"); add(1 / (1 + np.exp(-V2.GAMMA * z)), "G")
        add(np.log(np.maximum(d.n5.values, 1)), "ln5")
        for k in range(1, NLAG + 1):                     # độ trễ thô cho ML
            v = np.full(n, np.nan); v[k:] = lv[:-k]
            add(v, f"lrv_lag{k}")
        for k in ("fomc", V2.NHTW[p].lower(), "nfp", "cuoithang"):
            v = np.zeros(n); v[:-1] = lich[k][1:]        # sự kiện của ngày t+1
            add(v, f"ev_{k if k in ('fomc','nfp','cuoithang') else 'nhtw'}")
        for k in ("fomc", V2.NHTW[p].lower()):           # ngày kế tiếp sự kiện
            v0 = np.zeros(n); v0[1:] = lich[k][:-1]
            v = np.zeros(n); v[:-1] = v0[1:]
            add(v, f"ev_sau_{k if k=='fomc' else 'nhtw'}")
        for w in range(5):
            add((dow == w).astype(float), f"dow{w}")
        for j in range(len(V2.PAIRS)):
            add(np.full(n, float(j == ip)), f"pair{j}")
        X = np.column_stack(cols)
        y = np.empty(n); y[:-1] = lv[1:]; y[-1] = np.nan
        Xs.append(X); ys.append(y)
        pid.append(np.full(n, ip)); dts.append(d_.values)
        ten = nm
    return (np.vstack(Xs), np.concatenate(ys), ten,
            np.concatenate(pid), pd.DatetimeIndex(np.concatenate(dts)))


def qlike_tu_log(mu_log, s2, y_rv):
    """Doi du bao log sang phuong sai (hieu chinh log-chuan) roi cham QLIKE."""
    f = np.exp(np.clip(mu_log, -30, 0) + 0.5 * s2)
    ok = np.isfinite(f) & np.isfinite(y_rv) & (f > 0) & (y_rv > 0)
    r = y_rv[ok] / f[ok]
    return f, ok, (r - np.log(r) - 1)
