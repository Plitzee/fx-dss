"""ML TRUC TIEP TREN DAI LUONG RUI RO — huong `KEHOACH_2026Q4.md` muc 3.2.

Gia thuyet, dac trung, moc va tieu chi phu dinh DA CHOT TRUOC o `docs/RUIRO_ML.md`.

VI SAO. Moi lan huan luyen ML truoc day deu nham vao HUONG (thua) hoac PHUONG SAI
(thua). Chua ai huan luyen ML de du bao CHINH CAC DAI LUONG RUI RO. Da xac minh
bang grep: huong nay chi xuat hien trong file ke hoach, khong co dong ma nao.

HAI DICH:
  A  P(cham stop trong 5 phien)  — hien tinh bang NGUYEN LY PHAN XA duoi
     t-Student (`decision_record.p_cham_stop`). Moc do la mot HANG SO: no khong
     phu thuoc trang thai thi truong. Do dung la cho ML co the an tien.
  B  P(vi pham VaR 1%)           — bai toan phan loai nhi phan. Moc la muc danh
     nghia 0,01, cung la hang so.

Ca hai moc deu KHONG phu thuoc trang thai. Gia thuyet: co dieu kien hoa theo
trang thai (bien dong, nhay, gap, lich NHTW) thi du bao tot hon.

CHAM bang Brier + Brier Skill Score so khi hau hoc + ECE (do hieu chuan).
KHOP huan luyen · CHON kiem dinh · CHAM MOT LAN kiem tra.

Chay:  python src/ruiro_ml.py
Ghi:   output/ruiro_ml.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
EPS = 1e-12
K_STOP = 1.5          # stop o 1,5 sigma^ — chot truoc
H_STOP = 5            # trong 5 phien — chot truoc
A_VAR = 0.01          # muc VaR 1% — chot truoc

import balop as B                                   # noqa: E402
import volfc2 as V2                                 # noqa: E402
from split import doan                               # noqa: E402
from decision_record import p_cham_stop              # noqa: E402
from volfc import merge_thin_days                    # noqa: E402

# 8 dac trung CHOT TRUOC, tat ca biet tai thoi diem t
DAC_TRUNG = ["log_sig", "che_do", "z_t", "z_t1", "abs_z_tb5", "ti_nhay",
             "abs_gap", "phien_tu_hop"]


def nap():
    from api.main import noi_chuoi
    try:
        from volfc2 import nap_lich
        lich_co = True
    except Exception:                                # noqa: BLE001
        lich_co = False
    hang = []
    for p in B.PAIRS:
        m = merge_thin_days(noi_chuoi(p))
        ng = pd.DatetimeIndex(m.Date)
        g = doan(m.Date.values)
        h = V2.du_bao_san_xuat(m, p)                 # h[t] = du bao cho NGAY t
        sig = np.sqrt(np.maximum(h, 0.0))
        c = m.close.values
        o = m.open.values if "open" in m else c
        r = np.full(len(m), np.nan)
        r[1:] = np.log(np.maximum(c[1:], EPS) / np.maximum(c[:-1], EPS))
        z = r / np.maximum(sig, EPS)
        rv = np.maximum(m.rv5.values, EPS)
        bpv = np.minimum(np.maximum(m.bpv5.values, EPS), rv)
        ti_nhay = (rv - bpv) / rv
        gap = np.full(len(m), np.nan)
        gap[1:] = np.abs(np.log(np.maximum(o[1:], EPS)
                                / np.maximum(c[:-1], EPS)))
        # so phien ke tu ky hop gan nhat (lich biet truoc -> khong ro ri)
        tu_hop = np.full(len(m), np.nan)
        if lich_co:
            try:
                L = nap_lich(list(ng))
                hop = np.asarray(L.get("fomc", np.zeros(len(m)))) > 0
                d = 999
                for t in range(len(m)):
                    d = 0 if hop[t] else min(d + 1, 999)
                    tu_hop[t] = d
            except Exception:                        # noqa: BLE001
                tu_hop[:] = 0.0
        else:
            tu_hop[:] = 0.0
        abs_z5 = pd.Series(np.abs(z)).rolling(5, min_periods=4).mean().values

        for t in range(2, len(m) - H_STOP - 1):
            s1 = sig[t + 1]                          # du bao cho t+1, biet tai t
            if not (np.isfinite(s1) and s1 > 0):
                continue
            if not (np.isfinite(z[t]) and np.isfinite(z[t - 1])
                    and np.isfinite(abs_z5[t]) and np.isfinite(ti_nhay[t])
                    and np.isfinite(gap[t])):
                continue
            # DICH A: cham stop -1,5*s1 trong H_STOP phien ke tiep (loi suat luy tich)
            cum = np.cumsum(r[t + 1:t + 1 + H_STOP])
            if not np.isfinite(cum).all():
                continue
            y_stop = int(cum.min() <= -K_STOP * s1)
            # DICH B: vi pham VaR 1% o phien t+1
            hang.append(dict(pair=p, ngay=ng[t], doan=int(g[t]),
                             log_sig=float(np.log(s1)), z_t=float(z[t]),
                             z_t1=float(z[t - 1]), abs_z_tb5=float(abs_z5[t]),
                             ti_nhay=float(ti_nhay[t]), abs_gap=float(gap[t]),
                             phien_tu_hop=float(min(tu_hop[t], 30)),
                             sig1=float(s1), r1=float(r[t + 1]),
                             y_stop=y_stop))
    df = pd.DataFrame(hang)
    # che do = tam phan vi cua sigma^, NGUONG CHOT TREN HUAN LUYEN
    tr = df.doan == 0
    q = np.nanquantile(df.log_sig[tr], [1 / 3, 2 / 3])
    df["che_do"] = np.digitize(df.log_sig.values, q).astype(float)
    return df


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, k=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e, n = 0.0, len(y)
    for lo, hi in zip(np.linspace(0, 1, k + 1)[:-1], np.linspace(0, 1, k + 1)[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def cham(y, p, p_nen):
    bs, bn = brier(y, p), brier(y, p_nen)
    return dict(brier=bs, bss=float(1 - bs / bn) if bn > 0 else np.nan,
                ece=ece(y, p), tb_du_bao=float(np.mean(p)),
                tb_thuc=float(np.mean(y)), n=int(len(y)))


def mo_hinh(X_tr, y_tr, X, ten):
    if ten == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X_tr)
        m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X_tr), y_tr)
        return m.predict_proba(sc.transform(X))[:, 1]
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                           min_child_samples=80, subsample=0.8,
                           colsample_bytree=0.8, verbose=-1, random_state=7)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X)[:, 1]


def chay_dich(df, ten_dich, y, nen_hang_so, ra):
    tr, va, te = df.doan == 0, df.doan == 1, df.doan == 2
    X = df[DAC_TRUNG].values
    kh = float(y[tr].mean())                      # khi hau hoc tu HUAN LUYEN
    print(f"\n── {ten_dich} · tần suất nền (huấn luyện) {100*kh:.3f}% " + "─" * 36)
    print(f"{'mô hình':<26}{'Brier kđ':>11}{'BSS kđ':>9}{'Brier kt':>11}"
          f"{'BSS kt':>9}{'ECE kt':>9}")
    kq = {}
    for nhan, p_all in (("khí hậu học (nền)", np.full(len(df), kh)),
                        ("mốc giải tích", nen_hang_so)):
        a = cham(y[va], p_all[va], np.full(va.sum(), kh))
        b = cham(y[te], p_all[te], np.full(te.sum(), kh))
        kq[nhan] = dict(kiem_dinh=a, kiem_tra=b)
        print(f"{nhan:<26}{a['brier']:>11.5f}{a['bss']:>9.4f}"
              f"{b['brier']:>11.5f}{b['bss']:>9.4f}{b['ece']:>9.4f}")
    for ten in ("logistic", "lightgbm"):
        try:
            p_all = mo_hinh(X[tr], y[tr], X, ten)
        except Exception as e:                     # noqa: BLE001
            print(f"{ten:<26}không chạy được: {e}")
            continue
        a = cham(y[va], p_all[va], np.full(va.sum(), kh))
        b = cham(y[te], p_all[te], np.full(te.sum(), kh))
        kq[ten] = dict(kiem_dinh=a, kiem_tra=b)
        print(f"{ten:<26}{a['brier']:>11.5f}{a['bss']:>9.4f}"
              f"{b['brier']:>11.5f}{b['bss']:>9.4f}{b['ece']:>9.4f}")
    ml = [k for k in kq if k in ("logistic", "lightgbm")]
    if ml:
        chon = max(ml, key=lambda k: kq[k]["kiem_dinh"]["bss"])
        r = kq[chon]["kiem_tra"]
        gl = kq["mốc giải tích"]["kiem_tra"]
        dk1 = r["bss"] > 0
        dk2 = r["brier"] < gl["brier"]
        print(f"\n  CHỌN TRÊN KIỂM ĐỊNH: {chon}")
        print(f"    ĐK1 BSS > 0 so khí hậu học trên kiểm tra   "
              f"{'ĐẠT' if dk1 else 'TRƯỢT'}   (BSS {r['bss']:+.4f})")
        print(f"    ĐK2 Brier thấp hơn mốc giải tích           "
              f"{'ĐẠT' if dk2 else 'TRƯỢT'}   ({r['brier']:.5f} so {gl['brier']:.5f})")
        print(f"    → {'DƯƠNG' if dk1 and dk2 else 'ÂM'}")
        kq["chon"] = chon
        kq["dk"] = dict(dk1=bool(dk1), dk2=bool(dk2))
        kq["phan_quyet"] = "duong" if (dk1 and dk2) else "am"
    ra[ten_dich] = kq


def main():
    t0 = time.time()
    print("=" * 100)
    print("ML TRỰC TIẾP TRÊN ĐẠI LƯỢNG RỦI RO (KEHOACH_2026Q4 mục 3.2)")
    print("tiêu chí chốt trước: docs/RUIRO_ML.md")
    print("=" * 100)

    df = nap()
    print(f"bảng {len(df):,} hàng · {df.pair.nunique()} cặp · "
          f"{df.ngay.min().date()} → {df.ngay.max().date()}")
    for nhan, gid in (("huấn luyện", 0), ("kiểm định", 1), ("kiểm tra", 2)):
        print(f"  {nhan:<12}{int((df.doan == gid).sum()):>7,} hàng")

    tr = df.doan == 0
    z_tr = (df.r1 / df.sig1).values[tr]
    z_tr = z_tr[np.isfinite(z_tr)]
    ra = {}

    # ── DICH A: cham stop
    y_a = df.y_stop.values.astype(float)
    p_nen_a = np.full(len(df), p_cham_stop(K_STOP, z_tr, horizon=H_STOP))
    print(f"\nmốc giải tích A (nguyên lý phản xạ, t-Student): "
          f"{100*p_nen_a[0]:.3f}% — HẰNG SỐ, không phụ thuộc trạng thái")
    chay_dich(df, f"A · P(chạm stop {K_STOP}σ̂ trong {H_STOP} phiên)",
              y_a, p_nen_a, ra)

    # ── DICH B: vi pham VaR
    q01 = float(np.quantile(z_tr, A_VAR))
    y_b = (df.r1.values <= q01 * df.sig1.values).astype(float)
    p_nen_b = np.full(len(df), A_VAR)
    print(f"\nmốc B: mức danh nghĩa {100*A_VAR:.1f}% — HẰNG SỐ "
          f"(phân vị z trên huấn luyện q01 = {q01:.3f})")
    chay_dich(df, f"B · P(vi phạm VaR {100*A_VAR:.0f}%)", y_b, p_nen_b, ra)

    os.makedirs(OUT, exist_ok=True)
    json.dump(ra, open(os.path.join(OUT, "ruiro_ml.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/ruiro_ml.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
