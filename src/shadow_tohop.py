"""SHADOW-LOG cho tổ hợp HAR+GRU+CatBoost (hồi quy Granger-Ramanathan) —
GHI SONG SONG với HAR sản xuất, KHÔNG thay đổi gì trên production.

VI SAO FILE NAY TON TAI. `docs/ML_DL_VONG7.md` va `kiem_tohop3.py` tim ra to
hop nay tot hon HAR ~3,0% QLIKE tren DUNG mot cua so kiem tra da dung de chon
mo hinh. DM p=0,041 (co y nghia rieng le) nhung MCS van giu HAR trong tap
khong phan biet duoc — chua du manh de doi san xuat. Buoc hop ly tiep theo
la GHI LOG SONG SONG voi HAR that, ngoai dung cua so kiem tra do, xem loi the
co giu duoc khong truoc khi quyet dinh dua vao san xuat.

HE SO HOI QUY GR DUNG BANG DA CHOT — KHONG khop lai o day (se la data
snooping neu tinh chinh theo du lieu moi):

    log(h_tohop) = a + b_har*log(h_har) + b_gru*log(h_gru)
                     + b_catb*log(h_catboost) + hc

    a=-0,14905  b_har=0,38356  b_gru=0,39423  b_catb=0,21335  hc=0,09168

(tu `output/ketqua_tohop3.json`, khoa "GR hồi quy · HAR+GRU+CatB", khop tren
doan kiem dinh 08-09/2026, dung y HET cach `kiem_tohop3.gr_hoi_quy` lam).

BA MO HINH THANH PHAN, MOI NGAY:
  h_har      volfc2.du_bao_san_xuat — DIEM VAO SAN XUAT that, khong doi.
  h_gru      GRU (hid=48, lr=2e-3 — cau hinh tot nhat tu run_dl.py) khop
             MOT LAN tren TOAN BO lich su co y biet (khong walk-forward
             theo nam nhu luc do luong nghien cuu, vi o day chi can DU BAO
             HOM NAY, khong can danh gia backtest).
  h_catboost CatBoost (hp=(8, 0.03, 400) — cau hinh tot nhat tu run_ml2.py)
             khop MOT LAN tren toan bo lich su.

Ca hai deu tai khop MOI LAN CHAY (khong luu model) — CatBoost ~10s, GRU ~1-2
phut tren CPU. Neu chay hang ngay, day la chi phi chap nhan duoc.

GHI VAO `data/so_dubao/shadow_tohop.csv` (KHONG DUNG chung file voi
`dubao.csv` san xuat — schema khac hoan toan, day la muc tieu PHUONG SAI
khong phai xac suat huong). Moi lan chay:
  1. Ghi mot hang MOI cho du bao HOM NAY (rv_that de trong, cham sau).
  2. CHAM lai cac hang CU co ngay du bao da qua (rv_that gio da biet) — dien
     rv_that, qlike_har, qlike_tohop.

Chay:  python src/shadow_tohop.py
Ghi:   data/so_dubao/shadow_tohop.csv
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "so_dubao")
SO = os.path.join(D, "shadow_tohop.csv")

import volfc2 as V2                                            # noqa: E402
import ml_data as MD                                            # noqa: E402
from volfc import merge_thin_days                               # noqa: E402
from api.main import noi_chuoi                                  # noqa: E402
from run_dl import xay_chuoi, khop_mot_lan                      # noqa: E402
from run_ml2 import _fit_catboost                               # noqa: E402

HE_SO_GR = dict(a=-0.14905101400260934, hc=0.09168429953336128,
               b_har=0.3835619412837551, b_gru=0.39422778522424173,
               b_catb=0.21334823344442053)
HP_CATB = (8, 0.03, 400)
HID_GRU, LR_GRU = 48, 2e-3
EPS = 1e-12


def nap_bang_song():
    """Nhu `volfc2.nap_bang()` nhung dung CHUOI DA NOI (`api.main.noi_chuoi`
    — lich su HistData den 2025-12-31 + hien hanh Yahoo tu 2026-01-01), thay
    vi chi doc `data/rv_adv.csv` tinh (dung o 2025-12-31). Day la DUNG nguon
    ma san xuat that (`api.main.tinh`) dung, nen "hom nay" o day la hom nay
    THAT chu khong phai hom nay cua ban sao du lieu cu tren may dev."""
    raw = {}
    for p in V2.PAIRS:
        raw[p] = merge_thin_days(noi_chuoi(p))
    chung = raw[V2.PAIRS[0]].Date
    for p in V2.PAIRS[1:]:
        chung = pd.Index(chung).intersection(pd.Index(raw[p].Date))
    chung = pd.DatetimeIndex(sorted(chung))
    return {p: raw[p][raw[p].Date.isin(chung)].reset_index(drop=True)
            for p in V2.PAIRS}, chung


def _du_bao_har(bang):
    ra = {}
    for p in V2.PAIRS:
        f = V2.du_bao_san_xuat(bang[p], p)
        ra[p] = float(f[-1])
    return ra


def _du_bao_catboost(X, y, pid, dts):
    tr = np.isfinite(X).all(1) & np.isfinite(y)
    cot_all = np.arange(X.shape[1])
    Xtr, ytr = X[tr], y[tr]
    rv_tr = np.exp(ytr)
    pred = _fit_catboost(Xtr, ytr, rv_tr, HP_CATB, {"all": cot_all})
    resid = ytr - pred(Xtr)
    s2 = float(resid.var())
    ra = {}
    for j, p in enumerate(V2.PAIRS):
        idx_p = np.where(pid == j)[0]
        i_today = idx_p[-1]                     # hàng cuối = hôm nay
        mu = float(pred(X[i_today:i_today + 1])[0])
        ra[p] = float(np.exp(np.clip(mu, -30, 0) + 0.5 * s2))
    return ra


def _du_bao_gru(X, y, ten, pid, dts):
    S, F, _, _ = xay_chuoi(X, y, ten, pid, dts)
    hople_dt = np.isfinite(S).all((1, 2)) & np.isfinite(F).all(1)
    itr = np.where(hople_dt & np.isfinite(y))[0]
    ite = []
    for j in range(len(V2.PAIRS)):
        idx_p = np.where((pid == j) & hople_dt)[0]
        ite.append(idx_p[-1])
    ite = np.array(ite)
    mu, s2, ep = khop_mot_lan(S, F, y, itr, ite, "gru", HID_GRU, LR_GRU, seed=0)
    print(f"  GRU khớp {ep} epoch, s²={s2:.4f}")
    ra = {}
    for k, j in enumerate(range(len(V2.PAIRS))):
        ra[V2.PAIRS[j]] = float(np.exp(np.clip(mu[k], -30, 0) + 0.5 * s2))
    return ra


def tohop_gr(h_har, h_gru, h_catb):
    hs = HE_SO_GR
    lg = (hs["a"] + hs["b_har"] * np.log(max(h_har, EPS))
          + hs["b_gru"] * np.log(max(h_gru, EPS))
          + hs["b_catb"] * np.log(max(h_catb, EPS)) + hs["hc"])
    return float(np.exp(lg))


def qlike(rv_that, h):
    if not (np.isfinite(rv_that) and np.isfinite(h) and h > 0 and rv_that > 0):
        return np.nan
    r = rv_that / h
    return float(r - np.log(r) - 1)


def cham_hang_cu(df, bang, chung):
    """Dien rv_that/QLIKE cho cac hang co ngay du bao da qua ma con trong."""
    con_thieu = df["rv_that"].isna() & (pd.to_datetime(df["ngay"]) < chung.max())
    if not con_thieu.any():
        return df, 0
    da_cham = 0
    for idx in df[con_thieu].index:
        p, ngay = df.at[idx, "pair"], pd.Timestamp(df.at[idx, "ngay"])
        d = bang.get(p)
        if d is None:
            continue
        hang = d[d.Date == ngay]
        if hang.empty or not np.isfinite(hang.rv5.values[0]):
            continue
        rv = float(hang.rv5.values[0])
        df.at[idx, "rv_that"] = rv
        df.at[idx, "qlike_har"] = qlike(rv, df.at[idx, "h_har"])
        df.at[idx, "qlike_tohop"] = qlike(rv, df.at[idx, "h_tohop"])
        da_cham += 1
    return df, da_cham


def main():
    t0 = time.time()
    print("=" * 100)
    print("SHADOW-LOG — tổ hợp HAR+GRU+CatBoost so với HAR sản xuất")
    print("=" * 100)

    bang, chung = nap_bang_song()
    print(f"dữ liệu tới {chung.max().date()} (chuỗi đã nối, không phải bản tĩnh cũ)")

    X, y, ten, pid, dts = MD.xay(bang, chung)

    print("\nHAR (sản xuất)…", flush=True)
    h_har = _du_bao_har(bang)

    print("CatBoost (khớp lại trên toàn bộ lịch sử)…", flush=True)
    h_catb = _du_bao_catboost(X, y, pid, dts)

    print("GRU (khớp lại trên toàn bộ lịch sử)…", flush=True)
    h_gru = _du_bao_gru(X, y, ten, pid, dts)

    hang_moi = []
    ngay_du_bao = chung.max().date().isoformat()
    ghi_luc = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'cặp':<10}{'HAR':>12}{'GRU':>12}{'CatBoost':>12}{'Tổ hợp':>12}")
    print("-" * 58)
    for p in V2.PAIRS:
        ht = tohop_gr(h_har[p], h_gru[p], h_catb[p])
        print(f"{p:<10}{h_har[p]:>12.3e}{h_gru[p]:>12.3e}{h_catb[p]:>12.3e}{ht:>12.3e}")
        hang_moi.append(dict(ghi_luc=ghi_luc, pair=p, ngay=ngay_du_bao,
                             h_har=h_har[p], h_gru=h_gru[p], h_catboost=h_catb[p],
                             h_tohop=ht, rv_that=np.nan, qlike_har=np.nan,
                             qlike_tohop=np.nan,
                             ma_cau_hinh="gr_har_gru_catb_v1"))

    moi = pd.DataFrame(hang_moi)
    if os.path.exists(SO):
        cu = pd.read_csv(SO)
        cu, n_cham = cham_hang_cu(cu, bang, chung)
        da_co = set(zip(cu.pair, cu.ngay))
        moi = moi[~moi.apply(lambda r: (r.pair, r.ngay) in da_co, axis=1)]
        df = pd.concat([cu, moi], ignore_index=True)
        print(f"\nđã chấm điểm {n_cham} hàng cũ (nay biết RV thật)")
    else:
        df = moi
        print("\ntạo mới shadow_tohop.csv")

    df = df.sort_values(["ngay", "pair"]).reset_index(drop=True)
    os.makedirs(D, exist_ok=True)
    df.to_csv(SO, index=False)

    da_cham = df.dropna(subset=["rv_that"])
    if len(da_cham) >= 10:
        print(f"\nTÍCH LŨY {len(da_cham)} phiên đã chấm:")
        print(f"  QLIKE HAR   trung bình {da_cham.qlike_har.mean():.4f}")
        print(f"  QLIKE tổ hợp trung bình {da_cham.qlike_tohop.mean():.4f}")
        chenh = (da_cham.qlike_tohop.mean() / da_cham.qlike_har.mean() - 1) * 100
        print(f"  chênh: {chenh:+.1f}%")

    print(f"\n→ {SO} ({len(df)} hàng) · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
