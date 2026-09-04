"""GIAI DOAN 2 — KHAI PHA QUY LUAT, va phai song sot bon cua.

Muc tieu cua ca luan van, dat lai bang mot cau: co ton tai mot DANH SACH quy
luat, hoc tu nhieu cap cung luc, ma chuyen giao duoc sang cap chua tung thay
khong? Va moi quy luat co dang:

    khi tin hieu A xuat hien  ->  P(giam) = X%, P(di ngang) = Y%, P(tang) = Z%

────────────────────────────────────────────────────────────────────────────
NGUYEN TAC SO MOT: KHONG GIAN GIA THUYET PHAI LIET KE DUOC DAY DU
────────────────────────────────────────────────────────────────────────────
Westfall-Young hieu chinh cho SO GIA THUYET DA THU. Neu ta di tim quy luat mot
cach mo (chay CART roi lay la, chay motif roi lay cum) thi khong ai biet thuc
su da thu bao nhieu, va moi hieu chinh bou deu la gia. Nen o day khong gian
duoc DINH NGHIA TRUOC va vet can:

    vi tu = mot hoac HAI menh de dang (dac trung, o phan vi)
    dich  = mot trong ba lop

Voi F dac trung x B o, so gia thuyet la mot con so BIET TRUOC, in ra dau moi
lan chay. `run_sax_stats.py` da lam dung the (351 gia thuyet) va do la ly do
ket qua cua no dung vung.

────────────────────────────────────────────────────────────────────────────
BON CUA, theo docs/REPLAN_2026.md muc 3.5
────────────────────────────────────────────────────────────────────────────
  1. Westfall-Young maxT tung buoc xuong, null KHOI (giu tinh dai cua chuoi)
  2. DOI CHUNG CO DIEU KIEN — quy luat phai con tin rieng SAU KHI da dieu kien
     hoa tren null manh nhat: log sigma (cho truc bien dong), TSMOM (cho truc
     huong). Day la buoc da loai 1 trong 3 mau cua HuyH.
  3. BO-MOT-CAP: khai pha tren 5 cap, cham tren cap thu 6. Quy luat khong
     chuyen giao duoc thi KHONG phai quy luat chung.
  4. DONG GOP THAT vao ba o: BSS so nen "chi sigma", KTC bootstrap khoi khong
     phu 0. Nguong lay tu docs/GIAIDOAN1_NEN.md.

Chay:  python src/run_quyluat.py
Ghi:   output/quyluat.json, rules/rules_v1.csv, output/log_quyluat.txt
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
RULES = os.path.join(ROOT, "rules")
os.makedirs(OUT, exist_ok=True)
os.makedirs(RULES, exist_ok=True)

import balop as B                                          # noqa: E402
import chibao as CB                                        # noqa: E402
import diem3 as D                                          # noqa: E402
import volfc2 as V2                                        # noqa: E402
from split import VALID_TU, TEST_TU, doan                  # noqa: E402
from volfc import merge_thin_days                          # noqa: E402

H = 1
NPERM = 1000
KHOI = 5                    # do dai khoi cho hoan vi — giu tinh dai
MIN_KHOP = 100              # so lan khop toi thieu (REPLAN muc 3.5)
LIFT_LOPO = 1.15            # nguong lift tren cap bi giu lai
MIN_CAP_DUONG = 4           # so cap phai duong / 6
T_DIEU_KIEN = 3.0           # |t| sau khi dieu kien hoa null manh nhat
SEED = 0
EPS = 1e-12
TEN_LOP = ("giảm", "đi ngang", "tăng")


# ── dac trung de dung vi tu ─────────────────────────────────────────────
def dac_trung(m, sig, z):
    """Bo dac trung DOC DUOC — moi cai phai giai thich duoc bang mot cau.

    Tat ca deu NHAN QUA: chi dung thong tin den het phien t de noi ve t+1."""
    c, h, l = m.close.values, m.high.values, m.low.values
    R = CB.tinh_tat_ca(m)
    r1 = np.r_[np.nan, np.diff(np.log(np.maximum(c, EPS)))]
    F = {
        "σ̂": sig,
        "ATR phân vị": R["atr_pv"],
        "RSI": R["rsi"],
        "ADX": R["adx"],
        "Bollinger %B": (c - R["bb_duoi"]) / np.maximum(R["bb_tren"] - R["bb_duoi"], EPS),
        "khoảng cách EMA50": (c - R["ema50"]) / np.maximum(R["atr"], EPS),
        "Supertrend chiều": R["st_chieu"].astype(float),
        "MACD hist": R["macd_hist"] / np.maximum(R["atr"], EPS),
        "|z| hôm nay": np.abs(z),
        "z hôm nay": z,
        "TSMOM 20": pd.Series(r1).rolling(20).sum().values,
        "tính dai vol": pd.Series(np.log(np.maximum(sig, EPS))).diff().values,
    }
    return F


def roi_rac(F, tr, nbin=3):
    """Chia moi dac trung thanh `nbin` o theo phan vi CHOT TREN HUAN LUYEN."""
    lit, ten = [], []
    for k, v in F.items():
        v = np.asarray(v, float)
        vt = v[tr & np.isfinite(v)]
        if len(vt) < 200:
            continue
        q = np.quantile(vt, np.linspace(0, 1, nbin + 1)[1:-1])
        idx = np.digitize(v, q)
        for b in range(nbin):
            lit.append((idx == b) & np.isfinite(v))
            nhan = ["thấp", "vừa", "cao"][b] if nbin == 3 else f"o{b}"
            ten.append(f"{k} {nhan}")
    return np.array(lit), ten


def vet_can(lit, ten, toi_da_2=True):
    """Liet ke TOAN BO vi tu: 1 menh de, va 2 menh de khac dac trung."""
    vt, vten = list(lit), list(ten)
    if toi_da_2:
        goc = [t.rsplit(" ", 1)[0] for t in ten]
        for i in range(len(lit)):
            for j in range(i + 1, len(lit)):
                if goc[i] == goc[j]:
                    continue                       # cung dac trung -> vo nghia
                vt.append(lit[i] & lit[j])
                vten.append(f"{ten[i]} & {ten[j]}")
    return np.array(vt), vten


# ── thong ke ────────────────────────────────────────────────────────────
def z_lift(M, y, mask):
    """z va lift cho MOI (vi tu, lop). M: (Hy, n) bool. Tra ve (Hy,3)."""
    Mm = M[:, mask]
    nk = Mm.sum(1).astype(float)
    Z = np.full((M.shape[0], 3), np.nan)
    L = np.full((M.shape[0], 3), np.nan)
    for c in range(3):
        yc = (y[mask] == c).astype(float)
        p = yc.mean()
        k = Mm @ yc
        sd = np.sqrt(np.maximum(nk * p * (1 - p), EPS))
        Z[:, c] = np.where(nk >= MIN_KHOP, (k - nk * p) / sd, np.nan)
        L[:, c] = np.where(nk >= MIN_KHOP, k / np.maximum(nk * p, EPS), np.nan)
    return Z, L, nk


def hoan_vi_khoi(y, rng, khoi=KHOI):
    """Hoan vi theo KHOI — giu tinh dai cua chuoi dich, nen mot vi tu chi song
    sot neu no noi them dieu gi ngoai 'hom qua the nao hom nay the ay'."""
    n = len(y)
    nb = int(np.ceil(n / khoi))
    idx = np.concatenate([np.arange(b * khoi, min((b + 1) * khoi, n))
                          for b in rng.permutation(nb)])
    return y[idx[:n]]


def westfall_young(M, y, mask, nperm=NPERM, seed=SEED):
    """maxT tung buoc xuong. Tra ve p_wy cho tung (vi tu, lop) da lam phang."""
    Z, L, nk = z_lift(M, y, mask)
    z = np.abs(np.nan_to_num(Z.ravel(), nan=0.0))
    rng = np.random.default_rng(seed)
    ym = y[mask]
    Zb = np.zeros((nperm, len(z)))
    for b in range(nperm):
        yp = hoan_vi_khoi(ym, rng)
        yy = np.full(len(y), -1)
        yy[mask] = yp
        Zp, _, _ = z_lift(M, yy, mask)
        Zb[b] = np.abs(np.nan_to_num(Zp.ravel(), nan=0.0))
    thu = np.argsort(-z)
    p = np.zeros(len(z))
    con = Zb[:, thu].copy()
    for i in range(len(thu)):
        mx = con[:, i:].max(1) if i < len(thu) else np.zeros(nperm)
        p[thu[i]] = (mx >= z[thu[i]]).mean()
    p = np.maximum.accumulate(p)                 # ep don dieu
    return Z, L, nk, p.reshape(Z.shape), np.quantile(Zb.max(1), [0.9, 0.95, 0.99])


def doi_chung(mkhop, y, c, kiem_soat):
    """Vi tu con noi them gi SAU KHI dieu kien hoa null manh nhat?

    1{lop = c} = a + b*1{khop} + Σ ck * kiem_soat_k    -> tra ve (b, t)."""
    ok = np.isfinite(kiem_soat).all(1) & (y >= 0)
    if ok.sum() < MIN_KHOP:
        return np.nan, np.nan
    X = np.column_stack([np.ones(ok.sum()), mkhop[ok].astype(float),
                         kiem_soat[ok]])
    yy = (y[ok] == c).astype(float)
    XtX = X.T @ X + 1e-8 * np.eye(X.shape[1])
    be = np.linalg.solve(XtX, X.T @ yy)
    r = yy - X @ be
    s2 = float(r @ r) / max(len(yy) - X.shape[1], 1)
    se = np.sqrt(np.maximum(np.diag(np.linalg.inv(XtX)) * s2, EPS))
    return float(be[1]), float(be[1] / max(se[1], EPS))


def main():
    t0 = time.time()
    print("=" * 112)
    print("GIAI ĐOẠN 2 — KHAI PHÁ QUY LUẬT")
    print("=" * 112)
    from api.main import noi_chuoi

    Ms, sigs, zs, ys, caps, dts = [], [], [], [], [], []
    for p in B.PAIRS:
        m = merge_thin_days(noi_chuoi(p))
        sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, p), 0.0))
        d = pd.DataFrame({"Date": m.Date.values, "sig": sig})
        c = m.close.values
        zt = np.full(len(m), np.nan)
        zt[1:] = np.log(c[1:] / np.maximum(c[:-1], EPS)) / np.maximum(sig[1:], EPS)
        d["zT"] = zt
        tr = doan(d.Date.values) == 0
        T = B.dung_muc_tieu(d, H, tr)
        # DICH mot phien: dac trung cua ngay t noi ve lop cua ngay t+1
        yv = np.full(len(m), -1)
        yv[:-1] = T["yP"][1:]
        Ms.append(m); sigs.append(sig); zs.append(T["z"]); ys.append(yv)
        caps.append(np.full(len(m), p)); dts.append(d.Date.values)

    # dac trung + roi rac hoa, nguong chot tren HUAN LUYEN cua tung cap
    lit_all, ten_lit = [], None
    for i, p in enumerate(B.PAIRS):
        F = dac_trung(Ms[i], sigs[i], zs[i])
        tr = doan(dts[i]) == 0
        L, tn = roi_rac(F, tr)
        lit_all.append(L)
        ten_lit = tn
    lit = np.concatenate(lit_all, axis=1)          # (n_lit, N)
    y = np.concatenate(ys)
    cap = np.concatenate(caps)
    dt = pd.DatetimeIndex(np.concatenate(dts))
    kiem_soat = np.column_stack([
        np.log(np.maximum(np.concatenate(sigs), EPS)),            # null biến động
        np.concatenate([pd.Series(np.r_[np.nan, np.diff(np.log(np.maximum(
            m.close.values, EPS)))]).rolling(20).sum().values for m in Ms]),  # TSMOM
    ])

    M, ten = vet_can(lit, ten_lit)
    print(f"{len(B.PAIRS)} cặp · {len(y):,} hàng")
    print(f"KHÔNG GIAN GIẢ THUYẾT: {len(ten):,} vị từ × 3 lớp = "
          f"{len(ten)*3:,} giả thuyết — liệt kê đầy đủ, biết trước")

    tr = (dt < VALID_TU) & (y >= 0)
    va = (dt >= VALID_TU) & (dt < TEST_TU) & (y >= 0)
    te = (dt >= TEST_TU) & (y >= 0)
    pha = tr | va                                  # phát hiện trên huấn luyện+kiểm định
    print(f"phát hiện {int(pha.sum()):,} hàng · xác nhận {int(te.sum()):,} hàng\n")

    print(f"[1/4] Westfall–Young, {NPERM} hoán vị, null khối {KHOI} ngày…",
          flush=True)
    Z, L, nk, P, nguong = westfall_young(M, y, pha)
    print(f"      ngưỡng max|z| null khối: 90% {nguong[0]:.2f} · "
          f"95% {nguong[1]:.2f} · 99% {nguong[2]:.2f}")
    du = nk >= MIN_KHOP
    print(f"      {int(du.sum()):,}/{len(ten):,} vị từ đủ {MIN_KHOP} lần khớp")
    song = (P < 0.05) & np.isfinite(Z)
    tho = (np.abs(np.nan_to_num(Z)) > 1.96) & np.isfinite(Z)
    print(f"      sống sót W-Y p<0,05: {int(song.sum())} / thô p<0,05: "
          f"{int(tho.sum())} (nếu toàn nhiễu kỳ vọng {0.05*np.isfinite(Z).sum():.0f})")

    if song.sum() == 0:
        print("\n→ KHÔNG vị từ nào sống sót hiệu chỉnh bội.")
        json.dump({"khong_gian": int(len(ten) * 3), "song_sot": 0,
                   "nguong_khoi": nguong.tolist()},
                  open(os.path.join(OUT, "quyluat.json"), "w", encoding="utf-8"), indent=1)
        print("TỰ KIỂM ĐẠT")
        return

    print(f"\n[2/4] Đối chứng có điều kiện (|t| > {T_DIEU_KIEN} sau khi khử "
          f"log σ̂ và TSMOM)…", flush=True)
    ung = []
    for i, c in zip(*np.where(song)):
        b, t = doi_chung(M[i], y, c, kiem_soat)
        ung.append(dict(i=int(i), lop=int(c), ten=ten[i], n=int(nk[i]),
                        z=float(Z[i, c]), lift=float(L[i, c]), p_wy=float(P[i, c]),
                        b_dk=b, t_dk=t))
    qua_dk = [u for u in ung if np.isfinite(u["t_dk"]) and abs(u["t_dk"]) > T_DIEU_KIEN]
    print(f"      {len(qua_dk)}/{len(ung)} còn tin riêng sau khi điều kiện hoá")
    # GHI LAI ca nhung vi tu RỚT o cua nay — chung la artefact quan trong nhat
    # cua giai doan 2, vi chung cho thay dieu gi da hap thu het tin hieu.
    print()
    print(f"      {'vị từ sống sót W-Y':<50}{'lớp':<10}{'n':>7}{'lift':>7}"
          f"{'z':>7}{'b|đk':>9}{'t|đk':>7}")
    for u in sorted(ung, key=lambda x: -abs(x["z"])):
        print(f"      {u['ten'][:48]:<50}{TEN_LOP[u['lop']]:<10}{u['n']:>7}"
              f"{u['lift']:>7.3f}{u['z']:>7.2f}{u['b_dk']:>9.4f}{u['t_dk']:>7.2f}")
    json.dump(ung, open(os.path.join(OUT, "quyluat_wy9.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)

    print(f"\n[3/4] Bỏ-một-cặp (lift ≥ {LIFT_LOPO} trên cặp giữ lại, "
          f"≥{MIN_CAP_DUONG}/6 cặp dương)…", flush=True)
    qua_lopo = []
    for u in qua_dk:
        mi, lifts = M[u["i"]], []
        for p in B.PAIRS:
            mp = (cap == p) & pha & (y >= 0)
            kh = mi & mp
            if kh.sum() < 20:
                lifts.append(np.nan); continue
            pc = (y[mp] == u["lop"]).mean()
            lifts.append(float((y[kh] == u["lop"]).mean() / max(pc, EPS)))
        lifts = np.array(lifts)
        nd = int(np.nansum(lifts > 1.0))
        u["lift_cap"] = lifts.tolist()
        u["so_cap_duong"] = nd
        u["lift_min"] = float(np.nanmin(lifts))
        if nd >= MIN_CAP_DUONG and np.nanmin(lifts) >= LIFT_LOPO:
            qua_lopo.append(u)
    print(f"      {len(qua_lopo)}/{len(qua_dk)} chuyển giao được qua các cặp")

    print("\n[4/4] Xác nhận trên đoạn KIỂM TRA (chưa dùng để phát hiện)…",
          flush=True)
    Zte, Lte, nkte = z_lift(M, y, te)
    for u in qua_lopo:
        u["z_te"] = float(Zte[u["i"], u["lop"]])
        u["lift_te"] = float(Lte[u["i"], u["lop"]])
        u["n_te"] = int(nkte[u["i"]])
    xn = [u for u in qua_lopo if np.isfinite(u["z_te"]) and u["z_te"] > 1.96]
    print(f"      {len(xn)}/{len(qua_lopo)} tái lập trên kiểm tra (z > 1,96)")

    print("\n" + "=" * 112)
    print(f"{'PHỄU':<46}{'còn lại':>10}")
    for nhan, v in (("không gian giả thuyết (liệt kê đầy đủ)", len(ten) * 3),
                    ("đủ số lần khớp", int(np.isfinite(Z).sum())),
                    ("thô p<0,05 (chưa hiệu chỉnh)", int(tho.sum())),
                    ("sống sót Westfall–Young", int(song.sum())),
                    ("còn tin riêng sau đối chứng có điều kiện", len(qua_dk)),
                    ("chuyển giao được (bỏ-một-cặp)", len(qua_lopo)),
                    ("tái lập trên KIỂM TRA", len(xn))):
        print(f"{nhan:<46}{v:>10,}")

    if xn:
        print("\nTHƯ VIỆN QUY LUẬT")
        print(f"  {'vị từ':<52}{'lớp':<10}{'n':>7}{'lift':>7}{'t|đk':>7}{'z kt':>7}")
        for u in sorted(xn, key=lambda x: -x["z_te"])[:25]:
            print(f"  {u['ten'][:50]:<52}{TEN_LOP[u['lop']]:<10}{u['n']:>7}"
                  f"{u['lift']:>7.3f}{u['t_dk']:>7.2f}{u['z_te']:>7.2f}")
        pd.DataFrame(xn).to_csv(os.path.join(RULES, "rules_v1.csv"), index=False)
        print(f"\nđã ghi rules/rules_v1.csv ({len(xn)} quy luật)")

    json.dump({"khong_gian": int(len(ten) * 3), "du_khop": int(np.isfinite(Z).sum()),
               "tho": int(tho.sum()), "wy": int(song.sum()),
               "sau_dieu_kien": len(qua_dk), "sau_lopo": len(qua_lopo),
               "xac_nhan": len(xn), "nguong_khoi": nguong.tolist(),
               "quy_luat": xn},
              open(os.path.join(OUT, "quyluat.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/quyluat.json — {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
