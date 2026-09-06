"""GIAI DOAN 2 O TAM HAN GIO — pheu khai pha quy luat tren H1.

VI SAO. `src/kiem_pheu.py` do duoc: pheu tren D1 chi phat hien duoc quy luat co
lift >= 1,35 (luc 80%). O lift 1,20 luc chi 40%, o 1,15 con 8%. Nen ket luan
"0/1.890" hien tai CHI loai tru duoc quy luat manh. Nguyen nhan la CO MAU, khong
phai ban chat thi truong.

    panel D1   21.596 hang
    H1 co san  592.343 thanh      -> x27,4

Thong ke z tang theo can(n) voi cung co hieu ung, nen x27 du lieu cho z lon hon
khoang x5,2. Nguong Westfall-Young co tang theo so vi tu nhung khong x5. Uoc
MDES giam tu 1,35 xuong ~1,05-1,10.

BON THU BAT BUOC PHAI XU LY, khong thi ket qua la rac:

 1. MUA VU TRONG NGAY. Loi suat H1 co chu ky phien A/Au/My rat manh. Neu khong
    khu, moi "quy luat" tim duoc chi la hieu ung gio mo cua. Xu ly HAI TANG:
    (a) sigma^ duoc khu mua vu truoc khi uoc roi gan lai (Andersen-Bollerslev),
    (b) GIO-TRONG-NGAY nam trong bo kiem soat duoi dang bien gia 23 cot.

 2. NHIEU VI CAU TRUC. Bid-ask bounce tao tu tuong quan am gia o tan suat cao.
    Kiem bang cach chay lai o H4: tin hieu that phai SONG khi gop len, tin hieu
    do nhieu se bien mat.

 3. KHOI HOAN VI >= 24 THANH. KHOI=5 cua D1 khong giu duoc chu ky ngay o H1,
    dung nguyen se cho KTC hep gia.

 4. CUA CHI PHI — quan trong nhat. Spread trung vi ~1 pip ma bien do H1 dien
    hinh chi ~10 pip. Quy luat lift 1,05 hoan toan co the VO GIA TRI KINH TE sau
    chi phi. Nen co cua thu nam: loi the rong phai duong sau khi tru spread theo
    gio. Cua nay DAT TRUOC khi chay, khong phai sau khi thay ket qua.

Chay:  python src/quyluat_h1.py            (H1)
       python src/quyluat_h1.py --h4       (gop len H4, kiem nhieu vi cau truc)
Ghi:   output/quyluat_h1.json | output/quyluat_h4.json
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
GIA = os.path.join(ROOT, "data", "prices")

import balop as B                                           # noqa: E402
import run_quyluat as Q                                     # noqa: E402
from split import VALID_TU, TEST_TU                          # noqa: E402

NPERM = 400                 # it hon D1 (1.000) vi moi hoan vi ton x27 phep tinh
KHOI = 24                   # MOT NGAY — giu chu ky trong ngay
MIN_KHOP = 500              # nhieu du lieu hon thi doi so lan khop cao hon
T_DIEU_KIEN = 3.0
LAM = 0.97                  # he so EWMA cho sigma^ gio  (nua doi ~23 thanh)
CUA_SO_B = 24 * 60          # cua so trung vi de dung dai b (~60 ngay giao dich)
SEED = 0
EPS = 1e-12
TEN_LOP = ("giảm", "đi ngang", "tăng")


def nap_h1(gop4=False):
    """Doc H1 sau cap. gop4=True thi gop 4 thanh lien tiep thanh H4."""
    ra = {}
    for p in B.PAIRS:
        d = pd.read_csv(os.path.join(GIA, f"{p}_h1.csv"), parse_dates=["Date"])
        d = d.sort_values("Date").reset_index(drop=True)
        if gop4:
            d["nhom"] = np.arange(len(d)) // 4
            d = d.groupby("nhom").agg(Date=("Date", "first"), open=("open", "first"),
                                      high=("high", "max"), low=("low", "min"),
                                      close=("close", "last"),
                                      n_bars=("n_bars", "sum")).reset_index(drop=True)
        ra[p] = d
    return ra


def mua_vu(r, gio, tr):
    """He so mua vu trong ngay s(gio) = trung binh |r| tung gio, CHOT TREN
    HUAN LUYEN. Chuan hoa ve trung binh 1 de khong doi thang do chung."""
    s = np.ones(24)
    for g in range(24):
        m = tr & (gio == g) & np.isfinite(r)
        if m.sum() >= 100:
            s[g] = float(np.mean(np.abs(r[m])))
    s = s / max(float(np.mean(s)), EPS)
    return np.maximum(s, 1e-3)


def sigma_gio(r, gio, tr):
    """sigma^ cho thanh KE TIEP, uoc NHAN QUA.

    Khu mua vu -> EWMA phuong sai tren chuoi da khu -> gan mua vu tro lai.
    Gia tri tai t chi dung thong tin den het t-1."""
    s = mua_vu(r, gio, tr)
    rd = r / s[gio]                                   # da khu mua vu
    x = np.where(np.isfinite(rd), rd ** 2, np.nan)
    v = np.full(len(r), np.nan)
    cur, co = np.nan, False
    for i in range(len(r)):
        v[i] = cur                                    # du bao cho t dung tin < t
        if np.isfinite(x[i]):
            cur = x[i] if not co else LAM * cur + (1 - LAM) * x[i]
            co = True
    return np.sqrt(np.maximum(v, 0.0)) * s[gio], s


def dac_trung_h1(d, sig, z, gio):
    """Bo dac trung H1 — moi cai giai thich duoc bang mot cau, deu NHAN QUA.

    Khong dung `chibao.tinh_tat_ca` vi bo do chinh cho khung ngay; o day dung
    cac dai luong tuong duong tinh truc tiep tren thanh gio."""
    c = d.close.values
    r1 = np.r_[np.nan, np.diff(np.log(np.maximum(c, EPS)))]
    S = lambda a: pd.Series(a)
    ema = lambda a, n: S(a).ewm(span=n, adjust=False).mean().values
    tr_ = np.maximum(d.high.values - d.low.values,
                     np.abs(np.r_[np.nan, d.high.values[1:] - c[:-1]]))
    atr = S(tr_).rolling(24).mean().values
    up = S(r1).clip(lower=0).rolling(14).mean().values
    dn = (-S(r1).clip(upper=0)).rolling(14).mean().values
    rsi = 100 - 100 / (1 + up / np.maximum(dn, EPS))
    ma20 = S(c).rolling(24).mean().values
    sd20 = S(c).rolling(24).std().values
    return {
        "σ̂": sig,
        "|z|": np.abs(z),
        "z": z,
        "RSI 14": rsi,
        "%B (24 thanh)": (c - (ma20 - 2 * sd20)) / np.maximum(4 * sd20, EPS),
        "cách EMA48": (c - ema(c, 48)) / np.maximum(atr, EPS),
        "ATR / giá": atr / np.maximum(c, EPS),
        "TSMOM 24": S(r1).rolling(24).sum().values,
        "TSMOM 120": S(r1).rolling(120).sum().values,
        "biên độ thanh": (d.high.values - d.low.values) / np.maximum(atr, EPS),
        "vị trí đóng cửa": (c - d.low.values) / np.maximum(d.high.values - d.low.values, EPS),
        "tính dai vol": S(np.log(np.maximum(sig, EPS))).diff().values,
    }


def wy_nhanh(Mm, ym, nperm=NPERM, khoi=KHOI, seed=SEED, min_khop=MIN_KHOP):
    """Westfall-Young maxT tung buoc xuong, ban da CAT SAN theo mask.

    Khac `run_quyluat.westfall_young` o cho no khong cat lai ma tran moi lan —
    voi 592k hang thi viec cat lai moi hoan vi la phan ton nhat."""
    def zl(y):
        nk = Mm.sum(1).astype(float)
        Z = np.full((Mm.shape[0], 3), np.nan)
        L = np.full((Mm.shape[0], 3), np.nan)
        for c in range(3):
            yc = (y == c).astype(np.float32)
            p = float(yc.mean())
            k = Mm @ yc
            sd = np.sqrt(np.maximum(nk * p * (1 - p), EPS))
            Z[:, c] = np.where(nk >= min_khop, (k - nk * p) / sd, np.nan)
            L[:, c] = np.where(nk >= min_khop, k / np.maximum(nk * p, EPS), np.nan)
        return Z, L, nk

    Z, L, nk = zl(ym)
    z = np.abs(np.nan_to_num(Z.ravel(), nan=0.0))
    rng = np.random.default_rng(seed)
    n = len(ym)
    nb = int(np.ceil(n / khoi))
    Zb = np.zeros((nperm, len(z)), dtype=np.float32)
    t0 = time.time()
    for b in range(nperm):
        idx = np.concatenate([np.arange(k * khoi, min((k + 1) * khoi, n))
                              for k in rng.permutation(nb)])[:n]
        Zp, _, _ = zl(ym[idx])
        Zb[b] = np.abs(np.nan_to_num(Zp.ravel(), nan=0.0))
        if b == 4:
            print(f"      ({(time.time()-t0)/5:.2f}s/hoán vị → ước "
                  f"{(time.time()-t0)/5*nperm/60:.1f} phút)", flush=True)
    thu = np.argsort(-z)
    p = np.zeros(len(z))
    con = Zb[:, thu]
    for i in range(len(thu)):
        p[thu[i]] = (con[:, i:].max(1) >= z[thu[i]]).mean()
    p = np.maximum.accumulate(p)
    return Z, L, nk, p.reshape(Z.shape), np.quantile(Zb.max(1), [0.9, 0.95, 0.99])


def main():
    gop4 = "--h4" in sys.argv
    nhan = "H4" if gop4 else "H1"
    t0 = time.time()
    print("=" * 108)
    print(f"KHAI PHÁ QUY LUẬT Ở TẦM HẠN {nhan}")
    print("=" * 108, flush=True)

    D = nap_h1(gop4)
    lits, ys, gios, sigs, dts, caps, rs = [], [], [], [], [], [], []
    ten_lit = None
    for p in B.PAIRS:
        d = D[p]
        c = d.close.values
        r = np.r_[np.nan, np.diff(np.log(np.maximum(c, EPS)))]
        gio = d.Date.dt.hour.values
        tr = (d.Date.values < np.datetime64(VALID_TU)) & np.isfinite(r)
        sig, _ = sigma_gio(r, gio, tr)
        z = r / np.maximum(sig, EPS)
        # dai b: co dinh cham, lay trung vi sigma^ truot (giong muc tieu P o D1)
        med = pd.Series(sig).rolling(CUA_SO_B, min_periods=CUA_SO_B // 4).median()
        b1 = med.bfill().values
        m_tr = tr & np.isfinite(b1) & (b1 > EPS)
        kP = float(np.quantile(np.abs(r[m_tr]) / b1[m_tr], 1 / 3)) if m_tr.sum() > 500 else 0.4
        bnd = kP * b1
        # DICH mot thanh: dac trung tai t noi ve lop cua t+1
        y_ = B.gan_lop(r, bnd)
        yv = np.full(len(d), -1)
        yv[:-1] = y_[1:]
        F = dac_trung_h1(d, sig, z, gio)
        L, tn = Q.roi_rac(F, tr)
        lits.append(L); ten_lit = tn
        ys.append(yv); gios.append(gio); sigs.append(sig)
        dts.append(d.Date.values); caps.append(np.full(len(d), p)); rs.append(r)

    lit = np.concatenate(lits, axis=1)
    y = np.concatenate(ys)
    gio = np.concatenate(gios)
    sig = np.concatenate(sigs)
    dt = pd.DatetimeIndex(np.concatenate(dts))
    cap = np.concatenate(caps)
    r_all = np.concatenate(rs)

    M, ten = Q.vet_can(lit, ten_lit)
    del lits, lit
    pha = (dt < TEST_TU) & (y >= 0)
    print(f"{len(B.PAIRS)} cặp · {len(y):,} thanh {nhan} · "
          f"{len(ten):,} vị từ × 3 lớp = {len(ten)*3:,} giả thuyết")
    print(f"phát hiện {int(pha.sum()):,} thanh · ma trận {M.nbytes/1e6:.0f} MB",
          flush=True)

    # ── BO KIEM SOAT: log sigma, TSMOM, NHAN TO DO-LA, va GIO (bien giả) ──
    khung = {}
    for i, p in enumerate(B.PAIRS):
        dau = 1.0 if p.endswith("USD") else -1.0
        khung[p] = pd.Series(dau * rs[i], index=pd.DatetimeIndex(dts[i]))
    Fusd = pd.DataFrame(khung)
    Fusd = Fusd / Fusd.std()
    nt = Fusd.mean(axis=1, skipna=True)
    nt_all = np.concatenate([nt.reindex(pd.DatetimeIndex(d)).values for d in dts])
    gio_gia = np.zeros((len(y), 23), np.float32)
    for g in range(1, 24):
        gio_gia[gio == g, g - 1] = 1.0
    # sigma^ phai vao duoi dang MEM DEO (bien gia phan vi rieng tung cap), khong
    # phai `log sigma^` tuyen tinh — xem docstring cua Q.kiem_soat_sigma.
    tr_all = np.concatenate([np.asarray(d) < np.datetime64(VALID_TU) for d in dts])
    ks = np.column_stack([
        Q.kiem_soat_sigma(sig, cap, tr_all),
        pd.Series(r_all).rolling(24).sum().values,
        nt_all,
        gio_gia,
    ])
    # CUM cho sai so vung: cap x khoi 24 thanh (mot ngay)
    cum_nhan = np.concatenate([[f"{p}_{i//24}" for i in range(len(dts[j]))]
                               for j, p in enumerate(B.PAIRS)])
    print(f"bộ kiểm soát: σ̂ (9 biến giả phân vị × mỗi cặp), TSMOM 24, "
          f"nhân tố đô-la, 23 biến giả giờ — "
          f"{int(np.isfinite(ks).all(1).sum()):,} thanh đủ", flush=True)

    # ── CUA 1: Westfall-Young ────────────────────────────────────────────
    print(f"\n[1/4] Westfall–Young, {NPERM} hoán vị, khối {KHOI} thanh…", flush=True)
    Mm = np.ascontiguousarray(M[:, pha])
    Z, L, nk, P, nguong = wy_nhanh(Mm, y[pha])
    print(f"      ngưỡng max|z|: 90% {nguong[0]:.2f} · 95% {nguong[1]:.2f} · "
          f"99% {nguong[2]:.2f}")
    du = nk >= MIN_KHOP
    song = (P < 0.05) & np.isfinite(Z)
    tho = (np.abs(np.nan_to_num(Z)) > 1.96) & np.isfinite(Z)
    print(f"      {int(du.sum()):,}/{len(ten):,} vị từ đủ {MIN_KHOP} lần khớp")
    print(f"      sống sót W-Y p<0,05: {int(song.sum())} / thô p<0,05: "
          f"{int(tho.sum())} (nhiễu thuần kỳ vọng {0.05*np.isfinite(Z).sum():.0f})",
          flush=True)

    ra = {"tam_han": nhan, "n_thanh": int(len(y)), "n_phat_hien": int(pha.sum()),
          "khong_gian": int(len(ten) * 3), "nperm": NPERM, "khoi": KHOI,
          "nguong_khoi": [round(float(x), 3) for x in nguong],
          "tho": int(tho.sum()), "wy": int(song.sum())}

    if song.sum() == 0:
        print(f"\n→ KHÔNG vị từ nào sống sót hiệu chỉnh bội ở {nhan}.")
        ra.update(sau_dieu_kien=0, sau_chi_phi=0)
    else:
        # ── CUA 2: doi chung co dieu kien ────────────────────────────────
        print(f"\n[2/4] Đối chứng có điều kiện (|t| > {T_DIEU_KIEN})…", flush=True)
        ung = []
        for i, c in zip(*np.where(song)):
            b_, t_ = Q.doi_chung(M[i], y, c, ks, cum=cum_nhan)
            ung.append(dict(i=int(i), lop=int(c), ten=ten[i], n=int(nk[i]),
                            z=float(Z[i, c]), lift=float(L[i, c]),
                            p_wy=float(P[i, c]), b_dk=b_, t_dk=t_))
        n_nen = sum(1 for u in ung if Q.la_vi_tu_nen(u["ten"]))
        qua = [u for u in ung
               if not Q.la_vi_tu_nen(u["ten"])
               and np.isfinite(u["t_dk"]) and abs(u["t_dk"]) > T_DIEU_KIEN]
        print(f"      {n_nen}/{len(ung)} vị từ sống sót W-Y là CHÍNH σ̂ — loại "
              f"khỏi không gian quy luật theo nguyên tắc")
        print(f"      {len(qua)}/{len(ung)-n_nen} vị từ KHÔNG-phải-σ̂ còn tin riêng")
        print(f"\n      {'vị từ':<46}{'lớp':<10}{'n':>8}{'lift':>7}{'z':>8}{'t|đk':>8}")
        for u in sorted(ung, key=lambda x: -abs(x["z"]))[:25]:
            print(f"      {u['ten'][:44]:<46}{TEN_LOP[u['lop']]:<10}{u['n']:>8,}"
                  f"{u['lift']:>7.3f}{u['z']:>8.2f}{u['t_dk']:>8.2f}")
        ra["sau_dieu_kien"] = len(qua)

        # ── CUA 3: CHI PHI (dat truoc khi chay) ──────────────────────────
        print(f"\n[3/4] Cửa chi phí — lợi thế ròng sau spread…", flush=True)
        try:
            sp = pd.read_csv(os.path.join(ROOT, "data", "spread_hourly_all.csv"))
            sp_gio = sp.groupby("hour").spread_med.median()
            sp_tb = float(sp_gio.mean())
        except Exception:
            sp_tb = 1.0
        qua_cp = []
        for u in qua:
            m = M[u["i"]] & pha & (y >= 0)
            # loi the ky vong = |r| trung binh khi khop, quy ra pip, tru spread
            loi = float(np.nanmean(np.abs(r_all[m]))) * 1e4
            u["loi_pip"] = round(loi, 3)
            u["spread_pip"] = round(sp_tb, 3)
            u["rong_pip"] = round(loi * (u["lift"] - 1.0) - sp_tb, 3)
            if u["rong_pip"] > 0:
                qua_cp.append(u)
        print(f"      spread trung vị dùng làm ngưỡng: {sp_tb:.2f} pip")
        print(f"      {len(qua_cp)}/{len(qua)} còn lợi thế ròng dương sau chi phí")
        ra["sau_chi_phi"] = len(qua_cp)
        ra["ung_vien"] = ung[:60]

    print("\n" + "=" * 108)
    print(f"{'PHỄU ' + nhan:<50}{'còn lại':>12}")
    for t_, v in (("không gian giả thuyết", ra["khong_gian"]),
                  ("thô p<0,05 (chưa hiệu chỉnh)", ra["tho"]),
                  ("sống sót Westfall–Young", ra["wy"]),
                  ("còn tin riêng sau điều kiện hoá", ra.get("sau_dieu_kien", 0)),
                  ("còn lợi thế ròng sau chi phí", ra.get("sau_chi_phi", 0))):
        print(f"{t_:<50}{v:>12,}")
    print("=" * 108)
    f = os.path.join(OUT, f"quyluat_{nhan.lower()}.json")
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(ra, fh, ensure_ascii=False, indent=1, default=float)
    print(f"đã ghi {os.path.relpath(f, ROOT)} — {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
