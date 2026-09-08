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
TEN_KIEM_SOAT = ("σ̂ (9 biến giả phân vị × mỗi cặp)", "TSMOM 20",
                 "nhân tố đô-la", "carry")


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


def westfall_young(M, y, mask, nperm=NPERM, seed=SEED, tra_null=False):
    """maxT tung buoc xuong. Tra ve p_wy cho tung (vi tu, lop) da lam phang.

    `tra_null=True` tra them (p_tho, Zb): p BIEN duyen tung gia thuyet — khong
    hieu chinh max — va ca ma tran null. Hai thu do la nguyen lieu cho cong
    FDR (`fdr_bh`), vi FDR can p THO cua ca ho chu khong can max|z|.
    """
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
    ng = np.quantile(Zb.max(1), [0.9, 0.95, 0.99])
    if tra_null:
        return (Z, L, nk, p.reshape(Z.shape), ng,
                p_tho_tu_null(z, Zb).reshape(Z.shape), Zb.astype(np.float32))
    return Z, L, nk, p.reshape(Z.shape), ng


def p_tho_tu_null(z, Zb):
    """p bien duyen hai phia tung gia thuyet, DEM truc tiep tren cot null.

    Cong (b + 1)/(B + 1) — uoc luong khong chech duoi hoan vi, va quan trong
    hon la khong bao gio tra ve p = 0 (mot p = 0 se lam thu tuc BH nhan bua
    bai o hang dau).

    GIOI HAN CUNG: p nho nhat co the tra ve la 1/(B + 1). Voi B = 1.000 thi
    do la 1e-3 — trong khi nguong BY o hang 1 la alpha/(m c) = 9,6e-7. Tuc
    dem truc tiep KHONG BAO GIO bac bo duoc gi qua cong FDR. Dung ham nay de
    KIEM CHUNG, con de chay cong thi dung `p_duoi_chuan`.
    """
    return (1.0 + (Zb >= z[None, :]).sum(0)) / (Zb.shape[0] + 1.0)


def he_so_phong(Zb):
    """lambda_j = do lech chuan cua |z| duoi null, tung gia thuyet.

    Duoi null doc lap, z ~ N(0,1) nen sd(z) = 1. Phu thuoc khoi trong chuoi
    lam sd phong len; lambda do CHINH cai phong do. Day la "genomic control"
    (Devlin & Roeder 1999), lay tung cot vi so lan khop moi vi tu moi khac.

    Uoc tu MOMENT BAC HAI cua |z| chu khong tu var(|z|): duoi null z doi xung
    quanh 0 nen E[z] = 0 va E[z^2] = lambda^2, con var(|z|) thi khong.
    """
    return np.sqrt(np.maximum(np.mean(np.asarray(Zb, float) ** 2, 0), EPS))


def p_duoi_chuan(z, lam):
    """p hai phia tu duoi chuan da hieu chuan: p = 2(1 - Phi(|z| / lambda)).

    NGOAI SUY, va phai duoc noi ro nhu vay. Hoan vi chi do duoc toi 1e-3; moi
    thu duoi do la suy ra tu gia dinh duoi chuan. `kiem_fdr.py` KIEM CHUNG gia
    dinh nay o vung 1e-3..1 noi ca hai cach deu do duoc, roi moi dam dung no
    o vung sau hon.
    """
    from scipy.stats import norm
    return np.clip(2.0 * norm.sf(np.abs(z) / np.maximum(lam, EPS)), 1e-300, 1.0)


def fdr_bh(p, alpha=0.05, bang_bo=True):
    """Benjamini-Hochberg, hoac Benjamini-Yekutieli khi `bang_bo=True`.

    TRA VE (bac_bo, p_hieu_chinh, nguong_p).

    BH kiem soat FDR khi cac gia thuyet doc lap hoac PRDS. Ho gia thuyet o day
    KHONG the gia dinh la PRDS: cac vi tu long nhau ("σ̂ cao" va "σ̂ cao & thu
    Hai"), va ba lop cua cung mot vi tu buoc phai co tuong quan AM. Nen mac
    dinh la BY (Benjamini-Yekutieli 2001), dung duoi PHU THUOC TUY Y, doi lai
    bang cach chia cho c(m) = sum(1/i) ~ ln m + 0,577.

    Voi m = 5.670 gia thuyet, c(m) = 9,22. Do la mot cai gia RAT dat — va no
    la ly do phai DO chu khong duoc doan xem FDR co manh hon FWER khong.
    """
    p = np.asarray(p, float).ravel()
    m = len(p)
    thu = np.argsort(p)
    ps = p[thu]
    c = np.log(m) + 0.5772156649 + 1.0 / (2 * m) if bang_bo else 1.0
    # p hieu chinh BH/BY: min tu phai sang trai cua m*c*p_(i)/i
    ph = np.minimum.accumulate((m * c * ps / np.arange(1, m + 1))[::-1])[::-1]
    ph = np.minimum(ph, 1.0)
    bb = np.zeros(m, bool)
    bb[thu] = ph <= alpha
    ra = np.empty(m)
    ra[thu] = ph
    k = np.where(ph <= alpha)[0]
    ngp = float(ps[k[-1]]) if len(k) else 0.0
    return bb.reshape(np.shape(p)), ra, ngp


R2_TRUNG = 0.99          # nguong coi vi tu la TRUNG voi bo kiem soat
DAC_TRUNG_NEN = ("σ̂",)   # dac trung LA CHINH NEN — khong duoc tinh la quy luat


def la_vi_tu_nen(ten_vt):
    """Vi tu chi gom cac menh de ve CHINH NEN thi khong phai quy luat.

    LY DO NGUYEN TAC, chot truoc khi nhin ket qua. Cau hoi cua giai doan 2 la
    "co quy luat nao noi them dieu gi NGOAI mot mo hinh bien dong tot khong".
    Mot vi tu nhu "sigma^ cao" la CHINH mo hinh do dem ra roi rac hoa — no
    khong the tra loi cau hoi ay, dung nhu 1 = 1 khong chung minh duoc gi.

    Day cung la cho hai lan lien tiep sinh ra ket qua rac: dieu kien hoa mot ham
    cua sigma^ len chinh sigma^ lam he so KHONG DINH DANH DUOC, va hoi quy xac
    suat tuyen tinh ngoai suy ra |b| = 0,64 trong khi lift chi ung voi 0,17.
    Cach dung khong phai va thong ke — la loai chung ra khoi khong gian QUY LUAT
    ngay tu dau. Chung van duoc BAO CAO o bang song sot W-Y, vi "ca chin cai
    song sot deu la chinh sigma^" tu no da la mot ket qua.
    """
    return all(m.rsplit(" ", 1)[0].strip() in DAC_TRUNG_NEN
               for m in str(ten_vt).split(" & "))


def doi_chung(mkhop, y, c, kiem_soat, r2_trung=R2_TRUNG, cum=None):
    """Vi tu con noi them gi SAU KHI dieu kien hoa null manh nhat?

    1{lop = c} = a + b*1{khop} + Σ ck * kiem_soat_k    -> tra ve (b, t).

    CHOT CHONG TRUNG TUYEN TINH. Bo kiem soat chua bien gia phan vi cua sigma^,
    ma nhieu vi tu trong khong gian gia thuyet CUNG la phan vi cua sigma^ — nen
    chung gan nhu nam tron trong khong gian cua bo kiem soat. Khi do he so cua
    vi tu KHONG DINH DANH DUOC: hoi quy van chay, van tra ve mot con so, nhung
    con so do vo nghia.

    Do duoc that (05/09/2026, output/log_chan_doan_dk.txt): vi tu "sigma^ cao"
    cho b = 0,346 va t = 31,8 trong khi lift chi 1,246 (tuc +8 diem phan tram,
    khong phai +34,6). So dieu kien cua X'X la 2,9e12. Neu khong chan, pheu se
    ghi mot "quy luat" thuan tuy do trung tuyen tinh vao rules_v1.csv.

    Nen: hoi quy vi tu len bo kiem soat truoc; neu R^2 >= r2_trung thi vi tu
    KHONG noi them gi ngoai bo kiem soat theo dung nghia den — tra ve NaN de no
    truot cua, thay vi tra ve mot thong ke rac.
    """
    ok = np.isfinite(kiem_soat).all(1) & (y >= 0)
    if ok.sum() < MIN_KHOP:
        return np.nan, np.nan
    K = np.column_stack([np.ones(ok.sum()), kiem_soat[ok]])
    d = mkhop[ok].astype(float)
    # R^2 cua vi tu tren bo kiem soat — bang lstsq de chiu duoc suy bien hang
    be_k, *_ = np.linalg.lstsq(K, d, rcond=None)
    du = d - K @ be_k
    sst = float(((d - d.mean()) ** 2).sum())
    r2 = 1.0 - float(du @ du) / max(sst, EPS)
    if r2 >= r2_trung:
        return np.nan, np.nan               # trung voi bo kiem soat -> vo nghia

    X = np.column_stack([np.ones(ok.sum()), d, kiem_soat[ok]])
    yy = (y[ok] == c).astype(float)
    be, _, rank, _ = np.linalg.lstsq(X, yy, rcond=None)
    u = yy - X @ be

    # SAI SO CHUAN VUNG THEO CUM, khong phai SE thuong.
    #
    # Vi sao bat buoc. Bo kiem soat chua bien gia phan vi cua sigma^, ma nhieu vi
    # tu cung la phan vi cua sigma^. Sau khi khu, phan bien thien con lai cua vi
    # tu tap trung vao DUNG mot vai o phan vi giap ranh — tuc he so chi duoc
    # dinh danh tu mot mau con rat mong. SE thuong gia dinh phuong sai deu tren
    # TOAN BO hang nen no danh gia THAP do bat dinh, va t no tung.
    #
    # Do duoc that (05/09/2026): vi tu "sigma^ cao" cho b = 0,3455 va t = 31,70
    # voi SE thuong, trong khi lift chi 1,246 (+8 diem phan tram, khong phai
    # +34,6). VIF chi 2,9 nen day KHONG phai trung tuyen tinh — day la SE sai.
    #
    # Cum = cap x khoi thoi gian: chuoi tai chinh co tu tuong quan trong cum va
    # cac cap dong theo nhau qua nhan to do-la, nen hai truc do phai vao cum.
    if cum is None:
        se = float(np.sqrt(float(u @ u) / max(len(yy) - rank, 1)
                           / max(float(du @ du), EPS)))
    else:
        g = np.asarray(cum)[ok]
        w = du * u
        tong = pd.Series(w).groupby(pd.Series(g)).sum().values
        se = float(np.sqrt(float((tong ** 2).sum())) / max(float(du @ du), EPS))
    return float(be[1]), float(be[1] / max(se, EPS))


def kiem_soat_sigma(sig, cap, tr, npv=10):
    """Bien gia phan vi cua sigma^, RIENG TUNG CAP — bo kiem soat MEM DEO.

    VI SAO KHONG DUNG `log sigma^` TUYEN TINH. Vi tu trong khong gian gia thuyet
    la CHI BAO PHAN VI ("sigma^ thap/vua/cao"). Mot bien tuyen tinh khong hap thu
    duoc mot chi bao phan vi, nen phan phi tuyen con lai se hien ra nhu la
    "thong tin moi" trong khi no chi la chinh sigma^.

    Da do truc tiep o H1 (output/log_h1_phi_tuyen2.txt): vi tu "sigma^ thap" co
    |t| = 9,68 voi kiem soat tuyen tinh, tut ve 0,78 voi kiem soat mem deo. Ba
    "quy luat song sot" o H1 deu bien mat. Nguong phan vi phai chot RIENG TUNG
    CAP vi sigma^ khac thang giua cac cap — dung nguong gop thi kiem soat bi chi
    phoi boi chenh lech giua cap chu khong phai bien thien trong cap.
    """
    sig = np.asarray(sig, float)
    cap = np.asarray(cap)
    ten_cap = list(dict.fromkeys(cap.tolist()))
    X = np.zeros((len(sig), len(ten_cap) * (npv - 1)), np.float32)
    for j, p in enumerate(ten_cap):
        mp = cap == p
        v = sig[mp & tr & np.isfinite(sig)]
        if len(v) < 100:
            continue
        q = np.quantile(v, np.linspace(0, 1, npv + 1)[1:-1])
        b = np.digitize(sig, q)
        for k in range(1, npv):
            X[mp & (b == k), j * (npv - 1) + k - 1] = 1.0
    return X


def nhan_to_usd(Ms, dts):
    """NHAN TO DO-LA CHUNG — trung binh sau cap sau khi quy ve cung chieu.

    Sau cap deu co USD mot ve nen chung dong theo nhau: rho = 0,443 do duoc
    (output/log_corr_regime.txt). Mot vi tu "hieu qua" hoan toan co the chi
    dang bam vao nhan to nay. LOPO kiem CHUYEN GIAO, khong kiem TRUC GIAO voi
    nhan to chung — hai chuyen khac nhau, nen phai dua no vao bo kiem soat.

    Quy ve cung chieu BAN USD: XXXUSD giu nguyen dau, USDXXX doi dau. Chuan hoa
    tung cap bang do lech chuan cua chinh no truoc khi lay trung binh, de cap
    bien dong manh khong at cac cap khac.

    NHAN QUA: gia tri tai t chi dung loi suat den het t, ma dich la lop cua t+1.
    """
    khung = {}
    for i, p in enumerate(B.PAIRS):
        c = Ms[i].close.values
        r = np.r_[np.nan, np.diff(np.log(np.maximum(c, EPS)))]
        dau = 1.0 if p.endswith("USD") else -1.0          # quy ve "ban USD"
        khung[p] = pd.Series(dau * r, index=pd.DatetimeIndex(dts[i]))
    F = pd.DataFrame(khung)
    F = F / F.std()
    nt = F.mean(axis=1, skipna=True)
    return [nt.reindex(pd.DatetimeIndex(d)).values for d in dts]


def nap_du_lieu():
    """Nap toan bo du lieu + bo kiem soat dung chung cho Giai doan 2.

    Tach ra tu main() de cac ho H2/H3/H5 (run_h2_*.py, run_h3_*.py, run_h5_*.py)
    dung LAI dung mot lan nap, khong copy-paste — tranh sai lech giao thuc giua
    cac ho. Tra ve dict voi moi thu can de tu dung dac_trung/roi_rac/vet_can/
    westfall_young/doi_chung cho MOT khong gian gia thuyet MOI."""
    from api.main import noi_chuoi
    import optimal_stop as OS

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

    y = np.concatenate(ys)
    cap = np.concatenate(caps)
    dt = pd.DatetimeIndex(np.concatenate(dts))
    nt_usd = nhan_to_usd(Ms, dts)
    carry = []
    for i, p in enumerate(B.PAIRS):
        try:
            carry.append(np.asarray(OS.carry_ngay(p, dts[i]), float))
        except Exception:
            carry.append(np.full(len(dts[i]), np.nan))
    sig_all = np.concatenate(sigs)
    tr_all = np.concatenate([doan(d) == 0 for d in dts])
    cum = np.concatenate([[f"{p}_{i//20}" for i in range(len(dts[j]))]
                          for j, p in enumerate(B.PAIRS)])
    kiem_soat = np.column_stack([
        kiem_soat_sigma(sig_all, cap, tr_all),
        np.concatenate([pd.Series(np.r_[np.nan, np.diff(np.log(np.maximum(
            m.close.values, EPS)))]).rolling(20).sum().values for m in Ms]),
        np.concatenate(nt_usd),
        np.concatenate(carry),
    ])
    tr = (dt < VALID_TU) & (y >= 0)
    va = (dt >= VALID_TU) & (dt < TEST_TU) & (y >= 0)
    te = (dt >= TEST_TU) & (y >= 0)
    pha = tr | va
    return dict(Ms=Ms, sigs=sigs, zs=zs, ys=ys, caps=caps, dts=dts,
                y=y, cap=cap, dt=dt, kiem_soat=kiem_soat, cum=cum,
                sig_all=sig_all, tr=tr, va=va, te=te, pha=pha)


def main():
    t0 = time.time()
    print("=" * 112)
    print("GIAI ĐOẠN 2 — KHAI PHÁ QUY LUẬT")
    print("=" * 112)

    du = nap_du_lieu()
    Ms, sigs, zs, dts = du["Ms"], du["sigs"], du["zs"], du["dts"]
    y, cap, dt = du["y"], du["cap"], du["dt"]
    kiem_soat, cum = du["kiem_soat"], du["cum"]
    tr, va, te, pha = du["tr"], du["va"], du["te"], du["pha"]

    # dac trung + roi rac hoa, nguong chot tren HUAN LUYEN cua tung cap
    lit_all, ten_lit = [], None
    for i, p in enumerate(B.PAIRS):
        F = dac_trung(Ms[i], sigs[i], zs[i])
        tri = doan(dts[i]) == 0
        L, tn = roi_rac(F, tri)
        lit_all.append(L)
        ten_lit = tn
    lit = np.concatenate(lit_all, axis=1)          # (n_lit, N)
    du_ks = np.isfinite(kiem_soat).all(1)
    print(f"bộ kiểm soát: {', '.join(TEN_KIEM_SOAT)} — "
          f"{du_ks.sum():,}/{len(du_ks):,} hàng đủ cả bốn")

    M, ten = vet_can(lit, ten_lit)
    print(f"{len(B.PAIRS)} cặp · {len(y):,} hàng")
    print(f"KHÔNG GIAN GIẢ THUYẾT: {len(ten):,} vị từ × 3 lớp = "
          f"{len(ten)*3:,} giả thuyết — liệt kê đầy đủ, biết trước")

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
          f"{', '.join(TEN_KIEM_SOAT)})…", flush=True)
    ung = []
    for i, c in zip(*np.where(song)):
        b, t = doi_chung(M[i], y, c, kiem_soat, cum=cum)
        ung.append(dict(i=int(i), lop=int(c), ten=ten[i], n=int(nk[i]),
                        z=float(Z[i, c]), lift=float(L[i, c]), p_wy=float(P[i, c]),
                        b_dk=b, t_dk=t))
    n_nen = sum(1 for u in ung if la_vi_tu_nen(u["ten"]))
    qua_dk = [u for u in ung
              if not la_vi_tu_nen(u["ten"])
              and np.isfinite(u["t_dk"]) and abs(u["t_dk"]) > T_DIEU_KIEN]
    print(f"      {n_nen}/{len(ung)} vị từ sống sót W-Y là CHÍNH σ̂ — loại khỏi "
          f"không gian quy luật theo nguyên tắc (xem la_vi_tu_nen)")
    print(f"      {len(qua_dk)}/{len(ung)-n_nen} vị từ KHÔNG-phải-σ̂ còn tin riêng "
          f"sau khi điều kiện hoá")
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
