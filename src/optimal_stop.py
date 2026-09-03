"""TANG 6b — DUNG TOI UU: GIU LENH HAY DONG LENH?

VI SAO TANG NAY TON TAI. Phieu quyet dinh o tang 6 in ra bang tam han:
xac suat cham stop 1 phien 5%, 5 phien 37%, 10 phien 54%, 20 phien 66%.
Roi bo mac nguoi dung. He thong khong co bat ky quy tac nao ve viec KHI NAO
NEN DONG LENH. Day la bai toan TUAN TU that su duy nhat trong he thong.

VI SAO KHONG DUNG HOC TANG CUONG. Da thu o tang 4 va thua co so do
(docs/SIZING_COMPARISON.md): PPO hoc ra he so gan nhu hang so, bien do 0,018
so voi 0,800 cua quy tac tay, va pha san cao gap 26 lan o cung muc tang
truong. O day trang thai chi co HAI CHIEU nen quy hoach dong giai CHINH XAC
tren luoi — khong can xap xi ham, khong can tham do, khong can 10^5 episode,
va ket qua la mot BIEN GIOI DOC DUOC chu khong phai mot mang no-ron.

BAI TOAN. Vi the mua, don bay f do tang 4 chot, muc dung lo dat cach gia vao
k_stop lan do lech chuan du bao. Moi phien phai chon: DONG hay GIU.

TRANG THAI (hai chieu, deu khong thu nguyen):
    s  = ln(gia hien tai / muc dung lo) / sigma_du_bao   — con cach stop bao xa
    v  = che do bien dong (tam phan vi cua sigma tren doan huan luyen)
cong so phien con lai n trong tam han quyet dinh.

CHUYEN TRANG THAI LAY TU CHINH PANEL, khong gia dinh phan phoi:
    cham stop trong phien t  <=>  zL_t <= -s          (zL da o don vi sigma)
    neu song sot: s' = (s + zT_t) * (sigma_t / sigma_{t+1})
Bo ba (zT, zL, ty so sigma) rut mau thuc nghiem theo dung che do v.

GIA TRI. Tinh bang loi suat log tang them tren mot don vi phoi nhiem, ke tu
bay gio den luc dong han:
    DONG : -c_thoat
    GIU  : carry + E[ 1{cham} * (-s*sigma - truot - c_thoat)
                    + 1{song} * (zT*sigma + V_{n-1}(s', v')) ]

DIEU QUAN TRONG PHAI HIEU TRUOC KHI DOC KET QUA. Tang 1 da chung minh khong
co loi the ve huong, tuc E[zT] = 0. Theo dinh ly dung tuy y cua martingale,
neu KHONG co chi phi va KHONG co truot gia thi moi quy tac dung deu cho cung
ky vong — bai toan se tam thuong. No KHONG tam thuong o day dung ba ly do:

  1. TRUOT GIA LAM CHAM STOP DAT HON THOAT TU NGUYEN. Do tu 60.617 lan cham
     muc dung lo tren nen M1.
  2. CARRY la tin hieu FX duy nhat co bang chung ben vung trong tai lieu, va
     no la thu duy nhat tra cong cho viec GIU.
  3. XAC SUAT CHAM STOP PHU THUOC sigma DU BAO. Day la cho gia tri cua tang 2
     cuoi cung bien thanh mot QUYET DINH chu khong con la mot con so.

Nen quy tac tim duoc co dang: GIU khi carry kiem duoc con vuot chi phi truot
gia nhan voi rui ro bi stop; DONG khi bien dong du bao tang lam rui ro do
vuot len. Neu carry am thi quy tac dung noi "dong ngay" — va do la ket luan
DUNG, nhat quan voi tang 1, khong phai loi cua mo hinh.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
D = os.path.join(os.path.dirname(HERE), "data")

from split import VALID_TU, TEST_TU

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
N_CHE_DO = 3
S_MAX, S_STEP = 8.0, 0.05
LUOI_S = np.arange(0.0, S_MAX + 1e-9, S_STEP)
NGAY_NAM = 252


# ───────────────────────────── chuẩn bị ─────────────────────────────
def nap_panel():
    p = pd.read_csv(os.path.join(D, "panel2_6pairs.csv"), parse_dates=["Date"])
    return p.dropna(subset=["sig", "zT", "zL", "zH"]).reset_index(drop=True)


def carry_ngay(pair, dates):
    """Carry ngay, don vi loi suat log/phien. carry.csv la % NAM cho vi the MUA."""
    c = pd.read_csv(os.path.join(D, "carry.csv"), parse_dates=["DATE"])
    c = c[c.pair == pair].sort_values("DATE")
    s = pd.Series(c.carry.values, index=pd.DatetimeIndex(c.DATE))
    v = s.reindex(pd.DatetimeIndex(dates), method="ffill").values
    return v / 100.0 / NGAY_NAM


def truot_trung_binh_sigma(sig_dien_hinh=0.005, k_stop=2.0):
    """Truot gia KY VONG quy ve don vi sigma.

    slippage.csv ghi truot theo tung khoang cach stop (don vi ty le gia).
    Lay nhom co khoang cach gan k_stop*sigma nhat roi tinh TRUNG BINH (khong
    phai p95 — day la ky vong, khong phai tran rui ro).
    """
    d = pd.read_csv(os.path.join(D, "slippage.csv"))
    muc_tieu = k_stop * sig_dien_hinh
    dists = np.array(sorted(d.dist.unique()))
    dist = dists[np.argmin(np.abs(dists - muc_tieu))]
    g = d[d.dist == dist].truot_phut
    pip = 1e-4
    return float(g.mean() * pip / sig_dien_hinh), float(dist), int(len(g))


def chi_phi_thoat(pair, sig_dien_hinh=0.005):
    """Chi phi thoat mot chieu, quy ve don vi sigma."""
    import cost
    ngay = pd.Timestamp("2024-06-03")
    sp = np.mean([cost.spread_pip(pair, h, ngay, "med") for h in range(24)])
    pip = 0.01 if "JPY" in pair else 1e-4
    gia = 150.0 if "JPY" in pair else 1.0
    return float((sp * pip / gia + cost.COMMISSION_PIP * pip / gia) / 2 / sig_dien_hinh)


def rut_mau(pan, mask_train):
    """Bo ba thuc nghiem (zT, zL, ty_so, che_do_ke) nhom theo che do hien tai."""
    nguong = np.quantile(pan.sig.values[mask_train],
                         np.arange(1, N_CHE_DO) / N_CHE_DO)
    mau = {v: {"zT": [], "zL": [], "ty": [], "v2": []} for v in range(N_CHE_DO)}
    for p in PAIRS:
        g = pan[pan.pair == p].sort_values("Date").reset_index(drop=True)
        sub = pan[pan.pair == p].sort_values("Date")
        m = np.asarray(mask_train)[sub.index.values]
        idx = np.where(m)[0]
        sig = g.sig.values
        ch = np.digitize(sig, nguong)
        for i in idx:
            if i + 1 >= len(g):
                continue
            v = int(ch[i])
            mau[v]["zT"].append(g.zT.values[i])
            mau[v]["zL"].append(g.zL.values[i])
            mau[v]["ty"].append(sig[i] / sig[i + 1])
            mau[v]["v2"].append(int(ch[i + 1]))
    for v in mau:
        for k in mau[v]:
            mau[v][k] = np.asarray(mau[v][k], float if k != "v2" else int)
    return mau, nguong


# ───────────────────────────── quy hoạch động ─────────────────────────────
def sigma_che_do(pan, mask_train, nguong):
    """Sigma dien hinh (trung vi) cua tung che do, uoc luong tren huan luyen."""
    sig = pan.sig.values[np.asarray(mask_train)]
    ch = np.digitize(sig, nguong)
    return np.array([float(np.median(sig[ch == v])) for v in range(N_CHE_DO)])


def _buoc_gia_tri(r, f):
    """Quy doi mot buoc loi suat log TUYET DOI 'r' sang GIA TRI CO DON BAY f.

    f=None (mac dinh)  -> tra ve y nguyen r (E[log-return] khong don bay,
    hanh vi CU, tuong thich nguoc 100%).
    f la so (>=0)      -> tra ve ln(1 + f*(e^r - 1)), tuc loi suat log cua
    VON DA VAY f LAN. Day la buoc THEO TUNG PHIEN — cong don cac buoc nay lai
    (V la tong don bien) DUNG BANG ln(von_cuoi/von_dau) cua danh muc co don
    bay, vi von_{t+1} = von_t*(1+f*r_t) nen ln(von_T/von_0) = sum_t ln(1+f*r_t).
    KHONG duoc cong r truoc roi moi doi don vi mot lan — sai vi don bay
    khong tuyen tinh qua nhieu phien (loi lom cua ham log).
    f=0 -> luon tra ve 0 (khong vay thi khong lai khong lo, bat ke r).
    """
    if f is None:
        return r
    if f <= 1e-12:
        return np.zeros_like(np.asarray(r, float)) if hasattr(r, "shape") else 0.0
    # cat r o +-50 truoc expm1 chi de tranh tran so hoc; ve mat tai chinh r
    # o day luon nho (thang loi suat ngay/tuan), khong bao gio cham nguong nay
    rc = np.clip(np.asarray(r, float), -50.0, 50.0)
    arg = f * np.expm1(rc)
    # arg <= -1 nghia la mot buoc DUY NHAT da xoa sach von (chay tai khoan) —
    # gan gia tri rat am (khong phai NaN) de DP biet tranh, khong lam hong
    # trung binh ca mang boi mot NaN lan truyen
    return np.where(arg > -1.0 + 1e-12, np.log1p(np.maximum(arg, -1.0 + 1e-12)), -50.0)


def giai(mau, sig_v, carry_ngay_abs, c_thoat_sigma, slip_sigma, N=20,
         M=4000, seed=0, f_v=None):
    """Giai nguoc N buoc. Tra ve V (N+1, n_s, n_v) va bien gioi dong lenh.

    DON VI. Moi thu trong V la LOI SUAT LOG TUYET DOI, khong phai don vi
    sigma — day la diem de sai nhat cua bai toan nay: khoang cach toi stop
    do bang sigma CUA HOM NAY, con gia tri tiep dien lai o sigma CUA NGAY MAI.
    Cong hai thu do truc tiep voi nhau la sai. Nen tat ca quy ve loi suat that:

        pay_cham = (-s - slip) * sig_v  -  c_thoat * sig_v
        pay_song =  zT * sig_v          +  V_{n-1}(s', v')
        dong     = -c_thoat * sig_v

    sig_v            : sigma dien hinh cua tung che do (mang do dai N_CHE_DO)
    carry_ngay_abs   : carry moi phien, LOI SUAT LOG tuyet doi
    c_thoat_sigma    : chi phi thoat mot chieu, don vi sigma
    slip_sigma       : truot gia ky vong, don vi sigma
    f_v              : (MOI, tuy chon) don bay Kelly-carry moi che do — tang 4
        (mang do dai N_CHE_DO). None (mac dinh) = ham gia tri cu, toi da hoa
        E[loi suat log KHONG don bay] — hanh vi CU giu nguyen 100%, moi noi
        dang goi giai() ma khong truyen f_v khong doi hanh vi gi ca. Truyen
        vao thi ham gia tri doi thanh E[ln(1+f*R)] — KHOP NOI truc tiep don
        bay tang 4 voi quyet dinh giu/dong tang 6b (xem
        docs/TANG6B_DUNGTOIUU.md, muc "khop noi don bay").
    """
    rng = np.random.default_rng(seed)
    ns, nv = len(LUOI_S), N_CHE_DO
    DR = {}
    for v in range(nv):
        n = len(mau[v]["zT"])
        j = rng.integers(0, n, M) if n > M else np.arange(n)
        DR[v] = {k: mau[v][k][j] for k in ("zT", "zL", "ty", "v2")}

    fv = None if f_v is None else np.asarray(f_v, float)

    V = np.zeros((N + 1, ns, nv))
    for v in range(nv):
        f = None if fv is None else float(fv[v])
        V[0, :, v] = _buoc_gia_tri(-c_thoat_sigma * sig_v[v], f)
    bien = np.full((N + 1, nv), np.nan)
    for n in range(1, N + 1):
        for v in range(nv):
            f = None if fv is None else float(fv[v])
            zT, zL, ty, v2 = (DR[v][k] for k in ("zT", "zL", "ty", "v2"))
            sv = sig_v[v]
            cham = zL[None, :] <= -LUOI_S[:, None]
            s2 = np.clip((LUOI_S[:, None] + zT[None, :]) * ty[None, :], 0.0, S_MAX)
            Vn = np.empty_like(s2)
            for vv in range(nv):
                sel = v2 == vv
                if sel.any():
                    Vn[:, sel] = np.interp(s2[:, sel].ravel(), LUOI_S,
                                           V[n - 1, :, vv]).reshape(s2[:, sel].shape)
            pay_cham = _buoc_gia_tri(
                (-LUOI_S[:, None] - slip_sigma - c_thoat_sigma) * sv, f)
            pay_song = _buoc_gia_tri(zT[None, :] * sv, f) + Vn
            carry_buoc = _buoc_gia_tri(carry_ngay_abs, f)
            giu = carry_buoc + np.where(cham, pay_cham, pay_song).mean(1)
            dong = np.full(ns, _buoc_gia_tri(-c_thoat_sigma * sv, f))
            V[n, :, v] = np.maximum(giu, dong)
            gi = giu > dong
            bien[n, v] = LUOI_S[np.argmax(gi)] if gi.any() else np.inf
    return V, bien


def nen_giu(V, s, v, n, c_thoat_sigma, sig_v, f_v=None):
    """Chinh sach toi uu: GIU khi va chi khi gia tri giu vuot gia tri dong.

    f_v phai TRUYEN Y HET nhu luc goi giai() de ra V nay — nguong so sanh
    (gia tri "dong") cung phai quy doi qua don bay giong het, neu khong sai."""
    if n <= 0:
        return False
    i = int(np.clip(round(s / S_STEP), 0, len(LUOI_S) - 1))
    n = int(np.clip(n, 0, V.shape[0] - 1))
    v = int(v)
    f = None if f_v is None else float(np.asarray(f_v, float)[v])
    nguong = _buoc_gia_tri(-c_thoat_sigma * sig_v[v], f)
    return bool(V[n, i, v] > nguong + 1e-14)


# ───────────────────────────── mô phỏng ─────────────────────────────
def mo_phong(g, mask, nguong, V, carry, c_thoat_sigma, slip_sigma, sig_v,
             k_stop=2.0, N=20, f_v=None, f_v_diem=None):
    """Mo mot lenh MUA moi phien duoc danh dau, chay tung chinh sach.

    g     : DataFrame MOT cap, da sap theo Date, da reset_index
    mask  : mang bool do dai len(g)
    carry : mang loi suat log carry moi phien, do dai len(g)
    f_v   : (MOI, tuy chon) don bay dung de RA QUYET DINH giu/dong — PHAI
        trung voi f_v da dua vao giai() de ra V nay, neu khong chinh sach
        "DP" se so sai nguong. None (mac dinh) = V duoc giai KHONG don bay,
        nen so sanh cung KHONG don bay — dung hanh vi CU.
    f_v_diem : (MOI, tuy chon) don bay dung de CHAM DIEM loi suat (P&L bao
        cao ra). Mac dinh = f_v (quyet dinh va cham diem dung CHUNG mot don
        bay, truong hop thong thuong). Truyen RIENG khi muon so sanh "quyet
        dinh theo bien gioi KHONG don bay nhung cham diem CO don bay" —
        tuc do xem don bay tang 4 ap SAU LEN MOT bien gioi da chon truoc,
        khac voi de DP tu GIAI RA bien gioi biet truoc se co don bay
        (src/compare_leverage_dp.py dung ca hai de so sanh).
    Tra ve (loi suat tung lenh theo chinh sach, ty le bi stop).
    """
    TEN = ("DP", "giữ hết tầm hạn", "đóng ngay", "đóng sau 5 phiên", "đóng sau 10 phiên")
    ket = {t: [] for t in TEN}
    stop = {t: [] for t in TEN}
    sig = g.sig.values; zT = g.zT.values; zL = g.zL.values
    ch = np.digitize(sig, nguong)
    mask = np.asarray(mask)
    f_diem = f_v if f_v_diem is None else f_v_diem

    def buoc(r, v):
        return r if f_diem is None else float(_buoc_gia_tri(r, float(np.asarray(f_diem, float)[v])))

    for i0 in np.where(mask)[0]:
        if i0 + N + 1 >= len(g):
            continue
        for ten in TEN:
            s = float(k_stop); tong = 0.0; bi = 0; dong_som = False
            for h in range(N):
                i = i0 + h
                v = int(ch[i]); n_con = N - h
                if ten == "đóng ngay":
                    giu = False
                elif ten == "giữ hết tầm hạn":
                    giu = True
                elif ten == "đóng sau 5 phiên":
                    giu = h < 5
                elif ten == "đóng sau 10 phiên":
                    giu = h < 10
                else:
                    giu = nen_giu(V, s, v, n_con, c_thoat_sigma, sig_v, f_v=f_v)
                if not giu:
                    tong += buoc(-c_thoat_sigma * sig[i], v)
                    dong_som = True
                    break
                tong += buoc(carry[i], v)
                if zL[i] <= -s:
                    tong += buoc((-s - slip_sigma - c_thoat_sigma) * sig[i], v)
                    bi = 1; dong_som = True
                    break
                tong += buoc(zT[i] * sig[i], v)
                s = float(np.clip((s + zT[i]) * (sig[i] / sig[i + 1]), 0.0, S_MAX))
            if not dong_som:
                tong += buoc(-c_thoat_sigma * sig[i0 + N - 1], ch[i0 + N - 1])
            ket[ten].append(tong); stop[ten].append(bi)
    return ({t: np.asarray(v) for t, v in ket.items()},
            {t: float(np.mean(v)) if v else np.nan for t, v in stop.items()})
