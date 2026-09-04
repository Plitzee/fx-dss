"""BA LOP {giam, di ngang, tang} — MUC TIEU va BON NEN. Giai doan 1.

Cai dat docs/REPLAN_2026.md muc 2 (hinh thuc hoa) va muc 2.3 (nen phai thang).

────────────────────────────────────────────────────────────────────────────
HAI MUC TIEU, CHO HAI VIEC KHAC NHAU — day la diem thiet ke quan trong nhat
────────────────────────────────────────────────────────────────────────────

MUC TIEU R (nghien cuu):  lop theo |z_h| < k, voi z_h = r_h / sigma_h
    sigma_h da bi CHIA RA, nen moi ky nang do duoc o day la ky nang VUOT TREN
    tang 2. Day la phep thu sach cho cau hoi "quy luat co them gi ngoai sigma?"
    He qua BAT BUOC phai xay ra: mo hinh "chi sigma" tren muc tieu R la mot
    HANG SO theo ngay. Tu kiem duoi day kiem dung dieu do — neu no khong con
    dung thi gia thiet phan phoi chuan hoa dung da vo, va do la mot phat hien.

MUC TIEU P (san pham):    lop theo |r_h| < b_h, voi b_h KHONG phu thuoc ngay
    b_h lay tu trung vi sigma 12 thang truot (doi cham theo thang, khong theo
    ngay), nen P(di ngang) = P(|z_h| < b_h/sigma_h) DOI MOI NGAY theo sigma.
    Day la thu hien len giao dien, va thong tin cua no den tu dung cho tang 2
    manh nhat.

────────────────────────────────────────────────────────────────────────────
NHAN QUA
────────────────────────────────────────────────────────────────────────────
`sig` trong panel la DU BAO da dich mot phien (chi dung thong tin den t-1) —
xem docs/DATASET.md. Nen tai moc t ta duoc dung sig[t], va dich la khoang
[t, t+h-1]. Ngưỡng k, he so c_h, tham so cua ca bon nen: uoc luong CHI tren
doan huan luyen cua split.py roi dong bang.

Tu kiem:  python src/balop.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from split import doan  # noqa: E402

PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF")
HS = (1, 5, 20)
CUA_SO_B = 252          # 12 thang truot cho dai cua muc tieu P
MIN_CUA_SO_B = 60
N_CHE_DO = 3
EPS = 1e-12


def nap(duong=None):
    """Panel theo cap: Date, sig, zT — da sap xep theo ngay."""
    p = duong or os.path.join(ROOT, "data", "panel2_6pairs.csv")
    pan = pd.read_csv(p, parse_dates=["Date"])
    return {c: d.sort_values("Date").reset_index(drop=True)
            for c, d in pan.groupby("pair")}


# ── muc tieu ────────────────────────────────────────────────────────────
def loi_suat_h(d, h):
    """r_h[t] = tong loi suat log tu t den t+h-1. NaN o duoi khi thieu ngay."""
    r = d.zT.values * d.sig.values
    n = len(r)
    cs = np.concatenate([[0.0], np.nancumsum(r)])
    ra = np.full(n, np.nan)
    ra[: n - h + 1] = cs[h:] - cs[: n - h + 1]
    return ra


def hieu_chinh_h(d, h, tr, che_do=None):
    """c_h — he so hieu chinh cho quy tac can(h), UOC LUONG TREN HUAN LUYEN.

    Quy tac can(h) gia dinh bien dong DUNG YEN suot h phien. No khong dung:
    bien dong hoi quy ve trung binh, nen o che do EM thi thuc te se dong manh
    hon du bao, con o che do CANG thi yeu hon. Do duoc tren doan kiem tra,
    ty le (sd thuc / sd du bao):

        h    binh tinh   vua    cang thang
        1      1,000    0,985     1,003
        5      1,057    0,905     0,861
       20      1,085    0,903     0,770      <- lech toi 31 diem phan tram

    Sai so nay lon dan theo h, va no la nguyen nhan h=20 AM CO Y NGHIA o che do
    em (BSS -0,0234). Nen c_h phai uoc RIENG TUNG CHE DO.

    che_do : mang ma che do (0/1/2) moi hang. None -> mot he so chung (ban cu).
    Tra ve so vo huong neu che_do=None, nguoc lai tra ve mang cung do dai d."""
    n = len(d)
    if h == 1:
        return 1.0 if che_do is None else np.ones(n)
    r = loi_suat_h(d, h)
    s = d.sig.values * np.sqrt(h)
    ok = tr & np.isfinite(r) & (s > EPS)
    if ok.sum() < 100:
        return 1.0 if che_do is None else np.ones(n)
    chung = float(np.std(r[ok] / s[ok]))
    if che_do is None:
        return chung
    ra = np.full(n, chung)
    for v in np.unique(np.asarray(che_do)):
        m = ok & (np.asarray(che_do) == v)
        if m.sum() >= 100:                     # du mau moi tach rieng
            ra[np.asarray(che_do) == v] = float(np.std(r[m] / s[m]))
    return ra


def sigma_h(d, h, c_h):
    """c_h co the la so vo huong (ban cu) hoac mang theo che do (ban moi)."""
    return d.sig.values * np.sqrt(h) * np.asarray(c_h, float)


def dai_P(d, h, c_h, kP=1.0):
    """b_h[t] — dai co dinh cua muc tieu P, tinh tu trung vi sigma 12 thang
    TRUOT (nhan qua: cua so ket thuc tai t, ma sig[t] da la du bao dich mot
    phien). Doi cham theo thang chu khong theo ngay.

    kP: he so nhan, CHON TREN HUAN LUYEN de "luc binh thuong ba o can nhau"
    (docs/REPLAN_2026.md muc 2.2). Do thang b = trung vi sigma (kP = 1) cho o
    vang chiem ~71%, KHONG can — vi trung vi sigma la ~1 sigma, ma
    P(|z| < 1) ~ 0,68. Muon can thi phai nhan them kP ~ 0,4."""
    s = pd.Series(d.sig.values)
    med = s.rolling(CUA_SO_B, min_periods=MIN_CUA_SO_B).median()
    med = med.bfill().values
    return kP * med * np.sqrt(h) * c_h


def chon_kP(d, h, c_h, tr, muc=1.0 / 3.0):
    """kP sao cho lop 'di ngang' cua muc tieu P chiem dung `muc` tren HUAN
    LUYEN. Giai bang cach dua ve phan vi cua |r_h| / (trung vi sigma truot)."""
    r = loi_suat_h(d, h)
    b1 = dai_P(d, h, c_h, kP=1.0)
    m = tr & np.isfinite(r) & (b1 > EPS)
    if m.sum() < 100:
        return 1.0
    return float(np.quantile(np.abs(r[m]) / b1[m], muc))


def chon_k(z_tr, muc=1.0 / 3.0):
    """k sao cho lop 'di ngang' chiem dung `muc` tren HUAN LUYEN."""
    z = z_tr[np.isfinite(z_tr)]
    return float(np.quantile(np.abs(z), muc)) if len(z) else 0.43


def gan_lop(gia_tri, dai):
    """0 = giam, 1 = di ngang, 2 = tang. NaN -> -1."""
    y = np.full(len(gia_tri), -1, int)
    ok = np.isfinite(gia_tri) & np.isfinite(dai)
    y[ok & (gia_tri < -dai)] = 0
    y[ok & (np.abs(gia_tri) <= dai)] = 1
    y[ok & (gia_tri > dai)] = 2
    return y


def dung_muc_tieu(d, h, tr, k=None, kP=None, theo_che_do=True):
    """Tra ve dict day du cho MOT cap, MOT tam han.

    k va kP deu CHON TREN HUAN LUYEN roi dong bang — truyen vao de dung lai
    nguong da chot cho cap/tam han khac neu can.

    theo_che_do: uoc c_h RIENG TUNG CHE DO bien dong. Mac dinh BAT vi quy tac
    can(h) chung lech toi 31 diem phan tram giua che do em va che do cang o
    h=20 — xem docstring cua hieu_chinh_h()."""
    if theo_che_do:
        ok = np.isfinite(d.sig.values)
        ng = np.quantile(d.sig.values[tr & ok], [1 / 3, 2 / 3])
        cd = np.digitize(d.sig.values, ng)
    else:
        cd = None
    c_h = hieu_chinh_h(d, h, tr, cd)
    r = loi_suat_h(d, h)
    sg = sigma_h(d, h, c_h)
    z = r / np.maximum(sg, EPS)
    if k is None:
        k = chon_k(z[tr])
    if kP is None:
        kP = chon_kP(d, h, c_h, tr)
    b = dai_P(d, h, c_h, kP)
    c_h_ra = float(np.mean(np.asarray(c_h, float)))
    return dict(c_h=c_h_ra, c_h_mang=np.asarray(c_h, float) * np.ones(len(d)),
                k=k, kP=kP, r=r, sigma_h=sg, z=z, b=b,
                yR=gan_lop(z, np.full(len(z), k)),
                yP=gan_lop(r, b),
                canh_R=np.full(len(z), k) * sg,   # dai muc tieu R quy ve don vi gia
                canh_P=b)


# ── bon nen ─────────────────────────────────────────────────────────────
def _chuan(P):
    P = np.clip(np.asarray(P, float), 1e-9, None)
    return P / P.sum(1, keepdims=True)


class KhiHauHoc:
    """NEN 1 — tan suat lop vo dieu kien tren huan luyen. San tuyet doi."""

    ten = "khí hậu học"

    def khop(self, y_tr):
        c = np.bincount(y_tr[y_tr >= 0], minlength=3).astype(float)
        self.p = c / max(c.sum(), 1.0)
        return self

    def du_bao(self, n, **kw):
        return _chuan(np.tile(self.p, (n, 1)))


class QuanTinh:
    """NEN 2 — ma tran chuyen tu lop cua cua so LIEN TRUOC (khong chong lan).

    Chan duong "quy luat" hoa ra chi la tinh dai tam thuong."""

    ten = "quán tính"

    def khop(self, y_tr, y_truoc_tr):
        M = np.ones((3, 3))                      # lam tron Laplace
        m = (y_tr >= 0) & (y_truoc_tr >= 0)
        for a, b in zip(y_truoc_tr[m], y_tr[m]):
            M[a, b] += 1
        self.M = M / M.sum(1, keepdims=True)
        self.p0 = np.bincount(y_tr[y_tr >= 0], minlength=3) / max((y_tr >= 0).sum(), 1)
        return self

    def du_bao(self, n, y_truoc=None, **kw):
        P = np.tile(self.p0, (n, 1))
        if y_truoc is None:
            return _chuan(P)
        m = np.asarray(y_truoc, int) >= 0
        P[m] = self.M[np.asarray(y_truoc, int)[m]]
        return _chuan(P)


class ChiSigma:
    """NEN 3 — NEN QUYET DINH. Student-t khop tren zT huan luyen, khong mau nao.

    Thang duoc nen nay moi co luan van. Tren MUC TIEU R no suy bien thanh hang
    so — do la chu y, xem docstring dau file."""

    ten = "chỉ σ̂"

    def khop(self, z_tr):
        z = z_tr[np.isfinite(z_tr)]
        nu, _, sc = stats.t.fit(z, floc=0)
        self.nu, self.sc = float(np.clip(nu, 2.05, 60)), float(sc)
        return self

    def du_bao(self, n, canh=None, sigma_h=None, **kw):
        e = np.asarray(canh, float) / np.maximum(np.asarray(sigma_h, float), EPS)
        F = stats.t.cdf(e / max(self.sc, EPS), self.nu)
        giua = np.clip(2 * F - 1, 1e-9, 1 - 1e-9)
        duoi = (1.0 - giua) / 2.0
        return _chuan(np.column_stack([duoi, giua, duoi]))


class SigmaCheDo:
    """NEN 4 — nhu NEN 3 nhung phan phoi chuan hoa uoc luong RIENG TUNG CHE DO
    bien dong (tam phan vi sigma tren huan luyen), theo tinh than Mondrian.

    Day la nen KHO nhat: no da chua san ca sigma LAN tinh dai cua che do."""

    ten = "σ̂ + chế độ"

    def khop(self, z_tr, sig_tr):
        self.nguong = np.quantile(sig_tr[np.isfinite(sig_tr)], [1 / 3, 2 / 3])
        ch = np.digitize(sig_tr, self.nguong)
        self.mau = []
        for v in range(N_CHE_DO):
            z = z_tr[(ch == v) & np.isfinite(z_tr)]
            self.mau.append(np.sort(z) if len(z) >= 50 else np.sort(z_tr[np.isfinite(z_tr)]))
        return self

    def du_bao(self, n, canh=None, sigma_h=None, sig=None, **kw):
        e = np.asarray(canh, float) / np.maximum(np.asarray(sigma_h, float), EPS)
        ch = np.digitize(np.asarray(sig, float), self.nguong)
        P = np.zeros((n, 3))
        for v in range(N_CHE_DO):
            m = ch == v
            if not m.any():
                continue
            z = self.mau[v]
            lo = np.searchsorted(z, -e[m]) / len(z)
            hi = np.searchsorted(z, e[m]) / len(z)
            P[m, 0] = lo
            P[m, 1] = np.clip(hi - lo, 1e-9, None)
            P[m, 2] = 1.0 - hi
        return _chuan(P)


# ── cua so mo rong cho tang ba lop ──────────────────────────────────────
BUOC_KHOP = 21          # khop lai moi ~1 thang giao dich
DAM_TOI_THIEU = 750     # can bao nhieu quan sat truoc khi duoc phep du bao


class NenCoSan:
    """Boc mot mang du bao DA TINH SAN thanh doi tuong co .du_bao().

    Dung de dua ket qua cua so mo rong vao tang to hop ma khong phai khop lai."""

    def __init__(self, ten, P, mo_cuoi=None):
        self.ten, self.P = ten, np.asarray(P, float)
        self.mo_cuoi = mo_cuoi

    def khop(self, *a, **k):
        return self

    def du_bao(self, n, **kw):
        return self.P[:n]

    def du_bao_ke_tiep(self, **kw):
        """Phien CHUA MO CUA khong co san trong mang — phai goi mo hinh khop
        gan nhat. Khong co thi tra ve hang cuoi cua lich su, con hon la tra ve
        hang DAU tien nhu `du_bao(1, ...)` se lam."""
        if self.mo_cuoi is not None:
            return self.mo_cuoi.du_bao(1, **kw)
        return self.P[-1:][:]


def du_bao_cuon(T, sig, tao_nen, buoc=BUOC_KHOP, dam=DAM_TOI_THIEU, tra_mo=False):
    """Du bao ba lop bang CUA SO MO RONG — thay vi dong bang o cuoi huan luyen.

    Tai moi moc khop lai, uoc phan phoi z va nguong che do tren TOAN BO quan
    sat TRUOC moc do, roi du bao cho toi moc ke. Dung giao thuc ma tang sigma^
    da chay san (volfc2 dung window=None, tuc cua so mo rong khop lai moi
    phien) — nay hai tang moi nhat quan.

    VI SAO NHAN VAO T DA DUNG SAN, KHONG DUNG lai nhan. Dinh nghia lop ("di
    ngang" nghia la gi, dai b rong bao nhieu) la QUYET DINH SAN PHAM, khong
    phai bai toan uoc luong: o vang khong duoc doi nghia duoi chan nguoi dung,
    va giu nhan co dinh moi so sanh duoc dong bang voi cuon mot cach cong bang.
    Cai duoc cuon la phan UOC LUONG XAC SUAT.

    NHAN QUA: khoi [t0, t0+buoc) chi dung mo hinh khop tren < t0. Tu kiem ep.
    """
    z, canh, sh = T["z"], T["canh_P"], T["sigma_h"]
    n = len(z)
    P = np.full((n, 3), np.nan)
    sig = np.asarray(sig, float)
    mo_cuoi = None
    for t0 in range(dam, n, buoc):
        ok = np.zeros(n, bool)
        ok[:t0] = True
        ok &= np.isfinite(z) & np.isfinite(sig) & (sig > 0)
        if ok.sum() < dam:
            continue
        mo = tao_nen(z[ok], sig[ok])
        mo_cuoi = mo
        Pb = mo.du_bao(n, canh=canh, sigma_h=sh, sig=sig)
        P[t0:min(t0 + buoc, n)] = Pb[t0:min(t0 + buoc, n)]
    return (P, mo_cuoi) if tra_mo else P


class ToHopTrucTuyen:
    """NEN 5 — HOC TRUC TUYEN (Hedge / trong so mu).

    Day la cau tra loi that cho "hom nay du bao sai thi mai phai chinh xac
    hon". Moi phien:
        1. du bao = to hop CAC CHUYEN GIA theo trong so HIEN TAI
        2. cho toi khi ket cuc phien do da biet, moi ha trong so cua chuyen gia
           vua sai, theo ham mu cua ton that log
    Chan hoi tiec O(sqrt(T log N)) so voi chuyen gia tot nhat nhin lai — tuc no
    khong the te hon han chuyen gia gioi nhat, va no tu loai chuyen gia hong ma
    khong can ai can thiep.

    NHAN QUA. Cap nhat dung ket cuc cua du bao dat ra o t-h, la thu chi biet
    duoc tai t. Voi h > 1 dieu nay QUAN TRONG: ket cuc cua du bao 5 phien dat o
    t phai doi toi t+5. `tre` chinh la h. Tu kiem ep dieu nay.

    Do duoc (output/ml3.json, doan kiem tra, muc tieu P, h=1):
        khi hau hoc   BSS  0
        chi sigma     BSS +0,0063  [+0,0038; +0,0088]
        sigma+che do  BSS +0,0074  [+0,0046; +0,0106]
        HEDGE         BSS +0,0076  [+0,0047; +0,0112]   <- tot nhat
    """

    ten = "tổ hợp trực tuyến"

    def __init__(self, chuyen_gia, eta=0.5, tre=1):
        """chuyen_gia: [(ten, doi_tuong_co_du_bao)]. eta chot 0,5 — la gia tri
        da do o src/run_ml3.py, KHONG duoc chinh lai tren doan kiem tra."""
        self.cg = list(chuyen_gia)
        self.eta = float(eta)
        self.tre = max(1, int(tre))

    def khop(self, *a, **k):
        return self

    def du_bao(self, n, y_that=None, **kw):
        """y_that: lop thuc tung phien (-1 = chua co ket cuc). Chi duoc dung DE
        CAP NHAT SAU KHI da du bao xong phien do."""
        A = np.stack([m.du_bao(n, **kw) for _, m in self.cg])      # (N, n, 3)
        N = A.shape[0]
        w = np.ones(N) / N
        ra = np.zeros((n, 3))
        lich_su = np.zeros((n, N))
        y = None if y_that is None else np.asarray(y_that, int)
        for t in range(n):
            ra[t] = _chuan((w[:, None] * A[:, t, :]).sum(0)[None, :])[0]
            lich_su[t] = w
            if y is None:
                continue
            j = t - self.tre + 1          # du bao dat o j vua du ket cuc tai t
            if j < 0 or y[j] < 0:
                continue
            ton = -np.log(np.maximum(A[:, j, y[j]], 1e-9))
            w = w * np.exp(-self.eta * (ton - ton.min()))
            w = w / max(w.sum(), EPS)
        self.trong_so = {t: float(v) for (t, _), v in zip(self.cg, w)}
        self.lich_su = lich_su
        return _chuan(ra)

    def du_bao_ke_tiep(self, **kw):
        """Mot hang cho phien CHUA MO CUA — dung TRONG SO DA HOC toi hom nay.

        Khong duoc goi du_bao(1, ...): the la khoi dong lai trong so tu deu nhau
        va vut bo toan bo phan da hoc."""
        if not hasattr(self, "trong_so"):
            raise RuntimeError("phải chạy du_bao() trên lịch sử trước")
        A = np.stack([(m.du_bao_ke_tiep(**kw) if hasattr(m, "du_bao_ke_tiep")
                       else m.du_bao(1, **kw)) for _, m in self.cg])[:, 0, :]
        w = np.array([self.trong_so[t] for t, _ in self.cg], float)
        return _chuan((w[:, None] * A).sum(0)[None, :])


def lop_truoc(y, h):
    """Lop cua cua so LIEN TRUOC khong chong lan: y[t-h]."""
    ra = np.full(len(y), -1, int)
    if h < len(y):
        ra[h:] = y[:-h]
    return ra


def _tu_kiem_to_hop():
    """Ba dieu phai dung, khong thi lop tren khong duoc phep chay san xuat."""
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 3, n)

    class Gia:                                   # chuyen gia gia lap, hang so
        def __init__(self, p): self.p = np.asarray(p, float)
        def du_bao(self, n, **kw): return np.tile(self.p, (n, 1))

    tot = Gia([.2, .6, .2])
    xau = Gia([.6, .2, .2])

    # 1) NHAN QUA — doi ket cuc TUONG LAI khong duoc doi du bao hom nay.
    for tre in (1, 5, 20):
        mo = ToHopTrucTuyen([("tốt", tot), ("xấu", xau)], tre=tre)
        P1 = mo.du_bao(n, y_that=y)
        y2 = y.copy()
        y2[-1] = (y2[-1] + 1) % 3               # sua DUNG ket cuc cuoi cung
        P2 = ToHopTrucTuyen([("tốt", tot), ("xấu", xau)], tre=tre).du_bao(n, y_that=y2)
        assert np.allclose(P1, P2), f"RO RI: ket cuc cuoi doi du bao truoc do (tre={tre})"

    # 2) HOC THAT — phai don trong so ve phia chuyen gia dung.
    yg = np.ones(n, int)                        # thuc te luon la "di ngang"
    mo = ToHopTrucTuyen([("tốt", tot), ("xấu", xau)], tre=1)
    mo.du_bao(n, y_that=yg)
    assert mo.trong_so["tốt"] > 0.99, f"khong hoc duoc: {mo.trong_so}"

    # 3) CHAN HOI TIEC — khong duoc te hon han chuyen gia gioi nhat.
    P = mo.du_bao(n, y_that=yg)
    lt = -np.log(P[np.arange(n), yg]).sum()
    lc = min(-np.log(np.tile(g.p, (n, 1))[np.arange(n), yg]).sum() for g in (tot, xau))
    can = mo.eta and np.sqrt(n * np.log(2)) / mo.eta
    assert lt - lc <= can, f"hoi tiec {lt-lc:.1f} vuot chan {can:.1f}"
    return dict(hoi_tiec=float(lt - lc), chan=float(can), trong_so=mo.trong_so)


def _tu_kiem_cuon(d, h=1):
    """Nhan qua: sua du lieu o CUOI chuoi khong duoc doi du bao o dau chuoi."""
    tr = doan(d.Date.values) == 0
    T = dung_muc_tieu(d, h, tr)
    sig = d.sig.values
    tao = lambda z, s: ChiSigma().khop(z)
    P1 = du_bao_cuon(T, sig, tao)
    T2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in T.items()}
    n = len(sig)
    T2["z"][-BUOC_KHOP:] = T2["z"][-BUOC_KHOP:] * 3.0 + 7.0   # pha hoai duoi chuoi
    P2 = du_bao_cuon(T2, sig, tao)
    m = np.isfinite(P1[:, 0]) & np.isfinite(P2[:, 0])
    cat = n - 2 * BUOC_KHOP
    assert np.allclose(P1[:cat][m[:cat]], P2[:cat][m[:cat]]), "RO RI: tuong lai doi qua khu"
    phu = float(np.isfinite(P1[:, 0]).mean())
    return dict(phu=phu, so_lan_khop=len(range(DAM_TOI_THIEU, n, BUOC_KHOP)))


if __name__ == "__main__":
    import diem3 as D

    kq = _tu_kiem_to_hop()
    print(f"  tổ hợp trực tuyến: nhân quả ĐẠT · học ĐẠT · hối tiếc "
          f"{kq['hoi_tiec']:.2f} ≤ chặn {kq['chan']:.1f} ĐẠT")

    B = nap()
    print("TỰ KIỂM")
    print(f"  {len(B)} cặp, {sum(len(d) for d in B.values()):,} phiên panel")

    d = B["EURUSD"]
    kc = _tu_kiem_cuon(d)
    print(f"  cửa sổ mở rộng: nhân quả ĐẠT · {kc['so_lan_khop']} lần khớp lại · "
          f"phủ {kc['phu']:.1%} số phiên")
    tr = doan(d.Date.values) == 0
    va = doan(d.Date.values) == 1

    # 1) can(h) va he so hieu chinh
    print("\n  hệ số hiệu chỉnh tầm hạn c_h (ước trên huấn luyện):")
    for h in HS:
        cs = [hieu_chinh_h(B[p], h, doan(B[p].Date.values) == 0) for p in PAIRS]
        print(f"    h={h:2d}: " + " ".join(f"{c:.3f}" for c in cs))
        assert all(0.7 < c < 1.4 for c in cs), "c_h ra ngoai khoang hop ly"

    # 2) k chon tren huan luyen phai cho ~1/3 di ngang TREN HUAN LUYEN
    print("\n  ngưỡng k và tỷ lệ lớp (EURUSD, h=1):")
    T = dung_muc_tieu(d, 1, tr)
    for ten, y in (("R", T["yR"]), ("P", T["yP"])):
        m = y >= 0
        tl = np.bincount(y[m], minlength=3) / m.sum()
        print(f"    mục tiêu {ten}: giảm {tl[0]:.3f}  đi ngang {tl[1]:.3f}  tăng {tl[2]:.3f}")
    for ten, y, nh in (("R", T["yR"], "k"), ("P", T["yP"], "kP")):
        tl = np.bincount(y[tr & (y >= 0)], minlength=3).astype(float)
        tl /= tl.sum()
        assert abs(tl[1] - 1 / 3) < 0.02, f"{nh} phai cho ~1/3 di ngang tren huan luyen"
    print(f"    k = {T['k']:.4f}   kP = {T['kP']:.4f}")

    # dai muc tieu P phai DUNG YEN theo ngay (chi doi cham theo thang)
    b = T["b"]
    dao_b = float(np.std(np.diff(b)) / max(np.mean(b), EPS))
    dao_s = float(np.std(np.diff(d.sig.values)) / max(np.mean(d.sig.values), EPS))
    print(f"    dao động ngày-qua-ngày: dải b {dao_b:.5f}  so với σ̂ {dao_s:.5f}"
          f"  (b phải êm hơn nhiều)")
    assert dao_b < dao_s / 10, "dai b phai doi cham hon sigma it nhat 10 lan"

    # 3) LUAN DIEM MUC 2.1 — kiem truc tiep
    n = len(d)
    ns = ChiSigma().khop(T["z"][tr])
    PR = ns.du_bao(n, canh=T["canh_R"], sigma_h=T["sigma_h"], sig=d.sig.values)
    PP = ns.du_bao(n, canh=T["canh_P"], sigma_h=T["sigma_h"], sig=d.sig.values)
    print("\n  P(đi ngang) của nền 'chỉ σ̂' — độ dao động theo ngày:")
    print(f"    mục tiêu R: sd={PR[:,1].std():.2e}  min={PR[:,1].min():.4f}  max={PR[:,1].max():.4f}")
    print(f"    mục tiêu P: sd={PP[:,1].std():.2e}  min={PP[:,1].min():.4f}  max={PP[:,1].max():.4f}")
    assert PR[:, 1].std() < 1e-9, "muc tieu R: P(di ngang) PHAI la hang so"
    assert PP[:, 1].std() > 0.05, "muc tieu P: P(di ngang) phai dao dong that su"
    print("    → đúng như mục 2.1: dải co theo σ̂ làm ô vàng đứng im; dải cố định thì không")

    # 4) nhan qua — dich phai KHONG dung thong tin sau t+h-1
    for h in (1, 5):
        T2 = dung_muc_tieu(d, h, tr)
        r = T2["r"]
        r2 = loi_suat_h(d.iloc[: 2000].reset_index(drop=True), h)
        k = 2000 - h
        assert np.allclose(r[:k], r2[:k], atol=1e-12, equal_nan=True), \
            f"r_h[t] doi khi cat bo tuong lai -> co ro ri (h={h})"
    print("\n  nhân quả: r_h[:t] không đổi khi cắt bỏ dữ liệu tương lai — ĐẠT")

    # 5) bon nen chay duoc va thang khi hau hoc dung huong
    ytr, yva = T["yR"][tr], T["yR"][va]
    ok = yva >= 0
    kh = KhiHauHoc().khop(ytr)
    qt = QuanTinh().khop(ytr, lop_truoc(T["yR"], 1)[tr])
    cd = SigmaCheDo().khop(T["z"][tr], d.sig.values[tr])
    Pkh = kh.du_bao(int(va.sum()))[ok]
    Pqt = qt.du_bao(int(va.sum()), y_truoc=lop_truoc(T["yR"], 1)[va])[ok]
    Pns = ns.du_bao(n, canh=T["canh_R"], sigma_h=T["sigma_h"], sig=d.sig.values)[va][ok]
    Pcd = cd.du_bao(n, canh=T["canh_R"], sigma_h=T["sigma_h"], sig=d.sig.values)[va][ok]
    yy = yva[ok]
    print("\n  bốn nền trên KIỂM ĐỊNH (EURUSD, mục tiêu R, h=1):")
    for t, P in (("khí hậu học", Pkh), ("quán tính", Pqt), ("chỉ σ̂", Pns), ("σ̂ + chế độ", Pcd)):
        print(f"    {t:<14} log={D.diem_log(P, yy):.4f}  brier={D.brier(P, yy):.4f}"
              f"  ece={D.ece(P, yy):.4f}  BSS={D.bss(P, yy, Pkh):+.4f}")
    for t, P in (("quán tính", Pqt), ("chỉ σ̂", Pns), ("σ̂ + chế độ", Pcd)):
        assert np.isfinite(D.diem_log(P, yy)), f"{t} cho diem log khong hop le"

    print("\nTỰ KIỂM ĐẠT")
