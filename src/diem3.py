"""CHI SO CHAM DIEM CHO DU BAO BA LOP — tang 5 cua ke hoach 2026.

San pham giao cho nguoi dung la BA PHAN TRAM, nen HIEU CHUAN chinh la san pham.
File nay cai bo chi so ma docs/REPLAN_2026.md muc 4 doi hoi.

Nguyen tac lay tu bai hoc da tra gia cua repo (docs/CHISO_DANHGIA.md):

  1. DO PHU THOI THI CHUA DU. PIT + Kolmogorov-Smirnov bac bo gia dinh chuan o
     p = 0,0001 trong khi chinh no vuot MOI backtest VaR. Nen cham bang QUY TAC
     CHAM DIEM CHINH DANG (log score, Brier), khong cham bang do phu.
  2. TRUNG BINH GOP GIAU DUNG CHO QUAN TRONG NHAT. QLIKE Q5/Q1 cua nen cu la
     1,82. Nen moi chi so o day deu co ban theo TUNG CAP va TUNG CHE DO.

Quy uoc: lop 0 = giam, 1 = di ngang, 2 = tang. P la mang (n, 3) tong hang = 1.

Tu kiem:  python src/diem3.py
"""
import numpy as np

TEN_LOP = ("giảm", "đi ngang", "tăng")
EPS = 1e-12


def _kiem(P, y):
    P = np.asarray(P, float)
    y = np.asarray(y, int)
    assert P.ndim == 2 and P.shape[1] == 3, "P phai co dang (n, 3)"
    assert len(P) == len(y), "P va y phai cung do dai"
    assert np.all((y >= 0) & (y <= 2)), "y phai thuoc {0,1,2}"
    assert np.allclose(P.sum(1), 1.0, atol=1e-6), "moi hang cua P phai tong bang 1"
    return P, y


def diem_log(P, y):
    """Diem log (log-loss). Cang NHO cang tot. Quy tac cham diem chinh dang."""
    P, y = _kiem(P, y)
    return float(-np.mean(np.log(np.maximum(P[np.arange(len(y)), y], EPS))))


def brier(P, y):
    """Brier nhieu lop (tong binh phuong sai so tren ca ba lop). Cang NHO cang tot."""
    P, y = _kiem(P, y)
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    return float(np.mean(((P - Y) ** 2).sum(1)))


def bss(P, y, P_nen):
    """Brier skill score so voi mot nen. 0 = khong hon nen, 1 = hoan hao.

    Day la con so BAO CAO CHINH vi no doc duoc truc tiep: BSS = 0,05 nghia la
    giam duoc 5% Brier so voi nen."""
    b, b0 = brier(P, y), brier(P_nen, y)
    return float(1.0 - b / max(b0, EPS))


def ece(P, y, nbin=10):
    """Sai so hieu chuan ky vong, tinh tren TUNG LOP roi lay trung binh co
    trong so (one-vs-rest) — dung hon la chi nhin xac suat lon nhat, vi ba o
    tren giao dien hien CA BA con so chu khong chi con lon nhat."""
    P, y = _kiem(P, y)
    canh = np.linspace(0.0, 1.0, nbin + 1)
    tong, n = 0.0, 0
    for c in range(3):
        p, o = P[:, c], (y == c).astype(float)
        idx = np.clip(np.digitize(p, canh[1:-1]), 0, nbin - 1)
        for b in range(nbin):
            m = idx == b
            if m.sum() == 0:
                continue
            tong += m.sum() * abs(p[m].mean() - o[m].mean())
            n += int(m.sum())
    return float(tong / max(n, 1))


def mce(P, y, nbin=10, n_toi_thieu=30):
    """Sai so hieu chuan LON NHAT tren mot thung. Bo qua thung qua it mau."""
    P, y = _kiem(P, y)
    canh = np.linspace(0.0, 1.0, nbin + 1)
    xau = 0.0
    for c in range(3):
        p, o = P[:, c], (y == c).astype(float)
        idx = np.clip(np.digitize(p, canh[1:-1]), 0, nbin - 1)
        for b in range(nbin):
            m = idx == b
            if m.sum() >= n_toi_thieu:
                xau = max(xau, abs(p[m].mean() - o[m].mean()))
    return float(xau)


def do_tin_cay(P, y, lop, nbin=10):
    """Du lieu ve bieu do tin cay cho MOT lop: (p_tb, tan_suat_that, n) moi thung."""
    P, y = _kiem(P, y)
    p, o = P[:, lop], (y == lop).astype(float)
    canh = np.linspace(0.0, 1.0, nbin + 1)
    idx = np.clip(np.digitize(p, canh[1:-1]), 0, nbin - 1)
    ra = []
    for b in range(nbin):
        m = idx == b
        ra.append((float(p[m].mean()) if m.sum() else np.nan,
                   float(o[m].mean()) if m.sum() else np.nan, int(m.sum())))
    return ra


def _auc1(s, lab):
    """Mann-Whitney tren mot nhom. Tra ve (auc, n1*n0) hoac (nan, 0)."""
    n1, n0 = int(lab.sum()), int(len(lab) - lab.sum())
    if n1 == 0 or n0 == 0:
        return np.nan, 0
    hang = np.argsort(np.argsort(s)) + 1.0
    for v in np.unique(s):                            # hang trung binh khi bang nhau
        k = s == v
        if k.sum() > 1:
            hang[k] = hang[k].mean()
    return float((hang[lab == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)), n1 * n0


def auc_huong(P, y, nhom=None):
    """KY NANG HUONG DI, tach rieng khoi truc di-ngang.

    Chi lay nhung phien THUC SU di ra khoi dai (y != 1), roi hoi: diem
    p_tang/(p_giam+p_tang) co phan biet duoc tang voi giam khong? 0,5 = khong.
    Day la con so PHAI in canh ba o — xem docs/REPLAN_2026.md muc 2.4.

    `nhom` (BAT BUOC khi gop nhieu cap): tinh AUC TRONG TUNG NHOM roi lay
    trung binh co trong so n1*n0. Khong phan tang thi mot du bao HANG SO cung
    ra AUC != 0,5, vi moi cap co tan suat nen rieng va viec gop tao ra kha nang
    phan biet GIUA CAC CAP — day khong phai ky nang dinh thoi. Da do: khi hau
    hoc gop 6 cap cho AUC 0,60 o h=20 trong khi su that phai la 0,50."""
    P, y = _kiem(P, y)
    m = y != 1
    if m.sum() < 20:
        return np.nan
    s = P[m, 2] / np.maximum(P[m, 0] + P[m, 2], EPS)
    lab = (y[m] == 2).astype(int)
    if nhom is None:
        return _auc1(s, lab)[0]
    g = np.asarray(nhom)[m]
    tu, mau = 0.0, 0.0
    for v in np.unique(g):
        k = g == v
        a, w = _auc1(s[k], lab[k])
        if w > 0 and np.isfinite(a):
            tu += a * w
            mau += w
    return float(tu / mau) if mau > 0 else np.nan


def auc_ktc(P, y, nhom=None, nboot=400, khoi=20, seed=0):
    """Khoang tin cay bootstrap THEO KHOI cho auc_huong.

    Khoi vi chuoi thoi gian co tu tuong quan — bootstrap tung diem se cho KTC
    hep gia. Khi co `nhom`, lay mau khoi RIENG TRONG TUNG NHOM roi ghep lai,
    de cau truc phan tang duoc giu nguyen qua moi lan lap."""
    P, y = _kiem(P, y)
    n = len(y)
    if n < 100:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    g = np.zeros(n, int) if nhom is None else np.asarray(nhom)
    chi = [np.flatnonzero(g == v) for v in np.unique(g)]
    ra = []
    for _ in range(nboot):
        lay = []
        for c in chi:
            m = len(c)
            nk = int(np.ceil(m / khoi))
            b = rng.integers(0, max(m - khoi, 1), nk)
            lay.append(c[np.concatenate([np.arange(s, min(s + khoi, m))
                                         for s in b])[:m]])
        idx = np.concatenate(lay)
        v = auc_huong(P[idx], y[idx], None if nhom is None else g[idx])
        if np.isfinite(v):
            ra.append(v)
    if len(ra) < 20:
        return (np.nan, np.nan)
    return (float(np.quantile(ra, 0.025)), float(np.quantile(ra, 0.975)))


def bss_ktc(P, y, P_nen, nhom=None, nboot=400, khoi=20, seed=0):
    """KTC bootstrap THEO KHOI cho BSS.

    Bat buoc phai co: BSS o day chi co bien do ~1%, va voi tam han h>1 cac cua
    so CHONG LAN nen mau huu hieu chi bang n/h. Bao cao BSS tran khong kem KTC
    se noi qua muc bang chung. Lay mau khoi rieng trong tung nhom (cap)."""
    P, y = _kiem(P, y)
    n = len(y)
    if n < 100:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    g = np.zeros(n, int) if nhom is None else np.asarray(nhom)
    chi = [np.flatnonzero(g == v) for v in np.unique(g)]
    ra = []
    for _ in range(nboot):
        lay = []
        for c in chi:
            m = len(c)
            nk = int(np.ceil(m / khoi))
            b = rng.integers(0, max(m - khoi, 1), nk)
            lay.append(c[np.concatenate([np.arange(s, min(s + khoi, m))
                                         for s in b])[:m]])
        idx = np.concatenate(lay)
        v = bss(P[idx], y[idx], P_nen[idx])
        if np.isfinite(v):
            ra.append(v)
    if len(ra) < 20:
        return (np.nan, np.nan)
    return (float(np.quantile(ra, 0.025)), float(np.quantile(ra, 0.975)))


def bang(P, y, P_nen=None, nbin=10, nhom=None):
    """Mot dong chi so day du. Truyen `nhom` khi da gop nhieu cap."""
    r = dict(n=int(len(y)), log=diem_log(P, y), brier=brier(P, y),
             ece=ece(P, y, nbin), mce=mce(P, y, nbin),
             auc=auc_huong(P, y, nhom))
    if P_nen is not None:
        r["bss"] = bss(P, y, P_nen)
    return r


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 20000
    # sinh du lieu co hieu chuan HOAN HAO: P sinh ra truoc, y rut tu chinh P
    a = rng.dirichlet([2.0, 3.0, 2.0], n)
    y = np.array([rng.choice(3, p=p) for p in a])

    print("TỰ KIỂM")
    print(f"  hiệu chuẩn hoàn hảo : log={diem_log(a, y):.4f}  brier={brier(a, y):.4f}"
          f"  ece={ece(a, y):.4f}  mce={mce(a, y):.4f}")
    assert ece(a, y) < 0.02, "du bao hieu chuan hoan hao phai co ECE ~ 0"

    # khi hau hoc: hang so bang tan suat that
    tan = np.bincount(y, minlength=3) / len(y)
    kh = np.tile(tan, (n, 1))
    print(f"  khí hậu học         : log={diem_log(kh, y):.4f}  brier={brier(kh, y):.4f}"
          f"  ece={ece(kh, y):.4f}")
    assert diem_log(a, y) < diem_log(kh, y), "du bao that phai thang khi hau hoc"
    assert bss(a, y, kh) > 0, "BSS so khi hau hoc phai duong"
    assert abs(bss(kh, y, kh)) < 1e-9, "BSS cua chinh nen phai bang 0"

    # du bao qua tu tin: LAM NHON bang nhiet do (mu < 1), khong phai nhan hang
    # so — nhan ca ba cot voi cung mot so roi chuan hoa lai la phep DONG NHAT.
    tt = a ** 2.5
    tt /= tt.sum(1, keepdims=True)
    print(f"  quá tự tin          : log={diem_log(tt, y):.4f}  ece={ece(tt, y):.4f}")
    assert ece(tt, y) > ece(a, y), "du bao qua tu tin phai co ECE lon hon"
    assert diem_log(tt, y) > diem_log(a, y), "lam nhon qua da phai lam xau diem log"

    # ── AUC huong ────────────────────────────────────────────────────────
    # CO tin hieu: y rut tu chinh a, nen p_tang/(p_giam+p_tang) that su bao
    # duoc dau. Day la truong hop du bao huong CO KY NANG.
    u2 = auc_huong(a, y)
    lo2, hi2 = auc_ktc(a, y, nboot=200, seed=1)
    print(f"  AUC hướng (CÓ tín hiệu)      : {u2:.4f}  KTC 95% [{lo2:.3f}, {hi2:.3f}]")
    assert lo2 > 0.5, "co tin hieu thi KTC phai nam han tren 0,5"

    # KHONG co tin hieu: giu nguyen du bao a, nhung rut y tu ban DOI XUNG cua
    # a — cung xac suat di ngang, con tang/giam chia deu 50/50. Luc do cot
    # tang/giam cua a khong con noi gi ve dau, dung tinh huong ma tang 1 da do
    # duoc tren du lieu that (E[zT] = 0).
    doi_xung = a.copy()
    tb = (doi_xung[:, 0] + doi_xung[:, 2]) / 2.0
    doi_xung[:, 0] = doi_xung[:, 2] = tb
    y0 = np.array([rng.choice(3, p=p) for p in doi_xung])
    u = auc_huong(a, y0)
    lo, hi = auc_ktc(a, y0, nboot=200, seed=1)
    print(f"  AUC hướng (KHÔNG tín hiệu)   : {u:.4f}  KTC 95% [{lo:.3f}, {hi:.3f}]")
    assert lo <= 0.5 <= hi, "khong co tin hieu huong thi KTC phai phu 0,5"
    assert u2 > u, "truong hop co tin hieu phai cho AUC cao hon"

    # ── ao giac do GOP NHIEU CAP, va cach phan tang chua ──────────────────
    # Dung lai dung tinh huong da gap that: moi "cap" co tan suat tang RIENG,
    # du bao la HANG SO cua chinh cap do. Khong co ky nang dinh thoi nao ca —
    # AUC that phai la 0,50. Nhung neu gop ma khong phan tang thi no > 0,5.
    ncap, mc = 6, 3000
    ps = np.linspace(0.30, 0.55, ncap)          # cap khac nhau ve tan suat nen
    Pg, yg, gg = [], [], []
    for i, pu in enumerate(ps):
        pf = 0.32
        pr = np.array([1 - pf - pu, pf, pu])
        Pg.append(np.tile(pr, (mc, 1)))
        yg.append(rng.choice(3, size=mc, p=pr))
        gg.append(np.full(mc, i))
    Pg, yg, gg = np.vstack(Pg), np.concatenate(yg), np.concatenate(gg)
    a_gop = auc_huong(Pg, yg)
    a_pt = auc_huong(Pg, yg, nhom=gg)
    lo3, hi3 = auc_ktc(Pg, yg, nhom=gg, nboot=200, seed=2)
    print(f"  AUC hằng số, gộp KHÔNG phân tầng : {a_gop:.4f}   ← ảo giác")
    print(f"  AUC hằng số, CÓ phân tầng theo cặp: {a_pt:.4f}  KTC 95% [{lo3:.3f}, {hi3:.3f}]")
    assert a_gop > 0.53, "vi du phai tai hien duoc ao giac (neu khong thi test vo nghia)"
    assert abs(a_pt - 0.5) < 1e-9, "du bao hang so, phan tang dung, PHAI cho AUC = 0,5"
    assert lo3 <= 0.5 <= hi3, "KTC phan tang phai phu 0,5"

    # do_tin_cay: tong n moi lop phai bang n
    for c in range(3):
        assert sum(r[2] for r in do_tin_cay(a, y, c)) == n

    print("TỰ KIỂM ĐẠT")
