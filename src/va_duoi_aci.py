"""VA DUOI — HUONG THU TAM: DUOI TRUC TUYEN (ACI / AgACI).

Gia thuyet, cau hinh, giao thuc va tieu chi phu dinh DA CHOT TRUOC o
`docs/DUOI_ACI_TIEUCHI.md` (commit 8ab4eb5, truoc moi commit co so lieu).

DIEM KHAC BAY HUONG TRUOC — o LOP van de, khong o uoc luong duoi. Bay huong
truoc deu de xuat k cau hinh roi chon 1 tren KIEM DINH; TONG_QUAN muc 5.1 da
chan doan chinh buoc do la cho hong (DUOI_EVT A4h: lan thu tu chon-tren-kiem-
dinh chon sai). Huong nay BO buoc do: A1 la cau hinh de xuat, chot trong van
ban tieu chi truoc khi thay bat ky con so nao.

ACI (Gibbs & Candes 2021, NeurIPS 34:1660-1672) khong mo hinh hoa duoi. No la
vong dieu khien phan hoi tren chinh muc vi pham:

    alpha_{t+1} = alpha_t + gamma*(alpha - err_t)      err_t = 1{z_t <= q_t}
    q_t         = phan vi muc alpha_t cua z trong cua so hieu chuan

Bao dam do phu dai han duoi dich chuyen phan phoi TUY Y (Menh de 4.1) — la
DINH LY, khong phai ket qua thuc nghiem. Nhung xem muc 8 cua van ban tieu chi:
voi T=729 can la 0,274 o alpha=1%, LONG hon chinh alpha, nen phan dong gop
that o quy mo mau nay la hanh vi huu han mau cua vong phan hoi.

NAM CAU HINH (muc 4 van ban tieu chi):
  A1  ACI tren muc alpha, gamma=0,01   <- DE XUAT, ke thua tu conformal.py san xuat
  A2  bam quantile truc tiep (gradient pinball), eta = 0,01*sd(z) huan luyen
  A3  AgACI — Hedge tren gamma in {0,002 .. 0,05}
  A4  nhu A1, gamma=0,005              <- chi bao cao DO NHAY, khong duoc chon
  A5  nhu A1, gamma=0,02               <- chi bao cao DO NHAY, khong duoc chon

Chay:  python src/va_duoi_aci.py
       python src/va_duoi_aci.py --tu-kiem      (chi chay ba tu kiem)
       python src/va_duoi_aci.py --mo-kiem-tra   (mo DUNG MOT LAN)
Ghi:   output/va_duoi_aci.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                              # noqa: E402
from metrics import dq_test                                    # noqa: E402
from va_duoi import nap, cham, phan_vi_cuon, MUC, DAM, CUON     # noqa: E402

MO_KIEM_TRA = "--mo-kiem-tra" in sys.argv
EPS = 1e-12

GAMMA_CHINH = 0.01          # ke thua nguyen trang tu conformal.py::chay_aci san xuat
GAMMA_NHAY = (0.005, 0.02)  # A4/A5 — CHI bao cao do nhay
GAMMA_AGACI = (0.002, 0.005, 0.01, 0.02, 0.05)
ETA_HEDGE = 0.5             # ke thua tu balop.Hedge san xuat (KHOA_SO muc 4.2)
A_SAN = 1e-4                # chan duoi/tren cho alpha_t, nhu chay_aci


# ───────────────────────────────────────── loi: ba co che truc tuyen
def can_aci(a, T, gamma):
    """Can HAI PHIA cua Gibbs & Candes cho |do phu - alpha|.

    Cong don alpha_{t+1} = alpha_t + gamma*(alpha - err_t) cho
        (1/T)*sum(err_t) - alpha = (alpha_1 - alpha_{T+1}) / (T*gamma)
    nen can la |alpha_1 - alpha_{T+1}|/(T*gamma). Voi alpha_1 = alpha va
    alpha_{T+1} nam trong [0, 1] thi |alpha_1 - alpha_{T+1}| <= max(a, 1-a),
    cong them mot buoc gamma cho lan vuot cuoi:

        |do phu - alpha|  <=  (max(a, 1-a) + gamma) / (T*gamma)

    KHAI BAO SUA (15/09/2026, TRUOC khi cham bat ky so lieu that nao): muc 8
    cua `DUOI_ACI_TIEUCHI.md` ban dau viet `(alpha_1 + gamma)/(T*gamma)` —
    do la ban MOT PHIA, chi dung cho chieu thieu phu. Tu kiem 3 bat duoc ngay
    tren du lieu mo phong. Ket luan dinh tinh cua muc 8 KHONG doi (that ra con
    manh hon): can long hon chinh alpha o ca hai muc.

    CANH BAO ve gia thiet: phep dan tren gia dinh alpha_t KHONG bi chan. Cai
    dat nay (va ca `conformal.py::chay_aci` san xuat) co chan alpha_t vao
    [A_SAN, 1-A_SAN] vi khong lay duoc phan vi o muc am. Khi alpha_t cham san,
    dang thuc cong don khong con dung chinh xac — nen dong "trong can" phai
    doc kem ty le cham san bao cao o tu kiem 3.
    """
    return (max(a, 1.0 - a) + gamma) / (max(T, 1) * gamma)


def _qe_cua_so(cua, a):
    """Phan vi muc `a` va ES (trung binh duoi phan vi) tren mot cua so."""
    if len(cua) < 20:
        return np.nan, np.nan
    q = float(np.quantile(cua, min(max(a, A_SAN), 1 - A_SAN)))
    duoi = cua[cua <= q]
    return q, (float(duoi.mean()) if len(duoi) else q)


def aci_muc(z, a, gamma, dam=DAM, cuon=CUON):
    """A1/A4/A5 — ACI tren MUC alpha. Tra ve (qz, ez, alphas).

    NHAN QUA: q_t chi dung z[<t]. Ket cuc z_t chi duoc dung SAU khi da phat
    q_t cho phien do — dung nhu khi chay that.
    """
    n = len(z)
    qz = np.full(n, np.nan)
    ez = np.full(n, np.nan)
    al = np.full(n, np.nan)
    a_t = float(a)
    for t in range(dam, n):
        cua = z[max(0, t - cuon):t]
        cua = cua[np.isfinite(cua)]
        q, e = _qe_cua_so(cua, a_t)
        qz[t], ez[t], al[t] = q, e, a_t
        if not np.isfinite(z[t]) or not np.isfinite(q):
            continue
        err = 1.0 if z[t] <= q else 0.0
        a_t = float(np.clip(a_t + gamma * (a - err), A_SAN, 1 - A_SAN))
    return qz, ez, al


def bam_quantile(z, a, eta, dam=DAM, cuon=CUON):
    """A2 — gradient duoi cua ham mat pinball ngay tren muc quantile.

        q_{t+1} = q_t + eta*(alpha - 1{z_t <= q_t})

    Vi pham -> q lui xuong (noi ra); khong vi pham -> q nhich len (thu lai).
    Khoi tao q bang phan vi thuc nghiem tren cua so dau tien (khong phai 0,
    vi 0 se ton hang tram phien de bo toi vung dung).
    """
    n = len(z)
    qz = np.full(n, np.nan)
    ez = np.full(n, np.nan)
    cua0 = z[max(0, dam - cuon):dam]
    cua0 = cua0[np.isfinite(cua0)]
    q_t = float(np.quantile(cua0, a)) if len(cua0) >= 20 else np.nan
    for t in range(dam, n):
        cua = z[max(0, t - cuon):t]
        cua = cua[np.isfinite(cua)]
        qz[t] = q_t
        if len(cua) and np.isfinite(q_t):
            duoi = cua[cua <= q_t]
            ez[t] = float(duoi.mean()) if len(duoi) else q_t
        if not np.isfinite(z[t]) or not np.isfinite(q_t):
            continue
        err = 1.0 if z[t] <= q_t else 0.0
        q_t = q_t + eta * (a - err)
    return qz, ez


def _pinball(q, y, a):
    """Ham mat pinball cho muc quantile a."""
    d = y - q
    return np.where(d >= 0, a * d, (a - 1.0) * d)


def agaci(z, a, gammas=GAMMA_AGACI, eta_hedge=ETA_HEDGE, sd_chuan=1.0,
          dam=DAM, cuon=CUON):
    """A3 — AgACI (Zaffran et al. 2022): chay nhieu gamma SONG SONG roi to hop
    truc tuyen bang trong so mu, thay vi CHON mot gamma.

    Mat mat pinball duoc chuan hoa bang `sd_chuan` = sd(z) tren HUAN LUYEN de
    dai luong dua vao Hedge la khong thu nguyen; eta=0,5 ke thua tu
    `balop.Hedge` san xuat. Ca hai deu la hang so chot truoc, khong do tren du
    lieu — day la diem giu cho phat bieu "A3 khong co sieu tham so phai chon".
    """
    n = len(z)
    K = len(gammas)
    qz = np.full(n, np.nan)
    ez = np.full(n, np.nan)
    a_k = np.full(K, float(a))
    w = np.ones(K) / K
    ls_w = np.full((n, K), np.nan)
    for t in range(dam, n):
        cua = z[max(0, t - cuon):t]
        cua = cua[np.isfinite(cua)]
        q_k = np.empty(K)
        e_k = np.empty(K)
        for k in range(K):
            q_k[k], e_k[k] = _qe_cua_so(cua, a_k[k])
        ok = np.isfinite(q_k)
        if not ok.any():
            continue
        ww = np.where(ok, w, 0.0)
        s = ww.sum()
        if s <= EPS:
            continue
        ww = ww / s
        qz[t] = float(ww @ np.where(ok, q_k, 0.0))
        ez[t] = float(ww @ np.where(ok, e_k, 0.0))
        ls_w[t] = w
        if not np.isfinite(z[t]):
            continue
        ton = _pinball(q_k, z[t], a) / max(sd_chuan, EPS)       # khong thu nguyen
        ton = np.where(ok, ton, np.nanmax(ton[ok]) if ok.any() else 0.0)
        w = w * np.exp(-eta_hedge * (ton - ton.min()))
        w = w / max(w.sum(), EPS)
        for k in range(K):
            if not np.isfinite(q_k[k]) or not np.isfinite(z[t]):
                continue
            err = 1.0 if z[t] <= q_k[k] else 0.0
            a_k[k] = float(np.clip(a_k[k] + gammas[k] * (a - err), A_SAN, 1 - A_SAN))
    return qz, ez, ls_w


# ───────────────────────────────────────── chan doan DQ (canh bao muc 3)
def dq_chan_doan(z, sig, qz, mask, a):
    """Tra ve (ac1, b_lag): tu tuong quan tre-1 cua chuoi vi pham, va TONG he so
    tre trong hoi quy DQ.

    Muc 3 van ban tieu chi doi phan biet hai chieu: DUONG = vi pham DON CUM
    (khiem khuyet dang co), AM = vi pham DAY NHAU, tuc tu tuong quan am NHAN
    TAO do chinh vong phan hoi tao ra. Neu DQ bac bo vi chieu AM thi la THAT
    BAI cua H17b, khong duoc doc thanh "gan dat".
    """
    m = mask & np.isfinite(z) & np.isfinite(sig) & (sig > 0) & np.isfinite(qz)
    if m.sum() < 100:
        return None, None
    y = z[m] * sig[m]
    v = qz[m] * sig[m]
    h = (y <= v).astype(float)
    if h.sum() < 2 or h.std() < EPS:
        return None, None
    ac1 = float(np.corrcoef(h[:-1], h[1:])[0, 1]) if len(h) > 2 else np.nan
    lags = 4
    hc = h - a
    n = len(hc)
    X = [np.ones(n - lags)]
    for l in range(1, lags + 1):
        X.append(hc[lags - l:n - l])
    X.append(v[lags:])
    X = np.column_stack(X)
    Y = hc[lags:]
    try:
        b = np.linalg.solve(X.T @ X, X.T @ Y)
    except np.linalg.LinAlgError:
        return ac1, None
    return ac1, float(np.sum(b[1:1 + lags]))


# ───────────────────────────────────────── ba tu kiem BAT BUOC (muc 6)
def tu_kiem():
    print("TỰ KIỂM (docs/DUOI_ACI_TIEUCHI.md mục 6)")
    rng = np.random.default_rng(23)
    dat = True

    # ── tu kiem 1: NHAN QUA — cat du lieu tuong lai khong doi q_t qua khu
    z = rng.standard_t(6, 3000) / np.sqrt(6 / 4)
    q_full, _, _ = aci_muc(z, 0.01, GAMMA_CHINH)
    q_cut, _, _ = aci_muc(z[:2200], 0.01, GAMMA_CHINH)
    k = min(len(q_cut), 2200)
    doi = np.nansum(np.abs(q_full[:k] - q_cut[:k]) > 1e-12)
    ok = doi == 0
    dat &= ok
    print(f"  1 nhân quả: cắt 800 phiên cuối → {doi} giá trị q_t đổi ở tiền tố  "
          f"{'ĐẠT' if ok else 'HỎNG'}")

    # ── tu kiem 2: DINH LY dung duoi DICH CHUYEN PHAN PHOI biet truoc
    n = 6000
    z2 = np.empty(n)
    z2[:n // 2] = rng.standard_normal(n // 2)
    z2[n // 2:] = rng.standard_t(3, n - n // 2) * 1.8       # doi ca thang do lan duoi
    for a in (0.05, 0.01):
        q_aci, _, _ = aci_muc(z2, a, GAMMA_CHINH)
        q_tinh = float(np.quantile(z2[DAM - CUON:DAM], a))   # tinh, uoc mot lan
        m = np.isfinite(q_aci)
        phu_aci = float(np.mean(z2[m] <= q_aci[m]))
        phu_tinh = float(np.mean(z2[m] <= q_tinh))
        ok = abs(phu_aci - a) < abs(phu_tinh - a)
        dat &= ok
        print(f"  2 dịch chuyển α={a}: ACI {phu_aci:.4f} · tĩnh {phu_tinh:.4f} "
              f"· đích {a}  {'ĐẠT' if ok else 'HỎNG'}")

    # ── tu kiem 3: CAN ly thuyet, va CHAN DOAN bao hoa cua alpha_t
    for a in (0.05, 0.01):
        q_aci, _, al = aci_muc(z2, a, GAMMA_CHINH)
        m = np.isfinite(q_aci)
        T = int(m.sum())
        phu = float(np.mean(z2[m] <= q_aci[m]))
        can = can_aci(a, T, GAMMA_CHINH)
        ok = abs(phu - a) <= can
        dat &= ok
        sat = float(np.mean(al[np.isfinite(al)] <= A_SAN * 1.001))
        print(f"  3 cận α={a}: |{phu:.4f}−{a}| = {abs(phu-a):.4f} ≤ "
              f"{can:.4f} (T={T})  {'ĐẠT' if ok else 'HỎNG'}"
              f"   · α_t chạm sàn {100*sat:.1f}% phiên")

    print(f"→ TỰ KIỂM {'ĐẠT' if dat else 'HỎNG'}")
    return dat


# ───────────────────────────────────────── cau hinh CHOT TRUOC
CHINH = ["A1 ACI γ=0,01 (đề xuất)", "A2 bám quantile", "A3 AgACI"]
NHAY = ["A4 ACI γ=0,005 (độ nhạy)", "A5 ACI γ=0,02 (độ nhạy)"]
MOC = "V0 phân vị huấn luyện (mốc)"


def tinh_het(z, g, a, sd_hl):
    """Tra ve dict ten -> (qz, ez) cho moc + 5 cau hinh."""
    ra = {}
    qz, ez = phan_vi_cuon(z, g, a, "huan_luyen")
    ra[MOC] = (qz, ez)
    q, e, _ = aci_muc(z, a, GAMMA_CHINH)
    ra[CHINH[0]] = (q, e)
    q, e = bam_quantile(z, a, 0.01 * sd_hl)
    ra[CHINH[1]] = (q, e)
    q, e, _ = agaci(z, a, sd_chuan=sd_hl)
    ra[CHINH[2]] = (q, e)
    for ten, gm in zip(NHAY, GAMMA_NHAY):
        q, e, _ = aci_muc(z, a, gm)
        ra[ten] = (q, e)
    return ra


def main():
    if "--tu-kiem" in sys.argv:
        sys.exit(0 if tu_kiem() else 1)
    if not tu_kiem():
        sys.exit("tự kiểm HỎNG — dừng, không chấm số liệu thật")

    t0 = time.time()
    D = nap()
    print("\n" + "=" * 118)
    print("VÁ ĐUÔI — HƯỚNG THỨ TÁM: ĐUÔI TRỰC TUYẾN (ACI / AgACI)")
    print("tiêu chí chốt trước: docs/DUOI_ACI_TIEUCHI.md · A1 là cấu hình ĐỀ XUẤT,")
    print("KHÔNG chọn lại trên kiểm định (mục 5)")
    print("=" * 118)

    ra = {"gamma": GAMMA_CHINH, "kiem_dinh": {}, "kiem_tra": {}, "chan_doan": {}}
    TEN_HET = [MOC] + CHINH + NHAY

    cache = {}
    for p in B.PAIRS:
        z, g = D[p]["z"], D[p]["g"]
        sd_hl = float(np.std(z[(g == 0) & np.isfinite(z)]))
        for a in MUC:
            cache[(p, a)] = tinh_het(z, g, a, sd_hl)
        print(f"  {p}: sd(z) huấn luyện = {sd_hl:.4f} · đã tính 5 cấu hình")

    doan_chay = [("kiem_dinh", 1)] + ([("kiem_tra", 2)] if MO_KIEM_TRA else [])
    for nhan_doan, gid in doan_chay:
        print(f"\n── ĐOẠN {nhan_doan.upper()} " + "─" * 80)
        for a in MUC:
            print(f"\n  mức {1-a:.0%} (α={a})")
            print(f"  {'cặp':9}{'phương án':<26}{'vi phạm':>9}{'Kupiec':>9}"
                  f"{'Chris':>8}{'DQ':>8}{'tỷ lệ ES':>10}{'ac1':>8}{'Σb_trễ':>9}")
            for p in B.PAIRS:
                z, sig, g = D[p]["z"], D[p]["sig"], D[p]["g"]
                dau = True
                for ten in TEN_HET:
                    qz, ez = cache[(p, a)][ten]
                    r = cham(z, sig, qz, ez, g == gid, a)
                    if r is None:
                        continue
                    ac1, blag = dq_chan_doan(z, sig, qz, g == gid, a)
                    r = dict(r, ac1=None if ac1 is None or not np.isfinite(ac1)
                             else round(ac1, 4),
                             b_tre=None if blag is None or not np.isfinite(blag)
                             else round(blag, 4))
                    ra[nhan_doan].setdefault(f"{a}", {}).setdefault(p, {})[ten] = r
                    f = lambda v: "  —  " if v is None else f"{v:.3f}"
                    print(f"  {p if dau else '':9}{ten:<26}{100*r['vi_pham']:>8.2f}%"
                          f"{f(r['kupiec']):>9}{f(r['chris']):>8}{f(r['dq']):>8}"
                          f"{f(r['ty_le_es']):>10}{f(r['ac1']):>8}{f(r['b_tre']):>9}")
                    dau = False
                print()

    # ── H17a: Kupiec khong bac bo o >= 5/6 cap, ca hai muc (muc 3 van ban)
    print("=" * 118)
    print("H17a — HỆ QUẢ CỦA ĐỊNH LÝ: Kupiec không bác bỏ ở ≥ 5/6 cặp, cả hai mức")
    ra["h17a"] = {}
    for nhan_doan, _ in doan_chay:
        for ten in [MOC] + CHINH:
            dong = []
            for a in MUC:
                c = sum(1 for p in B.PAIRS
                        if ((ra[nhan_doan].get(f"{a}", {}).get(p, {}).get(ten) or {})
                            .get("kupiec") or 1.0) >= 0.05)
                dong.append((a, c))
            ok = all(c >= 5 for _, c in dong)
            ra["h17a"].setdefault(nhan_doan, {})[ten] = dict(
                theo_muc={str(a): c for a, c in dong}, dat=bool(ok))
            mo_ta = " · ".join(f"α={a}: {c}/6" for a, c in dong)
            print(f"  {nhan_doan:<11}{ten:<26}{mo_ta}   "
                  f"{'ĐẠT' if ok else 'TRƯỢT'}")

    # ── do phu thuc nghiem so voi CAN ly thuyet (muc 8 — can long, phai noi ro)
    print("\nĐỘ PHỦ THỰC NGHIỆM so với CẬN lý thuyết (mục 8: cận lỏng ở T này)")
    print(f"  {'cặp':9}{'α':>7}{'vi phạm A1':>13}{'|lệch|':>9}{'cận':>9}{'':>6}")
    for p in B.PAIRS:
        for a in MUC:
            r = ra["kiem_dinh"].get(f"{a}", {}).get(p, {}).get(CHINH[0])
            if not r or r["vi_pham"] is None:
                continue
            lech = abs(r["vi_pham"] - a)
            can = can_aci(a, r["n"], GAMMA_CHINH)
            print(f"  {p:9}{a:>7}{100*r['vi_pham']:>12.2f}%{lech:>9.4f}{can:>9.4f}"
                  f"{'  trong cận' if lech <= can else '  NGOÀI cận':>6}")

    if MO_KIEM_TRA and ra["kiem_tra"]:
        E = CHINH[0]
        dq_dat, hong, es_lech = {"USDJPY": None, "USDCHF": None}, [], 0
        am_nhan_tao = []
        for a in MUC:
            for p in B.PAIRS:
                rv = ra["kiem_tra"].get(f"{a}", {}).get(p, {}).get(MOC)
                re_ = ra["kiem_tra"].get(f"{a}", {}).get(p, {}).get(E)
                if not rv or not re_:
                    continue
                if p in dq_dat and re_.get("dq") is not None:
                    dq_dat[p] = (dq_dat[p] or False) or (re_["dq"] >= 0.05)
                    if re_["dq"] < 0.05 and (re_.get("b_tre") or 0) < 0:
                        am_nhan_tao.append(f"{p} α={a}")
                if rv["dat"] and not re_["dat"]:
                    hong.append(f"{p} α={a}")
                if re_.get("ty_le_es") is not None and abs(re_["ty_le_es"] - 1) > 0.10:
                    es_lech += 1
        dk_a = ra["h17a"].get("kiem_tra", {}).get(E, {}).get("dat", False) and \
            ra["h17a"].get("kiem_dinh", {}).get(E, {}).get("dat", False)
        dk_b = any(v for v in dq_dat.values() if v is not None) and not am_nhan_tao
        dk_c = not hong
        dk_d = es_lech < 3
        print("\n" + "=" * 118)
        print("PHÁN QUYẾT theo TIÊU CHÍ CHỐT TRƯỚC (docs/DUOI_ACI_TIEUCHI.md mục 7)")
        print("-" * 118)
        print(f"  (a) H17a đúng trên CẢ HAI đoạn                {'ĐẠT' if dk_a else 'TRƯỢT'}")
        print(f"  (b) DQ không bác bỏ ở ≥1 của USDJPY/USDCHF,   {'ĐẠT' if dk_b else 'TRƯỢT'}"
              f"   (USDJPY={dq_dat['USDJPY']} · USDCHF={dq_dat['USDCHF']}"
              f"{' · ÂM NHÂN TẠO: ' + ', '.join(am_nhan_tao) if am_nhan_tao else ''})")
        print(f"  (c) không làm hỏng ô V0 đang đạt              {'ĐẠT' if dk_c else 'TRƯỢT'}"
              f"   ({'không ô nào' if dk_c else ', '.join(hong)})")
        print(f"  (d) |tỷ lệ ES − 1| > 0,10 ở < 3/12 ô          {'ĐẠT' if dk_d else 'TRƯỢT'}"
              f"   ({es_lech}/12)")
        print("-" * 118)
        xong = dk_a and dk_b and dk_c and dk_d
        print(f"  → {'DƯƠNG' if xong else 'KHÔNG đủ điều kiện dương — KHÔNG đổi sản xuất'}")
        print("\n  Nhắc quy tắc 4 mục 5: nếu A2/A3 tốt hơn A1 ở đây, đó là QUAN SÁT")
        print("  được báo cáo, KHÔNG được đọc ngược thành 'đáng lẽ chọn A3'.")
        ra["phan_quyet"] = dict(a=bool(dk_a), b=bool(dk_b), c=bool(dk_c),
                                d=bool(dk_d), hong=hong, am_nhan_tao=am_nhan_tao,
                                es_lech=es_lech,
                                duong="duong" if xong else "am")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "va_duoi_aci.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/va_duoi_aci.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
