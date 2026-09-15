"""VA DUOI — HUONG THU BAY: GAS/DCS (Beta-t-EGARCH) cho quy mo dong cua z.

Gia thuyet, cau hinh va tieu chi phu dinh DA CHOT TRUOC o `docs/GAS_TIEUCHI.md`.

Beta-t-EGARCH (Harvey & Chakravarty 2008; Harvey 2013): z_t ~ t(0, s_t, nu),
dong hoc cua ln(s_t) cap nhat theo DIEM SO (score) cua log-likelihood:

    ln(s_{t+1}) = omega + alpha*u_t + beta*ln(s_t)
    u_t = (nu+1)*z_t^2 / (nu*s_t^2 + z_t^2) - 1

u_t BI CHAN trong [-1, nu] — ben vung voi ngoai lai, khac han cap nhat kieu
GARCH bac hai (z_t^2 khong chan) va khac CAViaR (tu hoi quy TUY Y tren muc
quantile, da that bai — KHOA_SO dong 15).

Chay:  python src/va_duoi_gas.py
       python src/va_duoi_gas.py --tu-kiem       (chi tu kiem GAS MLE)
       python src/va_duoi_gas.py --mo-kiem-tra    (mo dung mot lan)
Ghi:   output/va_duoi_gas.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                           # noqa: E402
import volfc2 as V2                                         # noqa: E402
from metrics import kupiec, christoffersen_ind, dq_test     # noqa: E402
from split import doan                                      # noqa: E402
from volfc import merge_thin_days                           # noqa: E402
from va_duoi import cham                                    # noqa: E402

MUC = (0.05, 0.01)
BUOC = 21
DAM = 750
CUON = 500
EPS = 1e-12
MO_KIEM_TRA = "--mo-kiem-tra" in sys.argv
HET_PHAT_TRIEN = "2025-12-31"     # KHOA_SO muc 2 — cat truoc khi cham (RUIRO_ML A4)


# ───────────────────────────────────────────── GAS Beta-t-EGARCH lõi
def diem_so(z, s, nu):
    """u_t = (nu+1)*z^2 / (nu*s^2 + z^2) - 1, bi chan trong [-1, nu]."""
    return (nu + 1.0) * z ** 2 / (nu * s ** 2 + z ** 2 + EPS) - 1.0


def de_quy_s(z, omega, alpha, beta, nu, s0):
    """Tra ve mang s[0..n-1]: s[t] la QUY MO DA BIET TAI THOI DIEM t (dung de
    du bao z_t), cap nhat SAU KHI da thay z_t (dung cho t+1)."""
    n = len(z)
    ls = np.empty(n)
    ls[0] = np.log(max(s0, EPS))
    for t in range(n - 1):
        u = diem_so(z[t], np.exp(ls[t]), nu)
        ls[t + 1] = omega + alpha * u + beta * ls[t]
        ls[t + 1] = np.clip(ls[t + 1], -10, 10)
    return np.exp(ls)


def loglik_am(theta, z, nu, s0):
    omega, alpha, beta = theta
    if not (0 <= alpha <= 2 and 0 <= beta <= 0.999):
        return 1e10
    s = de_quy_s(z, omega, alpha, beta, nu, s0)
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        return 1e10
    ll = stats.t.logpdf(z / s, nu) - np.log(s)
    if not np.all(np.isfinite(ll)):
        return 1e10
    return -float(np.sum(ll))


def khop_gas(z, nu):
    """MLE (omega, alpha, beta) tren mot doan z. Tra ve (theta, s_cuoi)."""
    z = z[np.isfinite(z)]
    if len(z) < 60:
        return None
    s0 = float(np.std(z))
    # khoi tao: beta gan 1 (dai), omega ~ (1-beta)*log(s0), alpha nho
    x0 = np.array([0.02 * (1 - 0.97), 0.05, 0.97])
    x0[0] = (1 - x0[2]) * np.log(max(s0, EPS))
    res = minimize(loglik_am, x0, args=(z, nu, s0), method="Nelder-Mead",
                   options=dict(maxiter=2000, xatol=1e-6, fatol=1e-6))
    if not res.success and res.fun >= 1e9:
        return None
    theta = res.x
    s_path = de_quy_s(z, *theta, nu, s0)
    return theta, float(s_path[-1])


def uoc_nu(z):
    nu, _, _ = stats.t.fit(z[np.isfinite(z)], floc=0)
    return float(np.clip(nu, 2.5, 40))


# ───────────────────────────────────────────── ba bien the CHOT TRUOC
def gas_du_bao(z, g, nu, kieu, buoc=BUOC, dam=DAM, cuon=CUON):
    """Tra ve mang s[t] = quy mo du bao CHO PHIEN t (biet tai t-1), dung
    NHAN QUA: chi dung z[<t0] de khop, roi de quy TIEN VE PHIA TRUOC toi t0
    voi tham so DA KHOP (khong nhin z sau t0 khi de quy)."""
    n = len(z)
    s_ra = np.full(n, np.nan)
    if kieu == "dong_bang":
        tr = g == 0
        r = khop_gas(z[tr], nu)
        if r is None:
            return s_ra
        theta, _ = r
        omega, alpha, beta = theta
        s0 = float(np.std(z[tr][np.isfinite(z[tr])]))
        ls = np.log(max(s0, EPS))
        for t in range(n):
            s_ra[t] = np.exp(ls)
            if np.isfinite(z[t]):
                u = diem_so(z[t], np.exp(ls), nu)
                ls = np.clip(omega + alpha * u + beta * ls, -10, 10)
        return s_ra

    for t0 in range(dam, n, buoc):
        lo = 0 if kieu == "mo_rong" else max(0, t0 - cuon)
        zt = z[lo:t0]
        r = khop_gas(zt, nu)
        if r is None:
            continue
        theta, s_last = r
        omega, alpha, beta = theta
        ls = np.log(max(s_last, EPS))
        t1 = min(t0 + buoc, n)
        for t in range(t0, t1):
            s_ra[t] = np.exp(ls)
            if np.isfinite(z[t]):
                u = diem_so(z[t], np.exp(ls), nu)
                ls = np.clip(omega + alpha * u + beta * ls, -10, 10)
    return s_ra


def qz_ez_tu_s(s, nu, a):
    """Quy doi quy mo GAS s[t] -> (qz, ez) o thang z, dung dinh dang cua
    va_duoi.cham() (nhan them sig san xuat ben trong cham)."""
    q_chuan = float(stats.t.ppf(a, nu))
    f_q = float(stats.t.pdf(q_chuan, nu))
    es_chuan = -f_q / a * (nu + q_chuan ** 2) / (nu - 1) if nu > 1 else q_chuan
    return s * q_chuan, s * es_chuan


BIEN_THE = ["GAS mở rộng", "GAS đóng băng", "GAS cuộn 500"]
_KIEU = {"GAS mở rộng": "mo_rong", "GAS đóng băng": "dong_bang",
        "GAS cuộn 500": "cuon"}


def diem_lien_tuc_dq(ra_doan):
    """S_DQ = S (nhu DUOI_EVT) + trung binh 1{DQ p < 0.05}."""
    d = {}
    for ten in ["V0 phân vị huấn luyện (mốc)"] + BIEN_THE:
        v, dq_phat = [], []
        for a in MUC:
            for p in B.PAIRS:
                r = ra_doan.get(f"{a}", {}).get(p, {}).get(ten)
                if not r or r["vi_pham"] is None:
                    continue
                s = abs(r["vi_pham"] / a - 1.0)
                if r["ty_le_es"] is not None:
                    s += abs(r["ty_le_es"] - 1.0)
                v.append(s)
                if r.get("dq") is not None:
                    dq_phat.append(1.0 if r["dq"] < 0.05 else 0.0)
        s_co_ban = float(np.mean(v)) if v else np.nan
        s_dq = float(np.mean(dq_phat)) if dq_phat else np.nan
        d[ten] = s_co_ban + s_dq
    return d


def tu_kiem():
    print("TỰ KIỂM `de_quy_s`/`khop_gas` trên dữ liệu mô phỏng có tham số biết trước")
    rng = np.random.default_rng(11)
    dat = True
    for om_th, al_th, be_th, nu_th in ((0.05, 0.08, 0.93, 6.0),
                                       (0.10, 0.12, 0.88, 8.0)):
        n = 6000
        s0 = float(np.exp(om_th / (1 - be_th)))
        ls = np.log(s0)
        z = np.empty(n)
        for t in range(n):
            s = np.exp(ls)
            z[t] = stats.t.rvs(nu_th, random_state=rng) * s
            u = diem_so(z[t], s, nu_th)
            ls = om_th + al_th * u + be_th * ls
        r = khop_gas(z, nu_th)
        if r is None:
            print("  KHÔNG khớp được — HỎNG"); dat = False; continue
        theta, _ = r
        lech = np.abs(theta - np.array([om_th, al_th, be_th]))
        ok = lech[1] < 0.15 and lech[2] < 0.10          # alpha, beta la dong hoc chinh
        dat &= ok
        print(f"  (ω,α,β)_thật=({om_th},{al_th},{be_th}) "
              f"khớp=({theta[0]:.3f},{theta[1]:.3f},{theta[2]:.3f})  "
              f"{'ĐẠT' if ok else 'HỎNG'}")
    # u_t bi chan
    u_lon = diem_so(1e6, 1.0, 5.0)
    ok = abs(u_lon - 5.0) < 1e-6
    dat &= ok
    print(f"  u_t bị chặn khi z cực đoan (→ν=5): u={u_lon:.4f}  "
          f"{'ĐẠT' if ok else 'HỎNG'}")
    print(f"→ TỰ KIỂM {'ĐẠT' if dat else 'HỎNG'}")
    return dat


def nap():
    from api.main import noi_chuoi
    ra = {}
    for p in B.PAIRS:
        m = merge_thin_days(noi_chuoi(p))
        m = m[pd.DatetimeIndex(m.Date) <= pd.Timestamp(HET_PHAT_TRIEN)].reset_index(drop=True)
        sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, p), 0.0))
        c = m.close.values
        z = np.full(len(m), np.nan)
        z[1:] = np.log(c[1:] / np.maximum(c[:-1], EPS)) / np.maximum(sig[1:], EPS)
        ra[p] = dict(z=z, sig=sig, g=doan(m.Date.values))
    return ra


def main():
    if "--tu-kiem" in sys.argv:
        sys.exit(0 if tu_kiem() else 1)
    if not tu_kiem():
        sys.exit("tự kiểm GAS HỎNG — dừng, không chấm số liệu thật")

    t0 = time.time()
    D = nap()
    print("\n" + "=" * 112)
    print("VÁ ĐUÔI — HƯỚNG THỨ BẢY: GAS/DCS (Beta-t-EGARCH)")
    print("tiêu chí chốt trước: docs/GAS_TIEUCHI.md")
    print("=" * 112)

    ra = {"kiem_dinh": {}, "kiem_tra": {}}
    s_theo_cap = {}
    for p in B.PAIRS:
        z, sig, g = D[p]["z"], D[p]["sig"], D[p]["g"]
        nu = uoc_nu(z[(g == 0) & np.isfinite(z)])
        s_theo_cap[p] = {"nu": nu}
        print(f"\n{p}: ν ước trên huấn luyện = {nu:.2f}")
        for ten in BIEN_THE:
            t1 = time.time()
            s = gas_du_bao(z, g, nu, _KIEU[ten])
            s_theo_cap[p][ten] = s
            print(f"  {ten:<16}khớp xong ({time.time()-t1:.0f}s)")

    doan_chay = [("kiem_dinh", 1)]
    if MO_KIEM_TRA:
        doan_chay.append(("kiem_tra", 2))

    for nhan_doan, gid in doan_chay:
        print(f"\n── ĐOẠN {nhan_doan.upper()} " + "─" * 60)
        for a in MUC:
            print(f"\n  mức {1-a:.0%} (α={a})")
            print(f"  {'cặp':9}{'phương án':<18}{'vi phạm':>9}{'Kupiec':>9}"
                  f"{'Chris':>8}{'DQ':>8}{'tỷ lệ ES':>10}")
            for p in B.PAIRS:
                z, sig, g = D[p]["z"], D[p]["sig"], D[p]["g"]
                nu = s_theo_cap[p]["nu"]
                dau = True
                # moc V0
                zt = z[(g == 0) & np.isfinite(z)]
                q0 = float(np.quantile(zt, a))
                e0 = float(np.mean(zt[zt <= q0])) if (zt <= q0).any() else q0
                qz0 = np.full(len(z), q0); ez0 = np.full(len(z), e0)
                r0 = cham(z, sig, qz0, ez0, g == gid, a)
                if r0:
                    ra[nhan_doan].setdefault(f"{a}", {}).setdefault(p, {})[
                        "V0 phân vị huấn luyện (mốc)"] = r0
                    f = lambda v: "  —  " if v is None else f"{v:.3f}"
                    print(f"  {p if dau else '':9}{'V0 mốc':<18}"
                          f"{100*r0['vi_pham']:>8.2f}%{f(r0['kupiec']):>9}"
                          f"{f(r0['chris']):>8}{f(r0['dq']):>8}{f(r0['ty_le_es']):>10}")
                    dau = False
                for ten in BIEN_THE:
                    s = s_theo_cap[p][ten]
                    qz, ez = qz_ez_tu_s(s, nu, a)
                    r = cham(z, sig, qz, ez, g == gid, a)
                    if not r:
                        continue
                    ra[nhan_doan].setdefault(f"{a}", {}).setdefault(p, {})[ten] = r
                    f = lambda v: "  —  " if v is None else f"{v:.3f}"
                    print(f"  {'':9}{ten:<18}{100*r['vi_pham']:>8.2f}%"
                          f"{f(r['kupiec']):>9}{f(r['chris']):>8}{f(r['dq']):>8}"
                          f"{f(r['ty_le_es']):>10}")
                print()

    diem = diem_lien_tuc_dq(ra["kiem_dinh"])
    ra["diem_kiem_dinh"] = diem
    print("=" * 112)
    print("ĐIỂM S_DQ trên KIỂM ĐỊNH (chốt trước — thấp hơn tốt hơn)")
    for ten, s in sorted(diem.items(), key=lambda kv: kv[1]):
        print(f"  {ten:<32}S_DQ = {s:.4f}")
    tot = min((t for t in BIEN_THE), key=lambda t: diem[t])
    moc_s = diem["V0 phân vị huấn luyện (mốc)"]
    thang_kd = diem[tot] < moc_s
    print(f"\n  mốc V0: S_DQ = {moc_s:.4f} · tốt nhất GAS: {tot} = {diem[tot]:.4f}")
    print(f"  → GAS {'THẮNG' if thang_kd else 'KHÔNG thắng'} mốc trên kiểm định")
    ra["chon_kiem_dinh"] = tot
    ra["gas_thang_kiem_dinh"] = bool(thang_kd)

    if MO_KIEM_TRA and thang_kd and ra["kiem_tra"]:
        E = tot
        dq_dat = {"USDJPY": None, "USDCHF": None}
        hong = []
        for a in MUC:
            for p in B.PAIRS:
                rv = ra["kiem_tra"].get(f"{a}", {}).get(p, {}).get(
                    "V0 phân vị huấn luyện (mốc)")
                re_ = ra["kiem_tra"].get(f"{a}", {}).get(p, {}).get(E)
                if not rv or not re_:
                    continue
                if p in dq_dat and re_.get("dq") is not None:
                    khong_bac_bo = re_["dq"] >= 0.05
                    dq_dat[p] = (dq_dat[p] or False) or khong_bac_bo
                if rv["dat"] and not re_["dat"]:
                    hong.append(f"{p} α={a}")
        dk1 = any(v for v in dq_dat.values() if v is not None)
        dk2 = not hong
        print("\n" + "=" * 112)
        print("PHÁN QUYẾT theo TIÊU CHÍ CHỐT TRƯỚC (docs/GAS_TIEUCHI.md mục 7)")
        print("-" * 112)
        print(f"  DQ USDJPY không bác bỏ ở ít nhất 1 mức: {dq_dat['USDJPY']}")
        print(f"  DQ USDCHF không bác bỏ ở ít nhất 1 mức: {dq_dat['USDCHF']}")
        print(f"  ĐK1 DQ không bác bỏ ở ≥1 trong hai cặp     "
              f"{'ĐẠT' if dk1 else 'TRƯỢT'}")
        print(f"  ĐK2 không làm hỏng cặp V0 đang đạt         "
              f"{'ĐẠT' if dk2 else 'TRƯỢT'}   "
              f"({'không ô nào' if dk2 else ', '.join(hong)})")
        print("-" * 112)
        xong = dk1 and dk2
        print(f"  → {'DƯƠNG' if xong else 'KHÔNG đủ điều kiện dương — KHÔNG đổi sản xuất'}")
        ra["phan_quyet"] = dict(dk1=bool(dk1), dk2=bool(dk2), hong=hong,
                                duong="duong" if xong else "am")
    elif MO_KIEM_TRA:
        print("\nGAS KHÔNG thắng trên kiểm định — theo tiêu chí mục 7, KHÔNG mở kiểm tra.")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "va_duoi_gas.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/va_duoi_gas.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
