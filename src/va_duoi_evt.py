"""VA DUOI DUOI — HUONG THU NAM: LY THUYET GIA TRI CUC BIEN (EVT/POT).

Tieu chi, gia thuyet va quy tac chon DA CHOT TRUOC o `docs/DUOI_EVT.md`.
Chay sau khi van ban do da commit.

VAN DE (da do, `CHISO_DANHGIA.md` muc 5b-5e). O muc 99%, USDJPY va USDCHF
truot backtest VaR/ES; 4/6 cap dat. BON huong da thu va that bai:
  5c  V1 mo rong / V2 cuon          5d  phan vi theo che do
  9.2 CAViaR (4 bien the)           5e  cua so hop NHTW/FOMC

VI SAO EVT KHAC HAN BON HUONG DO. Ca bon deu uoc phan vi 1% bang PHAN VI THUC
NGHIEM — tuc dua vao khoang 7 quan sat cuc tri tren ~750 phien. Bay nhieu diem
khong du de dinh hinh mot cai duoi. EVT/POT khong uoc phan vi truc tiep: no
khop PHAN PHOI PARETO TONG QUAT cho TOAN BO phan vuot nguong u (khoang 75 diem
o u = phan vi 90), roi NGOAI SUY ra muc 1% bang cong thuc giai tich. Tuc cung
mot mau nhung dung gap ~10 lan so quan sat de uoc cung mot con so.

  VaR_p = u + (beta/xi) * [ (p*n/N_u)^(-xi) - 1 ]
  ES_p  = (VaR_p + beta - xi*u) / (1 - xi)                (xi < 1)

Nen tang ly thuyet: Pickands-Balkema-de Haan (phan du vuot nguong hoi tu ve
GPD khi u -> vo cuc); ap cho VaR tai chinh: McNeil & Frey (2000),
*Estimation of tail-related risk measures for heteroscedastic financial time
series*, Journal of Empirical Finance 7(3-4):271-300 — dung so do "loc GARCH
roi EVT cho phan du", chinh la cau hinh o day (loc HAR roi EVT cho z).

GIA THUYET CHOT TRUOC:
  xi > 0 (duoi day) o ca 6 cap, va EVT dua ty le vi pham o muc 99% VE GAN 1%
  hon so voi phan vi thuc nghiem, manh nhat o USDJPY/USDCHF.

Ma `gpd_duoi` tai su dung y tuong cua `gpd_tail` trong `src/experiment2.py`
(script da co tu 29/08/2026, chua tung chay tren ho so, khong co dong nao
trong KHOA_SO) — nhung viet lai theo giao thuc hien tai va co tu kiem rieng.

Chay:  python src/va_duoi_evt.py
       python src/va_duoi_evt.py --tu-kiem      (chi chay tu kiem GPD)
Ghi:   output/va_duoi_evt.json
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                           # noqa: E402
from va_duoi import nap, cham, phan_vi_cuon, MUC, BUOC, DAM, CUON  # noqa: E402

MO_KIEM_TRA = "--mo-kiem-tra" in sys.argv   # mo DUNG MOT LAN, co tuong minh
U_MAC_DINH = 0.90      # nguong POT: phan vi 90 cua ton that
U_CAO = 0.95           # bien the do nhay nguong
TOI_THIEU_VUOT = 30    # so diem vuot nguong toi thieu moi cho khop GPD


def gpd_duoi(z, a, u_q=U_MAC_DINH):
    """Phan vi muc `a` va ES tu GPD khop cho duoi DUOI cua z.

    Lam viec tren ton that duong x = -z. Tra ve (qz, ez) o THANG z (am),
    hoac None neu khong du diem vuot nguong.
    """
    x = -np.asarray(z, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 100:
        return None
    u = float(np.quantile(x, u_q))
    vuot = x[x > u] - u
    if len(vuot) < TOI_THIEU_VUOT:
        return None
    xi, _, beta = stats.genpareto.fit(vuot, floc=0)
    nu_ = len(vuot)
    if not (np.isfinite(xi) and np.isfinite(beta)) or beta <= 0:
        return None
    if abs(xi) < 1e-6:                      # gioi han xi -> 0: duoi ham so
        q = u + beta * np.log((nu_ / n) / a)
        es = q + beta
    else:
        q = u + (beta / xi) * ((a * n / nu_) ** (-xi) - 1.0)
        es = (q + beta - xi * u) / (1.0 - xi) if xi < 1 else np.nan
    if not np.isfinite(q):
        return None
    if not np.isfinite(es) or es < q:
        es = q
    return -float(q), -float(es)            # ve thang z (am)


def evt_cuon(z, g, a, kieu, u_q=U_MAC_DINH, buoc=BUOC, dam=DAM, cuon=CUON):
    """Giong `phan_vi_cuon` nhung uoc bang GPD. kieu: huan_luyen|mo_rong|cuon."""
    n = len(z)
    qz = np.full(n, np.nan)
    ez = np.full(n, np.nan)
    ok = np.isfinite(z)

    if kieu == "huan_luyen":
        r = gpd_duoi(z[(g == 0) & ok], a, u_q)
        if r is None:
            return qz, ez
        qz[:], ez[:] = r
        return qz, ez

    for t0 in range(dam, n, buoc):
        lo = 0 if kieu == "mo_rong" else max(0, t0 - cuon)
        m = np.zeros(n, bool)
        m[lo:t0] = True
        m &= ok
        if m.sum() < dam // 2:
            continue
        r = gpd_duoi(z[m], a, u_q)
        if r is None:
            continue
        t1 = min(t0 + buoc, n)
        qz[t0:t1], ez[t0:t1] = r
    return qz, ez


# ── danh sach bien the CHOT TRUOC (5 moi + 1 moc, dem 5 vao KHOA_SO)
BIEN_THE = [
    ("V0 phân vị huấn luyện (mốc sản xuất)", ("thuc_nghiem", "huan_luyen", None)),
    ("V1 phân vị mở rộng", ("thuc_nghiem", "mo_rong", None)),
    ("E1 EVT huấn luyện (đóng băng)", ("evt", "huan_luyen", U_MAC_DINH)),
    ("E2 EVT mở rộng", ("evt", "mo_rong", U_MAC_DINH)),
    ("E3 EVT cuộn 500", ("evt", "cuon", U_MAC_DINH)),
    ("E4 EVT mở rộng, ngưỡng 95%", ("evt", "mo_rong", U_CAO)),
]


def uoc(z, g, a, dac):
    ho, kieu, u_q = dac
    if ho == "thuc_nghiem":
        return phan_vi_cuon(z, g, a, kieu)
    return evt_cuon(z, g, a, kieu, u_q)


def diem_lien_tuc(ra_doan):
    """Diem CHON chot truoc — LIEN TUC, vi muc 5e da chung minh dat/khong bi mu.

    S = trung binh tren (alpha, cap) cua |ty le vi pham/alpha - 1| + |ty le ES - 1|
    Thap hon = tot hon. Khong dung dat/KHONG lam tieu chi chon.
    """
    d = {}
    for ten in [t for t, _ in BIEN_THE]:
        v = []
        for a in MUC:
            for p in B.PAIRS:
                r = ra_doan.get(f"{a}", {}).get(p, {}).get(ten)
                if not r or r["vi_pham"] is None:
                    continue
                s = abs(r["vi_pham"] / a - 1.0)
                if r["ty_le_es"] is not None:
                    s += abs(r["ty_le_es"] - 1.0)
                v.append(s)
        d[ten] = float(np.mean(v)) if v else np.nan
    return d


def tu_kiem():
    """GPD khop dung chua — mo phong voi xi, beta biet truoc."""
    print("TỰ KIỂM `gpd_duoi` trên dữ liệu mô phỏng có nghiệm giải tích")
    rng = np.random.default_rng(7)
    dat = True
    for xi_th, beta_th in ((0.20, 1.0), (0.05, 1.5), (0.30, 0.8)):
        # z = -(u + GPD) cho phan duoi; phan than lay chuan de u_q=0.9 co nghia
        n = 200_000
        u_th = 2.0
        than = rng.normal(0, 1, n)
        than = than[than < u_th][: int(n * 0.9)]
        duoi = u_th + stats.genpareto.rvs(xi_th, loc=0, scale=beta_th,
                                          size=int(n * 0.1), random_state=11)
        x = np.concatenate([than, duoi])
        z = -x
        for a in (0.01,):
            r = gpd_duoi(z, a, U_MAC_DINH)
            # nghiem giai tich tren chinh mau
            u = float(np.quantile(x, U_MAC_DINH))
            vt = x[x > u] - u
            q_th = u + (beta_th / xi_th) * ((a * len(x) / len(vt)) ** (-xi_th) - 1)
            lech = abs(-r[0] - q_th) / q_th
            ok = lech < 0.10
            dat &= ok
            print(f"  ξ={xi_th:.2f} β={beta_th:.1f} α={a}: "
                  f"GPD {-r[0]:.3f} · giải tích {q_th:.3f} · "
                  f"lệch {100*lech:.2f}%  {'ĐẠT' if ok else 'HỎNG'}")
    # don dieu: ES phai sau hon VaR
    r = gpd_duoi(-np.abs(rng.standard_t(4, 5000)), 0.01)
    ok = r[1] <= r[0]
    dat &= ok
    print(f"  ES sâu hơn VaR: {r[1]:.3f} ≤ {r[0]:.3f}  {'ĐẠT' if ok else 'HỎNG'}")
    print(f"→ TỰ KIỂM {'ĐẠT' if dat else 'HỎNG'}")
    return dat


def main():
    if "--tu-kiem" in sys.argv:
        sys.exit(0 if tu_kiem() else 1)
    if not tu_kiem():
        sys.exit("tự kiểm GPD HỎNG — dừng, không chấm số liệu")

    D = nap()
    print("\n" + "=" * 112)
    print("VÁ ĐUÔI — HƯỚNG THỨ NĂM: EVT/POT (McNeil & Frey 2000)")
    print("tiêu chí chốt trước: docs/DUOI_EVT.md")
    print("=" * 112)

    # hinh dang duoi uoc tren HUAN LUYEN — kiem dau gia thuyet (xi > 0)
    print("\nHÌNH DẠNG ĐUÔI ước trên HUẤN LUYỆN (giả thuyết chốt trước: ξ > 0)")
    print(f"  {'cặp':<9}{'n vượt':>8}{'u':>8}{'ξ':>9}{'β':>8}")
    hinh_dang = {}
    for p in B.PAIRS:
        z, g = D[p]["z"], D[p]["g"]
        zt = z[(g == 0) & np.isfinite(z)]
        x = -zt
        u = float(np.quantile(x, U_MAC_DINH))
        vt = x[x > u] - u
        xi, _, be = stats.genpareto.fit(vt, floc=0)
        hinh_dang[p] = dict(n_vuot=int(len(vt)), u=float(u), xi=float(xi),
                            beta=float(be))
        print(f"  {p:<9}{len(vt):>8}{u:>8.3f}{xi:>9.4f}{be:>8.4f}")
    duong = sum(v["xi"] > 0 for v in hinh_dang.values())
    print(f"  → ξ > 0 ở {duong}/6 cặp")

    ra = {"u": U_MAC_DINH, "hinh_dang": hinh_dang, "kiem_dinh": {}, "kiem_tra": {}}

    for nhan_doan, gid in (("kiem_dinh", 1), ("kiem_tra", 2)):
        if nhan_doan == "kiem_tra" and not MO_KIEM_TRA:
            continue
        for a in MUC:
            print(f"\n── mức {1-a:.0%} (α = {a}) · ĐOẠN {nhan_doan.upper()} " + "─" * 52)
            print(f"{'cặp':9}{'phương án':<32}{'vi phạm':>9}{'Kupiec':>9}"
                  f"{'Chris':>8}{'DQ':>8}{'tỷ lệ ES':>10}")
            for p in B.PAIRS:
                z, sig, g = D[p]["z"], D[p]["sig"], D[p]["g"]
                dau = True
                for ten, dac in BIEN_THE:
                    qz, ez = uoc(z, g, a, dac)
                    r = cham(z, sig, qz, ez, g == gid, a)
                    if r is None:
                        continue
                    ra[nhan_doan].setdefault(f"{a}", {}).setdefault(p, {})[ten] = r
                    f = lambda v: "  —  " if v is None else f"{v:.3f}"
                    print(f"{p if dau else '':9}{ten:<32}{100*r['vi_pham']:>8.2f}%"
                          f"{f(r['kupiec']):>9}{f(r['chris']):>8}{f(r['dq']):>8}"
                          f"{f(r['ty_le_es']):>10}")
                    dau = False
                print()

    diem = diem_lien_tuc(ra["kiem_dinh"])
    ra["diem_kiem_dinh"] = diem
    print("=" * 112)
    print("ĐIỂM CHỌN LIÊN TỤC trên KIỂM ĐỊNH (chốt trước — thấp hơn là tốt hơn)")
    print(f"  {'phương án':<32}{'S':>10}{'số ô đạt':>11}")
    for ten, s in sorted(diem.items(), key=lambda kv: kv[1]):
        dat = sum(1 for a in MUC for p in B.PAIRS
                  if (ra["kiem_dinh"].get(f"{a}", {}).get(p, {}).get(ten) or {}).get("dat"))
        print(f"  {ten:<32}{s:>10.4f}{dat:>8}/12")
    moc = diem["V0 phân vị huấn luyện (mốc sản xuất)"]
    tot = min(diem, key=lambda k: diem[k])
    print(f"\n  mốc V0: S = {moc:.4f}")
    print(f"  tốt nhất: {tot} · S = {diem[tot]:.4f}")
    thang = tot != "V0 phân vị huấn luyện (mốc sản xuất)"
    ra["chon_kiem_dinh"] = tot
    ra["evt_thang_kiem_dinh"] = bool(thang and tot.startswith("E"))
    print(f"  → EVT {'THẮNG' if ra['evt_thang_kiem_dinh'] else 'KHÔNG thắng'} "
          f"mốc trên kiểm định")
    if not ra["evt_thang_kiem_dinh"]:
        print("  Theo quy tắc chốt trước: KHÔNG mở đoạn kiểm tra.")

    if MO_KIEM_TRA and ra["kiem_tra"]:
        E = tot
        dat_e = dat_v0 = 0
        hong = []
        for a in MUC:
            for p in B.PAIRS:
                rv = ra["kiem_tra"].get(f"{a}", {}).get(p, {}).get(
                    "V0 phân vị huấn luyện (mốc sản xuất)")
                re_ = ra["kiem_tra"].get(f"{a}", {}).get(p, {}).get(E)
                if not rv or not re_:
                    continue
                dat_v0 += int(rv["dat"]); dat_e += int(re_["dat"])
                if rv["dat"] and not re_["dat"]:
                    hong.append(f"{p} α={a}")
        dk1 = dat_e >= 5
        dk2 = not hong
        dk3 = duong >= 4
        print("
" + "=" * 112)
        print("PHÁN QUYẾT theo TIÊU CHÍ CHỐT TRƯỚC (docs/DUOI_EVT.md mục 6)")
        print("-" * 112)
        print(f"  mốc V0 trên kiểm tra: {dat_v0}/12 ô đạt · {E}: {dat_e}/12")
        print(f"  ĐK1 ô đạt ≥ 5/12                  {'ĐẠT' if dk1 else 'TRƯỢT'}   ({dat_e}/12)")
        print(f"  ĐK2 không làm hỏng ô V0 đang đạt  {'ĐẠT' if dk2 else 'TRƯỢT'}   "
              f"({'không ô nào' if dk2 else ', '.join(hong)})")
        print(f"  ĐK3 ξ > 0 ở ≥ 4/6 cặp             {'ĐẠT' if dk3 else 'TRƯỢT'}   ({duong}/6)")
        print("-" * 112)
        xong = dk1 and dk2 and dk3
        print(f"  → {'DƯƠNG' if xong else 'KHÔNG đủ điều kiện dương — KHÔNG đổi sản xuất'}")
        ra["phan_quyet"] = dict(dk1=bool(dk1), dk2=bool(dk2), dk3=bool(dk3),
                                dat_v0=dat_v0, dat_e=dat_e, hong=hong,
                                duong="duong" if xong else "am")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "va_duoi_evt.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("\n→ output/va_duoi_evt.json")


if __name__ == "__main__":
    main()
