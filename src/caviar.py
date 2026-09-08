"""CAViaR — PHAN VI TU HOI QUY CHO DUOI, thay cho phan vi tinh cua va_duoi.py.

VAN DE. Ba phuong an o `src/va_duoi.py` (V0 huan luyen, V1 mo rong, V2 cuon)
deu la PHAN VI KHONG DIEU KIEN cua z, chi khac nhau o cua so uoc. Ca ba deu
truot o USDCHF tren KIEM DINH:

    a = 0,01   USDCHF  V1:  vi pham 2,19% (ky vong 1%)  Kupiec 0,015
                            DQ 0,0031   ty le ES 1,325
    a = 0,05   USDCHF  V1:  Christoffersen 0,019  DQ 0,064

DQ va Christoffersen la hai phep kiem DOC LAP CUA VI PHAM. Chung hong co nghia
la vi pham hom nay DU BAO DUOC tu vi pham hom qua — vi pham don cum lai. Mot
phan vi khong dieu kien khong the sua duoc dieu do du co uoc lai bao nhieu lan:
no chi biet mot con so cho ca cua so.

CAViaR (Engle & Manganelli 2004, "CAViaR: Conditional Autoregressive Value at
Risk by Regression Quantiles") mo hinh THANG cai dong luc do — phan vi hom nay
phu thuoc phan vi hom qua va cu soc hom qua:

    SAV  q_t = b0 + b1 q_{t-1} + b2 |z_{t-1}|          doi xung
    AS   q_t = b0 + b1 q_{t-1} + b2 z+_{t-1} + b3 z-_{t-1}   bat don bay
    IG   q_t = -sqrt(b0 + b1 q^2_{t-1} + b2 z^2_{t-1})  GARCH gian tiep

Uoc bang cuc tieu ton that pinball — khong can gia dinh hinh dang phan phoi.

CHAY TREN z CHU KHONG TREN LOI SUAT THO. Day la lua chon thiet ke, khong phai
mac dinh: tang sigma^ da giai thich phan lon bien dong, nen cho CAViaR chay
tren loi suat tho la bat no lam lai viec da lam roi. Chay tren z la hoi dung
cau hoi ma DQ dang neu — SAU KHI da chia sigma^, phan du con dong luc duoi
nao khong. Neu co thi CAViaR bat duoc; neu khong thi b1 = b2 = 0 va no tu quy
ve hang so, tuc quy ve chinh V1.

GIAO THUC DOAN — DIEU KIEN BAT BUOC. Chon tren KIEM DINH. Doan KIEM TRA KHONG
mo o day. `docs/CHISO_DANHGIA.md` muc 5c ghi tang VaR da mo kiem tra HAI lan
(chan doan 5b, roi va_duoi). Lan thu ba chi duoc mo sau khi da chot cau hinh
va ghi bien ban — va chi mot lan.

Chay:  python src/caviar.py
Ghi:   output/caviar.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                           # noqa: E402
import va_duoi as VD                                        # noqa: E402
from metrics import kupiec, christoffersen_ind, dq_test     # noqa: E402

MUC = (0.05, 0.01)
BUOC = 63           # khop lai moi ~1 quy (CAViaR dat hon phan vi rong ~50 lan)
DAM = 750
N_KHOI_DAU = 12     # so diem khoi dau ngau nhien moi lan uoc
EPS = 1e-12
SEED = 11


# ── ba dang dong luc ────────────────────────────────────────────────────
# Ca ba deu la DE QUY TUYEN TINH BAC MOT (IG thi tuyen tinh o q^2), nen chay
# duoc bang lfilter trong C thay vi vong lap Python. Voi ~50 lan khop x 4 diem
# khoi dau x hang tram lan goi ham muc tieu, do la khac biet giua vai phut va
# vai gio. Tu kiem `_tu_kiem_dequy()` doi chieu voi vong lap tuong minh.
def _chay(rho, v, s0):
    """s[0] = s0 ; s[t] = rho s[t-1] + v[t]  voi t >= 1."""
    from scipy.signal import lfilter
    s = np.empty(len(v))
    s[0] = s0
    if len(v) > 1:
        s[1:] = lfilter([1.0], [1.0, -rho], v[1:], zi=[rho * s0])[0]
    return s


def _duong_sav(b, z, q0):
    v = np.empty(len(z))
    v[0] = 0.0
    v[1:] = b[0] + b[2] * np.abs(z[:-1])
    return _chay(b[1], v, q0)


def _duong_as(b, z, q0):
    v = np.empty(len(z))
    v[0] = 0.0
    v[1:] = (b[0] + b[2] * np.maximum(z[:-1], 0.0)
             + b[3] * np.maximum(-z[:-1], 0.0))
    return _chay(b[1], v, q0)


def _duong_ig(b, z, q0):
    v = np.empty(len(z))
    v[0] = 0.0
    v[1:] = abs(b[0]) + abs(b[2]) * z[:-1] ** 2
    s = _chay(abs(b[1]), v, q0 * q0)
    return -np.sqrt(np.maximum(s, EPS))


def _tu_kiem_dequy(seed=0):
    """lfilter phai ra DUNG bang vong lap tuong minh — neu khong thi moi con so
    ben duoi la con so cua mot mo hinh khac."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(400)
    for dang, b, q0 in (("SAV", [-0.05, 0.92, -0.08], -1.7),
                        ("AS", [-0.04, 0.90, -0.06, 0.11], -1.7),
                        ("IG", [0.12, 0.85, 0.09], -1.7)):
        b = np.array(b, float)
        q = np.empty(len(z))
        qt = q0
        for t in range(len(z)):                      # vong lap tuong minh
            q[t] = qt
            if dang == "SAV":
                qt = b[0] + b[1] * qt + b[2] * abs(z[t])
            elif dang == "AS":
                qt = (b[0] + b[1] * qt + b[2] * max(z[t], 0.0)
                      + b[3] * max(-z[t], 0.0))
            else:
                qt = -np.sqrt(max(abs(b[0]) + abs(b[1]) * qt * qt
                                  + abs(b[2]) * z[t] * z[t], EPS))
        lech = float(np.max(np.abs(q - DANG[dang][0](b, z, q0))))
        assert lech < 1e-9, f"{dang}: lệch {lech:.2e}"
    return True


DANG = {"SAV": (_duong_sav, 3), "AS": (_duong_as, 4), "IG": (_duong_ig, 3)}


def pinball(q, z, a):
    """Ton that kiem tra (tick loss) — ham muc tieu chinh danh cua phan vi."""
    e = z - q
    return float(np.mean(e * (a - (e < 0).astype(float))))


def uoc(z, a, dang, seed=SEED, n_dau=N_KHOI_DAU):
    """Uoc CAViaR tren mang z (khong NaN). Tra ve (beta, q_cuoi, ton_that).

    Cach uoc theo dung Engle-Manganelli: RAI nhieu diem khoi dau ngau nhien,
    lay vai diem tot nhat roi moi toi uu. Ham muc tieu pinball khong loi va co
    nhieu cuc tieu dia phuong; khoi dong tu mot diem duy nhat la cach chac chan
    nhat de ra ket qua khong tai lap duoc."""
    ham, k = DANG[dang]
    q0 = float(np.quantile(z[:min(300, len(z))], a))
    rng = np.random.default_rng(seed)
    if dang == "IG":
        thu = np.abs(rng.normal(0, 1, (n_dau * 8, k))) * [q0 * q0, 1, 1]
        thu[0] = [q0 * q0 * 0.1, 0.9, 0.05]
    else:
        thu = rng.normal(0, 1, (n_dau * 8, k)) * 0.3
        thu[:, 1] = rng.uniform(0.5, 0.99, len(thu))       # b1 gan 1: dai
        thu[0] = [q0 * 0.05, 0.9, -0.05] + ([0.05] if k == 4 else [])

    def f(b):
        # CHAN PHAN KY. b1 la he so dai; |b1| >= 1 thi de quy no theo ham mu.
        # Tren cua so khop ngan no co the chua kip lo ra, nhung khi chay tiep
        # ve phia truoc thi no nổ — do la nguon cua pinball 52.205 va ty le ES
        # -107.808 o ban chay dau tien. Chan ngay trong ham muc tieu, khong de
        # thuat toan toi uu di vao vung do.
        if not np.isfinite(b).all() or abs(b[1]) >= 0.999:
            return np.inf
        q = ham(b, z, q0)
        return np.inf if not np.isfinite(q).all() else pinball(q, z, a)
    diem = sorted(((f(b), b) for b in thu), key=lambda x: x[0])[:n_dau // 3]
    tot = (np.inf, None)
    for v0, b0 in diem:
        try:
            r = minimize(f, b0, method="Nelder-Mead",
                         options=dict(maxiter=1200, xatol=1e-6, fatol=1e-9))
            if r.fun < tot[0]:
                tot = (float(r.fun), r.x)
        except Exception:
            continue
    if tot[1] is None:
        return None, q0, np.inf
    q = ham(tot[1], z, q0)
    # mot buoc nua de lay q cua phien KE TIEP
    return tot[1], float(q[-1]), tot[0]


def duong_nhan_qua(z, a, dang, buoc=BUOC, dam=DAM):
    """q_t nhan qua: khop lai moi `buoc` phien tren du lieu DEN t0, roi chay
    de quy tien ve phia truoc. Cung giao thuc voi V1 cua va_duoi.py.

    Tra ve (qz, ez). ez la ky vong duoi q, uoc bang TY SO k = E[z|z<=q]/q tren
    cua so khop roi nhan lai — giu phep so sanh tap trung vao DONG LUC cua VaR,
    la thu ma DQ dang bat loi."""
    ham, _ = DANG[dang]
    n = len(z)
    qz = np.full(n, np.nan)
    ez = np.full(n, np.nan)
    ok = np.isfinite(z)
    for t0 in range(dam, n, buoc):
        zz = z[:t0][ok[:t0]]
        if len(zz) < dam // 2:
            continue
        b, _, _ = uoc(zz, a, dang, seed=SEED + t0)
        if b is None:
            continue
        # ty so ES uoc tren chinh cua so khop
        qin = ham(b, zz, float(np.quantile(zz[:min(300, len(zz))], a)))
        d = zz <= qin
        k_es = float(np.mean(zz[d]) / np.mean(qin[d])) if d.sum() >= 10 else 1.15

        t1 = min(t0 + buoc, n)
        # Chay de quy tren CHUOI DA NEN (chi cac phien co z huu han) roi rai
        # nguoc ve dung vi tri. Ban dau cho z thieu = 0, nhung nhu vay la bia
        # ra mot phien YEN TINH o moi cho thieu du lieu — de quy doc no nhu
        # tin hieu that va keo phan vi ve phia 0. Nen o day chi don gian la
        # BO QUA phien thieu, dung het cach ma buoc khop da nhin thay.
        vt = np.flatnonzero(ok[:t1])
        q = ham(b, z[vt], float(np.quantile(zz[:min(300, len(zz))], a)))
        gan = vt >= t0
        qz[vt[gan]] = q[gan]
        ez[vt[gan]] = q[gan] * k_es
    return qz, ez


def main():
    t0 = time.time()
    chi_cap = [c for c in sys.argv[1:] if c in B.PAIRS] or list(B.PAIRS)
    print("=" * 104)
    print("CAViaR — phân vị tự hồi quy cho đuôi, đo trên KIỂM ĐỊNH")
    print("=" * 104, flush=True)
    print("đoạn KIỂM TRA không mở ở đây (docs/CHISO_DANHGIA.md mục 5c)",
          flush=True)
    _tu_kiem_dequy()
    print("tự kiểm đệ quy (lfilter ≡ vòng lặp tường minh): ĐẠT", flush=True)

    D = VD.nap()
    ket = {"buoc": BUOC, "dam": DAM, "muc": list(MUC), "cap": chi_cap,
           "kiem_dinh": {}}

    for a in MUC:
        print(f"\n{'─'*104}\nα = {a}", flush=True)
        print(f"  {'cặp':9}{'biến thể':<14}{'n':>6}{'vi phạm':>9}"
              f"{'Kupiec':>9}{'Chris':>8}{'DQ':>8}{'ES':>8}{'pinball':>10}"
              f"{'đạt':>6}", flush=True)
        ket["kiem_dinh"][str(a)] = {}
        for p in chi_cap:
            z, sig, g = D[p]["z"], D[p]["sig"], D[p]["g"]
            kd = g == 1
            hang = {}
            # mốc: V1 mở rộng, đúng cái đang chạy
            qz, ez = VD.phan_vi_cuon(z, g, a, "mo_rong")
            r = VD.cham(z, sig, qz, ez, kd, a)
            if r:
                m = kd & np.isfinite(z) & np.isfinite(qz)
                r["pinball"] = round(pinball(qz[m], z[m], a), 5)
                hang["V1 mở rộng"] = r
            for dang in ("SAV", "AS", "IG"):
                qz, ez = duong_nhan_qua(z, a, dang)
                r = VD.cham(z, sig, qz, ez, kd, a)
                if r:
                    m = kd & np.isfinite(z) & np.isfinite(qz)
                    r["pinball"] = round(pinball(qz[m], z[m], a), 5)
                    # CHOT PHAN KY — ban chay dau tien co pinball 52.205 vi de
                    # quy no. Neu no tai dien thi phai HIEN RA chu khong duoc
                    # lan vao bang nhu mot con so binh thuong.
                    r["phan_ky"] = bool(r["pinball"] > 10 * hang["V1 mở rộng"]["pinball"]
                                        if "V1 mở rộng" in hang else False)
                    hang[f"CAViaR {dang}"] = r
            for bien, r in hang.items():
                print(f"  {p if bien.startswith('V1') else '':9}{bien:<14}"
                      f"{r['n']:>6}{r['vi_pham']:>9.4f}{r['kupiec']:>9.4f}"
                      f"{r['chris']:>8.4f}{r['dq']:>8.4f}{r['ty_le_es']:>8.3f}"
                      f"{r['pinball']:>10.5f}{'✓' if r['dat'] else '✗':>6}"
                      f"{'  ⚠ PHÂN KỲ' if r.get('phan_ky') else ''}",
                      flush=True)
            print(flush=True)
            ket["kiem_dinh"][str(a)][p] = hang

    # ── tổng kết ────────────────────────────────────────────────────────
    print("=" * 104)
    print("TỔNG KẾT — số cặp ĐẠT cả ba phép kiểm (Kupiec, Christoffersen, DQ)")
    print(f"  {'biến thể':<14}{'α=0,05':>9}{'α=0,01':>9}{'pinball TB':>13}",
          flush=True)
    biens = ["V1 mở rộng", "CAViaR SAV", "CAViaR AS", "CAViaR IG"]
    ket["tong_ket"] = {}
    for bien in biens:
        dat, pb = [], []
        for a in MUC:
            d = sum(1 for p in chi_cap
                    if ket["kiem_dinh"][str(a)].get(p, {}).get(bien, {}).get("dat"))
            dat.append(d)
            pb += [ket["kiem_dinh"][str(a)][p][bien]["pinball"]
                   for p in chi_cap if bien in ket["kiem_dinh"][str(a)].get(p, {})]
        ket["tong_ket"][bien] = {"dat_05": dat[0], "dat_01": dat[1],
                                 "pinball_tb": float(np.mean(pb)) if pb else None}
        print(f"  {bien:<14}{dat[0]:>6}/{len(chi_cap):<3}{dat[1]:>6}/{len(chi_cap):<3}"
              f"{np.mean(pb) if pb else float('nan'):>13.5f}", flush=True)

    ket["giay"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "caviar.json"), "w", encoding="utf-8") as f:
        json.dump(ket, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/caviar.json · {ket['giay']:.0f}s")


if __name__ == "__main__":
    main()
