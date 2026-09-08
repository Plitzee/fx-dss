"""CHIA SE CO DINH (Fixed-Share) CHO TO HOP HEDGE — co dang tien khong?

VAN DE. `balop.ToHopTrucTuyen` dung Hedge tran: w <- w exp(-eta l), chuan hoa.
Phep nhan do KHONG CO SAN. Mot chuyen gia thua lien tuc mot doan se bi dim
xuong 1e-6, va sau do du no tro thanh chuyen gia gioi nhat thi cung phai thang
rat nhieu phien lien tuc moi ngoi len lai. Neu CHE DO thi truong doi — ma thi
truong tien te thi doi — cai cham do la ton that thuc, moi lan doi che do mot
lan tra gia.

Fixed-Share (Herbster & Warmuth 1998, "Tracking the Best Expert") vá dung cho
do: moi phien, sau buoc Hedge, tron alpha phan trong so ve deu

    w <- (1 - alpha) w + alpha / N

Chan hoi tiec doi tu "so voi CHUYEN GIA tot nhat" thanh "so voi DAY chuyen gia
tot nhat co k lan chuyen" — dung thu ta muon khi co che do.

NHUNG PHAI DO. Fixed-Share khong mien phi: no keo trong so ve deu moi phien,
nen neu KHONG co doi che do — neu mot chuyen gia that su gioi nhat suot ca doan
— thi no chi lam loang mot to hop dang dung. Nen script nay do hai thu:

  CHAN DOAN   chuyen gia gioi nhat co DOI theo thoi gian khong? Neu khong doi
              lan nao thi Fixed-Share khong the an tien, va ket luan dung o day.
  QUET alpha  log-loss va BSS tren doan KIEM DINH, kem KTC bootstrap khoi cho
              HIEU SO so voi alpha = 0. Chenh lech khong vuot 0 thi la hoa.

KY LUAT DOAN. Chon alpha tren KIEM DINH. Doan kiem tra khong mo o day —
`docs/CHISO_DANHGIA.md` ghi so lan da mo, va moi lan mo la mot lan tieu.

Chay:  python src/kiem_fixshare.py
Ghi:   output/kiem_fixshare.json
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

import balop as B                                            # noqa: E402
import diem3 as D                                            # noqa: E402
import run_ml3 as M3                                         # noqa: E402
from balop import _chuan, EPS                                # noqa: E402
from split import VALID_TU, TEST_TU                          # noqa: E402

ALPHAS = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
NBOOT = 500
KHOI = 20
SEED = 7


def hedge(A, y, eta=0.5, alpha=0.0, nhom=None):
    """Hedge co chia se co dinh. A: (N, n, 3). Tra ve (P, lich_su_trong_so).

    Giu nguyen cau truc cua `run_ml3.trong_so_mu`: trong so RIENG tung cap, vi
    sau lan trong so mot cap voi cap khac la tron hai bai toan khac nhau."""
    N, n, _ = A.shape
    ra = np.zeros((n, 3))
    ls = np.zeros((n, N))
    g = np.zeros(n, int) if nhom is None else pd.factorize(np.asarray(nhom))[0]
    ws = {v: np.ones(N) / N for v in np.unique(g)}
    for t in range(n):
        wt = ws[g[t]]
        ra[t] = _chuan((wt[:, None] * A[:, t, :]).sum(0)[None, :])[0]
        ls[t] = wt
        if y[t] < 0:
            continue
        l = -np.log(np.maximum(A[:, t, y[t]], 1e-9))
        wt = wt * np.exp(-eta * (l - l.min()))
        wt = wt / max(wt.sum(), EPS)
        if alpha > 0:
            wt = (1.0 - alpha) * wt + alpha / N
        ws[g[t]] = wt
    return _chuan(ra), ls


def logloss(P, y):
    m = y >= 0
    return float(-np.log(np.maximum(P[m][np.arange(m.sum()), y[m]], 1e-12)).mean())


def chan_doan_doi_che_do(A, y, ten, nhom, cua_so=250):
    """Chuyen gia gioi nhat co DOI theo thoi gian khong?

    Cat doan kiem dinh thanh cac cua so roi nhau, moi cua so tim chuyen gia co
    log-loss thap nhat. Neu ra cung mot ten o moi cua so thi khong co gi de
    Fixed-Share theo duoi va ket luan dung o day."""
    n = A.shape[1]
    ra = []
    for a in range(0, n - cua_so // 2, cua_so):
        b = min(a + cua_so, n)
        yy = y[a:b]
        m = yy >= 0
        if m.sum() < 50:
            continue
        ll = [float(-np.log(np.maximum(
            A[k, a:b][m][np.arange(m.sum()), yy[m]], 1e-12)).mean())
            for k in range(A.shape[0])]
        j = int(np.argmin(ll))
        ra.append({"tu": a, "den": b, "n": int(m.sum()), "gioi_nhat": ten[j],
                   "log": round(ll[j], 4),
                   "cach_biet": round(float(np.sort(ll)[1] - ll[j]), 4)})
    return ra


def main():
    t0 = time.time()
    print("=" * 108)
    print("CHIA SẺ CỐ ĐỊNH (Fixed-Share) CHO TỔ HỢP HEDGE — quét α trên KIỂM ĐỊNH")
    print("=" * 108, flush=True)

    X, yR, yP, phu, ten_dt, pid, dts, hop_le = M3.nap()
    tr = (dts < VALID_TU) & hop_le
    va = (dts >= VALID_TU) & (dts < TEST_TU) & hop_le
    print(f"huấn luyện {int(tr.sum()):,} · kiểm định {int(va.sum()):,} "
          f"— đoạn kiểm tra KHÔNG mở ở đây", flush=True)

    ket = {"alphas": list(ALPHAS), "eta": 0.5, "nboot": NBOOT, "khoi": KHOI,
            "n_kiem_dinh": int(va.sum()), "muc_tieu": {}}

    for ten_mt, y in (("P", yP), ("R", yR)):
        print("\n" + "─" * 108)
        print(f"MỤC TIÊU {ten_mt}", flush=True)
        ok = y >= 0
        Xtr, ytr = X[tr & ok], y[tr & ok]

        # ── dựng dàn chuyên gia, tất cả khớp trên HUẤN LUYỆN ────────────
        Ps = {}
        for nen_ten, lop in (("chỉ σ̂", B.ChiSigma), ("σ̂ + chế độ", B.SigmaCheDo)):
            Pv = np.zeros((len(X), 3))
            for p in B.PAIRS:
                mp = phu.pair.values == p
                z, trp = phu.z.values[mp], tr[mp]
                mo = lop()
                (mo.khop(z[trp]) if lop is B.ChiSigma
                 else mo.khop(z[trp], phu.sig.values[mp][trp]))
                canh = phu["canh_R" if ten_mt == "R" else "canh_P"].values[mp]
                Pv[mp] = mo.du_bao(int(mp.sum()), canh=canh,
                                   sigma_h=phu.sigma_h.values[mp],
                                   sig=phu.sig.values[mp])
            Ps[nen_ten] = _chuan(np.where(np.isfinite(Pv), Pv, 1 / 3.0)[va])
        print("  đang khớp… ", end="", flush=True)
        for nhan, ham in (("logistic đa thức", M3.logistic_da_thuc),
                          ("LightGBM 3 lớp", M3.lightgbm_3lop),
                          ("GRU 3 lớp", M3.gru_3lop)):
            try:
                Ps[nhan], _ = ham(Xtr, ytr, X[va])
                print(f"{nhan} ✓ ", end="", flush=True)
            except Exception as e:
                print(f"{nhan} ✗({str(e)[:32]}) ", end="", flush=True)
        print(flush=True)

        ten = list(Ps)
        A = np.stack([Ps[k] for k in ten])
        yv, nhom = y[va], phu.pair.values[va]
        Pkh = M3.khi_hau_hoc(ytr, int(va.sum()))

        # ── CHẨN ĐOÁN: chuyên gia giỏi nhất có đổi không ────────────────
        cd = chan_doan_doi_che_do(A, yv, ten, nhom)
        so_ten = len({c["gioi_nhat"] for c in cd})
        print(f"\n  CHẨN ĐOÁN — chuyên gia giỏi nhất theo cửa sổ 250 phiên:")
        print(f"  {'cửa sổ':>14}{'n':>7}  {'giỏi nhất':<22}{'log':>8}"
              f"{'cách biệt':>11}")
        for c in cd:
            print(f"  {c['tu']:>6}–{c['den']:<7}{c['n']:>7}  "
                  f"{c['gioi_nhat']:<22}{c['log']:>8.4f}{c['cach_biet']:>11.4f}")
        print(f"  → {so_ten} chuyên gia khác nhau chiếm ngôi đầu trên "
              f"{len(cd)} cửa sổ", flush=True)

        # ── QUÉT α ──────────────────────────────────────────────────────
        print(f"\n  QUÉT α — log-loss và BSS trên kiểm định")
        print(f"  {'α':>7}{'log':>9}{'Δlog vs α=0':>14}{'BSS':>9}"
              f"{'ΔBSS':>9}{'KTC 95% của ΔBSS':>22}", flush=True)
        P0 = None
        hang = []
        for a in ALPHAS:
            P, ls = hedge(A, yv, eta=0.5, alpha=a, nhom=nhom)
            r = D.bang(P[yv >= 0], yv[yv >= 0], Pkh[yv >= 0],
                       nhom=nhom[yv >= 0])
            ll = logloss(P, yv)
            if P0 is None:
                P0, ll0, bss0 = P, ll, r["bss"]
                dlo = dhi = dbss = 0.0
            else:
                dbss = r["bss"] - bss0
                dlo, dhi = bss_hieu_ktc(P, P0, yv, Pkh, nhom)
            hang.append({"alpha": a, "log": ll, "bss": r["bss"],
                         "d_log": ll - ll0, "d_bss": dbss,
                         "d_bss_lo": dlo, "d_bss_hi": dhi,
                         "ece": r["ece"],
                         "trong_so_cuoi": dict(zip(ten, ls[-1].tolist()))})
            ktc = ("      —  (mốc)" if a == 0.0
                   else f"[{dlo:+.5f}; {dhi:+.5f}]")
            sao = " *" if (a > 0 and dlo > 0) else ""
            print(f"  {a:>7.3f}{ll:>9.4f}{ll-ll0:>+14.5f}{r['bss']:>+9.4f}"
                  f"{dbss:>+9.5f}{ktc:>22}{sao}", flush=True)

        tot = max(hang[1:], key=lambda h: h["d_bss"])
        an = tot["d_bss_lo"] > 0
        print(f"\n  → α tốt nhất {tot['alpha']:.3f}: ΔBSS {tot['d_bss']:+.5f} "
              f"KTC [{tot['d_bss_lo']:+.5f}; {tot['d_bss_hi']:+.5f}]")
        print(f"  → {'ĂN TIỀN — cận dưới KTC vượt 0' if an else 'HOÀ — KTC chứa 0, không có bằng chứng Fixed-Share giúp'}",
              flush=True)
        ket["muc_tieu"][ten_mt] = {"chuyen_gia": ten, "chan_doan": cd,
                                   "so_ten_dau_bang": so_ten, "quet": hang,
                                   "alpha_tot": tot["alpha"], "an_tien": an}

    ket["giay"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "kiem_fixshare.json"), "w",
              encoding="utf-8") as f:
        json.dump(ket, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/kiem_fixshare.json · {ket['giay']:.0f}s")


def bss_hieu_ktc(P, P0, y, Pkh, nhom, nboot=NBOOT, khoi=KHOI, seed=SEED):
    """KTC bootstrap KHOI cho HIEU BSS(P) - BSS(P0), GHEP CAP.

    Ghep cap la cho quan trong: hai to hop chi khac nhau o alpha nen chung phan
    lon phuong sai. Bootstrap rieng roi tru hai KTC se ra khoang rong gap may
    lan su that va giau mat moi khac biet nho."""
    m = y >= 0
    P, P0, y, Pkh = P[m], P0[m], y[m], Pkh[m]
    nhom = np.asarray(nhom)[m]
    n = len(y)
    nb = int(np.ceil(n / khoi))
    rng = np.random.default_rng(seed)
    ra = []
    for _ in range(nboot):
        idx = np.concatenate([np.arange(k * khoi, min((k + 1) * khoi, n))
                              for k in rng.integers(0, nb, nb)])[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            ra.append(D.bang(P[idx], y[idx], Pkh[idx], nhom=nhom[idx])["bss"]
                      - D.bang(P0[idx], y[idx], Pkh[idx], nhom=nhom[idx])["bss"])
        except Exception:
            continue
    if len(ra) < 50:
        return float("nan"), float("nan")
    return float(np.quantile(ra, 0.025)), float(np.quantile(ra, 0.975))


if __name__ == "__main__":
    main()
