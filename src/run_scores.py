"""BO CHI SO DANH GIA DAY DU — cham tren doan KIEM TRA.

Do phu va be rong khoang chi noi ve MOT muc tin cay. Mot he thong ho tro quyet
dinh dua ra ca mot PHAN PHOI du bao, nen phai cham bang thuoc do danh gia ca
phan phoi. File nay dung:

  CRPS          quy tac cham diem CHINH DANG cho toan phan phoi (Gneiting-Raftery)
  Pinball       mat mat phan vi tung muc — cho biet sai o dau tren phan phoi
  Log score     mat mat logarit cua mat do du bao
  PIT + KS      phep bien doi tich phan xac suat; neu phan phoi dung thi PIT
                phai phan bo DEU tren [0,1]
  Kupiec        H0: ty le vi pham VaR dung bang alpha
  Christoffersen H0: cac lan vi pham DOC LAP (khong dinh cum)
  DQ            Engle-Manganelli, manh hon ca hai cai tren
  FZ0           mat mat chung cho cap (VaR, ES) — Patton-Ziegel-Chen 2019
  Winkler       diem khoang, phat ca be rong lan lan truot

Moi tham so uoc luong tren doan HUAN LUYEN. Doan kiem tra chi cham diem.
"""
import os, sys
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
D = os.path.join(os.path.dirname(HERE), "data")
from scipy import stats
from split import doan
from metrics import (crps_from_quantiles, pinball, fz0, kupiec,
                     christoffersen_ind, dq_test, TAU_GRID)

P = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
pan = pd.read_csv(os.path.join(D, "panel2_6pairs.csv"), parse_dates=["Date"])
TAU_BAO = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


class PhanPhoi:
    """Phan phoi du bao cho loi suat mot phien, tam o 0, thang do theo sig."""

    def __init__(self, ten, z_tr, s_tr, kieu, n_bins=1, sig_cot="sig"):
        self.ten, self.kieu, self.sig_cot = ten, kieu, sig_cot
        self.n_bins = n_bins
        self.edges = np.quantile(s_tr, np.arange(1, n_bins) / n_bins) if n_bins > 1 else np.array([])
        g = np.digitize(s_tr, self.edges) if n_bins > 1 else np.zeros(len(s_tr), int)
        self.z = [z_tr[g == i] if (g == i).sum() >= 100 else z_tr for i in range(n_bins)]
        if kieu == "t":
            nu, _, sc = stats.t.fit(z_tr, floc=0)
            self.nu = float(np.clip(nu, 2.5, 40)); self.sc = float(sc)
        if kieu == "kinh nghiệm":
            self.kde = [stats.gaussian_kde(zz) for zz in self.z]

    def _bin(self, sig):
        return int(np.digitize([sig], self.edges)[0]) if self.n_bins > 1 else 0

    def qz(self, taus, b=0):
        """Phan vi CHUAN HOA (chua nhan sig) — vector hoa."""
        taus = np.asarray(taus, float)
        if self.kieu == "gauss":
            return stats.norm.ppf(taus)
        if self.kieu == "t":
            return stats.t.ppf(taus, self.nu) * self.sc
        return np.quantile(self.z[b], taus)

    def bins(self, sig):
        sig = np.asarray(sig, float)
        return (np.digitize(sig, self.edges) if self.n_bins > 1
                else np.zeros(len(sig), int))

    def Q(self, taus, sig):
        """Ma tran phan vi (n, len(taus))."""
        sig = np.asarray(sig, float); b = self.bins(sig)
        out = np.empty((len(sig), len(taus)))
        for i in range(self.n_bins):
            m = b == i
            if m.any():
                out[m] = self.qz(taus, i)[None, :] * sig[m][:, None]
        return out

    def q(self, tau, sig):
        """Phan vi cua loi suat tai muc tau (vo huong)."""
        return float(self.qz([tau], self._bin(sig))[0]) * sig

    def cdf(self, y, sig):
        z = y / sig
        if self.kieu == "gauss":
            return float(stats.norm.cdf(z))
        if self.kieu == "t":
            return float(stats.t.cdf(z / self.sc, self.nu))
        zz = self.z[self._bin(sig)]
        return float((zz <= z).mean())

    def logpdf(self, y, sig):
        z = y / sig
        if self.kieu == "gauss":
            return float(stats.norm.logpdf(z) - np.log(sig))
        if self.kieu == "t":
            return float(stats.t.logpdf(z / self.sc, self.nu) - np.log(self.sc) - np.log(sig))
        return float(np.log(max(self.kde[self._bin(sig)](z)[0], 1e-12)) - np.log(sig))

    def es(self, alpha, sig):
        """ES duoi muc alpha (gia tri AM)."""
        if self.kieu == "gauss":
            return float(-stats.norm.pdf(stats.norm.ppf(alpha)) / alpha * sig)
        if self.kieu == "t":
            x = stats.t.ppf(alpha, self.nu)
            return float(-stats.t.pdf(x, self.nu) * (self.nu + x ** 2) / ((self.nu - 1) * alpha)
                         * self.sc * sig)
        zz = self.z[self._bin(sig)]
        v = np.quantile(zz, alpha)
        t = zz[zz <= v]
        return float((t.mean() if len(t) else v) * sig)


def dm(x):
    n = len(x); mb = x.mean(); L = int(np.ceil(1.5 * n ** (1 / 3))); s = np.sum((x - mb) ** 2) / n
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * np.sum((x[k:] - mb) * (x[:-k] - mb)) / n
    return mb / np.sqrt(max(s, 1e-16) / n)


def chuan_bi():
    CAU = {}
    for p in P:
        d = pan[pan.pair == p].reset_index(drop=True)
        g = doan(d.Date.values)
        r = d.zT.values * d.sig.values                 # loi suat log thuc te
        s_new = d.sig.values; s_old = d.sig_old.values
        m = np.isfinite(s_old)
        tr = (g == 0) & m; te = (g == 2) & m
        CAU[p] = {"r": r, "sig": s_new, "sig_old": s_old, "tr": tr, "te": te}
    return CAU


CAU = chuan_bi()
MO_HINH = [
    ("Gauss", "gauss", 1, "sig"),
    ("Student-t", "t", 1, "sig"),
    ("Kinh nghiệm chung", "kinh nghiệm", 1, "sig"),
    ("Mondrian 2 (đang dùng)", "kinh nghiệm", 2, "sig"),
    ("Mondrian 3", "kinh nghiệm", 3, "sig"),
    ("Student-t + sig CŨ", "t", 1, "sig_old"),
]


def chay():
    KQ = {}
    for ten, kieu, nb, cot in MO_HINH:
        acc = dict(crps=[], logs=[], pin={t: [] for t in TAU_BAO}, wink=[], pit=[],
                   hit1=[], hit5=[], var1=[], var5=[], fz=[])
        for p in P:
            c = CAU[p]; s = c[cot]
            tr, te = c["tr"], c["te"]
            z_tr = c["r"][tr] / s[tr]
            m = PhanPhoi(ten, z_tr, s[tr], kieu, nb)
            y = c["r"][te]; sg = s[te]
            Q = m.Q(TAU_GRID, sg)
            acc["crps"].append(crps_from_quantiles(y, Q))
            acc["logs"].append(np.array([-m.logpdf(a, b) for a, b in zip(y, sg)]))
            QB = m.Q(np.array(TAU_BAO), sg)
            for j, t in enumerate(TAU_BAO):
                acc["pin"][t].append(pinball(y, QB[:, j], t))
            QE = m.Q(np.array([0.05, 0.95, 0.01, 0.025]), sg)
            lo = QE[:, 0]; hi = QE[:, 1]
            acc["wink"].append((hi - lo) + (2 / 0.10) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0)))
            acc["pit"].append(np.array([m.cdf(a, b) for a, b in zip(y, sg)]))
            v1 = QE[:, 2]; v5 = lo
            acc["hit1"].append((y <= v1).astype(int)); acc["hit5"].append((y <= v5).astype(int))
            acc["var1"].append(v1); acc["var5"].append(v5)
            e = np.array([m.es(0.025, x) for x in sg]); v = QE[:, 3]
            acc["fz"].append(fz0(y, v, e, 0.025))
        KQ[ten] = acc
    return KQ


KQ = chay()
pip = 1e4

print("=" * 108)
print("BANG A — QUY TAC CHAM DIEM CHINH DANG, doan KIEM TRA (4.316 quan sat, don vi pip)")
print("=" * 108)
print(f"{'phân phối dự báo':<26}{'CRPS':>10}{'log score':>12}{'điểm khoảng 90%':>18}{'FZ0 (VaR/ES 2,5%)':>20}")
print("-" * 108)
for ten, _, _, _ in MO_HINH:
    a = KQ[ten]
    print(f"{ten:<26}{np.mean(np.concatenate(a['crps']))*pip:>10.2f}"
          f"{np.mean(np.concatenate(a['logs'])):>12.4f}"
          f"{np.mean(np.concatenate(a['wink']))*pip:>18.2f}"
          f"{np.mean(np.concatenate(a['fz'])):>20.4f}")
print("-" * 108)
print("Thấp hơn là tốt hơn ở cả bốn cột. CRPS và điểm khoảng tính bằng pip.")

print("\n" + "=" * 108)
print("BANG B — MAT MAT PHAN VI (pinball, pip) — sai o dau tren phan phoi")
print("=" * 108)
print(f"{'phân phối dự báo':<26}" + "".join(f"{f'τ={t:g}':>11}" for t in TAU_BAO))
print("-" * 108)
for ten, _, _, _ in MO_HINH:
    a = KQ[ten]
    print(f"{ten:<26}" + "".join(f"{np.mean(np.concatenate(a['pin'][t]))*pip:>11.3f}" for t in TAU_BAO))

print("\n" + "=" * 108)
print("BANG C — DIEBOLD-MARIANO TREN CRPS, so voi ban dang dung (Mondrian 2)")
print("=" * 108)
BASE = "Mondrian 2 (đang dùng)"
print(f"{'phân phối dự báo':<26}" + "".join(f"{p:>12}" for p in P) + f"{'thua':>8}")
print("-" * 108)
for ten, _, _, _ in MO_HINH:
    if ten == BASE:
        continue
    line = f"{ten:<26}"; lose = 0
    for i, p in enumerate(P):
        t = dm(KQ[ten]["crps"][i] - KQ[BASE]["crps"][i]); pv = 2 * (1 - stats.norm.cdf(abs(t)))
        sg = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .1 else ""
        if t > 0 and pv < .05:
            lose += 1
        line += f"{f'{t:+.2f}{sg}':>12}"
    print(line + f"{lose:>5}/6")
print("-" * 108)
print("t dương = TỆ HƠN bản đang dùng.  *** p<0,01  ** p<0,05  * p<0,1")

print("\n" + "=" * 108)
print("BANG D — PIT: phan phoi du bao co DUNG DANG khong?")
print("=" * 108)
print("Nếu phân phối đúng thì PIT phải phân bố đều trên [0,1].")
print(f"\n{'phân phối dự báo':<26}{'KS stat':>10}{'p-value':>10}{'PIT<0,05':>11}{'PIT>0,95':>11}"
      f"{'trung bình':>12}{'kết luận':>12}")
print("-" * 108)
for ten, _, _, _ in MO_HINH:
    u = np.concatenate(KQ[ten]["pit"])
    ks, pv = stats.kstest(u, "uniform")
    kl = "ĐẠT" if pv > 0.05 else ("bác bỏ" if pv > 0.001 else "bác bỏ mạnh")
    print(f"{ten:<26}{ks:>10.4f}{pv:>10.4f}{(u<0.05).mean():>11.1%}{(u>0.95).mean():>11.1%}"
          f"{u.mean():>12.3f}{kl:>12}")
print("-" * 108)
print("Cột PIT<0,05 và PIT>0,95 phải xấp xỉ 5,0%. Cao hơn = đuôi quá mỏng.")

print("\n" + "=" * 108)
print("BANG E — BACKTEST VaR (Kupiec / Christoffersen / DQ), gop 6 cap")
print("=" * 108)
for al, key, vkey in ((0.01, "hit1", "var1"), (0.05, "hit5", "var5")):
    print(f"\nMức VaR {al:.0%}  — kỳ vọng {al:.1%} lần vi phạm")
    print(f"{'phân phối dự báo':<26}{'vi phạm':>10}{'Kupiec p':>11}{'Christof. p':>13}"
          f"{'DQ p':>9}{'kết luận':>14}")
    print("-" * 108)
    for ten, _, _, _ in MO_HINH:
        a = KQ[ten]
        pk = []; pc = []; pd_ = []; rate = []
        for i in range(len(P)):
            h = a[key][i]
            _, p1, ph = kupiec(h, al); _, p2 = christoffersen_ind(h)
            rate.append(ph); pk.append(p1); pc.append(p2)
            if vkey:
                _, p3 = dq_test(h, a[vkey][i], al)
                pd_.append(p3)
        f = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
        ok = (f(pk) > 0.05) and (f(pc) > 0.05) and (not pd_ or f(pd_) > 0.05)
        print(f"{ten:<26}{f(rate):>10.2%}{f(pk):>11.3f}{f(pc):>13.3f}"
              f"{(f(pd_) if pd_ else float('nan')):>9.3f}{('ĐẠT' if ok else 'không đạt'):>14}")
print("-" * 108)
print("p > 0,05 = không bác bỏ được giả thuyết mô hình đúng (đây là điều ta muốn).")
print("Kupiec: đúng tần suất.  Christoffersen: vi phạm không dính cụm.  DQ: cả hai.")


print("\n" + "=" * 108)
print("BANG F — CHI SO CHO DU BAO BIEN DONG DIEM (khong phai phan phoi), doan KIEM TRA")
print("=" * 108)
print("Mincer-Zarnowitz: hoi quy log(RV thuc) = a + b*log(du bao). Mo hinh khong")
print("thien lech thi a=0, b=1. R2_log cho biet giai thich duoc bao nhieu phuong sai.\n")
adv = pd.read_csv(os.path.join(D, "rv_adv.csv"), parse_dates=["Date"])
print(f"{'dự báo sig':<22}{'QLIKE':>10}{'MSE (×1e10)':>14}{'a':>9}{'b':>8}{'R² log':>9}{'corr log':>10}")
print("-" * 108)
for cot, ten in (("sig", "mới (tổ hợp HAR)"), ("sig_old", "cũ (MA20-GK)")):
    Q = []; M = []; A = []; B = []; R2 = []; CR = []
    for p in P:
        d = pan[pan.pair == p].reset_index(drop=True)
        g = doan(d.Date.values)
        m = (g == 2) & np.isfinite(d[cot].values) & np.isfinite(d.rv5.values) & (d.rv5.values > 0)
        f = d[cot].values[m] ** 2          # du bao PHUONG SAI
        yv = d.rv5.values[m]
        Q.append(np.mean(yv / f - np.log(yv / f) - 1))
        M.append(np.mean((yv - f) ** 2))
        x = np.log(f); yy = np.log(yv)
        X = np.column_stack([np.ones(len(x)), x])
        b = np.linalg.lstsq(X, yy, rcond=None)[0]
        res = yy - X @ b
        A.append(b[0]); B.append(b[1])
        R2.append(1 - res.var() / yy.var()); CR.append(np.corrcoef(x, yy)[0, 1])
    print(f"{ten:<22}{np.mean(Q):>10.4f}{np.mean(M)*1e10:>14.3f}{np.mean(A):>9.3f}"
          f"{np.mean(B):>8.3f}{np.mean(R2):>9.3f}{np.mean(CR):>10.3f}")
print("-" * 108)
print("QLIKE va MSE la hai thuoc do duy nhat BEN voi viec RV chi la thuoc do nhieu")
print("cua bien dong that (Patton 2011). Cac thuoc do khac co the xep hang sai.")
