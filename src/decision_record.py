"""TANG 6 — PHIEU QUYET DINH: bien ket qua tang 1-5 thanh mot to phieu doc duoc.

Ba phan cua phieu, va cai nao kiem dinh duoc:

  1. HANH DONG  (don bay khuyen nghi)  -> tu PositionSizer, tang 4
  2. RUI RO     (P cham stop, khoang gia) -> KIEM DINH DUOC, va da kiem
  3. GIAI THICH (rang buoc nao dang siet) -> tu PositionSizer.explain

Ket qua kiem dinh hieu chuan (du lieu giu rieng, 6 cap, n=8.957 moi nguong):

  * P(cham stop) theo nguyen ly phan xa: lech tuyet doi TB 1,44%,
    toi da 3,08% (o stop rat gan 0,5 sigma, va lech theo huong BI QUAN).
    O vung hay dung 2-3 sigma: lech 0,40% va 0,10%.

  * Khoang du bao, |lech| trung binh tren 4 muc 80/90/95/99%:
        Gauss     1,20%   (nhung 99% danh nghia chi phu 97,5% — hut duoi)
        Student-t 1,25%   (be rong 158,0 pip o muc 99%)
        Conformal 0,37%   (be rong 145,6 pip — vua chuan hon vua hep hon)

  * Bao dam cua conformal chi la BIEN. Do phu theo che do bien dong
    (muc danh nghia 90%):
        Conformal chung        88,4 / 90,8 / 91,9   -> lech max 1,9%
        Conformal theo che do  90,3 / 90,8 / 89,4   -> lech max 0,8%
    Vi vay tang 6 dung ban PHAN TANG THEO CHE DO BIEN DONG (Mondrian).

  * SO TANG PHU THUOC CHAT LUONG DU BAO sig. Sau khi doi sang du bao moi
    (src/volfc.py) thi 3 tang lam lech max XAU DI thanh 2,4%, con 2 tang cho
    1,1%. Mac dinh n_bins=2. Xem docs/TANG2_BIENDONG.md muc 6.

GIOI HAN DA DO, PHAI GHI VAO LUAN VAN: ca ba phuong phap deu PHU THIEU
khi tai khoan dang lo (Conformal 89,3% luc sut giam so voi 90,3% luc o
dinh von) — dung luc nguoi dung can con so chinh xac nhat.
"""
import numpy as np
from scipy import stats

from position_sizing import PositionSizer          # noqa: F401  (tai xuat)

# sai so hieu chuan DA DO — dung de in "do tin cay cua chinh con so nay"
LECH_DA_DO_KHOANG = 0.012        # conformal phan tang, lech toi da theo che do
LECH_DA_DO_PSTOP = 0.0144        # P(cham stop), lech tuyet doi trung binh
LECH_KHI_DANG_LO = 0.010         # phu thieu them khi dang sut giam


def pip_size(pair):
    return 0.01 if "JPY" in pair.upper() else 0.0001


# ───────────────────────── xac suat cham stop ─────────────────────────
def p_cham_stop(k_sigma, z_train, horizon=1):
    """P(gia cham stop dat cach k_sigma lan do lech chuan) trong 'horizon' phien.

    Nguyen ly phan xa: P(min_{t<=T} X_t <= -b) = 2 P(X_T <= -b) voi buoc di
    doi xung. Duoi t-Student khop tren tap huan luyen, khong phai Gauss."""
    nu, _, sc = stats.t.fit(np.asarray(z_train, float), floc=0)
    nu = float(np.clip(nu, 2.5, 40))
    return float(min(1.0, 2.0 * stats.t.cdf(-k_sigma / (sc * np.sqrt(horizon)), nu)))


# ───────────────────── khoang conformal phan tang ─────────────────────
class KhoangConformal:
    """Conformal chia doi, PHAN TANG theo phan vi bien dong (Mondrian).

    So tang la mot danh doi: nhieu tang thi bam sat che do hon nhung moi tang
    it mau hieu chuan hon. Voi du bao sig CU (MA20-GK) thi 3 tang la tot nhat;
    voi du bao MOI (volfc) thi 2 tang tot hon — do sac hon nen phan du con lai
    it khong dong nhat hon, va it tang cho nhieu mau hon:

        sig cu, 3 tang   89,5 / 90,3 / 88,8   lech max 1,2%
        sig moi, 2 tang  88,9 / 90,0          lech max 1,1%
        sig moi, 3 tang  87,6 / 90,7 / 89,3   lech max 2,4%

    Tang nao mong (< min_n mau) thi lui ve dung bo chung."""

    def __init__(self, z_train, sig_train, n_bins=2, min_n=50):
        z = np.asarray(z_train, float)
        s = np.asarray(sig_train, float)
        self.n_bins = int(n_bins)
        self.edges = np.quantile(s, np.arange(1, self.n_bins) / self.n_bins)
        g = np.digitize(s, self.edges)
        self.z_chung = z
        self.z_theo_che_do = [z[g == i] if (g == i).sum() >= min_n else z
                              for i in range(self.n_bins)]
        self.n_theo_che_do = [int((g == i).sum()) for i in range(self.n_bins)]

    def che_do(self, sig):
        return int(np.digitize([float(sig)], self.edges)[0])

    def ten_che_do(self, sig):
        i = self.che_do(sig)
        if self.n_bins == 2:
            return ("thấp", "cao")[i]
        if self.n_bins == 3:
            return ("thấp", "vừa", "cao")[i]
        return f"tầng {i+1}/{self.n_bins}"

    def nua_be_rong(self, muc, sig=None):
        """Nua be rong khoang, don vi 'so lan do lech chuan'."""
        z = self.z_chung if sig is None else self.z_theo_che_do[self.che_do(sig)]
        return float(np.quantile(np.abs(z), min(muc * (1 + 1 / len(z)), 0.9999)))

    def khoang(self, gia, sig, muc=0.90):
        h = self.nua_be_rong(muc, sig) * sig
        return float(gia * np.exp(-h)), float(gia * np.exp(h))


class KhoangACI:
    """CONFORMAL THICH UNG THEO TANG — ban dang dung o tang 6.

    Hai y tuong ghep lai:
      * MONDRIAN: hieu chuan rieng cho tung che do bien dong (2 tang)
      * ACI (Gibbs & Candes, NeurIPS 2021): cap nhat muc alpha TRUC TUYEN
            alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
        voi err_t = 1 neu quan sat that roi ra ngoai khoang. Bao dam: do phu
        dai han hoi tu ve dung muc danh nghia BAT KE phan phoi troi the nao.

    Do tren tap giu rieng, muc danh nghia 90%, panel moi (lech so voi 90%):

        phuong phap    chung   o dinh  dang lo  vol thap  vol cao  lech max
        tinh           89,6%    89,8%    89,2%     86,9%    90,2%      3,1%
        mondrian 3     89,6%    90,0%    89,2%     87,6%    89,3%      2,4%
        cua so truot   89,7%    89,9%    89,3%     87,0%    90,1%      3,0%
        ACI chung      90,2%    90,5%    89,8%     86,8%    91,5%      3,2%
        DtACI          89,8%    90,1%    89,5%     87,0%    91,1%      3,0%
        ACI THEO TANG  90,3%    90,7%    89,9%     88,8%    90,1%      1,2%   <-

    Diem khoang (Winkler) cung tot nhat: 285,0 so voi 290,0 cua ban tinh.

    CON LAI CHUA SUA DUOC: moi phuong phap deu phu thieu khoang 0,6-0,8 diem
    phan tram khi tai khoan dang lo. Da thu ca nam cach, khong cach nao xoa
    duoc khoang chenh do. Phai ghi trong luan van la gioi han da do."""

    def __init__(self, z_train, sig_train, n_bins=2, muc=0.90, gamma=0.01, cua_so=750):
        z = np.asarray(z_train, float); s = np.asarray(sig_train, float)
        self.n_bins, self.muc, self.gamma, self.cua_so = int(n_bins), float(muc), float(gamma), int(cua_so)
        self.edges = np.quantile(s, np.arange(1, self.n_bins) / self.n_bins)
        self.alpha = np.full(self.n_bins, 1.0 - self.muc)
        self.z_hist = list(z[-self.cua_so:])
        self.s_hist = list(s[-self.cua_so:])

    def che_do(self, sig):
        return int(np.digitize([float(sig)], self.edges)[0])

    def ten_che_do(self, sig):
        i = self.che_do(sig)
        return ("thấp", "cao")[i] if self.n_bins == 2 else f"tầng {i+1}/{self.n_bins}"

    def _cal(self, i):
        z = np.asarray(self.z_hist); s = np.asarray(self.s_hist)
        sel = z[np.digitize(s, self.edges) == i]
        return sel if len(sel) >= 60 else z

    def nua_be_rong(self, muc=None, sig=None):
        """Nua be rong, don vi 'so lan do lech chuan'. muc=None -> muc thich ung."""
        i = 0 if sig is None else self.che_do(sig)
        lev = (1.0 - self.alpha[i]) if muc is None else float(muc)
        c = self._cal(i)
        return float(np.quantile(np.abs(c), min(max(lev, 0.0) * (1 + 1 / len(c)), 0.9999)))

    def khoang(self, gia, sig, muc=None):
        h = self.nua_be_rong(muc, sig) * sig
        return float(gia * np.exp(-h)), float(gia * np.exp(h))

    def quan_sat(self, z, sig):
        """Goi SAU khi biet ket qua that: cap nhat alpha va bo nho hieu chuan."""
        i = self.che_do(sig)
        err = float(abs(float(z)) > self.nua_be_rong(None, sig))
        a = self.alpha[i] + self.gamma * ((1.0 - self.muc) - err)
        self.alpha[i] = float(np.clip(a, 1e-4, 0.5))
        self.z_hist.append(float(z)); self.s_hist.append(float(sig))
        if len(self.z_hist) > self.cua_so:
            self.z_hist.pop(0); self.s_hist.pop(0)
        return err

    @property
    def muc_hien_hanh(self):
        return 1.0 - self.alpha


# ───────────────────────────── to phieu ─────────────────────────────
class PhieuQuyetDinh:
    def __init__(self, sizer, khoang, pair, z_train):
        self.sizer, self.khoang, self.pair = sizer, khoang, pair
        self.z_train = np.asarray(z_train, float)

    def lap(self, gia, sig, mu, nu, dd=0.0, stop_sigma=2.0,
            muc=(0.80, 0.95), von=10000.0):
        ex = self.sizer.explain(sig, mu, nu, dd)
        p = p_cham_stop(stop_sigma, self.z_train)
        kh = {m: self.khoang.khoang(gia, sig, m) for m in muc}
        return dict(
            pair=self.pair, gia=float(gia), von=float(von),
            don_bay=ex["f"], von_dat=ex["f"] * von,
            rang_buoc=ex["rang_buoc"], k_vol=ex["k_vol"], k_dd=ex["k_dd"],
            kelly=ex["kelly"], tran_rui_ro=ex["ruin_cap"],
            muc_bien_dong=ex["muc_bien_dong"], sut_giam=float(dd),
            stop_sigma=float(stop_sigma),
            stop_gia=float(gia * (1 - stop_sigma * sig)),
            stop_pip=float(stop_sigma * sig * gia / pip_size(self.pair)),
            p_cham_stop=p, khoang=kh,
            n_mau_che_do=self.khoang.n_theo_che_do[self.khoang.che_do(sig)],
        )

    def in_ra(self, r, W=64):
        pip = pip_size(self.pair)
        L, a = [], None
        rows = []
        rows.append(f"PHIẾU QUYẾT ĐỊNH — {r['pair']}")
        rows.append(None)                                   # ke ngang
        rows.append(f"Giá tham chiếu {r['gia']:.5f}    Vốn {r['von']:,.0f}")
        rows.append(f"Đòn bẩy khuyến nghị {r['don_bay']:.2f}×  →  đặt {r['von_dat']:,.0f}")
        rows.append("")
        rows.append(f"VÌ SAO: ràng buộc đang siết là {r['rang_buoc'].upper()}")
        rows.append(f"  Kelly {r['kelly']:.2f}×    trần rủi ro {r['tran_rui_ro']:.2f}×")
        rows.append(f"  k biến động {r['k_vol']:.2f} (mức {r['muc_bien_dong']})"
                    f"   k sụt giảm {r['k_dd']:.2f} (−{r['sut_giam']:.0%})")
        rows.append("")
        rows.append(f"RỦI RO: stop {r['stop_sigma']:.1f}σ tại {r['stop_gia']:.5f}"
                    f" ({r['stop_pip']:.0f} pip)")
        rows.append(f"  Xác suất chạm stop trong 1 phiên: {r['p_cham_stop']:.2%}"
                    f"  (±{LECH_DA_DO_PSTOP:.1%})")
        for m, (lo, hi) in sorted(r["khoang"].items()):
            rows.append(f"  Khoảng giá {m:.0%}: {lo:.5f} – {hi:.5f}"
                        f"  ({(hi - lo) / pip:.0f} pip)")
        rows.append("")
        rows.append("ĐỘ TIN CẬY CỦA CHÍNH CÁC SỐ TRÊN")
        rows.append(f"  Khoảng: conformal phân tầng theo chế độ biến động,")
        rows.append(f"  lệch ≤{LECH_DA_DO_KHOANG:.1%}; mẫu hiệu chuẩn chế độ này:"
                    f" {r['n_mau_che_do']:,} phiên")
        rows.append(f"  ⚠ khi đang lỗ, độ phủ thực tế thấp hơn ghi ~{LECH_KHI_DANG_LO:.0%}")
        out = ["┌" + "─" * W + "┐"]
        for t in rows:
            if t is None:
                out.append("├" + "─" * W + "┤")
            else:
                out.append("│ " + t[: W - 2].ljust(W - 2) + " │")
        out.append("└" + "─" * W + "┘")
        return "\n".join(out)


# ───────────────────────────── tu kiem ─────────────────────────────
if __name__ == "__main__":
    import os, pandas as pd, warnings
    warnings.filterwarnings("ignore")
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    pan = pd.read_csv(os.path.join(d, "panel2_6pairs.csv"), parse_dates=["Date"])
    PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]

    print("TU KIEM TANG 6")
    # 1) do phu thuc te cua khoang phan tang, tren tap giu rieng
    for muc in (0.80, 0.90, 0.95):
        cov = []
        for p in PAIRS:
            g = pan[pan.pair == p].reset_index(drop=True)
            n = int(len(g) * 0.70)
            tr, te = g.iloc[:n], g.iloc[n:]
            kc = KhoangConformal(tr.zT.values, tr.sig.values)
            h = np.array([kc.nua_be_rong(muc, s) for s in te.sig.values])
            cov.append(float(np.mean(np.abs(te.zT.values) <= h)))
        m = float(np.mean(cov))
        print(f"  độ phủ ở mức {muc:.0%}: {m:.1%}  (lệch {m - muc:+.1%})")
        assert abs(m - muc) < 0.03, "khoang phai phu gan muc danh nghia"

    # 1b) ban thich ung theo tang — ban dang dung
    covA=[]; covM=[]
    for p in PAIRS:
        g = pan[pan.pair == p].reset_index(drop=True)
        n = int(len(g) * 0.70); tr, te = g.iloc[:n], g.iloc[n:]
        ka = KhoangACI(tr.zT.values, tr.sig.values)
        ok = []
        for _, row in te.iterrows():
            h = ka.nua_be_rong(None, row.sig)
            ok.append(abs(row.zT) <= h)
            ka.quan_sat(row.zT, row.sig)
        covA.append(float(np.mean(ok)))
        kc0 = KhoangConformal(tr.zT.values, tr.sig.values)
        hh = np.array([kc0.nua_be_rong(0.90, x) for x in te.sig.values])
        covM.append(float(np.mean(np.abs(te.zT.values) <= hh)))
    a, m = float(np.mean(covA)), float(np.mean(covM))
    print(f"  ACI theo tầng, độ phủ ở mức 90%: {a:.1%}   (bản tĩnh: {m:.1%})")
    assert abs(a - 0.90) < 0.02, "ACI phai phu gan 90%"

    # 2) khoang phai NO ra khi tang muc tin cay
    g = pan[pan.pair == "EURUSD"].reset_index(drop=True)
    n = int(len(g) * 0.70)
    tr, te = g.iloc[:n], g.iloc[n:]
    kc = KhoangConformal(tr.zT.values, tr.sig.values)
    assert kc.nua_be_rong(0.95, kc.edges[-1]) > kc.nua_be_rong(0.80, kc.edges[-1])
    print("  khoảng nở theo mức tin cậy: ĐẠT")

    # 3) P(cham stop) phai giam khi stop dat xa hon
    ps = [p_cham_stop(k, tr.zT.values) for k in (0.5, 1.0, 2.0, 3.0)]
    assert all(ps[i] > ps[i + 1] for i in range(len(ps) - 1))
    print(f"  P(chạm stop) 0,5σ→3σ: " + " > ".join(f"{x:.1%}" for x in ps) + "  ĐẠT")

    # 4) phieu thuc te
    sizer = PositionSizer(tr.sig.values)
    pq = PhieuQuyetDinh(sizer, kc, "EURUSD", tr.zT.values)
    gia = 1.0850
    r = pq.lap(gia, float(te.sig.iloc[0]), 0.0002, 6.0, dd=0.08, stop_sigma=2.0)
    print()
    print(pq.in_ra(r))
    r2 = pq.lap(gia, float(te.sig.iloc[0]), 0.0002, 6.0, dd=0.25, stop_sigma=2.0)
    assert r2["don_bay"] < r["don_bay"], "sut giam sau phai giam co"
    print(f"\n  cùng phiên nhưng sụt giảm 25%: đòn bẩy "
          f"{r['don_bay']:.2f}× → {r2['don_bay']:.2f}×  ĐẠT")
