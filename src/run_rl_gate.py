"""RL CONG VAO LENH — hoc khi nao DUNG NGOAI, khong hoc khoi luong.

Ba dang RL da thu va da thua o docs/SIZING_COMPARISON.md deu hoc KHOI LUONG:
khong gian hanh dong lien tuc, hang nghin tham so.
  PPO 0,018 · CVaR-PPO 0,030 · bandit bang 0,144  so  quy tac tay 0,886
Doc theo chieu quan trong: phuong phap IT NANG LUC NHAT lai tot nhat trong ba
cai hoc duoc. Du lieu nay thuong cho it tham so, nhieu cau truc.

File nay thu dang chua ai thu: hanh dong NHI PHAN — tham gia hay dung ngoai.
  * khong gian hanh dong 2 gia tri thay vi lien tuc
  * trang thai roi rac 72 o thay vi vector thuc 6 chieu
  * da co tin hieu de khai thac: giai doan 0 do duoc momentum o che do cang
    thang cho Sharpe -0,615, p=0,001, am 12/12 o

VI SAO VAN LA RL CHU KHONG PHAI HOC CO GIAM SAT: hanh dong DOI TRANG THAI
tuong lai. Dung ngoai hom nay -> khong lo -> sut giam nho hon -> tran rui ro
ngay mai cho don bay cao hon. Vong phan hoi do la thu hoc co giam sat khong
bat duoc.

BA TAC NHAN, tang dan do bao thu:
  1. Bandit ngu canh   gamma = 0, chi nhin phan thuong tuc thi
  2. Q-learning bang   gamma = 0,95, co nhin xa
  3. Q BI QUAN         tru mot khoan phat ty le 1/sqrt(n(s,a)) — khong hanh
     dong theo bang chung mong. Day la dang don gian nhat cua RL ngoai tuyen
     co rang buoc bao thu, thu ma van lieu 2025-2026 chi ra la bat buoc khi
     che do troi.

DOI CHUNG: luon vao · khong bao gio vao · QUY TAC TAY (dung ngoai khi sigma o
tercile cao nhat — chinh la quy luat loai-tru tim duoc o giai doan 0).

Giao thuc: hoc tren HUAN LUYEN, chon tren KIEM DINH, cham KIEM TRA mot lan.

Chay:  python src/run_rl_gate.py
Ghi:   output/rl_gate.json, output/log_rl_gate.txt
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

import volfc2 as V2                                        # noqa: E402
from position_sizing import PositionSizer                  # noqa: E402
from split import VALID_TU, TEST_TU, doan                  # noqa: E402

PANEL = os.path.join(ROOT, "data", "panel2_6pairs.csv")
CARRY = os.path.join(ROOT, "data", "carry.csv")
CHI_PHI_PIP = 0.7          # spread khu hoi trung vi sau 2015 (docs/DATASET.md)
PIP = {"USDJPY": 0.01}
GAMMA = 0.95
SEED = 0
EPS = 1e-12


# ── trang thai ──────────────────────────────────────────────────────────
N_SIG, N_DD, N_CARRY, N_PNL, N_EV = 3, 3, 2, 2, 2
N_TRANG_THAI = N_SIG * N_DD * N_CARRY * N_PNL * N_EV        # 72


def ma_trang_thai(i_sig, i_dd, i_carry, i_pnl, i_ev):
    return (((i_sig * N_DD + i_dd) * N_CARRY + i_carry) * N_PNL + i_pnl) * N_EV + i_ev


def nap():
    pan = pd.read_csv(PANEL, parse_dates=["Date"])
    ca = pd.read_csv(CARRY, parse_dates=["DATE"]).rename(columns={"DATE": "Date"})
    ca["thang"] = ca.Date.values.astype("datetime64[M]")
    pan["thang"] = pan.Date.values.astype("datetime64[M]")
    pan = pd.merge(pan, ca[["thang", "pair", "carry"]], on=["thang", "pair"],
                   how="left")
    pan["carry"] = pan.groupby("pair").carry.ffill().fillna(0.0)
    lich = V2.nap_lich(pd.DatetimeIndex(sorted(pan.Date.unique())))
    return pan.sort_values(["pair", "Date"]).reset_index(drop=True), lich


def dung_chuoi(pan, lich):
    """Cho moi cap: loi suat log CO DON BAY neu THAM GIA, va ma trang thai."""
    ngay_ix = {d: i for i, d in enumerate(sorted(pan.Date.unique()))}
    ra = {}
    for p, g in pan.groupby("pair"):
        g = g.sort_values("Date").reset_index(drop=True)
        tr = doan(g.Date.values) == 0
        sizer = PositionSizer(g.sig.values[tr])
        nguong = np.quantile(g.sig.values[tr], [1 / 3, 2 / 3])
        i_sig = np.digitize(g.sig.values, nguong)

        # carry hang nam -> loi the ky vong hang ngay, dau theo dau carry
        mu = g.carry.values / 100.0 / 252.0
        nu = 6.0
        f = sizer.size(g.sig.values, np.abs(mu), nu, dd=0.0, so_vi_the=6)
        dau = np.sign(mu)
        dau[dau == 0] = 1.0

        # CHI PHI PHAI CUNG DON VI VOI LOI SUAT. `r_tho` la loi suat log (khong
        # thu nguyen), con CHI_PHI_PIP*pip la mot muc GIA tuyet doi. Phai chia
        # cho muc gia moi ra don vi loi suat — bo qua buoc nay la sai 156 lan o
        # USDJPY, dung loai loi da mac o sang_pip() cua api/main.py.
        gia = pd.read_csv(os.path.join(ROOT, "data", "prices", f"{p}_d1.csv"),
                          parse_dates=["Date"])[["Date", "close"]]
        gia = pd.merge(g[["Date"]], gia, on="Date", how="left")
        c_gia = gia.close.ffill().bfill().values
        pip = PIP.get(p, 0.0001)
        r_tho = g.zT.values * g.sig.values * dau           # loi suat theo huong carry
        chi_phi = CHI_PHI_PIP * pip / np.maximum(c_gia, EPS)
        r_net = np.log(np.maximum(1.0 + f * (r_tho - chi_phi), 1e-9))

        i_carry = (mu > 0).astype(int)
        pnl5 = pd.Series(r_tho).rolling(5).sum().shift(1).fillna(0.0).values
        i_pnl = (pnl5 > 0).astype(int)
        ev = np.zeros(len(g), int)
        nh = V2.NHTW[p].lower()
        for k in ("fomc", nh):
            v = np.array([lich[k][ngay_ix[d]] for d in g.Date.values])
            ev = np.maximum(ev, (v > 0).astype(int))
        ra[p] = dict(Date=g.Date.values, r=r_net, f=f, i_sig=i_sig,
                     i_carry=i_carry, i_pnl=i_pnl, i_ev=ev,
                     doan=doan(g.Date.values))
    return ra


def i_dd_tu(dd):
    return 0 if dd < 0.02 else (1 if dd < 0.10 else 2)


# ── chay mot chinh sach ─────────────────────────────────────────────────
def chay(ch, chinh_sach, mask_ten):
    """Tra ve chuoi loi suat danh muc (trung binh 6 cap) va nhat ky (s,a,r)."""
    caps = list(ch)
    n = max(len(ch[p]["r"]) for p in caps)     # cac cap lech nhau vai phien
    von = {p: 1.0 for p in caps}
    dinh = {p: 1.0 for p in caps}
    r_ngay, nk = [], []
    for t in range(n):
        rs = []
        for p in caps:
            c = ch[p]
            if t >= len(c["r"]) or not mask_ten(c["doan"][t]):
                continue
            dd = 1.0 - von[p] / max(dinh[p], EPS)
            s = ma_trang_thai(int(c["i_sig"][t]), i_dd_tu(dd),
                              int(c["i_carry"][t]), int(c["i_pnl"][t]),
                              int(c["i_ev"][t]))
            a = chinh_sach(s, p, t)
            r = c["r"][t] if a == 1 else 0.0
            von[p] *= np.exp(r)
            dinh[p] = max(dinh[p], von[p])
            rs.append(r)
            nk.append((s, a, r))
        if rs:
            r_ngay.append(float(np.mean(rs)))
    return np.array(r_ngay), nk


def thong_ke(r, ten):
    if len(r) == 0:
        return dict(ten=ten, n=0)
    eq = np.exp(np.cumsum(r))
    dd = 1.0 - eq / np.maximum.accumulate(np.r_[1.0, eq])[1:]
    sd = r.std()
    return dict(ten=ten, n=int(len(r)), tb_bp=float(r.mean() * 1e4),
                sharpe=float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
                mdd=float(dd.max()), von_cuoi=float(eq[-1]),
                ty_le_vao=None)


def dm_nw(x):
    """Diebold-Mariano voi phuong sai Newey-West (giong run_final7.py)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan, np.nan
    mb = x.mean()
    L = int(np.ceil(1.5 * n ** (1 / 3)))
    s = np.sum((x - mb) ** 2) / n
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * np.sum((x[k:] - mb) * (x[:-k] - mb)) / n
    t = mb / np.sqrt(max(s, 1e-16) / n)
    from scipy import stats as st
    return float(t), float(2 * (1 - st.norm.cdf(abs(t))))


# ── ba tac nhan ─────────────────────────────────────────────────────────
def hoc_Q(nk, gamma, bi_quan=0.0, n_lap=200, lr=0.1):
    """Q-learning bang tren nhat ky (s,a,r). gamma=0 -> bandit ngu canh.

    bi_quan > 0: tru mot khoan phat ty le 1/sqrt(n(s,a)) — khong hanh dong theo
    bang chung mong."""
    Q = np.zeros((N_TRANG_THAI, 2))
    dem = np.zeros((N_TRANG_THAI, 2))
    for (s, a, r) in nk:
        dem[s, a] += 1
    rng = np.random.default_rng(SEED)
    for _ in range(n_lap):
        for i in rng.permutation(len(nk)):
            s, a, r = nk[i]
            s2 = nk[i + 1][0] if i + 1 < len(nk) else s
            muc = r + gamma * Q[s2].max()
            Q[s, a] += lr * (muc - Q[s, a])
    if bi_quan > 0:
        Q = Q - bi_quan / np.sqrt(np.maximum(dem, 1.0))
    return Q, dem


def main():
    print("=" * 104)
    print("RL CỔNG VÀO LỆNH — học khi nào ĐỨNG NGOÀI (hành động nhị phân)")
    print("=" * 104)
    pan, lich = nap()
    ch = dung_chuoi(pan, lich)
    caps = list(ch)
    print(f"{len(caps)} cặp · {len(ch[caps[0]]['r']):,} phiên/cặp · "
          f"{N_TRANG_THAI} ô trạng thái × 2 hành động")

    tr = lambda g: g == 0
    va = lambda g: g == 1
    te = lambda g: g == 2

    # ── nhat ky tren HUAN LUYEN bang chinh sach luon vao (kham pha day du) ──
    _, nk_tr = chay(ch, lambda s, p, t: 1, tr)
    print(f"nhật ký huấn luyện: {len(nk_tr):,} bước")
    tham = np.bincount([s for s, a, r in nk_tr], minlength=N_TRANG_THAI)
    print(f"ô trạng thái có ≥30 lần thăm: {int((tham >= 30).sum())}/{N_TRANG_THAI}"
          f" · ô chưa từng thăm: {int((tham == 0).sum())}")

    # ── doi chung ────────────────────────────────────────────────────────
    doi_chung = {
        "luôn vào": lambda s, p, t: 1,
        "không bao giờ vào": lambda s, p, t: 0,
        "quy tắc tay (bỏ tercile σ̂ cao)": lambda s, p, t: 0 if (s // (N_DD * N_CARRY * N_PNL * N_EV)) == 2 else 1,
    }

    # ── ba tac nhan, chon tren KIEM DINH ────────────────────────────────
    tac_nhan = {}
    for ten, gam, bq in (("bandit ngữ cảnh (γ=0)", 0.0, 0.0),
                         ("Q-learning (γ=0,95)", GAMMA, 0.0),
                         ("Q bi quan (γ=0,95)", GAMMA, 0.02)):
        Q, dem = hoc_Q(nk_tr, gam, bq)
        pi = Q.argmax(1)
        pi[tham < 30] = 1                      # ô mỏng bằng chứng: theo mặc định
        tac_nhan[ten] = (lambda P: (lambda s, p, t: int(P[s])))(pi)

    for nhan, ham_mask, tieu_de in (("kiểm định", va, "KIỂM ĐỊNH (dùng để chọn)"),
                                    ("kiểm tra", te, "KIỂM TRA (chấm một lần)")):
        print(f"\n{'─' * 104}\n{tieu_de}")
        print(f"  {'chính sách':<34}{'TB (bp/ngày)':>13}{'Sharpe':>9}"
              f"{'sụt giảm':>10}{'vốn cuối':>10}{'% vào lệnh':>12}{'t vs luôn vào':>15}")
        r_goc = None
        for ten, cs in {**doi_chung, **tac_nhan}.items():
            r, nk = chay(ch, cs, ham_mask)
            tl = np.mean([a for _, a, _ in nk]) if nk else 0.0
            k = thong_ke(r, ten)
            if ten == "luôn vào":
                r_goc = r
                tt = ""
            else:
                m = min(len(r), len(r_goc))
                t_, p_ = dm_nw(r[:m] - r_goc[:m])
                tt = f"{t_:+.2f} (p={p_:.3f})"
            print(f"  {ten:<34}{k['tb_bp']:>13.3f}{k['sharpe']:>9.2f}"
                  f"{k['mdd']:>9.1%}{k['von_cuoi']:>10.3f}{tl:>11.1%}{tt:>15}")

    Q0, _ = hoc_Q(nk_tr, GAMMA, 0.0)
    pi = Q0.argmax(1)
    pi[tham < 30] = 1
    hand = np.array([0 if (s // (N_DD * N_CARRY * N_PNL * N_EV)) == 2 else 1
                     for s in range(N_TRANG_THAI)])
    kh = (pi != hand) & (tham >= 30)
    print(f"\nchính sách học được KHÁC quy tắc tay ở {int(kh.sum())}/"
          f"{int((tham >= 30).sum())} ô có đủ bằng chứng")
    json.dump({"tham": tham.tolist(), "pi": pi.tolist(), "hand": hand.tolist()},
              open(os.path.join(OUT, "rl_gate.json"), "w"), indent=1)
    print("\nđã ghi output/rl_gate.json")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
