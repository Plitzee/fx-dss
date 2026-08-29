"""
THI NGHIEM 6 — BACKTEST THAT + DEFLATED SHARPE + PBO.

Tin hieu: momentum chuoi thoi gian (dau cua loi suat L phien gan nhat) — mot chien luoc
CO tai lieu trong FX, khong phai tu nghi ra. 4 do dai x 6 quy tac dinh co = 24 cau hinh.
Muc dich KHONG phai tim chien luoc sinh loi, ma la chay cho duoc hai chi so chong
overfitting ma giai doan 1 xep vao loai bat buoc, tren du lieu that.
"""
import numpy as np, pandas as pd, sys, json
from itertools import combinations
from scipy import stats
sys.path.insert(0, "/tmp/fx/src")
from fxdata import PAIRS
from sizing import f_fixed_risk, f_kelly, f_cvar, f_ruin_cap

LOOKBACKS = [20, 60, 120, 250]
COST_PIPS = 1.0
GAMMA = 0.5772156649015329          # hang so Euler-Mascheroni


def build_pnl():
    """Sinh chuoi P&L ngay cho tung (do dai momentum x quy tac dinh co), gop 12 cap."""
    frames = []
    for p in PAIRS:
        fc = pd.read_csv(f"/tmp/fx/fc_{p}.csv", parse_dates=["Date"]).dropna(subset=["HAR-RV"])
        fc = fc.reset_index(drop=True)
        fc["sig"] = np.sqrt(fc["HAR-RV"].clip(lower=1e-12))
        fc["pair"] = p
        frames.append(fc[["Date", "pair", "r", "sig"]])
    D = pd.concat(frames, ignore_index=True)
    es_z = -2.6                                    # ES 97,5% chuan hoa xap xi cua t(6)
    out = {}
    for L in LOOKBACKS:
        for p in PAIRS:
            s = D[D.pair == p].reset_index(drop=True)
            r = s.r.values
            sig = s.sig.values
            mom = pd.Series(r).rolling(L).sum().shift(1).values      # tin hieu, tre 1 phien
            mu_hat = pd.Series(r).rolling(L).mean().shift(1).values  # uoc luong loi the
            sgn = np.sign(mom)
            rules = {
                "Cố định 2%":        f_fixed_risk(sig, 0.02),
                "Kelly đầy đủ":      f_kelly(np.abs(mu_hat), sig, 1.0),
                "Kelly 1/2":         f_kelly(np.abs(mu_hat), sig, 0.5),
                "Kelly 1/4":         f_kelly(np.abs(mu_hat), sig, 0.25),
                "Ràng buộc CVaR":    f_cvar(sig, es_z, 0.02),
                "Kelly + trần rủi ro": np.minimum(f_kelly(np.abs(mu_hat), sig, 1.0),
                                                  f_ruin_cap(sig, 250, 0.01, 6.0)),
            }
            for name, f in rules.items():
                f = np.clip(np.nan_to_num(f, nan=0.0), 0, 30.0)
                pos = sgn * f
                turn = np.abs(np.diff(np.r_[0.0, pos]))
                pnl = pos * r - turn * (COST_PIPS * 1e-4 / 1.1)
                key = (L, name)
                df = pd.DataFrame({"Date": s.Date.values, "pnl": pnl})
                out.setdefault(key, []).append(df)
    # gop deu 12 cap
    series = {}
    for key, lst in out.items():
        m = pd.concat(lst).groupby("Date").pnl.mean()
        series[key] = m
    R = pd.DataFrame(series).dropna()
    R.columns = pd.MultiIndex.from_tuples(R.columns, names=["lookback", "rule"])
    return R


def sharpe(x, ann=250):
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return 0.0 if sd == 0 else float(x.mean() / sd * np.sqrt(ann))


def psr(sr_ann, x, sr_star_ann=0.0, ann=250):
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado)."""
    T = len(x)
    sr = sr_ann / np.sqrt(ann)                       # ve don vi moi quan sat
    sr0 = sr_star_ann / np.sqrt(ann)
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)
    den = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / den))


def expected_max_sr(var_sr_ann, N):
    """E[max SR] duoi gia thuyet khong co ky nang (Bailey & Lopez de Prado)."""
    s = np.sqrt(var_sr_ann)
    return float(s * ((1 - GAMMA) * stats.norm.ppf(1 - 1 / N)
                      + GAMMA * stats.norm.ppf(1 - 1 / (N * np.e))))


def pbo_cscv(R, S=16):
    """Probability of Backtest Overfitting qua Combinatorially Symmetric Cross-Validation."""
    X = R.values
    T, N = X.shape
    T2 = (T // S) * S
    X = X[:T2]
    blocks = X.reshape(S, T2 // S, N)
    s1 = blocks.sum(axis=1)                       # (S, N)
    s2 = (blocks ** 2).sum(axis=1)
    nb = T2 // S
    lams = []
    for comb in combinations(range(S), S // 2):
        m = np.zeros(S, bool); m[list(comb)] = True
        for sel in (m, ~m):
            pass
        for IS, OOS in ((m, ~m),):
            n_is = nb * IS.sum(); n_oos = nb * (~IS).sum()
            mu_is = s1[IS].sum(0) / n_is
            var_is = s2[IS].sum(0) / n_is - mu_is ** 2
            sr_is = mu_is / np.sqrt(np.maximum(var_is, 1e-18))
            mu_oo = s1[~IS].sum(0) / n_oos
            var_oo = s2[~IS].sum(0) / n_oos - mu_oo ** 2
            sr_oo = mu_oo / np.sqrt(np.maximum(var_oo, 1e-18))
            n_star = int(np.argmax(sr_is))
            rank = float((sr_oo <= sr_oo[n_star]).sum())      # 1..N
            w = rank / (N + 1)
            lams.append(np.log(w / (1 - w)))
    lams = np.array(lams)
    return float((lams <= 0).mean()), lams


if __name__ == "__main__":
    R = build_pnl()
    T, N = R.shape
    print("=" * 108)
    print(f"A. BACKTEST 24 CAU HINH — {T} phien, gop deu 12 cong cu, tru chi phi 1 pip khu hoi")
    print("=" * 108)
    srs = {c: sharpe(R[c].values) for c in R.columns}
    tab = pd.Series(srs).unstack(level=1)
    print(f"{'Momentum':>10}" + "".join(f"{c:>21}" for c in tab.columns))
    print("-" * 108)
    for L in tab.index:
        print(f"{f'{L} phiên':>10}" + "".join(f"{tab.loc[L, c]:>21.3f}" for c in tab.columns))
    print("-" * 108)
    best = max(srs, key=srs.get)
    print(f"Sharpe nam. Cau hinh tot nhat: momentum {best[0]} phien + {best[1]}  ->  SR = {srs[best]:.3f}")

    print("\n" + "=" * 108)
    print("B. SHARPE NAY CO THAT KHONG? — Probabilistic va Deflated Sharpe Ratio")
    print("=" * 108)
    x = R[best].values
    sr_ann = srs[best]
    var_sr = np.var(list(srs.values()), ddof=1)
    sr0 = expected_max_sr(var_sr, N)
    p_psr = psr(sr_ann, x, 0.0)
    p_dsr = psr(sr_ann, x, sr0)
    print(f"  Sharpe quan sat duoc                        : {sr_ann:>8.3f}")
    print(f"  Do lech (skew) cua chuoi P&L                : {stats.skew(x):>8.3f}")
    print(f"  Do nhon (kurtosis)                          : {stats.kurtosis(x, fisher=False):>8.3f}")
    print(f"  So cau hinh da thu  N                       : {N:>8d}")
    print(f"  Do lech chuan cua Sharpe qua cac cau hinh   : {np.sqrt(var_sr):>8.3f}")
    print(f"  Nguong ky vong E[max SR] duoi H0 khong ky nang: {sr0:>6.3f}")
    print("  " + "-" * 68)
    print(f"  PSR  (so voi nguong 0)                      : {p_psr:>8.1%}")
    print(f"  DSR  (so voi nguong E[max SR])              : {p_dsr:>8.1%}")
    print("  " + "-" * 68)
    print("  DSR la xac suat Sharpe that > 0 SAU KHI tinh den viec da thu N cau hinh.")
    print(f"  Ket luan: {'KHONG dat' if p_dsr < 0.95 else 'dat'} nguong 95% thuong duoc dung.")

    print("\n" + "=" * 108)
    print("C. QUY TRINH CHON MO HINH CO TONG QUAT HOA KHONG? — PBO qua CSCV")
    print("=" * 108)
    pbo, lams = pbo_cscv(R, S=16)
    print(f"  So to hop chia mau (S=16, chon 8)           : {len(lams):>8,d}")
    print(f"  Xac suat backtest overfitting (PBO)         : {pbo:>8.1%}")
    print(f"  Trung vi logit thu hang ngoai mau           : {np.median(lams):>8.3f}")
    print("  " + "-" * 68)
    print("  PBO = xac suat mo hinh tot nhat TRONG mau lai nam duoi trung vi NGOAI mau.")
    print("  PBO ~ 50% nghia la quy trinh chon mo hinh chi la tung dong xu.")
    print(f"  Ket luan: {'quy trinh chon KHONG tong quat hoa' if pbo > 0.4 else 'quy trinh chon co gia tri'}.")

    print("\n" + "=" * 108)
    print("D. SO SANH — Sharpe tho vs Sharpe da khu thien lech, tung quy tac")
    print("=" * 108)
    print(f"{'Quy tac':<22}{'SR tot nhat':>13}{'SR trung binh':>15}{'PSR':>9}{'DSR':>9}{'ket luan':>28}")
    print("-" * 108)
    for rule in tab.columns:
        cols = [c for c in R.columns if c[1] == rule]
        s_ = {c: srs[c] for c in cols}
        b_ = max(s_, key=s_.get)
        v_ = np.var(list(s_.values()), ddof=1)
        s0 = expected_max_sr(max(v_, 1e-9), len(cols))
        pp = psr(s_[b_], R[b_].values, 0.0)
        pd_ = psr(s_[b_], R[b_].values, s0)
        verdict = "khong phan biet duoc voi nhieu" if pd_ < 0.95 else "song sot"
        print(f"{rule:<22}{s_[b_]:>13.3f}{np.mean(list(s_.values())):>15.3f}"
              f"{pp:>9.1%}{pd_:>9.1%}{verdict:>28}")
    print("-" * 108)

    json.dump({"sharpe": {f"{k[0]}|{k[1]}": v for k, v in srs.items()},
               "best": f"{best[0]}|{best[1]}", "sr_best": sr_ann,
               "sr0": sr0, "psr": p_psr, "dsr": p_dsr, "pbo": pbo, "T": int(T), "N": int(N)},
              open("/tmp/fx/exp6.json", "w"), indent=1)
    R.to_csv("/tmp/fx/exp6_pnl.csv")
    print("\nDa luu exp6.json va exp6_pnl.csv")
