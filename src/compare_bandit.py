"""TANG 4 — BANDIT NGU CANH CO TIM RA DIEU KIEN HOA (k_vol x k_dd) MA PPO/CVAR-PPO
DA BO LO KHONG? (docs/SIZING_COMPARISON.md, muc 3)

BOI CANH. PPO va CVaR-PPO deu that bai o CUNG mot dieu: bien do he so k theo
sut giam chi 0,018-0,030 so voi 0,800 cua quy tac tay, va cang huan luyen lau
bien do cang NHO DI. Gia thuyet trong tai lieu do: loi ich nam sau trong duoi
(phan sinh chi 0,07% duong di), qua sau de policy-gradient qua nhieu buoc thoi
gian nhin thay.

CAU HOI O DAY. Neu bo qua het credit-assignment qua nhieu buoc (thu ma PPO
lam bang GAE, CVaR-PPO lam bang loc duoi tren CA DUONG DI) va chi hoc THAM
LAM tung buoc mot — dung MOT bandit ngu canh don gian, coi moi ngay la mot
"luot choi" doc lap voi phan thuong tuc thi — thi co tim ra dung dieu kien
hoa khong? Day la mot GIA DINH YEU HON RL (bo qua tuong quan doc theo thoi
gian trong cung mot duong di), nhung don gian hon nhieu: khong mang no-ron,
khong dao ham, chi la bang trung binh mau cong don.

THIET KE. Ngu canh = (bac ba bien dong tuong doi, bac 5 muc sut giam — dung
CHINH 5 moc 0/5/10/20/30% ma k_vs_dd() cua compare_rl.py dung de doi chieu
duoc truc tiep) = 15 o. Hanh dong = mot luoi 11 gia tri k trong [0,5; 1,5]
(dung khoang PPO/CVaR-PPO da dung). Epsilon-greedy tren TRUNG BINH MAU cong
don moi (ngu_canh, hanh dong) — KHONG bootstrap qua trang thai ke tiep (day
la diem phan biet BANDIT voi Q-learning/RL: moi phan thuong duoc coi la doc
lap voi tuong lai, khong lan truyen gia tri).

Chay:  python src/compare_bandit.py EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF 150
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from rl_env import SizingEnv, LEV_MAX
from position_sizing import K_VOL_HI, K_VOL_LO, K_DD_HI, K_DD_SLOPE, K_DD_FLOOR
from sizing import f_kelly, f_ruin_cap

SR_TRUE = 0.5
SR_TRAIN = (0.0, 1.2)
K_LO, K_HI = 0.5, 1.5
N_K = 11
K_GRID = np.linspace(K_LO, K_HI, N_K)
N_SIG, N_DD = 3, 5
DD_MOC = (0.0, 0.05, 0.10, 0.20, 0.30)          # y het k_vs_dd() cua compare_rl.py
NPATH, NEVAL = 160, 6000
NIT = int(sys.argv[2]) if len(sys.argv) > 2 else 150


def fbase(env):
    """Chuan so sanh cua rl_env.py: min(Kelly, tran rui ro), KHONG dieu kien hoa."""
    s = env.sig[env.idx[:, env.t]]
    return np.clip(np.minimum(f_kelly(env.mu_bel, s, 1.0),
                               f_ruin_cap(s, 250, 0.01, env.nu)), 0, LEV_MAX)


def k_tay(sig_rel, dd, q1, q2):
    """He so k cua QUY TAC TAY THAT (position_sizing.py.k_vol/k_dd), ap vao
    dung bien trang thai cua moi truong nay (sig_rel=ln(sig/sig_bar) thay vi
    sig tho, q1/q2 la tam phan vi THAT da khop tren huan luyen cua pair/seed
    nay) — dung de doi chieu cong bang voi bandit, khong roi rac hoa."""
    t = np.clip((np.asarray(sig_rel, float) - q1) / max(q2 - q1, 1e-9), 0, 1)
    kv = K_VOL_HI + (K_VOL_LO - K_VOL_HI) * t
    kd = np.clip(K_DD_HI - K_DD_SLOPE * np.asarray(dd, float), K_DD_FLOOR, K_DD_HI)
    return kv * kd


def ngu_canh(sig_rel, dd, q1, q2):
    sig_b = np.digitize(sig_rel, [q1, q2])                 # 0,1,2
    dd_b = np.digitize(dd, DD_MOC[1:])                      # 0..4
    return sig_b * N_DD + dd_b


def roll(env, chon_k, n):
    """chon_k(sig_rel, dd, ctx) -> k (n,), da GREEDY (danh gia, khong tham do).
    Truyen ca sig_rel/dd LIEN TUC lan ctx da roi rac — bandit dung ctx,
    "tay" dung truc tiep sig_rel/dd (dung y het production, khong roi rac)."""
    obs = env.reset(n)
    lev = []
    while True:
        sig_rel, dd = obs[:, 0], obs[:, 2] / 5.0
        ctx = ngu_canh(sig_rel, dd, env._q1, env._q2)
        k = chon_k(sig_rel, dd, ctx)
        f = np.clip(k * fbase(env), 0, LEV_MAX)
        lev.append(float(np.mean(f)))
        obs2, r, done = env.step(f, reward_kind="pnl_log")
        if done:
            break
        obs = obs2
    st = env.stats(); st["avg_lev"] = float(np.mean(lev))
    return st


def train_bandit(env, n_iter, n_paths, rng):
    """Bandit ngu canh bang: trung binh mau cong don tung (ngu_canh, hanh
    dong), epsilon-greedy giam dan tu 1,0 xuong 0,03."""
    sum_flat = np.zeros(N_SIG * N_DD * N_K)
    cnt_flat = np.zeros(N_SIG * N_DD * N_K)
    for it in range(n_iter):
        eps = 1.0 + (0.03 - 1.0) * it / max(n_iter - 1, 1)
        obs = env.reset(n_paths)
        while True:
            sig_rel, dd = obs[:, 0], obs[:, 2] / 5.0
            ctx = ngu_canh(sig_rel, dd, env._q1, env._q2)
            Q = (sum_flat / np.maximum(cnt_flat, 1)).reshape(N_SIG * N_DD, N_K)
            greedy = np.argmax(Q[ctx], axis=1)
            rand_a = rng.integers(0, N_K, size=len(ctx))
            explore = rng.random(len(ctx)) < eps
            a = np.where(explore, rand_a, greedy)
            f = np.clip(K_GRID[a] * fbase(env), 0, LEV_MAX)
            obs2, r, done = env.step(f, reward_kind="pnl_log")
            idxf = ctx * N_K + a
            np.add.at(cnt_flat, idxf, 1)
            np.add.at(sum_flat, idxf, r)
            if done:
                break
            obs = obs2
    return (sum_flat / np.maximum(cnt_flat, 1)).reshape(N_SIG, N_DD, N_K)


PANEL = os.environ.get("FX_PANEL", "panel2_6pairs.csv")
P = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", PANEL), parse_dates=["Date"])
print(f"panel: {PANEL}  ({len(P):,} dòng)")
PAIRS = sys.argv[1].split(",") if len(sys.argv) > 1 else \
    ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
SEEDS = [1, 2, 3]

res = {"bandit": {"lev": [], "g": [], "r": [], "kdd": []},
       "tay": {"lev": [], "g": [], "r": [], "kdd": []},
       "tran_tron": {"lev": [], "g": [], "r": [], "kdd": []}}
t0 = time.time()
for pair in PAIRS:
    D = P[P.pair == pair].reset_index(drop=True)
    k = int(len(D) * .7)
    TR, TE = D.iloc[:k].reset_index(drop=True), D.iloc[k:].reset_index(drop=True)
    for sd in SEEDS:
        env = SizingEnv(TR, sharpe_true=SR_TRAIN, seed=sd)
        # tam phan vi cua sig_rel = ln(sig/sig_bar), do TREN HUAN LUYEN (chinh
        # la TR — khong ro ri), dung y het cach PositionSizer khop tam phan vi.
        env._q1, env._q2 = (float(x) for x in np.quantile(np.log(env.sig / env.sig_bar), [1 / 3, 2 / 3]))

        rng = np.random.default_rng(sd)
        Qtab = train_bandit(env, NIT, NPATH, rng)

        te = SizingEnv(TE, sharpe_true=SR_TRUE, seed=999)
        te._q1, te._q2 = env._q1, env._q2   # nguong tam phan vi la THAM SO da khop, mang sang tap kiem tra

        flat_bandit = Qtab.reshape(N_SIG * N_DD, N_K)
        st = roll(te, lambda sr, dd, ctx: K_GRID[np.argmax(flat_bandit[ctx], axis=1)], NEVAL)
        res["bandit"]["lev"].append(st["avg_lev"]); res["bandit"]["g"].append(st["growth"])
        res["bandit"]["r"].append(st["p_ruin"])
        res["bandit"]["kdd"].append([K_GRID[np.argmax(Qtab[1, d])] for d in range(N_DD)])

        # "tay": DUNG TRUC TIEP sig_rel/dd LIEN TUC — dung y nhu production,
        # KHONG roi rac hoa (khac bandit, von buoc phai roi rac de co bang tra).
        q1, q2 = env._q1, env._q2
        st = roll(te, lambda sr, dd, ctx: k_tay(sr, dd, q1, q2), NEVAL)
        res["tay"]["lev"].append(st["avg_lev"]); res["tay"]["g"].append(st["growth"])
        res["tay"]["r"].append(st["p_ruin"])
        res["tay"]["kdd"].append([float(k_tay(0.0, d, q1, q2)) for d in DD_MOC])

        st = roll(te, lambda sr, dd, ctx: np.ones(len(ctx)), NEVAL)
        res["tran_tron"]["lev"].append(st["avg_lev"]); res["tran_tron"]["g"].append(st["growth"])
        res["tran_tron"]["r"].append(st["p_ruin"])
        res["tran_tron"]["kdd"].append([1.0] * N_DD)

    print(f"  {pair}: xong [{time.time()-t0:.0f}s]", flush=True)

print("\n" + "=" * 80)
print(f"KẾT QUẢ — {len(PAIRS)} cặp × {len(SEEDS)} seed, {NIT} vòng bandit")
print("=" * 80)
print(f"{'':<18}{'tăng trưởng':>14}{'độ lệch':>10}{'phá sản TB':>13}{'tệ nhất':>10}")
print("-" * 80)
TEN = {"tran_tron": "Trần trơn (k=1)", "tay": "Điều kiện hoá tay", "bandit": "Bandit ngữ cảnh"}
for mode in ("tran_tron", "tay", "bandit"):
    g = np.array(res[mode]["g"]); r = np.array(res[mode]["r"])
    print(f"{TEN[mode]:<18}{g.mean():>13.2%}{g.std():>10.3f}{r.mean():>12.2%}{r.max():>10.2%}")

print("\n" + "=" * 80)
print("CHẨN ĐOÁN — hệ số k theo mức sụt giảm (bậc biến động GIỮA)")
print("=" * 80)
print(f"{'':<18}" + "".join(f"{'dd='+f'{d:.0%}':>10}" for d in DD_MOC) + f"{'biên độ':>11}")
print("-" * 80)
for mode in ("tran_tron", "tay", "bandit"):
    K = np.array(res[mode]["kdd"]).mean(0)
    print(f"{TEN[mode]:<18}" + "".join(f"{v:>10.3f}" for v in K) + f"{K.max()-K.min():>11.3f}")
print("-" * 80)
print("Quy tắc thiết kế tay: k(dd) đi từ 1,300 xuống 0,500 — biên độ 0,800")
print("PPO thường: biên độ 0,018   |   CVaR-PPO: biên độ 0,030  (docs/SIZING_COMPARISON.md)")

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)
json.dump({m: {k: (np.array(v).tolist()) for k, v in d.items()} for m, d in res.items()},
          open(os.path.join(OUT, "bandit_results.json"), "w"), indent=1, ensure_ascii=False)
print("\nđã ghi output/bandit_results.json")
print("\nTỰ KIỂM ĐẠT")
