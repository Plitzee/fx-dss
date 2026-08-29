"""
THI NGHIEM 7 — HOC TANG CUONG CHO TANG DINH CO VI THE.

Bai hoc lon nhat cua giai doan nay duoc phat hien BANG CACH LAM SAI TRUOC:
mot lan chay mot seed cho ket qua rat dep va rat sai. Khi chay lai voi nhieu seed,
khoi luong ma agent hoc duoc bien thien tren duoi 90%. Vi vay moi con so o day
deu la TRUNG VI QUA NHIEU SEED, kem khoang bien thien.

Bon cau hoi:
  A. Do on dinh theo seed — mot ket qua RL mot-seed co dang tin khong?
  B. Bac thuan nhat cua phan thuong theo khoi luong f co quyet dinh gi khong?
  C. Chinh sach RL (hop nhieu seed) co danh bai quy tac tinh min(Kelly, tran) khong?
  D. Khi nha dau tu uoc luong sai loi the, RL sup nhu Kelly hay phang nhu quy tac co tran?
"""
import numpy as np, pandas as pd, sys, json, time
sys.path.insert(0, "/tmp/fx/src")
from rl_env import SizingEnv, LEV_MAX
from rl_agent import GaussianPolicy, train
from sizing import f_kelly, f_ruin_cap, f_fixed_risk

P = pd.read_csv("/tmp/fx/exp3_panel.csv", parse_dates=["Date"])
EU = P[P.pair == "EURUSD"].reset_index(drop=True)
SR_TRUE = 0.5
SR_TRAIN = (0.0, 1.2)      # loi the NGAU NHIEN khi huan luyen (xem ghi chu ve cong tuyen)
NIT, NPATH, NEVAL = 250, 192, 12000
SEEDS = [1, 2, 3, 4]
t0 = time.time()

STATIC = {
    "Cố định 2%":          lambda s, mu, nu: f_fixed_risk(s, 0.02),
    "Kelly đầy đủ":        lambda s, mu, nu: f_kelly(mu, s, 1.0),
    "Kelly 1/2":           lambda s, mu, nu: f_kelly(mu, s, 0.5),
    "Kelly + trần rủi ro": lambda s, mu, nu: np.minimum(f_kelly(mu, s, 1.0),
                                                        f_ruin_cap(s, 250, 0.01, nu)),
}
REWARDS = [
    ("P&L số học (bậc 1 theo f)", "pnl_arith", 0.0),
    ("P&L log (lõm theo f)", "pnl_log", 0.0),
    ("Differential Sharpe (bất biến thang)", "dsr", 0.0),
    ("P&L log + phạt sụt giảm", "pnl_log", 1.0),
]


def run_policy(env, n_paths, pols=None, rule=None, belief_mult=1.0):
    """pols: mot chinh sach, hoac DANH SACH chinh sach -> hop bang trung binh hanh dong."""
    obs = env.reset(n_paths)
    if belief_mult != 1.0:
        env.mu_bel = env.mu_bel * belief_mult
        obs = env._obs()[0]
    lev, done = [], False
    while not done:
        if pols is not None:
            ps = pols if isinstance(pols, (list, tuple)) else [pols]
            f = np.mean([p.act(obs, explore=False)[1] for p in ps], axis=0)
        else:
            s = env.sig[env.idx[:, env.t]]
            f = np.clip(rule(s, env.mu_bel, env.nu), 0, LEV_MAX)
        lev.append(float(np.mean(f)))
        obs2, _, done = env.step(f)
        if not done:
            obs = obs2
    st = env.stats(); st["avg_lev"] = float(np.mean(lev))
    return st


def fit(kind, pen, seed, noise=0.0):
    # Loi the phai NGAU NHIEN khi huan luyen: neu co dinh, dac trung "loi the tin"
    # cong tuyen hoan toan voi 1/sigma va agent khong the hoc cach phan ung voi niem tin.
    env = SizingEnv(EU, sharpe_true=SR_TRAIN, belief_noise=noise, seed=seed)
    pol = GaussianPolicy(seed=seed, lr=0.03)
    train(env, pol, n_iter=NIT, n_paths=NPATH, reward_kind=kind, cvar_pen=pen)
    return pol


# ══════════════════════════════════════════════════════════════════════
print("=" * 116)
print("A + B. DO ON DINH THEO SEED, VA BAC THUAN NHAT CUA PHAN THUONG")
print("=" * 116)
print("Cung kien truc, cung so vong huan luyen, cung du lieu. Chi doi ham phan thuong va seed.")
print(f"Moi o: {len(SEEDS)} seed doc lap. Tran don bay cua moi truong la {LEV_MAX:.0f}:1.\n")
print(f"{'Phần thưởng':<38}{'đòn bẩy theo seed':>34}{'hệ số biến thiên':>19}"
      f"{'tăng trưởng (trung vị)':>24}")
print("-" * 116)
BANK, res_ab = {}, []
for lab, kind, pen in REWARDS:
    pols, levs, grs, rus = [], [], [], []
    for sd in SEEDS:
        p_ = fit(kind, pen, sd)
        st = run_policy(SizingEnv(EU, sharpe_true=SR_TRUE, seed=999), NEVAL, pols=p_)
        pols.append(p_); levs.append(st["avg_lev"]); grs.append(st["growth"]); rus.append(st["p_ruin"])
        res_ab.append(dict(reward=lab, seed=sd, **st))
    BANK[lab] = pols
    L = np.array(levs)
    print(f"{lab:<38}{'  '.join(f'{x:5.1f}' for x in L):>34}"
          f"{L.std(ddof=1)/max(L.mean(),1e-9):>18.0%}"
          f"{np.median(grs):>23.1%}   [{time.time()-t0:.0f}s]", flush=True)
print("-" * 116)
print("KET QUA QUAN TRONG NHAT CUA GIAI DOAN NAY nam o cot 'he so bien thien':")
print("mot chinh sach RL huan luyen bang REINFORCE tren bai toan nay ket thuc o mot muc")
print("don bay bat ky trong khoang rong, chi phu thuoc seed. MOT ket qua mot-seed la vo nghia.")
print("\nVe bac thuan nhat: P&L so hoc TUYEN TINH theo f nen khong co cuc dai noi tai va")
print("agent bi keo len tran; P&L log LOM theo f nen co cuc dai duy nhat (chinh la Kelly);")
print("Differential Sharpe BAT BIEN voi phep nhan f nen ban than no khong noi gi ve khoi luong.")

# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 116)
print("C. HOP NHIEU SEED vs QUY TAC TINH   (loi the that = Sharpe 0,5/nam, uoc luong DUNG)")
print("=" * 116)
print("Cach chua don gian cho van de seed: hop trung binh hanh dong cua ca 4 chinh sach.\n")
print(f"{'Chính sách':<40}{'đòn bẩy TB':>13}{'tăng trưởng/năm':>18}{'P(phá sản)':>13}"
      f"{'MaxDD':>10}{'TT/MaxDD':>11}")
print("-" * 116)
res_c = []
for name, rule in STATIC.items():
    st = run_policy(SizingEnv(EU, sharpe_true=SR_TRUE, seed=999), NEVAL, rule=rule)
    res_c.append(dict(policy=name, kind="tĩnh", **st))
    print(f"{name:<40}{st['avg_lev']:>13.2f}{st['growth']:>17.1%}{st['p_ruin']:>12.1%}"
          f"{st['maxdd']:>10.1%}{st['growth']/max(st['maxdd'],1e-6):>11.2f}")
print("  " + "·" * 112)
for lab in ("P&L log (lõm theo f)", "P&L log + phạt sụt giảm"):
    st = run_policy(SizingEnv(EU, sharpe_true=SR_TRUE, seed=999), NEVAL, pols=BANK[lab])
    res_c.append(dict(policy="RL hợp 4 seed · " + lab, kind="RL", **st))
    print(f"{'RL hợp 4 seed · ' + lab:<40}{st['avg_lev']:>13.2f}{st['growth']:>17.1%}"
          f"{st['p_ruin']:>12.1%}{st['maxdd']:>10.1%}{st['growth']/max(st['maxdd'],1e-6):>11.2f}")
print("-" * 116)

# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 116)
print("D. PHEP THU SAI LECH NIEM TIN — RL sup nhu Kelly hay phang nhu quy tac co tran?")
print("=" * 116)
COLS = [("Kelly đầy đủ", None), ("Kelly 1/2", None), ("Kelly + trần rủi ro", None),
        ("RL hợp (P&L log)", "P&L log (lõm theo f)"),
        ("RL hợp (+phạt DD)", "P&L log + phạt sụt giảm")]
print(f"{'Tin/thật':>10}" + "".join(f"{c[0]:>21}" for c in COLS))
print("-" * 116)
res_d = []
for ratio in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
    cells = []
    for disp, key in COLS:
        env = SizingEnv(EU, sharpe_true=SR_TRUE, seed=999)
        st = (run_policy(env, 8000, rule=STATIC[disp], belief_mult=ratio) if key is None
              else run_policy(env, 8000, pols=BANK[key], belief_mult=ratio))
        res_d.append(dict(ratio=ratio, policy=disp, **st))
        cells.append(f"{st['growth']:+.0%} / {st['p_ruin']:.0%}")
    print(f"{ratio:>9.1f}x" + "".join(f"{c:>21}" for c in cells))
print("-" * 116)
print("Moi o: tang truong log/nam / xac suat pha san trong 250 phien.")

print("\n" + "=" * 116)
print("E. DON BAY MA TUNG CHINH SACH CHON, THEO MUC NIEM TIN")
print("=" * 116)
print("Chan doan quyet dinh: chinh sach co THUC SU dung dau vao 'loi the' khong,")
print("hay no phang vi mot ly do tam thuong (khong nhin dau vao do)?\n")
print(f"{'Tin/thật':>10}{'RL hợp (P&L log)':>20}{'Kelly đầy đủ':>16}{'Kelly + trần':>16}{'Cố định 2%':>14}")
print("-" * 116)
res_e = []
for ratio in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    row = {}
    for disp, key in [("RL", "P&L log (lõm theo f)"), ("Kelly đầy đủ", None),
                      ("Kelly + trần rủi ro", None), ("Cố định 2%", None)]:
        env = SizingEnv(EU, sharpe_true=SR_TRUE, seed=999)
        st = (run_policy(env, 8000, rule=STATIC[disp], belief_mult=ratio) if key is None
              else run_policy(env, 8000, pols=BANK[key], belief_mult=ratio))
        row[disp] = st["avg_lev"]
        res_e.append(dict(ratio=ratio, policy=disp, lev=st["avg_lev"]))
    print(f"{ratio:>9.2f}x{row['RL']:>20.2f}{row['Kelly đầy đủ']:>16.2f}"
          f"{row['Kelly + trần rủi ro']:>16.2f}{row['Cố định 2%']:>14.2f}")
print("-" * 116)
r0 = [x for x in res_e if x["ratio"] == 0.25]
r1 = [x for x in res_e if x["ratio"] == 8.0]
for d in ("RL", "Kelly đầy đủ", "Kelly + trần rủi ro"):
    a = [x["lev"] for x in r0 if x["policy"] == d][0]
    b = [x["lev"] for x in r1 if x["policy"] == d][0]
    print(f"   {d:<22} don bay tang {b/max(a,1e-9):>5.2f} lan khi niem tin tang 32 lan")

json.dump({"seed_ablation": res_ab, "vs_static": res_c,
           "sensitivity": res_d, "lev_response": res_e},
          open("/tmp/fx/exp7.json", "w"), indent=1)
print(f"\nDa luu exp7.json  ({time.time()-t0:.0f}s)")
