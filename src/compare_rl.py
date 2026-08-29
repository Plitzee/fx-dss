"""CVaR-PPO — gia thuyet: PPO thuong that bai vi loi ich cua viec giam co khi
sut giam nam gan het o PHAN DUOI. Toi uu duoi thay vi trung binh phai sua duoc.

Gradient chinh sach CVaR:  chi cac duong di trong duoi alpha dong gop,
voi loi the (G_i - VaR_alpha). Cac duong khac bi che hoan toan.
"""
import sys,time,json,warnings; warnings.filterwarnings("ignore")
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from rl_env import SizingEnv, LEV_MAX
from sizing import f_kelly, f_ruin_cap
from ppo import PPO, gae, sig_

SR_TRUE=0.5; SR_TRAIN=(0.0,1.2); K_LO,K_HI=0.5,1.5
NIT,NPATH,NEVAL=int(sys.argv[2]),160,6000
ALPHA=0.20                      # tap trung vao 20% duong di te nhat

def fbase(env):
    s=env.sig[env.idx[:,env.t]]
    return np.clip(np.minimum(f_kelly(env.mu_bel,s,1.0),f_ruin_cap(s,250,0.01,env.nu)),0,LEV_MAX)

def roll(env,pol,n,explore=True):
    obs=env.reset(n); S,A,M,R,V=[],[],[],[],[]; done=False; lev=[]
    while not done:
        a,m=pol.act(obs,explore); v,_=pol.val(obs)
        f=np.clip((K_LO+(K_HI-K_LO)*sig_(a))*fbase(env),0,LEV_MAX)
        S.append(obs);A.append(a);M.append(m);V.append(v); lev.append(float(np.mean(f)))
        o2,r,done=env.step(f,reward_kind="pnl_log"); R.append(r)
        if not done: obs=o2
    st=env.stats(); st["avg_lev"]=float(np.mean(lev))
    return (np.array(S),np.array(A),np.array(M),np.array(R),np.array(V)),st

def train(env,pol,n_iter,n_paths,mode):
    for _ in range(n_iter):
        (S,A,M,R,V),_=roll(env,pol,n_paths)
        if mode=="ppo":
            ADV,RET=gae(R,V)
            ADV=(ADV-ADV.mean())/(ADV.std()+1e-8)
        else:
            G=R.sum(0)                                    # loi ich ca duong di
            var=np.quantile(G,ALPHA)
            tail=(G<=var).astype(float)                   # chi phan duoi dong gop
            adv_ep=(G-var)*tail
            adv_ep=(adv_ep-adv_ep[tail>0].mean())/(adv_ep[tail>0].std()+1e-8) if tail.sum()>1 else adv_ep
            ADV=np.repeat(adv_ep[None,:],R.shape[0],axis=0)*tail[None,:]
            _,RET=gae(R,V)
        pol.update(S.reshape(-1,S.shape[-1]),A.ravel(),M.ravel(),ADV.ravel(),RET.ravel())

def k_vs_dd(pol,env):
    """CHAN DOAN THEN CHOT: he so k hoc duoc co phu thuoc sut giam khong?"""
    env.reset(200); base=env._obs()[0]
    out=[]
    for dd in (0.0,0.05,0.10,0.20,0.30):
        o=base.copy(); o[:,2]=dd*5                        # cot 2 la sut giam
        _,m=pol.act(o,explore=False)
        out.append(float(np.mean(K_LO+(K_HI-K_LO)*sig_(m))))
    return out

PANEL=os.environ.get("FX_PANEL","panel2_6pairs.csv")
P=pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data",PANEL),parse_dates=["Date"])
print(f"panel: {PANEL}  ({len(P):,} dòng)")
PAIRS=sys.argv[1].split(","); SEEDS=[1,2,3]
res={}
t0=time.time()
for mode in ("ppo","cvar"):
    res[mode]={"lev":[],"g":[],"r":[],"kdd":[]}
    for pair in PAIRS:
        D=P[P.pair==pair].reset_index(drop=True); k=int(len(D)*.7)
        TR,TE=D.iloc[:k].reset_index(drop=True),D.iloc[k:].reset_index(drop=True)
        for sd in SEEDS:
            env=SizingEnv(TR,sharpe_true=SR_TRAIN,seed=sd); pol=PPO(seed=sd)
            train(env,pol,NIT,NPATH,mode)
            te=SizingEnv(TE,sharpe_true=SR_TRUE,seed=999)
            _,st=roll(te,pol,NEVAL,explore=False)
            res[mode]["lev"].append(st["avg_lev"]); res[mode]["g"].append(st["growth"])
            res[mode]["r"].append(st["p_ruin"]); res[mode]["kdd"].append(k_vs_dd(pol,te))
        print(f"  {mode} {pair}: xong [{time.time()-t0:.0f}s]",flush=True)

print("\n"+"="*80); print(f"KET QUA — {len(PAIRS)} cap x {len(SEEDS)} seed, {NIT} vong"); print("="*80)
print(f"{'':<12}{'tăng trưởng':>14}{'độ lệch':>10}{'phá sản TB':>13}{'tệ nhất':>10}")
print("-"*80)
for mode in ("ppo","cvar"):
    g=np.array(res[mode]["g"]); r=np.array(res[mode]["r"])
    nm="PPO thường" if mode=="ppo" else "CVaR-PPO"
    print(f"{nm:<12}{g.mean():>13.2%}{g.std():>10.3f}{r.mean():>12.2%}{r.max():>10.2%}")

print("\n"+"="*80)
print("CHAN DOAN — he so k theo muc sut giam (co hoc duoc dieu kien hoa khong?)")
print("="*80)
print(f"{'':<12}"+"".join(f"{'dd='+f'{d:.0%}':>10}" for d in (0,.05,.10,.20,.30))+f"{'biên độ':>11}")
print("-"*80)
for mode in ("ppo","cvar"):
    K=np.array(res[mode]["kdd"]).mean(0)
    nm="PPO thường" if mode=="ppo" else "CVaR-PPO"
    print(f"{nm:<12}"+"".join(f"{v:>10.3f}" for v in K)+f"{K.max()-K.min():>11.3f}")
print("-"*80)
print("Quy tac thiet ke tay: k(dd) di tu 1,300 xuong 0,500 — bien do 0,800")
json.dump({m:{k:(np.array(v).tolist()) for k,v in d.items()} for m,d in res.items()},
          open("cvar_results.json","w"))
