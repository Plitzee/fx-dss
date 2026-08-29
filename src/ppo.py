"""PPO (numpy thuan) cho tang dinh co vi the — thay REINFORCE, giu tham so hoa phan du.
Muc tieu kep + ham gia tri lam duong co so + GAE. Dao ham viet tay, khong can thu vien."""
import numpy as np
from rl_env import LEV_MAX, N_STATE

def sig_(x): return 1.0/(1.0+np.exp(-np.clip(x,-30,30)))

class PPO:
    def __init__(self,n_state=N_STATE,n_hidden=32,seed=0,lr=3e-3,sigma_a=0.6,
                 clip=0.2,epochs=4,vf_coef=0.5):
        r=np.random.default_rng(seed); self.rng=r
        self.W1=r.normal(0,1/np.sqrt(n_state),(n_state,n_hidden)); self.b1=np.zeros(n_hidden)
        self.W2=r.normal(0,1/np.sqrt(n_hidden),(n_hidden,1))*0.1; self.b2=np.zeros(1)
        self.V1=r.normal(0,1/np.sqrt(n_state),(n_state,n_hidden)); self.c1=np.zeros(n_hidden)
        self.V2=r.normal(0,1/np.sqrt(n_hidden),(n_hidden,1))*0.1; self.c2=np.zeros(1)
        self.lr=lr; self.sigma_a=sigma_a; self.clip=clip; self.epochs=epochs; self.vf=vf_coef
        self._m={k:np.zeros_like(v) for k,v in self.params().items()}
        self._v={k:np.zeros_like(v) for k,v in self.params().items()}; self._t=0
    def params(self): return {"W1":self.W1,"b1":self.b1,"W2":self.W2,"b2":self.b2,
                              "V1":self.V1,"c1":self.c1,"V2":self.V2,"c2":self.c2}
    def pi(self,s):
        h=np.tanh(s@self.W1+self.b1); return (h@self.W2+self.b2).ravel(),h
    def val(self,s):
        h=np.tanh(s@self.V1+self.c1); return (h@self.V2+self.c2).ravel(),h
    def act(self,s,explore=True):
        m,_=self.pi(s)
        a=m+self.sigma_a*self.rng.normal(size=m.shape) if explore else m
        return a,m
    def update(self,S,A,Mold,ADV,RET,n_mb=4):
        n=len(S); idx=np.arange(n)
        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for mb in np.array_split(idx,n_mb):
                s,a,mo,adv,ret=S[mb],A[mb],Mold[mb],ADV[mb],RET[mb]
                m,h=self.pi(s); v,hv=self.val(s)
                # ty le xac suat (Gaussian, sigma co dinh)
                lr_=(-0.5*((a-m)**2-(a-mo)**2)/self.sigma_a**2)
                ratio=np.exp(np.clip(lr_,-20,20))
                un=ratio*adv; cl=np.clip(ratio,1-self.clip,1+self.clip)*adv
                mask=(un<=cl).astype(float)          # nhanh nao duoc chon boi min()
                # grad muc tieu (ta minimize -L)
                g_m=-(mask*adv*ratio*(a-m)/self.sigma_a**2)/len(mb)
                dW2=h.T@g_m[:,None]; db2=np.array([g_m.sum()])
                dh=np.outer(g_m,self.W2.ravel())*(1-h**2)
                dW1=s.T@dh; db1=dh.sum(0)
                g_v=self.vf*2*(v-ret)/len(mb)
                dV2=hv.T@g_v[:,None]; dc2=np.array([g_v.sum()])
                dhv=np.outer(g_v,self.V2.ravel())*(1-hv**2)
                dV1=s.T@dhv; dc1=dhv.sum(0)
                self.step({"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,
                           "V1":dV1,"c1":dc1,"V2":dV2,"c2":dc2})
    def step(self,g,clip_norm=1.0):
        self._t+=1; b1_,b2_,eps=0.9,0.999,1e-8
        tot=np.sqrt(sum(float((x**2).sum()) for x in g.values()))
        sc=min(1.0,clip_norm/(tot+1e-12))
        for k,p in self.params().items():
            gg=g[k]*sc
            self._m[k]=b1_*self._m[k]+(1-b1_)*gg
            self._v[k]=b2_*self._v[k]+(1-b2_)*gg**2
            mh=self._m[k]/(1-b1_**self._t); vh=self._v[k]/(1-b2_**self._t)
            p-=self.lr*mh/(np.sqrt(vh)+eps)

def gae(R,V,gamma=0.99,lam=0.95):
    T,n=R.shape; A=np.zeros_like(R); last=np.zeros(n)
    for t in range(T-1,-1,-1):
        nv=V[t+1] if t+1<T else np.zeros(n)
        delta=R[t]+gamma*nv-V[t]
        last=delta+gamma*lam*last; A[t]=last
    return A,A+V
