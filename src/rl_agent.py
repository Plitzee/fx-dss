"""
CHINH SACH GAUSSIAN + REINFORCE CO BASELINE, cai bang numpy thuan.
MLP 2 lop (6 -> 24 -> 1). Dao ham viet tay — minh bach, khong hop den, khong can GPU.
"""
import numpy as np
from rl_env import LEV_MAX, N_STATE


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class GaussianPolicy:
    def __init__(self, n_state=N_STATE, n_hidden=24, seed=0, lr=0.02, sigma_a=0.6):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1 / np.sqrt(n_state), (n_state, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = r.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, 1))
        self.b2 = np.zeros(1)
        self.lr = lr
        self.sigma_a = sigma_a
        self.rng = r
        # bo dem Adam
        self._m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self._t = 0

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, s):
        h = np.tanh(s @ self.W1 + self.b1)
        m = (h @ self.W2 + self.b2).ravel()
        return m, h

    def act(self, s, explore=True):
        m, h = self.forward(s)
        a = m + self.sigma_a * self.rng.normal(size=m.shape) if explore else m
        f = LEV_MAX * sigmoid(a)
        return a, f, m, h

    def grads(self, s, h, m, a, adv):
        """Gradient cua -J theo tham so. adv: (n,) loi the da chuan hoa."""
        dlogp_dm = (a - m) / self.sigma_a ** 2           # (n,)
        g = -(dlogp_dm * adv)                            # dau am: ta minimize -J
        n = len(g)
        dW2 = (h.T @ g[:, None]) / n
        db2 = np.array([g.mean()])
        dh = np.outer(g, self.W2.ravel()) * (1 - h ** 2)
        dW1 = (s.T @ dh) / n
        db1 = dh.mean(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def update(self, grads, clip=1.0):
        self._t += 1
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        tot = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        scale = min(1.0, clip / (tot + 1e-12))
        for k, p in self.params().items():
            g = grads[k] * scale
            self._m[k] = b1_ * self._m[k] + (1 - b1_) * g
            self._v[k] = b2_ * self._v[k] + (1 - b2_) * g ** 2
            mh = self._m[k] / (1 - b1_ ** self._t)
            vh = self._v[k] / (1 - b2_ ** self._t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)


def train(env, policy, n_iter=250, n_paths=256, reward_kind="dsr", cvar_pen=0.0,
          gamma=0.99, verbose=False):
    hist = []
    for it in range(n_iter):
        S, H, M, A, R = [], [], [], [], []
        obs = env.reset(n_paths)
        done = False
        while not done:
            a, f, m, h = policy.act(obs)
            S.append(obs); H.append(h); M.append(m); A.append(a)
            obs2, rew, done = env.step(f, reward_kind=reward_kind, cvar_pen=cvar_pen)
            R.append(rew)
            if not done:
                obs = obs2
        R = np.array(R)                                   # (T, n)
        T = R.shape[0]
        G = np.zeros_like(R)
        run = np.zeros(R.shape[1])
        for t in range(T - 1, -1, -1):
            run = R[t] + gamma * run
            G[t] = run
        base = G.mean(axis=1, keepdims=True)              # baseline theo buoc thoi gian
        ADV = G - base
        sd = ADV.std() + 1e-8
        ADV = ADV / sd
        gsum = {k: np.zeros_like(v) for k, v in policy.params().items()}
        for t in range(T):
            gt = policy.grads(S[t], H[t], M[t], A[t], ADV[t])
            for k in gsum:
                gsum[k] += gt[k]
        for k in gsum:
            gsum[k] /= T
        policy.update(gsum)
        st = env.stats()
        hist.append(st)
        if verbose and (it % 25 == 0 or it == n_iter - 1):
            print(f"    iter {it:>3}  tang truong {st['growth']:+.1%}  "
                  f"pha san {st['p_ruin']:.1%}  MaxDD {st['maxdd']:.1%}", flush=True)
    return hist


def evaluate(env, policy, n_paths=20000, reward_kind="dsr"):
    obs = env.reset(n_paths)
    lev = []
    done = False
    while not done:
        _, f, m, _ = policy.act(obs, explore=False)
        lev.append(f.mean())
        obs2, _, done = env.step(f, reward_kind=reward_kind)
        if not done:
            obs = obs2
    st = env.stats()
    st["avg_lev"] = float(np.mean(lev))
    return st
