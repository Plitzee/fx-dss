"""Bo chi so danh gia hien dai. Moi ham deu co kiem chung o cuoi file."""
import numpy as np
from scipy import stats

# ── 1. Ham chi so bien dong (Patton 2011: chi MSE va QLIKE ben voi thuoc do nhieu)
def qlike(proxy, h):
    """QLIKE = log h + proxy/h  (proxy, h la PHUONG SAI). Thap hon la tot hon."""
    h = np.asarray(h, float); proxy = np.asarray(proxy, float)
    return np.log(h) + proxy/h

def mse_var(proxy, h):
    return (np.asarray(proxy, float) - np.asarray(h, float))**2

# ── 2. Pinball va CRPS
def pinball(y, q, tau):
    """rho_tau(y,q) = (tau - 1{y<q})(y - q).  y, q vo huong hoac mang."""
    y = np.asarray(y, float); q = np.asarray(q, float)
    return (tau - (y < q).astype(float))*(y - q)

TAU_GRID = np.linspace(0.005, 0.995, 199)

def crps_from_quantiles(y, Q, taus=TAU_GRID):
    """CRPS = 2 * tich phan pinball tren moi muc phan vi.
    Q: mang (n, len(taus)) cac phan vi du bao."""
    y = np.asarray(y, float)[:, None]
    L = (taus[None, :] - (y < Q).astype(float))*(y - Q)
    return 2*np.trapezoid(L, taus, axis=1)

def crps_normal(y, mu, sd):
    """Dang dong kin cho phan phoi chuan — dung de KIEM CHUNG ham tren."""
    z = (np.asarray(y,float)-mu)/sd
    return sd*(z*(2*stats.norm.cdf(z)-1) + 2*stats.norm.pdf(z) - 1/np.sqrt(np.pi))

# ── 3. VaR / ES cho phan phoi t chuan hoa (phuong sai = 1)
def std_t_ppf(p, nu):
    return stats.t.ppf(p, nu)/np.sqrt(nu/(nu-2))

def std_t_es(p, nu):
    """ES duoi muc p cua t chuan hoa (p nho, duoi trai). Tra ve gia tri AM."""
    s = np.sqrt(nu/(nu-2))
    x = stats.t.ppf(p, nu)
    # ES cua t chuan (chua chuan hoa) o duoi trai
    es_raw = -stats.t.pdf(x, nu)*(nu + x**2)/((nu-1)*p)
    return es_raw/s

# ── 4. FZ0 loss cho cap (VaR, ES)  — Patton, Ziegel & Chen 2019
def fz0(y, v, e, alpha):
    """y loi suat; v = VaR (AM); e = ES (AM); alpha = xac suat duoi (vd 0,025)."""
    y = np.asarray(y,float); v = np.asarray(v,float); e = np.asarray(e,float)
    ind = (y <= v).astype(float)
    return -(1/(alpha*e))*ind*(v-y) + v/e + np.log(-e) - 1

# ── 5. Kiem dinh backtest VaR
def kupiec(hits, alpha):
    """H0: ty le vi pham = alpha. LR_uc ~ chi2(1)."""
    hits = np.asarray(hits, float); n = len(hits); x = hits.sum()
    if x == 0: return np.nan, np.nan, 0.0
    ph = x/n
    ll0 = (n-x)*np.log(1-alpha) + x*np.log(alpha)
    ll1 = (n-x)*np.log(1-ph)    + x*np.log(ph)
    lr = -2*(ll0-ll1)
    return lr, 1-stats.chi2.cdf(lr, 1), ph

def christoffersen_ind(hits):
    """H0: chuoi vi pham doc lap (Markov bac 1). LR_ind ~ chi2(1)."""
    h = np.asarray(hits, int)
    n00=n01=n10=n11=0
    for a,b in zip(h[:-1], h[1:]):
        if a==0 and b==0: n00+=1
        elif a==0 and b==1: n01+=1
        elif a==1 and b==0: n10+=1
        else: n11+=1
    if (n01+n11)==0 or (n00+n01)==0 or (n10+n11)==0: return np.nan, np.nan
    p01 = n01/(n00+n01); p11 = n11/(n10+n11); p = (n01+n11)/(n00+n01+n10+n11)
    def lg(x): return np.log(x) if x>0 else 0.0
    ll1 = n00*lg(1-p01)+n01*lg(p01)+n10*lg(1-p11)+n11*lg(p11)
    ll0 = (n00+n10)*lg(1-p)+(n01+n11)*lg(p)
    lr = -2*(ll0-ll1)
    return lr, 1-stats.chi2.cdf(lr, 1)

def dq_test(hits, var_series, alpha, lags=4):
    """Engle & Manganelli 2004 dynamic quantile test. Manh hon Kupiec/Christoffersen."""
    h = np.asarray(hits, float) - alpha
    n = len(h)
    X = [np.ones(n-lags)]
    for l in range(1, lags+1):
        X.append(h[lags-l:n-l])
    X.append(np.asarray(var_series, float)[lags:])
    X = np.column_stack(X)
    Y = h[lags:]
    try:
        b = np.linalg.solve(X.T@X, X.T@Y)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    stat = float(Y@X@b/(alpha*(1-alpha)))
    return stat, 1-stats.chi2.cdf(stat, X.shape[1])

# ── 6. Model Confidence Set (Hansen, Lunde & Nason 2011), thong ke T_R, bootstrap khoi
def _boot_means(losses, boot_idx, batch=40):
    """Trung binh bootstrap cua tung mo hinh, tinh theo lo de khong tran bo nho."""
    outs=[]
    for i in range(0, boot_idx.shape[0], batch):
        outs.append(losses[boot_idx[i:i+batch]].mean(axis=1))   # (batch, M)
    return np.vstack(outs)

def mcs(losses, alpha=0.10, B=1000, block=20, seed=0):
    """Model Confidence Set (Hansen, Lunde & Nason 2011), thong ke T_R, bootstrap khoi.
    losses: (T, M). Tra ve (chi so cac mo hinh song sot, danh sach bi loai kem p-value)."""
    rng = np.random.default_rng(seed)
    T, M = losses.shape
    nblk = int(np.ceil(T/block))
    starts = rng.integers(0, max(1, T-block+1), size=(B, nblk))
    boot_idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :T]
    boot_idx = np.clip(boot_idx, 0, T-1)
    alive = list(range(M)); elim = []
    while len(alive) > 1:
        L = np.ascontiguousarray(losses[:, alive])
        Lb = _boot_means(L, boot_idx)                  # (B, k)
        mbar = L.mean(axis=0)
        dbar = mbar[:, None] - mbar[None, :]           # (k, k)
        db_boot = Lb[:, :, None] - Lb[:, None, :]      # (B, k, k)
        var = db_boot.var(axis=0, ddof=1) + 1e-30
        t = dbar/np.sqrt(var)
        TR = np.nanmax(np.abs(t))
        t_boot = (db_boot - dbar[None])/np.sqrt(var)[None]
        TR_boot = np.nanmax(np.abs(t_boot), axis=(1, 2))
        pval = float((TR_boot >= TR).mean())
        if pval > alpha:
            break
        worst = int(np.nanargmax(np.nanmax(t, axis=1)))   # mo hinh te nhat so voi ke tot nhat
        elim.append((alive[worst], pval))
        alive.pop(worst)
    return alive, elim

# ── 7. Kiem chung noi bo
if __name__ == "__main__":
    print("KIEM CHUNG CAI DAT")
    print("-"*70)
    rng = np.random.default_rng(0)
    # CRPS: dang so vs dang dong kin
    y = rng.normal(0,1,5000); mu=0.1; sd=1.3
    Q = mu + sd*stats.norm.ppf(TAU_GRID)[None,:]*np.ones((len(y),1))
    a = crps_from_quantiles(y, Q); b = crps_normal(y, mu, sd)
    print(f"CRPS so vs dong kin (chuan): sai so tuong doi TB = {np.mean(np.abs(a-b)/b):.5f}")
    # pinball tai tau=0,5 = 1/2 * MAE
    q = np.full_like(y, mu)
    print(f"Pinball(0,5) vs MAE/2      : {np.mean(pinball(y,q,0.5)):.6f} vs {np.mean(np.abs(y-q))/2:.6f}")
    # ES cua t chuan hoa: so voi mo phong
    for nu in (5, 8):
        s = rng.standard_t(nu, 400000)/np.sqrt(nu/(nu-2))
        v = std_t_ppf(0.025, nu); es_emp = s[s<=v].mean()
        print(f"ES 2,5% t(nu={nu}) — cong thuc {std_t_es(0.025,nu):+.4f} | mo phong {es_emp:+.4f}")
    # Kupiec: duoi H0 ty le tu choi ~ 5%
    rej=[]
    for i in range(3000):
        h = (rng.random(1000) < 0.025).astype(int)
        _,p,_ = kupiec(h, 0.025); rej.append(p<0.05)
    print(f"Kupiec sai lam loai I (danh nghia 5%): {np.mean(rej):.3f}")
    # DQ: duoi H0
    rej=[]
    for i in range(1500):
        h = (rng.random(1000) < 0.025).astype(int)
        v = -2*np.ones(1000)
        _,p = dq_test(h, v, 0.025); rej.append(p<0.05)
    print(f"DQ     sai lam loai I (danh nghia 5%): {np.mean(rej):.3f}")
    # MCS: 3 mo hinh giong het nhau -> khong loai ai; 1 mo hinh te ro -> bi loai
    L = rng.normal(0,1,(500,3))
    keep,_ = mcs(L, alpha=0.10, B=400, seed=1)
    print(f"MCS voi 3 mo hinh tuong duong : giu lai {len(keep)}/3 (ky vong 3)")
    L2 = np.column_stack([rng.normal(0,1,500), rng.normal(0,1,500), rng.normal(3,1,500)])
    keep2,el = mcs(L2, alpha=0.10, B=400, seed=1)
    print(f"MCS khi mo hinh 3 te ro rang  : giu lai {sorted(keep2)} (ky vong [0, 1])")
