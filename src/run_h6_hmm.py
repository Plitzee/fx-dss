"""H6 — HMM / MARKOV SWITCHING, ho thu sau cua Giai doan 2.

Day la thuat toan DUY NHAT trong ke hoach Pha 1 cua HuyH (muc "Week 2 — A.
HMM / Markov Switching") ma repo CHUA he thu. Che do hien co trong he thong
(`SigmaCheDo`, `A3`, va `H5` hom nay) deu la che do QUAN SAT DUOC — chia theo
tam phan vi cua mot dai luong da tinh duoc (sigma^, tu tuong quan). HMM khac ve
ban chat: trang thai la AN, phai suy ra tu chuoi quan sat, va co ma tran
chuyen trang thai rieng.

QUAN SAT: loi suat log ngay tho `r_t` (KHONG chia sigma^). Chon nhu vay vi ke
hoach muon chinh cac trang thai mang y nghia thi truong doc duoc — "yen tinh
bien do thap", "xu huong len", "xu huong xuong", "cang thang bien dong cao" —
va mot HMM Gauss don bien tren r sinh ra dung phan loai do (cac trang thai
khac nhau ca o TRUNG BINH lan PHUONG SAI). Cau hoi "no co noi them gi ngoai
sigma^ khong" da duoc tra loi rieng o cua DOI CHUNG CO DIEU KIEN cua pheu, noi
bo kiem soat da chua bien gia phan vi cua sigma^ rieng tung cap.

KHONG GIAN GIA THUYET, CHOT TRUOC KHI CHAY:
    K = 2, 3, 4 trang thai  ->  2 + 3 + 4 = 9 vi tu  x  3 lop  =  27 gia thuyet
Ba cau hinh K deu duoc khai bao truoc, khong dua K nao ra khoi bang sau khi
nhin ket qua.

──────────────────────────────────────────────────────────────────────────
RO RI NHIN TRUOC — CHO DE SAI NHAT CUA HMM, VA CACH FILE NAY CHAN
──────────────────────────────────────────────────────────────────────────
`hmmlearn` co hai ham suy dien, VA CA HAI DEU RO RI neu dung de lam dac trung
du bao:

    model.predict(X)        Viterbi tren TOAN chuoi -> trang thai tai t phu
                            thuoc ca nhung quan sat SAU t
    model.predict_proba(X)  hau nghiem LAM TRON (smoothed), cung dung toan chuoi

Trang thai tai t vi the "biet" tuong lai. Dung lam dac trung thi moi ket qua
deu vo nghia.

File nay chi dung `hmmlearn` de KHOP tham so (Baum-Welch tren doan huan
luyen), con SUY DIEN thi tu viet BO LOC TIEN (forward filter):

    alpha_t(j) ~ P(trang thai_t = j | y_1..y_t)     <- chi dung qua khu

va chung minh tinh nhan qua bang ba tu kiem o `_tu_kiem()`:
  1. khoi phuc duoc tham so tu du lieu mo phong tu mot HMM da biet
  2. loc == lam tron TAI BUOC CUOI (dong nhat thuc toan hoc, phai dung khop)
  3. alpha tai t KHONG DOI khi noi them du lieu tuong lai vao sau  <- quyet dinh

Tham so khop RIENG TUNG CAP tren doan HUAN LUYEN, 5 lan khoi dong ngau nhien
lay lan co log-likelihood cao nhat (bai hoc tu CAViaR: mot diem khoi dau duy
nhat cho ket qua khong tai lap duoc).

Chay:  python src/run_h6_hmm.py
Ghi:   output/h6_hmm.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                             # noqa: E402
from split import doan                                        # noqa: E402
from run_quyluat import (                                     # noqa: E402
    nap_du_lieu, z_lift, westfall_young, doi_chung, TEN_LOP,
    MIN_KHOP, LIFT_LOPO, MIN_CAP_DUONG, T_DIEU_KIEN, NPERM, KHOI, EPS,
)

CAC_K = (2, 3, 4)          # CHOT TRUOC — khong bo K nao sau khi nhin ket qua
N_KHOI_DAU = 5             # so lan khoi dong ngau nhien moi lan khop
SEED = 0

# TRAN DO PHU — vi tu bat gan khap mau thi khong phai quy luat.
#
# Pheu cua repo von co SAN do phu toi thieu (MIN_KHOP = 100 lan khop) nhung
# KHONG co tran. Voi moi ho truoc day dieu do khong thanh van de: khong gian
# goc chia tam phan vi nen mot menh de phu toi da ~1/3; cum motif ~10%; H5 la
# giao cua hai dieu kien nen con nho hon. HMM thi khac han: trang thai bien
# dong THAP chiem 17.628/18.306 hang = 96% chuoi.
#
# Voi mot vi tu phu 96%, phan phoi null cua max|z| duoi hoan vi khoi gan nhu
# SUY BIEN (lay gan het mau thi z hau nhu khong doi khi xao tron), nen ngay ca
# |z| = 0,5 cung "vuot" nguong W-Y. Do la artefact cua phep kiem chu khong
# phai tin hieu — va no da lo ra bang mot dau hieu khong the bo qua: so vi tu
# song sot W-Y (12) LON HON so vi tu dat |z| > 1,96 (8), dieu khong the xay ra
# voi mot hieu chinh boi lanh manh.
#
# Nguong 2/3 chot theo LY DO CAU TRUC, khong phai sau khi nhin ket qua: khong
# gian gia thuyet chinh cua repo sinh ra vi tu phu toi da ~1/3, nen 2/3 la
# rong RAI GAP DOI muc do. No khong rang buoc bat ky ho nao khac (da kiem: H1,
# H2, H3, H5 khong co vi tu nao vuot 2/3), chi loai dung trang thai nen cua
# HMM — thu von khong phai "quy luat" ma la "binh thuong".
MAX_PHU = 2.0 / 3.0


# ── bo loc tien: NHAN QUA, tu viet ──────────────────────────────────────
def loc_tien(r, khoi_dau, chuyen, mu, sd):
    """alpha[t, j] = P(trang thai_t = j | y_1..y_t) — CHI dung den het t.

    Truy hoi chuan: du bao mot buoc bang ma tran chuyen, roi nhan mat do phat
    xa cua quan sat TAI t, roi chuan hoa. Khong co buoc lui nao, nen khong the
    dung thong tin sau t.

    NaN trong r: giu nguyen alpha cua phien truoc sau khi day qua ma tran
    chuyen (tuc "khong quan sat duoc gi hom nay" chu khong phai "quan sat thay
    0" — dung bai hoc cua loi bia loi suat bang 0, CHISO_DANHGIA.md muc 13).
    """
    n, K = len(r), len(khoi_dau)
    al = np.full((n, K), np.nan)
    truoc = np.asarray(khoi_dau, float)
    for t in range(n):
        du = truoc if t == 0 else truoc @ chuyen
        if np.isfinite(r[t]):
            ph = np.exp(-0.5 * ((r[t] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            hn = du * np.maximum(ph, 1e-300)
            s = hn.sum()
            hn = hn / s if s > 0 else du
        else:
            hn = du                                  # khong quan sat -> chi day tien
        al[t] = hn
        truoc = hn
    return al


def khop_hmm(r_tr, K, seed=SEED, n_dau=N_KHOI_DAU):
    """Baum-Welch tren doan HUAN LUYEN, `n_dau` lan khoi dong, lay lan co
    log-likelihood cao nhat. Tra ve (khoi_dau, chuyen, mu, sd)."""
    from hmmlearn.hmm import GaussianHMM
    X = np.asarray(r_tr, float).reshape(-1, 1)
    tot, tot_ll = None, -np.inf
    for i in range(n_dau):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=200,
                            random_state=seed + 17 * i, tol=1e-4)
            try:
                m.fit(X)
                ll = float(m.score(X))
            except Exception:
                continue
        if np.isfinite(ll) and ll > tot_ll:
            tot_ll, tot = ll, m
    if tot is None:
        return None
    return (np.asarray(tot.startprob_, float), np.asarray(tot.transmat_, float),
            tot.means_.ravel().astype(float),
            np.sqrt(tot.covars_.ravel()).astype(float))


def _tu_kiem():
    """Ba tu kiem, cai thu ba la cai quyet dinh."""
    rng = np.random.default_rng(7)
    # sinh du lieu tu mot HMM 2 trang thai DA BIET
    A = np.array([[0.95, 0.05], [0.10, 0.90]])
    mu_th, sd_th = np.array([0.0, 0.0]), np.array([0.4, 1.6])
    n = 6000
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = rng.choice(2, p=A[s[t - 1]])
    r = rng.normal(mu_th[s], sd_th[s])

    th = khop_hmm(r, 2, seed=3)
    assert th is not None, "khop that bai"
    kd, ch, mu, sd = th
    # (1) khoi phuc tham so: do lech chuan hai trang thai phai gan 0,4 va 1,6
    sd_sap = np.sort(sd)
    lech = max(abs(sd_sap[0] - 0.4), abs(sd_sap[1] - 1.6))
    assert lech < 0.15, f"khong khoi phuc duoc tham so, lech {lech:.3f}"

    # (2) LOC == LAM TRON tai buoc cuoi (dong nhat thuc)
    from hmmlearn.hmm import GaussianHMM
    m = GaussianHMM(n_components=2, covariance_type="diag")
    m.startprob_, m.transmat_ = kd, ch
    m.means_ = mu.reshape(-1, 1)
    m.covars_ = (sd ** 2).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lam_tron = m.predict_proba(r.reshape(-1, 1))
    al = loc_tien(r, kd, ch, mu, sd)
    d_cuoi = float(np.max(np.abs(al[-1] - lam_tron[-1])))
    assert d_cuoi < 1e-8, f"loc != lam tron o buoc cuoi: {d_cuoi:.2e}"

    # (3) NHAN QUA: alpha tai t khong doi khi noi them tuong lai
    t0 = 4000
    al_ngan = loc_tien(r[:t0 + 1], kd, ch, mu, sd)
    d_nq = float(np.max(np.abs(al_ngan[t0] - al[t0])))
    assert d_nq < 1e-12, f"RO RI: alpha[t] doi khi them tuong lai, lech {d_nq:.2e}"

    # doi chieu: lam tron THI CO doi -> chung to phep kiem (3) that su co luc
    d_lt = float(np.max(np.abs(m.predict_proba(r[:t0 + 1].reshape(-1, 1))[t0]
                               - lam_tron[t0])))
    return d_cuoi, d_nq, d_lt


def main():
    t0 = time.time()
    print("=" * 112)
    print("H6 — HMM / MARKOV SWITCHING (trạng thái ẩn), họ thứ sáu của Giai đoạn 2")
    print("=" * 112)

    print("tự kiểm bộ lọc tiến…", flush=True)
    d_cuoi, d_nq, d_lt = _tu_kiem()
    print(f"  (1) khôi phục tham số từ HMM mô phỏng: ĐẠT")
    print(f"  (2) lọc ≡ làm trơn ở bước cuối       : lệch {d_cuoi:.2e}  ĐẠT")
    print(f"  (3) NHÂN QUẢ — α[t] không đổi khi nối thêm tương lai: lệch {d_nq:.2e}  ĐẠT")
    print(f"      (đối chiếu: hậu nghiệm LÀM TRƠN lệch {d_lt:.3f} — tức phép kiểm (3) có lực)")

    du = nap_du_lieu()
    Ms, dts = du["Ms"], du["dts"]
    y, cap = du["y"], du["cap"]
    kiem_soat, cum = du["kiem_soat"], du["cum"]
    tr, va, te, pha = du["tr"], du["va"], du["te"], du["pha"]

    print(f"\nKhớp HMM (Baum-Welch, {N_KHOI_DAU} lần khởi động) trên HUẤN LUYỆN, "
          f"riêng từng cặp, K = {CAC_K}…", flush=True)
    lit_cap = {K: [] for K in CAC_K}
    mo_ta = {}
    for i, p in enumerate(B.PAIRS):
        c = Ms[i].close.values
        r = np.r_[np.nan, np.diff(np.log(np.maximum(c, EPS)))]
        tri = doan(dts[i]) == 0
        r_tr = r[tri & np.isfinite(r)]
        for K in CAC_K:
            th = khop_hmm(r_tr, K, seed=SEED)
            if th is None:
                lit_cap[K].append(np.zeros((K, len(r)), bool))
                continue
            kd, ch, mu, sd = th
            al = loc_tien(r, kd, ch, mu, sd)          # NHAN QUA
            # CHUAN HOA NHAN TRANG THAI — bat buoc khi gop nhieu cap.
            # Baum-Welch danh so trang thai TUY Y (label switching): "trang
            # thai 0" cua EURUSD co the la trang thai yen tinh, con cua GBPUSD
            # lai la trang thai cang thang. Gop theo chi so tho la tron lan hai
            # che do khac nghia nhau, va vi tu gop lai se vo nghia. Sap lai
            # theo sd TANG DAN de chi so luon co cung y nghia o moi cap:
            # 0 = bien dong thap nhat ... K-1 = bien dong cao nhat.
            thu = np.argsort(sd)
            al = al[:, thu]
            tt = np.argmax(al, axis=1)
            hop = np.isfinite(al).all(1)
            lit_cap[K].append(np.array([(tt == j) & hop for j in range(K)]))
            if i == 0:
                mo_ta[K] = [f"sd={sd[j]*1e4:.0f}pip, mu={mu[j]*1e4:+.1f}pip, "
                            f"dai={1/max(1-ch[j,j],1e-9):.0f} phiên" for j in thu]
    for K in CAC_K:
        print(f"  K={K} (EURUSD, xếp theo biến động): " + " · ".join(mo_ta.get(K, [])))

    M = np.concatenate([np.concatenate(lit_cap[K], axis=1) for K in CAC_K], axis=0)
    ten = [f"HMM K={K} trạng thái {j}" for K in CAC_K for j in range(K)]
    print(f"\nKHÔNG GIAN GIẢ THUYẾT H6: {len(ten)} vị từ × 3 lớp = {len(ten)*3} "
          f"giả thuyết — liệt kê đầy đủ, biết trước")

    # TRAN DO PHU — xem giai thich o hang so MAX_PHU
    phu = (M & pha[None, :]).sum(1) / max(int(pha.sum()), 1)
    qua_rong = phu > MAX_PHU
    if qua_rong.any():
        print(f"\nloại {int(qua_rong.sum())} vị từ phủ > {MAX_PHU:.0%} mẫu "
              f"(trạng thái NỀN, không phải quy luật — xem hằng số MAX_PHU):")
        for i in np.where(qua_rong)[0]:
            print(f"    {ten[i]:<26} phủ {phu[i]:.1%}")
        M = M[~qua_rong]
        ten = [t for t, b in zip(ten, qua_rong) if not b]
        print(f"  → còn {len(ten)} vị từ × 3 lớp = {len(ten)*3} giả thuyết")
    print(f"phát hiện {int(pha.sum()):,} hàng · xác nhận {int(te.sum()):,} hàng\n")

    print(f"[1/4] Westfall–Young, {NPERM} hoán vị, null khối {KHOI} ngày…", flush=True)
    Z, L, nk, P, nguong = westfall_young(M, y, pha)
    print(f"      ngưỡng max|z| null khối: 90% {nguong[0]:.2f} · "
          f"95% {nguong[1]:.2f} · 99% {nguong[2]:.2f}")
    du_khop = nk >= MIN_KHOP
    song = (P < 0.05) & np.isfinite(Z)
    tho = (np.abs(np.nan_to_num(Z)) > 1.96) & np.isfinite(Z)
    print(f"      {int(du_khop.sum())}/{len(ten)} vị từ đủ {MIN_KHOP} lần khớp")
    print(f"      sống sót W-Y p<0,05: {int(song.sum())} / thô p<0,05: {int(tho.sum())}"
          f" (nếu toàn nhiễu kỳ vọng {0.05*np.isfinite(Z).sum():.0f})")

    qua_dk, qua_lopo, xn = [], [], []
    if song.sum() > 0:
        print(f"\n[2/4] Đối chứng có điều kiện (|t| > {T_DIEU_KIEN})…", flush=True)
        ung = []
        for i, c in zip(*np.where(song)):
            b, t = doi_chung(M[i], y, c, kiem_soat, cum=cum)
            ung.append(dict(i=int(i), lop=int(c), ten=ten[i], n=int(nk[i]),
                            z=float(Z[i, c]), lift=float(L[i, c]),
                            p_wy=float(P[i, c]), b_dk=b, t_dk=t))
        qua_dk = [u for u in ung if np.isfinite(u["t_dk"]) and abs(u["t_dk"]) > T_DIEU_KIEN]
        print(f"      {len(qua_dk)}/{len(ung)} vị từ còn tin riêng sau điều kiện hoá")
        print(f"\n      {'vị từ':<26}{'lớp':<10}{'n':>7}{'lift':>7}{'z':>8}{'t|đk':>8}")
        for u in sorted(ung, key=lambda x: -abs(x["z"]))[:12]:
            print(f"      {u['ten']:<26}{TEN_LOP[u['lop']]:<10}{u['n']:>7}"
                  f"{u['lift']:>7.3f}{u['z']:>8.2f}{u['t_dk']:>8.2f}")

        print(f"\n[3/4] Bỏ-một-cặp (lift ≥ {LIFT_LOPO}, ≥{MIN_CAP_DUONG}/6 cặp dương)…",
              flush=True)
        for u in qua_dk:
            mi, lifts = M[u["i"]], []
            for p in B.PAIRS:
                mp = (cap == p) & pha & (y >= 0)
                kh = mi & mp
                if kh.sum() < 20:
                    lifts.append(np.nan); continue
                pc = (y[mp] == u["lop"]).mean()
                lifts.append(float((y[kh] == u["lop"]).mean() / max(pc, EPS)))
            lifts = np.array(lifts)
            nd = int(np.nansum(lifts > 1.0))
            u["lift_cap"], u["so_cap_duong"] = lifts.tolist(), nd
            u["lift_min"] = float(np.nanmin(lifts))
            if nd >= MIN_CAP_DUONG and np.nanmin(lifts) >= LIFT_LOPO:
                qua_lopo.append(u)
        print(f"      {len(qua_lopo)}/{len(qua_dk)} chuyển giao được qua các cặp")

        print("\n[4/4] Xác nhận trên đoạn KIỂM TRA…", flush=True)
        Zte, Lte, _ = z_lift(M, y, te)
        for u in qua_lopo:
            u["z_te"] = float(Zte[u["i"], u["lop"]])
            u["lift_te"] = float(Lte[u["i"], u["lop"]])
        xn = [u for u in qua_lopo if np.isfinite(u["z_te"]) and u["z_te"] > 1.96]
        print(f"      {len(xn)}/{len(qua_lopo)} tái lập trên kiểm tra")
    else:
        print("\n→ KHÔNG vị từ nào sống sót Westfall–Young.")

    print("\n" + "=" * 112)
    print(f"{'PHỄU H6 (HMM)':<46}{'còn lại':>10}")
    for nhan, v in (("không gian giả thuyết", len(ten) * 3),
                    ("đủ số lần khớp", int(du_khop.sum())),
                    ("thô p<0,05", int(tho.sum())),
                    ("sống sót Westfall–Young", int(song.sum())),
                    ("còn tin riêng sau đối chứng", len(qua_dk)),
                    ("chuyển giao được (LOPO)", len(qua_lopo)),
                    ("tái lập trên KIỂM TRA", len(xn))):
        print(f"{nhan:<46}{v:>10,}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(khong_gian=len(ten) * 3, du_khop=int(du_khop.sum()),
                    tho=int(tho.sum()), wy=int(song.sum()),
                    sau_dieu_kien=len(qua_dk), sau_lopo=len(qua_lopo),
                    xac_nhan=len(xn), quy_luat=xn, cac_K=list(CAC_K),
                    tu_kiem=dict(loc_vs_lamtron=d_cuoi, nhan_qua=d_nq)),
              open(os.path.join(OUT, "h6_hmm.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/h6_hmm.json — {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
