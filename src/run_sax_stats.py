"""VIEC 4 — KIEM DINH BOI CHO NHANH MAU KY HIEU.

Van de. `huyh_patterns.py` xac nhan ba mau cua HuyH co lift > 1 tren du lieu
cua minh. Nhung leave-one-pair-out chi kiem soat overfit THEO CAP; no khong
kiem soat SO LUONG MAU DA THU. Ham Hamalainen & Webb (DMKD 2019) noi thang:
lift THO xep hang cao cac mau gia, va phai hieu chinh da gia thuyet.

File nay lam ba viec:

  1. LIET KE TOAN BO khong gian mau (W = 2, 3, 4 trang thai tien de x 3 dich
     = 351 gia thuyet) thay vi chi ba mau da duoc chon san.

  2. HOAN VI WESTFALL-YOUNG kiem soat FWER (SPEck, Jenkins-Walzer-Goldfeld-
     Riondato, DMKD 2022). Mo hinh null: XOAY VONG chuoi trang thai cua tung
     cap mot khoang ngau nhien. Xoay vong bao toan CA phan phoi bien CA tu
     tuong quan cua chuoi — no chi pha vo su can le giua tien de va dich,
     dung thu ma gia thuyet noi ve. Thong ke max |z| tren toan bo 351 gia
     thuyet trong moi lan hoan vi cho ta phan phoi null co hieu chinh boi.
     Kem theo FDR Benjamini-Hochberg de so sanh (Sermpinis et al. 2021 dung
     FDR roi rac cho dung bai toan nay tren 21.000 quy tac ky thuat).

  3. DOI CHUNG CO DIEU KIEN. Hutchinson et al. (RIBAF 2022) cho thay toan bo
     loi suat bat thuong cua quy tac ky thuat tien te bi hap thu boi dong
     luong chuoi thoi gian. Mau cua minh la mau BIEN DONG chu khong phai mau
     HUONG, nen doi chung tuong duong khong phai TSMOM ma la: mau con noi
     them gi khi DA CO du bao HAR? Hoi quy trang thai dich len bien gia mau
     CONG log du bao HAR. Neu he so mau chet khi them HAR thi mau chi la
     cach ma hoa lai chinh du bao HAR.

Giao thuc: PHAT HIEN tren huan luyen + kiem dinh, XAC NHAN tren kiem tra.
"""
import os
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

from split import doan
import volfc2 as V2
from run_grid import bang_cache
from split import VALID_TU

TEN = ["LOW", "MEDIUM", "HIGH"]
WS = (2, 3, 4)
NPERM = 1000
MIN_KHOP = 20
MAU_HUYH = [(("MEDIUM", "HIGH", "HIGH"), "HIGH"),
            (("LOW", "MEDIUM", "LOW"), "LOW"),
            (("HIGH", "HIGH", "MEDIUM"), "HIGH")]


def nap():
    """Chuoi trang thai 3 muc cho moi cap; moc lay tu doan HUAN LUYEN."""
    adv = pd.read_csv(os.path.join(D, "rv_adv.csv"), parse_dates=["Date"])
    out = {}
    for p in V2.PAIRS:
        d = adv[adv.pair == p].sort_values("Date").reset_index(drop=True)
        d = d[d.n5 >= 100].reset_index(drop=True)
        g = doan(d.Date.values)
        v = d.rv5.values
        q = np.quantile(v[g == 0], [1 / 3, 2 / 3])
        s = np.digitize(v, q)                       # 0=LOW 1=MEDIUM 2=HIGH
        out[p] = (d.Date.values, s, g)
    return out


def ma_hoa(s, W):
    """Ma so nguyen cua cua so W trang thai KET THUC ngay t-1 (tien de cua t)."""
    n = len(s)
    code = np.full(n, -1)
    acc = np.zeros(n, int)
    for k in range(W):
        acc[W:] = acc[W:] * 3 + s[k:n - W + k]
    code[W:] = acc[W:]
    return code


def z_gop(DATA, W, seg_mask, dich_series=None):
    """Ma tran z cho moi (ma tien de, trang thai dich), gop 6 cap.

    z = (so lan trung - ky vong) / do lech chuan nhi thuc, voi xac suat nen
    lay tren CHINH doan dang xet.
    """
    nA = 3 ** W
    k = np.zeros((nA, 3)); nn = np.zeros(nA)
    pn = np.zeros(3); tot = 0
    for p in V2.PAIRS:
        _, s, g = DATA[p]
        code = ma_hoa(s, W)
        tgt = s if dich_series is None else dich_series[p]
        m = seg_mask(g) & (code >= 0)
        c = code[m]; t = tgt[m]
        for j in range(3):
            pn[j] += (t == j).sum()
        tot += m.sum()
        np.add.at(nn, c, 1)
        np.add.at(k, (c, t), 1)
    pn = pn / max(tot, 1)
    exp = nn[:, None] * pn[None, :]
    sd = np.sqrt(np.maximum(nn[:, None] * pn[None, :] * (1 - pn[None, :]), 1e-12))
    z = np.where(nn[:, None] >= MIN_KHOP, (k - exp) / sd, np.nan)
    lift = np.where(nn[:, None] >= MIN_KHOP, k / np.maximum(exp, 1e-12), np.nan)
    return z, lift, nn, k


def _khoi_bootstrap(s, rng, mean_block):
    """Bootstrap khoi dung (stationary bootstrap, Politis-Romano).

    Giu duoc tinh dai NGAN HAN cua chuoi trang thai (do dai khoi trung binh
    `mean_block`) nhung pha vo cau truc dai hon khoi. Mo hinh null nay tra
    loi dung cau hoi: mau ba ngay co noi them gi NGOAI tinh dai don gian?
    """
    n = len(s); pgeo = 1.0 / mean_block
    out = np.empty(n, s.dtype); i = 0
    while i < n:
        st = int(rng.integers(0, n)); L = int(rng.geometric(pgeo))
        L = min(L, n - i)
        idx = (st + np.arange(L)) % n
        out[i:i + L] = s[idx]; i += L
    return out


def z_phang(DATA, seg_mask, keys, dich=None):
    """Vector z theo dung thu tu `keys` = [(W, ma_tiende, dich), ...]."""
    Z = {}
    for W in WS:
        Z[W] = z_gop(DATA, W, seg_mask, dich)[0]
    return np.array([Z[W][a, j] for (W, a, j) in keys])


def chay_hoanvi(DATA, seg_mask, rng, nperm=NPERM, kieu="xoay", mean_block=2,
                keys=None):
    """Phan phoi null cua max|z| tren TOAN BO khong gian.

    kieu='xoay'  : XOAY chuoi DICH so voi chuoi TIEN DE. Pha vo moi phu
                   thuoc dan-tre. Null "khong co gi du bao duoc".
    kieu='khoi'  : bootstrap khoi dung do dai trung binh `mean_block`. Giu
                   tinh dai ngan han. Null "chi co tinh dai, khong co mau".
    """
    Zs = []
    for b in range(nperm):
        if kieu == "xoay":
            dich = {}
            for p in V2.PAIRS:
                s = DATA[p][1]
                dich[p] = np.roll(s, int(rng.integers(50, len(s) - 50)))
            Zs.append(z_phang(DATA, seg_mask, keys, dich))
        else:
            DP = {}
            for p in V2.PAIRS:
                dt, s, g = DATA[p]
                DP[p] = (dt, _khoi_bootstrap(s, rng, mean_block), g)
            Zs.append(z_phang(DP, seg_mask, keys))
    return np.abs(np.nan_to_num(np.vstack(Zs)))          # (nperm, m)


def westfall_young(z_obs, Zb):
    """maxT TUNG BUOC XUONG (Westfall & Young 1993). Tra ve p da hieu chinh.

    Buoc xuong moi la ban dung: sau khi bac bo cac gia thuyet manh nhat,
    thong ke max cua buoc sau chi lay tren PHAN CON LAI. Neu dung ban MOT
    BUOC thi cac mau tinh dai tam thuong (HIGH->HIGH => HIGH, z~69) chiem
    het thong ke max va lam moi gia thuyet khac khong bao gio bac bo duoc.
    """
    a = np.abs(np.nan_to_num(z_obs))
    o = np.argsort(-a)
    m = len(a)
    p = np.empty(m)
    for r in range(m):
        sub = o[r:]
        mx = Zb[:, sub].max(1)
        p[o[r]] = float((mx >= a[o[r]]).mean())
    # ep tinh don dieu theo thu tu bac bo
    run = 0.0
    for r in range(m):
        run = max(run, p[o[r]])
        p[o[r]] = run
    return p


def main():
    DATA = nap()
    rng = np.random.default_rng(3)

    def phat_hien(g):
        return g <= 1                       # huan luyen + kiem dinh

    def kiem_tra(g):
        return g == 2

    n_gt = sum(3 ** W * 3 for W in WS)
    print("=" * 100)
    print("VIỆC 4 — KIỂM ĐỊNH BỘI CHO NHÁNH MẪU KÝ HIỆU")
    print("=" * 100)
    print(f"không gian giả thuyết: W ∈ {WS} → {sum(3**W for W in WS)} tiền đề × 3 đích "
          f"= {n_gt} giả thuyết")
    print(f"Westfall-Young maxT từng bước xuống, {NPERM} hoán vị, ba mô hình null")
    print(f"phát hiện trên HUẤN LUYỆN + KIỂM ĐỊNH, xác nhận trên KIỂM TRA\n")

    goc = {W: z_gop(DATA, W, phat_hien) for W in WS}
    keys = []
    for W in WS:
        z, lift, nn, k = goc[W]
        for a in range(3 ** W):
            if nn[a] < MIN_KHOP:
                continue
            for j in range(3):
                if np.isfinite(z[a, j]):
                    keys.append((W, a, j))
    z_obs = np.array([goc[W][0][a, j] for (W, a, j) in keys])
    print(f"{len(keys)} giả thuyết đủ số khớp (≥{MIN_KHOP})\n")

    print("BA MÔ HÌNH NULL (SPEck, Jenkins et al. 2022: chọn null quyết định kết luận)")
    print("-" * 100)
    NUL = {}
    for ten, kieu, mb in (("xoay — không gì dự báo được", "xoay", 0),
                          ("khối 2 ngày — đã chứa sẵn tính dai AR(1)", "khoi", 2),
                          ("khối 5 ngày — đã chứa sẵn tính dai một tuần", "khoi", 5)):
        Zb = chay_hoanvi(DATA, phat_hien, np.random.default_rng(3), NPERM,
                         kieu, mb, keys)
        NUL[ten] = Zb
        q = np.quantile(Zb.max(1), [0.90, 0.95, 0.99])
        print(f"  {ten:<45} max|z| null: 90% {q[0]:>6.2f}  95% {q[1]:>6.2f}  99% {q[2]:>6.2f}")
    print(f"  {'(không hiệu chỉnh bội, hai phía 5%)':<45} ngưỡng           |z| =   1.96")
    print("-" * 100)
    KEY2 = "khối 2 ngày — đã chứa sẵn tính dai AR(1)"
    print("  Cột p W-Y dùng null KHỐI 2 NGÀY và thủ tục maxT TỪNG BƯỚC XUỐNG.")
    print("  Null này đã chứa sẵn tính dai của biến động, nên một mẫu chỉ sống sót nếu")
    print("  nó nói thêm điều gì NGOÀI 'hôm qua cao thì hôm nay cũng cao'.\n")
    p_wy_all = {}
    for ten, Zb in NUL.items():
        p_wy_all[ten] = westfall_young(z_obs, Zb)
    P_WY = {(W, a, j): p_wy_all[KEY2][i] for i, (W, a, j) in enumerate(keys)}
    P_XOAY = {(W, a, j): p_wy_all["xoay — không gì dự báo được"][i]
              for i, (W, a, j) in enumerate(keys)}
    P_K5 = {(W, a, j): p_wy_all["khối 5 ngày — đã chứa sẵn tính dai một tuần"][i]
            for i, (W, a, j) in enumerate(keys)}

    rows = []
    for W in WS:
        z, lift, nn, k = goc[W]
        for a in range(3 ** W):
            if nn[a] < MIN_KHOP:
                continue
            mau = tuple(TEN[(a // 3 ** (W - 1 - i)) % 3] for i in range(W))
            for j in range(3):
                if not np.isfinite(z[a, j]):
                    continue
                pr = 2 * (1 - stats.norm.cdf(abs(z[a, j])))
                rows.append(dict(W=W, mau=" → ".join(mau), dich=TEN[j],
                                 n=int(nn[a]), lift=float(lift[a, j]),
                                 z=float(z[a, j]), p_tho=pr,
                                 p_wy=float(P_WY[(W, a, j)]),
                                 p_wy_xoay=float(P_XOAY[(W, a, j)]),
                                 p_wy_k5=float(P_K5[(W, a, j)])))
    df = pd.DataFrame(rows).sort_values("z", key=lambda s: -s.abs()).reset_index(drop=True)

    # FDR Benjamini-Hochberg trên p thô
    m = len(df)
    o = np.argsort(df.p_tho.values)
    ranked = df.p_tho.values[o]
    bh = ranked * m / (np.arange(m) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    q = np.empty(m); q[o] = np.clip(bh, 0, 1)
    df["q_fdr"] = q

    n_tho = int((df.p_tho < 0.05).sum())
    n_fdr = int((df.q_fdr < 0.10).sum())
    print(f"BAO NHIÊU GIẢ THUYẾT SỐNG SÓT ({m} giả thuyết)")
    print("-" * 100)
    print(f"  không hiệu chỉnh gì (p<0,05):                       {n_tho:>4}   "
          f"— nếu toàn nhiễu thì kỳ vọng {0.05*m:.0f}")
    print(f"  FDR Benjamini-Hochberg trên p thô (q<0,10):         {n_fdr:>4}")
    print(f"  W-Y bước xuống, null XOAY (p<0,05):                 "
          f"{int((df.p_wy_xoay<0.05).sum()):>4}")
    print(f"  W-Y bước xuống, null KHỐI 2 NGÀY (p<0,05):          {int((df.p_wy<0.05).sum()):>4}")
    print(f"  W-Y bước xuống, null KHỐI 5 NGÀY (p<0,05):          "
          f"{int((df.p_wy_k5<0.05).sum()):>4}\n")

    print("15 giả thuyết mạnh nhất trên HUẤN LUYỆN + KIỂM ĐỊNH")
    print("-" * 100)
    print(f"{'W':>2}{'mẫu':<40}{'⇒ đích':<10}{'n':>7}{'lift':>8}{'z':>8}"
          f"{'p thô':>9}{'p W-Y':>9}{'q FDR':>9}")
    print("-" * 100)
    for _, r in df.head(15).iterrows():
        print(f"{r.W:>2}{r.mau:<40}{r.dich:<10}{r.n:>7}{r.lift:>8.3f}{r.z:>8.2f}"
              f"{r.p_tho:>9.4f}{r.p_wy:>9.4f}{r.q_fdr:>9.4f}")
    print("-" * 100)

    print("\nBA MẪU CỦA HuyH TRONG BỐI CẢNH TOÀN KHÔNG GIAN")
    print("-" * 100)
    print(f"{'mẫu':<40}{'lift':>8}{'z':>8}{'p thô':>9}{'p W-Y':>9}{'q FDR':>9}"
          f"{'hạng':>8}{'sống sót?':>12}")
    print("-" * 100)
    songsot = []
    for mau, dich in MAU_HUYH:
        nhan = " → ".join(mau)
        r = df[(df.mau == nhan) & (df.dich == dich)]
        if len(r) == 0:
            print(f"{nhan:<40}{'không đủ số khớp':>60}")
            continue
        r = r.iloc[0]
        hang = int(df.index[(df.mau == nhan) & (df.dich == dich)][0]) + 1
        ok = r.p_wy < 0.05
        if ok:
            songsot.append((mau, dich))
        print(f"{nhan+' ⇒ '+dich:<40}{r.lift:>8.3f}{r.z:>8.2f}{r.p_tho:>9.4f}"
              f"{r.p_wy:>9.4f}{r.q_fdr:>9.4f}{hang:>8}"
              f"{('CÓ' if ok else 'KHÔNG'):>12}")
    print("-" * 100)

    # ── xác nhận trên KIỂM TRA
    print("\nXÁC NHẬN TRÊN ĐOẠN KIỂM TRA (chưa dùng ở bước phát hiện)")
    print("-" * 100)
    print(f"{'mẫu':<40}{'n':>7}{'lift kiểm tra':>16}{'z':>8}{'p thô':>10}")
    print("-" * 100)
    for W in WS:
        zt, lt, nt, _ = z_gop(DATA, W, kiem_tra)
        for mau, dich in MAU_HUYH:
            if len(mau) != W:
                continue
            a = 0
            for c in mau:
                a = a * 3 + TEN.index(c)
            j = TEN.index(dich)
            if nt[a] < MIN_KHOP or not np.isfinite(zt[a, j]):
                print(f"{' → '.join(mau)+' ⇒ '+dich:<40}{int(nt[a]):>7}"
                      f"{'không đủ số khớp':>16}")
                continue
            pr = 2 * (1 - stats.norm.cdf(abs(zt[a, j])))
            print(f"{' → '.join(mau)+' ⇒ '+dich:<40}{int(nt[a]):>7}{lt[a,j]:>16.3f}"
                  f"{zt[a,j]:>8.2f}{pr:>10.4f}")
    print("-" * 100)

    # ── đối chứng có điều kiện: mẫu còn nói thêm gì khi ĐÃ CÓ dự báo HAR?
    print("\n\nĐỐI CHỨNG CÓ ĐIỀU KIỆN — mẫu còn nói thêm gì khi đã có dự báo HAR?")
    print("Thay cho đối chứng TSMOM của Hutchinson et al. (2022), vốn dành cho mẫu")
    print("HƯỚNG. Mẫu ở đây là mẫu BIẾN ĐỘNG nên đối chứng đúng là dự báo HAR.")
    print("Hồi quy:  1{trạng thái đích} = a + b·1{khớp mẫu} + c·log(dự báo HAR)")
    print("-" * 100)
    bang, chung = bang_cache()
    tr = np.asarray(chung < VALID_TU)
    import pickle
    with open(os.path.join(OUT, "cauhinh_chot.pkl"), "rb") as f:
        CH = pickle.load(f)
    FH = V2.chay(bang, chung, deseason=CH["deseason"],
                 crosspair=bool(CH["crosspair"]), event=CH["event"],
                 window=None, lams=(CH["lam"],), train_mask=tr,
                 recal=CH["recal"])[CH["lam"]]
    print(f"{'mẫu':<40}{'b (chỉ mẫu)':>14}{'t':>7}{'b (có HAR)':>13}{'t':>7}"
          f"{'t của HAR':>12}{'kết luận':>16}")
    print("-" * 100)
    for mau, dich in MAU_HUYH:
        W = len(mau); j = TEN.index(dich)
        a = 0
        for c in mau:
            a = a * 3 + TEN.index(c)
        yy, xx, hh = [], [], []
        for p in V2.PAIRS:
            dt, s, g = DATA[p]
            code = ma_hoa(s, W)
            f = np.asarray(FH[p])
            ser = pd.Series(f, index=pd.DatetimeIndex(chung)).reindex(pd.DatetimeIndex(dt))
            fv = ser.values
            m = (code >= 0) & np.isfinite(fv) & (fv > 0)
            yy.append((s[m] == j).astype(float))
            xx.append((code[m] == a).astype(float))
            hh.append(np.log(fv[m]))
        y = np.concatenate(yy); x = np.concatenate(xx); h = np.concatenate(hh)
        h = (h - h.mean()) / h.std()

        def ols(X, y):
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            r = y - X @ b
            s2 = r @ r / (len(y) - X.shape[1])
            se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
            return b, b / se
        X1 = np.column_stack([np.ones_like(y), x])
        b1, t1 = ols(X1, y)
        X2 = np.column_stack([np.ones_like(y), x, h])
        b2, t2 = ols(X2, y)
        kl = "còn tin riêng" if abs(t2[1]) > 2.0 else "bị HAR hấp thụ"
        print(f"{' → '.join(mau)+' ⇒ '+dich:<40}{b1[1]:>14.4f}{t1[1]:>7.2f}"
              f"{b2[1]:>13.4f}{t2[1]:>7.2f}{t2[2]:>12.2f}{kl:>16}")
    print("-" * 100)
    df.to_csv(os.path.join(OUT, "sax_kiemdinh_boi.csv"), index=False)
    print(f"\nđã ghi output/sax_kiemdinh_boi.csv ({len(df)} giả thuyết)")
    print("=" * 100)


if __name__ == "__main__":
    main()
