"""H7 — MATRIX PROFILE / ANALOG LICH SU, ho thu bay cua Giai doan 2.

Ke hoach Pha 1 cua HuyH (Week 2 muc B) doi Matrix Profile voi dung cac dac
trung: nearest_motif_distance, mean_top_k_distance, historical_successor_up_rate,
historical_successor_down_rate, motif_support. H2 (`run_h2_motif.py`) moi lam
mot PROXY bang codebook KMeans — no tra loi "cua so hom nay thuoc cum hinh
dang nao", KHONG tra loi duoc cau hoi cot loi cua analog forecasting:

    "nhung lan qua khu thi truong trong GIONG hom nay nhat, hom sau da xay ra
     chuyen gi?"

File nay lam dung cau hoi do.

──────────────────────────────────────────────────────────────────────────
RO RI NHIN TRUOC — VA VI SAO KHONG DUNG THANG stumpy.stump()
──────────────────────────────────────────────────────────────────────────
`stumpy.stump(T, m)` tinh tu-noi tren TOAN chuoi: "lang gieng gan nhat" cua
cua so tai t co the nam SAU t. Dung lam dac trung du bao thi ket cuc tuong lai
ro ri thang vao dac trung hien tai.

Nen o day tu viet tim kiem NHAN QUA, va dung stumpy lam DOI CHUNG cho cong
thuc khoang cach (tu kiem 3 ben duoi). Voi hang t, ung vien analog t' phai:

    t' + L <= t      cua so analog KHONG chong lan cua so hien tai
                     (loai "trung khop tam thuong" — cua so hom qua gan nhu
                      trung cua so hom nay)
    y[t'] >= 0       ket cuc cua analog DA BIET tai thoi diem t
                     (y[t'] la lop cua ngay t'+1, ma t'+1 <= t)

CONG THUC. Moi cua so da z-chuan hoa (trung binh 0, phuong sai 1) nen
    ||a - b||^2 = 2L - 2 a·b
tuc tim analog gan nhat = tim tich vo huong LON NHAT — mot phep nhan ma tran
vector duy nhat moi hang, thay vi vong lap tinh khoang cach.

KHONG GIAN GIA THUYET, CHOT TRUOC:
    3 do dai cua so (5, 10, 20) x 4 dac trung x 3 o phan vi = 36 vi tu
    x 3 lop = 108 gia thuyet
Bon dac trung: khoang cach toi analog gan nhat, va ty le ket cuc GIAM / DI
NGANG / TANG trong K = 20 analog gan nhat. Nguong phan vi chot tren HUAN
LUYEN, rieng tung cap.

Chay:  python src/run_h7_matrixprofile.py
Ghi:   output/h7_matrixprofile.json
"""
import json
import os
import sys
import time

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
from run_h2_motif import cua_so_chuan_hoa                      # noqa: E402

DO_DAI = (5, 10, 20)
K_ANALOG = 20              # so analog gan nhat — CHOT TRUOC
LICH_SU_TOI_THIEU = 250    # so phien qua khu toi thieu truoc khi duoc tim analog
MAX_PHU = 2.0 / 3.0        # tran do phu, cung ly do nhu run_h6_hmm.py
TEN_DT = ("khoảng cách analog gần nhất", "tỷ lệ analog → giảm",
          "tỷ lệ analog → đi ngang", "tỷ lệ analog → tăng")


def dac_trung_analog(S, y, L, k=K_ANALOG, toi_thieu=LICH_SU_TOI_THIEU):
    """Bon dac trung analog NHAN QUA cho tung hang.

    S: (n, L) cua so da z-chuan hoa, ket thuc tai t. y: nhan lop DA DICH.
    Tra ve (n, 4): [d1, ty_le_giam, ty_le_ngang, ty_le_tang].
    """
    n = len(S)
    ra = np.full((n, 4), np.nan)
    hop = np.isfinite(S).all(1)
    for t in range(toi_thieu, n):
        if not hop[t]:
            continue
        # UNG VIEN: khong chong lan (t' + L <= t) va ket cuc da biet
        gh = t - L + 1
        if gh <= 0:
            continue
        uv = np.flatnonzero(hop[:gh] & (y[:gh] >= 0))
        if len(uv) < k:
            continue
        dot = S[uv] @ S[t]                       # ||a-b||^2 = 2L - 2 a·b
        top = uv[np.argpartition(-dot, k - 1)[:k]]
        d1 = np.sqrt(max(2.0 * L - 2.0 * float(dot.max()), 0.0))
        yl = y[top]
        ra[t] = (d1, float((yl == 0).mean()), float((yl == 1).mean()),
                 float((yl == 2).mean()))
    return ra


def vi_tu_analog(zs, dts, y, pha, im_lang=False):
    """Dung vi tu analog: (M, ten). Tach ra de run_spa_ho2.py dung lai."""
    n_cap = [len(z) for z in zs]
    off = np.concatenate([[0], np.cumsum(n_cap)])
    lit, ten = [], []
    for L in DO_DAI:
        F_cap = []
        for i in range(len(B.PAIRS)):
            S = cua_so_chuan_hoa(zs[i], L)
            F_cap.append(dac_trung_analog(S, y[off[i]:off[i + 1]], L))
        F = np.concatenate(F_cap, axis=0)
        # roi rac hoa theo tam phan vi, nguong CHOT TREN HUAN LUYEN tung cap
        for j, ten_dt in enumerate(TEN_DT):
            v = F[:, j]
            idx = np.full(len(v), -1, int)
            for i, p in enumerate(B.PAIRS):
                sl = slice(off[i], off[i + 1])
                vi = v[sl]
                vt = vi[(doan(dts[i]) == 0) & np.isfinite(vi)]
                if len(vt) < 200:
                    continue
                q = np.quantile(vt, [1 / 3, 2 / 3])
                idx[sl] = np.where(np.isfinite(vi), np.digitize(vi, q), -1)
            for b in range(3):
                lit.append(idx == b)
                ten.append(f"{ten_dt} L={L} {['thấp','vừa','cao'][b]}")
        if not im_lang:
            print(f"  L={L} xong", flush=True)
    M = np.array(lit)
    phu = (M & pha[None, :]).sum(1) / max(int(pha.sum()), 1)
    qua_rong = phu > MAX_PHU
    if qua_rong.any():
        if not im_lang:
            print(f"loại {int(qua_rong.sum())} vị từ phủ > {MAX_PHU:.0%} mẫu")
        M = M[~qua_rong]
        ten = [t for t, b in zip(ten, qua_rong) if not b]
    return M, ten


def _tu_kiem():
    """Ba tu kiem: cong thuc khoang cach, tinh nhan qua, va doi chieu stumpy."""
    rng = np.random.default_rng(11)
    n, L = 600, 10
    x = rng.normal(0, 1, n).cumsum()
    S = cua_so_chuan_hoa(np.r_[np.nan, np.diff(x)], L)

    # (1) cong thuc 2L - 2a·b == khoang cach Euclid truc tiep
    ok = np.flatnonzero(np.isfinite(S).all(1))
    a, b = ok[100], ok[300]
    d_ct = np.sqrt(max(2.0 * L - 2.0 * float(S[a] @ S[b]), 0.0))
    d_tt = float(np.linalg.norm(S[a] - S[b]))
    assert abs(d_ct - d_tt) < 1e-8, f"cong thuc sai: {d_ct} vs {d_tt}"

    # (2) NHAN QUA: dac trung tai t khong doi khi cat bo tuong lai
    y = rng.integers(0, 3, n)
    t0 = 500
    f_day = dac_trung_analog(S, y, L, toi_thieu=300)
    f_cat = dac_trung_analog(S[:t0 + 1], y[:t0 + 1], L, toi_thieu=300)
    d_nq = float(np.nanmax(np.abs(f_day[t0] - f_cat[t0])))
    assert d_nq < 1e-12, f"RO RI: dac trung doi khi them tuong lai, {d_nq:.2e}"

    # (3) DOI CHIEU stumpy: khoang cach toi lang gieng gan nhat (KHONG nhan qua,
    #     chi de kiem chung cong thuc), dung cung vung loai tru
    import stumpy
    r = np.r_[np.nan, np.diff(x)]
    r_sach = r[np.isfinite(r)].astype(float)
    mp = stumpy.stump(r_sach, m=L)
    d_stumpy = float(mp[:, 0].min())
    Ss = cua_so_chuan_hoa(np.r_[np.nan, r_sach], L)
    okk = np.flatnonzero(np.isfinite(Ss).all(1))
    G = Ss[okk] @ Ss[okk].T
    D = np.sqrt(np.maximum(2.0 * L - 2.0 * G, 0.0))
    loai = int(np.ceil(L / 4))                  # vung loai tru mac dinh cua stumpy
    ii = np.abs(okk[:, None] - okk[None, :]) <= loai
    D[ii] = np.inf
    d_toi = float(D.min())
    lech = abs(d_stumpy - d_toi)
    assert lech < 1e-6, f"lech voi stumpy: {d_stumpy} vs {d_toi}"
    return d_nq, d_stumpy, d_toi


def main():
    t0 = time.time()
    print("=" * 112)
    print("H7 — MATRIX PROFILE / ANALOG LỊCH SỬ, họ thứ bảy của Giai đoạn 2")
    print("=" * 112)

    print("tự kiểm…", flush=True)
    d_nq, d_st, d_toi = _tu_kiem()
    print(f"  (1) công thức ||a−b||² = 2L − 2a·b khớp khoảng cách Euclid: ĐẠT")
    print(f"  (2) NHÂN QUẢ — đặc trưng tại t không đổi khi cắt bỏ tương lai: "
          f"lệch {d_nq:.2e}  ĐẠT")
    print(f"  (3) đối chiếu stumpy.stump: {d_st:.6f} vs tự tính {d_toi:.6f}  ĐẠT")

    du = nap_du_lieu()
    Ms, zs, dts = du["Ms"], du["zs"], du["dts"]
    y, cap = du["y"], du["cap"]
    kiem_soat, cum = du["kiem_soat"], du["cum"]
    tr, va, te, pha = du["tr"], du["va"], du["te"], du["pha"]

    print(f"\nTìm {K_ANALOG} analog gần nhất cho từng phiên (nhân quả, không "
          f"chồng lấn)…", flush=True)
    M, ten = vi_tu_analog(zs, dts, y, pha)
    print(f"\nKHÔNG GIAN GIẢ THUYẾT H7: {len(ten)} vị từ × 3 lớp = {len(ten)*3} "
          f"giả thuyết — liệt kê đầy đủ, biết trước")
    print(f"phát hiện {int(pha.sum()):,} hàng · xác nhận {int(te.sum()):,} hàng\n")

    print(f"[1/4] Westfall–Young, {NPERM} hoán vị, null khối {KHOI} ngày…", flush=True)
    Z, L_, nk, P, nguong = westfall_young(M, y, pha)
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
                            z=float(Z[i, c]), lift=float(L_[i, c]),
                            p_wy=float(P[i, c]), b_dk=b, t_dk=t))
        qua_dk = [u for u in ung if np.isfinite(u["t_dk"]) and abs(u["t_dk"]) > T_DIEU_KIEN]
        print(f"      {len(qua_dk)}/{len(ung)} vị từ còn tin riêng sau điều kiện hoá")
        print(f"\n      {'vị từ':<44}{'lớp':<10}{'n':>7}{'lift':>7}{'z':>8}{'t|đk':>8}")
        for u in sorted(ung, key=lambda x: -abs(x["z"]))[:12]:
            print(f"      {u['ten'][:42]:<44}{TEN_LOP[u['lop']]:<10}{u['n']:>7}"
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
    print(f"{'PHỄU H7 (Matrix Profile)':<46}{'còn lại':>10}")
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
                    xac_nhan=len(xn), quy_luat=xn, K_analog=K_ANALOG,
                    do_dai=list(DO_DAI)),
              open(os.path.join(OUT, "h7_matrixprofile.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/h7_matrixprofile.json — {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
