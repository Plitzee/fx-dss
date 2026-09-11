"""SPA cho tung ho cua Giai doan 2 (H2 motif, H3 rule-list, H5 che do).

Dong tiep phan con lai cua B1 (`docs/REPLAN_2026.md` muc 10.4): sau khi H2,
H3, H5 da co code sinh vi tu (run_h2_motif.py, run_h3_rulelist.py,
run_h5_chedo.py), file nay bien MOI vi tu cua tung ho thanh MOT ung vien du
bao xac suat ba lop, roi ap `metrics.spa_test()` — cau hoi: CA HO co ai thang
duoc nen mot cach co y nghia khong, kiem soat cho viec thu nhieu ung vien.

NEN LA "CHI SIGMA^" — dung nhu REPLAN_2026.md muc 10.4 doi hoi, KHONG PHAI
khi hau hoc. (Ban dau file nay dung khi hau hoc vi de plumbing hon, nhung do
la nen QUA YEU: gan nhu moi vi tu lien quan bien dong se "thang" khi hau hoc
mot cach tam thuong — chinh la dieu H1 da chi ra roi (sigma^ cao/thap/vua co
lift rat lon so khi hau hoc, nhung do LA CHINH NEN chu khong phai kham pha
moi). So sanh voi "chi sigma^" moi la cau hoi dung: co vi tu nao NGOAI thong
tin sigma^ da co khong.

Nen duoc DUNG LAI tren dung chi so hang cua run_quyluat.py (khac voi
run_balop.py — hang bi dich 1 phien va gop theo cach khac): tinh lai
B.dung_muc_tieu(d, H=1, tr) tung cap de lay canh_P/sigma_h, khop ChiSigma tren
huan luyen, du bao, roi DICH 1 phien giong het cach `nap_du_lieu()`/`yv` da
dich muc tieu — vi nen phai du bao CHO DUNG phien ma y (da dich) dang noi ve.

MOI vi tu j THANH MOT UNG VIEN: du bao 3 lop = tan suat CO DIEU KIEN tren
HUAN LUYEN neu vi tu active tai phien do, nguoc lai la du bao nen "chi sigma^"
cua CHINH phien do. Danh gia tren doan KIEM DINH (chua dung de phat hien
quy luat).

Chay:  python src/run_spa_ho2.py
Ghi:   output/spa_ho2.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                              # noqa: E402
from split import doan                                         # noqa: E402
from run_quyluat import nap_du_lieu, dac_trung, roi_rac, vet_can  # noqa: E402
from run_h2_motif import vi_tu_motif                            # noqa: E402
from run_h3_rulelist import xay_cay, MAX_DEPTH, MIN_LA          # noqa: E402
from run_h5_chedo import che_do_tu_tuong_quan, N_CD, CD_TEN     # noqa: E402
from metrics import spa_test                                    # noqa: E402

EPS = 1e-12


def loss_log3(P, y):
    P = np.asarray(P, float)
    y = np.asarray(y, int)
    return -np.log(np.maximum(P[np.arange(len(y)), y], EPS))


def nen_chi_sigma(Ms, sigs, dts):
    """Du bao "chi sigma^" (ChiSigma) DUNG chi so hang cua run_quyluat.py:
    dich 1 phien giong het `yv` — hang t noi ve lop cua ngay t+1.

    Dung lai DUNG cach nap_du_lieu() da dung d = DataFrame(Date, sig, zT) —
    KHONG the goi dung_muc_tieu truc tiep tren Ms[i] vi Ms[i] la panel gia
    goc, khong co cot sig/zT ma dung_muc_tieu can."""
    ra = []
    for i, p in enumerate(B.PAIRS):
        d = pd.DataFrame({"Date": Ms[i].Date.values, "sig": sigs[i]})
        c = Ms[i].close.values
        zt = np.full(len(Ms[i]), np.nan)
        zt[1:] = np.log(c[1:] / np.maximum(c[:-1], EPS)) / np.maximum(sigs[i][1:], EPS)
        d["zT"] = zt
        tri = doan(d.Date.values) == 0
        T = B.dung_muc_tieu(d, 1, tri)
        ns = B.ChiSigma().khop(T["z"][tri])
        Pn = ns.du_bao(len(d), canh=T["canh_P"], sigma_h=T["sigma_h"], sig=sigs[i])
        Pshift = np.full_like(Pn, np.nan)
        Pshift[:-1] = Pn[1:]
        ra.append(Pshift)
    return np.concatenate(ra, axis=0)


def ung_vien_tu_vi_tu(M, y, tr_mask, apply_mask, kh_forecast, min_n=100):
    """Voi MOI hang cua M (mot vi tu), tra ve (n_apply, 3) du bao: tan suat
    3-lop CO DIEU KIEN tren train (neu du min_n mau) khi vi tu active tai
    phien do, nguoc lai dung `kh_forecast` (nen "chi sigma^" DA CHIA se, cung
    chi so hang) cho phien do."""
    kh_tr = np.bincount(y[tr_mask & (y >= 0)], minlength=3).astype(float)
    kh_tr = kh_tr / max(kh_tr.sum(), 1.0)
    N = int(apply_mask.sum())
    nen_ap = kh_forecast[apply_mask]
    thieu = ~np.isfinite(nen_ap).all(1)
    nen_ap = np.where(thieu[:, None], kh_tr[None, :], nen_ap)  # phong ve khi thieu
    cac = []
    for j in range(M.shape[0]):
        act_tr = M[j] & tr_mask & (y >= 0)
        if act_tr.sum() >= min_n:
            f = np.bincount(y[act_tr], minlength=3).astype(float)
            f = f / f.sum()
        else:
            f = kh_tr
        act_ap = M[j][apply_mask]
        P = nen_ap.copy()
        P[act_ap] = f
        cac.append(P)
    return cac


def chay_spa(ten_ho, M, ten_vt, y, tr_mask, va_mask, kh_forecast, max_ung_vien=None):
    print(f"\n{'─'*100}\n{ten_ho}: {M.shape[0]:,} vị từ (ứng viên)")
    if max_ung_vien and M.shape[0] > max_ung_vien:
        # gioi han so ung vien de SPA khong qua nang — LAY THEO |z| tho, tuc
        # theo chinh do manh cua tin hieu THO, khong nhin p sau hieu chinh
        from run_quyluat import z_lift
        Z, L, nk = z_lift(M, y, tr_mask | va_mask)
        diem = np.nanmax(np.abs(np.nan_to_num(Z)), axis=1)
        top = np.argsort(-diem)[:max_ung_vien]
        M = M[top]
        ten_vt = [ten_vt[i] for i in top]
        print(f"  (giới hạn còn {max_ung_vien} ứng viên mạnh nhất theo |z| thô, "
              f"để SPA khả thi)")
    cac = ung_vien_tu_vi_tu(M, y, tr_mask, va_mask, kh_forecast)
    y_va = y[va_mask]
    Lcac = np.column_stack([loss_log3(P, y_va) for P in cac])
    nen_va = kh_forecast[va_mask]
    kh_tr = np.bincount(y[tr_mask & (y >= 0)], minlength=3).astype(float)
    kh_tr = kh_tr / kh_tr.sum()
    thieu = ~np.isfinite(nen_va).all(1)
    nen_va = np.where(thieu[:, None], kh_tr[None, :], nen_va)
    L_nen = loss_log3(nen_va, y_va)
    p, T_SPA = spa_test(L_nen, Lcac, B=1000, block=5, seed=7)
    dbar = (L_nen[:, None] - Lcac).mean(0)
    tot_i = int(np.argmax(dbar))
    print(f"  n(kiểm định)={len(y_va):,}  T_SPA={T_SPA:.4f}  p={p:.4f}  "
          f"{'BÁC BỎ H0' if p < 0.05 else 'không bác bỏ H0'}")
    print(f"  ứng viên tốt nhất (thô): {ten_vt[tot_i]}  Δlog={dbar[tot_i]:+.5f}")
    return dict(n_ung_vien=M.shape[0], n_va=int(len(y_va)), p=p, T_SPA=T_SPA,
                bac_bo=bool(p < 0.05), ung_vien_tot_nhat=ten_vt[tot_i],
                dbar_tot=float(dbar[tot_i]))


def main():
    print("=" * 100)
    print("SPA cho từng họ Giai đoạn 2 — so với nền \"chỉ σ̂\"")
    print("=" * 100)
    du = nap_du_lieu()
    Ms, sigs, zs, dts = du["Ms"], du["sigs"], du["zs"], du["dts"]
    y, tr, va = du["y"], du["tr"], du["va"]

    print("Đang dựng nền \"chỉ σ̂\" (dịch 1 phiên, đúng chỉ số hàng)…", flush=True)
    kh_forecast = nen_chi_sigma(Ms, sigs, dts)

    ket = {}

    print("\nĐang dựng lại vị từ H2 (motif)…", flush=True)
    M2, ten2, hop_le2 = vi_tu_motif(zs, dts)
    ket["H2_motif"] = chay_spa("H2 — motif", M2 & hop_le2[None, :], ten2, y, tr, va,
                               kh_forecast)

    print("\nĐang dựng lại vị từ H3 (rule-list)…", flush=True)
    from sklearn.tree import DecisionTreeClassifier
    X, ten_dt, ok = xay_cay(zs, dts, Ms, sigs, tr)
    cay = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LA,
                                  random_state=0)
    cay.fit(X[tr & ok], y[tr & ok])
    la = cay.apply(np.where(np.isfinite(X), X, 0.0))
    la_id = sorted(set(la[tr & ok].tolist()))
    M3 = np.array([(la == lid) & ok for lid in la_id])
    ten3 = [f"lá CART #{i}" for i in range(len(la_id))]
    ket["H3_rulelist"] = chay_spa("H3 — rule-list", M3, ten3, y, tr, va, kh_forecast)

    print("\nĐang dựng lại vị từ H5 (chế độ tự tương quan)…", flush=True)
    lit_all, ten_lit = [], None
    for i, p in enumerate(B.PAIRS):
        F = dac_trung(Ms[i], sigs[i], zs[i])
        tri = doan(dts[i]) == 0
        L, tn = roi_rac(F, tri)
        lit_all.append(L)
        ten_lit = tn
    lit = np.concatenate(lit_all, axis=1)
    M0, ten0 = vet_can(lit, ten_lit)
    che_do = che_do_tu_tuong_quan(Ms, dts)
    M5 = np.concatenate([M0 & (che_do == v)[None, :] for v in range(N_CD)], axis=0)
    ten5 = [f"{t} [{CD_TEN[v]}]" for v in range(N_CD) for t in ten0]
    ket["H5_chedo"] = chay_spa("H5 — chế độ tự tương quan", M5, ten5, y, tr, va,
                               kh_forecast, max_ung_vien=630)

    print("\nĐang dựng lại vị từ H6 (HMM)…", flush=True)
    from run_h6_hmm import vi_tu_hmm
    M6, ten6, _ = vi_tu_hmm(Ms, dts, du["pha"], im_lang=True)
    ket["H6_hmm"] = chay_spa("H6 — HMM", M6, ten6, y, tr, va, kh_forecast)

    print("\nĐang dựng lại vị từ H7 (Matrix Profile)…", flush=True)
    from run_h7_matrixprofile import vi_tu_analog
    M7, ten7 = vi_tu_analog(zs, dts, y, du["pha"], im_lang=True)
    ket["H7_matrixprofile"] = chay_spa("H7 — Matrix Profile", M7, ten7, y, tr, va,
                                       kh_forecast)

    # ── PHA 2 — ho dau tien co dung THONG TIN NGOAI GIA ────────────────
    print("\nĐang dựng lại vị từ H8 (nội dung thông cáo FOMC — PHA 2)…",
          flush=True)
    from run_h8_tintuc import nap_thong_cao, dac_trung_van_ban, vi_tu_tintuc
    tc = nap_thong_cao()
    if len(tc) >= 30:
        F8 = dac_trung_van_ban(tc)
        M8, ten8 = vi_tu_tintuc(F8, dts, tr, du["pha"])
        ket["H8_tintuc_FOMC"] = chay_spa("H8 — tin tức (FOMC)", M8, ten8, y,
                                         tr, va, kh_forecast)
    else:
        print("  bỏ qua — chưa có thông cáo; chạy collect/tin_tuc_nhtw.py trước")

    print("\nĐang dựng lại vị từ H8b (embedding ngữ nghĩa thông cáo FOMC — PHA 2)…",
          flush=True)
    from run_h8b_embedding import embed_thong_cao, vi_tu_embedding
    if len(tc) >= 30:
        F8b, cot_dt = embed_thong_cao(tc)
        M8b, ten8b = vi_tu_embedding(F8b, dts, tr, du["pha"], cot_dt)
        ket["H8b_embedding_FOMC"] = chay_spa("H8b — embedding (FOMC)", M8b, ten8b, y,
                                             tr, va, kh_forecast)
    else:
        print("  bỏ qua — chưa có thông cáo; chạy collect/tin_tuc_nhtw.py trước")

    print("\n" + "=" * 100)
    print("TỔNG KẾT SPA THEO HỌ (so nền \"chỉ σ̂\", kiểm soát nhiều ứng viên)")
    print(f"  {'họ':<20}{'n ứng viên':>12}{'p-value':>10}{'bác bỏ α=0,05':>15}")
    for k, v in ket.items():
        print(f"  {k:<20}{v['n_ung_vien']:>12,}{v['p']:>10.4f}"
              f"{'CÓ' if v['bac_bo'] else 'không':>15}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(ket, open(os.path.join(OUT, "spa_ho2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print("\n→ output/spa_ho2.json")


if __name__ == "__main__":
    main()
