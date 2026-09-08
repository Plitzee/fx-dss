"""FDR THAY FWER O CUA MOT CUA PHEU — co bat duoc nhieu quy luat that hon khong?

CAU HOI. `docs/KEHOACH_2026Q4.md` §1.2 tu lau da ghi "doi FWER sang FDR" nhung
chua bao gio chay. Pheu hien tai chan bang Westfall-Young: giu xac suat co DU
mot duong tinh gia <= 5%. Do la muc kiem soat NGHIEM NHAT co the. FDR chi giu
TY LE duong tinh gia trong so cai bao ra <= 5% — long hon, nen ve nguyen tac
manh hon.

NHUNG CO MOT CHO PHAI DO CHU KHONG DUOC DOAN. Voi DUNG MOT tin hieu that trong
ho m gia thuyet, nguong BH o hang 1 la alpha/m — tuc BANG Bonferroni. Va vi ho
gia thuyet nay khong the gia dinh PRDS (vi tu long nhau, ba lop tuong quan am)
nen phai dung BY, con chia them cho c(m) = 9,22. Tuc voi mot tin hieu, FDR
gan nhu chac chan YEU HON W-Y chu khong manh hon.

FDR chi an tien khi co NHIEU tin hieu that cung luc. Va do CHINH LA gia thuyet
dang tranh cai: neu that su co quy luat, no se khong xuat hien don doc — mot
vi tu that se keo theo cac vi tu long nhau va cac lop lan can. Nen bai kiem
phai chay CA HAI kich ban:

    MOT tin hieu   — kich ban kiem_pheu.py dang dung. Du bao: FDR thua.
    NAM tin hieu   — tiem vao 5 vi tu roi nhau. Du bao: FDR thang, va thang
                     cang nhieu khi so tin hieu cang lon.

Neu ket qua thuc te khac du bao thi du bao sai — bao cao so do.

TAI SAO DUNG LAI DUOC MA TRAN NULL. Giong het ly do o kiem_pheu.py: phan phoi
null cua |z| la tinh chat cua MA TRAN VI TU va cau truc khoi, khong phai cua
tin hieu ta tiem. Uoc mot lan (1.000 hoan vi) roi dung lai cho moi lan lap.

Chay:  python src/kiem_fdr.py
Ghi:   output/kiem_fdr.json
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

import run_quyluat as Q                                      # noqa: E402
import kiem_pheu as KP                                       # noqa: E402

LIFTS = (1.05, 1.10, 1.15, 1.20, 1.35)
N_LAP = 60
K_TIN_HIEU = (1, 5)         # so tin hieu that tiem dong thoi
ALPHA = 0.05
SEED = 12345


def tiem_nhieu(y, M, ung_vien, k, lift, pha, rng):
    """Tiem k tin hieu vao k vi tu ROI NHAU. Tra ve (y2, danh sach (i, c)).

    "Roi nhau" o day la roi nhau ve VI TU, khong phai ve hang khop — hai vi tu
    khac nhau van co the cung khop mot hang, va do la dung: quy luat that trong
    du lieu that cung chong len nhau nhu vay. Nhung tiem hai lan vao CUNG mot
    vi tu thi lan sau se de len lan truoc va do manh dat ra khong con doc duoc.
    """
    y2 = y.copy()
    dich = []
    for i in rng.choice(ung_vien, size=k, replace=False):
        c = int(rng.integers(0, 3))
        y2, _ = KP.tiem(y2, M[int(i)], c, lift, pha, rng)
        dich.append((int(i), c))
    return y2, dich


def cong_fdr(M, y, pha, lam, alpha=ALPHA, bang_bo=True):
    """Chay cong FDR, tra ve tap chi so phang bi bac bo."""
    Z, _, _ = Q.z_lift(M, y, pha)
    z = np.abs(np.nan_to_num(Z.ravel(), nan=0.0))
    bb, _, _ = Q.fdr_bh(Q.p_duoi_chuan(z, lam), alpha=alpha, bang_bo=bang_bo)
    return Z, set(np.where(bb)[0])


def kiem_xap_xi(Zb, lam):
    """Duoi chuan hieu chuan co khop null hoan vi o vung CA HAI cung do duoc?

    Day la buoc phai qua truoc khi duoc phep ngoai suy xuong 1e-6. So sanh o
    ba muc ma 1.000 hoan vi con phan giai duoc: 10%, 5%, 1%. Neu duoi chuan
    khop o day thi ngoai suy con co co so; neu lech nhieu thi ket qua FDR ben
    duoi phai doc voi dung mot hat muoi.
    """
    from scipy.stats import norm
    B = Zb.shape[0]
    ra = []
    for muc in (0.10, 0.05, 0.01):
        # phan vi (1 - muc) cua null hoan vi, chuan hoa theo lambda
        qn = np.quantile(Zb / lam[None, :], 1.0 - muc, axis=0)
        # phan vi tuong ung neu |z|/lambda that su ~ |N(0,1)|
        qc = norm.ppf(1.0 - muc / 2.0)
        ra.append({"muc": muc, "hoan_vi_trungvi": float(np.median(qn)),
                   "chuan": float(qc),
                   "lech_ty_le": float(np.median(qn) / qc - 1.0)})
    return ra, B


def main():
    t0 = time.time()
    print("=" * 100)
    print("FDR THAY FWER — đo lực ở cả hai kịch bản một tín hiệu và nhiều tín hiệu")
    print("=" * 100, flush=True)

    M, ten, y, kiem_soat, pha = KP.chuan_bi()
    nht = M.shape[0] * 3
    cm = np.log(nht) + 0.5772156649 + 1.0 / (2 * nht)
    print(f"{M.shape[0]:,} vị từ × 3 lớp = {nht:,} giả thuyết · "
          f"{int(pha.sum()):,} hàng phát hiện", flush=True)
    print(f"hằng số Benjamini-Yekutieli c(m) = {cm:.2f} "
          f"→ ngưỡng p hạng 1 = α/(m·c) = {ALPHA/(nht*cm):.2e}", flush=True)

    print(f"\n[0/3] Ma trận null {Q.NPERM} hoán vị khối (chạy MỘT lần)…",
          flush=True)
    _, _, _, _, ng, _, Zb = Q.westfall_young(M, y, pha, nperm=Q.NPERM,
                                             seed=SEED, tra_null=True)
    gtth = float(ng[1])
    lam = Q.he_so_phong(Zb)
    print(f"      giá trị tới hạn W-Y 95% = {gtth:.2f} · "
          f"null {Zb.shape[0]}×{Zb.shape[1]:,} ({Zb.nbytes/1e6:.0f} MB) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"      hệ số phồng λ: trung vị {np.median(lam):.3f} · "
          f"khoảng [{lam.min():.3f}, {lam.max():.3f}]", flush=True)

    ra = {"n_gia_thuyet": int(nht), "n_hang": int(pha.sum()), "alpha": ALPHA,
          "c_by": float(cm), "gia_tri_toi_han_wy": gtth,
          "lambda_trungvi": float(np.median(lam)),
          "lifts": list(LIFTS), "n_lap": N_LAP, "k_tin_hieu": list(K_TIN_HIEU)}

    # ── KIỂM CHỨNG XẤP XỈ ĐUÔI trước khi dám ngoại suy ──────────────────
    print(f"\n      kiểm chứng xấp xỉ đuôi (nơi cả hoán vị lẫn chuẩn đo được):",
          flush=True)
    kx, _ = kiem_xap_xi(Zb, lam)
    ra["kiem_xap_xi"] = kx
    print(f"      {'mức':>7}{'hoán vị':>10}{'chuẩn':>9}{'lệch':>9}", flush=True)
    for r in kx:
        print(f"      {r['muc']:>7.0%}{r['hoan_vi_trungvi']:>10.3f}"
              f"{r['chuan']:>9.3f}{r['lech_ty_le']:>8.1%}", flush=True)
    lech = max(abs(r["lech_ty_le"]) for r in kx)
    ra["lech_xap_xi_max"] = float(lech)
    nx = ("— chấp nhận được, ngoại suy có cơ sở" if lech < 0.05
          else "— LỚN, kết quả FDR bên dưới phải đọc dè dặt")
    print(f"      → lệch lớn nhất {lech:.1%} {nx}", flush=True)

    # Ngưỡng |z| tương đương của từng cổng, để so sánh trực tiếp
    from scipy.stats import norm
    zbh = float(np.median(lam)) * norm.ppf(1 - ALPHA / (2 * nht))
    zby = float(np.median(lam)) * norm.ppf(1 - ALPHA / (2 * nht * cm))
    ra["z_tuong_duong"] = {"wy_95": gtth, "bh_hang1": zbh, "by_hang1": zby}
    print(f"\n      ngưỡng |z| tương đương ở HẠNG 1 (một tín hiệu duy nhất):",
          flush=True)
    print(f"        W-Y 95%  {gtth:.2f}   ← cổng hiện tại", flush=True)
    print(f"        BH       {zbh:.2f}", flush=True)
    print(f"        BY       {zby:.2f}   ← chặt hơn cả hai, đúng như dự báo",
          flush=True)

    # ── ĐỐI CHỨNG ÂM cho cổng FDR ───────────────────────────────────────
    print(f"\n[1/3] ĐỐI CHỨNG ÂM cổng BY — xáo trộn khối, phải ra ~0", flush=True)
    rng = np.random.default_rng(SEED)
    am = []
    for lap in range(KP.N_AM):
        ys = KP.xao_tron_khoi(y, Q.KHOI, rng)
        _, bb = cong_fdr(M, ys, pha, lam)
        q = 0
        for j in bb:
            i, c = divmod(j, 3)
            _, t = Q.doi_chung(M[i], ys, c, kiem_soat)
            q += int(np.isfinite(t) and abs(t) > Q.T_DIEU_KIEN)
        am.append({"lan": lap, "bac_bo_by": len(bb), "qua_dieu_kien": q})
        print(f"      lần {lap+1:>2}: BY bác bỏ {len(bb):>3} · "
              f"qua điều kiện hoá {q}", flush=True)
    ra["doi_chung_am"] = am
    print(f"      → trung bình "
          f"{np.mean([a['qua_dieu_kien'] for a in am]):.1f} dương tính giả",
          flush=True)

    # ── ĐỐI CHỨNG DƯƠNG: W-Y vs BY, ở k = 1 và k = 5 ────────────────────
    nk = np.array([(M[i] & pha).sum() for i in range(M.shape[0])])
    uv = np.array([i for i in np.where(nk >= KP.MIN_KHOP_TIEM)[0]
                   if not Q.la_vi_tu_nen(ten[i])])
    print(f"\n[2/3] ĐỐI CHỨNG DƯƠNG — {len(uv):,} vị từ đủ khớp làm giá đỡ",
          flush=True)

    duong = []
    for k in K_TIN_HIEU:
        print(f"\n      ── k = {k} tín hiệu thật tiêm đồng thời ──", flush=True)
        print(f"      {'lift':>6}{'W-Y bắt':>12}{'lực W-Y':>10}"
              f"{'BY bắt':>10}{'lực BY':>9}{'BY bác bỏ':>12}", flush=True)
        for lift in LIFTS:
            bw = bb_ = 0
            nbb = []
            for lap in range(N_LAP):
                r2 = np.random.default_rng(SEED + 7919 * lap
                                           + 131 * k + int(lift * 1000))
                y2, dich = tiem_nhieu(y, M, uv, k, lift, pha, r2)
                Z2, tapby = cong_fdr(M, y2, pha, lam)
                nbb.append(len(tapby))
                for i, c in dich:
                    z = Z2[i, c]
                    qua_dk = None
                    # cổng W-Y: |z| > giá trị tới hạn, rồi điều kiện hoá
                    if np.isfinite(z) and abs(z) > gtth:
                        _, t = Q.doi_chung(M[i], y2, c, kiem_soat)
                        qua_dk = bool(np.isfinite(t) and abs(t) > Q.T_DIEU_KIEN)
                        bw += int(qua_dk)
                    # cổng BY: nằm trong tập bác bỏ, rồi CÙNG điều kiện hoá
                    if i * 3 + c in tapby:
                        if qua_dk is None:
                            _, t = Q.doi_chung(M[i], y2, c, kiem_soat)
                            qua_dk = bool(np.isfinite(t)
                                          and abs(t) > Q.T_DIEU_KIEN)
                        bb_ += int(qua_dk)
            n = N_LAP * k
            duong.append({"k": k, "lift": lift, "n_co_hoi": n,
                          "bat_wy": bw, "luc_wy": bw / n,
                          "bat_by": bb_, "luc_by": bb_ / n,
                          "by_bac_bo_tb": float(np.mean(nbb))})
            print(f"      {lift:>6.2f}{bw:>8}/{n:<4}{bw/n:>9.0%}"
                  f"{bb_:>7}/{n:<3}{bb_/n:>8.0%}{np.mean(nbb):>12.1f}",
                  flush=True)
    ra["doi_chung_duong"] = duong

    # ── KẾT LUẬN ────────────────────────────────────────────────────────
    print(f"\n[3/3] KẾT LUẬN", flush=True)

    def mdes(rows, khoa):
        """Lift nhỏ nhất đạt lực >= 80%, nội suy tuyến tính giữa hai mức."""
        rows = sorted(rows, key=lambda r: r["lift"])
        for a, b in zip(rows, rows[1:]):
            if a[khoa] < 0.8 <= b[khoa]:
                w = (0.8 - a[khoa]) / max(b[khoa] - a[khoa], 1e-9)
                return a["lift"] + w * (b["lift"] - a["lift"])
        return float("nan") if rows[-1][khoa] < 0.8 else rows[0]["lift"]

    ra["mdes"] = {}
    print(f"      {'k':>3}{'MDES W-Y':>12}{'MDES BY':>11}{'':>4}kết luận")
    for k in K_TIN_HIEU:
        r = [d for d in duong if d["k"] == k]
        mw, mb = mdes(r, "luc_wy"), mdes(r, "luc_by")
        ra["mdes"][str(k)] = {"wy": mw, "by": mb}
        if not np.isfinite(mb):
            kl = "BY không đạt lực 80% ở mọi mức — thua"
        elif not np.isfinite(mw):
            kl = "BY đạt, W-Y không — THẮNG"
        else:
            kl = ("BY nhạy hơn" if mb < mw - 0.005 else
                  "W-Y nhạy hơn" if mw < mb - 0.005 else "hoà")
        print(f"      {k:>3}{mw:>12.3f}{mb:>11.3f}    {kl}")

    ra["giay"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, "kiem_fdr.json")
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(ra, fh, ensure_ascii=False, indent=1)
    print(f"\n→ output/kiem_fdr.json · {ra['giay']:.0f}s")


if __name__ == "__main__":
    main()
