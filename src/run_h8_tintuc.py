"""PHA 2 / H8 — NOI DUNG THONG CAO FOMC co noi them gi ngoai gia khong?

Day la ho gia thuyet dau tien cua PHA 2 (Historical + News) trong roadmap cua
HuyH. No tra loi RQ4 — "Does adding headline/news information improve
prediction beyond historical-only features?" — bang DUNG cai pheu bon cua ma
bay ho H1-H7 da di qua, khong phai bang mot phep so sanh de hon.

──────────────────────────────────────────────────────────────────────────
CAU HOI PHAI DAT CHO SAC — NEU KHONG SE CHI KHAM PHA LAI CAI LICH
──────────────────────────────────────────────────────────────────────────
Repo DA BIET ngay hop NHTW lam gia dong manh hon: 18 loai su kien, ket luan
"su kien khuech dai bien do, khong chi ra chieu" (CHISO_DANHGIA.md muc 8). Nen
mot phep do ngay tho kieu "ngay co FOMC thi bien do lon hon" se DUONG, va no
khong noi len dieu gi moi ca — do la hieu ung LICH, da do roi.

Cau hoi that cua Pha 2 la: **NOI DUNG van ban co noi them gi ngoai viec da co
mot cuoc hop khong?** Nen bo kiem soat cua ho nay duoc THEM mot bien: chi bao
"phien nay nam trong cua so sau mot ky hop FOMC". Vi tu chi song neu no con
tin rieng SAU KHI da khu ca sigma^, TSMOM, nhan to do-la, carry, VA chinh cai
lich hop.

──────────────────────────────────────────────────────────────────────────
BA DAC TRUNG VAN BAN — CHOT TRUOC KHI CHAY
──────────────────────────────────────────────────────────────────────────
  thay_doi_cau_chu   1 - cosine(TF-IDF ky nay, TF-IDF ky truoc)
                     Do do LECH NGON NGU giua hai thong cao lien tiep. Day la
                     thuoc do kinh dien cua "chuyen huong chinh sach" (Acosta
                     2015; Hansen, McMahon & Prat 2018) va khong can tu dien
                     cam xuc nao — no chi hoi "lan nay ho noi khac lan truoc
                     bao nhieu".
  doi_do_dai         (so tu ky nay - ky truoc) / ky truoc
                     Thong cao dai them thuong di kem giai thich nhieu hon,
                     tuc bat dinh cao hon.
  giong_dieu         (dem tu DIEU HUONG THAT CHAT - tu NOI LONG) / tong tu
                     Tu dien nho, LIET KE DAY DU o HAWK/DOVE ben duoi. Chot
                     truoc, khong sua sau khi nhin ket qua.

KHONG GIAN GIA THUYET, LIET KE DAY DU:
  3 dac trung x 3 o tam phan vi x 2 cua so (1 va 5 phien sau thong cao)
  = 18 vi tu x 3 lop = 54 gia thuyet.

CAN LE THOI GIAN — CHO DE SAI NHAT. Thong cao ra 14:00 gio New York ngay hop.
Nen dac trung chi duoc ap tu phien KE TIEP tro di, khong bao gio ap cho chinh
phien hop. `_tu_kiem()` kiem dung dieu do.

Chay:  python src/run_h8_tintuc.py
Ghi:   output/h8_tintuc.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
TIN = os.path.join(ROOT, "data", "tin_tuc", "fomc")

import balop as B                                             # noqa: E402
from split import doan                                        # noqa: E402
from run_quyluat import (                                     # noqa: E402
    nap_du_lieu, z_lift, westfall_young, doi_chung, TEN_LOP,
    MIN_KHOP, LIFT_LOPO, MIN_CAP_DUONG, T_DIEU_KIEN, NPERM, KHOI, EPS,
)

CUA_SO = (1, 5)            # so phien sau thong cao ma dac trung con hieu luc
MAX_PHU = 2.0 / 3.0        # tran do phu, cung ly do nhu run_h6_hmm.py

# Tu dien CHOT TRUOC — liet ke day du, khong sua sau khi nhin ket qua.
HAWK = ("inflation", "inflationary", "tighten", "tightening", "restrictive",
        "raise", "raising", "increase", "increases", "elevated", "firm",
        "firmer", "overheating", "price pressures")
DOVE = ("accommodative", "accommodation", "ease", "easing", "lower", "lowering",
        "cut", "cuts", "weak", "weakened", "weakness", "downside", "moderate",
        "moderated", "slack", "soften", "softened")


def nap_thong_cao():
    """Tra ve DataFrame(ngay, van_ban) da sap theo ngay."""
    if not os.path.isdir(TIN):
        return pd.DataFrame(columns=["ngay", "van_ban"])
    hang = []
    for f in sorted(os.listdir(TIN)):
        if not f.endswith(".txt"):
            continue
        t = open(os.path.join(TIN, f), encoding="utf-8").read().strip()
        if len(t) > 500:
            hang.append(dict(ngay=pd.Timestamp(f[:-4]), van_ban=t))
    return pd.DataFrame(hang).sort_values("ngay").reset_index(drop=True)


def dac_trung_van_ban(tc):
    """Ba dac trung cho MOI thong cao, tinh tu no va thong cao LIEN TRUOC."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vt = TfidfVectorizer(stop_words="english", max_features=4000,
                         ngram_range=(1, 2))
    X = vt.fit_transform(tc.van_ban.tolist())
    n = len(tc)
    thay_doi = np.full(n, np.nan)
    for i in range(1, n):
        a, b = X[i].toarray().ravel(), X[i - 1].toarray().ravel()
        cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))
        thay_doi[i] = 1.0 - cos

    so_tu = tc.van_ban.str.split().str.len().values.astype(float)
    doi_dai = np.full(n, np.nan)
    doi_dai[1:] = (so_tu[1:] - so_tu[:-1]) / np.maximum(so_tu[:-1], 1.0)

    giong = np.empty(n)
    for i, t in enumerate(tc.van_ban):
        w = t.lower()
        h = sum(w.count(k) for k in HAWK)
        d = sum(w.count(k) for k in DOVE)
        giong[i] = (h - d) / max(len(w.split()), 1)

    return pd.DataFrame(dict(ngay=tc.ngay, thay_doi_cau_chu=thay_doi,
                             doi_do_dai=doi_dai, giong_dieu=giong))


def gan_vao_phien(F, dts_cap, cua_so):
    """Gan dac trung cua thong cao gan nhat vao cac phien SAU no.

    NHAN QUA: thong cao ra 14:00 gio New York ngay hop, nen chi ap tu phien
    KE TIEP (`ngay_phien > ngay_thong_cao`), toi da `cua_so` phien."""
    ng = pd.DatetimeIndex(dts_cap)
    ra = np.full((len(ng), 3), np.nan)
    cot = ["thay_doi_cau_chu", "doi_do_dai", "giong_dieu"]
    for _, r in F.iterrows():
        if not np.isfinite(r[cot].values.astype(float)).all():
            continue
        sau = np.flatnonzero(ng > r.ngay)          # NGHIEM NGAT: khong lay ngay hop
        if len(sau) == 0:
            continue
        ra[sau[:cua_so]] = r[cot].values.astype(float)
    return ra


TEN_DT = ("thay đổi câu chữ", "đổi độ dài", "giọng điệu")


def vi_tu_tintuc(F, dts, tr, pha):
    """Dung vi tu tu dac trung van ban. Tach ra de run_spa_ho2.py dung lai."""
    lit, ten = [], []
    for cs in CUA_SO:
        A = np.concatenate([gan_vao_phien(F, dts[i], cs)
                            for i in range(len(B.PAIRS))], axis=0)
        for j, tdt in enumerate(TEN_DT):
            v = A[:, j]
            vt = v[tr & np.isfinite(v)]
            if len(vt) < 100:
                continue
            q = np.quantile(vt, [1 / 3, 2 / 3])   # nguong CHOT tren huan luyen
            idx = np.where(np.isfinite(v), np.digitize(v, q), -1)
            for b in range(3):
                lit.append(idx == b)
                ten.append(f"{tdt} {['thấp','vừa','cao'][b]} [{cs} phiên sau]")
    M = np.array(lit)
    phu = (M & pha[None, :]).sum(1) / max(int(pha.sum()), 1)
    rong = phu > MAX_PHU
    if rong.any():
        M, ten = M[~rong], [t for t, b in zip(ten, rong) if not b]
    return M, ten


def _tu_kiem(F, dts_cap):
    """Chan ro ri: dac trung KHONG duoc xuat hien o chinh phien hop, va khong
    duoc xuat hien o bat ky phien nao TRUOC do."""
    ng = pd.DatetimeIndex(dts_cap)
    A = gan_vao_phien(F, dts_cap, 5)
    xau = 0
    for _, r in F.iterrows():
        if not np.isfinite(r[["thay_doi_cau_chu"]].values.astype(float)).all():
            continue
        i = np.flatnonzero(ng <= r.ngay)           # ngay hop va truoc do
        if len(i) and np.isfinite(A[i[-1]]).any():
            # phien cuoi cung <= ngay hop co dac trung? chi sai neu dung tu
            # CHINH thong cao nay — kiem bang cach so gia tri
            if np.allclose(np.nan_to_num(A[i[-1]]),
                           np.nan_to_num(r[["thay_doi_cau_chu", "doi_do_dai",
                                            "giong_dieu"]].values.astype(float))):
                xau += 1
    return xau


def main():
    t0 = time.time()
    print("=" * 112)
    print("PHA 2 / H8 — NỘI DUNG THÔNG CÁO FOMC (Historical + News)")
    print("=" * 112)

    tc = nap_thong_cao()
    if len(tc) < 30:
        print(f"CHỈ có {len(tc)} thông cáo — chạy `python collect/tin_tuc_nhtw.py` trước.")
        return
    print(f"{len(tc)} thông cáo FOMC · {tc.ngay.min().date()} → {tc.ngay.max().date()}")
    F = dac_trung_van_ban(tc)
    du = F.dropna()
    print(f"{len(du)} thông cáo có đủ ba đặc trưng (bản đầu thiếu vì không có kỳ trước)")
    print(f"  thay đổi câu chữ : trung vị {du.thay_doi_cau_chu.median():.3f} · "
          f"khoảng [{du.thay_doi_cau_chu.min():.3f}; {du.thay_doi_cau_chu.max():.3f}]")
    print(f"  đổi độ dài       : trung vị {du.doi_do_dai.median():+.3f}")
    print(f"  giọng điệu       : trung vị {du.giong_dieu.median():+.5f}")

    d = nap_du_lieu()
    Ms, dts = d["Ms"], d["dts"]
    y, cap = d["y"], d["cap"]
    kiem_soat, cum = d["kiem_soat"], d["cum"]
    tr, va, te, pha = d["tr"], d["va"], d["te"], d["pha"]

    print("\ntự kiểm rò rỉ — đặc trưng KHÔNG được xuất hiện ở chính phiên họp…",
          flush=True)
    xau = _tu_kiem(F, dts[0])
    print(f"  {xau} vi phạm  {'ĐẠT' if xau == 0 else '← RÒ RỈ'}")
    assert xau == 0, "đặc trưng rò rỉ vào phiên họp"

    M, ten = vi_tu_tintuc(F, dts, tr, pha)
    print()
    print(f"KHÔNG GIAN GIẢ THUYẾT H8: {len(ten)} vị từ × 3 lớp = "
          f"{len(ten)*3} giả thuyết — liệt kê đầy đủ, biết trước")

    # ── BO KIEM SOAT MANH HON: them chi bao "trong cua so sau ky hop" ────
    # Neu khong co bien nay, mot vi tu song sot chi dang noi "co hop FOMC" —
    # dieu repo DA BIET (muc 8). Them vao thi phep kiem hoi dung cau hoi cua
    # Pha 2: NOI DUNG co noi them gi ngoai viec DA CO mot cuoc hop khong.
    trong_cs = np.zeros(len(y), bool)
    off = np.concatenate([[0], np.cumsum([len(x) for x in dts])])
    for i in range(len(B.PAIRS)):
        ng = pd.DatetimeIndex(dts[i])
        for ngay_tc in F.ngay:
            sau = np.flatnonzero(ng > ngay_tc)[:max(CUA_SO)]
            trong_cs[off[i] + sau] = True
    ks2 = np.column_stack([kiem_soat, trong_cs.astype(float)])
    print(f"bộ kiểm soát: 4 biến gốc + chỉ báo lịch họp "
          f"({int(trong_cs.sum()):,} phiên trong cửa sổ sau họp)")
    print(f"phát hiện {int(pha.sum()):,} hàng · xác nhận {int(te.sum()):,} hàng\n")

    print(f"[1/4] Westfall–Young, {NPERM} hoán vị, null khối {KHOI} ngày…", flush=True)
    Z, L, nk, P, nguong = westfall_young(M, y, pha)
    print(f"      ngưỡng max|z| null khối: 90% {nguong[0]:.2f} · 95% {nguong[1]:.2f}")
    du_khop = nk >= MIN_KHOP
    song = (P < 0.05) & np.isfinite(Z)
    tho = (np.abs(np.nan_to_num(Z)) > 1.96) & np.isfinite(Z)
    print(f"      {int(du_khop.sum())}/{len(ten)} vị từ đủ {MIN_KHOP} lần khớp")
    print(f"      sống sót W-Y p<0,05: {int(song.sum())} / thô p<0,05: {int(tho.sum())}"
          f" (nếu toàn nhiễu kỳ vọng {0.05*np.isfinite(Z).sum():.0f})")

    qua_dk, qua_lopo, xn = [], [], []
    if song.sum() > 0:
        print(f"\n[2/4] Đối chứng có điều kiện (|t| > {T_DIEU_KIEN}, ĐÃ khử cả "
              f"lịch họp)…", flush=True)
        ung = []
        for i, c in zip(*np.where(song)):
            b, t = doi_chung(M[i], y, c, ks2, cum=cum)
            ung.append(dict(i=int(i), lop=int(c), ten=ten[i], n=int(nk[i]),
                            z=float(Z[i, c]), lift=float(L[i, c]),
                            p_wy=float(P[i, c]), b_dk=b, t_dk=t))
        qua_dk = [u for u in ung
                  if np.isfinite(u["t_dk"]) and abs(u["t_dk"]) > T_DIEU_KIEN]
        print(f"      {len(qua_dk)}/{len(ung)} vị từ còn tin riêng")
        print(f"\n      {'vị từ':<40}{'lớp':<10}{'n':>7}{'lift':>7}{'z':>8}{'t|đk':>8}")
        for u in sorted(ung, key=lambda x: -abs(x["z"]))[:10]:
            print(f"      {u['ten'][:38]:<40}{TEN_LOP[u['lop']]:<10}{u['n']:>7}"
                  f"{u['lift']:>7.3f}{u['z']:>8.2f}{u['t_dk']:>8.2f}")

        print(f"\n[3/4] Bỏ-một-cặp…", flush=True)
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
        print(f"      {len(qua_lopo)}/{len(qua_dk)} chuyển giao được")

        print("\n[4/4] Xác nhận trên KIỂM TRA…", flush=True)
        Zte, Lte, _ = z_lift(M, y, te)
        for u in qua_lopo:
            u["z_te"] = float(Zte[u["i"], u["lop"]])
            u["lift_te"] = float(Lte[u["i"], u["lop"]])
        xn = [u for u in qua_lopo if np.isfinite(u["z_te"]) and u["z_te"] > 1.96]
        print(f"      {len(xn)}/{len(qua_lopo)} tái lập")
    else:
        print("\n→ KHÔNG vị từ nào sống sót Westfall–Young.")

    print("\n" + "=" * 112)
    print(f"{'PHỄU H8 (nội dung thông cáo FOMC)':<46}{'còn lại':>10}")
    for nhan, v in (("không gian giả thuyết", len(ten) * 3),
                    ("đủ số lần khớp", int(du_khop.sum())),
                    ("thô p<0,05", int(tho.sum())),
                    ("sống sót Westfall–Young", int(song.sum())),
                    ("còn tin riêng sau đối chứng (đã khử lịch)", len(qua_dk)),
                    ("chuyển giao được (LOPO)", len(qua_lopo)),
                    ("tái lập trên KIỂM TRA", len(xn))):
        print(f"{nhan:<46}{v:>10,}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(khong_gian=len(ten) * 3, du_khop=int(du_khop.sum()),
                    tho=int(tho.sum()), wy=int(song.sum()),
                    sau_dieu_kien=len(qua_dk), sau_lopo=len(qua_lopo),
                    xac_nhan=len(xn), quy_luat=xn, n_thong_cao=len(tc),
                    cua_so=list(CUA_SO)),
              open(os.path.join(OUT, "h8_tintuc.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/h8_tintuc.json — {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
