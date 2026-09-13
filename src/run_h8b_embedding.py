"""PHA 2 / H8b — EMBEDDING NGU NGHIA cua thong cao FOMC co noi them gi
ngoai BA dac trung thu cong (TF-IDF cosine, do dai, tu dien HAWK/DOVE) da
thu o `run_h8_tintuc.py` khong?

BOI CANH. `PHA2_TINTUC.md` muc 7 (han che) ghi ro: "Chi ba dac trung, khong
co embedding... Embedding chua thu" — dung RQ5 cua roadmap HuyH ("which forms
of text representation are most useful: sentiment, event category,
EMBEDDINGS"). File nay tra loi phan con thieu do, DUNG chinh 129 thong cao
FOMC da tai (`data/tin_tuc/fomc/`), KHONG can thu them du lieu moi (da kiem
tra ECB/BOE/BOJ deu doi hoi ha tang crawl JS/API rieng, chua lam trong file
nay — xem ghi chu cuoi).

PHUONG PHAP:
  1. Ma hoa MOI thong cao bang mo hinh cau (sentence-transformers,
     `all-MiniLM-L6-v2`, 384 chieu, pretrained — KHONG huan luyen them tren
     du lieu FX) — dung dinh nghia "embedding" cua RQ5, khac han TF-IDF (dem
     tu) va tu dien HAWK/DOVE (dem tu theo danh sach).
  2. Giam chieu bang PCA xuong 3 THANH PHAN CHINH — CHOT TRUOC, khong doi
     sau khi nhin ket qua. Ba la so du de nam phan lon phuong sai van ban ma
     khong lam khong gian gia thuyet phinh to (dung tinh than "khong search
     moi bien" cua roadmap).
  3. PCA duoc khop tren TOAN BO 129 thong cao (ca qua khu va tuong lai) —
     GIONG HET quy uoc da dung cho TF-IDF trong `run_h8_tintuc.py` (vector
     hoa tren toan bo van ban). Day KHONG PHAI ro ri gia FX: PCA chi dung
     VAN BAN (thuoc tinh tinh cua tai lieu da cong bo), khong dung bat ky
     RV/gia/nhan nao — nhan qua ve GIA chi bi vi pham neu dac trung dung
     THONG TIN GIA tuong lai, khong phai neu dung VAN BAN tuong lai (van
     ban cua ky hop thang 3/2020 khong the "ro ri" tu van ban thang 6/2020
     ve GIA thang 3/2020).
  4. Cung logic gan-vao-phien (chi tu phien KE TIEP), cung 2 cua so (1, 5),
     cung tam phan vi, cung bo kiem soat (kiem_soat + chi bao "trong cua so
     sau ky hop"), cung Westfall-Young + doi chung dieu kien + LOPO + xac
     nhan kiem tra nhu H8 va toan bo phau H1-H7.
  5. KHONG GIAN GIA THUYET: 3 PC x 3 o tam phan vi x 2 cua so = 18 vi tu x
     3 lop = 54 gia thuyet — DUNG BANG H8, liet ke day du truoc khi chay.

Day la MOT HO GIA THUYET DOC LAP (H8b), khong gop chung voi H8 — moi ho van
duoc hieu chinh Westfall-Young rieng, dung quy uoc da co cho H1-H8.

Chay:  python src/run_h8b_embedding.py
Ghi:   output/h8b_embedding.json
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

import balop as B                                             # noqa: E402
from run_h8_tintuc import nap_thong_cao                        # noqa: E402
from run_quyluat import (                                     # noqa: E402
    nap_du_lieu, z_lift, westfall_young, doi_chung, TEN_LOP,
    MIN_KHOP, LIFT_LOPO, MIN_CAP_DUONG, T_DIEU_KIEN, NPERM, KHOI, EPS,
)

CUA_SO = (1, 5)
MAX_PHU = 2.0 / 3.0
N_PC = 3                       # CHOT TRUOC — xem docstring
MO_HINH = "sentence-transformers/all-MiniLM-L6-v2"


def embed_thong_cao(tc):
    """Ma hoa van ban -> embedding -> PCA N_PC thanh phan. Tra ve
    DataFrame(ngay, pc0..pc{N_PC-1})."""
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    m = SentenceTransformer(MO_HINH)
    E = m.encode(tc.van_ban.tolist(), show_progress_bar=False,
                normalize_embeddings=True)
    pca = PCA(n_components=N_PC, random_state=0)
    Z = pca.fit_transform(E)
    print(f"  PCA {N_PC} thành phần giải thích "
          f"{pca.explained_variance_ratio_.sum():.1%} phương sai embedding")
    cot = {f"pc{k}": Z[:, k] for k in range(N_PC)}
    return pd.DataFrame(dict(ngay=tc.ngay, **cot)), list(cot)


def gan_vao_phien(F, dts_cap, cua_so, cot_dt):
    """Gan dac trung cua thong cao gan nhat vao cac phien SAU no (nhan qua:
    chi tu phien ke tiep, nhu run_h8_tintuc.gan_vao_phien)."""
    ng = pd.DatetimeIndex(dts_cap)
    ra = np.full((len(ng), len(cot_dt)), np.nan)
    for _, r in F.iterrows():
        if not np.isfinite(r[cot_dt].values.astype(float)).all():
            continue
        sau = np.flatnonzero(ng > r.ngay)
        if len(sau) == 0:
            continue
        ra[sau[:cua_so]] = r[cot_dt].values.astype(float)
    return ra


def _tu_kiem(F, dts_cap, cot_dt):
    ng = pd.DatetimeIndex(dts_cap)
    A = gan_vao_phien(F, dts_cap, 5, cot_dt)
    xau = 0
    for _, r in F.iterrows():
        if not np.isfinite(r[cot_dt].values.astype(float)).all():
            continue
        i = np.flatnonzero(ng <= r.ngay)
        if len(i) and np.isfinite(A[i[-1]]).any():
            if np.allclose(np.nan_to_num(A[i[-1]]),
                           np.nan_to_num(r[cot_dt].values.astype(float))):
                xau += 1
    return xau


def vi_tu_embedding(F, dts, tr, pha, cot_dt):
    lit, ten = [], []
    for cs in CUA_SO:
        A = np.concatenate([gan_vao_phien(F, dts[i], cs, cot_dt)
                            for i in range(len(B.PAIRS))], axis=0)
        for j, tdt in enumerate(cot_dt):
            v = A[:, j]
            vt = v[tr & np.isfinite(v)]
            if len(vt) < 100:
                continue
            q = np.quantile(vt, [1 / 3, 2 / 3])
            idx = np.where(np.isfinite(v), np.digitize(v, q), -1)
            for b in range(3):
                lit.append(idx == b)
                ten.append(f"embedding {tdt} {['thấp','vừa','cao'][b]} [{cs} phiên sau]")
    M = np.array(lit)
    phu = (M & pha[None, :]).sum(1) / max(int(pha.sum()), 1)
    rong = phu > MAX_PHU
    if rong.any():
        M, ten = M[~rong], [t for t, b in zip(ten, rong) if not b]
    return M, ten


def main():
    t0 = time.time()
    print("=" * 112)
    print("PHA 2 / H8b — EMBEDDING NGỮ NGHĨA thông cáo FOMC (Historical + News)")
    print("=" * 112)

    tc = nap_thong_cao()
    if len(tc) < 30:
        print(f"CHỈ có {len(tc)} thông cáo — chạy `python collect/tin_tuc_nhtw.py` trước.")
        return
    print(f"{len(tc)} thông cáo FOMC · {tc.ngay.min().date()} → {tc.ngay.max().date()}")
    print(f"đang mã hoá bằng {MO_HINH} (pretrained, không huấn luyện thêm)…", flush=True)
    F, cot_dt = embed_thong_cao(tc)

    d = nap_du_lieu()
    Ms, dts = d["Ms"], d["dts"]
    y, cap = d["y"], d["cap"]
    kiem_soat, cum = d["kiem_soat"], d["cum"]
    tr, va, te, pha = d["tr"], d["va"], d["te"], d["pha"]

    print("\ntự kiểm rò rỉ — đặc trưng KHÔNG được xuất hiện ở chính phiên họp…",
          flush=True)
    xau = _tu_kiem(F, dts[0], cot_dt)
    print(f"  {xau} vi phạm  {'ĐẠT' if xau == 0 else '← RÒ RỈ'}")
    assert xau == 0, "đặc trưng rò rỉ vào phiên họp"

    M, ten = vi_tu_embedding(F, dts, tr, pha, cot_dt)
    print()
    print(f"KHÔNG GIAN GIẢ THUYẾT H8b: {len(ten)} vị từ × 3 lớp = "
          f"{len(ten)*3} giả thuyết — liệt kê đầy đủ, biết trước")

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
        print(f"\n      {'vị từ':<44}{'lớp':<10}{'n':>7}{'lift':>7}{'z':>8}{'t|đk':>8}")
        for u in sorted(ung, key=lambda x: -abs(x["z"]))[:10]:
            print(f"      {u['ten'][:42]:<44}{TEN_LOP[u['lop']]:<10}{u['n']:>7}"
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
    print(f"{'PHỄU H8b (embedding thông cáo FOMC)':<46}{'còn lại':>10}")
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
                    cua_so=list(CUA_SO), mo_hinh=MO_HINH, n_pc=N_PC),
              open(os.path.join(OUT, "h8b_embedding.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/h8b_embedding.json — {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
