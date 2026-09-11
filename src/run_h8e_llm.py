"""PHA 2 / H8e — TRICH XUAT CO CAU TRUC BANG LLM tu thong cao FOMC.

Day la HO E cua tuan 2 roadmap ("structured LLM extraction", danh dau
optional). Ba ho truoc da di het cac cach DEM va MA HOA MAY:

  H8   dem tu theo tu dien (HAWK/DOVE) + TF-IDF cosine + do dai   SPA p=0,162
  H8b  embedding pretrained + PCA                                  SPA p=0,861
  H8c  phan loai chu de theo tu dien                               0 quy luat

Ca ba deu KHONG doc hieu van ban — chung dem tu hoac do khoang cach vector.
Ho E hoi cau khac: mot mo hinh ngon ngu DOC HIEU toan van roi phan dinh co
bat duoc sac thai ma dem tu bo lo khong? Vi du sac thai ma tu dien khong
thay: "despite still-elevated inflation, participants saw scope for a more
gradual pace" — dem tu cho ra diem DIEU HAU (inflation, elevated) trong khi
y nghia tong the la BO CAU.

════════════════════════════════════════════════════════════════════════
SCHEMA — CHOT TRUOC KHI DOC BAT KY THONG CAO NAO
════════════════════════════════════════════════════════════════════════
File nay duoc commit TRUOC khi nhan duoc sinh ra (xem lich su git). Bon
truong, moi truong mot bien phan loai, khong co truong so lien tuc — de
khong the "tinh chinh" thang do sau khi nhin ket qua:

  dieu_huong          dieu_hau | bo_cau | trung_lap | hon_hop
                      Lap truong TONG THE cua ca van ban, khong phai dem
                      tu. "hon_hop" danh cho thong cao co tin hieu hai
                      chieu ro rang (vd thua nhan lam phat cao NHUNG bao
                      hieu san sang noi long).

  bat_dinh            thap | trung_binh | cao
                      Muc do ngon ngu phong thu/ra dieu kien: "uncertain",
                      "closely monitor", "remains attentive to risks"...

  thay_doi_lap_truong co | khong
                      So voi ky TRUOC do: van ban co bao hieu mot chuyen
                      huong that (doi cum tu dinh huong, bo/them cam ket)
                      hay chi lap lai?

  phu_thuoc_du_lieu   manh | vua | yeu
                      "manh" = tu choi cam ket truoc, moi thu tuy du lieu.
                      "yeu"  = dua dinh huong tuong doi ro cho ky toi.

KHONG GIAN GIA THUYET: 4+3+2+3 = 12 muc x 2 cua so (1, 5 phien) = 24 vi tu
x 3 lop = 72 gia thuyet — liet ke day du, doc lap voi H8 (54), H8b (54),
H8c (24).

NGUOI GAN NHAN. Nhan do CHINH Claude (mo hinh chay phien lam viec nay) doc
toan van 129 thong cao va gan, khong qua API rieng. Ghi ro vi day la han
che can neu: nhan mang phan dinh chu quan cua mot mo hinh, KHONG tai lap
duoc chinh xac neu doi mo hinh/phien ban. De bu lai, toan bo nhan duoc luu
ra `data/tin_tuc/fomc_llm_nhan.json` de kiem toan va tai su dung — dung
yeu cau cua roadmap ("LLM khong duoc dung nhu opaque final oracle. Output
phai duoc luu de reproducible/audit").

CAN LE THOI GIAN: giong het H8/H8b/H8c — nhan chi ap tu phien KE TIEP ngay
hop tro di, co tu kiem rieng.

Chay:  python src/run_h8e_llm.py
Ghi:   output/h8e_llm.json
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
NHAN = os.path.join(ROOT, "data", "tin_tuc", "fomc_llm_nhan.json")

import balop as B                                             # noqa: E402
from run_h8_tintuc import nap_thong_cao                        # noqa: E402
from run_quyluat import (                                     # noqa: E402
    nap_du_lieu, z_lift, westfall_young, doi_chung, TEN_LOP,
    MIN_KHOP, LIFT_LOPO, MIN_CAP_DUONG, T_DIEU_KIEN, NPERM, KHOI, EPS,
)

CUA_SO = (1, 5)
MAX_PHU = 2.0 / 3.0

# SCHEMA CHOT TRUOC — xem docstring
TRUONG = {
    "dieu_huong": ("dieu_hau", "bo_cau", "trung_lap", "hon_hop"),
    "bat_dinh": ("thap", "trung_binh", "cao"),
    "thay_doi_lap_truong": ("co", "khong"),
    "phu_thuoc_du_lieu": ("manh", "vua", "yeu"),
}
NHAN_VIET = {
    "dieu_huong": "điều hướng", "bat_dinh": "bất định",
    "thay_doi_lap_truong": "đổi lập trường", "phu_thuoc_du_lieu": "phụ thuộc dữ liệu",
}


def nap_nhan():
    """Doc nhan da gan. Moi phan tu: {ngay, dieu_huong, bat_dinh, ...}."""
    if not os.path.exists(NHAN):
        return None
    d = json.load(open(NHAN, encoding="utf-8"))
    df = pd.DataFrame(d["nhan"])
    df["ngay"] = pd.to_datetime(df.ngay)
    return df.sort_values("ngay").reset_index(drop=True)


def gan_vao_phien(F, dts_cap, cua_so, truong):
    ng = pd.DatetimeIndex(dts_cap)
    ra = np.full(len(ng), None, dtype=object)
    for _, r in F.iterrows():
        gt = r.get(truong)
        if gt is None or (isinstance(gt, float) and np.isnan(gt)):
            continue
        sau = np.flatnonzero(ng > r.ngay)
        if len(sau) == 0:
            continue
        ra[sau[:cua_so]] = gt
    return ra


def _tu_kiem(F, dts_cap):
    """Nhan KHONG duoc xuat hien o chinh phien hop."""
    ng = pd.DatetimeIndex(dts_cap)
    xau = 0
    for truong in TRUONG:
        A = gan_vao_phien(F, dts_cap, 5, truong)
        for _, r in F.iterrows():
            gt = r.get(truong)
            if gt is None:
                continue
            i = np.flatnonzero(ng <= r.ngay)
            if len(i) and A[i[-1]] is not None and A[i[-1]] == gt:
                xau += 1
    return xau


def vi_tu_llm(F, dts, tr, pha):
    lit, ten = [], []
    for cs in CUA_SO:
        for truong, muc in TRUONG.items():
            A = np.concatenate([gan_vao_phien(F, dts[i], cs, truong)
                                for i in range(len(B.PAIRS))], axis=0)
            for m in muc:
                v = (A == m)
                if v[tr].sum() < 100:
                    continue
                lit.append(v)
                ten.append(f"{NHAN_VIET[truong]}={m} [{cs} phiên sau]")
    M = np.array(lit)
    phu = (M & pha[None, :]).sum(1) / max(int(pha.sum()), 1)
    rong = phu > MAX_PHU
    if rong.any():
        M, ten = M[~rong], [t for t, b in zip(ten, rong) if not b]
    return M, ten


def main():
    t0 = time.time()
    print("=" * 112)
    print("PHA 2 / H8e — TRÍCH XUẤT CÓ CẤU TRÚC BẰNG LLM (Historical + News)")
    print("=" * 112)

    F = nap_nhan()
    if F is None:
        print(f"CHƯA có {NHAN} — cần gán nhãn trước.")
        return
    tc = nap_thong_cao()
    print(f"{len(F)} thông cáo đã gán nhãn / {len(tc)} thông cáo có văn bản")
    for truong in TRUONG:
        print(f"\n  {NHAN_VIET[truong]}:")
        print("    " + F[truong].value_counts().to_string().replace("\n", "\n    "))

    d = nap_du_lieu()
    dts = d["dts"]
    y, cap = d["y"], d["cap"]
    kiem_soat, cum = d["kiem_soat"], d["cum"]
    tr, va, te, pha = d["tr"], d["va"], d["te"], d["pha"]

    print("\n\ntự kiểm rò rỉ — nhãn KHÔNG được xuất hiện ở chính phiên họp…",
          flush=True)
    xau = _tu_kiem(F, dts[0])
    print(f"  {xau} vi phạm  {'ĐẠT' if xau == 0 else '← RÒ RỈ'}")
    assert xau == 0, "nhãn rò rỉ vào phiên họp"

    M, ten = vi_tu_llm(F, dts, tr, pha)
    print()
    print(f"KHÔNG GIAN GIẢ THUYẾT H8e: {len(ten)} vị từ × 3 lớp = "
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
        print(f"\n      {'vị từ':<36}{'lớp':<10}{'n':>7}{'lift':>7}{'z':>8}{'t|đk':>8}")
        for u in sorted(ung, key=lambda x: -abs(x["z"]))[:10]:
            print(f"      {u['ten'][:34]:<36}{TEN_LOP[u['lop']]:<10}{u['n']:>7}"
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
    print(f"{'PHỄU H8e (trích xuất LLM từ thông cáo FOMC)':<46}{'còn lại':>10}")
    for nhan_, v in (("không gian giả thuyết", len(ten) * 3),
                     ("đủ số lần khớp", int(du_khop.sum())),
                     ("thô p<0,05", int(tho.sum())),
                     ("sống sót Westfall–Young", int(song.sum())),
                     ("còn tin riêng sau đối chứng (đã khử lịch)", len(qua_dk)),
                     ("chuyển giao được (LOPO)", len(qua_lopo)),
                     ("tái lập trên KIỂM TRA", len(xn))):
        print(f"{nhan_:<46}{v:>10,}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(khong_gian=len(ten) * 3, du_khop=int(du_khop.sum()),
                    tho=int(tho.sum()), wy=int(song.sum()),
                    sau_dieu_kien=len(qua_dk), sau_lopo=len(qua_lopo),
                    xac_nhan=len(xn), quy_luat=xn, n_nhan=len(F),
                    cua_so=list(CUA_SO), schema={k: list(v) for k, v in TRUONG.items()}),
              open(os.path.join(OUT, "h8e_llm.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\nđã ghi output/h8e_llm.json — {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
