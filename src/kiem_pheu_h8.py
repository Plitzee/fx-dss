"""MDES cho PHA 2 (H8 + H8b) — pheu tin tuc FOMC co du luc de loai tru hieu
ung dang quan tam khong, hay chi la "khong tim thay gi vi pheu qua chat"?

`kiem_pheu.py` da lam dung viec nay cho Giai doan 2 (D1). File nay ap DUNG
CACH LAM DO cho ho gia thuyet tin tuc (H8 dac trung thu cong + H8b embedding,
`PHA2_TINTUC.md` muc 6: SPA p=0,162, khong bac bo H0) — cau hoi dat ra o do
con bo ngo: "khong tim thay" vi THAT SU khong co hieu ung, hay vi 129 thong
cao la qua it de pheu bat duoc gi?

Dung LAI cac ham tong quat cua `kiem_pheu.py` (doi chung am/duong, gia tri
toi han, tiem tin hieu) — chi thay ham `chuan_bi()` bang du lieu cua H8+H8b.

Chay:  python src/kiem_pheu_h8.py
Ghi:   output/kiem_pheu_h8.json
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
import run_quyluat as Q                                       # noqa: E402
from kiem_pheu import (                                       # noqa: E402
    LIFTS, N_LAP, N_AM, NPERM_NGUONG, MIN_KHOP_TIEM, SEED,
    xao_tron_khoi, tiem, gia_tri_toi_han, qua_hai_cua,
)
from run_h8_tintuc import nap_thong_cao, dac_trung_van_ban, vi_tu_tintuc  # noqa: E402
from run_h8b_embedding import embed_thong_cao, vi_tu_embedding           # noqa: E402

CUA_SO = (1, 5)


def chuan_bi_h8():
    """Dung LAI du lieu chung (`run_quyluat.nap_du_lieu`), ghep vi tu cua
    CA H8 (dac trung thu cong) va H8b (embedding) thanh MOT khong gian gia
    thuyet — dung nhu `run_spa_ho2.py` da lam khi so sanh hai ho."""
    d = Q.nap_du_lieu()
    dts = d["dts"]

    tc = nap_thong_cao()
    F8 = dac_trung_van_ban(tc)
    M8, ten8 = vi_tu_tintuc(F8, dts, d["tr"], d["pha"])

    F8b, cot8b = embed_thong_cao(tc)
    M8b, ten8b = vi_tu_embedding(F8b, dts, d["tr"], d["pha"], cot8b)

    M = np.concatenate([M8, M8b], axis=0)
    ten = ten8 + ten8b

    # bo kiem soat: 4 bien goc + chi bao "trong cua so sau ky hop" — GIONG HET
    # ca run_h8_tintuc.py va run_h8b_embedding.py
    trong_cs = np.zeros(len(d["y"]), bool)
    off = np.concatenate([[0], np.cumsum([len(x) for x in dts])])
    for i in range(len(B.PAIRS)):
        ng = pd.DatetimeIndex(dts[i])
        for ngay_tc in tc.ngay:
            sau = np.flatnonzero(ng > ngay_tc)[:max(CUA_SO)]
            trong_cs[off[i] + sau] = True
    ks2 = np.column_stack([d["kiem_soat"], trong_cs.astype(float)])

    return M, ten, d["y"], ks2, d["pha"], d["cum"]


def main():
    t0 = time.time()
    print("=" * 100)
    print("MDES CHO PHA 2 (H8 + H8b) — tin tức FOMC còn bỏ ngỏ gì sau SPA p=0,162?")
    print("=" * 100, flush=True)

    M, ten, y, kiem_soat, pha, cum = chuan_bi_h8()
    ngt = M.shape[0] * 3
    print(f"{M.shape[0]:,} vị từ (H8 + H8b) × 3 lớp = {ngt:,} giả thuyết · "
          f"{int(pha.sum()):,} hàng phát hiện", flush=True)
    ra = {"ho": "H8+H8b (FOMC: đặc trưng thủ công + embedding)",
          "n_gia_thuyet": int(ngt), "n_hang": int(pha.sum()),
          "nperm_nguong": NPERM_NGUONG, "lifts": list(LIFTS),
          "n_lap": N_LAP, "t_dieu_kien": Q.T_DIEU_KIEN}

    print(f"\n[0/2] Giá trị tới hạn max|z| dưới null khối (khối {Q.KHOI})…",
          flush=True)
    gtth = gia_tri_toi_han(M, y, pha)
    ra["gia_tri_toi_han"] = gtth
    print(f"      giá trị tới hạn 95% = {gtth:.2f}", flush=True)

    print(f"\n[1/2] ĐỐI CHỨNG ÂM — xáo trộn khối kết cục, phễu phải ra ~0",
          flush=True)
    rng = np.random.default_rng(SEED)
    am = []
    for lap in range(N_AM):
        ys = xao_tron_khoi(y, Q.KHOI, rng)
        Zs, _, _ = Q.z_lift(M, ys, pha)
        vuot = np.where(np.isfinite(Zs) & (np.abs(Zs) > gtth))
        q = 0
        for i, c in zip(*vuot):
            _, t = Q.doi_chung(M[i], ys, c, kiem_soat, cum=cum)
            if np.isfinite(t) and abs(t) > Q.T_DIEU_KIEN:
                q += 1
        am.append({"lan": lap, "vuot_nguong": int(len(vuot[0])), "qua_dieu_kien": q})
        print(f"      lần {lap+1:>2}: vượt ngưỡng {len(vuot[0]):>3} · "
              f"qua điều kiện hoá {q}", flush=True)
    ra["doi_chung_am"] = am
    tb_am = float(np.mean([a["qua_dieu_kien"] for a in am]))
    print(f"      → trung bình {tb_am:.1f} dương tính giả / {ngt:,} giả thuyết",
          flush=True)

    print(f"\n[2/2] ĐỐI CHỨNG DƯƠNG — tiêm quy luật đã biết, {N_LAP} lần mỗi mức",
          flush=True)
    nk = np.array([(M[i] & pha).sum() for i in range(M.shape[0])])
    ung_vien = np.array([i for i in np.where(nk >= MIN_KHOP_TIEM)[0]
                         if not Q.la_vi_tu_nen(ten[i])])
    print(f"      {len(ung_vien):,} vị từ đủ ≥{MIN_KHOP_TIEM} lần khớp làm giá đỡ\n",
          flush=True)
    if len(ung_vien) == 0:
        print("      KHÔNG có vị từ nào đủ số lần khớp — mẫu 129 thông cáo quá")
        print("      nhỏ để ước MDES qua chính khung tiêm-tín-hiệu này.")
        ra["mdes_khong_kha_thi"] = True
    else:
        print(f"      {'lift đặt':>9}{'lift thực':>11}{'|z| trung vị':>14}"
              f"{'bắt được':>12}{'lực':>9}", flush=True)
        duong = []
        for lift in LIFTS:
            bat, lts, zs_ = 0, [], []
            for lap in range(N_LAP):
                r2 = np.random.default_rng(SEED + 7919 * lap + int(lift * 1000))
                i = int(r2.choice(ung_vien))
                c = int(r2.integers(0, 3))
                y2, lt = tiem(y, M[i], c, lift, pha, r2)
                lts.append(lt)
                ok, z = qua_hai_cua(M, y2, kiem_soat, pha, i, c, gtth, cum)
                zs_.append(abs(z))
                bat += int(ok)
            luc = bat / N_LAP
            duong.append({"lift": lift, "lift_that": float(np.nanmean(lts)),
                          "z_trungvi": float(np.nanmedian(zs_)),
                          "bat": bat, "luc": luc})
            print(f"      {lift:>9.2f}{np.nanmean(lts):>11.3f}"
                  f"{np.nanmedian(zs_):>14.2f}{bat:>8}/{N_LAP:<4}{luc:>8.0%}",
                  flush=True)
        ra["doi_chung_duong"] = duong
        dat = [d for d in duong if d["luc"] >= 0.80]
        mdes = dat[0]["lift"] if dat else None
        ra["lift_nho_nhat_luc80"] = mdes
        print("\n" + "=" * 100)
        if mdes is not None:
            print(f"HIỆU ỨNG NHỎ NHẤT PHÁT HIỆN ĐƯỢC (lực 80%): lift = {mdes:.2f}")
            print(f"→ SPA p=0,162 (H8) nay phát biểu được thành: phễu tin tức")
            print(f"  bắt được hiệu ứng lift ≥ {mdes:.2f} với xác suất ≥ 80%, nên mọi")
            print(f"  quy luật mạnh hơn thế trong nội dung FOMC đã bị loại trừ.")
        else:
            print(f"KHÔNG mức lift nào trong {LIFTS} đạt lực 80%.")
            print("→ Với 129 thông cáo, phễu KHÔNG đủ lực để loại trừ hiệu ứng")
            print("  yếu — kết luận âm của H8/H8b phải kèm câu bảo lưu này.")
    print(f"Dương tính giả trên nhiễu thuần: {tb_am:.1f}/{ngt:,}")
    print("=" * 100)

    with open(os.path.join(OUT, "kiem_pheu_h8.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1, default=float)
    print(f"đã ghi output/kiem_pheu_h8.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
