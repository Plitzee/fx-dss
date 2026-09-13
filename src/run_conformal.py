"""CONFORMAL PREDICTION — tu BA XAC SUAT thanh TAP DU BAO CO BAO DAM.

VAN DE MA NO GIAI. Repo da DO hieu chuan rat ky (ECE, MCE, PIT-KS, bieu do tin
cay). Nhung "do" khac "bao dam": ECE = 0,013 noi rang tren trung binh xac suat
khop tan suat, no KHONG hua gi cho mot phien cu the, va khong hua gi khi thi
truong doi che do.

Conformal prediction (Vovk et al.; Angelopoulos & Bates 2023) doi lai mot thu
khac: thay vi mot con so xac suat, no tra ve mot TAP NHAN kem BAO DAM DO PHU
huu han mau, KHONG can gia dinh phan phoi:

    thay vi   P(giam)=0,34  P(di ngang)=0,31  P(tang)=0,35
    tra ve    {giam, tang}  voi bao dam phu 90%

Vi sao dieu do dung cho DU AN NAY hon la mot mo hinh manh hon: bao dam do phu
KHONG CAN CO TIN HIEU moi co gia tri. Khi bat dinh lon, tap rong ra — va do
chinh la thong tin trung thuc cho nguoi ra quyet dinh. Kich thuoc tap trung
binh cho biet he thong THUC SU loai tru duoc bao nhieu: gan 3 nghia la khong
loai duoc gi, gan 2 nghia la loai duoc mot kha nang, va do la gia tri that.

HAI DIEM SO KHONG PHU HOP (nonconformity), ca hai deu chuan, CHOT TRUOC:

  LAC   s(x,y) = 1 - p_y                     (Sadinle et al. 2019)
        Tap NHO NHAT voi cung do phu bien duyen. Doi lai do phu co dieu kien
        theo lop kem hon.
  APS   s(x,y) = tong xac suat da sap giam den khi cham y  (Romano et al. 2020)
        Tap LON hon nhung do phu co dieu kien tot hon.

HAI GIAO THUC:

  TINH (split conformal)  q^ chot mot lan tren tap hieu chuan. Bao dam dung
        khi du lieu HOAN VI DUOC — ma chuoi thoi gian tai chinh thi KHONG.
  ACI   (Gibbs & Candes 2021) cap nhat alpha truc tuyen theo do phu da thuc
        hien:  alpha_{t+1} = alpha_t + gamma * (alpha_muc_tieu - err_t)
        Bao dam do phu DAI HAN duoi troi phan phoi tuy y — dung cai ma
        walk-forward (CHISO_DANHGIA.md muc 14) cho thay la co that o day.

TU KIEM (o `_tu_kiem()`), cai thu hai la cai quyet dinh:
  1. tren du lieu i.i.d., split conformal phu dung muc tieu
  2. duoi TROI PHAN PHOI co chu y, split conformal PHU HUT, con ACI keo duoc
     ve muc tieu — tuc phep so sanh nay co luc that

Chay:  python src/run_conformal.py
Ghi:   output/conformal.json
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
from run_walkforward import du_bao_san_xuat                   # noqa: E402

ALPHA = 0.10               # muc tieu: phu 90% — CHOT TRUOC
GAMMA = 0.01               # buoc hoc cua ACI — gia tri chuan, CHOT TRUOC
CUA_SO_HC = 500            # so diem hieu chuan cuon cho ACI
TEN_LOP = ("giảm", "đi ngang", "tăng")
EPS = 1e-12


from conformal import (diem_lac, diem_aps, DIEM, nguong,   # noqa: E402
                       chay_tinh, chay_aci, cham)          # noqa: E402


def _tu_kiem():
    """Hai tu kiem; cai thu hai chung minh phep so sanh co luc."""
    rng = np.random.default_rng(0)

    # (1) I.I.D. — split conformal phai phu dung muc tieu
    n = 8000
    Pt = rng.dirichlet([2.0, 2.0, 2.0], n)
    y = np.array([rng.choice(3, p=p) for p in Pt])
    hc, dg = slice(0, 4000), slice(4000, n)
    tap, _ = chay_tinh(Pt[hc], y[hc], Pt[dg], diem_lac)
    phu_iid, _, _ = cham(tap, y[dg])
    assert abs(phu_iid - (1 - ALPHA)) < 0.02, f"i.i.d. phu {phu_iid:.3f}"

    # (2) TROI PHAN PHOI — tinh phai HUT, ACI phai keo lai duoc
    # hieu chuan tren giai doan "de", danh gia tren giai doan "kho" (du bao
    # cua mo hinh tro nen te hon han vi lop that bi xao tron mot phan)
    P_hc = rng.dirichlet([6.0, 2.0, 2.0], 3000)
    y_hc = np.array([rng.choice(3, p=p) for p in P_hc])
    P_dg = rng.dirichlet([6.0, 2.0, 2.0], 3000)
    y_dg = np.array([rng.choice(3, p=p) for p in P_dg])
    xao = rng.random(3000) < 0.45                       # 45% phien "sai che do"
    y_dg[xao] = rng.integers(0, 3, int(xao.sum()))
    t_tinh, _ = chay_tinh(P_hc, y_hc, P_dg, diem_lac)
    phu_tinh, _, _ = cham(t_tinh, y_dg)
    t_aci, _ = chay_aci(P_hc, y_hc, P_dg, y_dg, diem_lac)
    phu_aci, _, _ = cham(t_aci, y_dg)
    assert phu_tinh < 1 - ALPHA - 0.02, f"troi phai lam TINH hut, {phu_tinh:.3f}"
    assert abs(phu_aci - (1 - ALPHA)) < abs(phu_tinh - (1 - ALPHA)), \
        f"ACI phai gan muc tieu hon TINH: {phu_aci:.3f} vs {phu_tinh:.3f}"
    return phu_iid, phu_tinh, phu_aci


def main():
    t0 = time.time()
    print("=" * 104)
    print(f"CONFORMAL PREDICTION — tập dự báo có BẢO ĐẢM độ phủ {1-ALPHA:.0%}")
    print("=" * 104)

    print("tự kiểm…", flush=True)
    p_iid, p_tinh, p_aci = _tu_kiem()
    print(f"  (1) i.i.d. — split conformal phủ {p_iid:.3f} (mục tiêu {1-ALPHA:.2f})  ĐẠT")
    print(f"  (2) TRÔI phân phối — TĨNH phủ {p_tinh:.3f} (HỤT) · "
          f"ACI phủ {p_aci:.3f} (kéo lại được)  ĐẠT")
    print("      → phép so sánh có lực thật, không phải hai cách cho cùng kết quả")

    ket = {}
    for h in B.HS:
        P, Pkh, y, cap, ngay, g = du_bao_san_xuat(h)
        print(f"\n{'─'*104}\nTẦM HẠN h = {h} phiên")

        # hieu chuan = cuoi HUAN LUYEN · danh gia = KIEM DINH  (giao thuc chinh)
        # xac nhan   = KIEM DINH        · danh gia = KIEM TRA
        for ten_bo, m_hc, m_dg in (
                ("kiểm định", (g == 0), (g == 1)),
                ("kiểm tra", (g <= 1), (g == 2))):
            P_hc, y_hc = P[m_hc][-3000:], y[m_hc][-3000:]
            P_dg, y_dg = P[m_dg], y[m_dg]
            ngay_dg, cap_dg = ngay[m_dg], cap[m_dg]
            print(f"\n  ── đánh giá trên {ten_bo} (n = {len(y_dg):,}) ──")
            print(f"  {'điểm số':<9}{'giao thức':<11}{'độ phủ':>9}{'kích thước TB':>15}"
                  f"{'tập rỗng':>10}")
            for ten_d, ham in DIEM.items():
                for gt in ("tĩnh", "ACI"):
                    if gt == "tĩnh":
                        tap, al = chay_tinh(P_hc, y_hc, P_dg, ham)
                    else:
                        tap, al = chay_aci(P_hc, y_hc, P_dg, y_dg, ham)
                    phu, kt, rong = cham(tap, y_dg)
                    sao = "" if abs(phu - (1 - ALPHA)) < 0.02 else "  ← lệch mục tiêu"
                    print(f"  {ten_d:<9}{gt:<11}{phu:>9.3f}{kt:>15.2f}"
                          f"{rong:>10.3f}{sao}")
                    ket[f"h{h}_{ten_bo}_{ten_d}_{gt}"] = dict(
                        do_phu=phu, kich_thuoc=kt, tap_rong=rong,
                        n=int(len(y_dg)))

                    if gt == "ACI" and ten_d == "APS" and ten_bo == "kiểm tra":
                        # do phu theo NAM — co giu duoc khi che do doi khong
                        du = pd.DataFrame(dict(nam=ngay_dg.year,
                                               dung=tap[np.arange(len(y_dg)), y_dg],
                                               kt=tap.sum(1)))
                        gr = du.groupby("nam").agg(n=("dung", "size"),
                                                   phu=("dung", "mean"),
                                                   kt=("kt", "mean"))
                        print("             độ phủ theo năm (APS + ACI):", end=" ")
                        print(" · ".join(f"{int(i)}: {rr.phu:.3f}"
                                         for i, rr in gr.iterrows()))
                        ket[f"h{h}_theo_nam"] = {str(int(i)): dict(
                            n=int(rr.n), do_phu=float(rr.phu),
                            kich_thuoc=float(rr.kt)) for i, rr in gr.iterrows()}

            # MOC SO SANH: chay DUNG thu tuc do tren KHI HAU HOC (hang so).
            # Khong co con so nay thi "kich thuoc 2,5" khong doc duoc — phai
            # biet mot du bao KHONG CO THONG TIN GI thi ra bao nhieu.
            for ten_d, ham in DIEM.items():
                tap0, _ = chay_aci(Pkh[m_hc][-3000:], y_hc, Pkh[m_dg], y_dg, ham)
                phu0, kt0, _ = cham(tap0, y_dg)
                r = ket[f"h{h}_{ten_bo}_{ten_d}_ACI"]
                r["kich_thuoc_khhoc"] = kt0
                r["thong_tin"] = kt0 - r["kich_thuoc"]
                print(f"  {'mốc khí hậu học ' + ten_d:<20}{phu0:>9.3f}"
                      f"{kt0:>15.2f}{'':>10}"
                      f"  → thông tin thật {kt0 - r['kich_thuoc']:+.2f} lớp")

    print("\n" + "=" * 104)
    print("ĐỌC BẢNG: độ phủ phải ≈ 0,90. Kích thước TB là thứ nói hệ thống biết")
    print("bao nhiêu — 3,00 nghĩa là không loại trừ được gì; càng gần 2,00 càng")
    print("loại được một khả năng, và ĐÓ mới là giá trị sử dụng thật.")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(alpha=ALPHA, gamma=GAMMA, ket_qua=ket,
                   tu_kiem=dict(iid=p_iid, troi_tinh=p_tinh, troi_aci=p_aci)),
              open(os.path.join(OUT, "conformal.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/conformal.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
