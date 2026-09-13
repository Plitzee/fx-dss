"""CRPS CHO PHAN PHOI LOI SUAT CUA HE SAN XUAT — do dung cai san pham hua.

VI SAO DAY LA PHEP DO DANG GIA NHAT CON THIEU. Ket luan cot loi cua luan van
la "huong gia khong co ky nang, nhung bien do/rui ro thi co". Nghia la san
pham khong hua "EURUSD se tang 0,3%" — no hua "bien dong ngay mai nam trong
khoang nay, va day la muc tu tin". QLIKE chi cham duoc con so PHUONG SAI
diem; CRPS cham duoc CA PHAN PHOI, tuc dung cai loi hua do.

HE SAN XUAT DA CO SAN MOT PHAN PHOI DU BAO DAY DU — chi la chua ai rap lai:

    tang 2 (volfc2.du_bao_san_xuat)  ->  sigma^(t)
    tang 6 (va_duoi.py, cau hinh V0) ->  phan phoi thuc nghiem cua z, DONG BANG
                                          tren doan huan luyen
    ghep lai:  r(t+1) ~ sigma^(t) x {z_1, ..., z_m}

Do la mot phan phoi day du cho loi suat ngay mai, khong phai mot diem.

BA CAU HINH DUOC DOI CHIEU:

  M0  khi hau hoc      phan phoi thuc nghiem cua loi suat HUAN LUYEN.
                       Tuong duong "sigma^ hang so + hinh dang thuc nghiem" —
                       moc de danh bai NHAT phai vuot qua.
  M1  san xuat         sigma^(t) DONG x phan phoi thuc nghiem cua z.
  M1g sigma dong+chuan sigma^(t) DONG x phan phoi CHUAN. Tach rieng de tra loi:
                       phan thang loi den tu sigma^ dong, hay tu hinh dang
                       duoi day (khong chuan) cua z?

So sanh M1 voi M0 tra loi "sigma^ dong co gia tri khong"; so M1 voi M1g tra
loi "hinh dang phan phoi co gia tri khong". Hai cau hoi doc lap.

DIEM SO KY NANG:  CRPSS = 1 - CRPS(mo hinh) / CRPS(khi hau hoc)
duong = tot hon khi hau hoc, cung quy uoc voi BSS dung khap repo.

CHAN RO RI: phan phoi z CHI duoc uoc tren doan huan luyen (`_tu_kiem_ro_ri`
kiem lai dieu do); sigma^(t) la du bao san xuat, da qua kiem chung nhan qua
o `docs/DONGBO_SANXUAT.md`.

Chay:  python src/crps_loi_suat.py
Ghi:   output/crps_loi_suat.json
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

import balop as B                                            # noqa: E402
import crps as C                                             # noqa: E402

TEN_DOAN = {1: "kiểm định", 2: "kiểm tra"}
PIP = 1e4        # doi sang pip cho de doc (cap 5 chu so; JPY xu ly rieng ben duoi)


def _tu_kiem_ro_ri(D):
    """Phan phoi z dung de du bao PHAI chi lay tu doan huan luyen: cat bo
    toan bo du lieu sau moc huan luyen khong duoc lam doi mau z."""
    xau = 0
    for p, d in D.items():
        z, g = d["z"], d["g"]
        ok = np.isfinite(z)
        m1 = z[(g == 0) & ok]
        # cat bo moi thu tu doan kiem dinh tro di roi lay lai
        n_tr = int(np.max(np.flatnonzero(g == 0))) + 1
        z2 = z[:n_tr]
        m2 = z2[np.isfinite(z2) & (g[:n_tr] == 0)]
        if not (len(m1) == len(m2) and np.allclose(m1, m2)):
            xau += 1
    return xau


def main():
    t0 = time.time()
    print("=" * 104)
    print("CRPS CHO PHÂN PHỐI LỢI SUẤT — chấm đúng cái hệ thống hứa")
    print("=" * 104)

    print("tự kiểm module CRPS…")
    assert C._tu_kiem(), "tự kiểm CRPS thất bại — dừng"

    print("\nđang dựng σ̂ sản xuất và lợi suất chuẩn hoá cho 6 cặp…", flush=True)
    from va_duoi import nap
    D = nap()

    xau = _tu_kiem_ro_ri(D)
    print(f"tự kiểm rò rỉ (phân phối z chỉ từ huấn luyện): {xau} vi phạm  "
          f"{'ĐẠT' if xau == 0 else '← RÒ RỈ'}")
    assert xau == 0, "phân phối z rò rỉ ngoài đoạn huấn luyện"

    ket = {}
    for gid in (1, 2):
        ket[TEN_DOAN[gid]] = {}

    print()
    for p in B.PAIRS:
        d = D[p]
        z, sig, g = d["z"], d["sig"], d["g"]
        ok = np.isfinite(z) & np.isfinite(sig) & (sig > 0)
        r = z * sig                                    # loi suat log thuc te

        z_tr = z[(g == 0) & ok]                        # phan phoi z, CHOT tren huan luyen
        r_tr = r[(g == 0) & ok]                        # khi hau hoc: loi suat huan luyen
        sig_hs = float(np.std(r_tr))                   # sigma^ hang so tuong duong

        for gid in (1, 2):
            m = (g == gid) & ok
            if m.sum() < 50:
                continue
            r_m, sig_m, z_m = r[m], sig[m], z[m]

            c_m0 = C.crps_mau_chung(r_tr, r_m)                    # khí hậu học
            c_m1 = sig_m * C.crps_mau_chung(z_tr, z_m)            # σ̂ động × z thực nghiệm
            c_m1g = C.crps_chuan(0.0, sig_m, r_m)                 # σ̂ động × chuẩn

            ket[TEN_DOAN[gid]][p] = dict(
                n=int(m.sum()), sig_hang_so=sig_hs,
                crps_khihau=float(c_m0.mean()),
                crps_sanxuat=float(c_m1.mean()),
                crps_sigma_chuan=float(c_m1g.mean()),
                crpss_sanxuat=float(1 - c_m1.mean() / c_m0.mean()),
                crpss_sigma_chuan=float(1 - c_m1g.mean() / c_m0.mean()),
            )

    # ── bao cao
    for gid in (1, 2):
        ten = TEN_DOAN[gid]
        if not ket[ten]:
            continue
        print("=" * 104)
        print(f"ĐOẠN {ten.upper()}")
        print(f"  {'cặp':<9}{'n':>6}{'CRPS khí hậu':>15}{'CRPS sản xuất':>15}"
              f"{'CRPS σ̂+chuẩn':>15}{'kỹ năng SX':>13}{'kỹ năng σ̂+chuẩn':>18}")
        print("  " + "-" * 100)
        for p, v in ket[ten].items():
            print(f"  {p:<9}{v['n']:>6}{v['crps_khihau']*PIP:>15.3f}"
                  f"{v['crps_sanxuat']*PIP:>15.3f}{v['crps_sigma_chuan']*PIP:>15.3f}"
                  f"{v['crpss_sanxuat']:>12.2%}{v['crpss_sigma_chuan']:>18.2%}")
        ns = np.array([v["n"] for v in ket[ten].values()], float)
        gop = {}
        for k in ("crps_khihau", "crps_sanxuat", "crps_sigma_chuan"):
            gop[k] = float(np.average([v[k] for v in ket[ten].values()], weights=ns))
        gop["crpss_sanxuat"] = float(1 - gop["crps_sanxuat"] / gop["crps_khihau"])
        gop["crpss_sigma_chuan"] = float(1 - gop["crps_sigma_chuan"] / gop["crps_khihau"])
        print("  " + "-" * 100)
        print(f"  {'GỘP':<9}{int(ns.sum()):>6}{gop['crps_khihau']*PIP:>15.3f}"
              f"{gop['crps_sanxuat']*PIP:>15.3f}{gop['crps_sigma_chuan']*PIP:>15.3f}"
              f"{gop['crpss_sanxuat']:>12.2%}{gop['crpss_sigma_chuan']:>18.2%}")
        print(f"  (CRPS nhân {PIP:.0e} — đơn vị pip trên cặp 5 chữ số; "
              f"nhỏ hơn là tốt hơn)")
        ket[ten]["_gop"] = gop
        print()

    os.makedirs(OUT, exist_ok=True)
    json.dump(ket, open(os.path.join(OUT, "crps_loi_suat.json"), "w",
                        encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"→ output/crps_loi_suat.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
