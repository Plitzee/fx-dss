"""A3 — BO CHON MO HINH THEO CHE DO (docs/KEHOACH_CAITIEN.md).

BANG CHUNG (output/log_balop.txt, trich vao KEHOACH_CAITIEN.md):
  BSS so khi hau hoc, theo TUNG CHE DO bien dong (tam phan vi cua sigma^):
                che do em     che do vua    che do cang
    h=1  chi sigma^     +0,0249       -0,0081        +0,0182
    h=20 sigma^+che do  +0,0326       -0,0154        +0,0270

Che do "vua" la cho DUY NHAT mo hinh dang chon THUA khi hau hoc, lap lai o ca
hai tam han. Dang dung MOT mo hinh cho ca ba che do nen phan lai o hai che do
ngoai dang bi che do giua an bot.

Y TUONG: chon theo tung PHIEN — o che do "vua" tra ve du bao KHI HAU HOC thay
vi du bao cua mo hinh san xuat (NEN_THEO_H trong api/main.py), giu nguyen mo
hinh san xuat o hai che do con lai. Vi BSS cua chinh khi hau hoc luon bang 0
(dinh nghia), thay the co dinh khong the LAM XAU hon o che do vua — cau hoi
that su la GOP LAI ca ba che do, mo hinh moi co THANG duoc mo hinh cu KHONG,
va co Y NGHIA khong (KTC ghep cap, khong phai chi nhin diem).

GIAO THUC: chon tren KIEM DINH, dung dung bo cham diem/KTC da co
(src/diem3.py), tai su dung `chuan_bi()` cua run_balop.py de dam bao cung
giao thuc khop/cham voi Giai doan 1 — khong tinh lai tu dau.

Chay:  python src/run_balop_chedo.py
Ghi:   output/balop_chedo.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                            # noqa: E402
import diem3 as D                                             # noqa: E402
from run_balop import chuan_bi, CHE_DO_TEN, KHOI               # noqa: E402
from api.main import NEN_THEO_H                                # noqa: E402


def main():
    print("=" * 100)
    print("A3 — CHỌN MÔ HÌNH THEO CHẾ ĐỘ: chế độ \"vừa\" rơi về khí hậu học")
    print("=" * 100)
    print("Mô hình sản xuất mỗi h (api/main.py NEN_THEO_H):",
          {h: NEN_THEO_H[h] for h in B.HS})
    print()

    ket = {}
    for h in B.HS:
        dat, _ = chuan_bi(h)
        print("─" * 100)
        print(f"TẦM HẠN h = {h} phiên  ·  mô hình sản xuất: {NEN_THEO_H[h]}")
        for mt in ("P", "R"):
            y = dat[mt]["y"]
            cap = dat[mt]["cap"]
            che_do = dat[mt]["che_do"]
            Pkh = dat[mt]["P"]["khí hậu học"]
            P_sx = dat[mt]["P"][NEN_THEO_H[h]]

            P_cd = P_sx.copy()
            vua = che_do == 1
            P_cd[vua] = Pkh[vua]

            r_sx = D.bang(P_sx, y, Pkh, nhom=cap)
            r_cd = D.bang(P_cd, y, Pkh, nhom=cap)
            blo_sx, bhi_sx = D.bss_ktc(P_sx, y, Pkh, nhom=cap, nboot=300,
                                       khoi=KHOI[h], seed=7)
            blo_cd, bhi_cd = D.bss_ktc(P_cd, y, Pkh, nhom=cap, nboot=300,
                                       khoi=KHOI[h], seed=7)
            dlo, dhi = D.delta_bss_ktc(P_cd, P_sx, y, Pkh, nhom=cap, nboot=300,
                                       khoi=KHOI[h], seed=7)

            print(f"\n  ── mục tiêu {mt} ── n={len(y):,}")
            print(f"  {'phương án':<26}{'log':>9}{'BSS':>9}{'KTC 95% BSS':>20}")
            print(f"  {'sản xuất (' + NEN_THEO_H[h] + ')':<26}"
                  f"{r_sx['log']:>9.4f}{r_sx['bss']:>+9.4f}"
                  f"{f'[{blo_sx:+.4f}, {bhi_sx:+.4f}]':>20}")
            print(f"  {'vừa → khí hậu học':<26}"
                  f"{r_cd['log']:>9.4f}{r_cd['bss']:>+9.4f}"
                  f"{f'[{blo_cd:+.4f}, {bhi_cd:+.4f}]':>20}")
            sao = " *" if (np.isfinite(dlo) and dlo > 0) else (
                  " (xấu hơn có ý nghĩa)" if (np.isfinite(dhi) and dhi < 0) else "")
            print(f"  Δ BSS (chọn theo chế độ − sản xuất), KTC ghép cặp: "
                  f"[{dlo:+.4f}, {dhi:+.4f}]{sao}")

            print(f"\n  theo từng chế độ:")
            for v in range(3):
                m = che_do == v
                if m.sum() < 200:
                    continue
                r1 = D.bang(P_sx[m], y[m], Pkh[m], nhom=cap[m])
                r2 = D.bang(P_cd[m], y[m], Pkh[m], nhom=cap[m])
                print(f"    {CHE_DO_TEN[v]:<12} n={int(m.sum()):>6,}  "
                      f"sản xuất BSS={r1['bss']:+.4f}   chọn-theo-chế-độ BSS={r2['bss']:+.4f}")

            ket[f"h{h}_{mt}"] = dict(
                san_xuat=dict(**r_sx, bss_lo=blo_sx, bss_hi=bhi_sx),
                vua_kh=dict(**r_cd, bss_lo=blo_cd, bss_hi=bhi_cd),
                delta_bss_lo=dlo, delta_bss_hi=dhi)
        print()

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "balop_chedo.json"), "w", encoding="utf-8") as f:
        json.dump(ket, f, ensure_ascii=False, indent=1, default=float)
    print("=" * 100)
    print("→ output/balop_chedo.json")


if __name__ == "__main__":
    main()
