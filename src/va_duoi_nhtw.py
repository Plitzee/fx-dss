"""VA DUOI THEO CUA SO HOP NHTW — huong THU TU cho A1, dua tren chan doan
carry/BoJ moi (11/09/2026), sau khi ba huong truoc (V1/V2 vo dieu kien,
CAViaR dong, che do sigma^) deu that bai cho USDJPY (`CHISO_DANHGIA.md`
muc 5c, 5d, 9.2).

CHAN DOAN MOI DAN TOI HUONG NAY. Carry USDJPY (chenh lech lai suat, du
lieu co san o `data/carry.csv` qua `optimal_stop.carry_ngay`) nhay tu ~2%/
nam (2018-2022) len 5,18% (2023), giu 4,87%/3,36% (2024/2025) — dung giai
doan BoJ binh thuong hoa chinh sach, dung giai doan sd(z) USDJPY tang dot
bien (1,014 -> 1,136, huan luyen -> kiem tra). NHUNG carry cung da cao o
2022 (2,27%) ma khong vo duoi tuong tu — nen day nhieu kha nang la chuyen
"carry-unwind" QUANH NGAY HOP (nhu su kien thang 8/2024 that), khong phai
muc carry tron theo thoi gian.

Dieu do goi y DQ p=0,000 (vi pham du bao duoc tu vi pham truoc + tu muc
VaR — CHISO_DANHGIA muc 5c) co the la vi cau truc SU KIEN ROI RAC (ngay
hop NHTW/FOMC), khong phai dong luc TRON (CAViaR — da thu, khong an tien)
hay bac thang sigma^ (che do — da thu, khong an tien). Day la huong CHUA
thu: dieu kien hoa phan vi theo "co dang trong cua so K phien sau mot ky
hop NHTW (rieng cap) HOAC FOMC hay khong" — CHI HAI NHOM (khong phai ba
nhu che do), tranh dung loi phan manh mau da lam che do that bai.

CUA SO K=5 PHIEN (khoang mot tuan giao dich) — CHOT TRUOC, khong sua sau
khi nhin ket qua. Ngay hop + K phien sau duoc biet truoc CA NAM (lich NHTW
da cong bo), nen day khong phai ro ri — giong het cach `event=capday` da
dung cho tang bien dong san xuat.

HAI BIEN THE, CHON TREN KIEM DINH — dung KHUON va_duoi_chedo.py, chi doi
dinh nghia "che do" tu tam phan vi sigma^ (3 nhom) sang chi bao hop NHTW/
FOMC (2 nhom):
  V0 sự kiện   moi nhom mot phan vi CO DINH, uoc tren HUAN LUYEN
  V1 sự kiện   moi nhom mot phan vi CUA SO MO RONG, khop lai moi 21 phien

GIAO THUC MO DOAN — day la LAN MO THU TU cho van de VaR USDJPY (V0/V1/V2 la
lan 2, che do la lan ba nhung KHONG mo kiem tra vi thua tren kiem dinh). Nguoi
dung da xac nhan: NEU bien the moi thang tren KIEM DINH thi MO LUON KIEM TRA
de ket luan dut diem trong CUNG mot lan chay nay — khong quay lai sua sau khi
thay ket qua kiem tra (dung KHOA_SO.md muc 3).

Chay:  python src/va_duoi_nhtw.py
Ghi:   output/va_duoi_nhtw.json
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

import balop as B                                            # noqa: E402
import volfc2 as V2                                          # noqa: E402
import va_duoi as VD                                         # noqa: E402
from volfc import merge_thin_days                            # noqa: E402
from api.main import noi_chuoi                                # noqa: E402

MUC = (0.05, 0.01)
BUOC = 21
DAM = 750
DAM_SK = 100          # quan sat toi thieu TRONG nhom su kien de uoc rieng no
K_CUA_SO = 5           # phien sau ngay hop (CHOT TRUOC)
EPS = 1e-12


def nhom_su_kien(p, dates):
    """Chi bao (bool) — dang trong cua so K_CUA_SO phien KE TU (bao gom)
    ngay hop NHTW rieng cap HOAC FOMC. Lich biet truoc, khong ro ri."""
    d = pd.DatetimeIndex(dates)
    lich = V2.nap_lich(d)
    n = len(d)
    trong = np.zeros(n, bool)
    for ten_nh in ("fomc", V2.NHTW[p].lower()):
        idx = np.flatnonzero(lich[ten_nh] > 0)
        for i in idx:
            trong[i:i + K_CUA_SO] = True
    return trong


def _q_e(zz, a):
    q = float(np.quantile(zz, a))
    e = float(np.mean(zz[zz <= q])) if (zz <= q).any() else q
    return q, e


def phan_vi_v0_sk(z, g, sk, a):
    n = len(z)
    qz = np.full(n, np.nan); ez = np.full(n, np.nan)
    ok = np.isfinite(z)
    zt_all = z[(g == 0) & ok]
    q_gop, e_gop = _q_e(zt_all, a) if len(zt_all) else (np.nan, np.nan)
    for v in (False, True):
        zt = z[(g == 0) & ok & (sk == v)]
        q, e = _q_e(zt, a) if len(zt) >= DAM_SK else (q_gop, e_gop)
        m = ok & (sk == v)
        qz[m] = q; ez[m] = e
    return qz, ez


def phan_vi_v1_sk(z, g, sk, a, buoc=BUOC, dam=DAM, dam_sk=DAM_SK):
    n = len(z)
    qz = np.full(n, np.nan); ez = np.full(n, np.nan)
    ok = np.isfinite(z)
    for t0 in range(dam, n, buoc):
        cua = np.zeros(n, bool); cua[:t0] = True; cua &= ok
        if cua.sum() < dam // 2:
            continue
        t1 = min(t0 + buoc, n)
        q_gop, e_gop = _q_e(z[cua], a)
        for v in (False, True):
            m_sk = cua & (sk == v)
            q, e = _q_e(z[m_sk], a) if m_sk.sum() >= dam_sk else (q_gop, e_gop)
            dich = (sk[t0:t1] == v)
            if dich.any():
                qz[t0:t1][dich] = q
                ez[t0:t1][dich] = e
    return qz, ez


def _tu_kiem_khong_ro_ri(p, dates):
    """Chi bao su kien chi phu thuoc LICH (biet truoc ca nam) — cat du lieu
    SAU mot moc gia dinh khong duoc lam doi chi bao TRUOC moc do."""
    d = pd.DatetimeIndex(dates)
    moc = pd.Timestamp("2020-06-01")
    sk1 = nhom_su_kien(p, d)
    d2 = d[d < pd.Timestamp("2021-01-01")]
    sk2 = nhom_su_kien(p, d2)
    i1 = np.flatnonzero(d < moc)
    i2 = np.flatnonzero(d2 < moc)
    ok = np.array_equal(sk1[i1], sk2[i2])
    print(f"  tự kiểm không rò rỉ (chỉ báo họp NHTW): {'ĐẠT' if ok else 'THẤT BẠI'}")
    return ok


def main():
    mo_kiem_tra = "--kiem-tra" in sys.argv
    D = VD.nap()
    print("=" * 104)
    print("VÁ ĐUÔI THEO CỬA SỔ HỌP NHTW/FOMC — hướng thứ tư cho USDJPY")
    print("=" * 104)
    assert _tu_kiem_khong_ro_ri("USDJPY", noi_chuoi("USDJPY").Date.values), \
        "tự kiểm rò rỉ thất bại — dừng"

    print(f"\nα={MUC}, cửa sổ K={K_CUA_SO} phiên sau họp, "
          f"{'MỞ CẢ KIỂM ĐỊNH + KIỂM TRA' if mo_kiem_tra else 'CHỈ KIỂM ĐỊNH'}")

    ra = {"buoc": BUOC, "dam": DAM, "dam_sk": DAM_SK, "k_cua_so": K_CUA_SO,
          "muc": list(MUC), "kiem_dinh": {}}
    if mo_kiem_tra:
        ra["kiem_tra"] = {}
    kieu = {"V0 sự kiện": phan_vi_v0_sk, "V1 sự kiện": phan_vi_v1_sk}

    doans = [("kiem_dinh", 1)] + ([("kiem_tra", 2)] if mo_kiem_tra else [])

    for ten_doan, gid in doans:
        for a in MUC:
            print(f"\n{'─'*104}\n[{ten_doan}] α = {a}")
            print(f"  {'cặp':9}{'phương án':<14}{'n':>6}{'vi phạm':>9}"
                  f"{'Kupiec':>9}{'Chris':>8}{'DQ':>8}{'ES':>8}{'đạt':>6}"
                  f"   [mốc V1 vô đk]")
            ra[ten_doan][str(a)] = {}
            for p in B.PAIRS:
                z, sig, g = D[p]["z"], D[p]["sig"], D[p]["g"]
                dates = merge_thin_days(noi_chuoi(p)).Date.values
                sk = nhom_su_kien(p, dates)
                hang = {}
                for nhan, f in kieu.items():
                    qz, ez = f(z, g, sk, a)
                    r = VD.cham(z, sig, qz, ez, g == gid, a)
                    if r is None:
                        continue
                    hang[nhan] = r
                qz1, ez1 = VD.phan_vi_cuon(z, g, a, "mo_rong")
                moc = VD.cham(z, sig, qz1, ez1, g == gid, a)
                for nhan, r in hang.items():
                    print(f"  {p if nhan.startswith('V0') else '':9}{nhan:<14}"
                          f"{r['n']:>6}{r['vi_pham']:>9.4f}{r['kupiec']:>9.4f}"
                          f"{r['chris']:>8.4f}{r['dq']:>8.4f}{r['ty_le_es']:>8.3f}"
                          f"{'✓' if r['dat'] else '✗':>6}"
                          f"   (V1 vô đk: DQ={moc.get('dq', float('nan')) if moc else float('nan'):.4f}"
                          f" {'✓' if (moc or {}).get('dat') else '✗'})")
                print(flush=True)
                ra[ten_doan][str(a)][p] = {**hang, "V1 mốc": moc,
                                            "n_su_kien": int(sk.sum())}

    print("=" * 104)
    print("TỔNG KẾT — số cặp ĐẠT cả ba phép kiểm")
    for ten_doan, gid in doans:
        print(f"\n[{ten_doan}]  {'phương án':<14}{'α=0,05':>9}{'α=0,01':>9}")
        for bien in ("V1 mốc", "V0 sự kiện", "V1 sự kiện"):
            dat = []
            for a in MUC:
                d_ = sum(1 for p in B.PAIRS
                        if (ra[ten_doan][str(a)].get(p, {}).get(bien) or {}).get("dat"))
                dat.append(d_)
            print(f"  {bien:<14}{dat[0]:>6}/{len(B.PAIRS):<3}{dat[1]:>6}/{len(B.PAIRS):<3}")

    print("\nUSDJPY riêng — DQ theo phương án:")
    for ten_doan, gid in doans:
        print(f"  [{ten_doan}]")
        for a in MUC:
            for bien in ("V1 mốc", "V0 sự kiện", "V1 sự kiện"):
                r = ra[ten_doan][str(a)].get("USDJPY", {}).get(bien)
                if r:
                    print(f"    α={a} {bien:<14} DQ={r['dq']:.4f}  "
                          f"{'✓' if r['dat'] else '✗'}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "va_duoi_nhtw.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/va_duoi_nhtw.json")


if __name__ == "__main__":
    main()
