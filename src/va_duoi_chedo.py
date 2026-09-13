"""VA DUOI THEO CHE DO — huong con lai cua A1 (docs/KEHOACH_CAITIEN.md).

BOI CANH. `va_duoi.py` da thu ba phuong an phan vi VO DIEU KIEN (V0 huan
luyen, V1 mo rong, V2 cuon) — ca ba deu that bai o USDJPY: DQ p = 0,000 KHONG
DOI qua ca ba, nghia la vi pham VaR DU BAO DUOC tu vi pham truoc + tu chinh
muc VaR. Do la dau hieu cua CAU TRUC DONG ma mot phan vi vo dieu kien (mot con
so cho ca cua so) khong the bat duoc du uoc lai bao nhieu lan.

`caviar.py` da thu huong dong luc tu hoi quy (CAViaR) — cung khong an tien
(loi the duy nhat khong song sot qua doi hat giong, xem CHISO_DANHGIA.md 9.2).

Huong CON LAI ma A1 de xuat: PHAN VI CO DIEU KIEN THEO CHE DO bien dong, dung
tinh than `SigmaCheDo` (balop.py) nhung ap cho DUOI thay vi ca phan phoi. Y
tuong: neu sd(z) USDJPY troi theo thoi gian vi CHE DO bien dong thay doi ty
trong (vd che do "cang" xuat hien nhieu hon o doan sau), thi mot phan vi
UOC RIENG TUNG CHE DO se tu dong bam theo, khong can biet "thoi gian" la truc
nao ca — khac voi V1/V2 chi biet cua so THOI GIAN.

CHE DO: tam phan vi cua sig (sigma^ HAR san xuat) tren HUAN LUYEN, giong het
cong thuc `SigmaCheDo.khop()` — 3 nhom [1/3, 2/3], nguong CHOT tren huan luyen,
ap dung nguyen cho ca chuoi (khong leak).

HAI BIEN THE, CHON TREN KIEM DINH:
  V0 chế độ   moi che do mot phan vi CO DINH, uoc tren HUAN LUYEN
  V1 chế độ   moi che do mot phan vi CUA SO MO RONG, khop lai moi 21 phien
              (neu mau trong che do qua mong thi lui ve phan vi CHUNG ca cua
              so — tranh phoi bay vao mot vai chuc quan sat)

GIAO THUC DOAN — day la LAN MO THU BA neu tinh ca CAViaR (V0/V1/V2 la lan 2,
CAViaR khong dung doan kiem tra nen khong tinh la mot lan mo). Chi cham tren
KIEM DINH o day. Doan KIEM TRA CHUA duoc mo — chi mo sau khi chot cau hinh va
ghi bien ban (docs/KHOA_SO.md muc 4), va CHI MOT LAN.

Chay:  python src/va_duoi_chedo.py
Ghi:   output/va_duoi_chedo.json
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
import va_duoi as VD                                         # noqa: E402
from metrics import kupiec, christoffersen_ind, dq_test      # noqa: E402

MUC = (0.05, 0.01)
BUOC = 21           # khop lai moi ~1 thang, cung giao thuc voi va_duoi.py
DAM = 750           # quan sat toi thieu truoc khi duoc uoc (ca cua so)
DAM_CD = 150        # quan sat toi thieu TRONG MOT CHE DO de uoc rieng no
N_CHE_DO = 3
EPS = 1e-12


def che_do_cua(sig, g):
    """Nguong tam phan vi CHOT tren huan luyen (g==0), ap cho ca chuoi.
    Giong het cong thuc SigmaCheDo.khop() o balop.py — khong doi de so sanh
    duoc truc tiep voi che do dang dung trong san xuat."""
    tr = sig[(g == 0) & np.isfinite(sig)]
    nguong = np.quantile(tr, [1 / 3, 2 / 3])
    ch = np.full(len(sig), -1, int)
    ok = np.isfinite(sig)
    ch[ok] = np.digitize(sig[ok], nguong)
    return ch, nguong


def _q_e(zz, a):
    q = float(np.quantile(zz, a))
    e = float(np.mean(zz[zz <= q])) if (zz <= q).any() else q
    return q, e


def phan_vi_v0_chedo(z, g, che_do, a):
    """Moi che do mot phan vi CO DINH, uoc tren HUAN LUYEN (g==0)."""
    n = len(z)
    qz = np.full(n, np.nan)
    ez = np.full(n, np.nan)
    ok = np.isfinite(z)
    zt_all = z[(g == 0) & ok]
    q_gop, e_gop = _q_e(zt_all, a) if len(zt_all) else (np.nan, np.nan)
    for v in range(N_CHE_DO):
        zt = z[(g == 0) & ok & (che_do == v)]
        q, e = _q_e(zt, a) if len(zt) >= DAM_CD else (q_gop, e_gop)
        m = ok & (che_do == v)
        qz[m] = q
        ez[m] = e
    return qz, ez


def phan_vi_v1_chedo(z, g, che_do, a, buoc=BUOC, dam=DAM, dam_cd=DAM_CD):
    """Moi che do mot phan vi CUA SO MO RONG, khop lai moi `buoc` phien.
    Neu mau trong che do trong cua so con qua mong (< dam_cd) thi lui ve
    phan vi GOP ca cua so (nhu V1 vo dieu kien) cho khoi phien do — tranh
    phan vi nhay lung tung vi mau vai chuc diem."""
    n = len(z)
    qz = np.full(n, np.nan)
    ez = np.full(n, np.nan)
    ok = np.isfinite(z)
    for t0 in range(dam, n, buoc):
        cua = np.zeros(n, bool)
        cua[:t0] = True
        cua &= ok
        if cua.sum() < dam // 2:
            continue
        t1 = min(t0 + buoc, n)
        q_gop, e_gop = _q_e(z[cua], a)
        for v in range(N_CHE_DO):
            mcd = cua & (che_do == v)
            if mcd.sum() >= dam_cd:
                q, e = _q_e(z[mcd], a)
            else:
                q, e = q_gop, e_gop
            dich = (che_do[t0:t1] == v)
            if dich.any():
                qz[t0:t1][dich] = q
                ez[t0:t1][dich] = e
    return qz, ez


def main():
    D = VD.nap()
    print("=" * 104)
    print("VÁ ĐUÔI THEO CHẾ ĐỘ — biến thể có điều kiện, CHỌN TRÊN KIỂM ĐỊNH")
    print("=" * 104)
    print("đoạn KIỂM TRA không mở ở đây (lần mở thứ ba, phải chốt trước)")

    ra = {"buoc": BUOC, "dam": DAM, "dam_cd": DAM_CD, "muc": list(MUC),
          "kiem_dinh": {}}
    kieu = {"V0 chế độ": phan_vi_v0_chedo, "V1 chế độ": phan_vi_v1_chedo}

    for a in MUC:
        print(f"\n{'─'*104}\nα = {a}")
        print(f"  {'cặp':9}{'phương án':<14}{'n':>6}{'vi phạm':>9}"
              f"{'Kupiec':>9}{'Chris':>8}{'DQ':>8}{'ES':>8}{'đạt':>6}"
              f"   [mốc V0/V1 vô đk]")
        ra["kiem_dinh"][str(a)] = {}
        for p in B.PAIRS:
            z, sig, g = D[p]["z"], D[p]["sig"], D[p]["g"]
            che_do, nguong = che_do_cua(sig, g)
            hang = {}
            for nhan, f in kieu.items():
                qz, ez = f(z, g, che_do, a)
                r = VD.cham(z, sig, qz, ez, g == 1, a)
                if r is None:
                    continue
                hang[nhan] = r
            # moc de doi chieu ngay tren cung dong: V0/V1 vo dieu kien
            m0 = {}
            for nhan_moc, kieu_moc in (("V0 mốc", "huan_luyen"), ("V1 mốc", "mo_rong")):
                qz, ez = VD.phan_vi_cuon(z, g, a, kieu_moc)
                r = VD.cham(z, sig, qz, ez, g == 1, a)
                if r:
                    m0[nhan_moc] = r
            for nhan, r in hang.items():
                moc = m0.get("V1 mốc", {})
                print(f"  {p if nhan.startswith('V0') else '':9}{nhan:<14}"
                      f"{r['n']:>6}{r['vi_pham']:>9.4f}{r['kupiec']:>9.4f}"
                      f"{r['chris']:>8.4f}{r['dq']:>8.4f}{r['ty_le_es']:>8.3f}"
                      f"{'✓' if r['dat'] else '✗':>6}"
                      f"   (V1 mốc: DQ={moc.get('dq', float('nan')):.4f}"
                      f" {'✓' if moc.get('dat') else '✗'})")
            print(flush=True)
            ra["kiem_dinh"][str(a)][p] = {**hang, **m0,
                                           "ngưỡng_chế_độ": [round(float(x), 4) for x in nguong]}

    # tong ket
    print("=" * 104)
    print("TỔNG KẾT — số cặp ĐẠT cả ba phép kiểm (Kupiec, Christoffersen, DQ)")
    print(f"  {'phương án':<14}{'α=0,05':>9}{'α=0,01':>9}")
    biens = ["V0 mốc", "V1 mốc", "V0 chế độ", "V1 chế độ"]
    ra["tong_ket"] = {}
    for bien in biens:
        dat = []
        for a in MUC:
            d = sum(1 for p in B.PAIRS
                    if ra["kiem_dinh"][str(a)].get(p, {}).get(bien, {}).get("dat"))
            dat.append(d)
        ra["tong_ket"][bien] = {"dat_05": dat[0], "dat_01": dat[1]}
        print(f"  {bien:<14}{dat[0]:>6}/{len(B.PAIRS):<3}{dat[1]:>6}/{len(B.PAIRS):<3}")

    print("\nUSDJPY riêng — DQ theo phương án (α=0,01):")
    for bien in biens:
        r = ra["kiem_dinh"]["0.01"].get("USDJPY", {}).get(bien)
        if r:
            print(f"  {bien:<14} DQ={r['dq']:.4f}  {'✓' if r['dat'] else '✗'}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "va_duoi_chedo.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1, default=float)
    print("\n→ output/va_duoi_chedo.json")


if __name__ == "__main__":
    main()
