"""TAP TRUNG NGAN HAN — ky nang o h = 1 nam O DAU?

BOI CANH. Hai phep do doc lap da chi cung mot cho:

  walk-forward (CHISO_DANHGIA.md muc 14)   h=1: 14/14 nam BSS duong
                                           h=5: 10/14 · h=20: 8/14
  conformal    (CHISO_DANHGIA.md muc 16)   h=1: +0,10 lop thong tin
                                           h=5: -0,09 · h=20: -0,10

Ket luan: ky nang cua he thong nam o TAM HAN 1 PHIEN. Truoc khi dung san pham
quanh no, phai biet ky nang do PHAN BO THE NAO — neu no chi den tu mot hai cap
hoac mot che do, thi giao dien phai noi khac han so voi khi no deu.

Ba lat cat, deu tren muc tieu P (thu hien tren giao dien), mo hinh SAN XUAT
(to hop truc tuyen), cham tren KIEM DINH va KIEM TRA:

  1. theo CAP        — ky nang co deu 6 cap khong?
  2. theo CHE DO     — em / vua / cang thang
  3. thong tin conformal theo cap — kich thuoc tap so voi moc khi hau hoc

Chay:  python src/kiem_ngan_han.py
Ghi:   output/ngan_han.json
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

import balop as B                                             # noqa: E402
import diem3 as D                                             # noqa: E402
from run_walkforward import du_bao_san_xuat                   # noqa: E402
from run_conformal import chay_aci, diem_lac, cham            # noqa: E402

H = 1
CHE_DO_TEN = ("bình tĩnh", "vừa", "căng thẳng")


def che_do_tu_sig(P, ngay, cap, g):
    """Tam phan vi cua sigma^ tren HUAN LUYEN, rieng tung cap — cung cong thuc
    SigmaCheDo. Tra ve mang 0/1/2. Can sig, lay lai tu balop.nap()."""
    Bd = B.nap()
    ra = np.full(len(cap), -1, int)
    for p in B.PAIRS:
        d = Bd[p]
        from split import doan
        gp = doan(d.Date.values)
        sig = d.sig.values
        nguong = np.quantile(sig[gp == 0], [1 / 3, 2 / 3])
        m = cap == p
        # can theo NGAY vi du_bao_san_xuat da loc y >= 0
        idx = {np.datetime64(x, "D"): i for i, x in enumerate(d.Date.values)}
        vt = np.array([idx.get(np.datetime64(x, "D"), -1) for x in ngay[m]])
        s = np.where(vt >= 0, sig[np.clip(vt, 0, len(sig) - 1)], np.nan)
        ra[m] = np.where(np.isfinite(s), np.digitize(s, nguong), -1)
    return ra


def main():
    t0 = time.time()
    print("=" * 100)
    print(f"TẬP TRUNG NGẮN HẠN — kỹ năng ở h = {H} phiên nằm ở đâu?")
    print("=" * 100)

    P, Pkh, y, cap, ngay, g = du_bao_san_xuat(H)
    che_do = che_do_tu_sig(P, ngay, cap, g)
    ket = {}

    for ten_doan, m_doan in (("kiểm định", g == 1), ("kiểm tra", g == 2)):
        print(f"\n{'─'*100}\nĐOẠN {ten_doan.upper()} (n = {int(m_doan.sum()):,})")

        # ── 1. theo CAP ──────────────────────────────────────────────────
        print(f"\n  theo CẶP:")
        print(f"  {'cặp':<10}{'n':>7}{'log':>9}{'BSS':>9}{'KTC 95% của BSS':>22}"
              f"{'ECE':>8}{'AUC':>8}")
        theo_cap = {}
        for p in B.PAIRS:
            m = m_doan & (cap == p)
            if m.sum() < 100:
                continue
            r = D.bang(P[m], y[m], Pkh[m])
            lo, hi = D.bss_ktc(P[m], y[m], Pkh[m], nboot=300, khoi=20, seed=7)
            sao = " *" if np.isfinite(lo) and lo > 0 else ""
            print(f"  {p:<10}{int(m.sum()):>7,}{r['log']:>9.4f}{r['bss']:>+9.4f}"
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{r['ece']:>8.4f}"
                  f"{r['auc']:>8.4f}{sao}")
            theo_cap[p] = dict(**r, bss_lo=lo, bss_hi=hi)
        duong = sum(1 for v in theo_cap.values()
                    if np.isfinite(v["bss_lo"]) and v["bss_lo"] > 0)
        print(f"  → {duong}/{len(theo_cap)} cặp có BSS dương CÓ Ý NGHĨA "
              f"(KTC không phủ 0)")

        # ── 2. theo CHE DO ───────────────────────────────────────────────
        print(f"\n  theo CHẾ ĐỘ biến động:")
        print(f"  {'chế độ':<14}{'n':>7}{'log':>9}{'BSS':>9}"
              f"{'KTC 95% của BSS':>22}{'ECE':>8}")
        theo_cd = {}
        for v in range(3):
            m = m_doan & (che_do == v)
            if m.sum() < 100:
                continue
            r = D.bang(P[m], y[m], Pkh[m], nhom=cap[m])
            lo, hi = D.bss_ktc(P[m], y[m], Pkh[m], nhom=cap[m], nboot=300,
                               khoi=20, seed=7)
            sao = " *" if np.isfinite(lo) and lo > 0 else ""
            print(f"  {CHE_DO_TEN[v]:<14}{int(m.sum()):>7,}{r['log']:>9.4f}"
                  f"{r['bss']:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}"
                  f"{r['ece']:>8.4f}{sao}")
            theo_cd[CHE_DO_TEN[v]] = dict(**r, bss_lo=lo, bss_hi=hi)

        # ── 3. THONG TIN CONFORMAL theo cap ──────────────────────────────
        print(f"\n  THÔNG TIN CONFORMAL theo cặp (kích thước tập, LAC + ACI):")
        print(f"  {'cặp':<10}{'độ phủ':>9}{'mô hình':>10}{'khí hậu học':>14}"
              f"{'thông tin':>12}")
        m_hc = g == 0 if ten_doan == "kiểm định" else g <= 1
        theo_cap_cf = {}
        for p in B.PAIRS:
            mh = m_hc & (cap == p)
            md = m_doan & (cap == p)
            if md.sum() < 100 or mh.sum() < 200:
                continue
            t1, _ = chay_aci(P[mh][-1500:], y[mh][-1500:], P[md], y[md], diem_lac)
            t0_, _ = chay_aci(Pkh[mh][-1500:], y[mh][-1500:], Pkh[md], y[md],
                              diem_lac)
            phu1, kt1, _ = cham(t1, y[md])
            _, kt0, _ = cham(t0_, y[md])
            print(f"  {p:<10}{phu1:>9.3f}{kt1:>10.2f}{kt0:>14.2f}"
                  f"{kt0 - kt1:>+12.2f}")
            theo_cap_cf[p] = dict(do_phu=phu1, kich_thuoc=kt1,
                                  kich_thuoc_khhoc=kt0, thong_tin=kt0 - kt1)
        tt = np.array([v["thong_tin"] for v in theo_cap_cf.values()])
        print(f"  → {int((tt > 0).sum())}/{len(tt)} cặp có thông tin dương · "
              f"trung vị {np.median(tt):+.2f} lớp")

        ket[ten_doan] = dict(theo_cap=theo_cap, theo_che_do=theo_cd,
                             conformal_theo_cap=theo_cap_cf,
                             so_cap_duong=duong)

    print("\n" + "=" * 100)
    print("ĐỌC: nếu kỹ năng dồn vào 1–2 cặp hoặc 1 chế độ thì sản phẩm phải nói")
    print("rõ điều đó, không được trình bày như một con số chung cho cả sáu cặp.")

    os.makedirs(OUT, exist_ok=True)
    json.dump(ket, open(os.path.join(OUT, "ngan_han.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1,
              default=float)
    print(f"\n→ output/ngan_han.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
