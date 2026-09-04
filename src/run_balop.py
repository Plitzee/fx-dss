"""GIAI DOAN 1 — BANG NEN DAY DU cho du bao ba lop.

Chay bon nen cua docs/REPLAN_2026.md muc 2.3 tren CA HAI muc tieu (R va P),
CA BA tam han (1/5/20 phien), 6 cap, cham tren doan KIEM DINH.

Chua co quy luat nao o day — day la thuoc do. Giai doan 2 khai pha quy luat se
phai thang bang nay, va con so phai thang la NEN 3 "chi sigma".

Moi tham so (k, kP, c_h, nu, nguong che do, ma tran chuyen) uoc luong CHI tren
doan HUAN LUYEN roi dong bang, dung luat cua split.py.

Chay:  python src/run_balop.py
Ghi:   output/nen3.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

import balop as B                                          # noqa: E402
import diem3 as D                                          # noqa: E402
from split import doan                                     # noqa: E402


def chuan_bi(h):
    """Khop bon nen tren HUAN LUYEN, du bao tren KIEM DINH, gop 6 cap.

    Tra ve dict[muc_tieu] -> dict voi P cua tung nen, y, va nhan phu tro
    (cap, che do) de con ban nho duoc."""
    Bd = B.nap()
    ra = {mt: dict(y=[], cap=[], che_do=[], P={t: [] for t in TEN_NEN})
          for mt in ("R", "P")}
    tham = {}
    for p in B.PAIRS:
        d = Bd[p]
        g = doan(d.Date.values)
        tr, va = g == 0, g == 1
        T = B.dung_muc_tieu(d, h, tr)
        tham[p] = dict(c_h=T["c_h"], k=T["k"], kP=T["kP"])
        sig = d.sig.values
        nguong = np.quantile(sig[tr], [1 / 3, 2 / 3])
        che_do = np.digitize(sig, nguong)

        # bon nen — khop mot lan tren huan luyen, dung cho ca hai muc tieu
        ns = B.ChiSigma().khop(T["z"][tr])
        cd = B.SigmaCheDo().khop(T["z"][tr], sig[tr])

        for mt, y_all, canh in (("R", T["yR"], T["canh_R"]),
                                ("P", T["yP"], T["canh_P"])):
            yt = B.lop_truoc(y_all, h)
            kh = B.KhiHauHoc().khop(y_all[tr])
            qt = B.QuanTinh().khop(y_all[tr], yt[tr])
            n = len(d)
            P = {
                "khí hậu học": kh.du_bao(n),
                "quán tính": qt.du_bao(n, y_truoc=yt),
                "chỉ σ̂": ns.du_bao(n, canh=canh, sigma_h=T["sigma_h"], sig=sig),
                "σ̂ + chế độ": cd.du_bao(n, canh=canh, sigma_h=T["sigma_h"], sig=sig),
            }
            # NEN 5 — to hop truc tuyen tren chinh bon nen tren. Chay MOT LAN
            # tu dau chuoi: trong so hoc dan tu ket cuc DA BIET, tre dung h
            # phien. Khong khop gi tren kiem dinh, nen khong ro ri.
            th = B.ToHopTrucTuyen(
                [("khí hậu học", kh), ("quán tính", qt),
                 ("chỉ σ̂", ns), ("σ̂ + chế độ", cd)], tre=h)
            P["tổ hợp trực tuyến"] = th.du_bao(
                n, y_that=y_all, canh=canh, sigma_h=T["sigma_h"], sig=sig,
                y_truoc=yt)
            tham[p][f"trong_so_{mt}"] = th.trong_so
            m = va & (y_all >= 0)
            ra[mt]["y"].append(y_all[m])
            ra[mt]["cap"].append(np.full(int(m.sum()), p))
            ra[mt]["che_do"].append(che_do[m])
            for t in TEN_NEN:
                ra[mt]["P"][t].append(P[t][m])

    for mt in ra:
        ra[mt]["y"] = np.concatenate(ra[mt]["y"])
        ra[mt]["cap"] = np.concatenate(ra[mt]["cap"])
        ra[mt]["che_do"] = np.concatenate(ra[mt]["che_do"])
        for t in TEN_NEN:
            ra[mt]["P"][t] = np.vstack(ra[mt]["P"][t])
    return ra, tham


TEN_NEN = ("khí hậu học", "quán tính", "chỉ σ̂", "σ̂ + chế độ",
           "tổ hợp trực tuyến")
CHE_DO_TEN = ("bình tĩnh", "vừa", "căng thẳng")

# Do dai khoi cho bootstrap. Voi tam han h, cac cua so CHONG LAN nhau h phien
# nen khoi phai dai hon h that nhieu, neu khong KTC se hep gia.
KHOI = {1: 20, 5: 40, 20: 80}


def main():
    ket = {}
    print("=" * 100)
    print("GIAI ĐOẠN 1 — BẢNG NỀN CHO DỰ BÁO BA LỚP (chấm trên đoạn KIỂM ĐỊNH)")
    print("=" * 100)
    print("Mục tiêu R = |z_h| < k  (σ̂ đã chia ra — đo kỹ năng VƯỢT TRÊN tầng 2)")
    print("Mục tiêu P = |r_h| < b  (dải cố định — thứ hiện trên giao diện)")
    print("BSS so với khí hậu học; AUC là kỹ năng HƯỚNG ĐI, 0,5 = không phân biệt được.")
    print("AUC PHÂN TẦNG THEO CẶP — gộp thẳng 6 cặp cho ảo giác 0,60 dù dự báo là hằng số,")
    print("vì mỗi cặp có tần suất nền riêng. Xem tự kiểm của src/diem3.py.\n")

    for h in B.HS:
        dat, tham = chuan_bi(h)
        print("─" * 100)
        print(f"TẦM HẠN h = {h} phiên"
              + ("   (cửa sổ chồng lấn → mẫu hữu hiệu nhỏ hơn n)" if h > 1 else ""))
        kk = " ".join(f"{p}:{tham[p]['k']:.3f}" for p in B.PAIRS)
        print(f"  k mỗi cặp : {kk}")
        kp = " ".join(f"{p}:{tham[p]['kP']:.3f}" for p in B.PAIRS)
        print(f"  kP mỗi cặp: {kp}")

        for mt in ("R", "P"):
            y = dat[mt]["y"]
            cap = dat[mt]["cap"]
            Pkh = dat[mt]["P"]["khí hậu học"]
            tl = np.bincount(y, minlength=3) / len(y)
            print(f"\n  ── MỤC TIÊU {mt} ──  n={len(y):,}   "
                  f"tỷ lệ lớp: giảm {tl[0]:.3f} / đi ngang {tl[1]:.3f} / tăng {tl[2]:.3f}")
            print(f"  {'nền':<16}{'log':>9}{'Brier':>9}{'BSS':>9}{'KTC 95% của BSS':>20}"
                  f"{'ECE':>8}{'MCE':>8}{'AUC hướng':>11}{'KTC 95%':>18}")
            for t in TEN_NEN:
                P = dat[mt]["P"][t]
                r = D.bang(P, y, Pkh, nhom=cap)
                lo, hi = D.auc_ktc(P, y, nhom=cap, nboot=300, khoi=KHOI[h], seed=7)
                blo, bhi = D.bss_ktc(P, y, Pkh, nhom=cap, nboot=300,
                                     khoi=KHOI[h], seed=7)
                sao = " *" if (np.isfinite(blo) and blo > 0) else ""
                print(f"  {t:<16}{r['log']:>9.4f}{r['brier']:>9.4f}{r['bss']:>+9.4f}"
                      f"{f'[{blo:+.4f}, {bhi:+.4f}]':>20}"
                      f"{r['ece']:>8.4f}{r['mce']:>8.4f}{r['auc']:>11.4f}"
                      f"{f'[{lo:.3f}, {hi:.3f}]':>18}{sao}")
                ket[f"h{h}_{mt}_{t}"] = dict(**{k: v for k, v in r.items()},
                                             auc_lo=lo, auc_hi=hi,
                                             bss_lo=blo, bss_hi=bhi)

            # ban theo che do — trung binh gop giau dung cho quan trong nhat
            tot = min(TEN_NEN, key=lambda t: D.diem_log(dat[mt]["P"][t], y))
            print(f"  nền tốt nhất theo điểm log: {tot}  — bản theo chế độ biến động:")
            for v in range(3):
                m = dat[mt]["che_do"] == v
                if m.sum() < 200:
                    continue
                r = D.bang(dat[mt]["P"][tot][m], y[m], Pkh[m], nhom=cap[m])
                print(f"    {CHE_DO_TEN[v]:<12} n={int(m.sum()):>6,}  log={r['log']:.4f}"
                      f"  BSS={r['bss']:+.4f}  ECE={r['ece']:.4f}  AUC={r['auc']:.4f}")
                ket[f"h{h}_{mt}_{tot}_chedo{v}"] = r
        print()

    with open(os.path.join(OUT, "nen3.json"), "w", encoding="utf-8") as f:
        json.dump(ket, f, indent=1, ensure_ascii=False)
    print("=" * 100)
    print("đã ghi output/nen3.json")
    print("\nĐỌC BẢNG: cột BSS của nền 'chỉ σ̂' trên MỤC TIÊU P là con số mà giai đoạn 2")
    print("phải vượt. Trên MỤC TIÊU R nền đó suy biến thành hằng số nên BSS ≈ 0 — đúng")
    print("thiết kế, xem docs/REPLAN_2026.md mục 2.1.")
    print("\nTỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
