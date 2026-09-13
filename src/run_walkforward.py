"""WALK-FORWARD — hieu nang co TROI theo thoi gian khong?

Ke hoach Pha 1 cua HuyH (Week 3 muc 1) doi "di xa hon mot lan chia 80/20":
chia truc thoi gian thanh nhieu khoi lien tiep va do performance drift, regime
sensitivity, stability. Repo da co ba doan co dinh (huan luyen / kiem dinh /
kiem tra) nhung CHUA co bang trai theo tung nam.

VI SAO KHONG PHAI "HUAN LUYEN LAI NHIEU LAN". Mo hinh san xuat da la mo hinh
CUON: `du_bao_cuon` khop lai moi 21 phien tren cua so MO RONG, nen du bao tai
ngay t von da chi dung thong tin den t. Walk-forward o day vi the KHONG phai
huan luyen lai — no la viec CHIA chuoi du bao nhan qua da co san theo tung
nam roi cham diem tung khoi. Dung nghia "performance drift" cho mot mo hinh
truc tuyen.

DAY LA CHAN DOAN, KHONG PHAI CONG CU CHON. Khong mot lua chon mo hinh nao
trong repo phu thuoc bang nay; no chi tra loi "hieu nang co on dinh qua cac
nam khong, hay chi den tu vai nam may man".

Nen so sanh cho BSS la KHI HAU HOC KHOP TREN HUAN LUYEN (dong bang), khong
phai khi hau hoc cua tung nam — neu lay khi hau hoc tung nam thi moi nam se
tu chuan hoa ve 0 va bang mat sach y nghia.

Chay:  python src/run_walkforward.py
Ghi:   output/walkforward.json
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
import diem3 as D                                             # noqa: E402
from split import doan                                        # noqa: E402
from api.main import NEN_THEO_H                               # noqa: E402

TEN_DOAN = ("huấn luyện", "kiểm định", "kiểm tra")


def du_bao_san_xuat(h):
    """Du bao NHAN QUA cua mo hinh SAN XUAT cho toan truc thoi gian, gop 6 cap.

    Tra ve (P, Pkh, y, cap, ngay, doan_id) — Pkh la khi hau hoc khop tren
    HUAN LUYEN, dung lam moc BSS chung cho moi nam."""
    Bd = B.nap()
    Ps, Pkhs, ys, caps, ngays, gs = [], [], [], [], [], []
    for p in B.PAIRS:
        d = Bd[p]
        g = doan(d.Date.values)
        tr = g == 0
        T = B.dung_muc_tieu(d, h, tr)
        sig = d.sig.values
        n = len(d)
        y_all, canh = T["yP"], T["canh_P"]
        yt = B.lop_truoc(y_all, h)
        ns = B.ChiSigma().khop(T["z"][tr])
        cd = B.SigmaCheDo().khop(T["z"][tr], sig[tr])
        kh = B.KhiHauHoc().khop(y_all[tr])
        qt = B.QuanTinh().khop(y_all[tr], yt[tr])
        kw = dict(canh=canh, sigma_h=T["sigma_h"], sig=sig, y_truoc=yt)

        Pns = B.du_bao_cuon(T, sig, lambda z, sg: B.ChiSigma().khop(z), canh=canh)
        Pcd = B.du_bao_cuon(T, sig, lambda z, sg: B.SigmaCheDo().khop(z, sg),
                            canh=canh)
        for Pc, nen in ((Pns, ns), (Pcd, cd)):      # dam dau chuoi
            thieu = ~np.isfinite(Pc[:, 0])
            if thieu.any():
                Pc[thieu] = nen.du_bao(n, **kw)[thieu]

        if NEN_THEO_H[h] == "tổ hợp trực tuyến":
            th = B.ToHopTrucTuyen(
                [("khí hậu học", kh), ("quán tính", qt),
                 ("chỉ σ̂", B.NenCoSan("chỉ σ̂", Pns)),
                 ("σ̂ + chế độ", B.NenCoSan("σ̂ + chế độ", Pcd))], tre=h)
            P = th.du_bao(n, y_that=y_all, **kw)
        elif NEN_THEO_H[h] == "chỉ σ̂ (cuộn)":
            P = Pns
        else:
            P = Pcd

        m = y_all >= 0
        Ps.append(P[m]); Pkhs.append(kh.du_bao(n)[m])
        ys.append(y_all[m]); caps.append(np.full(int(m.sum()), p))
        ngays.append(pd.DatetimeIndex(d.Date.values)[m]); gs.append(g[m])
    return (np.vstack(Ps), np.vstack(Pkhs), np.concatenate(ys),
            np.concatenate(caps), pd.DatetimeIndex(np.concatenate(ngays)),
            np.concatenate(gs))


def main():
    t0 = time.time()
    print("=" * 104)
    print("WALK-FORWARD — hiệu năng theo từng năm (chẩn đoán độ ổn định)")
    print("=" * 104)
    print("Mô hình sản xuất mỗi h:", {h: NEN_THEO_H[h] for h in B.HS})
    print("BSS so với KHÍ HẬU HỌC KHỚP TRÊN HUẤN LUYỆN (đóng băng) — không phải")
    print("khí hậu học của từng năm, nếu không mỗi năm sẽ tự chuẩn hoá về 0.")

    ket = {}
    for h in B.HS:
        P, Pkh, y, cap, ngay, g = du_bao_san_xuat(h)
        print(f"\n{'─'*104}")
        print(f"TẦM HẠN h = {h} phiên · mô hình: {NEN_THEO_H[h]}"
              + ("   (cửa sổ chồng lấn → mẫu hữu hiệu ≈ n/h)" if h > 1 else ""))
        print(f"  {'năm':<7}{'đoạn':<12}{'n':>7}{'log':>9}{'BSS':>9}"
              f"{'ECE':>8}{'AUC hướng':>11}")
        nam_ket = {}
        for nam in sorted(set(ngay.year)):
            m = ngay.year == nam
            if m.sum() < 200:
                continue
            r = D.bang(P[m], y[m], Pkh[m], nhom=cap[m])
            gd = TEN_DOAN[int(np.bincount(g[m], minlength=3).argmax())]
            print(f"  {nam:<7}{gd:<12}{int(m.sum()):>7,}{r['log']:>9.4f}"
                  f"{r['bss']:>+9.4f}{r['ece']:>8.4f}{r['auc']:>11.4f}")
            nam_ket[str(nam)] = dict(doan=gd, **r)
        bss = np.array([v["bss"] for v in nam_ket.values()])
        duong = int((bss > 0).sum())
        print(f"  → {duong}/{len(bss)} năm có BSS dương · "
              f"trung vị {np.median(bss):+.4f} · "
              f"khoảng [{bss.min():+.4f}; {bss.max():+.4f}]")
        ket[f"h{h}"] = dict(mo_hinh=NEN_THEO_H[h], theo_nam=nam_ket,
                            so_nam_duong=duong, so_nam=len(bss),
                            bss_trung_vi=float(np.median(bss)),
                            bss_min=float(bss.min()), bss_max=float(bss.max()))

    print("\n" + "=" * 104)
    print("TỔNG KẾT ĐỘ ỔN ĐỊNH")
    print(f"  {'h':<5}{'số năm BSS dương':>20}{'BSS trung vị':>16}{'khoảng':>26}")
    for h in B.HS:
        v = ket[f"h{h}"]
        ty_le = f"{v['so_nam_duong']}/{v['so_nam']}"
        khoang = f"[{v['bss_min']:+.4f}; {v['bss_max']:+.4f}]"
        print(f"  {h:<5}{ty_le:>20}{v['bss_trung_vi']:>+16.4f}{khoang:>26}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(ket, open(os.path.join(OUT, "walkforward.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1,
              default=float)
    print(f"\n→ output/walkforward.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
