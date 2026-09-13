"""PHA 2/3 — BAT NGO CHINH SACH (market-implied) so voi NOI DUNG VAN BAN,
tren truc bien dong.

VI SAO. `run_m2_bien_dong.py` da thu BON cach bien van ban thong cao FOMC
thanh dac trung (nhan LLM, truc ngu nghia, embedding PCA, thu cong) — KHONG
cach nao thang M1 ngoai mau. Tra cuu tai lieu cho ly do co ban:

  Noi dung thong cao PHAN LON DA DUOC DU DOAN TRUOC. Cai lam gia dong khong
  phai thong cao NOI GI, ma la no KHAC BAO NHIEU so voi ky vong thi truong.
  Phan bat ngo do — theo dinh nghia — KHONG nam trong van ban; no nam o
  khoang cach giua van ban va ky vong, va chi doc duoc tu GIA THI TRUONG.

Do la ly do bon cach bieu dien deu that bai: chung ma hoa ngay cang tot MOT
THU VON KHONG CHUA THONG TIN MOI.

Cach lam chuan cua tai lieu (Kuttner 1998; Gurkaynak-Sack-Swanson 2005;
Nakamura-Steinsson 2018; Bauer-Swanson 2023) la HIGH-FREQUENCY IDENTIFICATION:
do bat ngo bang thay doi gia hop dong tuong lai lai suat trong cua so hep
quanh cong bo.

DU LIEU: U.S. Monetary Policy Event-Study Database (USMPD), SF Fed —
Acosta, Ajello, Bauer, Loria & Miranda-Agrippino (2025). Cong khai, mien phi.
`data/usmpd/USMPD.xlsx`, sheet "Monetary Events": cua so 100 phut tu 10 phut
truoc cong bo thong cao den 60 phut sau khi hop bao bat dau.

  MP1     bat ngo lai suat ky hop hien tai, suy tu fed funds futures
  UST2Y   thay doi loi suat trai phieu 2 nam trong cung cua so
  SP500   loi suat SP500 trong cung cua so
  EURUSD  phan ung ty gia EURUSD trong cung cua so

GIA THUYET — CHOT TRUOC: **DO LON** bat ngo (|MP1|), khong phai dau, du bao
BIEN DONG cao hon o cac phien sau. Co che: bat ngo lon -> dinh gia lai manh
-> bien dong thuc hien cao. Dung dau (MP1 co dau) van duoc dua vao mot bien
the rieng de kiem, nhung KHONG phai gia thuyet chinh.

KHONG RO RI: cua so ket thuc 15:00 gio New York ngay hop (thong cao 14:00 +
hop bao); dac trung chi ap tu phien KE TIEP tro di, giong het H8/H8b/H8c/H8e.

GIAO THUC: giong het `run_m2_bien_dong.py` — chi cac phien trong cua so K=5
sau hop, khop tren HUAN LUYEN, chon tren KIEM DINH, cham MOT LAN tren
KIEM TRA, QLIKE bat bien thang do, DM test so voi M1.

Chay:  python src/run_m2_batngo.py
Ghi:   output/m2_batngo.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
USMPD = os.path.join(ROOT, "data", "usmpd", "USMPD.xlsx")

import volfc2 as V2                                            # noqa: E402
from run_final7 import dm_nw                                    # noqa: E402
from run_h8_tintuc import nap_thong_cao, dac_trung_van_ban      # noqa: E402
from run_m2_bien_dong import dung_bang, khop_cham, K_CUA_SO     # noqa: E402


def dac_trung_batngo():
    """Doc USMPD, dung cac dac trung DO LON bat ngo (chot truoc)."""
    d = pd.read_excel(USMPD, sheet_name="Monetary Events")
    d = d[(d.Date >= "2009-01-01") & (d.Date <= "2026-12-31")].copy()
    F = pd.DataFrame({"ngay": pd.to_datetime(d.Date)})
    F["abs_mp1"] = np.abs(d.MP1.values) * 100.0          # diem co ban
    F["mp1"] = d.MP1.values * 100.0                       # co dau
    F["abs_ust2y"] = np.abs(d.UST2Y.values) * 100.0
    F["abs_sp500"] = np.abs(d.SP500.values)
    F["abs_eurusd"] = np.abs(d.EURUSD.values)
    return F.dropna(subset=["abs_mp1"]).reset_index(drop=True)


def main():
    t0 = time.time()
    print("=" * 104)
    print("BẤT NGỜ CHÍNH SÁCH (thị trường) so với NỘI DUNG VĂN BẢN — trục biến động")
    print("=" * 104)

    Fb = dac_trung_batngo()
    print(f"USMPD: {len(Fb)} kỳ FOMC · {Fb.ngay.min().date()} → {Fb.ngay.max().date()}")
    print(f"  |MP1| (điểm cơ bản): trung vị {Fb.abs_mp1.median():.2f} · "
          f"p90 {Fb.abs_mp1.quantile(.9):.2f} · tối đa {Fb.abs_mp1.max():.2f}")

    tc = nap_thong_cao()
    F4 = dac_trung_van_ban(tc)
    c4 = ["thay_doi_cau_chu", "doi_do_dai", "giong_dieu"]

    cb = ["abs_mp1", "mp1", "abs_ust2y", "abs_sp500", "abs_eurusd"]
    bang, chung = V2.nap_bang()
    df = dung_bang(bang, chung, [(Fb, cb), (F4, c4)])
    print(f"\nbảng dài {len(df):,} hàng (cửa sổ {K_CUA_SO} phiên sau họp)")
    co = df[cb].notna().all(1)
    print(f"  có đủ dữ liệu bất ngờ: {int(co.sum()):,} hàng")
    for nhan, gid in (("huấn luyện", 0), ("kiểm định", 1), ("kiểm tra", 2)):
        print(f"  {nhan:<12}{int(((df.doan == gid) & co).sum()):>7,} hàng")

    bien = [
        ("M1 (HAR + lịch họp)", []),
        ("M2·S1 |MP1| ĐỘ LỚN bất ngờ", ["abs_mp1"]),
        ("M2·S2 |MP1| + MP1 có dấu", ["abs_mp1", "mp1"]),
        ("M2·S3 độ lớn đa tài sản", ["abs_mp1", "abs_ust2y", "abs_sp500"]),
        ("M2·S4 |phản ứng EURUSD|", ["abs_eurusd"]),
        ("M2·R4 văn bản thủ công", c4),
        ("M2·S1+R4 bất ngờ + văn bản", ["abs_mp1"] + c4),
    ]
    kq = {}
    for ten, cot in bien:
        r = khop_cham(df, cot, ten)
        if r:
            kq[ten] = r

    m1 = kq["M1 (HAR + lịch họp)"]
    print("\n" + "=" * 104)
    print(f"{'biến thể':<30}{'k':>3}{'QLIKE kđ':>11}{'so M1':>8}"
          f"{'QLIKE kt':>11}{'so M1':>8}{'DM p (kt)':>11}")
    print("-" * 104)
    for ten, r in kq.items():
        vd = r.get("kiem_dinh", {}).get("qlike", np.nan)
        kt = r.get("kiem_tra", {}).get("qlike", np.nan)
        d_vd = (vd / m1["kiem_dinh"]["qlike"] - 1) * 100
        d_kt = (kt / m1["kiem_tra"]["qlike"] - 1) * 100
        if ten == m1["ten"]:
            pp = np.nan
        else:
            n = min(len(r["_ql_kiem_tra"]), len(m1["_ql_kiem_tra"]))
            _, pp = dm_nw(r["_ql_kiem_tra"][:n] - m1["_ql_kiem_tra"][:n])
        print(f"{ten:<30}{r['n_dac_trung']:>3}{vd:>11.4f}{d_vd:>7.1f}%"
              f"{kt:>11.4f}{d_kt:>7.1f}%{pp:>11.4f}")
    print("-" * 104)
    print("  (âm = TỐT HƠN M1; DM p<0,05 là chênh lệch có ý nghĩa)")

    tot = min((t for t in kq if t != m1["ten"]),
              key=lambda t: kq[t]["kiem_dinh"]["qlike"])
    d_kt = (kq[tot]["kiem_tra"]["qlike"] / m1["kiem_tra"]["qlike"] - 1) * 100
    print(f"\nTỐT NHẤT TRÊN KIỂM ĐỊNH (quy tắc chọn): {tot}")
    print(f"  → trên kiểm tra: {d_kt:+.1f}% so với M1 "
          f"({'TỐT HƠN' if d_kt < 0 else 'tệ hơn'})")

    os.makedirs(OUT, exist_ok=True)
    json.dump({t: {k: v for k, v in r.items() if not k.startswith("_ql")}
               for t, r in kq.items()},
              open(os.path.join(OUT, "m2_batngo.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/m2_batngo.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
