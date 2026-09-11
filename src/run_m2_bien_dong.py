"""PHA 2 / WEEK 3 — M1 vs M2 TREN TRUC BIEN DONG: bon cach bien tin hieu LLM
thanh dac trung HOC DUOC, so sanh truc tiep.

VI SAO FILE NAY TON TAI. H8/H8b/H8c/H8e deu dua tin tuc vao PHEU KHAI PHA QUY
LUAT tren truc HUONG GIA. Hai sai lam:

  1. SAI CONG CU. Pheu chi kiem "vi tu X co du bao lop Y khong" — no khong ket
     hop nhieu tin hieu, khong co trong so, khong hoc. Rieng H8e con nen ~500
     tu chinh sach xuong 4 nhan phan loai (~10 bit), vut gan het thu LLM hieu.
  2. SAI TRUC. Repo DA BIET "su kien khuech dai bien do, khong chi ra chieu"
     (CHISO_DANHGIA.md muc 8), va truc huong gia da can kiet (0/8.469). Neu
     tin tuc co tac dung thi phai o truc BIEN DONG.

File nay lam dung phep so sanh roadmap Week 3 doi ("M1 vs M2_magnitude"), voi
bon cach bien van ban thanh dac trung, tat ca LIEN TUC tru R1:

  R1  nhan phan loai LLM (H8e)          4 truong -> bien gia  [MOC DE SO]
  R2  TRUC NGU NGHIA (moi)              chieu embedding len truc do LLM dinh
                                        nghia bang cau mo neo -> diem lien tuc
                                        CO THU TU, CO CUONG DO
  R3  PCA embedding (H8b)               3 thanh phan chinh
  R4  dac trung thu cong (H8)           TF-IDF cosine, doi do dai, giong dieu

BA TRUC NGU NGHIA CHO R2 — CHOT TRUOC, cau mo neo liet ke day du:

  chinh_sach   (+) "The Committee signals tighter monetary policy, is concerned
                    about high inflation, and will raise interest rates."
               (-) "The Committee signals easier monetary policy, is concerned
                    about weak employment, and will lower interest rates."
  bat_dinh     (+) "The economic outlook is highly uncertain and downside risks
                    have increased significantly."
               (-) "The economic outlook is clear, stable, and risks are
                    roughly balanced."
  chuyen_huong (+) "The Committee is changing its policy stance and announcing
                    a new program or a new approach."
               (-) "The Committee is maintaining its existing policy stance
                    unchanged from the previous meeting."

  diem = cos(embedding_thong_cao, mo_neo_duong)
         - cos(embedding_thong_cao, mo_neo_am)

CAU HOI DUOC DAT: trong cua so K=5 phien SAU mot ky hop FOMC, biet NOI DUNG
thong cao co cai thien du bao BIEN DONG khong, NGOAI viec da biet co mot cuoc
hop (bien lich) va da co du bao HAR?

  M1  log_rv(t+1) ~ log h_HAR(t) + so phien ke tu hop
  M2  M1 + dac trung tin tuc (mot trong R1..R4)

GIAO THUC: chi danh gia tren cac phien TRONG cua so sau hop (noi tin tuc ton
tai). Khop tren HUAN LUYEN, CHON tren KIEM DINH, cham MOT LAN tren KIEM TRA.
Doi log -> phuong sai bang hieu chinh log-chuan (+0,5*var phan du huan luyen),
cham bang QLIKE bat bien thang do — dung y giao thuc vong 7.

Chay:  python src/run_m2_bien_dong.py
Ghi:   output/m2_bien_dong.json
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

import volfc2 as V2                                            # noqa: E402
from split import doan                                         # noqa: E402
from run_final7 import dm_nw                                    # noqa: E402
from run_h8_tintuc import nap_thong_cao, dac_trung_van_ban      # noqa: E402
from run_h8b_embedding import MO_HINH                           # noqa: E402
from run_h8e_llm import nap_nhan, TRUONG                        # noqa: E402

K_CUA_SO = 5
EPS = 1e-12

MO_NEO = {
    "chinh_sach": (
        "The Committee signals tighter monetary policy, is concerned about high "
        "inflation, and will raise interest rates.",
        "The Committee signals easier monetary policy, is concerned about weak "
        "employment, and will lower interest rates."),
    "bat_dinh": (
        "The economic outlook is highly uncertain and downside risks have "
        "increased significantly.",
        "The economic outlook is clear, stable, and risks are roughly balanced."),
    "chuyen_huong": (
        "The Committee is changing its policy stance and announcing a new "
        "program or a new approach.",
        "The Committee is maintaining its existing policy stance unchanged from "
        "the previous meeting."),
}


def qlike(rv, h):
    r = rv / np.maximum(h, EPS)
    return r - np.log(np.maximum(r, EPS)) - 1.0


def truc_ngu_nghia(tc):
    """R2 — chieu embedding thong cao len cac truc do cau mo neo dinh nghia."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MO_HINH)
    E = m.encode(tc.van_ban.tolist(), show_progress_bar=False,
                normalize_embeddings=True)
    cot = {}
    for ten, (duong, am) in MO_NEO.items():
        A = m.encode([duong, am], normalize_embeddings=True)
        cot[f"truc_{ten}"] = E @ A[0] - E @ A[1]
    return pd.DataFrame(dict(ngay=tc.ngay, **cot)), list(cot)


def dac_trung_llm(F_nhan):
    """R1 — bien gia tu nhan phan loai LLM (bo mot muc lam tham chieu)."""
    cot = {}
    for truong, muc in TRUONG.items():
        for m in muc[1:]:
            cot[f"{truong}={m}"] = (F_nhan[truong] == m).astype(float).values
    return pd.DataFrame(dict(ngay=F_nhan.ngay, **cot)), list(cot)


def dung_bang(bang, chung, F_list):
    """Bang dai: moi hang la mot phien TRONG cua so sau hop, kem dac trung."""
    lich = V2.nap_lich(chung)
    ngay_hop = pd.DatetimeIndex(chung)[lich["fomc"] > 0]
    hang = []
    for p in V2.PAIRS:
        d = bang[p]
        ng = pd.DatetimeIndex(d.Date)
        g = doan(d.Date.values)
        rv = np.maximum(d.rv5.values, EPS)
        h_har = V2.du_bao_san_xuat(d, p)
        lrv = np.log(rv)
        for ngay_tc in ngay_hop:
            sau = np.flatnonzero(ng > ngay_tc)[:K_CUA_SO]
            for k, t in enumerate(sau):
                if t + 1 >= len(d):
                    continue
                if not (np.isfinite(h_har[t]) and h_har[t] > 0
                        and np.isfinite(lrv[t + 1])):
                    continue
                hang.append(dict(pair=p, ngay=ng[t], ngay_hop=ngay_tc,
                                 buoc=k + 1, doan=int(g[t]),
                                 log_h_har=float(np.log(h_har[t])),
                                 y=float(lrv[t + 1]), rv_that=float(rv[t + 1])))
    df = pd.DataFrame(hang)
    for F, _ in F_list:
        df = df.merge(F, left_on="ngay_hop", right_on="ngay", how="left",
                      suffixes=("", "_x"))
        df = df.drop(columns=[c for c in df.columns if c.endswith("_x")])
    return df


def khop_cham(df, cot_them, ten):
    """OLS tren HUAN LUYEN, cham QLIKE tren kiem dinh va kiem tra."""
    cot = ["log_h_har", "buoc"] + list(cot_them)
    m_ok = df[cot + ["y"]].notna().all(1)
    d = df[m_ok]
    tr = d[d.doan == 0]
    if len(tr) < 100:
        return None
    X = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cot])
    beta, *_ = np.linalg.lstsq(X, tr.y.values, rcond=None)
    s2 = float(np.var(tr.y.values - X @ beta))
    ra = {"ten": ten, "n_dac_trung": len(cot), "s2": s2}
    for nhan, gid in (("kiem_dinh", 1), ("kiem_tra", 2)):
        s = d[d.doan == gid]
        if len(s) < 50:
            continue
        Xs = np.column_stack([np.ones(len(s))] + [s[c].values for c in cot])
        mu = Xs @ beta
        h = np.exp(np.clip(mu, -30, 0) + 0.5 * s2)
        ql = qlike(s.rv_that.values, h)
        ra[nhan] = dict(n=len(s), qlike=float(ql.mean()))
        ra[f"_ql_{nhan}"] = ql
    return ra


def main():
    t0 = time.time()
    print("=" * 104)
    print("PHA 2 / WEEK 3 — M1 vs M2 TRÊN TRỤC BIẾN ĐỘNG")
    print("=" * 104)

    tc = nap_thong_cao()
    F_nhan = nap_nhan()
    print(f"{len(tc)} thông cáo · {len(F_nhan)} có nhãn LLM")

    print("\nđang dựng bốn cách biểu diễn…", flush=True)
    F1, c1 = dac_trung_llm(F_nhan)
    F2, c2 = truc_ngu_nghia(tc)
    from run_h8b_embedding import embed_thong_cao
    F3, c3 = embed_thong_cao(tc)
    F4 = dac_trung_van_ban(tc)
    c4 = ["thay_doi_cau_chu", "doi_do_dai", "giong_dieu"]
    print(f"  R1 nhãn LLM: {len(c1)} biến giả")
    print(f"  R2 trục ngữ nghĩa: {c2}")
    print(f"  R3 PCA embedding: {len(c3)} thành phần")
    print(f"  R4 thủ công: {c4}")

    bang, chung = V2.nap_bang()
    df = dung_bang(bang, chung, [(F1, c1), (F2, c2), (F3, c3), (F4, c4)])
    print(f"\nbảng dài {len(df):,} hàng (chỉ phiên trong cửa sổ {K_CUA_SO} "
          f"phiên sau họp FOMC)")
    for nhan, gid in (("huấn luyện", 0), ("kiểm định", 1), ("kiểm tra", 2)):
        print(f"  {nhan:<12}{int((df.doan == gid).sum()):>7,} hàng")

    bien = [("M1 (HAR + lịch họp)", []),
            ("M2·R1 nhãn LLM", c1),
            ("M2·R2 trục ngữ nghĩa", c2),
            ("M2·R3 PCA embedding", c3),
            ("M2·R4 thủ công", c4)]
    kq = {}
    for ten, cot in bien:
        r = khop_cham(df, cot, ten)
        if r:
            kq[ten] = r

    m1 = kq["M1 (HAR + lịch họp)"]
    print("\n" + "=" * 104)
    print(f"{'biến thể':<26}{'k':>4}{'QLIKE kiểm định':>18}{'so M1':>9}"
          f"{'QLIKE kiểm tra':>17}{'so M1':>9}{'DM p (kt)':>11}")
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
        print(f"{ten:<26}{r['n_dac_trung']:>4}{vd:>18.4f}{d_vd:>8.1f}%"
              f"{kt:>17.4f}{d_kt:>8.1f}%{pp:>11.4f}")
    print("-" * 104)

    tot = min((t for t in kq if t != m1["ten"]),
              key=lambda t: kq[t]["kiem_dinh"]["qlike"])
    print(f"\nTỐT NHẤT TRÊN KIỂM ĐỊNH (quy tắc chọn): {tot}")
    d_kt = (kq[tot]["kiem_tra"]["qlike"] / m1["kiem_tra"]["qlike"] - 1) * 100
    print(f"  → trên kiểm tra: {d_kt:+.1f}% so với M1 "
          f"({'tốt hơn' if d_kt < 0 else 'TỆ HƠN'})")

    os.makedirs(OUT, exist_ok=True)
    json.dump({t: {k: v for k, v in r.items() if not k.startswith("_ql")}
               for t, r in kq.items()},
              open(os.path.join(OUT, "m2_bien_dong.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/m2_bien_dong.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
