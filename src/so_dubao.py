"""SO DU BAO — ghi truoc, cham sau, khong bao gio sua.

`docs/DONGBO_SANXUAT.md` goi day la thu "bien he thong tu mot script chay tu
dong thanh mot DSS that". Ly do: moi chi so hieu chuan dang hien tren giao dien
(ECE 0,0156 · MCE 0,0816 · BSS +0,0105) deu do tren DOAN KIEM DINH 2021-2023.
Chung khong theo doi gi dang xay ra hom nay. So nay sua dieu do.

BA LUAT, va tu kiem o cuoi file ep ca ba:

  1. CHI GHI DUOC CHO TUONG LAI. Tu choi ghi du bao cho mot phien ma ket cuc
     da biet. Khong co luat nay thi ca so chi la mot bai backtest doi ten.
  2. CHI GHI THEM. Mot dong da ghi khong bao gio bi sua hay xoa. Cham diem ghi
     sang FILE KHAC roi ghep lai khi doc.
  3. KHONG TRUNG LAP. Ghi lai cung (cap, ngay, tam han) thi bo qua, khong tao
     dong thu hai — de viec dinh ky chay lai nhieu lan van an toan.

Ghi vao:  data/so_dubao/dubao.csv    (du bao, chi ghi them)
          data/so_dubao/chamdiem.csv (ket cuc, chi ghi them)

Tu kiem:  python src/so_dubao.py
"""
import datetime as dt
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SO = os.path.join(ROOT, "data", "so_dubao")
F_DB = os.path.join(SO, "dubao.csv")
F_CD = os.path.join(SO, "chamdiem.csv")

COT_DB = ["ghi_luc", "pair", "ngay", "h", "p_giam", "p_ngang", "p_tang",
          "b_pip", "sigma_pip", "che_do", "mo_hinh", "ma_cau_hinh"]
COT_CD = ["cham_luc", "pair", "ngay", "h", "r_pip", "y_thuc"]
EPS = 1e-12


def _doc(f, cot):
    if not os.path.exists(f):
        return pd.DataFrame(columns=cot)
    d = pd.read_csv(f)
    for c in cot:
        if c not in d:
            d[c] = np.nan
    return d[cot]


def doc_dubao():
    return _doc(F_DB, COT_DB)


def doc_chamdiem():
    return _doc(F_CD, COT_CD)


def _them(f, cot, hang):
    os.makedirs(SO, exist_ok=True)
    d = pd.DataFrame(hang)[cot]
    d.to_csv(f, mode="a", header=not os.path.exists(f), index=False)


def ghi(du_bao, gia_den):
    """Ghi du bao cho cac phien CHUA co ket cuc.

    du_bao : list dict, moi dict co du cac khoa cua COT_DB (tru ghi_luc)
    gia_den: dict pair -> ngay CUOI CUNG da co gia (chuoi 'YYYY-MM-DD')

    Tra ve (so_ghi, so_bo_qua_trung, so_tu_choi_qua_khu).
    """
    cu = doc_dubao()
    da_co = set(zip(cu.pair.astype(str), cu.ngay.astype(str), cu.h.astype(int))) \
        if len(cu) else set()
    luc = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    moi, trung, qua_khu = [], 0, 0
    for r in du_bao:
        khoa = (str(r["pair"]), str(r["ngay"]), int(r["h"]))
        if khoa in da_co:
            trung += 1
            continue
        # LUAT 1: ket cuc cua [ngay, ngay+h-1] phai CHUA biet. Ta biet gia den
        # `gia_den[pair]`, nen phien duoc du bao phai nam SAU moc do.
        if str(r["ngay"]) <= str(gia_den.get(r["pair"], "0000-00-00")):
            qua_khu += 1
            continue
        moi.append({**{k: r.get(k) for k in COT_DB if k != "ghi_luc"},
                    "ghi_luc": luc})
        da_co.add(khoa)
    if moi:
        _them(F_DB, COT_DB, moi)
    return len(moi), trung, qua_khu


def _lop(r_pip, b_pip):
    if r_pip < -b_pip:
        return 0
    if abs(r_pip) <= b_pip:
        return 1
    return 2


def cham(gia):
    """Cham diem moi du bao da du ket cuc ma chua cham.

    gia : dict pair -> DataFrame co cot Date (datetime) va close.
    Tra ve so dong vua cham.
    """
    db = doc_dubao()
    if not len(db):
        return 0
    cd = doc_chamdiem()
    da = set(zip(cd.pair.astype(str), cd.ngay.astype(str), cd.h.astype(int))) \
        if len(cd) else set()

    luc = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    moi = []
    for p, g in db.groupby("pair"):
        if p not in gia:
            continue
        d = gia[p].sort_values("Date").reset_index(drop=True)
        ngay = d.Date.dt.strftime("%Y-%m-%d").tolist()
        vt = {x: i for i, x in enumerate(ngay)}
        c = d.close.values.astype(float)
        pip = 0.01 if p == "USDJPY" else 0.0001
        for _, r in g.iterrows():
            khoa = (str(p), str(r.ngay), int(r.h))
            if khoa in da:
                continue
            i, hh = vt.get(str(r.ngay)), int(r.h)
            # can i >= 1 de co gia dong cua TRUOC phien duoc du bao, va can
            # i+hh-1 nam trong chuoi de ket cuc da day du
            if i is None or i < 1 or i + hh - 1 > len(c) - 1:
                continue
            r_pip = float((c[i + hh - 1] - c[i - 1]) / pip)
            moi.append({"cham_luc": luc, "pair": p, "ngay": str(r.ngay),
                        "h": int(r.h), "r_pip": round(r_pip, 3),
                        "y_thuc": _lop(r_pip, float(r.b_pip))})
            da.add(khoa)
    if moi:
        _them(F_CD, COT_CD, moi)
    return len(moi)


def ghep():
    """Du bao da co ket cuc, dang bang de cham diem."""
    db, cd = doc_dubao(), doc_chamdiem()
    if not len(db) or not len(cd):
        return pd.DataFrame()
    for d in (db, cd):
        d["pair"] = d.pair.astype(str)
        d["ngay"] = d.ngay.astype(str)
        d["h"] = d.h.astype(int)
    return pd.merge(db, cd, on=["pair", "ngay", "h"], how="inner")


def thong_ke(h=1, n_toi_thieu=30):
    """Chi so hieu chuan TRUOT, tinh tu chinh so nay.

    Tra ve None khi chua du mau — KHONG doan, khong lay so cua doan kiem dinh
    ra thay the. Giao dien phai noi ro dang tich luy."""
    import sys
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import diem3 as D

    g = ghep()
    if not len(g):
        return {"n": 0, "du_mau": False}
    g = g[g.h == int(h)]
    if not len(g):
        return {"n": 0, "du_mau": False}
    P = g[["p_giam", "p_ngang", "p_tang"]].values.astype(float)
    P = P / np.maximum(P.sum(1, keepdims=True), EPS)
    y = g.y_thuc.values.astype(int)
    ra = {"n": int(len(g)), "du_mau": bool(len(g) >= n_toi_thieu),
          "tu": str(g.ngay.min()), "den": str(g.ngay.max()),
          "so_cap": int(g.pair.nunique())}
    if not ra["du_mau"]:
        return ra
    tan = np.bincount(y, minlength=3) / len(y)
    Pkh = np.tile(tan, (len(y), 1))
    ra.update({"log": round(D.diem_log(P, y), 4), "brier": round(D.brier(P, y), 4),
               "ece": round(D.ece(P, y), 4), "mce": round(D.mce(P, y), 4),
               "bss": round(D.bss(P, y, Pkh), 4),
               "auc": round(float(D.auc_huong(P, y, nhom=g.pair.values)), 4)})
    return ra


if __name__ == "__main__":
    import shutil
    import tempfile

    goc = SO
    SO = tempfile.mkdtemp(prefix="sodubao_")
    F_DB, F_CD = os.path.join(SO, "dubao.csv"), os.path.join(SO, "chamdiem.csv")
    print("TỰ KIỂM  (thư mục tạm, không đụng sổ thật)")

    ngay = pd.date_range("2026-01-01", periods=40, freq="D")
    rng = np.random.default_rng(0)
    gia = {"EURUSD": pd.DataFrame({"Date": ngay,
                                   "close": 1.10 + np.cumsum(rng.normal(0, 0.004, 40))})}
    den = {"EURUSD": "2026-01-30"}          # gia moi co den 30/01

    def db(n, b=30.0):
        return [{"pair": "EURUSD", "ngay": str(x.date()), "h": 1,
                 "p_giam": 0.3, "p_ngang": 0.4, "p_tang": 0.3, "b_pip": b,
                 "sigma_pip": 45.0, "che_do": 1, "mo_hinh": "chỉ σ̂",
                 "ma_cau_hinh": "test"} for x in n]

    # LUAT 1 — khong ghi duoc cho qua khu
    g1, t1, q1 = ghi(db(ngay[:30]), den)
    print(f"  ghi 30 phiên ĐÃ có kết cục : ghi {g1} · trùng {t1} · từ chối {q1}")
    assert g1 == 0 and q1 == 30, "phai tu choi toan bo qua khu"

    # ghi cho tuong lai thi duoc
    g2, t2, q2 = ghi(db(ngay[30:35]), den)
    print(f"  ghi 5 phiên CHƯA có kết cục: ghi {g2} · trùng {t2} · từ chối {q2}")
    assert g2 == 5 and q2 == 0

    # LUAT 3 — ghi lai thi bo qua, khong nhan doi
    g3, t3, _ = ghi(db(ngay[30:35]), den)
    assert g3 == 0 and t3 == 5, "ghi lai phai bo qua"
    assert len(doc_dubao()) == 5
    print(f"  ghi lại đúng 5 phiên đó     : ghi {g3} · trùng {t3} — sổ vẫn 5 dòng")

    # LUAT 2 — chi ghi them: noi dung dong cu khong doi
    truoc = doc_dubao().copy()
    ghi(db(ngay[35:38]), den)
    sau = doc_dubao()
    assert len(sau) == len(truoc) + 3, "phai them dung 3 dong"
    assert sau.head(len(truoc)).reset_index(drop=True).equals(
        truoc.reset_index(drop=True)), "dong cu bi sua — vi pham luat chi ghi them"
    print("  chỉ ghi thêm                : 5 dòng cũ nguyên vẹn sau khi ghi thêm")

    # cham diem — gio gia da du toi 09/02
    gia2 = {"EURUSD": pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=40, freq="D"),
                                    "close": gia["EURUSD"].close.values})}
    n = cham(gia2)
    g = ghep()
    print(f"  chấm điểm                   : {n} dòng · ghép được {len(g)} dòng")
    assert n > 0 and len(g) == n

    # lop thuc phai khop dinh nghia dai
    for _, r in g.iterrows():
        assert r.y_thuc == _lop(r.r_pip, r.b_pip)
    print("  lớp thực khớp định nghĩa dải — ĐẠT")

    # cham lai khong nhan doi
    assert cham(gia2) == 0, "cham lai phai khong tao dong moi"
    print("  chấm lại                    : 0 dòng mới")

    # thong ke: chua du mau thi PHAI noi chua du, khong duoc doan
    t = thong_ke(h=1, n_toi_thieu=30)
    print(f"  thống kê n={t['n']} (ngưỡng 30): đủ mẫu = {t['du_mau']}")
    assert t["du_mau"] is False and "ece" not in t, "chua du mau thi khong duoc tra chi so"
    t2_ = thong_ke(h=1, n_toi_thieu=3)
    assert t2_["du_mau"] and "ece" in t2_ and "bss" in t2_
    print(f"  hạ ngưỡng xuống 3           : ECE {t2_['ece']} · BSS {t2_['bss']} · AUC {t2_['auc']}")

    shutil.rmtree(SO, ignore_errors=True)
    SO = goc
    print("\nTỰ KIỂM ĐẠT")
