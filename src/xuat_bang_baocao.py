"""XUAT BANG TONG HOP KIEU BAI BAO — QLIKE, MSE, MAE, CRPS cho TOAN BO mo hinh
da thu trong vong 7 (HAR, ML co dien, DL, foundation model, to hop), de nguoi
dung dua vao bao cao voi giao vien huong dan.

QUY UOC CHI SO (ghi ro de khong hieu nham khi doc bang):
  * QLIKE — dang bat bien thang do da dung xuyen suot du an: r-log(r)-1,
    r=rv_that/du_bao. Day la chi so CHINH cua toan bo vong 7.
  * MSE, MAE — tinh tren RV*10^4 (RV goc ~1e-5..1e-4, nhan 10^4 de bang de
    doc, giong quy uoc thuong gap trong tai lieu bien dong thuc hien).
  * CRPS — VOI MO HINH DU BAO DIEM (tat ca tru Chronos): CRPS = MAE, vi
    phan phoi du bao la MOT DIEM (Dirac) va CRPS cua phan phoi Dirac tai f
    dung bang |y-f| (Gneiting & Raftery 2007, dinh ly co ban). KHONG phai
    xap xi — day la dang thuc CHINH XAC.
    VOI Chronos-bolt: co 9 phan vi that (0,1..0,9 cua log-RV, da doi sang
    thang RV), CRPS tinh tu cong thuc lien he pinball loss:
      CRPS(F,y) = 2 * tich_phan_0^1 pinball_tau(y, q_tau) d(tau)
    xap xi bang trung binh cong tren luoi 9 phan vi (Gneiting & Raftery
    2007, muc 4; day la xap xi lien tuc chuan trong cac bai bao dung
    luoi phan vi thua, vd M5/GluonTS mean_wQuantileLoss).

Nguon du bao:
  * HAR/ML/DL/XGBoost/CatBoost/TabPFN — qua `kiem_tohop2.nap_du_bao()` +
    `kiem_tohop3.nap_them()` (cung mang F da dung cho to hop).
  * Chronos-bolt, TTM — rieng, tu `output/chronos_raw.npz`,
    `output/ttm_raw.npz` (moi sinh tu ban chay lai co luu du bao tho).
  * To hop tot nhat — tai tao truc tiep bang ham co san trong kiem_tohop2/3.

Chay:  python src/xuat_bang_baocao.py
Ghi:   output/bang_baocao.json, output/bang_baocao.csv
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")

from split import VALID_TU, TEST_TU                                    # noqa: E402
from kiem_tohop2 import nap_du_bao, tb_hinh_hoc, gr_hoi_quy, qlike_arr, EPS  # noqa: E402
from kiem_tohop3 import nap_them                                        # noqa: E402

TL = 1e4   # he so nhan cho MSE/MAE/CRPS de bang de doc (RV goc rat nho)


def mse_mae(rv, h):
    e = (rv - h) * TL
    return float(np.mean(e ** 2)), float(np.mean(np.abs(e)))


def crps_diem(rv, h):
    """CRPS cho du bao DIEM = MAE (phan phoi Dirac tai h)."""
    return float(np.mean(np.abs(rv - h))) * TL


def crps_phanvi(rv, Q, muc=np.linspace(0.1, 0.9, 9)):
    """CRPS xap xi tu luoi phan vi qua trung binh pinball loss (Gneiting &
    Raftery 2007): CRPS ~ 2 * trung_binh_tau[pinball_tau(y, q_tau)]."""
    y = rv[:, None]
    err = y - Q
    pin = np.maximum(muc[None, :] * err, (muc[None, :] - 1) * err)
    return float(np.mean(2 * pin.mean(1))) * TL


def cham_tu_mang(rv_all, h_all, dts_all, ten):
    """Cham QLIKE/MSE/MAE/CRPS(=MAE) tren kiem dinh + kiem tra tu mang tho
    (dung cho Chronos/TTM co lich rieng, khong chung mang F)."""
    ra = {}
    for doan_, (lo, hi) in (("kiem_dinh", (VALID_TU, TEST_TU)),
                            ("kiem_tra", (TEST_TU, None))):
        m = (dts_all >= lo) & np.isfinite(h_all) & (h_all > 0) & np.isfinite(rv_all)
        if hi is not None:
            m &= dts_all < hi
        if m.sum() < 30:
            continue
        ql = float(qlike_arr(rv_all[m], h_all[m]).mean())
        mse, mae = mse_mae(rv_all[m], h_all[m])
        ra[doan_] = dict(n=int(m.sum()), qlike=ql, mse=mse, mae=mae, crps=mae)
    return ra


def cham_tu_F(F, rv, dts, ten):
    h = F[ten]
    ra = {}
    for doan_, (lo, hi) in (("kiem_dinh", (VALID_TU, TEST_TU)),
                            ("kiem_tra", (TEST_TU, None))):
        m = (dts >= lo) & np.isfinite(h) & (h > 0) & np.isfinite(rv) & (rv > 0)
        if hi is not None:
            m &= dts < hi
        if m.sum() < 30:
            continue
        ql = float(qlike_arr(rv[m], h[m]).mean())
        mse, mae = mse_mae(rv[m], h[m])
        ra[doan_] = dict(n=int(m.sum()), qlike=ql, mse=mse, mae=mae, crps=mae)
    return ra


def main():
    print("=" * 100)
    print("XUẤT BẢNG TỔNG HỢP KIỂU BÀI BÁO — QLIKE / MSE / MAE / CRPS")
    print("=" * 100)

    F, rv, dts = nap_du_bao()
    F = nap_them(F)
    log_rv = np.log(np.maximum(rv, EPS))
    va = (dts >= VALID_TU) & (dts < TEST_TU)

    BANG = {}
    DON_LE = {
        "HAR vòng 7": "HAR vòng 7 (khớp mỗi phiên)",
        "OLS HAR (khớp năm)": "OLS HAR (khớp năm)",
        "Ridge": "Ridge (toàn bộ đặc trưng)",
        "Random Forest": "Random Forest",
        "LightGBM (L2)": "LightGBM (L2 trên log)",
        "LightGBM (QLIKE)": "LightGBM (QLIKE trực tiếp)",
        "MLP": "MLP",
        "LSTM": "LSTM (khớp năm)",
        "GRU": "GRU (khớp năm)",
        "Transformer": "Transformer (PatchTST rút gọn) (khớp năm)",
        "XGBoost": "XGBoost",
        "CatBoost": "CatBoost",
        "TabPFN v2": "TabPFN v2 (ngữ cảnh 8k, khớp năm)",
    }
    for nhan, ten in DON_LE.items():
        if ten in F:
            BANG[nhan] = cham_tu_F(F, rv, dts, ten)
        else:
            print(f"  (thiếu {ten} — bỏ qua {nhan})")

    # Chronos-bolt (raw npz riêng, có phân vị thật cho CRPS)
    pc = os.path.join(OUT, "chronos_raw.npz")
    if os.path.exists(pc):
        dch = np.load(pc, allow_pickle=True)
        dts_c = pd.DatetimeIndex(dch["date"])
        rv_c, h_c, Q_c = dch["rv"], dch["h"], dch["q"]
        ra = {}
        for doan_, (lo, hi) in (("kiem_dinh", (VALID_TU, TEST_TU)),
                                ("kiem_tra", (TEST_TU, None))):
            m = (dts_c >= lo) & np.isfinite(h_c) & (h_c > 0) & np.isfinite(rv_c)
            if hi is not None:
                m &= dts_c < hi
            ql = float(qlike_arr(rv_c[m], h_c[m]).mean())
            mse, mae = mse_mae(rv_c[m], h_c[m])
            cr = crps_phanvi(rv_c[m], Q_c[m])
            ra[doan_] = dict(n=int(m.sum()), qlike=ql, mse=mse, mae=mae, crps=cr)
        BANG["Chronos-bolt-small"] = ra
    else:
        print("  (thiếu output/chronos_raw.npz — bỏ qua Chronos)")

    # TTM (raw npz riêng)
    pt = os.path.join(OUT, "ttm_raw.npz")
    if os.path.exists(pt):
        dtt = np.load(pt, allow_pickle=True)
        dts_t = pd.DatetimeIndex(dtt["date"])
        rv_t, h_t = dtt["rv"], dtt["h"]
        ra = {}
        for doan_, (lo, hi) in (("kiem_dinh", (VALID_TU, TEST_TU)),
                                ("kiem_tra", (TEST_TU, None))):
            m = (dts_t >= lo) & np.isfinite(h_t) & (h_t > 0) & np.isfinite(rv_t)
            if hi is not None:
                m &= dts_t < hi
            ql = float(qlike_arr(rv_t[m], h_t[m]).mean())
            mse, mae = mse_mae(rv_t[m], h_t[m])
            ra[doan_] = dict(n=int(m.sum()), qlike=ql, mse=mse, mae=mae, crps=mae)
        BANG["TTM"] = ra
    else:
        print("  (thiếu output/ttm_raw.npz — bỏ qua TTM)")

    # ── tổ hợp tốt nhất (tái tạo trực tiếp bằng hàm đã có)
    goc = "HAR vòng 7 (khớp mỗi phiên)"
    to_hop = {
        "Tổ hợp HAR+GRU (đều tay)": (["HAR vòng 7 (khớp mỗi phiên)", "GRU (khớp năm)"], "deu"),
        "Tổ hợp HAR+GRU+CatBoost (hồi quy GR)":
            (["HAR vòng 7 (khớp mỗi phiên)", "GRU (khớp năm)", "CatBoost"], "gr"),
    }
    for nhan, (ten_full, kieu) in to_hop.items():
        if not all(t in F for t in ten_full):
            continue
        if kieu == "deu":
            h = tb_hinh_hoc(F, ten_full)
        else:
            du_bao, _ = gr_hoi_quy(F, ten_full, log_rv, va)
            te = dts >= TEST_TU
            h = np.where(np.isnan(du_bao(te)), du_bao(va), du_bao(te))
        BANG[nhan] = cham_tu_F({nhan: h}, rv, dts, nhan)

    # ── in bảng kiểu bài báo + ghi file
    xep = sorted(BANG, key=lambda k: BANG[k].get("kiem_tra", {}).get("qlike", 9))
    print("\n" + "=" * 100)
    print(f"{'Mô hình':<38}{'QLIKE(vđ)':>10}{'QLIKE(kt)':>10}{'MSE(kt)':>10}"
          f"{'MAE(kt)':>10}{'CRPS(kt)':>10}")
    print("-" * 100)
    for k in xep:
        v = BANG[k]
        vd = v.get("kiem_dinh", {})
        kt = v.get("kiem_tra", {})
        print(f"{k:<38}{vd.get('qlike', float('nan')):>10.4f}"
              f"{kt.get('qlike', float('nan')):>10.4f}{kt.get('mse', float('nan')):>10.4f}"
              f"{kt.get('mae', float('nan')):>10.4f}{kt.get('crps', float('nan')):>10.4f}")
    print("-" * 100)
    print(f"(MSE/MAE/CRPS tính trên RV×{TL:.0e}; CRPS = MAE cho mô hình dự báo điểm,"
          f" xem docstring)")

    # ── bảng chi tiết theo từng cặp (QLIKE kiểm tra) cho TOP mô hình
    top8 = xep[:8]
    import volfc2 as V2
    Pn = V2.PAIRS
    n1 = len(dts) // len(Pn)
    pid_arr = np.concatenate([np.full(n1, j) for j in range(len(Pn))])
    te_mask = dts >= TEST_TU

    def h_theo_ten(k):
        """Tra ve mang du bao toan cuc cho mo hinh/to hop `k` (None neu la
        Chronos/TTM — xu ly rieng vi khong nam trong F)."""
        if k in to_hop:
            ten_full, kieu = to_hop[k]
            if kieu == "deu":
                return tb_hinh_hoc(F, ten_full)
            du_bao, _ = gr_hoi_quy(F, ten_full, log_rv, va)
            return np.where(np.isnan(du_bao(te_mask)), du_bao(va), du_bao(te_mask))
        ten = DON_LE.get(k)
        return F[ten] if ten in F else None

    BANG_CAP = {}
    for k in top8:
        BANG_CAP[k] = {}
        if k == "Chronos-bolt-small" and os.path.exists(pc):
            for p in Pn:
                m = (dts_c >= TEST_TU) & (dch["pair"] == p) & np.isfinite(h_c) & (h_c > 0)
                if m.sum() >= 10:
                    BANG_CAP[k][p] = float(qlike_arr(rv_c[m], h_c[m]).mean())
            continue
        if k == "TTM" and os.path.exists(pt):
            for p in Pn:
                m = (dts_t >= TEST_TU) & (dtt["pair"] == p) & np.isfinite(h_t) & (h_t > 0)
                if m.sum() >= 10:
                    BANG_CAP[k][p] = float(qlike_arr(rv_t[m], h_t[m]).mean())
            continue
        h_all = h_theo_ten(k)
        if h_all is None:
            continue
        for j, p in enumerate(Pn):
            m = te_mask & (pid_arr == j) & np.isfinite(h_all) & (h_all > 0) & np.isfinite(rv) & (rv > 0)
            if m.sum() >= 10:
                BANG_CAP[k][p] = float(qlike_arr(rv[m], h_all[m]).mean())

    print(f"\nQLIKE KIỂM TRA THEO TỪNG CẶP — {len(top8)} mô hình dẫn đầu")
    print("-" * 100)
    print(f"{'cặp':<10}" + "".join(f"{k[:14]:>15}" for k in top8))
    print("-" * 100)
    for p in Pn:
        line = f"{p:<10}"
        for k in top8:
            v = BANG_CAP.get(k, {}).get(p, float("nan"))
            line += f"{v:>15.4f}"
        print(line)
    print("-" * 100)

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(tong_hop=BANG, theo_cap=BANG_CAP),
              open(os.path.join(OUT, "bang_baocao.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    rows = []
    for k in xep:
        v = BANG[k]
        for doan_ in ("kiem_dinh", "kiem_tra"):
            if doan_ in v:
                rows.append(dict(mo_hinh=k, doan=doan_, **v[doan_]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bang_baocao.csv"), index=False)
    print("\n→ output/bang_baocao.json, output/bang_baocao.csv")


if __name__ == "__main__":
    main()
