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

from scipy import stats                                                # noqa: E402
from split import VALID_TU, TEST_TU                                    # noqa: E402
from kiem_tohop2 import nap_du_bao, tb_hinh_hoc, gr_hoi_quy, qlike_arr, EPS  # noqa: E402
from kiem_tohop3 import nap_them                                        # noqa: E402
import crps as C                                                        # noqa: E402

TL = 1e4   # he so nhan cho MSE/MAE/CRPS de bang de doc (RV goc rat nho)


def mse_mae(rv, h):
    e = (rv - h) * TL
    return float(np.mean(e ** 2)), float(np.mean(np.abs(e)))


def s2_tu_huan_luyen(h, rv, mask_tr):
    """Phuong sai cua phan du log-RV, uoc CHI tren doan HUAN LUYEN.

    Moi mo hinh trong bang nay deu du bao log-RV roi quy doi
    h = exp(mu + s^2/2) — tuc no NGAM dinh phan phoi log-chuan LN(mu, s^2)
    cho RV, khong phai mot diem. Vi log(h) = mu + s^2/2 nen phan du
    log(rv) - log(h) co dung phuong sai s^2, va mu = log(h) - s^2/2.

    Uoc tren huan luyen (khong phai kiem dinh/kiem tra) de khong ro ri."""
    m = mask_tr & np.isfinite(h) & (h > 0) & np.isfinite(rv) & (rv > 0)
    if m.sum() < 200:
        return np.nan
    return float(np.var(np.log(rv[m]) - np.log(h[m])))


def crps_lognormal_tu_h(rv, h, s2):
    """CRPS THAT cho phan phoi log-chuan ngam dinh boi (h, s^2)."""
    if not np.isfinite(s2) or s2 <= 0:
        return float("nan"), float("nan")
    s = np.sqrt(s2)
    mu = np.log(h) - s2 / 2.0
    cr = float(np.mean(C.crps_lognormal(mu, s, rv))) * TL
    # do phu khoang TRUNG TAM 80% (phan vi 10-90). Chon 80% chu khong 90% vi
    # Chronos chi cho 9 phan vi 0,1..0,9 — de moi mo hinh trong bang dung
    # CUNG mot muc, khong phai ngoai suy duoi cua Chronos.
    lo = np.exp(mu + s * stats.norm.ppf(0.10))
    hi = np.exp(mu + s * stats.norm.ppf(0.90))
    phu = float(np.mean((rv >= lo) & (rv <= hi)))
    return cr, phu


def _tu_kiem_tai_tao():
    """Tu kiem: tu (h, phan du huan luyen) co tai tao dung s^2 that khong."""
    rng = np.random.default_rng(3)
    n = 6000
    mu_that = rng.normal(-10.5, 0.6, n)
    s2_that = 0.35
    rv = np.exp(mu_that + rng.normal(0, np.sqrt(s2_that), n))
    h = np.exp(mu_that + s2_that / 2)                    # dung cach quy doi
    tr = np.zeros(n, bool); tr[: n // 2] = True
    s2_uoc = s2_tu_huan_luyen(h, rv, tr)
    lech = abs(s2_uoc - s2_that) / s2_that
    ok = lech < 0.08
    print(f"  tự kiểm tái tạo s²: ước {s2_uoc:.4f} so với thật {s2_that:.4f} "
          f"(lệch {lech:.1%})  {'ĐẠT' if ok else 'THẤT BẠI'}")
    return ok


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


def cham_tu_F(F, rv, dts, ten, s2_ngoai=None):
    h = F[ten]
    tr = dts < VALID_TU
    s2 = s2_tu_huan_luyen(h, rv, tr) if s2_ngoai is None else s2_ngoai
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
        cr, phu = crps_lognormal_tu_h(rv[m], h[m], s2)
        ra[doan_] = dict(n=int(m.sum()), qlike=ql, mse=mse, mae=mae,
                         crps=cr, phu90=phu, s2=float(s2))
    return ra


def main():
    print("=" * 100)
    print("XUẤT BẢNG TỔNG HỢP KIỂU BÀI BÁO — QLIKE / MSE / MAE / CRPS")
    print("=" * 100)
    assert C._tu_kiem(im_lang=True), "tự kiểm module CRPS thất bại — dừng"
    assert _tu_kiem_tai_tao(), "tự kiểm tái tạo s² thất bại — dừng"
    print()

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
            # do phu 80% tu CHINH phan vi 0,1 va 0,9 cua Chronos
            phu = float(np.mean((rv_c[m] >= Q_c[m][:, 0])
                                & (rv_c[m] <= Q_c[m][:, 8])))
            ra[doan_] = dict(n=int(m.sum()), qlike=ql, mse=mse, mae=mae,
                             crps=cr, phu90=phu, s2=float("nan"))
        BANG["Chronos-bolt-small"] = ra
    else:
        print("  (thiếu output/chronos_raw.npz — bỏ qua Chronos)")

    # TTM (raw npz riêng)
    pt = os.path.join(OUT, "ttm_raw.npz")
    if os.path.exists(pt):
        dtt = np.load(pt, allow_pickle=True)
        dts_t = pd.DatetimeIndex(dtt["date"])
        rv_t, h_t = dtt["rv"], dtt["h"]
        # s^2 = 2*he_so_hieu_chinh, uoc tren HUAN LUYEN trong chinh kiem_ttm.py
        s2_t = dtt["s2"] if "s2" in dtt else np.full(len(h_t), np.nan)
        ra = {}
        for doan_, (lo, hi) in (("kiem_dinh", (VALID_TU, TEST_TU)),
                                ("kiem_tra", (TEST_TU, None))):
            m = (dts_t >= lo) & np.isfinite(h_t) & (h_t > 0) & np.isfinite(rv_t)
            if hi is not None:
                m &= dts_t < hi
            ql = float(qlike_arr(rv_t[m], h_t[m]).mean())
            mse, mae = mse_mae(rv_t[m], h_t[m])
            # s^2 khac nhau theo cap -> tinh CRPS tung hang roi lay trung binh
            s2m = s2_t[m]
            mu_t = np.log(h_t[m]) - s2m / 2.0
            cr = float(np.mean(C.crps_lognormal(mu_t, np.sqrt(s2m), rv_t[m]))) * TL
            lo_ = np.exp(mu_t + np.sqrt(s2m) * stats.norm.ppf(0.10))
            hi_ = np.exp(mu_t + np.sqrt(s2m) * stats.norm.ppf(0.90))
            phu = float(np.mean((rv_t[m] >= lo_) & (rv_t[m] <= hi_)))
            ra[doan_] = dict(n=int(m.sum()), qlike=ql, mse=mse, mae=mae,
                             crps=cr, phu90=phu, s2=float(np.mean(s2m)))
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
            BANG[nhan] = cham_tu_F({nhan: h}, rv, dts, nhan)
        else:
            du_bao, coef = gr_hoi_quy(F, ten_full, log_rv, va)
            te = dts >= TEST_TU
            h = np.where(np.isnan(du_bao(te)), du_bao(va), du_bao(te))
            # To hop GR chi co du bao tren kiem dinh+kiem tra (he so khop tren
            # kiem dinh), nen KHONG uoc duoc s^2 tu doan huan luyen nhu cac mo
            # hinh khac. Dung chinh phuong sai du CUA HOI QUY do — `gr_hoi_quy`
            # tra ve hc = 0,5*var(phan du), nen s^2 = 2*hc. Day la cung mot
            # dai luong, uoc tren cung du lieu ma he so duoc khop.
            BANG[nhan] = cham_tu_F({nhan: h}, rv, dts, nhan,
                                    s2_ngoai=2.0 * float(coef["hc"]))

    # ── in bảng kiểu bài báo + ghi file
    xep = sorted(BANG, key=lambda k: BANG[k].get("kiem_tra", {}).get("qlike", 9))
    print("\n" + "=" * 112)
    print(f"{'Mô hình':<38}{'QLIKE(vđ)':>10}{'QLIKE(kt)':>10}{'MSE(kt)':>10}"
          f"{'MAE(kt)':>10}{'CRPS(kt)':>10}{'phủ 80%':>10}")
    print("-" * 112)
    for k in xep:
        v = BANG[k]
        vd = v.get("kiem_dinh", {})
        kt = v.get("kiem_tra", {})
        print(f"{k:<38}{vd.get('qlike', float('nan')):>10.4f}"
              f"{kt.get('qlike', float('nan')):>10.4f}{kt.get('mse', float('nan')):>10.4f}"
              f"{kt.get('mae', float('nan')):>10.4f}{kt.get('crps', float('nan')):>10.4f}"
              f"{kt.get('phu90', float('nan')):>10.1%}")
    print("-" * 112)
    print(f"(MSE/MAE/CRPS tính trên RV×{TL:.0e}. CRPS dùng PHÂN PHỐI dự báo:")
    print(" log-chuẩn LN(log h − s²/2, s²) với s² = var(phần dư log-RV) ước trên")
    print(" HUẤN LUYỆN; riêng Chronos dùng 9 phân vị thật. 'phủ 80%' = tỉ lệ RV")
    print(" thực rơi vào khoảng trung tâm 80% của phân phối đó — càng gần 80%")
    print(" càng hiệu chuẩn tốt.)")

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
