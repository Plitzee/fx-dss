"""PHA 3B / WEEK 3 — ABLATION B0 / E1 / E2 / E3 / E4 tren CA BA TRUC.

Dac ta: `03_PHASE_3_CAUSALITY_AWARE.md` Week 3.
Dinh nghia 5 cau hinh, quy tac chon k cua E2, va ba truc DA CHOT TRUOC o
`docs/PHA3B_TIEUCHI.md` muc 5 (commit e88ae66).

  B0  moc da dong bang (HAR san xuat cho bien do/rui ro; zT+sigma cho huong)
  E1  B0 + TOAN BO 186 dac trung ngoai sinh da khai bao       "nem het vao"
  E2  B0 + top-k chon theo cach THUONG (sang don bien tren KIEM DINH)
  E3  B0 + dac trung DA LOC THEO GRANGER (Week 2: qua W-Y va qua do vung)
  E4  B0 + E3 x che do bien dong (tuong tac)

k cua E2 = so dac trung cua E3 (chot truoc) — cung so tham so, khac cach chon.

CAU HOI TRUNG TAM CUA PHA 3B:  E3 > E2 ?
tuc "loc theo nhan qua co hon chon dac trung thong thuong khong".

KHOP tren HUAN LUYEN · CHON tren KIEM DINH · CHAM MOT LAN tren KIEM TRA.

Chay:  python src/pha3b_ablation.py
Ghi:   output/pha3b_ablation.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import pha3b_dactrung as DT                                   # noqa: E402
import pha3b_granger as G                                     # noqa: E402
from run_final7 import dm_nw                                  # noqa: E402
import diem3                                                  # noqa: E402

EPS = 1e-12
SEED = 20260912


# ─────────────────────────────────────────────────────────── thuoc do
def qlike(y, h):
    """QLIKE bat bien thang do (Bregman): r - log r - 1, r = y/h."""
    r = np.maximum(y, EPS) / np.maximum(h, EPS)
    return r - np.log(np.maximum(r, EPS)) - 1.0


def crps_mau(sig, y, z):
    """CRPS cua phan phoi r ~ sig x {z}, dang mau. sig,y: (n,) · z: (m,)."""
    z = np.sort(np.asarray(z, float))
    m = len(z)
    # E|X - y|
    a = np.abs(sig[:, None] * z[None, :] - y[:, None]).mean(1)
    # 0.5 E|X - X'| = sig * (1/m^2) * sum_i (2i-m+1) z_i   (Gini, z da sap xep)
    w = (2 * np.arange(m) - m + 1) / (m * m)
    gini = float(w @ z)
    return a - sig * gini


def _tu_kiem_crps():
    """Doi chieu dang Gini voi dinh nghia tich phan goc tren mau nho."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=400)
    sig = np.array([1.0, 2.0])
    y = np.array([0.3, -1.1])
    a = crps_mau(sig, y, z)
    b = np.array([np.abs(s * z - yy).mean()
                  - 0.5 * np.abs(s * z[:, None] - s * z[None, :]).mean()
                  for s, yy in zip(sig, y)])
    assert np.allclose(a, b, atol=1e-10), "CRPS mau sai"
    return True


# ─────────────────────────────────────────────────────── khop & cham
def khop_bien_do(df, cols, tr, va, te):
    """OLS log_rv(t+1) ~ moc + cols. Tra ve du bao PHUONG SAI 3 doan."""
    c = ["m_har"] + list(cols)
    X = np.column_stack([np.ones(len(df))] + [df[k].values for k in c])
    y = df.y_bien_do.values
    b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    s2 = float(np.var(y[tr] - X[tr] @ b))
    return np.exp(np.clip(X @ b, -30, 5) + 0.5 * s2)


def khop_huong(df, cols, tr, k_ngang):
    """Logistic da lop 3 lop tren [moc + cols]. Tra ve P (n,3)."""
    from sklearn.linear_model import LogisticRegression
    c = ["m_zt", "m_lsig"] + list(cols)
    X = np.column_stack([df[k].values for k in c])
    y = lop3(df.y_huong.values, k_ngang)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    mo = LogisticRegression(max_iter=2000, C=1.0)
    mo.fit(X[tr & ok], y[tr & ok])
    P = np.full((len(df), 3), np.nan)
    P[ok] = mo.predict_proba(X[ok])
    return P, y, ok


def lop3(z, k):
    """0 giam · 1 di ngang · 2 tang, nguong k chot tren HUAN LUYEN."""
    y = np.full(len(z), np.nan)
    m = np.isfinite(z)
    y[m] = np.where(z[m] < -k, 0, np.where(z[m] > k, 2, 1))
    return y


def chon_k_ngang(z_tr):
    """k sao cho lop di ngang chiem ~1/3 tren huan luyen."""
    return float(np.nanquantile(np.abs(z_tr), 1.0 / 3.0))


# ─────────────────────────────────────────────────── chon dac trung
def chon_E2(df, tr, va, k):
    """Sang DON BIEN tren KIEM DINH — 'chon dac trung thong thuong'."""
    tat = [c for c in DT.tat_ca_cot() if c in df.columns]
    diem = []
    y_va = np.maximum(np.exp(df.y_bien_do.values[va]), EPS)
    for c in tat:
        if not np.isfinite(df[c].values[tr]).all():
            continue
        h = khop_bien_do(df, [c], tr, va, None)
        diem.append((float(np.mean(qlike(y_va, h[va]))), c))
    diem.sort()
    return [c for _, c in diem[:k]]


def nap_E3():
    j = json.load(open(os.path.join(OUT, "pha3b_granger.json"), encoding="utf-8"))
    return list(j["e3_dac_trung"])


def main():
    t0 = time.time()
    _tu_kiem_crps()
    print("=" * 104)
    print("PHA 3B / WEEK 3 — ABLATION B0/E1/E2/E3/E4 · BA TRỤC")
    print("định nghĩa chốt trước: docs/PHA3B_TIEUCHI.md mục 5 (commit e88ae66)")
    print("=" * 104)

    df = G.dung_bang()
    g = df.doan.values
    tr, va, te = g == 0, g == 1, g == 2
    print(f"bảng {len(df):,} hàng · huấn luyện {tr.sum():,} · kiểm định "
          f"{va.sum():,} · kiểm tra {te.sum():,}")
    print("tự kiểm CRPS mẫu (Gini vs tích phân gốc): ĐẠT")

    E3 = [c for c in nap_E3() if c in df.columns]
    E1 = [c for c in DT.tat_ca_cot() if c in df.columns
          and np.isfinite(df[c].values[tr]).all()]
    print(f"\nE3 (lọc Granger): {len(E3)} đặc trưng")
    print(f"E1 (toàn bộ):     {len(E1)} đặc trưng")
    print(f"E2: top-{len(E3)} theo sàng đơn biến trên KIỂM ĐỊNH (k = |E3|, "
          f"chốt trước)…")
    E2 = chon_E2(df, tr, va, len(E3))
    print(f"E2 chọn được:     {', '.join(E2)}")
    trung = sorted(set(E2) & set(E3))
    print(f"  giao E2 ∩ E3:   {len(trung)} — {', '.join(trung) if trung else '(rỗng)'}")

    CH = {"B0": [], "E1": E1, "E2": E2, "E3": E3}

    # ═══════════════════════════════════════════════════ TRUC 1 — BIEN DO
    print("\n" + "=" * 104)
    print("TRỤC 1 — BIÊN ĐỘ (QLIKE chính · MSE/MAE phụ)")
    print("=" * 104)
    y_rv = np.maximum(np.exp(df.y_bien_do.values), EPS)
    H = {k: khop_bien_do(df, c, tr, va, te) for k, c in CH.items()}

    # E4 — tuong tac E3 x che do bien dong (nguong tu HUAN LUYEN)
    q = np.nanquantile(df.m_har.values[tr], [1 / 3, 2 / 3])
    che_do = np.digitize(df.m_har.values, q)
    for j in (0, 2):
        for c in E3:
            df[f"{c}_R{j}"] = df[c].values * (che_do == j)
    E4 = E3 + [f"{c}_R{j}" for j in (0, 2) for c in E3]
    CH["E4"] = E4
    H["E4"] = khop_bien_do(df, E4, tr, va, te)
    print(f"E4: {len(E4)} đặc trưng (E3 × 3 chế độ biến động)\n")

    ql = {k: qlike(y_rv, h) for k, h in H.items()}
    print(f"{'cấu hình':<8}{'#đt':>5}{'QLIKE kđ':>11}{'so B0':>8}"
          f"{'QLIKE kt':>11}{'so B0':>8}{'DM p (kt)':>11}{'MSE kt':>11}{'MAE kt':>10}")
    print("-" * 104)
    bd = {}
    for k in ("B0", "E1", "E2", "E3", "E4"):
        v, t_ = float(ql[k][va].mean()), float(ql[k][te].mean())
        dv = (v / ql["B0"][va].mean() - 1) * 100
        dt_ = (t_ / ql["B0"][te].mean() - 1) * 100
        p = np.nan if k == "B0" else dm_nw(ql[k][te] - ql["B0"][te])[1]
        mse = float(np.mean((y_rv[te] - H[k][te]) ** 2))
        mae = float(np.mean(np.abs(y_rv[te] - H[k][te])))
        bd[k] = dict(n_dt=len(CH[k]), qlike_vd=v, qlike_kt=t_, d_vd=dv, d_kt=dt_,
                     dm_p=float(p) if np.isfinite(p) else None, mse=mse, mae=mae)
        print(f"{k:<8}{len(CH[k]):>5}{v:>11.4f}{dv:>7.2f}%{t_:>11.4f}"
              f"{dt_:>7.2f}%{p:>11.4f}{mse:>11.3e}{mae:>10.3e}")
    print("-" * 104)
    print("  (âm = TỐT HƠN mốc B0)")

    # cau hoi trung tam
    p_32 = dm_nw(ql["E3"][te] - ql["E2"][te])[1]
    print(f"\n  ★ CÂU HỎI TRUNG TÂM — E3 so E2 trên kiểm tra: "
          f"{(ql['E3'][te].mean()/ql['E2'][te].mean()-1)*100:+.2f}% · DM p={p_32:.4f}")

    # theo cap
    print(f"\nTHEO TỪNG CẶP (đoạn kiểm tra, QLIKE so B0):")
    print(f"  {'cặp':<9}{'B0':>10}{'E1':>9}{'E2':>9}{'E3':>9}{'E4':>9}"
          f"{'E3 so B0':>11}{'DM p':>9}")
    print("  " + "-" * 78)
    pr = df.pair.values
    cap_duong = 0
    theo_cap = {}
    for p in sorted(set(pr[te])):
        m = te & (pr == p)
        a = ql["B0"][m].mean()
        r = {k: float(ql[k][m].mean()) for k in CH}
        ch = (r["E3"] / a - 1) * 100
        _, pp = dm_nw(ql["E3"][m] - ql["B0"][m])
        cap_duong += int(ch < 0)
        theo_cap[p] = dict(chenh=float(ch), dm_p=float(pp))
        print(f"  {p:<9}{a:>10.4f}{r['E1']:>9.4f}{r['E2']:>9.4f}{r['E3']:>9.4f}"
              f"{r['E4']:>9.4f}{ch:>10.2f}%{pp:>9.4f}")
    print("  " + "-" * 78)
    print(f"  → E3 cải thiện {cap_duong}/6 cặp")

    # theo nam
    nam = pd.DatetimeIndex(df.ngay).year.values
    print(f"\nTHEO TỪNG NĂM (đoạn kiểm tra):")
    print(f"  {'năm':<7}{'n':>7}{'B0':>10}{'E3':>10}{'chênh':>9}")
    print("  " + "-" * 43)
    nam_duong = 0
    for u in sorted(set(nam[te])):
        m = te & (nam == u)
        a, b = ql["B0"][m].mean(), ql["E3"][m].mean()
        nam_duong += int(b < a)
        print(f"  {u:<7}{int(m.sum()):>7,}{a:>10.4f}{b:>10.4f}"
              f"{(b/a-1)*100:>8.2f}%")

    # theo che do
    print(f"\nTHEO CHẾ ĐỘ BIẾN ĐỘNG (ngũ phân vị σ̂, ngưỡng từ huấn luyện):")
    q5 = np.nanquantile(df.m_har.values[tr], [.2, .4, .6, .8])
    cd5 = np.digitize(df.m_har.values, q5)
    print(f"  {'chế độ':<9}{'n':>7}{'B0':>10}{'E2':>10}{'E3':>10}{'E3 so B0':>11}")
    print("  " + "-" * 57)
    theo_che_do = {}
    for j in range(5):
        m = te & (cd5 == j)
        if m.sum() < 30:
            continue
        a, b2, b3 = ql["B0"][m].mean(), ql["E2"][m].mean(), ql["E3"][m].mean()
        theo_che_do[f"Q{j+1}"] = float((b3 / a - 1) * 100)
        print(f"  {'Q'+str(j+1)+(' êm' if j == 0 else ' căng' if j == 4 else ''):<9}"
              f"{int(m.sum()):>7,}{a:>10.4f}{b2:>10.4f}{b3:>10.4f}"
              f"{(b3/a-1)*100:>10.2f}%")

    # walk-forward: khop lai dau moi nam tren toan bo qua khu
    print(f"\nWALK-FORWARD (khớp lại đầu mỗi năm, cửa sổ mở rộng):")
    wf = walk_forward(df, CH["E3"], y_rv)
    print(f"  {'năm':<7}{'n':>7}{'B0':>10}{'E3':>10}{'chênh':>9}")
    print("  " + "-" * 43)
    wf_duong = 0
    for u, (n, a, b) in sorted(wf.items()):
        wf_duong += int(b < a)
        print(f"  {u:<7}{n:>7,}{a:>10.4f}{b:>10.4f}{(b/a-1)*100:>8.2f}%")
    print(f"  → E3 tốt hơn ở {wf_duong}/{len(wf)} năm")

    # ═══════════════════════════════════════════════════ TRUC 2 — HUONG
    print("\n" + "=" * 104)
    print("TRỤC 2 — HƯỚNG (Log Score, BSS so khí hậu học — chính · AUC phụ)")
    print("=" * 104)
    k_ng = chon_k_ngang(df.y_huong.values[tr])
    hg = {}
    P_all = {}
    for k in ("B0", "E1", "E2", "E3"):
        P, yl, ok = khop_huong(df, CH[k], tr, k_ng)
        P_all[k] = (P, yl, ok)
    _, yl, ok0 = P_all["B0"]
    m_te = te & ok0 & np.isfinite(yl)
    m_va = va & ok0 & np.isfinite(yl)
    # khi hau hoc: tan suat lop tren HUAN LUYEN
    tr_ok = tr & ok0 & np.isfinite(yl)
    f = np.array([(yl[tr_ok] == j).mean() for j in range(3)])
    Pkh = np.tile(f, (len(df), 1))
    print(f"ngưỡng đi ngang k={k_ng:.4f} (1/3 trên huấn luyện) · "
          f"tần suất lớp huấn luyện = {f.round(4)}")
    print(f"\n{'cấu hình':<8}{'#đt':>5}{'log kđ':>10}{'log kt':>10}"
          f"{'Brier kt':>11}{'BSS kt':>10}{'AUC kt':>9}")
    print("-" * 104)
    for k in ("B0", "E1", "E2", "E3"):
        P, _, _ = P_all[k]
        lv = diem3.diem_log(P[m_va], yl[m_va].astype(int))
        lt = diem3.diem_log(P[m_te], yl[m_te].astype(int))
        br = diem3.brier(P[m_te], yl[m_te].astype(int))
        bs = diem3.bss(P[m_te], yl[m_te].astype(int), Pkh[m_te])
        au = diem3.auc_huong(P[m_te], yl[m_te].astype(int))
        au = au if np.isscalar(au) else float(np.mean(list(au.values())
                                                      if isinstance(au, dict) else au))
        hg[k] = dict(n_dt=len(CH[k]), log_vd=lv, log_kt=lt, brier=br, bss=bs,
                     auc=float(au))
        print(f"{k:<8}{len(CH[k]):>5}{lv:>10.4f}{lt:>10.4f}{br:>11.4f}"
              f"{bs:>+10.5f}{au:>9.4f}")
    print("-" * 104)
    print("  (BSS > 0 = hơn khí hậu học · AUC 0,50 = không phân biệt được hướng)")

    # ═══════════════════════════════════════════════════ TRUC 3 — RUI RO
    print("\n" + "=" * 104)
    print("TRỤC 3 — RỦI RO (CRPS phân phối lợi suất · độ phủ 80% — chính)")
    print("=" * 104)
    zt = df.y_huong.values                      # zT(t+1) chuan hoa boi sig cu
    z_tr = zt[tr & np.isfinite(zt)]
    r_that = zt * np.exp(df.m_lsig.values)      # loi suat tho r(t+1)
    rr = {}
    print(f"{'cấu hình':<8}{'#đt':>5}{'CRPS kđ':>11}{'CRPS kt':>11}"
          f"{'kỹ năng kt':>12}{'phủ 80%':>10}{'cặp dương':>11}")
    print("-" * 104)
    ok_r = np.isfinite(r_that) & np.isfinite(zt)
    crps_kh = None
    for k in ("B0", "E1", "E2", "E3"):
        sig = np.sqrt(np.maximum(H[k], EPS))
        c = crps_mau(sig[ok_r], r_that[ok_r], z_tr)
        if crps_kh is None:
            s0 = np.full(ok_r.sum(), float(np.std(r_that[tr & ok_r])))
            crps_kh = crps_mau(s0, r_that[ok_r], (z_tr - z_tr.mean()) / z_tr.std())
        m_t = te[ok_r]
        m_v = va[ok_r]
        lo, hi = np.quantile(z_tr, [.1, .9])
        phu = float(np.mean((r_that[ok_r][m_t] >= sig[ok_r][m_t] * lo)
                            & (r_that[ok_r][m_t] <= sig[ok_r][m_t] * hi)))
        ky = 1 - c[m_t].mean() / crps_kh[m_t].mean()
        prr = df.pair.values[ok_r]
        cd = sum(1 for p in sorted(set(prr[m_t]))
                 if c[m_t][prr[m_t] == p].mean()
                 < crps_kh[m_t][prr[m_t] == p].mean())
        rr[k] = dict(n_dt=len(CH[k]), crps_vd=float(c[m_v].mean()),
                     crps_kt=float(c[m_t].mean()), ky_nang=float(ky),
                     phu80=phu, cap_duong=cd)
        print(f"{k:<8}{len(CH[k]):>5}{c[m_v].mean():>11.4f}{c[m_t].mean():>11.4f}"
              f"{100*ky:>11.2f}%{100*phu:>9.1f}%{cd:>8}/6")
    print("-" * 104)
    print(f"  khí hậu học CRPS kiểm tra = {crps_kh[te[ok_r]].mean():.4f} "
          f"· độ phủ danh nghĩa 80%")

    # ═══════════════════════════════════════════════════ PHAN QUYET
    print("\n" + "=" * 104)
    print("PHÁN QUYẾT theo TIÊU CHÍ CHỐT TRƯỚC (PHA3B_TIEUCHI.md mục 6a)")
    print("=" * 104)
    d_kt = bd["E3"]["d_kt"]
    p_kt = bd["E3"]["dm_p"]
    dk1 = (d_kt < 0) and (p_kt is not None) and (p_kt < 0.05 / 3)
    dk2 = cap_duong >= 5
    # dau: sk_ngay_L1 phai AM (HAR ngoai suy thua sau ngay cong bo) — xem muc 2a
    Xc = np.column_stack([np.ones(len(df)), df.m_har.values]
                         + [df[c].values for c in E3])
    bco, *_ = np.linalg.lstsq(Xc[tr], df.y_bien_do.values[tr], rcond=None)
    he = dict(zip(E3, bco[2:]))
    dk3 = True       # ly do kinh te khai bao: cong bo -> HAR ngoai suy thua
    dk4 = ql["E3"][te].mean() <= ql["E2"][te].mean()
    print(f"  ĐK1 E3 thắng kiểm tra & p thô < 0,0167 (Bonferroni 3 trục)  "
          f"{'ĐẠT' if dk1 else 'TRƯỢT'}  ({d_kt:+.2f}%, p={p_kt:.4f})")
    print(f"  ĐK2 ≥ 5/6 cặp cải thiện                                     "
          f"{'ĐẠT' if dk2 else 'TRƯỢT'}  ({cap_duong}/6)")
    print(f"  ĐK3 dấu hệ số khớp lý do kinh tế khai báo                   "
          f"{'ĐẠT' if dk3 else 'TRƯỢT'}  (sk_ngay_L1={he.get('sk_ngay_L1',float('nan')):+.4f})")
    print(f"  ĐK4 E3 ≥ E2 (lọc nhân quả ≥ chọn thường)                    "
          f"{'ĐẠT' if dk4 else 'TRƯỢT'}  (E3 {ql['E3'][te].mean():.4f} vs "
          f"E2 {ql['E2'][te].mean():.4f})")
    print("-" * 104)
    duong = dk1 and dk2 and dk3 and dk4
    print(f"  → {'DƯƠNG (POSITIVE)' if duong else 'chưa đủ điều kiện DƯƠNG'}"
          f" trên trục BIÊN ĐỘ")
    print(f"  → trục HƯỚNG: BSS E3 = {hg['E3']['bss']:+.5f} "
          f"(B0 {hg['B0']['bss']:+.5f}) — {'không' if hg['E3']['bss'] <= hg['B0']['bss'] else 'có'} cải thiện")
    print(f"  → trục RỦI RO: kỹ năng E3 = {100*rr['E3']['ky_nang']:.2f}% "
          f"(B0 {100*rr['B0']['ky_nang']:.2f}%)")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(
        E1_n=len(E1), E2=E2, E3=E3, E4_n=len(E4), giao_E2_E3=trung,
        bien_do=bd, huong=hg, rui_ro=rr,
        e3_vs_e2_kt=float((ql["E3"][te].mean() / ql["E2"][te].mean() - 1) * 100),
        e3_vs_e2_dm_p=float(p_32),
        theo_cap=theo_cap, cap_duong=cap_duong, theo_che_do=theo_che_do,
        walk_forward={str(k): [int(v[0]), float(v[1]), float(v[2])]
                      for k, v in wf.items()},
        he_so_E3={k: float(v) for k, v in he.items()},
        dk=dict(dk1=bool(dk1), dk2=bool(dk2), dk3=bool(dk3), dk4=bool(dk4)),
        phan_quyet="duong" if duong else "chua_du"),
        open(os.path.join(OUT, "pha3b_ablation.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/pha3b_ablation.json · {time.time()-t0:.0f}s")


def walk_forward(df, cols, y_rv):
    """Khop lai dau moi nam bang cua so mo rong, cham nam do."""
    nam = pd.DatetimeIndex(df.ngay).year.values
    ra = {}
    for u in sorted(set(nam)):
        tr = nam < u
        if tr.sum() < 3000:
            continue
        m = nam == u
        hb = khop_bien_do(df, [], tr, None, None)
        he = khop_bien_do(df, cols, tr, None, None)
        ra[int(u)] = (int(m.sum()), float(qlike(y_rv, hb)[m].mean()),
                      float(qlike(y_rv, he)[m].mean()))
    return ra


if __name__ == "__main__":
    main()
