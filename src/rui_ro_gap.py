"""HAI LOAI RUI RO HE THONG CHUA CO: gap qua dem/cuoi tuan, va thanh khoan.

`KEHOACH_2026Q4.md` muc 2.2 neu hai loai rui ro he thong ma he thong KHONG do:

  "Rui ro nhay gia (gap). Cuoi tuan va ngay le, gia mo cua nhay khoi gia dong.
   Stop-loss KHONG bao ve duoc trong khoang do — nha dau tu giu lenh qua cuoi
   tuan dang chiu rui ro he thong khong he do."
  "Rui ro thanh khoan. Hien chi co spread theo gio. Chua co: spread gian ra bao
   nhieu trong ngay su kien."

DAY LA PHEP DO MO TA, KHONG PHAI MOT THI NGHIEM.
  - khong chon mo hinh, khong mo doan kiem tra de chon gi
  - khong co gia thuyet can bac bo, nen khong dem vao KHOA_SO muc 5 nhu mot
    "cau hinh mo hinh" — no khong phai mo hinh
  - dung de BAO CAO mot rui ro dang ton tai ma nguoi dung khong thay

Phan vi stop lay tu `position_sizing`/`api` de con so khop voi thu he thong
thuc su hien len giao dien: stop o k * sigma^ voi k trong {1,0; 1,5; 2,0}.

Chay:  python src/rui_ro_gap.py
Ghi:   output/rui_ro_gap.json
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")
SPREAD = os.path.join(ROOT, "data", "spread_hourly_all.csv")
SUKIEN = os.path.join(ROOT, "data", "su_kien.csv")
EPS = 1e-12
K_STOP = (1.0, 1.5, 2.0)

import volfc2 as V2                                # noqa: E402


def bang_gap():
    """Gap = log(open_t / close_{t-1}), kem sigma^ du bao cho phien t va so ngay nghi."""
    bang, _ = V2.nap_bang()
    hang = []
    for p in V2.PAIRS:
        d = bang[p]
        ng = pd.DatetimeIndex(d.Date)
        o, c = d.open.values, d.close.values
        sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(d, p), 0.0))
        gap = np.full(len(d), np.nan)
        gap[1:] = np.log(np.maximum(o[1:], EPS) / np.maximum(c[:-1], EPS))
        nghi = np.full(len(d), np.nan)
        nghi[1:] = (ng[1:] - ng[:-1]).days
        hang.append(pd.DataFrame(dict(pair=p, ngay=ng, gap=gap, sig=sig,
                                      ngay_nghi=nghi, thu=ng.dayofweek)))
    df = pd.concat(hang, ignore_index=True)
    return df[np.isfinite(df.gap) & np.isfinite(df.sig) & (df.sig > 0)
              & np.isfinite(df.ngay_nghi)].copy()


def do_gap(df):
    df["z_gap"] = np.abs(df.gap) / df.sig
    cuoi_tuan = df.ngay_nghi >= 3          # bac qua >= 3 ngay lich = cuoi tuan/le
    ra = {}
    print("\n[1] ĐỘ LỚN GAP — |gap| tính theo đơn vị σ̂ của chính phiên đó")
    print(f"  {'nhóm':<22}{'n':>7}{'trung vị':>10}{'p90':>9}{'p99':>9}{'tối đa':>9}")
    print("  " + "-" * 66)
    for ten, m in (("trong tuần (1 ngày)", ~cuoi_tuan), ("cuối tuần/lễ (≥3 ngày)", cuoi_tuan)):
        z = df.z_gap[m].values
        ra[ten] = dict(n=int(len(z)), trung_vi=float(np.median(z)),
                       p90=float(np.percentile(z, 90)),
                       p99=float(np.percentile(z, 99)), toi_da=float(z.max()))
        print(f"  {ten:<22}{len(z):>7,}{np.median(z):>10.3f}"
              f"{np.percentile(z,90):>9.3f}{np.percentile(z,99):>9.3f}{z.max():>9.3f}")
    ty = (ra["cuối tuần/lễ (≥3 ngày)"]["trung_vi"]
          / ra["trong tuần (1 ngày)"]["trung_vi"])
    print(f"  → gap cuối tuần lớn hơn gap trong tuần {ty:.2f}× (trung vị)")
    ra["ty_cuoi_tuan_tren_trong_tuan"] = float(ty)
    return ra, cuoi_tuan


def do_stop(df, cuoi_tuan):
    """Xac suat gap NHAY QUA stop, va truot bao nhieu khi da nhay."""
    ra = {}
    print("\n[2] STOP-LOSS BỊ NHẢY QUA — gap vượt ngưỡng stop trước khi stop kịp khớp")
    print(f"  {'stop tại':<12}{'nhóm':<24}{'P(nhảy qua)':>13}{'trượt thêm TB':>16}")
    print("  " + "-" * 67)
    for k in K_STOP:
        for ten, m in (("trong tuần", ~cuoi_tuan), ("cuối tuần/lễ", cuoi_tuan)):
            z = df.z_gap[m].values
            qua = z > k
            pr = float(qua.mean())
            # truot them = phan vuot qua stop, tinh theo sigma^ (chi khi da nhay)
            tr = float(np.mean(z[qua] - k)) if qua.any() else 0.0
            ra[f"k={k} {ten}"] = dict(p_nhay=pr, truot_them_sigma=tr,
                                      n=int(len(z)))
            print(f"  {k:<12.1f}{ten:<24}{100*pr:>12.2f}%{tr:>15.3f}σ̂")
    print("  → \"trượt thêm\" là phần lỗ VƯỢT QUA mức stop, đơn vị σ̂ — stop không chặn được")
    return ra


def do_thanh_khoan():
    """Spread gian bao nhieu trong ngay su kien — PHAN TANG THEO GIO.

    Phai phan tang: spread co chu ky noi ngay rat manh, va su kien My tap trung
    vao vai gio nhat dinh. Neu gop ca ngay thi hieu ung gio se tron voi hieu ung
    su kien. So sanh trong TUNG o (cap, gio), roi gop ty le.
    """
    if not (os.path.exists(SPREAD) and os.path.exists(SUKIEN)):
        print()
        print("[3] THANH KHOẢN: thiếu dữ liệu, bỏ qua")
        return None
    sp = pd.read_csv(SPREAD, parse_dates=["Date"])
    sk = pd.read_csv(SUKIEN, parse_dates=["date"])
    ngay_sk = set(sk.date.dt.normalize().dropna())
    sp = sp.dropna(subset=["spread_med"])
    sp["co_sk"] = sp.Date.dt.normalize().isin(ngay_sk)
    print()
    print("[3] THANH KHOẢN — spread ngày CÓ sự kiện so ngày thường, "
          "PHÂN TẦNG theo (cặp, giờ)")
    n_sk = int(sp.co_sk.sum()); n_th = int((~sp.co_sk).sum())
    print(f"  {n_sk:,} ô giờ ngày sự kiện · {n_th:,} ô giờ ngày thường "
          f"({100*n_sk/(n_sk+n_th):.0f}% ngày có sự kiện — lịch dày)")
    ty = []
    for (p_, h), g in sp.groupby(["pair", "hour"]):
        a, b = g.spread_med[g.co_sk], g.spread_med[~g.co_sk]
        if len(a) < 30 or len(b) < 30 or b.mean() <= 0:
            continue
        ty.append(a.mean() / b.mean())
    if not ty:
        print("  không đủ mẫu trong ô nào")
        return None
    ty = np.array(ty)
    from scipy import stats as _st
    t, pv = _st.ttest_1samp(ty, 1.0)
    print(f"  {len(ty)} ô (cặp × giờ) đủ mẫu · tỉ lệ trung bình "
          f"{ty.mean():.4f} · trung vị {np.median(ty):.4f}")
    print(f"  khoảng 90%: [{np.percentile(ty,5):.4f}; {np.percentile(ty,95):.4f}]")
    print(f"  t-test tỉ lệ = 1: t={t:.2f}, p={pv:.4f}")
    qua = int((ty > 1.0).sum())
    print(f"  → {qua}/{len(ty)} ô có spread RỘNG HƠN trong ngày sự kiện "
          f"({100*qua/len(ty):.0f}%)")
    kl = ("spread giãn có ý nghĩa" if pv < 0.05 and ty.mean() > 1
          else "KHÔNG phân biệt được với không giãn")
    print(f"  → kết luận: {kl}")
    return dict(n_o=int(len(ty)), ty_tb=float(ty.mean()),
                ty_trung_vi=float(np.median(ty)), t=float(t), p=float(pv),
                o_rong_hon=qua, ty_le_ngay_su_kien=float(n_sk/(n_sk+n_th)),
                ket_luan=kl)


def main():
    t0 = time.time()
    print("=" * 96)
    print("HAI LOẠI RỦI RO HỆ THỐNG CHƯA ĐO — gap qua đêm/cuối tuần, và thanh khoản")
    print("PHÉP ĐO MÔ TẢ. Không chọn mô hình, không mở đoạn kiểm tra để chọn gì.")
    print("=" * 96)

    df = bang_gap()
    print(f"{len(df):,} phiên có đủ gap và σ̂ · {df.pair.nunique()} cặp")
    g, ct = do_gap(df)
    s = do_stop(df, ct)
    tk = do_thanh_khoan()

    # theo tung cap — gap cuoi tuan
    print("\n[4] GAP CUỐI TUẦN THEO TỪNG CẶP (|gap|/σ̂)")
    print(f"  {'cặp':<9}{'n':>6}{'trung vị':>10}{'p99':>9}{'P(>1,5σ̂)':>11}")
    print("  " + "-" * 47)
    theo_cap = {}
    for p in sorted(df.pair.unique()):
        z = df.z_gap[(df.pair == p) & ct].values
        if len(z) < 50:
            continue
        theo_cap[p] = dict(n=int(len(z)), trung_vi=float(np.median(z)),
                           p99=float(np.percentile(z, 99)),
                           p_qua_1_5=float((z > 1.5).mean()))
        print(f"  {p:<9}{len(z):>6}{np.median(z):>10.3f}"
              f"{np.percentile(z,99):>9.3f}{100*(z>1.5).mean():>10.2f}%")

    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(do_lon_gap=g, stop=s, thanh_khoan=tk, theo_cap=theo_cap,
                   k_stop=list(K_STOP)),
              open(os.path.join(OUT, "rui_ro_gap.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n→ output/rui_ro_gap.json · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
