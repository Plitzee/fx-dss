"""VÁ ĐUÔI — HƯỚNG THỨ SÁU: phát hiện trôi thang đo TRỰC TUYẾN (ADWIN) +
tái ước lượng thích nghi (EWMA), thay vì điều kiện hoá theo SỰ KIỆN cụ thể
(5 hướng trước, xem `docs/VA_DUOI_CANTHIEP_KETQUA.md`).

KHÁC BIỆT VỀ BẢN CHẤT so với 5 hướng trước: mọi hướng trước đều dựa vào
BIẾT TRƯỚC một mốc thời gian cụ thể (ngày họp, ngày can thiệp) — nên đều
vướng đúng một bế tắc: đoạn kiểm định không chứa đủ bằng chứng để phân biệt
cách vá nào đúng. Hướng này KHÔNG cần biết trước bất kỳ ngày nào — nó phát
hiện độ TRÔI THANG ĐO (đã chẩn đoán ở `va_duoi.py`: sd(z) USDJPY tăng đơn
điệu 1,014→1,100→1,136 qua ba đoạn) TỪ CHÍNH THỐNG KÊ NỘI TẠI của chuỗi,
bằng hai kỹ thuật học trực tuyến (online learning):

  1. ADWIN (Bifet & Gavaldà 2007, "Learning from Time-Changing Data with
     Adaptive Windowing") — thuật toán phát hiện điểm đổi phân phối trên
     luồng dữ liệu, so sánh trung bình hai nửa cửa sổ bằng cận Hoeffding.
     Cho |z| chạy qua ADWIN theo đúng THỨ TỰ THỜI GIAN (huấn luyện rồi
     kiểm định) — hoàn toàn không cần biết trước ngày sự kiện nào.
  2. EWMA (RiskMetrics, λ=0,94 — CÙNG HẰNG SỐ đã dùng cho nền EWMA biến
     động ở nơi khác trong repo, không tự chọn theo dữ liệu này) tái ước
     lượng PHƯƠNG SAI của z liên tục, để ngưỡng VaR "co giãn" theo độ biến
     động THỰC TẾ gần đây thay vì đóng băng vĩnh viễn ở mức ước trên huấn
     luyện.

GIAO THỨC CHỐNG RÒ RỈ: tại mỗi phiên t, CHỈ dùng dữ liệu đến hết phiên t-1
(ADWIN.update và EWMA đều cập nhật SAU khi đã dự báo cho t). Đánh giá HOÀN
TOÀN trên đoạn KIỂM ĐỊNH — không chạm kiểm tra.

Chạy:  python src/va_duoi_adwin.py
Ghi:   output/va_duoi_adwin.json
Cần:   pip install river
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from river.drift import ADWIN

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                             # noqa: E402
import volfc2 as V2                                           # noqa: E402
from metrics import kupiec, christoffersen_ind, dq_test        # noqa: E402
from split import doan                                         # noqa: E402
from volfc import merge_thin_days                               # noqa: E402

MUC = (0.05, 0.01)
LAM_EWMA = 0.94         # RiskMetrics chuan — CUNG hang so da dung cho nen EWMA khac trong repo
DELTA_ADWIN = 0.002     # mac dinh cua thu vien river, KHONG chinh theo du lieu nay
TOI_THIEU_SAU_RESET = 250   # can it nhat ngan nay quan sat sau diem doi de tin tuong phan vi moi
EPS = 1e-12


def nap(pair):
    from api.main import noi_chuoi
    m = merge_thin_days(noi_chuoi(pair))
    sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, pair), 0.0))
    c = m.close.values
    z = np.full(len(m), np.nan)
    z[1:] = np.log(c[1:] / np.maximum(c[:-1], EPS)) / np.maximum(sig[1:], EPS)
    g = doan(m.Date.values)
    return pd.DataFrame({"Date": m.Date.values, "z": z, "doan": g})


def cham(hits, var_series, alpha):
    _, pk, ty_le = kupiec(hits, alpha)
    _, pc = christoffersen_ind(hits)
    _, pdq = dq_test(hits, var_series, alpha)
    return dict(n_vi_pham=int(hits.sum()), ty_le_vi_pham=round(float(ty_le), 4),
               kupiec_p=None if pk is None or not np.isfinite(pk) else round(float(pk), 4),
               chris_p=None if pc is None or not np.isfinite(pc) else round(float(pc), 4),
               dq_p=None if pdq is None or not np.isfinite(pdq) else round(float(pdq), 4))


def chay_mot_cap(pair):
    d = nap(pair)
    ok = np.isfinite(d.z.values)
    d = d[ok].reset_index(drop=True)
    tr = d.doan.values == 0
    va = d.doan.values == 1
    z = d.z.values

    i_tr_cuoi = int(np.flatnonzero(tr)[-1])
    z_tr = z[tr]
    hinh_dang = z_tr / (z_tr.std() + EPS)     # "hinh dang" chuan hoa, dong bang tren huan luyen

    # ── 1. ADWIN chay QUA TOAN BO huan luyen + kiem dinh, THEO THU TU ────
    adwin = ADWIN(delta=DELTA_ADWIN)
    diem_doi = []          # chi so (trong mang da loc ok) noi ADWIN bao doi
    for t in range(len(z)):
        adwin.update(abs(float(z[t])))
        if adwin.drift_detected and t > i_tr_cuoi:   # chi quan tam diem doi RƠI VÀO kiem dinh tro di
            diem_doi.append(t)

    # ── 2. EWMA phuong sai — cap nhat SAU khi da dung cho phien hien tai ──
    var_ewma = np.full(len(z), np.nan)
    var_ewma[i_tr_cuoi] = float(np.var(z_tr))          # gieo hat = phuong sai huan luyen
    for t in range(i_tr_cuoi + 1, len(z)):
        var_ewma[t] = LAM_EWMA * var_ewma[t - 1] + (1 - LAM_EWMA) * z[t - 1] ** 2

    ra = {"pair": pair, "sd_z_huan_luyen": round(float(z_tr.std()), 4),
         "n_diem_doi_adwin_trong_kiem_dinh": len(diem_doi),
         "ngay_diem_doi_dau_tien": (str(pd.Timestamp(d.Date.values[diem_doi[0]]).date())
                                    if diem_doi else None),
         "muc": {}}

    z_va, idx_va = z[va], np.flatnonzero(va)

    for a in MUC:
        q_shape = float(np.quantile(hinh_dang, a))

        # V0 — moc dang chay: phan vi z co dinh tren huan luyen
        q_v0 = float(np.quantile(z_tr, a))
        hits_v0 = (z_va <= q_v0).astype(int)
        var_v0 = np.full(len(z_va), q_v0)

        # EWMA thich nghi — nguong = hinh dang x do lech chuan EWMA HIEN TAI
        sd_ewma_va = np.sqrt(np.maximum(var_ewma[idx_va], EPS))
        var_ewma_arr = q_shape * sd_ewma_va
        hits_ewma = (z_va <= var_ewma_arr).astype(int)

        # ADWIN-reset — phan vi uoc lai tren du lieu TU DIEM DOI GAN NHAT den t-1
        var_adwin = np.empty(len(z_va))
        for j, t in enumerate(idx_va):
            diem_truoc_t = [p for p in diem_doi if p < t]
            if diem_truoc_t and (t - diem_truoc_t[-1]) >= 0:
                bat_dau = diem_truoc_t[-1]
                cua_so = z[bat_dau:t]
                if len(cua_so) >= TOI_THIEU_SAU_RESET:
                    var_adwin[j] = np.quantile(cua_so, a)
                    continue
            var_adwin[j] = q_v0    # chua du du lieu sau diem doi -> lui ve V0
        hits_adwin = (z_va <= var_adwin).astype(int)

        ra["muc"][str(a)] = {
            "v0": cham(hits_v0, var_v0, a),
            "ewma_thich_nghi": cham(hits_ewma, var_ewma_arr, a),
            "adwin_reset": cham(hits_adwin, var_adwin, a),
        }

    return ra


def main():
    ra = {p: chay_mot_cap(p) for p in ("USDJPY", "USDCHF")}
    print(json.dumps(ra, ensure_ascii=False, indent=1))
    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "va_duoi_adwin.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("Đã ghi", outp)
    return ra


if __name__ == "__main__":
    main()
