"""VÁ ĐUÔI — HƯỚNG THỨ NĂM: lịch CAN THIỆP/KHỦNG HOẢNG (khác lịch họp ĐỊNH KỲ
đã thử và thất bại ở mục 5e `CHISO_DANHGIA.md`). Nới VaR/ES trong cửa sổ K
ngày sau một sự kiện can thiệp tiền tệ KHÔNG ĐỊNH KỲ, đã công khai, đã biết
trước — không phải lịch họp thường lệ.

DANH SÁCH SỰ KIỆN — CHỈ lấy từ sự kiện ĐÃ CÔNG KHAI, ĐÃ ĐƯỢC CHÍNH DỰ ÁN GHI
LẠI TRƯỚC (`docs/KETQUA_VONG7.md` mục "Về USDJPY — tại sao KHÔNG bỏ"), không
tự bịa hay suy đoán ngày mới:
    USDJPY: 2022-09-22 (BOJ can thiệp lần đầu từ 1998), 2022-10-21 (can thiệp
            lớn nhất), 2022-10-24 (can thiệp tiếp), 2022-12-20 (BOJ nới YCC)
    USDCHF: 2015-01-15 (SNB bỏ sàn EUR/CHF)

CẢNH BÁO PHƯƠNG PHÁP LUẬN — PHẢI ĐỌC TRƯỚC KHI TIN KẾT QUẢ:
  (1) Cả 4 ngày BOJ nằm trong đoạn KIỂM ĐỊNH (`VALID_TU`=2021-10-13 đến
      `TEST_TU`=2023-11-20) — dùng được để kiểm chứng thật. Nhưng ngày SNB
      2015 nằm trong đoạn HUẤN LUYỆN (trước 2021-10-13) — USDCHF do đó
      KHÔNG có sự kiện kiểm định thật nào để đối chiếu; kết quả cho USDCHF
      ở đây chỉ mang tính minh hoạ cơ chế, không phải kiểm chứng.
  (2) HỆ SỐ NỚI RỘNG: vì chỉ có 4 điểm dữ liệu cho USDJPY, KHÔNG thể ước
      lượng hệ số nới từ chính 4 ngày đó rồi test lại trên đúng 4 ngày đó —
      đó là quá khớp trần trụi. Script này dùng CÁC HỆ SỐ TRÒN, chọn TRƯỚC
      khi xem kết quả khớp thế nào (1,5×, 2×, 3×) — không tinh chỉnh theo
      dữ liệu. Đây vẫn là một dạng "biết trước sự kiện" (tác giả đã đọc
      docs/KETQUA_VONG7.md trước khi viết script này) — không phải một phép
      thử "mù" hoàn toàn, phải khai báo rõ, không giấu.
  (3) Đây KHÔNG giải quyết được cơ chế TRÔI THANG ĐO đơn điệu của USDJPY đã
      chẩn đoán ở `src/va_duoi.py` (sd(z) tăng dần theo đoạn, không phải
      chỉ vài ngày cực đoan) — cửa sổ can thiệp chỉ nhắm đúng các ngày sự
      kiện, không sửa được xu hướng trôi nền tảng.

Chạy:  python src/va_duoi_canthiep.py
Ghi:   output/va_duoi_canthiep.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import balop as B                                             # noqa: E402
import volfc2 as V2                                           # noqa: E402
from metrics import kupiec, christoffersen_ind, dq_test        # noqa: E402
from split import doan, VALID_TU, TEST_TU                      # noqa: E402
from volfc import merge_thin_days                               # noqa: E402

MUC = (0.05, 0.01)
K_CUA_SO = 5              # so phien sau su kien duoc coi la "trong can thiep"
HE_SO_NOI = (1.5, 2.0, 3.0)   # tron, chot TRUOC khi xem ket qua khop the nao
EPS = 1e-12

LICH_CAN_THIEP = {
    "USDJPY": ["2022-09-22", "2022-10-21", "2022-10-24", "2022-12-20"],
    "USDCHF": ["2015-01-15"],
}


def nap(pair):
    from api.main import noi_chuoi
    m = merge_thin_days(noi_chuoi(pair))
    sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, pair), 0.0))
    c = m.close.values
    z = np.full(len(m), np.nan)
    z[1:] = np.log(c[1:] / np.maximum(c[:-1], EPS)) / np.maximum(sig[1:], EPS)
    g = doan(m.Date.values)
    return pd.DataFrame({"Date": m.Date.values, "z": z, "sig": sig, "doan": g})


def co_can_thiep(dates, ngay_su_kien, k=K_CUA_SO):
    co = np.zeros(len(dates), dtype=bool)
    ds = pd.DatetimeIndex(dates)
    for ns in ngay_su_kien:
        t0 = pd.Timestamp(ns)
        co |= (ds >= t0) & (ds < t0 + pd.Timedelta(days=k * 2))  # ~k phien giao dich, du du ngay le
    return co


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
    tr = d.doan.values == 0
    va = d.doan.values == 1
    z_tr = d.z.values[tr]
    z_tr = z_tr[np.isfinite(z_tr)]
    ct = co_can_thiep(d.Date.values, LICH_CAN_THIEP.get(pair, []))
    n_su_kien_trong_kiem_dinh = int((ct & va).sum() > 0) and len(
        [s for s in LICH_CAN_THIEP.get(pair, []) if VALID_TU <= pd.Timestamp(s) < TEST_TU])

    ra = {"pair": pair, "n_su_kien_trong_kiem_dinh": n_su_kien_trong_kiem_dinh,
         "n_ngay_can_thiep_trong_kiem_dinh": int((ct & va).sum()), "muc": {}}

    z_va = d.z.values[va]
    ct_va = ct[va]
    ok = np.isfinite(z_va)

    for a in MUC:
        q = float(np.quantile(z_tr, a))          # V0 — dong bang tren huan luyen
        var_v0 = np.full(ok.sum(), q)
        hits_v0 = (z_va[ok] <= q).astype(int)
        ket_qua_a = {"v0": cham(hits_v0, var_v0, a)}

        for lam in HE_SO_NOI:
            var_ct = np.where(ct_va[ok], q * lam, q)     # noi q theo he so TRON trong cua so can thiep
            hits_ct = (z_va[ok] <= var_ct).astype(int)
            ket_qua_a[f"noi_x{lam}"] = cham(hits_ct, var_ct, a)

        ra["muc"][str(a)] = ket_qua_a

    return ra


def main():
    ra = {p: chay_mot_cap(p) for p in ("USDJPY", "USDCHF")}
    print(json.dumps(ra, ensure_ascii=False, indent=1))
    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "va_duoi_canthiep.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("Đã ghi", outp)
    return ra


if __name__ == "__main__":
    main()
