"""THỬ NGHIỆM — RuleFit (Friedman & Popescu 2008, "Predictive Learning via
Rule Ensembles," Annals of Applied Statistics) làm phương pháp THAY THẾ cho
CART nông của H3 (`run_h3_rulelist.py`) — cùng dữ liệu, cùng đặc trưng, cùng
mục tiêu, chỉ đổi lớp mô hình luật.

VÌ SAO SO SÁNH: H3 chỉ thử MỘT họ mô hình luật — cây quyết định nông
(max_depth=3, một cấu hình duy nhất, không đối chiếu). RuleFit là một họ
khác hẳn: sinh nhiều luật ứng viên từ RỪNG cây nông (không phải MỘT cây),
rồi dùng hồi quy Lasso để CHỌN LỌC tổ hợp luật + đặc trưng tuyến tính tốt
nhất — linh hoạt hơn một cây đơn, và là phương pháp chuẩn trong thống kê
học máy cho bài toán "tìm luật diễn giải được" mà H3 đang hỏi.

KHÁC BIỆT VỀ KHUNG ĐÁNH GIÁ — cần nói rõ: H3 dùng khung "liệt kê giả thuyết
trước + Westfall-Young" (đúng cho MỘT cây cố định, số lá biết trước). RuleFit
tự sinh tập luật ứng viên PHỤ THUỘC DỮ LIỆU (rừng cây ngẫu nhiên bên trong),
nên KHÔNG có một "không gian giả thuyết cố định biết trước" để áp Westfall-
Young theo đúng nghĩa. Ở đây đánh giá RuleFit theo cách khác, cũng đã dùng
xuyên suốt dự án: Brier Skill Score (BSS) so với khí hậu học trên đoạn KIỂM
ĐỊNH (chưa từng dùng để khớp), cộng kiểm định Diebold-Mariano cho hiệu số
điểm Brier giữa hai mô hình. Đây là so sánh CHẤT LƯỢNG DỰ BÁO, không phải
đếm giả thuyết sống sót — công bằng vì cả hai đều được chấm trên CÙNG một
đoạn chưa từng thấy.

GIAO THỨC: khớp trên HUẤN LUYỆN (đoạn 0), CHỌN không tham số nào trên kiểm
định (RuleFit tự chọn alpha Lasso bằng cross-validation NỘI BỘ trên chính
đoạn huấn luyện, tham số `cv=True` mặc định — không nhìn kiểm định), báo cáo
BSS/DM trên KIỂM ĐỊNH (đoạn 1). KHÔNG chạm đoạn kiểm tra.

Chạy:  python src/kiem_rulefit_h3.py
Ghi:   output/kiem_rulefit_h3.json
Cần:   pip install imodels
"""
import json
import os
import sys
import warnings

import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

from run_quyluat import nap_du_lieu                            # noqa: E402
from run_h3_rulelist import xay_cay, MAX_DEPTH, MIN_LA          # noqa: E402

from imodels import RuleFitClassifier                            # noqa: E402

SEED = 0
N_LOP = 3
TEN_LOP = ("giảm", "đi ngang", "tăng")


def dm_nw(x):
    """Diebold-Mariano, phuong sai Newey-West — dung lai nguyen ham chuan
    cua repo (src/run_final7.py) de nhat quan."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); mb = x.mean(); L = int(np.ceil(1.5 * n ** (1 / 3)))
    s = np.sum((x - mb) ** 2) / n
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * np.sum((x[k:] - mb) * (x[:-k] - mb)) / n
    t = mb / np.sqrt(max(s, 1e-16) / n)
    return t, 2 * (1 - stats.norm.cdf(abs(t)))


def brier_moi_hang(P, y):
    """Diem Brier nhieu-lop cho TUNG hang (khong gop trung binh) — de con
    dua vao DM test tren chuoi hieu so."""
    Y = np.eye(N_LOP)[y]
    return np.sum((P - Y) ** 2, axis=1)


def rulefit_ova(X_tr, y_tr, X_ev, seed=SEED):
    """RuleFit MOT-CHOI-PHAN-CON-LAI cho 3 lop, chuan hoa xac suat ve tong 1.

    imodels.RuleFitClassifier la bo phan loai NHI PHAN (dung Lasso tren tap
    luat sinh tu rung cay) — khop 3 lan (moi lop la "1" so voi "khong phai
    lop do"), roi chuan hoa. Day la ky thuat da-lop-tu-nhi-phan chuan, khong
    phai bien the tuy tien."""
    P_tr = np.zeros((len(X_ev), N_LOP))
    for k in range(N_LOP):
        yk = (y_tr == k).astype(int)
        if yk.sum() < 50 or (yk == 0).sum() < 50:      # qua it mau lop nay
            P_tr[:, k] = yk.mean() if len(yk) else 1 / N_LOP
            continue
        rf = RuleFitClassifier(max_rules=30, tree_size=4, random_state=seed, cv=True)
        rf.fit(X_tr, yk)
        p1 = np.asarray(rf.predict_proba(X_ev))[:, 1]
        P_tr[:, k] = p1
    P_tr = np.clip(P_tr, 1e-6, None)
    return P_tr / P_tr.sum(1, keepdims=True)


def main():
    du = nap_du_lieu()
    zs, dts, Ms, sigs = du["zs"], du["dts"], du["Ms"], du["sigs"]
    y, tr, va = du["y"], du["tr"], du["va"]

    print("Đang dựng đặc trưng hạng phân vị (giống hệt H3)…")
    X, ten_dt, ok = xay_cay(zs, dts, Ms, sigs, tr)
    tr_ok, va_ok = tr & ok, va & ok
    print(f"đặc trưng: {', '.join(ten_dt)}")
    print(f"huấn luyện: {tr_ok.sum():,} hàng · kiểm định (báo cáo): {va_ok.sum():,} hàng")

    X_tr, y_tr = X[tr_ok], y[tr_ok]
    X_ev, y_ev = X[va_ok], y[va_ok]

    # ── A: CART nong, dung DUNG cau hinh H3 (max_depth=3, min_samples_leaf=200) ──
    cay = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LA,
                                 random_state=SEED)
    cay.fit(X_tr, y_tr)
    P_cart = cay.predict_proba(X_ev)

    # ── B: RuleFit, mot-choi-phan-con-lai ────────────────────────────────
    print("Đang khớp RuleFit (3 lần, một-chọi-phần-còn-lại)…")
    P_rf = rulefit_ova(X_tr, y_tr, X_ev)

    # ── moc khi hau hoc: tan suat lop tren HUAN LUYEN ────────────────────
    tan_suat = np.bincount(y_tr, minlength=N_LOP) / len(y_tr)
    P_kh = np.tile(tan_suat, (len(y_ev), 1))

    b_kh = brier_moi_hang(P_kh, y_ev)
    b_cart = brier_moi_hang(P_cart, y_ev)
    b_rf = brier_moi_hang(P_rf, y_ev)

    bss_cart = 1 - b_cart.mean() / b_kh.mean()
    bss_rf = 1 - b_rf.mean() / b_kh.mean()
    t_cart, p_cart_vs_kh = dm_nw(b_kh - b_cart)
    t_rf, p_rf_vs_kh = dm_nw(b_kh - b_rf)
    t_rf_cart, p_rf_vs_cart = dm_nw(b_cart - b_rf)     # >0 nghia la RuleFit loi hon CART

    ra = dict(
        n_train=int(len(y_tr)), n_bao_cao=int(len(y_ev)),
        cart=dict(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LA,
                  n_la=int(len(set(cay.apply(X_tr).tolist()))),
                  bss=round(float(bss_cart), 5),
                  dm_vs_khihauhoc_p=round(float(p_cart_vs_kh), 5)),
        rulefit=dict(max_rules=30, tree_size=4,
                    bss=round(float(bss_rf), 5),
                    dm_vs_khihauhoc_p=round(float(p_rf_vs_kh), 5)),
        rulefit_vs_cart=dict(dm_t=round(float(t_rf_cart), 4),
                            dm_p=round(float(p_rf_vs_cart), 5),
                            chenh_bss=round(float(bss_rf - bss_cart), 5)),
        ket_luan=("RuleFit thắng CART có ý nghĩa (p<0,05)" if p_rf_vs_cart < 0.05 and bss_rf > bss_cart
                  else "CART thắng RuleFit có ý nghĩa" if p_rf_vs_cart < 0.05 and bss_cart > bss_rf
                  else "không khác biệt có ý nghĩa giữa hai họ mô hình luật"))
    print(json.dumps(ra, ensure_ascii=False, indent=1))

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "kiem_rulefit_h3.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print("Đã ghi", outp)
    return ra


if __name__ == "__main__":
    main()
