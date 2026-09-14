"""Logic tang rui ro duoi (VaR/ES + backtest) va xuat xu tung con so tren
phieu rui ro — tach khoi router de endpoint /risk chi con la lop dieu phoi
tham so + goi ham, khong lan logic tinh toan."""
import os

import numpy as np

from api.config import MUC_VAR, ROOT, doan
from api.utils import pip_size, sang_pip


def var_es(pair, K, z_tr, gia, von=10000.0, don_bay=1.0):
    """TANG RUI RO DUOI — VaR va ES, kem BACKTEST cua chinh no.

    VaR muc alpha = phan vi alpha cua loi suat phien toi. ES = ky vong loi suat
    KHI DA roi vao duoi VaR — tuc "neu ngay xau xay ra thi lo trung binh bao
    nhieu". VaR mot minh khong du: no noi nguong, khong noi do sau.

    Uoc bang PHAN VI THUC NGHIEM cua z (khong gia dinh chuan) nhan sigma^ hom
    nay. Day dung la cach `src/evaluate2.py` da lam va da dat het backtest.

    Kem backtest chay TAI CHO tren doan KIEM TRA:
      Kupiec        ty le vi pham co dung bang alpha khong
      Christoffersen  cac lan vi pham co dinh cum khong
      DQ            manh hon ca hai (Engle & Manganelli 2004)
    p >= 0,05 = khong bac bo duoc. Voi ~720 phien moi cap thi luc kiem dinh
    RAT THAP — phai noi ra, khong duoc doc "dat" thanh "da chung minh".
    """
    import sys as _s
    from api.config import SRC
    if SRC not in _s.path:
        _s.path.insert(0, SRC)
    from metrics import kupiec, christoffersen_ind, dq_test

    pan = K["pan"]
    sig = pan.sig.values
    sg = float(sig[-1])
    ps = pip_size(pair)
    g = doan(pan.Date.values)
    z_all = pan.zT.values
    te = (g == 2) & np.isfinite(z_all) & np.isfinite(sig) & (sig > 0)

    ra = {"von_mau": von, "muc": []}
    for a in MUC_VAR:
        qz = float(np.quantile(z_tr, a))               # phan vi z tren HUAN LUYEN
        ez = float(np.mean(z_tr[z_tr <= qz])) if (z_tr <= qz).any() else qz
        v_r, e_r = qz * sg, ez * sg                    # loi suat (am)
        # backtest tren doan kiem tra: nguong di dong theo sigma^ tung phien
        vt = qz * sig[te]
        y = z_all[te] * sig[te]
        hits = (y <= vt).astype(int)
        _, pk, ph = kupiec(hits, a)
        _, pi_ = christoffersen_ind(hits)
        _, pdq = dq_test(hits, vt, a)
        # ES co du sau khong: trong CHINH nhung phien vi pham, lo thuc te TB
        # co bang ES da du bao khong? < 1 la mo hinh danh gia THAP muc lo.
        m = hits.astype(bool)
        es_du = float(np.mean(ez * sig[te][m])) if m.any() else float("nan")
        es_th = float(np.mean(y[m])) if m.any() else float("nan")
        r4 = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 4)
        ra["muc"].append({
            "alpha": a,
            "var_pip": round(float(sang_pip(abs(v_r), gia, pair)), 1),
            "es_pip": round(float(sang_pip(abs(e_r), gia, pair)), 1),
            "var_usd": round(von * don_bay * abs(v_r), 0),
            "es_usd": round(von * don_bay * abs(e_r), 0),
            "n_kiem_tra": int(te.sum()), "n_vi_pham": int(hits.sum()),
            "ty_le_vi_pham": r4(ph), "ky_vong": a,
            "kupiec_p": r4(pk), "chris_p": r4(pi_), "dq_p": r4(pdq),
            "es_du_bao_pip": round(float(sang_pip(abs(es_du), gia, pair)), 1) if np.isfinite(es_du) else None,
            "es_thuc_te_pip": round(float(sang_pip(abs(es_th), gia, pair)), 1) if np.isfinite(es_th) else None,
            "es_ty_le": r4(es_du / es_th) if np.isfinite(es_du) and np.isfinite(es_th) and es_th != 0 else None,
            "dat": bool((pk is None or not np.isfinite(pk) or pk >= 0.05)
                        and (pi_ is None or not np.isfinite(pi_) or pi_ >= 0.05)
                        and (pdq is None or not np.isfinite(pdq) or pdq >= 0.05)),
        })
    ra["canh_bao_luc"] = (
        f"Ở mức 1% chỉ kỳ vọng ~{round(0.01*int(te.sum()))} lần vi phạm trên "
        f"{int(te.sum())} phiên — lực kiểm định THẤP. \"Không bác bỏ được\" "
        f"không có nghĩa là \"đã chứng minh đúng\".")
    return ra


def xuat_xu_rui_ro(pair, z_tr, nu, cr, sizer):
    """XUAT XU TUNG CON SO tren phieu rui ro.

    Nha dau tu chi tin duoc neu thay: con so nay ra tu CONG THUC nao, uoc tren
    BAO NHIEU mau, va cai gi CHUNG MINH no dung. Cot cuoi la chi so da do —
    QLIKE/MAE/CRPS/PIT cho tang sigma, ty le phu thuc te cho tang khoang, so
    lan cham stop cho tang truot gia. Khong dong nao la tham so dat tay.
    """
    import json
    f = os.path.join(ROOT, "output", "chiso_mohinh.json")
    C = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    q = ((C.get("cap") or {}).get(pair) or {}).get("kiểm tra") or {}
    g = (C.get("gop") or {}).get("kiểm tra") or {}
    r2 = lambda v, n=2: None if v is None else round(float(v), n)

    return {
        "doan_do": "kiểm tra (2023-11-20 → nay), chưa từng dùng để khớp",
        "n_sigma": q.get("n"), "n_z": int(len(z_tr)),
        "muc": [
            {"ten": "σ̂ — biên độ dao động dự kiến",
             "cong_thuc": "HAR vòng 7: log RV = f(ngày, tuần, tháng) + hiệu chỉnh "
                          "realized quarticity + semivariance ± + bipower/jump + "
                          "lịch NHTW riêng từng cặp",
             "de_hieu": "Dự đoán hôm nay giá sẽ dao động bao nhiêu pip, dựa trên mức "
                        "dao động của hôm qua, tuần qua, tháng qua, cộng thêm ngày họp "
                        "ngân hàng trung ương đã biết trước.",
             "uoc_tren": f"{q.get('n', 0):,} phiên đoạn kiểm tra",
             "chi_so": [("QLIKE", r2(q.get("qlike"), 4), "0 là hoàn hảo; gộp 6 cặp "
                         f"{r2(g.get('qlike'), 4)}"),
                        ("MAE", r2(q.get("mae_sigma_pip")), "pip — sai số tuyệt đối "
                         "trung bình của chính σ̂"),
                        ("RMSE", r2(q.get("rmse_sigma_pip")), "pip — phạt nặng lần "
                         "trượt lớn"),
                        ("CRPS", r2(q.get("crps_pip")), "pip — chấm CẢ PHÂN PHỐI, "
                         "không chỉ điểm giữa"),
                        ("PIT (KS p)", r2(q.get("pit_ks_p"), 4),
                         "p < 0,05 nghĩa là hình dạng phân phối bị BÁC BỎ"),
                        ("độ phủ 90%", r2(q.get("do_phu_90"), 4),
                         "phải gần 0,90; thấp hơn = σ̂ hụt, khoảng quá hẹp")]},
            {"ten": "Dừng lỗ = 2 σ̂",
             "cong_thuc": "quét 1,0–4,0 σ̂ trên lưới, chọn theo tiền cuối kỳ có "
                          "trừ trượt giá thật",
             "de_hieu": "Thử mọi khoảng dừng lỗ từ hẹp tới rộng trên toàn bộ lịch sử, "
                        "chọn khoảng cho nhiều tiền nhất SAU KHI đã trừ phí và trượt giá.",
             "uoc_tren": "docs/TANG6B_DUNGTOIUU.md — 60.617 lần chạm stop đã đo trượt giá",
             "chi_so": [("hệ số cắt", 0.92, "trượt giá thực làm mất 8% so với giả "
                         "định khớp đúng giá stop")]},
            {"ten": "P(chạm dừng lỗ) theo tầm hạn",
             "cong_thuc": "mô phỏng chạm rào trên PHÂN PHỐI z THỰC NGHIỆM (không "
                          "giả định chuẩn), ngưỡng 2σ̂/√h",
             "de_hieu": "Đếm trên dữ liệu thật xem giá đã chạm mức dừng lỗ đó bao nhiêu "
                        "lần, chứ không giả định giá đi theo đường cong lý thuyết.",
             "uoc_tren": f"{len(z_tr):,} phiên huấn luyện của chính cặp này",
             "chi_so": [("bậc tự do t", r2(nu), "đuôi càng dày ν càng nhỏ; ν<10 là "
                         "đuôi rất dày")]},
            {"ten": "Kelly",
             "cong_thuc": "f* = lợi thế / σ̂² — lợi thế lấy từ CARRY đo được, "
                          "KHÔNG dùng dự báo hướng",
             "de_hieu": "Cỡ lệnh tối ưu về dài hạn. Lợi thế duy nhất hệ thống dùng là "
                        "chênh lệch lãi suất giữa hai đồng tiền — KHÔNG dùng dự đoán hướng.",
             "uoc_tren": "carry trung vị 260 phiên gần nhất của cặp này",
             "chi_so": [("carry", r2(1e4 * cr), "bp/ngày")]},
            {"ten": "Trần rủi ro phá sản",
             "cong_thuc": "ngân sách phá sản 1% trên chuỗi tổn thất đuôi t(ν), "
                          "rồi nhân hệ số trượt giá 0,92",
             "de_hieu": "Trần cứng: cỡ lệnh lớn nhất mà xác suất cháy tài khoản vẫn "
                        "dưới 1%, đã tính cả những phiên giá nhảy bất thường.",
             "uoc_tren": "toàn mạch ~26 năm, mọi cấu hình đều không cháy tài khoản",
             "chi_so": [("vốn cuối", "1,004–1,037", "lần — lợi thế NHỎ, đây là sự thật")]},
            {"ten": "Hệ số danh mục 1/√(k+k(k−1)ρ)",
             "cong_thuc": "ρ hiệu dụng đo theo chế độ; căng thẳng thì ρ nhảy lên",
             "de_hieu": "Sáu cặp đều có USD nên chúng cùng thắng cùng thua. Mở nhiều "
                        "lệnh thì phải thu nhỏ từng lệnh, nếu không rủi ro cộng dồn.",
             "uoc_tren": "docs/TANG4_DANHMUC.md",
             "chi_so": [("không cắt", "73,6%", "xác suất phá sản khi mở 6 lệnh cùng "
                         "chiều USD mà không thu nhỏ — thay vì 1%")]},
        ]}
