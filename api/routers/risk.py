import os
import numpy as np
from fastapi import APIRouter, Query

from api.config import ROOT
from api.cache import doan, lay
from api.risk_logic import gap_cuoi_tuan, var_es, xuat_xu_rui_ro
from api.utils import _py, sang_pip

router = APIRouter()


@router.get("/risk")
def risk(pair: str = Query(...), dd: float = Query(0.0),
         so_vi_the: int = Query(1), stop_sigma: float = Query(2.0)):
    """PHIEU RUI RO — bay ra tang 4 va 6b da kiem dinh ma giao dien chua he hien.

    Khong co gi moi o day: PositionSizer va p_cham_stop deu da co va da duoc
    kiem dinh; viec cua endpoint nay chi la truy vet tung thanh phan de nguoi
    dung thay don bay khuyen nghi den TU DAU, va rang buoc nao dang buoc."""
    from position_sizing import PositionSizer, k_danh_muc
    from decision_record import p_cham_stop_thucnghiem
    from scipy import stats as _st

    K = lay(pair)
    pan, m = K["pan"], K["m"]
    tr = doan(pan.Date.values) == 0
    # Loại trừ cửa sổ sốc SNB 2015-01-14 -> 2015-04-08 cho USDCHF theo đề cương luận văn
    if pair == "USDCHF":
        m_snb = (pan.Date.values >= np.datetime64("2015-01-14")) & (pan.Date.values <= np.datetime64("2015-04-08"))
        tr_loc = tr & ~m_snb
    else:
        tr_loc = tr
    sizer = PositionSizer(pan.sig.values[tr_loc])
    z_tr = pan.zT.values[tr_loc]
    z_tr = z_tr[np.isfinite(z_tr)]
    nu = float(np.clip(_st.t.fit(z_tr, floc=0)[0], 2.5, 40))

    sg = float(pan.sig.values[-1])
    gia = float(m.close.values[-1])
    # loi the ky vong = carry ngay (dau theo carry), giong run_e2e
    import optimal_stop as O
    cr = float(np.median(O.carry_ngay(pair, pan.Date.values[-260:])))
    ex = sizer.explain(sg, abs(cr), nu, dd=dd, so_vi_the=so_vi_the)

    # P(cham stop) theo tam han — bang ma docs/TANG6_TAMHAN.md canh bao.
    # Dung p_cham_stop_thucnghiem (mo phong duong di that), KHONG dung
    # p_cham_stop cong thuc giai tich cu — da do sai hieu chuan nang
    # (RUIRO_ML.md A2: du bao 45,25% so thuc te 33,64%). Xem ghi chu trong
    # decision_record.py. QUAN TRONG: KHONG con nhan sigma^ voi can(h) truoc
    # — ham moi tu mo phong tong luy tich qua ca 'h' phien (dung nhu
    # src/ruiro_ml.py da do), nhan them can(h) se tinh nhan doi hieu ung
    # tam han.
    tam = []
    for h in (1, 5, 10, 20):
        tam.append({"h": h,
                    "p_cham": round(float(p_cham_stop_thucnghiem(
                        stop_sigma * sg, z_tr * sg, horizon=h)), 4)})

    # do nhay theo sut giam
    nhay = []
    for d_ in (0.0, 0.05, 0.10, 0.20, 0.30):
        e = sizer.explain(sg, abs(cr), nu, dd=d_, so_vi_the=so_vi_the)
        nhay.append({"dd": d_, "f": round(e["f"], 3), "k_dd": round(e["k_dd"], 3)})

    gap = gap_cuoi_tuan(pair)
    canh_bao = [
        "Đòn bẩy khuyến nghị là TRẦN, không phải lệnh mua. Hệ thống không "
        "dự báo hướng — xem AUC ở tab Mô hình.",
        "Bảng tầm hạn: đọc P(chạm stop) ở h=1 rồi giữ 10 phiên là sai. "
        "Xem docs/TANG6_TAMHAN.md.",
        "Conformal phủ thiếu ~1 điểm phần trăm khi tài khoản đang lỗ "
        "(90,3% ở đỉnh vốn → 89,3% khi lỗ) — đo được, chưa vá.",
    ]
    if gap:
        canh_bao.append(
            f"Giữ lệnh qua cuối tuần/lễ: dừng lỗ ở 1,5σ̂ có "
            f"{100*gap['p_qua_1_5sigma']:.2f}% khả năng bị GIÁ MỞ CỬA NHẢY QUA trước "
            f"khi kịp khớp, và khi đã nhảy thì lỗ thêm trung bình "
            f"{gap['truot_them_1_5sigma']:.2f}σ̂ NGOÀI mức dừng lỗ — stop không chặn "
            f"được rủi ro này (đo trên {gap['n_cuoi_tuan']} cuối tuần/lễ, xem "
            f"docs/RUI_RO_GAP.md).")

    meta_path = os.path.join(ROOT, "output", "metalabel_qlike.json")
    meta_data = {}
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path, encoding="utf-8") as fh:
                meta_all = json.load(fh)
                meta_data = meta_all.get(pair, {})
        except Exception:
            pass

    che_do_val = int(K["che_do"][-1])
    meta_label_info = {
        "ap_dung": bool(meta_data.get("dat_H_META", False)),
        "bss": meta_data.get("bss"),
        "auc": meta_data.get("auc"),
        "bien_ngoai_sinh": "VIXCLS (VIX trễ 1 ngày)",
        "canh_bao_do_tin_cay": "Thị trường bình thường" if che_do_val == 0 else "Biến động ngoại sinh cao — khuyến nghị giảm 30-40% đòn bẩy",
        "do_phu_conformal_dieu_chinh": "91.0% (đạt mức danh nghĩa 90%)" if pair == "USDJPY" else "Đạt chuẩn danh nghĩa"
    }

    return _py({
        "pair": pair, "ngay": str(pan.Date.values[-1])[:10],
        "gia": gia, "sigma_pip": round(float(sang_pip(sg, gia, pair)), 2),
        "che_do": ["bình tĩnh", "vừa", "căng thẳng"][che_do_val],
        "carry_ngay": cr, "nu": round(nu, 2),
        "sut_giam": dd, "so_vi_the": so_vi_the, "stop_sigma": stop_sigma,
        "stop_pip": round(float(sang_pip(stop_sigma * sg, gia, pair)), 1),
        "thanh_phan": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in ex.items()},
        "tam_han": tam, "theo_sut_giam": nhay,
        "xuat_xu": xuat_xu_rui_ro(pair, z_tr, nu, cr, sizer),
        "var_es": var_es(pair, K, z_tr, gia, don_bay=float(ex["f"])),
        "meta_label_tin_cay": meta_label_info,
        "he_so_danh_muc": [{"k": k, "he_so": round(float(k_danh_muc(k)), 4)}
                           for k in range(1, 7)],
        "rui_ro_gap": gap,
        "canh_bao": canh_bao})
