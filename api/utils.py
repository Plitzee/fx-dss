"""Ham thuan (khong I/O): doi don vi, ep kieu JSON, dinh dang tap conformal.

Tach rieng vi day la lop bao ve re nhat va bi goi nhieu nhat trong toan bo
API — moi con so hien tren giao dien deu di qua `sang_pip` va `_py` truoc khi
ra JSON. Xem tests/test_helpers.py.
"""
import numpy as np

from api.config import PIP


def pip_size(p):
    return PIP.get(p, 0.0001)


def sang_pip(v, gia, pair):
    """Doi mot dai luong TUONG DOI (do lech chuan cua loi suat log) sang pip.

    Phai nhan voi MUC GIA roi moi chia co pip. Chia thang co pip la sai: voi
    EURUSD gia ~1,16 thi gan dung nen loi khong lo ra, nhung voi USDJPY gia
    ~156 thi lech 156 lan — man hinh tung hien sigma 0,4 pip thay vi ~65."""
    return np.asarray(v, float) * np.asarray(gia, float) / pip_size(pair)


def _py(o):
    """Ep kieu numpy ve kieu Python thuan — lop chan cuoi truoc khi ra JSON.
    np.int64/np.float64 khong tuan tu hoa duoc, va NaN/Inf thi khong hop le
    trong JSON nen doi thanh null."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    return o


TEN_LOP = ("giảm", "đi ngang", "tăng")


def _tap_conformal(X, i):
    """Tap du bao conformal cho hang i — kem BAO DAM do phu 90%.

    Ba xac suat noi "kha nang bao nhieu"; tap nay noi "loai tru duoc gi". Kich
    thuoc tap la thu doc duoc ngay: 3 nghia la khong loai duoc gi, 2 nghia la
    loai duoc mot kha nang. Do phu 90% giu duoc ke ca khi thi truong doi che
    do — da do o docs/CHISO_DANHGIA.md muc 16."""
    tap = X.get("tap")
    if tap is None or i >= len(tap):
        return {"tap_du_bao": None, "tap_kich_thuoc": None,
                "conformal_alpha": None,
                "tap_ghi_chu": "chưa đủ dữ liệu hiệu chuẩn"}
    lop = [TEN_LOP[j] for j in range(3) if tap[i, j]]
    al = X.get("alpha_aci")
    return {"tap_du_bao": lop, "tap_kich_thuoc": len(lop),
            "conformal_alpha": round(float(al[i]), 4) if al is not None else None,
            "tap_ghi_chu": f"bảo đảm phủ 90% (ACI) — {len(lop)}/3 lớp còn lại"}
