"""Vercel Python Serverless Function — GET /api/decision

Tra ve JSON tuong duong PhieuQuyetDinh.lap() (src/decision_record.py) cong
muc "tang 6b" (khuyen nghi giu/dong), nhung tu du lieu DA TINH SAN
(web/api/_data/, xem src/export_ui_state.py) thay vi doc CSV+huan luyen lai
moi request — chi giai lai quy hoach dong (dp_core.giai) vi carry phu thuoc
NGAY nguoi dung chon.

Tham so query:
  pair        EURUSD | GBPUSD | USDJPY | AUDUSD | USDCAD | USDCHF
  date        YYYY-MM-DD, trong khoang [valid_tu, ngay cuoi chuoi]
  so_vi_the   so nguyen >=1, mac dinh 1
  dd          sut giam hien tai, 0..1, mac dinh 0
  stop_sigma  mac dinh 2.0
  von         mac dinh 10000

KHONG dung mu bia — mu (loi the Kelly) LUON la trung vi carry cua so mo rong
tinh DEN TRUOC ngay duoc chon, dung quy uoc "Vá 03/09/2026" cua
docs/TANG6_HIEU_CHUAN.md.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "_lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from scipy import stats

import dp_core as DP

_DATA = os.path.join(HERE, "_data")
_CACHE = {}

PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF")
N = 20  # tam han DP, dung quy uoc toan repo (optimal_stop.py, run_e2e.py...)

# HuyH — mau tuan tu bien dong (xem src/decision_record.py:LuatKyHieu — DONG
# BO voi bang o do, day chi la ban sao du lieu tinh, khong hoc lai).
LUAT = {
    (1, 2, 2): ("cao", 1.69, "vừa → cao → cao"),
    (0, 1, 0): ("thấp", 1.32, "thấp → vừa → thấp"),
    (2, 2, 1): ("cao", 1.69, "cao → cao → vừa"),
    (2, 2, 2): ("cao", 4.80, "cao → cao → cao"),
    (0, 0, 0): ("thấp", 1.41, "thấp → thấp → thấp"),
}


def _load_json(name):
    if name not in _CACHE:
        with open(os.path.join(_DATA, name), encoding="utf-8") as f:
            _CACHE[name] = json.load(f)
    return _CACHE[name]


def _meta():
    return _load_json("meta.json")


def _series(pair):
    key = f"series_{pair}"
    if key not in _CACHE:
        s = _load_json(f"{key}.json")
        _CACHE[key] = {
            "dates": s["dates"],  # list cua chuoi ISO, giu nguyen de so sanh lexicographic
            "sig": np.array(s["sig"], float),
            "zT": np.array(s["zT"], float),
            "zL": np.array(s["zL"], float),
            "rv5": np.array(s["rv5"], float),
            "close": np.array(s["close"], float),
            "carry": np.array(s["carry"], float),
        }
    return _CACHE[key]


def _regime():
    if "regime_np" not in _CACHE:
        r = _load_json("regime.json")
        nguong = np.array(r["nguong"], float)
        sig_v = np.array(r["sig_v"], float)
        mau = {int(v): {k: np.array(r["mau"][v][k], float if k != "v2" else int)
                        for k in ("zT", "zL", "ty", "v2")}
               for v in r["mau"]}
        _CACHE["regime_np"] = (nguong, sig_v, mau)
    return _CACHE["regime_np"]


def pip_size(pair):
    return 0.01 if "JPY" in pair.upper() else 0.0001


def f_kelly(mu, sig):
    return mu / sig ** 2


def f_ruin_cap(sig, horizon_days, budget, nu, ruin_level=0.5):
    s_h = sig * np.sqrt(horizon_days)
    z = stats.t.ppf(budget / 2.0, nu) / np.sqrt(nu / (nu - 2))
    b = -z * s_h
    loss_allowed = -np.log(ruin_level)
    return loss_allowed / max(b, 1e-9)


def k_danh_muc(so_vi_the, rho):
    k = int(so_vi_the)
    if k <= 1:
        return 1.0
    rho = float(np.clip(rho, -1.0 / (k - 1) + 1e-9, 1.0))
    return float(1.0 / np.sqrt(k + k * (k - 1) * rho))


def p_cham_stop(k_sigma, nu, sc, horizon=1):
    return float(min(1.0, 2.0 * stats.t.cdf(-k_sigma / (sc * np.sqrt(horizon)), nu)))


def che_do_2bin(sig, edge):
    return 1 if sig >= edge else 0


def che_do_nbin(sig, edges):
    return int(np.digitize([float(sig)], edges)[0])


def tinh_phieu(pair, ngay, so_vi_the, dd, stop_sigma, von, muc_list):
    meta = _meta()
    pm = meta["pairs"][pair]
    C = meta["const"]
    s = _series(pair)
    dates = s["dates"]

    if ngay not in dates:
        # lay ngay giao dich gan nhat <= ngay yeu cau (thi truong khong mo cuoi tuan/le)
        idx_arr = [i for i, d in enumerate(dates) if d <= ngay]
        if not idx_arr:
            raise ValueError(f"không có phiên nào trước {ngay} cho {pair}")
        idx = idx_arr[-1]
        ngay_dung = dates[idx]
    else:
        idx = dates.index(ngay)
        ngay_dung = ngay

    sig = float(s["sig"][idx])
    gia = float(s["close"][idx])
    nu = float(pm["nu"])
    t_scale = float(pm["t_scale"])

    # ── mu: trung vi carry CUA SO MO RONG, tinh DEN TRUOC ngay dang xet ──
    mask_truoc = np.array(dates) < ngay_dung
    if mask_truoc.sum() < 30:
        mu = float(np.median(s["carry"][: max(idx, 1)]))
    else:
        mu = float(np.median(s["carry"][mask_truoc]))

    # ── tang 4: PositionSizer.explain(), vá 03/09/2026 (chan f ve [0,lev_max]) ──
    s1, s2, s_stress = pm["s1"], pm["s2"], pm["s_stress"]
    t = float(np.clip((sig - s1) / max(s2 - s1, 1e-12), 0, 1))
    kv = C["k_vol_hi"] + (C["k_vol_lo"] - C["k_vol_hi"]) * t
    kd = float(np.clip(C["k_dd_hi"] - C["k_dd_slope"] * dd, C["k_dd_floor"], C["k_dd_hi"]))
    rho_eff = C["rho_cang_thang"] if sig >= s_stress else C["rho_mac_dinh"]
    kdm = k_danh_muc(so_vi_the, rho_eff)
    cap = C["k_slip"] * f_ruin_cap(sig, C["horizon"], C["budget"], nu)
    kelly = f_kelly(mu, sig)
    eff = kv * kd * kdm * cap
    lev_max = C["lev_max"]
    f = float(np.clip(min(kelly, eff), 0.0, lev_max))
    if kelly <= 0.0:
        rang_buoc = "không có lợi thế (Kelly ≤ 0)"
    elif f >= lev_max - 1e-9:
        rang_buoc = "trần đòn bẩy"
    else:
        rang_buoc = "Kelly" if kelly < eff else "trần rủi ro"
    muc_bien_dong = "thấp" if sig < s1 else ("cao" if sig > s2 else "vừa")

    # ── tang 6: P(cham stop), khoang gia, tam han ──
    p_stop = p_cham_stop(stop_sigma, nu, t_scale)
    reg2 = che_do_2bin(sig, pm["khoang_edges"][0])
    hw = pm["khoang_halfwidth"][str(reg2)]
    khoang = {}
    for m in muc_list:
        key = f"{m:.2f}"
        h = hw.get(key)
        if h is None:  # muc khong nam trong bang da tinh san -> noi suy don gian tu "chung"
            h = pm["khoang_halfwidth"]["chung"].get(key, hw["0.90"])
        h = h * sig
        khoang[m] = (gia * float(np.exp(-h)), gia * float(np.exp(h)))
    n_mau_che_do = pm["khoang_n_theo_che_do"][reg2]

    th = pm["tamhan"]
    reg3_tamhan = che_do_nbin(sig, th["edges"])
    bang_tam_han = []
    for h in th["tam_han"]:
        c_h = th["c"][str(h)][reg3_tamhan]
        sh_ratio = np.sqrt(h) * c_h
        bang_tam_han.append((h, p_cham_stop(stop_sigma / max(sh_ratio, 1e-12), nu, t_scale)))

    # ── luat ky hieu (HuyH) ──
    luat_ky_hieu = None
    if idx >= 2:
        q = pm["luat_q"]
        k3 = tuple(int(np.digitize([float(x)], q)[0]) for x in s["rv5"][idx - 2: idx + 1])
        if k3 in LUAT:
            dich, lift, nhan = LUAT[k3]
            luat_ky_hieu = (f"Ba phiên gần nhất: {nhan}. Trong lịch sử, sau mẫu này xác suất "
                            f"biến động {dich} cao gấp {lift:.2f} lần mức nền.")

    canh_bao_dong_luong = bool(sig >= s_stress)

    # ── tang 6b: giai DP khong don bay (chinh sach dang san xuat) ──
    nguong6b, sig_v6b, mau6b = _regime()
    c_thoat_sigma = pm["c_thoat_sigma"]
    slip_sigma = meta["slip_sigma"]
    V, bien = DP.giai(mau6b, sig_v6b, mu, c_thoat_sigma, slip_sigma, N=N, seed=1)
    reg6b = che_do_nbin(sig, nguong6b)
    giu_luc_vao = DP.nen_giu(V, stop_sigma, reg6b, N, c_thoat_sigma, sig_v6b)
    b = bien[N, reg6b]
    bien_gioi_sigma = None if not np.isfinite(b) else float(b)

    return dict(
        pair=pair, ngay=ngay_dung, gia=gia, von=float(von),
        mu_carry=mu, che_do_6b=reg6b,
        don_bay=f, von_dat=f * von, rang_buoc=rang_buoc,
        kelly=kelly, tran_rui_ro=cap, k_vol=kv, k_dd=kd, k_dm=kdm,
        rho_hieu_dung=rho_eff, che_do_cang_thang=canh_bao_dong_luong,
        so_vi_the=int(so_vi_the), muc_bien_dong=muc_bien_dong,
        sut_giam=float(dd), stop_sigma=float(stop_sigma),
        stop_gia=gia * (1 - stop_sigma * sig),
        stop_pip=stop_sigma * sig * gia / pip_size(pair),
        p_cham_stop=p_stop, khoang=khoang, n_mau_che_do=int(n_mau_che_do),
        bang_tam_han=bang_tam_han, luat_ky_hieu=luat_ky_hieu,
        canh_bao_dong_luong=canh_bao_dong_luong,
        khuyen_nghi_giu_dong=dict(
            giu_luc_vao=giu_luc_vao, bien_gioi_sigma=bien_gioi_sigma, carry_ngay=mu,
        ),
    )


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            q = parse_qs(urlparse(self.path).query)
            pair = q.get("pair", ["EURUSD"])[0].upper()
            if pair not in PAIRS:
                raise ValueError(f"cặp không hợp lệ: {pair}")
            ngay = q.get("date", [None])[0]
            if not ngay:
                raise ValueError("thiếu tham số date (YYYY-MM-DD)")
            so_vi_the = int(float(q.get("so_vi_the", ["1"])[0]))
            dd = float(q.get("dd", ["0"])[0])
            stop_sigma = float(q.get("stop_sigma", ["2.0"])[0])
            von = float(q.get("von", ["10000"])[0])
            muc_list = tuple(sorted(float(x) for x in
                              q.get("muc", ["0.80,0.95"])[0].split(",") if x))

            out = tinh_phieu(pair, ngay, so_vi_the, dd, stop_sigma, von, muc_list)
            body = json.dumps(out, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:  # noqa: BLE001 — tra loi JSON co loi ro rang cho frontend
            body = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
