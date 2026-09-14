"""Tang tinh toan + bo nho dem — noi chuoi lich su/hien hanh, dung san xuat
sigma + ba xac suat + tap conformal cho tung cap, va nap nen theo khung.

`_bo_nho` la cache toan cuc TRONG MOT TIEN TRINH: tinh mot lan cho moi cap,
dung lai cho moi request sau (endpoint /forecast, /series, /risk... deu goi
qua `lay()`). `_khoa` chi bao dam hai request dong thoi khong cung tinh lai
mot cap — KHONG bien no thanh cache da tien trinh hay ben ngoai tien trinh.
"""
import datetime as dt
import os
import threading

import numpy as np
import pandas as pd
from fastapi import HTTPException

from api.config import CF, CF_CUA_SO, D, LIVE, MOC_NOI, NEN_THEO_H, PAIRS, B, V2, doan, merge_thin_days

HS = (1, 5, 20)

# `_bo_nho` la cache toan cuc TRONG MOT TIEN TRINH: tinh mot lan cho moi cap,
# dung lai cho moi request sau. `_khoa` chi bao dam hai request dong thoi
# khong cung tinh lai mot cap.
_khoa = threading.Lock()
_bo_nho = {}


def _nap_lich_su(p):
    """Chuoi HistData: gia ngay + do luong noi ngay tu rv_adv.csv."""
    g = pd.read_csv(os.path.join(D, "prices", f"{p}_d1.csv"), parse_dates=["Date"])
    a = pd.read_csv(os.path.join(D, "rv_adv.csv"), parse_dates=["Date"])
    a = a[a.pair == p].drop(columns=["pair"])
    d = pd.merge(g[["Date", "open", "high", "low", "close"]], a, on="Date", how="inner")
    d["nguon"] = "histdata"
    return d


def _nap_hien_hanh(p):
    f = os.path.join(LIVE, f"{p}_d1.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, parse_dates=["Date"])
    d = d.rename(columns={"rsp5": "rsp", "rsn5": "rsn"})
    d["nguon"] = "yahoo"
    return d[["Date", "open", "high", "low", "close", "rv5", "rq5", "bpv5",
              "rsp", "rsn", "n5", "rv_uoc", "nguon"]]


def noi_chuoi(p):
    """Lich su den 2025-12-31 + hien hanh tu 2026-01-01."""
    ls = _nap_lich_su(p)
    ls = ls[ls.Date < MOC_NOI]
    hh = _nap_hien_hanh(p)
    if hh is not None:
        hh = hh[hh.Date >= MOC_NOI]
        d = pd.concat([ls, hh], ignore_index=True)
    else:
        d = ls
    if "rv_uoc" not in d:
        d["rv_uoc"] = 0
    d["rv_uoc"] = d.rv_uoc.fillna(0).astype(int)
    return d.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def tinh(p):
    """Tinh sigma + ba xac suat cho MOT cap. Ket qua duoc nho lai."""
    d = noi_chuoi(p)
    m = merge_thin_days(d)
    sig2 = V2.du_bao_san_xuat(m, p)                     # phuong sai du bao
    sig = np.sqrt(np.maximum(sig2, 0.0))

    pan = pd.DataFrame({"Date": m.Date.values, "sig": sig})
    zt = np.empty(len(m)); zt[:] = np.nan
    c = m.close.values
    zt[1:] = np.log(c[1:] / np.maximum(c[:-1], 1e-12)) / np.maximum(sig[1:], 1e-12)
    pan["zT"] = zt
    ok = np.isfinite(pan.sig.values) & (pan.sig.values > 0)
    pan = pan[ok].reset_index(drop=True)
    tr = doan(pan.Date.values) == 0

    xs = {}
    for h in HS:
        T = B.dung_muc_tieu(pan, h, tr)
        n = len(pan)
        yt = B.lop_truoc(T["yP"], h)
        ns = B.ChiSigma().khop(T["z"][tr])
        cd = B.SigmaCheDo().khop(T["z"][tr], pan.sig.values[tr])
        kh = B.KhiHauHoc().khop(T["yP"][tr])
        qt = B.QuanTinh().khop(T["yP"][tr], yt[tr])
        kw = dict(canh=T["canh_P"], sigma_h=T["sigma_h"], sig=pan.sig.values,
                  y_truoc=yt)
        # cua so mo rong cho hai nen sigma^ — khop lai moi ~21 phien
        Pns, mo_ns = B.du_bao_cuon(T, pan.sig.values,
                                   lambda z, sg: B.ChiSigma().khop(z),
                                   canh=T["canh_P"], tra_mo=True)
        Pcd, mo_cd = B.du_bao_cuon(T, pan.sig.values,
                                   lambda z, sg: B.SigmaCheDo().khop(z, sg),
                                   canh=T["canh_P"], tra_mo=True)
        for Pc, nen in ((Pns, ns), (Pcd, cd)):     # dam dau chuoi: dung ban dong bang
            thieu = ~np.isfinite(Pc[:, 0])
            if thieu.any():
                Pc[thieu] = nen.du_bao(n, **kw)[thieu]
        ns_c = B.NenCoSan("chỉ σ̂ (cuộn)", Pns, mo_ns or ns)
        cd_c = B.NenCoSan("σ̂ + chế độ (cuộn)", Pcd, mo_cd or cd)

        if NEN_THEO_H[h] == "σ̂ + chế độ (cuộn)":
            mo, P = cd_c, Pcd
        elif NEN_THEO_H[h] == "chỉ σ̂ (cuộn)":
            mo, P = ns_c, Pns
        elif NEN_THEO_H[h] == "tổ hợp trực tuyến":
            # Hoc truc tuyen: trong so cap nhat tu ket cuc DA BIET, tre dung h
            # phien. Du bao cho phien moi nhat dung trong so hoc tu toan bo qua
            # khu truoc no — dung nghia "hom qua sai thi hom nay chinh".
            mo = B.ToHopTrucTuyen([("khí hậu học", kh), ("quán tính", qt),
                                   ("chỉ σ̂", ns_c), ("σ̂ + chế độ", cd_c)], tre=h)
            P = mo.du_bao(n, y_that=T["yP"], **kw)
        else:
            mo = ns if NEN_THEO_H[h] == "chỉ σ̂" else cd
            P = mo.du_bao(n, **kw)
        xs[h] = dict(P=P, b=T["b"], sigma_h=T["sigma_h"], kP=T["kP"], c_h=T["c_h"],
                     mo=mo, trong_so=getattr(mo, "trong_so", None), yP=T["yP"])

    # ── TAP DU BAO CONFORMAL (ACI) ──────────────────────────────────────
    # Ba xac suat noi "kha nang bao nhieu"; tap conformal noi "loai tru duoc
    # gi, voi BAO DAM". Do la hai thu khac nhau, va cai thu hai la thu co the
    # HUA duoc: do phu 90% giu duoc ke ca khi thi truong doi che do.
    #
    # Vi sao ACI chu khong phai split conformal tinh: da do o
    # docs/CHISO_DANHGIA.md muc 16 — conformal tinh hong o CA HAI huong tren
    # chinh du lieu nay (LAC hut con 0,814; APS phong len 0,999), con ACI giu
    # 0,901-0,908 o moi tam han. Du lieu nay co troi phan phoi that.
    #
    # NHAN QUA: `chay_aci` phat tap cho phien t TRUOC, roi moi dung ket cuc
    # cua t de cap nhat alpha. Phien moi nhat chua co ket cuc — dung nhu khi
    # chay that.
    #
    # HIEU CHUAN RIENG TUNG CAP: do duoc +0,21 lop thong tin so voi +0,10 khi
    # gop chung (src/kiem_ngan_han.py). Ham nay von da chay rieng tung cap.
    for h in HS:
        yv = np.asarray(xs[h]["yP"], int)
        Ph = xs[h]["P"]
        hop = (yv >= 0) & np.isfinite(Ph).all(1)
        i_hc = np.flatnonzero(hop & tr)
        if len(i_hc) >= 200:
            tap, al = CF.chay_aci(Ph[i_hc][-CF_CUA_SO:], yv[i_hc][-CF_CUA_SO:],
                                  np.where(np.isfinite(Ph), Ph, 1 / 3),
                                  np.where(yv >= 0, yv, 0), CF.diem_lac)
            xs[h]["tap"], xs[h]["alpha_aci"] = tap, al
        else:
            xs[h]["tap"] = xs[h]["alpha_aci"] = None

    nguong = np.quantile(pan.sig.values[tr], [1 / 3, 2 / 3])
    return dict(m=m, pan=pan, sig=pan.sig.values, xs=xs,
                che_do=np.digitize(pan.sig.values, nguong), nguong=nguong,
                tinh_luc=dt.datetime.utcnow())


def lay(p, moi=False):
    if p not in PAIRS:
        raise HTTPException(404, f"không có cặp {p}")
    with _khoa:
        if moi or p not in _bo_nho:
            _bo_nho[p] = tinh(p)
        return _bo_nho[p]


def _idx(K, ngay):
    d = K["pan"].Date.values
    if ngay is None:
        return len(d) - 1
    t = np.datetime64(pd.Timestamp(ngay))
    i = int(np.searchsorted(d, t))
    if i >= len(d) or d[i] != t:
        i = min(max(i - 1, 0), len(d) - 1)
    return i


def gia_theo_ngay(K, ngay_arr):
    """Gia dong CUA K['m'] tuong ung tung ngay trong ngay_arr — tra theo
    NGAY, KHONG dung chung chi so voi K['pan'].

    K['m'] la gia tho DAY DU tu 2010; K['pan'] la panel nghien cuu tu
    2012-02-14 (552 dong it hon). Hai mang cung KET THUC o mot ngay nhung
    KHAC DIEM BAT DAU, nen mot chi so hop le cua pan (vi du chi so cuoi cung
    3777) tra ve mot ngay HOAN TOAN KHAC neu ap thang vao m (da do: ra ngay
    2024-07-24 thay vi 2026-09-11 — lech hon 2 nam). Ham nay thay cho kieu
    dung "K['m'].close.values[i]" voi i tinh tu _idx(K, ...)/pan — chi dung
    an toan khi lay dong CUOI CUNG (m.close.values[-1], hai mang cung ket
    thuc mot ngay nen -1 luon dung), con lay theo vi tri BAT KY thi PHAI qua
    ham nay."""
    m_ngay = K["m"].Date.values
    idx = np.searchsorted(m_ngay, np.asarray(ngay_arr, dtype="datetime64[ns]"))
    idx = np.clip(idx, 0, len(m_ngay) - 1)
    return K["m"].close.values[idx]


def nap_khung(pair, tf):
    """Nen cho MOT khung thoi gian.

      D1  lich su day (HistData 2010 -> 2025-12) + Yahoo tu 2026
      H1  lich su day (repo prices/{P}_h1.csv tu 2010) + Yahoo 730 ngay
      M15 chi Yahoo, 60 ngay      — gioi han cua nha cung cap
      M5  chi Yahoo, 60 ngay      — gioi han cua nha cung cap

    KHONG con "M1": da DO truc tiep tren feed Yahoo (12/09/2026) — nen 1 phut
    cua ho la ANH CHUP GIA (98,7% thanh co o=h=l=c), khong phai OHLC that; ve
    nen o do la ve mot day doji vo hinh. Nen 5 phut thi KHAC HAN: chi 20,8% suy
    bien (dung luc thanh khoan rat thap, gan dung), tuc ~79% la OHLC that. Nen
    khung nhanh nhat giao dien hien la M5, khong phai M1 — trung thuc voi cai
    nguon mien phi THAT SU co, thay vi ve M1 "gia".

    Do sau khac nhau la RANG BUOC CUA NGUON, khong phai lua chon thiet ke; ham
    tra ve `ghi_chu` de giao dien noi ro cho nguoi dung."""
    if tf == "D1":
        d = lay(pair)["m"][["Date", "open", "high", "low", "close", "nguon",
                            "rv_uoc", "n5"]]
        return d.rename(columns={"Date": "ts"}), "lịch sử đầy đủ từ 2010"
    if tf == "H1":
        ph = []
        f = os.path.join(D, "prices", f"{pair}_h1.csv")
        if os.path.exists(f):
            a = pd.read_csv(f, parse_dates=["Date"]).rename(columns={"Date": "ts"})
            a = a[a.ts < MOC_NOI]
            a["nguon"] = "histdata"
            ph.append(a[["ts", "open", "high", "low", "close", "nguon"]])
        g = os.path.join(LIVE, f"{pair}_H1.csv")
        if os.path.exists(g):
            b = pd.read_csv(g, parse_dates=["ts"])
            b = b[b.ts >= MOC_NOI]
            b["nguon"] = "yahoo"
            ph.append(b[["ts", "open", "high", "low", "close", "nguon"]])
        if not ph:
            raise HTTPException(503, "chưa có dữ liệu H1")
        d = pd.concat(ph, ignore_index=True).sort_values("ts")
        return d.drop_duplicates("ts").reset_index(drop=True), "lịch sử đầy đủ từ 2010"
    g = os.path.join(LIVE, f"{pair}_{tf}.csv")
    if not os.path.exists(g):
        raise HTTPException(503, f"chưa tải khung {tf} — chạy collect/live_fx.py")
    d = pd.read_csv(g, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    d["nguon"] = "yahoo"
    han = {"M15": "chỉ 60 ngày — giới hạn nhà cung cấp",
           "M5": "chỉ 60 ngày — giới hạn nhà cung cấp"}
    return d, han.get(tf, "")


def _tem(d, tf):
    """Tra ve (ngay, t).

    `ngay` luon la 'YYYY-MM-DD' — de giao dien do duoc ve du bao NGAY.
    `t` la moc cho BIEU DO: Lightweight Charts chi nhan chuoi 'YYYY-MM-DD'
    hoac SO GIAY UNIX. Chuoi kieu '2026-09-04 03:00' bi parse ra rac va bieu
    do hien trang — da gap that tren ban da trien khai."""
    ngay = [str(pd.Timestamp(x).date()) for x in d.ts.values]
    if tf == "D1":
        return ngay, ngay
    return ngay, [int(pd.Timestamp(x).timestamp()) for x in d.ts.values]


def _phien_ke_tiep(d):
    t = pd.Timestamp(d) + pd.Timedelta(days=1)
    while t.weekday() >= 5:                 # FX nghi thu 7 va Chu nhat
        t += pd.Timedelta(days=1)
    return t
