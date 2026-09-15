"""DO CHAT LUONG NGUON DU LIEU NOI NGAY — Dukascopy (tick / nen M1) va EODHD.

VI SAO CAN FILE NAY. `collect/live_fx.py` chot nguon hien hanh la Yahoo bang
mot doan do duoc ghi trong docstring cua no. Do la cach lam dung, nhung phep
do do khong tai lap duoc: no nam trong van xuoi, khong phai trong ma. File nay
dung lai phep do do thanh script chay duoc, va mo rong sang hai nguon moi.

DAI LUONG QUYET DINH la RV5 — phuong sai thuc hien 5 phut, chinh la thu
`volfc2.du_bao_san_xuat` tieu thu. Moc so sanh (su that) la RV5 dung tu TICK
Dukascopy: tick la ban ghi giao dich goc, moi thu khac deu la ban tom luoc cua
no.

DA DO 15/09/2026 (EURUSD, 2026-09-08, 84 nen M5 chung, 08-15 UTC, cung quy ve BID):

    RV5 tu tick Dukascopy   1,6651e-06   (moc)
    RV5 tu nen M1 Dukascopy 1,6651e-06   lech  0,0%
    RV5 tu EODHD            1,8374e-06   lech +10,3%

va 0/84 nen la "lang" that theo tick, trong khi EODHD bao 19,2% nen co
o=h=l=c — tuc EODHD mat chuyen dong trong nen, khong phai thi truong dung yen.

CANH BAO VE CO MAU. Con so tren la MOT ngay, MOT cap, gio London/NY. Du de
canh bao, CHUA du de ket luan dong theo chuan cua chinh du an nay. Dung
`--ngay` va `--cap` de mo rong truoc khi trich dan.

HAN MUC DUKASCOPY. Dukascopy cho khoang 15-20 request roi siet (quan sat ghi
trong ../dukas/dukas_v3.py, xac nhan lai 15/09/2026: ban 20 lan lien tuc ->
9/20 thanh cong, tre trung vi 14,2s; co nghi 5-6s giua cac lan -> lay deu).
Nen moi vong lap o day deu co NGHI. Dung bo di de "chay cho nhanh".

Chay:  python collect/probe_nguon.py --tu-kiem
       python collect/probe_nguon.py                       (EURUSD, 1 ngay)
       python collect/probe_nguon.py --cap EURUSD,USDJPY --ngay 2026-09-08,2026-09-09
Ghi:   output/probe_nguon.json
"""
import argparse
import json
import lzma
import math
import os
import struct
import sys
import time
import datetime as dt

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")

UA = {"User-Agent": "Mozilla/5.0 (compatible; fx-dss-thesis/1.0)"}
DUKA = "https://datafeed.dukascopy.com/datafeed"
EODHD = "https://eodhd.com/api"
REC = struct.Struct(">IIIIIf")     # nen: t(giay), o, c, l, h, volume
TICK = struct.Struct(">IIIff")     # tick: ms, ask, bid, volA, volB
NGHI = 5.0                         # giay nghi giua hai request Dukascopy
LAN = 8                            # so lan thu lai moi URL
GIO_MAC_DINH = list(range(8, 16))  # 08-15 UTC: London + dau NY, thanh khoan cao


def diem(pair):
    """Gia tri mot 'point' cua Dukascopy: 1e-3 cho JPY, 1e-5 cho con lai."""
    return 1e-3 if "JPY" in pair.upper() else 1e-5


def _lay(u, lan=LAN, nghi=NGHI):
    """GET co thu lai va NGHI. Tra ve bytes, b'' neu 404, None neu that bai."""
    for _ in range(lan):
        try:
            r = requests.get(u, headers=UA, timeout=25)
            if r.status_code == 200:
                time.sleep(nghi)
                return r.content
            if r.status_code == 404:
                time.sleep(nghi)
                return b""
        except requests.RequestException:
            pass
        time.sleep(4.0)
    return None


def tick_m5(pair, ngay, gio):
    """M5 tu TICK Dukascopy. Tra ve {chi_so_m5: gia_dong_BID} va so gio hong."""
    P = diem(pair)
    ra, hong = {}, []
    for h in gio:
        u = f"{DUKA}/{pair}/{ngay.year}/{ngay.month-1:02d}/{ngay.day:02d}/{h:02d}h_ticks.bi5"
        b = _lay(u)
        if b is None:
            hong.append(h)
            continue
        if not b:
            continue
        raw = lzma.LZMADecompressor().decompress(b)
        for i in range(len(raw) // TICK.size):
            ms, ask, bid, _, _ = TICK.unpack_from(raw, i * TICK.size)
            ra.setdefault(h * 12 + ms // 300_000, []).append(bid * P)
    return {k: v[-1] for k, v in ra.items()}, hong


def nen_m1(pair, ngay):
    """Nen M1 Dukascopy — MOT request cho ca ngay. Tra ve list ban ghi."""
    u = f"{DUKA}/{pair}/{ngay.year}/{ngay.month-1:02d}/{ngay.day:02d}/BID_candles_min_1.bi5"
    b = _lay(u)
    if not b:
        return None if b is None else []
    raw = lzma.LZMADecompressor().decompress(b)
    return [REC.unpack_from(raw, i * REC.size) for i in range(len(raw) // REC.size)]


def m1_sang_m5(nen, pair, gio):
    P = diem(pair)
    ra = {}
    for t, o, c, l, h_, v in nen:
        if v <= 0:
            continue
        k = t // 300
        if k // 12 in gio:
            ra.setdefault(k, []).append(c * P)
    return {k: v[-1] for k, v in ra.items()}


def eodhd_m5(pair, ngay, gio, khoa="demo"):
    """M5 cua EODHD cho mot ngay. Tra ve ({chi_so: dong}, so_nen_phang)."""
    t0 = int(dt.datetime.combine(ngay, dt.time(0)).replace(tzinfo=dt.timezone.utc).timestamp())
    t1 = t0 + 86_400
    u = (f"{EODHD}/intraday/{pair}.FOREX?interval=5m&api_token={khoa}"
         f"&fmt=json&from={t0}&to={t1}")
    try:
        j = requests.get(u, headers=UA, timeout=60).json()
    except Exception:                                    # noqa: BLE001
        return None, 0
    if not isinstance(j, list):
        return None, 0
    ra, phang = {}, 0
    for x in j:
        if x.get("open") is None:
            continue
        d0 = dt.datetime.utcfromtimestamp(x["timestamp"])
        if d0.date() != ngay or d0.hour not in gio:
            continue
        ra[d0.hour * 12 + d0.minute // 5] = float(x["close"])
        if x["open"] == x["high"] == x["low"] == x["close"]:
            phang += 1
    return ra, phang


def rv(chuoi, khoa):
    """RV5 = tong binh phuong loi suat log giua cac nen M5 lien tiep."""
    v = [chuoi[k] for k in khoa]
    return sum(math.log(v[i] / v[i - 1]) ** 2 for i in range(1, len(v)))


def mot_ngay(pair, ngay, gio, khoa_eodhd):
    t_m5, hong = tick_m5(pair, ngay, gio)
    nen = nen_m1(pair, ngay)
    m_m5 = m1_sang_m5(nen, pair, gio) if nen else {}
    e_m5, e_phang = eodhd_m5(pair, ngay, gio, khoa_eodhd)
    e_m5 = e_m5 or {}
    chung = sorted(set(t_m5) & set(m_m5) & set(e_m5))
    if len(chung) < 20:
        return dict(pair=pair, ngay=str(ngay), loi="qua it nen chung",
                    n_chung=len(chung), gio_hong=hong)
    r_t, r_m, r_e = rv(t_m5, chung), rv(m_m5, chung), rv(e_m5, chung)
    return dict(
        pair=pair, ngay=str(ngay), n_chung=len(chung), gio_hong=hong,
        rv_tick=r_t, rv_m1=r_m, rv_eodhd=r_e,
        ty_le_m1=(r_m / r_t if r_t else None),
        ty_le_eodhd=(r_e / r_t if r_t else None),
        eodhd_nen_phang=e_phang,
        nen_m1_tong=len(nen) if nen else 0)


def tu_kiem():
    """Kiem cac phep BIEN DOI, khong cham mang."""
    print("TỰ KIỂM (không gọi mạng)")
    dat = True
    # RV cua chuoi hang so phai bang 0
    ok = abs(rv({1: 1.1, 2: 1.1, 3: 1.1}, [1, 2, 3])) < 1e-18
    dat &= ok
    print(f"  RV của chuỗi không đổi = 0        {'ĐẠT' if ok else 'HỎNG'}")
    # RV cong dodn duoc: hai buoc +1% roi -1%
    v = {1: 1.0, 2: 1.01, 3: 1.0}
    mong = math.log(1.01) ** 2 + math.log(1 / 1.01) ** 2
    ok = abs(rv(v, [1, 2, 3]) - mong) < 1e-15
    dat &= ok
    print(f"  RV khớp nghiệm giải tích          {'ĐẠT' if ok else 'HỎNG'}")
    # point cua JPY khac han
    ok = diem("USDJPY") == 1e-3 and diem("EURUSD") == 1e-5
    dat &= ok
    print(f"  point JPY 1e-3, còn lại 1e-5      {'ĐẠT' if ok else 'HỎNG'}")
    # gop M1 -> M5: 5 nen M1 trong cung khoang phai ra MOT nen M5, lay gia DONG.
    # Dung so NGUYEN point (110000 + t) de khong lan voi sai so lam tron khi
    # dung fixture — day chinh la cho ban dau viet sai va bi tu kiem bat.
    nen = [(t * 60, 0, 110_000 + t, 0, 0, 1.0) for t in range(10)]
    g = m1_sang_m5(nen, "EURUSD", [0])
    ok = (len(g) == 2
          and abs(g[0] - 110_004 * 1e-5) < 1e-12      # nen M5 dau: dong o t=240
          and abs(g[1] - 110_009 * 1e-5) < 1e-12)     # nen M5 sau: dong o t=540
    dat &= ok
    print(f"  gộp M1→M5 lấy đúng giá đóng       {'ĐẠT' if ok else 'HỎNG'}")
    # nen volume = 0 (phien dong cua) phai bi bo qua
    g2 = m1_sang_m5([(0, 0, 110_000, 0, 0, 0.0), (60, 0, 110_005, 0, 0, 2.0)],
                    "EURUSD", [0])
    ok = g2 == {0: 110_005 * 1e-5}
    dat &= ok
    print(f"  bỏ qua nến volume = 0             {'ĐẠT' if ok else 'HỎNG'}")
    print(f"→ TỰ KIỂM {'ĐẠT' if dat else 'HỎNG'}")
    return dat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default="EURUSD")
    ap.add_argument("--ngay", default="2026-09-08")
    ap.add_argument("--gio", default="8-15", help="khoảng giờ UTC, ví dụ 8-15 hoặc 0-23")
    ap.add_argument("--khoa-eodhd", default=os.environ.get("EODHD_API_KEY", "demo"))
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args()

    if a.tu_kiem:
        sys.exit(0 if tu_kiem() else 1)
    if not tu_kiem():
        sys.exit("tự kiểm HỎNG — dừng, không đo số liệu thật")

    g0, g1 = (int(x) for x in a.gio.split("-"))
    gio = list(range(g0, g1 + 1))
    caps = [x.strip().upper() for x in a.cap.split(",") if x.strip()]
    ngays = [dt.date.fromisoformat(x.strip()) for x in a.ngay.split(",") if x.strip()]

    print("\n" + "=" * 96)
    print("ĐO CHẤT LƯỢNG NGUỒN NỘI NGÀY — mốc là RV5 dựng từ TICK Dukascopy")
    print(f"giờ UTC {g0:02d}–{g1:02d} · {len(caps)} cặp × {len(ngays)} ngày "
          f"· khoá EODHD: {'(môi trường)' if a.khoa_eodhd != 'demo' else 'demo'}")
    print("Dukascopy có hạn mức — script NGHỈ giữa các request, sẽ chậm. Đừng bỏ phần nghỉ.")
    print("=" * 96)

    ket = []
    for p in caps:
        for ng in ngays:
            t0 = time.time()
            r = mot_ngay(p, ng, gio, a.khoa_eodhd)
            r["giay"] = round(time.time() - t0, 1)
            ket.append(r)
            if r.get("loi"):
                print(f"  {p} {ng}: {r['loi']} (n={r['n_chung']}, giờ hỏng {r['gio_hong']})")
                continue
            ghi_hong = f", giờ hỏng {r['gio_hong']}" if r["gio_hong"] else ""
            print(f"\n  {p} {ng} — {r['n_chung']} nến M5 chung{ghi_hong} · {r['giay']}s")
            print(f"    RV5 tick   {r['rv_tick']:.4e}   (mốc)")
            print(f"    RV5 nến M1 {r['rv_m1']:.4e}   tỷ lệ {r['ty_le_m1']:.4f}"
                  f"   lệch {abs(r['ty_le_m1']-1)*100:5.1f}%")
            print(f"    RV5 EODHD  {r['rv_eodhd']:.4e}   tỷ lệ {r['ty_le_eodhd']:.4f}"
                  f"   lệch {abs(r['ty_le_eodhd']-1)*100:5.1f}%")
            print(f"    EODHD nến o=h=l=c: {r['eodhd_nen_phang']}/{r['n_chung']}")

    tot = [r for r in ket if not r.get("loi")]
    if tot:
        lm = [abs(r["ty_le_m1"] - 1) for r in tot]
        le = [abs(r["ty_le_eodhd"] - 1) for r in tot]
        print("\n" + "=" * 96)
        print(f"GỘP {len(tot)} ô (cặp × ngày)")
        print(f"  |lệch| RV5 nến M1 Dukascopy : trung vị {100*float(np.median(lm)):5.2f}%"
              f" · tối đa {100*max(lm):5.2f}%")
        print(f"  |lệch| RV5 EODHD            : trung vị {100*float(np.median(le)):5.2f}%"
              f" · tối đa {100*max(le):5.2f}%")
        print("\n  Nhắc: mốc là tick. Nến M1 Dukascopy dựng TỪ chính tick đó nên khớp")
        print("  gần tuyệt đối là điều PHẢI xảy ra — nếu nó lệch thì lỗi ở code này,")
        print("  không phải ở nguồn. EODHD là nhà cung cấp khác nên lệch là thông tin thật.")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "probe_nguon.json"), "w", encoding="utf-8") as f:
        json.dump({"gio": [g0, g1], "ket_qua": ket}, f, ensure_ascii=False, indent=1)
    print(f"\n→ output/probe_nguon.json")


if __name__ == "__main__":
    main()
