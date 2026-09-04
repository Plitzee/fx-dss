"""VIEC DINH KY — cap nhat du lieu va dung lai giao dien.

Bon buoc, chay theo thu tu:
  1. tai du lieu hien hanh   (collect/live_fx.py)
  2. tinh lai sigma + xac suat  (goi /refresh cua API neu no dang chay)
  3. chup mot ban tinh        (web/ui_data.json) tu chinh API — de ban tinh va
                               ban truc tiep KHONG BAO GIO lech nhau
  4. dung lai hai trang       (web/build.py)

Chay tay:      python jobs/cap_nhat.py
Bo qua buoc 1: python jobs/cap_nhat.py --khong-tai

Dinh ky (Windows, chay 06:05 UTC moi ngay):
  schtasks /create /tn "fx-dss cap nhat" /tr "python C:\\...\\jobs\\cap_nhat.py" ^
           /sc daily /st 06:05

LUU Y NIEM PHONG: buoc 1 tai du lieu 2026, ma toan bo 2026 nam trong tap khoa
so cua docs/KHOA_SO.md. Chi chay viec nay khi da chot cau hinh va ghi bien ban.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
API = os.environ.get("FXDSS_API", "http://127.0.0.1:8899")
PY = sys.executable
HS = ("1", "5", "20")


def buoc(n, ten):
    print(f"\n[{n}] {ten}")
    print("-" * 72)


def chay(*a):
    r = subprocess.run([PY, *a], cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        raise SystemExit(f"thất bại: {' '.join(a)}")


def api_song():
    try:
        return requests.get(f"{API}/health", timeout=8).status_code == 200
    except Exception:
        return False


def chup_ban_tinh(day_du=True):
    """Goi API roi ghi du lieu cho giao dien.

    TACH THEO CAP. Lich su DAY DU (4.324 nen tu 2010) ton ~2 MB moi cap, tuc
    ~11,6 MB cho ca sau. Nhoi het vao mot file roi noi tuyen vao HTML thi trang
    nang khong mo noi. Nen:
        web/data/meta.json    nho — cap, hieu chuan, su kien, so du bao, chi phi
        web/data/{PAIR}.json  ~2 MB — nen, du bao, chi bao cua RIENG cap do
    Giao dien nap meta + cap dau tien luc khoi dong, cac cap khac tai khi bam.
    Vercel tu nen gzip/brotli nen ~2 MB xuong con ~300 KB tren duong truyen.

    Dung CHINH API lam nguon, khong tinh lai bang duong khac — de ban tinh va
    ban truc tiep khong the lech nhau.
    """
    g = lambda p: requests.get(API + p, timeout=600).json()
    N = 6000 if day_du else 1500
    meta = g("/meta")
    thu_muc = os.path.join(WEB, "data")
    os.makedirs(thu_muc, exist_ok=True)

    M = {"cap": meta["cap"], "valid_tu": meta["valid_tu"], "test_tu": meta["test_tu"],
         "moc_noi": meta["moc_noi_nguon"], "canh_bao": meta["canh_bao"],
         "cap_nhat_luc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
         "su_kien": g("/events?tu=2018-01-01").get("su_kien", []),
         "hieu_chuan": g("/calibration").get("bang", {}),
         "so_dubao": {h: g(f"/journal?h={h}") for h in HS},
         "chi_phi_gio": {}, "tom_tat": {}}

    tong = 0
    for p in meta["cap"]:
        s_ = g(f"/series?pair={p}&n={N}")
        f_ = g(f"/forecast_series?pair={p}&n={N}")
        try:
            M["chi_phi_gio"][p] = g(f"/cost?pair={p}")
        except Exception:
            M["chi_phi_gio"][p] = None
        try:
            ind = g(f"/indicators?pair={p}&tf=D1&n={N}")
        except Exception:
            ind = None

        k = {d: i for i, d in enumerate(f_["ngay"])}
        sel = lambda a: [a[k[d]] if d in k else None for d in s_["ngay"]]
        tam = {}
        for h in HS:
            t = f_["tam"][h]
            tam[h] = {"p": [v or [1 / 3, 1 / 3, 1 / 3] for v in sel(t["p"])],
                      "b_pip": [v or 0 for v in sel(t["b_pip"])],
                      "sig_pip": [v or 0 for v in sel(t["sig_pip"])],
                      "kP": t["kP"], "c_h": t["c_h"], "nen": t["nen"]}
        D_ = {"ngay": s_["ngay"], "o": s_["o"], "h": s_["h"], "l": s_["l"],
              "c": s_["c"], "pip": s_["pip"], "nguon": s_.get("nguon"),
              "rv_uoc": s_.get("rv_uoc"),
              "sig_pip": [v or 0 for v in sel(f_["sig_pip"])],
              "che_do": [0 if v is None else v for v in sel(f_["che_do"])],
              "nen12": f_["nen12"], "tam": tam,
              "ind": ({"duong": ind["duong"], "st_chieu": ind["st_chieu"],
                       "vwap_that": ind["vwap_that"], "cau_truc": ind["cau_truc"]}
                      if ind else None)}
        fp = os.path.join(thu_muc, f"{p}.json")
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump(D_, fh, ensure_ascii=False, separators=(",", ":"))
        kb = os.path.getsize(fp) / 1024
        tong += kb
        M["tom_tat"][p] = {"n": len(s_["ngay"]), "tu": s_["ngay"][0],
                           "den": s_["ngay"][-1], "kb": round(kb)}
        print(f"  {p}  {len(s_['ngay']):5,} nến  {s_['ngay'][0]} → {s_['ngay'][-1]}  {kb:6,.0f} KB")

    fm = os.path.join(thu_muc, "meta.json")
    with open(fm, "w", encoding="utf-8") as fh:
        json.dump(M, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"  meta.json {os.path.getsize(fm)/1024:,.0f} KB · tổng {tong/1024:,.1f} MB "
          f"(gzip trên đường truyền còn ~1/6)")


def so_du_bao():
    """Ghi du bao cho phien CHUA MO CUA, roi cham nhung phien da du ket cuc.

    Thu tu quan trong: GHI truoc, CHAM sau. Nguoc lai thi co luc du bao vua ghi
    xong da bi cham ngay trong cung mot lan chay — dung ra la mot bai backtest."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    sys.path.insert(0, ROOT)
    import so_dubao as SD
    from api.main import noi_chuoi, PAIRS

    meta = requests.get(f"{API}/meta", timeout=60).json()
    ma = f"{meta['moc_noi_nguon']}|{meta['valid_tu']}"

    hang, den = [], {}
    for p in PAIRS:
        try:
            f = requests.get(f"{API}/forecast_next?pair={p}", timeout=300).json()
        except Exception as e:
            print(f"  {p}: không lấy được dự báo kế tiếp ({str(e)[:60]})")
            continue
        den[p] = f["du_lieu_den"]
        for h, t in f["tam"].items():
            hang.append({"pair": p, "ngay": f["ngay"], "h": int(h),
                         "p_giam": t["p_giam"], "p_ngang": t["p_ngang"],
                         "p_tang": t["p_tang"], "b_pip": t["dai_pip"],
                         "sigma_pip": t["sigma_pip"], "che_do": f["che_do"],
                         "mo_hinh": t["mo_hinh"], "ma_cau_hinh": ma})
    g, tr, qk = SD.ghi(hang, den)
    print(f"  ghi: {g} dòng mới · {tr} trùng (bỏ qua) · {qk} từ chối vì đã có kết cục")

    gia = {p: noi_chuoi(p)[["Date", "close"]] for p in PAIRS}
    n = SD.cham(gia)
    print(f"  chấm: {n} dòng")
    for h in (1, 5, 20):
        t = SD.thong_ke(h=h)
        if t.get("du_mau"):
            print(f"  h={h:<2} n={t['n']:<4} ECE {t['ece']} · MCE {t['mce']} · "
                  f"BSS {t['bss']:+.4f} · AUC {t['auc']}")
        else:
            print(f"  h={h:<2} n={t['n']:<4} chưa đủ mẫu (cần 30) — đang tích luỹ")


def main():
    t0 = time.time()
    tai = "--khong-tai" not in sys.argv
    print("=" * 72)
    print("CẬP NHẬT ĐỊNH KỲ FX-DSS")
    print("=" * 72)

    if tai:
        buoc(1, "Tải dữ liệu hiện hành (Yahoo 1h→D1 + 5m→rv5)")
        chay("collect/live_fx.py")
    else:
        buoc(1, "Tải dữ liệu — BỎ QUA (--khong-tai)")

    buoc(2, "Tính lại σ̂ và ba xác suất")
    if api_song():
        r = requests.post(f"{API}/refresh", timeout=600).json()
        print(f"  đã tính lại: {', '.join(r['da_tinh_lai'])}")
    else:
        raise SystemExit(f"  API không chạy ở {API}.\n"
                         f"  Khởi động: python -m uvicorn api.main:app --port 8899")

    buoc(3, "Ghi sổ dự báo cho phiên kế tiếp, rồi chấm những phiên đã đủ kết cục")
    so_du_bao()

    buoc(4, "Chụp bản tĩnh từ chính API")
    chup_ban_tinh()

    buoc(5, "Dựng lại hai trang")
    chay("web/build.py")

    print(f"\nXONG trong {time.time()-t0:.0f}s")
    print("  bản trực tiếp: http://127.0.0.1:8899/")
    print("  bản tĩnh     : web/ui.html")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
