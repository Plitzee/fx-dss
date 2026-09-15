"""VIEC DINH KY — cap nhat du lieu va dung lai giao dien.

Nam buoc, chay theo thu tu:
  1. tai du lieu hien hanh   (collect/live_fx.py)
  1b. KIEM TRA DO MOI du lieu vua tai — chan lai TRUOC khi tinh lai cache
      neu du lieu tro nen cu hon/rong hon truoc, tranh tinh ca cache tren
      du lieu tai loi/rong ma khong ai biet
  2. tinh lai sigma + xac suat  (goi /refresh cua API neu no dang chay)
  3. chup mot ban tinh        (web/ui_data.json) tu chinh API — de ban tinh va
                               ban truc tiep KHONG BAO GIO lech nhau. Ghi ra
                               THU MUC TAM roi doi ten NGUYEN KHOI (atomic) —
                               tranh web/data/ o trang thai nua-cu-nua-moi
                               neu tien trinh bi ngat giua chung khi dang chup
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
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
LIVE = os.path.join(ROOT, "data", "live")
API = os.environ.get("FXDSS_API", "http://127.0.0.1:8899")
PY = sys.executable
HS = ("1", "5", "20")


class LoiBuoc(SystemExit):
    """Loi co CHU DICH o mot buoc cu the — de thong bao luon ro buoc nao
    hong va trang thai du lieu bi bo lai la gi, thay vi mot traceback tho."""


def buoc(n, ten):
    print(f"\n[{n}] {ten}")
    print("-" * 72)


def chay(*a):
    r = subprocess.run([PY, *a], cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        raise LoiBuoc(f"thất bại: {' '.join(a)} (mã thoát {r.returncode})")


def kiem_tra_du_lieu_moi(ngay_truoc):
    """Sau khi tai (buoc 1), xac nhan data/live/*.csv MOI HON hoac BANG ngay
    da co truoc do — chan lai buoc 2 neu du lieu vua tai bi rong/loi/cu hon,
    thay vi am tham tinh lai cache tren du lieu hong. KHONG doi hanh vi neu
    moi thu binh thuong; chi la MOT lop chan an toan."""
    f_hien_tai = {}
    for f in glob.glob(os.path.join(LIVE, "*_d1.csv")):
        pair = os.path.basename(f).replace("_d1.csv", "")
        try:
            d = pd.read_csv(f, usecols=["Date"], parse_dates=["Date"])
            f_hien_tai[pair] = (d.Date.max(), len(d))
        except Exception as e:
            raise LoiBuoc(f"  data/live/{pair}_d1.csv đọc lỗi ngay sau khi tải: {e}\n"
                          f"  DỪNG trước bước 2 — không tính lại cache trên dữ liệu hỏng.")

    if not f_hien_tai:
        raise LoiBuoc("  data/live/ rỗng sau bước tải — collect/live_fx.py có vẻ đã "
                      "không ghi được gì. DỪNG trước bước 2.")

    canh_bao = []
    for pair, (ngay_moi, n) in f_hien_tai.items():
        if n < 5:
            canh_bao.append(f"{pair}: chỉ {n} dòng — nghi ngờ dữ liệu rỗng/lỗi")
        cu = ngay_truoc.get(pair)
        if cu is not None and ngay_moi < cu:
            canh_bao.append(f"{pair}: ngày mới nhất LÙI từ {cu.date()} về {ngay_moi.date()}")

    if canh_bao:
        raise LoiBuoc("  Dữ liệu vừa tải trông bất thường, DỪNG trước bước 2:\n    "
                      + "\n    ".join(canh_bao))
    print(f"  ĐẠT — {len(f_hien_tai)} cặp, dữ liệu mới hơn hoặc bằng lần chạy trước")


def _ngay_moi_nhat_hien_co():
    """Chup nhanh ngay moi nhat cua data/live/ TRUOC khi tai — de doi chieu
    sau buoc 1. Tra ve {} neu chua co gi (lan chay dau tien)."""
    ra = {}
    for f in glob.glob(os.path.join(LIVE, "*_d1.csv")):
        pair = os.path.basename(f).replace("_d1.csv", "")
        try:
            d = pd.read_csv(f, usecols=["Date"], parse_dates=["Date"])
            ra[pair] = d.Date.max()
        except Exception:
            pass
    return ra


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

    GHI NGUYEN KHOI (atomic): moi file ghi vao `data_new/` truoc; chi khi
    TOAN BO cap + meta.json thanh cong moi doi ten `data/` hien co thanh
    `data_prev/` (du phong 1 buoc) roi doi `data_new/` thanh `data/`. Neu
    tien trinh bi ngat giua chung (mat mang, crash), `web/data/` cu VAN CON
    NGUYEN — nguoi dung khong bao gio thay trang o trang thai nua-cu-nua-moi.
    """
    g = lambda p: requests.get(API + p, timeout=600).json()
    N = 6000 if day_du else 1500
    meta = g("/meta")
    thu_muc_cu = os.path.join(WEB, "data")
    thu_muc = os.path.join(WEB, "data_new")
    if os.path.isdir(thu_muc):
        shutil.rmtree(thu_muc)          # don ban _new dang do tu lan chay hong truoc
    os.makedirs(thu_muc, exist_ok=True)

    M = {"cap": meta["cap"], "valid_tu": meta["valid_tu"], "test_tu": meta["test_tu"],
         "moc_noi": meta["moc_noi_nguon"], "canh_bao": meta["canh_bao"],
         # rao chan RV5 that — ban tinh phai mang theo, neu khong banner canh bao
         # se khong bao gio hien tren Vercel (xem web/ui_template.html::paintRvBanner)
         "rv_rao_chan": meta.get("rv_rao_chan"),
         "cap_nhat_luc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
         "su_kien": g("/events?tu=2018-01-01").get("su_kien", []),
         "hieu_chuan": g("/calibration").get("bang", {}),
         "so_dubao": {h: g(f"/journal?h={h}") for h in HS},
         "mo_hinh": g("/models"),
         "ky_nang_theo_h": meta.get("ky_nang_theo_h", {}),
         "rui_ro": {},
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
            M["rui_ro"][p] = g(f"/risk?pair={p}&so_vi_the=6")
        except Exception:
            M["rui_ro"][p] = None
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
                      "kP": t["kP"], "c_h": t["c_h"], "nen": t["nen"],
                      "tap": (sel(t["tap"]) if t.get("tap") else None)}
        D_ = {"ngay": s_["ngay"], "o": s_["o"], "h": s_["h"], "l": s_["l"],
              "c": s_["c"], "pip": s_["pip"], "nguon": s_.get("nguon"),
              "rv_uoc": s_.get("rv_uoc"), "n5": s_.get("n5"),
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

    # ── doi ten NGUYEN KHOI: data/ -> data_prev/ (du phong), data_new/ -> data/ ──
    thu_muc_prev = os.path.join(WEB, "data_prev")
    if os.path.isdir(thu_muc_cu):
        if os.path.isdir(thu_muc_prev):
            shutil.rmtree(thu_muc_prev)
        os.rename(thu_muc_cu, thu_muc_prev)
    os.rename(thu_muc, thu_muc_cu)
    print(f"  đã đổi tên nguyên khối data_new/ -> data/ (bản trước đó giữ ở data_prev/)")


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
        ngay_truoc = _ngay_moi_nhat_hien_co()
        buoc(1, "Tải dữ liệu hiện hành (Yahoo 1h→D1 + 5m→rv5)")
        chay("collect/live_fx.py")
        buoc("1b", "Kiểm tra độ mới dữ liệu vừa tải")
        kiem_tra_du_lieu_moi(ngay_truoc)
    else:
        buoc(1, "Tải dữ liệu — BỎ QUA (--khong-tai)")

    buoc(2, "Tính lại σ̂ và ba xác suất")
    if api_song():
        r = requests.post(f"{API}/refresh", timeout=600).json()
        print(f"  đã tính lại: {', '.join(r['da_tinh_lai'])}")
    else:
        raise LoiBuoc(f"  API không chạy ở {API}.\n"
                      f"  Khởi động: python -m uvicorn api.main:app --port 8899")

    buoc(3, "Ghi sổ dự báo cho phiên kế tiếp, rồi chấm những phiên đã đủ kết cục")
    so_du_bao()

    buoc(4, "Chụp bản tĩnh từ chính API (ghi nguyên khối — xem chup_ban_tinh)")
    chup_ban_tinh()

    buoc(5, "Dựng lại hai trang")
    chay("web/build.py")

    print(f"\nXONG trong {time.time()-t0:.0f}s")
    print("  bản trực tiếp: http://127.0.0.1:8899/")
    print("  bản tĩnh     : web/ui.html")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    try:
        main()
    except LoiBuoc as e:
        print(f"\n{'!'*72}\nDỪNG GIỮA CHỪNG — {e}\n"
              f"Dữ liệu/trang từ lần chạy TRƯỚC vẫn còn nguyên (bước 4 ghi nguyên "
              f"khối), chỉ có thể thiếu bản cập nhật của lần chạy này.\n{'!'*72}")
        raise
