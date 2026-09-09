"""PHA 2 — THU THAP VAN BAN THONG CAO FOMC.

Day la buoc dau tien cua Pha 2 trong roadmap cua HuyH (Historical + News).
`docs/00_MASTER_ROADMAP.md` doi: "Do not start news integration until Phase 1
has a stable baseline" — Pha 1 da dat cong do (walk-forward, hieu chuan, hop
dong du bao on dinh), xem docs/KHOA_SO.md muc 4.

VI SAO CHON THONG CAO FOMC, KHONG PHAI TIN TUC CHUNG

Cong Pha 2 cua chinh roadmap doi "timestamp alignment between FX and news is
verified; no major leakage is detected". Do la rang buoc quyet dinh cach chon
nguon, va repo nay da dinh RO RI ba lan chi trong mot ngay (CHISO_DANHGIA.md
muc 13, va hai lan o H2/H6) — nen nguon nao co dau thoi gian mo ho thi khong
dung duoc.

Xep theo rui ro ro ri, tu thap len cao:
  1. thong cao NHTW    lich cong bo dinh TRUOC hang nam, gio cong bo co dinh
  2. so lieu vi mo     ngay cong bo biet truoc, nhung ban so lieu bi sua lai
  3. tin tuc tong hop  dau thoi gian la luc BAI DUOC INDEX, khong phai luc
                       su kien xay ra — GDELT chinh la dang nay

Da thu GDELT DOC 2.0 API truoc: no chan IP nay bat ke nhip gui (gioi han 1
request/5s nhung van tra ve thong bao gioi han sau 20s cho). Khong du tin cay
de dung lam du lieu luan van, va dau thoi gian cua no cung thuoc loai 3.

FOMC duoc chon trong so cac NHTW vi CA SAU CAP deu co USD mot ve — mot nguon
duy nhat lien quan den toan bo bang, thay vi phai ghep sau nguon roi giai
thich vi sao cap nay co cap kia khong.

CAN LE THOI GIAN. Thong cao FOMC ra luc 14:00 gio New York ngay hop cuoi.
Nen dac trung tu no chi duoc ap cho phien KE TIEP — `src/tin_tuc.py` lo viec
dich, file nay chi luu van ban kem ngay cong bo.

Chay:  python collect/tin_tuc_nhtw.py
Ghi:   data/tin_tuc/fomc/{YYYY-MM-DD}.txt  +  data/tin_tuc/fomc_index.csv
"""
import os
import re
import sys
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data")
OUT = os.path.join(D, "tin_tuc", "fomc")
UA = "Mozilla/5.0 (compatible; fx-dss-research/1.0)"
NGHI = 1.2                      # giay giua hai lan goi — lich su voi may chu
MAU = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{d}{s}.htm"


def lay(url, thu=3):
    for i in range(thu):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(r, timeout=45) as f:
                return f.read().decode("utf-8", errors="replace")
        except Exception:
            if i == thu - 1:
                return None
            time.sleep(2 * (i + 1))
    return None


DAU = ("Share",)                      # het phan tieu de, bat dau than thong cao
CUOI = ("Voting for", "Voting against", "Implementation Note",
        "For media inquiries", "Last Update",
        "Board of Governors of the Federal Reserve System 20th")


def loc_van_ban(html):
    """Rut THAN thong cao tu <div id="article">.

    KHONG cat theo the HTML. Trang cua Fed long nhieu <div> nen `</div>` dau
    tien roi vao ngay sau tieu de — ban dau viet the va chi lay duoc 103 ky tu
    (dung 16 tu tieu de). Nen cat theo MOC VAN BAN, chac chan hon nhieu:

      dau : sau "Share" (het khoi tieu de + nut chia se)
      cuoi: truoc "Voting for ..." — danh sach nguoi bo phieu tro di la thu
            tuc hanh chinh, khong phai noi dung chinh sach. Giu lai se lam
            nhieu phep do THAY DOI CAU CHU, vi ten thanh vien doi moi nam
            trong khi lap truong chinh sach thi khong.
    """
    if not html:
        return None
    i = html.find('id="article"')
    if i < 0:
        return None
    doan = re.sub(r"(?is)<(script|style|nav|figure)[^>]*>.*?</\1>", " ",
                  html[i:i + 40000])
    txt = re.sub(r"(?s)<[^>]+>", " ", doan)
    txt = (txt.replace("&nbsp;", " ").replace("&amp;", "&")
              .replace("&#39;", "'").replace("&quot;", '"')
              .replace("&#160;", " "))
    txt = re.sub(r"\s+", " ", txt).strip()

    for m in DAU:
        k = txt.find(m)
        if 0 <= k < 400:
            txt = txt[k + len(m):].strip()
            break
    cat = [txt.find(m) for m in CUOI]
    cat = [c for c in cat if c > 300]
    if cat:
        txt = txt[:min(cat)].strip()
    return txt if len(txt) > 500 else None


def main():
    os.makedirs(OUT, exist_ok=True)
    cb = pd.read_csv(os.path.join(D, "cb_dates.csv"), parse_dates=["date"])
    ngay = sorted(cb[cb.bank == "FOMC"].date.dt.date.unique())
    # 2026 nam trong TAP KHOA SO (docs/KHOA_SO.md muc 2) — khong tai ve
    ngay = [d for d in ngay if d.year <= 2025]
    print(f"{len(ngay)} ngày họp FOMC (2010–2025, đã cắt 2026 vì nằm trong tập khoá sổ)")

    hang, moi, sanco, hong = [], 0, 0, []
    for k, d in enumerate(ngay, 1):
        ds = d.strftime("%Y%m%d")
        fp = os.path.join(OUT, f"{d}.txt")
        if os.path.exists(fp) and os.path.getsize(fp) > 500:
            txt = open(fp, encoding="utf-8").read()
            sanco += 1
        else:
            txt = None
            for s in ("a", "b", "c"):
                txt = loc_van_ban(lay(MAU.format(d=ds, s=s)))
                time.sleep(NGHI)
                if txt and len(txt) > 500:
                    break
                txt = None
            if txt:
                open(fp, "w", encoding="utf-8").write(txt)
                moi += 1
            else:
                hong.append(str(d))
        if txt:
            hang.append(dict(ngay=str(d), so_tu=len(txt.split()), so_ky_tu=len(txt)))
        if k % 20 == 0:
            print(f"  {k}/{len(ngay)}  (mới {moi} · sẵn có {sanco} · hỏng {len(hong)})",
                  flush=True)

    df = pd.DataFrame(hang).sort_values("ngay")
    df.to_csv(os.path.join(D, "tin_tuc", "fomc_index.csv"), index=False,
              encoding="utf-8", lineterminator="\n")
    print(f"\nlấy được {len(df)}/{len(ngay)} thông cáo · mới {moi} · sẵn có {sanco}")
    if hong:
        print(f"KHÔNG lấy được {len(hong)}: {', '.join(hong[:8])}"
              + (" …" if len(hong) > 8 else ""))
    print(f"độ dài (số từ): trung vị {df.so_tu.median():.0f} · "
          f"khoảng [{df.so_tu.min()}; {df.so_tu.max()}]")
    print(f"→ data/tin_tuc/fomc/ · data/tin_tuc/fomc_index.csv")


if __name__ == "__main__":
    main()
