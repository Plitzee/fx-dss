# -*- coding: utf-8 -*-
"""Dung thu muc trien khai Vercel tu cac thanh phan da co.

Ket qua trong deploy/:
  public/index.html   giao dien (ban TINH + goi /api/intraday cho khung noi ngay)
  public/ui_data.json du bao da tinh san o may minh
  api/intraday.py     ham serverless: nen noi ngay + chi bao
  lib/chibao.py       ban sao src/chibao.py (Vercel chi dong goi thu muc du an)
  requirements.txt    requests, pandas, numpy  (KHONG scipy)
  vercel.json

Chay:  python deploy/build.py
Roi:   cd deploy && vercel deploy --prod
"""
import io
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "web")
PUB = os.path.join(HERE, "public")
LIB = os.path.join(HERE, "lib")

VERCEL = """{
  "version": 2,
  "functions": { "api/intraday.py": { "memory": 1024, "maxDuration": 30 } },
  "headers": [
    { "source": "/ui_data.json",
      "headers": [{ "key": "Cache-Control", "value": "public, max-age=300" }] }
  ]
}
"""
REQS = "requests>=2.31\npandas>=2.0\nnumpy>=1.26\n"


def main():
    os.makedirs(PUB, exist_ok=True)
    os.makedirs(LIB, exist_ok=True)

    # 1. chi bao — dung CHINH file o src/, khong viet lai (mot bo ma duy nhat)
    shutil.copy2(os.path.join(ROOT, "src", "chibao.py"), os.path.join(LIB, "chibao.py"))

    # 2. du lieu du bao da tinh san
    src = os.path.join(WEB, "ui_data.json")
    if not os.path.exists(src):
        raise SystemExit("thiếu web/ui_data.json — chạy `python jobs/cap_nhat.py` trước")
    shutil.copy2(src, os.path.join(PUB, "ui_data.json"))

    # 3. trang: ban TINH (du bao nuong san) nhung API tro toi /api de lay khung
    #    noi ngay. Hai nguon nay khong xung dot: du bao la dai luong NGAY, con
    #    nen noi ngay chi de ve bieu do.
    t = io.open(os.path.join(WEB, "ui_template.html"), encoding="utf-8").read()
    data = io.open(src, encoding="utf-8").read()
    html = t.replace("__DATA__", data).replace("__API__", '"VERCEL"')
    io.open(os.path.join(PUB, "index.html"), "w", encoding="utf-8", newline="\n").write(html)

    io.open(os.path.join(HERE, "vercel.json"), "w", encoding="utf-8", newline="\n").write(VERCEL)
    io.open(os.path.join(HERE, "requirements.txt"), "w", encoding="utf-8", newline="\n").write(REQS)

    mb = os.path.getsize(os.path.join(PUB, "index.html")) / 1048576
    print(f"  public/index.html    {mb:5.2f} MB")
    print(f"  public/ui_data.json  {os.path.getsize(src)/1048576:5.2f} MB")
    print(f"  api/intraday.py + lib/chibao.py + vercel.json + requirements.txt")
    print("\nTriển khai:  cd deploy && vercel deploy --prod")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
