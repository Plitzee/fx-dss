# -*- coding: utf-8 -*-
"""Dung hai ban giao dien tu CUNG mot mau `web/ui_template.html`.

  web/ui.html       ban TINH   — du lieu nuong san, mo duoc o bat ky dau,
                                 khong can server. Dung de xuat ban Artifact.
  web/ui_live.html  ban TRUC TIEP — goi FastAPI, du lieu toi hom nay.
                                 API phuc vu chinh no o duong dan "/".

Chay:  python web/build.py            (dung ca hai)
       python web/build.py live       (chi ban truc tiep)
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, "ui_template.html")


def dung(ten, data_js, api_js):
    t = io.open(TPL, encoding="utf-8").read()
    assert t.count("__DATA__") == 1 and t.count("__API__") == 1, "mau thieu cho thay the"
    out = t.replace("__DATA__", data_js).replace("__API__", api_js)
    p = os.path.join(HERE, ten)
    io.open(p, "w", encoding="utf-8", newline="\n").write(out)
    return p, len(out.encode("utf-8"))


def main():
    che_do = sys.argv[1] if len(sys.argv) > 1 else "ca_hai"

    if che_do in ("ca_hai", "live"):
        # cung goc voi API, nen duong dan tuong doi la du
        p, n = dung("ui_live.html", "null", "location.origin")
        print(f"  {os.path.basename(p):16s} {n/1024:8.0f} KB   bản TRỰC TIẾP (gọi API cùng gốc)")

    if che_do in ("ca_hai", "tinh"):
        f = os.path.join(HERE, "ui_data.json")
        if not os.path.exists(f):
            print("  ui_data.json chưa có — chạy `python src/xuat_ui.py` trước")
        else:
            p, n = dung("ui.html", io.open(f, encoding="utf-8").read(), "null")
            print(f"  {os.path.basename(p):16s} {n/1048576:8.2f} MB   bản TĨNH (dữ liệu nướng sẵn)")


if __name__ == "__main__":
    main()
