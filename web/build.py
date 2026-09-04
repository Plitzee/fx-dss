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
        # Bản tĩnh (Artifact) KHÔNG gọi mạng được, nên phải nội tuyến một khối
        # gộp. Cắt còn N_TINH nến để trang không quá nặng — bản trực tiếp và
        # bản Vercel vẫn có lịch sử đầy đủ.
        import json
        d = os.path.join(HERE, "data")
        if not os.path.exists(os.path.join(d, "meta.json")):
            print("  web/data/ chưa có — chạy `python jobs/cap_nhat.py` trước")
        else:
            N_TINH = 1500
            M = json.load(io.open(os.path.join(d, "meta.json"), encoding="utf-8"))
            goi = {"meta": {k: M[k] for k in ("cap", "valid_tu", "test_tu",
                                              "canh_bao", "cap_nhat_luc",
                                              "mo_hinh", "rui_ro") if k in M},
                   "cap": {}, "su_kien": M.get("su_kien", []),
                   "hieu_chuan": M.get("hieu_chuan", {}),
                   "so_dubao": M.get("so_dubao", {})}
            goi["meta"]["tu"] = "—"
            for c in M["cap"]:
                f = os.path.join(d, c + ".json")
                if not os.path.exists(f):
                    continue
                x = json.load(io.open(f, encoding="utf-8"))
                k = slice(max(0, len(x["ngay"]) - N_TINH), None)
                y = {q: (x[q][k] if isinstance(x.get(q), list) else x.get(q))
                     for q in ("ngay", "o", "h", "l", "c", "sig_pip", "che_do",
                               "nguon", "rv_uoc", "pip", "nen12")}
                y["tam"] = {h: {**t, "p": t["p"][k], "b_pip": t["b_pip"][k],
                                "sig_pip": t["sig_pip"][k]}
                            for h, t in x["tam"].items()}
                y["ind"] = None if not x.get("ind") else {
                    **x["ind"],
                    "duong": {a_: b_[k] for a_, b_ in x["ind"]["duong"].items()}}
                cp = M.get("chi_phi_gio", {}).get(c)
                y["chi_phi_gio"] = {"med": cp["med"], "p95": cp["p95"]} if cp else None
                goi["cap"][c] = y
            p, n = dung("ui.html", json.dumps(goi, ensure_ascii=False,
                                              separators=(",", ":")), "null")
            print(f"  {os.path.basename(p):16s} {n/1048576:8.2f} MB   bản TĨNH ({N_TINH} nến/cặp)")


if __name__ == "__main__":
    main()
