"""Server LOCAL phục vụ /api/decision và /api/meta khi chạy `next dev`
KHÔNG qua `vercel dev` (vercel dev cần đăng nhập tài khoản Vercel — server
này cho phép test giao diện mà không cần điều đó).

Dùng LẠI đúng logic tính toán trong api/decision.py và api/meta.py (import
làm module, gọi hàm — không viết lại công thức); chỉ phần đọc query string
là lặp lại ngắn gọn vì hai handler kia được viết cho khuôn của Vercel
(BaseHTTPRequestHandler độc lập theo path), không tiện gọi chéo trực tiếp.

Vercel thật sự KHÔNG dùng file này — nó dùng thẳng api/*.py qua Python
runtime riêng của nó. File này chỉ có tác dụng khi phát triển local.

Chạy:
    python web/dev_server.py          (mặc định cổng 8787)
Rồi ở một cửa sổ dòng lệnh khác:
    cd web && npm run dev
next.config.js tự động proxy /api/* sang server này khi KHÔNG chạy trên Vercel.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "api"), os.path.join(HERE, "api", "_lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

import decision as D  # noqa: E402
import meta as M  # noqa: E402

PORT = int(os.environ.get("PORT", "8787"))


class Dispatch(BaseHTTPRequestHandler):
    def _json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path.startswith("/api/meta"):
                with open(os.path.join(M._DATA, "meta.json"), encoding="utf-8") as f:
                    meta = json.load(f)
                pairs = list(meta["pairs"].keys())
                ranges = {}
                for p in pairs:
                    with open(os.path.join(M._DATA, f"series_{p}.json"), encoding="utf-8") as f:
                        s = json.load(f)
                    ranges[p] = [s["dates"][0], s["dates"][-1]]
                self._json(200, {"pairs": pairs, "valid_tu": meta["valid_tu"], "ranges": ranges})
                return

            if path.startswith("/api/decision"):
                q = parse_qs(urlparse(self.path).query)
                pair = q.get("pair", ["EURUSD"])[0].upper()
                ngay = q.get("date", [None])[0]
                if not ngay:
                    raise ValueError("thiếu tham số date (YYYY-MM-DD)")
                so_vi_the = int(float(q.get("so_vi_the", ["1"])[0]))
                dd = float(q.get("dd", ["0"])[0])
                stop_sigma = float(q.get("stop_sigma", ["2.0"])[0])
                von = float(q.get("von", ["10000"])[0])
                muc_list = tuple(sorted(float(x) for x in
                                  q.get("muc", ["0.80,0.95"])[0].split(",") if x))
                out = D.tinh_phieu(pair, ngay, so_vi_the, dd, stop_sigma, von, muc_list)
                self._json(200, out)
                return

            self._json(404, {"error": f"không có route {path}"})
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": str(e)})

    def log_message(self, fmt, *args):
        sys.stderr.write("[dev_server] " + (fmt % args) + "\n")


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Dispatch)
    print(f"dev_server: đang chạy tại http://127.0.0.1:{PORT}  (Ctrl+C để dừng)")
    print("mở một cửa sổ dòng lệnh khác, vào thư mục web/ rồi chạy: npm run dev")
    srv.serve_forever()
