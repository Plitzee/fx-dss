"""GET /api/meta — danh sách cặp tiền + khoảng ngày dùng được cho UI."""
import json
import os
from http.server import BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(HERE, "_data")


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with open(os.path.join(_DATA, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        pairs = list(meta["pairs"].keys())
        ranges = {}
        for p in pairs:
            with open(os.path.join(_DATA, f"series_{p}.json"), encoding="utf-8") as f:
                s = json.load(f)
            ranges[p] = [s["dates"][0], s["dates"][-1]]
        out = {"pairs": pairs, "valid_tu": meta["valid_tu"], "ranges": ranges}
        body = json.dumps(out, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
