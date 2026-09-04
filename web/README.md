# FX-DSS — Tầng 7 (UI)

Giao diện web cho phiếu quyết định (tầng 6) + khuyến nghị giữ/đóng (tầng 6b).
Next.js (frontend) + một hàm Python serverless (`api/decision.py`,
`api/meta.py`) đọc dữ liệu đã tính sẵn trong `api/_data/` — không cần
pandas/CSV gốc lúc chạy, chỉ `numpy` + `scipy` (xem `requirements.txt`).

Toàn bộ tham số (`api/_data/*.json`) sinh ra bởi `src/export_ui_state.py`
ở gốc repo — chạy lại script đó (`python src/export_ui_state.py`) và
copy `web/api/_data/` vào đây mỗi khi dữ liệu/mô hình ở tầng 2-6b đổi.

## Chạy thử local (không cần tài khoản Vercel)

Cần: Node.js 18+, Python 3.9-3.12 (đã có `numpy`, `scipy`).

Mở **hai cửa sổ dòng lệnh**, cùng trong thư mục `web/`:

```bash
# cửa sổ 1 — server Python phục vụ /api/*
pip install -r requirements.txt      # nếu chưa có numpy/scipy
python dev_server.py                 # http://127.0.0.1:8787

# cửa sổ 2 — Next.js
npm install
npm run dev                          # http://localhost:3000
```

Mở `http://localhost:3000` — chọn cặp tiền + ngày (từ 2021-10-13, sau đoạn
huấn luyện chính thức), điền số vị thế/sụt giảm/stop/vốn, bấm "Tính phiếu".
Mỗi lần bấm sẽ giải lại quy hoạch động cho tầng 6b (vài giây, vì carry phụ
thuộc ngày được chọn) — không phải lỗi treo.

`next.config.js` tự nhận biết: khi chạy trên Vercel thật (biến môi trường
`VERCEL` được Vercel tự đặt) nó KHÔNG proxy — `api/*.py` được Vercel Python
runtime phục vụ trực tiếp. `dev_server.py` chỉ tồn tại cho việc phát triển
local không qua `vercel dev`.

## Triển khai lên Vercel

```bash
npm i -g vercel     # nếu chưa có
cd web
vercel              # đăng nhập + link project lần đầu
vercel --prod       # triển khai bản chính thức
```

Vercel tự nhận diện Next.js (frontend) và `api/*.py` (Python Serverless
Functions) trong cùng một project — không cần cấu hình thêm.

## Giới hạn đã biết (nói rõ, không giấu)

* **Khoảng dự báo dùng bản Conformal tĩnh (Mondrian 2 tầng), không phải ACI
  thích ứng** — dù `docs/TANG6_HIEU_CHUAN.md` ghi ACI là lựa chọn sản xuất.
  Lý do: ACI là mô hình *trực tuyến* (cập nhật dần theo từng quan sát thật),
  khó đóng băng đúng trạng thái cho một ngày lịch sử bất kỳ người dùng chọn.
  Bản tĩnh cũng đã kiểm định (độ phủ 90,3/90,8/89,4%, xem tài liệu trên) và
  là đúng bản mà tự kiểm `decision_record.py` hiện dùng — không phải một lựa
  chọn tuỳ tiện mới.
* **Không có `k_danh_mục` liên cặp thời gian thực** — số vị thế mở là do
  người dùng tự khai (như phiếu gốc), hệ thống không biết bạn có đang mở
  lệnh ở cặp khác hay không.
* **Đây là công cụ minh hoạ/học thuật**, dùng dữ liệu lịch sử đến
  2025-12-31, không có nguồn giá trực tiếp (live feed).
