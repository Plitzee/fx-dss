# Data pipeline — `collect/`

11 script trong thư mục này **không cùng vai trò** — trước giờ không có tài
liệu nào tách chúng ra, khiến người đọc mới (kể cả hội đồng) không biết cái
nào chạy hằng ngày, cái nào chỉ chạy một lần. Bảng dưới đây tách rõ.

## 1. Bootstrap lịch sử — chạy MỘT LẦN khi dựng máy mới, không nằm trong CI

| script | việc gì | đầu ra |
|---|---|---|
| `histdata_dl.py` | tải nến M1 gốc từ HistData.com, 1 request/năm | `data/prices_raw/` |
| `histdata_dl2.py` | đợt tải bổ sung 6 cặp chéo (2010–2025) + 2026 YTD cho đủ 12 cặp | như trên |
| `prep_fx.py` | xử lý dữ liệu thô từ HistData/Forex Tester/Forexite thành định dạng chuẩn | `data/prices/{PAIR}_d1.csv`, `_h1.csv` |
| `finish_dataset.py` | vá 3 mảnh dữ liệu còn thiếu sau đợt tải chính | như trên |
| `rv5.py` | tính realized variance ở nhiều tần suất lấy mẫu từ nến M1 gốc | `data/rv_multi.csv` |
| `rv_advanced.py` | đo lường nội ngày nâng cao (RQ, BPV, semivariance) cho HAR hiện đại | `data/rv_adv.csv` |
| `ngoai_sinh.py` | tải 9 chuỗi thị trường ngoại sinh (Pha 3B) kèm ngữ nghĩa dấu thời gian | `data/ngoai_sinh/` |
| `tin_tuc_nhtw.py` | thu thập văn bản thông cáo FOMC (Pha 2) | `data/tin_tuc/` |
| `slippage.py` | đo trượt giá qua mức dừng lỗ trên toàn bộ lịch sử tick | `data/slippage.csv` |
| `tick_spread.py` | đo spread thật từ tick HistData, thay hằng số giả định | `data/spread_hourly_all.csv` |

## 2. Chẩn đoán / thăm dò — không phải bước pipeline, chỉ chạy tay khi cần

`tick_probe.py`, `probe_aggregates.py` — kiểm tra HistData có phát hành gì,
đo mức độ gộp của Dukascopy. Không sinh ra dữ liệu sản xuất.

## 3. Vận hành hằng ngày — nằm trong CI (`.github/workflows/capnhat.yml`)

| script | tần suất | vai trò |
|---|---|---|
| `live_fx.py` | 4 lần/ngày | lấp đầy dữ liệu từ 2026-01-01 đến hôm nay (nguồn Yahoo, thay HistData đã dừng cập nhật) → `data/live/` |
| `lich_su_kien.py` | 4 lần/ngày (nếu có `FRED_TOKEN`) | cập nhật lịch công bố vĩ mô/NHTW → `data/su_kien.csv` |

Hai script này là ĐẦU VÀO DUY NHẤT của tầng tính toán sản xuất
(`api/cache.py:noi_chuoi`) — mọi script bootstrap ở mục 1 chỉ chạy một lần
để dựng nền, không bao giờ được CI gọi lại.

**Cổng kiểm tra độ mới (14/09/2026):** [jobs/cap_nhat.py](../jobs/cap_nhat.py)
kiểm tra `data/live/*.csv` ngay sau khi tải (bước 1b) — nếu rỗng, lỗi đọc,
hoặc ngày mới nhất LÙI so với lần chạy trước, dừng lại TRƯỚC khi tính lại
cache (bước 2), thay vì âm thầm tính trên dữ liệu tải hỏng. Bước chụp bản
tĩnh (bước 4) ghi nguyên khối (`web/data_new/` → đổi tên `web/data/`) nên
một lần chạy bị ngắt giữa chừng không để lại trang ở trạng thái nửa cũ nửa
mới.

## 4. Sơ đồ luồng đầy đủ

Xem [`docs/KIEN_TRUC_HE_THONG.md`](../docs/KIEN_TRUC_HE_THONG.md) cho sơ đồ
nối toàn bộ pipeline này với tầng dự báo, tầng rủi ro, API và giao diện.
