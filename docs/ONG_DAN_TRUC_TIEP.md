# Ống dẫn trực tiếp — dữ liệu tới hôm nay, API, hai bản giao diện

*04/09/2026. Tái lập: `python jobs/cap_nhat.py`.*

Trước bước này hệ thống dừng ở 2025-12-31 và giao diện là ảnh chụp tĩnh. Nay có
ống dẫn chạy được: tải → tính → phục vụ, mất **8 giây** cho cả 6 cặp.

---

## 1. Nguồn dữ liệu — bốn thứ đã thử, một thứ dùng được

`docs/DONGBO_SANXUAT.md` ghi chỗ tắc nghẽn: HistData phát hành **theo năm** nên
không cập nhật hằng ngày được, còn bộ tải Dukascopy `.bi5` thì chưa ai viết. Đã
đo từ máy này:

| Nguồn | Kết quả | Dùng được? |
|---|---|---|
| Dukascopy `datafeed` | **timeout 15–21 giây**, mọi lần thử | không |
| Stooq | trả về trang kiểm tra bot (proof-of-work) | không — không vượt cơ chế chống bot |
| FRED | được, chính thức, ổn định | **đối chứng** — trễ ~1 tuần, giá trưa NY, không OHLC |
| Yahoo chart | được, có OHLC + thanh 5 phút | **nguồn chính** |

**Yahoo là endpoint không chính thức, không có cam kết dịch vụ.** Đủ cho luận
văn và bản trình diễn; hệ thống chạy thật nên mua nhà cung cấp có hợp đồng.

### 1.1 Một lỗi đã mắc và đã sửa — đừng lặp lại

Bản đầu lấy thẳng `interval=1d` của Yahoo. Sai:

- **37,1% số ngày có `close == open`**, và `high < open` — thanh nến **không hợp lệ**
- lệch trung bình **16,6 pip** so HistData

Cách đúng là làm **đúng như repo vẫn làm**: lấy thanh **giờ** rồi tự gộp lên
ngày (M1→H1→D1 trong `collect/prep_fx.py`; ở đây là H1→D1). Sau khi đổi:

| | nến ngày của Yahoo | tự gộp từ thanh giờ |
|---|---|---|
| thanh không hợp lệ | 37,1% | **0** |
| lệch trung vị vs HistData | — | **0,20–3,40 pip** tuỳ cặp |
| tương quan | — | 0,99987–0,99999 |

### 1.2 Mối nối hai nhà cung cấp — con số phải công bố

`docs/KHOA_SO.md` từng **từ chối** vá dữ liệu bằng nguồn thứ hai, vì "vá sẽ cấy
một mối nối giữa hai nhà cung cấp vào giữa chuỗi". Ở đây buộc phải nối, nên phải
**đo và công bố**:

| cặp | lệch trung vị (pip) | p95 | tương quan |
|---|---|---|---|
| USDCHF | **0,20** | 5,37 | 0,9999862 |
| USDJPY | 0,40 | 8,77 | 0,9999558 |
| USDCAD | 0,60 | 5,00 | 0,9999644 |
| GBPUSD | 1,06 | 8,74 | 0,9999652 |
| AUDUSD | 2,58 | 7,29 | 0,9998686 |
| EURUSD | **3,40** | 8,23 | 0,9999855 |

Đối chiếu: kiểm tra chéo tick-vs-tick cũ của repo cho **0,350 pip**. Số ở đây
lớn hơn vì Yahoo là **báo giá chỉ dẫn**, không phải dữ liệu tick.
**Chuỗi trước 2026-01-01 là HistData; từ đó là Yahoo.** Mọi kết luận bắc qua mốc
đó phải tính đến độ lệch này.

---

## 2. σ̂ chạy bằng đúng mô hình sản xuất, không phải proxy

`volfc2.du_bao_san_xuat` đòi `Date, open, high, low, close, rv5, rq5, bpv5, rsp,
rsn, n5`. Tất cả tính được từ lợi suất log 5 phút:

```
rv5  = Σ r²              rq5  = n/3 · Σ r⁴            bpv5 = π/2 · Σ|rᵢ||rᵢ₋₁|
rsp  = Σ r²·1{r>0}       rsn  = Σ r²·1{r<0}           n5   = số lợi suất
```

Thanh 5 phút của Yahoo phủ **60 ngày** — mà độ trễ dài nhất của HAR là **22
phiên** — nên **dự báo cho hôm nay dùng rv5 thật hoàn toàn**, không cần ước
lượng thay thế. Đoạn 2026-01 → trước cửa sổ 60 ngày chỉ để **vẽ biểu đồ**, được
ước từ thanh giờ và đánh dấu `rv_uoc=1`.

**Một lỗi đã bắt trong lúc làm.** Ban đầu co giãn `rsp`/`rsn` bằng hai hệ số
riêng, làm vỡ đẳng thức `rsp + rsn = rv5` khoảng **7%** ở các dòng ước — mà
`volfc2.thiet_ke` dựa vào đúng đẳng thức đó ("khử cùng một hệ số"). Sửa: lấy
`rv5` rồi **phân bổ** theo tỷ lệ dương/âm đo từ khung giờ. Sau khi sửa, sai số
đẳng thức ≤ 9,6×10⁻¹³ ở cả 6 cặp.

Nối chuỗi xong: **5.167 ngày** (4.994 HistData + 173 Yahoo), 2010-01-03 →
2026-09-03. Tính σ̂ + ba xác suất cho một cặp mất **1,2 giây**.

---

## 3. API

`api/main.py`, FastAPI. Chạy: `python -m uvicorn api.main:app --port 8899`

| Endpoint | Trả về |
|---|---|
| `GET /health` | dữ liệu đến ngày nào, bao nhiêu ngày rv5 thật, trễ mấy ngày |
| `GET /meta` | cấu hình, mối nối, **4 cảnh báo** (nguồn không chính thức, niêm phong 2026, mối nối, chưa có sổ dự báo) |
| `GET /series` | nến OHLC + cột `nguon` (histdata/yahoo) + cờ `rv_uoc` |
| `GET /forecast` | ba xác suất cho một ngày, kèm dải, σ̂, chế độ |
| `GET /forecast_series` | cả chuỗi ba xác suất cho mọi tầm hạn — để giao diện rê chuột |
| `GET /cost` | spread theo giờ UTC |
| `GET /calibration` | bảng giai đoạn 1 |
| `GET /events` | lịch ngân hàng trung ương |
| `POST /refresh` | tính lại từ dữ liệu **trên đĩa** |

`/refresh` **không** gọi ra mạng — việc tải là của `jobs/cap_nhat.py`. Chủ ý:
một request của người dùng không được kích hoạt lời gọi ra ngoài.

---

## 4. Hai bản giao diện, một mẫu

`web/build.py` dựng cả hai từ **cùng** `web/ui_template.html`:

| Bản | Dữ liệu | Dùng khi |
|---|---|---|
| `web/ui.html` | nướng sẵn (1,5 MB) | mở ở bất kỳ đâu, không cần server — bản đem đi trình bày |
| `web/ui_live.html` | gọi API cùng gốc | chạy thật, API phục vụ nó ở `/` |

`jobs/cap_nhat.py` bước 3 **chụp bản tĩnh từ chính API**, nên hai bản không thể
lệch nhau.

---

## 5. Việc định kỳ

```
python jobs/cap_nhat.py            # tải → tính lại → chụp → dựng, ~8 giây
python jobs/cap_nhat.py --khong-tai  # bỏ bước tải
```

Windows, 06:05 UTC mỗi ngày:

```
schtasks /create /tn "fx-dss cap nhat" /tr "python <ROOT>\jobs\cap_nhat.py" /sc daily /st 06:05
```

---

## 6. Cảnh báo phải đọc trước khi bật chạy định kỳ

**Toàn bộ 2026 nằm trong tập khoá sổ** (`docs/KHOA_SO.md` mục 2: "cộng thêm
2026-01 → 2026-08, cả 12 cặp"). Phục vụ dữ liệu 2026 qua API này **tiêu một phần
niêm phong đó**.

Điều này **không** tự động làm hỏng luận văn — ngược lại, nếu cấu hình đã đóng
băng trước thì dự báo tiến về phía trước trên 2026 là bằng chứng ngoài mẫu
**mạnh hơn** một tập giữ lại hồi cố. Điều kiện, cả ba phải giữ:

1. cấu hình tầng 2 đã chốt (`CAUHINH_SANXUAT` — đã chốt từ vòng 7),
2. ghi biên bản **trước** khi bật,
3. **không** quay lại sửa mô hình vì thấy 2026 xấu.

Sáu cặp chéo trong tập khoá sổ vẫn **chưa** bị chạm và nên giữ nguyên.

---

## 7. Còn thiếu

| Thiếu | Hệ quả |
|---|---|
| **Sổ dự báo** | chưa lưu dự báo hôm nay để mai chấm, nên thanh hiệu chuẩn trên giao diện vẫn là số tĩnh đo trên kiểm định |
| **Xác thực thật** | màn hình đăng nhập là minh hoạ, không có ô mật khẩu, không gửi và không lưu gì |
| **Thư viện quy luật** | giai đoạn 2 — hiện chỉ có nền "chỉ σ̂" và "σ̂ + chế độ" |
| **Nhà cung cấp có hợp đồng** | Yahoo có thể đổi hoặc chặn bất cứ lúc nào |
| **Chỉ báo ở backend** | EMA/Bollinger/RSI vẫn tính ở trình duyệt; kế hoạch là đưa về backend để chỉ báo và quy luật dùng chung một bộ mã |
