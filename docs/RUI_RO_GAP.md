# HAI LOẠI RỦI RO HỆ THỐNG CHƯA ĐO — đã đo (13/09/2026)

*Tái lập: `python src/rui_ro_gap.py`. Kết quả: `output/rui_ro_gap.json`.*

`KEHOACH_2026Q4.md` mục 2.2 nêu hai loại rủi ro hệ thống mà hệ thống **không**
đo, dù dữ liệu có sẵn. Văn bản này đo chúng.

**Đây là PHÉP ĐO MÔ TẢ, không phải thí nghiệm.** Không chọn mô hình, không mở
đoạn kiểm tra để chọn gì, không có giả thuyết cần bác bỏ — nên **không đếm vào
`KHOA_SO.md` mục 5** như một cấu hình mô hình. Mục đích là **báo cáo một rủi ro
đang tồn tại mà người dùng không thấy**.

---

## 1. Gap qua đêm và cuối tuần

21.588 phiên, 6 cặp. Gap = `log(mở_t / đóng_{t−1})`, quy về đơn vị **σ̂ của
chính phiên đó** để so được giữa các cặp và các thời kỳ.

| nhóm | n | trung vị | p90 | p99 | tối đa |
|---|---|---|---|---|---|
| trong tuần (1 ngày) | 17.244 | 0,002 | 0,006 | 0,067 | 1,296 |
| **cuối tuần / lễ (≥3 ngày)** | 4.344 | **0,132** | **0,431** | **1,346** | **4,379** |

**Gap trong tuần gần như bằng 0 — và đó là đúng, không phải lỗi.** FX giao dịch
24 giờ nên ranh giới "ngày" chỉ là quy ước; giá mở của phiên sau nối liền giá
đóng phiên trước. Con số 0,002σ̂ chính là phép tự kiểm cho toàn bộ bảng này.

Rủi ro thật nằm **trọn vẹn ở cuối tuần**, khi thị trường thực sự đóng.

## 2. Stop-loss bị nhảy qua — con số đáng báo lên giao diện

Stop đặt ở `k·σ̂`. Khi gap vượt `k` thì lệnh khớp ở giá mở, **không** ở mức stop.

| stop tại | nhóm | P(nhảy qua) | trượt thêm trung bình |
|---|---|---|---|
| 1,0σ̂ | trong tuần | 0,01% | 0,249σ̂ |
| **1,0σ̂** | **cuối tuần/lễ** | **1,98%** | **0,526σ̂** |
| 1,5σ̂ | trong tuần | 0,00% | — |
| **1,5σ̂** | **cuối tuần/lễ** | **0,83%** | **0,508σ̂** |
| 2,0σ̂ | trong tuần | 0,00% | — |
| **2,0σ̂** | **cuối tuần/lễ** | **0,32%** | **0,546σ̂** |

"Trượt thêm" là phần lỗ **vượt quá** mức stop, đơn vị σ̂.

**Đọc thành câu dùng được:** với stop ở 1,5σ̂, giữ lệnh qua cuối tuần có xác
suất **0,83%** bị nhảy qua stop, và khi đã nhảy thì lỗ thêm trung bình **0,51σ̂
ngoài mức stop**. Trên 724 cuối tuần mỗi cặp (14 năm) là khoảng **6 lần mỗi
cặp**. Hiếm — nhưng **stop không chặn được**, và hiện không có chỗ nào trong hệ
thống nói điều đó.

Đáng chú ý: mức trượt thêm **không giảm** khi đặt stop xa hơn (0,53 → 0,51 →
0,55σ̂). Đặt stop rộng hơn làm **giảm tần suất** bị nhảy nhưng **không giảm độ
sâu** khi đã bị nhảy. Đó là đặc trưng của rủi ro gap và là lý do nó không hedge
được bằng cách nới stop.

### Theo từng cặp (cuối tuần)

| cặp | trung vị | p99 | P(> 1,5σ̂) |
|---|---|---|---|
| **USDJPY** | 0,135 | **1,725** | **1,38%** |
| EURUSD | 0,116 | 1,509 | 1,10% |
| AUDUSD | 0,146 | 1,421 | 0,97% |
| USDCAD | 0,124 | 1,054 | 0,69% |
| GBPUSD | 0,129 | 1,111 | 0,41% |
| USDCHF | 0,150 | 1,175 | 0,41% |

**USDJPY tệ nhất ở cả hai thước đo đuôi** (p99 = 1,725; P(>1,5σ̂) = 1,38%) — và
đây là **cặp thứ ba độc lập** cho thấy đuôi USDJPY là chỗ yếu nhất của hệ thống,
sau backtest VaR/ES (`CHISO_DANHGIA.md` mục 5b) và EVT (`DUOI_EVT.md` A4).

## 3. Thanh khoản — spread trong ngày sự kiện

**Phải phân tầng theo giờ.** Spread có chu kỳ nội ngày rất mạnh và sự kiện Mỹ
tập trung vào vài giờ nhất định; gộp cả ngày sẽ trộn hiệu ứng giờ với hiệu ứng
sự kiện. Nên so sánh trong **từng ô (cặp × giờ)** rồi gộp tỉ lệ.

*(Bản đo đầu tiên không phân tầng cho trung vị **giống hệt nhau** ở hai nhóm —
dấu hiệu phép đo sai, không phải "không có hiệu ứng". Đã sửa.)*

103.504 ô giờ; **56% số ngày có sự kiện** (lịch FRED rất dày — 4.139 sự kiện).

| | |
|---|---|
| ô (cặp × giờ) đủ mẫu | 144 |
| tỉ lệ spread ngày sự kiện / ngày thường, trung bình | **1,0213** |
| trung vị | 1,0354 |
| khoảng 90% | [0,9111 ; 1,0831] |
| t-test tỉ lệ = 1 | **t = 3,99 · p = 0,0001** |
| số ô spread rộng hơn | **118/144 (82%)** |

**Spread giãn có ý nghĩa, nhưng nhỏ: khoảng 2%.** Kết luận trung thực: hiệu ứng
tồn tại và nhất quán (82% ô cùng chiều) nhưng **về độ lớn thì không đáng kể** so
với chi phí giao dịch mà `cost_table.csv` đã tính. Nó **không** đổi được kết luận
nào về khả năng sinh lời.

## 4. Điều này đổi gì trong hệ thống

| | |
|---|---|
| Đổi mô hình? | **Không.** Đây là phép đo, không phải mô hình. |
| Đổi định cỡ vị thế? | **Không** — tầng 4/6b đang tạm dừng theo yêu cầu. |
| **Nên hiện lên giao diện?** | **Có** — hai con số: `P(nhảy qua stop cuối tuần)` và `trượt thêm trung bình`, theo cặp. Đó là rủi ro người dùng đang chịu mà không được nói. |
| Đổi kết luận nào đã có? | **Không** — nhưng nó **củng cố** chẩn đoán USDJPY bằng một thước đo độc lập thứ ba. |

## 5. Giới hạn

- Gap đo bằng `open` của panel ngày, tức **giá mở đầu tiên có dữ liệu** — nếu
  nhà môi giới mở muộn hơn thì gap thực tế có thể khác.
- Chỉ 724 cuối tuần mỗi cặp, nên p99 của gap cuối tuần tựa lên ~7 quan sát.
  Con số p99 phải đọc là chỉ dấu, không phải ước lượng chắc.
- Lịch sự kiện phủ 56% số ngày, nên "ngày thường" không thật sự là ngày không có
  tin. Hiệu ứng 2% vì vậy là **chặn dưới**, không phải ước lượng đầy đủ.
