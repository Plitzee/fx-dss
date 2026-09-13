# BỘ LỌC CHU KỲ NỘI TUẦN (BCL) — tiêu chí CHỐT TRƯỚC

Lập 13/09/2026. Nhánh `replan-2026`. **Commit chứa văn bản này nằm trước mọi
commit có số liệu.**

---

## 1. Lỗ hổng nhắm tới — và xác minh rằng nó thật

`MAU_HINH_FX.md` mục A2 gọi đây là *"cải tiến tầng 2 có cơ sở tài liệu mạnh
nhất còn chưa làm"*:

> *"Chuyện Chủ nhật đã sửa (`merge_thin_days`) mới chỉ là phần thô nhất của vấn
> đề này. Phần tinh hơn — chu kỳ theo **giờ trong ngày** và theo **ngày trong
> tuần** bên trong mỗi phiên — vẫn chưa được khử trong `rv_adv.csv`."*

**Đã xác minh nó không trùng với `deseason` đã bị loại.** `KETQUA_VONG7.md` ghi
khử mùa vụ nội tuần là *"âm — giúp khi chưa có lịch, thừa khi đã có"*. Nhưng
`volfc2.he_so_mua_vu` nhận `lv = log(rv5)` — tức khử mùa vụ **theo thứ, trên
chuỗi RV NGÀY**. BCL làm việc trên **lợi suất NỘI NGÀY, TRƯỚC khi cộng dồn
thành RV ngày**. Hai thứ khác nhau; cái sau chưa bao giờ chạm vào.

## 2. Nguồn

- **Boudt, Croux & Laurent (2011)**, *Robust Estimation of Intraweek
  Periodicity in Volatility and Jump Detection*, **Journal of Empirical
  Finance** 18(2):353–367.
- Áp cho FX tần suất cao: **Yi (2023)**, *Finance Research Letters* 55:103821.

Chữ **intraweek** (không chỉ *intraday*) là điểm cốt yếu: nó xử lý tương tác
**ngày-trong-tuần × giờ-trong-ngày**, đúng thứ FX cần vì tuần FX là 24×5.

## 3. Cách dùng ĐÚNG — và vì sao không phải "RV thay thế"

BCL **không** là một ước lượng RV thay thế. Chia RV cho hệ số chu kỳ sẽ ước một
**đại lượng khác** với phương sai tích hợp thật của ngày đó, nên mục tiêu sẽ
không còn so sánh được. BCL là **bộ lọc chuẩn hoá** để ngưỡng phát hiện nhảy áp
đồng đều qua mọi ô trong tuần:

> Một lợi suất 3σ lúc 14:00 New York (ô có phương sai cao **dự đoán được**) ít
> khả năng là nhảy hơn cùng lợi suất đó lúc 22:00.

Nên sản phẩm của nó là một phép **tách liên tục/nhảy (C/J) tốt hơn**, rồi đưa
vào HAR dưới dạng hai số hạng — đúng sơ đồ **HAR-CJ** (Andersen, Bollerslev &
Diebold 2007). Mục tiêu giữ nguyên `log rv5(t+1)`, nên QLIKE so sánh được.

## 4. Giả thuyết CHỐT TRƯỚC

> **H11.** Tách C/J bằng ngưỡng **đã lọc chu kỳ nội tuần** cho dự báo biến động
> tốt hơn mốc HAR sản xuất (vốn tách C/J bằng **bipower**, bất biến với thứ tự
> thời gian trong ngày).
>
> **H11b — điều kiện ghi công.** P1 (có lọc) phải tốt hơn **P2 (cùng phép tách,
> KHÔNG lọc chu kỳ)**. Nếu P2 cũng thắng tương đương thì thứ ăn tiền là *phát
> hiện nhảy bằng ngưỡng*, **không phải** lọc chu kỳ, và **BCL không được ghi
> công** — phải viết đúng như vậy.
>
> **Dấu dự kiến:** hệ số của `log C` **lớn hơn** hệ số của `log(1 + J/C)` —
> phần liên tục dai hơn phần nhảy, đúng phát hiện chuẩn của HAR-CJ.

## 5. Năm cấu hình — liệt kê đầy đủ TRƯỚC khi chạy

| | đặc trưng thêm vào `log h_HAR` |
|---|---|
| **B0** mốc | — (HAR sản xuất, đã có bipower C/J bên trong) |
| **P1** BCL | `log c_bcl`, `log(1 + j_bcl/c_bcl)` |
| **P2** ngưỡng thô *(đối chứng)* | `log c_tho`, `log(1 + j_tho/c_tho)` |
| **P3** cả hai | P1 + P2 |
| **P4** chỉ tỉ lệ chu kỳ | `ti_chuky` = Σ(r/s)² / Σr² |

**Đếm vào `KHOA_SO.md`: 4 cấu hình mới.** Không thêm cấu hình nào ngoài bảng này.

Tham số chốt tại đây, không điều chỉnh sau: ô = (thứ) × (288 ô 5 phút) ·
ước lượng scale **robust MAD** (`/0,6745`) · chuẩn hoá BCL
`mean(s²) = 1` · ngưỡng nhảy **c = 4** · tối thiểu 60 quan sát mỗi ô · tối
thiểu 100 ô mỗi ngày.

### 5a. ĐÍNH CHÍNH ĐẶC TẢ — ghi TRƯỚC khi thấy bất kỳ số liệu nào

Bản đầu của mục 5 viết **1.440 ô = (thứ 0–4) × 288**, dựa trên câu *"tuần FX
là 24×5"* của `MAU_HINH_FX.md`. **Sai trên dữ liệu thật.** Nến M1 của HistData,
sau khi chuyển sang UTC, **có cả ô thứ Bảy và Chủ nhật** — phiên Sydney mở vào
chiều Chủ nhật giờ New York. Chạy lần đầu báo `IndexError: index 1981 out of
bounds for size 1440`, tức có thật ô ở thứ 6.

Sửa thành **2.016 ô = 7 × 288**. Các ô cuối tuần ít mẫu sẽ không đạt ngưỡng 60
quan sát và rơi về trung vị, đúng cơ chế dự phòng đã chốt ở mục 5.

Đây là đính chính **đặc tả**, không phải tinh chỉnh: nó được phát hiện bằng một
lỗi chạy chương trình, sửa **trước khi** bất kỳ QLIKE nào được chấm, và nó làm
bộ lọc **bao phủ hơn** chứ không làm nó dễ thắng hơn. Ghi lại để người đọc thấy
rõ thứ tự, và vì chính câu "24×5" trong `MAU_HINH_FX.md` cũng cần đính chính.

## 6. Chống rò rỉ

Hệ số chu kỳ `s(thứ, ô)` ước **CHỈ trên đoạn huấn luyện** (< 2021-10-13), đóng
băng, rồi áp cho cả ba đoạn. Ba tự kiểm bắt buộc, dừng nếu hỏng:

1. Chuẩn hoá đúng: `mean(s²) = 1` tới sai số 1e−9.
2. **Cắt bỏ toàn bộ dữ liệu sau 2023-01-01 không làm đổi một hệ số nào** trong
   1.440 ô (vì hệ số chỉ dùng dữ liệu huấn luyện).
3. Chu kỳ có thật: `max(s)/min(s) > 2` — nếu không thì không có gì để khử, và
   mọi kết quả sau đó vô nghĩa.

## 7. Giao thức chấm

Khung so sánh **giống hệt** `kiem_nen_ablation.py` và `run_tang2_taptrung.py`,
để kết quả đặt cạnh `|gap|` và `hhi` được: `log rv5(t+1) ~ log h_HAR + đặc
trưng`, OLS trên **huấn luyện**, chọn trên **kiểm định**, chấm **một lần** trên
**kiểm tra**, QLIKE bất biến thang đo, DM + Newey–West, MCS, tách theo cặp.

## 8. TIÊU CHÍ PHỦ ĐỊNH

Kết luận **"lọc chu kỳ nội tuần không mang lại giá trị đo được"** khi bất kỳ
điều nào sau đây đúng:

1. Cấu hình tốt nhất trên **kiểm định** không thắng B0 trên **kiểm tra**, hoặc
   thắng nhưng DM p ≥ 0,05.
2. Hoặc P1 **không** tốt hơn P2 — khi đó BCL không được ghi công kể cả khi
   dự báo tốt lên (H11b).
3. Hoặc không đạt ≥ 5/6 cặp cải thiện.

Kết luận **dương** đòi **cả ba**: thắng kiểm tra với DM p < 0,0125 (Bonferroni
4 cấu hình) **và** P1 tốt hơn P2 **và** ≥ 5/6 cặp.

Không sửa tiêu chí sau khi thấy số. Nếu tiêu chí soạn dở thì khai báo chỗ dở,
giữ nguyên phán quyết — đúng như đã làm ở `DUOI_EVT.md` mục A5a.

## 9. Lực phát hiện — khai báo TRƯỚC

Bảng ~21.600 hàng (3.282 ở mỗi đoạn ngoài mẫu), cùng cỡ với thí nghiệm `|gap|`
và `hhi`. Ở cỡ này, `|gap|` đạt DM p = 0,0445 với cải thiện 0,67%, còn `hhi`
không đạt với 0,76% theo chiều xấu. Nên **ngưỡng phát hiện thực tế ở khung này
vào khoảng 0,5–0,7% QLIKE**. Cải thiện nhỏ hơn mức đó sẽ không phân biệt được
với nhiễu, và kết quả âm phải phát biểu kèm con số này.

## 10. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 4 (giả thuyết, kèm điều
kiện ghi công H11b), mục 8 (tiêu chí phủ định) và mục 9 (khai báo lực) **trước
khi** số liệu được chấm:

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không phải
> công cụ điền hộ.
