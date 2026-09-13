# ML TRỰC TIẾP TRÊN ĐẠI LƯỢNG RỦI RO — tiêu chí CHỐT TRƯỚC

Lập 13/09/2026. Nhánh `replan-2026`. **Commit chứa văn bản này nằm trước mọi
commit có số liệu.**

---

## 1. Lỗ hổng nhắm tới — và xác minh

`KEHOACH_2026Q4.md` mục 3.2: *"Mọi lần huấn luyện ML trước đây đều nhắm vào
**hướng** (thua) hoặc **phương sai** (thua). Chưa ai huấn luyện ML để dự báo
**chính các đại lượng rủi ro**."*

**Đã xác minh bằng grep toàn repo**: `P(chạm dừng lỗ)` / `vi phạm VaR … phân
loại` chỉ xuất hiện trong chính file kế hoạch. Không một dòng mã nào.

## 2. Vì sao hướng này khác — và vì sao nó có cơ hội thật

Hai mốc đang chạy **đều là HẰNG SỐ**, không phụ thuộc trạng thái thị trường:

| đích | mốc hiện tại | bản chất |
|---|---|---|
| A `P(chạm stop)` | nguyên lý phản xạ dưới t-Student (`decision_record.p_cham_stop`) | hàm của `k` và `ν` — **không** nhận tham số trạng thái |
| B `P(vi phạm VaR 1%)` | mức danh nghĩa 0,01 | hằng số theo định nghĩa |

Đó là một điểm yếu **cấu trúc**, không phải điểm yếu của ước lượng. Nếu xác suất
chạm stop thật sự phụ thuộc chế độ biến động, thành phần nhảy, gap hay lịch họp
NHTW, thì mốc hằng số **không thể** bắt được, và một mô hình có điều kiện sẽ
thắng. Đây là lý do hướng này có cơ hội ở nơi mà ML trên hướng và trên phương
sai đều đã thua.

## 3. Giả thuyết CHỐT TRƯỚC

> **H13.** Xác suất rủi ro (chạm stop, vi phạm VaR) **phụ thuộc trạng thái**, nên
> một mô hình có điều kiện cho Brier thấp hơn mốc hằng số và **BSS > 0** so
> khí hậu học.
>
> **Dấu dự kiến:** `log σ̂` và `ti_nhay` (tỉ trọng nhảy) làm **tăng** xác suất
> chạm stop; `abs_gap` cũng **tăng**. Nếu hệ số ngược dấu thì ghi là bác bỏ.

## 4. Định nghĩa đích — chốt trước, không đổi

```
A   y = 1 nếu min của lợi suất LUỸ TÍCH trong 5 phiên kế tiếp ≤ −1,5·σ̂(t+1)
B   y = 1 nếu lợi suất phiên t+1 ≤ q01·σ̂(t+1),  q01 = phân vị 1% của z trên HUẤN LUYỆN
```

`σ̂(t+1)` là dự báo cho phiên t+1, **biết tại thời điểm t** (HAR dùng dữ liệu
đến hết t). Stop ở **1,5σ̂**, tầm hạn **5 phiên**, mức VaR **1%** — cả ba chốt
tại đây, không điều chỉnh sau.

## 5. Tám đặc trưng — liệt kê đầy đủ TRƯỚC khi chạy

Tất cả biết tại thời điểm *t*: `log_sig` (log σ̂(t+1)) · `che_do` (tam phân vị
của σ̂, **ngưỡng chốt trên huấn luyện**) · `z_t` · `z_t1` · `abs_z_tb5` ·
`ti_nhay` = (rv−bpv)/rv · `abs_gap` · `phien_tu_hop` (số phiên kể từ kỳ họp, lịch
biết trước cả năm nên không rò rỉ, chặn ở 30).

Hai mô hình: **logistic** (có chuẩn hoá) và **LightGBM**.

**Đếm vào `KHOA_SO.md`: 4 cấu hình** (2 mô hình × 2 đích). Không thêm.

## 6. Thước đo

**Brier score** (chính) · **Brier Skill Score** so khí hậu học ước trên huấn
luyện · **ECE** 10 ô (độ hiệu chuẩn). Xác suất phải **được hiệu chuẩn**, không
chỉ xếp hạng đúng — đây là hệ hỗ trợ quyết định, con số hiện lên giao diện phải
đọc được như xác suất thật.

## 7. Giao thức và TIÊU CHÍ PHỦ ĐỊNH

Khớp trên **huấn luyện**, chọn mô hình theo **BSS trên kiểm định**, chấm **một
lần** trên **kiểm tra**.

Kết luận **dương** cho mỗi đích đòi **cả hai**:

1. **BSS > 0** so khí hậu học trên kiểm tra.
2. **Và** Brier thấp hơn **mốc giải tích** hiện đang chạy.

Trượt bất kỳ điều nào → **âm**, không đổi sản xuất. Không sửa tiêu chí sau khi
thấy số.

## 8. Lực phát hiện — khai báo TRƯỚC

Đích B là **rất mất cân bằng**: tần suất nền ~1%, nên đoạn kiểm tra (~3.200
hàng/6 cặp) chỉ có khoảng **32 ca dương**. Với cỡ đó, Brier bị thống trị bởi
phần lớn các ca âm và BSS có phương sai rất lớn. Khai báo trước: **đích B là
phép thử lực thấp**, và một kết luận âm ở B chỉ nói *"không phát hiện được trên
~32 ca dương"*.

Đích A cân bằng hơn nhiều (tần suất nền sẽ in ra trong kết quả) nên nó là phép
thử **chính**; B là phép thử phụ.

## 9. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 3 (giả thuyết), mục 7 (tiêu
chí phủ định) và mục 8 (khai báo lực) **trước khi** số liệu được chấm:

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không phải
> công cụ điền hộ.
