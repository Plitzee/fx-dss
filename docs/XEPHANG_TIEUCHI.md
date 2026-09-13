# XẾP HẠNG CHÉO SÁU ĐỒNG TIỀN — tiêu chí CHỐT TRƯỚC

Lập 13/09/2026. Nhánh `replan-2026`. **Commit chứa văn bản này nằm trước mọi
commit có số liệu.**

---

## 1. Lỗ hổng nhắm tới — và xác minh rằng nó thật

`KEHOACH_2026Q4.md` mục 1.3 đề xuất đổi câu hỏi từ *"EURUSD sẽ lên hay xuống"*
sang *"trong 6 cặp, cặp nào mạnh nhất"*, và ghi *"repo chưa thử lần nào"*.

**Đã xác minh.** `grep` toàn bộ `src/` và `docs/` cho *xếp hạng chéo /
cross-sectional / rank IC / Spearman*: chỉ khớp **đúng một file — chính file kế
hoạch đó**. Không một dòng mã nào. Lỗ hổng thật.

## 2. Vì sao đây không phải lần thứ 14 của cùng một bài toán

Trục hướng đã cạn kiệt qua **8.652 giả thuyết, 13 nhánh độc lập, 0 sống sót**.
Nhưng **tất cả** đều đo hướng **TUYỆT ĐỐI của từng cặp**. Nhân tố đô-la chung
(ρ = 0,443 đã đo trong repo) chiếm gần nửa biến thiên và làm nhiễu mọi phép đo
đó.

Xếp hạng chéo **triệt tiêu nhân tố đô-la bằng cấu trúc**: mỗi phiên, lợi suất
của 6 đồng được **khử trung bình ngang**, nên thứ còn lại là *cường độ tương
đối*, không còn chứa hướng của đô-la. Đó là một bài toán khác, có đích khác
(thứ hạng, không phải dấu) và thước đo khác (rank IC, không phải AUC).

Văn liệu cho hướng này mạnh hơn hẳn dự báo hướng từng cặp:

- **Menkhoff, Sarno, Schmeling & Schrimpf (2012)**, *Currency Momentum
  Strategies*, **Journal of Financial Economics** 106(3):660–684.
- **Lustig, Roussanov & Verdelhan (2011)**, *Common Risk Factors in Currency
  Markets*, **Review of Financial Studies** 24(11):3731–3777.

## 3. Quy đổi về cùng gốc — chốt trước

Lợi suất của đồng X so USD:

```
cặp niêm yết XXXUSD (EUR, GBP, AUD):  r_X = +log(P_t / P_{t−1})
cặp niêm yết USDXXX (JPY, CAD, CHF):  r_X = −log(P_t / P_{t−1})
```

Sáu đồng: EUR, GBP, AUD, JPY, CAD, CHF. Không thêm đồng nào.

## 4. Giả thuyết CHỐT TRƯỚC

> **H12.** Xếp hạng chéo 6 đồng theo động lượng và/hoặc carry cho **rank IC
> dương có ý nghĩa** với thứ hạng lợi suất (đã khử đô-la) của phiên kế tiếp.
>
> **Dấu dự kiến:** động lượng **dương** (đồng mạnh tiếp tục mạnh) và carry
> **dương** (đồng lãi suất cao tăng giá) — đúng chiều hai bài báo ở mục 2.
> Đảo chiều 1 tuần vào với dấu **âm của lợi suất** theo định nghĩa.

## 5. Sáu tín hiệu — liệt kê đầy đủ TRƯỚC khi chạy

| | định nghĩa |
|---|---|
| `mom_1m` | tổng lợi suất 21 phiên |
| `mom_3m` | tổng lợi suất 63 phiên |
| `mom_12m` | tổng lợi suất 252 phiên |
| `dao_1w` | **trừ** tổng lợi suất 5 phiên (đảo chiều ngắn) |
| `carry` | chênh lệch lãi suất đồng X − USD, **trễ 2 tháng** (y hệt Pha 3) |
| `gop` | trung bình z-score ngang của `mom_1m`, `mom_3m`, `mom_12m`, `carry` |

**Đếm vào `KHOA_SO.md`: 6 cấu hình.** Không thêm tín hiệu nào ngoài bảng này.

## 6. CỬA CHI PHÍ — đặt TRƯỚC, không phải sau khi thấy kết quả

`KEHOACH_2026Q4.md` mục 1.1 điểm 4 gọi đây là điều *"quan trọng nhất"*, và nó
áp nguyên ở đây. Danh mục: **mua đồng xếp đỉnh, bán đồng xếp đáy**, 1 đồng mỗi
bên, lợi suất chia 2 để thành trên mỗi đơn vị vốn gộp.

Chi phí: mỗi lần thành phần đổi, trừ **1 pip một chiều cho mỗi chân đổi**
(`spread = 1e−4` trên lợi suất log, lấy từ trung vị spread đã đo trong repo).
Đây là chi phí **bảo thủ theo hướng bất lợi** cho giả thuyết.

Một tín hiệu có IC dương nhưng **lãi ròng ≤ 0** thì **không** được tính là dương.

## 7. Chống rò rỉ

Mọi tín hiệu tại phiên *t* chỉ dùng dữ liệu **đến hết phiên t**, và dự báo lợi
suất phiên *t+1*. Carry trễ 2 tháng như Pha 3. Tự kiểm bắt buộc, dừng nếu hỏng:
**cắt bỏ toàn bộ dữ liệu sau 2023-01-01 không làm đổi một giá trị tín hiệu nào**
ở các phiên trước mốc.

## 8. Giao thức và TIÊU CHÍ PHỦ ĐỊNH

Chọn tín hiệu có IC cao nhất trên **kiểm định**, rồi đọc kết quả của **đúng tín
hiệu đó** trên **kiểm tra** (chấm một lần).

Kết luận **dương** đòi **cả hai**:

1. Rank IC trên kiểm tra **> 0 với t > 1,96**.
2. **Và** lãi **ròng sau chi phí** của danh mục đỉnh−đáy **> 0 với t > 1,96**.

Trượt bất kỳ điều nào → **kết luận âm**: xếp hạng chéo không mang lại giá trị
đo được. Không sửa tiêu chí sau khi thấy số.

## 9. Lực phát hiện — khai báo TRƯỚC

Đoạn kiểm tra có ~547 phiên × 6 đồng. Với 6 quan sát mỗi phiên, sai số chuẩn
của Spearman IC từng phiên vào khoảng `1/√5 ≈ 0,45`; trung bình 547 phiên cho
sai số chuẩn khoảng **0,019**. Nên ngưỡng phát hiện (t > 1,96) là **IC ≈ 0,038**.

Khai báo trước: đây là phép thử **lực thấp về IC** — chỉ 6 tài sản nên mỗi phiên
cho rất ít thông tin xếp hạng. Một kết luận âm ở đây phát biểu được là *"không
phát hiện được IC lớn hơn ~0,04 trên 547 phiên với 6 đồng"*, **không** phát biểu
được là *"động lượng tiền tệ không tồn tại"* — văn liệu dùng hàng chục đồng, nơi
mỗi phiên cho nhiều thông tin xếp hạng hơn hẳn.

## 10. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 4 (giả thuyết), mục 6 (cửa
chi phí), mục 8 (tiêu chí phủ định) và mục 9 (khai báo lực) **trước khi** số
liệu được chấm:

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không phải
> công cụ điền hộ.
