# PHA 3 — TIÊU CHÍ CHỐT TRƯỚC (viết TRƯỚC khi chạy bất kỳ thí nghiệm nào)

Lập 12/09/2026. Nhánh `replan-2026`.

Văn bản này tồn tại vì `NHANXET_ROADMAP.md` mục 3.2 và đề xuất 8 chỉ ra rằng
roadmap của HuyH **không có tiêu chí phủ định cho Pha 2 và Pha 3**:

> *"Thiếu nó thì hai pha sau sẽ luôn tìm ra một cái gì đó — vì không ai định
> nghĩa trước thế nào là 'không có gì'."*

Pha 2 đã đóng mà không có văn bản như thế viết trước (tiêu chí được suy ra
sau). Pha 3 không lặp lại điều đó. **Commit chứa file này phải nằm TRƯỚC mọi
commit chứa số liệu Pha 3.**

---

## 1. Tầng thông tin của Pha 3

| tầng | nội dung | trạng thái |
|---|---|---|
| M0 | nền không mô hình (chỉ σ̂, ngẫu nhiên) | xong — Giai đoạn 0 |
| M1 | **lịch sử** giá/biến động của chính cặp | xong — Pha 1 |
| M2 | **tin tức** (thông cáo NHTW, bất ngờ thị trường) | xong — `PHA2_KETQUA.md` |
| **M3** | **bối cảnh vĩ mô / xuyên tài sản** — chênh lệch lãi suất, lập trường chính sách | **Pha 3 — bắt đầu ở đây** |

## 2. Câu hỏi nghiên cứu — kể cả RQ về TRỤC (đề xuất 9)

- **RQ8** — Bối cảnh vĩ mô có cải thiện dự báo vượt M1 (và M2) không?
- **RQ9** *(RQ về trục, roadmap gốc không có)* — **Trên trục nào** (hướng /
  biên độ / rủi ro) thông tin vĩ mô mang lại giá trị **đo được**?

RQ9 là câu hỏi trung tâm của cả dự án. Kết quả Pha 1–2 đã trả lời cho hai
tầng đầu: **hướng đi không có kỹ năng, biên độ có.** Pha 3 phải trả lời cùng
một câu cho tầng vĩ mô, chứ không chỉ "dự báo có tốt lên không".

## 3. Giả thuyết CHỐT TRƯỚC

Chỉ chạy trên **trục biên độ** (biến động). Lý do đã có bằng chứng: trục hướng
giá đã cạn kiệt qua 8.652 giả thuyết, 0 sống sót, trên 12 nhánh độc lập. Mở
lại trục đó ở Pha 3 là tiêu thêm ngân sách kiểm định bội cho một câu hỏi đã
được trả lời.

> **H9.** *Phân kỳ lập trường chính sách tiền tệ* — độ lớn thay đổi của chênh
> lệch lãi suất giữa hai đồng — dự báo **biến động cao hơn** cho cặp đó.
>
> Cơ chế: chênh lệch lãi suất đổi chiều buộc định giá lại vị thế carry; tháo
> carry là cơ chế đã biết sinh biến động (Brunnermeier, Nagel & Pedersen 2009,
> *Carry Trades and Currency Crashes*).
>
> **Dấu dự kiến: DƯƠNG.** Hệ số âm = giả thuyết bị bác bỏ, kể cả khi dự báo
> có tốt lên (đúng như đã xảy ra với S4 ở `PHA2_KETQUA.md` mục 3b).

### 3a. Danh sách đặc trưng — liệt kê đầy đủ TRƯỚC khi chạy

Nguồn: `data/fred_rates.csv` (lãi suất liên ngân hàng 3 tháng, 8 đồng) và
`data/carry.csv`. Cả hai là chuỗi **tháng**.

| # | đặc trưng | định nghĩa |
|---|---|---|
| 1 | `chenh_ls` | chênh lệch lãi suất của cặp (mức) |
| 2 | `d_chenh_ls` | thay đổi 3 tháng của `chenh_ls` (có dấu) |
| 3 | `abs_d_chenh_ls` | **độ lớn** thay đổi 3 tháng — biến của H9 |
| 4 | `ls_usd` | mức lãi suất USD (đại diện điều kiện tiền tệ toàn cầu) |
| 5 | `abs_d_ls_usd` | độ lớn thay đổi 3 tháng của lãi suất USD |

**Biến thể sẽ chạy: 7** (1 mốc + 5 đơn lẻ + 1 gộp `abs_d_chenh_ls +
abs_d_ls_usd`). Con số này chốt tại đây và đếm vào `KHOA_SO.md`.

## 4. Chống rò rỉ — chốt trước

Chuỗi FRED theo tháng được công bố **có độ trễ**. Quy tắc áp dụng:

> Giá trị của tháng *m* chỉ được dùng cho các phiên từ tháng *m+2* trở đi
> (trễ **2 tháng**).

Trễ 2 tháng là bảo thủ hơn độ trễ công bố thật (thường 1–4 tuần) và không cần
tra cứu lịch công bố từng chuỗi. Tự kiểm bắt buộc: **cắt bỏ toàn bộ tương lai
không được làm đổi giá trị đặc trưng của bất kỳ phiên nào** — đúng phép kiểm
đã dùng cho H8/H8b/H8c/H8e.

## 5. Giao thức chấm — y hệt Pha 1 và Pha 2

Mốc **B1 = HAR sản xuất** (`V2.du_bao_san_xuat`), hồi quy lại bằng OLS. Khớp
trên **huấn luyện**, chọn trên **kiểm định**, chấm **một lần** trên **kiểm
tra**. QLIKE bất biến thang đo; DM + Newey–West; tách theo từng cặp.

Chia dữ liệu không đổi: huấn luyện 2010-01→2021-10 · kiểm định 2021-10→2023-11
· kiểm tra 2023-11→2025-12.

## 6. TIÊU CHÍ PHỦ ĐỊNH — thứ Pha 2 đã thiếu

Pha 3 kết luận **"tầng vĩ mô không mang lại giá trị đo được, và dừng"** khi
**tất cả** các điều sau đúng:

1. Biến thể tốt nhất trên **kiểm định** **không** thắng B1 trên **kiểm tra**
   (chênh QLIKE ≥ 0); **hoặc** thắng nhưng DM p ≥ 0,05.
2. **Và** không đạt ≥ 5/6 cặp cải thiện.
3. **Và** dấu hệ số của `abs_d_chenh_ls` không khớp giả thuyết ở mục 3,
   **hoặc** khớp nhưng độ lớn không đổi được kết quả ở (1).

Kết luận **dương** đòi: thắng trên kiểm tra với **DM p < 0,05 sau Bonferroni
cho 6 biến thể** (tức p thô < 0,0083), **và** ≥ 5/6 cặp cải thiện, **và** dấu
hệ số khớp giả thuyết chốt trước.

Ba điều kiện, không phải một. Bài học trực tiếp từ S4: S4 đạt điều kiện 1 và
2 nhưng **trượt điều kiện 3**, và phép thử tổng quát sau đó cho kết quả âm.

## 7. Lực phát hiện — khai báo TRƯỚC, không phải sau

Đặc trưng là chuỗi **tháng**, nên số quan sát **độc lập** không phải 21.582
phiên mà là số tháng:

| đoạn | phiên | **tháng độc lập** |
|---|---|---|
| huấn luyện | 3.054 | **~142** |
| kiểm định | 547 | **~26** |
| kiểm tra | 548 | **~26** |

**26 tháng ở đoạn kiểm tra là rất ít.** Khai báo trước: phép thử này **lực
yếu theo thiết kế**, và một kết quả âm ở đây **yếu hơn** kết luận âm của Pha 1
(8.652 giả thuyết, MDES lift 1,20) lẫn của Pha 2 (183 giả thuyết, MDES 1,35).

Hệ quả phải ghi trong luận văn: kết luận âm của Pha 3 phát biểu được là *"không
phát hiện được hiệu ứng với dữ liệu tháng trên 26 tháng kiểm tra"*, **không**
phát biểu được là *"tầng vĩ mô không chứa thông tin"*. Đó là giới hạn của dữ
liệu sẵn có, không phải của phương pháp.

## 8. Danh sách KHÔNG LÀM ở Pha 3

- **Không** mở lại trục hướng giá (đã cạn kiệt, 8.652 giả thuyết).
- **Không** mở bộ niêm phong (`KHOA_SO.md`; giữ cho lần chạy cuối toàn hệ thống).
- **Không** thêm đặc trưng ngoài 5 cái ở mục 3a mà không mở biên bản mới.
- **Không** đổi mốc B1, đổi cách chia, hay đổi thước đo sau khi thấy số.

## 9. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 3 (giả thuyết), mục 6
(tiêu chí phủ định) và mục 7 (khai báo lực) **trước khi** số liệu được chấm:

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không phải
> công cụ điền hộ.
