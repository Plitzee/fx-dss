# BIÊN BẢN KHÓA SỔ DỮ LIỆU

**Ngày lập:** 28/08/2026
**Luận văn:** Hệ thống hỗ trợ quyết định giao dịch ngoại hối (luận văn chung)
**Lý do lập:** Từ thời điểm này nhóm bắt đầu giai đoạn *lặp cải tiến* pipeline —
thử nhiều cấu hình cho tới khi kết quả tốt. Nếu không tách một phần dữ liệu ra
trước, mọi kết quả cuối cùng đều mang lỗi *data snooping*: mô hình được chọn vì
nó hợp với chính bộ dữ liệu dùng để đánh giá nó.

---

## 1. TẬP PHÁT TRIỂN — được phép dùng tự do

| Mục | Nội dung |
|---|---|
| Cặp tiền | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF |
| Khoảng | 2010-01-03 → 2025-12-31 |
| Số phiên | 4.994 / cặp (29.843 dòng panel) |
| Nguồn | HistData M1 → H1 → D1, đã chuyển giờ New York sang UTC có xử lý DST |

Trên tập này được phép: đổi mô hình, đổi siêu tham số, đổi đặc trưng, đổi hàm
thưởng RL, thiết kế tầng fuzzy, chạy lại bao nhiêu lần tùy ý.

## 2. TẬP KHÓA SỔ — KHÔNG được chạm cho tới lần chạy cuối

| Mục | Nội dung |
|---|---|
| Cặp tiền | EURGBP, EURJPY, GBPJPY, AUDJPY, EURCHF, NZDUSD |
| Khoảng | 2010-01-01 → 2025-12-31 |
| Cộng thêm | 2026-01 → 2026-08, **cả 12 cặp** |

Hai tập này khác nhau về **hai chiều độc lập**: sáu cặp chéo không có USD hai vế
nên kiểm tra được tính tổng quát ngoài nhóm cặp chính; dữ liệu 2026 nằm hoàn toàn
sau mọi phép thử đã thực hiện nên kiểm tra được tính bền theo thời gian.

### Vị trí trên đĩa

| Tập | Thư mục | Ai đọc |
|---|---|---|
| Phát triển | `histdata_raw/` → `fx_clean/` | `prep_fx.py` mặc định |
| **Khóa sổ** | `histdata_seal/` → `fx_seal/` | **không script nào**, cho tới lần chạy cuối |

Hai thư mục tách rời có chủ đích: `prep_fx.py` tự quét `histdata_raw/`, nên nếu
để chung thì dữ liệu khóa sổ sẽ chảy vào pipeline ngay và biên bản này vô nghĩa.
Lần chạy cuối mở niêm phong bằng: `py prep_fx.py --src histdata_seal --out fx_seal`

### Quyết định về dữ liệu — ghi ngày 28/08/2026, TRƯỚC mọi phân tích

**AUDJPY năm 2012 bị loại khỏi tập khóa sổ.**

Phát hiện: kho lưu trữ HistData cho `AUDJPY_2012` chỉ chứa 33.047 dòng phủ
07/10/2012 → 22/10/2012, thay vì ~370.000 dòng cả năm. Đã thử tải lại theo năm
(`--force`) — vẫn cho kết quả như cũ, nên đây là khiếm khuyết của kho lưu trữ
chứ không phải lỗi tải. Tải theo tháng không khả dụng: HistData chỉ mở trang
theo tháng cho năm hiện hành.

Quyết định: **loại năm 2012 của AUDJPY, giữ nguyên cặp này cho 15 năm còn lại.**

Lý do không vá bằng nguồn khác (Dukascopy / Forexite / TrueFX đều có dữ liệu này):
vá sẽ cấy một mối nối giữa hai nhà cung cấp vào giữa chuỗi của một cặp, ngay
trong bộ dữ liệu tồn tại để xác nhận kết luận. Mối nối đó chưa được hiệu chuẩn
cho cặp JPY và có thể tạo bước nhảy giả ở hai điểm nối. Phần dữ liệu mất là
~250 phiên trên tổng ~24.000 của tập khóa sổ (1%), không đủ để đổi kết luận nào.
Một lỗ hổng được khai báo trung thực hơn một mối nối chưa kiểm chứng.

Lưu ý phương pháp luận: quy tắc này **không** được tổng quát thành "loại mọi cặp
có dữ liệu thiếu". Lỗ hổng trong kho dữ liệu tài chính tập trung ở các cặp thanh
khoản thấp và giai đoạn thị trường căng thẳng; loại bỏ theo tiêu chí đó sẽ lọc
mẫu về phía những trường hợp dễ. Ở đây loại đúng một năm của một cặp, vì lý do
kỹ thuật đã xác minh, và được ghi lại trước khi phân tích.

**Rào chắn liền mạch (áp cho cả hai tập).** Mọi cửa sổ trượt — MA20, HAR 5 ngày
và 22 ngày, Yang-Zhang 5 ngày — chỉ được tính khi toàn bộ cửa sổ không bắc qua
khoảng trống lớn hơn 4 ngày lịch. Vượt ngưỡng thì trả về rỗng. Lý do: nhóm đã
từng dính đúng lỗi này ở HAR-RV, khi cửa sổ 22 ngày âm thầm nối hai đoạn cách
nhau nhiều tháng. Đã kiểm chứng tập phát triển không có khoảng trống nào trên
4 ngày (khoảng cách lớn nhất là 3 ngày, tức cuối tuần), nên rào chắn là vô hiệu
ở đó — đúng như mong đợi, và đó chính là phép tự kiểm.

## 3. QUY TẮC

1. Tập khóa sổ được chạy **đúng một lần**, sau khi cấu hình cuối đã chốt và ghi
   vào mục 4 dưới đây.
2. Nếu chạy tập khóa sổ rồi mà kết quả xấu, **không được** quay lại sửa mô hình
   rồi chạy lại. Kết quả xấu là một phát hiện, phải báo cáo đúng như nó xảy ra.
3. Nếu bắt buộc phải chạy lần hai (ví dụ phát hiện lỗi code, không phải kết quả
   xấu), phải ghi lại lý do vào mục 5 và nêu trong luận văn.
4. Mọi cấu hình đã thử trên tập phát triển phải được đếm và ghi vào mục 5, để
   phần báo cáo hiệu chỉnh được đa kiểm định.

## 4. CẤU HÌNH CHỐT — điền trước khi mở tập khóa sổ

- Mô hình biến động: ................................................
- Quy tắc định cỡ: ..................................................
- Tham số tầng fuzzy: ...............................................
- Ngày chốt: ........................................................
- Ký xác nhận: ......................................  /  ....................

## 5. NHẬT KÝ THỬ NGHIỆM TRÊN TẬP PHÁT TRIỂN

| # | Ngày | Thay đổi gì | Kết quả chính |
|---|---|---|---|
| 1 | 22/08/2026 | 10 mô hình biến động, walk-forward, MIN_TRAIN=250 | MA20-GK hạng TB 1,3; HAR thua 6/6 (DM p<0,01) |
| 2 | 28/08/2026 | RL REINFORCE đòn bẩy tuyệt đối, 3 seed | Biến thiên seed 56% — không dùng được |
| 3 | 28/08/2026 | RL REINFORCE tham số hóa phần dư | Biến thiên seed 30%; phá sản trung vị 1,3% |
| 4 | 28/08/2026 | PPO tham số hóa phần dư, 6 seed | Biến thiên seed 3%; k học được = 1,08 |
| 5 |  |  |  |

---

*Biên bản này lập TRƯỚC khi tập khóa sổ được tải về máy. Đưa vào phụ lục luận văn.*
