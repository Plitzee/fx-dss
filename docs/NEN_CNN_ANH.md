# MẪU HÌNH NẾN QUA ẢNH CNN — THÍ ĐIỂM PHƯƠNG PHÁP HIỆN ĐẠI

*Lập 14/09/2026. Nhánh `replan-2026`. Bổ sung SAU khi nhánh K1-K4 (hình học
+ mẫu có tên, `docs/NEN_TIEUCHI.md`) đã đóng băng — **không mở lại phán
quyết đó**. Đây là phương pháp THỨ 14, khác hẳn 13 nhánh trước về bản chất.*

---

## 0. Vì sao thử phương pháp này

13 nhánh trước (SAX/motif/matrix-profile/rule-list/K1-K4 hình học+mẫu có
tên) đều dùng **đặc trưng số** (thống kê tóm tắt của nến) làm đầu vào cho
kiểm định tuyến tính/hồi quy. Không nhánh nào học trực tiếp trên **hình dạng
không gian** của nến — đúng cách một nhà giao dịch "đọc biểu đồ" hay đúng
cách bài báo hiện đại dưới đây tiếp cận vấn đề.

**Tài liệu tham khảo:** Chen, J.-H., & Tsai, Y.-C. (2020). "Encoding
candlesticks as images for pattern classification using convolutional
neural networks." *Financial Innovation*, 6, Article 26.
https://doi.org/10.1186/s40854-020-00187-0 — mã hoá cửa sổ nến thành ẢNH rồi
dùng CNN học trực tiếp trên hình dạng, thay vì trích đặc trưng thủ công.

Script: [`src/kiem_cnn_nen.py`](../src/kiem_cnn_nen.py)

---

## 1. Thiết kế

- **Ảnh:** cửa sổ 10 nến liên tiếp → ảnh 2 kênh (kênh bấc, kênh thân
  tăng/giảm), 32 pixel cao, chuẩn hoá theo biên độ giá CỦA CHÍNH cửa sổ đó
  (không dùng thông tin ngoài cửa sổ — không rò rỉ).
- **Mục tiêu:** học PHẦN DƯ so với log HAR sản xuất (không học lại từ đầu),
  để trả lời đúng câu "hình ảnh nến có thêm thông tin NGOÀI mốc HAR không".
- **CNN:** 2 lớp tích chập nhỏ + pooling + 2 lớp tuyến tính — kiến trúc tối
  giản, không cố ép tham số lớn hơn cần thiết cho bài toán.
- **Giao thức chống rò rỉ:** khớp trên đoạn HUẤN LUYỆN; đoạn KIỂM ĐỊNH tách
  làm HAI nửa theo thời gian — nửa đầu **chỉ để chọn epoch dừng sớm**, nửa
  sau **chưa từng được mô hình nhìn thấy ở bất kỳ bước nào**, dùng DUY NHẤT
  để báo cáo QLIKE/DM. Đoạn kiểm tra hoàn toàn không chạm.

**Lưu ý minh bạch về quy trình:** lần chạy đầu tiên dùng CHUNG một đoạn kiểm
định vừa để chọn epoch vừa để báo cáo kết quả — cho p=0,0003 trên EURUSD,
trông rất đẹp. Phát hiện ngay đây là lỗi rò rỉ kinh điển (mô hình được chọn
DỰA TRÊN chính đoạn dùng để chấm điểm nó) và sửa lại bằng cách tách hai nửa
độc lập trước khi tin bất kỳ con số nào — đúng đúng loại lỗi mà toàn bộ
triết lý Westfall-Young/sealed-set của luận văn tồn tại để ngăn.

---

## 2. Kết quả — sau khi sửa rò rỉ, chạy trên cả 6 cặp

| cặp | QLIKE mốc HAR | QLIKE HAR+CNN | chênh (%) | DM p thô | Holm | Bonferroni |
|---|---|---|---|---|---|---|
| AUDUSD | −8,8694 | −9,0940 | **−2,44** | 0,0000 | **ĐẠT** | **ĐẠT** |
| GBPUSD | −9,3513 | −9,4749 | **−1,33** | 0,0003 | **ĐẠT** | **ĐẠT** |
| USDCAD | −9,8580 | −10,0933 | **−2,41** | 0,0000 | **ĐẠT** | **ĐẠT** |
| USDCHF | −9,4118 | −9,5625 | −1,69 | 0,0213 | không | không |
| EURUSD | −9,5813 | −9,8210 | −2,53 | 0,0987 | không | không |
| USDJPY | −9,2554 | −9,2627 | −0,09 | 0,8694 | không | không |

(chênh âm = QLIKE thấp hơn = tốt hơn; công thức chia cho |QLIKE mốc| để dấu
% khớp với dấu cải thiện thực tế — bản chạy đầu tiên có lỗi tiểu tiết ở đây:
QLIKE ở thang này luôn âm nên chia trực tiếp cho mốc sẽ làm % ĐỔI DẤU và đọc
nhầm thành "tệ hơn"; đã sửa trước khi đưa vào bảng này)

**Lưu ý về độ lặp lại:** huấn luyện CNN trên GPU (CUDA/cuDNN) không hoàn
toàn tất định dù đã cố định seed — chạy lại cho p-value lệch nhẹ vài phần
nghìn giữa các lần (vd EURUSD p=0,0987 so với 0,1119 ở lần chạy trước), tuy
kết luận sống sót/không sống sót qua hiệu chỉnh KHÔNG đổi giữa hai lần chạy.
Đây là giới hạn cần nêu rõ, không giấu.

**3/6 cặp (AUDUSD, GBPUSD, USDCAD) sống sót qua cả Holm và Bonferroni** —
kiểm soát sai số họ (family-wise) nghiêm ngặt trên 6 phép kiểm đồng thời.
Đây là nhánh KHAI PHÁ MẪU HÌNH ĐẦU TIÊN trong toàn bộ 14 nhánh (8.652+ giả
thuyết cũ, cộng 600 giả thuyết K1-K4) sống sót qua hiệu chỉnh đa kiểm định
trên bất kỳ cặp nào. Hướng cải thiện nhất quán ở cả 6/6 cặp (không cặp nào
cho dấu ngược lại), kể cả 3 cặp không đạt ngưỡng thống kê.

---

## 3. Vì sao CHƯA đưa vào sản xuất — chưa đạt ngưỡng đóng băng của chính dự án

Tiêu chí độ vững đã dùng xuyên suốt Pha3B (`do_vung`, `pha3b_granger.py`)
đòi **≥5/6 cặp** đồng thuận dấu trước khi một quan hệ được coi là "vững".
**3/6 KHÔNG đạt ngưỡng đó.** Ngoài ra:

- Mới chạy **một kiến trúc, một cỡ cửa sổ (10 nến), một seed ngẫu nhiên**
  — chưa kiểm độ nhạy qua nhiều seed/kiến trúc như phần còn lại của dự án
  vẫn yêu cầu trước khi kết luận.
- Chưa có đối chứng âm (permutation trên nhãn) để đo tỷ lệ dương tính giả
  của CHÍNH quy trình chọn-epoch-rồi-báo-cáo này — cần trước khi tin p-value
  ở mức tuyệt đối.
- Chưa chạy leave-one-pair-out hay kiểm tra qua các năm riêng lẻ.
- Hoàn toàn chưa chạm đoạn kiểm tra hay tập niêm phong.

**Kết luận trung thực:** đây là tín hiệu ĐÁNG THEO ĐUỔI nhất trong toàn bộ
lịch sử khai phá mẫu hình của dự án — nhưng theo đúng thanh chắn mà chính dự
án đặt ra cho các nhánh khác, nó **CHƯA đủ điều kiện** để tuyên bố là phát
hiện vững. Hướng tiếp theo (nếu luận văn có thời gian): lặp lại với 3-5 seed,
thêm đối chứng âm hoán vị, kiểm LOPO — đúng quy trình đã áp dụng cho 13
nhánh trước, trước khi cân nhắc đưa vào Level 2 Granger chính thức.
