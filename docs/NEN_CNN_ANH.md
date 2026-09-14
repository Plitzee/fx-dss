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

## 2b. So sánh kiến trúc — thử Vision Transformer trên CÙNG ảnh, CÙNG quy mô tham số

*Thêm 14/09/2026.* Văn liệu 2024-2026 (Stanford CS231n 2025 "Learning
Predictive Candlestick Patterns: Vision Transformers for Technical
Analysis"; arXiv 2605.00875 "Visual Chart Representations for
Cryptocurrency Regime Prediction") báo cáo Vision Transformer (ViT) thường
vượt CNN trên bài toán ảnh nến — cơ chế tự chú ý bắt được quan hệ XA giữa
các nến (nến 1 và nến 10) mà tích chập cục bộ khó bắt trực tiếp.

Script: [`src/kiem_vit_nen.py`](../src/kiem_vit_nen.py) — patch embedding
4×4 + CLS + vị trí học được + 2 lớp Transformer Encoder, **20.865 tham số**
(gần đúng bằng 21.393 tham số của CNN) để so sánh công bằng chỉ đổi kiến
trúc, không đổi quy mô mô hình. Dùng lại NGUYÊN VẸN `dung_du_lieu()` và giao
thức huấn luyện/chọn-epoch/báo cáo của `kiem_cnn_nen.py`.

**Kết quả — ngược hẳn CNN:**

| cặp | CNN chênh QLIKE | ViT chênh QLIKE | ViT DM p | ViT Holm+Bonferroni |
|---|---|---|---|---|
| AUDUSD | −2,44% (cải thiện) | **+1,63% (xấu đi)** | 0,0007 | **XẤU ĐI có ý nghĩa** |
| EURUSD | −2,53% (cải thiện) | **+2,04% (xấu đi)** | 0,0006 | **XẤU ĐI có ý nghĩa** |
| GBPUSD | −1,33% (cải thiện, ĐẠT) | +0,40% (xấu đi) | 0,1604 | không đáng kể |
| USDCAD | −2,41% (cải thiện, ĐẠT) | +1,03% (xấu đi) | 0,0654 | không đáng kể (biên) |
| USDCHF | −1,69% (cải thiện) | **+2,11% (xấu đi)** | 0,0003 | **XẤU ĐI có ý nghĩa** |
| USDJPY | −0,09% (gần như không đổi) | **+2,52% (xấu đi)** | 0,0000 | **XẤU ĐI có ý nghĩa** |

**6/6 cặp ViT làm QLIKE XẤU ĐI về hướng, 4/6 xấu đi CÓ Ý NGHĨA** sau cả Holm
và Bonferroni — trong khi CNN (cùng quy mô tham số, cùng dữ liệu, cùng giao
thức) cải thiện ở 6/6 cặp và đạt ý nghĩa ở 3/6. Đây là kết quả **hoàn toàn
ngược** với xu hướng báo cáo trong văn liệu 2024-2026.

**Diễn giải hợp lý nhất:** với chỉ ~3.100 cửa sổ huấn luyện/cặp, kiên kiến
trúc tự chú ý (không có thiên kiến quy nạp về TÍNH CỤC BỘ KHÔNG GIAN như
tích chập) cần NHIỀU dữ liệu hơn để học đúng, hoặc cần huấn luyện trước
(pretrain) trên tập lớn — đúng như thực tế đã biết rộng rãi trong thị giác
máy tính (ViT gốc, Dosovitskiy et al. 2021, cũng chỉ vượt CNN khi có
pretrain quy mô lớn). Các bài báo 2024-2026 báo cáo ViT thắng CNN thường
dùng trọng số pretrain trên ImageNet rồi tinh chỉnh — ở đây huấn luyện ViT
TỪ ĐẦU (không pretrain, vì ảnh nến 2 kênh trừu tượng không phải ảnh tự
nhiên nên đặc trưng ImageNet chưa chắc chuyển giao được) trên dữ liệu ít.

**Kết luận cho hệ thống:** ở quy mô dữ liệu và thiết kế hiện tại, **CNN là
lựa chọn tốt hơn ViT** cho bài toán mã hoá ảnh nến — không phải vì CNN
"hiện đại hơn" mà vì thiên kiến quy nạp của nó (cục bộ, bất biến tịnh tiến)
phù hợp hơn với lượng dữ liệu sẵn có. Đây là minh chứng cụ thể cho nguyên
tắc "phương pháp hiện đại hơn không tự động tốt hơn" — phải kiểm bằng số,
không suy luận từ danh tiếng kiến trúc.

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
