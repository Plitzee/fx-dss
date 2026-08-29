# Tầng 6 — Lớp đầu ra và kiểm định hiệu chuẩn

## 1. Tầng 6 làm gì

Tầng 6 biến kết quả của tầng 1–5 thành một **phiếu quyết định** người dùng đọc
được. Ba phần:

| Phần | Nội dung | Kiểm định được? |
|---|---|---|
| Hành động | đòn bẩy khuyến nghị, vốn đặt | không (phụ thuộc lợi thế giả định) |
| Rủi ro | P(chạm stop), khoảng giá dự báo | **có** |
| Giải thích | ràng buộc nào đang siết, k_vol, k_dd | không (truy vết, không dự báo) |

Chỉ phần rủi ro là kiểm định được, và nó cũng là phần quan trọng nhất: nếu
phiếu ghi "xác suất chạm stop 15%" mà thực tế là 30% thì cả hệ thống chỉ là
đồ trang trí. Toàn bộ kiểm định dưới đây chạy trên **tập giữ riêng** (30%
cuối của mỗi cặp), tham số ước lượng trên 70% đầu.

## 2. Kiểm định 1 — xác suất chạm stop

Công thức: nguyên lý phản xạ, `P(min ≤ −b) = 2·P(X_T ≤ −b)`, dưới t-Student
khớp trên tập huấn luyện (không phải Gauss).

| Ngưỡng stop | Lệch dự báo − thực tế |
|---|---|
| 0,5σ | +3,08% |
| 2,5σ | +0,40% |
| 3,0σ | +0,10% |

Lệch tuyệt đối trung bình trên 6 ngưỡng: **1,44%**; n = 8.957 mỗi ngưỡng.

Kết luận: con số headline đáng tin, và đáng tin nhất ở đúng vùng hay dùng
(stop 2–3σ). Ở stop rất gần thì dự báo **cao hơn** thực tế — lệch theo hướng
bi quan, tức an toàn.

## 3. Kiểm định 2 — khoảng dự báo giá

Ba cách dựng khoảng, đo trên 4 mức danh nghĩa 80/90/95/99%:

| Phương pháp | \|lệch\| TB | Ghi chú | Bề rộng @99% |
|---|---|---|---|
| Gauss | 1,20% | 99% danh nghĩa chỉ phủ **97,5%** — hụt đuôi | 119,3 pip |
| Student-t | 1,25% | sửa được đuôi nhưng rộng nhất | 158,0 pip |
| **Conformal** | **0,37%** | vừa chuẩn hơn vừa hẹp hơn Student-t | 145,6 pip |

Conformal thắng thẳng, không đánh đổi.

## 4. Kiểm định 3 — độ phủ **có điều kiện**

Bảo đảm của conformal chỉ là **biên** (đúng trung bình toàn mẫu), không phải
có điều kiện. Kiểm ở mức danh nghĩa 90%, chia theo tam phân vị biến động:

| Phương pháp | vol thấp | vol vừa | vol cao | \|lệch\| max |
|---|---|---|---|---|
| Gauss | 87,7% | 90,3% | 91,0% | 2,3% |
| Student-t | 87,0% | 89,3% | 90,1% | 3,0% |
| Conformal (biên) | 88,4% | 90,8% | 91,9% | 1,9% |
| **Conformal phân tầng theo vol** | **90,3%** | **90,8%** | **89,4%** | **0,8%** |

Điểm yếu là thật, và cách chữa cũng thật: phân tầng (Mondrian) kéo lệch tối đa
từ 1,9% xuống 0,8%.

## 5. Giới hạn đã đo — phải ghi trong luận văn

Cả ba phương pháp **phủ thiếu khi tài khoản đang lỗ**:

| Phương pháp | ở đỉnh vốn | đang lỗ |
|---|---|---|
| Gauss | 89,7% | 88,6% |
| Student-t | 88,8% | 87,8% |
| Conformal | 90,3% | 89,3% |

Đây đúng là lúc người dùng cần con số chính xác nhất. Nguyên nhân hợp lý: sụt
giảm tương quan với chế độ biến động đang chuyển, phần dư lịch sử chưa kịp
phản ánh. Hướng mở rộng: thêm trạng thái sụt giảm vào biến phân tầng Mondrian.
Phiếu quyết định in thẳng cảnh báo này thay vì giấu.

## 6. Chốt vào pipeline

* Khoảng dự báo: **đổi từ Student-t sang conformal phân tầng theo chế độ biến động**.
* P(chạm stop): giữ nguyên công thức phản xạ.
* Cài đặt: `src/decision_record.py` (`KhoangConformal`, `p_cham_stop`,
  `PhieuQuyetDinh`). Chạy `python src/decision_record.py` để tự kiểm — nó
  kiểm lại độ phủ trên tập giữ riêng mỗi lần chạy, không tin vào số chép tay.
