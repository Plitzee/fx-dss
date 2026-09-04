# Đánh giá cuối — chia huấn luyện / kiểm định / kiểm tra

Lập 29/08/2026. File này sửa một khiếm khuyết **phương pháp luận** của mọi kết
quả trước đó trong dự án.

## 1. Vấn đề: cho tới vòng này, chọn và chấm trên cùng một tập

Toàn dự án chỉ có 70/30. Nghĩa là mọi lựa chọn — chọn mô hình biến động trong
14 ứng viên, chọn số tầng của conformal, chọn `K_SLIP`, chọn dạng hệ số danh
mục — đều được chấm điểm trên đúng cái tập 30% mà sau đó lại dùng để báo cáo.

Đó là **rò rỉ lựa chọn**. Con số báo cáo sẽ lạc quan hơn sự thật, và không ai
biết lạc quan bao nhiêu. Với một luận văn thì đây là chỗ phản biện sẽ đánh, và
đánh trúng.

## 2. Cách chia mới (`src/split.py`)

Chia theo **ngày**, không theo chỉ số, để mọi cặp cùng một mốc:

| Đoạn | Từ | Đến | Phiên/cặp | Tổng |
|---|---|---|---|---|
| huấn luyện | 2012-02-14 | 2020-06-12 | 2.160 | 12.960 |
| kiểm định | 2020-06-15 | 2023-03-23 | 720 | 4.320 |
| kiểm tra | 2023-03-24 | 2025-12-31 | 719 | 4.316 |

Luật: **mọi lựa chọn chỉ được nhìn đoạn kiểm định. Đoạn kiểm tra chỉ được chấm
điểm.** Tập khóa sổ (6 cặp chéo + toàn bộ 2026) vẫn nằm ngoài, mở ở P3 — đây là
lớp bảo vệ thứ hai, không thay thế lớp thứ nhất.

## 3. Tầng 2 — chọn trên kiểm định, chấm trên kiểm tra

| Mô hình | QLIKE kiểm định | hạng | QLIKE kiểm tra | hạng |
|---|---|---|---|---|
| **STHARQ** | **0,1086** | **1** | **0,1587** | **1** |
| EN(STHARQ,HARQ,SHAR) | 0,1086 | 2 | 0,1593 | 2 |
| EN(tất cả) | 0,1093 | 3 | 0,1597 | 3 |
| SHAR | 0,1099 | 4 | 0,1615 | 5 |
| HARQ | 0,1100 | 5 | 0,1615 | 6 |
| HAR | 0,1107 | 6 | 0,1621 | 9 |
| STHAR | 0,1112 | 8 | 0,1609 | 4 |
| MA5-RV5 | 0,1227 | 10 | 0,1797 | 10 |
| MA20-GK | 0,1460 | 11 | 0,1977 | 11 |

Kiểm định chọn **STHARQ**, và STHARQ cũng đứng nhất trên kiểm tra — **giá phải
trả cho việc chọn mô hình là 0,0%**. Lựa chọn không phải may mắn.

**Con số headline phải sửa.** Trước đây README ghi cải thiện **24%** so với
MA20-GK. Con số đó đo trên tập đã dùng để chọn. Trên đoạn kiểm tra sạch, cải
thiện là **19,7%** (0,1587 so với 0,1977). Vẫn lớn, vẫn có ý nghĩa, nhưng thấp
hơn 4,3 điểm — và đó chính là phần lạc quan mà rò rỉ lựa chọn tạo ra.

Diebold–Mariano trên đoạn kiểm tra, so với MA20-GK: STHARQ thắng **6/6 cặp ở
p<0,05** (t từ −2,35 đến −2,92). Model Confidence Set trên kiểm tra giữ STHARQ,
cả hai tổ hợp, và HAR-CJ ở 6/6.

**Bản đang chạy là tổ hợp, không phải STHARQ đơn lẻ.** Trên kiểm tra tổ hợp cho
0,1593 so với 0,1587 — đắt hơn **0,4%**. Giữ tổ hợp vì lý do đã tuyên bố trước
khi mở tập kiểm tra: DM không phân biệt được hai cái, và tổ hợp không đặt cược
vào một dạng hàm cụ thể trước khi tập khóa sổ được mở. Nếu hội đồng muốn con số
tốt nhất thì đổi sang STHARQ đơn lẻ, chênh 0,4%.

## 4. Tầng 6 — khoảng dự báo

Chọn theo tiêu chí **|lệch| tối đa** trên năm trạng thái (chung, vol thấp, vol
cao, ở đỉnh vốn, đang lỗ).

Đoạn kiểm định:

| Phương pháp | phủ chung | vol thấp | vol cao | ở đỉnh | đang lỗ | \|lệch\| max | điểm khoảng |
|---|---|---|---|---|---|---|---|
| **Mondrian 2** | 90,1% | 89,7% | 89,9% | 90,0% | 90,2% | **0,3%** | **281,3** |
| Mondrian 3 | 90,3% | 89,9% | 89,8% | 90,1% | 90,5% | 0,5% | 281,1 |
| ACI-tầng 3 | 89,5% | 89,2% | 89,8% | 89,6% | 89,4% | 0,8% | 286,5 |
| tĩnh | 90,5% | 89,7% | 90,8% | 90,3% | 90,7% | 0,8% | 280,2 |
| ACI-tầng 2 | 89,5% | 88,2% | 90,3% | 89,5% | 89,6% | 1,8% | 283,2 |
| ACI | 89,8% | 87,7% | 91,8% | 89,7% | 89,9% | 2,3% | 283,9 |

Đoạn kiểm tra:

| Phương pháp | phủ chung | vol thấp | vol cao | ở đỉnh | đang lỗ | \|lệch\| max | điểm khoảng |
|---|---|---|---|---|---|---|---|
| ACI-tầng 3 | 91,0% | 91,1% | 91,1% | 91,2% | 90,7% | 1,2% | 257,2 |
| ACI-tầng 2 | 90,8% | 90,0% | 90,7% | 91,4% | 90,2% | 1,4% | 257,2 |
| **Mondrian 2 (được chọn)** | 90,7% | 88,7% | 91,5% | 91,2% | 90,2% | **1,5%** | 259,0 |
| tĩnh | 90,3% | 88,1% | 91,5% | 90,6% | 89,9% | 1,9% | 260,6 |
| ACI | 90,5% | 87,8% | 92,9% | 90,9% | 90,0% | 2,9% | 258,7 |

**Conformal thích ứng (ACI) KHÔNG tự chứng minh được.** Vòng trước tôi kết luận
ACI-tầng thắng, nhưng kết luận đó đo trên đúng tập rồi dùng để báo cáo. Dưới
quy trình sạch, kiểm định xếp ACI-tầng 2 thứ **năm** trên cả hai tiêu chí, và
chọn Mondrian 2 tĩnh. Trên kiểm tra ACI-tầng có nhỉnh hơn (1,2–1,4% so với
1,5%) nhưng chênh lệch giữa năm phương án đầu là 1,2–1,9% — trong nhiễu, và
chọn theo đoạn kiểm tra chính là thứ quy trình này cấm.

Bản đang chạy vẫn là **Mondrian 2** (`KhoangConformal` mặc định `n_bins=2`),
tức trùng với lựa chọn của kiểm định. `KhoangACI` giữ trong mã như một phương
án đã thử, không dùng làm mặc định.

Giá phải trả cho việc chọn: **+0,3 điểm phần trăm** |lệch| max.

## 5. Các con số rủi ro trên đoạn kiểm tra

**Xác suất chạm stop** (tham số ước lượng trên huấn luyện, chấm trên kiểm tra,
n = 4.316 mỗi ngưỡng):

| Ngưỡng | Dự báo | Thực tế | Lệch |
|---|---|---|---|
| 0,5σ | 57,2% | 55,4% | +1,8% |
| 1,0σ | 27,2% | 27,3% | −0,1% |
| 1,5σ | 11,4% | 12,2% | −0,8% |
| 2,0σ | 4,6% | 5,0% | −0,4% |
| 2,5σ | 1,8% | 2,2% | −0,4% |
| 3,0σ | 0,8% | 1,0% | −0,2% |

Lệch tuyệt đối trung bình **0,62%** — tốt hơn con số 1,44% báo trước đây.

**Bảng tầm hạn** (hệ số hiệu chỉnh ước lượng trên huấn luyện, stop 2σ cố định):

| Tầm hạn | Dự báo | Thực tế | Lệch |
|---|---|---|---|
| 1 phiên | 4,9% | 5,2% | −0,2% |
| 5 phiên | 33,2% | 34,5% | −1,3% |
| 10 phiên | 49,2% | 50,4% | −1,2% |
| 20 phiên | 62,3% | 62,1% | +0,3% |

**Hệ số danh mục** (mô phỏng trên đoạn kiểm tra, ngân sách phá sản 1%):

| Số cặp | k_danh_mục | Phá sản khi áp | Nếu không áp |
|---|---|---|---|
| 1 | 1,00 | 0,22% | 0,22% |
| 2 | 0,59 | 0,28% | 8,72% |
| 3 | 0,42 | 0,30% | 23,75% |
| 6 | 0,23 | 0,50% | **66,68%** |

Luật `k_danh_mục` giữ mọi cấu hình dưới ngân sách trên dữ liệu chưa từng dùng
để hiệu chỉnh nó.

## 6. Tổng kết — con số nào là con số thật

| Chỉ số | Báo trước đây | Trên đoạn kiểm tra sạch |
|---|---|---|
| QLIKE tốt hơn MA20-GK | 24% | **19,7%** |
| Diebold–Mariano | 6/6 p<0,05 | 6/6 p<0,05 |
| \|lệch\| hiệu chuẩn khoảng | 1,2% | **1,5%** |
| Lệch P(chạm stop) | 1,44% | **0,62%** |
| Độ phủ khoảng ở mức 90% | 89,6% | **90,7%** |

Ba con số xấu đi, hai con số tốt lên. Cái xấu đi là những cái tôi đã **chọn**
trên tập chấm điểm; cái tốt lên là những cái không có gì để chọn (công thức
phản xạ không có siêu tham số). Đó đúng là dấu vân tay của rò rỉ lựa chọn, và
giờ nó đã được đo chứ không còn ẩn.

## 7. Tái lập

```bash
python src/split.py             # tự kiểm cách chia
python src/run_final_eval.py    # tầng 2: chọn trên kiểm định, chấm trên kiểm tra (~40s)
python src/run_final_eval2.py   # tầng 4 và 6 (~3 phút)
```
