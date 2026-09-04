# Đồng bộ sản xuất — vòng 7

*01/09/2026. Bước chuẩn bị trước khi làm tầng 7 (giao diện) và ống dẫn dữ liệu hằng ngày.*

Trước bước này, cấu hình thắng vòng 7 mới chỉ tồn tại trong `volfc2.py` như một
điểm trong lưới thí nghiệm; panel rủi ro và toàn bộ tầng 3–6 vẫn chạy trên dự báo
cũ. Nghĩa là báo cáo nói một đằng còn hệ thống chạy một nẻo. Bước này khép lại
khoảng cách đó.

## 1. Điểm vào sản xuất

`volfc2.du_bao_san_xuat(d, pair)` — **một hàm, một cặp, không phụ thuộc cặp khác.**

```python
from volfc  import merge_thin_days
from volfc2 import du_bao_san_xuat, CAUHINH_SANXUAT

d = merge_thin_days(df_mot_cap)      # Date, OHLC, rv5, rq5, bpv5, rsp, rsn, n5
f = du_bao_san_xuat(d, "EURUSD")     # mảng phương sai dự báo, f[i] là dự báo CHO ngày i
```

Cấu hình đóng băng trong `CAUHINH_SANXUAT`:

```
deseason  = none      khử mùa vụ theo thứ không còn tác dụng khi đã có lịch
crosspair = off       thông tin chéo cặp đã nằm trong lịch sử của chính cặp
event     = capday    lịch NHTW RIÊNG từng cặp + FOMC + ngày kế tiếp + NFP + cuối tháng
window    = exp       cửa sổ mở rộng
recal     = off       hiệu chuẩn Mincer–Zarnowitz không giúp
lam       = 0.0       ước lượng riêng từng cặp, không co ngót về panel
```

**Hai tính chất khiến nó hợp với ống dẫn hằng ngày** — và cả hai là hệ quả tình cờ
của việc cấu hình thắng lại là cấu hình đơn giản nhất:

- `lam = 0` và `crosspair = off` ⟹ **mỗi cặp độc lập hoàn toàn.** Cập nhật EURUSD
  không cần dữ liệu của USDJPY. Ống dẫn có thể xử lý từng cặp, thất bại từng cặp,
  chạy lại từng cặp.
- Biến lịch là của ngày *t+1* và biết trước nhiều năm ⟹ **chạy được TRƯỚC khi
  phiên t+1 mở cửa.** Không phải chờ tới hết ngày mới có dự báo cho ngày đó.

Đã kiểm chứng hàm này trùng khớp engine lưới tới sai số **0** trên 6/6 cặp
(`python src/volfc2.py`), và bài kiểm tra rò rỉ nhìn trước vẫn đạt.

## 2. Panel đã dựng lại

`data/panel2_6pairs.csv` sinh lại bằng dự báo mới. Bản cũ giữ ở
`data/panel2_v6_6pairs.csv`.

Mức thay đổi của `sig` trên 21.596 hàng chung: **trung vị 2,6%, p95 18,4%, tối đa 88,8%.**
Đủ lớn để mọi kết luận của tầng 3–6 phải chạy lại — và đã chạy lại.

## 3. Tầng 3–6 sau khi chạy lại

| Kiểm tra | Kết quả |
|---|---|
| `split.py` | 2.503 / 547 / 549 phiên mỗi cặp — không đổi |
| `position_sizing.py` (tầng 4) | ĐẠT. Ngưỡng biến động 0,00407 / 0,00518; đòn bẩy 9,07× ở đỉnh → 3,49× khi sụt giảm 30%; hệ số danh mục 1→1,00, 6→0,23 |
| `decision_record.py` (tầng 6) | ĐẠT. Độ phủ 78,7 / 89,4 / 94,5% ở mức danh nghĩa 80 / 90 / 95% |
| `run_final_eval2.py` | Bảng chạm stop lệch trung bình **0,94%** trên kiểm tra; hệ số danh mục giữ phá sản ở 0,27% cho 6 cặp so với **63,77%** nếu không áp |
| `run_scores.py` | Kupiec / Christoffersen / DQ **đạt 6/6** ở cả hai mức VaR 1% và 5%. Mincer–Zarnowitz: b = 0,944, R²_log = 0,439 (cũ: 0,704 và 0,248) |

## 4. Một thay đổi thật ở tầng 6 — và một cảnh báo

Khi σ̂ tốt lên thì **lựa chọn phương pháp khoảng cũng đổi.** Chạy lại
`run_final_eval2.py` trên panel mới:

| phương pháp | lệch max (kiểm định) | lệch max (kiểm tra) |
|---|---|---|
| **ACI phân tầng 3** | **2,0%** ← chọn | 1,5% |
| Mondrian 3 | 2,9% | 1,3% |
| ACI phân tầng 2 | 3,1% | **1,1%** |
| tĩnh / Mondrian 2 / ACI chung | 4,0% | 1,5–2,0% |

Với panel vòng 6, bản **tĩnh Mondrian 2** thắng. Với panel vòng 7, **ACI phân tầng
3** thắng. Mặc định sản xuất đã đổi theo, qua một bộ tạo duy nhất
`decision_record.khoang_mac_dinh()`.

**Cảnh báo phải ghi vào luận văn:** lựa chọn này *không ổn định*, nó bám theo chất
lượng của σ̂. Khoảng cách giữa sáu phương pháp trên đoạn kiểm tra chỉ 1,1–2,0%, nên
đây không phải lựa chọn lớn — nhưng nó phải được **chọn trên kiểm định**, không
được đặt tay, và phải chạy lại mỗi lần tầng 2 đổi. Giá phải trả cho việc chọn lần
này là **+0,4 điểm phần trăm** (1,5% so với 1,1% tốt nhất có thể).

## 5. Còn lại trước khi có ống dẫn hằng ngày

- **Nguồn dữ liệu hằng ngày.** HistData phát hành theo **năm** (`1 request = 1 năm`),
  không dùng cho cập nhật hằng ngày được. Phải viết bộ tải Dukascopy `.bi5` theo giờ
  cho ngày hôm trước. Chưa có mã.
- **Lịch ngân hàng trung ương hết hạn cuối 2026.** `data/cb_dates.csv` phủ tới
  2026-12. Cần một việc định kỳ lấy lịch năm sau — các ngân hàng đều công bố trước
  ~1 năm.
- **Sổ dự báo.** Chưa có nơi lưu dự báo hôm nay để hôm sau chấm điểm. Đây là thứ
  biến hệ thống từ script chạy tự động thành DSS thật: QLIKE trượt, tỷ lệ vi phạm
  trượt, PIT trượt, và phiếu quyết định tự hạ độ tin cậy khi hiệu chuẩn trôi.
- **Bài kiểm tra rò rỉ phải chạy ở MỖI lần triển khai.** Hệ thống tự khớp lại hằng
  ngày là đúng chỗ rò rỉ nhìn trước và trôi ngầm chui vào.
- **Tập khoá sổ tuyệt đối không được nằm trong đường đi của ống dẫn hằng ngày.**
