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

## 7. Vá 03/09/2026 — ba lỗ hổng nghiêm trọng nhất từ đợt soát toàn hệ thống

Trước khi lên tầng 7 (UI), toàn bộ pipeline được soát lại một lượt tìm chỗ
kém hiệu quả / phase này hại phase kia. Ba lỗ hổng nghiêm trọng nhất, đều nằm
ở chính **phiếu quyết định** — thứ người dùng thật sự đọc — được vá cùng lúc.

### 7.1. `mu` (lợi thế Kelly) trong tự kiểm là hằng số **bịa**

Trước bản vá, mọi lệnh gọi `PhieuQuyetDinh.lap(...)` trong tự kiểm truyền
`mu=0,0002` — một con số không đến từ đâu cả. Tầng 1 (dự báo hướng đi) đã bị
bác bỏ (`E[zT]=0`), nên **không có cơ sở nào** cho một `mu` hướng dương cố
định. Lợi thế Kelly hợp lệ **duy nhất** trong toàn hệ thống là carry (xem
`optimal_stop.py`, và mục "khớp nối đòn bẩy" trong
`docs/TANG6B_DUNGTOIUU.md`) — thứ `compare_leverage_dp.py` và `run_e2e.py`
đã dùng đúng, nhưng `decision_record.py` (nơi ra phiếu thật) thì chưa.

Vá: `mu` giờ tính từ `optimal_stop.carry_ngay()`, trung vị trên cửa sổ mở
rộng (đúng quy ước dùng xuyên suốt repo). Chạy tự kiểm với dữ liệu EURUSD
hiện có, carry huấn luyện đo được là **−0,257 bp/phiên** — ÂM, không phải một
con số bịa dương cố định.

### 7.2. Ngân sách phá sản mặc định 3%, trong khi mọi nơi khác ghi 1%

`PositionSizer.__init__` mặc định `budget=0,03` từ commit đầu tiên của file.
Không nơi nào trong repo truyền `budget=` để ghi đè — kể cả chính phiếu quyết
định, tự in ra chữ "ngân sách 1%" trong phần "ĐIỀU KIỆN ĐỂ CON SỐ ĐÒN BẨY CÒN
ĐÚNG". `docs/TANG4_DANHMUC.md` cũng ghi rõ ngân sách đặt ra là 1%. Tức là suốt
từ đầu, mọi đòn bẩy tính ra trong sản xuất đều **âm thầm chạy ở ngân sách gấp
3 lần** con số đã công bố.

Vá: đổi mặc định về `budget=0,01`, khớp `f_ruin_cap()` (đã đúng 0,01 từ đầu)
và mọi tài liệu đã công bố.

**Đã kiểm tra ảnh hưởng ngược dòng:** chạy lại `compare_leverage_dp.py` và
`run_e2e.py` (hai kịch bản duy nhất khác gọi `PositionSizer()` không ghi đè
budget) sau khi đổi mặc định — **kết quả không đổi một số nào** so với bản đã
ghi trong `docs/TANG6B_DUNGTOIUU.md` và `docs/TOANMACH_E2E.md`. Lý do: đòn bẩy
carry-Kelly trong mọi trường hợp đã thử đều nhỏ hơn cả hai mức trần rủi ro (1%
lẫn 3%) — bị Kelly chặn trước khi chạm trần, nên đổi trần không đổi kết quả.
Đây chính là câu đã in sẵn trong `compare_leverage_dp.py`: "đòn bẩy carry-Kelly
quá nhỏ để đổi kết luận". Hai tài liệu đó **không cần sửa số**.

### 7.3. Tầng 6b (giữ/đóng) giải xong nhưng chưa bao giờ lên phiếu

`optimal_stop.py` giải quy hoạch động đầy đủ, đã kiểm định (xem
`docs/TANG6B_DUNGTOIUU.md`), nhưng `decision_record.py` trước bản vá **không
import** `optimal_stop` — người dùng đọc phiếu không có cách nào biết hệ
thống có khuyến nghị giữ hay đóng.

Vá: thêm lớp `KhuyenNghiGiuDong` — giải DP một lần (bản **không đòn bẩy**,
đúng chính sách đang sản xuất, xem giới hạn ở `TANG6B_DUNGTOIUU.md`) cho một
carry cụ thể, rồi cho `PhieuQuyetDinh` một tham số `tang6b` tùy chọn. Phiếu
giờ in thêm mục "TẦNG 6b — GIỮ HAY ĐÓNG": khuyến nghị lúc vừa vào lệnh, và
biên giới đóng lệnh theo σ ở chế độ biến động hiện tại.

### 7.4. Lỗi phát sinh khi vá 7.1: đòn bẩy ÂM khi carry âm

Bỏ `mu` bịa (luôn dương) làm lộ một lỗi có sẵn: `PositionSizer.size()` chặn
kết quả về `[0, lev_max]`, nhưng `PositionSizer.explain()` (thứ phiếu quyết
định dùng) thì **không** — khi Kelly âm (carry âm), `explain()` trả đòn bẩy
âm thẳng, và vì Kelly âm luôn nhỏ hơn trần rủi ro dương nên `min(kelly, trần)`
luôn chọn Kelly — **trần rủi ro và hệ số sụt giảm mất tác dụng hoàn toàn**.
Với carry EURUSD huấn luyện hiện âm, đây không phải kịch bản giả định — phiếu
thật sự sẽ in "đòn bẩy khuyến nghị −1,84×" nếu không vá.

Vá: `explain()` giờ chặn `f` về `[0, lev_max]` giống hệt `size()`, và nhãn
ràng buộc thêm nhánh "không có lợi thế (Kelly ≤ 0)". Phiếu giờ đúng đắn in
đòn bẩy 0× khi không có lợi thế, thay vì một số âm vô nghĩa.

### Tái lập

```bash
python src/position_sizing.py     # ngân sách 1%, tự kiểm ĐẠT
python src/decision_record.py     # mu = carry thật, mục TẦNG 6b, tự kiểm ĐẠT
```
