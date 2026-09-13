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

---

# PHỤ LỤC A — KẾT QUẢ (13/09/2026)

*Tái lập: `python src/ruiro_ml.py`. Kết quả: `output/ruiro_ml.json`.*

21.578 hàng · 6 cặp · 2012-02 → 2025-12 · huấn luyện 15.000 / kiểm định 3.282 /
kiểm tra 3.296.

**Đã cắt 1.016 hàng sau 2025-12-31.** `api.main.noi_chuoi` nối thêm dữ liệu live
tới 2026-09, nhưng `KHOA_SO.md` mục 2 **dành riêng 2026-01 → 2026-08 cho bộ niêm
phong**. Nguồn live khác `histdata_seal` về mặt tệp, nhưng **cùng giai đoạn** mà
niêm phong dành riêng, nên chấm trên đó sẽ làm xói mòn đúng thứ niêm phong tồn
tại để bảo vệ. Cắt đi. *(Lưu ý: `va_duoi.py` và `shadow_tohop.py` cũng dùng
`noi_chuoi`; xem mục A4.)*

## A1. Đích A — `P(chạm stop 1,5σ̂ trong 5 phiên)`, tần suất nền 33,64%

| mô hình | Brier kđ | BSS kđ | Brier **kt** | BSS kt | ECE kt |
|---|---|---|---|---|---|
| khí hậu học (nền) | 0,22572 | 0,0000 | 0,22097 | 0,0000 | **0,0069** |
| **mốc giải tích** *(đang chạy)* | 0,23744 | **−0,0519** | **0,23606** | **−0,0683** | **0,1230** |
| logistic | 0,22577 | −0,0002 | **0,22063** | **+0,0016** | 0,0113 |
| LightGBM | 0,22895 | −0,0143 | 0,22461 | −0,0165 | 0,0415 |

Theo đúng chữ của mục 7: ĐK1 ĐẠT (BSS +0,0016 > 0) và ĐK2 ĐẠT (0,22063 <
0,23606) → **"DƯƠNG"**.

### A1a. Nhưng không được đọc đó là một thắng lợi — ba lý do

1. **Độ lớn bằng không.** BSS **+0,0016** là cải thiện 0,16%. Mốc giải tích thua
   khí hậu học tới 6,8%; logistic chỉ nhúc nhích trên khí hậu học.
2. **Dấu đảo khi đổi 1.016 hàng.** Khi chưa cắt giai đoạn niêm phong, BSS kiểm
   tra là **−0,0002**; sau khi cắt thành **+0,0016**. Một kết quả đổi dấu vì
   1.016 hàng thì nằm trong nhiễu, không phải hiệu ứng.
3. **H13 bị bác bỏ ở biến chi phối.** Hệ số logistic (đã chuẩn hoá):

   | đặc trưng | hệ số | H13 dự kiến |
   |---|---|---|
   | **`log_sig`** | **−0,1157** | **DƯƠNG** → **bác bỏ** |
   | `che_do` | +0,0392 | — |
   | `abs_gap` | +0,0053 | dương, nhưng ≈ 0 |
   | `ti_nhay` | +0,0011 | dương, nhưng ≈ 0 |

   `log_sig` là hệ số **lớn nhất** và nó **âm**. Cơ chế thật dễ thấy sau khi biết:
   stop đặt ở **1,5·σ̂**, tức đã chuẩn hoá theo σ̂. σ̂ cao hôm nay thì, do biến
   động hoàn nguyên, thường **thổi phồng** biến động ngày mai — nên cái stop
   *theo tỉ lệ σ̂* lại tương đối **xa hơn**. Đúng ngược chiều đã chốt trước.

**Lỗi của chính văn bản này:** mục 7 viết ĐK1 là `BSS > 0` mà **không có ngưỡng
độ lớn tối thiểu**, dù mục 8 đã khai báo lực. Với một tiêu chí như thế, bất kỳ
cải thiện dù bằng nhiễu cũng "đạt". Đây là cùng loại lỗi đã khai báo ở
`DUOI_EVT.md` mục A5a. **Giữ nguyên phán quyết theo chữ viết, nhưng kết luận
thực hành là: không đưa vào sản xuất.**

## A2. PHÁT HIỆN DÙNG ĐƯỢC — mốc giải tích đang chạy bị sai hiệu chuẩn

Đây là thứ giá trị nhất của thí nghiệm, và nó không phải ML.

| | dự báo TB | thực tế | Brier | ECE |
|---|---|---|---|---|
| **`p_cham_stop`** *(đang hiện trên phiếu quyết định)* | **45,25%** | 33,64% | 0,23606 | **0,1230** |
| khí hậu học (hằng số = tần suất huấn luyện) | 33,64% | 33,64% | 0,22097 | 0,0069 |

**Mốc đang chạy vượt ước xác suất chạm stop 11,6 điểm phần trăm — tức cao hơn
thực tế khoảng 35% tương đối.** Nó **tệ hơn một hằng số**: Brier cao hơn 6,8%,
ECE cao hơn **18 lần**.

Nguyên nhân: `p_cham_stop` dùng **nguyên lý phản xạ** `P(min ≤ −b) = 2·P(X_T ≤ −b)`.
Đẳng thức đó đúng cho **bước đi đối xứng có gia số độc lập cùng phân phối**. Lợi
suất FX chuẩn hoá bằng σ̂ thì **không** độc lập cùng phân phối — có co cụm biến
động — nên hệ số 2,0 và phép chia `√h` đều thổi phồng. Đây là sai **đặc tả mô
hình**, không phải sai ước lượng, nên không sửa được bằng cách khớp lại.

| | |
|---|---|
| Có nên đổi? | **Có**, nhưng đây là **con số sản xuất** đang hiện trên giao diện → cần người chịu trách nhiệm quyết, không phải công cụ tự đổi. |
| Đổi thành gì thì an toàn nhất? | **Tần suất nền ước trên huấn luyện**, phân tầng theo cặp. Brier 0,236 → 0,221, ECE 0,123 → 0,007. Một tham số, không mô hình mới. |
| Có cần thí nghiệm mới? | **Không** — bảng A1 đã là phép so có đủ mốc. |

## A3. Đích B — `P(vi phạm VaR 1%)`: **ÂM**, và đúng như đã khai báo lực

| mô hình | Brier kt | BSS kt | ECE kt |
|---|---|---|---|
| khí hậu học / mốc danh nghĩa | 0,01170 | 0,0000 | 0,0018 |
| logistic | 0,01171 | −0,0011 | 0,0018 |
| LightGBM | 0,01191 | −0,0184 | 0,0055 |

ĐK1 và ĐK2 đều **TRƯỢT**. Đúng như mục 8 khai báo trước: với tần suất nền 1%,
đoạn kiểm tra chỉ có **~33 ca dương**, nên đây là phép thử **lực thấp**. Kết luận
chỉ phát biểu được là *"không phát hiện được phụ thuộc trạng thái trên ~33 ca
dương"*.

Đáng ghi: mốc danh nghĩa 0,01 **trùng khớp** khí hậu học tới 5 chữ số — tức
phân vị VaR ước trên huấn luyện đang hiệu chuẩn **đúng ở mức gộp**. Khiếm khuyết
VaR là ở **từng cặp** (USDJPY/USDCHF) và ở **động học**, không ở mức gộp — nhất
quán với `DUOI_EVT.md` A4.

## A4. Một việc tồn đọng phát hiện ra trong lúc làm

`va_duoi.py`, `va_duoi_evt.py` và `shadow_tohop.py` đều dùng `api.main.noi_chuoi`,
nên **đoạn "kiểm tra" của chúng có thể bao gồm dữ liệu live 2026** — giai đoạn mà
`KHOA_SO.md` mục 2 dành riêng cho bộ niêm phong. Script này đã cắt; **ba script
kia thì chưa kiểm**.

Việc cần làm (chưa làm): rà cả ba, xác định đoạn kiểm tra của chúng dừng ở đâu,
và nếu có chấm trên 2026 thì ghi rõ trong tài liệu tương ứng. Không phải lỗi
nghiêm trọng — nguồn live khác tệp niêm phong — nhưng nó làm mờ đúng cái ranh
giới mà biên bản khoá sổ dựng lên.
