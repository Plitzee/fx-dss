# TRỌNG SỐ NGÀY TRONG TỔ HỢP SẢN XUẤT — tiêu chí CHỐT TRƯỚC

Lập 14/09/2026. Nhánh `replan-2026`. **Commit chứa văn bản này nằm trước mọi
commit có số liệu.**

---

## 1. Vì sao đây là bước tiếp theo hợp lý — bằng chứng loại trừ

Hai thí nghiệm liên tiếp đêm qua/tối nay đã thử **giải thích** khoảng cách
~7–8% QLIKE mà `docs/BCL_TIEUCHI.md` phát hiện (thêm `log rv5(t)` thô vào mốc
cho −7,78%), bằng cách tách **thành phần nhảy**:

| thí nghiệm | cách tách nhảy | kết quả |
|---|---|---|
| BCL (`kiem_bcl.py`) | ngưỡng đếm thô, lọc chu kỳ nội tuần | **không đóng được khoảng cách** — C/J thô sụp khi có mốc đã sửa |
| HAR-J (`kiem_harj.py`) | bipower thật (Andersen–Bollerslev–Diebold 2007) | **không cải thiện tổ hợp**, dù dấu hệ số đúng lý thuyết 6/6 cặp |

**Bằng phép loại trừ**: đây nhiều khả năng không phải "thiếu thông tin nhảy",
mà là **cách tổ hợp đang định trọng số ngày thấp hơn mức tối ưu**. Ba mô hình
con định trọng số ngày theo ba cách khác nhau:

```python
X["STHARQ"] = [1, lv, lw, lm, lv·G, lw·G, lm·G, lq, lq·lv]   # smooth-transition
X["HARQ"]   = [1, lv, lw, lm, lq, lq·lv]                      # giảm theo quarticity
X["SHAR"]   = [1, lp, ln_, lw, lm]                             # KHÔNG dùng lv — dùng semivariance
```

rồi **trung bình cộng đơn giản** ba log-dự báo (`g = L.mean(0)`,
`volfc2.du_bao_san_xuat`). SHAR không hề dùng `lv` trực tiếp; HARQ giảm hệ số
ngày khi RV nhiễu (đúng thiết kế, nhưng có thể giảm quá tay cho dữ liệu 5 phút
FX); STHARQ nhân thêm hệ số chuyển tiếp. Ba cách khác nhau, cộng đều — chưa
ai kiểm xem trọng số kết quả có tối ưu không.

## 2. Giả thuyết CHỐT TRƯỚC

> **H15.** Tổ hợp sản xuất đang **định trọng số dưới tối ưu** cho thông tin
> RV ngày gần nhất. Sửa bằng (a) thêm một mô hình HAR cổ điển thuần
> (Corsi 2009, không quarticity/không bất đối xứng) làm thành viên thứ tư,
> và/hoặc (b) thay trung bình đều bằng **trọng số thích ứng theo tổn thất quá
> khứ** (Hedge — cùng cơ chế đã dùng cho tầng xác suất, `balop.ToHopTrucTuyen`,
> BSS +0,0076 ở đó) — sẽ cải thiện QLIKE so với mốc.

## 3. Ba cấu hình mới — liệt kê đầy đủ TRƯỚC khi chạy

Mô hình mới `LV` — HAR cổ điển thuần (Corsi 2009), **đúng bằng thiết kế của
HARQ trừ hai số hạng quarticity**:

```python
X["LV"] = [1, lv, lw, lm]
```

| | thành viên | trọng số |
|---|---|---|
| **B0** mốc *(không đổi)* | STHARQ, HARQ, SHAR | đều (1/3 mỗi) |
| **B0+LV** | STHARQ, HARQ, SHAR, LV | đều (1/4 mỗi) |
| **B0 Hedge** | STHARQ, HARQ, SHAR | Hedge — tổn thất QLIKE quá khứ |
| **B0+LV Hedge** | STHARQ, HARQ, SHAR, LV | Hedge — tổn thất QLIKE quá khứ |

**Đếm vào `KHOA_SO.md`: 3 cấu hình mới.** Không thêm biến thể nào ngoài bảng
này (không thử η khác, không thử trọng số nghịch-đảo-QLIKE kiểu khác).

### 3a. Cơ chế Hedge — chốt tham số trước

Trọng số mũ trên tổn thất QLIKE, **cùng η = 0,5** đã dùng và đã đo tốt cho
tầng xác suất (`balop.ToHopTrucTuyen`, mặc định `eta=0.5`):

```
w_i(t+1) ∝ w_i(t) · exp(−η · QLIKE_i(t))     rồi chuẩn hoá tổng = 1
```

Cập nhật dùng tổn thất QLIKE của **chính phiên t vừa biết kết cục** (không
nhìn tương lai — đúng nguyên tắc "hôm nay sai thì mai chỉnh" đã ghi trong
`balop.py`). Khởi tạo trọng số đều `1/k`.

## 4. Giao thức chấm

Tái tạo đúng cơ chế `du_bao_san_xuat()` (như `kiem_harj.py` đêm qua): mỗi mô
hình con khớp OLS **cửa sổ mở rộng** (`he_so_cuon`), dự báo log-phương sai
riêng; sau đó tổ hợp theo cách của từng cấu hình (đều hoặc Hedge). Mọi tham
số khác **giữ nguyên `CAUHINH_SANXUAT`**. QLIKE trên kiểm định rồi kiểm tra,
DM + Newey–West so B0, theo từng cặp.

## 5. Chống rò rỉ

`LV` chỉ dùng `lv/lw/lm` đã có sẵn trong `thiet_ke()`, không thêm dữ liệu
mới. Trọng số Hedge cập nhật bằng tổn thất của phiên **đã qua**, dùng để tổ
hợp dự báo cho phiên **kế tiếp** — tự kiểm bắt buộc: trọng số tại thời điểm
dùng để tổ hợp dự báo cho ngày *t+1* chỉ được phép phụ thuộc tổn thất tại các
ngày ≤ *t*.

## 6. TIÊU CHÍ PHỦ ĐỊNH

Kết luận **"trọng số hiện tại đã đủ tốt, không sửa"** khi bất kỳ điều nào
đúng:

1. Cấu hình tốt nhất không thắng B0 trên **kiểm tra**, hoặc thắng nhưng DM
   p ≥ 0,05.
2. Hoặc không đạt ≥ 5/6 cặp cải thiện.

Kết luận **dương** đòi **cả hai**: thắng kiểm tra với DM p < 0,0167
(Bonferroni 3 cấu hình) **và** ≥ 5/6 cặp. Không cần điều kiện dấu hệ số ở
đây (khác EVT/HAR-J) vì đây không phải giả thuyết về cơ chế mới mà là **hiệu
chỉnh trọng số** của cùng một tập thông tin đã biết là có ích.

Không sửa tiêu chí sau khi thấy số.

## 7. Lực phát hiện — khai báo TRƯỚC

Cùng bảng D1, ~3.282 hàng mỗi đoạn ngoài mẫu như mọi ablation tầng 2. Ngưỡng
phát hiện thực tế đã đo ở khung này là **khoảng 0,5–0,7% QLIKE**
(`|gap|` đạt p=0,0445 với 0,67%). Nếu bằng chứng loại trừ ở mục 1 đúng, kỳ
vọng ở đây lớn hơn nhiều mức đó — vì đích nhắm chính là khoảng cách 7,78% đã
đo trực tiếp, không phải một đặc trưng phụ nhỏ.

## 8. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 2 (giả thuyết), mục 6
(tiêu chí phủ định) và mục 7 (khai báo lực) **trước khi** số liệu được chấm:

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không
> phải công cụ điền hộ.

---

# PHỤ LỤC A — KẾT QUẢ (14/09/2026, chạy SAU khi mục 1–8 đã chốt)

*Tái lập: `python src/kiem_trongso.py`. Kết quả: `output/trongso.json`.*

| cấu hình | QLIKE kiểm định | QLIKE kiểm tra | so B0 | DM p | cặp cải thiện |
|---|---|---|---|---|---|
| B0 mốc | 0,1162 | 0,1585 | — | — | — |
| B0+LV | 0,1167 | 0,1588 | +0,17% | 0,441 | 2/6 |
| **B0 Hedge** *(tốt nhất trên kiểm định)* | 0,1164 | **0,1581** | **−0,25%** | 0,551 | 4/6 |
| B0+LV Hedge | 0,1165 | 0,1581 | −0,23% | 0,583 | 4/6 |

## Phán quyết — **ÂM**

| điều kiện | kết quả | |
|---|---|---|
| ĐK1 thắng kiểm tra, p<0,0167 | **TRƯỢT** | −0,25%, p=0,551 |
| ĐK2 ≥5/6 cặp | **TRƯỢT** | 4/6 |

## Điều quan trọng nhất: giả thuyết loại trừ thứ ba cũng thất bại — phải xét lại chính chẩn đoán ban đầu

Không cấu hình nào tách khỏi nhiễu, và quan trọng hơn: **không cấu hình nào
tiệm cận mức 7,78%** mà chẩn đoán ở `BCL_TIEUCHI.md` mục A4 đã đo được khi
thêm `log rv5(t)` thô vào mốc. Mức tốt nhất ở đây chỉ **−0,25%**.

Đây là lần **thứ ba liên tiếp** một giả thuyết nhắm đúng vào khoảng cách đó
thất bại:

| giả thuyết | kết quả |
|---|---|
| Thiếu tách nhảy bằng ngưỡng (BCL) | không đóng được khoảng cách |
| Thiếu tách nhảy bằng bipower thật (HAR-J) | không cải thiện, dù cơ chế đúng |
| **Trọng số tổ hợp dưới tối ưu (ở đây)** | **không cải thiện, dù đổi cả thành viên lẫn cách tổ hợp** |

**Phải đọc lại đúng phát hiện ban đầu.** Ba giả thuyết đều nhắm vào việc
*sửa cách tổ hợp ra dự báo σ̂*, và cả ba đều thất bại khi kiểm trong đúng cơ
chế tổ hợp thật (`du_bao_san_xuat`). Trong khi đó, con số 7,78% ở
`BCL_TIEUCHI.md` được đo trong **một khung khác hẳn**: hồi quy MỘT phương
trình `log_rv(t+1) ~ log(h_HAR đã hiệu chỉnh log-chuẩn) + log_rv(t)` — tức
`log(h_HAR)` ở đó là **đầu ra cuối cùng đã qua hiệu chỉnh log-normal**
(`exp(fit + 0,5·s²)`), không phải trung bình log-dự báo thô của ba mô hình
con. Khả năng cao nhất: `log_rv(t)` trong khung BCL đang sửa cho **sai lệch
của chính phép hiệu chỉnh log-chuẩn** (hoặc một hiệu ứng thuộc về khung hồi
quy một phương trình đó), **không phải** một khoảng trống thông tin trong
cách ba mô hình con định trọng số ngày.

## Kết luận cho luận văn

> *Ba hướng độc lập nhắm vào khoảng cách QLIKE ~7,78% được chẩn đoán khi làm
> đối chứng cho một thí nghiệm khác (tách nhảy bằng ngưỡng, tách nhảy bằng
> bipower, và trọng số tổ hợp thích ứng) đều **không tái tạo được cải thiện
> đó** khi kiểm trong đúng cơ chế tổ hợp sản xuất thật. Kết luận thận trọng:
> con số 7,78% phản ánh đặc thù của khung so sánh dùng để chẩn đoán nó (hồi
> quy một phương trình trên đầu ra đã hiệu chỉnh log-chuẩn), không phải một
> khiếm khuyết có thể sửa được trong cách tổ hợp ba mô hình con hiện tại. Nên
> rút lại phát biểu "tầng 2 để lại ~8-10% QLIKE trên bàn" cho tới khi có bằng
> chứng trực tiếp hơn.*

Việc còn treo (chưa làm, ngoài phạm vi ba thí nghiệm này): kiểm trực tiếp
xem chính phép hiệu chỉnh log-chuẩn (`s2_tu_huan_luyen`/hiệu chỉnh
`exp(fit+0,5s²)`) có thiên lệch hay không — đó mới là biến khác duy nhất
giữa khung BCL và khung tổ hợp thật chưa được cô lập.
