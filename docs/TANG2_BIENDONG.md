# Tầng 2 — Dự báo biến động: chẩn đoán, so sánh, và mô hình được chọn

Lập 29/08/2026. Thay thế kết luận cũ trong README ("MA20-GK và GARCH(1,1)-t
ngang nhau, cùng thắng mọi biến thể HAR ở p<0,01"). Kết luận cũ **sai**, và
mục 1 nói rõ sai vì đâu.

## 1. Lỗi trong so sánh cũ — phiên Chủ nhật nằm trong chuỗi hồi quy

Thị trường FX mở lại lúc 22:00 UTC Chủ nhật. Dữ liệu HistData cắt theo ngày
lịch, nên phiên đó thành một "ngày" riêng dài khoảng 2 giờ:

| | ngày giao dịch đầy đủ | phiên Chủ nhật |
|---|---|---|
| số thanh 5 phút | 287 (trung vị) | 23 |
| realized variance | 2,14e−05 (trung vị) | 8,77e−07 |
| số ngày (mỗi cặp) | ~4.150 | ~843 (**17%**) |

Phương sai của phiên Chủ nhật nhỏ hơn **24 lần**. Hai hệ quả:

**(a) Điểm QLIKE bị thổi phồng.** Mọi mô hình đều dự báo cho phiên Chủ nhật
bằng mức của ngày thường, nên bị phạt khoảng 2,0 QLIKE trên mỗi phiên đó.
QLIKE của mô hình tốt nhất là 0,51 khi tính cả Chủ nhật, nhưng chỉ 0,18 khi
chỉ tính ngày đầy đủ.

**(b) Và đây mới là lỗi thật: HAR sụp đổ.** MA20 lấy trung bình 20 ngày nên
vài phiên Chủ nhật bị pha loãng. HAR thì hồi quy trực tiếp trên log của chuỗi
đó — chuỗi bị lưỡng thức, và khi dự báo thứ Hai từ giá trị Chủ nhật bé tí, số
hạng trễ ngày kéo dự báo xuống rất thấp, đúng vào chỗ QLIKE phạt nặng nhất.

Đo trực tiếp giả thuyết này (QLIKE, chỉ chấm trên ngày đầy đủ, trung bình 6 cặp):

| | QLIKE |
|---|---|
| HAR như cũ (chuỗi có Chủ nhật) | 0,4616 |
| HAR + biến giả cho ngày mỏng | 2,5416 |
| **HAR sau khi bỏ Chủ nhật khỏi chuỗi** | **0,1648** |

Bỏ ra khỏi chuỗi cải thiện **64%**. Biến giả *không* cứu được, vì thứ hỏng
không chỉ là mức của ngày Chủ nhật mà là các trung bình trượt 5 và 22 ngày
được tính trên chuỗi trộn.

**Cách xử lý đã chọn:** không bỏ dữ liệu mà **gộp phiên Chủ nhật vào ngày giao
dịch kế tiếp** (`volfc.merge_thin_days`), đúng quy ước ngày giao dịch FX bắt
đầu 22:00 UTC Chủ nhật. Mở cửa lấy của phiên mỏng, cao/thấp bao cả hai, đóng
cửa lấy của ngày chính, các đo lường nội ngày cộng lại. Giữ được 99,997%
tổng phương sai; phần mất là vài phiên mỏng sát ngày lễ không có ngày kế tiếp.

## 2. Dữ liệu mới cần có

`collect/rv_advanced.py` tính lại từ 34,9 triệu nến M1, cho mỗi ngày mỗi cặp:

| Đo lường | Dùng cho |
|---|---|
| `rv5` realized variance 5 phút | mục tiêu, và biến trễ |
| `rq5` realized quarticity | HARQ — hiệu chỉnh sai số đo lường |
| `bpv5` bipower variation | tách nhảy khỏi phần liên tục (HAR-CJ) |
| `rsp`,`rsn` semivariance dương/âm | SHAR |
| `n5` số thanh | phát hiện phiên mỏng |

`rv5` khớp `rv_m5` cũ tới sai số 6,7e−16 — tức script mới tái lập đúng script cũ.

## 3. So sánh 14 mô hình

Walk-forward, cửa sổ mở rộng, ước lượng lại mỗi phiên, chỉ dùng thông tin tới
t. 6 cặp × ~3.600 phiên chấm điểm. Mọi hồi quy ở không gian log với hiệu chỉnh
log-chuẩn +0,5·var(phần dư) khi đổi về mức.

| Mô hình | QLIKE TB | Hạng TB |
|---|---|---|
| **STHARQ** | **0,1645** | **2,7** |
| **EN(STHARQ, HARQ, SHAR)** | **0,1645** | **2,7** |
| EN(tất cả HAR) | 0,1649 | 4,5 |
| EN(HARQ, CJ, SHAR) | 0,1653 | 4,7 |
| HARQ | 0,1664 | 4,8 |
| HAR-full | 0,1667 | 6,8 |
| SHAR | 0,1665 | 7,3 |
| HAR-CJ | 0,1664 | 7,5 |
| STHAR | 0,1668 | 7,8 |
| THAR | 0,1670 | 8,2 |
| HAR | 0,1676 | 9,0 |
| GBM (cây tăng cường) | 0,1828 | 12,2 |
| MA5-RV5 | 0,1856 | 12,8 |
| **MA20-GK (đang nuôi panel)** | **0,2161** | **14,0** |

**Kiểm định.** Diebold–Mariano so với MA20-GK: mọi biến thể HAR thắng **6/6
cặp ở p<0,05**. Model Confidence Set (α=0,10, bootstrap khối): có mặt ở cả 6/6
cặp chỉ có ba mô hình — HARQ, STHARQ, EN(STHARQ,HARQ,SHAR). MA20-GK sống sót
1/6, GBM 1/6.

**Học máy không thắng.** GBM thua mọi biến thể HAR ở 5/6 cặp có ý nghĩa thống
kê. Kết quả này khớp với tài liệu 2024–2025: Fed FEDS 2025-061 kết luận mô
hình chuyển chế độ THAR/STHAR "consistently outperform ML and linear models",
và Branco et al. (2024) đặt tên bài là *"Does anything beat linear models?"*.
Đây là một kết luận âm đáng đưa vào luận văn, không phải một thất bại.

## 4. Mô hình được chọn

`src/volfc.py` — trung bình hình học của ba mô hình **STHARQ, HARQ, SHAR**.

Vì sao lấy tổ hợp thay vì STHARQ đơn lẻ, khi hai cái ngang điểm: DM giữa
chúng là 0/6 cả hai chiều, tức không phân biệt được. Khi không phân biệt
được thì tổ hợp an toàn hơn — nó không đặt cược vào một dạng hàm cụ thể, và
tập khóa sổ (6 cặp chéo + 2026) chưa mở nên rủi ro chọn nhầm mô hình là có
thật.

Ba thành phần và lý do có mặt:

- **STHARQ** — HAR với hệ số chuyển mượt theo chế độ biến động (biến chuyển
  là log RV ngày chuẩn hóa bằng cửa sổ 250 phiên trước), cộng hiệu chỉnh sai
  số đo lường bằng realized quarticity.
- **HARQ** — Bollerslev–Patton–Quaedvlieg (2016): ngày mà RV đo được kém chính
  xác thì hệ số trễ ngày phải bị co lại.
- **SHAR** — Patton–Sheppard (2015): tách semivariance âm và dương; biến động
  âm dự báo tốt hơn.

Panel mới bắt đầu 14/02/2012 thay vì 26/01/2010. Đây **không** phải lựa chọn
— STHARQ cần 250 phiên để chuẩn hóa biến chuyển chế độ cộng 300 quan sát để
ước lượng. Đổi `min_train` từ 320 lên 750 không làm đổi QLIKE tới chữ số thứ
tư, nên ràng buộc thật là cửa sổ 250 phiên đó.

## 5. Cải thiện có xuyên xuống dưới không — đo, không đoán

Đây là phần quan trọng nhất, và câu trả lời trung thực là **có, nhưng nhỏ đi
rất nhanh**.

**Chất lượng bản thân dự báo (tầng 2)** — cải thiện lớn:

| Cặp | corr(sig, biến động thực) cũ | mới | corr trên log cũ | mới |
|---|---|---|---|---|
| EURUSD | 0,624 | 0,711 | 0,696 | 0,760 |
| GBPUSD | 0,561 | 0,688 | 0,683 | 0,768 |
| USDJPY | 0,543 | 0,665 | 0,658 | 0,770 |
| AUDUSD | 0,624 | 0,720 | 0,686 | 0,758 |
| USDCAD | 0,647 | 0,720 | 0,704 | 0,758 |
| USDCHF | 0,435 | 0,522 | 0,655 | 0,736 |

Con số 0,511 ghi trong `DATASET.md` mục "đầu ra mô hình" nay là 0,52–0,72.

**Tầng 4 (định cỡ vị thế)** — cải thiện nhỏ:

| | cũ | mới |
|---|---|---|
| ngày rủi ro thực vượt 3σ dự báo | 0,97% | 0,93% |
| lỗ 1% xấu nhất của vị thế | 7,47% | 7,05% |
| sụt giảm tối đa (mô phỏng, cùng đường giá) | 40,9% | 39,1% |

**Tầng 6 (khoảng dự báo)** — cải thiện rất nhỏ: điểm khoảng Winkler giảm 1,4%
(254,8 → 251,3 pip), độ phủ giữ nguyên ở 89,8% so với 89,6%.

**Vì sao nhỏ dần.** Tầng 4 và tầng 6 không tiêu thụ σ̂ trực tiếp mà tiêu thụ
**phân vị đuôi của lợi suất đã chuẩn hóa**. Phân vị đó bị chi phối bởi hình
dạng đuôi dày của phân phối lợi suất FX, không phải bởi mức phương sai có điều
kiện. Cải thiện 24% ở phương sai chỉ đổi được vài phần trăm ở đuôi.

Đây là một kết luận nên nói thẳng trong luận văn, vì nó chặn được một lời hứa
sai: *dự báo biến động tốt hơn không tự động thành hệ thống quyết định tốt hơn*.

## 6. Một hồi quy phải sửa — số tầng của conformal

Dự báo sắc hơn làm **hỏng** cách phân tầng cũ của tầng 6:

| Cấu hình | độ phủ theo tầng (danh nghĩa 90%) | \|lệch\| max |
|---|---|---|
| sig cũ, 3 tầng | 89,5 / 90,3 / 88,8 | 1,2% |
| **sig mới, 2 tầng** | **88,9 / 90,0** | **1,1%** |
| sig mới, 3 tầng | 87,6 / 90,7 / 89,3 | 2,4% |
| sig mới, 4 tầng | 87,7 / 89,6 / 90,7 / 89,3 | 2,3% |
| sig mới, 5 tầng | 86,6 / 89,8 / 90,7 / 89,5 / 89,0 | 3,4% |

Giải thích: σ̂ sắc hơn thì phần dư còn lại ít không đồng nhất hơn, nên chia
nhỏ chỉ làm giảm số mẫu hiệu chuẩn mỗi tầng mà không mua thêm được gì.
`KhoangConformal` đổi mặc định sang `n_bins=2`.

## 7. Tái lập

```bash
python collect/rv_advanced.py --pair EURUSD   # từng cặp, ~15 giây, cần histdata_raw
python src/volfc.py EURUSD                    # tự kiểm, gồm kiểm rò rỉ nhìn trước
python src/build_panel2.py                    # dựng lại panel  (~20 giây)
python src/run_volbake.py                     # so 14 mô hình   (~110 giây)
python src/run_volstats.py                    # DM + MCS
python src/decision_record.py                 # tự kiểm tầng 6 trên panel mới
```

## Nguồn

- Bollerslev, Patton, Quaedvlieg (2016), *Exploiting the errors: A simple approach for improved volatility forecasting*, Journal of Econometrics.
- Patton, Sheppard (2015), *Good volatility, bad volatility: signed jumps and the persistence of volatility*, Review of Economics and Statistics.
- Andersen, Bollerslev, Diebold (2007), *Roughing it up: including jump components in the measurement, modeling and forecasting of return volatility*.
- Branco et al. (2024), *Forecasting realized volatility: Does anything beat linear models?*, Journal of Empirical Finance.
- Federal Reserve FEDS 2025-061, *Linear and nonlinear econometric models against machine learning models*.
- Hansen, Lunde, Nason (2011), *The Model Confidence Set*, Econometrica.
