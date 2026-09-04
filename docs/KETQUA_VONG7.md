# Vòng 7 — Kết quả sáu việc cải tiến pipeline

*Chạy 01/09/2026. Mọi con số dưới đây sinh ra từ mã trong repo và tái lập được bằng
`python src/run_grid2.py && python src/run_final7.py && python src/run_sax_stats.py
&& python src/run_poolability.py`.*

Sáu việc lấy từ bảng cuối `docs/MAU_HINH_FX.md`, xếp theo tỷ lệ giá trị trên chi phí.
**Bốn trong sáu việc cho kết quả ÂM.** Đó không phải thất bại — nó là phần lớn giá trị
của vòng này, vì mỗi kết quả âm là một câu trả lời sẵn cho một câu hỏi phản biện.

---

## 0. Chia lại dữ liệu 70/15/15

| đoạn | từ | đến | phiên/cặp | vai trò |
|---|---|---|---|---|
| huấn luyện | 2012-02-14 | 2021-10-12 | 2.503 | xây dựng, gỡ lỗi, nhìn thoải mái |
| kiểm định | 2021-10-13 | 2023-11-17 | 547 | **chọn** mô hình và siêu tham số |
| kiểm tra | 2023-11-20 | 2025-12-31 | 549 | **chấm điểm một lần** |

Tỷ lệ tính trên 3.649 phiên *có dự báo* chung cho cả 6 cặp (sau khi trừ 500 phiên đệm
của cửa sổ mở rộng). Tầng 2 ước lượng lại tham số mỗi phiên bằng cửa sổ mở rộng, nên
"huấn luyện" ở đây không phải một tập khớp cố định mà là đoạn thời gian **được phép
nhìn vào dự báo**.

**Ghi chú trung thực.** Mốc kiểm tra cũ là 2023-03-24; đoạn kiểm tra mới nằm *trong*
đoạn kiểm tra cũ, và đoạn cũ đã từng được chấm một lần ở vòng trước. Tập này không còn
hoàn toàn trinh nguyên. Lớp bảo vệ thật sự vẫn là **tập khoá sổ** (6 cặp chéo + toàn bộ
2026), chưa hề mở.

---

## 1. Việc 6 (hoá ra quan trọng nhất) — lịch ngân hàng trung ương RIÊNG cho từng cặp

Kế hoạch xếp việc này thứ sáu. Thực tế nó là **cải tiến duy nhất có tác dụng thật**.

Ý tưởng: mỗi cặp chịu **hai** ngân hàng trung ương, và cặp nào cũng khác cặp nào.

| cặp | NHTW của đồng không phải USD | + luôn có |
|---|---|---|
| EURUSD | ECB | FOMC |
| GBPUSD | BOE | FOMC |
| USDJPY | BOJ | FOMC |
| AUDUSD | RBA | FOMC |
| USDCAD | BOC | FOMC |
| USDCHF | SNB | FOMC |

Đã thu thập **901 ngày công bố quyết định** từ trang chính thức của bảy ngân hàng
(FOMC 137, ECB 157, RBA 179, BOC 136, BOJ 119, BOE 101, SNB 72), giai đoạn 2010–2026,
lưu ở `data/cb_dates.csv`. Biến đưa vào mô hình gồm: ngày họp FOMC, ngày họp NHTW của
cặp, **ngày kế tiếp** của cả hai (Lee & Wang 2025: đảo chiều tập trung trong cửa sổ
12–24 giờ sau công bố), thứ Sáu đầu tháng (NFP) và hai phiên cuối tháng.

Lịch biết trước nhiều năm nên **không thể rò rỉ** — đã kiểm chứng bằng bài kiểm tra
bóp méo tương lai (mục 8).

Bốn cách mã hoá sự kiện được so trên đoạn kiểm định:

| cách mã hoá | QLIKE kiểm định (trung bình mọi cấu hình khác) |
|---|---|
| `capday` — NHTW riêng từng cặp + ngày kế tiếp | **0,1170** |
| `cap` — NHTW riêng từng cặp | 0,1175 |
| `cbonly` — chỉ FOMC + NHTW của cặp | 0,1181 |
| `chung` — FOMC + ECB cho mọi cặp | 0,1235 |
| `off` — không có lịch | 0,1272 |

Đọc bảng này theo chiều dọc là thấy đúng luận điểm "mẫu riêng từng đồng tiền":
dùng **chung** một lịch ECB+FOMC cho cả 6 cặp chỉ ăn được 3% so với không có gì,
còn dùng lịch **đúng** của từng cặp ăn được 8%.

*Lỗ hổng dữ liệu đã biết:* lịch BOE 2010–2014 và BOJ 2011–2014 không lấy được từ nguồn
chính thức (BoE không công bố ngày *công bố* trước 2015; trang lưu trữ BOJ bị cắt).
Hai lỗ này nằm **hoàn toàn trong đoạn huấn luyện**, gây suy giảm hệ số chứ không gây
rò rỉ. Đoạn kiểm định và kiểm tra phủ đầy đủ.

---

## 2. Việc 2 — khử chu kỳ nội tuần: **ÂM**

Đã thử ba mức: không khử; khử theo **thứ trong tuần** (hệ số trung vị bền, ước lượng
chỉ trên huấn luyện, kiểu Boudt–Croux–Laurent *intraweek*); và khử thêm **độ phủ**
theo số thanh 5 phút `n5`.

| | QLIKE kiểm định (trung bình) | tốt nhất |
|---|---|---|
| khử theo thứ | 0,1250 | 0,1214 |
| không khử | 0,1279 | 0,1231 |
| khử thứ + độ phủ | **0,1372** | 0,1315 |

Khử theo thứ giúp **khi chưa có lịch sự kiện** (−2,3%). Nhưng khi đã đưa lịch NHTW vào,
lợi ích biến mất: cấu hình thắng có `deseason=none`. Cách đọc hợp lý là hai thứ **trùng
thông tin** — phần lớn "hiệu ứng thứ" trong RV của FX thực chất là hệ quả của việc các
ngân hàng trung ương họp vào thứ Tư (FOMC) và thứ Năm (ECB, BOE), cộng NFP thứ Sáu.
Lịch sự kiện là biến nhân quả; biến giả theo thứ chỉ là cái bóng của nó.

Khử theo **độ phủ** thì làm hỏng hẳn (+7,3%). Lý do rõ: `n5` thấp không phải nhiễu đo
lường mà là **thông tin thật** — ngày lễ, ngày ít giao dịch quả thật có phương sai thấp.
Chia nó đi là vứt tín hiệu.

---

## 3. Việc 3 — gộp hay tách 6 cặp: **hệ số khác nhau, nhưng co ngót vẫn ÂM**

Kiểm định poolability (Chow/Roy–Zellner) trên đoạn huấn luyện, mô hình HARQ:

```
N = 18.198   k = 6   ràng buộc q = 30
F(30, 18.162) = 2,82   p = 4,3 × 10⁻⁷   →  BÁC BỎ tính gộp được
```

Hệ số quả thật khác nhau, và khác đúng chỗ đáng chú ý:

| hệ số | EURUSD | GBPUSD | USDJPY | AUDUSD | USDCAD | USDCHF | độ tán/\|TB\| |
|---|---|---|---|---|---|---|---|
| log RV(d) | 0,400 | 0,374 | **0,514** | **0,519** | 0,425 | 0,384 | 0,14 |
| log RV(w) | 0,335 | 0,394 | **0,213** | 0,282 | 0,288 | 0,357 | 0,19 |
| log Q | −0,543 | **+0,362** | −0,828 | **−1,221** | −0,930 | −0,454 | 0,83 |

USDJPY và AUDUSD dồn trọng số vào thành phần **ngày**, EURUSD và USDCHF dồn vào thành
phần **tuần** — tức JPY và AUD phản ứng nhanh hơn, quên nhanh hơn. Hệ số hiệu chỉnh sai
số đo (`log Q`) của GBPUSD đổi dấu so với năm cặp còn lại.

**Nhưng khác nhau trong mẫu không có nghĩa là tách thì dự báo tốt hơn.** Quét toàn dải
co ngót λ (0 = riêng hoàn toàn, 1 = gộp hoàn toàn) trên đoạn kiểm định:

| λ | ý nghĩa | QLIKE kiểm định |
|---|---|---|
| 0,00 | riêng hoàn toàn từng cặp | 0,1162 |
| 0,30 | co ngót 30% | 0,1158 |
| 0,70 | co ngót 70% | 0,1156 |
| 1,00 | gộp hoàn toàn | 0,1158 |

Toàn dải chỉ chênh **0,0006 QLIKE (0,5%)** — nhỏ hơn nhiều so với lợi ích của lịch sự
kiện (7,3%). Với ~2.500 quan sát mỗi cặp, sai số ước lượng đã đủ nhỏ để không cần mượn
thông tin từ cặp khác. Đây đúng là đánh đổi thiên lệch–phương sai mà Pesaran–Pick–
Timmermann mô tả, và ở phía T lớn thì ước lượng riêng thắng.

---

## 4. Việc 5 — biến RV chéo cặp: **ÂM**

Thêm một hệ số cho log RV trung bình của 5 cặp còn lại tại ngày *t*. Trung bình mọi cấu
hình: 0,1208 khi bật so với 0,1206 khi tắt. Cấu hình tốt nhất có nó **tắt**. Lan toả
biến động giữa các cặp là có thật (Rubaszek et al. 2025 đo được trên tần suất 5 phút),
nhưng ở tần suất **ngày** và với chân trời một bước, thông tin đó đã nằm sẵn trong lịch
sử RV của chính cặp đó.

---

## 5. Chọn cấu hình — và cách tránh overfit chính đoạn kiểm định

Đã backtest **1.024 cấu hình** (5 trục × 7 mức λ). Vấn đề: chênh lệch giữa cấu hình
thứ 1 và thứ 20 chỉ 0,0006 QLIKE. Chọn cái nhỏ nhất trong 1.024 là overfit đoạn kiểm định.

Giao thức đã dùng:
1. Lấy 40 cấu hình đầu, chạy **Model Confidence Set** (Hansen–Lunde–Nason 2011) trên
   chuỗi tổn thất kiểm định → **cả 40 đều sống sót**, không phân biệt được về thống kê.
2. Trong tập MCS, chọn cấu hình **đơn giản nhất** (ít trục bật nhất), không phải cấu
   hình QLIKE nhỏ nhất.

**Cấu hình chốt:** `deseason=none, crosspair=off, event=capday, window=expanding,
recal=off, lambda=0` — tức chỉ thêm **một** thứ so với bản cũ: lịch NHTW riêng từng cặp.

QLIKE kiểm định 0,1162 so với 0,1281 của bản cũ (**−9,3%**). Cấu hình QLIKE nhỏ nhất
đạt 0,1156 — chênh 0,0006, nằm trong sai số, nên không đáng đổi lấy độ phức tạp.

---

## 6. Kết quả trên đoạn kiểm tra (mở đúng một lần)

**QLIKE, 2023-11-20 → 2025-12-31, 548 phiên × 6 cặp**

| cặp | MA20-GK (sản xuất cũ) | HAR gốc | **HAR vòng 7** | v7 so với gốc |
|---|---|---|---|---|
| EURUSD | 0,1893 | 0,1575 | **0,1454** | −7,7% |
| GBPUSD | 0,1509 | 0,1231 | **0,1127** | −8,5% |
| USDJPY | 0,3795 | 0,3116 | **0,3006** | −3,5% |
| AUDUSD | 0,1861 | 0,1363 | **0,1175** | −13,8% |
| USDCAD | 0,1788 | 0,1430 | **0,1332** | −6,8% |
| USDCHF | 0,2187 | 0,1644 | **0,1417** | −13,8% |
| **trung bình 6 cặp** | **0,2172** | **0,1726** | **0,1585** | **−8,2%** |
| trung bình 5 cặp (bỏ USDJPY) | 0,1847 | 0,1449 | 0,1301 | −10,2% |

So với MA20-GK đang nuôi panel cũ: **−27,0%**.

**Diebold–Mariano (Newey–West):** HAR vòng 7 thắng có ý nghĩa p<0,05 ở **4/6 cặp** so
với HAR gốc (USDJPY p=0,24 và USDCAD p=0,086 không đủ ý nghĩa) và **6/6 cặp** so với
MA20-GK.

**Model Confidence Set (α=0,10):** chỉ còn **{HAR vòng 7}**. MA20-GK bị loại ở p=0,0013,
HAR gốc bị loại ở p=0,0037.

**Bộ chỉ số phân phối trên đoạn kiểm tra** (lợi suất/σ̂ ~ t Student, tham số ước lượng
trên huấn luyện; VaR danh nghĩa 2,5%):

| cặp | CRPS×10⁴ | pinball 5%×10⁵ | log score | PIT-KS p | vi phạm | Kupiec p | Christoffersen p | DQ p | FZ0 |
|---|---|---|---|---|---|---|---|---|---|
| EURUSD | 23,40 | 46,71 | −4,082 | 0,757 | 2,92% | 0,540 | 0,326 | 0,722 | −4,516 |
| GBPUSD | 23,77 | 47,66 | −4,050 | 0,361 | 2,74% | 0,726 | 0,358 | 0,833 | −4,539 |
| USDJPY | 34,52 | 70,74 | −3,683 | 0,115 | 3,29% | 0,261 | 0,268 | 0,191 | −3,985 |
| AUDUSD | 29,96 | 64,33 | −3,829 | 0,582 | 2,92% | 0,540 | 0,326 | 0,151 | −4,157 |
| USDCAD | 17,51 | 33,82 | −4,365 | 0,258 | 1,83% | 0,288 | 0,542 | 0,909 | −4,859 |
| USDCHF | 26,56 | 57,46 | −3,955 | 0,674 | 3,47% | 0,170 | 0,242 | 0,193 | −4,258 |

**Kupiec 6/6, Christoffersen 6/6, DQ 6/6, PIT-KS 6/6 đều không bác bỏ.** Hiệu chuẩn
phân phối sạch trên toàn bộ 6 cặp.

---

## 7. Việc 1 — chấm điểm phân tầng theo chế độ

Ngũ phân vị của **biến động dự báo**; ngưỡng lấy từ đoạn huấn luyện của từng cặp nên
biết trước, không rò rỉ.

| chế độ | n | MA20-GK | HAR gốc | HAR v7 | thiên lệch log | tỷ lệ dự báo thiếu |
|---|---|---|---|---|---|---|
| Q1 êm nhất | 979 | 0,1813 | 0,1717 | 0,1691 | −0,098 | 33,7% |
| Q2 | 807 | 0,2040 | 0,1672 | 0,1585 | −0,130 | 33,2% |
| Q3 | 628 | 0,1962 | 0,1544 | 0,1345 | −0,088 | 37,3% |
| Q4 | 492 | 0,2488 | 0,1995 | 0,1592 | −0,149 | 31,5% |
| Q5 căng nhất | 382 | **0,3309** | 0,1820 | **0,1700** | −0,170 | 31,9% |
| **Q5 / Q1** | | **1,82** | 1,06 | **1,01** | | |

Đây là bảng đáng đưa vào luận văn nhất, và nó nói một điều mà con số trung bình gộp
giấu kín: **khoảng cách giữa hai mô hình rộng gấp 13 lần khi thị trường căng.**
MA20-GK trừ HAR v7 là **+0,0122 ở Q1** nhưng **+0,1609 ở Q5**. Nói cách khác, ở chế độ
êm hai mô hình gần như ngang nhau; toàn bộ giá trị của tầng 2 nằm ở chế độ căng — đúng
chỗ mà hệ thống quyết định cần nó nhất.

*Đọc đúng hai cột cuối.* QLIKE là thước đo **tương đối** nên không tự tăng theo mức biến
động; điều đáng đọc là độ bền theo chế độ, không phải mức tuyệt đối. Thiên lệch log âm ở
mọi chế độ là **đúng** chứ không phải lỗi: QLIKE chấm kỳ vọng có điều kiện, mà phân phối
RV lệch phải, nên trung vị tỷ số thực/dự báo phải nhỏ hơn 1.

### Tập trung tổn thất ở đuôi

| cặp | QLIKE trung bình | trung vị | 1% ngày tệ nhất chiếm | 5% tệ nhất chiếm |
|---|---|---|---|---|
| EURUSD | 0,1454 | 0,0537 | 21,0% | 48,1% |
| GBPUSD | 0,1127 | 0,0420 | 16,9% | 42,1% |
| **USDJPY** | **0,3006** | 0,0671 | **53,8%** | **68,3%** |
| AUDUSD | 0,1175 | 0,0393 | 18,8% | 46,6% |
| USDCAD | 0,1332 | 0,0417 | 24,7% | 51,7% |
| USDCHF | 0,1417 | 0,0456 | 20,4% | 50,5% |

### Về USDJPY — tại sao KHÔNG bỏ

USDJPY có QLIKE cao gấp đôi các cặp khác. Nhưng **53,8% tổn thất của nó đến từ 1% số
ngày** — nghĩa là 5 ngày trên 548. Trên đoạn kiểm định, 12 ngày trên 547 chiếm 52% tổng
tổn thất, và 12 ngày đó là:

| ngày | thực/dự báo | chuyện gì xảy ra |
|---|---|---|
| 2022-09-22 | 12,3× | BOJ can thiệp tỷ giá lần đầu từ 1998 |
| 2022-10-21 | **31,7×** | ngày can thiệp lớn nhất |
| 2022-10-24 | 7,9× | can thiệp tiếp |
| 2022-12-20 | 9,6× | BOJ bất ngờ nới biên độ YCC |

USDCHF y hệt: QLIKE năm 2015 là **1,669** vì đúng một ngày — 15/01/2015, SNB bỏ sàn.

Đây là **cú sốc chính sách không dự báo được**, không phải khuyết tật của cặp hay của
mô hình. Bỏ USDJPY để làm đẹp con số là bỏ bằng chứng chứ không phải bỏ vấn đề, và hội
đồng sẽ hỏi ngay tại sao mất một cặp. Cách xử lý đúng là **báo cáo cả trung bình lẫn
trung vị lẫn bảng phân tầng** — cả ba đều có ở trên. Con số 5 cặp (0,1301) có sẵn nếu
cần, nhưng nên đứng cạnh con số 6 cặp chứ không thay nó.

Trớ trêu là: vòng này lại **cải thiện USDJPY nhiều hơn** trên kiểm định (−7,1%) nhờ đúng
lịch BOJ.

### Kiểm định forecast breakdown (Giacomini–Rossi 2009, bản rút gọn, một phía)

| cặp | L̄ huấn luyện | L̄ kiểm tra | chênh | t | p | kết luận |
|---|---|---|---|---|---|---|
| EURUSD | 0,1138 | 0,1454 | +0,0316 | 1,87 | 0,031 | **gãy** |
| GBPUSD | 0,1361 | 0,1127 | −0,0234 | −2,35 | 0,99 | không bác bỏ |
| USDJPY | 0,1803 | 0,3006 | +0,1203 | 1,12 | 0,132 | không bác bỏ |
| AUDUSD | 0,1137 | 0,1175 | +0,0038 | 0,28 | 0,390 | không bác bỏ |
| USDCAD | 0,0962 | 0,1332 | +0,0370 | 1,38 | 0,084 | không bác bỏ |
| USDCHF | 0,2252 | 0,1417 | −0,0835 | −4,63 | 1,00 | không bác bỏ |

Chỉ EURUSD bị bác bỏ, và sát ngưỡng. Bản rút gọn chưa có số hạng hiệu chỉnh sai số ước
lượng tham số nên hơi rộng rãi; dùng để so sánh giữa các cặp là chính.

---

## 8. Kiểm chứng không rò rỉ

Với **đúng cấu hình đã chốt**: nhân toàn bộ dữ liệu sau mốc 75% lên 9 lần rồi chạy lại.
**2.561 dự báo quá khứ ở cả 6/6 cặp giữ nguyên từng chữ số.** Engine mới cũng khớp bản
`volfc.py` cũ tới sai số tương đối 1,1 × 10⁻¹¹ trên cấu hình gốc.

---

## 9. Việc 4 — kiểm định bội cho nhánh mẫu ký hiệu

Đây là lỗ hổng phản biện lớn nhất của nhánh SAX, và giờ đã bịt.

Thay vì chỉ kiểm ba mẫu đã được chọn sẵn, đã **liệt kê toàn bộ không gian**: W ∈ {2,3,4}
trạng thái tiền đề × 3 đích = 351 giả thuyết, 336 trong đó đủ số khớp.

**Ba mô hình null** (SPEck, Jenkins et al. 2022 — chọn null quyết định kết luận):

| null | ý nghĩa | ngưỡng max\|z\| 95% |
|---|---|---|
| xoay chuỗi đích | "không gì dự báo được" | 11,51 |
| **khối 2 ngày** | **"đã chứa sẵn tính dai AR(1)"** | **29,51** |
| khối 5 ngày | "đã chứa sẵn tính dai một tuần" | 53,41 |
| — không hiệu chỉnh bội — | | **1,96** |

Null khối 2 ngày là cái đúng để dùng: nó **đã chứa sẵn** tính dai của biến động, nên một
mẫu chỉ sống sót nếu nó nói thêm điều gì ngoài "hôm qua cao thì hôm nay cũng cao".
Thủ tục là **maxT từng bước xuống** của Westfall & Young — bản một bước sẽ bị các mẫu
tính dai tầm thường (HIGH→HIGH ⇒ HIGH, z ≈ 69) chiếm hết thống kê max và làm mọi giả
thuyết khác không bao giờ bác bỏ được.

**Bao nhiêu giả thuyết sống sót trong 336:**

| thủ tục | số sống sót |
|---|---|
| không hiệu chỉnh gì (p<0,05) | **248** — nếu toàn nhiễu thì kỳ vọng 17 |
| FDR Benjamini–Hochberg (q<0,10) | 262 |
| W-Y bước xuống, null xoay | 158 |
| **W-Y bước xuống, null khối 2 ngày** | **41** |
| W-Y bước xuống, null khối 5 ngày | 20 |

**Ba mẫu của HuyH:**

| mẫu | lift | z | p thô | p W-Y | hạng /336 | sống sót? |
|---|---|---|---|---|---|---|
| MEDIUM → HIGH → HIGH ⇒ HIGH | 1,935 | 20,06 | <10⁻⁴ | <0,001 | 24 | **CÓ** |
| LOW → MEDIUM → LOW ⇒ LOW | 1,979 | 18,49 | <10⁻⁴ | <0,001 | 27 | **CÓ** |
| HIGH → HIGH → MEDIUM ⇒ HIGH | 1,430 | 9,34 | <10⁻⁴ | 1,000 | 81 | **KHÔNG** |

Xác nhận trên đoạn kiểm tra (chưa dùng ở bước phát hiện): cả ba đều tái lập lift thô
(3,420 / 1,405 / 1,709, đều p<0,01) — nhưng lift thô không đủ.

**Đối chứng có điều kiện.** Hutchinson et al. (2022) cho thấy toàn bộ lợi suất bất
thường của quy tắc kỹ thuật tiền tệ bị động lượng chuỗi thời gian hấp thụ. Mẫu ở đây là
mẫu **biến động** chứ không phải mẫu **hướng**, nên đối chứng tương đương không phải
TSMOM mà là **chính dự báo HAR**:

`1{trạng thái đích} = a + b·1{khớp mẫu} + c·log(dự báo HAR)`

| mẫu | b chỉ mẫu | t | b khi có HAR | t | kết luận |
|---|---|---|---|---|---|
| MEDIUM → HIGH → HIGH ⇒ HIGH | 0,3675 | 22,6 | 0,1236 | **8,94** | còn tin riêng |
| LOW → MEDIUM → LOW ⇒ LOW | 0,2756 | 16,9 | 0,0812 | **6,05** | còn tin riêng |
| HIGH → HIGH → MEDIUM ⇒ HIGH | 0,1599 | 9,9 | −0,0043 | **−0,32** | bị HAR hấp thụ |

**Hai thủ tục hoàn toàn độc lập chỉ vào cùng một mẫu phải loại.** Đó là bằng chứng mạnh
hơn nhiều so với việc chỉ có một trong hai.

**Đề nghị:** giữ hai mẫu đầu ở tầng 6 làm lời giải thích, **bỏ mẫu thứ ba**, và ghi
đúng như trên vào luận văn. Kèm câu: trong 336 giả thuyết, 248 "có ý nghĩa" nếu không
hiệu chỉnh nhưng chỉ 41 sống sót khi hiệu chỉnh đúng — hệ số thổi phồng 6 lần.

---

## 10. Tổng kết sáu việc

| # | việc | kết quả | vào hệ thống? |
|---|---|---|---|
| 1 | chấm điểm phân tầng theo chế độ | **thành công** — Q5/Q1 là 1,82 (cũ) so với 1,01 (mới); khoảng cách rộng gấp 13 lần ở chế độ căng | có, thành bảng báo cáo chuẩn |
| 2 | khử chu kỳ nội tuần | **âm** — giúp khi chưa có lịch, thừa khi đã có; khử độ phủ làm hỏng hẳn | không |
| 3 | co ngót hệ số về panel | **âm** — poolability bị bác bỏ (p=4,3e−7) nhưng toàn dải λ chỉ chênh 0,5% | không |
| 4 | kiểm định bội cho mẫu SAX | **thành công** — loại 1 trong 3 mẫu, hai thủ tục độc lập cùng chỉ một mẫu | có, bỏ mẫu thứ ba |
| 5 | biến RV chéo cặp | **âm** — thông tin đã nằm trong lịch sử của chính cặp đó | không |
| 6 | lịch sự kiện riêng từng cặp | **thành công** — −9,3% kiểm định, −8,2% kiểm tra, MCS chỉ còn một mình | **có** |

**Một dòng cho hội đồng:** trong sáu cải tiến có cơ sở tài liệu, đúng một cái có tác dụng
— và nó là cái *rẻ nhất về mặt mô hình nhưng đắt nhất về mặt công thu thập dữ liệu*:
biết ngày nào ngân hàng trung ương **của đúng đồng tiền đó** họp.

---

## 11. Còn lại

- **Chưa làm:** đo hiệu ứng ngày-trong-tuần bằng bộ lọc Boudt–Croux–Laurent *intraweek*
  trên dữ liệu nội ngày thật (repo chỉ có RV ngày và thanh giờ, chưa có M1/M5).
- **Chưa làm:** lấp lịch BOE 2010–2014 và BOJ 2011–2014.
- **Chưa làm:** P3 — chốt cấu hình, mở tập khoá sổ (6 cặp chéo + 2026), chạy một lần.
- **Nên làm trước P3:** cập nhật `panel2_6pairs.csv` bằng dự báo của cấu hình vòng 7,
  rồi chạy lại tầng 3–6 để xem cải thiện tầng 2 chảy xuống bao nhiêu.
