# BỘ LỌC CHU KỲ NỘI TUẦN (BCL) — tiêu chí CHỐT TRƯỚC

Lập 13/09/2026. Nhánh `replan-2026`. **Commit chứa văn bản này nằm trước mọi
commit có số liệu.**

---

## 1. Lỗ hổng nhắm tới — và xác minh rằng nó thật

`MAU_HINH_FX.md` mục A2 gọi đây là *"cải tiến tầng 2 có cơ sở tài liệu mạnh
nhất còn chưa làm"*:

> *"Chuyện Chủ nhật đã sửa (`merge_thin_days`) mới chỉ là phần thô nhất của vấn
> đề này. Phần tinh hơn — chu kỳ theo **giờ trong ngày** và theo **ngày trong
> tuần** bên trong mỗi phiên — vẫn chưa được khử trong `rv_adv.csv`."*

**Đã xác minh nó không trùng với `deseason` đã bị loại.** `KETQUA_VONG7.md` ghi
khử mùa vụ nội tuần là *"âm — giúp khi chưa có lịch, thừa khi đã có"*. Nhưng
`volfc2.he_so_mua_vu` nhận `lv = log(rv5)` — tức khử mùa vụ **theo thứ, trên
chuỗi RV NGÀY**. BCL làm việc trên **lợi suất NỘI NGÀY, TRƯỚC khi cộng dồn
thành RV ngày**. Hai thứ khác nhau; cái sau chưa bao giờ chạm vào.

## 2. Nguồn

- **Boudt, Croux & Laurent (2011)**, *Robust Estimation of Intraweek
  Periodicity in Volatility and Jump Detection*, **Journal of Empirical
  Finance** 18(2):353–367.
- Áp cho FX tần suất cao: **Yi (2023)**, *Finance Research Letters* 55:103821.

Chữ **intraweek** (không chỉ *intraday*) là điểm cốt yếu: nó xử lý tương tác
**ngày-trong-tuần × giờ-trong-ngày**, đúng thứ FX cần vì tuần FX là 24×5.

## 3. Cách dùng ĐÚNG — và vì sao không phải "RV thay thế"

BCL **không** là một ước lượng RV thay thế. Chia RV cho hệ số chu kỳ sẽ ước một
**đại lượng khác** với phương sai tích hợp thật của ngày đó, nên mục tiêu sẽ
không còn so sánh được. BCL là **bộ lọc chuẩn hoá** để ngưỡng phát hiện nhảy áp
đồng đều qua mọi ô trong tuần:

> Một lợi suất 3σ lúc 14:00 New York (ô có phương sai cao **dự đoán được**) ít
> khả năng là nhảy hơn cùng lợi suất đó lúc 22:00.

Nên sản phẩm của nó là một phép **tách liên tục/nhảy (C/J) tốt hơn**, rồi đưa
vào HAR dưới dạng hai số hạng — đúng sơ đồ **HAR-CJ** (Andersen, Bollerslev &
Diebold 2007). Mục tiêu giữ nguyên `log rv5(t+1)`, nên QLIKE so sánh được.

## 4. Giả thuyết CHỐT TRƯỚC

> **H11.** Tách C/J bằng ngưỡng **đã lọc chu kỳ nội tuần** cho dự báo biến động
> tốt hơn mốc HAR sản xuất (vốn tách C/J bằng **bipower**, bất biến với thứ tự
> thời gian trong ngày).
>
> **H11b — điều kiện ghi công.** P1 (có lọc) phải tốt hơn **P2 (cùng phép tách,
> KHÔNG lọc chu kỳ)**. Nếu P2 cũng thắng tương đương thì thứ ăn tiền là *phát
> hiện nhảy bằng ngưỡng*, **không phải** lọc chu kỳ, và **BCL không được ghi
> công** — phải viết đúng như vậy.
>
> **Dấu dự kiến:** hệ số của `log C` **lớn hơn** hệ số của `log(1 + J/C)` —
> phần liên tục dai hơn phần nhảy, đúng phát hiện chuẩn của HAR-CJ.

## 5. Năm cấu hình — liệt kê đầy đủ TRƯỚC khi chạy

| | đặc trưng thêm vào `log h_HAR` |
|---|---|
| **B0** mốc | — (HAR sản xuất, đã có bipower C/J bên trong) |
| **P1** BCL | `log c_bcl`, `log(1 + j_bcl/c_bcl)` |
| **P2** ngưỡng thô *(đối chứng)* | `log c_tho`, `log(1 + j_tho/c_tho)` |
| **P3** cả hai | P1 + P2 |
| **P4** chỉ tỉ lệ chu kỳ | `ti_chuky` = Σ(r/s)² / Σr² |

**Đếm vào `KHOA_SO.md`: 4 cấu hình mới.** Không thêm cấu hình nào ngoài bảng này.

Tham số chốt tại đây, không điều chỉnh sau: ô = (thứ) × (288 ô 5 phút) ·
ước lượng scale **robust MAD** (`/0,6745`) · chuẩn hoá BCL
`mean(s²) = 1` · ngưỡng nhảy **c = 4** · tối thiểu 60 quan sát mỗi ô · tối
thiểu 100 ô mỗi ngày.

### 5a. ĐÍNH CHÍNH ĐẶC TẢ — ghi TRƯỚC khi thấy bất kỳ số liệu nào

Bản đầu của mục 5 viết **1.440 ô = (thứ 0–4) × 288**, dựa trên câu *"tuần FX
là 24×5"* của `MAU_HINH_FX.md`. **Sai trên dữ liệu thật.** Nến M1 của HistData,
sau khi chuyển sang UTC, **có cả ô thứ Bảy và Chủ nhật** — phiên Sydney mở vào
chiều Chủ nhật giờ New York. Chạy lần đầu báo `IndexError: index 1981 out of
bounds for size 1440`, tức có thật ô ở thứ 6.

Sửa thành **2.016 ô = 7 × 288**. Các ô cuối tuần ít mẫu sẽ không đạt ngưỡng 60
quan sát và rơi về trung vị, đúng cơ chế dự phòng đã chốt ở mục 5.

Đây là đính chính **đặc tả**, không phải tinh chỉnh: nó được phát hiện bằng một
lỗi chạy chương trình, sửa **trước khi** bất kỳ QLIKE nào được chấm, và nó làm
bộ lọc **bao phủ hơn** chứ không làm nó dễ thắng hơn. Ghi lại để người đọc thấy
rõ thứ tự, và vì chính câu "24×5" trong `MAU_HINH_FX.md` cũng cần đính chính.

## 6. Chống rò rỉ

Hệ số chu kỳ `s(thứ, ô)` ước **CHỈ trên đoạn huấn luyện** (< 2021-10-13), đóng
băng, rồi áp cho cả ba đoạn. Ba tự kiểm bắt buộc, dừng nếu hỏng:

1. Chuẩn hoá đúng: `mean(s²) = 1` tới sai số 1e−9.
2. **Cắt bỏ toàn bộ dữ liệu sau 2023-01-01 không làm đổi một hệ số nào** trong
   1.440 ô (vì hệ số chỉ dùng dữ liệu huấn luyện).
3. Chu kỳ có thật: `max(s)/min(s) > 2` — nếu không thì không có gì để khử, và
   mọi kết quả sau đó vô nghĩa.

## 7. Giao thức chấm

Khung so sánh **giống hệt** `kiem_nen_ablation.py` và `run_tang2_taptrung.py`,
để kết quả đặt cạnh `|gap|` và `hhi` được: `log rv5(t+1) ~ log h_HAR + đặc
trưng`, OLS trên **huấn luyện**, chọn trên **kiểm định**, chấm **một lần** trên
**kiểm tra**, QLIKE bất biến thang đo, DM + Newey–West, MCS, tách theo cặp.

## 8. TIÊU CHÍ PHỦ ĐỊNH

Kết luận **"lọc chu kỳ nội tuần không mang lại giá trị đo được"** khi bất kỳ
điều nào sau đây đúng:

1. Cấu hình tốt nhất trên **kiểm định** không thắng B0 trên **kiểm tra**, hoặc
   thắng nhưng DM p ≥ 0,05.
2. Hoặc P1 **không** tốt hơn P2 — khi đó BCL không được ghi công kể cả khi
   dự báo tốt lên (H11b).
3. Hoặc không đạt ≥ 5/6 cặp cải thiện.

Kết luận **dương** đòi **cả ba**: thắng kiểm tra với DM p < 0,0125 (Bonferroni
4 cấu hình) **và** P1 tốt hơn P2 **và** ≥ 5/6 cặp.

Không sửa tiêu chí sau khi thấy số. Nếu tiêu chí soạn dở thì khai báo chỗ dở,
giữ nguyên phán quyết — đúng như đã làm ở `DUOI_EVT.md` mục A5a.

## 9. Lực phát hiện — khai báo TRƯỚC

Bảng ~21.600 hàng (3.282 ở mỗi đoạn ngoài mẫu), cùng cỡ với thí nghiệm `|gap|`
và `hhi`. Ở cỡ này, `|gap|` đạt DM p = 0,0445 với cải thiện 0,67%, còn `hhi`
không đạt với 0,76% theo chiều xấu. Nên **ngưỡng phát hiện thực tế ở khung này
vào khoảng 0,5–0,7% QLIKE**. Cải thiện nhỏ hơn mức đó sẽ không phân biệt được
với nhiễu, và kết quả âm phải phát biểu kèm con số này.

## 10. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 4 (giả thuyết, kèm điều
kiện ghi công H11b), mục 8 (tiêu chí phủ định) và mục 9 (khai báo lực) **trước
khi** số liệu được chấm:

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không phải
> công cụ điền hộ.

---

# PHỤ LỤC A — KẾT QUẢ (13/09/2026)

*Tái lập: `python src/xay_bcl.py` rồi `python src/kiem_bcl.py`.
Kết quả: `data/bcl_noituan.csv`, `output/bcl_heso.json`, `output/bcl_ablation.json`.*

## A1. Chu kỳ nội tuần CÓ THẬT — và BCL hoạt động đúng như mô tả

Ba tự kiểm **ĐẠT** cho cả 6 cặp: `mean(s²)=1` tới 1e−10 · cắt dữ liệu sau
2023-01-01 đổi **0/2016** hệ số · biên độ chu kỳ `max/min` = **4,06–7,28**.

1.446/2.016 ô đủ 60 quan sát (phần thiếu là ô cuối tuần, rơi về trung vị).

Bộ lọc làm đúng việc nó hứa: nó phát hiện **ít** nhảy hơn ngưỡng thô, vì nó
không còn gán nhãn "nhảy" cho các lợi suất lớn ở những ô vốn có phương sai cao
dự đoán được.

| cặp | nhảy/ngày BCL | nhảy/ngày thô |
|---|---|---|
| EURUSD | 2,54 | 5,25 |
| GBPUSD | 2,77 | 5,56 |
| USDJPY | 2,94 | 4,25 |
| USDCHF | 3,09 | 5,86 |

## A2. Ablation — và phán quyết ÂM

| cấu hình | #đt | QLIKE kiểm định | QLIKE kiểm tra | so B0 | DM p |
|---|---|---|---|---|---|
| B0 mốc | 0 | 0,1569 | 0,1872 | — | — |
| P1 BCL C/J | 2 | 0,1435 | 0,1742 | −6,97% | 0,0102 |
| **P2 ngưỡng thô** *(đối chứng)* | 2 | **0,1413** | 0,1712 | **−8,59%** | 0,0022 |
| P3 cả hai | 4 | 0,1419 | 0,1717 | −8,30% | 0,0028 |
| P4 chỉ tỉ lệ chu kỳ | 1 | 0,1593 | 0,1872 | −0,03% | 0,966 |

| điều kiện | kết quả | |
|---|---|---|
| ĐK1 thắng kiểm tra, p < 0,0125 | **ĐẠT** | −8,59%, p=0,0022 |
| **ĐK2 P1 tốt hơn P2 — H11b** | **TRƯỢT** | P1 −6,97% **kém hơn** P2 −8,59% |
| ĐK3 ≥ 5/6 cặp | **ĐẠT** | 6/6 |

→ **H11b bác bỏ. BCL KHÔNG được ghi công.** Đối chứng không lọc chu kỳ **tốt
hơn** bản có lọc. Lọc chu kỳ nội tuần không mang lại giá trị đo được ở đây.

## A3. Nhưng −8,59% quá lớn để tin — và nó đúng là không thật

Con số −8,59% lớn **gấp 13 lần** `|gap|` (−0,67%), ứng viên tốt nhất dự án từng
có. Trong một repo mà mọi thứ đều thất bại, một con số như thế phải bị nghi là
rò rỉ trước khi được ghi. Bốn phép kiểm:

| phép kiểm | kết quả |
|---|---|
| Đặc trưng có tương quan bất thường với **mục tiêu** `log rv(t+1)`? | **không** — `log c_tho` 0,748 so `m_har` 0,733 |
| `rv_kiem` (tính từ M1) có khớp `rv5` của panel? | **khớp** — trung vị tỉ lệ **1,000**, corr log-log **0,9969**, đúng cả 16 năm |
| Lệch ngày khi nối bảng? | **không** — 24.894/24.902 hàng khớp đúng ngày |
| **Đối chứng còn thiếu: `log rv5(t)` THÔ, không tách C/J gì cả** | **đây mới là nguyên nhân** |

| cấu hình | QLIKE kiểm định | QLIKE kiểm tra | so B0 |
|---|---|---|---|
| B0 mốc | 0,1569 | 0,1872 | — |
| **chỉ thêm `log rv5(t)` thô** | **0,1381** | **0,1727** | **−7,78%** |
| P2 ngưỡng thô (C/J) | 0,1413 | 0,1712 | −8,59% |
| P1 BCL | 0,1435 | 0,1742 | −6,97% |

**Gần như toàn bộ −8,59% chỉ là "đưa RV của chính ngày t vào hồi quy".** Phép
tách C/J thêm được 0,81 điểm phần trăm; BCL thì **tệ hơn cả `log rv(t)` thô**.

Đối chứng này **không** có trong mục 5. Nó được thêm vào sau khi thấy số — điều
đó hợp lệ vì nó chỉ có thể **làm yếu** kết luận, không làm mạnh; đó là ngược
hẳn với p-hacking. Nhưng phải ghi rõ thứ tự như vậy.

## A4. PHÁT HIỆN THẬT — khung so sánh của chính repo bị thiếu một số hạng

Đây quan trọng hơn BCL rất nhiều, nên phải nói riêng.

Khung `log rv(t+1) ~ log h_HAR(t)` là khung đã dùng cho **mọi** ablation của dự
án: `hhi` (`run_tang2_taptrung.py`), mẫu hình nến và `|gap|`
(`kiem_nen_ablation.py`), Pha 3 vĩ mô, và B0 của Pha 3B. Trên đúng cùng bảng,
cùng hàng:

| | QLIKE kiểm định | QLIKE kiểm tra |
|---|---|---|
| HAR sản xuất chấm **trực tiếp** (không hồi quy lại) | 0,1587 | 0,1914 |
| khung B0 (hồi quy lại 1 hệ số) | 0,1569 | 0,1872 |
| **khung + `log rv5(t)`** | **0,1381** | **0,1727** |

Khung B0 **tốt hơn** HAR sản xuất chấm trực tiếp (−2,17%) — việc hồi quy lại
giúp, không hại. Nhưng thêm **một** biến hiển nhiên, RV của chính ngày hôm đó,
lại cải thiện **−7,78%** so khung B0 và **−9,78%** so HAR sản xuất.

Tức mô hình tầng 2 đang **để lại khoảng 8–10% QLIKE trên bàn**, thu lại được
bằng một số hạng duy nhất đã nằm trong tập thông tin của chính nó. Đây là một
khiếm khuyết **của mô hình**, phát hiện ra một cách tình cờ khi làm đối chứng
cho một thí nghiệm khác, và nó có hệ quả ngược lên mọi kết luận đã dùng khung
này: mọi phát biểu *"X cải thiện 0,x%"* đều được đo so với một mốc tự nó đã
lệch ~8%.

## A5. Hệ quả kiểm lại ngay: `|gap|` **mạnh lên** dưới mốc đã sửa

Phép kiểm đúng cho mọi ứng viên cũ là: nó còn thêm được gì **sau khi** `log
rv(t)` đã nằm trong mốc?

| cấu hình | QLIKE kiểm tra | so B0 cũ | **so B1 đã sửa** | DM p vs B1 |
|---|---|---|---|---|
| B0 khung cũ | 0,1872 | — | +8,43% | 0,0127 |
| B0 + \|gap\| *(kết quả đã công bố)* | 0,1860 | −0,67% | +7,71% | 0,0232 |
| **B1 = B0 + `log rv(t)`** | 0,1727 | −7,78% | — | — |
| **B1 + \|gap\|** | **0,1706** | **−8,88%** | **−1,20%** | **0,0027** |
| B1 + C/J thô | 0,1719 | −8,22% | −0,48% | 0,657 |
| B1 + C/J BCL | 0,1750 | −6,53% | **+1,35%** | 0,343 |

Ba điều đọc được, theo thứ tự quan trọng:

1. **`|gap|` sống sót và mạnh lên.** Dưới mốc đã sửa, nó cho **−1,20%** với
   **DM p = 0,0027** — so với −0,67% và p = 0,0445 dưới khung cũ. Nó **không**
   là biến đại diện cho RV ngày t; nó là thông tin cộng thêm thật, đúng như cơ
   chế đã khai báo trước (RV loại gap theo thiết kế nên HAR không bao giờ thấy).
2. **Tách C/J bằng ngưỡng sụp hẳn** khi `log rv(t)` đã có mặt: −0,48%, p = 0,657.
3. **BCL tệ hơn mốc** đã sửa: +1,35%.

**Lưu ý kỷ luật, quan trọng:** điều 1 **không** đảo được phán quyết của
`NEN_TIEUCHI.md`. Phán quyết đó được ra theo tiêu chí chốt trước trong **khung
cũ**, và nó đứng nguyên: `|gap|` đã trượt ĐK2 của nó. Bảng trên là một **phép
thử MỚI, trong một khung KHÁC**, nên nó phải được đếm là cấu hình mới trong
`KHOA_SO.md` và phát biểu là bằng chứng **bổ sung**, không phải cho điểm lại.
Điều nó củng cố là khuyến nghị đã có: đưa `|gap|` vào danh sách chốt của lần mở
tập niêm phong cuối cùng.

## A6. Kết luận cho luận văn

> *Bộ lọc chu kỳ nội tuần của Boudt–Croux–Laurent (2011) — dù chu kỳ **được
> xác nhận có thật** trên cả 6 cặp, biên độ 4–7 lần — **không** cải thiện dự báo biến động ngày ở đây: phép tách C/J có lọc chu kỳ **kém hơn** chính phép tách
> đó không lọc, và cả hai đều sụp xuống mức không ý nghĩa khi mốc so sánh đã
> bao gồm RV của ngày hiện tại.*
>
> *Giá trị thật của thí nghiệm này nằm ở đối chứng: nó phát hiện rằng khung so
> sánh dùng cho mọi ablation của dự án thiếu một số hạng đáng ~8% QLIKE, và khi
> sửa mốc đó thì `|gap|` — ứng viên mạnh nhất của dự án — **mạnh lên** thay vì
> biến mất (p 0,0445 → 0,0027), trong khi mọi cách tách nhảy đều biến mất.*
