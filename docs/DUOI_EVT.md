# VÁ ĐUÔI — HƯỚNG THỨ NĂM: EVT/POT. Tiêu chí CHỐT TRƯỚC

Lập 13/09/2026. Nhánh `replan-2026`. **Commit chứa văn bản này nằm trước mọi
commit có số liệu.**

---

## 1. Khiếm khuyết nhắm tới

`CHISO_DANHGIA.md` mục 5b–5e: ở mức **99%**, USDJPY và USDCHF trượt backtest
VaR/ES (USDJPY vi phạm 2,07% thay vì 1,0%; tỷ lệ ES 0,799; PIT-KS 0,0036).
**4/6 cặp đạt.** Đây là khiếm khuyết **duy nhất còn chưa vá** của hệ thống.

Bốn hướng đã thử và thất bại:

| hướng | tài liệu | kết quả |
|---|---|---|
| V1 mở rộng / V2 cuộn 500 | mục 5c | V1 thua V0 trên kiểm tra |
| phân vị theo chế độ biến động | mục 5d | kém hơn mốc |
| CAViaR (4 biến thể) | mục 9.2 | không ăn tiền |
| cửa sổ họp NHTW/FOMC | mục 5e | không thắng trên kiểm định |

## 2. Vì sao EVT khác hẳn bốn hướng đó — lý lẽ chốt trước

Cả bốn hướng trên đều ước phân vị 1% bằng **phân vị thực nghiệm**: với ~750
quan sát, phân vị 1% tựa lên khoảng **7 điểm cực trị**. Bảy điểm không đủ để
định hình một cái đuôi; mọi biến thể chỉ đổi *cửa sổ nào* được dùng, chứ không
đổi *bao nhiêu thông tin* được dùng để ước cùng con số đó.

EVT/POT đổi đúng chỗ đó. Nó không ước phân vị trực tiếp mà khớp **phân phối
Pareto tổng quát** cho **toàn bộ** phần vượt ngưỡng `u` (ở `u` = phân vị 90 thì
có ~75 điểm), rồi **ngoại suy giải tích** ra mức 1%:

```
VaR_p = u + (β/ξ)·[ (p·n/N_u)^(−ξ) − 1 ]
ES_p  = (VaR_p + β − ξ·u) / (1 − ξ)            với ξ < 1
```

Tức **cùng một mẫu, nhưng dùng gấp ~10 lần số quan sát** để ước cùng con số.

Nền lý thuyết: Pickands–Balkema–de Haan (phần dư vượt ngưỡng hội tụ về GPD).
Áp cho VaR tài chính: **McNeil & Frey (2000)**, *Estimation of tail-related
risk measures for heteroscedastic financial time series*, **Journal of
Empirical Finance** 7(3–4):271–300 — sơ đồ "lọc GARCH rồi EVT cho phần dư".
Cấu hình ở đây là **lọc HAR rồi EVT cho z**, cùng sơ đồ.

### 2a. Mã đã có sẵn trong repo mà chưa ai chạy

`src/experiment2.py` (sửa lần cuối **29/08/2026**) đã cài `gpd_tail` — FHS+EVT
với Kupiec/Christoffersen/DQ/MCS. Nhưng nó đọc `/tmp/fx/fc_*.csv` (đường dẫn
chết), kết quả **không có trong bất kỳ tài liệu nào**, và **không có dòng nào
trong `KHOA_SO.md`**. `va_duoi_evt.py` viết lại theo giao thức hiện tại và có
tự kiểm riêng, không gọi script cũ.

## 3. Giả thuyết CHỐT TRƯỚC

> **H10.** Phân phối z có đuôi dày: **ξ > 0** ở cả 6 cặp. Và EVT đưa tỷ lệ vi
> phạm ở mức 99% **về gần 1% hơn** so với phân vị thực nghiệm, **mạnh nhất ở
> USDJPY và USDCHF** — hai cặp mà phân vị thực nghiệm đang hụt.
>
> ξ ≤ 0 ở đa số cặp = giả thuyết về hình dạng bị bác bỏ, và phải ghi như vậy
> kể cả khi backtest có tốt lên.

## 4. Năm biến thể — liệt kê đầy đủ TRƯỚC khi chạy

| | ước lượng | cửa sổ | ngưỡng u |
|---|---|---|---|
| V0 *(mốc sản xuất)* | phân vị thực nghiệm | huấn luyện, đóng băng | — |
| V1 *(tham chiếu)* | phân vị thực nghiệm | mở rộng, khớp lại mỗi 21 phiên | — |
| **E1** | **GPD/POT** | huấn luyện, đóng băng | 90% |
| **E2** | **GPD/POT** | mở rộng, khớp lại mỗi 21 phiên | 90% |
| **E3** | **GPD/POT** | cuộn 500 phiên | 90% |
| **E4** | **GPD/POT** | mở rộng | **95%** (độ nhạy ngưỡng) |

**Đếm vào `KHOA_SO.md`: 5 cấu hình** (V0 và V1 đã đếm ở dòng 10; E1–E4 mới, cộng
biến thể ngưỡng). Không thêm biến thể nào ngoài bảng này.

## 5. Quy tắc chọn — và vì sao nó phải LIÊN TỤC

Mục 5e đã chứng minh một điều then chốt:

> *"Đoạn kiểm định về cấu trúc **không thể phân biệt được** bất kỳ cách vá nào
> cho vấn đề USDJPY, vì chính vấn đề đó chỉ tồn tại trên kiểm tra… USDJPY đạt
> ở MỌI phương án trên kiểm định, kể cả mốc."*

Nên tiêu chí **đạt/KHÔNG** (nhị phân) bị mù — nó không thể xếp hạng các biến
thể. Đó là lý do tiêu chí chọn ở đây là **liên tục**, chốt tại đây:

```
S = trung bình trên (α ∈ {5%, 1%}) × (6 cặp) của
      | tỷ lệ vi phạm / α − 1 |  +  | tỷ lệ ES − 1 |
```

Thấp hơn là tốt hơn. S là **liên tục và có hướng** (0 là hoàn hảo), nên nó xếp
hạng được cả khi mọi biến thể đều "đạt". Nó dùng đúng hai đại lượng mà khiếm
khuyết được định nghĩa bằng: tần suất vi phạm sai, và ES hụt.

Đây là lựa chọn (a) trong hai đánh đổi mà mục 5e nêu — **không** phải mở kiểm
tra để chẩn đoán; vẫn chọn ngoài mẫu, chỉ đổi thước đo từ nhị phân sang liên tục.

## 6. TIÊU CHÍ PHỦ ĐỊNH

**Không mở đoạn kiểm tra** nếu không có biến thể EVT nào đạt S thấp hơn mốc V0
trên kiểm định. Hướng EVT đóng lại, ghi là hướng thứ năm thất bại.

Mở kiểm tra **đúng một lần** khi: có E* với `S(E*) < S(V0)` trên kiểm định.
Khi đó phán quyết **dương** đòi **cả ba**:

1. Trên **kiểm tra**, E* có số ô đạt **≥ 5/12** (mốc hiện tại: 4/6 cặp ở mức
   99% — phải nói rõ con số mốc đo lại trong chính lần chạy này).
2. **Và** E* **không** làm hỏng cặp nào mà V0 đang đạt.
3. **Và** ξ > 0 ở ≥ 4/6 cặp (dấu của giả thuyết H10).

Trượt bất kỳ điều nào → **không đổi sản xuất**, ghi kết quả đúng như nó xảy ra.
Không sửa tiêu chí sau khi thấy số.

## 7. Lực phát hiện — khai báo TRƯỚC

Mỗi cặp có ~720 phiên ở mỗi đoạn. Ở α = 1% kỳ vọng **~7 lần vi phạm**. Kupiec/
Christoffersen/DQ ở cỡ mẫu này có **lực rất thấp** — `CHISO_DANHGIA.md` mục 5
đã ghi rõ: *"không bác bỏ được" không có nghĩa "đã chứng minh đúng"*.

Hệ quả, khai báo trước: nếu EVT thắng thì phát biểu mạnh nhất cho phép là
*"EVT đưa tỷ lệ vi phạm và tỷ lệ ES về gần mức danh nghĩa hơn"* — **không**
phát biểu được *"đuôi đã được hiệu chuẩn đúng"*. Và nếu EVT thua, kết luận âm
cũng chỉ mạnh ở mức 7 vi phạm kỳ vọng cho phép.

## 8. Tự kiểm bắt buộc

Chạy trước khi chấm bất kỳ số liệu thật nào, và dừng nếu hỏng:

- GPD khớp trên dữ liệu mô phỏng có ξ, β biết trước: phân vị 1% từ `gpd_duoi`
  lệch dưới **10%** so với nghiệm giải tích, ở ba bộ (ξ, β).
- Đơn điệu: ES phải **sâu hơn hoặc bằng** VaR ở mọi cấu hình.

## 9. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 3 (giả thuyết), mục 5 (quy
tắc chọn liên tục và lý do), mục 6 (tiêu chí phủ định) và mục 7 (khai báo lực)
**trước khi** số liệu được chấm:

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không phải
> công cụ điền hộ.

---

# PHỤ LỤC A — KẾT QUẢ (13/09/2026, chạy SAU khi mục 1–9 đã chốt)

*Tái lập: `python src/va_duoi_evt.py` (chỉ kiểm định) rồi
`python src/va_duoi_evt.py --mo-kiem-tra`. Kết quả: `output/va_duoi_evt.json`.*

**Tự kiểm ĐẠT** trước khi chấm số liệu thật: GPD lệch 0,38–0,46% so nghiệm
giải tích ở ba bộ (ξ, β); ES sâu hơn VaR.

## A1. Hình dạng đuôi — **giả thuyết H10 bị BÁC BỎ**

ξ ước trên huấn luyện (u = phân vị 90, 251 điểm vượt ngưỡng mỗi cặp):

| cặp | ξ | | cặp | ξ |
|---|---|---|---|---|
| EURUSD | **−0,055** | | AUDUSD | **−0,110** |
| GBPUSD | +0,077 | | USDCAD | **−0,107** |
| USDJPY | **−0,039** | | USDCHF | +0,193 |

**ξ > 0 chỉ ở 2/6 cặp.** H10 chốt trước đòi cả 6. Bác bỏ.

Đây không phải chi tiết kỹ thuật — nó là một phát hiện: **sau khi chuẩn hoá
bằng σ̂ của HAR, đuôi dưới của z gần như HÀM SỐ (ξ ≈ 0), không phải luỹ thừa.**
Phần "đuôi dày" mà văn liệu gán cho lợi suất FX phần lớn là **co cụm biến
động**, và HAR đã hút hết phần đó rồi. Nói cách khác: mô hình tầng 2 làm đúng
việc của nó, và cái còn lại trong z thì không dày như người ta tưởng.

## A2. Kiểm định — EVT thắng, nên mở kiểm tra theo đúng mục 6

| phương án | S kiểm định | ô đạt |
|---|---|---|
| V0 *(mốc sản xuất)* | 0,3722 | 8/12 |
| V1 phân vị mở rộng | 0,3252 | 10/12 |
| E1 EVT huấn luyện | 0,2468 | 10/12 |
| **E2 EVT mở rộng** | **0,2308** | 10/12 |
| E3 EVT cuộn 500 | 0,2577 | 9/12 |
| E4 EVT ngưỡng 95% | 0,3146 | 10/12 |

**Cả bốn biến thể EVT đều thắng mốc.** Điều kiện mở kiểm tra thoả → mở **một
lần**, bằng cờ tường minh `--mo-kiem-tra` để lần mở nằm rõ trong lịch sử lệnh.

## A3. PHÁN QUYẾT — **KHÔNG đủ điều kiện dương**

| điều kiện | kết quả | |
|---|---|---|
| ĐK1 ô đạt ≥ 5/12 trên kiểm tra | **ĐẠT** | 8/12 |
| ĐK2 không làm hỏng ô V0 đang đạt | **TRƯỢT** | USDCHF α=5% |
| ĐK3 ξ > 0 ở ≥ 4/6 cặp | **TRƯỢT** | 2/6 |

→ **KHÔNG đổi cấu hình sản xuất.** Không sửa tiêu chí sau khi thấy số.

## A4. Nhưng phải ghi thẳng: trên chính hai ô đang hỏng, EVT vá được

Mức **99%**, đoạn **kiểm tra** — hai cặp mà khiếm khuyết được định nghĩa bằng:

| | vi phạm | Kupiec p | DQ p | tỷ lệ ES |
|---|---|---|---|---|
| **USDJPY** V0 *(đang chạy)* | 2,20% | **0,005** ✗ | **0,000** ✗ | 0,813 |
| USDJPY E2 EVT mở rộng | 1,79% | 0,055 | 0,000 ✗ | 0,842 |
| **USDJPY E3 EVT cuộn 500** | **1,51%** | **0,198** ✓ | 0,006 ✗ | **0,937** |
| **USDCHF** V0 *(đang chạy)* | 1,79% | 0,054 | 0,015 ✗ | 1,131 |
| **USDCHF E1/E2 EVT** | 1,65% | 0,106 | 0,015 ✗ | **1,008** |

Lần đầu tiên trong năm hướng, Kupiec của USDJPY đi từ **bác bỏ (0,005)** sang
**không bác bỏ (0,198)**, và tỷ lệ ES từ 0,813 lên 0,937. USDCHF thì tỷ lệ ES
được vá gần như trọn vẹn: 1,131 → **1,008**.

**Nhưng DQ vẫn bác bỏ ở cả hai.** Đó đúng là điều mục 5e đã chẩn đoán:
Engle–Manganelli bác bỏ vì vi phạm **dự báo được từ vi phạm trước** — tức đuôi
có **cấu trúc ĐỘNG**. EVT sửa *hình dạng* và *độ chính xác* của đuôi, nó không
sửa *động học*. Kết quả này **xác nhận chẩn đoán 5e** bằng một cơ chế độc lập:
ngay cả khi đuôi đã được ước tốt hơn hẳn, DQ vẫn hỏng.

## A5. Theo thước đo liên tục thì EVT thắng cả trên kiểm tra

| phương án | S kiểm tra |
|---|---|
| V0 *(mốc)* | 0,4913 |
| V1 | 0,4120 |
| E1 | 0,4295 |
| E2 | 0,3779 |
| **E3 EVT cuộn 500** | **0,2749** |
| E4 | 0,3977 |

**Cả bốn biến thể EVT đều thấp hơn mốc**, E3 giảm **44%**.

### A5a. LỖI TRONG CHÍNH VĂN BẢN CHỐT TRƯỚC NÀY — phải ghi ra

Mục 5 lập luận dài rằng tiêu chí **nhị phân** đạt/KHÔNG **bị mù** (dẫn mục 5e),
và vì thế quy tắc **chọn** phải liên tục. Nhưng mục 6 lại viết ĐK1 và ĐK2 **bằng
chính đếm nhị phân đó**. Đó là **không nhất quán nội tại**, và nó là lỗi của
tôi khi soạn văn bản, phát hiện ra sau khi đã thấy số.

Xử lý: **giữ nguyên phán quyết theo đúng chữ của mục 6** — không đủ điều kiện
dương, không đổi sản xuất. Sửa tiêu chí sau khi thấy số là việc mà toàn bộ giao
thức này tồn tại để ngăn, và nó không có ngoại lệ cho trường hợp tiêu chí bị
soạn dở.

Nhưng ghi lại đầy đủ để người đọc tự đánh giá: **trên thước đo mà mục 5 lập
luận là đúng, EVT thắng ở cả hai đoạn, và thắng ở đúng hai ô đang hỏng.**

### A5b. Lần thứ tư: chọn trên kiểm định chọn sai

E2 tốt nhất trên kiểm định (S=0,2308) nhưng **E3 tốt nhất trên kiểm tra**
(S=0,2749 so 0,3779 của E2), và E3 mới là biến thể vá được USDJPY. E3 chỉ xếp
**thứ ba** trên kiểm định.

Hình mẫu này nay đã lặp **bốn lần, bốn cơ chế độc lập**: Pha 3B (E3 lọc nhân
quả hơn E2 chọn thường, p=0,0010) · qlikeHAR · mẫu hình nến (kiểm định chọn K2,
thứ sống sót phễu là \|gap\|) · và ở đây.

## A6. Cơ chế thật — khác cơ chế đã chốt trước

Mục 2 nói EVT giúp vì **ngoại suy được cái đuôi dày**. Với ξ ≈ 0 ở 4/6 cặp thì
lý lẽ đó **sai**. Cơ chế thật là phần còn lại của chính lập luận đó:

> EVT dùng **251 điểm vượt ngưỡng** để ước cùng con số mà phân vị thực nghiệm
> ước bằng **~7 thống kê thứ tự**. Lợi ích là **giảm phương sai của ước lượng**,
> không phải ngoại suy hình dạng. Đó là lý do nó giúp dù đuôi không dày.

Hệ quả dùng được: với cỡ mẫu mà luận văn này có, **sai số ước lượng đuôi lớn hơn
sai số dạng hàm**. Đây là phát biểu định lượng về *vì sao* vá đuôi khó ở đây, và
nó giải thích gọn vì sao bốn hướng trước — đều chỉ đổi cửa sổ, không đổi lượng
thông tin — không hướng nào ăn tiền.

## A7. Kết luận cho luận văn

> *Hướng thứ năm (EVT/POT theo sơ đồ McNeil–Frey) là hướng **duy nhất trong
> năm** làm dịch chuyển được backtest đuôi của USDJPY (Kupiec 0,005 → 0,198)
> và vá gần trọn tỷ lệ ES của USDCHF (1,131 → 1,008), và nó thắng mốc theo
> thước đo liên tục ở **cả hai** đoạn. Nhưng nó **không được triển khai**: giả
> thuyết về hình dạng đuôi bị bác bỏ (ξ > 0 chỉ 2/6 cặp), nó làm hỏng một ô
> mốc đang đạt, và DQ vẫn bác bỏ ở cả hai cặp — xác nhận rằng khiếm khuyết là
> **động học đuôi**, thứ mà ước lượng đuôi tốt hơn không sửa được.*
