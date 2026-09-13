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
