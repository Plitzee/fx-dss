# GAS/DCS — QUY MÔ ĐỘNG THEO SCORE CHO ĐUÔI. Tiêu chí CHỐT TRƯỚC

Lập 14/09/2026. Nhánh `replan-2026`. **Commit chứa văn bản này nằm trước mọi
commit có số liệu.**

---

## 1. Khiếm khuyết nhắm tới — khiếm khuyết DUY NHẤT còn mở hoàn toàn

Sau **sáu** hướng độc lập cho đuôi VaR/ES của USDJPY/USDCHF (mục 5b–5e
`CHISO_DANHGIA.md`; EVT `DUOI_EVT.md`), tất cả đều thất bại ở đúng một phép
kiểm: **DQ (Engle–Manganelli 2004) vẫn bác bỏ** — vi phạm VaR dự đoán được
từ vi phạm trước đó, tức đuôi có **động học** mà chưa cách nào bắt được.

| hướng | cơ chế | kết quả DQ |
|---|---|---|
| V1 mở rộng / V2 cuộn | cửa sổ, khớp lại định kỳ | bác bỏ |
| CAViaR (4 biến thể) | tự hồi quy tham số trên quantile | bác bỏ, thua ngay trên kiểm định |
| Phân vị theo chế độ | phân tầng theo σ̂ | bác bỏ |
| Cửa sổ họp NHTW | phân tầng theo lịch | bác bỏ |
| **EVT/POT** | phân phối GPD tĩnh, khớp lại định kỳ | **USDJPY: Kupiec 0,005→0,198 (vá được!) nhưng DQ vẫn bác bỏ (0,006)** |

EVT (`DUOI_EVT.md` A4) là hướng gần nhất — nó vá được **tần suất** vi phạm
nhưng không vá được **động học**. Đây là khiếm khuyết cuối cùng còn mở sau
sáu hướng.

## 2. Vì sao GAS/DCS khác hẳn CAViaR (đã thất bại)

CAViaR tự hồi quy **trực tiếp trên mức quantile**, dạng
`VaR_t = a + b·VaR_{t-1} + c·|r_{t-1}|` — một đặc tả **tuỳ ý** (ad hoc), cập
nhật theo độ lớn residual thô, không có nền tảng suy diễn xác suất.

**GAS (Generalized Autoregressive Score, Creal, Koopman & Lucas 2013,
*Journal of Applied Econometrics* 28(5):777–795)** cập nhật tham số theo
**điểm số (score)** — đạo hàm của log-likelihood — được chứng minh là hướng
cập nhật **tối ưu cục bộ theo Kullback–Leibler** cho một họ phân phối đã
chọn. Với z chuẩn hoá theo phân phối Student-t (đã dùng xuyên suốt hệ thống,
`decision_record.py`), công thức cụ thể là **Beta-t-EGARCH**
(Harvey & Chakravarty 2008; Harvey 2013, *Dynamic Models for Volatility and
Heavy Tails*, Cambridge):

```
ln(s_{t+1}) = ω + α·u_t + β·ln(s_t)
u_t = [(ν+1)·z_t² / (ν·s_t² + z_t²)] − 1        (điểm số đã chuẩn hoá)
```

Điểm mấu chốt: `u_t` **bị chặn** (bounded) — khi `z_t` cực đoan, `u_t → ν`,
không phát nổ như cập nhật kiểu GARCH bậc hai (`z_t²`). Đây chính là cơ chế
**bền vững** (robust) với ngoại lai mà CAViaR không có, và là lý do nó có cơ
hội bắt được động học đuôi mà không bị chính các cú sốc lớn (thứ đang gây ra
DQ) làm hỏng việc ước lượng.

## 3. Giả thuyết CHỐT TRƯỚC

> **H16.** GAS-t cho tỷ lệ vi phạm VaR/ES **và** kiểm định DQ tốt hơn mốc V0
> (phân vị thực nghiệm tĩnh) — đặc biệt ở USDJPY/USDCHF, nơi DQ đã bác bỏ ở
> mọi hướng trước.
>
> **Điều kiện quan trọng nhất, khác mọi hướng trước:** đây là hướng ĐẦU TIÊN
> nhắm trực tiếp vào **DQ**, không chỉ tần suất vi phạm. Nếu GAS-t đạt tần
> suất tốt nhưng DQ vẫn bác bỏ (như EVT), phải ghi nhận **không đạt mục tiêu
> chính** dù các chỉ số khác có cải thiện.

## 4. Ba cấu hình — liệt kê đầy đủ TRƯỚC khi chạy

Cùng cấu trúc V0/V1/V2 đã dùng cho mọi hướng đuôi trước — để so sánh trực
tiếp được:

| | ν (bậc tự do) | ω, α, β | khớp lại |
|---|---|---|---|
| **GAS mở rộng** | ước trên huấn luyện, đóng băng | MLE, khớp lại mỗi 21 phiên trên cửa sổ mở rộng | mỗi 21 phiên |
| **GAS đóng băng** | ước trên huấn luyện, đóng băng | MLE một lần trên huấn luyện, đóng băng | không |
| **GAS cuộn 500** | ước trên huấn luyện, đóng băng | MLE, khớp lại mỗi 21 phiên trên cửa sổ cuộn 500 phiên | mỗi 21 phiên, cuộn |

`ν` ước một lần bằng `scipy.stats.t.fit` trên `z` huấn luyện (đúng cách
`decision_record.py` đã làm), giữ cố định cho cả ba biến thể — để phép so
chỉ khác nhau ở phần **động học** (ω, α, β và cách khớp lại), không lẫn với
khác biệt do ước lại bậc tự do.

`ω, α, β` ước bằng cực đại hoá log-likelihood Student-t có đệ quy, giới hạn
`α ∈ [0, 2]`, `β ∈ [0, 0,999]` (ổn định), khởi tạo `s_0` = độ lệch chuẩn mẫu
của đoạn khớp.

**Đếm vào `KHOA_SO.md`: 3 cấu hình mới.**

## 5. Giao thức chấm

Y hệt mọi hướng đuôi trước: `s_t` (đã ước) thay cho phân vị thực nghiệm để
tính `VaR = s_t · t_ν^{-1}(α)`, `ES = s_t · ES_ν(α)` (công thức đóng cho
Student-t chuẩn). Backtest bằng **Kupiec, Christoffersen, DQ** trên **kiểm
định**, chọn theo điểm liên tục **giống hệt `DUOI_EVT.md` mục 5** (đã chứng
minh nhị phân đạt/KHÔNG bị mù ở USDJPY):

```
S = trung bình trên (α, cặp) của |tỷ lệ vi phạm/α − 1| + |tỷ lệ ES − 1|
```

cộng thêm **một số hạng DQ** vào điểm số, vì đây là mục tiêu chính của
hướng này (khác EVT — EVT không có mục tiêu DQ nên không cần số hạng này):

```
S_DQ = S + trung bình trên (α, cặp) của 1{DQ p < 0,05}
```

Chọn cấu hình có `S_DQ` thấp nhất trên **kiểm định**, chấm **một lần** trên
**kiểm tra**.

## 6. Chống rò rỉ

`s_t` (dự báo cho phiên t) chỉ dùng `z_1, ..., z_{t-1}` qua đệ quy GAS — tự
kiểm bắt buộc: cắt bỏ toàn bộ dữ liệu sau một mốc không được làm đổi `s_t`
tại các phiên trước mốc đó (đúng phép tự kiểm đã dùng cho mọi hướng trước).

`noi_chuoi()` nối dữ liệu live tới 2026-09 — **cắt tại 2025-12-31** trước khi
chấm (đúng `KHOA_SO.md` mục 2, sửa đúng lỗ hổng đã ghi ở `RUIRO_ML.md` mục
A4 cho `va_duoi.py`/`va_duoi_evt.py`).

## 7. TIÊU CHÍ PHỦ ĐỊNH

Kết luận **"GAS/DCS không mang lại giá trị đo được"** khi bất kỳ điều nào
đúng:

1. Cấu hình tốt nhất trên kiểm định không có `S_DQ` thấp hơn mốc V0 —
   **không mở kiểm tra**.
2. Trên kiểm tra: **DQ vẫn bác bỏ ở USDJPY VÀ USDCHF** (mục tiêu chính không
   đạt, dù các chỉ số khác cải thiện).
3. Hoặc làm hỏng ≥ 1 cặp mà V0 đang đạt.

Kết luận **dương** đòi: mở kiểm tra (điều kiện 1 không xảy ra), **và** DQ
không bác bỏ ở **ít nhất một trong hai** USDJPY/USDCHF (cải thiện thật sự so
với toàn bộ sáu hướng trước — không cần cả hai, vì đây là mục tiêu chưa từng
đạt được dù một phần), **và** không làm hỏng cặp nào V0 đang đạt.

Không sửa tiêu chí sau khi thấy số.

## 8. Lực phát hiện — khai báo TRƯỚC

Cùng ~720 phiên mỗi đoạn, ~7 vi phạm kỳ vọng ở α=1% — lực thấp cho Kupiec,
nhưng **DQ dùng toàn bộ chuỗi vi phạm** (không chỉ đếm tần suất) nên có lực
cao hơn hẳn cho đúng câu hỏi "có động học không". Đây là lý do DQ được chọn
làm mục tiêu chính thay vì chỉ tần suất.

## 9. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 3 (giả thuyết, đặc biệt
điều kiện DQ), mục 7 (tiêu chí phủ định) và mục 8 (khai báo lực) **trước
khi** số liệu được chấm:

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không
> phải công cụ điền hộ.
