# MẪU HÌNH NẾN — TIÊU CHÍ CHỐT TRƯỚC

*Lập 12/09/2026. Nhánh `replan-2026`.
**Commit chứa file này phải nằm TRƯỚC mọi commit chứa số liệu.***

---

## 0. Vì sao đây là lỗ hổng thật, không phải việc lặp lại

Repo đã khai phá mẫu hình rất nhiều — 8.652 giả thuyết, 12 nhánh — nhưng **mọi
nhánh đều dùng chuỗi đóng-đóng hoặc đại lượng từ nến 5 phút**:

| nhánh đã làm | dùng gì |
|---|---|
| SAX biến động / SAX hướng | lợi suất đóng-đóng rời rạc hoá theo tercile |
| H2 motif / H7 Matrix Profile | đường giá **chuẩn hoá z** (chỉ close) |
| H3 rule-list, H5 chế độ, H6 HMM | đặc trưng kỹ thuật từ close |
| Tầng 2 (HAR sản xuất) | `rv5, rq5, bpv5, rsp, rsn` — **toàn bộ từ nến 5 phút** |

**Hình học OHLC của nến NGÀY chưa bao giờ vào mô hình nào.** Bốn cột
`open/high/low/close` có trong bảng sản xuất nhưng `volfc2.thiet_ke` không
dùng cột nào. `vol.py` có sẵn Parkinson / Garman-Klass / Rogers-Satchell nhưng
chúng chỉ phục vụ panel cũ (MA20-GK) đã bị thay từ vòng 7.

Grep toàn repo cho `doji|hammer|engulf|harami|marubozu|candlestick`: **1 lần
duy nhất**, và là trong đặc tả *vẽ* biểu đồ (`PROMPT_REPLAN_EN.md` dòng 204).

## 0a. Văn liệu — hai tín hiệu ngược chiều, và chúng định hình phép thử

**Trục hướng — đồng thuận ÂM, mạnh:**

- **Hernández-Nieves et al. (2020), *Mathematics* 8(5):802** — *"Predictive
  Power of Adaptive Candlestick Patterns in Forex Market. EURUSD Case"*. Đúng
  miền (FX, EURUSD), đúng cách tiếp cận (suy luận thống kê): **không tìm được
  lợi suất trung bình dương ròng trong bất kỳ trường hợp nào sau chi phí giao
  dịch.**
- Kiểm định trên 29 mã OMXS30 (2007–2015): không có sức dự báo.
- Phân tích 02/2026: phân phối lợi suất tích luỹ sau mẫu hình **không phân biệt
  được với vào lệnh ngẫu nhiên**.

**Nhưng có một luồng khớp đúng trục mà repo CÓ kỹ năng:**

Văn liệu 2026 về học sâu trên nến nhấn rằng **bóng nến mang thông tin biến động
trong thanh**, và *"candlesticks contain mode information reflecting the shape
of the curve and volatility patterns within a time unit; if candlestick data
were processed separately, this mode information would be lost"*
(ScienceDirect S1059056026002716; Springer 978-3-032-23544-2_33).

Đó là lý do phép thử này **không chỉ chạy trên trục hướng**.

## 0b. Cơ chế cụ thể, có thể sai, và đo được

Ba cơ chế khiến hình học OHLC ngày có thể mang thông tin mà `rv5` không có:

1. **RV 5 phút hụt cực trị trong thanh.** `rv5` cộng bình phương lợi suất từng
   thanh 5 phút; một cú nhọn *bên trong* một thanh 5 phút không vào `rv5`
   nhưng **có** vào `high`/`low` của ngày. Nên chênh lệch giữa ước lượng theo
   phạm vi (Parkinson/GK/RS) và `rv5` là một đại lượng đo được.
2. **Gap qua đêm bị loại theo thiết kế.** `DATASET.md` ghi rõ RV *"không chứa
   gap qua đêm — nhưng gap qua đêm chỉ chiếm 1,7–3,1% tổng phương sai ở FX nên
   bỏ qua được"*. Đó là một **giả định chưa ai kiểm**.
3. **Vị trí đóng trong phạm vi là thông tin thứ tự, không phải phương sai.**
   `rv5`, `bpv5`, `rsp`, `rsn` đều bất biến với hoán vị thứ tự các lợi suất
   trong ngày. `(C−L)/(H−L)` thì không. Đây là chiều thông tin mà **không** đại
   lượng nào của tầng 2 chạm tới.

**Đối trọng phải ghi trước:** `PHA2_KETQUA.md` mục 3d đã thử **độ tập trung RV
trong ngày** (hhi, top1, top12 từ 288 bin 5 phút) và ra **ÂM**, vì `hhi` tương
quan **+0,742** với tỉ trọng nhảy mà HAR đã có. Nếu hình học nến cũng chỉ là
thành phần nhảy trá hình thì kết quả sẽ lặp lại. Cơ chế 3 là cơ chế **duy
nhất** không bị lập luận đó bác trước.

---

## 1. Không gian giả thuyết — liệt kê đầy đủ TRƯỚC khi chạy

### 1a. Bốn họ, 25 đặc trưng

**K1 · Hình học nến chuẩn hoá** (5) — chia cho phạm vi nên vô thứ nguyên:

| đặc trưng | định nghĩa |
|---|---|
| `than` | \|C−O\| / (H−L) |
| `bong_tren` | (H − max(O,C)) / (H−L) |
| `bong_duoi` | (min(O,C) − L) / (H−L) |
| `vi_tri_dong` | (C−L) / (H−L) |
| `huong` | dấu(C−O) |

**K2 · Chênh lệch ước lượng phạm vi so với RV 5 phút** (3) — cơ chế 1:

`log(park/rv5)` · `log(gk/rv5)` · `log(rs/rv5)`

**K3 · Mẫu hình nến có tên** (15) — chỉ báo nhị phân, định nghĩa cổ điển:

*một thanh (5):* `doji` · `bua` (hammer) · `sao_bang` (shooting star) ·
`marubozu` · `con_quay` (spinning top)
*hai thanh (6):* `nhan_chim_tang` · `nhan_chim_giam` (engulfing) ·
`harami_tang` · `harami_giam` · `xuyen_tham` (piercing) · `may_den` (dark cloud)
*ba thanh (4):* `sao_mai` (morning star) · `sao_hom` (evening star) ·
`ba_linh_trang` · `ba_qua_den`

**K4 · Gap qua đêm** (2) — cơ chế 2: `gap` = log(O_t / C_{t−1}) · `abs_gap`

**Tổng: 25 đặc trưng.** Ngưỡng của K3 (ví dụ "thân < 5% phạm vi" cho doji) chốt
theo quy ước cổ điển, **không** hiệu chỉnh theo dữ liệu.

### 1b. Không gian đầy đủ

```
25 đặc trưng × 2 lag {1, 2} × 6 cặp × 2 trục = 600 phép kiểm
```

Cộng **8 cấu hình ablation** (mốc + 4 họ riêng + gộp K1K2K4 + gộp tất cả +
tương tác chế độ). Đếm vào `KHOA_SO.md` sau khi chạy.

### 1c. Hai trục, mỗi trục một mốc riêng

| trục | y | mốc ("past Y / baseline state") |
|---|---|---|
| **biên độ** | log rv5(t+1) | log ĥ_HAR(t) — dự báo sản xuất |
| **hướng** | zT(t+1) | zT(t), log σ̂(t) |

---

## 2. Chống rò rỉ — chốt trước

Đặc trưng tại phiên `t` với lag `L` dùng **nến của phiên t−L**, `L ≥ 1`. Không
bao giờ dùng nến của chính phiên `t`. Tự kiểm bắt buộc: cắt bỏ toàn bộ dữ liệu
sau một mốc **không được làm đổi** giá trị đặc trưng của bất kỳ phiên nào trước
mốc. Ngưỡng đạt: **0 giá trị đổi**.

Riêng `gap` = log(O_t / C_{t−1}) dùng giá mở cửa của phiên t — nên với lag 1 nó
là gap của phiên t−1, hoàn toàn nhân quả.

---

## 3. Kiểm soát bội

Y hệt Pha 3B, dùng lại bộ máy đã kiểm (`pha3b_granger`):
Westfall–Young **maxT từng bước xuống**, null hoán vị **khối 5 phiên**,
**1.000 hoán vị**. FDR-BH báo cáo song song làm tham chiếu, **không** dùng làm
cửa.

## 4. Màn lọc độ vững

Ứng viên sống sót W-Y còn phải qua: dấu nhất quán ≥ **5/6 cặp**, và ≥ **4/5
năm** của đoạn huấn luyện+kiểm định.

---

## 5. TIÊU CHÍ PHÁN QUYẾT — chốt trước

### 5a. DƯƠNG đòi **cả bốn**

1. Có ≥ 1 đặc trưng sống sót Westfall–Young **và** màn lọc độ vững.
2. **Và** cấu hình ablation tốt nhất trên **kiểm định** thắng mốc trên **kiểm
   tra** với DM p < 0,05 **sau Bonferroni cho 2 trục** (p thô < 0,025).
3. **Và** ≥ 5/6 cặp cải thiện trên trục đó.
4. **Và** dấu hệ số khớp cơ chế đã khai báo ở mục 0b.

### 5b. ÂM ĐÁNG TIN khi **tất cả**

1. Không đặc trưng nào qua cả W-Y lẫn độ vững.
2. **Và** không cấu hình ablation nào thắng mốc theo ngưỡng 5a.2.
3. **Và** MDES đã báo cáo.

### 5c. Điều kiện dừng

Không thêm mẫu hình nào ngoài 25 đặc trưng mục 1a. Không đổi ngưỡng của K3 sau
khi thấy số. Không mở tập niêm phong.

---

## 6. Khai báo lực — TRƯỚC

Đặc trưng theo **ngày**, nên số quan sát độc lập = số phiên: ~2.500 / ~547 /
~548 mỗi cặp. Ngang Pha 3B, mạnh hơn hẳn Pha 3 cũ (26 tháng).

Giới hạn phải ghi: K3 là **chỉ báo nhị phân hiếm** — một số mẫu ba thanh có thể
chỉ khớp vài chục lần mỗi cặp. Với những mẫu đó, lực thấp theo thiết kế, và
kết luận âm chỉ phát biểu được ở mức *"không phát hiện được với số lần khớp
này"*. Số lần khớp của từng mẫu phải in ra cùng kết quả.

---

## 7. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 1 (không gian giả thuyết),
mục 5 (tiêu chí phán quyết) và mục 6 (khai báo lực) **trước khi** số liệu được
chấm:

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích.
