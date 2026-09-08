# BIÊN BẢN KHÓA SỔ DỮ LIỆU

**Ngày lập:** 28/08/2026
**Luận văn:** Hệ thống hỗ trợ quyết định giao dịch ngoại hối (luận văn chung)
**Lý do lập:** Từ thời điểm này nhóm bắt đầu giai đoạn *lặp cải tiến* pipeline —
thử nhiều cấu hình cho tới khi kết quả tốt. Nếu không tách một phần dữ liệu ra
trước, mọi kết quả cuối cùng đều mang lỗi *data snooping*: mô hình được chọn vì
nó hợp với chính bộ dữ liệu dùng để đánh giá nó.

---

## 1. TẬP PHÁT TRIỂN — được phép dùng tự do

| Mục | Nội dung |
|---|---|
| Cặp tiền | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF |
| Khoảng | 2010-01-03 → 2025-12-31 |
| Số phiên | 4.994 / cặp (29.843 dòng panel) |
| Nguồn | HistData M1 → H1 → D1, đã chuyển giờ New York sang UTC có xử lý DST |

Trên tập này được phép: đổi mô hình, đổi siêu tham số, đổi đặc trưng, đổi hàm
thưởng RL, thiết kế tầng fuzzy, chạy lại bao nhiêu lần tùy ý.

## 2. TẬP KHÓA SỔ — KHÔNG được chạm cho tới lần chạy cuối

| Mục | Nội dung |
|---|---|
| Cặp tiền | EURGBP, EURJPY, GBPJPY, AUDJPY, EURCHF, NZDUSD |
| Khoảng | 2010-01-01 → 2025-12-31 |
| Cộng thêm | 2026-01 → 2026-08, **cả 12 cặp** |

Hai tập này khác nhau về **hai chiều độc lập**: sáu cặp chéo không có USD hai vế
nên kiểm tra được tính tổng quát ngoài nhóm cặp chính; dữ liệu 2026 nằm hoàn toàn
sau mọi phép thử đã thực hiện nên kiểm tra được tính bền theo thời gian.

### Vị trí trên đĩa

| Tập | Thư mục | Ai đọc |
|---|---|---|
| Phát triển | `histdata_raw/` → `fx_clean/` | `prep_fx.py` mặc định |
| **Khóa sổ** | `histdata_seal/` → `fx_seal/` | **không script nào**, cho tới lần chạy cuối |

Hai thư mục tách rời có chủ đích: `prep_fx.py` tự quét `histdata_raw/`, nên nếu
để chung thì dữ liệu khóa sổ sẽ chảy vào pipeline ngay và biên bản này vô nghĩa.
Lần chạy cuối mở niêm phong bằng: `py prep_fx.py --src histdata_seal --out fx_seal`

### Quyết định về dữ liệu — ghi ngày 28/08/2026, TRƯỚC mọi phân tích

**AUDJPY năm 2012 bị loại khỏi tập khóa sổ.**

Phát hiện: kho lưu trữ HistData cho `AUDJPY_2012` chỉ chứa 33.047 dòng phủ
07/10/2012 → 22/10/2012, thay vì ~370.000 dòng cả năm. Đã thử tải lại theo năm
(`--force`) — vẫn cho kết quả như cũ, nên đây là khiếm khuyết của kho lưu trữ
chứ không phải lỗi tải. Tải theo tháng không khả dụng: HistData chỉ mở trang
theo tháng cho năm hiện hành.

Quyết định: **loại năm 2012 của AUDJPY, giữ nguyên cặp này cho 15 năm còn lại.**

Lý do không vá bằng nguồn khác (Dukascopy / Forexite / TrueFX đều có dữ liệu này):
vá sẽ cấy một mối nối giữa hai nhà cung cấp vào giữa chuỗi của một cặp, ngay
trong bộ dữ liệu tồn tại để xác nhận kết luận. Mối nối đó chưa được hiệu chuẩn
cho cặp JPY và có thể tạo bước nhảy giả ở hai điểm nối. Phần dữ liệu mất là
~250 phiên trên tổng ~24.000 của tập khóa sổ (1%), không đủ để đổi kết luận nào.
Một lỗ hổng được khai báo trung thực hơn một mối nối chưa kiểm chứng.

Lưu ý phương pháp luận: quy tắc này **không** được tổng quát thành "loại mọi cặp
có dữ liệu thiếu". Lỗ hổng trong kho dữ liệu tài chính tập trung ở các cặp thanh
khoản thấp và giai đoạn thị trường căng thẳng; loại bỏ theo tiêu chí đó sẽ lọc
mẫu về phía những trường hợp dễ. Ở đây loại đúng một năm của một cặp, vì lý do
kỹ thuật đã xác minh, và được ghi lại trước khi phân tích.

**Rào chắn liền mạch (áp cho cả hai tập).** Mọi cửa sổ trượt — MA20, HAR 5 ngày
và 22 ngày, Yang-Zhang 5 ngày — chỉ được tính khi toàn bộ cửa sổ không bắc qua
khoảng trống lớn hơn 4 ngày lịch. Vượt ngưỡng thì trả về rỗng. Lý do: nhóm đã
từng dính đúng lỗi này ở HAR-RV, khi cửa sổ 22 ngày âm thầm nối hai đoạn cách
nhau nhiều tháng. Đã kiểm chứng tập phát triển không có khoảng trống nào trên
4 ngày (khoảng cách lớn nhất là 3 ngày, tức cuối tuần), nên rào chắn là vô hiệu
ở đó — đúng như mong đợi, và đó chính là phép tự kiểm.

## 3. QUY TẮC

1. Tập khóa sổ được chạy **đúng một lần**, sau khi cấu hình cuối đã chốt và ghi
   vào mục 4 dưới đây.
2. Nếu chạy tập khóa sổ rồi mà kết quả xấu, **không được** quay lại sửa mô hình
   rồi chạy lại. Kết quả xấu là một phát hiện, phải báo cáo đúng như nó xảy ra.
3. Nếu bắt buộc phải chạy lần hai (ví dụ phát hiện lỗi code, không phải kết quả
   xấu), phải ghi lại lý do vào mục 5 và nêu trong luận văn.
4. Mọi cấu hình đã thử trên tập phát triển phải được đếm và ghi vào mục 5, để
   phần báo cáo hiệu chỉnh được đa kiểm định.

## 4. CẤU HÌNH CHỐT — điền trước khi mở tập khóa sổ

**Điền ngày 08/09/2026.** Mọi dòng dưới đây là cấu hình *đang chạy trên
production*, không phải cấu hình mong muốn. Mỗi lựa chọn đều được chốt trên
đoạn KIỂM ĐỊNH và có tài liệu kèm số đo; không mục nào chọn trên đoạn kiểm tra.

### 4.1 Mô hình biến động (tầng 2)

| | |
|---|---|
| Họ mô hình | Tổ hợp **STHARQ + HARQ + SHAR** (`src/volfc2.py`, `MODELS`) |
| Cấu hình chốt | `deseason="none"` · `crosspair=False` · `event="capday"` · `window=None` (cửa sổ **mở rộng**) · `recal="off"` · `lam=0.0` (`CAUHINH_SANXUAT`) |
| Ước lượng | OLS cửa sổ mở rộng, Gram tích luỹ, khớp lại mỗi phiên; log-space + hiệu chỉnh log-chuẩn |
| Rào chắn | cửa sổ trượt bị vô hiệu khi bắc qua khoảng trống > 4 ngày lịch (`MAX_GAP`) |
| Đã loại có đo | khử mùa vụ nội tuần, chéo cặp, co ngót, cửa sổ cuộn — 560 cấu hình quét trên kiểm định (`output/grid2_valid.csv`) |

### 4.2 Nền ba lớp theo tầm hạn (tầng 5 — thứ hiện trên giao diện)

| tầm hạn | mô hình chốt |
|---|---|
| h = 1 | **tổ hợp trực tuyến** (Hedge trơn trên 4 nền, η = 0,5, trễ 1 phiên) |
| h = 5 | **σ̂ + chế độ (cuộn)** — khớp lại mỗi 21 phiên, cửa sổ mở rộng |
| h = 20 | **σ̂ + chế độ (cuộn)** |

(`api/main.py::NEN_THEO_H`). Chế độ = tam phân vị của σ̂, ngưỡng chốt trên
huấn luyện. Đã đo và **không** đổi: Fixed-Share (α = 0 tối ưu), chọn mô hình
theo chế độ (xấu hơn có ý nghĩa ở h=5), hiệu chuẩn lại isotonic/vector scaling.

### 4.3 Tầng VaR/ES (rủi ro)

**V0 — phân vị thực nghiệm của z ước trên HUẤN LUYỆN**, đóng băng
(`src/va_duoi.py`). Giữ nguyên sau khi V1/V2, CAViaR và phân vị-theo-chế-độ đều
không vượt được nó (`CHISO_DANHGIA.md` mục 5c, 5d, 9.2). Cảnh báo USDJPY và
USDCHF **giữ trên giao diện** — 4/6 cặp đạt cả ba phép kiểm ở mức 99%.

### 4.4 Quy tắc định cỡ (tầng 6b)

```
đòn bẩy = min(Kelly, trần phá sản 1%) × 0,92 × k_vol × k_dd × k_danh_mục
```

(`src/position_sizing.py`). `K_SLIP = 0,92` (trượt giá). `k_danh_mục` dùng
ρ = 0,44 mặc định, 0,55 khi σ̂ vượt phân vị 95% — dùng đúng độ phân giải đã đo,
không dùng tam phân vị vì quá thô để thấy hiệu ứng này.

### 4.5 Tham số tầng fuzzy — **KHÔNG có tầng fuzzy trong sản xuất**

Trường này có trong biên bản gốc ngày 28/08/2026 vì lúc đó fuzzy còn nằm trong
kế hoạch. Nó **đã được thử và bị loại có đo**: fuzzy Mamdani hơn một tích đơn
giản của hai hệ số tuyến tính đúng **+0,08%** (`src/compare_sizing.py`), không
đủ để trả cho phần phức tạp thêm. Sản xuất dùng quy tắc tuyến tính ở 4.4.

Ghi lại thay vì để trống, vì một trường bỏ trắng trong biên bản khoá sổ sau này
không phân biệt được với "quên điền".

### 4.6 Chốt

- **Ngày chốt:** 08/09/2026
- **Trạng thái:** cấu hình đã khoá. Từ thời điểm này **không sửa mô hình nữa**;
  mọi thay đổi tiếp theo (nếu có) phải ghi vào mục 5 kèm lý do.
- **Ký xác nhận:** ......................................  /  ....................

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không phải
> công cụ điền hộ. Tập khoá sổ chỉ được mở **sau** khi dòng này có chữ ký.

## 5. NHẬT KÝ THỬ NGHIỆM TRÊN TẬP PHÁT TRIỂN

Quy tắc 4 đòi **đếm** mọi cấu hình đã thử, để phần báo cáo hiệu chỉnh được đa
kiểm định. Bảng dưới đây đếm đến ngày chốt 08/09/2026.

| # | Ngày | Thay đổi gì | Số cấu hình | Kết quả chính |
|---|---|---|---|---|
| 1 | 22/08/2026 | 10 mô hình biến động, walk-forward, MIN_TRAIN=250 | 10 | MA20-GK hạng TB 1,3; HAR thua 6/6 (DM p<0,01) |
| 2 | 28/08/2026 | RL REINFORCE đòn bẩy tuyệt đối, 3 seed | 1 | Biến thiên seed 56% — không dùng được |
| 3 | 28/08/2026 | RL REINFORCE tham số hóa phần dư | 1 | Biến thiên seed 30%; phá sản trung vị 1,3% |
| 4 | 28/08/2026 | PPO tham số hóa phần dư, 6 seed | 1 | Biến thiên seed 3%; k học được = 1,08 |
| 5 | 29/08–02/09 | **Lưới biến động vòng 7** — 5 trục cấu hình × 3 họ mô hình | **560** | Chốt STHARQ+HARQ+SHAR, `deseason=none`, cửa sổ mở rộng (`grid2_valid.csv`) |
| 6 | 02/09/2026 | 14 mô hình ML/DL cho biến động | 14 | Mô hình cây thua **mọi** biến thể HAR (`ML_DL_VONG7.md`) |
| 7 | 03/09/2026 | Bảng nền ba lớp: 7 nền × 2 mục tiêu × 3 tầm hạn | 42 | Chốt `NEN_THEO_H`; BSS h=1 +0,0107 trên kiểm định |
| 8 | 04/09/2026 | ML/DL ba lớp (logistic, LightGBM, GRU) + Hedge | 4 | LightGBM −0,0149 · GRU −0,1083 — trần GBM **dưới** nền |
| 9 | 04–05/09 | Định cỡ: Kelly, trần phá sản, fuzzy Mamdani, tích tuyến tính | 4 | Fuzzy hơn tích tuyến tính +0,08% → **loại**; chốt quy tắc 4.4 |
| 10 | 05/09/2026 | Vá đuôi VaR/ES: V0 / V1 mở rộng / V2 cuộn | 3 | V1 thua V0 trên kiểm tra (8/12 so 9/12) → **giữ V0** |
| 11 | 05/09/2026 | Hiệu chuẩn lại: nhiệt độ 1 tham số, vector scaling | 2 | h=20 xấu đi trên kiểm tra → **không triển khai** |
| 12 | 04–06/09 | **Giai đoạn 2 — khai phá quy luật**, 3 nhánh đầu | **2.577 giả thuyết** | SAX biến động 41 sống sót (2 độc lập HAR); SAX hướng 0; ngưỡng đặc trưng 0 |
| 13 | 06–07/09 | Lan truyền biến động chéo cặp (Diebold–Yilmaz), D1 và H1 | 2 | Không cải thiện; tệ hơn ở H1 |
| 14 | 07/09/2026 | Phản ứng quanh sự kiện vĩ mô, mở rộng 7 → 18 loại | 18 | 0/18 có thiên lệch hướng có ý nghĩa |
| 15 | 08/09/2026 | Ba phương pháp hiện đại: FDR(BY), CAViaR (4 biến thể), Fixed-Share (9 mức α) | 14 | Cả ba **không ăn tiền** — giữ W-Y, phân vị tĩnh, Hedge trơn |
| 16 | 08/09/2026 | A1 hướng cuối: phân vị đuôi theo chế độ (2 biến thể) | 2 | Kém hơn mốc → **A1 đóng lại** |
| 17 | 08/09/2026 | A3: chọn mô hình theo chế độ (chế độ vừa → khí hậu học) | 1 | Xấu hơn **có ý nghĩa** ở h=5 → giữ nguyên |
| 18 | 08/09/2026 | Hansen SPA cho họ mô hình Giai đoạn 1 | — | Bác bỏ H0 ở h=5, h=20 (nền đang chạy **thắng** "chỉ σ̂") |
| 19 | 08/09/2026 | **Giai đoạn 2 — ba họ còn thiếu**: H2 motif, H3 rule-list, H5 chế độ tự tương quan | **5.766 giả thuyết** | 0 quy luật; SPA không bác bỏ ở cả ba họ |
| 20 | 08/09/2026 | **Lỗi bịa lợi suất bằng 0** — phát hiện, sửa, chạy lại toàn bộ Giai đoạn 2 | — | Rút lại 1 quy luật sai; **lực phễu 1,35 → 1,20**; Giai đoạn 1 **không đổi** (đã kiểm chứng 2 lần độc lập) |
| 21 | 08/09/2026 | **H6 HMM** (K = 2, 3, 4) và **H7 Matrix Profile** (K=20 analog, L=5/10/20) | **126 giả thuyết** | 0 quy luật; SPA p = 1,000 và 0,972 |
| 22 | 08/09/2026 | Walk-forward theo từng năm, 3 tầm hạn | — | h=1 dương **14/14 năm**; h=20 chỉ 8/14 |

### Tổng kết cho phần hiệu chỉnh bội của luận văn

| khoản | số lượng |
|---|---|
| Cấu hình **mô hình** đã thử trên tập phát triển | **~680** |
| Giả thuyết **quy luật** đã liệt kê và kiểm định | **8.469** (8 nhánh độc lập) |
| Quy luật sống sót toàn bộ phễu bốn cửa | **0** |
| Lực phát hiện của phễu (MDES, lực 80%) | lift **1,20** |
| Dương tính giả trên nhiễu thuần (đối chứng âm) | **0,0 / 1.890** |

Con số 8.469 giả thuyết là con số **biết trước và liệt kê đầy đủ**, không phải
đếm ngược sau khi chạy — đó là điều kiện để Westfall–Young có nghĩa. Con số ~680
cấu hình mô hình là lý do mọi kết luận về **mô hình** đều chỉ được phát biểu trên
đoạn kiểm định, và vì sao tập khoá sổ tồn tại.

---

*Biên bản này lập TRƯỚC khi tập khóa sổ được tải về máy. Đưa vào phụ lục luận văn.*
