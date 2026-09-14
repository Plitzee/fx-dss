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
| 23 | 09/09/2026 | **TabPFN v2** (mô hình nền cho dữ liệu bảng, Nature 2025) | 1 | +0,0085 — **ML tốt nhất từng thử**, vượt xa LightGBM/GRU, nhưng vẫn thua nền σ̂ (+0,0110) → không đổi mô hình |
| 24 | 09/09/2026 | **Conformal prediction**: 2 điểm số (LAC, APS) × 2 giao thức (tĩnh, ACI) | 4 | **ACI ĐẠT** — độ phủ 0,901–0,908 ở mọi tầm hạn; tĩnh hỏng (0,814–1,000) → **áp dụng vào sản xuất** |
| 25 | 09/09/2026 | REINFORCE so với giám sát cho xác suất hướng, cùng kiến trúc (`kiem_rl_xacsuat.py`) | 2 | REINFORCE sụp đổ (BSS −0,576) so với giám sát (+0,0053) → không dùng RL cho bài toán này |
| 26 | 09/09/2026 | Chronos-bolt-small (mô hình nền chuỗi thời gian, zero-shot) cho biến động vòng 7 | 1 | Thua HAR v7 ~17–19% (kiểm định +19,0%, kiểm tra +16,8%) — khớp Brini (2607.05291) |
| 27 | 10/09/2026 | TTM (Tiny Time Mixers, zero-shot), 2 cách hiệu chỉnh (log-chuẩn đơn giản + hồi quy Mincer-Zarnowitz) | 2 | Cũng thua HAR ~17–19%, gần giống hệt Chronos; MZ không cải thiện đáng kể so với hiệu chỉnh đơn giản |
| 28 | 10/09/2026 | Mở rộng họ mô hình vòng 7: XGBoost (3 hp), CatBoost (2 hp) | 5 | CatBoost 0,1192, XGBoost 0,1214 (kiểm định) — cạnh tranh nhưng không vượt LightGBM/Ridge |
| 29 | 10–11/09/2026 | TabPFN v2 cho HỒI QUY biến động vòng 7 (khác dòng 23 — đó là phân loại P/R), ngữ cảnh 8k, khớp lại theo năm, GPU | 1 | QLIKE kiểm định 0,1164 — gần bằng HAR (0,1162), tốt nhất trong các mô hình bảng/cây đã thử |
| 30 | 10/09/2026 | Tổ hợp dự báo vòng 7 — 3 cách kết hợp (đều tay, trọng số 1/QLIKE, hồi quy Granger-Ramanathan) × mọi tập con của {HAR, LightGBM, GRU, LSTM, Ridge} | 46 | Tốt nhất ban đầu: HAR+GRU+LSTM đều tay, thắng HAR nhưng MCS (α=0,10) vẫn giữ HAR trong tập không phân biệt được |
| 31 | 10/09/2026 | Mở rộng tổ hợp — thêm XGBoost/CatBoost/TabPFN vào tập ứng viên + stacking phi tuyến (LightGBM meta-learner), mọi tập con × 4 cách kết hợp | 509 | Tốt nhất: hồi quy GR của HAR+GRU+CatBoost (kiểm tra −3,0% so HAR, DM p=0,041 — tổ hợp DUY NHẤT có ý nghĩa riêng lẻ); **stacking phi tuyến TỆ NHẤT** (#383/509, tệ hơn HAR đơn); MCS vẫn giữ HAR trong tập |
| 32 | 09/09/2026 | **H8 — nội dung thông cáo FOMC** (Pha 2, `run_h8_tintuc.py`): TF-IDF cosine, đổi độ dài, giọng điệu HAWK/DOVE | **54 giả thuyết** | 0 quy luật qua phễu; SPA p=0,162 — thấp nhất trong mọi họ đã thử nhưng xa ngưỡng 0,05 |
| 33 | 11/09/2026 | **H8b — embedding ngữ nghĩa thông cáo FOMC** (Pha 2, `run_h8b_embedding.py`): sentence-transformers pretrained, PCA 3 thành phần | **54 giả thuyết** | 0 quy luật qua phễu; SPA p=0,861 — TỆ HƠN đặc trưng thủ công (H8), không hỗ trợ giả thuyết "embedding tốt hơn thủ công" |
| 34 | 11/09/2026 | **MDES cho H8+H8b** (`kiem_pheu_h8.py`): đối chứng âm/dương, tiêm hiệu ứng biết trước vào 126 vị từ gộp | — | lift ≥ **1,35** đạt lực 80% (yếu hơn lift 1,20 của phễu Giai đoạn 1, do mẫu chỉ 129 thông cáo); đối chứng âm 0,0/108 — phễu không rò rỉ |
| 35 | 11/09/2026 | **Vá đuôi VaR/ES — hướng thứ tư**: phân vị điều kiện theo cửa sổ K=5 phiên sau họp NHTW/FOMC (`va_duoi_nhtw.py`), 2 biến thể | 2 | KHÔNG thắng mốc V1 vô điều kiện trên kiểm định (3-5/6 so 5/6, làm hỏng thêm EURUSD/AUDUSD); USDJPY đạt ở MỌI phương án trên kiểm định — không mở đoạn kiểm tra theo đúng quy tắc. Phát hiện thêm: giao thức chọn-trên-kiểm-định về cấu trúc không thể phân biệt cách vá cho lỗi chỉ hiện ở kiểm tra (xem `CHISO_DANHGIA.md` mục 5e) |

| 36 | 11/09/2026 | **H8c — chủ đề chính thông cáo FOMC** (Pha 2, `run_h8c_chude.py`): 4 chủ đề theo từ điển chốt trước | **24 giả thuyết** | 0 quy luật; SPA p=0,669. Phân bố rất lệch (109/129 kỳ là "lạm phát") |
| 37 | 11/09/2026 | **H8e — trích xuất có cấu trúc bằng LLM** (Pha 2, `run_h8e_llm.py`): schema 4 trường CHỐT TRƯỚC và commit trước khi đọc văn bản (commit cc4df90); nhãn do chính model phiên này đọc toàn văn 129 thông cáo, lưu ở `data/tin_tuc/fomc_llm_nhan.json` | **51 giả thuyết** | 0 quy luật; **SPA p=1,000 — YẾU NHẤT trong 4 cách biểu diễn văn bản**. Thứ tự đơn điệu: càng hiểu sâu văn bản, tín hiệu càng yếu (xem `PHA2_TINTUC.md` mục 8b) |

| 38 | 11/09/2026 | **M1 vs M2 trên trục BIẾN ĐỘNG** (`run_m2_bien_dong.py`): 4 cách biểu diễn văn bản (nhãn LLM / trục ngữ nghĩa / PCA embedding / thủ công) đưa vào mô hình học được, thay vì phễu quy luật | 5 | KHÔNG cách nào thắng M1. Hai cách tốt nhất trên kiểm định đều tệ hơn trên kiểm tra (overfit mẫu chọn 510 hàng) |
| 39 | 11/09/2026 | **Bất ngờ chính sách market-implied** (`run_m2_batngo.py`), dữ liệu USMPD/SF Fed (Acosta et al. 2025) — cửa sổ 100 phút quanh công bố FOMC | 7 | **KẾT QUẢ DƯƠNG ĐẦU TIÊN của nhánh tin tức**: |phản ứng EURUSD| thắng M1 −2,7% trên kiểm tra (DM p=0,0009), chọn đúng luật trên kiểm định trước; 6/6 cặp cải thiện. Nhưng |MP1| thuần lãi suất THẤT BẠI (+2,5%). Xem `PHA2_TINTUC.md` mục 8c–8d |
| 40 | 11/09/2026 | **Độ vững của S4** (`run_m2_vung.py`) — CHẨN ĐOÁN của mô hình đã chấm ở dòng 39, biến GIỮ NGUYÊN | **0** (không cấu hình mới, không chọn lại) | Vững: 4/5 năm · 3/3 nhóm độ mạnh · walk-forward −2,1% (p=0,0031). **Nhưng hệ số ÂM (−0,122, ổn định mọi cửa sổ) trong khi giả thuyết chốt trước là DƯƠNG** → cơ chế đăng ký bị bác bỏ; biến hoạt động như chiết khấu phần ngoại suy thừa của HAR, không phải thước đo cú sốc. Xem `PHA2_TINTUC.md` mục 8e |
| 41 | 11/09/2026 | **Đóng băng Pha 2** — `docs/PHA2_KETQUA.md` (biên bản PHASE2_NEWS_RESULT theo roadmap Week 3–4) | — | Quyết định TÁCH ĐÔI: **STOP** nhánh nội dung văn bản (kết luận âm đáng tin, 183 giả thuyết, MDES lift ≥1,35); **CHUYỂN** phát hiện S4 sang Tầng 2 vì cơ chế thật là vi cấu trúc trong ngày, không phải tín hiệu tin tức. **KHÔNG mở bộ niêm phong** |
| 42 | 11/09/2026 | **Độ tập trung RV trong ngày** (Tầng 2, `xay_tap_trung.py` + `run_tang2_taptrung.py`), dựng lại từ 34,9tr nến M1 gốc; giả thuyết CHỐT TRƯỚC ở commit 07bbe1b | 7 | **ÂM.** 0/6 biến thể thắng mốc HAR sản xuất (+0,34% đến +0,76% trên kiểm tra, mọi p>0,09). Dấu hệ số ĐÚNG như chốt trước (đều âm) nhưng độ lớn bằng không. `hhi` tương quan +0,742 với tỉ trọng nhảy — độ tập trung phần lớn là nhảy trá hình, mà HAR sản xuất đã có. **Hệ quả: hạ mức tin cậy của S4 (dòng 39) và KHÔNG đưa vào hệ thống** — xem `PHA2_KETQUA.md` mục 3d |

| 43 | 12/09/2026 | **Pha 3 — tầng vĩ mô (M3)** trên trục biên độ (`run_pha3_vimo.py`), lãi suất liên ngân hàng 3 tháng từ FRED, trễ 2 tháng; giả thuyết + **tiêu chí phủ định** + **khai báo lực** CHỐT TRƯỚC ở `docs/PHA3_TIEUCHI.md` (commit 5651be0) | 7 | **ÂM — cả ba điều kiện phủ định đều trượt.** Biến thể tốt nhất trên kiểm định chỉ +0,01% trên kiểm tra (p=0,821), 3/6 cặp, và hệ số H9 **âm** (−0,0212) trong khi giả thuyết đòi dương. Tự kiểm rò rỉ 0/14.665. Lặp đúng hình mẫu của S4: tương quan thô **+0,144** nhưng sau khi điều kiện trên HAR thì **đảo dấu** — HAR đã hấp thụ trọn phần thông tin đó. Xem `PHA3_TIEUCHI.md` Phụ lục A |

| 44 | 12/09/2026 | **Pha 3B — CAUSALITY-AWARE đầy đủ theo `03_PHASE_3_CAUSALITY_AWARE.md`** (`collect/ngoai_sinh.py`, `src/pha3b_{dactrung,granger,ablation,mdes}.py`): 11 biến / 6 họ, 9 chuỗi FRED theo **NGÀY**, 6 biến đổi × lag {1,2,5} = 186 đặc trưng; Level 1 hữu ích dự báo → Level 2 Granger (W-Y maxT từng bước, null khối 5, 1.000 hoán vị) → Level 3 PCMCI rút gọn; ablation B0/E1/E2/E3/E4 trên **cả ba trục**. Biến, không gian giả thuyết, tiêu chí phán quyết, khai báo lực, và quyết định niêm phong CHỐT TRƯỚC ở `docs/PHA3B_TIEUCHI.md` (commit e88ae66) | **594 giả thuyết + 5 cấu hình** | **PHÁN QUYẾT TÁCH ĐÔI.** (a) Week 2 **DƯƠNG** — **14/594 sống sót Westfall–Young VÀ màn lọc độ vững** (6/6 cặp, 12/12 năm cùng dấu), lần đầu trong toàn dự án; hệ số thổi phồng đo được **7,1 lần** (214 thô so kỳ vọng 30). Toàn bộ 14 nằm trên **trục biên độ**; trục hướng 0/198 (p W-Y tốt nhất 1,000), trục rủi ro 0/198. (b) Week 3 **ÂM** — không cấu hình nào thắng B0 trên kiểm tra: E1 +19,9% (p<0,0001), E2 +4,0% (p=0,032), E3 +0,8% (p=0,746), E4 +0,2%. (c) **Câu hỏi trung tâm DƯƠNG**: E3 (lọc nhân quả) hơn E2 (chọn thường) **−3,07%, DM p=0,0010**; giảm một nửa độ trôi kiểm định→kiểm tra (10,1 → 5,0 điểm). Cơ chế đã đo: E2 chọn biến vĩ mô đang **trôi** (DGS2 lệch **4,54 sd** giữa hai đoạn), màn lọc độ vững loại chúng. (d) MDES **≈1,0% QLIKE** (lực 91,3% qua cửa W-Y thật), đối chứng âm 0,0%. Kiểm hiệu đính ALFRED: VIXCLS **có** 4/3.914 giá trị bị sửa (0,102%, dưới ngưỡng) — bác giả định "chuỗi thị trường không hiệu đính". Kiểm cắt tương lai 0/3.148.236. **KHÔNG mở niêm phong** (quy tắc chốt trước: chỉ mở nếu DƯƠNG). Xem `docs/PHA3B_KETQUA.md` |

| 44b | 12/09/2026 | **Chẩn đoán** tách `su_kien_phi_nhtw` thành NFP và phần còn lại (chẩn đoán của biến ĐÃ KHAI BÁO, cùng quy ước `run_m2_vung.py`) | **0** (không cấu hình mới) | Phần vững duy nhất là **NFP dùng ngày công bố THẬT** thay cho quy tắc "thứ Sáu đầu tháng" của B0: **−0,43% QLIKE kiểm tra, DM p<0,0001**, tốt hơn trên cả kiểm định lẫn kiểm tra. Các công bố khác NFP thắng kiểm định (0,1525) nhưng **thua kiểm tra** (+2,29%). Đây là bản vá **chất lượng dữ liệu** cho B0, không phải thông tin ngoại sinh mới — đúng như `KEHOACH_2026Q4.md` dự đoán trước khi có dữ liệu |

> **Ghi chú chuẩn bị ký (thêm 14/09/2026, không đổi nội dung nào ở trên):**
> bốn dòng tiếp theo (48–51), ngày 13/09/2026, nằm SAU ngày chốt 08/09/2026 ở
> mục 4.6 — về bản chất đều là "thay đổi sau ngày chốt theo quy tắc 3" giống
> các dòng 45–47 và mục (1)–(4) bên dưới, dù xuất hiện TRƯỚC tiêu đề đó trong
> bảng vì được thêm vào trình tự đánh số liên tục thay vì chèn sau tiêu đề.
> Đã gộp đúng vào bảng tổng kết hiệu chỉnh bội cuối trang (dòng "~1.322").
> Xem mục (5) bổ sung bên dưới mục (4) để có tường thuật đầy đủ cho cả bốn
> dòng này trước khi ký.

| 48 | 13/09/2026 | **Vá đuôi — HƯỚNG THỨ NĂM: EVT/POT** (`va_duoi_evt.py`), sơ đồ McNeil & Frey (2000) *J. Empirical Finance* 7(3–4):271–300 — lọc HAR rồi khớp GPD cho phần vượt ngưỡng của z. Giả thuyết H10, 5 biến thể, **quy tắc chọn LIÊN TỤC** (vì mục 5e đã chứng minh đạt/KHÔNG bị mù) và tiêu chí phủ định CHỐT TRƯỚC ở `docs/DUOI_EVT.md` (commit afa284a). Tự kiểm GPD lệch 0,38–0,46% so nghiệm giải tích | **5** | **KHÔNG đổi sản xuất — nhưng là hướng duy nhất trong năm làm dịch chuyển được đuôi.** (a) **H10 BỊ BÁC BỎ**: ξ > 0 chỉ **2/6** cặp (EURUSD −0,055 · USDJPY −0,039 · AUDUSD −0,110 · USDCAD −0,107). Phát hiện: sau chuẩn hoá bằng σ̂ của HAR, đuôi dưới của z **gần hàm số (ξ≈0), không phải luỹ thừa** — phần “đuôi dày” của FX phần lớn là co cụm biến động, HAR đã hút hết. (b) Kiểm định: **cả 4 biến thể EVT thắng mốc** (S: V0 0,3722 → E2 0,2308), điều kiện mở kiểm tra thoả → mở **một lần** bằng cờ tường minh `--mo-kiem-tra`. (c) Phán quyết: ĐK1 ĐẠT (8/12) · **ĐK2 TRƯỢT** (E2 làm hỏng USDCHF α=5%) · **ĐK3 TRƯỢT** (ξ 2/6) → không đủ điều kiện dương. (d) Nhưng trên **đúng hai ô đang hỏng**: USDJPY α=1% Kupiec **0,005 → 0,198** (E3), tỷ lệ ES 0,813 → 0,937; USDCHF tỷ lệ ES **1,131 → 1,008** (E1/E2). **DQ vẫn bác bỏ cả hai** — xác nhận chẩn đoán 5e bằng cơ chế độc lập: khiếm khuyết là **động học đuôi**, ước lượng đuôi tốt hơn không sửa được. (e) Theo thước đo liên tục EVT thắng **cả trên kiểm tra** (V0 0,4913 → E3 0,2749, giảm 44%). (f) **Khai báo lỗi của chính văn bản chốt trước**: mục 5 lập luận nhị phân bị mù nhưng mục 6 lại viết ĐK1/ĐK2 bằng đếm nhị phân — không nhất quán nội tại. Vẫn giữ phán quyết theo đúng chữ của mục 6; không sửa tiêu chí sau khi thấy số. (g) Cơ chế thật khác cơ chế chốt trước: lợi ích đến từ **giảm phương sai ước lượng** (251 điểm vượt ngưỡng so ~7 thống kê thứ tự), không từ ngoại suy đuôi dày — giải thích vì sao 4 hướng trước, đều chỉ đổi cửa sổ, không hướng nào ăn tiền. (h) Lần thứ **tư** chọn-trên-kiểm-định chọn sai: E2 nhất kiểm định nhưng **E3** mới vá được USDJPY. Kèm đính chính `CHISO_DANHGIA.md` mục 5e (ghi sai V1 thành cấu hình sản xuất; thực tế là V0). Xem `docs/DUOI_EVT.md` |
| 49 | 13/09/2026 | **Bộ lọc chu kỳ nội tuần BCL** (`xay_bcl.py` + `kiem_bcl.py`) — Boudt, Croux & Laurent (2011) *J. Empirical Finance* 18(2):353–367, dựng từ 34,9tr nến M1; hệ số chu kỳ 2.016 ô (7×288) ước **chỉ trên huấn luyện**. Giả thuyết H11, **điều kiện ghi công H11b** (phải hơn đối chứng không lọc) và tiêu chí phủ định CHỐT TRƯỚC ở `docs/BCL_TIEUCHI.md`. Gồm 4 cấu hình chốt trước + **5 cấu hình chẩn đoán thêm sau khi thấy số** (khai báo ở Phụ lục A3/A5) | **9** | **ÂM cho BCL — nhưng phát hiện một khiếm khuyết của chính khung so sánh.** (a) Chu kỳ **có thật**: biên độ max/min 4,06–7,28 ở cả 6 cặp; tự kiểm rò rỉ 0/2016 hệ số đổi. Bộ lọc hoạt động đúng mô tả (phát hiện **ít** nhảy hơn ngưỡng thô: 2,3–3,1 so 4,3–5,9/ngày). (b) **H11b BỊ BÁC BỎ**: P1 có lọc (−6,97%) **kém hơn** P2 không lọc (−8,59%) → **BCL không được ghi công**. (c) **−8,59% là giả**: đối chứng còn thiếu — chỉ thêm `log rv5(t)` thô đã cho **−7,78%**. Gần như toàn bộ cải thiện là "đưa RV ngày t vào hồi quy", không phải tách nhảy. Đã loại trừ rò rỉ trước đó: `rv_kiem` khớp `rv5` panel (trung vị tỉ lệ **1,000**, corr log-log 0,9969), không lệch ngày, đặc trưng không tương quan bất thường với mục tiêu. (d) **PHÁT HIỆN CHÍNH**: khung `log rv(t+1) ~ log h_HAR` — dùng cho **mọi** ablation của dự án (`hhi`, mẫu hình nến, `|gap|`, Pha 3, B0 của Pha 3B) — **thiếu một số hạng đáng ~8% QLIKE** (−7,78% so khung, **−9,78%** so HAR sản xuất chấm trực tiếp). Mọi phát biểu "X cải thiện 0,x%" trước đây đều đo so với mốc tự nó lệch ~8%. (e) **Kiểm lại dưới mốc đã sửa (B1 = B0 + log rv(t))**: `|gap|` **MẠNH LÊN** — −1,20%, **DM p=0,0027** (so −0,67%, p=0,0445 ở khung cũ) → nó **không** là biến đại diện cho RV ngày t mà là thông tin cộng thêm thật. Tách C/J thô **sụp** (−0,48%, p=0,657); C/J BCL **tệ hơn mốc** (+1,35%). (f) **Kỷ luật**: (e) **KHÔNG** đảo phán quyết của `NEN_TIEUCHI.md` — đó là phép thử trong khung KHÁC, đếm là cấu hình mới, phát biểu là bằng chứng **bổ sung**, không cho điểm lại. Nó củng cố khuyến nghị đưa `|gap|` vào danh sách chốt của lần mở niêm phong cuối. Xem `docs/BCL_TIEUCHI.md` Phụ lục A |
| 50 | 13/09/2026 | **Xếp hạng chéo sáu đồng tiền** (`xep_hang_cheo.py`) — hướng `KEHOACH_2026Q4.md` mục 1.3 ghi "repo chưa thử lần nào", đã xác minh bằng grep toàn repo: **không một dòng mã nào**. Quy đổi 6 cặp về lợi suất đồng X so USD, **khử trung bình ngang mỗi phiên** để triệt tiêu nhân tố đô-la (ρ=0,443). 6 tín hiệu, **cửa chi phí spread đặt TRƯỚC**, tiêu chí phủ định và khai báo lực CHỐT TRƯỚC ở `docs/XEPHANG_TIEUCHI.md`. Văn liệu: Menkhoff, Sarno, Schmeling & Schrimpf (2012) *JFE* 106(3); Lustig, Roussanov & Verdelhan (2011) *RFS* 24(11) | **6** | **ÂM — nhánh thứ 14 cho kết luận âm trên trục hướng, và là nhánh duy nhất đo hướng TƯƠNG ĐỐI.** Tự kiểm rò rỉ 0 giá trị đổi. Tín hiệu được chọn (`mom_12m`) cho rank IC kiểm tra **+0,0121 (t=0,58)** và **lỗ ròng 2,02 bp/phiên** sau chi phí. Không tín hiệu nào có lãi ròng dương đạt ý nghĩa. (a) **`mom_12m` là ca overfit kiểm định mẫu mực**: đạt p=0,024 trên kiểm định (IC 0,0458, t=2,21) rồi sụp còn t=0,58 trên kiểm tra — đúng điều ngẫu nhiên phải tạo ra với 6 giả thuyết. (b) **Dấu của H12 bị bác bỏ ở tầm hạn ngắn**: `mom_1m` cho IC **âm** nhất quán cả hai đoạn (−0,029 / −0,038, t=−1,78) — tức **đảo chiều**, không phải động lượng; nhưng lỗ nhiều nhất (−3,20 bp) vì vòng quay cao. (c) **Cửa chi phí làm đúng việc**: `mom_12m` có IC dương mà vẫn lỗ ròng — không đặt cửa trước thì bảng sẽ bị đọc thành "có tín hiệu yếu" thay vì "không dùng được". Giới hạn đã khai báo trước: chỉ 6 đồng → ngưỡng phát hiện IC ≈ 0,038; kết luận âm **không** phát biểu được thành "động lượng tiền tệ không tồn tại". Xem `docs/XEPHANG_TIEUCHI.md` Phụ lục A |
| 51 | 13/09/2026 | **ML trực tiếp trên đại lượng rủi ro** (`ruiro_ml.py`) — hướng `KEHOACH_2026Q4.md` mục 3.2, đã xác minh bằng grep là chưa có dòng mã nào. Hai đích: A `P(chạm stop 1,5σ̂/5 phiên)`, B `P(vi phạm VaR 1%)`; 8 đặc trưng, 2 mô hình, tiêu chí phủ định và khai báo lực CHỐT TRƯỚC ở `docs/RUIRO_ML.md` | **4** | **Không đưa vào sản xuất — nhưng tìm ra một khiếm khuyết SẢN XUẤT đang hoạt động.** (a) **Đích A “đạt” theo chữ viết nhưng không được đọc là thắng lợi**: BSS +0,0016 (bằng không), **đảo dấu** khi cắt 1.016 hàng (−0,0002 → +0,0016), và **H13 bị bác bỏ ở biến chi phối** — `log_sig` hệ số **−0,1157** (lớn nhất, ÂM) trong khi H13 đòi dương. Cơ chế thật: stop đặt ở 1,5·σ̂ nên đã chuẩn hoá theo σ̂; σ̂ cao hôm nay thổi phồng biến động ngày mai do hoàn nguyên, nên stop **tương đối xa hơn**. Khai báo lỗi của chính văn bản chốt trước: ĐK1 viết `BSS > 0` **không có ngưỡng độ lớn**, cùng loại lỗi đã khai ở `DUOI_EVT.md` A5a. (b) **PHÁT HIỆN DÙNG ĐƯỢC, và nó không phải ML**: mốc giải tích `decision_record.p_cham_stop` — **đang hiện trên phiếu quyết định** — dự báo **45,25%** trong khi thực tế **33,64%**, tức vượt ước ~35% tương đối. Nó **tệ hơn một hằng số**: Brier 0,23606 so 0,22097, **ECE 0,1230 so 0,0069 (18 lần)**. Nguyên nhân là sai **đặc tả**: nguyên lý phản xạ `P(min≤−b)=2P(X_T≤−b)` đòi gia số độc lập cùng phân phối, mà z chuẩn hoá bằng σ̂ có co cụm biến động. Đề nghị thay bằng **tần suất nền trên huấn luyện, phân tầng theo cặp** (1 tham số) — nhưng là con số sản xuất nên **cần người chịu trách nhiệm quyết**. (c) Đích B **ÂM** (BSS −0,0011), đúng như khai báo lực: chỉ ~33 ca dương. Mốc danh nghĩa 0,01 trùng khí hậu học tới 5 chữ số → VaR hiệu chuẩn **đúng ở mức gộp**; khiếm khuyết là ở từng cặp và ở động học, nhất quán `DUOI_EVT.md` A4. (d) **Đã cắt 1.016 hàng sau 2025-12-31**: `noi_chuoi` nối live tới 2026-09 nhưng mục 2 dành riêng 2026-01→08 cho niêm phong. **Việc tồn: `va_duoi.py`, `va_duoi_evt.py`, `shadow_tohop.py` cũng dùng `noi_chuoi` và CHƯA được rà điểm này.** Xem `docs/RUIRO_ML.md` Phụ lục A |
### Thay đổi sau ngày chốt — theo quy tắc 3

| 45 | 12/09/2026 | **qlikeHAR** (`kiem_qlikehar.py`) — Puke & Schweikert 2026, *J. Forecasting*: khớp HAR bằng chính QLIKE thay vì OLS. Bốn cấu hình CHỐT TRƯỚC ở commit f0e83d1, gồm một dòng đối chứng A0 (y hệt mốc nhưng khớp theo năm) để **tách bạch** tần suất khớp khỏi hàm mất | **4** | **ÂM.** Q_log −1,40% trên kiểm tra nhưng **DM p = 0,4450**; so đúng dòng đối chứng A0 thì −1,59%, **p = 0,3815**. Trên **kiểm định** mốc B0 vẫn tốt nhất (0,1162 so 0,1165) nên theo quy tắc chọn của repo **không chọn được**. Chỉ **1/6 cặp** cải thiện; toàn bộ phần thắng đến từ USDJPY (−6,71%), còn GBPUSD **xấu đi có ý nghĩa** (+2,95%, p=0,0007). Lợi thế ở chế độ êm (Q1 −4,55%) nhưng thiệt ở chế độ căng (Q5 +4,01%) — lặp đúng hình mẫu của nhánh học sâu. MCS giữ cả B0 lẫn Q_log. Bản sát bài báo (HAR thang mức 4 hệ số) +10,02%, bị loại khỏi MCS — nhưng đó là mô hình đơn giản hơn hẳn nên không dùng để bác bài báo. **Không đổi sản xuất.** Xem `docs/PHUONGPHAP_NGOAI.md` mục 1 |

| 46 | 12/09/2026 | **Conformal phân tầng theo trạng thái sụt giảm** (`kiem_conformal_dd.py`) — trả món nợ `TANG6_HIEU_CHUAN.md` mục 5 (hướng vá đã ghi, chưa bao giờ làm). Hai cấu hình CHỐT TRƯỚC: ACI-dd 2 (chỉ sụt giảm) và ACI-2D 2×2 (biến động × sụt giảm). Biến phân tầng của mô hình **trễ một phiên**, tự kiểm rò rỉ 0/3.000 + đối chứng dương | **2** | **Không đổi sản xuất, NHƯNG rút lại được một "giới hạn đã biết".** ACI-dd 2 **đóng được khe** đỉnh−lỗ (−0,03% so 0,29–0,96% của sáu cách cũ) đúng như lý thuyết Mondrian có điều kiện nói — nhưng đổi lấy độ phủ theo chế độ biến động (vol thấp 88,5% / vol cao 92,4%), `|lệch|` max 2,4% tệ nhất trong tám cách, và trên kiểm định cũng tệ nhất nên **không được chọn**. **Phát hiện quan trọng hơn:** bootstrap khối 20 phiên × 2.000 lần cho thấy khe đỉnh−lỗ **phủ 0 ở MỌI cấu hình** (KTC rộng ±2 điểm, khe báo cáo chỉ 0,6–0,8 điểm), và dấu đảo chiều giữa kiểm định/kiểm tra. Tức **giới hạn ghi ở TANG6_HIEU_CHUAN mục 5 nằm trong nhiễu** — giải thích luôn vì sao năm cách vá trước đều "không xoá được khoảng chênh". Đã trỏ đính chính vào tài liệu gốc. Xem `docs/PHUONGPHAP_NGOAI.md` mục 2 |

| 47 | 12/09/2026 | **Mẫu hình NẾN — hình học OHLC ngày** (`kiem_nen.py`, `kiem_nen_ablation.py`). Lỗ hổng thật: 12 nhánh trước đều dùng chuỗi đóng-đóng hoặc đại lượng nến 5 phút; 4 cột OHLC có trong bảng sản xuất nhưng `volfc2.thiet_ke` không dùng cột nào. 25 đặc trưng / 4 họ × 2 lag × 6 cặp × 2 trục, CHỐT TRƯỚC ở `docs/NEN_TIEUCHI.md` (commit 9643f80) | **600 giả thuyết + 8 cấu hình** | **TÁCH ĐÔI.** (a) **Mẫu hình nến cổ điển CHẾT SẠCH**: K3 (15 mẫu có tên) **0/360** sống sót W-Y trên cả hai trục, và đưa vào mô hình làm QLIKE kiểm tra **xấu đi có ý nghĩa** (+0,60%, p=0,0451). K1 hình học 0/120, K2 phạm vi−RV 0/72. Trục hướng **0/300**, p W-Y tốt nhất 0,9950 — nhánh thứ 13 cho kết quả âm về hướng, và lần này **khớp đúng đồng thuận văn liệu** (Orquín-Serrano 2020, *Mathematics* 8(5), art. 802). (b) **Nhưng 7/600 sống sót, CẢ BẢY là `abs_gap`** (độ lớn gap qua đêm), trục biên độ, F tới **67,84**, p W-Y 0,0010, dấu **dương** đúng cơ chế chốt trước. Cấu hình `chỉ |gap| L1` (**1 tham số**): kiểm tra **−0,67%**, DM p=0,0445, **5/6 cặp**, hệ số +50,56, **trong MCS**, walk-forward 7/11 năm, vẫn thắng sau khi bỏ 1% gap lớn nhất (−0,36%, p=0,0411). **Trượt ĐK2** vì p=0,0445 > ngưỡng Bonferroni 0,025 → **KHÔNG đổi sản xuất**. Nhưng đây là **ứng viên cải thiện tầng 2 mạnh nhất dự án từng tạo ra** — đề nghị đưa vào danh sách chốt của lần mở tập niêm phong cuối. Kèm đính chính `DATASET.md`: gap qua đêm "bỏ qua được" là phần chưa chứng minh. Xem `docs/PHUONGPHAP_NGOAI.md` mục 4 |

Cấu hình ở mục 4 chốt ngày 08/09/2026. Hai thay đổi sau đó, đều ghi lại kèm lý
do trước khi mở tập khoá sổ:

**(1) Thêm tầng tập dự báo conformal (ACI) vào `api/main.py` — 09/09/2026.**

*Không* đổi mô hình nào ở mục 4. Đây là một tầng **cộng thêm** trên chính ba xác
suất đã có: cùng đầu vào, cùng mô hình, chỉ thêm một đầu ra mới. Lý do đưa vào:
nó cho thứ mà toàn bộ mục 4 không cho được — **bảo đảm** độ phủ 90% thay vì
*đo được* hiệu chuẩn tốt, và bảo đảm đó giữ ngay cả khi thị trường đổi chế độ
(`CHISO_DANHGIA.md` mục 16). Đã kiểm chứng conformal **tĩnh** hỏng trên chính dữ
liệu này (0,814–1,000) trong khi ACI giữ 0,901–0,908.

**(2) Công bố mức kỹ năng ĐO ĐƯỢC riêng cho từng tầm hạn — 09/09/2026.**

`KY_NANG_THEO_H` trong `api/main.py`. Hai phép đo độc lập (walk-forward theo
năm, và lượng thông tin conformal) đều cho cùng kết luận: kỹ năng nằm ở **h = 1**
(BSS dương 14/14 năm, 6/6 cặp có ý nghĩa trên kiểm tra, tập conformal nhỏ hơn mốc
khí hậu học 0,10–0,21 lớp), còn ở h = 5 và h = 20 thì tập dự báo **không** nhỏ
hơn tập của một hằng số (−0,09 và −0,10 lớp). Giao diện phải nói đúng như vậy
thay vì trình bày ba ô như nhau. Đây là thay đổi về **cách công bố**, không phải
về mô hình.

**(3) 566 cấu hình mô hình bổ sung cho biến động vòng 7, 09–11/09/2026 (dòng
25–31).** Đây KHÔNG phải thay đổi cấu hình sản xuất ở mục 4 — mô hình chốt sản
xuất (STHARQ+HARQ+SHAR, `CAUHINH_SANXUAT`) giữ nguyên. Đây là công việc nghiên
cứu thêm sau khi HuyH gửi roadmap v2, kiểm tra xem mô hình hiện đại hơn
(foundation model, tổ hợp dự báo) có đáng thay HAR sản xuất không. Lý do phải
ghi vào biên bản: quy tắc 4 đòi *"mọi cấu hình đã thử trên tập phát triển phải
được đếm"*, và riêng dòng 31 (509 tổ hợp) làm tổng số cấu hình mô hình nhảy từ
685 lên **1.251** — thay đổi đáng kể cho phần hiệu chỉnh đa kiểm định của luận
văn. Kết luận không đổi cấu hình sản xuất: không tổ hợp/mô hình mới nào vượt
qua được MCS so với HAR đơn (xem `docs/ML_DL_VONG7.md`).

**(4) 183 giả thuyết quy luật bổ sung — Pha 2 / tin tức (dòng 32–33, 36–37).**
Bốn nhánh MỚI trên cùng 129 thông cáo FOMC, đều đến SAU ngày chốt 08/09/2026
nên chưa từng được cộng vào tổng 8.469 của Giai đoạn 1:

| nhánh | cách biểu diễn | giả thuyết | SPA p |
|---|---|---|---|
| H8 (09/09) | từ điển HAWK/DOVE + TF-IDF + độ dài | 54 | 0,162 |
| H8b (11/09) | embedding pretrained + PCA | 54 | 0,861 |
| H8c (11/09) | chủ đề chính theo từ điển | 24 | 0,669 |
| H8e (11/09) | trích xuất có cấu trúc bằng LLM | 51 | 1,000 |

Tổng giả thuyết quy luật nay là 8.469 + 54 + 54 + 24 + 51 = **8.652**
(12 nhánh độc lập). Cả bốn đều 0 quy luật qua phễu. **Biểu diễn càng hiểu sâu
văn bản, SPA p càng cao** — thứ tự đơn điệu, nhất quán với giả thuyết không có
tín hiệu nào để bắt (xem `PHA2_TINTUC.md` mục 8b), trả lời RQ5 của roadmap
Pha 2: không đại diện văn bản nào trong bốn cách mang lại giá trị đo được.

**(5) Bốn nhánh bổ sung 13/09/2026 — dòng 48–51.** Tất cả SAU ngày chốt
08/09/2026, không cấu hình nào đổi sản xuất, đều đã đếm vào tổng ~1.322:

| dòng | nhánh | giả thuyết/cấu hình | kết quả |
|---|---|---|---|
| 48 | EVT/POT cho đuôi VaR/ES (`va_duoi_evt.py`) | 5 | H10 bác bỏ (ξ 2/6); vá được 2 ô hỏng riêng lẻ nhưng trượt ĐK2/ĐK3 gộp — không đổi sản xuất, ứng viên cho lần mở niêm phong cuối |
| 49 | Bộ lọc chu kỳ nội tuần BCL (`xay_bcl.py`) | 9 | H11b bác bỏ; phát hiện phụ quan trọng hơn: khung mốc `log rv(t+1)~log h_HAR` dùng cho MỌI ablation trước đó lệch ~8% QLIKE — không đổi phán quyết các nhánh trước (đếm là bằng chứng bổ sung trong khung khác) |
| 50 | Xếp hạng chéo 6 đồng tiền (`xep_hang_cheo.py`) | 6 | ÂM — nhánh thứ 14 cho kết luận âm trên trục hướng; `mom_12m` là ca overfit kiểm định mẫu mực (p=0,024 kiểm định → t=0,58 kiểm tra) |
| 51 | ML trực tiếp trên đại lượng rủi ro (`ruiro_ml.py`) | 4 | Không đưa vào sản xuất; phát hiện phụ: `decision_record.p_cham_stop` đang chạy sản xuất có ECE tệ hơn hằng số 18 lần — cần người chịu trách nhiệm quyết có vá hay không TRƯỚC khi ký mục 4.6 (xem mục kiểm tra trước khi ký bên dưới) |

Tổng bốn dòng: **24 giả thuyết/cấu hình mới**, không dòng nào đổi cấu hình sản
xuất ở mục 4. Dòng 51(b) là phát hiện duy nhất trong nhóm này ảnh hưởng tới
một con số ĐANG HIỂN THỊ cho người dùng — cần quyết định trước khi khoá.

| 52 | 14/09/2026 | **Tam giác hoá nhân quả bằng thư viện gốc** — PCMCI thật (`tigramite`, `pha3b_pcmci_doclap.py`), Double ML (`doubleml`, `pha3b_dml_causal.py`), Causal Forest (`econml`, `pha3b_causal_forest.py`). Cả ba chạy trên biến **ĐÃ khai báo** trong Pha3B (VIXCLS/GVZCLS/DFII10, 11 biến gốc mục 2a của `PHA3B_TIEUCHI.md`) — không đề xuất biến mới, không mở lại không gian giả thuyết 594 đã đóng | **0 cấu hình mới** (chẩn đoán/tam giác hoá trên biến đã khai báo, cùng quy ước dòng 40/44b) | PCMCI thật xác nhận VIX trên 3/3 cặp thử độc lập với Level 2/3; minh hoạ confounding cụ thể (GVZCLS mất ý nghĩa khi điều kiện trên VIXCLS). Double ML: **6/6 cặp** hiệu ứng VIX vững qua Holm — mạnh và nhất quán nhất trong mọi phép tam giác hoá. Causal Forest: hiệu ứng có đổi theo chế độ nhưng **không** theo một khuôn mẫu chung (4/6 cặp CATE tăng theo chế độ căng thẳng). Xem `docs/PHA3B_TAMGIAC_NHANQUA.md` |
| 53 | 14/09/2026 | **Lớp phủ nhân quả DML trên dự báo sản xuất** (`pha3b_dml_tichhop.py`) — ĐỀ XUẤT MỚI, không phải chẩn đoán: θ ước trên huấn luyện, đóng băng, áp lên σ̂_HAR trên kiểm định (chưa từng thấy), 6 cặp | **6** (1 cấu hình lớp phủ × 6 cặp) | **ÂM.** 4/6 cặp cải thiện QLIKE về hướng, **0/6 có ý nghĩa sau Holm**. Hiệu ứng nhân quả có thật (dòng 52) nhưng mốc HAR đã hấp thụ phần lớn qua chính cấu trúc tự hồi quy — **không đưa vào sản xuất** |
| 54 | 14/09/2026 | **CausalImpact sự kiện ECB** (`pycausalimpact`, `su_kien_causalimpact.py`, Brodersen et al. 2015) — thí điểm phương pháp thứ ba cho hạng mục "sự kiện theo từng cặp" đã hứa ở đề cương, KHÔNG phải quyết định Pha3B | **20** (20 phiên họp ECB gần nhất, huấn luyện+kiểm định) | 6/20 (30%) có hiệu ứng tăng biến động có ý nghĩa sau FDR-BH, toàn bộ cùng chiều dương. Giới hạn tự khai báo: USDCAD (hiệp biến) không tách biệt hoàn toàn khỏi ECB — thí điểm phương pháp, chưa phải kết luận đóng |
| 55 | 14/09/2026 | **Nhánh pattern-discovery thứ 14 — CNN mã hoá nến thành ảnh** (`kiem_cnn_nen.py`, Chen & Tsai 2020, *Financial Innovation* 6:26) — khác 13 nhánh trước (đặc trưng số): học trực tiếp trên hình dạng không gian của cửa sổ 10 nến. θ/trọng số CNN khớp trên huấn luyện, epoch chọn trên NỬA ĐẦU kiểm định, báo cáo trên NỬA SAU (chưa từng thấy) — sau khi phát hiện và sửa một lỗi rò rỉ ở lần chạy đầu (chọn epoch và báo cáo dùng chung một đoạn) | **6** (1 kiến trúc × 6 cặp) | **3/6 cặp (AUDUSD, GBPUSD, USDCAD) sống sót Holm VÀ Bonferroni** — nhánh pattern-discovery ĐẦU TIÊN trong toàn dự án (14 nhánh: 12 quy luật + mẫu hình nến K1-K4 + CNN nến, 8.652+600+6=9.258 giả thuyết) vượt qua hiệu chỉnh đa kiểm định trên bất kỳ cặp nào. **Chưa đạt ngưỡng ≥5/6 cặp** mà chính Pha3B/NEN_TIEUCHI đặt ra cho "quan hệ vững" → **chưa đưa vào sản xuất**, ghi nhận là hướng đáng theo đuổi nhất, cần thêm seed/kiến trúc/LOPO trước khi cân nhắc lại. Xem `docs/NEN_CNN_ANH.md` |
| 56 | 14/09/2026 | **Khai phá quy luật ở h=5, h=20 và mục tiêu R** (`run_giaidoan2_tamhan.py`) — đóng nốt hạng mục Assignment 2 còn treo từ đề cương. Sửa một lỗi rò rỉ trước khi chạy: `nap_du_lieu()` trước đây dùng CHUNG một lời gọi `dung_muc_tieu(d,H,tr)` cho cả đặc trưng lẫn nhãn — với h>1, đặc trưng sẽ nhìn thấy h−1 ngày tương lai. Đã tách riêng (đặc trưng LUÔN h=1, theo đúng mẫu có sẵn ở `kiem_h3.py`), kiểm chứng bằng số, và tăng khối hoán vị theo h | **7.941–7.950/tổ hợp × 5 tổ hợp mới (h1×R, h5×R, h5×P, h20×R, h20×P)** | **0/6 tổ hợp có quy luật sống sót Westfall-Young** — không đổi kết luận sản xuất. NHƯNG Hansen SPA bác bỏ ở **5/54 phép kiểm sau Holm, TOÀN BỘ ở mục tiêu R** (0 ở mục tiêu P, kể cả h=1×P gốc) — câu chữ tiêu chí dừng mục 10.4 REPLAN không thoả mãn tuyệt đối trên mục tiêu R. Cơ chế khả dĩ: mốc "chỉ σ̂" suy biến thành khí hậu học trên R (đúng thiết kế đã ghi ở REPLAN mục 3a), khiến SPA (màn lọc mềm hơn W-Y) nhạy hơn với mốc yếu. Độ lớn kinh tế nhỏ (dbar 0,0047–0,0152) ở cả 5 trường hợp. **Cần người chịu trách nhiệm quyết cách viết** trước khi ký mục 4.6 — xem `docs/GIAIDOAN2_TAMHAN_KETQUA.md` mục 4 |

**Ghi chú của phiên làm việc 14/09/2026 (không phải người chịu trách nhiệm
luận văn):** năm dòng 52–56 trên do một phiên trợ lý AI thực hiện theo yêu
cầu "làm cho phần nhân quả hiệu quả và chủ chốt hơn" và "hoàn tất khai phá
quy luật h=5/h=20". Đã áp dụng đúng kỷ luật đã có của tài liệu này (tách
huấn luyện/báo cáo, hiệu chỉnh đa kiểm định, không mở lại quyết định đã
đóng băng) nhưng **KHÔNG có văn bản _TIEUCHI.md chốt trước** cho các dòng
này như phần lớn các nhánh khác — đây là công việc thăm dò trong phiên,
không phải giả thuyết đăng ký trước theo đúng nghĩa Westfall-Young. Người
chịu trách nhiệm luận văn nên tự xác nhận lại cách đếm ở năm dòng này (cột
4) VÀ quyết định cách viết cho phát hiện SPA ở dòng 56 trước khi coi bảng
tổng kết bên dưới là số liệu cuối cùng để ký mục 4.6.

### Tổng kết cho phần hiệu chỉnh bội của luận văn

| khoản | số lượng |
|---|---|
| Cấu hình **mô hình** đã thử trên tập phát triển | **~1.334** (1.322 đến dòng 51 + 6 dòng 53 lớp phủ DML + 6 dòng 55 CNN nến) |
| Giả thuyết **quy luật** đã liệt kê và kiểm định | **8.652** (12 nhánh độc lập) |
| Quy luật sống sót toàn bộ phễu bốn cửa | **0** |
| Lực phát hiện của phễu quy luật (MDES, lực 80%) | lift **1,20** |
| Dương tính giả trên nhiễu thuần (đối chứng âm) | **0,0 / 1.890** |
| **Giả thuyết NHÂN QUẢ** đã liệt kê và kiểm định (Pha 3B, dòng 44) | **594** (11 biến × 3 lag × 6 cặp × 3 trục) |
| **Giả thuyết MẪU HÌNH NẾN** — cổ điển K1-K4 (dòng 47) + CNN ảnh (dòng 55) | **606** (600 + 6) — nâng tổng lên **14 nhánh** pattern-discovery |
| Mẫu hình nến **có tên (K3)** sống sót phễu | **0 / 360** |
| Mẫu hình nến **CNN ảnh** sống sót Holm+Bonferroni | **3 / 6 cặp** — chưa đạt ngưỡng ≥5/6, chưa đưa vào sản xuất |
| **Quan hệ nhân quả sống sót** W-Y **và** màn lọc độ vững | **14** — toàn bộ trên trục biên độ |
| Lực phát hiện của phễu nhân quả (MDES, lực 80%) | ≈ **1,0% QLIKE** |
| Tam giác hoá nhân quả bằng thư viện gốc (dòng 52) | PCMCI 3/3 cặp, **Double ML 6/6 cặp** — hiệu ứng VIX vững nhất trong mọi phép đo |
| CausalImpact sự kiện ECB, thí điểm (dòng 54, ngoài phễu chính) | 6/20 sự kiện có ý nghĩa sau FDR-BH |
| **Giả thuyết quy luật bổ sung** — h=5/h=20 và mục tiêu R (dòng 56) | **39.735** (5 tổ hợp × ~7.947, 12 nhánh mỗi tổ hợp) |
| Quy luật sống sót W-Y ở 5 tổ hợp bổ sung | **0 / 39.735** |
| Hansen SPA bác bỏ sau Holm, toàn bộ 54 phép kiểm (6 tổ hợp × 9 họ) | **5 / 54 — TOÀN BỘ ở mục tiêu R, 0 ở mục tiêu P** |

Hai con số phải đọc TÁCH NHAU: 8.652 là giả thuyết **quy luật** (phễu khai phá,
0 sống sót); 594 là giả thuyết **nhân quả** (phễu Pha 3B, 14 sống sót). Chúng
dùng hai phễu khác nhau, hai loại mục tiêu khác nhau, nên cộng lại là sai.

Con số 8.652 giả thuyết là con số **biết trước và liệt kê đầy đủ**, không phải
đếm ngược sau khi chạy — đó là điều kiện để Westfall–Young có nghĩa. Con số
~1.334 cấu hình mô hình là lý do mọi kết luận về **mô hình** đều chỉ được phát
biểu trên đoạn kiểm định, và vì sao tập khoá sổ tồn tại.

*(Sửa 14/09/2026: con số ở đoạn này từng ghi "~1.265", lệch với bảng tổng kết
ở trên — đã đồng bộ lại. Không có thay đổi nội dung nào khác từ việc sửa này.)*

---

## 6. DANH SÁCH KIỂM TRA TRƯỚC KHI KÝ MỤC 4.6

*Thêm 14/09/2026 theo yêu cầu chuẩn bị biên bản sẵn sàng để ký — KHÔNG tự ý
ký thay hay mở tập khoá sổ. Đánh dấu ✅/❌ là việc của người chịu trách nhiệm
luận văn, không phải của phiên làm việc này.*

| # | Điều kiện | Trạng thái tại 14/09/2026 | Bằng chứng |
|---|---|---|---|
| 1 | Mục 4 (cấu hình chốt) không còn thay đổi sản xuất nào đang chờ quyết định | ⚠️ **Cần xác nhận** | Dòng 51(b): `decision_record.p_cham_stop` đang chạy sản xuất có ECE tệ hơn hằng số 18 lần — cần người chịu trách nhiệm quyết có vá trước khi khoá hay ghi nhận như giới hạn |
| 2 | Khai phá quy luật đã đóng đúng tiêu chí dừng ở CẢ BA tầm hạn (h=1, h=5, h=20) | ⚠️ **Một phần — cần quyết định cách viết** | Đã chạy đủ 6 tổ hợp (h×mục tiêu), 0/6 sống sót Westfall-Young — nhưng Hansen SPA bác bỏ ở 5/54 phép kiểm sau Holm, TOÀN BỘ ở mục tiêu R (0 ở mục tiêu P). Câu chữ tiêu chí dừng mục 10.4 REPLAN ("cả năm họ không bác bỏ SPA... trên CẢ HAI mục tiêu") không thoả mãn tuyệt đối. Xem `docs/GIAIDOAN2_TAMHAN_KETQUA.md` mục 4 — cần người chịu trách nhiệm quyết cách viết trước khi ký |
| 3 | Mọi cấu hình/giả thuyết đã thử trên tập phát triển được đếm vào mục 5 | ✅ Đã cập nhật đến dòng 55 | Bảng tổng kết ở trên, ~1.334 cấu hình + 9.258 giả thuyết pattern + 594 giả thuyết nhân quả |
| 4 | Không còn ý định "thử thêm rồi mở lại" tập khoá sổ | ⚠️ **Cần xác nhận bằng lời** | Quy tắc 2, mục 3 — đây là cam kết của người ký, không kiểm chứng được bằng code |
| 5 | Dòng ký ở mục 4.6 còn để trống đúng như chủ đích | ✅ Đúng | Mục 4.6 — chưa có chữ ký |
| 6 | Tập khoá sổ (`histdata_seal/`) chưa bị bất kỳ script nào chạm | ✅ Đúng tại 14/09/2026 | Không phiên làm việc nào (kể cả phiên hiện tại) đã đọc/copy dữ liệu từ `histdata_seal/` |

**Sau khi cả 6 mục trên đều ✅, và chỉ khi đó**, người chịu trách nhiệm luận
văn ký vào mục 4.6, rồi mới chạy lệnh mở niêm phong ở mục 2 ("Vị trí trên
đĩa"). Không có bước nào trong danh sách này được thực hiện thay bởi công cụ
tự động — đây là các quyết định thuộc về con người theo đúng thiết kế ban
đầu của biên bản này (mục 4.6: *"người chịu trách nhiệm luận văn ký, không
phải công cụ điền hộ"*).

---

*Biên bản này lập TRƯỚC khi tập khóa sổ được tải về máy. Đưa vào phụ lục luận văn.*
