# TỔNG QUAN DỰ ÁN FX-DSS — TÀI LIỆU CHO AI KHÁC ĐỌC ĐỂ ĐỀ XUẤT PHƯƠNG PHÁP

*Lập 14/09/2026. Mục đích: cung cấp đủ ngữ cảnh để một AI khác (không có lịch
sử hội thoại với dự án này) có thể đọc, hiểu toàn cảnh, và đề xuất phương
pháp cải thiện — KHÔNG lặp lại những gì đã thử. Viết cho người đọc là AI,
nên ưu tiên liệt kê đầy đủ, có cấu trúc, hơn là văn phong tự nhiên.*

---

## 1. DỰ ÁN LÀ GÌ

Luận văn thạc sĩ ngành Hệ thống thông tin: xây dựng hệ thống hỗ trợ quyết
định giao dịch ngoại hối (FX) cho 6 cặp tiền chính (EURUSD, GBPUSD, USDJPY,
AUDUSD, USDCAD, USDCHF), trả lời ba câu hỏi mỗi phiên giao dịch:

1. **Biên độ** — ngày mai giá dao động mạnh hay yên ắng? (tầng biến động)
2. **Hướng** — ngày mai giá tăng hay giảm? (tầng ba xác suất)
3. **Rủi ro** — nếu vào lệnh thì nên đặt bao nhiêu, dừng lỗ ở đâu, lỗ tối đa
   bao nhiêu? (tầng quyết định)

**Triết lý cốt lõi**: mọi con số hiển thị phải truy được về một phép đo cụ
thể; kết quả ÂM (không tìm được gì) phải báo cáo trung thực như kết quả
dương, kèm lực phát hiện (MDES) để phân biệt "không có gì" với "phép thử
không đủ mạnh để tìm ra".

## 2. DỮ LIỆU

| Nguồn | Nội dung | Vai trò |
|---|---|---|
| HistData.com | Nến M1, 6 cặp chính, 2010-01-03 → 2025-12-31, ~34,9 triệu nến | Xương sống — mọi mô hình biến động dựng từ đây |
| Yahoo Finance | Giá từ 2026-01-01 (HistData ngừng cập nhật) | Nối tiếp cho vận hành sản xuất, cập nhật 4 lần/ngày |
| FRED (9 chuỗi vĩ mô hàng ngày) | DGS2, DGS10, T10Y2Y, DFII10, VIXCLS, GVZCLS, OVXCLS, DCOILBRENTEU, NIKKEI225 | Lớp ngoại sinh cho phân tích nhân quả (Pha 3B) |
| Lịch công bố NHTW/vĩ mô | `data/su_kien.csv`, `data/cb_dates.csv` | Điều kiện hoá sự kiện |
| **Tập niêm phong (CHƯA MỞ)** | 6 cặp chéo không-USD (EURGBP, EURJPY, GBPJPY, AUDJPY, EURCHF, NZDUSD) + dữ liệu 2026 của cả 12 cặp | Chấm điểm cuối cùng, đúng một lần — KHÔNG được đụng tới cho tới khi mọi thứ khác đã đóng băng |
| Chia thời gian | Huấn luyện: → 2021-10-13 · Kiểm định: → 2023-11-20 · Kiểm tra: 2023-11-20 → nay | Cố định trong `src/split.py`, không đổi |

## 3. KIẾN TRÚC PHẦN MỀM (tóm tắt — không phải trọng tâm cần AI khác góp ý)

- `collect/` — script thu thập dữ liệu (bootstrap một lần + vận hành hằng ngày qua CI)
- `src/` — toàn bộ logic tính toán, mô hình, thực nghiệm (~180 file)
- `api/` — FastAPI, tách package (`config/cache/utils/risk_logic/schemas/routers`)
- `web/` — giao diện (1 template HTML, biên dịch ra 3 bản: tĩnh/trực tiếp/Vercel)
- Triển khai: local/CI chạy đầy đủ (có scipy); Vercel chạy tĩnh + 1 hàm nhẹ (dự báo tính sẵn, đóng gói JSON)
- CI/CD: GitHub Actions, 4 lần/ngày, tự động cập nhật + build + deploy

---

## 4. PHƯƠNG PHÁP ĐÃ ÁP DỤNG — THEO TỪNG THÀNH PHẦN (đầy đủ, để tránh đề xuất trùng)

### 4.1 Tầng biến động (HAR) — THÀNH PHẦN ĐÃ THỬ NHIỀU PHƯƠNG PHÁP NHẤT

**Đang dùng sản xuất**: tổ hợp STHARQ + HARQ + SHAR (họ mô hình HAR mở rộng,
Corsi 2009 + Bollerslev-Patton-Quaedvlieg 2016 + jump/semivariance).

**Đã thử và THUA HAR** (không đổi sản xuất):
- Cổ điển: MA20-Garman-Klass, EWMA(RiskMetrics), Parkinson, Garman-Klass,
  Rogers-Satchell, Yang-Zhang, GARCH(1,1)-t
- ML/DL (14 mô hình): LightGBM (2 hàm mất mát), GRU, LSTM, Random Forest,
  MLP, Transformer/PatchTST rút gọn, Ridge, XGBoost, CatBoost, TabPFN v2
  (phân loại lẫn hồi quy)
- Foundation model chuỗi thời gian (zero-shot): Chronos-bolt-small, TTM
  (Tiny Time Mixers) — cả hai thua HAR ~17-19%, khớp văn liệu (Brini 2026)
- qlikeHAR (Puke & Schweikert 2026 — khớp bằng QLIKE thay OLS) — âm
- 4 cách tổ hợp dự báo (đều tay, 1/QLIKE, Granger-Ramanathan, stacking phi
  tuyến) trên tập {HAR, LightGBM, GRU, LSTM, Ridge, XGBoost, CatBoost,
  TabPFN} — 509 tổ hợp; tốt nhất (HAR+GRU+CatBoost, DM p=0,041) **không
  đưa vào sản xuất** vì lý do giải thích được, dù thắng có ý nghĩa

**~1.334 cấu hình mô hình đã thử cho riêng tầng này.**

### 4.2 Tầng ba xác suất (hướng đi) — KẾT LUẬN: HƯỚNG ĐI KHÔNG ĐOÁN ĐƯỢC

**Đang dùng sản xuất**: tổ hợp trực tuyến Hedge (exponential weights) trên 4
"chuyên gia": khí hậu học, quán tính, chỉ σ̂, σ̂+chế độ.

**Đã thử thay thế**:
- Fixed-Share (Herbster-Warmuth 1998) thay Hedge — thua, α=0 tối ưu
- Stacking (hồi quy logistic đa thức, khớp tĩnh) thay Hedge — **thua có ý
  nghĩa** (DM p=0,029) — Hedge thắng vì thích nghi liên tục theo thời gian
- Dùng LightGBM/GRU làm "chuyên gia" trong Hedge — Hedge tự hạ trọng số về 0

**14 nhánh khai phá quy luật hướng đi độc lập** (chi tiết mục 4.4) — TẤT CẢ
đều 0 quy luật sống sót. Đây là kết luận vững chắc nhất của toàn dự án.

### 4.3 Tầng rủi ro VaR/ES — **ĐIỂM NGHẼN LỚN NHẤT, XEM MỤC 5**

**Đang dùng sản xuất**: V0 — phân vị thực nghiệm của z, ước trên huấn luyện,
đóng băng.

**Đã thử và THUA V0** (6 hướng độc lập):
1. V1 — cửa sổ mở rộng, khớp lại mỗi 21 phiên
2. V2 — cửa sổ cuộn 500 phiên
3. Phân vị theo chế độ biến động (2 biến thể)
4. CAViaR (Engle-Manganelli 2004, 3 dạng SAV/AS/IG) — không ổn định qua seed
5. Cửa sổ họp NHTW/FOMC định kỳ (2 biến thể)
6. EVT/POT (McNeil-Frey 2000, 4 biến thể ngưỡng 90%/95%) — thắng theo thước
   đo liên tục nhưng trượt tiêu chí nhị phân đã chốt trước
7. **[Mới 14/09]** Lịch can thiệp KHÔNG định kỳ (BOJ 2022, SNB 2015), nới
   ngưỡng 1,5×/2×/3× — cũng thất bại, xem mục 5

### 4.4 Khai phá quy luật (14 nhánh, ~9.858 giả thuyết, 0 sống sót ở h=1)

| # | Nhánh | Phương pháp |
|---|---|---|
| 1 | SAX biến động | Rời rạc hoá tercile |
| 2 | SAX hướng giá | Rời rạc hoá tercile |
| 3 | Ngưỡng đặc trưng 3 lớp | Vị từ đơn/đôi trên 12 đặc trưng đọc được |
| 4 | H2 Motif | KMeans trên cửa sổ hình dạng chuẩn hoá |
| 5 | H3 Rule-list | CART nông (đã thử RuleFit thay thế — **thua, tệ hơn cả khí hậu học**) |
| 6 | H5 Chế độ tự tương quan | Tam phân vị tự tương quan lag-1 |
| 7 | H6 HMM | Baum-Welch tự viết, K=2/3/4 trạng thái |
| 8 | H7 Matrix Profile | Tìm analog gần nhất (tự viết, không có stumpy) |
| 9-12 | H8/H8b/H8c/H8e (tin tức FOMC) | Từ điển HAWK/DOVE, embedding pretrained, chủ đề, LLM extraction |
| 13 | K1-K4 mẫu hình nến | Hình học chuẩn hoá + mẫu có tên cổ điển (doji/hammer/engulfing — 0/360) |
| 14 | **[Mới]** CNN/ViT mã hoá ảnh nến | CNN (Chen&Tsai 2020): 3/6 cặp sống sót Holm+Bonferroni nhưng CHƯA đạt ngưỡng ≥5/6. ViT (patch+self-attention, cùng quy mô tham số): **thua CNN rõ rệt**, 6/6 cặp xấu đi |

**Đã chạy đủ ở CẢ BA tầm hạn (h=1/5/20) và CẢ HAI mục tiêu (P=lợi suất
tuyệt đối, R=chuẩn hoá theo σ̂)** — 0/6 tổ hợp có quy luật sống sót
Westfall-Young. Nhưng Hansen SPA (kiểm cấp-họ, mềm hơn) bác bỏ ở 5/54 phép
kiểm sau hiệu chỉnh Holm, **toàn bộ nằm ở mục tiêu R** — cần quyết định cách
viết, xem mục 5.

### 4.5 Nhân quả (Pha 3B) — 5 phương pháp độc lập, kết quả MẠNH

VIX (chỉ số sợ hãi thị trường) → biến động FX: đã xác nhận **nhân quả thật**
qua 4/5 phương pháp độc lập:

| Phương pháp | Kết quả |
|---|---|
| Granger (nested-regression F-test) + Westfall-Young | 14/594 giả thuyết sống sót, toàn bộ trục biên độ |
| PCMCI thật (`tigramite`, Runge) | Xác nhận VIX trên 3/3 cặp; phát hiện GVZCLS chỉ là tương quan giả (biến mất khi điều kiện trên VIX) |
| Double ML (`doubleml`) | **6/6 cặp** hiệu ứng VIX vững qua Holm — mạnh nhất |
| Causal Forest (`econml`) | Hiệu ứng không đồng nhất theo chế độ, nhưng KHÔNG theo khuôn mẫu chung |
| CausalImpact (Bayesian structural time series) | 6/20 sự kiện ECB có ý nghĩa sau FDR-BH — thí điểm |

**Nhưng**: dùng hiệu ứng này làm lớp phủ chỉnh dự báo trực tiếp — **0/6 cặp
cải thiện có ý nghĩa**. Kết luận: nhân quả có vai trò XÁC THỰC/CHẨN ĐOÁN
(biết biến nào thật, biến nào giả), KHÔNG có vai trò cải thiện dự báo trực
tiếp — mốc HAR đã hấp thụ phần lớn thông tin đó qua tự hồi quy.

### 4.6 Tin tức (Pha 2) — kết luận âm nhất quán

4 cách biểu diễn văn bản thông cáo FOMC (từ điển HAWK/DOVE+TF-IDF, embedding
pretrained+PCA, chủ đề chính, trích xuất bằng LLM) — 183 giả thuyết, 0 sống
sót. **Phát hiện đáng chú ý**: biểu diễn càng "hiểu sâu" văn bản thì tín
hiệu càng YẾU đi (LLM tệ nhất, SPA p=1,000) — nhất quán với "không có tín
hiệu để bắt", không phải do phương pháp biểu diễn chưa đủ tốt.

Một hướng phụ (bất ngờ chính sách market-implied, USMPD/SF Fed data) ban
đầu dương nhưng cơ chế đã đăng ký bị bác bỏ khi kiểm chứng (hệ số sai dấu).

### 4.7 Định cỡ vị thế & Hiệu chuẩn — đã đa dạng, không cần góp ý thêm

- Định cỡ: Kelly+trần phá sản (đang dùng) thắng PPO, CVaR-PPO, REINFORCE,
  fuzzy Mamdani, bandit ngữ cảnh (phá sản thấp hơn PPO 26 lần)
- Hiệu chuẩn: Adaptive Conformal Inference (ACI, Gibbs & Candès 2021) thắng
  conformal tĩnh (LAC/APS) rõ rệt — độ phủ 90,1-90,8% so với 81,4-100% của
  bản tĩnh

---

## 5. ĐIỂM NGHẼN LỚN NHẤT — CẦN Ý TƯỞNG MỚI THẬT SỰ

### 5.1 Tail-risk USDJPY/USDCHF — BẾ TẮC PHƯƠNG PHÁP LUẬN, không phải thiếu ý tưởng thống kê

**Vấn đề đo được**: USDJPY vi phạm VaR 1% ở tỷ lệ 2,07% (kỳ vọng 1,0%),
USDCHF ES thừa 15%. Nguyên nhân đã CHẨN ĐOÁN RÕ:
- USDJPY: 4 ngày BOJ can thiệp tỷ giá (2022-09-22, 2022-10-21, 2022-10-24,
  2022-12-20 — lần đầu can thiệp từ 1998) chiếm 52% tổn thất dự báo trên
  kiểm định. Ngoài ra: `sd(z)` (độ lệch chuẩn lợi suất chuẩn hoá) TĂNG ĐƠN
  ĐIỆU qua các đoạn (1,014 → 1,100 → 1,136, +12%) — một xu hướng TRÔI THANG
  ĐO dần dần, KHÔNG chỉ do vài ngày cực đoan.
- USDCHF: SNB bỏ sàn EUR/CHF 15/01/2015 (nằm ở đoạn HUẤN LUYỆN, trước
  2021-10-13).

**BẾ TẮC THẬT SỰ** (đã xác nhận qua 5 hướng vá độc lập, tất cả đều thất
bại vì đúng MỘT lý do): **vấn đề tail-risk chỉ lộ ra ở đoạn KIỂM TRA
(2023-11-20 → nay), KHÔNG BAO GIỜ lộ ra ở đoạn KIỂM ĐỊNH.** Mọi cấu hình đã
thử — kể cả chính mốc V0 đang chạy — đều "ĐẠT" ba phép kiểm định (Kupiec,
Christoffersen, DQ) trên kiểm định. Vì quy tắc dự án là "chọn cấu hình dựa
trên hiệu năng đo được trên kiểm định, chấm kiểm tra một lần duy nhất",
KHÔNG có tín hiệu nào trên dữ liệu được phép dùng để phân biệt cách vá nào
thật sự sửa được lỗi và cách nào không — mọi hướng đều trông "đạt" như
nhau trước khi mở kiểm tra.

**5 hướng đã thử, TẤT CẢ chết vì đúng lý do trên**: phân vị mở rộng/cuộn,
CAViaR động, phân vị theo chế độ, cửa sổ họp NHTW định kỳ, và (mới nhất)
lịch can thiệp không định kỳ (BOJ/SNB) với hệ số nới 1,5×/2×/3×.

**Hai lựa chọn thật, không phải vấn đề kỹ thuật thuần túy** (đã tự nhận
diện trong tài liệu dự án):
(a) Chọn cấu hình dựa trên LÝ DO KINH TẾ thay vì đo trên kiểm định — đánh
    đổi: không còn là lựa chọn ngoài mẫu (out-of-sample) đúng nghĩa.
(b) Mở đoạn kiểm tra để CHẨN ĐOÁN (không phải chọn mô hình), cam kết KHÔNG
    quay lại sửa dù kết quả ra sao.

**CÂU HỎI CHO AI KHÁC**: có kỹ thuật thống kê/machine learning nào cho phép
VẪN giữ kỷ luật "chọn trên kiểm định, không nhìn kiểm tra" MÀ VẪN phát hiện
được một lỗi CHỈ tồn tại ngoài mẫu chọn? Ví dụ: có framework nào về "đánh
giá độ vững của mô hình rủi ro dưới phân phối dịch chuyển" (distribution
shift robustness) không cần nhìn trước dữ liệu thật của giai đoạn dịch
chuyển? Có cách nào mô phỏng/tạo synthetic một "giai đoạn kiểm định thứ
hai" có tính chất tương tự (regime dịch chuyển, biến động dồn cụm hiếm) mà
không lấy dữ liệu thật từ kiểm tra? Domain adaptation, distributionally
robust optimization (DRO), hay các kỹ thuật uncertainty quantification dưới
covariate shift có áp dụng được ở đây không?

### 5.2 Kết quả Hansen SPA ở mục tiêu R (h=1/5/20) — cần quyết định cách viết

5/54 phép kiểm SPA (kiểm cấp-họ) bác bỏ giả thuyết không sau hiệu chỉnh
Holm, TOÀN BỘ ở mục tiêu R (chuẩn hoá theo σ̂), KHÔNG có ở mục tiêu P. Trong
khi đó, Westfall-Young (kiểm cấp-cá-thể, nghiêm hơn) cho 0/0 sống sót ở MỌI
tổ hợp. Cơ chế khả dĩ: mô hình tham chiếu "chỉ σ̂" trên mục tiêu R suy biến
thành khí hậu học (theo đúng thiết kế), khiến SPA nhạy hơn với mốc yếu.
**Không cần AI khác góp ý phương pháp mới cho việc này** — đây là vấn đề
diễn giải/viết văn, không phải thiếu kỹ thuật.

### 5.3 Sổ dự báo tích luỹ — bị chặn bởi lịch, không phải kỹ thuật

Cần ~30 phiên giao dịch thật để tính hiệu chuẩn trên dự báo THẬT hệ thống
đã đưa ra (không phải backtest). Mới có ~10/30. Không có giải pháp kỹ thuật
nào rút ngắn — chỉ chờ thời gian.

---

## 6. ĐÃ THỬ VÀ THẤT BẠI — DANH SÁCH TỔNG HỢP (để AI khác KHÔNG đề xuất lại)

- Bất kỳ biến thể nào của: cửa sổ ước lượng tĩnh/mở rộng/cuộn cho phân vị đuôi
- CAViaR (mọi dạng chuẩn: SAV, AS, IG)
- Phân tầng theo chế độ biến động (bất kỳ số nhóm nào)
- Điều kiện hoá theo lịch họp NHTW ĐỊNH KỲ
- Điều kiện hoá theo lịch can thiệp KHÔNG định kỳ (đã biết trước)
- EVT/POT với ngưỡng cố định (90%, 95%)
- Bất kỳ mô hình ML/DL/foundation-model nào cho dự báo BIẾN ĐỘNG (đã thử 14+ mô hình, HAR luôn thắng)
- Bất kỳ cách biểu diễn văn bản nào cho tin tức FOMC (4 cách, tất cả âm)
- RuleFit, Fixed-Share, Stacking (đều thua phương án đơn giản hơn đang dùng)
- Vision Transformer cho ảnh nến (thua CNN)
- Named candlestick patterns cổ điển (doji/hammer/engulfing...)

## 7. CÂU HỎI MỞ CHO AI KHÁC — NƠI Ý TƯỞNG MỚI THẬT SỰ CÓ GIÁ TRỊ

1. **[Ưu tiên cao nhất]** Mục 5.1 — kỹ thuật nào cho phép kiểm chứng một
   cách vá rủi ro KHÔNG cần nhìn dữ liệu kiểm tra, khi lỗi chỉ biểu hiện
   ngoài mẫu chọn?
2. Có phương pháp thống kê nào để ĐO ĐƯỢC "độ trôi thang đo" (scale drift)
   của `sd(z)` theo thời gian cho USDJPY MÀ KHÔNG cần biết trước ngày xảy
   ra sự kiện cụ thể — tức phát hiện được xu hướng drift từ chính cấu trúc
   thống kê nội tại, không dựa vào lịch sự kiện ngoại sinh?
3. Ngưỡng EVT/POT hiện dùng cố định (90%/95%) — có đáng thử phương pháp
   chọn ngưỡng tự động/dữ liệu-dẫn-dắt (đồ thị mean-excess, ước lượng Hill)
   không, và nó có giải quyết được đúng bế tắc ở mục 5.1 không (khả năng
   thấp, vì bế tắc là về GIAO THỨC KIỂM CHỨNG chứ không phải về ước lượng
   đuôi) — cần AI khác đánh giá độc lập.
4. Pha 3 (vĩ mô đơn giản, chỉ dùng lãi suất liên ngân hàng tháng) — độc
   canh thật, nhưng ưu tiên thấp vì Pha 3B đã mở rộng gấp nhiều lần. Có
   nguồn dữ liệu vĩ mô "độ bất ngờ" nào (Economic Policy Uncertainty Index,
   Citi Economic Surprise Index) đáng thử không?
