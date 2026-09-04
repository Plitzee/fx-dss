# Ai đang làm cùng thứ với mình — và họ tìm ra gì

Lập 30/08/2026. Xếp theo tầng của hệ thống. Mỗi mục ghi rõ: bài đó làm gì, nó
**xác nhận** hay **cạnh tranh** với ta, và nên làm gì với nó.

Ba mức đánh dấu:

- 🟢 **Xác nhận** — họ ra cùng kết luận bằng đường khác, trích dẫn để củng cố
- 🟡 **Lấp chỗ trống** — họ có thứ ta chưa có, nên đọc kỹ
- 🔴 **Cạnh tranh trực tiếp** — làm gần y hệt, phải trích dẫn và nói rõ mình khác chỗ nào

---

## Tầng 2 — dự báo biến động

### 🔴 Foundation model có thắng được HAR không?
**"Forecasting Realized Volatility with Time Series Foundation Models: A Comparison with Econometric Benchmarks"** — [arXiv:2607.05291](https://arxiv.org/html/2607.05291), 7/2026.

Bài gần nhất với tầng 2 của ta, và dùng **đúng bộ chỉ số ta dùng**: QLIKE, hồi
quy Mincer–Zarnowitz, Diebold–Mariano, Model Confidence Set.

Kết luận của họ: foundation model **không** thắng được Log-HAR một cách nhất
quán. Chỉ Tiny Time Mixers — mô hình **nhỏ nhất** — hơn được 1,3–1,8%. Và câu
quan trọng nhất: *"chọn đúng kiến trúc quan trọng hơn chuyện chọn foundation
model hay mô hình kinh tế lượng."* Trung bình đều tay của TTM và Log-HAR vào
được MCS ở 98–100% tài sản.

**Nghĩa là gì với ta:** kết quả GBM thua HAR của ta không phải cá biệt, và lựa
chọn **tổ hợp** của ta trùng đúng khuyến nghị của họ. Đây là bài phải trích dẫn
ở phần bảo vệ tầng 2.

### 🟢 Mô hình phi tuyến chuyển chế độ thắng cả ML
**Federal Reserve FEDS 2025-061**, *Linear and nonlinear econometric models against machine learning models* — [PDF](https://www.federalreserve.gov/econres/feds/files/2025061pap.pdf)

THAR/STHAR thắng cả ML lẫn tuyến tính, nhất là giai đoạn biến động cao. Đây là
cơ sở cho việc ta chọn **STHARQ** chứ không phải HAR trơn.

### 🟢 "Does anything beat linear models?"
Branco et al. (2024), *Journal of Empirical Finance* — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0927539824000598)

Tiêu đề đã là câu trả lời. Trích dẫn cùng bài trên.

### 🟡 Tổ hợp cho biến động tỷ giá — đúng bối cảnh FX
**"Forecasting exchange rate volatility: An amalgamation approach"** — *Journal of International Financial Markets* (2024), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1042443124001331)

FX-specific, và về **tổ hợp mô hình** — đúng thứ ta làm. Nên đọc để đối chiếu
cách họ chọn trọng số tổ hợp (ta dùng trung bình hình học đơn giản).

### Nền tảng (đã dùng)
- Bollerslev, Patton, Quaedvlieg (2016), *Exploiting the errors* → HARQ
- Patton, Sheppard (2015), *Good volatility, bad volatility* → SHAR
- Patton (2011), *Volatility forecast comparison using imperfect volatility proxies* → vì sao chỉ QLIKE và MSE được dùng để xếp hạng

---

## Tầng 6 — hiệu chuẩn và khoảng dự báo

### 🔴 Đây là bài gần nhất với công việc của bạn
**"Taming Tail Risk: Regime-Weighted Conformal Calibration for Nonstationary Value-at-Risk"** — [arXiv:2602.03903v2](https://arxiv.org/html/2602.03903v2), 7/2026.

Họ làm gần y hệt tầng 6 của ta:

| | Họ (RWC) | Ta |
|---|---|---|
| Ý tưởng lõi | conformal có **trọng số theo chế độ** | conformal **phân tầng** theo chế độ biến động |
| Trọng số | nhân suy giảm thời gian × nhân Gauss theo đặc trưng thị trường | phân tầng cứng theo tam/nhị phân vị |
| Backtest | Kupiec, Christoffersen, DQ | **giống hệt** |
| Điểm đặc biệt | **tỷ lệ vi phạm theo ngũ phân vị biến động** | **độ phủ theo tam phân vị biến động** |

Kết luận của họ đáng chú ý: *"trọng số theo chế độ cải thiện hiệu chuẩn ở giai
đoạn căng thẳng khi mô hình nền thích ứng chậm, nhưng gần như không thêm gì khi
mô hình nền đã thích ứng nhanh."*

**Đó chính xác là thứ ta đo được**: sau khi đổi sang dự báo σ̂ tốt hơn (thích ứng
nhanh hơn), ACI và phân tầng nhiều lớp **hết tác dụng** — kiểm định chọn
Mondrian 2 tầng tĩnh. Hai bên độc lập ra cùng một quy luật.

**Phải làm:** trích dẫn bài này ở mục tầng 6, và nói rõ ta khác ở đâu — ta dùng
phân tầng cứng chứ không dùng nhân, và ta có thêm bảng tầm hạn mà họ không có.

### 🟡 Conformal thời gian với điều chỉnh trực tuyến
**"Temporal Conformal Prediction (TCP)"** — [arXiv:2507.05470](https://arxiv.org/abs/2507.05470v1)

Cơ chế Robbins–Monro điều chỉnh độ phủ theo thời gian thực, so với GARCH và
Historical Simulation. Cùng họ với ACI mà ta đã cài. Đọc để biết biến thể.

### 🟡 Conformal cho VaR một phía
**"Proxy-Reliance Control in Conformal Recalibration of One-Sided Value-at-Risk"** — [arXiv:2603.22569](https://arxiv.org/html/2603.22569)

VaR là bài toán **một phía**, còn khoảng dự báo của ta là hai phía. Với phiếu
quyết định thì cái ta thật sự cần là một phía (đuôi trái). Đây là chỗ trống của
ta, đáng đọc.

### 🟡 Khung tổng quát
**Angelopoulos, Bates, Candès et al., "Conformal Risk Control"** — [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)

Tổng quát hoá conformal từ độ phủ sang **bất kỳ hàm rủi ro nào**. Nếu muốn kiểm
soát trực tiếp kỳ vọng lỗ thay vì độ phủ thì đây là khung.

### Nền tảng (đã dùng)
- Gibbs, Candès (2021), *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS
- Gibbs, Candès (2024), *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts*, JMLR

---

## Tầng 5 — bộ chỉ số đánh giá

### 🔴 Họ dùng gần đúng bộ chỉ số ta vừa xây
**"Probabilistic Forecasting Cryptocurrencies Volatility: From Point to Quantile Forecasts"** — [arXiv:2508.15922](https://arxiv.org/html/2508.15922v1)

Dùng CRPS (xấp xỉ qua pinball 99 mức), pinball, MARFE (sai lệch hiệu chuẩn),
Winkler Score. So 12 mô hình nền + 3 meta.

Điểm quan trọng nhất với ta: họ chuyển **dự báo điểm thành phân phối** bằng
*Quantile Estimation through Residual Simulation* (QRS) — và QRS **thắng** cả
Quantile Regression Forests lẫn Quantile Linear Regression, dù đơn giản hơn.

**Đó đúng là thứ tôi làm cho TSF** (`src/metrics/recalib.py`): lấy phần dư trên
tập hiệu chuẩn rồi dựng phân phối. Bài này là cơ sở học thuật cho lựa chọn đó.

### Nền tảng (đã dùng)
- Gneiting, Raftery (2007), quy tắc chấm điểm chính đáng
- Diebold, Gunther, Tay (1998), PIT
- Kupiec (1995) · Christoffersen (1998) · Engle & Manganelli (2004)
- Patton, Ziegel, Chen (2019), FZ0 cho cặp (VaR, ES)
- Hansen, Lunde, Nason (2011), Model Confidence Set

---

## Tầng 4 — định cỡ vị thế

### 🔴 Kelly kết hợp với hệ số theo biến động
**"Sizing the Risk: Kelly, VIX, and Hybrid Approaches in Put-Writing on Index Options"** — [arXiv:2508.16598](https://arxiv.org/html/2508.16598v1), 8/2025.

So Kelly với cách chia cỡ theo chế độ biến động (dùng VIX), rồi đề xuất **lai
hai cái**. Kết luận: bản lai "cân bằng tốt hơn giữa sinh lời và kiểm soát sụt
giảm".

**Đây gần đúng quy tắc của ta** — `min(Kelly, k_vol × trần)`. Khác: họ dùng VIX
(biến động ngụ ý), ta dùng σ̂ dự báo từ realized variance vì FX không có VIX
tương đương phổ biến. Phải trích dẫn và nêu rõ điểm khác đó.

**Chỗ trống của họ mà ta có:** họ không có hệ số danh mục. Lỗ hổng "trần rủi ro
chỉ đúng cho một vị thế" mà ta đo được (6 lệnh → phá sản 73,6%) là đóng góp
riêng, tôi chưa thấy bài nào xử lý.

---

## Nhánh TSF — mô hình sâu

### 🔴 Benchmark lớn nhất, và kết luận ngược trực giác
**"Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance"** — [arXiv:2603.01820](https://arxiv.org/html/2603.01820v1), 3/2026.

15+ kiến trúc: DLinear, NLinear, **iTransformer, PatchTST**, Mamba/Mamba2,
LSTM/xLSTM, TFT, và các bản lai. Dữ liệu 2010–2025.

Kết luận: **transformer thuần hoạt động không nhất quán**. Thắng cuộc là
**VSN+LSTM** (Sharpe cao nhất) và **xLSTM** (bền nhất trước chi phí giao dịch).
Mô hình có thiên kiến quy nạp về khử nhiễu và bộ nhớ thích ứng thắng rõ.

**Nghĩa là gì với nhánh TSF:** kết quả CAIFormer thua SPACE không phải bất
thường — nó khớp với xu hướng chung mà bài này ghi nhận. Nên trích dẫn ở phần
thảo luận, và cân nhắc thêm **xLSTM** vào benchmark.

### 🟡 Giải thích được cho dự báo tỷ giá
**"Enhancing Exchange Rate Forecasting with Explainable Deep Learning Models"** — [arXiv:2410.19241](https://arxiv.org/html/2410.19241v1)

Đúng bối cảnh FX + đúng hướng giải thích được. Liên quan trực tiếp tới tầng 6.

---

## Nhánh khai phá mẫu

### Nền tảng
**Lin, Keogh, Wei, Lonardi (2007)**, *Experiencing SAX* — [Springer](https://link.springer.com/article/10.1007/s10618-007-0064-z) · [PDF](https://cs.gmu.edu/~jessica/SAX_DAMI_preprint.pdf)

Bài HuyH đang dùng làm nền.

### 🟡 SAX trong tài chính, bản gần đây
**"CPC-SAX: Data mining of financial chart patterns with symbolic aggregate approXimation and instance-based multilabel classification"** — *Machine Learning with Applications* (2024), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405918824000175)

SAX + phân loại đa nhãn cho mẫu biểu đồ. Đây là bản hiện đại hoá của đúng thứ
HuyH làm — nên đọc và trích, vì hội đồng sẽ hỏi "SAX 2007 có cũ quá không".

### 🟡 SAX kết hợp thuật toán di truyền
**"Multi-dimensional pattern discovery in financial time series using SAX-GA with extended robustness"** — [ACM](https://dl.acm.org/doi/10.1145/2464576.2464664)

Khai phá mẫu **đa chiều** — hướng mở rộng tự nhiên cho nhánh HuyH (hiện chỉ một
chiều mỗi lần).

---

## Bối cảnh chung — hệ hỗ trợ quyết định và XAI

### 🟡 Khảo sát XAI cho chuỗi thời gian tài chính
**"A Survey of Explainable Artificial Intelligence (XAI) in Financial Time Series Forecasting"** — *ACM Computing Surveys* (2025), [ACM](https://dl.acm.org/doi/abs/10.1145/3729531)

Bài khảo sát để đặt tầng 6 vào bản đồ tài liệu. Phần "hệ hỗ trợ quyết định khác
mô hình dự báo ở chỗ nào" nên dựa vào đây.

### 🟡 Báo cáo ngành
**CFA Institute, "Explainable AI in Finance" (2025)** — [RPC](https://rpc.cfainstitute.org/research/reports/2025/explainable-ai-in-finance)

Góc nhìn người hành nghề, hữu ích cho phần mở đầu và phần ứng dụng.

---

## Ba khoảng trống mà tôi không tìm thấy ai làm

Đây là chỗ luận văn có thể tuyên bố đóng góp:

1. **Hệ số danh mục cho trần rủi ro.** Mọi bài về định cỡ đều xét một vị thế.
   Việc đo được rằng mở 6 lệnh cùng hướng USD ở đúng cỡ khuyến nghị cho phá sản
   **73,6%** thay vì 1%, và luật `1/√(k + k(k−1)ρ)` sửa được, tôi chưa thấy ai
   trình bày.
2. **Bảng tầm hạn trên phiếu quyết định.** Conformal nhiều tầm hạn thì có, nhưng
   việc chỉ ra rằng người dùng đọc "P(chạm stop) 5%" rồi giữ 10 phiên thì thực
   tế là **50%** — và in thẳng bảng đó ra — là vấn đề thiết kế DSS, không phải
   vấn đề mô hình. Ít bài chạm tới.
3. **Trượt giá đo từ M1 thay vì giả định.** Nhiều bài giả định khớp đúng tại mức
   dừng lỗ. Đo 60.617 lần chạm stop và cho thấy p95 bằng 35% khoảng cách stop —
   rồi quy ra hệ số cắt 0,92 — là đóng góp thực nghiệm cụ thể.

## Thứ tự đọc tôi khuyên

Nếu chỉ đọc được ba bài trước khi viết:

1. **arXiv 2602.03903** (RWC) — gần nhất với tầng 6, phải trích
2. **arXiv 2607.05291** (foundation models vs HAR) — hợp thức hoá tầng 2 và kết luận âm về ML
3. **arXiv 2603.01820** (benchmark deep learning) — hợp thức hoá kết quả CAIFormer của nhánh TSF
