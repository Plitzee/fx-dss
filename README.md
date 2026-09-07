# FX-DSS — Hệ hỗ trợ quyết định giao dịch ngoại hối

**Trang chạy thật:** https://fx-dss.vercel.app
**Luận văn MIS.** Repo này chứa dữ liệu đã xử lý, toàn bộ mã, và nhật ký đầy đủ
mọi thí nghiệm — kể cả những cái thất bại.

---

## 1. Tóm tắt cho người không đọc code

Hệ thống lấy dữ liệu giá của 6 cặp ngoại hối chính (EUR/USD, GBP/USD, USD/JPY,
AUD/USD, USD/CAD, USD/CHF) từ 2010 đến nay, rồi mỗi ngày trả lời ba câu hỏi cho
nhà đầu tư:

1. **Phiên tới giá sẽ động mạnh hay yên ắng?** — đây là câu **có** trả lời
   đáng tin, đo được bằng số.
2. **Phiên tới giá sẽ tăng hay giảm?** — đã kiểm bằng nhiều phương pháp độc
   lập, và câu trả lời trung thực là: **không đoán được**, ít nhất là bằng các
   công cụ đã thử trên dữ liệu này.
3. **Nếu vào lệnh thì rủi ro bao nhiêu?** — trả lời bằng cỡ lệnh khuyến nghị,
   khoảng dừng lỗ, và ước tính lỗ tối đa trong ngày xấu (VaR/ES).

Giao diện có hai chế độ: **"Nhà đầu tư"** (mặc định, không thuật ngữ, đọc xong
là biết nên làm gì) và **"Phân tích"** (đầy đủ chỉ số kỹ thuật, cho ai muốn xem
tận gốc con số ra từ đâu).

Điều quan trọng nhất về triết lý dự án: **mọi con số trên giao diện đều phải
truy được về một phép đo cụ thể**, và **kết quả âm được báo cáo đầy đủ như kết
quả dương** — nếu một phương pháp không mang lại gì, giao diện nói thẳng điều
đó thay vì giấu đi.

---

## 2. Sơ đồ hệ thống

```mermaid
flowchart TB
    subgraph DATA["📊 Dữ liệu"]
        D1[Giá D1/H1 6 cặp<br/>2010 → nay]
        D2[Lịch sự kiện<br/>FRED · 18 loại]
        D3[Lãi suất, spread,<br/>trượt giá]
    end

    subgraph TANG2["🌊 Tầng biến động — HAR vòng 7"]
        V1[Realized variance<br/>từ nến 5 phút]
        V2[Mô hình HAR<br/>ngày/tuần/tháng + jump]
        V3["σ̂ — dự báo biên độ<br/>dao động phiên tới"]
        V1 --> V2 --> V3
    end

    subgraph TANG3["🎲 Tầng ba xác suất"]
        P1[4 mô hình con:<br/>khí hậu học · quán tính ·<br/>chỉ σ̂ · σ̂+chế độ]
        P2["Tổ hợp trực tuyến<br/>(Hedge) — tự học,<br/>hạ trọng số mô hình sai"]
        P3["Ba ô: Giảm / Đi ngang / Tăng"]
        P1 --> P2 --> P3
    end

    subgraph RUI_RO["⚠️ Tầng rủi ro"]
        R1[VaR & ES<br/>+ backtest Kupiec/DQ]
        R2[P chạm dừng lỗ<br/>theo thời gian giữ]
        R3[Cỡ lệnh Kelly<br/>có trần phá sản]
    end

    subgraph NGHIEN_CUU["🔬 Nhánh nghiên cứu — khai phá quy luật"]
        N1[1.890 giả thuyết<br/>kỹ thuật, liệt kê đầy đủ]
        N2[Westfall–Young<br/>+ đối chứng có điều kiện]
        N3["Kết quả: 0 quy luật<br/>sống sót (D1 và H1)"]
        N1 --> N2 --> N3
    end

    DATA --> TANG2
    DATA --> RUI_RO
    D2 --> P1
    V3 --> P1
    V3 --> RUI_RO
    TANG2 --> NGHIEN_CUU
    D1 --> NGHIEN_CUU

    subgraph API["⚙️ API — FastAPI"]
        A1["/forecast /risk /events /models"]
    end
    TANG3 --> API
    RUI_RO --> API
    NGHIEN_CUU -.kết luận âm, không nuôi API.-> API

    subgraph UI["🖥️ Giao diện web"]
        U1[Biểu đồ nến<br/>+ chỉ báo kỹ thuật]
        U2[Ba ô xác suất<br/>+ cảnh báo sự kiện]
        U3[Phiếu rủi ro<br/>+ xuất xứ từng con số]
    end
    API --> UI
    UI --> DEPLOY[Vercel<br/>+ GitHub Actions 4 lần/ngày]
```

**Đọc sơ đồ này thế nào:** dữ liệu chảy vào hai tầng chính — **tầng biến động**
(cho ra σ̂, đã chứng minh có tín hiệu thật) và **tầng rủi ro** (VaR/ES, cỡ
lệnh). Ba ô xác suất được tính từ σ̂ cộng với lịch sự kiện. **Nhánh khai phá quy
luật** chạy song song để trả lời câu "có chỉ báo kỹ thuật nào giúp đoán hướng
không" — nó **không nuôi API sản xuất**, vì kết luận của nó là "không có gì cả"
(xem mục 5).

---

## 3. Phương pháp — bằng ngôn ngữ đơn giản

| Câu hỏi | Phương pháp | Vì sao chọn cách này |
|---|---|---|
| Phiên tới giá động mạnh cỡ nào? | **HAR vòng 7** trên realized variance (dữ liệu 5 phút) | Mô hình kinh điển trong tài chính lượng, thắng cả các mạng nơ-ron hiện đại đã thử (đo trong `docs/ML_DL_VONG7.md`) |
| Kết hợp nhiều mô hình xác suất thế nào? | **Tổ hợp trực tuyến (thuật toán Hedge)** | Trọng số tự hạ với mô hình vừa đoán sai — đúng nghĩa "học từ bài học trước", không cần huấn luyện lại từ đầu |
| Ngày họp ngân hàng trung ương / NFP / CPI ảnh hưởng thế nào? | Đo **tỷ lệ biến động thật** so với ngày thường, kèm khoảng tin cậy | Không dùng nhãn "tác động cao/thấp" theo quy ước như các lịch kinh tế khác — chỉ tin số đã đo |
| Cỡ lệnh bao nhiêu là an toàn? | **Kelly có trần phá sản**, điều chỉnh theo biến động và sụt giảm | Không dùng dự báo hướng làm lợi thế — chỉ dùng carry (chênh lệch lãi suất) đo được thật |
| Lỗ tối đa ngày xấu là bao nhiêu? | **VaR / ES** với backtest Kupiec, Christoffersen, DQ | Ba phép kiểm định thống kê chuẩn để biết con số có đáng tin không, không chỉ đưa ra suông |
| Có chỉ báo kỹ thuật nào giúp đoán hướng không? | **Khai phá quy luật có kiểm định bội** (Westfall–Young), đối chứng có điều kiện, thử cả ở D1 và H1 (592.343 quan sát) | Tránh "thấy quy luật giả" — hiện tượng rất phổ biến khi thử hàng nghìn giả thuyết mà không hiệu chỉnh |
| Nhiều cặp tiền có ảnh hưởng lẫn nhau không? | **Lan truyền biến động Diebold-Yilmaz** (phân rã phương sai từ VAR) | Phương pháp kinh tế lượng chuẩn cho câu hỏi "biến động của cặp này có phải nguyên nhân của cặp kia" |

---

## 4. Nguồn tham khảo — mỗi phương pháp gắn đúng bài gốc

Ưu tiên link **mở, đọc được không cần trả phí** (bản thảo tác giả, arXiv, kho
lưu trữ trường/ngân hàng trung ương) thay vì link tạp chí trả phí, để ai cũng
tra cứu được.

**Tầng biến động (σ̂)**
- Corsi (2009), *A Simple Approximate Long-Memory Model of Realized
  Volatility* — mô hình HAR gốc.
  [PDF (Oxford Academic)](https://academic.oup.com/jfec/article-pdf/7/2/174/2543795/nbp001.pdf) ·
  [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1365738)
- Bollerslev, Patton, Quaedvlieg (2016), *Exploiting the Errors: A Simple
  Approach for Improved Volatility Forecasting* — nguồn của HARQ, thành phần
  trong HAR vòng 7.
  [ResearchGate](https://www.researchgate.net/publication/283686848_Exploiting_the_errors_A_simple_approach_for_improved_volatility_forecasting)
- Barndorff-Nielsen & Shephard (2004), *Power and Bipower Variation with
  Stochastic Volatility and Jumps* — cơ sở cho bipower variation/semivariance
  dùng trong HAR vòng 7.
  [PDF (Duke)](https://public.econ.duke.edu/~get/browse/courses/883/Spr16/COURSE-MATERIALS/Z_Papers/BNSJFEC2004.pdf)

**Tổ hợp trực tuyến (thuật toán Hedge)**
- Freund & Schapire (1997), *A Decision-Theoretic Generalization of On-Line
  Learning and an Application to Boosting* — nguồn gốc thuật toán Hedge/trọng
  số mũ mà `balop.ToHopTrucTuyen` dùng.
  [PDF](https://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf)

**Khai phá quy luật & kiểm định bội**
- Westfall & Young (1993), *Resampling-Based Multiple Testing* (sách) — thủ
  tục step-down maxT dùng trong `run_quyluat.py`/`quyluat_h1.py`. Không có bản
  mở của sách; mô tả kỹ thuật mở tương đương:
  [Dudoit & van der Laan, kỹ thuật báo cáo Berkeley](https://statistics.berkeley.edu/sites/default/files/tech-reports/633.pdf)
- Hansen (2005), *A Test for Superior Predictive Ability* (SPA) — tiêu chí
  dừng của giai đoạn khai phá quy luật (`docs/REPLAN_2026.md` §10.4).
  [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569)
- Hansen, Lunde & Nason (2011), *The Model Confidence Set* — dùng để chọn cấu
  hình HAR vòng 7 trong lưới 1.024 cấu hình.
  [PDF (tác giả)](https://www.kevinsheppard.com/files/teaching/mfe/advanced-econometrics/Hansen_Lunde_Nason.pdf)
- Hutchinson, Kyziropoulos, O'Brien, O'Reilly & Sharma (2022), *Technical
  Trading Rule Profitability in Currencies: It's All About Momentum* — bằng
  chứng nền cho việc dùng TSMOM làm đối chứng null của trục hướng.
  [PDF mở (Queen's University Belfast)](https://pureadmin.qub.ac.uk/ws/portalfiles/portal/370624058/1_s2.0_S0275531922001659_main_1_.pdf)

**Rủi ro — VaR/ES, hiệu chuẩn**
- Kelly (1956), *A New Interpretation of Information Rate* — công thức cỡ
  lệnh Kelly dùng trong `position_sizing.py`.
  [PDF](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf)
- Kupiec (1995), *Techniques for Verifying the Accuracy of Risk Measurement
  Models* — kiểm định tỷ lệ vi phạm VaR.
  [Bản thảo Fed (RePEc, mở)](https://ideas.repec.org/p/fip/fedgfe/95-24.html)
- Christoffersen (1998), *Evaluating Interval Forecasts* — kiểm định vi phạm
  có dính cụm.
  [Trang RePEc/EconPapers](https://econpapers.repec.org/RePEc:ier:iecrev:v:39:y:1998:i:4:p:841-62)
- Engle & Manganelli (2004), *CAViaR: Conditional Autoregressive Value at Risk
  by Regression Quantiles* — nguồn của kiểm định DQ và hướng vá đuôi phân phối
  còn để ngỏ cho USDJPY/USDCHF.
  [Bản thảo NBER (mở)](https://www.nber.org/papers/w7341)
- Patton, Ziegel & Chen (2019), *Dynamic Semiparametric Models for Expected
  Shortfall (and Value-at-Risk)* — kiểm định FZ0 cho cặp (VaR, ES).
  [arXiv](https://arxiv.org/pdf/1707.05108)
- Diebold, Gunther & Tay (1998), *Evaluating Density Forecasts with
  Applications to Financial Risk Management* — kiểm định PIT.
  [PDF (NYU, mở)](https://archive.nyu.edu/bitstream/2451/14779/1/SOR-98-6.pdf)

**Lan truyền biến động giữa các cặp**
- Diebold & Yilmaz (2012), *Better to Give than to Receive: Predictive
  Directional Measurement of Volatility Spillovers* — công thức FEVD dùng
  trong `spillover_dy.py`.
  [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1536123)
- Rubaszek, Szafranek & Uddin (2025), *Intraday Volatility Connectedness on
  the Forex Market: The Role of Uncertainty* — bài dùng đúng bộ 5/6 cặp tiền
  của repo này, là lý do thử nghiệm này được tiến hành.
  [Trang RePEc/IDEAS](https://ideas.repec.org/a/eee/jimfin/v157y2025ics0261560625001330.html)
- Baruník & Křehlík (2016), *Asymmetric Volatility Connectedness on Forex
  Markets* — bản tiền thân mở truy cập của hướng lan truyền biến động FX.
  [arXiv](https://arxiv.org/pdf/1607.08214)

**Nguồn dữ liệu**
- FRED (Federal Reserve Bank of St. Louis) — lịch công bố kinh tế và lãi suất.
  [Tài liệu API](https://fred.stlouisfed.org/docs/api/fred/) ·
  [Đăng ký khoá miễn phí](https://fredaccount.stlouisfed.org/apikey)

---

## 5. Kết quả — nói thẳng, không tô hồng

### Có thật, nhưng nhỏ

Ô giữa (đi ngang), đo **ngoài mẫu** — tức trên dữ liệu chưa từng dùng để chọn
mô hình:

| tầm hạn | BSS (so với đoán theo tần suất lịch sử) | Khoảng tin cậy 95% |
|---|---|---|
| 1 phiên | **+0,0152** | [+0,0103; +0,0209] — có ý nghĩa |
| 5 phiên | **+0,0074** | [+0,0009; +0,0144] — có ý nghĩa |
| 20 phiên | −0,0058 | [−0,0193; +0,0166] — chưa chứng minh được |

Đọc là: hệ thống nhỉnh hơn "cứ đoán theo tần suất lịch sử" khoảng 1,5% ở tầm
hạn 1 phiên. Có thật, nhưng **không phải một lợi thế lớn**.

### Đã đo và xác nhận là KHÔNG có

Dự báo **hướng giá** (tăng/giảm) không có tín hiệu — xác nhận độc lập bằng
5 phương pháp khác nhau:

- Momentum: Sharpe −0,16 · Carry: Sharpe −0,05
- AUC hướng: 0,46–0,53 (không phân biệt được với việc tung đồng xu)
- Khai phá quy luật: 1.890 giả thuyết kỹ thuật, thử ở cả D1 (21.596 quan sát)
  và H1 (592.343 quan sát) — **0 quy luật sống sót** sau kiểm định bội
- Phản ứng quanh sự kiện: 18 loại (NFP, CPI, GDP, họp NHTW…) — **0/18** có
  thiên lệch hướng có ý nghĩa
- Lan truyền biến động chéo cặp (Diebold-Yilmaz): thử ở cả D1 và H1, không cải
  thiện dự báo, thậm chí tệ hơn ở H1 (`docs/KETQUA_VONG7.md`)

### Trong quá trình thử, đã bắt được và sửa các lỗi thống kê thật

Đáng nói vì đây chính là kỷ luật giúp kết quả đáng tin: một lần biến kiểm soát
đặt sai dạng làm hệ số hồi quy phóng đại giả (t = 31,7 từ một quan hệ vô nghĩa
về mặt logic), một lần lỗi định tuyến tham số làm 9 dòng kết quả tính sai. Cả
hai đều được phát hiện, sửa, và ghi lại công khai trong `docs/`.

### Rủi ro — 4/6 cặp đạt, 2/6 chưa

Backtest VaR/ES (Kupiec + Christoffersen + DQ) ở mức 99%: EURUSD, GBPUSD,
AUDUSD, USDCAD đạt cả ba. **USDJPY và USDCHF chưa đạt** — đã thử vá 2 lần,
thất bại, nguyên nhân xác định là đuôi phân phối có cấu trúc động mà mô hình
hiện tại (ước lượng vô điều kiện) chưa bắt được.

### Toàn mạch — không phải máy in tiền

Backtest ~26 năm: vốn cuối kỳ **1,004–1,037 lần** vốn ban đầu, không cấu hình
nào cháy tài khoản. Gần như hoà vốn. Đây là hệ **hỗ trợ quyết định trung thực**,
không phải hệ thống kiếm lời.

---

## 6. Cấu trúc repo

```
fx-dss/
├── data/               dữ liệu đã xử lý (giá, sự kiện, lãi suất, spread…)
├── src/                mã tính toán cốt lõi
│   ├── volfc2.py          HAR vòng 7 — dự báo σ̂ (tầng biến động)
│   ├── balop.py           ba mô hình xác suất + tổ hợp trực tuyến (Hedge)
│   ├── position_sizing.py định cỡ vị thế Kelly có trần
│   ├── sukien_profile.py  đo phản ứng giá quanh sự kiện
│   ├── run_quyluat.py     phễu khai phá quy luật ở D1
│   ├── quyluat_h1.py      phễu khai phá quy luật ở H1 (592k quan sát)
│   ├── kiem_pheu.py       kiểm chứng độ nhạy của phễu (đối chứng âm/dương)
│   ├── spillover_dy.py    lan truyền biến động Diebold-Yilmaz giữa các cặp
│   ├── va_duoi.py         thử vá đuôi phân phối USDJPY/USDCHF
│   ├── hieuchuan_lai.py   thử hiệu chuẩn lại xác suất (isotonic/nhiệt độ)
│   ├── metrics.py         VaR/ES, Kupiec, Christoffersen, DQ, MCS…
│   └── split.py           chia huấn luyện/kiểm định/kiểm tra — chống rò rỉ
├── api/main.py         FastAPI — mọi endpoint phục vụ giao diện
├── web/                giao diện: HTML/CSS/JS thuần + Lightweight Charts
├── collect/            thu thập dữ liệu (giá, sự kiện FRED, lãi suất)
├── jobs/cap_nhat.py    việc định kỳ: tải giá, tính lại, ghi sổ dự báo
├── docs/               MỌI thí nghiệm, kết quả, kể cả thất bại — xem mục 7
└── .github/workflows/  tự động cập nhật 4 lần/ngày + triển khai Vercel
```

## 7. Tài liệu chi tiết — nếu cần đào sâu

| Muốn biết gì | Đọc file nào |
|---|---|
| Toàn bộ kế hoạch nghiên cứu, đã duyệt | `docs/REPLAN_2026.md` |
| Kết quả tầng biến động (HAR vòng 7 vs ML/DL) | `docs/ML_DL_VONG7.md`, `docs/KETQUA_VONG7.md` |
| Kết quả khai phá quy luật, đủ cả D1 và H1 | `docs/GIAIDOAN2_QUYLUAT.md` |
| Backtest VaR/ES chi tiết từng cặp | `docs/CHISO_DANHGIA.md` |
| Việc còn phải làm, xếp theo ưu tiên | `docs/KEHOACH_2026Q4.md` |
| Quy tắc niêm phong dữ liệu (rất quan trọng, đọc trước khi chạy) | `docs/KHOA_SO.md` |
| Giải thích UI theo lối nói chuyện, dùng để báo cáo | `docs/BAOCAO_UI.md` |

## 8. Chạy thử

```bash
git clone <repo> && cd fx-dss
pip install numpy pandas scipy statsmodels requests fastapi "uvicorn[standard]"

# tự kiểm các module cốt lõi
python src/balop.py
python src/split.py

# chạy API + giao diện tại chỗ
python -m uvicorn api.main:app --port 8899
python web/build.py
```

Triển khai thật chạy tự động qua GitHub Actions 4 lần/ngày (`.github/workflows/capnhat.yml`) —
tải giá mới, cập nhật lịch sự kiện, ghi sổ dự báo, dựng lại và đẩy lên Vercel.

## 9. Ba điều phải biết trước khi động vào dữ liệu

1. **Đọc `docs/KHOA_SO.md` trước.** Có một tập dữ liệu bị niêm phong (6 cặp
   tiền chéo + toàn bộ 2026) để giữ tính ngoài mẫu cho lần chấm điểm cuối
   cùng. Mở sớm là mất vĩnh viễn giá trị kiểm chứng của nó.
2. **Sổ dự báo (`data/so_dubao/`) chỉ được thêm, không được sửa.** Đây là bằng
   chứng sống — mỗi dự báo được ghi trước khi biết kết quả, rồi chấm điểm sau.
3. **Mọi con số trên giao diện phải truy được về một phép đo trong `docs/`.**
   Không thêm chỉ số nào chỉ vì "nhìn có vẻ hợp lý" — nếu chưa đo, đừng hiển thị.
