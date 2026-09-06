# Kế hoạch cải tiến — trọng tâm mới, 05/09/2026

Thay cho `docs/KEHOACH_CAITIEN.md` sau khi đổi ưu tiên.

**Tạm dừng:** tầng 4 và 6b (định cỡ vị thế, đòn bẩy theo vốn). Giữ nguyên code
và giao diện, không phát triển tiếp. Mọi công sức dồn vào ba trục dưới đây.

---

## Trục 1 — Xác suất xu hướng (ba ô)

### Vấn đề hiện tại, nói thẳng

Ba ô đang chạy tốt ở **ô giữa** (BSS +0,0107 đến +0,0148, có ý nghĩa thống kê)
nhưng **hai ô ngoài không mang tin hướng**. Đã kiểm 5 cách độc lập, đều âm.

Nhưng `src/kiem_pheu.py` vừa đo được điều quan trọng: **phễu chỉ phát hiện được
quy luật có lift ≥ 1,35** (lực 80%). Ở lift 1,20 lực chỉ 40%, ở 1,15 còn 8%.
Nghĩa là kết luận âm hiện tại **chỉ loại trừ quy luật mạnh**. Nguyên nhân là cỡ
mẫu, không phải bản chất thị trường.

### 1.1 Hạ tầm hạn xuống H1 — đòn bẩy lớn nhất

| | số quan sát |
|---|---|
| panel D1 hiện tại | 21.596 hàng |
| **H1 có sẵn trên đĩa** (`data/prices/*_h1.csv`) | **592.343 thanh** |
| tỷ lệ | **×27,4** |

Thống kê z tăng theo √n với cùng cỡ hiệu ứng, nên ×27 dữ liệu cho z lớn hơn
khoảng ×5,2. Ngưỡng Westfall–Young có tăng theo số vị từ nhưng không tăng ×5.
**Ước tính MDES giảm từ 1,35 xuống khoảng 1,05–1,10.** Đó là bước sang một
loại phát biểu khác hẳn.

**Bốn thứ bắt buộc phải xử lý, nếu không kết quả sẽ là rác:**

1. **Mùa vụ trong ngày.** Lợi suất H1 có chu kỳ phiên Á/Âu/Mỹ rất mạnh. Phải
   đưa giờ-trong-ngày vào bộ kiểm soát, nếu không mọi "quy luật" tìm được chỉ
   là hiệu ứng giờ mở cửa.
2. **Nhiễu vi cấu trúc.** Bid-ask bounce tạo tự tương quan âm giả ở tần suất
   cao. Phải kiểm bằng cách so kết quả ở H1 với H4 — nếu tín hiệu biến mất khi
   gộp lên thì đó là nhiễu, không phải quy luật.
3. **Khối hoán vị phải dài hơn.** Hiện `KHOI = 5` cho D1. Ở H1 phải ≥ 24 thanh
   để giữ nguyên chu kỳ ngày, nếu không KTC sẽ hẹp giả.
4. **CỬA CHI PHÍ — quan trọng nhất.** Spread trung vị ~1 pip, mà biên độ H1
   điển hình chỉ ~10 pip. Một quy luật lift 1,05 ở H1 hoàn toàn có thể **vô
   giá trị kinh tế sau chi phí**. Nên thêm một cửa thứ năm vào phễu: lợi thế
   ròng phải dương sau khi trừ spread theo giờ (`data/spread_hourly_all.csv`)
   và trượt giá (`data/slippage.csv`). Cửa này đặt **trước** khi chạy, không
   phải sau khi thấy kết quả.

**Kết quả có thể xảy ra, cả hai đều là kết quả:**
- tìm được quy luật yếu sống sót → thư viện quy luật có nội dung thật
- vẫn 0 → kết luận âm mạnh hơn hẳn, vì nay loại trừ được tới lift ~1,05

### 1.2 FDR thay FWER

Westfall–Young khống chế sai lầm loại I **toàn cục** (FWER) — rất chặt. Chuyển
sang khống chế **tỷ lệ phát hiện sai** (FDR, Benjamini–Yekutieli dưới phụ
thuộc) sẽ hạ ngưỡng đáng kể, đổi lại chấp nhận vài dương tính giả.

Chạy lại `kiem_pheu.py` để **đo** MDES mới thay vì lập luận. Kết hợp với 1.1
thì hai cái cộng dồn.

### 1.3 Xu hướng tương đối thay vì tuyệt đối — hướng chưa ai thử

Thay câu hỏi "EURUSD sẽ lên hay xuống" bằng **"trong 6 cặp, cặp nào mạnh nhất"**.

- Đây là bài toán **khác hẳn**: xếp hạng chéo, không phải dự báo mức tuyệt đối.
- Nó **triệt tiêu nhân tố đô-la chung** (ρ = 0,443) — thứ chiếm gần một nửa
  biến thiên và làm nhiễu mọi phép đo hướng tuyệt đối.
- Văn liệu FX gọi là *currency momentum / cross-sectional carry*, có bằng chứng
  mạnh hơn nhiều so với dự báo hướng từng cặp.
- **Repo chưa thử lần nào.** Đây là lỗ hổng thật, không phải việc lặp lại.
- Chấm bằng: Spearman rank IC, và hit-rate của cặp xếp đầu so với ngẫu nhiên.

---

## Trục 2 — Phân tích rủi ro

### 2.1 Đuôi dưới USDJPY — đã biết nguyên nhân, lần thử sau phải khác

Lần thử ngày 05/09 **thất bại** (`CHISO_DANHGIA.md` mục 5c), nhưng nó chỉ ra
nguyên nhân: `DQ p = 0,000` **không đổi** qua cả hai phương án. Engle–Manganelli
bác bỏ vì vi phạm **dự báo được** từ vi phạm trước đó — tức đuôi có **cấu trúc
động** mà mô hình bỏ sót.

Cả hai phương án đã thử đều là ước lượng **vô điều kiện** nên không chạm tới
nguyên nhân. Lần sau phải là mô hình **có điều kiện**:
- phân vị theo chế độ biến động (như `SigmaCheDo` nhưng cho đuôi), hoặc
- **CAViaR** (Engle–Manganelli 2004) — VaR tự hồi quy, thiết kế đúng cho hiện
  tượng này

Chốt phương án **trước**, rồi mở đoạn kiểm tra lần ba. Mỗi lần mở là một lần tiêu.

### 2.2 Hai loại rủi ro hệ thống chưa có

- **Rủi ro nhảy giá (gap).** Cuối tuần và ngày lễ, giá mở cửa nhảy khỏi giá
  đóng. Stop-loss **không bảo vệ được** trong khoảng đó — nhà đầu tư giữ lệnh
  qua cuối tuần đang chịu rủi ro hệ thống không hề đo. Dữ liệu có sẵn, chỉ cần
  đo phân phối gap thứ Hai và báo lên giao diện.
- **Rủi ro thanh khoản.** Hiện chỉ có spread theo giờ. Chưa có: spread giãn ra
  bao nhiêu trong ngày sự kiện, và độ sâu thị trường.

### 2.3 Hiệu chuẩn trượt từ sổ dự báo

Sổ đang tích luỹ (0/30 phiên đã chấm, cần ~6 tuần). Khi đủ, thay số hiệu chuẩn
tĩnh của đoạn kiểm định bằng số **đo trên chính dự báo hệ thống đã đưa ra**.
Không cần code mới, chỉ cần chờ và nối vào.

---

## Trục 3 — Huấn luyện mô hình

### 3.1 Biến vĩ mô cho BIẾN ĐỘNG (không phải hướng) — trục duy nhất còn tin

fx-dss hiện **không có một biến vĩ mô nào**. Bộ dữ liệu ở `TSF/data/raw/`
có VIX, DXY, US2Y/10Y, Brent, Gold, SP500 — 2015–2026.

- Causal discovery của CAIFormer đã loại sạch chúng cho **hướng**. Nhưng cho
  **biên độ** thì chưa ai thử, và văn liệu nói VIX với độ dốc đường cong lãi
  suất có tác dụng thật.
- Cách làm: thêm làm biến ngoại sinh vào HAR vòng 7, đo QLIKE trên kiểm định.
- Ràng buộc: dữ liệu bắt đầu 2015 (mất 3 năm lịch sử), kéo tới 2026-08 (chồng
  tập khoá sổ, phải cắt).
- σ̂ tốt hơn → **ô giữa tốt hơn trực tiếp**, vì ô giữa là hàm đơn điệu của σ̂.
  Đây là đường ngắn nhất để cải thiện ba ô.

### 3.2 ML trực tiếp trên đại lượng rủi ro — chưa ai thử

Mọi lần huấn luyện ML trước đây đều nhắm vào **hướng** (thua) hoặc **phương sai**
(thua). Chưa ai huấn luyện ML để dự báo **chính các đại lượng rủi ro**:
- `P(chạm dừng lỗ trong h phiên)` — hiện tính bằng mô phỏng, có thể học trực tiếp
- `P(vi phạm VaR)` — bài toán phân loại nhị phân, chấm bằng Brier/reliability

Đây là bài toán khác hẳn, và nó nằm đúng trong định hướng "hệ phân tích rủi ro".

### 3.3 Lớp hiệu chuẩn lại

MCE hiện: 0,061 (h=1) · 0,047 (h=5) · **0,206 (h=20)**. Isotonic hoặc vector
scaling khớp trên kiểm định. Kỹ thuật chuẩn, không phải nghiên cứu. Nó **gỡ bớt
một lời cảnh báo** trên giao diện thay vì thêm tính năng.

---

## Thứ tự đề nghị

| # | việc | vì sao trước/sau |
|---|---|---|
| 1 | **3.3 hiệu chuẩn lại** | rẻ nhất, gỡ ngay một cảnh báo, không đụng gì khác |
| 2 | **1.1 phễu xuống H1** | đòn bẩy lớn nhất cho đúng thứ đang muốn tập trung; tốn nhất về thời gian chạy |
| 3 | **1.2 FDR + đo lại MDES** | cộng dồn với 1.1, chung một lần chạy |
| 4 | **3.1 vĩ mô cho σ̂** | cải thiện ô giữa trực tiếp; độc lập với trục 1 nên chạy song song được |
| 5 | **1.3 xu hướng tương đối** | hướng mới hoàn toàn, nên làm sau khi 1.1 cho biết dữ liệu H1 sạch tới đâu |
| 6 | **2.1 CAViaR cho USDJPY** | phải chốt phương án trước, mỗi lần mở kiểm tra là một lần tiêu |
| 7 | **2.2 rủi ro gap** | rẻ, dữ liệu có sẵn, bổ sung một loại rủi ro thật đang thiếu |
| 8 | **3.2 ML trên đại lượng rủi ro** | thăm dò, chưa rõ có gì |

Không đụng: tầng 4/6b định cỡ vị thế (tạm dừng theo yêu cầu), tập khoá sổ
(6 cặp chéo — vẫn nguyên vẹn cho lần chạy cuối).

---

## Điều phải nói trước

Trục 1 có thể vẫn ra 0. Nếu H1 với FDR đưa MDES xuống ~1,05 mà vẫn không tìm
thấy quy luật nào, thì kết luận âm trở nên **rất mạnh** — mạnh hơn nhiều so với
bản hiện tại — và đó là một kết quả tốt cho luận văn, không phải thất bại.

Điều **không** nên kỳ vọng: rằng thêm dữ liệu hay thêm mô hình sẽ làm hai ô
ngoài trở nên có ích. Năm phép kiểm độc lập đã nói không, và cửa chi phí ở mục
1.1 nhiều khả năng sẽ giết bất cứ tín hiệu yếu nào tìm được ở H1.

---

## Nối lịch sự kiện vĩ mô (05/09/2026) — đã dựng, chờ khoá

`collect/lich_su_kien.py`. Mở rộng lịch từ **7 NHTW / 901 ngày** sang cả các công
bố vĩ mô tác động cao: **NFP, CPI, GDP, bán lẻ, PCE, PPI, JOLTS, ISM**.

**Vì sao.** `sukien_profile.py` đo được σ̂ hụt **11–20%** đúng ở ngày sự kiện:

| | \|r\| so nền | σ̂ dự báo | hụt |
|---|---|---|---|
| FOMC | 1,454× | 1,232× | 15% |
| BOE | 1,321× | 1,075× | 19% |
| ECB | 1,274× | 1,022× | 20% |

Và hệ thống **không biết gì** về NFP/CPI — riêng NFP đã là ~190 ngày tác động cao
mỗi 16 năm. Nối vào là vá đúng chỗ σ̂ đang hụt, tức cải thiện **trục biến động** —
trục duy nhất đã chứng minh có thông tin (BSS +0,0152 ngoài mẫu ở h=1).

**Nguồn: FRED `releases/dates`** (Fed St. Louis) — chính thức, miễn phí, lịch sử
đầy đủ. Đây là **ngày công bố thật**, không phải suy ra từ quy tắc "thứ Sáu đầu
tháng": quy tắc đó sai vài lần mỗi năm, và ngày sai sẽ **làm loãng** phản ứng đo
được, khiến mô hình tệ đi chứ không tốt lên.

**Cần một khoá miễn phí.** Đăng ký 30 giây ở
`https://fredaccount.stlouisfed.org/apikey`, rồi đặt vào **biến môi trường**
(không đặt trong mã, không commit):

```
Windows :  setx FRED_API_KEY "khoa_cua_ban"
bash    :  export FRED_API_KEY=khoa_cua_ban
Actions :  Settings → Secrets → FRED_API_KEY
```

Chưa có khoá thì script dừng với hướng dẫn rõ ràng, không làm hỏng gì.

**Chuỗi đã nối sẵn.** `src/sukien_profile.py`, `api/main.py::/events` đều ưu tiên
`data/su_kien.csv` (bản mở rộng) rồi mới rơi về `data/cb_dates.csv`. Chạy xong
collector là toàn bộ chuỗi — hồ sơ phản ứng, thanh cảnh báo, dấu ⚡ trên biểu đồ —
tự phủ luôn NFP và CPI.

**Bước bắt buộc sau khi có dữ liệu:** chạy lại `python src/sukien_profile.py`.
**Không được gán nhãn "tác động cao" cho NFP/CPI trước khi đo.** Giao diện xếp
mức độ bằng **tỷ lệ đã đo** kèm cỡ mẫu, không bằng quy ước — đó là điểm khác biệt
so với các nền tảng khác, và nó chỉ đúng nếu con số đến từ phép đo.

**FRED không cho gì:** consensus forecast. Muốn đo "bất ngờ" (`|thực − dự báo|`)
thì phải có nguồn trả phí (Trading Economics). Ngày công bố đã đủ cho mục tiêu
của pha này.
