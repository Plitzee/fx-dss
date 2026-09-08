# Bộ chỉ số đánh giá — và vì sao độ phủ thôi thì chưa đủ

Lập 29/08/2026. Mọi số trong file này đo trên **đoạn kiểm tra** (2023-03-24 →
2025-12-31, 4.316 quan sát), tham số ước lượng trên đoạn huấn luyện.
Chạy lại: `python src/run_scores.py`.

## 1. Vì sao cần nhiều hơn độ phủ

Trước file này, tầng 6 được chấm bằng **độ phủ** và **bề rộng khoảng**. Cả hai
chỉ nói về một mức tin cậy. Một hệ thống hỗ trợ quyết định đưa ra cả một *phân
phối* dự báo, nên phải chấm bằng thước đo đánh giá được cả phân phối.

Bộ chỉ số dùng ở đây, và mỗi cái trả lời câu gì:

| Chỉ số | Trả lời câu hỏi |
|---|---|
| **CRPS** | Toàn bộ phân phối dự báo tốt đến đâu? (quy tắc chấm điểm chính đáng) |
| **Pinball** | Sai ở **đâu** trên phân phối — đuôi hay giữa? |
| **Log score** | Mật độ dự báo đặt bao nhiêu xác suất vào chỗ thực sự xảy ra? |
| **PIT + KS** | Phân phối có **đúng dạng** không? |
| **Kupiec** | Tần suất vi phạm VaR có đúng bằng α không? |
| **Christoffersen** | Các lần vi phạm có **dính cụm** không? |
| **DQ** (Engle–Manganelli) | Cả hai cùng lúc, mạnh hơn |
| **FZ0** | Cặp (VaR, ES) có nhất quán không? (Patton–Ziegel–Chen 2019) |
| **Winkler** | Khoảng vừa hẹp vừa phủ đúng chưa? |
| **QLIKE / MSE** | Dự báo biến động điểm — hai thước đo duy nhất bền khi RV chỉ là thước đo nhiễu (Patton 2011) |
| **Mincer–Zarnowitz** | Dự báo có thiên lệch hệ thống không? |

## 2. Quy tắc chấm điểm chính đáng (đơn vị pip)

| Phân phối dự báo | CRPS | Log score | Điểm khoảng 90% | FZ0 (VaR/ES 2,5%) |
|---|---|---|---|---|
| Gauss | 26,28 | −3,9548 | 208,19 | −4,3908 |
| Student-t | **26,22** | −3,9748 | **207,32** | −4,3976 |
| Kinh nghiệm chung | 26,23 | −3,9630 | 207,56 | −4,3975 |
| **Mondrian 2 (đang dùng)** | 26,23 | −3,9584 | 207,60 | **−4,3988** |
| Mondrian 3 | 26,23 | −3,9554 | 207,67 | −4,4005 |
| Student-t + σ̂ **cũ** | 26,33 | −3,9651 | 210,44 | −4,3607 |

Năm cách dựng phân phối trên **cùng một σ̂ mới** gần như không phân biệt được
theo CRPS (26,22–26,28). Cái tách ra là **dòng cuối**: dùng σ̂ cũ thì CRPS xấu
hơn 0,4% và điểm khoảng xấu hơn 1,5%. Nói cách khác, ở tầng 6 thứ quyết định là
**chất lượng σ̂**, không phải cách dựng đuôi phân phối.

Đây là một kết luận đáng đưa vào luận văn vì nó đi ngược trực giác: người ta
hay tranh luận Gauss hay Student-t hay conformal, nhưng đo bằng quy tắc chấm
điểm chính đáng thì chênh lệch giữa chúng nhỏ hơn chênh lệch do đổi mô hình
biến động.

## 3. Mất mát phân vị — sai ở đâu trên phân phối (pip)

| Phân phối dự báo | τ=0,01 | τ=0,05 | τ=0,25 | τ=0,5 | τ=0,75 | τ=0,95 | τ=0,99 |
|---|---|---|---|---|---|---|---|
| Gauss | 1,614 | 5,329 | 15,001 | 18,227 | 14,566 | 5,081 | 1,421 |
| Student-t | 1,602 | 5,311 | 14,947 | 18,227 | 14,526 | 5,055 | 1,425 |
| **Mondrian 2** | **1,596** | **5,306** | 14,948 | 18,234 | 14,529 | 5,074 | 1,439 |
| Mondrian 3 | 1,585 | 5,312 | 14,952 | 18,238 | 14,526 | 5,071 | 1,428 |
| Student-t + σ̂ cũ | 1,652 | 5,444 | 15,028 | 18,227 | 14,600 | 5,077 | 1,478 |

Bản đang dùng tốt nhất ở **đuôi trái** (τ=0,01 và 0,05) — đúng phía mà một hệ
thống quản trị rủi ro cần đúng. Gauss tệ nhất ở đúng chỗ đó.

## 4. PIT — phân phối có đúng dạng không

Nếu phân phối dự báo đúng thì PIT (phép biến đổi tích phân xác suất) phải phân
bố **đều** trên [0,1]. Kiểm định Kolmogorov–Smirnov:

| Phân phối dự báo | KS | p-value | PIT<0,05 | PIT>0,95 | Kết luận |
|---|---|---|---|---|---|
| **Gauss** | 0,0334 | **0,0001** | 4,4% | 4,6% | **bác bỏ mạnh** |
| Student-t | 0,0194 | 0,0771 | 5,0% | 5,0% | đạt |
| Kinh nghiệm chung | 0,0122 | 0,5422 | 4,7% | 5,1% | đạt |
| **Mondrian 2 (đang dùng)** | **0,0122** | **0,5323** | 4,7% | 4,9% | **đạt** |
| Mondrian 3 | 0,0171 | 0,1579 | 4,6% | 5,2% | đạt |
| Student-t + σ̂ cũ | 0,0201 | 0,0594 | 5,0% | 4,6% | đạt |

**Đây là bảng quan trọng nhất trong file.** Giả định chuẩn bị **bác bỏ ở
p=0,0001** — trong khi ở mục 5 dưới đây nó **vượt qua mọi backtest VaR**. Tức
là: một mô hình có thể đúng tần suất vi phạm ở vài mức mà vẫn sai dạng phân
phối. Nếu luận văn chỉ báo cáo độ phủ thì sẽ kết luận Gauss ổn, và kết luận đó
sai.

## 5. Backtest VaR

**Mức 1%** (kỳ vọng 1,0% vi phạm):

| Phân phối dự báo | Vi phạm | Kupiec p | Christoffersen p | DQ p | Kết luận |
|---|---|---|---|---|---|
| Gauss | 1,30% | 0,532 | 0,625 | 0,695 | đạt |
| Student-t | 1,04% | 0,468 | 0,693 | 0,716 | đạt |
| Mondrian 2 (đang dùng) | 1,11% | 0,545 | 0,674 | 0,748 | đạt |
| Student-t + σ̂ cũ | 1,04% | 0,523 | 0,692 | 0,550 | đạt |

**Mức 5%** (kỳ vọng 5,0%):

| Phân phối dự báo | Vi phạm | Kupiec p | Christoffersen p | DQ p | Kết luận |
|---|---|---|---|---|---|
| Gauss | 4,40% | 0,465 | 0,588 | 0,755 | đạt |
| Student-t | 5,00% | 0,819 | 0,504 | 0,690 | đạt |
| Mondrian 2 (đang dùng) | 4,73% | 0,595 | 0,501 | 0,677 | đạt |
| Student-t + σ̂ cũ | 5,00% | 0,737 | 0,400 | 0,443 | đạt |

Tất cả đều đạt — **nhưng đây là số gộp sáu cặp, và mục 5b cho thấy gộp đã
giấu mất hai cặp hỏng.** Phải nói thẳng giới hạn của kết luận này: với 720 phiên
mỗi cặp, các kiểm định này có lực rất thấp — ở mức 1% chỉ kỳ vọng 7 lần vi
phạm mỗi cặp. "Không bác bỏ được" không có nghĩa là "đã chứng minh đúng". Đó
chính là lý do PIT ở mục 4 có giá trị: nó dùng **toàn bộ** quan sát chứ không
chỉ những lần vượt ngưỡng.

## 6. Dự báo biến động điểm

| Dự báo σ̂ | QLIKE | MSE (×1e10) | a | b | R² log | corr log |
|---|---|---|---|---|---|---|
| **mới (tổ hợp HAR)** | **0,1593** | **6,139** | −1,129 | **0,906** | **0,354** | **0,593** |
| cũ (MA20-GK) | 0,1977 | 8,009 | −3,319 | 0,698 | 0,231 | 0,476 |

Mincer–Zarnowitz hồi quy log(RV thực) = a + b·log(dự báo); không thiên lệch thì
a=0, b=1. Dự báo mới có **b = 0,906** so với 0,698 của dự báo cũ — gần vô thiên
lệch hơn hẳn, và R² log tăng từ 0,231 lên 0,354.

Chỉ QLIKE và MSE được dùng để **xếp hạng** mô hình, theo Patton (2011): khi
biến thực (biến động) chỉ quan sát được qua một thước đo nhiễu (realized
variance), hai thước đo này là hai thước đo duy nhất bảo toàn thứ hạng đúng.
Các thước đo khác (MAE, MAPE, R²) có thể xếp hạng sai.

## 7. Diebold–Mariano trên CRPS

So với bản đang dùng (Mondrian 2), trên đoạn kiểm tra: không phương án nào
thắng ở cặp nào (0/6 cho mọi ứng viên). Gauss, Mondrian 3 và σ̂ cũ mỗi cái thua
ở 1/6 cặp tại p<0,05. Kết luận: **các cách dựng đuôi không phân biệt được nhau
theo CRPS**, phù hợp với mục 2.

## 8. Nguồn

- Gneiting, Raftery (2007), *Strictly proper scoring rules, prediction, and estimation*, JASA.
- Patton (2011), *Volatility forecast comparison using imperfect volatility proxies*, Journal of Econometrics.
- Patton, Ziegel, Chen (2019), *Dynamic semiparametric models for expected shortfall (and Value-at-Risk)*, Journal of Econometrics.
- Christoffersen (1998), *Evaluating interval forecasts*, International Economic Review.
- Kupiec (1995), *Techniques for verifying the accuracy of risk measurement models*, Journal of Derivatives.
- Engle, Manganelli (2004), *CAViaR*, Journal of Business & Economic Statistics.
- Diebold, Gunther, Tay (1998), *Evaluating density forecasts*, International Economic Review. (PIT)

---

## 5b. Backtest VaR **riêng từng cặp** (bổ sung 04/09/2026)

Mục 5 ở trên báo "đạt hết" — nhưng đó là số **gộp sáu cặp**. Tách ra thì hai cặp
trượt. Gộp lại đã giấu mất điều đó, đúng kiểu bẫy gộp repo này từng mắc một lần
với AUC.

Đo trên đoạn **kiểm tra**, mức 1%, ngưỡng di động theo σ̂ từng phiên, phân vị
thực nghiệm của z (không giả định chuẩn). Tái lập: `GET /risk?pair=…`, mã ở
`api/main.py::var_es`.

| cặp | VaR 99% (pip) | ES 99% (pip) | vi phạm | Kupiec p | DQ p | tỷ lệ ES | kết luận |
|---|---|---|---|---|---|---|---|
| EURUSD | 73,9 | 88,6 | 1,52% | 0,189 | 0,663 | 0,926 | đạt |
| GBPUSD | 116,9 | 151,6 | 0,42% | 0,074 | 0,787 | 1,064 | đạt |
| **USDJPY** | 162,6 | 199,1 | **2,07%** | **0,011** | **0,000** | **0,799** | **KHÔNG ĐẠT** |
| AUDUSD | 65,9 | 77,6 | 1,39% | 0,326 | 0,889 | 0,855 | đạt |
| USDCAD | 107,5 | 128,7 | 1,25% | 0,519 | 0,072 | 0,964 | đạt |
| **USDCHF** | 85,6 | 135,7 | **1,94%** | **0,024** | **0,012** | 1,150 | **KHÔNG ĐẠT** |

**USDJPY hỏng theo cả ba hướng, và ba phép đo độc lập cùng chỉ một chỗ:**

- backtest VaR 1%: vi phạm 2,07% thay vì 1% — ngày rất xấu đến **gấp đôi** mức mô hình nói
- tỷ lệ ES 0,799 — khi ngày đó xảy ra, lỗ thực tế **sâu hơn 25%** so với ES dự báo
- PIT-KS p = 0,0036 và độ phủ 90% chỉ đạt 85,2% (mục 4)

Kết luận: **σ̂ đánh giá thấp đuôi dưới của USDJPY.** Chưa vá. Giao diện in cảnh
báo này ở tab Rủi ro cho đúng cặp đó, kèm khuyến nghị giảm cỡ lệnh và nới dừng lỗ.

USDCHF trượt vì lý do khác: tỷ lệ ES 1,150 (> 1) nghĩa là ES **thừa** chứ không
thiếu — đuôi của nó bị chi phối bởi một biến cố duy nhất, ngày SNB bỏ neo
15/01/2015. Độ lệch z của USDCHF trên huấn luyện là −5,61 với KTC [−9,53; +0,04],
tức khoảng tin cậy rộng đúng kiểu một quan sát chi phối cả mẫu.

**Lực kiểm định.** Với ~722 phiên mỗi cặp, ở mức 1% chỉ kỳ vọng ~7 lần vi phạm.
"Đạt" ở bảng trên **không** có nghĩa là đã chứng minh đúng — nó chỉ có nghĩa là
chưa bác bỏ được. Bốn dòng "đạt" yếu hơn hai dòng "không đạt" nhiều.

---

## 5c. Vá đuôi dưới — **đã thử, KHÔNG thành công** (05/09/2026)

Ghi lại đầy đủ vì đây là một kết quả âm, và giấu nó đi thì mọi con số khác trong
tài liệu này mất giá trị.

### Chẩn đoán

`sd(z)` theo đoạn — chỉ **một** cặp có xu hướng:

| cặp | huấn luyện | kiểm định | kiểm tra |
|---|---|---|---|
| EURUSD | 0,958 | 1,000 | 0,991 |
| GBPUSD | 1,005 | 0,977 | 0,949 |
| **USDJPY** | **1,014** | **1,100** | **1,136** |
| AUDUSD | 0,938 | 0,980 | 0,942 |
| USDCAD | 0,933 | 0,949 | 0,889 |
| USDCHF | 1,052 | 0,947 | 0,995 |

USDJPY tăng **đơn điệu +12%**. Và đuôi **trái** sâu thêm trong khi đuôi phải
đứng yên: q01 −2,476 → **−3,742**, q99 2,572 → 2,559. Nên đây là **thang đo
trôi theo thời gian** cho riêng JPY, không phải hình dạng phân phối sai — hệ số
cố định ước trên huấn luyện sẽ vô ích vì `sd(z)` huấn luyện đã là 1,014.

### Ba phương án, chọn trên KIỂM ĐỊNH

| phương án | ô đạt / 12 | \|tỷ lệ ES − 1\| TB |
|---|---|---|
| V0 phân vị trên huấn luyện (đang chạy) | 8/12 | 0,0695 |
| **V1 cửa sổ mở rộng, khớp lại mỗi 21 phiên** | **10/12** | 0,0636 |
| V2 cửa sổ cuộn 500 phiên | 9/12 | **0,0362** |

Quy tắc đã cài sẵn trong `src/va_duoi.py` (số ô đạt nhiều nhất, hoà thì xét sai
lệch ES) chọn **V1**. Chốt, rồi mới mở đoạn kiểm tra.

**Điểm phải nêu:** trên kiểm định, USDJPY **đạt cả ba phương án**. Vấn đề JPY
chỉ lộ ra ở đoạn kiểm tra (2023–2025, giai đoạn BoJ bình thường hoá chính sách).
Nên về nguyên tắc **không thể** chọn cách vá cho một lỗi mà đoạn chọn không nhìn
thấy. Vẫn làm đúng giao thức, và ghi lại giới hạn này.

### Đoạn KIỂM TRA — chấm một lần

Đây là **lần mở thứ hai** của tầng VaR trên đoạn kiểm tra; lần đầu là chẩn đoán
ở mục 5b.

| | ô đạt / 12 | \|tỷ lệ ES − 1\| TB |
|---|---|---|
| V0 cũ | **9/12** | 0,0942 |
| V1 đã chọn | **8/12** | 0,0852 |

USDJPY mức 99%, chi tiết:

| | vi phạm | Kupiec | DQ | tỷ lệ ES |
|---|---|---|---|---|
| V0 | 2,07% | 0,011 | **0,000** | 0,799 |
| V1 | 1,94% | 0,025 | **0,000** | 0,888 |

**Kết luận: không sửa được.** Tỷ lệ vi phạm gần như không đổi (2,07% → 1,94%),
Kupiec vẫn dưới 0,05, và **DQ vẫn bằng 0,000**. Tỷ lệ ES có cải thiện thật
(0,799 → 0,888) nhưng chưa đạt ngưỡng 0,9. Ngoài ra V1 còn **làm hỏng** USDCAD
mức 99% (DQ 0,072 → 0,043).

### Quyết định: GIỮ NGUYÊN V0

Không đẩy V1 lên sản xuất. Nó không chứng minh được cải thiện trên đoạn kiểm tra
(8/12 so với 9/12), nên thay đổi mô hình lúc này là thêm rủi ro mà không đổi lại
được gì. Cảnh báo USDJPY trên giao diện **giữ nguyên**.

### Cái học được, dùng cho lần sau

`DQ p = 0,000` không đổi qua cả hai phương án là thông tin quan trọng: phép kiểm
Engle–Manganelli bác bỏ vì các lần vi phạm **dự báo được từ vi phạm trước đó và
từ chính mức VaR**. Tức đuôi JPY có **cấu trúc động** mà mô hình bỏ sót — không
phải chuyện ước lượng phân vị vô điều kiện cho chính xác hơn. Cả V0 lẫn V1 đều
là ước lượng vô điều kiện, nên cả hai đều không chạm tới nguyên nhân.

Hướng cho lần thử sau, **phải chốt trước khi mở kiểm tra lần ba**: phân vị **có
điều kiện theo chế độ** (như `SigmaCheDo` nhưng cho đuôi), hoặc mô hình đuôi
động dạng CAViaR. V2 (cuộn 500) có sai lệch ES tốt nhất trên kiểm định (0,0362)
nhưng đã thua ở quy tắc chọn — **không được** lấy nó ra dùng bây giờ chỉ vì V1
hỏng, vì đó đúng là data snooping mà cả giao thức này sinh ra để chặn.

---

## 5d. Phân vị theo chế độ — hướng cuối của A1, **cũng không ăn tiền** (08/09/2026)

`src/va_duoi_chedo.py`, `output/va_duoi_chedo.json`. CAViaR (mục 9.2) đã thử
hướng "đuôi động"; đây là hướng còn lại — "đuôi có điều kiện theo chế độ" —
đóng nốt danh sách hướng đề nghị ở mục 5c.

**Chế độ**: tam phân vị của σ̂ trên huấn luyện, đúng công thức `SigmaCheDo`
(`balop.py`), 3 nhóm êm/vừa/căng, ngưỡng chốt trên huấn luyện áp cho cả chuỗi.

**Hai biến thể**, so trực tiếp với mốc V0/V1 vô điều kiện, chọn trên kiểm định:

| phương án | α=0,05 | α=0,01 |
|---|---|---|
| V0 mốc (vô điều kiện, huấn luyện) | 4/6 | 4/6 |
| **V1 mốc (vô điều kiện, mở rộng — đang chạy)** | **5/6** | **5/6** |
| V0 chế độ (cố định theo chế độ) | 4/6 | 3/6 |
| V1 chế độ (mở rộng theo chế độ) | 4/6 | 4/6 |

Cả hai biến thể theo chế độ đều **kém hơn** mốc đang chạy (V1), không phải chỉ
hoà. Phân tổ theo chế độ làm hỏng thêm những cặp mà V1 vô điều kiện đang đạt
(EURUSD, GBPUSD, AUDUSD ở α=0,01) — mẫu bị chia nhỏ theo ba nhóm làm phân vị
kém ổn định hơn, mà đổi lại không sửa được gì ở đúng chỗ cần sửa.

Về USDJPY: **cả bốn phương án đều đạt** trên kiểm định (DQ 0,978–0,996) — nhất
quán với ghi chú ở mục 5c rằng vấn đề JPY chỉ **lộ ra ở đoạn kiểm tra**, không
phải kiểm định. Vì đoạn kiểm tra chưa được mở (đúng luật), không có cách nào
xác nhận trên dữ liệu này liệu phân vị theo chế độ có sửa được lỗi thật hay
không — nhưng vì bản thân nó đã thua ngay trên kiểm định, không có cơ sở để
đẩy nó đi xa hơn.

**Quyết định: giữ nguyên V1 mở rộng vô điều kiện — hướng A1 coi như đã đóng.**
Ba hướng độc lập cho đuôi (phân vị vô điều kiện cuộn/mở rộng, CAViaR động,
phân vị theo chế độ) đều không vượt được mốc trên kiểm định. Cảnh báo USDJPY
trên giao diện tiếp tục giữ nguyên; A1 trong `docs/KEHOACH_CAITIEN.md` chuyển
từ "còn một hướng chưa thử" sang "đã thử hết ba hướng, không hướng nào ăn
tiền" — nếu muốn tiếp tục thì cần dữ liệu mới (biến ngoại sinh, xem mục "Còn
để ngỏ" của `KEHOACH_CAITIEN.md`) chứ không phải biến thể thống kê khác của
cùng một phân vị.

---

## 7. Phản ứng đo được của 14 loại sự kiện (05/09/2026)

Sau khi nối FRED (`collect/lich_su_kien.py`), lịch mở rộng từ **7 NHTW / 901
ngày** lên **14 loại / 2.404 ngày**, và `src/sukien_profile.py` đo lại trên
**13.944 quan sát** (cặp × ngày sự kiện).

Tỷ lệ = trung vị `|r|` ngày sự kiện chia trung vị `|r|` của 20 phiên liền trước.

| loại | n | \|r\| so nền | KTC 95% | σ̂ dự báo | có tác động? |
|---|---|---|---|---|---|
| **FOMC** | 792 | **1,454** | [1,347; 1,516] | 1,232 | ✓ cao |
| **BOE** | 588 | **1,321** | [1,169; 1,480] | 1,075 | ✓ cao |
| **ECB** | 918 | **1,274** | [1,188; 1,409] | 1,022 | ✓ cao |
| **NFP** | 1.206 | **1,269** | [1,177; 1,364] | 1,103 | ✓ cao |
| **SNB** | 432 | **1,264** | [1,116; 1,445] | 1,123 | ✓ cao |
| BOC | 798 | 1,160 | [1,041; 1,252] | 1,013 | ✓ |
| RBA | 1.056 | 1,157 | [1,105; 1,251] | 0,984 | ✓ |
| BANLE | 1.254 | 1,119 | [1,030; 1,196] | 0,995 | ✓ |
| GDP | 1.338 | 1,101 | [1,048; 1,190] | 1,003 | ✓ |
| JOLTS | 1.158 | 1,097 | [1,020; 1,158] | 0,986 | ✓ |
| BOJ | 690 | 1,084 | [0,981; 1,176] | 1,037 | **phủ 1,0** |
| **CPI** | 1.266 | 1,082 | **[0,991; 1,159]** | 0,966 | **phủ 1,0** |
| PPI | 1.248 | 0,980 | [0,900; 1,051] | 0,998 | **phủ 1,0** |
| **PCE** | 1.200 | **0,943** | **[0,891; 0,991]** | 1,001 | **ÊM HƠN có ý nghĩa** |

### Ba điều đáng nói

**1. NFP thật sự cao — 1,269×, hạng tư trong 14 loại.** Đây là bổ sung có giá
trị nhất: 1.206 quan sát, KTC không phủ 1,0, và σ̂ chỉ dự báo 1,103× nên **hụt
13%**. Trước khi nối FRED, hệ thống hoàn toàn mù về ngày này.

**2. CPI KHÔNG khác ngày thường — trái với mọi lịch kinh tế.** 1,082× với KTC
**[0,991; 1,159] phủ 1,0**, trên 1.266 quan sát. Forex Factory, TradingView,
Investing.com đều tô **đỏ** cho CPI. Đo trên dữ liệu này thì nó **không** làm
giá động mạnh hơn ngày thường một cách phân biệt được.

Giao diện vì thế **không đánh dấu CPI**. Đây chính là điểm khác biệt của hệ
thống: mức độ đến từ **phép đo kèm khoảng tin cậy**, không từ quy ước.

**3. PCE ÊM HƠN ngày thường, có ý nghĩa.** 0,943× với KTC [0,891; 0,991] — cận
trên dưới 1,0. Giả thuyết: PCE công bố cuối tháng, và nền 20 phiên liền trước
của nó rơi vào giai đoạn thường có nhiều sự kiện khác. **Chưa kiểm chứng** —
ghi lại làm quan sát, không làm kết luận.

### Hướng đi: vẫn 0/14

**Không một loại nào trong 14** có thiên lệch hướng có ý nghĩa — mọi khoảng tin
cậy của dấu trung bình đều phủ 0. Trước là 0/7, nay là **0/14** với gấp đôi số
loại và gấp 2,6 lần số quan sát. Kết luận "sự kiện khuếch đại *biên độ*, không
chỉ ra *chiều*" mạnh lên đáng kể.

### Hai ngưỡng dùng trên giao diện

- **"có tác động"** — KTC 95% không phủ 1,0 → được đánh dấu chấm tròn
- **"tác động cao"** — điểm ước lượng ≥ 1,25 → biểu tượng ⚡ đỏ

Cả hai dựa trên khoảng tin cậy chứ không chỉ điểm ước lượng, nên BOJ (1,084×,
phủ 1,0) và CPI **không** được đánh dấu dù thị trường quen coi chúng là quan trọng.

---

## 8. Mở rộng thêm — 18 loại sự kiện, và Nhà ở lộ ra phản ứng ngược (06/09/2026)

Thêm 4 loại vào `collect/lich_su_kien.py`: **JOBLESS** (đơn xin trợ cấp thất
nghiệp hàng tuần), **DTC** (đơn hàng hoá lâu bền — release "Manufacturer's
Shipments, Inventories, and Orders"), **NHA** (bán nhà mới), **UMCSI** (khảo sát
tâm lý tiêu dùng Michigan). Bỏ **ISM** — nó là dữ liệu của một tổ chức tư nhân,
không nằm trong `releases/dates` của FRED theo đúng nghĩa "ngày công bố".

Lịch nay: **18 loại, 4.138 dòng** (từ 14 loại / 2.404 dòng). `sukien_profile.py`
đo lại trên số quan sát lớn hơn nhiều — **JOBLESS một mình đã có 5.190 quan sát**
vì ra hàng tuần thay vì hàng tháng.

| loại | n | \|r\| so nền | KTC 95% | σ̂ dự báo | |
|---|---|---|---|---|---|
| JOBLESS | 5.190 | 1,043 | [1,009; 1,084] | 0,998 | ✓ có tác động, nhẹ |
| DTC | 2.460 | 1,033 | [0,985; 1,104] | 0,992 | phủ 1,0 |
| **NHA** | 1.170 | **0,927** | **[0,849; 0,989]** | 0,964 | **ÊM HƠN có ý nghĩa** |
| UMCSI | 1.176 | 0,954 | [0,897; 1,012] | 0,997 | phủ 1,0 |

### Phát hiện: bán nhà mới (NHA) làm giá **êm hơn** ngày thường, có ý nghĩa

0,927× với KTC **[0,849; 0,989]** — cận trên dưới 1,0, trên 1.170 quan sát. Đây
là loại thứ hai (sau PCE) cho kết quả **ngược chiều** so với trực giác "sự kiện
nào cũng làm giá động mạnh hơn".

**Giả thuyết, chưa kiểm chứng:** New Residential Sales công bố cùng khung thời
gian với nhiều số liệu nhà ở khác đã biết trước (housing starts, building
permits thường ra sớm hơn vài ngày), nên tới lúc NHA công bố thị trường có thể
đã "tiêu hoá" phần lớn thông tin liên quan. Cũng có thể là ngẫu nhiên do KTC sát
1,0. Cần thêm dữ liệu hoặc kiểm chéo với dữ liệu nhà ở khác trước khi kết luận.

### JOBLESS: mẫu lớn nhất nhưng tác động nhỏ nhất trong nhóm "có tác động"

1,043× là mức thấp nhất trong các loại có KTC không phủ 1,0. Hợp lý — đây là số
liệu ra **hàng tuần**, thị trường ít bất ngờ hơn số liệu hàng tháng như NFP.

### Hướng đi: 0/18, không đổi

Vẫn không loại nào có thiên lệch hướng có ý nghĩa. Số lần kiểm độc lập tăng từ
14 lên 18 loại, kết luận "sự kiện khuếch đại biên độ, không chỉ ra chiều" tiếp
tục đứng vững.

---

## 9. Ba phương pháp hiện đại — đã thử, **cả ba không ăn tiền** (08/09/2026)

Rà lại xem các phương pháp đang dùng có lỗi thời không, ba ứng viên được chọn vì
đều đo được bằng công cụ sẵn có, không cần hạ tầng mới. Mục này ghi cả ba, kể cả
hai cái thất bại — vì cái học được từ chúng đáng giá hơn bản thân phép thử.

| # | Phương pháp | Thay cho | Kết quả |
|---|---|---|---|
| 1 | FDR Benjamini–Yekutieli | Westfall–Young FWER | **Không ăn tiền** — sai chỗ nghẽn |
| 2 | CAViaR (Engle–Manganelli 2004) | Phân vị tĩnh `va_duoi.py` | **Không ăn tiền** — lãi chỉ đến từ nhiễu seed |
| 3 | Fixed-Share (Herbster–Warmuth 1998) | Hedge trơn | **Không ăn tiền** — đo được, xấu đơn điệu |

### 9.1 FDR thay FWER — cải thiện 1,2%, và cải thiện đó là giả

`src/kiem_fdr.py`, `output/kiem_fdr.json`. 1.890 giả thuyết, 21.606 hàng.

Trước hết phải vượt một rào kỹ thuật: ngưỡng BY ở hạng 1 là α/(m·c) = 3,3e-6,
trong khi 1.000 hoán vị chỉ phân giải được tới 1e-3. Nên p thô đếm trực tiếp
**không bao giờ** bác bỏ được gì qua cổng FDR. Cách vá là hiệu chuẩn đuôi chuẩn
bằng hệ số phồng λ đo từ null (genomic control), rồi **kiểm chứng xấp xỉ đó** ở
vùng cả hai cách đều đo được — lệch lớn nhất **1,3%**, chấp nhận được.

Ngưỡng |z| tương đương ở hạng 1: W-Y 4,72 · BH 4,25 · **BY 4,71**. BY gần như
trùng W-Y, đúng như lý thuyết dự báo (với một tín hiệu, BH = Bonferroni, BY còn
chia thêm cho c(m) = 8,12).

MDES đo bằng đối chứng dương, 60 lần lặp mỗi mức:

| số tín hiệu tiêm | MDES W-Y | MDES BY |
|---|---|---|
| 1 | 1,306 | 1,290 |
| 5 | 1,312 | 1,299 |

BY nhạy hơn 1,2%. **Nhưng con số đó không dùng được**, vì ba lý do đo được:

1. **k = 5 không khá hơn k = 1.** Nếu FDR đang ăn tiền nhờ nhiều tín hiệu cùng
   lúc thì k = 5 phải hơn hẳn. Nó không hơn.
2. **BY bác bỏ trung bình 1.056 giả thuyết** trên dữ liệu thật. Ngưỡng BH/BY
   nới ra tỷ lệ thuận với số bị bác bỏ, nên cả lợi thế của BY đến từ 1.056 cái
   đó — mà chúng là gì? 892 giả thuyết có |z| > 4,71, max |z| = 28, và chúng là
   **các biến thể của σ̂**: "tính dai vol cao", "ATR phân vị cao", "|z| hôm nay
   thấp" — tất cả đều là thước đo biến động, tất cả đều chỉ vào lớp *đi ngang*.
   Lớp đi ngang định nghĩa bằng dải pip cố định, nên biến động thấp → đi ngang.
   Đó đúng là phép lặp thừa mà `la_vi_tu_nen` sinh ra để chặn — nhưng nó chỉ bắt
   đúng chữ "σ̂", bỏ lọt 11 đặc trưng biến động khác. **BY đang cưỡi lên phép
   lặp thừa để nới ngưỡng của chính nó.**
3. **Cổng đa kiểm định không phải chỗ nghẽn.** Đo trực tiếp trên dữ liệu thật:

   | cổng | còn lại |
   |---|---|
   | 1.890 giả thuyết | 1.890 |
   | cửa 1 — đa kiểm định, \|z\| > 4,72 | **891** (47%) |
   | cửa 2 — điều kiện hoá theo σ̂ | **31** (3,5% số qua cửa 1) |
   | *nếu bỏ hẳn cửa 1*, cửa 2 một mình | 55 |

   Cửa 1 đang loại thêm đúng **24** giả thuyết mà cửa 2 sẽ cho qua. Tức toàn bộ
   dư địa của việc nới cửa 1 — dù nới bằng FDR, e-value hay bỏ hẳn — là 24 giả
   thuyết. Tinh chỉnh một cái cửa cho qua 47% là tinh chỉnh sai chỗ.

**Quyết định: giữ Westfall–Young.** Ghi vào đây để `KEHOACH_2026Q4.md` §1.2
được đóng lại, không phải vì quên mà vì đã đo.

**Việc nên làm thay vào đó** (chưa làm): `la_vi_tu_nen` hiện chỉ loại 3/630 vị
từ. Đúng ra phải loại mọi vị từ chỉ nói về biến động — thêm 11 đặc trưng nữa.
Đó không phải mẹo thống kê mà là sửa **định nghĩa không gian quy luật**, và nó
ảnh hưởng tới con số "0/1.890" đang báo cáo.

### 9.2 CAViaR — 6/6 so với 5/6, và cái hơn đó không đứng vững

`src/caviar.py`, `output/caviar.json`. Ba dạng động lực (SAV, AS, IG — Engle &
Manganelli 2004) khớp trên **z** (sau khi đã chia σ̂), giao thức cuộn giống hệt
V1 mở rộng của `va_duoi.py`, chọn trên kiểm định.

Chạy đầu (`output/log_caviar.txt`) lộ ra một lỗi trước khi so được gì: đệ quy
`b1` không bị chặn dưới 1, nên vài cửa sổ nổ theo hàm mũ — SAV ở USDJPY ra
pinball 52.205 (đáng lẽ ~0,04), AS ở USDCHF ra tỷ lệ ES −4061. Vá bằng cách
chặn phân kỳ ngay trong hàm mục tiêu (`abs(b[1]) >= 0.999 → inf`), chạy lại
(`log_caviar2.txt`) mới có bảng sạch để so.

Tổng kết trên kiểm định, sáu cặp:

| biến thể | α=0,05 | α=0,01 | pinball TB |
|---|---|---|---|
| V1 mở rộng (mốc, phân vị tĩnh) | 5/6 | 5/6 | 0,07050 |
| CAViaR SAV | 4/6 | 4/6 | 0,07086 |
| CAViaR AS | 4/6 | 4/6 | 0,07050 |
| **CAViaR IG** | 5/6 | **6/6** | 0,07067 |

IG là dạng duy nhất **hơn** mốc — 6/6 so với 5/6 ở α=0,01, nhờ đúng một cặp
đổi từ trượt sang đạt: USDCHF (DQ 0,0031 → 0,0620). Nhìn thoáng qua đây là
bằng chứng CAViaR bắt được động lực đuôi mà phân vị tĩnh bỏ lỡ.

**Nhưng kiểm tra ổn định theo hạt giống ngẫu nhiên lật lại kết luận đó.**
`output/log_ondinh.txt` chạy lại đúng cấu hình IG với 5 seed (11, 101, 202,
303, 404) trên ba cặp — hai cặp "có vấn đề" (USDCHF, USDCAD) và một cặp làm
đối chứng (USDJPY):

| cặp | seed đạt DQ | khoảng DQ quan sát |
|---|---|---|
| USDJPY (đối chứng) | **5/5** | [0,9966; 0,9981] — ổn định |
| USDCHF | 4/5 | [0,0034; 0,0640] — sát ngưỡng 0,05 |
| USDCAD | **2/5** | [0,0163; 0,8822] — đổi kết luận theo seed |

USDCAD đã đạt ở mốc V1 (DQ 0,8715) — CAViaR IG không thêm gì ở đây, chỉ là
phép đo bấp bênh. Còn đúng cặp làm nên lợi thế "6/6" — USDCHF — có DQ dao
động ba bậc độ lớn (0,0034 → 0,0640) chỉ vì đổi hạt giống khởi tạo Nelder-Mead,
tức "đạt" hay "trượt" phụ thuộc may rủi tối ưu hoá chứ không phải tín hiệu.

Đã thử loại giả thuyết "thiếu điểm khởi động": `output/log_ondinh2.txt` chạy
lại với `n_dau=60` (gấp 5 lần số điểm khởi đầu ngẫu nhiên mặc định 12) — kết
quả **không đổi**, vẫn 4/5 và 2/5. Vậy không phải do tối ưu hoá chưa hội tụ đủ
sâu; bất định nằm ở chính bài toán (bề mặt pinball loss nhiều cực tiểu địa
phương gần nhau, giống hệt tình huống Fixed-Share ở 9.3 — "hạng nhất đổi tay
nhưng các phương án gần hoà nhau").

**Quyết định: giữ phân vị tĩnh (V1 mở rộng) của `va_duoi.py`.** CAViaR không
sai về lý thuyết — chạy trên z, có tự kiểm đệ quy, có chặn phân kỳ — nhưng cái
"hơn" duy nhất nó tạo ra so với mốc không sống sót qua đổi hạt giống. Không có
script CAViaR nào được gọi trong pipeline sản xuất; `src/caviar.py` giữ lại
như tài liệu đối chứng, không bật.

### 9.3 Fixed-Share — đo được, và xấu đơn điệu

`src/kiem_fixshare.py`, `output/kiem_fixshare.json`. Chọn trên **kiểm định**
(3.282 hàng); đoạn kiểm tra không mở.

Chẩn đoán trước đã: chuyên gia giỏi nhất **có** đổi — 3 chuyên gia khác nhau
chiếm ngôi đầu qua 13 cửa sổ 250 phiên. Nên tiền đề của Fixed-Share đúng. Nhưng
*cách biệt* giữa hạng nhất và hạng nhì chỉ 0,0002–0,0086 log-loss: ngôi đầu đổi
tay, mà các chuyên gia gần như hoà nhau. Đuổi theo kẻ dẫn đầu thì không có gì
để ăn, còn chi phí kéo trọng số về đều thì trả thật.

ΔBSS so với α = 0, KTC bootstrap khối **ghép cặp**:

| α | mục tiêu P | mục tiêu R |
|---|---|---|
| 0,001 | −0,00008 [−0,00029; +0,00012] | −0,00019 [−0,00037; +0,00001] |
| 0,010 | −0,00110 [−0,00248; +0,00016] | −0,00162 [−0,00270; −0,00037] |
| 0,050 | −0,00335 [−0,00612; −0,00081] | −0,00404 [−0,00661; −0,00146] |
| 0,200 | −0,00487 [−0,00864; −0,00150] | −0,00550 [−0,00875; −0,00236] |

Đơn điệu, cả hai mục tiêu, không có cực đại trong: **α = 0 là tối ưu**. Ở α lớn
KTC nằm hẳn dưới 0 — không phải "không đo được", mà là *đo được rằng nó hại*.

**Quyết định: giữ Hedge trơn.** `ToHopTrucTuyen.alpha` được thêm vào `balop.py`
với mặc định 0,0 (đúng hành vi cũ, tự kiểm đạt) và giữ lại để lần sau có thêm
chuyên gia thực sự khác nhau thì đo lại — không phải để bật.

### Cái học được

Cả ba thất bại có chung một hình dạng: **phương pháp mới đúng, nhưng vá vào chỗ
không đau — hoặc cái "hơn" nó tạo ra không sống sót qua kiểm tra thứ hai.** FDR
nới cửa 1 trong khi cửa 2 mới chặn. CAViaR hơn mốc đúng 1 cặp, và cặp đó đổi
kết luận theo hạt giống ngẫu nhiên — lợi thế là nhiễu đo, không phải tín hiệu.
Fixed-Share đuổi theo đổi chế độ trong khi các chuyên gia hoà nhau nên đổi chế
độ chẳng tốn gì. Trước khi thay một phương pháp, phải đo xem **ràng buộc đang
nằm ở đâu**, và phải **kiểm tra lại bằng seed/lát cắt khác** trước khi tin một
con số hơn — nếu không thì hiện đại hoá chỉ là thay đồ.

---

## 10. A3 — chọn mô hình theo chế độ, **cũng không ăn tiền trên mô hình đang chạy** (08/09/2026)

`src/run_balop_chedo.py`, `output/balop_chedo.json`. Bằng chứng gốc trong
`docs/KEHOACH_CAITIEN.md` mục A3: bảng BSS theo chế độ của **`run_balop.py`**
cho thấy chế độ "vừa" là nơi duy nhất mô hình thua khí hậu học, ở cả h=1 (chỉ
σ̂: −0,0081) và h=20 (σ̂+chế độ: −0,0154). Ý tưởng: ở chế độ "vừa", trả về dự
báo khí hậu học thay vì dự báo mô hình, giữ nguyên mô hình ở hai chế độ còn
lại — rẻ, không cần huấn luyện gì mới.

**Điểm mấu chốt bị bỏ sót lúc viết kế hoạch**: bảng bằng chứng đo trên mô hình
**tĩnh** (chỉ σ̂ đóng băng huấn luyện / σ̂+chế độ đóng băng huấn luyện), nhưng
mô hình **thật sự đang chạy sản xuất** (`api/main.py NEN_THEO_H`) là bản
**cuộn** (khớp lại mỗi 21 phiên) hoặc **tổ hợp trực tuyến** — khác hẳn. Đo lại
đúng trên mô hình sản xuất, mục tiêu P (mục tiêu hiện trên giao diện), KTC
bootstrap khối **ghép cặp** (`diem3.delta_bss_ktc`, viết thêm cho việc này):

| h | mô hình sản xuất | BSS "vừa" (sản xuất) | Δ BSS gộp cả 3 chế độ (chọn theo chế độ − sản xuất) |
|---|---|---|---|
| 1 | tổ hợp trực tuyến | −0,0072 | [−0,0000; +0,0053] — hoà |
| 5 | σ̂ + chế độ (cuộn) | **+0,0139** | **[−0,0093; −0,0008] — xấu hơn có ý nghĩa** |
| 20 | σ̂ + chế độ (cuộn) | −0,0113 | [−0,0016; +0,0092] — hoà |

Ở h=5, mô hình sản xuất **đã ổn** ở chế độ "vừa" (BSS dương +0,0139) — khác
hẳn số liệu tĩnh (−0,0154) mà kế hoạch trích dẫn. Khớp lại định kỳ (cuộn) tự
nó đã thích nghi với chế độ đổi, nên ép về khí hậu học ở đây là **bỏ đi lợi
thế đang có**, và KTC ghép cặp xác nhận điều đó có ý nghĩa (không phủ 0). Ở
h=1 và h=20, gộp lại là hoà — "vừa" cải thiện (về 0) nhưng bù lại mẫu bị chia
nhỏ theo chế độ ở nơi khác làm KTC rộng ra, không đủ để kết luận hơn.

**Quyết định: giữ nguyên mô hình sản xuất, không chọn theo chế độ.** Bài học
lặp lại đúng hình dạng của mục 9: bằng chứng ban đầu đúng, nhưng đo trên biến
thể **tĩnh** trong khi hệ thống thật đã dùng biến thể **cuộn/động** — chẩn
đoán không tự động chuyển sang toa thuốc nếu không đo lại trên đúng cấu hình
đang chạy. `docs/KEHOACH_CAITIEN.md` mục A3 cập nhật theo kết quả này.

---

## 11. B1 (một phần) — Hansen SPA cài xong, áp cho họ mô hình Giai đoạn 1 (08/09/2026)

`src/metrics.py::spa_test()` (mới), `src/run_spa.py`, `output/spa.json`.
`docs/REPLAN_2026.md` mục 10.4 đòi Hansen SPA (2005) để đóng tiêu chí dừng của
Giai đoạn 2; trước đây `metrics.py` chỉ có MCS (Hansen-Lunde-Nason 2011), thiếu
SPA. Đã viết `spa_test()` theo đúng phiên bản **p-value nhất quán** của Hansen
(2005) — tái tâm (recentering) ứng viên rám trước khi dựng phân phối null
bootstrap khối, cùng phong cách code với `mcs()`. Tự kiểm: tỷ lệ bác bỏ dưới
H0 ≈ danh nghĩa (0,057 so với 0,05 kỳ vọng), bác bỏ 97% khi có ứng viên thắng
nền rõ ràng.

**Phạm vi trung thực — chưa đóng được toàn bộ B1.** Kiến trúc "năm họ" H1–H5
mà `REPLAN_2026.md` mục 3.1 mô tả (SAX, motif, rule-list, trần GBM, chế độ)
phần lớn **chưa tồn tại thành code**: thư mục `rules/mining/` không có trong
repo, H2 (motif) và H3 (rule-list) chưa viết, H5 (chế độ) mới làm một nửa.
Viết đủ năm họ là khối lượng công việc khác hẳn "cài thêm một phép kiểm" —
không làm giả trong phiên này.

Thay vào đó, `run_spa.py` áp `spa_test()` ngay cho dữ liệu đã có sẵn và THẬT:
họ mô hình biến động của chính `run_balop.py` (Giai đoạn 1) so nền "chỉ σ̂" —
đúng nền mà REPLAN định nghĩa là mức phải vượt. Kết quả, mục tiêu P (giao
protocol KHOI theo tầm hạn, bootstrap khối theo cặp):

| h | T_SPA | p-value | bác bỏ H0 ở α=0,05 |
|---|---|---|---|
| 1 | 0,805 | 0,467 | không — đúng dự kiến, "tổ hợp trực tuyến" gần như không khác nền ở h=1 |
| 5 | 2,213 | **0,024** | **có** — "σ̂ + chế độ (cuộn)" thắng nền có ý nghĩa |
| 20 | 3,253 | **0,000** | **có** — cùng ứng viên, thắng rõ hơn |

Đọc: nền tảng mô hình biến động (Giai đoạn 1) **đã** vượt qua SPA ở h=5, h=20
— khớp với lý do `api/main.py` chọn "σ̂ + chế độ (cuộn)" cho hai tầm hạn đó.
Ở h=1, không có ứng viên nào thắng nền có ý nghĩa — cũng khớp với thực tế
"tổ hợp trực tuyến" ở h=1 gần như hoà với nền (Δlog thô +0,00008).

**Đây KHÔNG PHẢI là điều kiện "cả năm họ SPA" ở mục 10.4** — đó vẫn treo, chờ
H2/H3/H5 được viết thành code sinh chuỗi tổn thất. Nhưng `spa_test()` giờ là
công cụ sẵn có trong `metrics.py`, sẵn sàng dùng ngay khi các họ đó có dữ liệu.
`docs/GIAIDOAN2_QUYLUAT.md` mục 5 cập nhật dòng "Hansen SPA cho cả họ" thành
"đã cài công cụ, mới chạy được cho Giai đoạn 1 — Giai đoạn 2 (H2/H3/H5) còn
treo".
