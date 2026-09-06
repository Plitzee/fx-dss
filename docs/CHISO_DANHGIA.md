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
