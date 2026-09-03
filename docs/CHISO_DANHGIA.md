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

Tất cả đều đạt. **Phải nói thẳng giới hạn của kết luận này:** với 720 phiên
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
