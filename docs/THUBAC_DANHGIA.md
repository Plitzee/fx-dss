# Thứ bậc đánh giá chính thức — Task 3, đóng Phase 1

**Ngày lập:** 11/09/2026
**Theo:** `01_PHASE_1_HISTORICAL_ONLY.md` mục 5, Task 3 ("Formalize evaluation
hierarchy"); `00_MASTER_ROADMAP.md` mục 6 ("Evaluation hierarchy v2").

Tài liệu này gom lại thứ bậc chỉ số đã dùng RẢI RÁC khắp dự án thành MỘT bảng
tham chiếu chính thức, theo đúng khung roadmap: **chỉ số chính là điểm số xác
suất/phân phối đúng đắn (proper scoring rule)**, chỉ số phụ là các thước đo
điểm-đơn (point metrics) dễ đọc nhưng dễ đánh lừa nếu dùng một mình.

---

## 1. Nguyên tắc chọn

Sản phẩm trả về **phân phối xác suất / phân phối dự báo**, không phải một số
điểm. Một điểm số chỉ đáng tin làm chỉ số CHÍNH nếu nó là *proper scoring
rule* — tối thiểu hoá đúng khi và chỉ khi dự báo bằng phân phối thật (Gneiting
& Raftery 2007). Accuracy, MSE thô, hay "% đúng hướng" không có tính chất đó
— chúng có thể bị làm giả bằng cách hét dự báo cực đoan hoặc quá an toàn.

## 2. Bảng chỉ số theo trục

| Trục | Chỉ số CHÍNH (proper scoring) | Chỉ số PHỤ (báo cáo kèm) | Cài đặt |
|---|---|---|---|
| **Hướng giá** | Log Score, Brier Score, BSS, hiệu chuẩn (ECE/MCE/PIT) | AUC, Balanced Accuracy, Accuracy, Macro F1 | `balop.py`, `docs/GIAIDOAN0_KETQUA.md` |
| **Biên độ (σ̂)** | QLIKE (Bregman bất biến thang đo), CRPS | MSE, MAE | `volfc2.py`, `docs/ML_DL_VONG7.md`, `src/crps.py` |
| **Rủi ro (phân phối lợi suất)** | CRPS (mẫu/log-chuẩn), độ phủ khoảng | VaR/ES điểm, tỷ lệ ES | `src/crps_loi_suat.py`, `va_duoi.py` |
| **Mọi trục — độ tin cậy thống kê** | DM test, Model Confidence Set (Hansen-Lunde-Nason 2011), Hansen SPA | — | `metrics.py` |
| **Mọi trục — độ vững** | ổn định theo năm/cặp, walk-forward, leave-one-pair-out | — | `CHISO_DANHGIA.md` mục 14 |
| **Kinh tế** | kết quả sau chi phí (trượt giá + phí thoát) | Sharpe, sụt giảm tối đa, CVaR | `docs/TOANMACH_E2E.md` |

## 3. Quy tắc đọc — tránh bẫy đã từng mắc

- **Không đọc accuracy/AUC một mình cho hướng giá.** Dự án từng thấy
  accuracy tăng nhẹ trong khi Log Score xấu đi (`01_PHASE_1...md` mục 2:
  "Accuracy chỉ tăng ít, trong khi proper probabilistic scores thể hiện skill
  rõ hơn").
- **Không đọc MAE một mình cho biên độ.** MAE bằng CRPS khi dự báo là một
  điểm (Dirac) — nó KHÔNG phân biệt được mô hình biết-mình-không-chắc với mô
  hình chỉ đoán liều (`docs/ML_DL_VONG7.md` mục CRPS). Ví dụ đo được: Chronos
  MAE tệ nhất bảng nhưng CRPS tốt nhất — vì phân phối phân vị thật của nó
  hiệu chuẩn tốt hơn giá trị điểm không thể hiện.
- **Không đọc kết quả gộp 6 cặp một mình.** `docs/CHISO_DANHGIA.md` mục 5b:
  gộp "đạt hết" trong khi tách riêng USDJPY/USDCHF trượt cả ba phép kiểm.
- **Không đọc một con số kiểm định một mình.** Mọi kết luận mô hình phải
  chấm lại đúng MỘT LẦN trên kiểm tra, không quay lại chọn tiếp
  (`docs/KHOA_SO.md`).

## 4. Đối chiếu Task 3 với thực tế đã đo (11/09/2026)

| trục | đã có phép đo chính (proper score)? |
|---|---|
| Hướng giá | ✅ Log Score + BSS, `01_PHASE_1...md` mục 2 |
| Biên độ | ✅ QLIKE (`ML_DL_VONG7.md`) + CRPS log-chuẩn (`crps.py`, hôm nay) |
| Rủi ro | ✅ CRPS phân phối lợi suất (`crps_loi_suat.py`, hôm nay) — +1,70% kỹ năng so khí hậu học, 6/6 cặp |
| Kinh tế | ✅ sau chi phí, `TOANMACH_E2E.md` |

**Task 3 hoàn tất**: cả bốn hạng mục đã có phép đo proper-scoring, không hạng
mục nào chỉ dựa vào point metric.
