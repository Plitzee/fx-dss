# Điều tra tail-risk USDJPY/USDCHF — kết quả thăm dò, chưa đóng băng

*Lập 14/09/2026. Nhánh `replan-2026`. Tiếp theo bế tắc đã ghi trong
`TONG_QUAN_CHO_AI_KHAC.md` mục 5.1: 5 lần vá tail-risk trước đều thất bại
vì lỗi chỉ lộ ra ở đoạn KIỂM TRA, không bao giờ lộ ở KIỂM ĐỊNH, nên không
có tín hiệu hợp lệ để CHỌN cách vá nào đúng mà không phá kỷ luật "chọn
trên kiểm định, chấm kiểm tra một lần".*

**TRẠNG THÁI: THĂM DÒ, CHƯA CHỐT.** Không script nào trong tài liệu này
chấm điểm trên đoạn kiểm tra hay tập khoá sổ. Đây là thiết kế chuẩn bị sẵn
— quyết định đóng băng thuộc về người chịu trách nhiệm luận văn.

---

## 0. Lối ra khỏi bế tắc — đổi CÁCH kiểm chứng, không chỉ đổi công thức

Vì không có tín hiệu hợp lệ để *chọn* giữa các cách vá bằng cách đo hiệu
năng, hai cách vá dưới đây được thiết kế dựa trên **lý do đã biết trước**
(sự kiện lịch sử có thật, dịch chuyển phân phối đã đo bằng công cụ chẩn
đoán không đụng tới kết quả VaR) — không phải chọn theo hiệu năng đo
được. Việc chấm điểm cuối cùng nên dồn vào đúng một lần mở TẬP KHOÁ SỔ
của luận văn, không mở kiểm tra thêm một lần riêng cho việc này.

## 1. Adversarial Validation — `src/va_duoi_dichchuyen.py`

**Câu hỏi**: đoạn kiểm định và kiểm tra có thật sự khác phân phối, hay chỉ
là cảm giác từ "sd(z) tăng dần"?

**Cách làm**: gộp đặc trưng đuôi (không dùng nhãn/kết quả VaR) của hai
đoạn, huấn luyện một bộ phân loại đoán hàng nào thuộc đoạn nào. AUC cao +
có ý nghĩa (Mann-Whitney U) = hai đoạn khác phân phối thật.

**Kết quả**: AUC 0,95–0,99 cho **cả ba cặp thử** (USDJPY, USDCHF, và
**EURUSD đối chứng**) — dịch chuyển là hiện tượng **chung toàn thị
trường** (khớp chu kỳ tăng lãi suất toàn cầu 2022–2023), **không** riêng
cho USDJPY/USDCHF. Placebo-test đặc trưng "gần ngày BOJ can thiệp": quan
trọng như nhau cho cả USDJPY và EURUSD → chỉ là biến ngày-tháng đội lốt,
không phải tín hiệu riêng.

## 2. Điểm gãy cấu trúc (PELT, kiểu Bai–Perron) — `src/va_duoi_diemgay.py`

**Phát hiện quan trọng nhất**: USDJPY và USDCHF có **cơ chế khác hẳn nhau**.

- **USDCHF**: chỉ 5 điểm gãy trong 12+ năm. Điểm lớn nhất
  (2015-01-14 → 2015-04-08, σ̂(z) cuộn 60 phiên +200,3%/−71,0%) khớp
  **chính xác** với SNB bỏ sàn EUR/CHF — thuật toán tự tìm ra, không mớm.
  Không điểm gãy đáng kể nào gần mốc kiểm tra (2023-11-20).
- **USDJPY**: 35 điểm gãy rải đều suốt lịch sử; điểm gần mốc kiểm tra
  nhất (2023-12-08, +15,2%) không lớn hơn hàng chục điểm gãy khác trong
  quá khứ → khớp với chẩn đoán "trôi dần", không phải một cú gãy đơn lẻ.

→ **USDCHF cần loại trừ một cửa sổ ô nhiễm; USDJPY cần một cơ chế theo
dõi dịch chuyển liên tục.** Đây là lý do hai cặp được vá bằng hai phương
pháp khác nhau ở mục 3–4.

## 3. USDJPY — conformal & EVT có trọng số

### 3a. Conformal có trọng số (Tibshirani 2019) — `va_duoi_conformal_trongso.py`

Trọng số Tibshirani `w(x) = p(x)/(1−p(x))` từ bộ phân loại ở mục 1.
**Lần đầu dùng GradientBoosting**: cỡ mẫu hiệu dụng sập còn 26,9/547 —
không dùng được. **Sửa bằng logistic regression đã chuẩn hoá + cắt ngưỡng
xác suất [0,02; 0,98]** (thực hành chuẩn cho trọng số cực đoan) → cỡ mẫu
hiệu dụng 395,1/547, đáng tin. Kết quả ở mức 99% (đúng mức VaR 1% đang vi
phạm): **không cải thiện rõ ràng** so với phân vị thực nghiệm hiện tại.

### 3b. EVT/POT có trọng số — `va_duoi_evt_trongso.py`

Cùng trọng số, áp cho Generalized Pareto trên phần vượt ngưỡng thay vì
Student-t toàn thân. Kết quả **nhất quán qua cả hai biến thể ngưỡng**
(90%, 95%): đuôi phồng ra +3,9%/+4,0% khi có trọng số, và ở mức 99% cho
kết quả **ngang hoặc nhỉnh hơn** mốc thực nghiệm hiện tại (+0,4% ở ngưỡng
90%). **Đây là ứng viên khả dĩ nhất cho USDJPY** trong 3 lần thử.

## 4. USDCHF — loại trừ cửa sổ SNB — `va_duoi_snb_loaitru.py`

Loại 61 phiên (2,44% đoạn huấn luyện) trong cửa sổ SNB đã xác định ở mục
2. Kết quả: bậc tự do Student-t tăng 7,26→9,86 (đuôi mỏng lại — đúng
hướng), và ở mức 99%: phân vị **giảm 5,2%** — đúng hướng cần để sửa
"ES thừa 15%" (mô hình hiện quá bảo thủ vì cú sốc SNB làm phồng đuôi đã
khớp). **Kết quả sạch nhất trong toàn bộ điều tra.**

## 5. Tổng kết — đề xuất đóng băng

| Cặp | Cơ chế | Hướng vá đề xuất | Bằng chứng |
|---|---|---|---|
| USDJPY | Trôi dần liên tục | EVT có trọng số (ngưỡng 90%) | Nhất quán 2 ngưỡng, ≥ mốc hiện tại ở 99% |
| USDCHF | Ô nhiễm dữ liệu huấn luyện 2015 | Loại trừ cửa sổ SNB | ν tăng đúng hướng, phân vị 99% giảm 5,2% |

**Việc còn lại trước khi tích hợp**: viết biên bản chốt chính thức (đúng
thứ tự tiêu chí-trước-số-liệu), rồi chấm điểm CÙNG LÚC với lần mở tập
khoá sổ của luận văn — không mở kiểm tra thêm một lần riêng cho việc này.

## 6. Tái lập

```bash
python src/va_duoi_dichchuyen.py
python src/va_duoi_diemgay.py
python src/va_duoi_conformal_trongso.py
python src/va_duoi_evt_trongso.py
python src/va_duoi_snb_loaitru.py
```

Kết quả: `output/va_duoi_{dichchuyen,diemgay,conformal_trongso,evt_trongso,snb_loaitru}.json`
