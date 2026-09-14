# Meta-Labeling — kết quả thăm dò, chưa đóng băng

*Lập 14/09/2026. Nhánh `replan-2026`. Dùng khung meta-labeling
(López de Prado 2018): mô hình THỨ CẤP không dự báo lại — chỉ quyết định
có nên tin mô hình SƠ CẤP (HAR/conformal đã có) ở phiên này hay không.*

**TRẠNG THÁI: THĂM DÒ, CHƯA ĐÓNG BĂNG, CHƯA TÍCH HỢP SẢN XUẤT.** Ba lý do,
nói thẳng:

1. **Không đúng thứ tự tiêu chí-trước-số-liệu.** Bản không-VIX (3/6 cặp
   đạt) chạy trước; VIX được thêm SAU khi thấy kết quả đó ở mức biên —
   có lý do chính đáng (VIX đã được tam giác hoá nhân quả từ trước, không
   chọn bừa) nhưng vẫn khác quy ước `RUIRO_ML.md` (tiêu chí commit trước
   số liệu).
2. **Chỉ đo trên kiểm định**, chưa chạm kiểm tra/tập khoá sổ.
3. **Mới kiểm được "dự đoán được ngày tệ"**, chưa kiểm "hành động dựa
   trên dự đoán đó có thực sự cải thiện hệ thống không" (ví dụ: tự động
   nới khoảng tin cậy khi độ tin thấp — chưa thiết kế/kiểm).

---

## 0. Vì sao nhắm vào biên độ, không nhắm vào hướng

Trục hướng đã xác nhận **198/198** giả thuyết không sống sót qua kiểm
soát bội — thử meta-label "dự báo hướng có đúng không" gần như chắc chắn
thất bại vì chính cái nó dựa trên (hướng) đã chứng minh không đoán được.
Trục biên độ CÓ kỹ năng thật (QLIKE thắng MCS) nên "phiên nào dự báo biên
độ kém tin cậy hơn" là câu hỏi có cơ sở để hỏi.

## 1. Meta-label QLIKE — `src/metalabel_qlike.py`

**Mục tiêu**: dự đoán phiên nào QLIKE (σ̂) sẽ rơi vào nhóm 20% tệ nhất
(ngưỡng chốt trên huấn luyện, áp sang kiểm định).

### 1a. Hai lần rò rỉ bắt được và sửa — quan trọng hơn cả con số

- **Lần 1**: đưa `sig` (σ̂ hôm nay) làm đặc trưng → BSS giả **+0,37 đến
  +0,55**, AUC tới 0,90 ở **6/6 cặp**. Nghi ngờ vì "quá đẹp", kiểm tra:
  `corr(log(σ̂²), QLIKE) = 0,64` — vì `QLIKE = log(σ̂²) + rv5/σ̂²` chứa
  chính σ̂ trong công thức. Bỏ `sig` đi: BSS **-0,11** (âm). Xác nhận
  đây là rò rỉ vòng tròn, không phải tín hiệu thật.
- **Lần 2**: các đặc trưng cuộn (`rolling`) mặc định tính CẢ ngày hiện
  tại, trong khi mục tiêu là dự báo TRƯỚC khi biết kết quả hôm nay. Sửa
  bằng `shift(1)` trước khi cuộn. Kết quả gần như không đổi sau sửa
  (USDJPY BSS +0,23 → +0,17) — xác nhận phát hiện chính không phải do
  rò rỉ này.

### 1b. Kết quả sau khi sửa cả hai lỗi

| Cặp | BSS (không VIX) | **BSS (có VIX)** |
|---|---|---|
| EURUSD | −0,0909 | **+0,0627** |
| GBPUSD | +0,0701 | **+0,2199** |
| USDJPY | +0,1714 | +0,1737 |
| AUDUSD | +0,0953 | **+0,2283** |
| USDCAD | −0,0476 | +0,0085 |
| USDCHF | −0,1227 | −0,0548 |

Không VIX: **3/6 cặp đạt** (BSS>0). Thêm VIX (bien nhân quả đã xác nhận
qua tam giác hoá 5 phương pháp, không phải chọn ngẫu nhiên) làm đặc trưng
phụ, tra độ trễ 1 ngày: **5/6 cặp đạt** — vượt ngưỡng ≥5/6 mà dự án dùng
ở nơi khác (khai phá quy luật, CNN nến) để coi một phát hiện là đủ mạnh.

**USDCHF vẫn không đạt** — khớp với `DUOI_DIEUTRA_KETQUA.md`: cơ chế của
USDCHF là ô nhiễm TĨNH từ 2015, không phải hiện tượng động theo điều
kiện, nên một mô hình điều kiện (dù thêm VIX) không có lý do bắt được nó.
Đây là bằng chứng chéo thứ hai (sau PELT) cho việc USDCHF cần một hướng
vá khác hẳn USDJPY.

## 2. Meta-label vi phạm conformal — `src/metalabel_phuconformal.py`

**Mục tiêu**: dự đoán phiên nào khoảng conformal 90% sẽ bị vi phạm — nhắm
thẳng vào lỗ hổng đã biết "phủ thiếu khi đang lỗ" (`KhoangACI`, đã thử 5
cách không sửa được).

**Kết quả**: **0/6 cặp đạt**, BSS âm ở mọi cặp, AUC 0,52–0,56 (gần như
không phân biệt được). Đúng logic phép thử lực thấp: vi phạm 90% là sự
kiện hiếm (~10% cơ sở, ~55 ca dương trên kiểm định) — khớp với hiện tượng
đã gặp ở `RUIRO_ML.md` đích B (P(vi phạm VaR 1%), ~33 ca dương, cũng âm).
Không tiếp tục thử biến thể của loại bài toán này — bài học đã lặp lại
hai lần, thêm lần nữa không thêm giá trị.

## 3. Kiểm chứng hành động — `src/metalabel_hanhdong.py`

**Câu hỏi mục 3 (bản trước) đặt ra**: biết trước "hôm nay khó tin σ̂" có
tự động cải thiện gì không, nếu không thiết kế hành động cụ thể? Đã kiểm.

**Hành động thử**: quy tắc đơn giản nhất có thể — khi xác suất meta-label
báo "ngày tệ" vượt trung vị (chia đôi mẫu, không nhìn độ phủ để chọn
ngưỡng), nới nửa-bề-rộng conformal lên **1,15×**.

| Cặp | Độ phủ TĨNH | Độ phủ ĐỘNG | Kết quả |
|---|---|---|---|
| EURUSD | 86,5% | **90,3%** | Gần 90% hơn |
| GBPUSD | 92,0% | 93,0% | Không cải thiện (đã over-cover từ đầu) |
| USDJPY | 87,4% | **89,2%** | Gần 90% hơn |
| AUDUSD | 87,8% | **90,1%** | Gần 90% hơn |
| USDCAD | 90,0% | 91,4% | Không cải thiện (đã đạt từ đầu) |
| USDCHF | 90,3% | 91,8% | Không cải thiện (đã đạt từ đầu) |

**3/6 cặp cải thiện** — và có mẫu hình rõ: hành động chỉ giúp **những cặp
đang PHỦ THIẾU** (EURUSD, USDJPY, AUDUSD, độ phủ tĩnh < 90%). Với những
cặp đã phủ đủ/thừa từ đầu (GBPUSD, USDCAD, USDCHF), nới thêm chỉ đẩy xa
hơn khỏi mức danh nghĩa — đúng như dự đoán được về mặt logic, không phải
thất bại của phương pháp.

**Kết luận trung thực**: quy tắc "nới đều một hệ số cố định" là **quá
thô** — cần biết cả mức phủ NỀN của từng cặp trước khi quyết định nới hay
không, không chỉ dựa vào xác suất "ngày tệ". Đây là hướng tinh chỉnh tiếp
theo nếu muốn đưa vào sản xuất, không phải lý do bỏ hướng này.

## 4. Việc còn thiếu trước khi tích hợp

1. Viết lại phát hiện VIX thành biên bản đúng thứ tự (tiêu chí trước số
   liệu) — bản này chỉ ghi lại quá trình thật, không thay thế biên bản
   chính thức.
2. Chấm điểm trên kiểm tra/tập khoá sổ, cùng đợt với tail-risk.
3. Tinh chỉnh quy tắc hành động theo mức phủ nền từng cặp (mục 3), rồi đo
   lại — quy tắc cố định 1,15× cho mọi cặp đã cho thấy không đủ.

## 5. Tái lập

```bash
python src/metalabel_qlike.py
python src/metalabel_phuconformal.py
python src/metalabel_hanhdong.py
```

Kết quả: `output/metalabel_{qlike,phuconformal,hanhdong}.json`
