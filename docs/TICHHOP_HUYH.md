# Nhánh khai phá mẫu của HuyH — tóm tắt và tích hợp

Lập 29/08/2026. Nguồn: 6 notebook `01_profile` → `06_final_temporal_test`,
và bài báo nền `DMKD07b.pdf` = Lin, Keogh, Wei & Lonardi (2007),
*Experiencing SAX: a novel symbolic representation of time series*,
Data Mining and Knowledge Discovery.

## 1. HuyH đã làm gì

Rời rạc hoá chuỗi tỷ giá thành **chuỗi ký hiệu** rồi khai phá **mẫu tuần tự độ
dài 3**, theo tinh thần SAX nhưng dùng ngưỡng phân vị trên lợi suất thay vì
PAA + điểm ngắt Gauss của SAX gốc.

Dữ liệu: FRED daily, 6 đồng (CAD, JPY, CHF, AUD, EUR, GBP), 1971–2026.
CNY bị loại có lý do ghi rõ (chế độ neo tỷ giá → nhiều lợi suất bằng 0 → ngưỡng
phân vị suy biến).

Ba cách biểu diễn: hướng đi 3 trạng thái, biến động 3 trạng thái, lợi suất 5
trạng thái. Ngưỡng học **chỉ từ TRAIN**, đóng băng rồi áp lên VALIDATION/TEST.

Chia: TRAIN ≤ 2016-12-31 · VALIDATION 2017–2021 · TEST 2022-01-01 → 2026-08-14.

Phễu lọc bốn bước, mỗi bước siết hơn:

| Bước | Nội dung | Còn lại |
|---|---|---|
| NB03 | khai phá trong từng cặp | 1.074 dòng thống kê TRAIN |
| NB04 | so sánh xuyên cặp, tìm mẫu chung (có kiểm định FDR cho tính không đồng nhất) | **11** mẫu chung / 4.722 tổ hợp |
| NB05 | leave-one-pair-out: phát hiện trên 5 cặp, kiểm trên cặp thứ 6 | **7** mẫu bền xuyên chuỗi |
| NB06 | mở TEST đúng một lần, luật đóng băng trước | **3** mẫu qua |

## 2. Kết quả — và vì sao nó quan trọng với luận văn

Ba mẫu sống sót, **cả ba đều là mẫu biến động**:

| Mẫu | Đích | Lift TEST | Số cặp dương |
|---|---|---|---|
| MEDIUM → HIGH → HIGH | HIGH | 1,322 | 5/6 |
| LOW → MEDIUM → LOW | LOW | 1,285 | 5/6 |
| HIGH → HIGH → MEDIUM | HIGH | 1,150 | 5/6 |

**Không một mẫu hướng đi nào lọt tới vòng cuối** — biểu diễn `direction_3state`
có **0** ứng viên được đóng băng sau LOPO. Hai mẫu `return_5state` lọt vào nhưng
**trượt** TEST.

Đây là kết luận âm thứ ba về khả năng dự báo hướng đi, và nó có giá trị đặc biệt
vì **hoàn toàn độc lập** với nhánh của tôi:

| | Nhánh mô hình liên tục | Nhánh khai phá mẫu |
|---|---|---|
| Dữ liệu | HistData M1 → realized variance 5 phút | FRED daily |
| Phương pháp | HAR/STHARQ, hồi quy | chuỗi ký hiệu, mẫu tuần tự |
| Chia dữ liệu | 2012-06 / 2020-06 / 2023-03 | 2016-12 / 2021-12 / 2022-01 |
| Kết luận hướng đi | momentum Sharpe −0,16; carry −0,05 | 0/… mẫu sống sót |
| Kết luận biến động | dự báo được, QLIKE tốt hơn nền 19,7% | 3/3 mẫu sống sót đều là biến động |

Hai phương pháp không chia sẻ dữ liệu, thước đo hay giả định, mà ra cùng một
kết luận. Trong luận văn đây là **tam giác hoá**, mạnh hơn nhiều so với hai kết
quả rời rạc.

## 3. Tôi đã kiểm chứng lại ba mẫu trên dữ liệu của mình

`src/huyh_patterns.py`. Khác ba điểm: thước đo biến động là **realized variance
5 phút** (chính xác hơn hẳn |lợi suất| ngày), khoảng 2012–2025, cách chia của
`split.py`. Chấm trên đoạn kiểm tra:

| Mẫu | Lift của HuyH | Lift ở đây | Số cặp dương |
|---|---|---|---|
| MEDIUM → HIGH → HIGH ⇒ HIGH | 1,322 | **1,690** | 3/3 |
| LOW → MEDIUM → LOW ⇒ LOW | 1,285 | **1,318** | 5/6 |
| HIGH → HIGH → MEDIUM ⇒ HIGH | 1,150 | **1,690** | 3/3 |

**3/3 tái lập**, và lift còn cao hơn — đúng như kỳ vọng khi thay thước đo nhiễu
bằng thước đo tốt hơn.

Bảng đối chứng (toàn bộ 48 tổ hợp mẫu × đích đủ mẫu): trung vị lift 0,967,
phân vị 75% là 1,270, phân vị 90% là 1,380. Ba mẫu của HuyH nằm ở phân vị
**81%, 96% và 96%** — thật sự mạnh, không phải mức thường gặp.

**Nhưng bảng đối chứng cũng cho một cảnh báo:** mẫu mạnh nhất trên dữ liệu này
là `HIGH → HIGH → HIGH ⇒ HIGH` với lift **4,80**, gấp gần ba lần mẫu tốt nhất
của HuyH. Phễu lọc của HuyH không đưa nó ra. Nói cách khác: cả ba mẫu sống sót
đều là **quán tính biến động** — một stylized fact đã biết từ lâu — và mẫu quán
tính thuần tuý nhất lại là mẫu mạnh nhất.

## 4. Tích hợp được cái nào?

### 4a. Vào tầng 2 (dự báo biến động) — ĐÃ THỬ, KHÔNG DÙNG

`src/run_symbolic.py`: thêm đặc trưng ký hiệu vào STHARQ ở bốn mức, chấm theo
đúng quy trình huấn luyện/kiểm định/kiểm tra.

| Mô hình | QLIKE kiểm định | QLIKE kiểm tra | So với gốc |
|---|---|---|---|
| STHARQ | 0,1086 | 0,1587 | — |
| STHARQ + trạng thái | 0,1089 | 0,1585 | 0,14% |
| STHARQ + cặp trạng thái | 0,1092 | 0,1579 | 0,49% |
| STHARQ + 3 mẫu HuyH | 0,1085 | 0,1585 | 0,15% |
| STHARQ + tất cả ký hiệu | 0,1093 | 0,1580 | 0,46% |

Diebold–Mariano: **0/6 cặp** cho mọi biến thể. Model Confidence Set giữ cả năm
ở 6/6 — không phân biệt được.

**Kết luận: không tích hợp vào tầng 2.** Lý do có ý nghĩa chứ không phải thất
bại: mô hình liên tục **đã bắt hết** thông tin này rồi. Biểu diễn ký hiệu là một
phiên bản **mất mát** của chính thứ mà số hạng chuyển chế độ của STHARQ dùng ở
dạng liên tục. Rời rạc hoá thành ba trạng thái vứt đi thông tin, không thêm.

Đây là câu trả lời sạch cho câu hỏi hai nhánh liên hệ thế nào, và nên vào luận
văn nguyên như vậy.

### 4b. Vào tầng 6 (phiếu quyết định) — ĐÃ TÍCH HỢP

Đây mới là chỗ nó đáng giá. Mẫu ký hiệu là thứ **người đọc hiểu được**, còn hệ
số của STHARQ thì không. `LuatKyHieu` trong `src/decision_record.py` đọc ba
phiên gần nhất và in một câu:

```
│ LUẬT KÝ HIỆU (nhánh khai phá mẫu)                              │
│   Ba phiên gần nhất: thấp → thấp → thấp. Trong lịch sử, sau    │
│   mẫu này xác suất biến động thấp cao gấp 1.41 lần mức nền.    │
```

Không tham gia tính đòn bẩy, không tham gia tính khoảng. Chỉ giải thích. Đúng
vai trò một hệ hỗ trợ **quyết định**: con số đến từ mô hình liên tục, lời giải
thích đến từ mẫu ký hiệu.

### 4c. Nên mượn — thủ tục leave-one-pair-out

Nhánh của tôi chưa có phép kiểm này. Walk-forward trong từng cặp trả lời "mô
hình có ổn định theo thời gian không" nhưng **không** trả lời "mô hình có chuyển
được sang cặp chưa từng thấy không". LOPO của HuyH trả lời đúng câu đó, và áp
được nguyên xi cho tầng 2: khớp trên 5 cặp, chấm trên cặp thứ 6.

Đây là mục đáng làm tiếp theo cho tầng 2.

## 5. Ba điểm cần ghi trong luận văn khi trình bày nhánh này

1. **Nền của lift tính trên chính đoạn TEST** (`test_baseline_probability` lấy
   từ phân phối trạng thái kế tiếp trong TEST). Hợp lý — nó biến lift thành đại
   lượng tương đối trong cùng thời kỳ — nhưng phải nói rõ, vì nó hấp thụ luôn
   phần trôi phân phối. Nếu dùng nền của TRAIN thì lift sẽ khác.
2. **`test_target_is_max_lift_state_rate` chỉ 0,50–0,83**: ngay cả với mẫu sống
   sót, trạng thái được dự đoán thường **không** phải trạng thái có lift cao
   nhất. HuyH đã ghi đúng đây là chẩn đoán, không phải luật — nên giữ nguyên
   cách trình bày đó.
3. **Ba mẫu sống sót đều là quán tính biến động.** Đó là điểm mạnh (nhất quán
   với tài liệu và với nhánh kia) đồng thời là điểm yếu (không mới). Trình bày
   nó như **xác nhận độc lập một stylized fact bằng một phương pháp khác**, chứ
   đừng trình bày như phát hiện mới.

## 6. Tái lập

```bash
python src/huyh_patterns.py    # kiểm chứng 3 mẫu + bảng đối chứng 48 tổ hợp
python src/run_symbolic.py     # thử đưa ký hiệu vào tầng 2 (~60s)
python src/decision_record.py  # phiếu có dòng luật ký hiệu
```

## Nguồn

- Lin, Keogh, Wei, Lonardi (2007), *Experiencing SAX: a novel symbolic representation of time series*, Data Mining and Knowledge Discovery 15(2), 107–144.
