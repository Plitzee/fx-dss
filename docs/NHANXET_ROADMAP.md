# Nhận xét roadmap của HuyH — đối chiếu với những gì repo đã đo

*09/09/2026. Nhận xét hai văn bản `00_MASTER_ROADMAP.md` và
`01_PHASE_1_HISTORICAL_ONLY.md`.*

Văn bản này không phản đối kế hoạch. Phần khung của nó tốt hơn phần lớn kế
hoạch nghiên cứu mà một đồ án ở mức này thường có. Nhưng nó được viết như thể
dự án bắt đầu từ số không, trong khi repo đã đi qua ~10 tuần và **đã trả giá
để học đúng những thứ kế hoạch còn thiếu**. Mục đích văn bản này là ghép hai
thứ đó lại.

---

## 1. Những gì kế hoạch làm ĐÚNG, và nên giữ nguyên

| điều | vì sao đáng giữ |
|---|---|
| **Cổng giữa các pha** — "không thêm tầng thông tin mới khi tầng trước chưa có nền ổn định" | Đây là ý tưởng tốt nhất trong cả hai văn bản. Không có nó thì không quy được cải thiện về đúng nguồn thông tin, và mọi Δ(M2,M1) đều vô nghĩa. |
| **Nền bắt buộc** (§7) và câu *"The goal is not to prove that a complex model is sophisticated"* | Đúng tinh thần. Repo đã xác nhận bằng số: TabPFN v2 (SOTA 2026) thua nền "chỉ σ̂". |
| **Không chấm bằng một chỉ số** (§6) | Đúng, dù danh sách chỉ số cần sửa — xem mục 2.5. |
| **Hợp đồng dự báo ổn định** (§5) | Nhờ nó mà thêm tầng conformal vào không phải viết lại giao diện. |
| **Danh sách "không làm"** rõ ràng ở mỗi pha | Chặn phình phạm vi. Hiếm kế hoạch nào có. |
| **Bảng theo dõi thí nghiệm** (§11) kèm artefact cụ thể | Đã dựng: `results/experiment_summary.csv`. |

---

## 2. Sáu lỗ hổng — xếp theo mức nghiêm trọng

### 2.1 KHÔNG có kiểm soát đa kiểm định ở bất kỳ đâu — lỗ hổng lớn nhất

Kế hoạch bảo so Accuracy / Macro F1 / Log Loss / Brier giữa các bộ đặc trưng
(Bảng B), và so với nền (§7). Nhưng **không chỗ nào nói phải hiệu chỉnh cho
việc đã thử bao nhiêu thứ.**

Repo đã đo con số đó:

> Trên 1.890 giả thuyết liệt kê đầy đủ, **522 đạt p < 0,05 thô** — trong khi
> kỳ vọng dưới nhiễu thuần là 87. Hệ số thổi phồng **6,0 lần**. Sau
> Westfall–Young: **0 giả thuyết sống sót.**

Nghĩa là: **nếu làm đúng theo kế hoạch, sẽ "tìm ra" hàng trăm quy luật và
không cái nào là thật.** Đây không phải rủi ro lý thuyết — đó là con số đo
được trên chính bộ dữ liệu này.

**Cần thêm:** một mục ngang hàng với §6 và §7, quy định mọi so sánh nhiều giả
thuyết phải qua kiểm soát bội (Westfall–Young maxT với null khối, đã có sẵn
trong `src/run_quyluat.py`), và **không gian giả thuyết phải liệt kê được đầy
đủ TRƯỚC khi chạy** — nếu không thì mọi hiệu chỉnh đều giả.

### 2.2 KHÔNG có phân tích lực (power / MDES)

Kế hoạch không hỏi *"hiệu ứng nhỏ đến đâu thì ta vẫn phát hiện được?"* Thiếu
nó thì một kết quả âm **không phát biểu được thành gì cả** — không phân biệt
được "không có gì" với "có nhưng ta không đủ sức thấy".

Repo đã dựng `src/kiem_pheu.py` để trả lời, và nhờ nó kết luận đổi từ

> *"chúng tôi không tìm thấy gì"*  (không dùng được)

thành

> *"chúng tôi không tìm thấy gì, **và phễu này bắt được quy luật có lift ≥ 1,20
> với xác suất 80%**, nên mọi quy luật mạnh hơn thế đã bị loại trừ trên dữ liệu
> này"*  (một phát biểu khoa học)

**Cần thêm:** với mọi kết quả âm, MDES là **sản phẩm bắt buộc**, không phải
tuỳ chọn.

### 2.3 KHÔNG có chi phí giao dịch ở bất kỳ đâu

Cả hai văn bản không nhắc chi phí một lần nào. Với FX khung ngày và biên lợi
thế cỡ này, đó là thiếu sót quyết định:

| | đo được trong repo |
|---|---|
| spread trung vị | **0,3 – 1,2 pip** tuỳ cặp |
| dải mục tiêu h=1 (EURUSD) | ~14 pip |
| toàn mạch 26 năm | Sharpe **0,25–0,46**, vốn cuối **1,004–1,037** |

Một kế hoạch chỉ chấm bằng Accuracy/LogLoss/Brier **có thể tuyên bố thành công
trên một thứ lỗ tiền sau chi phí**. Repo có sẵn `data/cost_table.csv`.

**Cần thêm:** chi phí vào §6, và một tiêu chí "lợi thế có sống sót sau chi phí
không" ở cổng mỗi pha.

### 2.4 Rò rỉ chỉ được nhắc MỘT dòng, và chỉ ở cổng Pha 2

§12 ghi *"no major leakage is detected"* như một ô tích. Nhưng rò rỉ là **kiểu
hỏng chiếm ưu thế** của loại bài toán này. Chỉ riêng ngày 08–09/09, repo dính
ba lỗi:

| lỗi | hậu quả |
|---|---|
| `nancumsum` bịa lợi suất bằng 0 cho 3.306 hàng | tạo ra **một "quy luật" giả qua được cả bốn cửa kiểm định** |
| mẫu số H2 gồm cả phiên thiếu cửa sổ (99,7% "đi ngang") | 72/72 giả thuyết "sống sót" — 100%, vô lý |
| nhãn trạng thái HMM hoán vị giữa các cặp | gộp lẫn hai chế độ khác nghĩa nhau |

Cộng thêm lần trước đó ở `run_ml3.py` (lệch một phiên giữa đặc trưng và đích,
cho AUC 0,967 — con số không thể thật).

**Cần thêm:** một **giao thức chống rò rỉ thường trực**, không phải ô tích ở
cổng. Tối thiểu: mỗi đặc trưng mới phải có tự kiểm *"giá trị tại t không đổi
khi cắt bỏ toàn bộ dữ liệu sau t"*. Repo có ba ví dụ đã viết sẵn
(`run_h6_hmm.py`, `run_h7_matrixprofile.py`, `run_h8_tintuc.py`).

### 2.5 Danh sách chỉ số dẫn đầu bằng hai chỉ số SAI cho bài toán này

§6 liệt kê Accuracy và Macro F1 **trước tiên**. Đo được trên chính dữ liệu này
(kiểm định, mục tiêu P, h=1):

| mô hình | log | BSS | **chính xác** |
|---|---|---|---|
| khí hậu học (không kỹ năng) | 1,0980 | +0,0000 | **0,3620** |
| tổ hợp trực tuyến (sản xuất) | 1,0866 | +0,0107 | **0,3751** |

Từ "không có kỹ năng gì" lên "mô hình sản xuất", **độ chính xác chỉ nhúc nhích
1,3 điểm phần trăm** trong khi log score và BSS tách bạch rõ ràng với KTC không
phủ 0. Lý do: hai chỉ số đó chỉ đọc cột lớn nhất rồi vứt bỏ phân phối — mà sản
phẩm giao cho người dùng lại là **ba con số**.

**Cần sửa:** đưa quy tắc chấm điểm chính đáng (log score, Brier, hiệu chuẩn)
lên trước; giữ Accuracy/F1 ở mục "báo cáo cho quen thuộc", kèm cảnh báo.

### 2.6 `expected_return` và `confidence` trong hợp đồng dự báo — nguy hiểm nhất vì nó nằm trên mặt sản phẩm

§5 và mẫu đầu ra ở §2 hiện:

```
Expected return: +0.18%
Confidence: Medium
```

Repo đã đo: **AUC hướng 0,46–0,53, và 24/24 ô đều có KTC phủ 0,50.** Không
phân biệt được tăng với giảm. In một con số lợi nhuận kỳ vọng là **công bố một
kỹ năng đã được đo là không tồn tại**.

**Cần sửa:** bỏ `expected_return` khỏi hợp đồng, hoặc bắt buộc kèm khoảng tin
cậy (mà ở đây sẽ phủ 0). Thay bằng thứ hệ thống thật sự biết: biên độ dự kiến,
và **tập dự báo conformal có bảo đảm** — đã có trong API từ 09/09.

---

## 3. Ba thiếu sót về cấu trúc

### 3.1 Không có tập niêm phong

Kế hoạch có walk-forward và cross-pair — tốt. Nhưng không có thứ tương đương
`docs/KHOA_SO.md`: một tập **không được chạm cho tới lần chạy cuối cùng**.

Mà chính kế hoạch lại khuyến khích lặp nhiều vòng ("hyperparameter
optimization", "model selection"). Repo đã đếm: **~685 cấu hình mô hình** đã
thử trên tập phát triển. Không có tập niêm phong thì con số cuối cùng đã bị
nhiễm, bất kể walk-forward chặt đến đâu.

### 3.2 Pha 2 và Pha 3 không có tiêu chí phủ định

Pha 1 có tiêu chí dừng (`REPLAN_2026.md` §10.4: cả năm họ trượt SPA + không
quy luật nào qua LOPO + đúng trên cả hai mục tiêu và ba tầm hạn). Pha 2 và Pha
3 **không có gì tương đương**.

Thiếu nó thì hai pha sau sẽ *luôn* tìm ra một cái gì đó — vì không ai định
nghĩa trước thế nào là "không có gì".

**Cần thêm:** với mỗi pha, viết trước câu *"kết quả nào sẽ khiến ta kết luận
tầng thông tin này không giúp gì, và dừng"*.

### 3.3 Danh sách câu hỏi nghiên cứu không có chỗ cho phát hiện chính của dự án

RQ1–RQ7 đều hỏi *"dự báo có tốt lên không"*. Nhưng kết quả trung tâm của repo
là một câu hỏi khác:

> **Kỹ năng nằm ở TRỤC NÀO?** Hướng đi: không có (AUC phủ 0,50 ở 24/24 ô, 0 quy
> luật trên 8.469 giả thuyết). Biên độ: có (BSS dương 14/14 năm, 6/6 cặp có ý
> nghĩa trên kiểm tra).

Không RQ nào diễn đạt được điều đó. Đây là thiếu sót **cấu trúc**, không phải
chi tiết: kế hoạch không có chỗ để phát biểu kết quả chính của chính dự án.

**Cần thêm:** một RQ dạng *"Trên trục nào (hướng / biên độ / rủi ro) thông tin
X mang lại giá trị đo được?"* — áp cho cả ba pha.

---

## 4. Chỗ kế hoạch đã lạc hậu so với repo

| kế hoạch nói | thực tế |
|---|---|
| Pha 1 là 4 tuần, bắt đầu từ pipeline | Pha 1 **đã xong 5/6 điều kiện cổng**; chỉ còn chữ ký biên bản |
| Week 2: "thêm HMM và Matrix Profile rồi đo xem có cải thiện" | **Đã làm cả hai** (H6, H7) — 0 quy luật, SPA p = 1,000 và 0,972 |
| Mô hình cây: "XGBoost hoặc LightGBM" | LightGBM −0,0149, GRU −0,1083, TabPFN v2 +0,0085 — **cả ba thua nền σ̂ (+0,0110)** |
| "Do not prioritize intraday microstructure" | H1 **đã thử rồi** (592.343 quan sát) — 0 quy luật |
| RQ2: pattern/regime có hơn nền đơn giản không | **Đã trả lời: không.** Và toàn bộ tầng mô hình chỉ thêm **+0,00013** BSS so với "chỉ σ̂" trần |

---

## 5. Đề xuất cụ thể

**Sửa ngay (rẻ, ảnh hưởng lớn):**

1. Bỏ `expected_return` khỏi hợp đồng dự báo, hoặc bắt kèm KTC.
2. Đảo thứ tự §6: quy tắc chấm điểm chính đáng trước, Accuracy/F1 sau kèm cảnh báo.
3. Thêm chi phí giao dịch vào §6 và vào cổng mỗi pha.

**Thêm mục mới:**

4. **§ Kiểm soát đa kiểm định** — không gian giả thuyết liệt kê trước, Westfall–Young, đối chứng âm/dương.
5. **§ Lực phát hiện** — MDES là sản phẩm bắt buộc của mọi kết quả âm.
6. **§ Giao thức chống rò rỉ** — tự kiểm "cắt tương lai không đổi giá trị" cho mọi đặc trưng mới.
7. **§ Tập niêm phong** — chạy đúng một lần, ghi biên bản trước.

**Sửa cấu trúc:**

8. Viết tiêu chí phủ định cho Pha 2 và Pha 3.
9. Thêm RQ về **trục nào có kỹ năng**, áp cho cả ba pha.
10. Viết lại Pha 1 theo trạng thái thật (đã xong), thay vì kế hoạch 4 tuần từ đầu.

---

## 6. Kết luận

Kế hoạch **đúng về khung, thiếu về phòng tuyến**. Nó mô tả tốt việc *phải làm
gì*, nhưng gần như không nói *làm sao để không tự lừa mình* — mà với bài toán
có tỷ lệ tín hiệu trên nhiễu thấp như FX khung ngày, đó mới là phần khó.

Bốn thứ repo có mà kế hoạch không có — kiểm soát bội, phân tích lực, giao thức
chống rò rỉ, tập niêm phong — **chính là bốn thứ khiến kết quả âm của dự án
này đáng tin**. Nếu ghép chúng vào kế hoạch thì Pha 2 và Pha 3 sẽ cho ra kết
luận dùng được, dù kết luận đó là dương hay âm.
