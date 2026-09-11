# Pha 2 — Historical + News: kết quả họ đầu tiên (09/09/2026)

*Tái lập: `python collect/tin_tuc_nhtw.py` rồi `python src/run_h8_tintuc.py`.
Kết quả: `output/h8_tintuc.json`, `output/spa_ho2.json`, log
`output/log_h8_tintuc.txt`.*

Đây là bước đầu của **Pha 2** trong `00_MASTER_ROADMAP.md` của HuyH, trả lời
**RQ4**: *"Does adding headline/news information improve prediction beyond
historical-only features?"*

---

## 1. Cổng Pha 1 → Pha 2 đã mở

Roadmap đặt điều kiện: *"Never add a new information layer before the previous
phase has a stable baseline and measurable evaluation."* Đối chiếu:

| điều kiện | trạng thái |
|---|---|
| pipeline lịch sử chạy end-to-end | ✅ |
| chọn được ít nhất một mô hình lịch sử mạnh | ✅ `NEN_THEO_H` |
| có walk-forward | ✅ `CHISO_DANHGIA.md` mục 14 |
| xác suất được hiệu chuẩn hoặc đo | ✅ ECE/MCE/PIT + **bảo đảm conformal** (mục 16) |
| frontend dùng được prediction contract | ✅ |
| đóng băng chỉ số nền Pha 1 | ⚠️ biên bản đã điền, **chưa ký** |

Năm trên sáu điều kiện đã đạt; điều kiện còn lại là chữ ký, không phải kỹ thuật.

## 2. Chọn nguồn tin — quyết định bởi rủi ro rò rỉ, không bởi độ phong phú

Cổng Pha 2 của chính roadmap đòi *"timestamp alignment between FX and news is
verified; no major leakage is detected"*. Đó là ràng buộc quyết định, vì repo
này đã dính rò rỉ **ba lần trong một ngày** (`CHISO_DANHGIA.md` mục 13, và hai
lần ở H2/H6).

Xếp các nguồn theo rủi ro rò rỉ, từ thấp lên cao:

| nguồn | dấu thời gian | rủi ro |
|---|---|---|
| **thông cáo NHTW** | lịch định trước cả năm, giờ công bố cố định | **thấp nhất** |
| số liệu vĩ mô | ngày công bố biết trước, nhưng **bản số liệu bị sửa lại** | trung bình |
| tin tức tổng hợp (GDELT…) | dấu thời gian là lúc **bài được index**, không phải lúc sự kiện | cao nhất |

Đã thử **GDELT DOC 2.0 API** trước: nó chặn IP này bất kể nhịp gửi (giới hạn
công bố là 1 request/5 giây, nhưng vẫn trả về thông báo giới hạn sau 20 giây
chờ). Không đủ tin cậy để dựng dữ liệu luận văn — và dấu thời gian của nó cũng
thuộc loại rủi ro cao nhất.

**Chọn FOMC** trong số các NHTW vì **cả sáu cặp đều có USD một vế** — một nguồn
duy nhất liên quan tới toàn bộ bảng, thay vì phải ghép sáu nguồn rồi giải thích
vì sao cặp này có mà cặp kia không.

**Dữ liệu:** 129/129 thông cáo, 2010-01-27 → 2025-12-10, trung vị 408 từ
(khoảng [86; 798]). Năm 2026 **không tải** vì nằm trong tập khoá sổ
(`KHOA_SO.md` mục 2).

## 3. Ba đặc trưng văn bản — chốt trước khi chạy

| đặc trưng | định nghĩa | cơ sở |
|---|---|---|
| **thay đổi câu chữ** | 1 − cosine(TF-IDF kỳ này, kỳ trước) | thuốc đo kinh điển của "chuyển hướng chính sách" (Acosta 2015; Hansen, McMahon & Prat 2018) — không cần từ điển cảm xúc nào |
| **đổi độ dài** | (số từ kỳ này − kỳ trước) / kỳ trước | thông cáo dài thêm thường đi kèm giải thích nhiều hơn |
| **giọng điệu** | (đếm từ thắt chặt − từ nới lỏng) / tổng từ | từ điển nhỏ, liệt kê đầy đủ trong `HAWK`/`DOVE`, chốt trước |

Phần danh sách người bỏ phiếu bị **cắt bỏ** khỏi văn bản: tên thành viên đổi
mỗi năm trong khi lập trường chính sách thì không, giữ lại sẽ làm nhiễu đúng
phép đo *thay đổi câu chữ*.

Đo được trên 128 thông cáo: thay đổi câu chữ trung vị **0,183**, khoảng
[0,021; 0,882] — tức có kỳ gần như lặp lại nguyên văn kỳ trước, có kỳ viết lại
gần hết. Đặc trưng này **có biến thiên thật**, không phải hằng số.

**Không gian giả thuyết:** 3 đặc trưng × 3 ô tam phân vị × 2 cửa sổ (1 và 5
phiên sau thông cáo) = **18 vị từ × 3 lớp = 54 giả thuyết**, liệt kê đầy đủ.

## 4. Câu hỏi phải đặt cho sắc — nếu không sẽ chỉ khám phá lại cái lịch

Repo **đã biết** ngày họp NHTW làm giá động mạnh hơn: 18 loại sự kiện, kết luận
*"sự kiện khuếch đại biên độ, không chỉ ra chiều"* (`CHISO_DANHGIA.md` mục 8).
Nên một phép đo ngây thơ kiểu "ngày có FOMC thì biên độ lớn hơn" **sẽ dương**,
và nó không nói lên điều gì mới — đó là hiệu ứng **lịch**, đã đo rồi.

Câu hỏi thật của Pha 2 là: **nội dung văn bản có nói thêm gì ngoài việc đã có
một cuộc họp không?**

Nên bộ kiểm soát của họ này được **thêm một biến**: chỉ báo *"phiên này nằm
trong cửa sổ sau một kỳ họp FOMC"* (3.870 phiên). Một vị từ chỉ sống nếu nó còn
tin riêng **sau khi đã khử cả σ̂, TSMOM, nhân tố đô-la, carry, VÀ chính cái
lịch họp**.

**Chặn rò rỉ:** thông cáo ra 14:00 giờ New York ngày họp, nên đặc trưng chỉ
được áp từ phiên **kế tiếp** trở đi. Tự kiểm xác nhận **0 vi phạm** — không đặc
trưng nào xuất hiện ở chính phiên họp.

## 5. Kết quả

| bước phễu | còn lại |
|---|---|
| không gian giả thuyết | **54** |
| đủ số lần khớp | 18/18 vị từ |
| thô p<0,05 | **6** (kỳ vọng nhiễu 3) |
| **sống sót Westfall–Young** | **0** |
| còn tin riêng sau đối chứng (đã khử lịch) | 0 |
| chuyển giao được (LOPO) | 0 |
| tái lập trên KIỂM TRA | 0 |

Ngưỡng max|z| null khối: 90% 3,00 · 95% 3,21.

**Hansen SPA so nền "chỉ σ̂"** — phép trả lời trực tiếp RQ4:

| họ | p-value | bác bỏ H0 |
|---|---|---|
| H2 motif | 0,990 | không |
| H3 rule-list | 0,590 | không |
| H5 chế độ tự tương quan | 0,872 | không |
| H6 HMM | 1,000 | không |
| H7 Matrix Profile | 0,972 | không |
| **H8 tin tức (FOMC)** | **0,162** | **không** |

## 6. Đọc kết quả

**Trả lời RQ4: không — nội dung thông cáo FOMC không cải thiện dự báo một cách
có ý nghĩa.** 0 quy luật qua phễu, và SPA không bác bỏ H0 ở α = 0,05.

Nhưng có một điểm đáng ghi, và nó là điểm khác biệt duy nhất trong toàn bộ tám
họ đã thử: **p = 0,162 là p-value thấp nhất của mọi họ**. Các họ chỉ dùng dữ
liệu giá đều nằm ở 0,59–1,00; riêng họ đầu tiên dùng **thông tin ngoài giá** rơi
xuống 0,162. Vẫn xa ngưỡng 0,05, nên **không được đọc như một phát hiện** — bài
học từ quy luật H3 bị rút lại (`GIAIDOAN2_QUYLUAT.md` mục 6.2) là kết quả gần
ngưỡng phải bị nghi ngờ nhất, không phải mừng nhất.

Cách đọc trung thực: nó **nhất quán với giả thuyết rằng trần của hệ thống nằm ở
DỮ LIỆU, không ở phương pháp**. Bảy họ vắt kiệt dữ liệu giá đều cho ~1,0; nguồn
thông tin mới đầu tiên lập tức cho con số thấp hơn ba lần. Đó là dấu hiệu định
hướng cho phần còn lại của Pha 2 và cho Pha 3, không phải một kết luận.

## 7. Hạn chế — phải nói trước khi ai đó hỏi

1. **Chỉ một ngân hàng trung ương.** ECB (157 kỳ), BOE (101), BOJ (119) chưa
   làm. Với EURUSD thì ECB quan trọng ngang FOMC; kết quả hiện tại chỉ nói về
   vế USD.
2. **Chỉ ba đặc trưng, không có embedding.** RQ5 của roadmap hỏi *"which forms
   of text representation are most useful: sentiment, event category,
   embeddings"* — mới trả lời được hai (sentiment qua `giọng điệu`, và một dạng
   *thay đổi biểu diễn* qua TF-IDF cosine). Embedding chưa thử.
3. **129 quan sát là ít.** Mỗi thông cáo chỉ cho 1–5 phiên có đặc trưng, nên số
   lần khớp tuy đủ ngưỡng 100 nhưng biến thiên độc lập thì chỉ ~128 điểm.
4. **Từ điển `HAWK`/`DOVE` là do người viết chọn.** Đã chốt trước khi chạy và
   liệt kê đầy đủ trong mã, nhưng nó không phải từ điển đã kiểm chứng
   (Loughran–McDonald chẳng hạn).

Bốn hạn chế này là việc còn lại của Pha 2, không phải lý do để nghi ngờ con số
đã đo.

## 8. H8b — embedding ngữ nghĩa (11/09/2026), trả lời phần "embedding" của RQ5

*Tái lập: `python src/run_h8b_embedding.py`. Kết quả: `output/h8b_embedding.json`.*

Hạn chế #2 ở mục 7 ghi "chưa thử embedding". Dùng CHÍNH 129 thông cáo FOMC đã
có (không thu thêm dữ liệu mới), mã hoá bằng mô hình câu pretrained
(`sentence-transformers/all-MiniLM-L6-v2`, 384 chiều, không huấn luyện thêm
trên dữ liệu FX), giảm chiều còn 3 thành phần chính (PCA, chốt trước, giải
thích 51,4% phương sai embedding). Cùng giao thức hệt H8: cùng cửa sổ (1, 5
phiên sau họp), cùng tam phân vị, cùng bộ kiểm soát (đã khử lịch họp), cùng
phễu bốn cửa. Không gian giả thuyết: 3 PC × 3 ô × 2 cửa sổ = 18 vị từ × 3 lớp
= **54 giả thuyết** — độc lập với 54 giả thuyết của H8.

**Kết quả: 0 quy luật qua phễu — giống H8.** Nhưng khác biệt quan trọng nằm ở
Hansen SPA (`run_spa_ho2.py`):

| họ | p-value SPA | ứng viên tốt nhất (thô) |
|---|---|---|
| H8 — đặc trưng thủ công (TF-IDF, độ dài, HAWK/DOVE) | **0,162** | giọng điệu vừa [1 phiên sau], Δlog=+0,00056 |
| H8b — embedding pretrained (PCA 3 thành phần) | **0,861** | PC2 vừa [1 phiên sau], Δlog=+0,00010 |

**Đọc kết quả:** embedding pretrained **không** tốt hơn ba đặc trưng thủ công
tự thiết kế — thực ra tệ hơn đáng kể (p tăng từ 0,162 lên 0,861, Δlog tốt
nhất giảm hơn 5 lần). Đây là câu trả lời cụ thể cho RQ5 của roadmap ("which
forms of text representation are most useful: sentiment, event category, hay
embeddings"): với đúng 129 quan sát và đúng bài toán này, **biểu diễn có mục
đích rõ ràng (thay đổi câu chữ, độ dài, giọng điệu — mỗi cái có lý do kinh tế
cụ thể) mang thông tin đậm đặc hơn một embedding tổng quát 384 chiều bị nén
xuống 3 thành phần bằng phương sai** — không phải vì embedding "kém", mà vì
với ~128 điểm dữ liệu độc lập, một biểu diễn tổng quát cần nhiều dữ liệu hơn
để "học" được đâu là hướng thông tin liên quan, trong khi đặc trưng thủ công
đã mã hoá sẵn giả thuyết kinh tế vào đúng 3 con số.

**Cập nhật `KHOA_SO.md`:** +54 giả thuyết (H8b), +1 nhánh độc lập — tổng giả
thuyết quy luật của toàn dự án nay là 8.577 (10 nhánh), xem dòng 33 mục 5.

**Hạn chế còn lại sau H8b:** vẫn chỉ một ngân hàng trung ương (FOMC); một mô
hình embedding duy nhất (chưa thử OpenAI/Voyage hay embedding tài chính
chuyên biệt như FinBERT); PCA tuyến tính có thể bỏ lỡ cấu trúc phi tuyến mà
một probe phi tuyến nhỏ (không phải deep) có thể bắt được — nhưng với n≈128,
rủi ro overfit của một probe phức tạp hơn lớn hơn lợi ích kỳ vọng.

## 9. MDES cho H8+H8b (11/09/2026) — "không tìm thấy gì" mạnh tới đâu?

*Tái lập: `python src/kiem_pheu_h8.py`. Kết quả: `output/kiem_pheu_h8.json`.*

Theo đúng roadmap mục 7.2 ("mọi kết quả âm phải đi kèm MDES"), và đúng cách
`kiem_pheu.py` đã làm cho Giai đoạn 1: tiêm quy luật tổng hợp có độ mạnh biết
trước vào 126 vị từ gộp của H8+H8b, xem phễu (thô → Westfall–Young → đối
chứng có điều kiện đã khử lịch họp) bắt được ở mức lift nào với lực 80%.

**Kết quả: lift ≥ 1,35 mới đạt lực 80%** — so với **lift ≥ 1,20** của phễu
Giai đoạn 1 (Direction, 8.469 giả thuyết trên hàng chục nghìn quan sát).
Phễu tin tức **yếu hơn** vì mẫu nhỏ hơn nhiều (129 thông cáo so với hàng
nghìn phiên). Đối chứng âm cho 0,0/108 dương tính giả — phễu không rò rỉ.

**Đọc kết quả:** SPA p=0,162 (H8) và p=0,861 (H8b) nay phát biểu được đầy đủ:
*"nội dung/embedding thông cáo FOMC không cho thấy hiệu ứng có ý nghĩa, và
phễu đủ lực để loại trừ mọi hiệu ứng mạnh hơn lift 1,35 với xác suất 80% —
nhưng KHÔNG loại trừ được hiệu ứng yếu hơn mức đó (giữa 1,20 và 1,35), điều
mà phễu Giai đoạn 1 loại trừ được."* Đây là giới hạn trung thực cần nêu khi
đóng Phase 2 Week 3 (quyết định STOP/GO theo roadmap): kết luận âm cho FOMC
đứng vững nhưng với lực yếu hơn phần còn lại của dự án.
