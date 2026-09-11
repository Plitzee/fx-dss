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

## 8b. H8c — chủ đề chính, và H8e — trích xuất có cấu trúc bằng LLM (11/09/2026)

Roadmap Pha 2 (`02_PHASE_2_NEWS_AWARE.md`, Week 2) liệt kê năm họ biểu diễn
văn bản. Bốn họ nay đã thử hết trên cùng 129 thông cáo FOMC:

| họ | cách biểu diễn | script |
|---|---|---|
| B | cảm xúc/giọng điệu (từ điển HAWK/DOVE) + TF-IDF + độ dài | `run_h8_tintuc.py` |
| C | **chủ đề chính** (4 nhóm theo từ điển, chốt trước) | `run_h8c_chude.py` |
| D | **embedding** ngữ nghĩa pretrained + PCA | `run_h8b_embedding.py` |
| E | **trích xuất có cấu trúc bằng LLM** (4 trường phân loại) | `run_h8e_llm.py` |

*(Họ A — số lượng/độ mới/đa dạng nguồn — không áp dụng được: FOMC chỉ có
MỘT thông cáo mỗi kỳ họp từ MỘT nguồn, nên "đếm tin" và "đa dạng nguồn" là
hằng số. Cần dữ liệu tin tức dạng dòng (wire) mới đo được, mà nguồn đó đã bị
loại vì rủi ro dấu thời gian — xem mục 2.)*

**H8c — chủ đề chính.** Bốn chủ đề chốt trước (lạm phát / việc làm / tăng
trưởng / ổn định tài chính), lấy chủ đề có mật độ từ khoá cao nhất. Phân bố
rất lệch: **109/129 kỳ là "lạm phát"**, 19 kỳ "việc làm", 1 kỳ "ổn định tài
chính", 0 kỳ "tăng trưởng" — phản ánh đúng cấu trúc ngôn ngữ mục tiêu ổn
định giá của FOMC, nhưng cũng khiến chỉ 3/8 vị từ đủ 100 lần khớp. Kết quả:
**0 quy luật qua phễu, SPA p=0,669**.

**H8e — trích xuất có cấu trúc bằng LLM.** Đây là họ E của roadmap, và là
họ DUY NHẤT trong bốn họ **đọc hiểu toàn văn** thay vì đếm từ hay đo khoảng
cách vector. Schema bốn trường, **chốt trước và commit trước khi đọc bất kỳ
thông cáo nào** (commit `cc4df90`, xem lịch sử git — đây là điểm kiểm toán
quan trọng nhất của họ này):

| trường | các mức | phân bố trên 129 kỳ |
|---|---|---|
| điều hướng | diều hâu / bồ câu / trung lập / hỗn hợp | 31 / 48 / 0 / 50 |
| bất định | thấp / trung bình / cao | 7 / 84 / 38 |
| đổi lập trường | có / không | 69 / 60 |
| phụ thuộc dữ liệu | mạnh / vừa / yếu | 110 / 11 / 8 |

Nhãn do **chính mô hình chạy phiên làm việc này (Claude Opus 5) đọc toàn văn
129 thông cáo và gán**, lưu đầy đủ ở `data/tin_tuc/fomc_llm_nhan.json` để
kiểm toán — đúng yêu cầu roadmap *"LLM không được dùng như opaque final
oracle; output phải được lưu để reproducible/audit"*. Phân bố cân đối hơn
hẳn H8c (không có mức nào chiếm >85%), nên 17/17 vị từ đều đủ số lần khớp.

Kết quả: **0 quy luật qua phễu** (4 thô p<0,05, kỳ vọng nhiễu 3 — đúng mức
ngẫu nhiên).

### Bốn cách biểu diễn, xếp theo Hansen SPA — một quy luật ngược đời

| họ | cách biểu diễn | độ "hiểu" văn bản | SPA p |
|---|---|---|---|
| **H8** | từ điển HAWK/DOVE + TF-IDF + độ dài | thấp nhất (đếm từ) | **0,162** |
| H8c | chủ đề chính (từ điển chủ đề) | thấp | 0,669 |
| H8b | embedding pretrained + PCA | trung bình (vector ngữ nghĩa) | 0,861 |
| H8e | trích xuất có cấu trúc bằng LLM | **cao nhất (đọc hiểu toàn văn)** | **1,000** |

**Biểu diễn càng "hiểu" văn bản sâu, tín hiệu đo được càng YẾU** — thứ tự
đơn điệu hoàn hảo, ngược hẳn trực giác thông thường.

Cách đọc đúng: đây **không** phải bằng chứng rằng LLM đọc hiểu kém hơn đếm
từ. Nó nhất quán với giả thuyết **không có tín hiệu nào để bắt** ngay từ
đầu. Khi không có tín hiệu, cái quyết định p-value là **phương sai của
phép đo**, không phải chất lượng biểu diễn: ba đặc trưng liên tục của H8
chia theo tam phân vị cho các vị từ cân bằng và biến thiên mượt; còn nhãn
phân loại của H8e (và H8c) gộp mỗi kỳ họp vào một trong vài nhóm, làm mất
độ phân giải mà không thêm thông tin thật. Nói cách khác: p=0,162 của H8
nhiều khả năng là **nhiễu may mắn nhất trong bốn lần thử**, chứ không phải
tín hiệu mà ba họ kia bỏ lỡ — và kết luận này càng được củng cố bởi MDES ở
mục 9 (phễu chỉ bắt được hiệu ứng ≥1,35, không loại trừ được hiệu ứng yếu).

**Hạn chế phải nêu:** nhãn LLM mang phán định chủ quan của một mô hình tại
một thời điểm; đổi mô hình hoặc phiên bản có thể cho nhãn khác, nên họ này
**không tái lập được chính xác** như ba họ kia (vốn thuần thuật toán). Đây
là đánh đổi cố hữu của phương pháp, không phải lỗi triển khai — và là lý do
nhãn được lưu ra file thay vì sinh lại mỗi lần chạy.

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
