# Giai đoạn 2 — khai phá quy luật: phễu 1.890 → 0

*04/09/2026. Tái lập: `python src/run_quyluat.py`.
Kết quả: `output/quyluat.json`, `output/quyluat_wy9.json`, log `output/log_quyluat.txt`.*

Đây là câu hỏi trung tâm của luận văn: **có tồn tại một danh sách quy luật, học
từ nhiều cặp cùng lúc, chuyển giao được sang cặp chưa từng thấy không?**

Câu trả lời trên dữ liệu này: **không** — và cách nó không mới là phần đáng viết.

---

## 1. Thiết kế: không gian giả thuyết phải liệt kê được đầy đủ

Westfall–Young hiệu chỉnh cho **số giả thuyết đã thử**. Nếu đi tìm quy luật một
cách mở — chạy CART rồi lấy lá, chạy motif rồi lấy cụm — thì không ai biết thực
sự đã thử bao nhiêu, và mọi hiệu chỉnh bội đều là giả.

Nên không gian được định nghĩa trước và vét cạn:

```
vị từ  = một hoặc HAI mệnh đề dạng (đặc trưng, ô phân vị)
đích   = một trong ba lớp
```

12 đặc trưng đọc được × 3 ô = 36 mệnh đề → **630 vị từ × 3 lớp = 1.890 giả
thuyết**, con số biết trước, in ra mỗi lần chạy.

Đặc trưng: σ̂ · ATR phân vị · RSI · ADX · Bollinger %B · khoảng cách EMA50 ·
Supertrend chiều · MACD hist · |z| hôm nay · z hôm nay · TSMOM 20 · tính dai vol.

Ngưỡng phân vị chốt trên **đoạn huấn luyện**. Đích **dịch một phiên** — đặc
trưng của ngày t nói về lớp của ngày t+1 (bài học từ rò rỉ ở `run_ml3.py`).

---

## 2. Phễu

> **Số đã cập nhật 08/09/2026** sau khi sửa lỗi bịa lợi suất bằng 0
> (`CHISO_DANHGIA.md` mục 13). Mẫu phát hiện giảm từ 21.606 xuống 18.306 hàng
> vì 3.306 hàng khởi động có lợi suất bịa ra đã bị loại đúng cách. Kết luận
> cuối (**0 quy luật**) không đổi; mọi con số trung gian đều đổi, và phễu mạnh
> lên rõ rệt.

| bước | còn lại | *(số cũ, sai)* |
|---|---|---|
| không gian giả thuyết (liệt kê đầy đủ) | **1.890** | 1.890 |
| đủ 100 lần khớp | 1.749 | *1.764* |
| thô p<0,05 (chưa hiệu chỉnh) | **522** | *1.186* |
| sống sót Westfall–Young | **3** | *9* |
| còn tin riêng sau đối chứng có điều kiện | **0** | *0* |
| chuyển giao được (bỏ-một-cặp) | 0 | *0* |
| tái lập trên kiểm tra | 0 | *0* |

Ngưỡng `max|z|` của null khối 5 ngày: 90% **3,70** · 95% **3,89** · 99% **4,36**.

**Hệ số thổi phồng: 522 so với 87 kỳ vọng nếu toàn nhiễu — gấp 6,0 lần.**

Đây là con số phải đưa vào luận văn. Nó nói: nếu ai đó chạy đúng bộ đặc trưng
này, không hiệu chỉnh bội, họ sẽ "tìm ra" **522 quy luật có ý nghĩa thống kê**
và không cái nào là thật.

---

## 3. Ba vị từ sống sót — và tất cả đều là cùng một thứ

| vị từ | lớp | n | lift | z | b sau điều kiện | **t sau điều kiện** |
|---|---|---|---|---|---|---|
| σ̂ thấp | đi ngang | 5.622 | 1,179 | 9,41 | +0,0366 | **1,70** |
| σ̂ thấp | tăng | 5.622 | 0,901 | −5,33 | −0,0160 | **−0,80** |
| σ̂ thấp | giảm | 5.622 | 0,923 | −4,04 | −0,0206 | **−1,04** |

**Cả ba đều là σ̂ rời rạc hoá.** Không một chỉ báo kỹ thuật nào lọt vào: không
RSI, không ADX, không Bollinger, không MACD, không Supertrend, không khoảng
cách EMA, không ATR phân vị, không TSMOM.

Và cả ba đều **rớt ở cửa đối chứng có điều kiện** — |t| lớn nhất là 1,70,
ngưỡng là 3,0. Hiển nhiên, vì vị từ **chính là** σ̂, mà biến kiểm soát cũng là
log σ̂. Chúng không mang thông tin độc lập nào.

Hướng của con số cũng đúng trực giác: **σ̂ thấp → nhiều "đi ngang" hơn**
(lift 1,179), ít "tăng"/"giảm" hơn. Bảng cũ (trước khi sửa lỗi ở
`CHISO_DANHGIA.md` mục 13) cho cả ba ô σ̂ đều có lift < 1 với "đi ngang" —
điều **không thể xảy ra** với một phân hoạch phủ kín mẫu, và đó chính là dấu
vết đã dẫn tới việc tìm ra lỗi.

**Đọc theo cách khác:** trong toàn bộ không gian 1.890 giả thuyết, thứ duy nhất
sống sót hiệu chỉnh bội đúng cách là **chính dự báo biến động mà hệ thống đã
có**. Khai phá không tìm thêm được gì.

---

## 4. Tám nhánh độc lập, cùng một kết luận

| nhánh | không gian | sống sót W-Y | còn tin riêng | SPA vs nền |
|---|---|---|---|---|
| SAX **biến động** (`run_sax_stats.py`) | 336 | **41** | **2** (t = 8,94 và 6,05 khi có HAR) | — |
| SAX **hướng giá** (`run_sax_gia.py`) | 351 | **0** | — | — |
| Ngưỡng đặc trưng **ba lớp** (file này) | 1.890 | 3 | **0** | — |
| **H2 motif** (`run_h2_motif.py`) | 72 | **0** | — | p = 0,990 |
| **H3 rule-list** (`run_h3_rulelist.py`) | 24 | **0** | — | p = 0,590 |
| **H5 chế độ tự tương quan** (`run_h5_chedo.py`) | 5.670 | **0** | — | p = 0,872 |
| **H6 HMM** (`run_h6_hmm.py`) | 18 | 6 | **0** | p = 1,000 |
| **H7 Matrix Profile** (`run_h7_matrixprofile.py`) | 108 | **0** | — | p = 0,972 |

Bất đối xứng rất rõ và rất nhất quán: **trục biến động có cấu trúc khai phá
được; trục hướng đi và trục ba-lớp thì không.**

Nhánh biến động vượt một cái rào cao hơn hẳn (max|z| ≈ 69 so ngưỡng 29,51) và
để lại hai mẫu mang thông tin **độc lập với HAR**. Hai nhánh còn lại không để
lại gì.

---

## 5. Tiêu chí dừng ở mục 10.4 — đã kích hoạt chưa?

`REPLAN_2026.md` mục 10.4 đòi **cả ba** điều kiện: mọi họ trượt Hansen SPA, không
quy luật nào qua ngưỡng 3.5 sau LOPO, **và** đúng trên cả hai mục tiêu lẫn cả ba
tầm hạn.

Cập nhật 08/09/2026 — **cả năm họ H1–H5 giờ đã có code và đã chạy** (xem mục 6),
số đã sửa sau lỗi bịa lợi suất (`CHISO_DANHGIA.md` mục 13):

| điều kiện | trạng thái |
|---|---|
| không quy luật nào qua ngưỡng sau LOPO | **ĐÚNG** — cả sáu nhánh đều 0 quy luật. (Bản đầu của mục này ghi H3 có một ngoại lệ; ngoại lệ đó là artefact của lỗi dữ liệu, đã rút lại — mục 6.2) |
| trên cả hai mục tiêu | mới đo mục tiêu P, h=1 |
| trên cả ba tầm hạn | mới đo h=1 |
| Hansen SPA cho cả họ | **ĐÚNG, đóng được cho cả 5 họ ở cấu hình đã chạy** (`run_spa.py` cho Giai đoạn 1, `run_spa_ho2.py` cho H2/H3/H5) — **không họ nào bác bỏ được H0** so nền "chỉ σ̂" (mục 6.3) |

**Đọc kết quả này thế nào.** Hai trong ba điều kiện của mục 10.4 giờ đã **đúng
hoàn toàn** ở cấu hình h=1/mục tiêu P: không quy luật nào qua LOPO, và không họ
nào thắng nền qua SPA. Điều kiện còn lại (đúng trên **cả hai** mục tiêu và **cả
ba** tầm hạn) chưa đo hết — mới có h=1/mục tiêu P. Nên tiêu chí dừng **chưa
kích hoạt đủ theo nghĩa đen**, nhưng phần khó nhất đã đóng và bằng chứng nhất
quán qua sáu nhánh độc lập.

Và bằng chứng giờ **mạnh hơn trước**, không chỉ sạch hơn: phễu sau khi sửa lỗi
bắt được quy luật có lift ≥ **1,20** với lực 80% (trước là 1,35) — nên câu "không
tìm thấy gì" nay loại trừ được một dải quy luật rộng hơn hẳn.

Cộng với `run_ml3.py` (LightGBM −0,0148 và GRU −0,0594 — trần GBM nằm **dưới**
nền) thì H4 cũng đã trả lời: trần khai thác được của bộ đặc trưng này thấp hơn
chính nền σ̂.

---

## 6. H2, H3, H5 — ba họ còn lại đã chạy (08/09/2026)

Tiếp tục đóng B1 (`docs/KEHOACH_CAITIEN.md` mục B1): viết code cho ba họ chưa
tồn tại (`src/run_h2_motif.py`, `src/run_h3_rulelist.py`, `src/run_h5_chedo.py`),
dùng lại **đúng** bộ máy WY/đối chứng/LOPO của `run_quyluat.py` (nay tách ra
`nap_du_lieu()` để ba script này gọi lại, không viết lại).

### 6.1 H2 — motif (VQ codebook): 0 sống sót, nhưng bắt được một lỗi rò rỉ đáng nói

Ý tưởng: 3 độ dài cửa sổ (5/10/20 phiên) × K=8 cụm hình dạng (KMeans trên z
chuẩn hoá trong cửa sổ, học trên huấn luyện) = 24 vị từ × 3 lớp = 72 giả thuyết.

**Lần chạy đầu tiên cho kết quả vô lý: 72/72 giả thuyết "sống sót" W-Y** — tức
100%, trong khi kỳ vọng dưới nhiễu chỉ ~4. Truy ra nguyên nhân: phiên **THIẾU**
cửa sổ hợp lệ (do `merge_thin_days` gộp ngày mỏng dữ liệu) có tỷ lệ lớp "đi
ngang" **99,3%**, trong khi tỷ lệ nền toàn tập là 30,8% — tức bản thân việc "có
đủ 20 phiên liên tục không thiếu dữ liệu" **đã là một dự báo gần hoàn hảo**,
không liên quan gì đến hình dạng cửa sổ. Vì mọi cụm đều kế thừa đúng điều kiện
"có cửa sổ" giống hệt nhau, tất cả cùng "có tín hiệu" như nhau — giả.

**Sửa**: giới hạn phép so sánh về đúng tập phiên có đủ cả ba độ dài cửa sổ
(mẫu số đồng nhất cho mọi giả thuyết). Sau khi sửa: **0/72 sống sót
Westfall–Young**. Không có quy luật.

**Đây là lần đầu của cùng một lỗi, chưa nhận ra tận gốc.** Lúc đó vá bằng cách
thu hẹp mẫu so sánh cho riêng H2, và tưởng nguyên nhân là `merge_thin_days`.
Nguyên nhân thật sâu hơn một tầng — 3.306 hàng có lợi suất **bịa thành 0** vì
`nancumsum`, ảnh hưởng tới **mọi** nhánh chứ không riêng H2. Chỉ khi soi lại
quy luật H3 (mục 6.2) mới lần ra được. Sau khi sửa tận gốc, H2 còn 2/72 thô
p<0,05 — đúng bằng kỳ vọng nhiễu ~4, sạch hơn cả bản vá cục bộ (9/72).

### 6.2 H3 — rule-list (CART nông): 0 sống sót, sau khi RÚT LẠI một phát hiện sai

Cấu hình **chốt trước khi chạy** (không dò nhiều độ sâu rồi chọn): CART
`max_depth=3, min_samples_leaf=200` trên 12 đặc trưng đã có (chuyển hạng phân
vị trong từng cặp để gộp công bằng), sinh 8 lá → 24 giả thuyết.

Phễu sau khi sửa lỗi dữ liệu: 17 thô p<0,05 → **0 sống sót Westfall–Young**.
Không quy luật nào.

**Đã rút lại một phát hiện.** Bản chạy đầu tiên (trước khi sửa lỗi ở
`CHISO_DANHGIA.md` mục 13) báo 3 sống sót W-Y và **một quy luật qua hết bốn
cửa**: `σ̂ rất thấp VÀ ATR rất thấp → đi ngang`, lift 1,67, t|đk 4,34, LOPO 6/6,
tái lập kiểm tra z=4,31. Nó đã được ghi vào `rules/rules_h3.csv` và báo cáo là
quy luật đầu tiên của cả Giai đoạn 2.

**Nó không có thật.** Nguyên nhân: 3.306 hàng khởi động có lợi suất bị *bịa*
thành 0 và bị gán hết vào lớp "đi ngang" (99,7%). Chúng nằm trong mẫu nền nhưng
không nằm trong bất kỳ vị từ nào, nên kéo tỷ lệ nền của "đi ngang" lệch hẳn:
30,8% trên toàn mẫu so với 18,4% trên phần đủ đặc trưng. Một lá có tỷ lệ hoàn
toàn bình thường (30,7%) vì thế trông như lift 1,67 khi so với 18,4%. Toàn bộ
"phát hiện" là chênh lệch giữa hai mẫu nền. Chi tiết đầy đủ: `CHISO_DANHGIA.md`
mục 13.6.

**Bốn phép kiểm chéo ở `src/kiem_h3.py` đã bác nó trước cả khi tìm ra nguyên
nhân gốc** — đáng ghi lại vì đó là bộ khung dùng lại được cho mọi quy luật ứng
viên sau này:

| phép kiểm | ý tưởng | kết quả |
|---|---|---|
| A. đối chứng mạnh hơn | thêm biến giả phân vị **ATR** vào bộ kiểm soát (không chỉ σ̂) | t 4,34 → 3,22 — yếu đi rõ |
| B. **mục tiêu R** | dải chia theo σ̂ nên khử quan hệ cơ học "biến động thấp → nằm trong dải" | **chết** (t = 1,20) |
| C. tầm hạn 5, 20 | quy luật thật phải để lại dấu vết ở tầm hạn dài hơn | **chết, và đảo dấu** (lift 0,71 · z = −5,9) |
| D. đóng góp thật | thêm quy luật vào nền, đo ΔBSS trên kiểm định | KTC chứa 0; chỉ bật 0,6% số phiên |

Phép B là phép sắc nhất: mục tiêu R chia biên độ cho σ̂, nên nếu "quy luật" chỉ
là quan hệ cơ học *biến động thấp → biên độ nhỏ → nằm trong dải pip cố định*
thì nó phải biến mất — và nó biến mất.

### 6.3 SPA cho cả năm họ — không họ nào thắng nền "chỉ σ̂"

`src/run_spa_ho2.py`, `output/spa_ho2.json`. Biến mỗi vị từ của một họ thành
MỘT ứng viên dự báo ba lớp (tần suất có điều kiện trên huấn luyện nếu vị từ
đang active, ngược lại dùng chính dự báo nền "chỉ σ̂" của phiên đó — dịch đúng
1 phiên để khớp chỉ số hàng của `run_quyluat.py`), rồi áp `spa_test()` cho
TOÀN BỘ ứng viên của họ đó cùng lúc trên đoạn kiểm định:

| họ | số ứng viên | p-value | bác bỏ H0 ở α=0,05 |
|---|---|---|---|
| H2 (motif) | 24 | 0,990 | không |
| H3 (rule-list) | 8 | 0,590 | không |
| H5 (chế độ tự tương quan) | 630 (giới hạn từ 1.890 theo \|z\| thô) | 0,872 | không |
| H6 (HMM) | 6 | **1,000** | không |
| H7 (Matrix Profile) | 36 | 0,972 | không |

Cộng với Giai đoạn 1 (`CHISO_DANHGIA.md` mục 11: bác bỏ ở h=5/h=20 — nhưng đó
LÀ nền đang chạy, không phải một họ quy luật thay thế), **cả năm họ ứng viên
của Giai đoạn 2 đều không thắng nền "chỉ σ̂" có ý nghĩa** khi dùng làm hệ dự
báo toàn diện.

### 6.4 H5 — chế độ tự tương quan làm lớp điều kiện: 0 sống sót

Trục chế độ MỚI (khác σ̂): tam phân vị của tự tương quan lag-1 cuộn 20 phiên
(âm/trung tính/dương), giao với toàn bộ 630 vị từ gốc → 1.890 vị từ × 3 lớp =
5.670 giả thuyết. 1.734/1.890 đủ mẫu, 938 thô p<0,05 (kỳ vọng nhiễu ~260 — vẫn
thổi phồng, đúng như dự đoán khi tăng không gian giả thuyết), nhưng
**0/5.670 sống sót Westfall–Young**. Chế độ tự tương quan không mở khoá thêm
quy luật nào.

### 6.5 H6 — HMM / Markov switching: trạng thái ẩn có thật, nhưng là σ̂ trá hình

`src/run_h6_hmm.py`, `output/h6_hmm.json`. Đây là thuật toán **duy nhất** mà kế
hoạch Pha 1 của HuyH (Week 2, mục A) đòi mà repo chưa từng thử. Mọi "chế độ"
đang có (`SigmaCheDo`, A3, H5) đều là chế độ **quan sát được** — chia theo phân
vị của một đại lượng đã tính được. HMM khác về bản chất: trạng thái **ẩn**, có
ma trận chuyển riêng, phải suy ra từ chuỗi.

**Chặn rò rỉ nhìn trước — chỗ dễ sai nhất.** `hmmlearn.predict()` chạy Viterbi
trên **toàn** chuỗi và `predict_proba()` trả hậu nghiệm **làm trơn**; cả hai đều
để trạng thái tại t phụ thuộc quan sát **sau** t. Dùng làm đặc trưng dự báo thì
mọi kết quả đều vô nghĩa. Nên chỉ dùng `hmmlearn` để **khớp** tham số
(Baum–Welch trên huấn luyện), còn **suy diễn thì tự viết bộ lọc tiến**. Ba tự
kiểm, cái thứ ba là cái quyết định:

| tự kiểm | kết quả |
|---|---|
| khôi phục tham số từ HMM mô phỏng | ĐẠT |
| lọc ≡ làm trơn **tại bước cuối** (đồng nhất thức toán học) | lệch 3·10⁻¹³ |
| **α[t] không đổi khi nối thêm tương lai** | lệch **0,00** |
| *đối chiếu*: hậu nghiệm làm trơn ở cùng vị trí | lệch **0,092** ← phép kiểm có lực |

**Hai lỗi đã bắt trong lúc làm** — cả hai đều thuộc loại "nếu không để ý thì
kết quả trông vẫn hợp lý":

1. **Label switching.** Baum–Welch đánh số trạng thái tuỳ ý, nên "trạng thái 0"
   của EURUSD và của GBPUSD có thể là hai chế độ khác nghĩa nhau. Gộp theo chỉ
   số thô là trộn lẫn. Sửa: sắp lại theo σ tăng dần, để chỉ số luôn cùng nghĩa.
2. **Không có trần độ phủ.** Phễu có sẵn *sàn* (≥100 lần khớp) nhưng không có
   *trần*. Trạng thái nền của HMM phủ **96%** chuỗi, làm phân phối null của
   max|z| gần như suy biến — lấy gần hết mẫu thì xáo trộn cũng chẳng đổi z, nên
   ngay cả |z| = 0,5 cũng "vượt" ngưỡng W-Y. Dấu hiệu lộ ra không thể bỏ qua:
   **số sống sót W-Y (12) lớn hơn số vị từ đạt |z| > 1,96 (8)** — bất khả với
   một hiệu chỉnh bội lành mạnh. Thêm trần `MAX_PHU = 2/3`, chốt theo lý do cấu
   trúc (không gian chính của repo phủ tối đa ~1/3), không ràng buộc họ nào khác.

**Kết quả.** K = 2, 3, 4 chốt trước → 18 giả thuyết sau khi loại trạng thái nền.
HMM tìm được trạng thái biến động cao **có thật và đọc được**: σ = 109 pip so
với 49 pip, kỳ vọng kéo dài 14 phiên so với 102 — đúng phân loại mà kế hoạch mô
tả ("low-volatility sideways" / "high-volatility stress"). Trạng thái đó **có**
liên hệ với lớp: lift 0,672 cho "đi ngang" (|z| = 6,0), 6 vị từ sống sót W-Y.

Nhưng **0/6 còn tin riêng sau đối chứng có điều kiện** (|t| lớn nhất 2,16 <
3,0). Tức trạng thái ẩn của HMM chủ yếu là **khám phá lại chính biến động** mà
HAR đã mô hình hoá tốt hơn. K = 3 và K = 4 còn thoái hoá: các trạng thái phụ có
kỳ vọng kéo dài **1 phiên** và σ tới 4.195 pip — chúng chỉ đang gán mỗi ngày cực
đoan vào một trạng thái riêng, không phải chế độ.

### 6.6 H7 — Matrix Profile: kết quả âm sạch nhất của cả dự án

`src/run_h7_matrixprofile.py`, `output/h7_matrixprofile.json`. H2 mới làm
**proxy** bằng codebook KMeans — nó trả lời "cửa sổ hôm nay thuộc cụm hình dạng
nào", chứ không trả lời được câu hỏi cốt lõi của analog forecasting:

> những lần quá khứ thị trường trông **giống hôm nay nhất**, hôm sau đã xảy ra
> chuyện gì?

H7 làm đúng câu đó, với bốn đặc trưng kế hoạch liệt kê: khoảng cách tới analog
gần nhất, và tỷ lệ kết cục giảm / đi ngang / tăng trong K = 20 analog gần nhất.

**Chặn rò rỉ.** `stumpy.stump()` tự nối trên **toàn** chuỗi nên "láng giềng gần
nhất" của cửa sổ tại t có thể nằm **sau** t. Nên tự viết tìm kiếm nhân quả: ứng
viên t′ phải thoả `t′ + L ≤ t` (không chồng lấn — loại trùng khớp tầm thường
với cửa sổ hôm qua) **và** `y[t′] ≥ 0` (kết cục đã biết tại t). Dùng đồng nhất
thức `‖a−b‖² = 2L − 2a·b` trên cửa sổ đã z-chuẩn hoá nên mỗi hàng chỉ tốn một
phép nhân ma trận–vector. Ba tự kiểm: công thức khớp khoảng cách Euclid trực
tiếp; đặc trưng tại t **không đổi** khi cắt bỏ tương lai (lệch 0,00); và **đối
chiếu với `stumpy.stump`** ở cùng vùng loại trừ — **0,937739 so với 0,937739**.

**Kết quả**: 108 giả thuyết, 36/36 vị từ đủ mẫu, và chỉ **1** giả thuyết đạt
p < 0,05 **thô** — *ít hơn* cả kỳ vọng nhiễu thuần (~5). **0 sống sót W-Y.**

Đây là kết quả âm sạch nhất của cả dự án: analog lịch sử gần như **không mang
thông tin gì** về lớp của phiên kế tiếp. Đáng nói vì analog forecasting là trực
giác lâu đời nhất của phân tích kỹ thuật ("lịch sử lặp lại"), và ở đây nó được
đo bằng đúng công cụ hiện đại của nó, nhân quả, với kiểm soát bội — và không
còn lại gì.

## 6. Sản phẩm nếu tiền đề không đứng — theo đúng mục 10.4

Không có `rules_v1.csv` vì không quy luật nào sống sót. Sản phẩm xuất xưởng là
thứ mục 10.4 đã viết sẵn:

1. **Ba ô chạy trên mục tiêu P**, sinh từ tầng 2 + hiệu chuẩn — đang chạy tại
   `fx-dss.vercel.app`, BSS +0,0074 [+0,0046; +0,0106] trên đoạn kiểm tra.
2. **41 quy luật biến động** đã sống sót kiểm soát bội (2 trong đó mang thông
   tin độc lập với HAR) làm **tầng giải thích**, không làm nguồn dự báo.
3. **Quy luật loại-trừ** từ giai đoạn 0: không mở vị thế theo đà khi σ̂ ở tercile
   cao nhất (Sharpe −0,615, p = 0,001, âm 12/12 ô).
4. **Chương kết quả** là một kết quả âm có giá trị công bố, với con số cụ thể:
   1.890 giả thuyết liệt kê đầy đủ, hệ số thổi phồng 13,5 lần, và chín vị từ
   sống sót hoá ra đều là chính biến số mô hình đã có.

---

## 7. Còn thiếu

| việc | ghi chú |
|---|---|
| mục tiêu R, và h = 5, 20 | cần cho tiêu chí dừng đầy đủ |
| Hansen SPA / White Reality Check | phát biểu "cả họ không thắng nền" |
| Motif (matrix profile) | họ H2, chưa chạy — nhưng nó **không** liệt kê được đầy đủ nên phải xử lý bội khác |
| Tập khoá sổ | **chưa mở**, đúng luật — mở một lần ở cuối |

---

## Kiểm chứng chính cái phễu (05/09/2026)

Con số "0/1.890" tự nó **mơ hồ** giữa hai khả năng khác hẳn nhau: (a) không có
quy luật nào, hay (b) phễu quá chặt nên không bắt được gì. Không phân biệt được
thì nó không phát biểu thành câu gì cả. `src/kiem_pheu.py` phân biệt bằng hai
phép chạy chuẩn của thiết kế thí nghiệm.

Giá trị tới hạn `max|z|` dưới null khối, 1.000 hoán vị: **4,72**.

### Đối chứng âm — xáo trộn khối kết cục, 10 lần

| lần | vượt ngưỡng | qua điều kiện hoá |
|---|---|---|
| 1–10 | **0** | **0** |

Không một dương tính giả nào trên 1.890 giả thuyết, ở cả 10 lần. **Phễu không
rò rỉ.** Nếu bước này ra số dương thì mọi kết quả giai đoạn 2 đều phải vứt.

### Đối chứng dương — tiêm quy luật đã biết, 60 lần mỗi mức

Cách tiêm: chọn một vị từ **thật** có ≥300 lần khớp làm giá đỡ, rồi ở đúng
những hàng nó khớp, đổi kết cục sang lớp đích với xác suất vừa đủ đạt lift mong
muốn. Hàng không khớp giữ nguyên, nên cấu trúc tự tương quan của chuỗi được giữ.

| lift đặt | lift thực | \|z\| trung vị | bắt được | **lực** |
|---|---|---|---|---|
| 1,02 | 1,090 | 2,02 | 1/60 | 2% |
| 1,05 | 1,095 | 1,76 | 0/60 | 0% |
| 1,10 | 1,128 | 3,51 | 2/60 | 3% |
| 1,15 | 1,163 | 4,81 | 5/60 | 8% |
| 1,20 | 1,201 | 6,33 | 24/60 | **40%** |
| 1,35 | 1,346 | 10,76 | 60/60 | **100%** |
| 1,50 | 1,504 | 14,51 | 60/60 | 100% |

*Ghi chú trung thực:* phép tiêm có **sàn ~1,09** — ở nhiều vị từ, tỷ lệ hậu
nghiệm sẵn có đã cao hơn mức đặt, nên hai dòng đầu thực chất đo cùng một mức
với dòng 1,10. Ba dòng đó nhất quán với nhau (0–3%), nên không đổi kết luận.

### Kết luận phải phát biểu lại

**Hiệu ứng nhỏ nhất phát hiện được ở lực 80%: lift = 1,35.**

Nên câu đúng **không phải** *"không tồn tại quy luật nào"*, mà là:

> Phễu này bắt được quy luật có lift ≥ 1,35 với xác suất ≥ 80%, và không tìm
> thấy quy luật nào. Do đó **mọi quy luật mạnh hơn 1,35 đã bị loại trừ** trên
> dữ liệu này. **Quy luật yếu hơn 1,20 thì KHÔNG loại trừ được** — ở mức đó lực
> phát hiện chỉ 40%, và ở 1,15 chỉ còn 8%.

Đây là một hạn chế thật, và nó **không nhỏ**. Lift 1,35 nghĩa là xác suất lớp
đó cao hơn nền 35% — với nền ~1/3 thì là **33% → 45%**, một lợi thế rất lớn
theo chuẩn ngoại hối. Còn một quy luật lift 1,15 (33% → 38%), nếu có thật và
ổn định, vẫn có thể có ý nghĩa kinh tế mà phễu này **không thấy**.

Nguyên nhân là cỡ mẫu: 21.606 hàng phát hiện chia cho 1.890 giả thuyết, sau khi
Westfall–Young đẩy ngưỡng lên `|z| > 4,72` để khống chế sai lầm loại I toàn cục.
Đó là cái giá phải trả cho việc kiểm định bội trung thực — và nó phải được nêu
trong luận văn, chứ không được giấu sau con số 0.

Muốn hạ ngưỡng phát hiện thì có ba đường, đều tốn: thu hẹp không gian giả thuyết
(ít giả thuyết hơn → ngưỡng thấp hơn), thêm cặp/thêm năm dữ liệu, hoặc chuyển
sang khống chế FDR thay vì FWER (chấp nhận vài dương tính giả để đổi lấy lực).

---

## Hạ phễu xuống H1 — 27× dữ liệu, kết quả vẫn 0 (05/09/2026)

Mục 1.1 của `docs/KEHOACH_2026Q4.md`. `src/quyluat_h1.py`.

### Vì sao làm

`src/kiem_pheu.py` đo được: phễu D1 chỉ phát hiện quy luật lift ≥ 1,35 (lực 80%).
Ở 1,20 lực chỉ 40%, ở 1,15 còn 8%. Nên "0/1.890" chỉ loại trừ được quy luật
**mạnh** — nguyên nhân là cỡ mẫu, không phải bản chất thị trường.

| | số quan sát |
|---|---|
| panel D1 | 21.596 |
| **H1** (`data/prices/*_h1.csv`) | **592.343** — ×27,4 |

### Bốn thứ đã xử lý

1. **Mùa vụ trong ngày** — hai tầng: σ̂ khử mùa vụ trước khi ước EWMA rồi gắn lại
   (Andersen–Bollerslev), **và** 23 biến giả giờ trong bộ kiểm soát.
2. **Khối hoán vị 24 thanh** = một ngày, giữ chu kỳ trong ngày.
3. **Cửa chi phí** — cửa thứ năm, chốt trước khi chạy: lợi thế ròng phải dương
   sau spread (trung vị đo được 0,96 pip).
4. Kiểm nhiễu vi cấu trúc ở H4 — chưa chạy, để lần sau.

### Kết quả

| cửa | D1 | **H1** |
|---|---|---|
| không gian giả thuyết | 1.890 | 1.890 |
| thô p<0,05 | 1.186 | **1.683** |
| *(nhiễu thuần kỳ vọng)* | *88* | *94* |
| *(hệ số thổi phồng)* | *13,5×* | ***17,9×*** |
| sống sót Westfall–Young | 9 | **3** |
| **là chính σ̂** | **9/9** | **3/3** |
| không-phải-σ̂ còn tin riêng | **0** | **0** |
| qua cửa chi phí | — | **0** |

**Không một chỉ báo kỹ thuật nào lọt vào, kể cả với 27 lần dữ liệu.** Ba cái sống
sót ở H1 đều là "σ̂ thấp": lift 1,289 cho *đi ngang*, và 0,846 / 0,867 cho
*giảm* / *tăng* — **gần đối xứng và cả hai đều < 1**, tức vẫn không có tín hiệu
hướng.

### Ba lỗi đã bắt được trong quá trình này

Ghi lại vì cả ba đều suýt đưa một "quy luật" không tồn tại vào luận văn.

**1. `du_bao_cuon()` ghi cứng `canh_P`.** Mọi hàng `h*_R_*(cuộn)` trong
`nen3.json` tính bằng dải của mục tiêu P. 9 hàng sai, 0 hàng P bị ảnh hưởng.

**2. Biến kiểm soát đặt sai dạng.** `log σ̂` **tuyến tính** không hấp thụ nổi một
chỉ báo **phân vị** của chính σ̂. Ở H1, "σ̂ thấp" có |t| = 9,68 với kiểm soát
tuyến tính, tụt về **0,74** với biến giả phân vị riêng từng cặp. *(Lần đầu tôi
kiểm bằng phân vị **gộp** sáu cặp và nó vẫn sống — sai, vì σ̂ khác thang giữa các
cặp nên ngưỡng gộp bị chi phối bởi chênh lệch giữa cặp, không phải biến thiên
trong cặp.)*

**3. Vị từ về chính σ̂ không bao giờ nên là ứng viên.** Sau khi sửa (2), phễu D1
lại đẻ ra "σ̂ cao → tăng" với `t = 31,70` và **ghi thẳng vào `rules_v1.csv`**.
Ba lần vá thống kê đều không cứu được:

| vá | t |
|---|---|
| SE thường + kiểm soát tuyến tính | 1,26 |
| SE thường + kiểm soát mềm dẻo | 31,70 |
| chốt trùng tuyến R² ≥ 0,99 | 31,70 *(không kích hoạt — VIF chỉ 2,9)* |
| SE vững theo cụm (cặp × khối) | 31,59 |

Bằng chứng nội tại: `b = −0,6425` cho "σ̂ cao → đi ngang" tức −64 điểm phần trăm,
trong khi lift 0,482 chỉ ứng với −17 điểm — hồi quy xác suất tuyến tính ngoại suy
ra ngoài [0, 1].

Nguyên nhân là **thiết kế**, không phải thống kê. Câu hỏi của giai đoạn 2 là
*"có quy luật nào nói thêm gì **ngoài** một mô hình biến động tốt không"*. Vị từ
"σ̂ cao" **chính là** mô hình đó rời rạc hoá; điều kiện hoá nó lên chính nó là
hỏi *"σ̂ có nói thêm gì ngoài σ̂ không"*. Đã thêm `la_vi_tu_nen()` loại chúng
khỏi không gian quy luật **theo nguyên tắc**, chốt trước khi nhìn kết quả — nhưng
vẫn **báo cáo** chúng ở bảng sống sót, vì *"cả chín cái sống sót đều là chính σ̂"*
tự nó là kết quả trung tâm.

Hai cải tiến kỹ thuật giữ lại vì đúng trong tổng quát: kiểm soát σ̂ mềm dẻo riêng
từng cặp, và SE vững theo cụm.

### Phát biểu được gì

**"Không có quy luật nào"** vẫn chưa nói được — `kiem_pheu.py` mới chỉ đo lực ở
D1. Phải chạy lại nó ở H1 để biết MDES mới. Nhưng có thể nói:

> Với 592.343 quan sát và 1.890 giả thuyết liệt kê đầy đủ, sau kiểm định bội,
> đối chứng có điều kiện, và cửa chi phí — **không một chỉ báo kỹ thuật nào
> mang thông tin vượt trên một mô hình biến động tốt**. Thứ duy nhất sống sót
> là chính mô hình biến động đó.

---

## Lực phát hiện của phễu H1 — MDES rơi từ 1,35 xuống 1,10

`python src/kiem_pheu.py --h1`. Cùng thiết lập với phễu thật: khối 24 thanh, SE
vững theo cụm, và vị từ làm giá đỡ **không được là σ̂** (dùng `la_vi_tu_nen`) —
nếu tiêm tín hiệu vào chính σ̂ thì đang đo lực phát hiện σ̂, không phải lực phát
hiện quy luật.

Giá trị tới hạn max|z| dưới null khối: **5,36** (D1 là 4,72). Ngưỡng **cao hơn**
là đúng: với 513k quan sát, |z| dưới null cũng trải rộng hơn, nên Westfall–Young
phải nâng ngưỡng để giữ nguyên mức khống chế sai lầm loại I toàn cục. Đây là lý
do 27× dữ liệu **không** cho 27× lực.

### Đối chứng âm — 10 lần xáo trộn khối

**0 dương tính giả** cả 10 lần, trên 1.890 giả thuyết. Phễu H1 không rò rỉ.

### Đối chứng dương — 40 lần mỗi mức

| lift đặt | lift thực | \|z\| trung vị | bắt được | **lực** |
|---|---|---|---|---|
| 1,02 | 1,033 | 2,47 | 6/40 | 15% |
| 1,05 | 1,082 | 8,12 | 25/40 | 62% |
| **1,10** | 1,126 | 14,54 | **37/40** | **92%** |
| 1,15 | 1,153 | 22,61 | 38/40 | 95% |
| 1,20 | 1,202 | 30,07 | 40/40 | 100% |
| 1,35 | 1,350 | 51,57 | 40/40 | 100% |
| 1,50 | 1,500 | 73,87 | 40/40 | 100% |

**Hiệu ứng nhỏ nhất phát hiện được (lực 80%): lift = 1,10.**

| | D1 | **H1** |
|---|---|---|
| quan sát | 21.596 | 592.343 |
| giá trị tới hạn | 4,72 | 5,36 |
| **MDES (lực 80%)** | **1,35** | **1,10** |
| lực ở lift 1,20 | 40% | **100%** |
| lực ở lift 1,15 | 8% | **95%** |
| lực ở lift 1,05 | 0% | **62%** |
| dương tính giả / 1.890 | 0,0 | 0,0 |

### Phát biểu được gì — bản mạnh nhất hiện có

> Trên **592.343 quan sát** và **1.890 giả thuyết liệt kê đầy đủ**, phễu này phát
> hiện được quy luật có lift **≥ 1,10** với xác suất **≥ 80%** (và ≥ 1,20 với xác
> suất 100%), trong khi cho **0 dương tính giả** trên nhiễu thuần. Sau kiểm định
> bội, đối chứng có điều kiện, và cửa chi phí — **không một chỉ báo kỹ thuật nào
> sống sót**. Thứ duy nhất sống sót là chính mô hình biến động.
>
> Do đó: **không tồn tại quy luật kỹ thuật nào có lợi thế từ 10% trở lên** trên
> bộ dữ liệu này. Quy luật yếu hơn 1,05 thì vẫn **không** loại trừ được — ở mức
> đó lực chỉ 62%.

Lift 1,10 nghĩa là xác suất lớp cao hơn nền 10% — với nền ~1/3 thì là **33% →
37%**. Đó là ngưỡng đủ thấp để phát biểu có sức nặng: lợi thế nhỏ hơn thế, sau
khi trừ spread 0,96 pip trên biên độ H1 ~4,2 pip, gần như chắc chắn không còn ý
nghĩa kinh tế.
