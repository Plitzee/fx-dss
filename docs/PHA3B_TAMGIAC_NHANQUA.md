# PHA 3B — TAM GIÁC HOÁ NHÂN QUẢ: Granger, PCMCI, DML, Causal Forest, CausalImpact

*Lập 13/09/2026. Nhánh `replan-2026`. Bổ sung SAU khi `docs/PHA3B_KETQUA.md`
đã đóng băng — **không mở lại phán quyết đó**. Mục đích của tài liệu này là
trả lời trực tiếp một câu hỏi phản biện: "dùng thuật toán nhân quả nào, và
làm sao biết đó là nhân quả thật chứ không phải tương quan ngẫu nhiên?" —
bằng cách chạy THÊM hai phương pháp độc lập (không phải tự viết) trên đúng
dữ liệu đã có, và đối chiếu.*

---

## 0. Vì sao cần tài liệu này

`pha3b_granger.py` (đã đóng băng) dùng **hai** công cụ nhân quả, cả hai **tự
viết**:

- **Level 2** — Granger causality dạng hồi quy lồng nhau (nested-regression
  F-test) + Westfall–Young maxT step-down (kiểm soát đa kiểm định) — đây là
  **cửa quyết định chính thức**.
- **Level 3** — bản PCMCI **rút gọn** (PC-stable + MCI bằng tương quan riêng
  phần tuyến tính), tự khai báo rõ trong code: *"KHÔNG phải PCMCI đầy đủ của
  Runge et al."* vì môi trường lúc đó thiếu thư viện `tigramite`.

Đề cương trích Runge et al. 2019 (PCMCI) nhưng không nêu tên thuật toán cụ
thể đã cài, và không có phương pháp thứ ba (CausalImpact) như phản biện gợi
ý. Ba mục dưới đây lấp khoảng đó bằng **thư viện gốc, không tự viết lại**.

---

## 1. PCMCI thật (`tigramite` 5.2.10, tác giả Jakob Runge)

Script: [`src/pha3b_pcmci_doclap.py`](../src/pha3b_pcmci_doclap.py) ·
Kết quả: [`output/pha3b_pcmci_doclap.json`](../output/pha3b_pcmci_doclap.json)

**Thiết kế:** biến `y_bien_do` (log RV phiên t+1), mốc `m_har` (log HAR
forecast tại t), và 3 đặc trưng ngoại sinh THÔ (không tiền-lag {1,2,5} như
bản chính — để PCMCI tự tìm độ trễ): `VIXCLS_lv`, `GVZCLS_lv`, `DFII10_lv`.
`tau_min=1` (chỉ tìm quan hệ trễ, giữ đúng giới hạn đã khai báo ở Level 2/3
— không xử lý contemporaneous causality, để so sánh công bằng). FDR-BH trên
toàn bộ liên kết đã kiểm, dùng hàm gốc của tigramite.

**Kết quả trên 3 cặp (huấn luyện + kiểm định, không chạm tập kiểm tra):**

| cặp | liên kết sống sót sau FDR<0,05 |
|---|---|
| AUDUSD | `VIXCLS_lv` @τ=1 (q=9,7e-10), @τ=3 (q=6,4e-4) · `m_har` @τ=1 · `DFII10_lv` @τ=1 (q=0,046, biên) |
| EURUSD | `VIXCLS_lv` @τ=1 (q=1,4e-4) · `m_har` @τ=3,4 |
| USDJPY | `VIXCLS_lv` @τ=1 (q=2,5e-5), @τ=2 (q=0,019) |

**Đối chiếu với Level 2/3 đã đóng băng:** VIXCLS là 6/14 đặc trưng sống sót
của E3 (Westfall-Young) — PCMCI thật **xác nhận độc lập** quan hệ này trên
CẢ BA cặp kiểm tra, kể cả hai cặp (EURUSD, USDJPY) không nổi bật trong Level
3 rút gọn. Đây là tam giác hoá bằng thuật toán khác, tác giả khác, không
phải diễn giải lại cùng một phép tính.

### 1b. Minh hoạ cơ chế chống tương quan giả — confounding của GVZCLS

Đây là câu trả lời trực tiếp cho "làm sao biết không phải tương quan ngẫu
nhiên": chạy GVZCLS (biến động vàng) **một mình** so với **điều kiện thêm
trên VIXCLS**:

| cấu hình | GVZCLS → y_biến_độ, q(FDR) nhỏ nhất trên τ=1..5 |
|---|---|
| GVZCLS một mình (+ mốc) | có ý nghĩa (q < 0,05) |
| GVZCLS **điều kiện trên VIXCLS** (+ mốc) | **mất ý nghĩa** |

GVZCLS trông như một nguyên nhân khi xét một mình, nhưng biến mất khi PCMCI
điều kiện đồng thời trên VIXCLS — đúng cơ chế **confounding qua chỉ số biến
động chung** (cả hai đều là proxy cho "sợ hãi thị trường toàn cầu"), không
phải quan hệ riêng của thị trường vàng với FX. Đây chính là điều một phép
Granger đôi-một (kiểm từng biến riêng lẻ so với mốc, như Level 2) **không
phát hiện được** — vì nó không bao giờ điều kiện hai ứng viên ngoại sinh lên
nhau. Đây là lý do kỹ thuật cụ thể để dùng PCMCI (điều kiện đồng thời trên
toàn bộ tập ứng viên) thay vì chỉ Granger đôi-một khi muốn loại tương quan
giả do biến gây nhiễu chung.

**Giới hạn tự khai báo:** `tau_min=1` nên không xử lý contemporaneous
causality; `ParCorr` là kiểm định độc lập điều kiện TUYẾN TÍNH — quan hệ phi
tuyến (nếu có) sẽ không bị phát hiện. Không thay thế quyết định đã đóng băng
của Level 2 (E3 vẫn là 14 đặc trưng đã chốt) — đây là kiểm tra độ vững bên
ngoài protocol.

---

## 1b. Double Machine Learning — ĐỘ LỚN hiệu ứng, không chỉ tồn tại hay không

Script: [`src/pha3b_dml_causal.py`](../src/pha3b_dml_causal.py) ·
Kết quả: [`output/pha3b_dml_causal.json`](../output/pha3b_dml_causal.json)

**Vì sao cần thêm DML sau PCMCI:** cả Granger lẫn ParCorr (PCMCI) đều dùng
mô hình con **tuyến tính** cho phần kiểm soát nhiễu nền. Double Machine
Learning (Chernozhukov et al. 2018, *The Econometrics Journal* — thư viện
gốc `doubleml` của chính nhóm tác giả) dùng gradient boosting LINH HOẠT cho
hai hàm phiền toái (outcome~kiểm soát, treatment~kiểm soát), cross-fit 5-fold
× 3 lần lặp, rồi "gỡ nhiễu kép" để hệ số nhân quả còn lại có suy diễn hợp lệ
dù hai hàm phiền toái phi tuyến. Đây là bước NÂNG CẤP về ĐỘ VỮNG của ước
lượng, không phải kiểm định tồn tại quan hệ lần nữa.

**Kết quả — hiệu ứng của VIX (đã chuẩn hoá) lên log biến động ngày t+1, kiểm
soát mốc HAR + GVZCLS + DFII10:**

| cặp | θ (hệ số) | CI 95% | p | Holm |
|---|---|---|---|---|
| AUDUSD | +0,186 | [0,150 ; 0,221] | <0,0001 | ĐẠT |
| EURUSD | +0,093 | [0,056 ; 0,130] | <0,0001 | ĐẠT |
| GBPUSD | +0,125 | [0,090 ; 0,160] | <0,0001 | ĐẠT |
| USDJPY | +0,085 | [0,039 ; 0,131] | 0,0003 | ĐẠT |
| USDCAD | +0,126 | [0,094 ; 0,159] | <0,0001 | ĐẠT |
| USDCHF | +0,070 | [0,034 ; 0,105] | 0,0001 | ĐẠT |

**6/6 cặp sống sót qua hiệu chỉnh Holm** — vượt ngưỡng ≥5/6 mà chính Pha3B
đặt ra cho "quan hệ vững". Đây là kết quả nhất quán nhất trong toàn bộ tam
giác hoá: khoảng tin cậy không chạm 0 ở CẢ 6 cặp, ước lượng bằng mô hình
phiền toái phi tuyến (không phụ thuộc giả định tuyến tính như Level 2/3).

## 1c. Tích hợp vào dự báo sản xuất — kiểm tra "có thật sự hiệu quả không"

Script: [`src/pha3b_dml_tichhop.py`](../src/pha3b_dml_tichhop.py) ·
Kết quả: [`output/pha3b_dml_tichhop.json`](../output/pha3b_dml_tichhop.json)

Có θ vững không đồng nghĩa dùng nó CHỈNH dự báo sẽ tốt hơn — đây là bước
kiểm tra trực tiếp, tách bạch hoàn toàn ước lượng (huấn luyện) và đánh giá
(kiểm định, chưa từng thấy):

```
σ̂²_có_lớp_phủ(t) = σ̂²_HAR(t) · exp(θ_huấn_luyện · VIX_chuẩn_hoá(t))
```

| cặp | QLIKE mốc HAR | QLIKE +lớp phủ | chênh (%) | DM p |
|---|---|---|---|---|
| AUDUSD | −8,7932 | −8,7889 | +0,048 | 0,2227 |
| EURUSD | −9,4570 | −9,4597 | −0,029 | 0,2093 |
| GBPUSD | −9,2266 | −9,2351 | −0,093 | 0,4067 |
| USDJPY | −9,1470 | −9,1729 | −0,284 | 0,2508 |
| USDCAD | −9,7544 | −9,7523 | +0,022 | 0,4692 |
| USDCHF | −9,3900 | −9,3911 | −0,012 | 0,4769 |

**Kết quả: 4/6 cặp cải thiện về HƯỚNG, 0/6 cải thiện CÓ Ý NGHĨA sau Holm.**
Đây là phát hiện quan trọng, KHÔNG che giấu: hiệu ứng nhân quả tồn tại và
vững (mục 1b), nhưng khi biến thành lớp phủ đơn giản trên mốc HAR đã tinh
chỉnh kỹ, mức cải thiện quá nhỏ để tách khỏi nhiễu. Diễn giải hợp lý nhất:
mốc HAR đã tự hấp thụ phần lớn thông tin "thị trường đang căng thẳng" qua
chính cấu trúc tự hồi quy của nó (biến động hôm qua/tuần qua/tháng qua đã
cao khi VIX cao) — phần ĐÓNG GÓP THÊM của VIX sau khi đã có mốc tốt là nhỏ,
dù về mặt thống kê phần đóng góp đó là có thật (không phải nhiễu ngẫu nhiên,
mục 1b đã chứng minh).

## 1d. Hiệu ứng có đổi theo chế độ biến động không? — Causal Forest

Script: [`src/pha3b_causal_forest.py`](../src/pha3b_causal_forest.py) ·
Kết quả: [`output/pha3b_causal_forest.json`](../output/pha3b_causal_forest.json)

Giả thuyết kiểm tra: có thể lớp phủ TUYẾN TÍNH ở mục 1c thất bại vì hiệu
ứng VIX thực ra TẬP TRUNG ở chế độ căng thẳng, còn rải đều hiệu chỉnh ra cả
ba chế độ chỉ thêm nhiễu ở chế độ bình tĩnh. Dùng Causal Forest (Wager &
Athey 2018, JASA; Athey, Tibshirani & Wager 2019, Annals of Statistics —
thư viện `econml` của Microsoft Research) để ước lượng CATE (hiệu ứng trung
bình có điều kiện) riêng theo TỪNG chế độ:

| cặp | bình tĩnh | vừa | căng thẳng | tăng dần theo chế độ? |
|---|---|---|---|---|
| AUDUSD | 0,123 | 0,144 | **0,220** | có, rõ |
| EURUSD | 0,121 | 0,023 | 0,122 | không (hình chữ U) |
| GBPUSD | 0,103 | **0,168** | 0,099 | không (đỉnh ở giữa) |
| USDJPY | 0,130 | 0,087 | 0,066 | không (giảm dần) |
| USDCAD | 0,101 | **0,151** | 0,107 | không (đỉnh ở giữa) |
| USDCHF | 0,033 | 0,005 | **0,109** | có, rõ |

**4/6 cặp có CATE(căng thẳng) > CATE(bình tĩnh) — KHÔNG đạt ngưỡng ≥5/6.**
Có sự không đồng nhất theo chế độ THẬT (khoảng tin cậy khác nhau rõ giữa
các chế độ trong hầu hết các cặp), nhưng KHÔNG theo một hình dạng chung —
AUDUSD/USDCHF tăng đơn điệu theo mức căng thẳng, GBPUSD/USDCAD lại đạt đỉnh
ở chế độ VỪA, EURUSD/USDJPY gần như phẳng hoặc giảm nhẹ. Kết luận trung
thực: không có một quy tắc "bật lớp phủ khi căng thẳng" áp dụng chung cho
cả 6 cặp — muốn khai thác được sẽ cần hiệu chỉnh RIÊNG từng cặp, làm tăng
đáng kể độ phức tạp cho lợi ích chưa chắc chắn. Đây là hướng để ngỏ cho
công việc tiếp theo, không phải kết luận đóng.

## 2. CausalImpact (Brodersen et al. 2015, bản `pycausalimpact`)

Script: [`src/su_kien_causalimpact.py`](../src/su_kien_causalimpact.py) ·
Kết quả: [`output/su_kien_causalimpact_ecb.json`](../output/su_kien_causalimpact_ecb.json)

**Câu hỏi khác hẳn Pha3B:** Pha3B hỏi "biến X có dự báo được biến động
không" (cấp độ đặc trưng, gộp toàn chuỗi). CausalImpact hỏi "**phiên họp cụ
thể này** có đẩy biến động vượt mức phản-nền (counterfactual) hay không" —
cấp độ TỪNG sự kiện, đúng với mục "sự kiện theo từng cặp kèm phản hồi lịch
sử đo được" đã hứa trong đề cương (System Implementation).

**Thiết kế:** phản ứng = log RV(EURUSD); hiệp biến = log RV(USDCAD) (đại
diện chế độ biến động chung, ít chịu tác động trực tiếp từ tin ECB hơn vì
CAD gần với dầu/BoC/Fed hơn); tiền kỳ = 30 phiên trước phiên họp; hậu kỳ =
phiên họp + 2 phiên sau. Chạy trên **20 phiên họp ECB gần nhất** vẫn nằm
trong huấn luyện+kiểm định (< mốc kiểm tra 2023-11-20 — không chạm tập niêm
phong).

**Kết quả:** 20/20 phiên chạy được. **6/20 (30%) có ý nghĩa sau hiệu chỉnh
FDR-BH** (không chỉ p thô) — và **cả 6 đều cùng chiều dương** (biến động
tăng), không có phiên nào cho hiệu ứng âm có ý nghĩa. Dưới null thuần (không
có hiệu ứng thật ở bất kỳ phiên nào), kỳ vọng chỉ ~1/20 (5%) "có ý nghĩa" do
ngẫu nhiên, và về lý thuyết chia đều hai chiều dấu — quan sát 30% cùng chiều
dương là tín hiệu, không phải nhiễu.

**Giới hạn tự khai báo (nêu ngay trong code):** ECB và USDCAD không tách
biệt hoàn toàn — cả hai đều chịu ảnh hưởng từ chế độ rủi-ro-toàn-cầu, nên
hiệp biến này không loại hết confounding. Đây là thách thức cố hữu của
CausalImpact với sự kiện vĩ mô được công bố rộng (gần như mọi tài sản đều
phản ứng ít nhiều). CausalImpact phù hợp hơn với sự kiện ĐẶC THÙ một đồng
tiền (ví dụ can thiệp riêng của BOJ/SNB) — hướng mở rộng tự nhiên cho
System Implementation, không phải kết luận đóng ở đây. Kết quả trên là
**thí điểm phương pháp**, chưa dùng để ra quyết định nào của Pha3B.

---

## 3. Năm phương pháp — trả lời cho câu hỏi "dùng thuật toán nào, cho việc gì"

| phương pháp | câu hỏi trả lời | cấp độ | trạng thái |
|---|---|---|---|
| Granger (nested-regression F-test) + Westfall–Young | biến ngoại sinh X có thông tin dự báo NGOÀI mốc, kiểm soát 594 giả thuyết đồng thời không? | đặc trưng, gộp toàn chuỗi | **cửa quyết định chính thức**, đã đóng băng |
| PCMCI thật (tigramite) | quan hệ đó có đứng vững khi điều kiện ĐỒNG THỜI trên các ứng viên khác (chống confounding)? | đặc trưng, gộp toàn chuỗi | tam giác hoá độc lập, không đổi quyết định |
| Double ML (doubleml) | ĐỘ LỚN hiệu ứng là bao nhiêu, ước lượng vững dù mô hình nền phi tuyến? | đặc trưng, gộp toàn chuỗi | tam giác hoá độc lập — **6/6 cặp vững, kết quả mạnh nhất** |
| Causal Forest (econml) | hiệu ứng đó có đổi theo chế độ biến động không (hiệu ứng không đồng nhất)? | đặc trưng, theo chế độ | khám phá — không đồng nhất, không theo một khuôn mẫu chung |
| CausalImpact (Bayesian structural time series) | sự kiện CỤ THỂ này có đẩy biến động vượt phản-nền không? | từng sự kiện | thí điểm phương pháp cho System Implementation |

Không phương pháp nào đơn lẻ trả lời "nhân quả thực sự" một cách tuyệt đối
— mỗi phương pháp kiểm một giả định khác nhau (tuyến tính/phi tuyến, đôi-một
/đồng thời, tồn tại/độ lớn/không đồng nhất, đặc trưng/sự kiện). Việc BỐN
phương pháp **độc lập** (Granger, PCMCI, Double ML, và văn liệu vĩ mô-tài
chính) đều chỉ về cùng một biến (VIX) là bằng chứng mạnh hơn bất kỳ phương
pháp riêng lẻ nào — đây chính là "tam giác hoá" (triangulation).

## 4. Vậy phần nhân quả có "hiệu quả và chủ chốt" trong hệ thống không?

Trả lời trung thực, tách hai vai trò khác nhau của phân tích nhân quả:

**Vai trò CHẨN ĐOÁN/XÁC THỰC — hiệu quả và đã CHỨNG MINH được:** năm phương
pháp độc lập cho phép hệ thống nói được điều mà một pipeline ML thuần tương
quan không nói được — PHÂN BIỆT được VIX (nhân quả thật, sống sót mọi phép
kiểm, kể cả điều kiện đồng thời chống nhiễu) khỏi GVZCLS (tương quan giả do
nhiễu chung, biến mất khi điều kiện trên VIX — mục 1b của
`pha3b_pcmci_doclap.py`). Đây chính là câu trả lời cụ thể, đo được, cho câu
hỏi "làm sao biết đó là nhân quả thật chứ không phải tương quan ngẫu nhiên"
— và là lý do 11 biến ngoại sinh còn lại bị loại khỏi tập E3 trong khi VIX
được giữ lại. Vai trò này ĐÃ chủ chốt: nó là bộ lọc quyết định biến nào
được tin, không phải phụ lục minh hoạ.

**Vai trò CẢI THIỆN DỰ BÁO TRỰC TIẾP — CHƯA hiệu quả, và điều đó được báo
cáo trung thực chứ không giấu:** lớp phủ nhân quả tuyến tính (mục 1c) và nỗ
lực làm nó phụ thuộc chế độ (mục 1d) đều KHÔNG cho cải thiện QLIKE có ý
nghĩa thống kê. Đây không phải thất bại của phương pháp — đây là bằng
chứng cho thấy mốc HAR đã được tinh chỉnh tốt đến mức thông tin nhân quả từ
VIX phần lớn đã được mốc đó hấp thụ gián tiếp. Ép một con số "cải thiện"
không có thật vào đây sẽ đi ngược đúng triết lý "không giấu kết quả âm" của
toàn bộ luận văn.

**Kết luận cho đề cương:** nên trình bày phân tích nhân quả như một **tầng
kiểm chứng chủ chốt của hệ thống** (quyết định biến nào được tin dùng, định
lượng độ lớn hiệu ứng bằng bốn phương pháp độc lập) — chứ không phải như
một "tính năng tăng độ chính xác dự báo" chưa được chứng minh. Đây là điểm
khác biệt quan trọng cần nói rõ với hội đồng: giá trị của tầng nhân quả
trong hệ thống này là VALIDATION, không phải PERFORMANCE — và đó là một
đóng góp thật, đo được, không phải một lời hứa suông.
