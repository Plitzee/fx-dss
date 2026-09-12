# PHA 3B — KẾT QUẢ VÀ BIÊN BẢN ĐÓNG BĂNG (PHASE3_CAUSALITY_RESULT)

*Lập 12/09/2026. Nhánh `replan-2026`.
Theo `03_PHASE_3_CAUSALITY_AWARE.md` Week 4 (sealed validation + freeze).
Tiêu chí, biến, và phán quyết đã **chốt trước** ở `docs/PHA3B_TIEUCHI.md`
(commit `e88ae66`) — commit đó nằm TRƯỚC mọi commit chứa số liệu dưới đây.*

---

## 0. Phán quyết một dòng

> **Không tầng thông tin ngoại sinh nào cải thiện được mốc đã đóng băng. Nhưng
> lọc theo nhân quả đánh bại chọn đặc trưng thông thường một cách có ý nghĩa
> thống kê (−3,07%, DM p = 0,0010) — giá trị của nó là PHÒNG THỦ, không phải
> tấn công.**

Đây là **phán quyết tách đôi**, giống tiền lệ `PHA2_KETQUA.md` mục 3. Bằng
chứng không cho một nhãn duy nhất, và ép nó thành một sẽ là bóp méo — chi tiết
ở mục 6.

---

## 1. Đóng băng — những gì KHÔNG được đổi nữa

| hạng mục | giá trị chốt |
|---|---|
| **Biến ngoại sinh** | 11 biến / 6 họ, `docs/PHA3B_TIEUCHI.md` mục 2a |
| **Bộ dữ liệu** | `data/ngoai_sinh/chuoi_ngay.csv` — 9 chuỗi ngày FRED, 4.420 ngày, 2009-01 → 2025-12 |
| **Ngữ nghĩa dấu thời gian** | `data/ngoai_sinh/ngu_nghia.csv` — 38.321 dòng × 9 cột |
| **Biến đổi / độ trễ** | 6 dạng (`lv d1 ad1 d5 ad5 rv20`) × lag {1, 2, 5} → **186 đặc trưng** |
| **Không gian giả thuyết** | **594** (11 biến × 3 lag × 6 cặp × 3 trục) — khai báo trước, liệt kê đầy đủ |
| **Kiểm soát bội** | Westfall–Young maxT **từng bước xuống**, null hoán vị **khối 5 phiên**, 1.000 hoán vị |
| **Mốc B0** | HAR sản xuất `V2.du_bao_san_xuat` — QLIKE kiểm định **0,1569**, kiểm tra **0,1872** (khớp chính xác `PHA3_TIEUCHI.md` Phụ lục A) |
| **Chia dữ liệu** | `split.py`, không đổi từ Giai đoạn 1 |
| **Quy tắc chấm** | khớp huấn luyện · chọn kiểm định · chấm **một lần** kiểm tra |

Đếm vào `KHOA_SO.md`: **594 giả thuyết nhân quả** + **5 cấu hình** (B0/E1/E2/E3/E4).

---

## 2. WEEK 1 — Dữ liệu và ngữ nghĩa dấu thời gian

### 2a. Đã lấy gì

9 chuỗi thị trường theo **ngày** từ FRED (không phải tháng — đây là điểm khác
căn bản với `PHA3_TIEUCHI.md`):

| chuỗi | họ | n | lịch công bố (FRED) |
|---|---|---|---|
| `DGS2` `DGS10` `DFII10` | A · lợi suất TPCP Mỹ | 4.251 | H.15 Selected Interest Rates |
| `T10Y2Y` | A · độ dốc đường cong | 4.251 | Interest Rate Spreads |
| `VIXCLS` `GVZCLS` `OVXCLS` | B · ẩn ý biến động | 4.278–4.303 | CBOE Market Statistics |
| `DCOILBRENTEU` | C · hàng hoá | 4.300 | Spot Prices |
| `NIKKEI225` | D · cổ phiếu | 4.158 | Nikkei Indexes |

Cộng họ E (chênh lãi suất chính sách, tháng) và họ F (lịch công bố phi NHTW).

### 2b. Hai phép tự kiểm bắt buộc — **cả hai ĐẠT**

| phép kiểm | ngưỡng chốt trước | kết quả |
|---|---|---|
| **Cắt tương lai** — đặc trưng trước mốc 2023-01-01 không được đổi khi cắt bỏ mọi dữ liệu sau mốc | 0 giá trị đổi | **0 / 3.148.236** ✓ |
| **Hiệu đính (ALFRED vintage @ 2024-06-28)** | ≤ 0,5% số quan sát | DGS10 **0/3.875** · VIXCLS **4/3.914 (0,102%)** · Brent **0/3.914** ✓ |

Ghi nhận trung thực: **VIXCLS CÓ hiệu đính** — 4 giá trị được CBOE sửa lại sau
khi công bố. Nhỏ, dưới ngưỡng, nhưng nó chứng minh vì sao phép kiểm này bắt
buộc: giả định *"chuỗi thị trường không bao giờ hiệu đính"* là **sai**, và
biên bản đã đòi phải đo thay vì khẳng định suông (mục 3c).

### 2c. Ba thứ bị loại — lý do chốt trước, nhắc lại để không ai hỏi lại

| loại bỏ | lý do |
|---|---|
| `DTWEXBGS` (chỉ số USD rộng) | **chứa chính 6 cặp đích** → không phải ngoại sinh |
| `EVZCLS` (ẩn ý biến động EUR/USD) | **ngừng công bố 03/2025** → khuyết 9 tháng cuối đoạn kiểm tra. Đây là biến **hứa hẹn nhất** mà ta không dùng được — xem mục 8 |
| bộ `TSF/data/raw/` (Yahoo) | xuất xứ không truy được · chỉ từ 2015 · phủ tới 08/2026 → **chồng tập khoá sổ** |

---

## 3. WEEK 2 — Phát hiện quan hệ dẫn báo: **KẾT QUẢ DƯƠNG ĐẦU TIÊN**

### 3a. Bốn cửa, và hệ số thổi phồng đo được

| cửa | số sống sót / 594 | ghi chú |
|---|---|---|
| p tiệm cận thô < 0,05 | **214** | kỳ vọng dưới nhiễu thuần ≈ 30 → **hệ số thổi phồng 7,1 lần** |
| p hoán vị khối < 0,05 | 130 | chưa hiệu chỉnh bội |
| FDR-BH q < 0,05 | 51 | tham chiếu, **không** dùng làm cửa |
| **Westfall–Young p < 0,05** | **14** | **cửa quyết định** |
| **W-Y ∩ màn lọc độ vững** | **14** | — |

Hệ số thổi phồng **7,1 lần** là con số quan trọng cho luận văn: nó nói rằng
nếu làm đúng theo roadmap gốc (không có kiểm soát bội — lỗ hổng 2.1 mà
`NHANXET_ROADMAP.md` chỉ ra), ta sẽ "tìm ra" **214 quan hệ nhân quả** và
**200 trong số đó là giả**.

### 3b. 14 quan hệ sống sót — **toàn bộ nằm trên trục BIÊN ĐỘ**

| trục | biến | lag | số cặp | F lớn nhất | p W-Y |
|---|---|---|---|---|---|
| biên độ | `su_kien_phi_nhtw` | 1 | **6/6** | **67,52** | **0,0030** |
| biên độ | `su_kien_phi_nhtw` | 2 | 5 | 39,62 | 0,0070 |
| biên độ | `su_kien_phi_nhtw` | 5 | 2 | 22,32 | 0,0140 |
| biên độ | `VIXCLS` | 1 | 1 | 15,71 | 0,0420 |

**Trục hướng: 0/198 sống sót** (p W-Y tốt nhất = **1,000**).
**Trục rủi ro: 0/198 sống sót** (p W-Y tốt nhất = 0,857, dù p hoán vị thô
= 0,001).

Đây là lần đầu tiên trong toàn dự án có quan hệ vượt qua Westfall–Young. Và
nó trả lời **RQ9 / RQ7 của HuyH** dứt khoát: **thông tin ngoại sinh chỉ có giá
trị đo được trên trục BIÊN ĐỘ** — đúng cùng một trục mà Pha 1 và Pha 2 đã kết
luận, bằng một tầng thông tin hoàn toàn khác.

### 3c. Độ vững — ngưỡng chốt trước ≥5/6 cặp, ≥4/5 năm

| tổ hợp | cặp cùng dấu | năm cùng dấu | đạt |
|---|---|---|---|
| `su_kien_phi_nhtw` lag 1 | **6/6** | **12/12** | ✓ |
| `su_kien_phi_nhtw` lag 2 | 6/6 | 12/12 | ✓ |
| `su_kien_phi_nhtw` lag 5 | 6/6 | 11/12 | ✓ |
| `VIXCLS` lag 1 | 6/6 | 11/12 | ✓ |

**Dấu là ÂM** trên `sk_ngay_L1` (hệ số −0,117): *phiên sau ngày công bố vĩ mô
có RV **thấp hơn** mức HAR ngoại suy*. Đây đúng là cơ chế mà `PHA2_KETQUA.md`
mục 3b đã đề ra cho S4 rồi bị mục 3d bác bỏ — nay được xác nhận trên **toàn bộ
phiên**, 6 cặp, 12 năm, thay vì chỉ 129 kỳ FOMC.

### 3d. Level 1 và Level 3

**Level 1** (hữu ích dự báo, gộp cặp, điều kiện trên mốc):

| trục | mạnh nhất | F | p thô | R² riêng |
|---|---|---|---|---|
| biên độ | `DGS2` lag 2 | 22,92 | <1e−5 | 0,746% |
| hướng | `su_kien_phi_nhtw` lag 2 | 4,56 | 0,0034 | 0,075% |
| rủi ro | `su_kien_phi_nhtw` lag 1 | 25,46 | <1e−5 | 0,416% |

**Level 3 — PCMCI rút gọn** (PC-stable + MCI tuyến tính, tự viết; `tigramite`
không có trong môi trường). **Khai báo thẳng: đây là bản rút gọn tuyến tính,
không phải PCMCI đầy đủ** — HuyH Week 2 ghi rõ nó không bắt buộc.

Giữ lại 2–4 cha mỗi cặp. Đáng chú ý là chúng **khớp lý do kinh tế đã khai báo
trước**: `DCOILBRENTEU → USDCAD` (r = −0,080), `DGS10 → USDJPY` (+0,077),
`VIXCLS → GBPUSD/AUDUSD` (+0,130/+0,120), `GVZCLS → USDCHF` (+0,083).

---

## 4. WEEK 3 — Ablation: **không cấu hình nào thắng mốc**

### 4a. Trục biên độ (QLIKE)

| cấu hình | #đt | QLIKE kiểm định | so B0 | **QLIKE kiểm tra** | so B0 | DM p |
|---|---|---|---|---|---|---|
| **B0** mốc | 0 | 0,1569 | — | **0,1872** | — | — |
| **E1** ném hết vào | 171 | 0,1995 | +27,1% | **0,2245** | **+19,9%** | <0,0001 |
| **E2** chọn thường | 15 | **0,1473** | **−6,2%** | **0,1947** | **+4,0%** | 0,0324 |
| **E3** lọc nhân quả | 15 | 0,1504 | −4,2% | **0,1887** | +0,8% | 0,7462 |
| **E4** E3 × chế độ | 45 | 0,1510 | −3,8% | 0,1876 | +0,2% | 0,9377 |

*(âm = tốt hơn mốc)*

### 4b. ★ Câu hỏi trung tâm của Pha 3B — và câu trả lời

> *"Does causal filtering help beyond ordinary feature selection?"* (HuyH Week 3)

**E3 so E2 trên kiểm tra: −3,07% · DM p = 0,0010.**

**CÓ.** Lọc theo nhân quả tốt hơn chọn đặc trưng thông thường, có ý nghĩa
thống kê, trên đoạn kiểm tra chưa dùng để chọn gì.

Cách đọc sắc hơn — **độ trôi từ kiểm định sang kiểm tra**:

| | kiểm định | kiểm tra | **độ trôi** |
|---|---|---|---|
| E2 chọn thường | −6,15% | +3,99% | **10,14 điểm** |
| E3 lọc nhân quả | −4,19% | +0,80% | **4,99 điểm** |

Lọc nhân quả **giảm một nửa** mức sụp đổ ngoài mẫu. Nó không tìm thêm tín
hiệu — nó **ngăn ta phá hỏng mốc bằng đặc trưng giả**.

### 4c. Vì sao — cơ chế đã đo, không phải suy đoán

E2 và E3 chọn gần như hai tập rời nhau (**giao chỉ 4/15**):

| | E2 chọn gì | E3 chọn gì |
|---|---|---|
| chi phối bởi | `DGS2`, `DGS10`, `DFII10` (9/15 cột) | `VIXCLS` (6) + lịch công bố (9) |

Đo độ trôi phân phối của chính các chuỗi, đơn vị = độ lệch chuẩn của đoạn
huấn luyện:

| chuỗi | TB huấn luyện | TB kiểm tra | \|lệch\|/sd |
|---|---|---|---|
| `DGS2` | 0,87 | 4,12 | **4,54** |
| `NIKKEI225` | 16.608 | 39.702 | **3,84** |
| `DGS10` | 2,29 | 4,24 | **2,63** |
| `DFII10` | 0,35 | 1,95 | **2,22** |
| `T10Y2Y` | 1,43 | 0,13 | **1,61** |
| `VIXCLS` | 19,16 | 17,00 | 0,26 |
| `OVXCLS` / `GVZCLS` / `DCOILBRENTEU` | — | — | 0,14 / 0,10 / **0,03** |

Đoạn kiểm tra (2023–2025, lãi suất 4–5%) nằm **ngoài hẳn** vùng dữ liệu huấn
luyện (2012–2021, lãi suất 0–2%). Biến **mức** của chuỗi phi dừng ngoại suy
hỏng. Sàng đơn biến trên kiểm định **không thấy điều đó** vì đoạn kiểm định
(2021–2023) nằm giữa quá trình trôi.

Màn lọc độ vững của E3 — đòi **cùng dấu ở ≥5/6 cặp và ≥4/5 năm** — loại đúng
những biến đó, vì một chuỗi đang trôi không thể giữ cùng dấu qua 12 năm.

**Quy công cho đúng:** phần lớn giá trị của E3 đến từ **màn lọc độ vững**,
không phải từ bản thân kiểm định Granger. Biên bản định nghĩa E3 là *"qua
được mục 4b **và** 4e"* nên quy công này nằm trong đúng thứ đã khai báo — nhưng
phải nói rõ, không được để người đọc tưởng Granger một mình làm nên chuyện.

### 4d. Độ vững của E3

| lát cắt | kết quả |
|---|---|
| theo cặp (kiểm tra) | **4/6** cải thiện — *trượt ngưỡng 5/6* |
| theo năm (kiểm tra) | 2023 −3,7% · 2024 −1,2% · 2025 **+4,0%** |
| theo chế độ | Q2 −4,9% · Q3 −2,9% tốt hơn; Q1 êm +3,1% · Q5 căng +3,8% tệ hơn |
| **walk-forward** (khớp lại đầu mỗi năm) | **E3 tốt hơn 9/11 năm**, từ −8,2% đến +4,6% |

Walk-forward dương 9/11 năm nhưng chấm một lần trên kiểm tra lại hoà — hai con
số **không mâu thuẫn**: walk-forward khớp lại mỗi năm nên tự thích nghi với
trôi phân phối, còn cấu hình chấm một lần thì khớp cố định trên huấn luyện.
Đây chính là bằng chứng thứ hai cho cơ chế ở 4c.

### 4e. Trục hướng và trục rủi ro

**Hướng** (Log Score / BSS so khí hậu học):

| cấu hình | log kiểm tra | BSS kiểm tra | AUC |
|---|---|---|---|
| B0 | 1,0980 | **+0,00041** | 0,515 |
| E1 | 1,1199 | −0,02152 | 0,505 |
| E2 | 1,0994 | −0,00108 | 0,521 |
| E3 | 1,0981 | +0,00024 | 0,522 |

**Không cải thiện.** AUC 0,505–0,522 — vẫn không phân biệt được hướng. Kết
luận âm thứ năm về trục hướng, nay bằng tầng thông tin ngoại sinh.

**Rủi ro** (CRPS phân phối lợi suất, độ phủ 80%):

| cấu hình | CRPS kiểm tra | kỹ năng | phủ 80% | cặp dương |
|---|---|---|---|---|
| B0 | 0,00257 | 2,15% | 79,9% | 6/6 |
| E3 | 0,00257 | **2,19%** | **80,0%** | 6/6 |

Cải thiện **+0,04 điểm phần trăm** — thật nhưng không đáng kể, không đạt
ngưỡng nào.

---

## 5. MDES — kết luận âm này mạnh tới đâu

*Bắt buộc theo biên bản mục 5c. Tái lập: `python src/pha3b_mdes.py`.*

Hai quyết định thiết kế, cả hai để tránh đo lực giả:

1. **Cửa phải là cửa thật** — dùng phân phối null của thống kê **max** lấy từ
   chính lần chạy Week 2 (`output/pha3b_null_max.npy`, 1.000 hoán vị khối trên
   594 giả thuyết), ngưỡng F(95%) = **15,082**. Xấp xỉ Šidák trên p từng giả
   thuyết **không dùng được**: với số hoán vị hữu hạn nó không bao giờ xuống
   dưới 0,05 và sẽ báo "lực 0%" cho mọi cỡ hiệu ứng — một kết quả giả (đã mắc
   và đã sửa trong lúc làm).
2. **Vật mang phải là đặc trưng thật** — tiêm qua thành phần chính thứ nhất
   của khối `DGS2` lag 1 (biến **không** sống sót Week 2), nên cấu trúc cộng
   tuyến và phụ thuộc chuỗi giữ nguyên như dữ liệu thật.

| β | ΔQLIKE kiểm tra | lực (cửa W-Y thật) |
|---|---|---|
| **0,00** | +5,48% | **0,0%** ← đối chứng âm **ĐẠT** |
| 0,04 | +3,26% | 0,0% |
| 0,06 | +1,38% | 8,3% |
| **0,08** | **−0,97%** | **91,3%** ← **MDES** |
| 0,12 | −6,86% | 100,0% |
| 0,24 | −29,36% | 100,0% |

> **Phát biểu dùng được:** *"Phễu Pha 3B phát hiện được, với xác suất 80%, một
> quan hệ ngoại sinh đủ mạnh để cải thiện QLIKE khoảng **1,0%** (ròng, sau khi
> trừ chi phí ước lượng của khối). Mọi hiệu ứng mạnh hơn thế đã bị loại trừ
> trên dữ liệu này."*

So sánh sức mạnh ba kết luận âm của dự án:

| pha | không gian giả thuyết | quan sát độc lập ở kiểm tra | MDES |
|---|---|---|---|
| Giai đoạn 1 | 8.469 | 548 phiên | lift **1,20** |
| Pha 2 (tin tức) | 183 | 129 thông cáo | lift **1,35** |
| Pha 3 cũ (vĩ mô tháng) | 7 | **26 tháng** | *không khai báo được* |
| **Pha 3B** | **594** | **548 phiên** | **≈1,0% QLIKE** |

Ghi chú cách đọc: ở β = 0, thêm khối 6 biến vô ích vẫn làm QLIKE **xấu đi
5,48%** — đó là **chi phí ước lượng** của việc thêm biến mức phi dừng, đúng
cùng cơ chế ở mục 4c. Nên con số MDES là **ròng**: hiệu ứng phải đủ mạnh để
vừa trả chi phí đó vừa còn lãi 1%.

---

## 6. PHÁN QUYẾT theo tiêu chí chốt trước

### 6a. Đối chiếu từng điều kiện (biên bản mục 6a)

| điều kiện | kết quả | |
|---|---|---|
| ĐK1 E3 thắng kiểm tra & p thô < 0,0167 (Bonferroni 3 trục) | **TRƯỢT** | +0,80%, p = 0,7462 |
| ĐK2 ≥ 5/6 cặp cải thiện | **TRƯỢT** | 4/6 |
| ĐK3 dấu hệ số khớp lý do kinh tế khai báo | **ĐẠT** | `sk_ngay_L1` = −0,117 |
| ĐK4 **E3 ≥ E2** (lọc nhân quả ≥ chọn thường) | **ĐẠT** | −3,07%, p = 0,0010 |

### 6b. Vì sao phán quyết phải TÁCH ĐÔI

Biên bản mục 6b định nghĩa `CREDIBLE NEGATIVE` đòi **cả ba**, trong đó điều
kiện 1 là *"không ứng viên nào sống sót Westfall–Young, hoặc có sống sót nhưng
trượt màn lọc độ vững"*. **Điều kiện đó KHÔNG đúng ở đây** — 14 ứng viên sống
sót cả hai cửa.

Nên kết quả **không phải POSITIVE** (trượt ĐK1, ĐK2) và cũng **không phải
CREDIBLE NEGATIVE** như đã định nghĩa. Hai nhãn chốt trước không phủ hết không
gian kết quả. Ghi nhận thẳng đó là **thiếu sót của chính biên bản chốt trước**,
không phải lý do để bẻ cong số liệu — và xử lý đúng như tiền lệ Pha 2:

| phần | phán quyết |
|---|---|
| **(a) Phát hiện quan hệ** (Week 2) | **DƯƠNG** — 14/594 sống sót W-Y **và** màn lọc độ vững; 6/6 cặp, 12/12 năm cùng dấu. Lần đầu trong toàn dự án |
| **(b) Tích hợp vào hệ thống** (Week 3) | **ÂM** — không cấu hình nào (E1/E2/E3/E4) thắng mốc đã đóng băng trên kiểm tra |
| **(c) Câu hỏi trung tâm E3 vs E2** | **DƯƠNG** — lọc nhân quả hơn chọn thường 3,07%, p = 0,0010. Giá trị **phòng thủ**: giảm một nửa độ trôi kiểm định→kiểm tra |
| **(d) Trục hướng / rủi ro** | **ÂM** — 0/198 và 0/198 sống sót; BSS và CRPS không cải thiện |

### 6c. Chẩn đoán bổ sung — phần nào của tín hiệu lịch là thật?

*Chẩn đoán của biến ĐÃ KHAI BÁO `su_kien_phi_nhtw`, không phải giả thuyết mới —
đếm 0 cấu hình mới, cùng quy ước `run_m2_vung.py`.*

B0 đã có đại diện NFP theo **quy tắc** ("thứ Sáu đầu tháng"), nên phải tách:

| cấu hình | QLIKE kiểm định | QLIKE kiểm tra | so B0 | DM p |
|---|---|---|---|---|
| B0 | 0,1569 | 0,1872 | — | — |
| + **NFP** (ngày công bố THẬT, L1+L2) | 0,1564 | **0,1864** | **−0,43%** | **<0,0001** |
| + công bố khác NFP (L1+L2) | 0,1525 | 0,1915 | +2,29% | 0,1995 |
| + cả hai | 0,1512 | 0,1901 | +1,54% | 0,3975 |

**Phần thật sự vững là NFP, và nó là một bản vá CHẤT LƯỢNG DỮ LIỆU chứ không
phải thông tin ngoại sinh mới:** B0 dùng quy tắc "thứ Sáu đầu tháng" — sai vài
lần mỗi năm — còn ở đây dùng **ngày công bố thật từ FRED**. Thay quy tắc bằng
ngày thật cải thiện 0,43%, p < 0,0001, tốt hơn trên **cả** kiểm định lẫn kiểm
tra.

Đây đúng là điều `KEHOACH_2026Q4.md` đã dự đoán trước khi có dữ liệu: *"quy tắc
đó sai vài lần mỗi năm, và ngày sai sẽ làm loãng phản ứng đo được"*.

Các công bố **không phải NFP** thắng kiểm định (0,1525) nhưng **thua kiểm tra**
(+2,29%) — lại đúng hình mẫu overfit đoạn chọn.

---

## 7. WEEK 4 — Tập niêm phong: **KHÔNG MỞ**

Quy tắc đã chốt trước ở biên bản mục 8:

> DƯƠNG theo mục 6a → mở một lần. ÂM → không mở.

Phán quyết **không đạt DƯƠNG** (trượt ĐK1 và ĐK2), nên **tập niêm phong không
được mở**. Giữ nguyên cho lần chạy cuối của toàn hệ thống, đúng `KHOA_SO.md` và
đúng tiền lệ `PHA2_KETQUA.md` mục 3c.

Quyết định này ghi **trước** khi thấy số — không phải bào chữa viết sau.

---

## 8. Những gì Pha 3B **không** làm được — khai báo thẳng

| hạng mục | tình trạng |
|---|---|
| **Ẩn ý biến động FX** (`EVZCLS`) | **không dùng được** — CBOE ngừng công bố 03/2025. Đây là biến ngoại sinh hứa hẹn **nhất** cho mục tiêu biến động FX (ẩn ý biến động là dự báo của chính thị trường về đúng đại lượng ta đang dự báo). Không có nó, kết luận âm của Pha 3B **không** loại trừ được kênh này |
| PCMCI đầy đủ | **không** — bản rút gọn tuyến tính tự viết; thiếu kiểm định độc lập phi tuyến và xử lý trễ đồng thời |
| Lợi suất TPCP nước ngoài theo ngày | **không có trên FRED** (chỉ chuỗi tháng `IRLTLT01*`) → chênh lệch lợi suất **riêng từng cặp** chỉ có ở tần suất tháng |
| Giá trị công bố vĩ mô (số thực tế, số dự báo đồng thuận) | **không dùng** — chỉ dùng ngày công bố. Đo "bất ngờ" (\|thực − dự báo\|) cần nguồn trả phí |
| Tính độc lập của 6 cặp | **không có** — phần lớn biến ngoại sinh là biến Mỹ/toàn cầu dùng chung, nên 6 cặp **không phải** 6 quan sát độc lập. Đã khai báo trước ở mục 7 của biên bản |
| Khoảng hiệu ứng dưới 1% QLIKE | **không loại trừ được** (MDES) |

---

## 9. Hệ quả cho luận văn

**Ba câu phát biểu được, đã có bằng chứng đầy đủ:**

1. *Thông tin ngoại sinh có quan hệ dẫn báo thống kê với biến động FX — 14/594
   quan hệ sống sót kiểm soát FWER và màn lọc độ vững xuyên cặp, xuyên năm —
   nhưng **không** chuyển hoá thành cải thiện dự báo ngoài mẫu so với một mô
   hình HAR đã hiệu chỉnh tốt.*

2. *Lọc theo nhân quả có giá trị **phòng thủ** đo được: so với chọn đặc trưng
   thông thường, nó giảm một nửa mức sụp đổ từ kiểm định sang kiểm tra
   (10,1 → 5,0 điểm), −3,07% QLIKE, p = 0,0010. Cơ chế đã đo: sàng theo hiệu
   năng kiểm định chọn đúng những biến vĩ mô đang trôi (DGS2 lệch 4,5 độ lệch
   chuẩn giữa hai đoạn), còn màn lọc độ vững loại chúng.*

3. *Kết luận "kỹ năng chỉ nằm ở trục biên độ" nay đã được xác nhận trên **bốn
   tầng thông tin độc lập**: lịch sử giá (Pha 1) · tin tức văn bản (Pha 2) ·
   vĩ mô tháng (Pha 3) · ngoại sinh ngày + nhân quả (Pha 3B).*

**Đóng góp mới mà repo chưa từng có:** đây là lần đầu dự án trả lời được câu
*"lọc nhân quả có hơn chọn đặc trưng thường không"* bằng một thí nghiệm đối
chứng có chốt trước — và câu trả lời là **có**, dù cả hai đều không thắng được
mốc.

---

## 10. Tái lập

```bash
python collect/ngoai_sinh.py      # Week 1 — tải + ngữ nghĩa + kiểm hiệu đính (~40s)
python src/pha3b_dactrung.py      # Week 1 — đặc trưng + kiểm cắt tương lai (~60s)
python src/pha3b_granger.py       # Week 2 — Level 1/2/3 + W-Y (~95s)
python src/pha3b_ablation.py      # Week 3 — B0/E1/E2/E3/E4 × 3 trục (~35s)
python src/pha3b_mdes.py          # Week 3 — MDES (~20s)
```

Kết quả: `output/pha3b_{granger,quanhe,ablation,mdes,null_max}.*` ·
`data/ngoai_sinh/{chuoi_ngay,ngu_nghia,kiem_hieu_dinh}.*`

---

## 11. Đối chiếu tiêu chí ra khỏi Pha 3 (HuyH §Phase 3 exit criteria)

- [x] Exogenous data reproducible and time-aligned — mục 2a, 10
- [x] Revision leakage controlled — mục 2b, **đo bằng ALFRED vintage**
- [x] Hypothesis/lag space declared before final testing — 594, `PHA3B_TIEUCHI.md` mục 2f
- [x] Multiple-testing correction applied — W-Y maxT từng bước, null khối
- [x] Candidate stability tested — mục 3c
- [x] All-exogenous vs causal-filtered ablation exists — **E1 vs E2 vs E3**, mục 4
- [x] Direction/magnitude/risk reported separately — mục 4a, 4e
- [x] MDES reported for negative findings — mục 5
- [x] Sealed test used once after freeze — **không mở**, có lý do chốt trước, mục 7
- [x] Product evidence uses non-overclaiming language — không dùng từ "nguyên nhân" ở bất kỳ đâu; dùng "quan hệ dẫn báo theo thời gian"
- [x] Final M3 conclusion documented — tài liệu này

---

## 12. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 6 (phán quyết tách đôi),
mục 7 (không mở niêm phong), và mục 8 (khai báo hạn chế):

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không phải
> công cụ điền hộ.
