# PHA 3B — CAUSALITY-AWARE: TIÊU CHÍ CHỐT TRƯỚC

*Lập 12/09/2026. Nhánh `replan-2026`.
**Commit chứa file này phải nằm TRƯỚC mọi commit chứa số liệu Pha 3B.***

---

## 0. Vì sao có văn bản này — và quan hệ với `PHA3_TIEUCHI.md`

Roadmap của HuyH (`03_PHASE_3_CAUSALITY_AWARE.md`, 4 tuần) đặt ra một Pha 3
**rộng hơn nhiều** so với thứ repo đã chạy ngày 12/09/2026:

| | `PHA3_TIEUCHI.md` (đã chạy) | **Pha 3B — văn bản này** |
|---|---|---|
| Biến ngoại sinh | 5 đặc trưng, **1 họ** (lãi suất chính sách) | **11 biến, 6 họ** (lợi suất TPCP, ẩn ý biến động, hàng hoá, cổ phiếu, chênh lãi suất, lịch công bố) |
| Tần suất | **tháng** (26 tháng độc lập ở kiểm tra) | **ngày** (548 phiên độc lập ở kiểm tra) |
| Bước phát hiện nhân quả | **không có** | **Week 2 đầy đủ**: Level 1 (hữu ích dự báo) → Level 2 (Granger + kiểm soát bội) → Level 3 (PCMCI rút gọn) |
| Ablation | B0 vs 6 biến thể đơn | **B0 / E1 / E2 / E3 / E4** — tách bạch *"lọc nhân quả có hơn chọn đặc trưng thường không"* |
| Trục đánh giá | chỉ biên độ | **cả ba**: hướng · biên độ · rủi ro (HuyH §Week 3 bắt buộc tách riêng) |
| Ngữ nghĩa dấu thời gian | trễ 2 tháng cố định | bảng `available_time` đầy đủ + kiểm hiệu đính |

Pha 3B **không thay thế** `PHA3_TIEUCHI.md` — kết quả âm của nó giữ nguyên giá
trị và được trích dẫn lại như một lát cắt của Family E. Pha 3B là bản thi hành
**đúng đặc tả HuyH**, thứ mà bản trước chỉ làm được một phần.

---

## 1. Mục tiêu và bốn mô hình so sánh (HuyH §1)

> Kiểm tra liệu thông tin **ngoại sinh** và **lọc theo nhân quả/thời gian** có
> thêm giá trị, độ vững, hoặc tính giải thích được ngoài các tầng đã có hay không.

```
M1  = lịch sử (HAR sản xuất, đã đóng băng — KHOA_SO.md mục 4)
M2  = lịch sử + tin tức            → PHA2_KETQUA.md: ÂM, nhánh văn bản đã STOP
M3a = M1 + ngoại sinh                        (E1: ném hết vào)
M3b = M1 + ngoại sinh ĐÃ LỌC theo nhân quả   (E3)
```

Vì Pha 2 kết luận tin tức không hữu ích, HuyH §1 cho phép mốc là `M1 + Exogenous`.
**Chốt: dùng M1 làm B0.** M2 vẫn được báo cáo lại để giữ mạch nghiên cứu.

**Câu hỏi trung tâm của Pha 3B, chốt tại đây:**

> Không chỉ *"ngoại sinh có giúp không"*, mà **"lọc theo nhân quả có hơn việc
> chọn đặc trưng thông thường không"** — tức `E3 > E2` hay không.

Đây là câu hỏi mà cả `PHA3_TIEUCHI.md` lẫn bất kỳ tài liệu nào trong repo
**chưa từng** đặt ra.

---

## 2. WEEK 1 — Họ biến ngoại sinh chốt trước (HuyH Week 1 §1)

Nguyên tắc HuyH: *"Không ingest toàn bộ macro universe"* và *"mỗi variable phải
có lý do liên quan đến currency/pair"*. Danh sách dưới đây **đóng** — thêm bất
kỳ biến nào sau ngày lập phải mở biên bản mới và đếm lại vào `KHOA_SO.md`.

### 2a. Danh sách 11 biến, 6 họ

| # | mã FRED | họ | tần suất | lý do kinh tế gắn với cặp |
|---|---|---|---|---|
| 1 | `DGS2` | A · lợi suất TPCP Mỹ | ngày | kỳ vọng chính sách Fed — mẫu số của mọi cặp USD |
| 2 | `DGS10` | A | ngày | lợi suất dài hạn, kênh định giá lại tài sản |
| 3 | `T10Y2Y` | A | ngày | độ dốc đường cong — chỉ báo chu kỳ chính sách; văn liệu gắn với biến động FX |
| 4 | `DFII10` | A | ngày | lợi suất **thực** 10 năm (TIPS) — kênh lãi suất thực, tách khỏi kỳ vọng lạm phát |
| 5 | `VIXCLS` | B · ẩn ý biến động | ngày | khẩu vị rủi ro toàn cầu; kênh tháo carry → JPY, CHF (trú ẩn), AUD (rủi ro) |
| 6 | `GVZCLS` | B | ngày | ẩn ý biến động vàng — đại diện cầu trú ẩn, gắn CHF |
| 7 | `OVXCLS` | B | ngày | ẩn ý biến động dầu — gắn CAD |
| 8 | `DCOILBRENTEU` | C · hàng hoá | ngày | dầu Brent — CAD là đồng hàng hoá, kênh điều kiện thương mại |
| 9 | `NIKKEI225` | D · cổ phiếu | ngày | đại diện rủi ro/carry cho JPY |
| 10 | `chenh_ls` | E · chênh lãi suất chính sách | tháng | `fred_rates.csv` — **giữ nguyên định nghĩa của `PHA3_TIEUCHI.md`** để so được |
| 11 | `su_kien_phi_nhtw` | F · lịch công bố | ngày (sự kiện) | CPI/NFP/GDP/PPI/bán lẻ/PCE/JOLTS — **ngày công bố, không dùng giá trị** |

### 2b. Ba thứ **loại bỏ**, và lý do — chốt trước

| loại bỏ | lý do |
|---|---|
| `DTWEXBGS` (chỉ số USD rộng) | **chứa chính 6 cặp đích** → không phải ngoại sinh. Repo đã thử thông tin chéo cặp và ra âm (`KETQUA_VONG7.md` mục 4) |
| `EVZCLS` (ẩn ý biến động EUR/USD) | **ngừng công bố 03/2025**, sẽ khuyết 9 tháng cuối đoạn kiểm tra → thiên lệch. Đây là biến **hứa hẹn nhất** mà ta không dùng được; ghi vào phần giới hạn |
| bộ `TSF/data/raw/forex_macro_raw_2015_2026.csv` | nguồn Yahoo, xuất xứ không truy được; chỉ từ 2015 (mất 3 năm); phủ tới 08/2026 → **chồng tập khoá sổ** |

### 2c. Họ F đã nằm một phần trong B0 — khai báo để không tự lừa

B0 (HAR sản xuất) **đã có** lịch ngân hàng trung ương riêng từng cặp
(`event=capday`, `KETQUA_VONG7.md` mục 1). Nên họ F **chỉ được tính các công bố
KHÔNG phải NHTW** (CPI, NFP, GDP, PPI, bán lẻ, PCE, JOLTS, UMCSI, nhà ở) — phần
NHTW đã nằm trong mốc và đưa lại là đếm hai lần.

### 2d. Bộ biến đổi chốt trước — 6 dạng

Với mỗi chuỗi thị trường (biến 1–9), đúng **6** biến đổi, không hơn:

| dạng | định nghĩa | động cơ |
|---|---|---|
| `lv` | mức (chuẩn hoá bằng trung bình/độ lệch **của đoạn huấn luyện**) | trạng thái |
| `d1` | thay đổi 1 phiên | cú sốc có dấu |
| `ad1` | \|thay đổi 1 phiên\| | **độ lớn cú sốc** — kênh chính cho mục tiêu biến động |
| `d5` | thay đổi 5 phiên | xu thế ngắn |
| `ad5` | \|thay đổi 5 phiên\| | độ lớn xu thế |
| `rv20` | độ lệch chuẩn 20 phiên của `d1` | **biến động của chính biến ngoại sinh** — kênh lan truyền biến động |

Biến 10 (`chenh_ls`, tháng) dùng lại đúng 5 đặc trưng của `PHA3_TIEUCHI.md` mục
3a. Biến 11 (lịch) dùng 3 dạng: ngày công bố, ngày kế tiếp, đếm công bố trong
5 phiên gần nhất.

### 2e. Độ trễ ứng viên — chốt trước

`lag ∈ {1, 2, 5}` phiên giao dịch. `lag=1` nghĩa là giá trị của phiên `t−1`
dùng để dự báo `rv(t+1)` — đã bảo thủ hơn mức tối thiểu cần thiết.

### 2f. Không gian giả thuyết — **liệt kê đầy đủ TRƯỚC khi chạy**

```
Week 2 (Granger, Level 2):
   11 biến gốc × 3 lag × 6 cặp × 3 trục = 594 phép kiểm
Week 3 (ablation):
   5 cấu hình (B0, E1, E2, E3, E4) × 3 trục = 15 phép chấm
```

Tổng **594 giả thuyết nhân quả + 15 cấu hình**, đếm vào `KHOA_SO.md` sau khi chạy.

---

## 3. WEEK 1 — Ngữ nghĩa dấu thời gian và chống rò rỉ (HuyH Week 1 §2–3)

### 3a. Bảng ngữ nghĩa bắt buộc

Mỗi quan sát ngoại sinh lưu đủ:

```
observation_period   kỳ quan sát của giá trị
release_time         thời điểm công bố lần đầu
available_time       thời điểm HỆ THỐNG được phép dùng  ← cột quyết định
revision_time        thời điểm hiệu đính (nếu có)
value_as_released    giá trị lúc công bố lần đầu
revised_value        giá trị hiện hành
source               FRED series id
```

### 3b. Chính sách `available_time` — chốt trước

| loại chuỗi | quy tắc |
|---|---|
| thị trường ngày (`DGS*`, `T10Y2Y`, `DFII10`, `VIXCLS`, `GVZCLS`, `OVXCLS`, `DCOILBRENTEU`, `NIKKEI225`) | `available_time` = **phiên kế tiếp**. Giá trị ngày `t` chỉ dùng cho dự báo từ `t+1` |
| lãi suất chính sách theo tháng (`chenh_ls`) | **trễ 2 tháng**, giữ nguyên quy tắc đã chốt ở `PHA3_TIEUCHI.md` mục 4 |
| lịch công bố (`su_kien`) | ngày công bố là **biết trước**, không rò rỉ (lịch công bố trước nhiều tháng) |

### 3c. Hiệu đính — phân loại chốt trước

| nhóm | có hiệu đính? | xử lý |
|---|---|---|
| chuỗi thị trường (giá đóng cửa, lợi suất niêm yết) | **không** | dùng trực tiếp; **phải chứng minh bằng kiểm ALFRED vintage**, không được khẳng định suông |
| lãi suất liên ngân hàng tháng | có thể | trễ 2 tháng đã bao trùm |
| giá trị công bố vĩ mô (CPI, NFP…) | **có, mạnh** | **không dùng giá trị** — chỉ dùng ngày công bố |

### 3d. Hai phép tự kiểm bắt buộc (HuyH Week 1 §3)

1. **Kiểm cắt tương lai**: `feature(t)` tính từ toàn bộ CSDL phải **bằng đúng**
   `feature(t)` tính sau khi cắt bỏ mọi thông tin lộ ra sau `t`. Mốc cắt:
   `2023-01-01`. Ngưỡng đạt: **0 giá trị đổi**.
2. **Kiểm hiệu đính (ALFRED vintage)**: với ≥ 3 chuỗi thị trường, tải bản
   vintage tại một thời điểm quá khứ và so với bản hiện hành. Ngưỡng đạt:
   sai khác ≤ 0,5% số quan sát (cho phép sai số làm tròn/đăng muộn).

**Nếu bất kỳ phép nào hỏng → dừng, sửa, chạy lại. Không báo cáo kết quả.**

---

## 4. WEEK 2 — Ba mức phát hiện nhân quả (HuyH Week 2)

### 4a. Level 1 — hữu ích dự báo

> *Does past X add information beyond existing baseline features?*

Hồi quy lồng nhau, điều kiện hoá trên chính dự báo của mốc:

```
y(t+1) = a + b·log ĥ_HAR(t) + Σ_l c_l · X(t−l) + e
H0: mọi c_l = 0
```

Đây đúng là **đối chứng có điều kiện** mà repo đã dùng từ Giai đoạn 2
(`REPLAN_2026.md` mục 3.3) và là null mạnh nhất đã biết cho trục biên độ.

### 4b. Level 2 — Granger / nhân quả thời gian

> *Does past X improve prediction of Y beyond past Y / baseline state?*

Kiểm định F trên mô hình lồng nhau, hiệp phương sai **Newey–West** (HAC), bậc
trễ `L = ⌈1,5·n^(1/3)⌉` — cùng quy ước `dm_nw` đang dùng khắp repo.

Mỗi phép kiểm ghi đủ, đúng danh mục HuyH đòi:

```
lag · effect sign · effect size · raw statistic · adjusted statistic
· period stability · pair stability
```

**Không diễn giải Granger như nhân quả cấu trúc** (HuyH Week 2 và §13).
Từ dùng trong mọi báo cáo: *"quan hệ dẫn báo theo thời gian"*, không dùng
*"nguyên nhân"*.

### 4c. Level 3 — PCMCI rút gọn

`tigramite` **không có** trong môi trường. Cài bản rút gọn tự viết:
PC-stable chọn tập cha bằng tương quan riêng phần + bước MCI, tuyến tính.
HuyH §Week 2 ghi rõ *"Không bắt buộc phải có PCMCI để Phase 3 thành công"* —
nên đây là **bổ sung**, và phải **khai báo thẳng là bản rút gọn tuyến tính**,
không được trình bày như PCMCI đầy đủ.

### 4d. Kiểm soát bội — bắt buộc

```
liệt kê họ giả thuyết (594, mục 2f)
   → Westfall–Young maxT từng bước xuống, null hoán vị KHỐI
   → thống kê đã hiệu chỉnh
```

Độ dài khối: **5 phiên** (quy ước `KHOI=5` của repo cho dữ liệu ngày).
Số hoán vị: **1.000**. FDR-BH báo cáo **song song làm tham chiếu**, không dùng
làm cửa (lý do: `REPLAN_2026.md` mục 3.2 — FDR để lọt 262/336 trong khi W-Y
khối chỉ để lọt 41).

### 4e. Màn lọc độ vững — mọi ứng viên phải qua

```
sign consistency · lag consistency · year/subperiod stability
· pair stability · regime sensitivity
```

Ngưỡng đạt chốt trước: dấu nhất quán ≥ **5/6 cặp**, và ≥ **4/5 năm** của đoạn
huấn luyện+kiểm định.

---

## 5. WEEK 3 — Ablation B0/E1/E2/E3/E4 (HuyH Week 3)

| cấu hình | nội dung | trả lời câu gì |
|---|---|---|
| **B0** | HAR sản xuất đã đóng băng | mốc |
| **E1** | B0 + **toàn bộ** đặc trưng ngoại sinh đã khai báo | *ngoại sinh có giúp không?* |
| **E2** | B0 + đặc trưng **chọn theo cách thường** (sàng đơn biến trên **kiểm định**, lấy top-k theo QLIKE) | *chọn đặc trưng thường được gì?* |
| **E3** | B0 + đặc trưng **đã lọc theo Granger** (qua được mục 4b + 4e) | **câu hỏi trung tâm: lọc nhân quả có hơn E2 không?** |
| **E4** | B0 + tương tác ngoại sinh × chế độ biến động | *hiệu ứng có phụ thuộc chế độ không?* |

**k của E2 chốt trước = số đặc trưng mà E3 chọn được** (so sánh công bằng: cùng
số tham số, khác cách chọn). Nếu E3 chọn 0 đặc trưng thì E2 lấy `k = 3`.

### 5a. Ba trục phải báo cáo riêng (HuyH Week 3 — bắt buộc)

| trục | mốc B0 | chỉ số chính | chỉ số phụ |
|---|---|---|---|
| **Biên độ** | HAR sản xuất | **QLIKE** (Bregman bất biến thang đo) | MSE, MAE |
| **Hướng** | tổ hợp trực tuyến đang chạy (`balop`) | **Log Score, BSS** so khí hậu học | AUC, độ chính xác |
| **Rủi ro** | σ̂ × phân vị z thực nghiệm (V0) | **CRPS phân phối lợi suất**, độ phủ conformal | VaR/ES điểm |

### 5b. Độ vững bắt buộc

`walk-forward` · từng cặp · từng năm · chế độ thường vs chế độ căng
(ngũ phân vị σ̂, ngưỡng lấy từ **huấn luyện**).

### 5c. MDES — bắt buộc nếu âm (HuyH Week 3)

Nếu không thấy cải thiện, **phải** báo cáo độ lớn hiệu ứng nhỏ nhất phát hiện
được ở lực 80%, bằng tiêm hiệu ứng biết trước vào chính phễu — đúng cách
`kiem_pheu.py` đã làm cho Giai đoạn 1 (MDES lift 1,20) và `kiem_pheu_h8.py`
cho Pha 2 (1,35).

HuyH §Week 3: *"Không mở thêm hàng chục causal methods chỉ vì result âm."*
**Chốt: không thêm phương pháp nào ngoài Level 1–3 ở mục 4.**

---

## 6. TIÊU CHÍ PHÁN QUYẾT — chốt trước

### 6a. DƯƠNG (`POSITIVE`) đòi **cả bốn**

1. E3 thắng B0 trên **kiểm tra** ở ít nhất **một trục**, với DM p < 0,05 **sau
   Bonferroni cho 3 trục** (p thô < 0,0167).
2. **Và** ≥ 5/6 cặp cải thiện trên trục đó.
3. **Và** dấu hệ số **khớp** lý do kinh tế đã khai báo ở mục 2a cho biến đó.
4. **Và** — điều kiện riêng của Pha 3B — **E3 ≥ E2** trên chính trục đó. Nếu
   E3 ≈ E2 thì kết luận đúng là *"ngoại sinh có giúp, nhưng lọc nhân quả không
   thêm gì so với chọn đặc trưng thường"* — một phán quyết **khác**, phải ghi
   riêng, không được gộp vào "dương".

### 6b. ÂM ĐÁNG TIN (`CREDIBLE NEGATIVE`) khi **tất cả**

1. Không ứng viên nào sống sót Westfall–Young ở mục 4d, **hoặc** có sống sót
   nhưng trượt màn lọc độ vững 4e.
2. **Và** E1, E2, E3 đều không thắng B0 trên kiểm tra theo ngưỡng 6a.1.
3. **Và** MDES đã được báo cáo (mục 5c).

### 6c. Điều kiện dừng (HuyH §Phase 3 stop criteria)

Dừng mở rộng nhánh ngoại sinh/nhân quả khi: không hiệu ứng khai báo trước nào
sống sót hiệu chỉnh + độ vững · **hoặc** MDES đủ loại trừ hiệu ứng có ý nghĩa
thực tế · **hoặc** chất lượng dấu thời gian không đủ để phát biểu kết luận
nhân quả đáng tin.

> *"Không được tiếp tục thử method cho tới khi một p-value đẹp xuất hiện."*

---

## 7. Khai báo lực — TRƯỚC, không phải sau (bài học `PHA3_TIEUCHI.md` mục 7)

| đoạn | phiên/cặp | hàng (6 cặp) | **quan sát độc lập của biến ngoại sinh** |
|---|---|---|---|
| huấn luyện | ~2.503 | ~15.018 | **~2.503 phiên** |
| kiểm định | ~547 | ~3.282 | **~547 phiên** |
| kiểm tra | ~548 | ~3.282 | **~548 phiên** |

Đây là bước nhảy **×21** về số quan sát độc lập ở đoạn kiểm tra so với
`PHA3_TIEUCHI.md` (26 tháng → 548 phiên). Hệ quả chốt trước: **kết luận âm của
Pha 3B, nếu xảy ra, mạnh hơn hẳn kết luận âm của Pha 3 cũ**, và phát biểu được
ở mức *"tầng ngoại sinh ngày không thêm thông tin đo được"* thay vì chỉ
*"không phát hiện được với dữ liệu tháng"*.

Giới hạn còn lại phải ghi: các cặp dùng chung phần lớn biến ngoại sinh (đều là
biến Mỹ/toàn cầu), nên 6 cặp **không phải** 6 quan sát độc lập cho những biến
đó — đây là lý do màn lọc độ vững 4e đòi tính nhất quán xuyên cặp thay vì
cộng dồn cỡ mẫu.

---

## 8. WEEK 4 — Tập niêm phong: quyết định chốt trước

HuyH Week 4 §2 yêu cầu chạy sealed test một lần. `KHOA_SO.md` giữ tập niêm
phong cho **lần chạy cuối của toàn hệ thống**, và `PHA2_KETQUA.md` mục 3c đã
lập tiền lệ **không tiêu** lần mở đó cho một pha đơn lẻ.

**Quy tắc chốt trước cho Pha 3B:**

| kết quả Week 3 | hành động Week 4 |
|---|---|
| **DƯƠNG** theo mục 6a | **mở tập niêm phong một lần**, sau khi đóng băng đầy đủ biến/lag/phương pháp chọn/mô hình/thước đo/luật quyết định |
| **ÂM** theo mục 6b | **không mở** — lý do đúng như `PHA2_KETQUA.md` mục 3c: *"không có gì" trên kiểm tra không mạnh thêm khi có thêm "không có gì" trên tập niêm phong* |

Quyết định này ghi **trước** khi thấy số, nên không phải là bào chữa viết sau.

---

## 9. Danh sách KHÔNG LÀM ở Pha 3B

- **Không** thêm biến ngoại sinh ngoài 11 biến mục 2a.
- **Không** thêm biến đổi ngoài 6 dạng mục 2d, hay lag ngoài {1, 2, 5}.
- **Không** thêm phương pháp nhân quả ngoài Level 1–3.
- **Không** mở lại trục hướng giá bằng phễu quy luật (đã cạn kiệt, 8.652 giả
  thuyết) — trục hướng ở đây **chỉ** được chấm như một trục báo cáo của
  ablation, không phải một cuộc khai phá mới.
- **Không** đổi B0, đổi cách chia, hay đổi thước đo sau khi thấy số.
- **Không** dùng từ *"nguyên nhân"* / *"proven cause"* trong bất kỳ báo cáo
  hay giao diện nào (HuyH Week 4 §3).

---

## 10. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 2 (biến chốt trước),
mục 6 (tiêu chí phán quyết), mục 7 (khai báo lực) và mục 8 (quyết định niêm
phong) **trước khi** bất kỳ số liệu nào được chấm:

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không phải
> công cụ điền hộ.
