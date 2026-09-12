# Phương pháp từ văn liệu ngoài — thử, đo, và kết luận

*Bắt đầu 12/09/2026. Nhánh `replan-2026`.
Mục đích: tìm trong văn liệu 2025–2026 những phương pháp nhắm đúng **điểm yếu
đã đo được** của hệ thống, cài lại, chạy đúng giao thức của repo, và báo cáo
kết quả ra sao thì đúng như vậy.*

Nguyên tắc không đổi: **chốt cấu hình trước khi chạy · khớp huấn luyện · chọn
trên kiểm định · chấm một lần trên kiểm tra · đếm mọi cấu hình vào `KHOA_SO.md`.**

---

## 0. Điểm yếu nào đáng vá — xếp theo mức đã đo được

| # | điểm yếu | bằng chứng | có phương pháp ngoài không |
|---|---|---|---|
| 1 | **Hàm mất khớp ≠ hàm mất chấm** — HAR khớp bằng OLS trên log, chấm bằng QLIKE | repo tự đo: LightGBM tối ưu QLIKE hơn bản L2 **1,5%** (`ML_DL_VONG7.md` mục 4), rồi **không bao giờ áp cho HAR** | **CÓ** — mục 1 dưới |
| 2 | **Conformal phủ thiếu khi tài khoản đang lỗ** — 89,3% so 90,3% ở đỉnh vốn | `TANG6_HIEU_CHUAN.md`; hướng vá đã ghi nhưng **chưa làm** | **CÓ** — mục 2 |
| 3 | **Đuôi USDJPY/USDCHF trượt VaR/ES ở mức 99%** (4/6 cặp đạt) | `CHISO_DANHGIA.md` mục 5b–5e | có (EVT động), **chưa thử** |
| 4 | **h = 20 hiệu chuẩn kém** — MCE 0,206 | `KEHOACH_2026Q4.md` mục 3.3 | có, kỹ thuật chuẩn |
| 5 | **Đặc trưng ngoại sinh hỏng vì trôi phân phối** | Pha 3B: DGS2 lệch **4,54 độ lệch chuẩn** giữa hai đoạn | có (phát hiện điểm ngắt), chưa thử |

---

## 1. qlikeHAR — khớp bằng chính hàm mất dùng để chấm

**Nguồn:** Puke & Schweikert (2026), *Coherent Forecasting of Realized
Volatility*, **Journal of Forecasting** 45(1). SSRN 5233349.
Mã tái lập của tác giả: `github.com/marius-cp/qlikeHAR_replications`.

**Luận điểm của bài.** Văn liệu **chấm** dự báo biến động bằng QLIKE nhưng lại
**ước lượng** HAR bằng bình phương nhỏ nhất. Đó là một sự không khớp. Khớp lại
cho đúng — ước lượng bằng chính QLIKE — cho *"massive forecast performance
gains"* trên dữ liệu của họ.

Hàm mất của bài (QLIKE bỏ hằng số), lấy đúng từ `qlikeHAR.R`:

```
S(h, y) = log(h) + y/h
```

**Vì sao đáng thử ở đây** — không phải vì bài mới, mà vì repo **đã có bằng
chứng riêng cùng chiều và bỏ lỡ**: LightGBM tối ưu QLIKE trực tiếp hơn bản L2
1,5%, và điều đó chưa bao giờ được áp cho chính HAR sản xuất.

### 1a. Bốn cấu hình — chốt trước ở commit `f0e83d1`

| | mô tả |
|---|---|
| **B0** | mốc sản xuất: OLS trên log-RV, khớp **mỗi phiên**, hiệu chỉnh log-chuẩn |
| **A0** | **y hệt B0** nhưng khớp lại đầu mỗi năm — để **tách bạch** ảnh hưởng của tần suất khớp ra khỏi ảnh hưởng của hàm mất |
| **Q_log** | cùng thiết kế, cùng tần suất khớp như A0, hệ số cực tiểu **QLIKE trực tiếp** với `h = exp(Xβ)` |
| **Q_lvl** | bản sát bài báo: HAR **thang mức** 4 hệ số `[1, RV_d, RV_w, RV_m]`, cực tiểu QLIKE |

Dòng **A0** là dòng quan trọng nhất về mặt phương pháp: không có nó thì mọi
chênh lệch đều không quy được về đâu.

**Chi tiết kỹ thuật đáng ghi.** Với `h = exp(Xβ)` bài toán là **lồi**:

```
∂S/∂β  = X'(1 − y·e^{−Xβ})
∂²S/∂β² = X'diag(y·e^{−Xβ})X        (nửa xác định dương khi y > 0)
```

nên giải bằng **Newton** cho nghiệm toàn cục trong vài vòng lặp, thay vì
Nelder-Mead như mã R của tác giả. Đây đúng là cặp gradient/hessian mà repo
**đã tự dẫn** cho LightGBM — nay dùng lại cho HAR. Có tự kiểm trên dữ liệu mô
phỏng có nghiệm biết trước: lệch hệ số lớn nhất **0,0084**, và QLIKE của
nghiệm Newton ≤ QLIKE của nghiệm OLS — cả hai đạt.

### 1b. Kết quả

| cấu hình | QLIKE kiểm định | so B0 | **QLIKE kiểm tra** | so B0 | DM p |
|---|---|---|---|---|---|
| **B0** mốc sản xuất | **0,1162** | — | 0,1585 | — | — |
| A0 y hệt B0, khớp theo năm | 0,1167 | +0,44% | 0,1588 | +0,20% | 0,0059 |
| **Q_log** QLIKE trực tiếp | 0,1165 | +0,22% | **0,1563** | **−1,40%** | 0,4450 |
| Q_lvl HAR mức (bài báo) | 0,1332 | +14,64% | 0,1744 | +10,02% | 0,0226 |

**Model Confidence Set (α = 0,10) trên kiểm tra: `[B0, Q_log]`** — A0 và Q_lvl
bị loại.

Tách bạch hai nguồn chênh lệch trên đoạn kiểm tra:

| nguồn | độ lớn | |
|---|---|---|
| do **tần suất khớp** (A0 so B0) | +0,20% | khớp theo năm tệ hơn khớp mỗi phiên — khớp với con số 0,5% đã đo ở `ML_DL_VONG7.md` |
| do **hàm mất** (Q_log so A0) | **−1,59%** | DM p = **0,3815** — không có ý nghĩa |

### 1c. Phán quyết: **KHÔNG đưa vào sản xuất.** Ba lý do độc lập

1. **Trượt chính quy tắc chọn của repo.** Chọn phải trên **kiểm định**. Trên
   kiểm định, B0 tốt nhất (0,1162) còn Q_log là 0,1165. Con số −1,40% trên
   kiểm tra là thứ chỉ thấy được **sau khi** nhìn đoạn kiểm tra — không dùng
   để chọn được.
2. **Không có ý nghĩa thống kê.** DM p = 0,4450 so mốc; p = 0,3815 khi so đúng
   dòng đối chứng A0. MCS giữ cả B0 lẫn Q_log trong tập không phân biệt được.
3. **Không rộng.** Chỉ **1/6 cặp** cải thiện. Toàn bộ phần thắng gộp đến từ
   **USDJPY (−6,71%)**; năm cặp còn lại đều tệ hơn (+0,08% đến +2,95%, riêng
   GBPUSD +2,95% với p = 0,0007 — tức **xấu đi có ý nghĩa**).

Và phân tầng theo chế độ cho thấy lợi thế nằm sai chỗ:

| chế độ | B0 | Q_log | chênh |
|---|---|---|---|
| Q1 êm | 0,1809 | 0,1727 | **−4,55%** |
| Q2 | 0,1655 | 0,1596 | −3,58% |
| Q3 | 0,1349 | 0,1331 | −1,31% |
| Q4 | 0,1479 | 0,1532 | +3,58% |
| **Q5 căng** | 0,1502 | 0,1562 | **+4,01%** |

Lợi thế ở chế độ **êm**, thiệt ở chế độ **căng** — đúng chỗ hệ thống quyết định
cần nhất thì nó tệ hơn. Đây là **cùng một hình mẫu** đã thấy ở nhánh học sâu
(`ML_DL_VONG7.md`: *"lợi thế của học sâu không nằm ở đuôi"*), nay lặp lại với
một cơ chế hoàn toàn khác.

### 1d. Về bản sát bài báo (Q_lvl) — phải nói cho công bằng

Q_lvl thua đậm (+10% kiểm tra, bị loại khỏi MCS), **nhưng đó không phải bằng
chứng bác bỏ bài báo.** Q_lvl là HAR **4 hệ số**, còn B0 là tổ hợp
STHARQ+HARQ+SHAR có hiệu chỉnh sai số đo, semivariance, biến chuyển chế độ và
lịch ngân hàng trung ương riêng từng cặp. Một mô hình đơn giản hơn nhiều thua
là điều phải xảy ra. Phép so đúng để đánh giá **ý tưởng** của bài là
**Q_log so A0** — cùng lớp mô hình, cùng tần suất khớp, chỉ khác hàm mất — và
kết quả đó là **−1,59%, p = 0,3815**.

**Ghi lại một lỗi đã mắc trong lúc làm:** bản đầu tôi áp tuyến-tính-thang-mức
lên ma trận thiết kế của repo, vốn đã ở thang **log**. Đó là sai đặc tả và cho
QLIKE ≈ 32.897 (tệ hơn mốc 20 triệu phần trăm). Đã sửa bằng cách dựng ma trận
thiết kế thang mức riêng (`thiet_ke_muc`). Ghi lại để người sau không lặp lại,
và để thấy con số tham hoạ đó **không** phải kết quả của bài báo.

### 1e. Kết luận dùng được cho luận văn

> *Ý tưởng khớp hàm mất cho khớp thước đo — đã được chứng minh trên dữ liệu
> cổ phiếu (Puke & Schweikert 2026) và trên chính cây tăng cường của repo này
> (+1,5%) — **không chuyển giao được sang HAR trên 6 cặp ngoại hối**: chênh
> lệch −1,59% không đạt ý nghĩa (p = 0,38), không chọn được trên đoạn kiểm
> định, và tập trung ở chế độ êm cùng một cặp duy nhất.*

Đây là **kết quả âm có giá trị**: nó thu hẹp phạm vi áp dụng của một phát hiện
2026 vừa công bố, bằng một phép thử có đối chứng tách bạch đúng nguồn hiệu ứng.

Tái lập: `python src/kiem_qlikehar.py` → `output/qlikehar.json`.

---

## 2. Conformal phân tầng theo trạng thái sụt giảm — và một đính chính

**Điểm yếu nhắm tới.** `TANG6_HIEU_CHUAN.md` mục 5 ghi một **giới hạn đã đo**:
mọi cách dựng khoảng đều phủ thiếu khi tài khoản đang lỗ (Conformal 90,3% ở
đỉnh vốn so với 89,3% khi đang lỗ), và *"đây đúng là lúc người dùng cần con số
chính xác nhất"*. Tài liệu đó cũng ghi sẵn hướng vá: **thêm trạng thái sụt
giảm vào biến phân tầng Mondrian** — nhưng chưa bao giờ làm.

`decision_record.py` ghi đã thử **năm cách** (tĩnh, Mondrian 3, cửa sổ trượt,
ACI chung, DtACI, ACI theo tầng) và *"không cách nào xoá được khoảng chênh
đó"*. Nhưng **cả năm đều phân tầng theo biến động** — không cách nào phân tầng
theo chính cái biến mà độ phủ đang lệch trên nó. Đó đúng là cách chữa chuẩn của
Mondrian có điều kiện trong văn liệu conformal (Vovk et al.; và các bản
2025–2026 về *risk-conditional coverage disparity*, ví dụ arXiv 2512.11779).

### 2a. Hai cấu hình mới — chốt trước

| | phân tầng theo |
|---|---|
| **ACI-dd 2** | **chỉ** trạng thái sụt giảm (2 tầng) |
| **ACI-2D 2×2** | (biến động 2 tầng) × (sụt giảm 2 trạng thái) — đúng hướng vá đã ghi |

Ràng buộc chống rò rỉ: biến phân tầng của **mô hình** phải trễ một phiên. Bảng
đánh giá cũ dùng `rolling(20).sum()` không dịch — đúng khi chỉ để **báo cáo**
(nó chỉ là cách nhóm), nhưng sai nếu dùng làm biến phân tầng vì chứa lợi suất
của chính phiên t. Tự kiểm ép: cắt bỏ tương lai đổi **0/3.000** giá trị; đối
chứng dương xác nhận bản dịch khác bản không dịch.

### 2b. Khe đỉnh−lỗ, chấm một lần trên kiểm tra

| phương pháp | khe kiểm định | **khe kiểm tra** |
|---|---|---|
| tĩnh | −0,42% | +0,58% |
| Mondrian 2 | −0,11% | +0,29% |
| Mondrian 3 | −0,05% | +0,37% |
| ACI | +0,46% | +0,57% |
| ACI-tầng 2 | −0,16% | +0,64% |
| ACI-tầng 3 | −0,11% | +0,96% |
| **ACI-dd 2** *(mới)* | +0,36% | **−0,03%** |
| **ACI-2D 2×2** *(mới)* | +0,03% | +0,83% |

Đọc theo điểm ước lượng thì bản vá **có tác dụng đúng như lý thuyết nói**:
phân tầng theo chính biến sụt giảm kéo khe từ 0,29–0,96% về **−0,03%**.

**Nhưng nó đổi một chênh lệch lấy một chênh lệch khác.** ACI-dd 2 bỏ phân tầng
theo biến động, nên độ phủ theo chế độ biến động hỏng đi (vol thấp 88,5%, vol
cao 92,4%) và `|lệch| max` trên kiểm tra thành **2,4% — tệ nhất trong tám
cách**. Trên kiểm định nó cũng tệ nhất (4,1%), nên theo đúng quy tắc chọn của
repo nó **không được chọn**. Bản 2D thì không đóng được khe (+0,83%).

### 2c. ĐÍNH CHÍNH — khe đó chưa bao giờ phân biệt được với 0

Dấu của khe **đảo chiều** giữa kiểm định và kiểm tra ở phần lớn các cách
(ACI-tầng 3: −0,11% → +0,96%; tĩnh: −0,42% → +0,58%). Đó là dấu hiệu của nhiễu,
không phải của một hiệu ứng. Nên phải kiểm — bootstrap **khối 20 phiên**,
2.000 lần, gộp 6 cặp:

| phương pháp | đoạn | khe | KTC 95% | phủ 0? |
|---|---|---|---|---|
| ACI-tầng 3 | kiểm định | −0,23% | [−2,42%; +1,86%] | **có** |
| ACI-tầng 3 | kiểm tra | +0,95% | [−0,89%; +2,81%] | **có** |
| Mondrian 2 | kiểm định | −0,28% | [−2,48%; +1,82%] | **có** |
| Mondrian 2 | kiểm tra | +0,25% | [−1,69%; +2,14%] | **có** |
| ACI-dd 2 | kiểm định | +0,31% | [−1,89%; +2,26%] | **có** |
| ACI-dd 2 | kiểm tra | +0,02% | [−1,81%; +1,91%] | **có** |
| ACI-2D 2×2 | kiểm định | +0,19% | [−1,74%; +2,09%] | **có** |
| ACI-2D 2×2 | kiểm tra | +0,81% | [−1,01%; +2,71%] | **có** |

**Không một cấu hình nào có khe phân biệt được với 0.** Khoảng tin cậy rộng
khoảng ±2 điểm phần trăm, trong khi khe được báo cáo chỉ 0,6–0,8 điểm.

> **Hệ quả: "giới hạn đã đo" ghi ở `TANG6_HIEU_CHUAN.md` mục 5 nằm TRONG
> NHIỄU.** Con số 90,3% so 89,3% là thật với tư cách một số đo mẫu, nhưng nó
> không đủ để phát biểu rằng hệ thống phủ thiếu khi tài khoản đang lỗ.

Và điều này **giải thích luôn** vì sao năm cách trước đều "không xoá được
khoảng chênh": không có khoảng chênh nào để xoá.

### 2d. Phán quyết

| | |
|---|---|
| Đổi cấu hình sản xuất? | **KHÔNG.** Cả hai cấu hình mới đều trượt quy tắc chọn trên kiểm định, và bản đóng được khe thì làm hỏng độ phủ theo chế độ biến động |
| Giữ cảnh báo trên giao diện? | **KHÔNG nên giữ nguyên dạng cũ** — nó công bố một hiệu ứng chưa chứng minh được |
| Thu được gì | **một món nợ tài liệu được trả, và một "giới hạn đã biết" được rút lại có bằng chứng** |

Phát biểu đúng để đưa vào luận văn:

> *Độ phủ conformal đo được ở trạng thái đang lỗ thấp hơn ở đỉnh vốn khoảng
> 0,3–1,0 điểm phần trăm tuỳ cấu hình, nhưng khoảng tin cậy bootstrap khối
> rộng ±2 điểm và phủ 0 ở mọi cấu hình. Với cỡ mẫu hiện có, **không kết luận
> được** rằng độ phủ phụ thuộc trạng thái vốn. Phân tầng Mondrian theo chính
> biến sụt giảm kéo được điểm ước lượng về 0 nhưng đánh đổi bằng độ phủ theo
> chế độ biến động, nên không được chọn.*

Tái lập: `python src/kiem_conformal_dd.py` → `output/conformal_dd.json`.

---

## 3. Tổng kết hai hướng đã thử

| hướng | nguồn | kết quả | có tích hợp không |
|---|---|---|---|
| **qlikeHAR** — khớp bằng chính hàm mất dùng để chấm | Puke & Schweikert 2026, *J. Forecasting* | −1,59% nhưng **p = 0,38**, không chọn được trên kiểm định, 1/6 cặp, lợi thế nằm ở chế độ êm | **không** |
| **Conformal phân tầng theo sụt giảm** | Mondrian có điều kiện; văn liệu 2025–2026 về risk-conditional coverage | đóng được khe (−0,03%) nhưng hỏng độ phủ theo biến động; **và khe vốn không phân biệt được với 0** | **không** — nhưng **rút lại được một "giới hạn đã biết"** |

Cả hai đều là **kết quả âm có đối chứng**, và cả hai đều thu được thứ dùng
được cho luận văn: một cái thu hẹp phạm vi áp dụng của một phát hiện 2026 vừa
công bố; một cái đính chính chính tài liệu của dự án.

**Còn chưa thử** (xếp theo mức hứa hẹn): EVT động cho đuôi USDJPY/USDCHF
(mục 0 dòng 3) · hiệu chuẩn lại cho h = 20 (dòng 4) · phát hiện điểm ngắt để
đặt lại cửa sổ hiệu chuẩn khi biến ngoại sinh trôi (dòng 5).

---

## 4. Mẫu hình nến — lỗ hổng thật, và một ứng viên bất ngờ

*Tiêu chí chốt trước: `docs/NEN_TIEUCHI.md` (commit `9643f80`).
Tái lập: `python src/kiem_nen.py && python src/kiem_nen_ablation.py`.*

### 4a. Vì sao đây không phải việc lặp lại

Repo đã khai phá 8.652 giả thuyết qua 12 nhánh, nhưng **mọi nhánh đều dùng
chuỗi đóng-đóng hoặc đại lượng từ nến 5 phút**. Hình học OHLC của nến **ngày**
chưa bao giờ vào mô hình nào: bốn cột `open/high/low/close` có trong bảng sản
xuất nhưng `volfc2.thiet_ke` không dùng cột nào. Grep toàn repo cho
`doji|hammer|engulf|harami|candlestick`: **một lần duy nhất**, trong đặc tả
*vẽ* biểu đồ.

Văn liệu cho hai tín hiệu ngược chiều, và chúng định hình phép thử:

- **Trục hướng — đồng thuận ÂM.** Orquín-Serrano (2020), *Mathematics*
  8(5), art. 802, đúng miền FX/EURUSD: *không có lợi suất dương ròng trong bất kỳ
  trường hợp nào sau chi phí*. Phân tích 02/2026: không phân biệt được với vào
  lệnh ngẫu nhiên.
- **Nhưng** văn liệu 2026 về học sâu trên nến nhấn rằng **bóng nến mang thông
  tin biến động trong thanh** — đúng trục mà repo *có* kỹ năng.

25 đặc trưng, 4 họ, × 2 lag × 6 cặp × 2 trục = **600 phép kiểm**, liệt kê đầy
đủ trước. Tự kiểm cắt tương lai: **0/1.011.600** giá trị đổi.

### 4b. Kết quả phễu — tách đôi rất sạch

| họ | phép kiểm | sống sót W-Y | p W-Y nhỏ nhất | F lớn nhất |
|---|---|---|---|---|
| K1 hình học nến (thân, bóng, vị trí đóng) | 120 | **0** | 0,9351 | 8,16 |
| K2 phạm vi−RV (Parkinson/GK/RS so `rv5`) | 72 | **0** | 0,2767 | 12,54 |
| **K3 mẫu hình có tên** (doji, hammer, engulfing…) | 360 | **0** | 0,5075 | 10,93 |
| **K4 gap qua đêm** | 48 | **7** | **0,0010** | **67,84** |

Theo trục: biên độ **7/300** · **hướng 0/300**, p W-Y tốt nhất **0,9950**.

> **Mẫu hình nến cổ điển chết sạch** — 0/360, trên cả hai trục, với 15 mẫu và
> số lần khớp đủ lớn (doji 1.220 lần, engulfing ~1.750, con quay 4.490). Đây là
> nhánh thứ 13 của dự án cho kết quả âm trên trục hướng, và lần này nó **khớp
> đúng đồng thuận văn liệu** thay vì đi ngược.

**Cả bảy cái sống sót đều là `abs_gap`** — độ lớn gap qua đêm — trên trục biên
độ, dấu **dương**, ở 5/6 cặp.

### 4c. Vì sao `|gap|` lại có thông tin — cơ chế đã khai báo TRƯỚC

`DATASET.md` ghi rõ, từ lâu:

> *"RV = tổng bình phương lợi suất **trong ngày**, bỏ lợi suất bắc qua ranh giới
> ngày… gap qua đêm chỉ chiếm **1,7–3,1%** tổng phương sai ở FX nên **bỏ qua
> được**."*

Đó là một **giả định chưa ai kiểm**. Vì `rv5` loại gap theo thiết kế, HAR
**không bao giờ nhìn thấy** đại lượng đó — nên nó là thông tin ngoài tập thông
tin của tầng 2, chứ không phải một biến đổi của thứ tầng 2 đã có. Khác hẳn
`hhi` ở `PHA2_KETQUA.md` mục 3d, vốn tương quan +0,742 với tỉ trọng nhảy mà HAR
đã có sẵn.

Hệ số ước lượng: **+50,56** (cấu hình `chỉ |gap| L1`) — **dương**, đúng dấu đã
khai báo ở mục 0b cơ chế 2.

### 4d. Ablation — và cái bẫy lặp lại lần thứ ba

| cấu hình | #đt | QLIKE kiểm định | so B0 | **QLIKE kiểm tra** | so B0 | DM p |
|---|---|---|---|---|---|---|
| **B0 mốc** | 0 | 0,1569 | — | 0,1872 | — | — |
| K1 hình học | 10 | 0,1571 | +0,11% | 0,1871 | −0,09% | 0,7262 |
| **K2 phạm vi−RV** | 6 | **0,1562** | **−0,48%** | 0,1877 | **+0,25%** | 0,1752 |
| K3 mẫu có tên | 30 | 0,1578 | +0,56% | 0,1884 | **+0,60%** | **0,0451** |
| K4 gap đêm | 4 | 0,1569 | −0,00% | 0,1860 | −0,64% | 0,2232 |
| **chỉ \|gap\| L1** | **1** | 0,1571 | +0,11% | **0,1860** | **−0,67%** | **0,0445** |
| K1+K2+K4 | 20 | 0,1565 | −0,31% | 0,1869 | −0,21% | 0,7169 |
| tất cả 25 | 50 | 0,1568 | −0,08% | 0,1875 | +0,12% | 0,8307 |

**MCS (α = 0,10): `[B0, K1, K4, chỉ |gap| L1, K1+K2+K4]`** — K2, K3 và "tất cả
25" bị **loại**.

Hai điều đáng đọc kỹ:

1. **K3 làm mô hình xấu đi CÓ Ý NGHĨA** (+0,60%, p = 0,0451). Mẫu hình nến cổ
   điển không chỉ vô dụng — đưa vào là có hại. Đây cũng là đối chứng cho thấy
   bộ máy không dễ dãi: nếu nó "cải thiện" cả K3 thì phải nghi ngờ.
2. **Quy tắc chọn trên kiểm định lại chọn sai.** Nó chọn **K2** (tốt nhất trên
   kiểm định, −0,48%) — và K2 thua trên kiểm tra (+0,25%) **và bị MCS loại**.
   Trong khi thứ sống sót phễu (`|gap|`) thì thắng kiểm tra và nằm trong MCS.

> Đây là **lần thứ ba trong cùng một phiên** hình mẫu này lặp lại: Pha 3B (E2
> chọn-trên-kiểm-định thua E3 lọc-nhân-quả, p = 0,0010) · qlikeHAR (kiểm định
> chọn B0, Q_log tốt hơn trên kiểm tra) · và ở đây. **Chọn theo hiệu năng kiểm
> định thua lọc theo bằng chứng cấu trúc** — ba lần, ba cơ chế độc lập.

### 4e. `|gap|` vững tới đâu

| phép kiểm | kết quả |
|---|---|
| Theo cặp (kiểm tra) | **5/6 cải thiện** — chỉ USDCAD xấu hơn (+0,32%) |
| Walk-forward, khớp lại đầu mỗi năm | tốt hơn **7/11 năm**, trung vị **−0,74%** |
| Bỏ 1% phiên có gap lớn nhất | vẫn thắng: **−0,36%, p = 0,0411** — không phải do vài ngày ngoại lai |
| Dấu hệ số | **+50,56**, đúng cơ chế chốt trước |
| MCS | **nằm trong tập** |
| Số tham số | **1** — gần như không có chỗ để quá khớp |

### 4f. Phán quyết: **KHÔNG đổi sản xuất** — nhưng đây là ứng viên mạnh nhất từ trước tới nay

| điều kiện chốt trước | kết quả |
|---|---|
| ĐK1 có đặc trưng qua W-Y + độ vững | **ĐẠT** (7/600) |
| ĐK2 cấu hình **được chọn** thắng kiểm tra, p < 0,025 | **TRƯỢT** (K2: +0,25%, p = 0,1752) |
| ĐK3 ≥ 5/6 cặp cải thiện | **TRƯỢT** cho cấu hình được chọn (2/6) |
| ĐK4 dấu hệ số \|gap\| dương | **ĐẠT** (+50,56) |

Theo đúng luật, phán quyết là **chưa đủ điều kiện dương** → không đổi cấu hình
sản xuất. Không được sửa luật sau khi thấy số.

**Nhưng phải ghi thẳng, vì nó đúng:** nếu xét riêng cấu hình `chỉ |gap| L1` —
thứ duy nhất sống sót phễu — thì nó đạt ĐK1, ĐK3 (5/6), ĐK4, và **chỉ trượt
ĐK2 vì p = 0,0445 so ngưỡng Bonferroni 0,025**. Nó thắng ở p < 0,05 thô, vững
qua bốn lát cắt, và chỉ tốn **một tham số**.

> Đây là **ứng viên cải thiện tầng 2 mạnh nhất mà dự án từng tạo ra** — mạnh
> hơn tổ hợp HAR+GRU+CatBoost (−3,0% nhưng MCS giữ HAR), mạnh hơn qlikeHAR
> (p = 0,38), và là cái duy nhất đi kèm một cơ chế được nêu trước, kiểm trước,
> và có dấu đúng.

**Việc đúng phải làm tiếp**, không phải đổi sản xuất ngay: đưa `|gap|` vào danh
sách chốt của **lần mở tập niêm phong cuối cùng** (`KHOA_SO.md`). Nó là đúng
loại giả thuyết mà tập niêm phong tồn tại để trả lời — một biến, một cơ chế, đã
khai báo trước, chưa hề chạm dữ liệu niêm phong.

Và bất kể quyết định mô hình ra sao, `DATASET.md` phải sửa một câu: gap qua đêm
chiếm 1,7–3,1% phương sai, **nhưng "bỏ qua được" thì chưa chứng minh** — độ lớn
của nó dự báo được biến động phiên kế tiếp vượt trên HAR, F tới 67,84.
