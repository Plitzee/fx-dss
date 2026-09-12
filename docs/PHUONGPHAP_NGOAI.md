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

*Mục 2 (conformal có điều kiện theo trạng thái sụt giảm) — xem bên dưới khi chạy xong.*
