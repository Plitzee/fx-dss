# ĐUÔI TRỰC TUYẾN (ACI/AgACI) — HƯỚNG THỨ TÁM. Tiêu chí CHỐT TRƯỚC

Lập 15/09/2026. Nhánh `replan-2026`. **Commit chứa văn bản này nằm trước mọi
commit có số liệu.**

---

## 1. Khiếm khuyết nhắm tới — và vì sao bảy hướng trước đều chết vì cùng một lý do

Bảy hướng độc lập đã thử cho đuôi VaR/ES của USDJPY/USDCHF:

| # | hướng | cơ chế | kết quả |
|---|---|---|---|
| 1 | V1 mở rộng / V2 cuộn | đổi cửa sổ ước lượng | thua V0 trên kiểm tra |
| 2 | phân vị theo chế độ | phân tầng theo σ̂ | kém hơn mốc |
| 3 | CAViaR (4 biến thể) | tự hồi quy tham số trên quantile | thua ngay trên kiểm định |
| 4 | cửa sổ họp NHTW định kỳ | phân tầng theo lịch | không thắng kiểm định |
| 5 | EVT/POT (4 biến thể) | GPD cho phần vượt ngưỡng | vá được **tần suất**, DQ vẫn bác bỏ |
| 6 | lịch can thiệp không định kỳ | nới ngưỡng 1,5×/2×/3× | thất bại |
| 7 | GAS/DCS Beta-t-EGARCH | quy mô động theo score | *chốt trước, chưa chấm* |

`TONG_QUAN_CHO_AI_KHAC.md` mục 5.1 đã tự chẩn đoán đúng nguyên nhân chung, và
nó **không phải** vấn đề ước lượng đuôi:

> Vấn đề tail-risk chỉ lộ ra ở đoạn KIỂM TRA, KHÔNG BAO GIỜ lộ ra ở đoạn KIỂM
> ĐỊNH. Mọi cấu hình đã thử — kể cả chính mốc V0 đang chạy — đều "ĐẠT" ba phép
> kiểm trên kiểm định. Vì quy tắc dự án là "chọn cấu hình dựa trên hiệu năng đo
> được trên kiểm định", KHÔNG có tín hiệu nào trên dữ liệu được phép dùng để
> phân biệt cách vá nào thật sự sửa được lỗi.

`DUOI_EVT.md` A4(h) đo được hệ quả của đúng điều đó: **lần thứ tư
chọn-trên-kiểm-định chọn sai** — E2 nhất trên kiểm định nhưng E3 mới là cấu hình
vá được USDJPY.

**Bảy hướng trên đều thuộc cùng một lớp: mỗi hướng đề xuất k cấu hình, rồi dùng
đoạn kiểm định để chọn một.** Khi bản thân đoạn kiểm định không chứa thông tin
phân biệt, mọi hướng trong lớp đó đều thất bại — bất kể ước lượng đuôi tinh vi
tới đâu. Đó là lý do hướng thứ tám phải đổi **lớp**, không phải đổi ước lượng.

## 2. Vì sao ACI khác hẳn — nó bỏ luôn bước chọn-trên-kiểm-định

**Adaptive Conformal Inference** (Gibbs & Candès 2021, *Adaptive conformal
inference under distribution shift*, NeurIPS 34:1660–1672) không ước phân phối
đuôi. Nó là một **vòng điều khiển phản hồi** trên chính mức vi phạm:

```
α_{t+1} = α_t + γ·(α_mục_tiêu − err_t)        err_t = 1{z_t ≤ q_t}
q_t     = phân vị mức α_t của z trong cửa sổ hiệu chuẩn
```

Tính chất quyết định (Gibbs & Candès 2021, Mệnh đề 4.1): cộng dồn phép cập
nhật cho `(1/T)·Σ err_t − α = (α_1 − α_{T+1})/(T·γ)`, nên

```
| (1/T)·Σ_t err_t  −  α |  ≤  (max{α_1, 1−α_1} + γ) / (T·γ)   →  0  khi T → ∞
```

**với MỌI chuỗi phân phối, kể cả đối kháng** — không giả định dừng, không giả
định i.i.d., không giả định hoán vị được. Đây là **định lý**, không phải kết
quả thực nghiệm.

Hệ quả đúng vào chỗ đang kẹt: **không cần đoạn kiểm định để biết phương pháp có
hợp lệ hay không.** Bảo đảm không đến từ việc nó "đạt backtest trên kiểm định";
nó đến từ chứng minh. Vì vậy hướng này **không có bước chọn cấu hình trên kiểm
định** — thứ đã làm hỏng bốn lần trước.

### 2a. Khác CAViaR ở đâu (hướng 3, đã thất bại)

Cả hai đều cập nhật quantile theo kết cục quá khứ, nên phải nói rõ khác biệt
**trước** khi chạy:

| | CAViaR | ACI |
|---|---|---|
| cập nhật theo | độ lớn residual thô `\|r_{t-1}\|` | **chỉ dấu hiệu nhị phân** `1{vi phạm}` |
| dạng | đặc tả tham số tuỳ ý (SAV/AS/IG) | tích phân sai số độ phủ (khâu I của bộ điều khiển) |
| khớp bằng | hồi quy quantile, MLE trên mẫu | **không khớp gì** — một tham số γ chốt trước |
| tính hợp lệ | phải kiểm bằng backtest | **định lý, không điều kiện phân phối** |
| hỏng khi | đặc tả sai | không hỏng theo nghĩa độ phủ dài hạn |

CAViaR hỏng vì nó vẫn là một mô hình phải đúng. ACI không mô hình hoá đuôi.

### 2b. Đã có sẵn trong repo — nhưng đặt sai tầng

`src/conformal.py::chay_aci` đã cài đúng vòng lặp này và **đang chạy sản xuất**
cho tập dự báo 90% của tầng ba xác suất, với kết quả đã đo: độ phủ 0,901–0,908
so với 0,814–1,000 của conformal tĩnh (`KHOA_SO.md` dòng 24, thay đổi (1)).

Tức cơ chế này **đã được chứng minh hoạt động trên chính dữ liệu này**, chỉ chưa
bao giờ được áp cho tầng VaR/ES. Tầng VaR/ES vẫn đang dùng V0 — phân vị thực
nghiệm tĩnh ước trên huấn luyện, đóng băng từ 2021. Đây là lỗ hổng thật, không
phải lặp lại: đã xác minh bằng grep toàn repo, không dòng mã nào nối ACI với
`va_duoi*`.

### 2c. AgACI — bỏ nốt tham số γ

Zaffran, Féron, Goude, Josse & Dieuleveut (2022), *Adaptive conformal
predictions for time series*, ICML 162:25834–25866: chạy nhiều γ song song rồi
**tổ hợp trực tuyến** bằng trọng số mũ, thay vì chọn một γ. Dự án đã có sẵn cơ
chế này ở tầng khác (Hedge trơn, `balop.py`, đang chạy sản xuất cho h=1). Biến
thể AgACI vì vậy **không có một siêu tham số nào phải chọn** — mức độc lập cao
nhất với đoạn kiểm định mà hướng này có thể đạt.

## 3. Giả thuyết CHỐT TRƯỚC — và ranh giới giữa ĐỊNH LÝ và PHỎNG ĐOÁN

Phải tách bạch, vì trộn hai thứ này lại chính là lỗi đã khai ở `DUOI_EVT.md`
A5a (tiêu chí không có ngưỡng độ lớn) và `RUIRO_ML.md` A1a.

> **H17a — hệ quả của định lý, KHÔNG phải phỏng đoán.** Tỷ lệ vi phạm của ACI
> hội tụ về α trên **mọi** đoạn, nên **Kupiec** không bị bác bỏ ở ≥ 5/6 cặp tại
> cả hai mức α, ở **cả** kiểm định lẫn kiểm tra. Nếu điều này sai thì hoặc cài
> đặt sai, hoặc T quá ngắn so với cận `(α_1+γ)/(T·γ)` — phải phân biệt được hai
> khả năng đó bằng tự kiểm ở mục 6.
>
> **H17b — PHỎNG ĐOÁN, có thể sai.** Định lý ACI bảo đảm **tần suất**, KHÔNG
> bảo đảm **tính độc lập** của chuỗi vi phạm. Phỏng đoán ở đây là cơ chế phản
> hồi (sau một vi phạm, `α_t` giảm → quantile nới ra → vi phạm kế tiếp ít khả
> năng hơn) sẽ **giảm** hiện tượng vi phạm dồn cụm mà DQ đang bắt được, nên
> **DQ không bác bỏ ở ít nhất một trong hai** USDJPY/USDCHF.
>
> **Cảnh báo tự đặt ra trước:** cùng cơ chế đó cũng có thể tạo tự tương quan
> **âm** nhân tạo trong chuỗi vi phạm, và DQ bắt cả hai chiều. Nếu DQ bị bác bỏ
> **vì** tự tương quan âm, phải ghi nhận là **thất bại của H17b**, không được
> đọc thành "gần đạt". Dấu của hệ số DQ phải được báo cáo, không chỉ p.

**H17b là mục tiêu chính của hướng này** — cùng mục tiêu với hướng 7 (GAS), vì
đó là khiếm khuyết duy nhất còn mở sau EVT.

## 4. Năm cấu hình — liệt kê đầy đủ TRƯỚC khi chạy

| | cơ chế | tham số | vai trò |
|---|---|---|---|
| **A1** | ACI trên mức α (đọc phân vị thực nghiệm ở `α_t`) | **γ = 0,01** | **chính** |
| **A2** | bộ bám quantile trực tiếp (gradient hàm mất pinball) | η = 0,01·sd(z) huấn luyện | chính |
| **A3** | **AgACI** — Hedge trên γ ∈ {0,002; 0,005; 0,01; 0,02; 0,05} | không có | chính |
| A4 | như A1 | γ = 0,005 | **độ nhạy** |
| A5 | như A1 | γ = 0,02 | **độ nhạy** |

**γ = 0,01 của A1 KHÔNG được chọn bằng cách dò trên dữ liệu.** Nó là đúng giá
trị mặc định mà `conformal.py::chay_aci` đang chạy sản xuất từ 09/09/2026
(`KHOA_SO.md` dòng 24), kế thừa nguyên trạng. A4/A5 chỉ để **báo cáo độ nhạy**,
và theo mục 5 chúng **không được phép** trở thành cấu hình đề xuất.

Cửa sổ hiệu chuẩn cho A1/A4/A5: 500 phiên gần nhất (bằng `CUON` của V2 và bằng
`cua_so` mặc định của `chay_aci` sản xuất). ES ước bằng trung bình thực nghiệm
của z dưới `q_t` trên cùng cửa sổ.

**Đếm vào `KHOA_SO.md`: 5 cấu hình mới.**

## 5. Giao thức chấm — điểm khác biệt quan trọng nhất của hướng này

Mọi hướng trước: đề xuất k cấu hình → chọn 1 trên kiểm định → chấm kiểm tra.
**Hướng này KHÔNG làm bước giữa.**

1. **A1 là cấu hình đề xuất, chốt ngay tại văn bản này**, trước khi thấy bất kỳ
   con số nào. A2 và A3 là hai cơ chế thay thế đã khai báo trước, được chấm và
   báo cáo song song — **không** để chọn lại.
2. Đoạn **kiểm định** dùng cho đúng một việc: **kiểm chứng cài đặt** (mục 6) và
   đối chiếu độ phủ thực nghiệm với cận lý thuyết. Nó **không** được dùng để
   chọn giữa A1/A2/A3, cũng không để chọn γ.
3. Đoạn **kiểm tra** chấm **một lần**, bằng cờ tường minh `--mo-kiem-tra`, với
   A1 là cấu hình đề xuất đã chốt.
4. Nếu A2 hoặc A3 tốt hơn A1 trên kiểm tra, điều đó được **báo cáo như một quan
   sát**, không được đọc ngược thành "đáng lẽ nên chọn A3" và **không** đổi sản
   xuất sang A3 trong cùng lần chạy này.

Lý do quy tắc 4 phải viết ra trước: `DUOI_EVT.md` A4(h) cho thấy khi thấy E3 vá
được USDJPY còn E2 thì không, sức ép đọc ngược là rất lớn. Quy tắc này chặn
trước đúng sức ép đó.

## 6. Chống rò rỉ và tự kiểm bắt buộc

**Tự kiểm 1 — nhân quả.** `q_t` chỉ được dùng `z_1..z_{t-1}`. Cắt toàn bộ dữ
liệu sau một mốc không được làm đổi `q_t` tại mọi phiên trước mốc đó.

**Tự kiểm 2 — định lý đúng trên dữ liệu mô phỏng.** Sinh chuỗi có **dịch chuyển
phân phối biết trước** (đổi thang đo giữa chừng, và một đoạn t-3 bậc tự do).
Kiểm rằng độ phủ dài hạn của ACI hội tụ về α trong khi phân vị tĩnh thì không.
Nếu tự kiểm này hỏng thì dừng, không chấm số liệu thật.

**Tự kiểm 3 — cận lý thuyết.** Kiểm `|độ phủ − α| ≤ (α_1+γ)/(T·γ)` trên chính
đoạn mô phỏng. Đây là phép phân biệt "cài sai" với "T quá ngắn" mà H17a đòi.

**Cắt niêm phong.** `noi_chuoi()` nối dữ liệu live tới 2026-09; cắt tại
**2025-12-31** trước khi chấm (`KHOA_SO.md` mục 2). Đã đo ngày 15/09/2026:
không cắt thì **179–180/729 phiên (24,6%)** của đoạn kiểm tra là dữ liệu niêm
phong — việc tồn đã khai ở `RUIRO_ML.md` A4, nay đã vá ở `va_duoi.py::nap()`.

## 7. TIÊU CHÍ PHỦ ĐỊNH — có ngưỡng độ lớn, học từ hai lỗi đã khai

`DUOI_EVT.md` A5a và `RUIRO_ML.md` A1a đều khai báo cùng một lỗi: tiêu chí viết
"tốt hơn mốc" mà không có ngưỡng độ lớn, khiến cải thiện bằng nhiễu cũng "đạt".
Tiêu chí dưới đây có ngưỡng.

Kết luận **"ACI không mang lại giá trị đo được"** khi bất kỳ điều nào đúng:

1. **H17a sai**: Kupiec bị bác bỏ ở > 1/6 cặp tại bất kỳ mức α nào trên kiểm
   định, **và** tự kiểm 3 cho thấy T đủ dài so với cận. (Đây là dấu hiệu cài
   sai — phải dừng và sửa code, không phải kết quả nghiên cứu.)
2. **H17b sai**: DQ vẫn bác bỏ (p < 0,05) ở **cả** USDJPY **và** USDCHF trên
   kiểm tra.
3. A1 làm hỏng ≥ 1 ô mà V0 đang đạt trên kiểm tra.
4. Tỷ lệ ES của A1 lệch khỏi 1 quá **0,10** (độ lớn tối thiểu — V0 hiện lệch
   trung bình 0,0695 trên kiểm định) ở ≥ 3/12 ô.

Kết luận **dương** đòi **đồng thời**: (a) H17a đúng trên cả hai đoạn; (b) DQ
không bác bỏ ở ít nhất một trong hai USDJPY/USDCHF trên kiểm tra, **và** hệ số
DQ không cho thấy tự tương quan âm nhân tạo (cảnh báo ở mục 3); (c) không làm
hỏng ô nào V0 đang đạt; (d) tiêu chí 4 không kích hoạt.

Không sửa tiêu chí sau khi thấy số.

## 8. Lực phát hiện — khai báo TRƯỚC

~729 phiên mỗi đoạn sau khi cắt niêm phong (đo 15/09/2026), tức ~7 vi phạm kỳ
vọng ở α = 1% và ~36 ở α = 5%.

- **Kupiec**: lực thấp ở α=1% — đã biết, và là lý do H17a đòi ≥5/6 cặp thay vì
  từng cặp riêng lẻ.
- **DQ**: dùng toàn bộ chuỗi vi phạm nên có lực cao hơn cho đúng câu hỏi "có
  động học không" — đây là lý do DQ là mục tiêu chính.
- **Cận ACI**: với γ=0,01 và T=729, cận là (0,99+0,01)/(729·0,01) = **0,137** ở
  α=1%, và (0,95+0,01)/(729·0,01) = **0,132** ở α=5%. Cả hai **lỏng hơn chính
  α**. Nói thẳng: trên một đoạn 729 phiên, định lý ACI **chưa siết được gì** —
  bảo đảm là tiệm cận, và T ở đây chưa đủ để cận có nghĩa.

**Hệ quả phải ghi trước, không được lờ đi:** nếu ACI đạt ở α=1%, phần đóng góp
của **định lý** là nhỏ; phần lớn đến từ **hành vi hữu hạn mẫu** của vòng phản
hồi. Vì vậy mục 2 **không** được trích dẫn như "đã chứng minh sẽ đạt trên dữ
liệu này"; nó chỉ chứng minh phương pháp không cần đoạn kiểm định để **hợp lệ
hoá**, và đó vẫn là điểm mới thật so với bảy hướng trước.

### 8a. KHAI BÁO SỬA CÔNG THỨC CẬN — 15/09/2026, trước mọi số liệu thật

Bản đầu của mục 2 và mục 8 viết cận là `(α₁ + γ)/(T·γ)`, cho 0,274 ở α=1%. Đó
là bản **một phía**, chỉ đúng cho chiều thiếu phủ. Cận hai phía đúng dùng
`max{α₁, 1−α₁}` vì `α_{T+1}` có thể trôi về phía trên.

Lỗi bị **tự kiểm 3 bắt trên dữ liệu mô phỏng**, trước khi chạm bất kỳ số liệu
thật nào — nên đây là sửa văn bản chốt trước ở trạng thái còn hợp lệ, không
phải sửa tiêu chí sau khi thấy số. Kết luận định tính của mục 8 **không đổi**
và trên thực tế còn mạnh hơn: cận lỏng hơn α ở **cả hai** mức, không chỉ mức 1%.

**Giả thiết bị vi phạm, phải ghi kèm.** Phép dẫn giả định `α_t` không bị chặn.
Cài đặt này — và cả `conformal.py::chay_aci` đang chạy sản xuất — chặn `α_t`
vào [10⁻⁴, 1−10⁻⁴] vì không lấy được phân vị ở mức âm. Khi `α_t` chạm sàn,
đẳng thức cộng dồn không còn đúng chính xác. Vì vậy mọi dòng "trong cận" phải
đọc kèm **tỷ lệ phiên `α_t` chạm sàn**, và con số đó được báo cáo cùng kết quả.
Đo trên mô phỏng của tự kiểm 3: chạm sàn 0,1% phiên ở α=5% và 0,5% ở α=1%.

## 9. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 3 (ranh giới định lý/phỏng
đoán, đặc biệt cảnh báo tự tương quan âm), mục 5 (giao thức không chọn trên
kiểm định, đặc biệt quy tắc 4), mục 7 (tiêu chí phủ định) và mục 8 (khai báo
lực, đặc biệt đoạn về cận lỏng) **trước khi** số liệu được chấm:

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không phải
> công cụ điền hộ.
