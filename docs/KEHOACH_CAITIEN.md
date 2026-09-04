# Kế hoạch cải tiến — lập 05/09/2026

Viết sau khi giai đoạn 0–2 đã chạy xong, giao diện đã lên production, và phễu
khai phá đã được **kiểm chứng bằng đối chứng âm/dương**. Mọi việc dưới đây đều
kèm con số đã đo làm lý do, không có việc nào đưa vào vì "nên có".

---

## 0. Ràng buộc quyết định toàn bộ thứ tự

`docs/KHOA_SO.md` mục 3 quy tắc 1–2:

> Tập khoá sổ được chạy **đúng một lần**, sau khi cấu hình cuối đã chốt và ghi
> vào mục 4. Nếu chạy rồi mà kết quả xấu, **không được** quay lại sửa mô hình
> rồi chạy lại.

Nên kế hoạch có một hình dạng bắt buộc:

```
mọi thay đổi mô hình  →  CHỐT CẤU HÌNH  →  mở niêm phong (một lần)  →  hết
```

Không có đường quay lại. Mọi việc ở Đợt A và B phải xong **trước** Đợt C.
Mục 4 của biên bản khoá sổ hiện **đang để trống** — đó là việc phải làm, không
phải thủ tục hình thức.

---

## Đợt A — vá những chỗ đã biết hỏng

Ba việc này đều có bằng chứng đo được, và đều nằm ở phần *sản phẩm* chứ không
phải phần *nghiên cứu*.

### A1. Đuôi dưới của USDJPY — ưu tiên cao nhất

**Bằng chứng.** Ba phép đo độc lập cùng chỉ một chỗ:

| phép đo | USDJPY | mong đợi |
|---|---|---|
| vi phạm VaR 1% | **2,07%** | 1,0% |
| Kupiec p / DQ p | **0,011 / 0,000** | ≥ 0,05 |
| tỷ lệ ES (dự báo / thực tế) | **0,799** | 1,00 |
| PIT-KS p | **0,0036** | ≥ 0,05 |
| độ phủ 90% | **85,2%** | 90% |

Nghĩa là: ngày rất xấu đến **gấp đôi** mức hệ thống công bố, và khi nó đến thì
lỗ thực **sâu hơn 25%** so với ES dự báo. Đây là một **hệ phân tích rủi ro**;
con số rủi ro sai cho một trong sáu cặp là lỗi nặng nhất đang tồn tại.

**Chẩn đoán.** `var_es` đã dùng phân vị **thực nghiệm** của z riêng từng cặp
(không giả định chuẩn) và ν đã ước riêng từng cặp (8,23 cho JPY so với 11,8 cho
EUR). Vậy vấn đề **không** ở dạng phân phối mà ở **σ̂ hụt**: độ phủ 85,2% nói
σ̂ đánh giá thấp biến động JPY một cách hệ thống.

**Ba hướng, phải chọn trên đoạn kiểm định:**
1. phân vị z **theo chế độ** riêng từng cặp (giống `SigmaCheDo` nhưng cho đuôi)
2. thành phần nhảy giá riêng cho JPY — can thiệp và tháo carry tạo nhảy, mà
   bipower/jump hiện có trong HAR chưa hấp thụ hết
3. hệ số nở đuôi hiệu chuẩn trên huấn luyện, đơn giản nhất, dễ biện minh nhất

**Đo bằng gì:** chính `var_es` — Kupiec, Christoffersen, DQ, tỷ lệ ES. Đạt là
cả ba p ≥ 0,05 và tỷ lệ ES ∈ [0,9; 1,1] trên đoạn kiểm định.

**Ghi chú USDCHF.** Cũng trượt (Kupiec 0,024, DQ 0,012) nhưng **theo hướng
ngược**: tỷ lệ ES 1,150, tức ES *thừa*. Đuôi của nó bị chi phối bởi đúng một
biến cố — SNB bỏ neo 15/01/2015, độ lệch z trên huấn luyện −5,61 với KTC
[−9,53; +0,04]. Đây **không** phải lỗi mô hình mà là một quan sát chi phối cả
mẫu. Cách xử lý đúng là **khai báo**, không phải vá.

### A2. Lớp hiệu chuẩn lại — đánh vào MCE

**Bằng chứng.** MCE trên kiểm định: 0,061 (h=1) · 0,047 (h=5) · **0,206 (h=20)**.
Con số h=20 vừa **xấu đi** từ 0,138 khi chuyển sang cửa sổ mở rộng — đổi lấy
BSS +0,0011. Đó là bước lùi ở đúng chỉ số mà giao diện phải cảnh báo.

**Làm gì.** Isotonic hoặc vector scaling, khớp trên **kiểm định**, áp cho sản
xuất. Đây là kỹ thuật chuẩn, không phải nghiên cứu.

**Nó mua gì.** Không thêm tính năng — nó **gỡ bớt một lời cảnh báo**. Với một
hệ thống bán niềm tin thì việc đó đáng hơn.

### A3. Bộ chọn mô hình theo chế độ

**Bằng chứng.** Chế độ "vừa" là chỗ duy nhất mô hình **thua** khí hậu học, và
nó lặp lại ở cả hai tầm hạn:

| | chế độ êm | **chế độ vừa** | chế độ căng |
|---|---|---|---|
| h=1, chỉ σ̂ | +0,0249 | **−0,0081** | +0,0182 |
| h=20, σ̂+chế độ | +0,0326 | **−0,0154** | +0,0270 |

Hiện dùng **một** mô hình cho cả ba chế độ, nên phần lãi ở hai chế độ ngoài
đang bị chế độ giữa ăn bớt.

**Làm gì.** Chọn mô hình theo chế độ (chế độ vừa rơi về khí hậu học), chọn trên
kiểm định. Rẻ, dư địa nhìn thấy được.

---

## Đợt B — đóng tiêu chí dừng của giai đoạn 2

### B1. Hansen SPA — chưa chạy, mà nó là điều kiện

`REPLAN_2026.md` §10.4 định nghĩa tiền đề khai phá bị coi là không đứng được
nếu **cả ba** điều sau đúng, điều đầu tiên là:

> cả năm họ đều không bác bỏ được **Hansen SPA** so nền chỉ-σ̂ ở α = 0,05

`src/metrics.py` có **MCS** (Hansen–Lunde–Nason 2011) nhưng **không có SPA**
(Hansen 2005). Nên tiêu chí dừng hiện **chưa đóng được** — luận văn không phát
biểu được "cả họ mô hình không thắng nền" một cách chính thức.

Việc: cài SPA vào `metrics.py`, chạy trên năm họ, ghi vào `GIAIDOAN2_QUYLUAT.md`.

### B2. Nâng lực phát hiện của phễu

**Bằng chứng.** `kiem_pheu.py` đo được: hiệu ứng nhỏ nhất phát hiện được ở lực
80% là **lift 1,35**. Ở 1,20 lực chỉ 40%, ở 1,15 còn 8%. Nên kết luận hiện tại
chỉ loại trừ được quy luật **mạnh**.

**Ba đường, đều tốn:**

| đường | được gì | mất gì |
|---|---|---|
| FDR (BH/BY) thay FWER | ngưỡng thấp hơn nhiều → lực cao hơn | chấp nhận vài dương tính giả |
| thu hẹp không gian giả thuyết | ít giả thuyết → ngưỡng thấp hơn | phải biện minh vì sao bỏ phần nào, **trước** khi chạy |
| thêm cặp / thêm năm | lực cao hơn thật | đụng tập khoá sổ — **không được** |

Khuyến nghị: thử **FDR**, rồi chạy lại `kiem_pheu.py` để **đo** lực mới. Đây là
điểm mạnh mới của repo — từ nay mọi thay đổi phương pháp đều **cân đo được**
bằng chính công cụ đó, thay vì lập luận suông.

Nếu FDR đưa MDES từ 1,35 xuống ~1,20 thì phát biểu của luận văn mạnh lên rõ rệt.
Nếu không, đó cũng là kết quả — và phải ghi.

---

## Đợt C — chốt cấu hình và ghi biên bản

Điền `docs/KHOA_SO.md` mục 4, hiện đang trống:

- mô hình biến động: HAR vòng 7 + lịch NHTW riêng từng cặp
- nền ba lớp theo tầm hạn: h=1 tổ hợp trực tuyến · h=5, h=20 σ̂ + chế độ (cuộn)
- quy tắc định cỡ: min(Kelly, trần phá sản 1% × 0,92) × k_vol × k_dd × k_danh_mục
- ngày chốt + ký

Sau khi ký thì **không sửa mô hình nữa**. Mọi thứ ở Đợt A và B phải xong trước.

---

## Đợt D — mở niêm phong, chạy đúng một lần

Sáu cặp chéo: EURGBP, EURJPY, GBPJPY, AUDJPY, EURCHF, NZDUSD. 2,1 GB ở
`../dukas/histdata_seal/`, chưa script nào chạm.

```
py prep_fx.py --src histdata_seal --out fx_seal
```

Đây là **lớp bảo vệ cuối** và là kết quả đắt giá nhất của luận văn: nó trả lời
câu hỏi trung tâm — quy luật có chuyển giao sang cặp **không có USD hai vế**
không. Kết quả xấu cũng phải báo cáo đúng như nó xảy ra (quy tắc 2).

Nhớ: AUDJPY đã loại năm 2012 vì kho lưu trữ thiếu, ghi từ 28/08/2026 **trước**
mọi phân tích.

---

## Đợt E — chạy song song, không đụng mô hình

Ba việc này không nằm trong ràng buộc thứ tự ở mục 0.

| việc | trạng thái | ghi chú |
|---|---|---|
| **Sổ dự báo tích luỹ** | **0/30 phiên đã chấm** | cần ~30 phiên giao dịch ≈ 6 tuần → khoảng giữa 10/2026. Đây là bằng chứng mạnh nhất ở buổi bảo vệ: hiệu chuẩn trượt đo trên **chính những dự báo hệ thống đã đưa ra**, không phải backtest. Đã bắt đầu chạy 4 lần/ngày từ 04/09 |
| **`VERCEL_TOKEN`** | **hỏng** | token không hợp lệ; bước tự triển khai trong GitHub Action đang thất bại. Dữ liệu vẫn tải, sổ vẫn ghi — chỉ là trang web không tự cập nhật |
| **Merge PR #2** | đang mở | Action chạy từ `main`; chưa merge thì lượt chạy tự động sẽ dựng lại trang bằng mã cũ |

---

## Không làm — và lý do

| việc | vì sao không |
|---|---|
| thêm kiến trúc ML/DL cho hướng đi | đã đo hai lần: 14 mô hình biến động thua HAR, ba lớp thua nền |
| tích hợp kiến trúc CAIFormer | ablation của chính nó cho thấy bản **bỏ hết** bộ máy nhân quả tốt hơn bản đầy đủ ở 3/3 cặp (MASE 0,86 → 0,70) |
| mô hình nền (Chronos, MOIRAI) | bằng chứng ngược, chi phí cao |
| RL sâu (PPO…) | đã loại có đo; văn liệu 2025–2026 xác nhận cùng lý do |
| mở tập khoá sổ để huấn luyện | mất vĩnh viễn lớp bảo vệ duy nhất còn lại |

**Còn để ngỏ, chưa xếp lịch:** dữ liệu vĩ mô (VIX, DXY, đường cong lãi suất) làm
biến ngoại sinh cho **biến động** — trục duy nhất còn thông tin, và fx-dss hiện
không có một biến vĩ mô nào. Nếu làm thì phải vào Giai đoạn 2 như một họ giả
thuyết mới, qua đúng phễu bốn cửa. Ràng buộc: dữ liệu bắt đầu 2015 (mất 3 năm
lịch sử) và kéo tới 2026-08 (chồng tập khoá sổ, phải cắt).

---

## Thứ tự đề nghị

```
A1 đuôi USDJPY  →  A2 hiệu chuẩn lại  →  A3 chọn theo chế độ
                        ↓
              B1 Hansen SPA  →  B2 FDR + đo lại lực
                        ↓
              C chốt cấu hình, ký biên bản
                        ↓
              D mở niêm phong — MỘT LẦN

     (E chạy song song suốt: sổ dự báo, token, merge)
```

A1 trước vì nó là lỗi sản phẩm đang tồn tại trên giao diện. B1 trước B2 vì SPA
là điều kiện đã ghi trong kế hoạch, còn B2 là cải tiến tuỳ chọn.
