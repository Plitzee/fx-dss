# Giai đoạn 0 — chạy và lập hồ sơ bốn script chưa có tài liệu

*03/09/2026. Theo `docs/REPLAN_2026.md` mục 10, giai đoạn 0.
Tái lập: `python src/{run_sax_gia,run_momentum_regime,run_corr_regime,run_pdv}.py`.
Log thô: `output/log_{sax_gia,momentum_regime,corr_regime,pdv}.txt`.*

Bốn script này đã được viết xong nhưng chưa từng chạy trên hồ sơ: không tài liệu
nào ghi kết quả, không script nào lưu output, cả bốn còn untracked trong git.
Giai đoạn 0 chạy chúng trước khi thiết kế bất cứ thứ gì mới lên trên.

**Ghi chú kỹ thuật:** cả bốn phải chạy với `PYTHONIOENCODING=utf-8` — chúng in
tiếng Việt và ký tự `→`, mà console mặc định trên Windows là cp1252 nên sẽ chết
giữa chừng. Không sửa dòng mã nào; chỉ đặt biến môi trường.

---

## 1. `run_sax_gia.py` — mẫu hình HƯỚNG GIÁ: **0/351 sống sót**

Đây là câu hỏi trung tâm của hướng đi mới. Kết quả dứt khoát.

Thiết lập: trạng thái = tercile lợi suất ngày (XUỐNG / ĐI NGANG / LÊN), ngưỡng
chốt trên đoạn **huấn luyện**; liệt kê **toàn bộ** không gian W ∈ {2,3,4} × 3
đích = **351 giả thuyết**, tất cả đủ số khớp (≥ 20); Westfall–Young maxT từng
bước xuống, 1.000 hoán vị, null khối 2 ngày; phát hiện trên huấn luyện+kiểm
định, xác nhận trên kiểm tra.

```
null khối 2 ngày — max|z|:  90% 3.65   95% 3.86   99% 4.29
Sống sót Westfall-Young (p<0,05) trên 351 giả thuyết: 0/351
```

**Không phải trượt sát — trượt xa.** Mẫu mạnh nhất trong toàn bộ 351 giả thuyết:

| \|z\| | z | lift | n | mẫu ⇒ đích |
|---|---|---|---|---|
| 3,13 | +3,13 | 1,138 | 1.035 | HIGH → HIGH → HIGH ⇒ LOW |
| 2,94 | +2,94 | 1,233 | 313 | LOW → MEDIUM → HIGH → HIGH ⇒ HIGH |
| 2,92 | −2,92 | 0,923 | 2.952 | HIGH → HIGH ⇒ MEDIUM |
| 2,89 | +2,89 | 1,132 | 974 | LOW → HIGH → MEDIUM ⇒ MEDIUM |
| 2,87 | +2,87 | 1,224 | 329 | HIGH → HIGH → HIGH → HIGH ⇒ LOW |

`max|z|` quan sát = **3,13** so ngưỡng **3,86**. Không giả thuyết nào chạm tới.
Nếu không hiệu chỉnh bội thì 31/351 "có ý nghĩa" ở p<0,05 — trong khi thuần
nhiễu kỳ vọng 17,6. Hơi cao hơn nhiễu, nhưng thủ tục từng bước xuống nói rõ:
không cái nào đứng vững.

### 1.1 So sánh với nhánh biến động — đây mới là phần thuyết phục

Cùng bộ máy, cùng dữ liệu, cùng thủ tục, chỉ đổi biến đích:

| | **biến động** (`run_sax_stats.py`) | **hướng** (`run_sax_gia.py`) |
|---|---|---|
| số giả thuyết | 336 | 351 |
| ngưỡng null khối 2 ngày (95%) | **29,51** | **3,86** |
| max\|z\| quan sát | ≈ 69 | **3,13** |
| sống sót W-Y | **41** | **0** |

Đọc bảng này theo đúng thứ tự thì nó rất mạnh: **rào của hướng đi thấp hơn rào
của biến động 7,6 lần** — vì lợi suất ngày gần như không tự tương quan nên null
khối 2 ngày gần như không thêm gì so với mức 1,96 — **mà vẫn không ai vượt
được.** Trong khi biến động vượt một cái rào cao gấp 7,6 lần, 41 lần.

Đây là **kết luận âm thứ tư về hướng đi**, và là cái sạch nhất: không gian giả
thuyết liệt kê đầy đủ, kiểm soát FWER đúng cách, ngưỡng đặt trước.

### 1.2 Một chi tiết đáng chú ý dù không sống sót

Bốn trong năm mẫu mạnh nhất mang hơi hướng **đảo chiều**, không phải nối tiếp:
`HIGH→HIGH→HIGH ⇒ LOW` (lift 1,138), `HIGH→HIGH→HIGH→HIGH ⇒ LOW` (1,224), và
`HIGH→HIGH ⇒ MEDIUM` với z âm (tức *ít* khả năng đi ngang hơn nền). Cùng chiều
với kết luận momentum âm ở mục 2. Không đủ để tuyên bố gì — ghi lại để nếu vòng
sau có thử đảo chiều thì biết bắt đầu từ đâu.

---

## 2. `run_momentum_regime.py` — ràng buộc #6: **xác nhận**

`docs/REPLAN_2026.md` mục 1.3 ghi con số "Sharpe −0,62, p=0,001" là **không kiểm
chứng được**. Nay chạy được, và con số đúng:

```
Ý nghĩa thống kê trên đoạn KIỂM ĐỊNH (gộp mọi span, Newey-West):
  bình tĩnh    n= 2732  Sharpe= +0.161  t= +0.55  p=0.582
  vừa          n= 3824  Sharpe= +0.475  t= +1.50  p=0.133
  căng thẳng   n= 6020  Sharpe= -0.615  t= -3.30  p=0.001  ***
```

Bonferroni với 3 chế độ cho ngưỡng 0,05/3 = 0,0167; p = 0,001 vượt thoải mái.
**Ràng buộc #6 của spec được khôi phục vào danh sách.**

**Nhưng thứ chắc hơn con số p là tính nhất quán.** Chế độ căng thẳng âm ở **cả
12 ô** — 4 độ dài tín hiệu × 3 đoạn dữ liệu:

| span | huấn luyện | kiểm định | kiểm tra |
|---|---|---|---|
| 5 | −0,402 | −1,271 | −0,817 |
| 10 | −0,534 | −0,709 | −1,042 |
| 20 | −0,367 | −0,204 | −1,090 |
| 60 | −0,473 | −0,277 | −0,354 |

Không sót ô nào, kể cả trên đoạn **kiểm tra** vốn không dùng để chọn gì.

**Hệ quả cho thư viện quy luật:** đây là ứng viên đầu tiên đủ tư cách — một
**quy luật loại-trừ** dạng *"không mở vị thế theo đà khi σ̂ nằm ở tercile cao
nhất"*. Nó không dự báo hướng; nó chỉ ra một chỗ **không nên tham gia**, và điều
đó có giá trị vận hành thật.

**Và nó đóng nốt một cánh cửa:** hai chế độ còn lại dương nhưng p = 0,582 và
p = 0,133 — momentum-theo-chế-độ **không** cứu được tầng 1. Docstring của script
đã viết sẵn kết luận này ("dừng ở đây, khỏi thử Deep Momentum Network /
meta-labeling — đắt hơn nhiều, cùng tập thông tin") và dữ liệu ủng hộ nó.

---

## 3. `run_corr_regime.py` — **phát hiện mới, và là một lỗ hổng an toàn**

Câu hỏi: hằng số `RHO_MAC_DINH = 0,44` trong `position_sizing.py` có sập theo
chế độ biến động không?

**Ở mức tercile: không sập.** ρ = 0,434 / 0,457 / 0,444 cho bình tĩnh / vừa /
căng thẳng, khoảng tin cậy chồng lên nhau, toàn mẫu 0,443. Hằng số 0,44 mô tả
đúng.

**Ở đuôi thật: sập.**

| | n ngày | ρ | KTC 95% |
|---|---|---|---|
| top 5% biến động nhất | 217 | **0,544** | [0,461 – 0,625] |
| top 1% biến động nhất | 48 | **0,594** | [0,497 – 0,684] |
| phần còn lại | | 0,424 – 0,436 | |

KTC của top 1% **không phủ 0,44**. Đây đúng là hiện tượng văn liệu mô tả — đa
dạng hoá biến mất đúng lúc cần nhất — và lý do `TANG4_DANHMUC.md` không bắt được
là nó đo ρ **một lần trên toàn mẫu**.

**Hệ quả định lượng.** Với luật `k_danh_mục = 1/√(k + k(k−1)ρ)` và k = 6 vị thế
cùng chiều USD:

| ρ dùng | k_danh_mục | đòn bẩy so hằng số 0,44 |
|---|---|---|
| 0,44 (đang dùng) | 0,2282 | — |
| 0,544 (top 5%) | 0,2117 | **+7,3% quá cao** |
| 0,594 (top 1%) | 0,2049 | **+10,2% quá cao** |
| 0,684 (biên trên KTC) | 0,1942 | +14,9% quá cao |

Nghĩa là hệ số danh mục hiện tại cho phép đòn bẩy **cao hơn mức an toàn thật
7–10% đúng vào những ngày căng thẳng nhất** — sai đúng chiều nguy hiểm. Đây là
lỗ hổng phạm vi thứ ba của trần rủi ro, cùng loại với hai lỗ hổng đã vá ở
`TANG4_DANHMUC.md`.

**ĐÍNH CHÍNH 03/09/2026 — bản vá đã có sẵn, tôi báo sai ở bản đầu.**
`src/position_sizing.py` **đã** cài đúng cách vá này, trong phần sửa chưa commit:
`RHO_CANG_THANG = 0,55`, ngưỡng `NGUONG_CANG_THANG_PCTL = 0,95` chốt trên đoạn
huấn luyện, và `_rho_hieu_dung()` nâng ρ khi σ̂ chạm ngưỡng đó. Chú thích trong
mã trích dẫn đúng `run_corr_regime.py` và đúng những con số ở bảng trên. Tự kiểm
đạt: `hệ số danh mục (k=6): bình thường 0,228 (ρ=0,44) → vùng căng thẳng 0,211
(ρ=0,55)`.

Tôi kết luận nhầm vì đã grep `docs/` chứ không đọc mã: thứ thiếu là **tài liệu**,
không phải bản vá. Nay tài liệu là mục này.

Một khe hở nhỏ còn lại, có chủ đích: 0,55 nằm **giữa** 0,544 (top 5%) và 0,594
(top 1%) — chú thích trong mã ghi rõ là làm tròn giữa hai mức đo. Với riêng
những ngày top 1% thì nó vẫn lỏng hơn số đo một chút.

**Lỗ hổng THẬT nằm chỗ khác — `run_e2e.py` dòng 108 truyền `so_vi_the=1`.**
Nghĩa là bài kiểm tra toàn mạch — nguồn của con số tiêu đề "không cháy tài khoản
ở mọi cấu hình, vốn cuối 1,004–1,027" — **chưa hề áp hệ số danh mục**. Sáu khoang
vốn được cộng bình quân như thể độc lập. Chính docstring của script (dòng 25) và
`docs/TOANMACH_E2E.md` giới hạn 1 đều đã ghi đây là "giới hạn quan trọng nhất".
Bản vá ở tầng 4 không tự chảy xuống đây được, vì đòn bẩy nuôi vào bài toán quy
hoạch động vốn quyết định mở/đóng, mà số cặp đang mở lại phụ thuộc ngược lại
quyết định đó — một vòng lặp.

**ĐÃ ĐO 04/09/2026 — và kết quả ngược trực giác.** Chạy chặn trên bi quan nhất
(`E2E_SO_VI_THE=6`, luôn định cỡ như cả 6 cặp đang mở): **mọi chỉ số tốt lên** —
TB 0,25 → 0,33 bp/ngày, Sharpe 0,19 → 0,46, sụt giảm tối đa 5,8% → 2,9%, CVaR5%
−50,5 → −27,0 bp, vốn cuối 1,027 → 1,037, và vẫn không cháy tài khoản ở mọi
phần Kelly.

Nguyên nhân đo được, không phải suy đoán: trong `position_sizing.size()`,
`k_danh_mục` nhân vào **trần rủi ro** chứ không nhân vào Kelly, nên nó là nhát
cắt **có chọn lọc** — chỉ cắt đúng những ngày trần đang là ràng buộc, tức những
ngày đòn bẩy cao nhất. Tỷ lệ `f(k=6)/f(k=1)` đi từ **1,000** (carry nhỏ, Kelly
buộc) đến **0,228** (carry lớn, trần buộc). Khác hẳn hạ `frac`, vốn cắt đều tay
mọi ngày. Đầy đủ ở `docs/TOANMACH_E2E.md`, mục "Chặn trên".

Kèm một chi tiết phải ghi: ρ vùng căng thẳng **không kích hoạt** trên đường đi
này — `s_stress` = 0,00820 so với σ̂ đại diện chế độ cao nhất chỉ 0,00655 — nên
bản vá ở mục này **không đóng góp gì** vào bảng E2E. Nó chỉ bén ở những đường đi
truyền σ̂ ngày thật: phiếu quyết định và giao diện.

---

## 4. `run_pdv.py` — tái lập kết luận đã có

Thêm đặc trưng biến động phụ thuộc đường đi (Guyon–Lekeufack) vào họ HAR.

| mô hình | TB QLIKE | hạng TB |
|---|---|---|
| **STHARQ-PD** | **0,1599** | 2,7 |
| EN + PDV | 0,1601 | 3,3 |
| STHARQ | 0,1604 | 4,2 |
| EN hiện tại | 0,1604 | 4,2 |

STHARQ-PD tốt hơn STHARQ **0,3%**, tốt nhất ở 4/6 cặp — khớp đúng điều README
đã ghi. Kết luận giữ nguyên: **không đổi mô hình sản xuất**.

**Một đính chính về cách đọc bảng:** dấu `*` trong output chỉ đánh dấu **cột tốt
nhất** (`'*' if k==b` trong mã), **không** phải mức ý nghĩa thống kê. Script này
không chạy kiểm định nào. Nên nó xác nhận **độ lớn** hiệu ứng, không xác nhận ý
nghĩa; câu "không cặp nào đạt p<0,05" trong README đến từ phép đo khác.

---

## 5. Hệ quả cho `docs/REPLAN_2026.md`

| Mục | Thay đổi |
|---|---|
| 1.3 | ràng buộc #6 **không còn treo** — đã xác nhận, xem mục 2 |
| 1.4 | `run_sax_gia.py` **đã chạy** — xem mục 1 |
| 3.1, họ H1 (ký hiệu/SAX) | trên **hướng, tầm 1 phiên: đã trả lời = thất bại**. Trên biến động: 41 ứng viên |
| 3.2 | dự đoán "null khối sẽ quá bảo thủ cho hướng" **sai** — ngưỡng chỉ 3,86, gần mức chưa hiệu chỉnh 1,96. Không cần đổi null |
| 10, GĐ 2 | đã tiêu một phần: 1 họ × 1 mục tiêu × 1 tầm hạn đã xong trong nửa tuần thay vì 3 tuần |
| 10.4 tiêu chí dừng | **chưa kích hoạt** — mới thất bại 1/5 họ, trên 1/2 mục tiêu, ở 1/3 tầm hạn |
| thư viện quy luật | có **ứng viên đầu tiên**: quy luật loại-trừ momentum-chế-độ-căng (mục 2) |

**Chưa được kết luận gì thêm.** Còn chưa thử: bốn họ còn lại (motif, học quy tắc
đọc được, trần GBM, chế độ) trên hướng; **toàn bộ** mục tiêu R (đi ngang so
không đi ngang — nơi lý thuyết nói kỹ năng thật sự nằm); và tầm hạn 5 và 20
phiên. Kết quả mục 1 nói *biểu diễn ký hiệu trên lợi suất ngày rời rạc hoá theo
tercile* không chứa thông tin hướng — nó **không** nói mọi biểu diễn đều thế.
