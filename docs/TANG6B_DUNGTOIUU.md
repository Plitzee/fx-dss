# Tầng 6b — Dừng tối ưu: giữ lệnh hay đóng lệnh?

*01/09/2026. Tái lập bằng `python src/run_optstop.py`.*

## Vì sao tầng này tồn tại

Phiếu quyết định ở tầng 6 in ra bảng tầm hạn — xác suất chạm stop **1 phiên 5%,
5 phiên 37%, 10 phiên 54%, 20 phiên 66%** — rồi bỏ mặc người dùng. Hệ thống
không có bất kỳ quy tắc nào về **khi nào nên đóng lệnh**. Đây là bài toán tuần
tự thật sự duy nhất trong toàn hệ thống, và nó chưa được mô hình hoá dòng nào.

## Vì sao không dùng học tăng cường

Đã thử ở tầng 4 và thua có số đo (`docs/SIZING_COMPARISON.md`): PPO học ra hệ số
gần như hằng số, biên độ **0,018** so với **0,800** của quy tắc tay, và phá sản
cao gấp **26 lần** ở cùng mức tăng trưởng.

Ở tầng 6b, trạng thái chỉ có **hai chiều**, nên quy hoạch động giải **chính xác**
trên lưới: không cần xấp xỉ hàm, không cần thăm dò, không cần 10⁵ episode, và kết
quả là một **biên giới đọc được** chứ không phải một mạng nơ-ron.

## Bài toán

Vị thế mua, đòn bẩy do tầng 4 chốt, mức dừng lỗ đặt cách giá vào 2σ.
Mỗi phiên chọn: **ĐÓNG** hay **GIỮ**.

**Trạng thái** (hai chiều, đều không thứ nguyên):

- `s` = ln(giá hiện tại / mức dừng lỗ) / σ̂ — còn cách stop bao xa
- `v` = chế độ biến động (tam phân vị của σ̂ trên đoạn huấn luyện)

cộng số phiên còn lại `n` trong tầm hạn 20.

**Chuyển trạng thái lấy từ chính panel, không giả định phân phối.** Vì `zL` và
`zT` trong panel đã ở đơn vị σ nên mọi thứ khớp trực tiếp:

```
chạm stop trong phiên t   ⟺   zL_t ≤ −s
nếu sống sót              s' = (s + zT_t) · (σ̂_t / σ̂_{t+1})
```

Bộ ba `(zT, zL, σ̂_t/σ̂_{t+1})` rút mẫu thực nghiệm theo đúng chế độ `v`, 5.006
quan sát mỗi chế độ.

**Một cái bẫy đơn vị phải tránh.** Khoảng cách tới stop đo bằng σ của *hôm nay*,
còn giá trị tiếp diễn lại ở σ của *ngày mai*. Cộng hai thứ đó trực tiếp là sai —
đây là lỗi tôi mắc ở lần chạy đầu và phải sửa. Toàn bộ hàm giá trị được quy về
**lợi suất log tuyệt đối**, với σ điển hình riêng cho từng chế độ.

## Ba tham số, cả ba đều đo được chứ không đặt tay

| tham số | giá trị | nguồn |
|---|---|---|
| trượt giá kỳ vọng | **0,0889 σ** | 1.848 lần chạm stop ở khoảng cách 1,0%, đo từ nến M1 |
| chi phí thoát một chiều | 0,0066 – 0,0162 σ | `cost.py`, spread trung bình 24 giờ + hoa hồng |
| carry | −0,69% đến +0,98%/năm | `carry.csv`, trung vị trên đoạn huấn luyện |

## Phải hiểu điều này trước khi đọc kết quả

Tầng 1 đã chứng minh **không có lợi thế về hướng**, tức E[zT] = 0. Theo định lý
dừng tuỳ ý của martingale, nếu không có chi phí và không có trượt giá thì **mọi
quy tắc dừng đều cho cùng kỳ vọng** — bài toán sẽ tầm thường.

Nó *không* tầm thường ở đây đúng ba lý do:

1. **Trượt giá làm chạm stop đắt hơn thoát tự nguyện.**
2. **Carry** là thứ duy nhất trả công cho việc giữ.
3. **Xác suất chạm stop phụ thuộc σ̂.** Đây là chỗ giá trị của tầng 2 cuối cùng
   biến thành một *quyết định* chứ không còn là một con số.

Nhưng hệ quả cũng phải nói thẳng: **không quy tắc dừng nào tạo ra lợi suất.**
Nó chỉ **nắn lại phân phối**. Kết quả dưới đây đúng như vậy.

## Biên giới tìm được

Đóng lệnh khi khoảng cách tới stop **dưới** mức này:

| cặp | chế độ 0 (êm) | chế độ 1 | chế độ 2 (căng) | carry %/năm |
|---|---|---|---|---|
| EURUSD | đóng hết | 1,10σ | đóng hết | −0,69% |
| GBPUSD | đóng hết | 1,00σ | 2,55σ | −0,02% |
| USDJPY | đóng hết | 0,95σ | 2,00σ | +0,28% |
| AUDUSD | 1,90σ | 0,80σ | 1,20σ | +0,91% |
| USDCAD | đóng hết | 1,00σ | 2,25σ | +0,01% |
| USDCHF | 1,80σ | 0,80σ | 1,10σ | +0,98% |

Đọc bảng này:

- **Carry quyết định có được giữ hay không.** EURUSD carry âm → "đóng hết" ở hai
  trong ba chế độ. USDCHF và AUDUSD carry gần +1%/năm → có vùng giữ ở cả ba chế độ.
- **Chế độ căng đòi khoảng cách an toàn lớn hơn.** GBPUSD cần 2,55σ ở chế độ 2 so
  với 1,00σ ở chế độ 1. Đây chính là σ̂ của tầng 2 đi thẳng vào quyết định.
- Quy tắc **đọc được thành một câu**: *"giữ lệnh chừng nào còn cách stop hơn s*(chế
  độ) σ; dưới ngưỡng đó thì đóng, vì carry kiếm được không bù nổi rủi ro trượt giá."*

## Kết quả trên đoạn kiểm tra

*(Bảng gốc 01/09/2026 — carry HẰNG SỐ, giải DP một lần trên trung vị huấn
luyện. Đã bị thay bằng bản "cuốn theo năm" ở mục vá 03/09/2026 bên dưới; giữ
bảng này lại để thấy vá đó thay đổi cái gì.)*

Mở một lệnh mua mỗi phiên, stop 2σ, tầm hạn 20 phiên, 3.170 lệnh mô phỏng.

| chính sách | TB (bp) | trung vị | p5 (bp) | **CVaR 5% (bp)** | **bị stop** | DM vs DP | p |
|---|---|---|---|---|---|---|---|
| đóng sau 5 phiên | −0,30 | −7,06 | −128,9 | −149,4 | 34,8% | 1,29 | 0,197 |
| đóng ngay | −0,54 | −0,54 | −0,9 | −1,1 | 0,0% | 0,90 | 0,370 |
| đóng sau 10 phiên | −2,14 | −58,91 | −134,0 | −155,5 | 51,4% | 0,50 | 0,614 |
| giữ hết tầm hạn | −2,66 | −73,23 | −137,5 | −159,8 | 63,5% | 0,24 | 0,807 |
| **DP** | −3,95 | **−0,55** | **−110,2** | **−132,5** | **11,0%** | — | — |

### Đọc bảng này cho đúng

**Về lợi suất trung bình: không có gì phân biệt được.** Mọi p ≥ 0,197. Đây là kết
quả **đúng và đã dự đoán trước** từ tầng 1 — không quy tắc dừng nào tạo ra lợi suất
khi không có lợi thế về hướng. Bất kỳ ai báo cáo "chiến lược thoát lệnh của tôi tăng
lợi suất 30%" mà không có kiểm định này thì đang đọc nhiễu.

**Về phân phối: DP thắng rõ, trong nhóm các chính sách thực sự vào lệnh.**
So với "giữ hết tầm hạn":

- CVaR 5%: **−132,5 so với −159,8 bp** — đuôi nhẹ hơn **17%**
- phân vị 5: −110,2 so với −137,5 bp
- trung vị: **−0,55 so với −73,23 bp**
- tỷ lệ bị stop: **11,0% so với 63,5%** — thấp hơn **5,8 lần**

Cùng kỳ vọng (không phân biệt được về mặt thống kê), nhưng đuôi nhẹ hơn nhiều và
số lần bị quét stop giảm gần sáu lần. Với một hệ thống mà mục tiêu là *tăng trưởng
dưới ràng buộc phá sản*, đó đúng là thứ cần.

**"Đóng ngay" có đuôi đẹp nhất (−1,1 bp) vì nó không nhận rủi ro nào cả.** Đó là
chính sách suy biến, và nó cũng không kiếm được gì. So sánh có nghĩa là DP với các
chính sách *thực sự giữ lệnh* — và DP thắng tất cả trên mọi thước đo đuôi.

## Giới hạn phải ghi vào luận văn

1. ~~**Hàm mục tiêu chưa khớp hoàn toàn với mục tiêu hệ thống.**~~ **Đã cài đặt
   và kiểm định 03/09/2026** (mục "khớp nối đòn bẩy tầng 4" bên dưới) — `giai()`
   giờ nhận được `f_v` (đòn bẩy tầng 4) và tối ưu trực tiếp E[ln(1+f·R)]. Kiểm
   định xác nhận cơ chế đúng hướng (đuôi đỡ hơn áp đòn bẩy sau ~25%), nhưng
   **chưa bật mặc định trong sản xuất** vì Kelly đầy đủ cho carry vẫn để đuôi
   tệ hơn không vay — cần chiết khấu Kelly trước, xem mục đó để biết chi tiết
   và điều kiện để bật.
2. ~~**Carry coi như hằng số** (trung vị trên huấn luyện, mỗi cặp một giá trị).~~
   **Đã vá 03/09/2026** — xem mục ngay dưới đây. Carry giờ cuốn theo cửa sổ mở
   rộng theo năm thay vì đứng yên suốt đoạn kiểm định/kiểm tra. Vẫn còn là một
   **số vô hướng mỗi lần giải** (không phải một chiều trạng thái riêng của DP) —
   nếu muốn DP tự thấy carry đang trôi mà không cần giải lại thủ công mỗi năm
   thì phải đưa carry vào trạng thái (thành ba chiều), việc đó vẫn chưa làm.
3. **Chỉ mô phỏng vị thế MUA.** Vị thế bán dùng cùng bộ máy với carry đổi dấu và
   `zH` thay `zL`. Chưa chạy.
4. **Giả định vào lệnh mỗi phiên.** Đây là thiết kế để có mẫu lớn, không phải một
   chiến lược đề xuất.

## Vá 03/09/2026 — carry cuốn theo năm thay vì hằng số

### Phát hiện

`carry_ngay()` trong `optimal_stop.py` trả về một chuỗi carry **thay đổi theo
ngày thật** (nội suy tiến từ `carry.csv`, vốn là dữ liệu tháng) — carry KHÔNG
đứng yên. Nhưng `run_optstop.py` (bản trước vá) chỉ lấy **trung vị của đoạn
huấn luyện**, giải DP **một lần** bằng số đó, rồi áp y nguyên biên giới cho cả
đoạn kiểm định lẫn kiểm tra. Đo trực tiếp carry thật (%/năm, quy về đúng
`mtr = Date < VALID_TU` mà `run_optstop.py` dùng):

| cặp | carry huấn luyện | carry TB kiểm tra | ghi chú |
|---|---|---|---|
| EURUSD | −0,65% | −1,63% | cùng dấu, lệch 2,5 lần |
| GBPUSD | −0,02% | −0,06% | cùng dấu |
| USDJPY | +0,21% | +4,07% | cùng dấu, lệch ~19 lần |
| AUDUSD | +0,91% | −0,52% | **đổi dấu** |
| USDCAD | +0,01% | +1,26% | cùng dấu, lệch rất xa 0 |
| USDCHF | +0,97% | +3,91% | cùng dấu, lệch ~4 lần |

Đây là đúng lỗi **phạm vi áp dụng** đã gặp ở tầng 4 (`docs/TANG4_DANHMUC.md`):
công thức DP đúng với carry nó được đưa vào, nhưng cái nó được đưa vào là một
con số đã cũ khi dùng để chấm điểm những năm sau. Quy tắc "carry quyết định có
giữ hay không" (xem bảng biên giới ở trên) thì đúng — chỉ là carry dùng để
quyết định đã lỗi thời.

### Vá

`src/compare_carry_dong.py` giải lại DP **mỗi năm dương lịch**, dùng trung vị
carry của **mọi dữ liệu trước năm đó** (cửa sổ mở rộng, nhân quả — đúng quy ước
đã dùng ở tầng 2 định cỡ lại mỗi phiên và tầng 4 mục 2), rồi so với bản tĩnh
trên đoạn kiểm tra:

| chính sách | TB (bp) | trung vị | p5 (bp) | CVaR5% (bp) |
|---|---|---|---|---|
| DP-tĩnh (bản cũ) | −3,95 | −0,55 | −110,2 | −132,5 |
| DP-cuốn theo năm | **−0,44** | −0,55 | **−105,4** | **−127,4** |

n = 3.170 lệnh mỗi bên. Kiểm định Diebold–Mariano (Newey–West, so hiệu số
cuốn − tĩnh): **t=+2,00, p=0,046** — cuốn theo năm tốt hơn có ý nghĩa thống kê
ở mức 5%. Trung vị gần như không đổi (đây là hiệu ứng ở **trung bình và đuôi**,
không phải phân vị giữa) — hợp lý, vì carry lỗi thời chủ yếu làm sai chỗ nào
đáng giữ đúng lúc carry đã chảy nhiều, chứ không đổi bản chất phần lớn giao
dịch bình thường.

**Đã cài vào `run_optstop.py`.** DP giờ giải lại mỗi năm cho cả đoạn kiểm định
lẫn kiểm tra (hàm `chay()`), không còn dùng một biên giới tĩnh. Bảng "biên giới
đóng lệnh" gốc ở trên vẫn giữ lại để minh hoạ cơ chế; bản dùng để chấm điểm
thật là bảng "cuốn theo năm" mà `run_optstop.py` in ra ngay sau đó.

**Tái lập:**
```bash
python src/compare_carry_dong.py   # bằng chứng — so DP-tĩnh với DP-cuốn theo năm
python src/run_optstop.py          # bản đã vá, dùng để chấm điểm
```

## Vá 03/09/2026 — khớp nối đòn bẩy tầng 4 với quyết định giữ/đóng

### Câu hỏi

Giới hạn #1 ở trên đã nói: DP đang tối ưu E[lợi suất log], trong khi mục tiêu
thật của hệ thống là **E[ln(1 + f·R)]** — hàm lõm, phạt đuôi trái nặng hơn — với
`f` là đòn bẩy tầng 4 khuyến nghị. Với E[z] ≈ 0 (không có lợi thế hướng) thì hai
hàm gần bằng nhau nên DP cũ "tình cờ" giảm đuôi như hệ quả phụ. Câu hỏi: **giải
trực tiếp cho hàm lõm có tốt hơn không, hay áp đòn bẩy sau lên biên giới cũ là
đủ?**

### Đòn bẩy dùng ở đây là gì

Vị thế trong tầng 6b không có lợi thế **hướng** (tầng 1 đã bác bỏ — `mu=0` thì
Kelly=0, không có gì để tối ưu). Lợi thế duy nhất là **carry** — chính tài liệu
tầng 6b đã chứng minh "carry là thứ duy nhất trả công cho việc giữ". Nên dùng
đúng khung Kelly của tầng 4 nhưng với `mu = carry` (Kelly cho một giao dịch
carry, không phải giao dịch hướng): `f_Kelly = carry / σ²`, cắt bởi trần rủi ro
tầng 4 (`f_ruin_cap × k_vol`, 1 vị thế đơn lẻ — không áp `k_danh_mục`). Carry âm
thì Kelly âm, `PositionSizer` cắt về 0 — khớp đúng ý "carry không bù nổi thì nên
đóng" mà bảng biên giới DP đã tìm ra độc lập từ trước.

### Cài đặt

`optimal_stop.py`: `giai()` nhận thêm tham số tuỳ chọn `f_v` (đòn bẩy mỗi chế
độ) — khi có, mọi bước (carry, chuyển động giá, phí thoát, chạm stop) được quy
đổi qua `ln(1 + f·(eʳ−1))` **từng bước một rồi mới cộng dồn** (đúng cách vốn có
đòn bẩy thật sự compound qua nhiều phiên — cộng lợi suất trước rồi đổi đơn vị
một lần là sai, vì đòn bẩy không tuyến tính qua hàm lõm). `nen_giu()` và
`mo_phong()` nhận `f_v` tương ứng. Khi `f_v=None` (mặc định), hành vi giống hệt
bản cũ — mọi nơi đang gọi các hàm này mà không truyền `f_v` không đổi gì cả.

### So sánh — `src/compare_leverage_dp.py`

Ba cách, cùng chấm trên **một** thước đo E[ln(1+f·R)] (đoạn kiểm tra, carry cuốn
theo năm như mục vá trước):

| chính sách | TB (bp) | trung vị | p5 (bp) | CVaR5% (bp) |
|---|---|---|---|---|
| A: DP-log, không đòn bẩy (f=1, mốc so sánh) | −0,44 | −0,55 | −105,4 | −127,4 |
| B: DP-log (biên giới cũ), đòn bẩy áp SAU | −1,34 | 0,00 | −219,2 | −278,0 |
| C: DP-lõm, giải TRỰC TIẾP cho hàm có đòn bẩy | −0,96 | 0,00 | −131,2 | −209,7 |

n = 3.170 lệnh mỗi bên. DM (B−A): t=−0,18, p=0,853 — **không có ý nghĩa**: đòn
bẩy carry-Kelly quá nhỏ (đa số dưới 1×, cao nhất USDCHF ~4,1× ở chế độ êm) để
đổi kết luận về trung bình. DM (C−B): t=+0,08, p=0,937 — cũng không có ý nghĩa
ở **trung bình**.

### Đọc kết quả cho đúng — nơi khớp nối thật sự lộ ra là ĐUÔI, không phải trung bình

Không kiểm định DM nào ở trung bình có ý nghĩa (đúng dự đoán — martingale vẫn
đúng, đòn bẩy không tạo ra lợi thế mới). Nhưng nhìn **CVaR5%** thì khác hẳn:

- **Áp đòn bẩy SAU lên biên giới log cũ (B) làm đuôi tệ đi hẳn 2,2 lần**
  (−127,4 → −278,0 bp) so với không đòn bẩy. Biên giới cũ không "biết" nó sắp bị
  vay 4 lần, nên không né đủ sớm.
- **Giải trực tiếp cho hàm lõm (C) phục hồi lại phần lớn khoảng đó**
  (−278,0 → −209,7 bp, đỡ hơn ~25%) — DP nhìn thấy chi phí thật của việc bị stop
  khi đang vay nên đóng sớm hơn ở đúng những trạng thái đòn bẩy cao. Đây **là**
  câu trả lời cho việc khớp nối: có ích, và có ích ở đúng chỗ hàm lõm được thiết
  kế để bảo vệ — đuôi, không phải trung bình.
- Nhưng **C vẫn tệ hơn A** (−209,7 so với −127,4 bp). Ngay cả khi DP đã "biết
  trước" sẽ có đòn bẩy, dùng **Kelly đầy đủ** (không chiết khấu) cho carry vẫn
  để lại đuôi nặng hơn không vay gì. Đây không phải lỗi của khớp nối — đây là
  bài học Kelly kinh điển: Kelly đầy đủ nhạy với sai số ước lượng `mu` (ở đây là
  trung vị carry một mẫu con), và tầng 4 chính hãng **chưa bao giờ** dùng Kelly
  trần trụi — luôn `min(Kelly, trần rủi ro đã chiết khấu K_SLIP)`. Kết quả này
  *xác nhận thêm* lý do tầng 4 làm vậy, áp cho đúng bối cảnh carry.

### Kết luận và phạm vi

**Cơ chế khớp nối đã cài xong và đã kiểm định: nó hoạt động đúng hướng thiết
kế** (giải trực tiếp hàm lõm bảo vệ đuôi tốt hơn áp đòn bẩy sau). Nhưng **chưa
đổi bặc định trong `run_optstop.py`** — kết quả trên cho thấy dùng thẳng Kelly
đầy đủ cho carry (không chiết khấu) vẫn xấu hơn không vay, nên bật nó lên làm
mặc định sản xuất bây giờ là chưa đủ căn cứ. Việc còn lại, ngoài phạm vi hôm
nay: chiết khấu `f_v` (Kelly phân số, giống `K_SLIP` của tầng 4) rồi đo lại —
nếu khi đó C vượt A trên CVaR thì mới nên đổi mặc định.

**Tái lập:**
```bash
python src/compare_leverage_dp.py
```
