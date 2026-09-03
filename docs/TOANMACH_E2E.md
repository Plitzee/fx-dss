# Kiểm tra toàn mạch (end-to-end) — tầng 2 → 3 → 4 → 5 → 6b

*03/09/2026. Tái lập bằng `python src/run_e2e.py`.*

## Vì sao cần bài kiểm tra này, và vì sao nó khác mọi phép đo trước giờ

Mọi kiểm định trước đây trong dự án — kể cả `run_optstop.py`,
`compare_carry_dong.py`, `compare_leverage_dp.py` — đều đo trên **từng lệnh
độc lập**: mở một lệnh mới **mỗi phiên được đánh dấu**, chạy tối đa `N=20`
phiên, không quan tâm lệnh trước còn mở hay không. Cách đó đúng để đo **phân
phối lợi suất một lệnh** (cần cho DM test, CVaR mỗi lệnh…), nhưng **không trả
lời được câu hỏi thật của luận văn**: *nếu vận hành hệ thống liên tục — mở khi
đáng mở, giữ theo tầng 6b, đóng khi tầng 6b bảo đóng, mở lại khi lại đáng — vốn
có tăng trưởng ổn định không, có bao giờ cháy tài khoản không?*

`src/run_e2e.py` trả lời đúng câu đó: chạy **liên tục theo từng ngày thật**,
xâu chuỗi cả 5 tầng đã kiểm định riêng lẻ:

| tầng | vai trò trong vòng lặp |
|---|---|
| 2 | σ̂ mỗi ngày (đã có sẵn trong `panel2_6pairs.csv`, định cỡ lại mỗi phiên) |
| 4 | đòn bẩy `f` mỗi chế độ biến động — khung Kelly với `mu = carry` (carry-Kelly, xem `docs/TANG6B_DUNGTOIUU.md` mục khớp nối), **chiết khấu theo phần Kelly** (quét dưới đây) |
| 6b | quyết định GIỮ/ĐÓNG mỗi ngày — DP đã khớp nối đòn bẩy (mục trên), tra ở trạng thái ổn định `n=N` |
| 3, 5 | trượt giá kỳ vọng + chi phí thoát, tính khi đóng lệnh (đúng quy ước cũ: không tính phí lúc mở) |

## Quy ước cần biết trước khi đọc số

- **Danh mục 6 cặp, vốn chia đều 1/6 mỗi cặp.** Đây là 6 "khoang vốn" độc lập
  cộng bình quân, **CHƯA** áp `k_danh_mục` liên cặp (hệ số co khi nhiều cặp mở
  cùng lúc, đã vá ở tầng 4 — `docs/TANG4_DANHMUC.md`). Đây là giới hạn quan
  trọng nhất của lần chạy này, xem mục "Giới hạn" bên dưới.
- **Carry-Kelly ĐẦY ĐỦ đã được chứng minh (mục trước) rủi ro hơn không vay ở
  CVaR mỗi lệnh riêng lẻ** — nên ở đây quét luôn **phần Kelly** (1, 1/2, 1/4,
  1/8), đúng thực hành chuẩn (Kelly phân số) và đúng cách `K_SLIP` của tầng 4
  từng được chọn: đo thực nghiệm rồi chọn, không đặt tay.

## Kết quả — quét phần Kelly, gộp kiểm định + kiểm tra (6.578 phiên/cặp)

| phần Kelly | TB (bp/ngày) | Sharpe (năm) | sụt giảm tối đa | CVaR5% (bp) | cháy? | vốn cuối |
|---|---|---|---|---|---|---|
| 1 (đầy đủ) | 0,25 | 0,19 | 5,8% | −50,5 | không | 1,027 |
| 1/2 | 0,15 | 0,23 | 2,8% | −25,4 | không | 1,017 |
| 1/4 | 0,07 | 0,23 | 1,4% | −12,8 | không | 1,008 |
| 1/8 | 0,04 | 0,25 | 0,7% | −6,4 | không | 1,004 |

**Không phần Kelly nào cháy tài khoản** (ngưỡng `RUIN_LEVEL=50%` của
`sizing.py`) trên toàn bộ ~26 năm dữ liệu kiểm định+kiểm tra. Sụt giảm tối đa
cao nhất chỉ 5,8% (Kelly đầy đủ) — rất xa ngưỡng cháy 50%. Tăng trưởng tuyệt
đối **rất khiêm tốn** ở mọi mức (vốn cuối 1,004–1,027 sau 26 năm) — đúng như dự
đoán từ đầu dự án: carry là một lợi thế **nhỏ**, hệ thống không hứa hẹn lợi
nhuận lớn, chỉ hứa **không cháy** trong khi khai thác lợi thế nhỏ đó.

### Chọn phần Kelly, kiểm tra nhất quán kiểm định / kiểm tra

Phần Kelly **đầy đủ (1,0)** cho tăng trưởng cao nhất trong số không cháy tài
khoản, nên được chọn để soi riêng hai đoạn:

| đoạn | TB (bp/ngày) | Sharpe | sụt giảm tối đa | CVaR5% (bp) | cháy? | vốn cuối |
|---|---|---|---|---|---|---|
| kiểm định | 0,12 | 0,14 | 2,5% | −33,6 | không | 1,007 |
| kiểm tra | 0,37 | 0,23 | 5,8% | −60,0 | không | 1,021 |

Cả hai đoạn đều **dương và không cháy** — kiểm tra (ngoài mẫu chọn cấu hình)
thậm chí tốt hơn kiểm định, không có dấu hiệu quá khớp rõ rệt.

## Điều có vẻ mâu thuẫn với mục trước, và vì sao KHÔNG mâu thuẫn

Mục "khớp nối đòn bẩy" (`docs/TANG6B_DUNGTOIUU.md`) đo trên **lệnh ép mở mỗi
ngày** đã kết luận Kelly đầy đủ cho CVaR **tệ hơn không vay** (−210 so với
−127 bp mỗi lệnh). Ở đây, cùng Kelly đầy đủ, CVaR5% **theo ngày** của cả danh
mục chỉ −50,5 bp và không hề cháy tài khoản. Hai con số đo **hai thứ khác
nhau**, không phải một kết quả phủ định kết quả kia:

- Phép đo cũ **ép mở lệnh mỗi ngày bất kể tầng 6b có muốn hay không** — một bài
  kiểm tra sức chịu đựng (stress test) cố tình bỏ qua quyền chọn không tham
  gia.
- Phép đo mới để tầng 6b **tự chọn khi nào tham gia và khi nào rút sớm** —
  đúng cách hệ thống thật sự vận hành. Chính khả năng "không mở khi không đáng,
  đóng sớm khi hết đáng" là thứ hấp thụ phần lớn rủi ro đuôi mà Kelly đầy đủ
  tạo ra khi bị ép tham gia liên tục.

**Bài học phương pháp:** đo rủi ro của một quy tắc định cỡ mà không cho quy
tắc dừng đi kèm nó cơ hội hoạt động thì sẽ đánh giá quá tay. Tầng 6b không chỉ
"nắn lại phân phối" từng lệnh (kết luận cũ, vẫn đúng) — nó còn là **lớp bảo vệ
rủi ro hệ thống** khi Kelly đầy đủ được dùng cho danh mục liên tục.

## Giới hạn phải ghi vào luận văn

1. **Chưa áp `k_danh_mục` liên cặp.** Đây là hạn chế lớn nhất. Sáu khoang vốn
   được cộng bình quân như thể độc lập; thực tế các cặp carry-dương có xu
   hướng mở **cùng lúc** (carry là biến vĩ mô, tương quan giữa các cặp không
   phải 0 — xem `docs/TANG4_DANHMUC.md`). Nếu áp đúng `k_danh_mục` theo số cặp
   đang mở mỗi ngày, đòn bẩy mỗi cặp sẽ bị co lại khi nhiều cặp cùng mở, và kết
   quả (đặc biệt CVaR, sụt giảm tối đa) sẽ **thận trọng hơn** bảng trên. Đây là
   việc nên làm tiếp theo trước khi coi con số ở đây là "đã đủ an toàn để
   dùng thật".
2. **Vẫn trên đoạn kiểm định+kiểm tra, chưa chạm tập khóa sổ.** Theo đúng luật
   của `split.py`: mọi con số ở đây dùng để **chọn** cấu hình (ở đây là chọn
   phần Kelly), nên chưa phải con số báo cáo cuối cùng.
3. **Chính sách giữ/đóng dùng xấp xỉ trạng thái ổn định** (tra bảng tại
   `n=N` mỗi ngày thay vì đúng chỉ số ngày còn lại của một lệnh hữu hạn-hạn) —
   hợp lý cho vận hành vô hạn-hạn liên tục, nhưng là một xấp xỉ, chưa chứng
   minh hình thức là tối ưu cho bài toán vô hạn-hạn.
4. **Không tính phí lúc MỞ lệnh** (giữ đúng quy ước `mo_phong()` cũ để so sánh
   được) — nếu tính phí hai chiều thật, tăng trưởng sẽ thấp hơn bảng trên một
   chút.

## Tái lập

```bash
python src/run_e2e.py
```
