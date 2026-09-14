# GIAI ĐOẠN 2 — KHAI PHÁ QUY LUẬT Ở h=5, h=20 VÀ MỤC TIÊU R (KẾT QUẢ)

*Lập 14/09/2026. Nhánh `replan-2026`. Đóng nốt hạng mục còn treo: đề cương
chỉ mới khai phá đầy đủ ở h=1 × mục tiêu P (`docs/GIAIDOAN2_QUYLUAT.md`).
Tài liệu này chạy 5 tổ hợp còn thiếu để đủ điều kiện dừng ở
`docs/REPLAN_2026.md` mục 10.4 ("cả hai mục tiêu R/P, cả ba tầm hạn 1/5/20").*

---

## 0. Phán quyết một dòng

> **0 quy luật sống sót Westfall-Young ở BẤT KỲ tổ hợp nào trong 6 tổ hợp
> (12 nhánh × 6 tổ hợp ≈ 47.700 lượt giả thuyết)** — kết luận sản xuất
> KHÔNG đổi. Nhưng kiểm định Hansen SPA (màn lọc cấp-họ, mềm hơn W-Y) bác
> bỏ giả thuyết không ở **5/54 phép kiểm sau hiệu chỉnh Holm, TOÀN BỘ ở mục
> tiêu R** (0 ở mục tiêu P) — nghĩa là câu chữ mục 10.4 REPLAN
> ("cả năm họ đều không bác bỏ SPA... trên CẢ HAI mục tiêu") **không được
> thoả mãn tuyệt đối**. Cần người chịu trách nhiệm luận văn quyết định cách
> viết phần này trong luận văn — xem mục 3.

---

## 1. Sửa một lỗi rò rỉ trước khi chạy — bắt buộc phải nói rõ

`src/run_quyluat.py:nap_du_lieu()` (dùng chung cho 9/10 script qua
`from run_quyluat import ...`) trước đây gọi `B.dung_muc_tieu(d, H, tr)`
**một lần duy nhất**, dùng kết quả đó cho CẢ đặc trưng `zs` (dùng để dựng
vị từ — cửa sổ motif, tìm analog Matrix Profile, đặc trưng "z hôm nay")
LẪN nhãn `y`. Với `H=1` điều này vô tình không rò rỉ, nhưng nếu đổi thẳng
`H` sang 5/20, `T["z"]` (đặc trưng) sẽ là lợi suất chuẩn hoá của **cả cửa
sổ [t, t+h-1]** — tức nhìn thấy h−1 ngày tương lai ngay trong đặc trưng
dùng để dự báo.

**Đã sửa** theo đúng mẫu đã có sẵn trong repo (`src/kiem_h3.py:muc_tieu()`,
viết từ trước cho đúng mục đích này): tách hai lời gọi
`B.dung_muc_tieu()` — một LUÔN ở h=1 cho đặc trưng `zs`, một theo tầm hạn
yêu cầu chỉ để lấy nhãn. Đã kiểm chứng bằng số: `zs` giống hệt bit-for-bit
dù đổi `QUYLUAT_H` sang 1/5/20. Cũng tăng độ dài khối hoán vị Westfall-Young
theo tầm hạn (`KHOI = max(5, h)`) vì cửa sổ mục tiêu chồng lấp ở h>1 tạo tự
tương quan MA(h−1) mạnh hơn — khối quá ngắn sẽ phóng đại sai lầm loại I.

Script điều phối: [`src/run_giaidoan2_tamhan.py`](../src/run_giaidoan2_tamhan.py)
(có sao lưu tự động, tự nhận diện phần đã chạy để chống mất việc khi máy
bị ngắt giữa chừng — đã xảy ra thật một lần trong lúc chạy, phục hồi đúng).

---

## 2. Bảng tổng hợp Westfall-Young — 6 tổ hợp

| tổ hợp | tổng giả thuyết (10 nhánh) | sống sót W-Y | ghi chú |
|---|---|---|---|
| h=1 × P (gốc, không chạy lại) | 8.469 | 0 | đã đóng băng, xem `GIAIDOAN2_QUYLUAT.md` |
| h=1 × R | 7.950 | **0** | |
| h=5 × R | 7.947 | **0** | |
| h=5 × P | 7.947 | **0** | |
| h=20 × R | 7.941 | **0** | |
| h=20 × P | 7.950 | **0** | |

**0/6 tổ hợp có bất kỳ quy luật nào sống sót phễu Westfall-Young** — đây là
chỉ số RA QUYẾT ĐỊNH SẢN XUẤT chính thức xuyên suốt dự án. Kết luận không
đổi so với trước khi chạy mở rộng này.

---

## 3. Nhưng: Hansen SPA bác bỏ ở 5/54 phép kiểm sau Holm — toàn bộ ở mục tiêu R

SPA là màn lọc CẤP HỌ (kiểm "ứng viên tốt nhất trong họ có thắng mô hình
tham chiếu 'chỉ σ̂' không"), MỀM HƠN Westfall-Young (kiểm TỪNG ứng viên có
sống sót sau khi hiệu chỉnh cho TOÀN BỘ ứng viên đã thử). Hai phép kiểm trả
lời hai câu hỏi khác nhau — SPA bác bỏ không mâu thuẫn với W-Y = 0 sống sót.

Chạy Holm trên **toàn bộ 54 phép kiểm SPA** (9 họ × 6 tổ hợp — bao gồm cả
h=1×P gốc):

| tổ hợp | họ | p thô | q (Holm) | dbar (độ lớn) | sống sót Holm |
|---|---|---|---|---|---|
| h=1×R | H2_motif | 0,0000 | 0,0000 | 0,0047 | **CÓ** |
| h=1×R | H3_rulelist | 0,0000 | 0,0000 | 0,0099 | **CÓ** |
| h=1×R | H6_hmm | 0,0010 | 0,0500 | 0,0056 | **CÓ** (biên) |
| h=1×R | H7_matrixprofile | 0,0000 | 0,0000 | 0,0141 | **CÓ** |
| h=5×R | H7_matrixprofile | 0,0000 | 0,0000 | 0,0152 | **CÓ** |

**Tất cả 5 đều ở mục tiêu R, 0 ở mục tiêu P** (kể cả h=1×P gốc: 0/9 họ bác
bỏ). h=20×R có 4 họ bác bỏ THÔ (p 0,018–0,028) nhưng KHÔNG sống sót Holm
khi xét trên toàn bộ 54 phép kiểm.

**Cơ chế khả dĩ, khớp đúng thiết kế đã ghi trong `REPLAN_2026.md` mục 3a**:
mục tiêu R chuẩn hoá lợi suất bằng σ̂ (z = r/σ̂), nên theo đúng lời tài liệu
gốc, mô hình tham chiếu "chỉ σ̂" trên R **"suy biến thành khí hậu học"** —
tức là một mốc YẾU. SPA vốn nhạy hơn Westfall-Young với mốc yếu vì nó kiểm
"có ứng viên nào thắng mốc" thay vì "ứng viên NÀY có đứng vững sau hiệu
chỉnh toàn bộ". Độ lớn kinh tế (`dbar`) ở cả 5 trường hợp đều **rất nhỏ**
(0,0047–0,0152) — cùng bậc độ lớn với các phát hiện "có ý nghĩa thống kê
nhưng không đáng kể kinh tế" đã gặp nhiều lần trong dự án (vd `PHA3B_KETQUA.md`).

**Đây KHÔNG phải khẳng định "tìm ra quy luật trên mục tiêu R"** — 0 ứng
viên cụ thể nào trong 4 họ này sống sót Westfall-Young (xem bảng mục 2),
kể cả họ H6_hmm có 0 ứng viên đạt p thô < 0,05 dù SPA vẫn bác bỏ. Đây là
tín hiệu Ở CẤP ĐỘ TỔNG HỢP của cả họ, không quy được về một quy tắc cụ thể
nào để đưa vào sản xuất.

---

## 4. Đối chiếu với tiêu chí dừng — cần quyết định của người chịu trách nhiệm

`docs/REPLAN_2026.md` mục 10.4 yêu cầu CẢ BA điều kiện đồng thời để tuyên bố
tiền đề khai phá quy luật không đứng được:

1. Cả năm họ đều không bác bỏ SPA ở α=0,05 — **KHÔNG thoả mãn tuyệt đối**
   (5/54 bác bỏ sau Holm, xem mục 3)
2. Không quy luật nào đạt ngưỡng sau LOPO — **THOẢ MÃN** (0/6 tổ hợp, mục 2)
3. Đúng trên cả hai mục tiêu và cả ba tầm hạn — điều kiện 1 chỉ vi phạm ở
   **mục tiêu R**, mục tiêu P sạch ở cả ba tầm hạn

**Đề xuất cách viết trung thực cho luận văn** (không phải quyết định cuối,
cần người chịu trách nhiệm xác nhận): tuyên bố tiêu chí dừng đạt trên
**mục tiêu P** (dùng cho sản phẩm hiển thị UI) ở cả ba tầm hạn, và báo cáo
riêng phát hiện SPA cấp-họ trên mục tiêu R như một quan sát phụ đã giải
thích được cơ chế (mốc suy biến) và đã xác nhận không có quy luật cụ thể
nào sống sót — KHÔNG tuyên bố "tiêu chí dừng đạt tuyệt đối trên mọi mục
tiêu" vì câu đó không khớp số liệu.

---

## 5. Việc chưa làm trong phạm vi này

- **SAX (nhánh 1, 2)** — `run_sax_stats.py`, `run_sax_gia.py` không dùng
  chung `nap_du_lieu()`, cần viết lại code (không chỉ đổi tham số) để dựng
  mục tiêu ở h=5/20. CHƯA làm trong lượt này — nếu cần cho đủ "12/12 nhánh
  ở mọi tổ hợp", đây là việc còn lại duy nhất.
- Chưa chạy `kiem_pheu.py`/`kiem_fdr.py` (phễu-kiểm-lực MDES) cho các tổ
  hợp mới — REPLAN mục 10.4 không bắt buộc việc này để đóng tiêu chí dừng,
  nhưng nên làm nếu luận văn muốn báo cáo lực phát hiện ở h=5/20 giống như
  đã làm ở h=1.
