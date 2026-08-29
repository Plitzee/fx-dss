# Kế hoạch giai đoạn tiếp theo

Lập 29/08/2026. Cập nhật khi hoàn thành từng mục.

## Trạng thái

| Phần | Mức | Ghi chú |
|---|---|---|
| Nghiên cứu (tầng 0–5) | **98%** | Tầng 2, 2b, 3 đã xong; còn lịch sự kiện |
| Đóng gói (tầng 6–7) | **55%** | Tầng 6 có mã + kiểm hiệu chuẩn; còn giao diện |
| Tổng | **85%** | |

Dataset đã chốt. Từ đây không cần tải thêm dữ liệu cho bất kỳ mục nào
dưới đây, trừ mục P4 (lịch sự kiện).

### ~~P0 · Tầng 2 — dự báo biến động~~ — XONG 29/08/2026

Phát hiện lỗi phiên Chủ nhật trong chuỗi hồi quy; đổi sang tổ hợp
STHARQ+HARQ+SHAR (`src/volfc.py`), QLIKE tốt hơn 24%, DM 6/6, MCS 6/6.
Panel mới `data/panel2_6pairs.csv`. Xem `docs/TANG2_BIENDONG.md`.

Còn lại ở mục này: chạy lại các kết luận của tầng 3 và 4 trên panel mới để
xem có phụ thuộc lựa chọn mô hình không (mục S4 cũ giờ đã trả lời một nửa).

## Đường găng

Bốn mục dưới đây phải làm tuần tự. Mọi thứ khác chạy song song được.

### ~~P1 · Tầng fuzzy~~ — XONG 29/08/2026, và kết luận là KHÔNG dùng fuzzy

So sánh 9 phương pháp trên biên hiệu quả. Fuzzy Mamdani không hơn một tích
hai hệ số tuyến tính (+0,08%, trong nhiễu). RL không tìm ra điều kiện hóa.
Quy tắc được chọn: `f = min(Kelly, k_vol × k_dd × trần)`, cài trong
`src/position_sizing.py`. Xem `docs/SIZING_COMPARISON.md`.

Còn lại ở mục này: bảng độ nhạy hệ số để đưa vào luận văn (hệ số đến từ phán
đoán chuyên môn, nhiễu ±30% cho phá sản 0,01%–1,82%).

### ~~P1 cũ~~ (giữ để đối chiếu)
**Ước lượng: 1 ngày. Chặn: P3.**

Đầu vào từ `panel_6pairs.csv` và `cost.py`:
- `sig` — độ lệch chuẩn dự báo
- `P(chạm stop)` — theo nguyên lý phản xạ
- sụt giảm hiện tại
- **mới:** chi phí kỳ vọng tại giờ giao dịch, từ `cost.py`

Mốc hàm thuộc lấy từ **tam phân vị của phân phối trong tập huấn luyện**,
không đặt tay. Luật Mamdani, giải mờ ra hệ số `k` nhân vào trần rủi ro.

Bắt buộc kèm **phân tích độ nhạy ±20%** trên các mốc — "sao chọn mốc đó"
là câu phản biện chắc chắn có.

Tiêu chí xong: `src/fuzzy.py` có self-test; bảng so sánh `k` fuzzy với
`k = 1,08` mà PPO học được; kết luận nói rõ fuzzy thêm được gì so với
quy tắc tĩnh, hoặc thừa nhận là không.

### P2 · Tầng 6 — phiếu quyết định và tầng giải thích
**Ước lượng: 2 ngày. Đây là phần lõi MIS và hiện chưa ai nhận.**

Không phụ thuộc P1: dựng bản đầu bằng quy tắc tĩnh (Kelly + trần rủi ro),
cắm fuzzy vào sau.

Mỗi phiên sinh ra một phiếu gồm: σ̂ dự báo, P(chạm stop), đòn bẩy khuyến
nghị **quy ra lot**, mức dừng lỗ gợi ý, chi phí kỳ vọng tại giờ giao dịch,
và mức tin cậy.

Tầng giải thích nêu rõ luật nào kích hoạt và với độ khớp bao nhiêu. Đây
là thứ phân biệt một *hệ thống hỗ trợ quyết định* với một *mô hình dự báo*
— không có nó thì luận văn là hai mô hình ghép lại.

Ba luật đã có bằng chứng sẵn sàng đưa vào:
1. Không mở vị thế lúc 21:00 UTC (đắt gấp 2–4 lần, thanh khoản tụt 2,2–6,4 lần)
2. Khi biến động cao, dùng chi phí phân vị 95 chứ không phải trung vị
3. Trần rủi ro giữ 0% phá sản ở mọi mức niềm tin sai — không vượt trần

Tiêu chí xong: sinh được phiếu cho một phiên bất kỳ trong tập phát triển,
mỗi con số truy được về nguồn.

### P3 · Chốt cấu hình, mở niêm phong, chạy đúng một lần
**Ước lượng: 4 giờ. Chặn: P1.**

Ghi cấu hình vào mục 4 của `KHOA_SO.md` **trước khi** chạy. Rào chắn liền
mạch bắt buộc bật (AUDJPY thiếu 2012). Kết quả ra sao báo cáo đúng như vậy
— không được quay lại sửa mô hình rồi chạy lại.

### P4 · Viết luận văn
**Ước lượng: 2–3 tuần. Chặn: P2, P3.**

Chất liệu có khoảng 70%. Repo, `DATASET.md` và `KHOA_SO.md` đã ở dạng
gần với phụ lục.

## Việc song song, không chặn đường găng

| Mã | Việc | Công | Giá trị |
|---|---|---|---|
| ~~S1~~ | ~~Kiểm định carry~~ **XONG 29/08** — Sharpe 2010–2025 là −0,05, không đạt ngưỡng 0,30. `src/carry_test.py` | — | Đã thu: kết luận âm thứ hai về hướng đi |
| **S2** | Lịch sự kiện (FOMC, ECB, BOJ, NFP, CPI) | 1 ngày | **Cao nhất còn lại**: đo trượt giá cho thấy 12–13h UTC nguy hiểm nhất, lịch sự kiện giải thích vì sao |
| ~~S3~~ | ~~Mô hình trượt giá~~ **XONG 29/08** — đo 60.617 lần chạm stop, K_SLIP=0,92. `src/slippage_model.py` | — | Đã thu: phá sản thật cao gấp 2,5 lần giả định cũ |
| S4 | Dựng lại panel bằng GARCH-t, chạy song song | 4 giờ | Trung bình: kiểm tra kết luận tầng 4 có phụ thuộc lựa chọn mô hình không |
| S5 | Gộp mẫu xuyên cặp cho nhánh HuyH | 2 ngày | Trung bình: HuyH mới khai phá trong từng cặp riêng |
| S6 | Phân tích riêng cú sốc SNB 15/01/2015 | 4 giờ | Trung bình: USD/CHF là ngoại lệ duy nhất ở tầng 2 |
| S7 | Tải spread liên tục 16 năm thay vì 8 mốc | 2 giờ máy | Thấp: mô hình hai chế độ đã bắt đúng cấu trúc |

## Không làm

**Sổ lệnh.** Không có nguồn miễn phí, và không cần: cách xử lý đúng là
đóng khung giả định — chứng minh cỡ vị thế khuyến nghị nằm dưới ngưỡng mà
báo giá tốt nhất còn khớp được.

**Thêm cặp tiền vào tập phát triển.** Sáu cặp × 16 năm đã cho gần 28.400
phiên ngoài mẫu; khoảng tin cậy đã rất hẹp. Sáu cặp chéo nằm trong tập
khóa sổ, mở ở P3.

## Rủi ro

**Tầng 6 chưa ai nhận.** Phần kỹ thuật đã xong và đã kiểm hiệu chuẩn
(`src/decision_record.py`, `docs/TANG6_HIEU_CHUAN.md`); phần còn lại là giao
diện và viết luận văn. Rủi ro giờ là tổ chức: cần chốt phân công trước P2.

**Cám dỗ chạy lại tập khóa sổ.** Nếu P3 cho kết quả xấu, áp lực sửa mô
hình rồi chạy lại sẽ rất lớn. `KHOA_SO.md` mục 3 đã ghi luật cấm; đọc lại
trước khi mở niêm phong.
