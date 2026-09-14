# Vá đuôi — Lịch can thiệp tiền tệ (HƯỚNG THỨ NĂM, KẾT QUẢ)

*Lập 14/09/2026. Nối tiếp 4 hướng đã thất bại ở `docs/CHISO_DANHGIA.md` mục
5c/5d/5e và `docs/DUOI_EVT.md`. Khác các hướng trước ở chỗ dùng lịch CAN
THIỆP KHÔNG ĐỊNH KỲ (BOJ 2022, SNB 2015) thay vì lịch họp ĐỊNH KỲ.*

## Thiết kế

4 ngày BOJ can thiệp (2022-09-22, 2022-10-21, 2022-10-24, 2022-12-20 — đã
công khai, đã ghi trong `docs/KETQUA_VONG7.md`) và 1 ngày SNB bỏ sàn
(2015-01-15). Nới ngưỡng VaR bằng hệ số TRÒN (1,5×/2×/3×, chọn trước khi
xem khớp thế nào) trong cửa sổ ~5 phiên sau mỗi sự kiện. Script:
[`src/va_duoi_canthiep.py`](../src/va_duoi_canthiep.py).

## Kết quả — GẶP ĐÚNG BẾ TẮC ĐÃ CHẨN ĐOÁN Ở MỤC 5e

**USDJPY: V0 (mốc, KHÔNG có bất kỳ điều chỉnh nào) đã "ĐẠT" cả ba backtest
trên kiểm định, ở CẢ HAI mức:**

| mức | Kupiec p | Christoffersen p | DQ p |
|---|---|---|---|
| α=0,05, V0 | 0,609 | 0,596 | 0,863 |
| α=0,01, V0 | 0,529 | 0,670 | 0,988 |

Không có gì để sửa — mốc đã "đạt" trước khi thêm bất kỳ lớp can thiệp nào.
Thêm hệ số nới (1,5×/2×/3×) chỉ làm số vi phạm dao động nhẹ (7→5 ở α=0,01),
không có xu hướng rõ ràng, và ở một cấu hình (nới 1,5× tại α=0,01) DQ p còn
**giảm mạnh xuống 0,013** — tức đáng ngờ, không phải cải thiện.

**USDCHF: 0 sự kiện nằm trong kiểm định** (SNB 2015 nằm ở đoạn HUẤN LUYỆN,
trước 2021-10-13) — nới ngưỡng không đổi được gì (số liệu giống hệt V0 ở
mọi hệ số nới, vì không có ngày nào được đánh dấu "trong cửa sổ can thiệp"
trong đoạn đang chấm).

## Kết luận — xác nhận đúng bế tắc đã cảnh báo trước khi chạy

Đây **chính xác là bế tắc đã ghi ở `CHISO_DANHGIA.md` mục 5e**: vấn đề tail
của USDJPY chỉ lộ ra ở đoạn KIỂM TRA, không bao giờ lộ ra ở đoạn KIỂM ĐỊNH.
Vì V0 đã "đạt" trên kiểm định ngay từ đầu, không có tín hiệu nào để phân
biệt "lịch can thiệp có sửa được vấn đề thật hay không" — đúng hiện tượng
đã lặp lại ở CẢ NĂM hướng đã thử (phân vị mở rộng/cuộn, CAViaR, theo chế độ,
cửa sổ họp định kỳ, và giờ là lịch can thiệp không định kỳ).

**Đây KHÔNG phải lý do để nản** — nó xác nhận thêm một lần nữa rằng bế tắc
là ở CHÍNH GIAO THỨC (chọn-trên-kiểm-định không thể phân biệt được lỗi chỉ
tồn tại ngoài mẫu chọn), không phải do thiếu ý tưởng phương pháp. Ý tưởng
"lịch can thiệp" của phiên này VỀ MẶT LÝ THUYẾT là hợp lý hơn "lịch họp định
kỳ" (nhắm đúng cơ chế kinh tế đã chẩn đoán — can thiệp chính sách bất ngờ,
không phải họp thường lệ) — nhưng hợp lý về lý thuyết không giúp vượt qua
được bế tắc kiểm chứng.

**Khuyến nghị giữ nguyên như mục 5e đã kết luận**: chỉ có hai lựa chọn thật
— (a) chọn theo lý do kinh tế mà chấp nhận không kiểm chứng được, hoặc (b)
mở kiểm tra để CHẨN ĐOÁN (không chọn mô hình) và cam kết không quay lại sửa
dù kết quả thế nào. Đây là quyết định của người chịu trách nhiệm luận văn,
không phải việc kỹ thuật giải quyết được bằng thêm một biến thể nữa.
