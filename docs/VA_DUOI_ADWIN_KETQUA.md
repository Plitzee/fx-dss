# Vá đuôi — Phát hiện trôi thang đo trực tuyến ADWIN/EWMA (HƯỚNG THỨ SÁU, KẾT QUẢ)

*Lập 14/09/2026. Khác 5 hướng trước ở một điểm quan trọng: KHÔNG dựa vào
biết trước bất kỳ ngày sự kiện nào — dùng thuật toán phát hiện trôi trực
tuyến (ADWIN, Bifet & Gavaldà 2007) và tái ước lượng thích nghi (EWMA,
λ=0,94) để tự phát hiện độ trôi thang đo `sd(z)` đã chẩn đoán ở
`src/va_duoi.py`, hoàn toàn từ thống kê nội tại của chuỗi.*

Script: [`src/va_duoi_adwin.py`](../src/va_duoi_adwin.py)

## Kết quả 1 — ADWIN KHÔNG phát hiện được điểm đổi nào trong kiểm định

Chạy ADWIN trên luồng |z| theo đúng thứ tự thời gian (huấn luyện → kiểm
định), thử cả tham số mặc định (δ=0,002) lẫn độ nhạy cao hơn nhiều
(δ=0,05/0,1/0,2): **0 điểm đổi được phát hiện trong đoạn kiểm định, ở CẢ
HAI cặp, ở MỌI mức độ nhạy đã thử.**

**Diễn giải trung thực**: đây KHÔNG phải do chọn sai tham số — đã thử cả
dải độ nhạy rộng. Cơ chế thật: ADWIN được thiết kế để bắt **điểm đổi rõ
ràng giữa hai chế độ tương đối ổn định** (mean-shift), không phải **trôi
liên tục, chậm, đơn điệu** như đã chẩn đoán (sd(z) tăng dần 1,014→1,100→
1,136 qua BA đoạn dài, không phải một bước nhảy trong đoạn kiểm định). Bản
thân phát hiện "0 điểm đổi" này CŨNG là thông tin: nó củng cố thêm rằng độ
trôi thật sự KHÔNG đủ mạnh để thấy được chỉ bằng riêng đoạn kiểm định —
nhất quán với toàn bộ 5 chẩn đoán trước rằng vấn đề chủ yếu lộ ra khi so
sánh QUA CÁC ĐOẠN LỚN (huấn luyện/kiểm định/kiểm tra), không phải NGAY BÊN
TRONG một đoạn.

## Kết quả 2 — EWMA thích nghi: không cải thiện, làm USDCHF XẤU ĐI

| cặp | mức | V0 (mốc) | EWMA thích nghi |
|---|---|---|---|
| USDJPY | α=0,05 | Kupiec 0,609 · Chris 0,596 · DQ 0,863 | Kupiec 0,503 · Chris 0,146 · DQ 0,369 |
| USDJPY | α=0,01 | Kupiec 0,529 · DQ 0,988 | Kupiec 0,309 · DQ 0,923 |
| **USDCHF** | α=0,05 | Kupiec 0,609 · Chris **0,019** · DQ 0,066 | Kupiec **0,0045** · Chris 0,342 · DQ **0,0014** |
| **USDCHF** | α=0,01 | Kupiec **0,0154** · DQ **0,0031** | Kupiec **0,0060** · DQ **0,0012** |

USDJPY: mốc V0 vẫn "đạt" mọi chỉ số (như đã biết) — EWMA không cải thiện
rõ ràng, một vài chỉ số nhích lên, một vài nhích xuống, không có xu hướng.

**USDCHF: EWMA làm mọi thứ TỆ HƠN** — số vi phạm tăng (30→43 ở α=0,05,
12→13 ở α=0,01), Kupiec/DQ giảm mạnh hơn. Đáng chú ý: **USDCHF là cặp DUY
NHẤT trong toàn bộ 6 hướng đã thử có vấn đề LỘ RA THẬT trên kiểm định**
(Christoffersen/DQ đã thất bại ngay ở V0) — nghĩa là đây là lần hiếm hoi có
thể kiểm chứng công bằng, và kết quả kiểm chứng đó là ÂM rõ ràng.

## Kết luận

Hướng thứ 6 — **không ăn tiền**, nhưng khác 5 hướng trước ở một điểm quan
trọng: đây là hướng ĐẦU TIÊN không cần biết trước sự kiện cụ thể, và với
USDCHF, đây là lần HIẾM HOI có một phép kiểm chứng THẬT (không bị mù trên
kiểm định) — và kết quả kiểm chứng thật đó cho thấy EWMA thích nghi **làm
xấu đi**, không phải cải thiện. Với USDJPY, vẫn gặp lại đúng bế tắc cũ (V0
đã đạt sẵn, không có gì để phân biệt).

**Bài học phương pháp luận**: "phát hiện trôi tự động, không cần biết
trước sự kiện" là một ý tưởng đúng hướng về mặt lý thuyết (né được bế tắc
kiểm chứng), nhưng bản thân độ trôi của USDJPY quá CHẬM/YẾU để bất kỳ thuật
toán phát hiện điểm-đổi nào (kể cả rất nhạy) bắt được trong phạm vi một
đoạn dữ liệu ~2 năm. Đây không phải lỗi thuật toán — là bằng chứng thêm
rằng bản chất vấn đề là trôi CHẬM, DÀI HẠN (qua nhiều năm), không phải một
sự kiện hay giai đoạn ngắn có thể phát hiện cục bộ.

**Tổng cộng 6 hướng đã thử cho tail-risk USDJPY/USDCHF — không hướng nào
đạt.** Khuyến nghị giữ nguyên như đã kết luận ở `docs/CHISO_DANHGIA.md` mục
5e: đây là giới hạn cần trình bày trung thực trong luận văn, không phải
việc còn thiếu ý tưởng kỹ thuật.
