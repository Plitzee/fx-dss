# Tổ hợp ba xác suất — Stacking làm phương pháp thay thế cho Hedge (KẾT QUẢ)

*Lập 14/09/2026. Trước đây tầng tổ hợp chỉ đối chiếu 2 cách TRỰC TUYẾN
(Hedge, Fixed-Share). Thử thêm STACKING (hồi quy logistic đa thức, khớp một
lần trên huấn luyện) — một họ phương pháp OFFLINE hoàn toàn khác.*

## Thiết kế

Bốn nền giống hệt sản xuất (khí hậu học, quán tính, chỉ σ̂, σ̂+chế độ), khớp
tĩnh một lần trên huấn luyện (đơn giản hoá có chủ đích so với bản cuộn của
sản xuất, để so sánh công bằng CÁCH TỔ HỢP, không lẫn với cách khớp nền).
Hai cách tổ hợp:
- **Hedge** (đang sản xuất): trọng số mũ cập nhật trực tuyến.
- **Stacking**: hồi quy logistic đa thức trên 12 đặc trưng (xác suất 4 nền
  × 3 lớp), khớp một lần trên huấn luyện.

Script: [`src/kiem_stacking_bahop.py`](../src/kiem_stacking_bahop.py) ·
Kết quả: [`output/kiem_stacking_bahop.json`](../output/kiem_stacking_bahop.json)

## Kết quả

| cách tổ hợp | BSS so khí hậu học | so khí hậu học |
|---|---|---|
| Hedge (sản xuất) | **+0,0106** | p=0,0022 |
| Stacking | +0,0091 | p=0,0157 |

**Hedge thắng Stacking có ý nghĩa** (DM p=0,0291, chênh BSS −0,0015). Cả
hai đều thắng khí hậu học, nhưng Hedge tốt hơn.

**Diễn giải**: Hedge thích nghi LIÊN TỤC theo thời gian (hạ trọng số chuyên
gia vừa sai), trong khi stacking khớp CỐ ĐỊNH một lần trên huấn luyện —
không bắt được thay đổi tương đối giữa các nền theo thời gian (vd nền nào
tốt hơn ở từng giai đoạn thị trường). Đây khớp đúng lý do gốc dự án chọn
Hedge: khả năng thích nghi trực tuyến quan trọng hơn sức mạnh biểu diễn
tĩnh của một mô hình hồi quy.

## Kết luận

**Xác nhận lựa chọn Hedge là hợp lý** — đã thử thêm một họ phương pháp khác
hẳn (offline/có giám sát thay vì trực tuyến), và Hedge vẫn thắng có ý nghĩa.
