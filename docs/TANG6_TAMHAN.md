# Tầng 6 — Nhiều tầm hạn, và một phương pháp hiện đại đã thử rồi loại

Lập 29/08/2026.

## 1. Lỗ hổng: phiếu hiệu chuẩn cho một phiên, người dùng giữ lệnh nhiều ngày

Mọi con số rủi ro trên phiếu — P(chạm stop), khoảng giá — đều là con số **một
phiên**. Không chỗ nào nói ra điều đó, và người dùng nào giữ lệnh qua tuần sẽ
đọc nhầm.

Mức độ nghiêm trọng, đo trên tập giữ riêng (6 cặp, ~21.500 quan sát mỗi tầm hạn),
stop đặt **cố định** tại k·σ rồi giữ h phiên:

| Tầm hạn | stop 1σ | stop 2σ | stop 3σ |
|---|---|---|---|
| 1 phiên | 27,2% | **5,1%** | 1,0% |
| 5 phiên | 62,4% | 33,8% | 16,2% |
| 10 phiên | 73,0% | **49,7%** | 31,5% |
| 20 phiên | 81,0% | 63,1% | 47,3% |

Với stop 2σ: đọc "5%" rồi giữ mười phiên thì thực tế là **50%**. Sai mười lần,
và sai theo hướng nguy hiểm.

## 2. Công thức phản xạ có kéo dài được không? — Có

| Tầm hạn | Lệch tuyệt đối trung bình | Hướng |
|---|---|---|
| 1 phiên | 0,4% | — |
| 5 phiên | 1,3% | dự báo **thấp hơn** thực tế |
| 10 phiên | 1,7% | dự báo thấp hơn thực tế |
| 20 phiên | 1,7% | dự báo thấp hơn thực tế |

Sai số vẫn dưới 2 điểm phần trăm, và lệch theo hướng lạc quan — phải ghi rõ.
Không cần mô hình mới; chỉ cần **in con số ra**. Phiếu quyết định giờ có dòng:

```
│   Nếu GIỮ LỆNH lâu hơn, xác suất chạm stop tăng nhanh:         │
│     1 phiên: 5%   5 phiên: 35%   10 phiên: 52%   20 phiên: 65% │
```

## 3. Quy tắc √h đúng trung bình nhưng lệch theo chế độ

σ_h = σ_1·√h là giả định ngầm ở khắp nơi. Đo độ lệch chuẩn thực chia cho σ·√h:

| Tầm hạn | Toàn bộ | vol thấp | vol vừa | vol cao |
|---|---|---|---|---|
| 1 | 1,010 | 1,032 | — | 1,001 |
| 5 | 1,007 | 1,068 | 0,953 | 0,894 |
| 10 | 1,011 | 1,093 | 0,972 | 0,860 |
| 20 | 1,006 | 1,137 | 1,015 | 0,839 |

Trung bình gần như hoàn hảo (1,006–1,011) nhưng **có điều kiện thì lệch tới
±14%**: đang yên thì rủi ro tương lai lớn hơn √h, đang căng thẳng thì nhỏ hơn.
Đó là biến động hồi quy về trung bình, và √h không biết điều đó.

Hệ số hiệu chỉnh c(h, chế độ) ước lượng **trên tập huấn luyện**, áp lên tập giữ
riêng:

| Tầm hạn | vol thấp trước → sau | vol cao trước → sau |
|---|---|---|
| 5 | 1,124 → 1,049 | 0,894 → 0,895 |
| 10 | 1,118 → 1,013 | 0,860 → 0,890 |
| 20 | 1,120 → 0,959 | 0,839 → 0,962 |

Biên độ lệch từ 0,839–1,124 về 0,890–1,057 — giảm khoảng **một nửa**. Cài trong
`decision_record.TamHan`.

## 4. Một phương pháp hiện đại đã thử và KHÔNG dùng — biến động phụ thuộc đường đi

Tài liệu 2023–2025 về *path-dependent volatility* (Guyon–Lekeufack) cho rằng
biến động phụ thuộc cả **quỹ đạo giá**, không chỉ mức biến động quá khứ: một
đặc trưng xu hướng R₁ (tổng có trọng số mũ của lợi suất **có dấu**) và một đặc
trưng biến động R₂, mỗi cái hai thang thời gian.

Đã cài và đưa vào so sánh, cùng thủ tục walk-forward như tầng 2:

| Mô hình | QLIKE TB | Hạng TB |
|---|---|---|
| STHARQ-PD | **0,1599** | **2,7** |
| EN + PDV | 0,1601 | 3,3 |
| STHARQ | 0,1604 | 4,2 |
| **EN hiện tại** | **0,1604** | **4,2** |
| HARQ | 0,1624 | 6,5 |
| HAR-PD | 0,1628 | 8,2 |

Cải thiện **0,3%** — nhất quán (5/6 cặp có t âm) nhưng **không cặp nào đạt
p<0,05**, và kiểm định dấu cũng không đạt (P(≥5/6) = 0,109). Model Confidence
Set giữ cả hai.

**Quyết định: KHÔNG đổi tầng 2.** Lý do: 0,3% so với 24% mà lần đổi trước thu
được, tức nhỏ hơn 80 lần; và đổi mô hình sản xuất dựa trên một cải thiện không
có ý nghĩa thống kê, đo trên chính tập phát triển, đúng là kiểu điều chỉnh mà
`KHOA_SO.md` được viết ra để ngăn. Ghi lại như một phương pháp đã thử.

## 5. Tái lập

```bash
python src/decision_record.py     # tự kiểm, gồm bảng tầm hạn
```
