# H3 — RuleFit làm phương pháp thay thế cho CART (KẾT QUẢ)

*Lập 14/09/2026. Thử phương pháp thay thế cho nhánh H3 (rule-list), theo
đúng gợi ý tối ưu hoá đã đề xuất — H3 trước đây chỉ thử MỘT họ mô hình luật
(CART nông), chưa đối chiếu với phương pháp cạnh tranh nào.*

## Thiết kế

- **RuleFit** (Friedman & Popescu 2008, *Annals of Applied Statistics*) qua
  thư viện `imodels` — sinh luật ứng viên từ rừng cây nông, chọn lọc bằng
  Lasso. Một-chọi-phần-còn-lại cho 3 lớp (giảm/đi ngang/tăng), chuẩn hoá
  xác suất về tổng 1.
- Dùng LẠI nguyên vẹn đặc trưng, mục tiêu, huấn luyện/kiểm định của H3
  (`run_h3_rulelist.py:xay_cay`, `run_quyluat.py:nap_du_lieu`) — chỉ đổi
  lớp mô hình.
- Đánh giá bằng Brier Skill Score (BSS) so khí hậu học + kiểm định
  Diebold-Mariano trên đoạn KIỂM ĐỊNH (RuleFit tự chọn tham số Lasso bằng
  cross-validation NỘI BỘ trên huấn luyện, không nhìn kiểm định).

Script: [`src/kiem_rulefit_h3.py`](../src/kiem_rulefit_h3.py) ·
Kết quả: [`output/kiem_rulefit_h3.json`](../output/kiem_rulefit_h3.json)

## Kết quả — CART thắng rõ ràng

| mô hình | BSS so khí hậu học | so khí hậu học |
|---|---|---|
| CART (H3 gốc, max_depth=3) | **+0,0066** | p=0,035 (chưa hiệu chỉnh đa kiểm định) |
| RuleFit (30 luật, one-vs-rest) | **−0,0227** | p=0,0001 (**tệ hơn** khí hậu học, có ý nghĩa) |

**RuleFit thua CART có ý nghĩa mạnh** (DM p=0,0000, chênh BSS −0,029) —
RuleFit không chỉ không cải thiện, mà còn **tệ hơn cả một mô hình hằng số**
(khí hậu học). Diễn giải hợp lý nhất: với tín hiệu tài chính gần như nhiễu
thuần (đã xác nhận qua 14 nhánh khai phá khác), một họ mô hình LINH HOẠT
HƠN (RuleFit sinh nhiều luật ứng viên hơn một cây đơn) có xu hướng **quá
khớp nhiễu trên huấn luyện** nặng hơn, trong khi ràng buộc nghiêm của CART
(độ sâu 3, tối thiểu 200 mẫu/lá) hoạt động như một bộ chính quy hoá mạnh,
tình cờ phù hợp hơn với dữ liệu gần-nhiễu-thuần này.

**Lưu ý về con số BSS của CART**: p=0,035 ở đây là kiểm định TỔNG THỂ (một
lần, không hoán vị/hiệu chỉnh đa kiểm định), khác với phễu Westfall-Young
cấp-lá mà `run_h3_rulelist.py` gốc đã dùng (kết luận: 0/24 lá sống sót). Con
số này chỉ dùng làm MỐC SO SÁNH tương đối với RuleFit, không phải bằng
chứng mới cho việc mở lại quyết định H3 đã đóng.

## Kết luận

**Xác nhận lựa chọn CART của H3 là hợp lý** — không phải vì thiếu thử
nghiệm, mà vì đã kiểm chứng: phương pháp thay thế linh hoạt hơn (RuleFit)
cho kết quả tệ hơn rõ rệt trên chính bài toán này. Đây là câu trả lời cụ
thể, đo được, cho câu hỏi "H3 có nên thử phương pháp khác không" — có thử,
và CART vẫn là lựa chọn tốt hơn.
