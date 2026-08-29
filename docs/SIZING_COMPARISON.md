# So sánh phương pháp định cỡ vị thế

Thực hiện 29/08/2026. Tái lập bằng `src/compare_sizing.py` và `src/compare_rl.py`.

## Câu hỏi

Tầng 4 nhận σ̂ dự báo, xác suất chạm stop, sụt giảm hiện tại và mức lợi thế
mà nhà đầu tư tin mình có, rồi phải trả về một cỡ vị thế. Ý tưởng ban đầu là
dùng logic mờ. Câu hỏi thật: **logic mờ có thực sự tốt hơn các phương pháp
khác không, hay chỉ là một lựa chọn nghe hợp lý?**

## Thiết kế

Chín phương pháp, cùng môi trường, cùng đoạn giữ riêng 30%, cùng seed.

Điểm then chốt của thiết kế: **so sánh ở cùng mức xác suất phá sản**, không
so ở tham số tùy chọn. Mỗi họ phương pháp được quét qua dải tham số để dựng
biên hiệu quả, rồi nội suy tăng trưởng tại các mức rủi ro chung. So ở một
điểm tham số duy nhất sẽ cho kết luận tùy tiện — đó là lỗi tôi mắc ở lần
chạy đầu và phải làm lại.

Harness mở rộng (`sizing2.py`) truyền thêm trạng thái đường đi cho quy tắc.
Đã kiểm chứng nó **tái lập trùng khớp tuyệt đối** kết quả của `sizing.py`
trên ba quy tắc cũ trước khi dùng.

## Kết quả — tăng trưởng tại cùng mức phá sản

| Phương pháp | 0,1% | 0,3% | 1,0% | 3,0% |
|---|---|---|---|---|
| Trần trơn (không điều kiện hóa) | 12,59% | 13,37% | 15,99% | 17,24% |
| Trần × 1,08 — hệ số PPO học được | 12,91% | 13,79% | 15,37% | 17,43% |
| **Trần × tích hai hệ số** | **15,88%** | 16,19% | 17,28% | 17,56% |
| Trần × fuzzy Mamdani | 14,96% | **16,22%** | 17,24% | **18,81%** |

Ở cùng ngân sách rủi ro b = 0,03:

| Phương pháp | Tăng trưởng | Phá sản |
|---|---|---|
| PPO (k = 1,073) | 16,73% | 1,54% |
| Trần trơn | 15,97% | 0,97% |
| CVaR-PPO | 15,80% | 0,72% |
| **Điều kiện hóa tay** | **15,81%** | **0,06%** |

Cùng tăng trưởng, phá sản thấp hơn 16 lần so với trần trơn và 26 lần so với PPO.

## Ba kết luận

### 1. Điều kiện hóa theo trạng thái là thứ tạo ra khác biệt

Cho hệ số co giãn theo sụt giảm và biến động cho thêm khoảng **3,3 điểm phần
trăm** tăng trưởng ở mức phá sản 0,1%. Đây là mức tăng lớn và nhất quán qua
mọi mức rủi ro.

### 2. Fuzzy không đáng dùng

Chênh giữa fuzzy Mamdani và một tích hai hệ số tuyến tính: **+0,08%** — nằm
trong nhiễu giữa các seed.

Phép phân rã còn cho thấy fuzzy **không siêu cộng tính**. Đóng góp riêng lẻ:
hệ số sụt giảm −1,06% tăng trưởng (nhưng phá sản 1,07% → 0,05%); hệ số biến
động +1,64% (phá sản → 2,80%). Cộng lại +0,58%; fuzzy kết hợp cho −0,00%.
Không có tương tác nào được tạo ra. Chín luật Mamdani, ba hàm thuộc và phép
giải mờ trọng tâm chỉ tái tạo `k = k_vol × k_dd`.

### 3. Học tăng cường không tìm ra điều kiện hóa này

Hệ số k mà chính sách học được, đo tại các mức sụt giảm khác nhau:

| | dd=0% | dd=10% | dd=30% | biên độ |
|---|---|---|---|---|
| PPO thường | 1,073 | 1,070 | 1,055 | **0,018** |
| CVaR-PPO (α=0,20) | 0,994 | 0,982 | 0,964 | **0,030** |
| Quy tắc tay | 1,30 | 0,98 | 0,50 | **0,800** |

Sụt giảm **có** trong vector trạng thái (`rl_env.py`, cột 3). Nên đây không
phải vấn đề thiếu thông tin.

Giả thuyết ban đầu — PPO hỏng vì lợi ích nằm ở phần đuôi, nên mục tiêu nhạy
đuôi sẽ sửa được — **đúng về hướng, sai về độ lớn**. CVaR-PPO giảm k khi lỗ
sâu (đúng chiều, trong khi PPO thường gần như phẳng và hơi ngược chiều),
nhưng biên độ vẫn kém 27 lần.

Và đáng chú ý: **huấn luyện lâu hơn làm biên độ nhỏ đi**, không lớn lên
(0,098 ở 40 vòng → 0,030 ở 100 vòng). Chính sách hội tụ *rời xa* điều kiện
hóa. Lý do hợp lý: phá sản chỉ chiếm 0,07% số đường đi, nằm sâu bên trong
phần đuôi 20% mà CVaR-PPO tối ưu — ngay cả mục tiêu nhạy đuôi cũng gần như
không nhìn thấy nó.

## Giới hạn của kết luận này

**Hệ số của quy tắc tay có ảnh hưởng.** Nhiễu ±30% quanh giá trị thiết kế cho
tăng trưởng 14,52%–18,46% và phá sản lên tới 1,82%. Phần lớn biến thể chỉ
trượt dọc biên hiệu quả chứ không rơi khỏi nó, nhưng quy tắc **không** miễn
nhiễm với việc chọn tham số. Thứ bền vững là *hướng* điều kiện hóa.

**Quy tắc tay được cho một lợi thế mà RL không có: kiến thức miền.** Tôi chọn
các hệ số bằng phán đoán, không phải tối ưu từ dữ liệu. Đó vừa là điểm yếu
của phép so sánh, vừa chính là kết luận — hai dòng kiến thức miền đánh bại
100 vòng huấn luyện RL trong bài toán này.

**Lợi thế trong thí nghiệm là mô phỏng** (Sharpe rút ngẫu nhiên 0–1,2 khi
huấn luyện), theo đúng thiết kế: câu hỏi là *cho trước một lợi thế, định cỡ
ra sao*, không phải *có tìm được lợi thế không*.

## Quyết định

Đưa vào pipeline: `src/position_sizing.py` — `f = min(Kelly, k_vol × k_dd ×
trần_rủi_ro)`.

Không dùng fuzzy. Không dùng RL ở tầng quyết định.

**Giữ toàn bộ thí nghiệm RL trong luận văn.** Ba cấu hình (REINFORCE, PPO,
CVaR-PPO) thất bại theo cùng một cơ chế là một kết quả âm có giá trị, và nó
trả lời được câu *vì sao* — không phải một nhánh chết.
