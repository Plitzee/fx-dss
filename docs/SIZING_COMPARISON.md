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

## Bổ sung 03/09/2026 — bandit ngữ cảnh có tìm ra điều kiện hoá không?

### Câu hỏi

Giả thuyết trong tài liệu (mục 3) là RL thất bại vì **credit assignment** —
lợi ích của việc giảm cỡ nằm sâu trong đuôi hiếm (0,07% đường đi), quá xa về
tương lai để policy-gradient qua GAE/lọc đuôi trên 250 bước nhìn thấy. Nếu
đúng, một phương pháp **bỏ hẳn credit assignment qua thời gian** — coi mỗi
ngày là một lượt chơi độc lập, học tham lam từ phần thưởng tức thời — có nên
tìm ra điều kiện hoá **tốt hơn** RL không, dù nó là một mô hình yếu hơn hẳn
(không thấy được toàn bộ đường đi)?

### Thiết kế

`src/compare_bandit.py`: bandit ngữ cảnh dạng bảng, 15 ô ngữ cảnh (3 bậc biến
động tương đối × 5 bậc sụt giảm — đúng 5 mốc 0/5/10/20/30% mà bảng RL đã
dùng), 11 mức hành động k ∈ [0,5; 1,5] (đúng khoảng PPO/CVaR-PPO), cập nhật
bằng trung bình mẫu cộng dồn (epsilon-greedy 1,0→0,03), cùng môi trường
`rl_env.SizingEnv`, cùng phần thưởng `pnl_log`, cùng 6 cặp × 3 seed × đoạn
kiểm tra 30% với PPO/CVaR-PPO.

### Kết quả

| | tăng trưởng | phá sản TB | phá sản tệ nhất |
|---|---|---|---|
| Trần trơn (k=1) | 7,41% | 0,14% | 0,38% |
| Điều kiện hoá tay | 7,40% | **0,00%** | **0,00%** |
| **Bandit ngữ cảnh** | **7,76%** | 1,09% | 2,25% |

| | dd=0% | dd=5% | dd=10% | dd=20% | dd=30% | **biên độ** |
|---|---|---|---|---|---|---|
| Điều kiện hoá tay | 1,440 | 1,263 | 1,085 | 0,731 | 0,554 | **0,886** |
| **Bandit ngữ cảnh** | 1,311 | 1,356 | 1,322 | 1,256 | 1,211 | **0,144** |
| PPO thường | — | — | — | — | — | 0,018 |
| CVaR-PPO | — | — | — | — | — | 0,030 |

### Đọc kết quả

**Giả thuyết credit-assignment được củng cố, không bị bác bỏ.** Bỏ hẳn việc
lan truyền phần thưởng qua nhiều bước — chính thứ PPO/CVaR-PPO cố làm bằng
GAE và lọc đuôi — cho biên độ điều kiện hoá **4,8–8 lần lớn hơn** RL (0,144 so
với 0,018–0,030). Cùng một phần thưởng, cùng một môi trường: khác biệt duy
nhất là bandit không phải giải bài toán "quy công cho hành động nào, ở bước
nào, trong 250 bước" — nó chỉ hỏi "hành động này, ở đúng ngữ cảnh này, trung
bình cho gì". Bài toán tín dụng dài hạn — chứ không phải bản thân tín hiệu
đuôi hiếm — mới là thứ chặn RL.

**Nhưng bandit vẫn thua xa quy tắc tay, và thua ở đúng chỗ quan trọng nhất.**
Biên độ 0,144 chỉ bằng **16%** của 0,886, và hướng cũng **không đơn điệu**
(tăng nhẹ từ dd=0% sang dd=5% rồi mới giảm — quy tắc tay giảm dần đều ngay từ
đầu). Quan trọng hơn: bandit có **tăng trưởng cao nhất** (7,76%) nhưng **phá
sản tệ nhất** (1,09% trung bình, 2,25% tệ nhất) — đắt đổi lấy tăng trưởng
bằng rủi ro đuôi, đúng thứ hệ thống này không được phép làm. Lý do hợp lý:
phần thưởng tức thời `pnl_log` chỉ phạt phá sản **đúng ngày nó xảy ra** (một
lần, `RUIN_PEN`); bandit không có khái niệm "trạng thái này đang tích luỹ rủi
ro cho tương lai" — nó thực sự đoản hạn theo đúng nghĩa đen của một bandit.

### Kết luận

**Không thay quy tắc tay.** Bandit ngữ cảnh là một cải tiến thật so với RL
(bằng chứng ủng hộ giả thuyết credit-assignment), nhưng không phải một ứng
viên thay thế: nó học *có* điều kiện hoá, nhưng vừa yếu vừa lệch đúng hướng
gây rủi ro nhất (tăng trưởng đổi lấy đuôi). Kết luận cũ của tài liệu này vẫn
đứng vững, giờ có thêm một điểm dữ liệu: **kiến thức miền không chỉ thắng
RL, nó thắng cả một phương pháp học đơn giản hơn RL nhưng nhìn thấy tín hiệu
rõ hơn.**

**Tái lập:**
```bash
python src/compare_bandit.py EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF 150
```
