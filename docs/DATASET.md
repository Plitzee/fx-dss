# Từ điển dữ liệu và bản đồ luồng

Cập nhật 29/08/2026. Mọi file nằm trong `data/`.

## 1. Bản đồ nhanh — file nào nuôi tầng nào

| File | Tầng tiêu thụ | Vai trò |
|---|---|---|
| `prices/{PAIR}_d1.csv` | 2, 3 | OHLC ngày → 5 ước lượng biến động |
| `prices/{PAIR}_h1.csv` | 2 | Thanh giờ → RV cũ, giữ để tái lập |
| `rv_multi.csv` | **2** | `rv_m5` là **biến mục tiêu** của toàn bộ so sánh mô hình |
| `panel_6pairs.csv` | **3, 4** | Trạng thái rủi ro đã chuẩn hóa — đầu vào của RL và fuzzy |
| `cost_table.csv` | **5, 6** | Chi phí giao dịch tra cứu (chế độ × cặp × giờ) |
| `cost_elasticity.json` | 5 | Hiệu chỉnh chi phí theo biến động |
| `spread_hourly_all.csv` | 5 | Dữ liệu thô sinh ra hai file trên; dùng khi cần đo lại |
| `carry.csv` | 2b | Chênh lệch lãi suất — tín hiệu hướng đi chưa khai thác |
| `fred_rates.csv` | 2b | Lãi suất gốc, để tự tính carry cho cặp khác |
| `dukas_volume.csv` | — | Chưa tầng nào dùng |
| `rv_adv.csv` | **2** | RV + quarticity + bipower + semivariance 5 phút — đầu vào của `volfc.py` |
| `panel2_6pairs.csv` | **3, 4, 6** | Panel rủi ro ĐANG DÙNG, dựng bằng dự báo mới |

## 2. Chi tiết từng file

### `prices/{PAIR}_d1.csv` — 4.994 dòng/cặp, 6 cặp
```
Date, open, high, low, close, n_bars
2010-01-03, 1.4301, 1.4337, 1.4298, 1.4316, 100
```
`Date` chuỗi `YYYY-MM-DD`, **giờ UTC**. `n_bars` là số nến M1 gộp lại
trong ngày — chỉ báo độ đầy đủ dữ liệu. Cột này từng tên là
`tick_volume`, đã đổi ngày 29/08/2026 vì cái tên cũ khiến người đọc
tưởng là khối lượng giao dịch (tương quan với số nến M1 đúng bằng
1,000000).

Sinh ra bằng gộp 34,9 triệu nến M1 HistData, nên high/low **chính xác ở
độ phân giải phút** thay vì tin vào nến ngày của nhà cung cấp. Đây là lý
do phải tải M1 thay vì tải thẳng D1.

Liền mạch tuyệt đối: khoảng cách lớn nhất giữa hai phiên liên tiếp là 3
ngày (cuối tuần), không lỗ hổng nào trên 4 ngày ở cả 6 cặp × 16 năm.

### `prices/{PAIR}_h1.csv` — ~98.700 dòng/cặp
Cùng lược đồ, `Date` là `YYYY-MM-DD HH:MM:SS`. Trung vị 24 thanh/ngày.
Giữ lại để tái lập kết quả cũ; mục tiêu RV hiện dùng khung 5 phút.

### `rv_multi.csv` — 29.961 dòng
```
Date, pair, rv_m1, n_m1, rv_m5, n_m5, rv_m15, n_m15, rv_h1, n_h1
2010-01-04, EURUSD, 0.000040, 1351, 0.000039, 287, 0.000045, 95, 0.0000376, 23
```
Realized variance ngày ở bốn tần suất lấy mẫu. `rv_*` là **phương sai**
(không phải độ lệch chuẩn), đơn vị là bình phương lợi suất log.
`n_*` là số quan sát dùng để tính — **lọc theo cột này**, ngày Chủ nhật
chỉ có `n_m5` khoảng 20–60 thay vì ~287.

**`rv_m5` là mục tiêu chuẩn.** Lý do chọn 5 phút: nhiễu vi cấu trúc ở
khung 1 phút làm RV phồng 5%, còn khung giờ hụt 8–13%. Khung 5 phút là
điểm cân bằng, và là chuẩn trong tài liệu từ Andersen–Bollerslev.

Quy ước: RV = tổng bình phương lợi suất **trong ngày**, bỏ lợi suất bắc
qua ranh giới ngày. Nghĩa là nó **không** chứa gap qua đêm — nhưng gap
qua đêm chỉ chiếm 1,7–3,1% tổng phương sai ở FX nên bỏ qua được.

### `panel_6pairs.csv` — 29.843 dòng
```
Date, pair, sig, zT, zL, zH
2010-01-26, EURUSD, 0.005393, -0.866955, -1.394362, 0.405811
```
Trạng thái rủi ro đã chuẩn hóa, **đầu vào trực tiếp của tầng 4**:

- `sig` — độ lệch chuẩn ngày dự báo, từ MA20 của Garman-Klass, **đã dịch
  một phiên** nên hoàn toàn nhân quả (chỉ dùng thông tin tới t−1)
- `zT` — lợi suất đóng-đóng chia cho `sig`
- `zL` — mức thấp nhất trong ngày chia cho `sig` (luôn ≤ 0)
- `zH` — mức cao nhất trong ngày chia cho `sig` (luôn ≥ 0)

Bắt đầu từ 26/01/2010 vì cần 20 phiên đệm cho MA20.

**Cần biết:** panel này nhúng sẵn một lựa chọn mô hình — MA20-GK. Kết quả
mới cho thấy GARCH(1,1)-t ngang bằng về mặt thống kê (Diebold–Mariano
0/6 cặp). Nếu muốn kiểm tra kết luận tầng 4 có phụ thuộc lựa chọn này
không, phải dựng lại panel bằng GARCH-t và chạy song song.

### `cost_table.csv` — 288 dòng (2 chế độ × 6 cặp × 24 giờ)
```
regime, pair, hour, spread_med, spread_p95, spread_p99, n_ticks, n_obs
post2015, AUDUSD, 0, 0.905, 1.205, 1.905, 2220, 591
```
`hour` là **giờ UTC**. Spread tính bằng **pip**. `regime` là `pre2015`
hoặc `post2015` — spread FX nén lại một lần giữa 2014 và 2016 rồi phẳng
mười năm (trung bình 6 cặp: 3,50 → 2,45 → 0,70 pip), nên hai chế độ mô
tả đúng hơn một đường nội suy.

**Dùng `spread_med` cho chi phí thông thường, `spread_p95` cho thoát
buộc.** Tháng 3/2020 trung vị chỉ tăng 2,1–3,3 lần nhưng p95 tăng
19–115 lần. Khi lệnh dừng lỗ bị kích hoạt giữa khủng hoảng, thứ phải
trả là đuôi.

Đừng đọc file này trực tiếp — dùng `src/cost.py`, nó xử lý luôn phần
hiệu chỉnh theo biến động và cộng hoa hồng.

**Cảnh báo diễn giải:** số của `pre2015` toàn giá trị tròn (3,00 / 4,00)
và không phản ứng với biến động (R² 0,01–0,10). Đó gần như chắc chắn là
spread **cố định do nhà môi giới niêm yết**, không phải spread thị
trường. Với hệ thống dành cho nhà đầu tư cá nhân thì vẫn là chi phí phải
trả thật, nhưng phải nói rõ trong luận văn.

### `cost_elasticity.json`
```json
"EURUSD": {"beta_med": 0.432, "beta_p95": 0.589, "r2_med": 0.214,
           "r2_p95": 0.106, "sig_ref": 41.47, "n": 601}
```
Độ co giãn log-log của spread theo biến động, ước lượng trên dữ liệu từ
2015. `sig_ref` là mức biến động tham chiếu (đơn vị `sqrt(rv_m5)*1e4`).

Quan hệ này **yếu** — R² 0,00 đến 0,33 tùy cặp. Dùng làm hiệu chỉnh,
đừng dùng làm trụ cột. Điểm đáng chú ý: `beta_p95` lớn hơn `beta_med` ở
cả sáu cặp (tỷ lệ 1,4× đến 11,7×) — đuôi giãn nhanh hơn trung vị khi
biến động tăng.

### `spread_hourly_all.csv` — 103.504 dòng
```
Date, hour, pair, n_ticks, spread_med, spread_p95
2010-03-01, 5, AUDUSD, 284, 3.005, 4.005
```
Dữ liệu gốc sinh ra `cost_table.csv`. Tám thời kỳ mẫu: 2010, 2014, 2016,
2018, 2020, 2022, 2024, 2025 (riêng 2024 đủ 12 tháng, còn lại 3 tháng
mỗi năm). Đo từ tick thật có bid/ask, tổng khoảng 200 triệu tick.

`n_ticks` là **chỉ báo thanh khoản tốt nhất trong bộ dữ liệu** — số báo
giá mỗi giờ. Giờ 21 UTC có số tick thấp hơn giờ ban ngày 2,2–6,4 lần.

**Lọc cuối tuần trước khi phân tích.** Phiên Chủ nhật chỉ có 4 giờ mở
cửa và spread rất rộng; nếu không lọc sẽ ra kết luận sai — tôi đã dính
đúng lỗi này một lần khi tưởng Chủ nhật là ngày sốc COVID.

### `carry.csv` — 3.828 dòng
```
DATE, pair, carry
1994-01-01, EURUSD, 3.76
```
Chênh lệch lãi suất = lãi suất đồng **cơ sở** trừ lãi suất đồng **định
giá**, đơn vị %/năm. Với EURUSD là EUR trừ USD; với USDJPY là USD trừ
JPY.

**Dữ liệu tháng**, không phải ngày — forward-fill khi ghép vào panel
ngày. Chuẩn cho carry vì lãi suất chính sách đổi chậm.

Kiểm chứng dấu: AUD +1,98%/năm và NZD +3,14% (hai đồng carry kinh
điển), USDJPY +1,73% (JPY là đồng đi vay). Khớp với thực tế thị trường.

### `fred_rates.csv` — 4.751 dòng
Lãi suất ngắn hạn gốc của 8 đồng tiền: `DATE, rate, cur, code`. Giữ để
tự tính carry cho cặp không có sẵn trong `carry.csv`.

### `dukas_volume.csv` — 29.090 dòng
Số tick mỗi ngày trên feed Dukascopy (cột `n_ticks`, đổi từ `tick_volume`). **Thiếu 3/96 năm-cặp** (AUDUSD 2023,
EURUSD 2018, USDJPY 2025) — chạy `collect/finish_dataset.py --phase 2`
để bù. Hiện chưa tầng nào dùng; `n_ticks` trong spread là biến đại diện
thanh khoản tốt hơn.

## 3. Luồng dữ liệu

```
HistData M1 (34,9tr nến)          HistData tick (~200tr tick, có bid/ask)
        │                                        │
        ├─ prep_fx.py ──► prices/*_d1, *_h1      ├─ tick_spread.py
        │                                        │       │
        └─ rv5.py ──────► rv_multi.csv           │       ▼
                                │                │  spread_hourly_all.csv
                                │                │       │
                                ▼                │       ▼
    prices d1 ──► vol.py ──► 5 ước lượng         │  cost_table.csv
                                │                │  cost_elasticity.json
        ┌───────────────────────┤                │       │
        ▼                       ▼                │       ▼
   TẦNG 2 dự báo biến động   MA20-GK             │  TẦNG 5 chi phí
   (mục tiêu = rv_m5)           │                │       │
                                ▼                │       │
                        panel_6pairs.csv         │       │
                                │                │       │
                    ┌───────────┴────────┐       │       │
                    ▼                    ▼       │       │
              TẦNG 3 rủi ro       TẦNG 4 định cỡ ◄───────┘
              đường đi            (RL + fuzzy)
                                         │
    FRED ──► carry.csv ──► TẦNG 2B ──────┤
                          hướng đi        ▼
                                    TẦNG 6 phiếu quyết định
```

## 4. Những gì KHÔNG có, và hệ quả

**Không có sổ lệnh (order book).** Chỉ có báo giá tốt nhất. Không mô hình
hóa được tác động giá khi vào lệnh lớn. Không chặn đường vì hệ thống
nhắm nhà đầu tư cá nhân, nhưng phải nêu trong giới hạn.

**Không có khối lượng giao dịch thật.** FX là thị trường phi tập trung,
không tồn tại khối lượng tổng hợp. `n_ticks` là thứ gần nhất.

**Không có hoa hồng thật.** `src/cost.py` giả định 0,35 pip khứ hồi cho
tài khoản ECN. **Chưa kiểm chứng** — phải chạy phân tích độ nhạy 0,0 /
0,35 / 0,70 pip trước khi đưa vào kết luận nào.

**Không có tin tức hay sự kiện.** Không có lịch kinh tế, nên không tách
được biến động do sự kiện khỏi biến động nền.

**Spread không liên tục theo thời gian.** Tám thời kỳ mẫu chứ không phải
16 năm liền. Mô hình hai chế độ là cách xử lý; nếu ai cần độ chính xác
theo từng năm thì phải tải thêm.

## 5. Cạm bẫy đã biết

1. **Lọc cuối tuần** trước khi phân tích spread (`dayofweek < 5`).
2. **Lọc `n_m5`** trước khi dùng `rv_m5` — ngày Chủ nhật có rất ít quan sát.
3. **Rào chắn liền mạch** (`src/contig.py`) bắt buộc khi chạy trên tập
   khóa sổ, vì AUDJPY thiếu năm 2012. Trên tập phát triển nó vô hiệu.
4. **Giờ trong `cost_table` là UTC**, giờ trong tick gốc là New York.
   Việc chuyển đổi đã làm rồi, đừng chuyển lần nữa.
5. **`rv_*` là phương sai**, không phải độ lệch chuẩn. Lấy căn trước khi
   so với `sig` trong panel.
6. **Tập khóa sổ không nằm trong repo.** Đọc `KHOA_SO.md` trước khi tìm.


## 6. Cái gì là số đo, cái gì là suy ra, cái gì là giả định

Kiểm toán ngày 29/08/2026. Phân biệt này quan trọng vì trong luận văn,
số đo và giả định phải được phát biểu khác nhau.

### Số đo thật — lấy nguyên từ nguồn

| Đại lượng | Nguồn | Ghi chú |
|---|---|---|
| OHLC ngày và giờ | HistData M1 gộp lên | High/low chính xác ở độ phân giải phút |
| `n_bars` | Đếm nến M1 | Chỉ báo đầy đủ dữ liệu |
| `spread_med`, `spread_p95` | Tick HistData có bid/ask | Đo trực tiếp, ~200tr tick |
| `n_ticks` (spread_hourly) | Đếm tick | Biến đại diện thanh khoản tốt nhất |
| `n_ticks` (dukas_volume) | Feed Dukascopy | Của riêng một nhà môi giới, không phải thị trường |
| `fred_rates`, `DEX*` | FRED công bố | Số chính thức |

### Suy ra — tính từ số đo, có lựa chọn phương pháp

| Đại lượng | Lựa chọn đã làm | Hệ quả |
|---|---|---|
| `rv_m5` và các `rv_*` | Bỏ lợi suất bắc qua ranh giới ngày | Không chứa gap qua đêm (chỉ 1,7–3,1% phương sai) |
| `carry` | Lãi suất cơ sở trừ định giá, dữ liệu tháng | Phải forward-fill khi ghép vào panel ngày |
| `cost_elasticity` | Hồi quy log-log, mẫu từ 2015 | R² chỉ 0,00–0,33 — quan hệ yếu |

### **Đầu ra mô hình — KHÔNG phải số đo**

**`sig`** là một *dự báo*, không phải số đo, và toàn bộ `zT`, `zL`, `zH`
được chuẩn hóa bằng nó — nên mọi kết luận của tầng 3 và 4 đứng trên chất
lượng của nó.

| | panel cũ (`panel_6pairs.csv`) | panel mới (`panel2_6pairs.csv`) |
|---|---|---|
| mô hình | MA20-Garman-Klass, dịch 1 phiên | tổ hợp STHARQ+HARQ+SHAR (`src/volfc.py`) |
| tương quan với biến động thực | 0,435–0,647 | **0,522–0,720** |
| QLIKE ngoài mẫu | 0,2161 | **0,1645** |
| số hàng | 29.843 | 21.596 (phiên Chủ nhật đã gộp; đệm 250 phiên) |

Panel cũ giữ lại để đối chiếu, **không dùng cho kết quả mới**. Xem
`docs/TANG2_BIENDONG.md` về vì sao kết luận cũ ("HAR thua mọi mô hình
đơn giản") là sai.

### Giả định thuần túy — không đo được từ dữ liệu

| Tham số | Giá trị | Trạng thái |
|---|---|---|
| `COMMISSION_PIP` (`cost.py`) | 0,35 pip khứ hồi | **Chưa kiểm chứng**, nhưng đã chứng minh không load-bearing — xem `cost_sensitivity.py` |
| `REGIME_SPLIT` (`cost.py`) | 2015 | Chọn từ hình dạng dữ liệu, không phải từ sự kiện định trước |
| `max_gap_days` (`contig.py`) | 4 ngày | Cuối tuần là 3, cuối tuần có lễ là 4 |
| `MIN_TRAIN` | 250 phiên | Đánh đổi giữa độ dài đệm và số phiên chấm điểm |

### Cảnh báo về mã cũ

`src/experiment*.py` là các script từ giai đoạn đầu, còn chứa hằng số đã
lỗi thời — đáng chú ý nhất là `COST_PIPS = 1.0` trong `experiment6.py`,
đã bị `cost.py` thay thế. Đừng chạy chúng để lấy con số mới; chúng được
giữ lại để tái lập kết quả cũ.

`SR_TRUE = 0.5` trong `experiment7.py` là lợi thế **mô phỏng** cho môi
trường học tăng cường — đúng theo thiết kế, vì câu hỏi ở tầng 4 là *cho
trước một lợi thế, agent định cỡ ra sao*, không phải *có tìm được lợi
thế không*.
