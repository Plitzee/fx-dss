# Hệ thống hỗ trợ quyết định giao dịch ngoại hối — dữ liệu & pipeline

Luận văn tốt nghiệp MIS. Repo này chứa **dữ liệu đã xử lý và toàn bộ mã**.
Dữ liệu thô (khoảng 9,5 GB) không nằm trong git — tái tạo bằng `collect/`.

## Bắt đầu nhanh

```bash
git clone <repo> && cd fx-dss
pip install pandas numpy scipy arch
python src/cost.py          # tự kiểm mô hình chi phí
python src/contig.py        # tự kiểm rào chắn liền mạch
```

## Dữ liệu — `data/`

| File | Nội dung | Quy mô |
|---|---|---|
| `prices/{CAP}_d1.csv` | OHLC ngày, 6 cặp | 4.994 phiên/cặp, 2010-01-03 → 2025-12-31 |
| `prices/{CAP}_h1.csv` | Thanh giờ | ~98.700/cặp |
| `rv_multi.csv` | Realized variance 4 tần suất (1/5/15 phút, 1 giờ) | 29.961 dòng |
| `panel_6pairs.csv` | Panel rủi ro: sig, zT, zL, zH | 29.843 dòng |
| `cost_table.csv` | Chi phí giao dịch (chế độ × cặp × giờ) | 288 dòng |
| `cost_elasticity.json` | Độ co giãn spread theo biến động | 6 cặp |
| `spread_hourly_all.csv` | Spread thật theo giờ, 8 thời kỳ mẫu | 103.504 dòng |
| `carry.csv` | Chênh lệch lãi suất theo cặp (tháng) | 3.828 dòng |
| `fred_rates.csv` | Lãi suất ngắn hạn 8 đồng tiền | 4.751 dòng |
| `dukas_volume.csv` | Tick volume ngày | 29.090 dòng |
| `fred/DEX*.csv` | Tỷ giá ngày FRED, 6 cặp, 1971–2026 | dùng cho kiểm định suy giảm |

**Nguồn.** Giá và tick: HistData (nến M1 và tick quotes có bid/ask riêng),
34,9 triệu nến M1 gộp thành H1 và D1. Lãi suất: FRED. Khối lượng: Dukascopy.
Toàn bộ đã chuyển từ giờ New York sang UTC **có xử lý giờ mùa hè** — hiệu chuẩn
bằng thực nghiệm, không tin tài liệu; sai số còn lại 0,350 pip khi đối chiếu
chéo EUR/USD giữa hai nhà cung cấp trên 4.994 ngày.

## Ba điều cần biết trước khi dùng

**1. Đọc `docs/KHOA_SO.md` trước khi chạy bất cứ thứ gì.**
Có một tập dữ liệu bị **niêm phong** (6 cặp chéo + toàn bộ 2026) và nó *không*
nằm trong repo này, có chủ đích. Nó chỉ được mở đúng một lần, sau khi cấu hình
cuối đã chốt và ghi vào biên bản. Nếu bạn phân tích nó sớm thì toàn bộ kết quả
mất tính ngoài mẫu và không ai biết.

**2. Chi phí giao dịch không phải hằng số.**
Spread FX nén lại một lần giữa 2014 và 2016 rồi phẳng (trung bình 6 cặp:
3,50 → 2,45 → 0,70 pip), và chênh nhau 6 lần giữa cặp rẻ nhất và đắt nhất.
Dùng `src/cost.py`, đừng dùng số 0,91 pip trong mã cũ.

**3. Dùng phân vị 95 cho chi phí thoát buộc, không dùng trung vị.**
Tháng 3/2020 trung vị chỉ tăng 2,1–3,3 lần nhưng p95 tăng **19–115 lần**
(EUR/USD ngày 09/03: trung vị 0,51 pip, p95 33,91 pip). Khi lệnh dừng lỗ bị
kích hoạt giữa khủng hoảng, thứ phải trả là đuôi.

## Mã — `src/`

| File | Vai trò |
|---|---|
| `fxdata.py` | Nạp dữ liệu. `realized_var(pair, freq="m5")` là mục tiêu chuẩn |
| `cost.py` | Mô hình chi phí `spread_pip(cặp, giờ, ngày, q)` — có tự kiểm |
| `contig.py` | Rào chắn liền mạch: cửa sổ trượt không bắc qua lỗ hổng — có tự kiểm |
| `vol.py` | Ước lượng biến động: cc, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang |
| `sizing.py` | Quy tắc định cỡ vị thế, gồm Kelly + trần rủi ro |
| `rl_env.py`, `rl_agent.py` | Môi trường và tác tử học tăng cường |
| `metrics.py` | QLIKE, MSE, Model Confidence Set |
| `run_guard.py` | Chạy walk-forward 6 cặp, chấm điểm trên cả hai mục tiêu |
| `momentum_decay.py` | Suy giảm momentum qua 55 năm — có tự kiểm |
| `cost_sensitivity.py` | Độ nhạy hoa hồng, chứng minh không load-bearing — có tự kiểm |

## Thu thập lại dữ liệu thô — `collect/`

Chạy trên máy có mạng ra ngoài (không chạy được trong sandbox nghiên cứu):

```bash
python collect/histdata_dl.py                 # nến M1, 6 cặp × 16 năm  (~10 phút)
python collect/prep_fx.py                     # M1 → H1 → D1, hiệu chuẩn múi giờ
python collect/rv5.py --pair EURUSD           # realized variance, từng cặp (~19s/cặp)
python collect/tick_spread.py --years 2024    # spread thật từ tick   (~15 phút)
python collect/finish_dataset.py --phase 1    # lãi suất FRED         (~1 phút)
```

Mọi script đều có resume: ngắt giữa chừng rồi chạy lại chỉ tải phần còn thiếu.

## Kết quả chính tính đến 29/08/2026

Không dự báo được **hướng đi**, và lý do quan trọng hơn kết luận: momentum ngoại hối
đã **suy giảm đơn điệu suốt bốn thập kỷ** — Sharpe gộp +1,05 giai đoạn 1971–1985, +0,50
giai đoạn 1986–2000, −0,08 giai đoạn 2001–2009, **−0,16 giai đoạn 2010–2025**. Trên đúng
khoảng thời gian bộ dữ liệu này phủ, momentum âm **trước cả khi trừ chi phí**. Chi phí
giao dịch không phải nguyên nhân — đã kiểm chứng bằng phân tích độ nhạy: cho hoa hồng
chạy từ 0,00 đến 0,70 pip, Sharpe chỉ dịch 0,015–0,076. Tái lập bằng
`src/momentum_decay.py` và `src/cost_sensitivity.py`.

Dự báo được **biến động**, nhưng không cần mô hình cầu kỳ: MA20-GK và
GARCH(1,1)-t ngang nhau (mỗi mô hình tốt nhất ở 3/6 cặp, Diebold–Mariano 0/6),
cùng thắng mọi biến thể HAR ở p<0,01 — kể cả HAR được cho ăn chính chuỗi
realized variance 5 phút đầy đủ.

**Học tăng cường hội tụ về đúng quy tắc giải tích**: PPO được tự do chọn hệ số
k ∈ [0,50; 1,50] nhân vào trần rủi ro, trên 6 cặp × 6 seed độc lập, hội tụ về
1,01–1,11 (trung vị 1,08).

## Chưa làm

Tầng fuzzy; phiếu quyết định và tầng giải thích (tầng 6 — phần lõi MIS, chưa ai
nhận); giao diện; module báo cáo backtest; văn bản luận văn.
