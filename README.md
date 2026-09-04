# Xây dựng hệ thống hỗ trợ quyết định giao dịch ngoại hối dựa trên dự báo biến động và định cỡ vị thế theo ràng buộc rủi ro

*(tên đề tài; repo này chứa dữ liệu & pipeline)*

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
| `rv_adv.csv` | RV + quarticity + bipower + semivariance, 5 phút | 29.961 dòng |
| `panel2_6pairs.csv` | Panel rủi ro dựng bằng dự báo MỚI (đang dùng) | 21.596 dòng |
| `panel_6pairs.csv` | Panel rủi ro CŨ (MA20-GK), giữ để đối chiếu | 29.843 dòng |
| `cost_table.csv` | Chi phí giao dịch (chế độ × cặp × giờ) | 288 dòng |
| `cost_elasticity.json` | Độ co giãn spread theo biến động | 6 cặp |
| `spread_hourly_all.csv` | Spread thật theo giờ, 8 thời kỳ mẫu | 103.504 dòng |
| `carry.csv` | Chênh lệch lãi suất theo cặp (tháng) | 3.828 dòng |
| `slippage.csv` | Trượt giá qua mức dừng lỗ, đo từ M1 | 60.617 dòng |
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
| `position_sizing.py` | **Tầng 4 dùng cái này** — quy tắc được chọn sau khi so 9 phương pháp |
| `sizing.py`, `sizing2.py` | Quy tắc cơ sở và harness mô phỏng |
| `compare_sizing.py` | So sánh 9 phương pháp trên biên hiệu quả |
| `compare_rl.py` | PPO so với CVaR-PPO, kèm chẩn đoán điều kiện hóa |
| `split.py` | **Chia huấn luyện/kiểm định/kiểm tra** — có tự kiểm |
| `run_final_eval.py`, `run_final_eval2.py` | Chấm điểm cuối theo quy trình sạch |
| `run_scores.py` | **Bộ chỉ số đầy đủ**: CRPS, pinball, log score, PIT+KS, Kupiec, Christoffersen, DQ, FZ0, Mincer–Zarnowitz |
| `metrics.py` | Cài đặt các chỉ số trên — có tự kiểm |
| `volfc.py` | **Tầng 2 dùng cái này** — tổ hợp STHARQ+HARQ+SHAR — có tự kiểm |
| `build_panel2.py` | Dựng lại panel rủi ro bằng dự báo mới |
| `run_volbake.py`, `run_volstats.py` | So 14 mô hình biến động; DM + MCS |
| `carry_test.py` | Kiểm định carry — có tự kiểm ngưỡng đặt trước |
| `huyh_patterns.py` | Kiểm chứng lại 3 mẫu ký hiệu của HuyH trên dữ liệu này |
| `run_symbolic.py` | Thử đưa đặc trưng ký hiệu vào tầng 2 (kết luận: không dùng) |
| `slippage_model.py` | Trượt giá qua stop, đo từ 60.617 lần chạm — có tự kiểm |
| `decision_record.py` | **Tầng 6** — phiếu quyết định + khoảng conformal phân tầng — có tự kiểm |
| `rl_env.py`, `rl_agent.py` | Môi trường và tác tử học tăng cường |

| `run_guard.py` | Chạy walk-forward 6 cặp, chấm điểm trên cả hai mục tiêu |
| `momentum_decay.py` | Suy giảm momentum qua 55 năm — có tự kiểm |
| `cost_sensitivity.py` | Độ nhạy hoa hồng, chứng minh không load-bearing — có tự kiểm |

## Thu thập lại dữ liệu thô — `collect/`

Chạy trên máy có mạng ra ngoài (không chạy được trong sandbox nghiên cứu):

```bash
python collect/histdata_dl.py                 # nến M1, 6 cặp × 16 năm  (~10 phút)
python collect/prep_fx.py                     # M1 → H1 → D1, hiệu chuẩn múi giờ
python collect/rv5.py --pair EURUSD           # realized variance, từng cặp (~19s/cặp)
python collect/rv_advanced.py --pair EURUSD   # quarticity/bipower/semivariance (~15s/cặp)
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

Dự báo được **biến động**, và mô hình cầu kỳ CÓ đáng. Con số dưới đây đo trên
**đoạn kiểm tra sạch** (2023-11-20 → 2025-12-31) dưới quy trình 70/15/15, sau
khi chọn cấu hình trên đoạn kiểm định — xem `docs/KETQUA_VONG7.md`:

| | QLIKE trên kiểm tra |
|---|---|
| **HAR vòng 7** (thêm lịch NHTW riêng từng cặp) | **0,1585** |
| HAR gốc (tổ hợp STHARQ+HARQ+SHAR) | 0,1726 |
| MA20-GK (nền cũ) | 0,2172 |

Cải thiện **27,0%** so với nền cũ và **8,2%** so với bản HAR trước đó.
Diebold–Mariano thắng **6/6 cặp** so với MA20-GK và **4/6 cặp** so với HAR gốc
(p<0,05). Model Confidence Set ở α=0,10 chỉ còn lại **một mình HAR vòng 7**.
Hiệu chuẩn phân phối sạch 6/6 cặp trên cả bốn kiểm định (Kupiec, Christoffersen,
DQ, PIT-KS). Cây tăng cường (GBM) và các mô hình học máy thua mọi biến thể HAR.

**Cải tiến đến từ đâu.** Đã backtest 1.024 cấu hình trên đoạn kiểm định; đúng
**một** trục có tác dụng: mỗi cặp chịu ngân hàng trung ương *của riêng đồng tiền
đó* cộng FOMC, và đưa lịch họp thật của cả bảy ngân hàng (901 ngày, 2010–2026,
`data/cb_dates.csv`) vào mô hình. Dùng chung một lịch ECB+FOMC cho cả 6 cặp chỉ
ăn 3%; dùng đúng lịch của từng cặp ăn 8%. Bốn cải tiến khác có cơ sở tài liệu —
khử chu kỳ nội tuần, co ngót hệ số về panel, biến RV chéo cặp, hiệu chuẩn
Mincer–Zarnowitz — đều cho **kết quả âm**; chi tiết và lý do ở `docs/KETQUA_VONG7.md`.

**Toàn bộ giá trị nằm ở chế độ căng.** Chấm điểm phân tầng theo ngũ phân vị biến
động dự báo: MA20-GK có QLIKE Q5/Q1 = **1,82**, HAR vòng 7 là **1,01**. Khoảng
cách giữa hai mô hình là +0,0122 ở chế độ êm nhất nhưng **+0,1609 ở chế độ căng
nhất** — rộng gấp 13 lần. Trung bình gộp giấu kín đúng chỗ này.

*Lịch sử con số này: 24% đo trên tập vừa dùng để chọn vừa dùng để báo cáo; dưới
quy trình huấn luyện/kiểm định/kiểm tra 60/20/20 nó là 19,7%; nay dưới 70/15/15
với lịch NHTW là 27,0%. Chênh 24% → 19,7% chính là phần lạc quan do rò rỉ lựa
chọn — đã đo được thay vì ẩn.*

*Kết luận cũ ở đây — "MA20-GK và GARCH-t cùng thắng mọi biến thể HAR ở
p<0,01" — là **sai**, và sai vì một lỗi xử lý dữ liệu: phiên Chủ nhật của FX
chỉ dài 2 giờ, phương sai nhỏ hơn 24 lần, chiếm 17% số hàng, và nằm nguyên
trong chuỗi hồi quy. Gộp nó vào ngày giao dịch kế tiếp làm QLIKE của HAR đi
từ 0,4616 xuống 0,1648. Chi tiết và cách phát hiện: `docs/TANG2_BIENDONG.md`.*

**Nhưng cải thiện nhỏ dần khi đi xuống dưới.** Tương quan dự báo/thực tế lên
0,52–0,72 (trước là 0,435–0,647), lỗ đuôi 1% của vị thế giảm 7,47% → 7,05%,
còn điểm khoảng của tầng 6 chỉ tốt hơn 1,4%. Lý do: tầng 4 và 6 tiêu thụ phân
vị đuôi của lợi suất đã chuẩn hóa, thứ bị chi phối bởi độ dày đuôi chứ không
phải bởi mức phương sai. Dự báo biến động tốt hơn **không** tự động thành hệ
thống quyết định tốt hơn.

**Nhánh khai phá mẫu của HuyH ra cùng kết luận bằng phương pháp khác.** Chuỗi
ký hiệu + mẫu tuần tự trên FRED daily, phễu lọc bốn bước (4.722 → 11 → 7 → 3):
ba mẫu sống sót qua kiểm tra ngoài thời gian 2022–2026 **đều là mẫu biến động**,
**không** mẫu hướng đi nào sống sót. Tôi kiểm chứng lại cả ba trên realized
variance 5 phút: **3/3 tái lập**, lift còn cao hơn (1,69 · 1,32 · 1,69). Nhưng
thêm đặc trưng ký hiệu vào STHARQ chỉ cải thiện QLIKE 0,15–0,49% và thắng 0/6
cặp — mô hình liên tục đã bắt hết thông tin đó. Dùng làm **lời giải thích** trên
phiếu quyết định, không dùng để dự báo. `docs/TICHHOP_HUYH.md`.

**Carry cũng không phải tín hiệu hướng đi.** Sharpe sau chi phí trên đúng
khoảng hệ thống vận hành (2010–2025) là **−0,05**; mẫu cân bằng 2002–2025 cho
+0,09. Trước đó nó có thật: 1994–2000 là +1,05. Độ lệch −1,56, đúng đặc trưng
"carry crash". Ngưỡng đặt trước là Sharpe > 0,30 nên carry **không** được đưa
vào tầng quyết định. `src/carry_test.py`.

**Trượt giá qua mức dừng lỗ giờ là số đo, không còn là giả định.** 60.617 lần
chạm mức dừng lỗ đo từ nến M1: trượt p95 bằng **35% khoảng cách dừng lỗ**. Đưa
phân phối thật vào mô phỏng làm xác suất phá sản **tăng 2,5 lần**; hệ số cắt
0,92 đưa nó về mức cũ. Giờ trượt tệ nhất là 12–13h UTC, *không* trùng giờ
spread đắt nhất (21h UTC). `src/slippage_model.py`.

**Phiếu chỉ hiệu chuẩn cho một phiên — đã sửa.** Stop 2σ: đọc "5%" rồi giữ mười
phiên thì thực tế là **50%**. Công thức phản xạ kéo dài được (lệch 1,3–1,7% ở
h=5/10/20, lệch theo hướng lạc quan) nên chỉ cần in ra, và phiếu giờ có in.
Quy tắc √h đúng trung bình (1,006–1,011) nhưng lệch tới ±14% theo chế độ biến
động; hệ số hiệu chỉnh ước lượng trên tập huấn luyện giảm biên độ đó một nửa.
`docs/TANG6_TAMHAN.md`.

**Biến động phụ thuộc đường đi (Guyon–Lekeufack) đã thử và không dùng.** Cải
thiện QLIKE 0,3%, nhất quán 5/6 cặp nhưng không cặp nào đạt p<0,05 — nhỏ hơn 80
lần so với lần đổi mô hình trước. Đổi mô hình sản xuất vì một cải thiện không có
ý nghĩa thống kê là đúng kiểu điều chỉnh mà `KHOA_SO.md` ngăn.

**Trần rủi ro có hai lỗ hổng phạm vi, cả hai đã vá.** (a) Nó tính cho MỘT vị
thế: mở 6 lệnh cùng hướng USD ở đúng cỡ khuyến nghị cho phá sản **73,6%** trong
khi phiếu ghi 1%. Sáu cặp nhìn như tự phòng hộ (tương quan −0,09) nhưng đó là
ảo giác của quy ước yết giá — quy về cùng chiều USD thì tương quan là **+0,44**.
Luật `k_danh_mục = 1/√(k + k(k−1)ρ)` đưa mọi cấu hình về dưới 1%. (b) Bảo đảm
"250 phiên" chỉ đúng nếu định cỡ lại **mỗi phiên**: đặt một lần rồi giữ cho
1,95%, mỗi tháng cho 1,15%, mỗi phiên cho 0,41%. `docs/TANG4_DANHMUC.md`.

**Học tăng cường không tìm ra điều kiện hóa theo trạng thái.** PPO và CVaR-PPO
đều có sụt giảm trong vector trạng thái nhưng học ra hệ số gần như hằng số —
biên độ 0,018 và 0,030, so với 0,800 của một quy tắc thiết kế tay. Chạy lại trên
panel mới cho cùng kết luận (biên độ 0,045 và 0,026), và PPO còn học **sai
hướng**: hệ số TĂNG khi sụt giảm sâu hơn (1,013 ở đỉnh → 1,057 khi lỗ 30%). Huấn luyện
lâu hơn làm biên độ **nhỏ đi**. Quy tắc tay cho phá sản thấp hơn 26 lần ở cùng
tăng trưởng. Fuzzy Mamdani không hơn một tích hai hệ số tuyến tính (+0,08%).
Chi tiết: `docs/SIZING_COMPARISON.md`.

**Khoảng dự báo của tầng 6 phải là conformal phân tầng, không phải Student-t.**
Trên tập giữ riêng, conformal lệch hiệu chuẩn trung bình 0,37% so với 1,25% của
Student-t, và còn hẹp hơn ở mức 99% (145,6 so với 158,0 pip). Bảo đảm của
conformal chỉ là biên, nên phải phân tầng theo chế độ biến động: lệch tối đa
theo chế độ giảm từ 1,9% xuống 0,8%. Xác suất chạm stop theo nguyên lý phản xạ
lệch trung bình 1,44% (0,10% ở stop 3σ). Cả ba phương pháp đều phủ thiếu ~1%
khi tài khoản đang lỗ — giới hạn đã đo, in thẳng trên phiếu.
Bản đang dùng là **conformal thích ứng theo tầng** (ACI của Gibbs–Candès 2021
ghép với phân tầng Mondrian): lệch tối đa theo chế độ 1,2% so với 2,4–3,2% của
năm phương án còn lại, và điểm khoảng cũng tốt nhất.
**Chấm bằng quy tắc chấm điểm chính đáng thì cách dựng đuôi gần như không quan
trọng — chất lượng σ̂ mới quan trọng.** CRPS trên đoạn kiểm tra: Gauss 26,28 ·
Student-t 26,22 · Mondrian 2 26,23 pip; nhưng dùng σ̂ **cũ** thì 26,33 và điểm
khoảng xấu hơn 1,5%. Ngược lại, PIT + Kolmogorov–Smirnov **bác bỏ giả định
chuẩn ở p = 0,0001** trong khi chính nó vượt qua mọi backtest VaR — bằng chứng
rằng độ phủ thôi thì chưa đủ. Chi tiết: `docs/CHISO_DANHGIA.md`,
`docs/TANG6_HIEU_CHUAN.md`, `docs/DANHGIA_CUOI.md`.

## Chưa làm

Tầng fuzzy; phiếu quyết định và tầng giải thích (tầng 6 — phần lõi MIS, chưa ai
nhận); giao diện; module báo cáo backtest; văn bản luận văn.
