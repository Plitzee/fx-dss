# ML / DL / học trực tuyến cho ba ô — khảo sát 2026 và kết quả huấn luyện

*04/09/2026. Tái lập: `python src/run_ml3.py`.
Kết quả: `output/ml3.json`, log thô `output/log_ml3.txt`.*

Hai phần: (1) hiện trạng lĩnh vực, (2) **số đo thật** từ huấn luyện trên dữ liệu
của repo. Phần 2 mới là phần quyết định.

---

## Phần 1 — hiện trạng lĩnh vực, 2026

### 1.1 Cái gì đang lên

| Hướng | Đại diện | Ý tưởng cốt lõi |
|---|---|---|
| **Mô hình nền thời gian** | Chronos-2 (Amazon, nền T5), MOIRAI-2 (Salesforce, decoder-only, *Any-Variate Attention*) | huấn luyện trước trên nhiều miền, dùng lại không cần khớp riêng |
| **Không gian trạng thái** | TSMamba, FLDmamba | độ phức tạp **tuyến tính** theo độ dài chuỗi thay vì bình phương như Transformer |
| **Lai** | SST (Mamba + Transformer nhiều thang) | Mamba bắt cấu trúc dài, Transformer bắt động lực ngắn |
| **Hồi quy nhớ có cổng** | xLSTM, VSN+LSTM | thiên kiến quy nạp về khử nhiễu và bộ nhớ thích ứng |
| **Học trực tuyến dưới trôi khái niệm** | OneNet, SAOCP, AdaWeather | tổ hợp mô hình với **trọng số cập nhật mỗi bước** theo sai số vừa mắc |

### 1.2 Cái gì đang xuống

**Transformer thuần cho tài chính.** Benchmark lớn nhất hiện có —
[arXiv:2603.01820](https://arxiv.org/pdf/2603.01820), 15+ kiến trúc, futures đa
lớp tài sản gồm FX, 2010–2025 — kết luận thẳng: triết lý *"càng lớn càng tốt"*
của Transformer **thất bại ở thị trường tài chính**. Dẫn đầu là **VSN+LSTM** và
**xLSTM** (Sharpe > 2,30).

**Nhưng phải đọc kỹ bài đó, vì nó mâu thuẫn với kết quả của repo này.** Bài kết
luận mô hình chuỗi **phi tuyến vượt rõ** các nền tuyến tính. Còn đo trên dữ liệu
của ta thì HAR tuyến tính ngang ngửa mọi thứ. Hai điều này **không đối lập** —
chúng đo hai thứ khác nhau:

| | arXiv 2603.01820 | repo này |
|---|---|---|
| Đích | tín hiệu **định cỡ vị thế**, chấm bằng Sharpe | **QLIKE của phương sai**, rồi **log-loss của ba lớp** |
| Tài sản | futures đa lớp (hàng hoá, chỉ số, trái phiếu, FX) | **chỉ FX giao ngay** |
| Cái được thưởng | bắt được xu hướng chéo tài sản | bắt được biến động trong một lớp tài sản |

FX giao ngay là lớp tài sản **ít xu hướng nhất** trong rổ đó. Nên "phi tuyến
thắng" ở bài kia rất có thể đến từ hàng hoá và trái phiếu, chứ không phải FX.
Đây là điểm phải ghi trong phần thảo luận của luận văn, không được lờ đi.

### 1.3 "RL" — phải tách hai nghĩa

`docs/SIZING_COMPARISON.md` đã loại **PPO** — nhưng đó là RL học **chính sách
định cỡ vị thế**. Cái khác hẳn là **học trực tuyến có phản hồi**: hôm nay dự báo
sai thì mai điều chỉnh. Họ thuật toán đúng cho việc đó là **trọng số mũ**
(Hedge / exponential weights):

```
mỗi phiên:  dự báo = Σ wᵢ · Pᵢ        rồi mới nhìn kết cục
            wᵢ ← wᵢ · exp(−η · tổn thất log của chuyên gia i)
```

Có **chặn hối tiếc** `O(√(T log N))` so với chuyên gia tốt nhất nhìn lại, không
cần mô phỏng môi trường, và rẻ. Văn liệu tổng quan RL tài chính 2025–2026 đều
ghi cùng một vấn đề với RL sâu: **overfit và không tổng quát hoá được khi chế độ
trôi** — đúng cái Hedge tránh được bằng thiết kế.

---

## Phần 2 — kết quả huấn luyện trên dữ liệu của repo

### 2.1 Một rò rỉ đã mắc và đã sửa — ghi lại để không lặp

Lần chạy đầu cho **BSS +0,35 và AUC 0,967** trên mục tiêu R. Con số không thể
thật: giai đoạn 1 đã đo rằng **không nền nào thắng nổi khí hậu học** trên mục
tiêu đó.

Nguyên nhân: `ml_data.xay()` đặt `y[t] = log rv5[t+1]` — hàng `t` là đặc trưng
**để dự báo ngày t+1**. Còn `balop.dung_muc_tieu()` trả về lớp **của chính ngày
t**. Ghép thẳng hai cái là rò rỉ trực tiếp: `X[t]` chứa `lrv_d` = phương sai
thực hiện *của ngày t*, cộng `lrsp`/`lrsn` — bán phương sai dương/âm của ngày
đó, tức **cho thẳng dấu của phiên**. Mô hình không dự báo gì, nó đọc đáp án.

Sửa: dịch đích lên một phiên, và thêm chốt chặn ngay trong `nap()`:

```python
assert (phu.ngay_dich[m].values > phu.Date[m].values).all(), \
    "dich khong nam sau dac trung — con ro ri"
```

Cộng dòng in ra mỗi lần chạy: `căn chỉnh: đặc trưng ngày X → đích ngày Y`.

### 2.2 Thiết lập

21.900 hàng × 62 đặc trưng, 6 cặp gộp, cùng tập thông tin với HAR **cộng thêm**
22 độ trễ thô, thứ trong tuần, mã cặp — cho ML lợi thế có chủ đích. Khớp trên
huấn luyện, chọn trên kiểm định, **chấm kiểm tra đúng một lần**.

### 2.3 Mục tiêu R — kỹ năng vượt trên tầng 2

| mô hình | log | BSS | KTC 95% | AUC hướng | KTC 95% |
|---|---|---|---|---|---|
| khí hậu học | 1,0983 | 0 | — | 0,500 | — |
| chỉ σ̂ | 1,0990 | −0,0006 | [−0,0013; −0,0001] | 0,500 | — |
| σ̂ + chế độ | 1,0978 | +0,0005 | [−0,0008; +0,0022] | 0,498 | [0,480; 0,516] |
| logistic đa thức | 1,0977 | +0,0003 | [−0,0033; +0,0041] | **0,531** | **[0,511; 0,558]** |
| LightGBM 3 lớp | 1,1171 | **−0,0180** | [−0,0238; −0,0111] | 0,513 | [0,495; 0,536] |
| GRU 3 lớp | 1,1873 | **−0,0771** | [−0,0912; −0,0637] | 0,524 | [0,503; 0,548] |
| **học trực tuyến (Hedge)** | **1,0966** | **+0,0016** | [−0,0003; +0,0038] | 0,528 | [0,505; 0,553] |

### 2.4 Mục tiêu P — thứ hiển thị trên giao diện

| mô hình | log | BSS | KTC 95% | AUC hướng |
|---|---|---|---|---|
| khí hậu học | 1,0535 | 0 | — | 0,500 |
| chỉ σ̂ | 1,0450 | +0,0063 | [+0,0038; +0,0088] ✓ | 0,500 |
| σ̂ + chế độ | 1,0438 | +0,0074 | [+0,0046; +0,0106] ✓ | 0,501 |
| logistic đa thức | 1,0445 | +0,0072 | [+0,0038; +0,0121] ✓ | 0,527 |
| LightGBM 3 lớp | 1,0670 | **−0,0148** | [−0,0225; −0,0077] | 0,502 |
| GRU 3 lớp | 1,1185 | **−0,0594** | [−0,0724; −0,0482] | 0,515 |
| **học trực tuyến (Hedge)** | **1,0435** | **+0,0076** | [+0,0047; +0,0112] ✓ | 0,517 |

### 2.5 Bốn kết luận

**1. Phi tuyến THUA, và thua có ý nghĩa.** LightGBM và GRU âm rõ trên **cả hai**
mục tiêu, khoảng tin cậy không chạm 0. Đây không phải "ngang ngửa" như ở bảng
QLIKE — đây là **kém hơn hẳn**. Cùng tập thông tin, còn được cho thêm 22 độ trễ
thô. Chúng overfit.

**2. Học trực tuyến là mô hình tốt nhất trên cả hai mục tiêu** — nhưng lợi thế
so với `σ̂ + chế độ` rất mỏng: +0,0076 so +0,0074 ở mục tiêu P, khoảng tin cậy
chồng gần hết. Chưa đủ để thay nền.

**3. Nhưng Hedge làm được đúng việc bạn muốn.** Trọng số cuối trên đoạn kiểm tra:

```
mục tiêu P:  chỉ σ̂ 0,45 · σ̂+chế độ 0,18 · logistic 0,37 · LightGBM 0,00 · GRU 0,00
mục tiêu R:  chỉ σ̂ 0,20 · σ̂+chế độ 0,42 · logistic 0,38 · LightGBM 0,00 · GRU 0,00
```

Nó **tự tìm ra hai mô hình overfit và ép trọng số về 0**, không ai bảo. Đó chính
xác là "hôm nay sai thì mai bớt tin". Giá trị của nó nằm ở **tính bền**, không
phải ở điểm số.

**4. Lần đầu có dấu hiệu hướng đi — nhưng đừng mừng vội.** Logistic đa thức cho
AUC 0,531 [0,511; 0,558] ở mục tiêu R và 0,527 [0,508; 0,552] ở mục tiêu P —
**khoảng tin cậy không phủ 0,50**. Sau bốn kết luận âm liên tiếp, đây là lần đầu.

Nhưng: chạy **14 phép thử AUC** (7 mô hình × 2 mục tiêu), ở mức 5% thì kỳ vọng
0,7 lần dương giả. Hai lần dương, biên rất mỏng (0,511 so 0,500). Bonferroni cho
14 phép thử đòi mức 0,0036 — **gần như chắc chắn không sống sót**. Nên đây là
**ứng viên cho giai đoạn 2**, không phải kết luận. Muốn tuyên bố thì phải chạy
lại như một giả thuyết đặt trước, một phép thử, trên tập khoá sổ.

---

## Phần 3 — tích hợp vào giai đoạn nào

| Hướng | Giai đoạn | Vì sao |
|---|---|---|
| **Hedge làm tầng tổ hợp** | **3** (hiệu chuẩn) | Đã đo là tốt nhất và tự loại mô hình hỏng. Nối thẳng vào **sổ dự báo** vừa dựng: sổ ghi tổn thất mỗi phiên → Hedge cập nhật trọng số. Rẻ, có chặn hối tiếc, giải thích được. **Đây là việc nên làm trước.** |
| **Logistic làm ứng viên hướng** | **2** | AUC 0,53 chưa sống sót hiệu chỉnh bội. Đưa vào phễu Westfall–Young như một giả thuyết đặt trước. |
| **LightGBM + SHAP làm trần** | **2**, họ H4 | Đúng như spec: báo cáo làm **trần trên**, không xuất xưởng. Nay biết trần đó **thấp hơn nền** — bản thân điều đó là kết quả. |
| **xLSTM / VSN+LSTM** | **2**, tuỳ chọn | Bài 2603.01820 nói chúng thắng. Nhưng GRU đã thua rõ ở đây, mà xLSTM cùng họ. Thử một lần, đừng đầu tư nhiều. |
| **Mô hình nền (Chronos-2, MOIRAI-2)** | **hoãn** | `TAI_LIEU_LIEN_QUAN.md` đã ghi: foundation model **thua HAR** ở dự báo biến động. Chi phí cao, bằng chứng ngược. |
| **RL sâu (PPO…)** | **không** | Đã loại có đo. Văn liệu 2025–2026 xác nhận cùng lý do: overfit khi chế độ trôi. |

---

## Phần 4 — nói gì với hội đồng

> *"Đã huấn luyện logistic đa thức, LightGBM và GRU trực tiếp trên bài toán ba
> lớp, cùng tập thông tin với mô hình tuyến tính và còn được cho thêm 22 độ trễ
> thô. Hai mô hình phi tuyến **kém hơn có ý nghĩa thống kê** trên cả hai định
> nghĩa mục tiêu. Mô hình tốt nhất là **tổ hợp học trực tuyến** — nó tự đưa
> trọng số của hai mô hình phi tuyến về 0. Đây là kết quả đo được trên đoạn kiểm
> tra chấm một lần, không phải phán đoán."*

Và phần trung thực nhất: **lần chạy đầu cho AUC 0,967 vì rò rỉ căn chỉnh một
phiên.** Nó được phát hiện vì con số mâu thuẫn với nền đã đo ở giai đoạn 1, chứ
không phải vì ai đọc lại mã. Đó là lý do phải có nền trước khi có mô hình.

## Sources

- [Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance (arXiv 2603.01820)](https://arxiv.org/pdf/2603.01820)
- [A Mamba Foundation Model for Time Series Forecasting (arXiv 2411.02941)](https://arxiv.org/abs/2411.02941)
- [SST: Multi-Scale Hybrid Mamba-Transformer Experts (CIKM 2025)](https://dl.acm.org/doi/10.1145/3746252.3761394)
- [The 2026 Time Series Toolkit: 5 Foundation Models](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/)
- [OneNet: Enhancing Time Series Forecasting under Concept Drift by Online Ensembling](https://openreview.net/forum?id=Q25wMXsaeZ)
- [Improved Online Conformal Prediction via Strongly Adaptive Online Learning (SAOCP)](https://arxiv.org/pdf/2402.01139)
- [Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting](https://arxiv.org/html/2412.08435v5)
- [The Evolution of Reinforcement Learning in Quantitative Finance: A Survey (ACM CSUR)](https://dl.acm.org/doi/full/10.1145/3733714)
- [A Review of Reinforcement Learning in Financial Applications (arXiv 2411.12746)](https://arxiv.org/pdf/2411.12746)

---

## Phần 5 — RL đúng nghĩa: cổng vào lệnh (đã train, 04/09/2026)

*Tái lập: `python src/run_rl_gate.py`. Log: `output/log_rl_gate.txt`.*

Ở phần 3 tôi viết "RL cổng vào lệnh **sẽ** thua" mà chưa chạy. Đó là tiên
nghiệm, không phải kết quả. Nay đã chạy.

### 5.1 Vì sao đây vẫn là RL chứ không phải học có giám sát

Hành động **đổi trạng thái tương lai**: đứng ngoài hôm nay → không lỗ → sụt
giảm nhỏ hơn → trần rủi ro ngày mai cho đòn bẩy cao hơn. Vòng phản hồi đó là
thứ học có giám sát không bắt được. Không gian hành động 2 giá trị, trạng thái
rời rạc 72 ô (tercile σ̂ × 3 mức sụt giảm × dấu carry × dấu P&L 5 phiên × cờ
họp NHTW). Chi phí spread thật, quy đúng về đơn vị lợi suất.

### 5.2 Kết quả

**Kiểm định** (dùng để chọn):

| chính sách | TB (bp/ngày) | Sharpe | sụt giảm | % vào lệnh |
|---|---|---|---|---|
| luôn vào | 0,577 | 0,36 | 6,0% | 100% |
| **quy tắc tay (bỏ tercile σ̂ cao)** | **0,693** | **0,60** | **3,0%** | 52,8% |
| bandit ngữ cảnh (γ=0) | 0,164 | 0,25 | 2,5% | 34,8% |
| Q-learning (γ=0,95) | 0,508 | 0,36 | 5,7% | 92,5% |
| Q bi quan (γ=0,95) | 0,577 | 0,36 | 6,0% | 100% |

**Kiểm tra** (chấm một lần):

| chính sách | TB (bp/ngày) | Sharpe | sụt giảm | % vào lệnh | t so luôn vào |
|---|---|---|---|---|---|
| luôn vào | −0,821 | −0,60 | 7,0% | 100% | — |
| không bao giờ vào | 0,000 | 0,00 | 0,0% | 0% | +0,74 (p=0,460) |
| quy tắc tay | −0,585 | −0,46 | 5,7% | 79,6% | +0,74 (p=0,461) |
| bandit ngữ cảnh | **−0,200** | −0,31 | 3,2% | 33,4% | +0,75 (p=0,455) |
| Q-learning | −0,596 | −0,51 | 5,3% | 84,3% | +0,63 (p=0,529) |
| Q bi quan | −0,821 | −0,60 | 7,0% | 100% | +0,00 (p=1,000) |

### 5.3 Năm điều rút ra

**1. Không một hiệu số nào có ý nghĩa thống kê.** Mọi p > 0,45. Kết luận đúng
là **"không phân biệt được"**, không phải "RL thua".

**2. Chiến lược nền vốn đã lỗ trên kiểm tra** (−0,821 bp/ngày). Đúng như repo
đã biết: carry Sharpe −0,05 từ 2010, dưới ngưỡng 0,30 đặt trước. Nên thí nghiệm
này thực chất đo *"RL có học được cách ngừng giao dịch một chiến lược đang lỗ
không"* — chứ không phải *"RL có tạo ra lợi nhuận không"*. Cổng chỉ giảm lỗ
được, không sinh lãi được.

**3. Bandit lỗ ít nhất trong nhóm có giao dịch** (−0,200 so −0,821), bằng cách
hạ tỷ lệ tham gia xuống 33,4%. Lại là **phương pháp ít năng lực nhất** dẫn đầu
trong nhóm học được — lần thứ hai, sau bảng định cỡ vị thế.

**4. Q bi quan sập về "luôn vào".** Khi trừ khoản phạt tỷ lệ `1/√n(s,a)`, không
ô trạng thái nào đủ bằng chứng để lệch khỏi mặc định. Đó là biến thể bảo thủ
làm **đúng việc của nó**: từ chối hành động theo bằng chứng mỏng. Và nó nói
thẳng một điều — với 15.018 bước và 72 ô, dữ liệu này **không đủ** để học một
chính sách khác mặc định.

**5. Chính sách học được khác quy tắc tay ở 23/47 ô có đủ bằng chứng** — tức nó
không chỉ tìm lại quy luật thủ công, nó làm khác. Và cái khác đó không tốt hơn.

### 5.4 Cái này giải quyết gì, không giải quyết gì

**Đã giải quyết:** RL dạng bảng cho cổng vào lệnh, trên không gian trạng thái
này, trên chiến lược nền này → **không cải thiện có ý nghĩa** so với một quy tắc
tay một dòng. Đây là dạng RL **thứ tư** được thử và là lần thứ tư không thắng
được quy tắc thủ công.

**Chưa giải quyết:** RL trên một chiến lược nền **có lợi thế thật** (hiện chưa
có); trạng thái giàu hơn; RL sâu (đã thử ở bảng định cỡ, thua nặng hơn).

Mẫu hình lặp lại đủ nhiều lần để đáng ghi vào luận văn: **trên dữ liệu này, mỗi
lần tăng năng lực mô hình lại làm kết quả xấu đi.** PPO < bandit < quy tắc tay ở
định cỡ; GRU < LightGBM < logistic < nền ở ba lớp; Q-learning < bandit ở cổng
vào lệnh. Ba nhánh độc lập, cùng một chiều.
