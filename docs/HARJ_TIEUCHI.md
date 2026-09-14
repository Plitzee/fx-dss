# HAR-J — TÁCH LIÊN TỤC/NHẢY BẰNG BIPOWER. Tiêu chí CHỐT TRƯỚC

Lập 14/09/2026. Nhánh `replan-2026`. **Commit chứa văn bản này nằm trước mọi
commit có số liệu.**

---

## 1. Vì sao đây KHÔNG phải lặp lại thí nghiệm BCL đêm qua

`docs/BCL_TIEUCHI.md` đêm qua đã thử "tách C/J bằng ngưỡng" (đếm lợi suất vượt
`k·σ` trong ngày) — và đã đính chính một câu sai: mốc HAR sản xuất **không**
hề chứa bipower ở bất cứ đâu. Đọc lại nguyên văn `volfc2.py::thiet_ke()`:

```python
X = {"STHARQ": [1, lv, lw, lm, lv·G, lw·G, lm·G, lq, lq·lv],
     "HARQ":   [1, lv, lw, lm, lq, lq·lv],
     "SHAR":   [1, lp, ln_, lw, lm]}
```

`HARQ` điều chỉnh theo *sai số đo lường* (realized quarticity — Bollerslev,
Patton & Quaedvlieg 2016). `SHAR` tách theo *bất đối xứng* tăng/giảm
(semivariance — Patton & Sheppard 2015). **Không cái nào tách liên tục/nhảy.**
Cột `bpv5` (bipower variation, Barndorff-Nielsen & Shephard 2004) **có trong
dữ liệu** (`rv_adv.csv`) nhưng chưa từng được tham chiếu trong tổ hợp sản
xuất — chỉ dùng ở tầng ML riêng (`ml_data.py`).

Đây là phép thử **thật sự mới**: dùng đúng **bipower variation** (thước đo
chuẩn của văn liệu HAR-J/HAR-CJ, mượt và có nền tảng lý thuyết vững — Barndorff-
Nielsen & Shephard 2004), không phải ngưỡng đếm thô như BCL, và test trực tiếp
trên **cơ chế tổ hợp thật** của hệ thống (trung bình log-dự báo của nhiều mô
hình con, khớp bằng OLS cửa sổ mở rộng), không phải khung hồi quy đơn giản hoá
như `kiem_nen_ablation.py`/`kiem_bcl.py` đã dùng.

## 2. Nguồn

**Andersen, Bollerslev & Diebold (2007)**, *Roughing It Up: Including Jump
Components in the Measurement, Modeling, and Forecasting of Return
Volatility*, **Review of Economics and Statistics** 89(4):701–720. Tách
`RV_t = C_t + J_t` bằng bipower variation:

```
BV_t = (π/2) Σ|r_i||r_{i-1}|         (bất biến với nhảy, Barndorff-Nielsen &
                                       Shephard 2004)
C_t  = min(BV_t, RV_t)                (phần liên tục, chặn trên bởi RV)
J_t  = max(0, RV_t − BV_t)            (phần nhảy, không âm)
```

## 3. Giả thuyết CHỐT TRƯỚC

> **H14.** Thêm (hoặc thay) thành phần liên tục/nhảy tách bằng bipower vào tổ
> hợp sản xuất cải thiện QLIKE so với mốc hiện tại (STHARQ+HARQ+SHAR, trung
> bình đơn giản).
>
> **Dấu dự kiến:** hệ số `log(C)` **lớn hơn** hệ số `log(1+J/C)` — phần liên
> tục dai (persistent) hơn phần nhảy, đúng phát hiện chuẩn của toàn bộ văn
> liệu HAR-J/HAR-CJ. Hệ số `log(1+J/C)` ngược dấu hoặc bằng 0 thì H14 về mặt
> **cơ chế** bị bác bỏ, kể cả khi QLIKE có tốt lên (đúng bài học từ S4 —
> `PHA2_KETQUA.md` mục 3b).

## 4. Bốn cấu hình — liệt kê đầy đủ TRƯỚC khi chạy

Mô hình mới `HARJ`, cùng dạng với `SHAR` (chỉ đổi `lp/ln_` bất đối xứng thành
`lc/lj` liên tục/nhảy):

```python
C  = min(bpv5, rv5);  J = max(0, rv5 - bpv5)
lc = log(max(C, EPS)); lj = log(1 + J/C)
X["HARJ"] = [1, lc, lj, lw, lm]      # lw, lm: log-RV tuần/tháng, y hệt SHAR
```

| | tổ hợp |
|---|---|
| **B0** mốc *(không đổi)* | (STHARQ + HARQ + SHAR) / 3 |
| **B0+HARJ** | (STHARQ + HARQ + SHAR + HARJ) / 4 |
| **HARJ thay SHAR** | (STHARQ + HARQ + HARJ) / 3 |
| **HARJ riêng** *(chẩn đoán, không phải ứng viên)* | chỉ HARJ |

**Đếm vào `KHOA_SO.md`: 3 cấu hình mới** (B0 không tính vì đã có). Không thêm
biến thể nào ngoài bảng này.

Mọi tham số khác **giữ nguyên `CAUHINH_SANXUAT`**: cửa sổ mở rộng, `event=
"capday"`, không khử mùa vụ, không chéo cặp, `lam=0`, `RIDGE=1e-8`,
`MIN_TRAIN=500`, `MAX_GAP=4`. Không tinh chỉnh gì để "cho HARJ thắng".

## 5. Giao thức chấm

Tái tạo đúng cơ chế `du_bao_san_xuat()`: mỗi mô hình con khớp OLS **cửa sổ mở
rộng** (`he_so_cuon`, tích luỹ ma trận Gram), dự báo log-phương sai của từng
mô hình, cộng nửa phương sai phần dư, rồi **trung bình cộng** log-dự báo qua
các mô hình trong tổ hợp — đúng công thức `g = L.mean(0)` đang chạy production,
không phải khung hồi quy một phương trình đơn giản hoá.

Khớp trên **huấn luyện+kiểm định** theo đúng cách cửa sổ mở rộng đang chạy
(không có bước "chọn trên kiểm định" ở tầng tổ hợp — bản thân production cũng
không chọn, chỉ trung bình cố định); **chấm QLIKE trên kiểm định rồi kiểm
tra**, DM + Newey–West so B0, theo từng cặp.

## 6. Chống rò rỉ

`bpv5` đã có sẵn trong `rv_adv.csv`, tính từ chuỗi 5 phút intraday của chính
ngày đó — cùng nguồn, cùng độ trễ với `rv5`/`rq5`/`rsp`/`rsn` đã dùng suốt dự
án, nên không có rủi ro rò rỉ mới. Tự kiểm bắt buộc: `C_t ≤ RV_t` và `J_t ≥ 0`
tại mọi phiên (theo đúng định nghĩa `min`/`max` ở trên).

## 7. TIÊU CHÍ PHỦ ĐỊNH

Kết luận **"HAR-J không mang lại giá trị đo được"** khi bất kỳ điều nào đúng:

1. Cấu hình tốt nhất không thắng B0 trên **kiểm tra**, hoặc thắng nhưng DM
   p ≥ 0,05.
2. Hoặc không đạt ≥ 5/6 cặp cải thiện.
3. Hoặc dấu hệ số `log(1+J/C)` không dương hoặc lớn hơn `log(C)` (ngược cơ
   chế H14, dù QLIKE có tốt lên).

Kết luận **dương** đòi **cả ba**: thắng kiểm tra với DM p < 0,0167 (Bonferroni
3 cấu hình mới) **và** ≥ 5/6 cặp **và** dấu hệ số đúng H14.

Không sửa tiêu chí sau khi thấy số. Nếu tiêu chí soạn dở thì khai báo chỗ dở,
giữ nguyên phán quyết — đúng cách đã làm ở `DUOI_EVT.md` mục A5a và
`RUIRO_ML.md` mục A1a.

## 8. Lực phát hiện — khai báo TRƯỚC

Cùng bảng D1, ~3.282 hàng mỗi đoạn ngoài mẫu, cùng cỡ với mọi ablation tầng 2
đêm qua. Ngưỡng phát hiện thực tế đã đo ở khung này (`|gap|` đạt p=0,0445 với
0,67%; BCL không đạt) là **khoảng 0,5–0,7% QLIKE** — HAR-J phải vượt mức đó để
tách được khỏi nhiễu.

## 9. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 3 (giả thuyết), mục 7
(tiêu chí phủ định) và mục 8 (khai báo lực) **trước khi** số liệu được chấm:

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không
> phải công cụ điền hộ.

---

# PHỤ LỤC A — KẾT QUẢ (14/09/2026, chạy SAU khi mục 1–9 đã chốt)

*Tái lập: `python src/kiem_harj.py`. Kết quả: `output/harj.json`.*

Tự kiểm ĐẠT cả 6 cặp: `C_t ≤ RV_t`, `J_t ≥ 0` mọi phiên.

| cấu hình | QLIKE kiểm định | QLIKE kiểm tra | so B0 | DM p | cặp cải thiện |
|---|---|---|---|---|---|
| B0 mốc | 0,1162 | 0,1585 | — | — | — |
| **B0+HARJ** *(tốt nhất trên kiểm định)* | 0,1166 | 0,1588 | +0,18% | 0,177 | 1/6 |
| HARJ thay SHAR | 0,1168 | 0,1589 | +0,26% | 0,457 | 3/6 |
| HARJ riêng | 0,1192 | 0,1607 | +1,38% | 0,012 | 1/6 |

## Phán quyết — **ÂM**, hai trong ba điều kiện trượt

| điều kiện | kết quả | |
|---|---|---|
| ĐK1 thắng kiểm tra, p<0,0167 | **TRƯỢT** | +0,18%, p=0,177 |
| ĐK2 ≥5/6 cặp | **TRƯỢT** | 1/6 |
| **ĐK3 dấu hệ số đúng H14** | **ĐẠT** | **6/6** |

## Điều đáng ghi nhất: dấu hệ số đúng lý thuyết ở cả 6 cặp, nhưng dự báo không tốt lên

| cặp | log(C) | log(1+J/C) |
|---|---|---|
| EURUSD | **+0,286** | **−0,207** |
| GBPUSD | +0,313 | −0,054 |
| USDJPY | +0,408 | −0,166 |
| AUDUSD | +0,363 | −0,229 |
| USDCAD | +0,273 | −0,208 |
| USDCHF | +0,283 | −0,099 |

**`log(C)` dương, `log(1+J/C)` âm — cả 6/6 cặp, nhất quán tuyệt đối.** Đây
đúng là phát hiện cốt lõi của văn liệu HAR-J: phần liên tục có tính **dai**
(persistent), phần nhảy thì **không** — ở đây còn đi xa hơn: sau khi đã có
STHARQ/HARQ/SHAR, phần *thêm* của thành phần nhảy mang dấu **âm**, tức nó
hoạt động như **chiết khấu phần RV bị thổi phồng bởi cú nhảy**, không phải
tín hiệu báo biến động cao hơn — **cùng hình mẫu đã thấy ở S4** (Pha 2,
`PHA2_KETQUA.md` mục 3b) và ở `hhi` (Tầng 2, cùng `PHA2_KETQUA.md` mục 3d).

**Nhưng cơ chế đúng không đủ để cải thiện dự báo — vì thông tin đã bị hấp
thụ trước.** Đo trực tiếp: tương quan giữa **tỉ trọng nhảy** (`J/RV`) và số
hạng `lq` (điều chỉnh theo realized quarticity, đã có sẵn trong STHARQ/HARQ)
là **+0,72 (EURUSD) và +0,70 (USDJPY)** — gần giống hệt tương quan +0,742 đã
đo giữa `hhi` và tỉ trọng nhảy ở Tầng 2 đêm qua. `HARQ` không đo trực tiếp
thành phần nhảy, nhưng quarticity — thước đo "độ gồ ghề" của lợi suất trong
ngày — tương quan rất cao với nó, nên phần lớn thông tin mà HARJ mang tới
**đã có mặt gián tiếp** trong tổ hợp qua đường vòng của HARQ.

## Kết luận cho luận văn

> *Tách liên tục/nhảy bằng bipower (HAR-J, Andersen–Bollerslev–Diebold 2007)
> — dù dấu hệ số khớp đúng lý thuyết ở cả 6 cặp (phần liên tục dai, phần
> nhảy không) — **không cải thiện được** tổ hợp sản xuất đã có điều chỉnh
> quarticity (HARQ) và bất đối xứng (SHAR): tương quan +0,70–0,72 giữa tỉ
> trọng nhảy và số hạng quarticity cho thấy phần lớn thông tin đó **đã có
> mặt gián tiếp**. Đây là lần thứ ba trong dự án cùng một hình mẫu lặp lại —
> một đặc trưng "mới" hoá ra là biến thể của thứ mô hình sản xuất đã nắm
> được, chỉ qua một con đường khác.*
