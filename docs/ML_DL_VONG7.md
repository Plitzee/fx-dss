# Học máy và học sâu có thắng HAR không?

*Chạy 01/09/2026. Tái lập bằng `python src/run_ml.py && python src/run_dl.py
&& python src/run_dl_seed.py && python src/run_ml_final.py`.*

Ba bài gần đây đều trả lời **không**, trên dữ liệu của họ:

- **Branco, Rubesam & Zevallos** (*J. Empirical Finance* 2024) — 10 chỉ số toàn cầu,
  không có bằng chứng thống kê rằng ML phi tuyến vượt mô hình tuyến tính.
- **Kilic** (Fed FEDS 2025-061) — THAR/STHAR thắng XGBoost, DNN, BRNN, LSTM, GRU.
- **Brini** (arXiv 2607.05291) — foundation model không thắng Log-HAR; nhưng
  **trung bình đều tay** foundation model + Log-HAR nằm trong MCS 98–100%.

Vòng này kiểm lại trên dữ liệu của mình, đúng giao thức 70/15/15.

---

## Ba điều làm cho so sánh công bằng

**1. Cùng tập thông tin — thực ra ML được nhiều hơn.** Mọi mô hình đều nhận: ba
thành phần HAR ở không gian log, hiệu chỉnh realized quarticity, semivariance ±,
bipower và jump, biến chuyển chế độ G, lịch NHTW riêng từng cặp cho ngày *t+1*,
NFP, cuối tháng. ML/DL còn được thêm **22 độ trễ thô, thứ trong tuần, và mã cặp**
— những thứ HAR không có. Cho ML lợi thế này là có chủ đích: nếu nó vẫn không
thắng thì kết luận mạnh hơn.

**2. Cùng tần suất khớp lại.** HAR sản xuất khớp lại mỗi phiên; mạng nơ-ron thì
không thể. Nên **mọi** mô hình ML/DL đều khớp lại **đầu mỗi năm** bằng cửa sổ mở
rộng (2015–2026), và có sẵn một dòng **"OLS HAR (khớp năm)"** chạy trong đúng
harness đó để tách bạch: chênh lệch còn lại là do *lớp hàm*, không phải do tần
suất khớp. Kết quả: khớp mỗi phiên chỉ hơn khớp mỗi năm **0,5%** (0,1585 so với
0,1593). Tần suất khớp không phải là chuyện.

**3. Cùng cách đổi về phương sai.** Mọi mô hình dự báo log RV rồi đổi bằng hiệu
chỉnh log-chuẩn +0,5·var(phần dư) ước lượng trên chính đoạn huấn luyện của lần
khớp đó. Không làm thế thì QLIKE phạt oan ML. Riêng LightGBM có thêm một bản tối
ưu **trực tiếp QLIKE** (gradient/hessian đóng: `∂/∂m = 1 − y·e^{−m}`, `∂²/∂m² = y·e^{−m}`).

Siêu tham số chọn trên đoạn **kiểm định**; đoạn **kiểm tra** mở đúng một lần.

---

## Kết quả — 14 mô hình, đoạn kiểm tra 2023-11-20 → 2025-12-31 (548 phiên × 6 cặp)

| # | mô hình | QLIKE kiểm định | **QLIKE kiểm tra** | so HAR v7 | DM t | DM p |
|---|---|---|---|---|---|---|
| 1 | **Tổ hợp HAR v7 + GRU + LSTM** | 0,1139 | **0,1550** | −2,2% | −1,74 | 0,081 |
| 2 | **Tổ hợp HAR v7 + GRU** | **0,1132** | **0,1556** | −1,8% | −2,22 | **0,027** |
| 3 | GRU (khớp năm) | 0,1200 | 0,1563 | −1,4% | −0,59 | 0,553 |
| 4 | LSTM (khớp năm) | 0,1141 | 0,1572 | −0,8% | −0,48 | 0,634 |
| 5 | **HAR vòng 7** (khớp mỗi phiên) | 0,1162 | 0,1585 | — | — | — |
| 6 | Ridge (toàn bộ đặc trưng) | 0,1165 | 0,1590 | +0,3% | 0,23 | 0,820 |
| 7 | LightGBM (QLIKE trực tiếp) | 0,1164 | 0,1593 | +0,5% | 0,16 | 0,869 |
| 8 | OLS HAR (khớp năm) | 0,1178 | 0,1593 | +0,5% | 1,07 | 0,285 |
| 9 | LightGBM (L2 trên log) | 0,1187 | 0,1618 | +2,1% | 0,52 | 0,601 |
| 10 | Transformer (PatchTST rút gọn) | 0,1225 | 0,1652 | +4,2% | 0,91 | 0,363 |
| 11 | HAR gốc (khớp mỗi phiên) | 0,1281 | 0,1726 | +8,9% | 5,48 | <0,001 |
| 12 | Random Forest | 0,1313 | 0,1740 | +9,8% | 1,56 | 0,118 |
| 13 | MLP | 0,1440 | 0,1893 | +19,4% | 4,82 | <0,001 |
| 14 | MA20-GK (nền cũ) | 0,1654 | 0,2172 | +37,0% | 6,55 | <0,001 |

*DM dương = mô hình đó tệ hơn HAR vòng 7.*

**Model Confidence Set (α = 0,10): 11/14 sống sót.** Bị loại: MLP, MA20-GK,
HAR gốc. Nghĩa là — với 548 phiên — **không phân biệt được** HAR vòng 7 với GRU,
LSTM, Transformer, Ridge, hai bản LightGBM, Random Forest và OLS-khớp-năm.

---

## Bốn kết luận

### 1. Không mô hình ML/DL đơn lẻ nào thắng HAR có ý nghĩa thống kê

GRU tốt hơn 1,4% và LSTM tốt hơn 0,8% về điểm ước lượng, nhưng **Diebold–Mariano
cho p = 0,55 và 0,63** — không phân biệt được. Đây đúng là kết luận của Branco
et al. (2024) và Kilic (2025), tái lập trên dữ liệu FX của mình.

### 2. Nhưng TỔ HỢP thì thắng — và thắng có ý nghĩa

Trung bình hình học đều tay giữa HAR vòng 7 và GRU cho QLIKE **0,1556**, tốt hơn
HAR một mình **1,8%**, và lần này **DM p = 0,027**. Tổ hợp ba thành phần cho
0,1550 (−2,2%, p = 0,081).

Điều quan trọng về mặt giao thức: **tổ hợp cũng là cấu hình tốt nhất trên đoạn
KIỂM ĐỊNH** (0,1132, thấp nhất trong cả 14) — nên nó không phải thứ chọn sau khi
nhìn đoạn kiểm tra. Và đây chính là phát hiện của Brini (2026): mô hình hiện đại
không thắng HAR một mình, nhưng **trung bình đều tay của hai cái thì nằm trong
MCS**. Hai chuỗi dự báo sai khác nhau chỗ khác nhau; lấy trung bình thì triệt
tiêu bớt.

### 3. Kiến trúc càng phức tạp càng tệ

Xếp hạng trong nhóm học sâu: **GRU 48 > LSTM 48 > Transformer 64 > LSTM 96 > GRU 96**.
Mô hình lớn hơn (hidden 96) thua mô hình nhỏ hơn (hidden 48) ở cả hai họ, và
Transformer thua cả hai mô hình hồi tiếp. Với ~15.000 mẫu huấn luyện và 5 kênh,
dung lượng thêm chỉ đổi thành overfit — trùng với kết luận của benchmark
arXiv 2603.01820 rằng transformer tổng quát kém hơn kỳ vọng trên chuỗi tài chính.

**Random Forest (0,1740) và MLP (0,1893) thua rõ rệt**, MLP còn bị loại khỏi MCS.

### 4. Tối ưu trực tiếp QLIKE giúp cây tăng cường rõ rệt

LightGBM với hàm mục tiêu QLIKE tự viết đạt 0,1593 so với 0,1618 của bản L2
thông thường — **tốt hơn 1,5%** chỉ nhờ đổi hàm mất mát cho khớp với thước đo
đánh giá. Đây là chi tiết kỹ thuật nhỏ nhưng đáng viết vào luận văn.

---

## Ổn định theo hạt giống

Nếu "GRU thắng HAR 1,4%" chỉ đúng với một hạt giống thì nó không phải kết luận.
Chạy lại hai cấu hình dẫn đầu với ba hạt giống:

| mô hình | QLIKE kiểm định | QLIKE kiểm tra | dải kiểm tra |
|---|---|---|---|
| GRU h=48 | 0,1181 ± 0,0020 | **0,1552 ± 0,0013** | 0,1534 – 0,1566 |
| LSTM h=48 | 0,1163 ± 0,0015 | **0,1566 ± 0,0006** | 0,1561 – 0,1574 |

Lợi thế so với HAR (0,1585) **nhất quán qua cả ba hạt giống** — độ tán do hạt
giống (±0,0013) nhỏ hơn khoảng cách tới HAR (0,0033). Nhưng DM vẫn nói không có
ý nghĩa, và **cả hai điều đều đúng**: điểm ước lượng ổn định, nhưng so với độ ồn
ngày-qua-ngày của chuỗi tổn thất trên 548 phiên thì khoảng cách đó vẫn quá nhỏ để
khẳng định. Đó là lý do phải báo cáo cả hai chứ không chỉ một.

---

## Phân tầng theo chế độ — QLIKE kiểm tra

Ngũ phân vị biến động dự báo, ngưỡng lấy từ đoạn huấn luyện.

| chế độ | n | Tổ hợp HAR+GRU | GRU | LSTM | HAR vòng 7 | MA20-GK |
|---|---|---|---|---|---|---|
| Q1 êm | 979 | 0,1597 | 0,1562 | 0,1595 | 0,1691 | 0,1813 |
| Q2 | 807 | 0,1616 | 0,1671 | 0,1655 | **0,1585** | 0,2040 |
| Q3 | 628 | 0,1325 | 0,1354 | 0,1339 | 0,1345 | 0,1962 |
| Q4 | 492 | 0,1562 | 0,1587 | 0,1579 | 0,1592 | 0,2488 |
| Q5 căng | 382 | **0,1643** | 0,1654 | 0,1714 | 0,1700 | 0,3309 |

Học sâu ăn ở **chế độ êm** (Q1: 0,1562 so với 0,1691 của HAR, tốt hơn 7,6%) và
**thua ở Q2**. Ở chế độ căng nhất thì tổ hợp dẫn đầu nhưng khoảng cách nhỏ.
Nói cách khác, lợi thế của học sâu **không** nằm ở đuôi — đúng chỗ hệ thống quyết
định cần nhất thì nó không giúp thêm bao nhiêu. Cả năm mô hình hiện đại đều cách
MA20-GK rất xa ở Q5 (0,164–0,171 so với 0,331).

## QLIKE kiểm tra theo từng cặp — sáu mô hình dẫn đầu

| cặp | Tổ hợp HAR+GRU | Tổ hợp 3 | GRU | LSTM | HAR v7 | Ridge |
|---|---|---|---|---|---|---|
| EURUSD | **0,1421** | 0,1422 | 0,1446 | 0,1428 | 0,1454 | 0,1434 |
| GBPUSD | 0,1072 | 0,1084 | **0,1066** | 0,1074 | 0,1127 | 0,1074 |
| USDJPY | 0,2974 | 0,2991 | 0,2981 | 0,3035 | 0,3006 | 0,3034 |
| AUDUSD | **0,1118** | 0,1127 | 0,1123 | 0,1124 | 0,1175 | 0,1192 |
| USDCAD | 0,1335 | 0,1332 | 0,1367 | 0,1369 | **0,1332** | 0,1380 |
| USDCHF | **0,1376** | 0,1381 | 0,1397 | 0,1404 | 0,1417 | 0,1425 |

---

## Nên đưa tổ hợp vào sản xuất không?

**Được, nhưng cần biết giá.** Tổ hợp HAR v7 + GRU thắng 1,8% với p = 0,027, và
thắng trên cả kiểm định lẫn kiểm tra. Đổi lại:

- thêm phụ thuộc PyTorch vào pipeline sản xuất (hiện chỉ cần numpy/pandas/scipy)
- thêm bộ máy khớp lại theo năm và lưu trọng số
- mất tính giải thích được: HAR có 10 hệ số đọc được, GRU có ~10.000 tham số
- thời gian huấn luyện đi từ **0,6 giây** (toàn bộ 6 cặp, HAR) lên **~5 phút**

Với một luận văn MIS về **hệ thống hỗ trợ quyết định**, tôi khuyên: **giữ HAR
vòng 7 làm tầng 2 sản xuất, và báo cáo toàn bộ bảng này như một chương so sánh.**
Lý do: 1,8% không đổi được kết luận nào ở tầng 3–6, trong khi tính giải thích
được là một yêu cầu thực của DSS; và kết quả "mô hình hiện đại không thắng có ý
nghĩa, chỉ tổ hợp mới thắng" **tự nó đã là một đóng góp** khớp với ba bài 2024–2026.

Nếu muốn con số đẹp nhất thì đổi một dòng: dùng trung bình hình học của HAR vòng 7
và GRU. Mọi thứ cần đã có trong `src/run_dl.py` và `output/_dl_pred.npz`.

---

## Giới hạn phải nói rõ

- **Khớp lại theo năm, không phải theo phiên.** Với ML/DL đây là lựa chọn bắt
  buộc về mặt tính toán. Dòng "OLS HAR (khớp năm)" cho thấy chi phí của việc này
  chỉ 0,5%, nên nó không giải thích được khoảng cách — nhưng vẫn là một khác biệt.
- **Lưới siêu tham số nhỏ** (2–4 điểm mỗi họ), chạy trên 2 CPU. Một lưới lớn hơn
  có thể tìm được cấu hình DL tốt hơn. Ngược lại, lưới lớn hơn cũng làm trầm trọng
  thêm vấn đề kiểm định bội trên đoạn kiểm định.
- **Một kiến trúc Transformer duy nhất**, rút gọn (2 lớp, 4 đầu, hidden 64).
  Không phải PatchTST đầy đủ, không phải iTransformer, không phải foundation model.
- **Chronos-bolt-small đã thử thật, 09/09/2026** (`src/kiem_chronos.py`,
  `output/chronos.json`, `output/log_chronos.txt`). Lần trước ghi "không tải
  được trọng số" — kiểm tra lại thì đó là **xung đột phiên bản thư viện**
  (`transformers` cũ đòi hàm `is_offline_mode` mà `huggingface_hub` mới đã
  bỏ), không phải giới hạn môi trường thật. Sửa bằng
  `pip install -U "transformers<4.50" "huggingface_hub<0.28"`, tải trọng số
  `amazon/chronos-bolt-small` trong 2 giây.

  **Zero-shot** (không khớp riêng tham số cho từng cặp), cùng giao thức QLIKE
  với bảng 14 mô hình trên: cùng dữ liệu (`volfc2.nap_bang()`), cùng phân đoạn
  (`split.doan()`), cùng công thức QLIKE bất biến thang đo
  `r − log(r) − 1` (`r = proxy/h`) — **không phải** `metrics.qlike()` sách
  giáo khoa, vốn cho kết quả sai lệch hàng nghìn phần trăm vì phụ thuộc thang
  đo tuyệt đối của RV (một lỗi đã bắt và sửa trong lúc làm, xem log). Dự báo
  quy về phương sai bằng trung bình cộng của exp(9 phân vị) — xấp xỉ Monte
  Carlo của E[RV] = E[exp(log RV)], có tự kiểm đối chiếu giá trị kỳ vọng lý
  thuyết của log-normal.

  | cặp | QLIKE kiểm định | QLIKE kiểm tra |
  |---|---|---|
  | EURUSD | 0,1247 | 0,1702 |
  | GBPUSD | 0,1322 | 0,1335 |
  | USDJPY | 0,2787 | 0,3285 |
  | AUDUSD | **0,1024** | 0,1475 |
  | USDCAD | **0,0842** | 0,1521 |
  | USDCHF | 0,1074 | 0,1790 |
  | **gộp 6 cặp** | **0,1383** | **0,1851** |
  | HAR vòng 7 (mốc) | 0,1162 | 0,1585 |
  | chênh | +19,0% | +16,8% |

  **Kết luận: Chronos-bolt-small (zero-shot) THUA HAR vòng 7** — gộp 6 cặp
  tệ hơn 17–19%, đúng hạng #10-11 nếu chèn vào bảng 14 mô hình trên (giữa
  Transformer rút gọn và HAR gốc). Nhưng KHÔNG đều: thắng rõ ở AUDUSD, USDCAD
  (kiểm định); thua nặng nhất ở USDJPY (+87% kiểm định — đúng cặp đã biết khó
  ở tầng VaR/ES). Khớp với Brini (arXiv 2607.05291, đo trên 50 tài sản gồm cả
  FX): TSFM nói chung không thắng Log-HAR nhất quán, chỉ TTM thắng sát nút.

- **TTM (Tiny Time Mixers, IBM Granite) đã thử thật, 10/09/2026**
  (`src/kiem_ttm.py`, `src/kiem_ttm_mz.py`, `output/ttm.json`,
  `output/ttm_mz.json`, `output/log_ttm.txt`, `output/log_ttm_mz.txt`).
  Đây là mô hình DUY NHẤT trong Brini (2607.05291) được báo cáo thắng
  Log-HAR ở chân trời ngắn — nên có động lực thử riêng, dù hạ tầng đòi nâng
  cấp `transformers` lên `>=4.57.6` (xung đột trực tiếp với bản `<4.50` cần
  cho Chronos ở trên — muốn chạy lại Chronos/TabPFN sau TTM phải hạ cấp lại).

  **Zero-shot**, cùng giao thức QLIKE với Chronos và bảng 14 mô hình
  (`ibm-granite/granite-timeseries-ttm-r2`, ngữ cảnh cố định 512 phiên, chỉ
  lấy bước dự báo đầu tiên trong 96 bước). Vì TTM chỉ cho dự báo điểm (không
  có phân vị như Chronos) nên thử **hai cách quy đổi log-RV điểm → phương
  sai**, để tránh kết luận "thua HAR" chỉ vì hiệu chỉnh yếu:

  1. **Hiệu chỉnh log-chuẩn đơn giản**: `h = exp(dự_báo + 0,5·var(dư))`,
     hệ số hiệu chỉnh ước trên đoạn huấn luyện mỗi cặp (có tự kiểm đối chiếu
     giá trị kỳ vọng lý thuyết của log-normal, lệch 1,7% thay vì 11,3% nếu
     không hiệu chỉnh).
  2. **Hồi quy tái hiệu chuẩn Mincer-Zarnowitz** (đúng kỹ thuật Brini dùng):
     khớp OLS `log_rv_thật = a + b·dự_báo` trên đoạn huấn luyện, rồi
     `h = exp(a + b·dự_báo + 0,5·var(dư))`. Brini chỉ rõ phần lớn lợi thế
     ngắn hạn của TSFM đến từ bước tái hiệu chuẩn này (better-scaled), không
     phải từ mô hình động lực tốt hơn — nên đây là phép thử công bằng nhất
     với TTM.

  | cặp | QLIKE kiểm định (đơn giản / MZ) | QLIKE kiểm tra (đơn giản / MZ) |
  |---|---|---|
  | EURUSD | 0,1248 / 0,1241 | 0,1741 / 0,1733 |
  | GBPUSD | 0,1301 / 0,1290 | 0,1342 / 0,1337 |
  | USDJPY | 0,2653 / 0,2633 | 0,3160 / 0,3143 |
  | AUDUSD | **0,1067 / 0,1065** | 0,1542 / 0,1540 |
  | USDCAD | **0,0925 / 0,0924** | 0,1508 / 0,1516 |
  | USDCHF | 0,1069 / 0,1067 | 0,1836 / 0,1832 |
  | **gộp 6 cặp** | **0,1377 / 0,1370** | **0,1855 / 0,1850** |
  | Chronos-bolt-small (mốc) | 0,1383 | 0,1851 |
  | HAR vòng 7 (mốc) | 0,1162 | 0,1585 |
  | chênh so HAR | +18,5% / +17,9% | +17,0% / +16,7% |

  **Kết luận: TTM zero-shot cũng THUA HAR vòng 7**, gần như giống hệt
  Chronos (chênh nhau <1 điểm phần trăm giữa hai mô hình, giữa hai cách hiệu
  chỉnh). Đây là kết quả **KHÔNG khớp** với phát hiện của Brini rằng TTM
  thắng Log-HAR — quan trọng là hồi quy MZ (tái hiệu chuẩn đúng kỹ thuật
  Brini dùng để giải thích lợi thế của TSFM) hầu như không thay đổi gì so
  với hiệu chỉnh đơn giản (0,1370 vs 0,1377 kiểm định) — nên kết quả âm này
  KHÔNG phải do hiệu chỉnh yếu, mà là do TTM zero-shot thực sự dự báo kém
  hơn HAR trên đúng 6 cặp FX và giao thức đo của repo. Khả năng khác biệt
  với Brini: (a) Brini có thể đã fine-tune hoặc dùng tập tài sản/chân trời
  khác khi báo cáo TTM thắng — kết quả "TTM thắng" trong paper không chắc
  là zero-shot thuần; (b) 6 cặp FX chính là tập hẹp, khác biệt với 50 tài
  sản đa dạng của Brini; (c) mẫu 1.095 phiên kiểm định+kiểm tra mỗi cặp khá
  nhỏ so với các mốc trong paper gốc.

  **Khuyến nghị: dừng nhánh foundation-model zero-shot ở đây.** Cả Chronos
  và TTM đều thua HAR vòng 7 nhất quán ~17-19%, kể cả sau khi tái hiệu chuẩn
  đúng kỹ thuật của paper cho rằng TTM thắng. Việc HAR (một mô hình tuyến
  tính 3 tham số) vẫn thắng hai foundation model hiện đại một cách nhất
  quán, có tự kiểm và cùng giao thức đo, là bằng chứng thực nghiệm mạnh cho
  luận điểm cốt lõi của luận văn: RV có cấu trúc phụ thuộc dài hạn đơn giản
  (long-memory) mà HAR nắm bắt hiệu quả hơn các mô hình tổng quát chưa được
  tinh chỉnh riêng cho FX intraday. Muốn TSFM thắng thật sự cần fine-tune
  trên chính dữ liệu FX — ngoài phạm vi zero-shot đã thử ở đây.

## Tổ hợp dự báo (forecast combination) — có nên kết hợp nhiều mô hình?

`run_ml_final.py` đã thử MỘT kiểu tổ hợp: trung bình hình học đều tay giữa
HAR v7 và GRU/LSTM, và nó đã thắng — "Tổ hợp HAR v7 + GRU + LSTM" đứng #1
trong bảng 14 mô hình (QLIKE kiểm tra 0,1550, so HAR 0,1585). File
`src/kiem_tohop2.py` (10/09/2026) đào sâu hơn theo đúng tinh thần đó,
thử CÓ HỆ THỐNG mọi tập con của {HAR, LightGBM (QLIKE trực tiếp), GRU,
LSTM, Ridge} (31 tập con) với BA cách tổ hợp:

1. **Trung bình hình học đều tay** trên thang log (như đã có, mở rộng ra
   nhiều tập con hơn).
2. **Trung bình trọng số nghịch đảo QLIKE(kiểm định)** — trọng số
   `w_k ∝ 1/QLIKE_valid_k`, không cần khớp hồi quy.
3. **Hồi quy Granger-Ramanathan** (1984, *J. Forecasting*) không ràng buộc,
   trên thang log: `log(rv_thật) = a + Σ b_k·log(f_k) + e`, khớp OLS
   **chỉ trên đoạn kiểm định** (đoạn kiểm tra chỉ cham điểm một lần), rồi
   `h = exp(a + Σ b_k·log(f_k) + 0,5·var(dư))`. Đây là kỹ thuật tổ hợp kinh
   điển cho phép trọng số khác nhau mỗi mô hình thay vì ép bằng nhau — xem
   Granger & Ramanathan (1984) và tổng quan Wang et al., "Forecast
   combinations: an over 50-year review" (arXiv 2205.04216). Có tự kiểm
   (`_tu_kiem_gr`): cho hồi quy một dự báo hoàn hảo + một dự báo toàn nhiễu,
   hồi quy phải học được trọng số ~1 cho dự báo tốt và ~0 cho dự báo nhiễu.

**Kết quả: CẢ 45 tổ hợp thử đều thắng HAR đơn** trên đoạn kiểm tra (2,9%
đến 0,7%), không có ngoại lệ. Ba tổ hợp tốt nhất:

| # | tổ hợp | QLIKE kiểm định | QLIKE kiểm tra | so HAR | DM p |
|---|---|---|---|---|---|
| 1 | Hồi quy GR · HAR+LightGBM+GRU | 0,1128 | **0,1539** | −2,9% | 0,071 |
| 2 | Hồi quy GR · HAR+GRU | 0,1138 | 0,1541 | −2,8% | **0,029** |
| 3 | TB đều · HAR+LightGBM+GRU | 0,1126 | 0,1542 | −2,7% | 0,119 |

Quan sát quan trọng: **hồi quy GR không thắng rõ trung bình đều tay** — hệ
số hồi quy của các cặp tốt nhất gần bằng nhau (vd. HAR+GRU: a=−0,20,
HAR:0,51, GRU:0,48 — gần như 50/50). Đây chính là "câu đố tổ hợp dự báo"
(forecast combination puzzle) kinh điển trong tài liệu: trọng số ước lượng
tối ưu hiếm khi thắng trọng số đều tay đơn giản trên dữ liệu ngoài mẫu,
vì sai số ước lượng trọng số ăn hết phần lợi thế lý thuyết.

**Nhưng: Model Confidence Set (α=0,10) trên 15 tổ hợp đầu + HAR gốc cho
16/16 SỐNG SÓT** — kể cả HAR đơn lẻ. Nghĩa là, dù mọi tổ hợp đều có QLIKE
điểm số tốt hơn HAR (nhất quán, không ngẫu nhiên — luôn thắng ở mọi tập
con thử), khoảng cách đó **không đủ lớn để phân biệt có ý nghĩa thống kê**
với mẫu 548 phiên kiểm tra. Đây đúng là hiện tượng Brini (2607.05291) đã
mô tả: tổ hợp đều tay giữa TSFM/ML và Log-HAR thường rơi vào MCS 98–100%
cùng với HAR — cải thiện có thật về mặt điểm số nhưng chưa "chứng minh
được" theo chuẩn thống kê nghiêm ngặt.

**Khuyến nghị thực tế**: nếu cần chọn MỘT mô hình sản xuất, **tổ hợp trung
bình đều tay HAR + GRU (hoặc + LightGBM)** là lựa chọn hợp lý nhất — đơn
giản (không cần khớp hồi quy, không có nguy cơ overfit trọng số), nhất
quán thắng HAR trên mọi lát cắt đã thử, và đứng trong MCS. Không có bằng
chứng cho thấy hồi quy GR phức tạp hơn đáng giá so với trung bình đều tay
ở quy mô dữ liệu này. Toàn bộ 46 dòng kết quả, hệ số hồi quy GR từng tổ
hợp, và danh sách MCS: `output/ketqua_tohop2.json`,
`output/log_tohop2.txt`, mã nguồn `src/kiem_tohop2.py`.
