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
- **Không thử foundation model** (Chronos, TimesFM, Moirai, TTM) — không tải được
  trọng số trong môi trường này. Brini (2026) đã đo giúp: chỉ TTM hơn Log-HAR
  1,3–1,8%, các mô hình khác không thắng.
