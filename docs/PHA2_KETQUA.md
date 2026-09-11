# PHA 2 — BIÊN BẢN ĐÓNG BĂNG VÀ QUYẾT ĐỊNH (PHASE2_NEWS_RESULT)

Lập 11/09/2026. Nhánh `replan-2026`.
Theo `02_PHASE_2_NEWS_AWARE.md` Week 3 (quyết định STOP/GO) và Week 4 (đóng băng).

Văn bản này **đóng băng** những gì Pha 2 đã thử và kết luận được. Sau khi ký,
không script nào của nhánh tin tức được thêm giả thuyết mới mà không mở một
biên bản mới và đếm lại vào `KHOA_SO.md`.

---

## 1. Đóng băng — những gì KHÔNG được đổi nữa

| hạng mục | giá trị chốt |
|---|---|
| **Kho văn bản** | 129 thông cáo FOMC, 2010-01 → 2025-12, `data/tin_tuc/fomc/` |
| **Nguồn bất ngờ thị trường** | USMPD (SF Fed), `data/usmpd/USMPD.xlsx`, sheet *Monetary Events* |
| **Cửa sổ tin** | thông cáo 30 phút · họp báo 70 phút · sự kiện tiền tệ 100 phút (định nghĩa của USMPD, không tự đặt) |
| **Chống rò rỉ** | cửa sổ đóng ~15:00 New York ngày họp; đặc trưng chỉ áp **từ phiên kế tiếp** |
| **Trục đích** | hướng giá (H8, H8b, H8c, H8e) và **biên độ/biến động** (M1 vs M2) |
| **Mô hình nền M1** | `log_rv(t+1) ~ log h_HAR(t) + số phiên kể từ họp`, OLS trên huấn luyện |
| **Thước đo** | QLIKE bất biến thang đo (Bregman: `r − log r − 1`); DM + Newey–West; SPA (Hansen 2005); Westfall–Young maxT |
| **Chia dữ liệu** | `split.py` — huấn luyện / kiểm định / kiểm tra, không đổi từ Giai đoạn 1 |
| **Cửa sổ sau họp** | K = 5 phiên |
| **Quy tắc chọn** | chọn trên **kiểm định**, chấm **một lần** trên kiểm tra |
| **Ngưỡng quyết định** | dương = thắng M1 ngoài mẫu với DM p < 0,05 **và** vững qua ≥ 3/4 lát cắt |

Tổng đã đếm vào `KHOA_SO.md`: **183 giả thuyết tin tức** (H8 54 · H8b 54 ·
H8c 24 · H8e 51) và **11 biến thể M1/M2**.

---

## 2. Đã thử gì — bảng tổng

### 2a. Trục hướng giá — năm họ biểu diễn văn bản

| họ | cách biểu diễn | giả thuyết | sống sót | SPA p |
|---|---|---|---|---|
| H8 | đặc trưng thủ công (thay đổi câu chữ, độ dài, giọng điệu) | 54 | **0** | 0,162 |
| H8b | embedding câu (MiniLM-L6-v2) + PCA(3) | 54 | **0** | 0,861 |
| H8c | phân loại chủ đề chính | 24 | **0** | 0,669 |
| H8e | trích xuất có cấu trúc bằng LLM (4 trường chốt trước) | 51 | **0** | 1,000 |
| — | *đếm/độ mới/đa dạng nguồn* | — | **không áp dụng** | — |

Họ thứ năm của roadmap (news count / recency / source diversity) không áp dụng:
kho là **một nguồn, một văn bản mỗi kỳ họp**, nên cả ba đại lượng đó là hằng số.
Ghi nhận là giới hạn của dữ liệu, không phải bỏ qua.

**Thứ tự SPA p đáng chú ý:** 0,162 → 0,669 → 0,861 → **1,000**. Biểu diễn càng
hiểu văn bản sâu, tín hiệu càng yếu. Đó đúng là dạng người ta kỳ vọng khi
**không có tín hiệu nào tồn tại** — nếu có tín hiệu thật, biểu diễn tốt hơn
phải bắt được nhiều hơn.

### 2b. Trục biên độ — M1 vs M2 bằng mô hình học được

| nhóm | biến thể tốt nhất | kiểm tra so M1 | DM p |
|---|---|---|---|
| **văn bản** (R1–R4) | R1 nhãn LLM (tốt nhất trên kiểm định) | **+3,5%** (tệ hơn) | 0,154 |
| **bất ngờ lãi suất** (S1–S3) | S1 \|MP1\| | **+2,5%** (tệ hơn) | 0,0058 |
| **phản ứng FX** (S4) | **\|phản ứng EURUSD\| 100 phút** | **−2,7%** (tốt hơn) | **0,0009** |

Độ vững của S4 (mục 8e của `PHA2_TINTUC.md`): 6/6 cặp · 4/5 năm · 3/3 nhóm độ
mạnh · walk-forward −2,1% (p=0,0031). Đạt ngưỡng quyết định ở mục 1.

### 2c. MDES — kết luận âm mạnh tới đâu

Phễu tin tức đạt lực 80% ở **lift ≥ 1,35** (so với **1,20** của phễu Giai đoạn
1). Đối chứng âm: 0,0/108 dương tính giả. Vậy kết luận âm loại trừ được mọi
hiệu ứng mạnh hơn 1,35, **không** loại trừ được khoảng 1,20–1,35.

---

## 3. QUYẾT ĐỊNH — kết luận **tách đôi**

Roadmap đòi một phán quyết POSITIVE hoặc CREDIBLE-NEGATIVE. Bằng chứng không
cho một phán quyết duy nhất, và ép nó thành một sẽ là bóp méo:

### 3a. **KẾT LUẬN ÂM ĐÁNG TIN** cho *nội dung* tin tức — **STOP**

183 giả thuyết, bốn họ biểu diễn độc lập (từ đếm từ tới LLM), hai trục đích,
**không một cái nào sống sót**; SPA p tệ dần khi biểu diễn tốt dần; MDES loại
trừ được hiệu ứng ≥ 1,35.

Lý do căn bản, nhất quán với tài liệu: **nội dung thông cáo phần lớn đã được
dự đoán trước**; phần bất ngờ theo định nghĩa không nằm trong văn bản
(Kuttner 1998; Gürkaynak–Sack–Swanson 2005; Nakamura–Steinsson 2018;
Bauer–Swanson 2023).

→ **DỪNG** nhánh biểu diễn văn bản. Không mở rộng sang ECB/BOE/BOJ: sẽ chỉ
nhân bốn cùng một phép thử với cùng một lỗi khái niệm.

### 3b. **DƯƠNG** cho *độ lớn phản ứng thị trường* — nhưng **không thuộc Pha 2**

\|phản ứng EURUSD\| thắng M1 −2,7% trên kiểm tra (p=0,0009), vững qua bốn lát
cắt. Đây là kết quả dương đầu tiên và duy nhất của nhánh.

**Nhưng cơ chế chốt trước đã bị bác bỏ.** Giả thuyết đăng ký là *bất ngờ lớn →
biến động sau CAO hơn*; hệ số ước lượng **âm** (−0,122) và ổn định trên mọi cửa
sổ mở rộng. Thứ biến này làm là **chiết khấu phần ngoại suy thừa của HAR**: cú
nhảy trong cửa sổ công bố làm phồng RV ngày họp, HAR mang sang ngày sau, nhưng
cú nhảy đó không dai.

Vậy nó **không phải tín hiệu tin tức** theo nghĩa Pha 2 định nghĩa. Nó là hiệu
chỉnh **vi cấu trúc biến động trong ngày** mà nến ngày gộp mất — về bản chất
thuộc **Tầng 2**, và lẽ ra phải đến từ dữ liệu phút của chính repo (đã có sẵn),
không phải từ một sự kiện FOMC.

→ **CHUYỂN** phát hiện này sang Tầng 2 dưới dạng giả thuyết mới, chốt trước và
đếm riêng: *"phần RV rơi vào một cửa sổ hẹp quanh sự kiện lịch nên bị chiết
khấu khi ngoại suy"*. Phép thử đúng là dùng RV trong ngày của chính repo cho
mọi phiên, không chỉ 135 kỳ FOMC. **Chưa làm.**

### 3c. Không mở bộ niêm phong

Week 4 nêu chạy bộ niêm phong một lần sau khi đóng băng. **Không chạy ở đây**,
vì hai lý do:

1. Phần *dương* của Pha 2 vừa được kết luận là **không thuộc Pha 2**. Tiêu tốn
   lần mở duy nhất cho một biến chưa được đặt đúng tầng là lãng phí không thu hồi.
2. Phần *âm* không cần bộ niêm phong: "không có gì" trên kiểm tra không mạnh
   thêm khi có thêm "không có gì" trên bộ niêm phong.

Bộ niêm phong giữ nguyên cho lần chạy cuối của toàn hệ thống, theo `KHOA_SO.md`.

---

## 4. Những gì Pha 2 **không** làm được — khai báo thẳng

| hạng mục | tình trạng |
|---|---|
| ECB / BOE / BOJ | **không làm** — kho lưu trữ ECB dựng bằng JS; và theo 3a thì không nên làm |
| họ "đếm tin / độ mới / đa dạng nguồn" | **không áp dụng** với kho một nguồn |
| khoảng lift 1,20–1,35 | **không loại trừ được** (giới hạn mẫu 129 thông cáo) |
| ý nghĩa từng cặp riêng của S4 | **không đạt** (n=85/cặp, p 0,13–0,42); chỉ có ý nghĩa khi gộp |
| đa kiểm định cho S4 | 6 biến thể; Bonferroni p = 0,0054 — vẫn có ý nghĩa, nhưng đã khai báo |
| cơ chế của S4 | **bác bỏ giả thuyết chốt trước**; giải thích thay thế chưa được kiểm độc lập |

---

## 5. Tái lập

```bash
python src/run_h8_tintuc.py       # H8  — đặc trưng thủ công
python src/run_h8b_embedding.py   # H8b — embedding + PCA
python src/run_h8c_chude.py       # H8c — chủ đề
python src/run_h8e_llm.py         # H8e — trích xuất LLM
python src/kiem_pheu_h8.py        # MDES của phễu tin tức
python src/run_m2_bien_dong.py    # M1 vs M2, bốn biểu diễn văn bản
python src/run_m2_batngo.py       # M1 vs M2, bất ngờ thị trường (S1–S4)
python src/run_m2_vung.py         # độ vững của S4 (năm / độ mạnh / walk-forward)
```

---

## 6. Chữ ký

Người chịu trách nhiệm luận văn xác nhận đã đọc mục 1 (đóng băng), mục 3
(quyết định) và mục 4 (khai báo hạn chế):

Họ tên: ______________________  Ngày: ____________  Ký: ______________________

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký, không phải
> công cụ điền hộ.
