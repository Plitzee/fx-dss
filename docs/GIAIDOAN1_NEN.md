# Giai đoạn 1 — hình thức hoá ba lớp và bảng nền

*04/09/2026. Theo `docs/REPLAN_2026.md` mục 10, giai đoạn 1.
Tái lập: `python src/diem3.py && python src/balop.py && python src/run_balop.py`.
Kết quả: `output/nen3.json`, log thô `output/log_balop.txt`.*

Giai đoạn này **không khai phá quy luật nào**. Nó dựng thước đo, để khi giai
đoạn 2 khai phá thì đã có sẵn cái để chấm và một con số cụ thể phải vượt.

| File | Vai trò |
|---|---|
| `src/diem3.py` | chỉ số chấm điểm ba lớp — điểm log, Brier, BSS, ECE, MCE, AUC hướng, bootstrap khối — có tự kiểm |
| `src/balop.py` | định nghĩa hai mục tiêu R và P, bốn nền — có tự kiểm |
| `src/run_balop.py` | bảng nền đầy đủ: 2 mục tiêu × 3 tầm hạn × 4 nền, 6 cặp, chấm trên kiểm định |

---

## 1. Luận điểm mục 2.1 — đã kiểm trực tiếp, và nó đúng

Kế hoạch nói: dải "đi ngang" co giãn theo σ̂ sẽ làm ô vàng **đứng im**. Nay đo
được, trên EURUSD, h=1, nền "chỉ σ̂":

| | độ lệch chuẩn của P(đi ngang) theo ngày | thấp nhất | cao nhất |
|---|---|---|---|
| **mục tiêu R** (dải `k·σ̂`) | **1,10 × 10⁻¹⁶** | 0,3429 | 0,3429 |
| **mục tiêu P** (dải `b` cố định) | **7,88 × 10⁻²** | 0,0815 | 0,5812 |

Mục tiêu R cho **đúng một con số, mọi ngày, tới sai số máy**. Nếu giao diện chạy
trên định nghĩa đó thì ô vàng in 34,29% từ nay đến hết đời. Mục tiêu P chạy từ
8% đến 58%.

Đây là lý do phải giữ **cả hai**: R để đo kỹ năng vượt trên tầng 2, P để hiển
thị. Tự kiểm của `src/balop.py` khoá luận điểm này lại bằng `assert`, nên nếu
sau này ai đó đổi định nghĩa mà làm hỏng nó thì tự kiểm sẽ báo.

---

## 2. Một chỗ công thức trong kế hoạch không đạt được ý định của chính nó

`REPLAN_2026.md` mục 2.2 ghi: *"`b` mặc định = trung vị `σ̂` 12 tháng trượt của
từng cặp, **để lúc bình thường ba ô cân nhau**"*. Đo ra thì không cân:

| `b` | giảm | đi ngang | tăng |
|---|---|---|---|
| = trung vị σ̂ (đúng chữ trong kế hoạch) | 0,150 | **0,715** | 0,135 |
| = `kP` × trung vị σ̂, `kP` chọn trên huấn luyện | 0,342 | **0,325** | 0,333 |

Lý do đơn giản: trung vị σ̂ **chính là** ~1σ, mà `P(|z| < 1) ≈ 0,68`. Muốn ba ô
cân thì phải nhân thêm `kP ≈ 0,35–0,42`.

Đã sửa theo **ý định** chứ không theo chữ: `chon_kP()` chọn `kP` trên đoạn huấn
luyện sao cho lớp đi ngang chiếm 1/3, rồi đóng băng. Dải `b` vẫn giữ tính chất
cốt lõi — **đổi chậm hơn σ̂ 56 lần** ngày qua ngày (0,00249 so 0,13929), nên
thông tin trong ô vàng vẫn đến từ σ̂ chứ không từ dải.

---

## 3. Một ảo giác thống kê đã bắt được và vá

Bản chạy đầu cho **khí hậu học — một dự báo hằng số — ra AUC hướng 0,60 với KTC
[0,540; 0,680]**, tức "có ý nghĩa". Hằng số thì không thể phân biệt được gì; con
số đó là sai.

Nguyên nhân: AUC tính trên **6 cặp gộp thẳng**. Mỗi cặp có tần suất nền riêng,
nên điểm số nhận 6 giá trị khác nhau, và nếu cặp nào có nền P(tăng) cao hơn cũng
thật sự tăng nhiều hơn trong đoạn kiểm định thì AUC gộp vọt lên. Đó là khả năng
phân biệt **giữa các cặp**, không phải kỹ năng **định thời** — thứ duy nhất mà ô
đỏ/xanh tuyên bố.

Vá: `auc_huong()` và `auc_ktc()` nay nhận tham số `nhom`, tính AUC **trong từng
cặp** rồi lấy trung bình có trọng số `n₁n₀`. Tự kiểm của `diem3.py` dựng lại đúng
tình huống này (0,6507 khi gộp thẳng → **0,5000 chính xác** khi phân tầng) để nó
không quay lại.

Bài học phải mang sang giai đoạn 2: **mọi chỉ số gộp nhiều cặp đều phải phân
tầng theo cặp.** Nếu không, khai phá quy luật sẽ "phát hiện" ra chênh lệch tiết
diện ngang và tưởng là quy luật.

---

## 4. Bảng nền — con số giai đoạn 2 phải vượt

Chấm trên **đoạn kiểm định**, 6 cặp gộp, n = 3.282 mỗi ô. BSS so khí hậu học,
KTC bootstrap khối (khối 20/40/80 phiên cho h = 1/5/20, dài hơn `h` để không cho
KTC hẹp giả vì cửa sổ chồng lấn).

### Mục tiêu P — thứ hiện trên giao diện

| h | nền | BSS | KTC 95% của BSS | ECE | MCE |
|---|---|---|---|---|---|
| 1 | **chỉ σ̂** | **+0,0105** | **[+0,0039; +0,0169]** | 0,0156 | 0,0816 |
| 1 | σ̂ + chế độ | +0,0098 | [+0,0028; +0,0166] | 0,0135 | 0,0834 |
| 5 | **σ̂ + chế độ** | **+0,0121** | **[+0,0022; +0,0202]** | 0,0145 | 0,0726 |
| 5 | chỉ σ̂ | +0,0086 | [−0,0003; +0,0167] | 0,0152 | 0,0951 |
| 20 | σ̂ + chế độ | +0,0014 | [−0,0126; +0,0158] | 0,0364 | 0,1059 |
| 20 | chỉ σ̂ | −0,0019 | [−0,0140; +0,0189] | 0,0420 | 0,1351 |

**Ngưỡng cho giai đoạn 2:** quy luật khai phá được phải cho BSS **> +0,0105** ở
h=1 và **> +0,0121** ở h=5, với KTC không phủ ngưỡng đó. Ở h=20 nền không thắng
được khí hậu học nên bất kỳ BSS dương có ý nghĩa nào cũng đã là tiến bộ.

### Mục tiêu R — kỹ năng vượt trên tầng 2

| h | nền tốt nhất | BSS | KTC 95% |
|---|---|---|---|
| 1 | (khí hậu học) | +0,0000 | — |
| 5 | σ̂ + chế độ | +0,0005 | [−0,0026; +0,0022] |
| 20 | (khí hậu học) | +0,0000 | — |

**Không nền nào thắng khí hậu học trên mục tiêu R ở bất kỳ tầm hạn nào.** Đúng
thiết kế: σ̂ đã bị chia ra nên "chỉ σ̂" suy biến thành hằng số. Ngưỡng ở đây
sạch nhất có thể — **BSS > 0 với KTC không phủ 0**. Thắng được nghĩa là quy luật
biết điều gì đó mà mô hình biến động không biết.

---

## 5. Kỹ năng hướng đi: 24/24 ô không phân biệt được

Sau khi phân tầng, **mọi ô trong bảng đều có KTC 95% phủ 0,50** — 4 nền × 3 tầm
hạn × 2 mục tiêu. Ô cao nhất là σ̂+chế độ ở h=20 mục tiêu R: 0,5260 [0,450; 0,555].

Nền "chỉ σ̂" cho đúng 0,5000 theo cấu tạo (nó đối xứng: P(giảm) = P(tăng)). Đây
là điều **đúng đắn** với một mô hình không có tín hiệu hướng, và là mặc định mà
giao diện sẽ dùng cho tới khi có quy luật hướng nào chứng minh được điều ngược
lại. Nhắc lại: `run_sax_gia.py` đã cho 0/351 ở giai đoạn 0.

---

## 6. Một cảnh báo cho giai đoạn 3 (hiệu chuẩn)

Nền "chỉ σ̂" có **ECE tốt nhưng MCE xấu**: ECE 0,0156 so MCE **0,0816** ở h=1,
và MCE lên **0,1351** ở h=20. Nghĩa là trung bình thì hiệu chuẩn ổn, nhưng **có
thùng lệch 8–13 điểm phần trăm** — và các thùng lệch nhất là thùng xác suất cực
đoan, đúng lúc giao diện nói một con số mạnh.

Khí hậu học có MCE thấp hơn (0,024) chỉ vì nó là hằng số. Nên đây là đánh đổi
thật: nền σ̂ mua điểm trung bình tốt hơn bằng đuôi hiệu chuẩn xấu hơn. Đúng bài
học của `docs/CHISO_DANHGIA.md` — **trung bình tốt không có nghĩa là đuôi tốt**.
Giai đoạn 3 phải đo MCE, không chỉ ECE.

---

## 7. Tiêu chí xong của giai đoạn 1

| Tiêu chí (REPLAN_2026 mục 10) | Trạng thái |
|---|---|
| bảng nền đầy đủ trên kiểm định | **đạt** — 2 mục tiêu × 3 tầm hạn × 4 nền, kèm KTC |
| hàm tính lớp có tự kiểm | **đạt** — `balop.py` và `diem3.py` đều tự kiểm, gồm cả kiểm nhân quả |

Kiểm nhân quả trong `balop.py`: cắt bỏ toàn bộ dữ liệu sau mốc t rồi tính lại
`r_h[:t]` — phải trùng khít. Đạt cho h = 1 và 5.

**Chưa làm, ghi vào việc tồn:** đoạn kiểm tra chưa mở (đúng luật `split.py` —
chỉ mở một lần ở giai đoạn 4); chưa có bản theo từng cặp riêng trong bảng in
(đã lưu đủ trong `output/nen3.json` để dựng lại).
