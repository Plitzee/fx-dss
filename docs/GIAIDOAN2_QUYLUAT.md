# Giai đoạn 2 — khai phá quy luật: phễu 1.890 → 0

*04/09/2026. Tái lập: `python src/run_quyluat.py`.
Kết quả: `output/quyluat.json`, `output/quyluat_wy9.json`, log `output/log_quyluat.txt`.*

Đây là câu hỏi trung tâm của luận văn: **có tồn tại một danh sách quy luật, học
từ nhiều cặp cùng lúc, chuyển giao được sang cặp chưa từng thấy không?**

Câu trả lời trên dữ liệu này: **không** — và cách nó không mới là phần đáng viết.

---

## 1. Thiết kế: không gian giả thuyết phải liệt kê được đầy đủ

Westfall–Young hiệu chỉnh cho **số giả thuyết đã thử**. Nếu đi tìm quy luật một
cách mở — chạy CART rồi lấy lá, chạy motif rồi lấy cụm — thì không ai biết thực
sự đã thử bao nhiêu, và mọi hiệu chỉnh bội đều là giả.

Nên không gian được định nghĩa trước và vét cạn:

```
vị từ  = một hoặc HAI mệnh đề dạng (đặc trưng, ô phân vị)
đích   = một trong ba lớp
```

12 đặc trưng đọc được × 3 ô = 36 mệnh đề → **630 vị từ × 3 lớp = 1.890 giả
thuyết**, con số biết trước, in ra mỗi lần chạy.

Đặc trưng: σ̂ · ATR phân vị · RSI · ADX · Bollinger %B · khoảng cách EMA50 ·
Supertrend chiều · MACD hist · |z| hôm nay · z hôm nay · TSMOM 20 · tính dai vol.

Ngưỡng phân vị chốt trên **đoạn huấn luyện**. Đích **dịch một phiên** — đặc
trưng của ngày t nói về lớp của ngày t+1 (bài học từ rò rỉ ở `run_ml3.py`).

---

## 2. Phễu

| bước | còn lại |
|---|---|
| không gian giả thuyết (liệt kê đầy đủ) | **1.890** |
| đủ 100 lần khớp | 1.764 |
| thô p<0,05 (chưa hiệu chỉnh) | **1.186** |
| sống sót Westfall–Young | **9** |
| còn tin riêng sau đối chứng có điều kiện | **0** |
| chuyển giao được (bỏ-một-cặp) | 0 |
| tái lập trên kiểm tra | 0 |

Ngưỡng `max|z|` của null khối 5 ngày: 90% **4,58** · 95% **4,83** · 99% **5,41**.

**Hệ số thổi phồng: 1.186 so với 88 kỳ vọng nếu toàn nhiễu — gấp 13,5 lần.**

Đây là con số phải đưa vào luận văn. Nó nói: nếu ai đó chạy đúng bộ đặc trưng
này, không hiệu chỉnh bội, họ sẽ "tìm ra" **1.186 quy luật có ý nghĩa thống kê**
và không cái nào là thật.

---

## 3. Chín vị từ sống sót — và tất cả đều là cùng một thứ

| vị từ | lớp | n | lift | z | b sau điều kiện | **t sau điều kiện** |
|---|---|---|---|---|---|---|
| σ̂ cao | đi ngang | 6.559 | 0,482 | −28,02 | +0,0038 | **0,47** |
| σ̂ vừa | đi ngang | 6.119 | 0,608 | −20,47 | −0,0041 | **−0,75** |
| σ̂ cao | tăng | 6.559 | 1,246 | 14,68 | +0,0128 | **1,25** |
| σ̂ thấp | đi ngang | 5.622 | 0,715 | −14,27 | +0,0047 | **0,61** |
| σ̂ cao | giảm | 6.559 | 1,216 | 12,52 | −0,0166 | **−1,64** |
| σ̂ vừa | tăng | 6.119 | 1,174 | 10,07 | +0,0005 | **0,08** |
| σ̂ vừa | giảm | 6.119 | 1,175 | 9,81 | +0,0036 | **0,52** |
| σ̂ thấp | giảm | 5.622 | 1,144 | 7,73 | +0,0082 | **0,84** |
| σ̂ thấp | tăng | 5.622 | 1,111 | 6,13 | −0,0129 | **−1,31** |

**Cả chín đều là σ̂ rời rạc hoá.** Không một chỉ báo kỹ thuật nào lọt vào:
không RSI, không ADX, không Bollinger, không MACD, không Supertrend, không
khoảng cách EMA, không ATR phân vị, không TSMOM.

Và cả chín đều **rớt ở cửa đối chứng có điều kiện** — |t| lớn nhất là 1,64,
ngưỡng là 3,0. Hiển nhiên, vì vị từ **chính là** σ̂, mà biến kiểm soát cũng là
log σ̂. Chúng không mang thông tin độc lập nào.

**Đọc theo cách khác:** trong toàn bộ không gian 1.890 giả thuyết, thứ duy nhất
sống sót hiệu chỉnh bội đúng cách là **chính dự báo biến động mà hệ thống đã
có**. Khai phá không tìm thêm được gì.

---

## 4. Ba nhánh độc lập, cùng một kết luận

| nhánh | không gian | sống sót W-Y | còn tin riêng |
|---|---|---|---|
| SAX **biến động** (`run_sax_stats.py`) | 336 | **41** | **2** (t = 8,94 và 6,05 khi có HAR) |
| SAX **hướng giá** (`run_sax_gia.py`) | 351 | **0** | — |
| Ngưỡng đặc trưng **ba lớp** (file này) | 1.890 | 9 | **0** |

Bất đối xứng rất rõ và rất nhất quán: **trục biến động có cấu trúc khai phá
được; trục hướng đi và trục ba-lớp thì không.**

Nhánh biến động vượt một cái rào cao hơn hẳn (max|z| ≈ 69 so ngưỡng 29,51) và
để lại hai mẫu mang thông tin **độc lập với HAR**. Hai nhánh còn lại không để
lại gì.

---

## 5. Tiêu chí dừng ở mục 10.4 — đã kích hoạt chưa?

`REPLAN_2026.md` mục 10.4 đòi **cả ba** điều kiện: mọi họ trượt Hansen SPA, không
quy luật nào qua ngưỡng 3.5 sau LOPO, **và** đúng trên cả hai mục tiêu lẫn cả ba
tầm hạn.

Hiện có:

| điều kiện | trạng thái |
|---|---|
| không quy luật nào qua ngưỡng sau LOPO | **đúng** — 0 quy luật thậm chí chưa tới được cửa LOPO |
| trên cả hai mục tiêu | mới đo mục tiêu P, h=1 |
| trên cả ba tầm hạn | mới đo h=1 |
| Hansen SPA cho cả họ | chưa chạy |

**Chưa kích hoạt đầy đủ, nhưng đã nghiêng hẳn.** Cộng với `run_ml3.py` (LightGBM
−0,0148 và GRU −0,0594 — trần GBM nằm **dưới** nền) thì họ H4 cũng đã trả lời:
trần khai thác được của bộ đặc trưng này thấp hơn chính nền σ̂.

---

## 6. Sản phẩm nếu tiền đề không đứng — theo đúng mục 10.4

Không có `rules_v1.csv` vì không quy luật nào sống sót. Sản phẩm xuất xưởng là
thứ mục 10.4 đã viết sẵn:

1. **Ba ô chạy trên mục tiêu P**, sinh từ tầng 2 + hiệu chuẩn — đang chạy tại
   `fx-dss.vercel.app`, BSS +0,0074 [+0,0046; +0,0106] trên đoạn kiểm tra.
2. **41 quy luật biến động** đã sống sót kiểm soát bội (2 trong đó mang thông
   tin độc lập với HAR) làm **tầng giải thích**, không làm nguồn dự báo.
3. **Quy luật loại-trừ** từ giai đoạn 0: không mở vị thế theo đà khi σ̂ ở tercile
   cao nhất (Sharpe −0,615, p = 0,001, âm 12/12 ô).
4. **Chương kết quả** là một kết quả âm có giá trị công bố, với con số cụ thể:
   1.890 giả thuyết liệt kê đầy đủ, hệ số thổi phồng 13,5 lần, và chín vị từ
   sống sót hoá ra đều là chính biến số mô hình đã có.

---

## 7. Còn thiếu

| việc | ghi chú |
|---|---|
| mục tiêu R, và h = 5, 20 | cần cho tiêu chí dừng đầy đủ |
| Hansen SPA / White Reality Check | phát biểu "cả họ không thắng nền" |
| Motif (matrix profile) | họ H2, chưa chạy — nhưng nó **không** liệt kê được đầy đủ nên phải xử lý bội khác |
| Tập khoá sổ | **chưa mở**, đúng luật — mở một lần ở cuối |

---

## Kiểm chứng chính cái phễu (05/09/2026)

Con số "0/1.890" tự nó **mơ hồ** giữa hai khả năng khác hẳn nhau: (a) không có
quy luật nào, hay (b) phễu quá chặt nên không bắt được gì. Không phân biệt được
thì nó không phát biểu thành câu gì cả. `src/kiem_pheu.py` phân biệt bằng hai
phép chạy chuẩn của thiết kế thí nghiệm.

Giá trị tới hạn `max|z|` dưới null khối, 1.000 hoán vị: **4,72**.

### Đối chứng âm — xáo trộn khối kết cục, 10 lần

| lần | vượt ngưỡng | qua điều kiện hoá |
|---|---|---|
| 1–10 | **0** | **0** |

Không một dương tính giả nào trên 1.890 giả thuyết, ở cả 10 lần. **Phễu không
rò rỉ.** Nếu bước này ra số dương thì mọi kết quả giai đoạn 2 đều phải vứt.

### Đối chứng dương — tiêm quy luật đã biết, 60 lần mỗi mức

Cách tiêm: chọn một vị từ **thật** có ≥300 lần khớp làm giá đỡ, rồi ở đúng
những hàng nó khớp, đổi kết cục sang lớp đích với xác suất vừa đủ đạt lift mong
muốn. Hàng không khớp giữ nguyên, nên cấu trúc tự tương quan của chuỗi được giữ.

| lift đặt | lift thực | \|z\| trung vị | bắt được | **lực** |
|---|---|---|---|---|
| 1,02 | 1,090 | 2,02 | 1/60 | 2% |
| 1,05 | 1,095 | 1,76 | 0/60 | 0% |
| 1,10 | 1,128 | 3,51 | 2/60 | 3% |
| 1,15 | 1,163 | 4,81 | 5/60 | 8% |
| 1,20 | 1,201 | 6,33 | 24/60 | **40%** |
| 1,35 | 1,346 | 10,76 | 60/60 | **100%** |
| 1,50 | 1,504 | 14,51 | 60/60 | 100% |

*Ghi chú trung thực:* phép tiêm có **sàn ~1,09** — ở nhiều vị từ, tỷ lệ hậu
nghiệm sẵn có đã cao hơn mức đặt, nên hai dòng đầu thực chất đo cùng một mức
với dòng 1,10. Ba dòng đó nhất quán với nhau (0–3%), nên không đổi kết luận.

### Kết luận phải phát biểu lại

**Hiệu ứng nhỏ nhất phát hiện được ở lực 80%: lift = 1,35.**

Nên câu đúng **không phải** *"không tồn tại quy luật nào"*, mà là:

> Phễu này bắt được quy luật có lift ≥ 1,35 với xác suất ≥ 80%, và không tìm
> thấy quy luật nào. Do đó **mọi quy luật mạnh hơn 1,35 đã bị loại trừ** trên
> dữ liệu này. **Quy luật yếu hơn 1,20 thì KHÔNG loại trừ được** — ở mức đó lực
> phát hiện chỉ 40%, và ở 1,15 chỉ còn 8%.

Đây là một hạn chế thật, và nó **không nhỏ**. Lift 1,35 nghĩa là xác suất lớp
đó cao hơn nền 35% — với nền ~1/3 thì là **33% → 45%**, một lợi thế rất lớn
theo chuẩn ngoại hối. Còn một quy luật lift 1,15 (33% → 38%), nếu có thật và
ổn định, vẫn có thể có ý nghĩa kinh tế mà phễu này **không thấy**.

Nguyên nhân là cỡ mẫu: 21.606 hàng phát hiện chia cho 1.890 giả thuyết, sau khi
Westfall–Young đẩy ngưỡng lên `|z| > 4,72` để khống chế sai lầm loại I toàn cục.
Đó là cái giá phải trả cho việc kiểm định bội trung thực — và nó phải được nêu
trong luận văn, chứ không được giấu sau con số 0.

Muốn hạ ngưỡng phát hiện thì có ba đường, đều tốn: thu hẹp không gian giả thuyết
(ít giả thuyết hơn → ngưỡng thấp hơn), thêm cặp/thêm năm dữ liệu, hoặc chuyển
sang khống chế FDR thay vì FWER (chấp nhận vài dương tính giả để đổi lấy lực).
