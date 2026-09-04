# Tầng 4 — Hai lỗ hổng của trần rủi ro, và cách vá

Lập 29/08/2026. Cả hai đều là lỗi **phạm vi áp dụng**: công thức trần rủi ro
đúng trong điều kiện của nó, nhưng phiếu quyết định đưa con số ra mà không nói
điều kiện, nên người dùng vi phạm mà không biết.

## 1. Lỗ hổng lớn — trần rủi ro chỉ đúng cho MỘT vị thế

`f_ruin_cap` trả về đòn bẩy sao cho P(vốn tụt dưới 50% trong 250 phiên) ≤ 1%.
Nhưng nó tính cho **một** vị thế đứng một mình. Phiếu quyết định in con số đó
ra, người dùng mở ba lệnh, và ngân sách phá sản bị tiêu ba lần.

Mức độ nghiêm trọng, đo trên chính dữ liệu của hệ thống (mỗi vị thế ở trần
đầy đủ, ngân sách đặt ra là **1%**):

| Số cặp mở | Phá sản (hướng ngẫu nhiên) | Phá sản (cùng hướng USD) |
|---|---|---|
| 1 | 0,56% | 0,56% |
| 2 | 14,94% | 14,94% |
| 3 | 36,96% | 36,96% |
| 4 | 20,03% | 55,23% |
| 6 | 11,70% | **73,60%** |

Mở 6 lệnh cùng hướng USD ở đúng cỡ mà hệ thống khuyến nghị cho **73,6%** xác
suất cháy tài khoản, trong khi phiếu ghi 1%. Sai 70 lần.

### Vì sao không được dùng 1/√k

Sáu cặp trong tập phát triển đều có USD một vế. Nếu để nguyên chiều yết giá
thì tương quan trung bình là **−0,09** — nhìn như đã tự phòng hộ. Nhưng đó là
ảo giác của quy ước yết giá: mua EURUSD và **bán** USDJPY đều là "bán USD".
Quy về cùng chiều USD thì tương quan trung bình là **+0,44**:

| | AUDUSD | EURUSD | GBPUSD | USDCAD | USDCHF | USDJPY |
|---|---|---|---|---|---|---|
| AUDUSD | 1,00 | 0,52 | 0,55 | 0,66 | 0,37 | 0,27 |
| EURUSD | 0,52 | 1,00 | 0,62 | 0,45 | 0,66 | 0,39 |
| GBPUSD | 0,55 | 0,62 | 1,00 | 0,47 | 0,43 | 0,27 |
| USDCAD | 0,66 | 0,45 | 0,47 | 1,00 | 0,32 | 0,17 |
| USDCHF | 0,37 | 0,66 | 0,43 | 0,32 | 1,00 | 0,45 |
| USDJPY | 0,27 | 0,39 | 0,27 | 0,17 | 0,45 | 1,00 |

Biến động danh mục 6 cặp chia cho biến động một vị thế là **0,73**, không phải
0,41 như giả định độc lập. Phân tán chỉ bằng hơn nửa mức người ta tưởng.

### Luật đề xuất và kiểm chứng

```
k_danh_mục = 1 / sqrt( k + k(k−1)·ρ )
```

với ρ là tương quan trung bình **sau khi quy về cùng chiều USD**. Đo lại sau
khi áp:

| Số cặp | ρ TB | 1/√k (ngây thơ) | k_danh_mục | Phá sản khi áp | Nếu không áp |
|---|---|---|---|---|---|
| 1 | — | 1,00 | 1,00 | 0,56% | 0,56% |
| 2 | 0,52 | 0,71 | 0,57 | 0,44% | 14,94% |
| 3 | 0,56 | 0,58 | 0,40 | 0,47% | 36,96% |
| 4 | 0,54 | 0,50 | 0,31 | 0,54% | 55,23% |
| 6 | 0,44 | 0,41 | 0,23 | 0,76% | 73,60% |

Mọi cấu hình về dưới ngân sách 1%. Đã cài `k_danh_muc()` trong
`position_sizing.py`; `PositionSizer.size(..., so_vi_the=k)` và phiếu quyết
định in hệ số ra để người dùng thấy nó đang bị cắt vì cái gì.

**Mặc định `so_vi_the=1` là một cái bẫy đã biết.** Nếu người dùng quên khai
thì hệ thống lặng lẽ cho lại con số sai. Phiếu quyết định vì thế in thẳng dòng
"Đã khai N vị thế" cùng cảnh báo. Bản giao diện sau này nên bắt buộc khai, đừng
để mặc định.

## 2. Lỗ hổng thứ hai — bảo đảm 250 phiên chỉ đúng nếu định cỡ lại mỗi phiên

Trần rủi ro lấy σ̂ **hôm nay** rồi tuyên bố một bảo đảm cho **250 phiên tới**.
Biến động không đứng yên: tỷ số σ sau 250 phiên chia σ hôm nay có trung vị
0,97 nhưng p95 là 2,06 và p99 là 2,85. **16% số ngày** có biến động về sau cao
hơn 1,5 lần.

Hệ quả, đo bằng cách cho đường σ thật chạy thay vì giữ nguyên:

| Cách dùng | Phá sản thật |
|---|---|
| Đặt cỡ một lần rồi giữ 250 phiên | **1,95%** |
| Định cỡ lại mỗi tháng | 1,15% |
| Định cỡ lại mỗi phiên | 0,41% |

Ngân sách là 1,00%. Đặt một lần thì vi phạm gấp đôi; mỗi tháng vẫn vi phạm;
**chỉ định cỡ lại mỗi phiên mới đúng**. Đây không phải lỗi công thức mà là
điều kiện áp dụng chưa được nói ra. Phiếu quyết định giờ in điều kiện này.

## 3. Một quan sát chưa xử lý

Khi định cỡ lại mỗi phiên, phá sản thật là 0,41% so với ngân sách 1,00% — tức
trần đang **thận trọng quá 2,4 lần**, và đang bỏ lỡ tăng trưởng. Có thể nới
ngân sách để lấy lại phần đó.

Chưa làm, có lý do: `K_SLIP` và `k_danh_mục` vừa siết vào theo hai hướng khác,
và tập khóa sổ chưa mở. Nới ngân sách dựa trên chính tập phát triển là đúng
kiểu điều chỉnh sẽ không sống sót ngoài mẫu. Việc này nên làm **sau** P3, nếu
kết quả trên tập khóa sổ khớp với tập phát triển.

## 4. Lỗ hổng thứ ba — tương quan vùng cực đoan (đã vá 03/09/2026)

ρ=0,44 là trung bình **toàn mẫu**. Chia theo tercile biến động thì ρ gần như
phẳng (0,43–0,46, CI chồng lấn) — nhìn thoáng tưởng không có gì. Nhưng tách
đúng **đuôi cực đoan** (`src/run_corr_regime.py`) thì khác hẳn:

| Vùng | n ngày | ρ đo được | CI 95% |
|---|---|---|---|
| Toàn mẫu (= hằng số đang dùng) | — | 0,443 | — |
| Top 5% biến động nhất | 217 | **0,544** | [0,461, 0,625] |
| Top 1% biến động nhất | 48 | **0,594** | [0,497, 0,684] |

CI không chạm 0,44 — tăng thật, đúng lúc rủi ro cần siết chặt nhất. Với danh
mục 6 vị thế cùng hướng USD, `k_danh_mục` đúng ra phải là 0,205 (top 1%) thay
vì 0,228 (hằng số) — đòn bẩy đang được cho phép cao hơn thật khoảng 10% đúng
lúc căng thẳng nhất.

**Vá:** `PositionSizer` giờ tự nâng ρ lên `RHO_CANG_THANG = 0,55` (làm tròn
giữa hai mức đo được) khi σ̂ chạm ngưỡng top 5% của tập huấn luyện — ngưỡng
và hệ số đều tính trên huấn luyện, không rò rỉ. Phiếu quyết định in thêm dòng
cảnh báo "VÙNG CĂNG THẲNG" khi kích hoạt. Bài học phương pháp: câu hỏi "có
sập theo chế độ không" phải hỏi đúng độ phân giải đuôi phân phối — tercile
quá thô để thấy hiệu ứng này, giống hệt bài học chi phí giao dịch (trung vị
COVID chỉ tăng 2 lần, p95 tăng 67 lần).

## 5. Tái lập

```bash
python src/position_sizing.py     # tự kiểm, gồm bảng hệ số danh mục + vùng căng thẳng
python src/decision_record.py     # phiếu có in k danh mục và điều kiện áp dụng
python src/run_corr_regime.py     # đo ρ theo chế độ + đuôi cực đoan (nguồn của mục 4)
```
