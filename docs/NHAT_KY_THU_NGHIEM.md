# NHẬT KÝ THỬ NGHIỆM PHƯƠNG PHÁP THAY THẾ

*Bắt đầu 14/09/2026. Sổ này KHÁC `docs/KHOA_SO.md`: KHOA_SO.md đếm giả
thuyết/cấu hình cho phần hiệu chỉnh đa kiểm định của luận văn (mục đích học
thuật). Sổ NÀY để TRA CỨU NHANH — trước khi thử một phương pháp mới cho một
thành phần, xem đã ai thử chưa, thử gì rồi, kết quả sao — mục đích kỹ thuật,
phục vụ việc chỉnh sửa hệ thống liên tục. Mỗi dòng nên trỏ tới tài liệu chi
tiết riêng nếu có.*

## Cách dùng

Trước khi thử phương pháp mới cho một thành phần: tìm thành phần đó trong
bảng dưới, xem cột "đã thử". Nếu đã có nhiều dòng cho cùng thành phần, đọc
"quyết định" của lần gần nhất trước khi thử thêm — tránh lặp lại đúng phép
thử đã làm.

Sau khi chạy MỘT thử nghiệm mới (dù nhỏ), thêm MỘT dòng vào bảng — kể cả khi
kết quả âm. Format dòng:

```
| YYYY-MM-DD | <thành phần> | <phương pháp gốc/đang dùng> | <phương pháp thử> | <số liệu chính> | <ĐẠT/KHÔNG ĐẠT/CHƯA RÕ> | <link doc/script> |
```

## Bảng

| Ngày | Thành phần | Phương pháp gốc | Phương pháp thử | Số liệu chính | Quyết định | Chi tiết |
|---|---|---|---|---|---|---|
| 14/09 | H3 rule-list (Giai đoạn 2) | CART nông (max_depth=3) | RuleFit (Friedman & Popescu 2008) | BSS: CART +0,0066 vs RuleFit −0,0227; DM p=0,0000 | **KHÔNG ĐẠT** — giữ CART | [H3_RULEFIT_KETQUA.md](H3_RULEFIT_KETQUA.md) · [kiem_rulefit_h3.py](../src/kiem_rulefit_h3.py) |
| 14/09 | Tổ hợp ba xác suất (Giai đoạn 1b) | Hedge (học trực tuyến) | Stacking (logistic đa thức, khớp tĩnh) | BSS: Hedge +0,0106 vs Stacking +0,0091; DM p=0,029 | **KHÔNG ĐẠT** — giữ Hedge | [STACKING_VS_HEDGE_KETQUA.md](STACKING_VS_HEDGE_KETQUA.md) · [kiem_stacking_bahop.py](../src/kiem_stacking_bahop.py) |
| 14/09 | Mẫu hình nến — mã hoá ảnh | CNN (Chen & Tsai 2020) | Vision Transformer (patch embed + self-attention) | 6/6 cặp CNN cải thiện vs 6/6 cặp ViT xấu đi, 4/6 có ý nghĩa | **KHÔNG ĐẠT** — giữ CNN | [NEN_CNN_ANH.md](NEN_CNN_ANH.md) mục 2b · [kiem_vit_nen.py](../src/kiem_vit_nen.py) |
| 14/09 | Nhân quả VIX→biến động (Pha 3B) | Granger + Westfall-Young | PCMCI thật, Double ML, Causal Forest, CausalImpact | Double ML 6/6 cặp vững qua Holm; lớp phủ dự báo 0/6 cải thiện có ý nghĩa | **XÁC NHẬN tồn tại quan hệ, KHÔNG ĐẠT để tích hợp dự báo** | [PHA3B_TAMGIAC_NHANQUA.md](PHA3B_TAMGIAC_NHANQUA.md) |
| 14/09 | Khai phá quy luật h=5/h=20/mục tiêu R | (chưa từng chạy ở tầm hạn/mục tiêu này) | 12 nhánh Giai đoạn 2, đủ 5 tổ hợp còn thiếu | 0/6 tổ hợp sống sót W-Y; SPA bác bỏ 5/54 sau Holm (toàn bộ mục tiêu R) | **Một phần** — cần quyết định cách viết | [GIAIDOAN2_TAMHAN_KETQUA.md](GIAIDOAN2_TAMHAN_KETQUA.md) |
| 14/09 | VaR/ES USDJPY/USDCHF — hướng thứ 5 | V0 (phân vị thực nghiệm huấn luyện) | Lịch can thiệp KHÔNG định kỳ (BOJ 2022, SNB 2015), nới ngưỡng 1,5×/2×/3× | V0 đã "đạt" cả 3 backtest trên kiểm định TRƯỚC khi thêm nới — không có gì để phân biệt | **KHÔNG ĐẠT** — gặp đúng bế tắc chọn-trên-kiểm-định đã biết ở mục 5e | [VA_DUOI_CANTHIEP_KETQUA.md](VA_DUOI_CANTHIEP_KETQUA.md) · [va_duoi_canthiep.py](../src/va_duoi_canthiep.py) |
| 14/09 | VaR/ES USDJPY/USDCHF — hướng thứ 6 | V0 (phân vị thực nghiệm huấn luyện) | ADWIN (phát hiện trôi trực tuyến, không cần biết sự kiện) + EWMA thích nghi λ=0,94 | ADWIN: 0 điểm đổi phát hiện được (đã thử cả δ nhạy cao); EWMA: USDCHF xấu đi rõ (Kupiec 0,015→0,006, DQ 0,003→0,001) | **KHÔNG ĐẠT** — USDJPY vẫn mù trên kiểm định; USDCHF kiểm chứng được thật (hiếm) và cho kết quả ÂM | [VA_DUOI_ADWIN_KETQUA.md](VA_DUOI_ADWIN_KETQUA.md) · [va_duoi_adwin.py](../src/va_duoi_adwin.py) |

## Kết quả chung của đợt 14/09: "hiện có" thắng cả 3/3 lần

Ba phép thử đầu (RuleFit, Stacking, ViT) đều cho CÙNG một kết luận: phương
pháp **đơn giản/hiện có** thắng phương pháp **linh hoạt/hiện đại hơn**. Với
dữ liệu tài chính gần-nhiễu-thuần, mô hình càng linh hoạt càng dễ quá khớp
nhiễu trên huấn luyện — các ràng buộc "thô" (CART nông, trọng số trực
tuyến, tích chập cục bộ) vô tình đóng vai trò chính quy hoá phù hợp hơn.
**Không có thay đổi cấu hình sản xuất nào từ đợt này** — cả 3 xác nhận lựa
chọn hiện tại đã đúng, bằng số đo được chứ không chỉ do mặc định.

## Phase còn "độc canh", chưa thử phương pháp thay thế

Chỉ còn **một** trường hợp đã xác minh trực tiếp bằng cách đọc file:

- **Pha 3 — tầng vĩ mô đơn giản** (`docs/PHA3_TIEUCHI.md` mục 3a, 8): chỉ
  dùng lãi suất liên ngân hàng 3 tháng (2 chuỗi gốc, tần suất THÁNG). Mục 8
  "Danh sách KHÔNG LÀM" tự ghi: *"Không thêm đặc trưng ngoài 5 cái ở mục
  3a."* — **Độ ưu tiên THẤP**: Pha 3B (ngay sau đó) đã mở rộng thành 9
  chuỗi FRED tần suất NGÀY, lấp gần hết khoảng trống này. Chỉ đáng làm nếu
  luận văn muốn nói rõ ràng "đã thử cả tần suất tháng lẫn ngày."

**Đính chính một gợi ý trước đó của tôi**: tôi từng gợi ý "H5 chế độ tự
tương quan" là độc canh. Sau khi đọc lại kỹ `src/run_h5_chedo.py` — gợi ý
đó SAI. Đội dự án đã cố ý chọn trục "tự tương quan" cho H5 để KHÔNG trùng
lặp với chế độ biến động (đã có ở `SigmaCheDo`/A3) — nghĩa là ở cấp độ toàn
dự án, "chế độ thị trường" đã có **3 định nghĩa khác nhau** được thử: tam
phân vị biến động (SigmaCheDo), tự tương quan lag-1 (H5), và trạng thái ẩn
HMM (H6). Không phải một trường hợp độc canh — xin lỗi vì gợi ý vội trước
đó.

**Kết luận sau khi rà lại toàn bộ**: dự án hiện tại **gần như không còn
phase nào độc canh đáng kể** — 3 lần thử phương pháp thay thế gần đây nhất
đều xác nhận lựa chọn cũ, và các "phase gần-độc-canh" khác khi kiểm lại kỹ
đều hoá ra đã có đối chiếu ở một nhánh liền kề. Đã kiểm nhanh một ứng viên khác: **ngưỡng EVT/POT** (`src/va_duoi_evt.py`)
— đã thử 2 ngưỡng CỐ ĐỊNH (phân vị 90% mặc định, 95% làm biến thể độ nhạy),
nhưng CHƯA thử phương pháp chọn ngưỡng TỰ ĐỘNG/DỮ LIỆU-DẪN-DẮT (đồ thị
mean-excess, ước lượng Hill) — đây là ứng viên còn lại có khả năng cao
nhất, nhưng độ ưu tiên thấp vì EVT vốn đã không đổi sản xuất (trượt 2/3
tiêu chí, xem `docs/DUOI_EVT.md`).
