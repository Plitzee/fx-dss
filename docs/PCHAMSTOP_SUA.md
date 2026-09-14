# Sửa `p_cham_stop` — lệch hiệu chuẩn đã đo, vá 14/09/2026

*Chẩn đoán gốc: `docs/RUIRO_ML.md` mục A2, 13/09/2026. Tài liệu này ghi lại
bản vá thực tế, phạm vi, và những gì CỐ Ý không đụng tới.*

## Vấn đề

`decision_record.p_cham_stop()` — con số "P(chạm dừng lỗ)" hiện trên
phiếu quyết định — dùng nguyên lý phản xạ `P(min ≤ −b) = 2·P(X_T ≤ −b)`
dưới Student-t khớp trên huấn luyện. Đẳng thức này giả định **bước đi độc
lập cùng phân phối** — lợi suất FX chuẩn hoá có **co cụm biến động thật**,
vi phạm giả định đó.

Đo trên dữ liệu giữ riêng: dự báo trung bình **45,25%** so với thực tế
**33,64%** — vượt ước **11,6 điểm phần trăm**. Brier tệ hơn cả một hằng
số ngây thơ (khí hậu học); ECE cao gấp **18 lần**. Đây là sai **đặc tả mô
hình**, không sửa được bằng khớp lại tham số.

## Phạm vi sửa — CỐ Ý hẹp

`p_cham_stop()` được **giữ nguyên, không sửa** — các script nghiên cứu
(`src/run_final_eval2.py`, `src/ruiro_ml.py`) dùng nó cho những con số **đã
báo cáo/đóng băng**; sửa thẳng trong hàm dùng chung sẽ âm thầm đổi số liệu
đã công bố mà không ai biết.

Thêm hàm **mới** `p_cham_stop_thucnghiem()` — không giả định phân phối
hay tính độc lập nào, đếm trực tiếp trên dữ liệu: với mỗi điểm bắt đầu,
mô phỏng tổng luỹ tích qua `horizon` phiên, kiểm tra có chạm ngưỡng
`-k_sigma` tại bất kỳ lúc nào trong đường đi không (đúng cách `RUIRO_ML.md`
đo "tần suất nền" — khớp công thức trong `src/ruiro_ml.py`).

**Chỉ dùng hàm mới ở `api/routers/risk.py`** (số hiển thị live) — không
đụng tới các script nghiên cứu.

## Một chi tiết kỹ thuật phải sửa kèm

Cách gọi cũ trong `api/routers/risk.py` nhân sẵn σ̂ với `√h` RỒI mới gọi
hàm ở `horizon=1` mặc định — một phép xấp xỉ (đã biết lệch ±14% theo chế
độ, `TANG6_TAMHAN.md`). Hàm mới mô phỏng đường đi thật qua `horizon`
phiên nên **không được nhân thêm `√h`** nữa — làm vậy sẽ tính nhân đôi
hiệu ứng tầm hạn. Đã sửa: dùng kỹ thuật **filtered historical simulation**
(Barone-Adesi, Giannopoulos & Vosper 1999) — nhân lại shock chuẩn hoá lịch
sử (`z_train`) theo σ̂ hôm nay (`z_train * sg`), mô phỏng đường đi trên
đơn vị pip thật thay vì đơn vị sigma trừu tượng.

## Kiểm chứng

- `pytest tests/` — 22/22 pass sau khi sửa.
- Gọi thật `/risk?pair=EURUSD`: bảng tầm hạn cho
  `{1: 1,72%, 5: 24,73%, 10: 40,54%, 20: 56,52%}` — tăng đơn điệu theo
  tầm hạn, cùng bậc độ lớn với ví dụ minh hoạ trong `TANG6_TAMHAN.md`
  (~5/35/52/65% ở stop 2σ).

## Tái lập

```bash
python -m uvicorn api.main:app --port 8899 &
curl "http://127.0.0.1:8899/risk?pair=EURUSD" | python -m json.tool
```
