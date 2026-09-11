# Biên bản đóng Phase 1 — Historical-Only FX-DSS

**Ngày lập:** 11/09/2026
**Theo:** `00_MASTER_ROADMAP.md`, `01_PHASE_1_HISTORICAL_ONLY.md` (HuyH, roadmap v2,
gửi 11/09/2026), mục "Task 6 — Sign-off".
**Vai trò tài liệu này:** đây là bản NHÁP biên bản đóng Phase 1 theo đúng khuôn
HuyH yêu cầu (what worked / what did not / what was ruled out / what remains
uncertain / which M1 is frozen / what Phase 2 is allowed to change). Đây
**không phải** biên bản khoá sổ dữ liệu (`KHOA_SO.md`) — hai tài liệu độc lập,
phục vụ hai câu hỏi khác nhau: `KHOA_SO.md` khoá **dữ liệu** để tránh data
snooping; tài liệu này khoá **kết luận khoa học của Phase 1** để Phase 2 có một
đối chứng cố định.

Dòng ký ở cuối để trống có chủ đích, giống quy ước của `KHOA_SO.md` — người
chịu trách nhiệm luận văn ký, công cụ không điền hộ.

---

## 1. Cái gì hiệu quả (what worked)

**Magnitude/Risk (biến động) có kỹ năng đo được, ổn định, và khó đánh bại
bằng mô hình phức tạp hơn:**

- HAR mở rộng (STHARQ+HARQ+SHAR, `volfc2.py`) đạt BSS dương **14/14 năm**,
  có ý nghĩa trên kiểm tra ở **6/6 cặp** (`CHISO_DANHGIA.md` mục 14).
- Lưới 560 cấu hình biến động (`grid2_valid.csv`) xác nhận cấu hình đơn giản
  nhất trong Model Confidence Set là đủ tốt — không cần cấu hình phức tạp.
- Conformal ACI đảm bảo độ phủ 90% thực sự (0,901–0,908) ở mọi tầm hạn, trong
  khi conformal tĩnh hỏng (0,814–1,000) — đây là năng lực sản phẩm thật, không
  chỉ là số đo nghiên cứu.
- Kỹ năng đo được tập trung ở **h=1** (walk-forward theo năm và tập conformal
  đều đồng thuận); h=5/h=20 không có bằng chứng bổ sung tương đương.

**Bốn lỗi phương pháp luận đã bắt được và biến thành quy tắc thường trực**
(chi tiết: `01_PHASE_1_HISTORICAL_ONLY.md` mục 4): `nancumsum` tạo lợi suất
giả, lỗi mẫu số H2 làm 72/72 giả thuyết sống giả, hoán vị nhãn trạng thái HMM
giữa các cặp, và lệch pha đặc trưng/mục tiêu một phiên (AUC giả 0,967). Bốn
lỗi này nay là bốn bài tự kiểm bắt buộc cho mọi đặc trưng/mô hình mới — bao
gồm cả các thử nghiệm ML/DL/foundation-model bổ sung 09–11/09/2026 (dòng 25–31,
`KHOA_SO.md`), tất cả đều có tự kiểm rò rỉ riêng trước khi báo cáo kết quả.

## 2. Cái gì không hiệu quả (what did not work)

**Direction (hướng giá) không có bằng chứng đáng tin:**

- AUC = 0,46–0,53; **24/24** khoảng tin cậy phủ 0,50.
- **0/8.469** giả thuyết quy luật sống sót qua hiệu chỉnh Westfall–Young
  (8 nhánh độc lập: SAX biến động/hướng, ngưỡng đặc trưng, phản ứng sự kiện,
  H2 motif, H3 rule-list, H5 chế độ tự tương quan, H6 HMM, H7 Matrix Profile).
- HMM: 0 quy luật, SPA p=1,000. Matrix Profile: 0 quy luật, SPA p=0,972.
- Toàn bộ tầng model (logistic, LightGBM, GRU) cho mục tiêu xác suất hướng chỉ
  cộng thêm **+0,00013 BSS** so với baseline chỉ dùng σ̂ — nằm trong sai số.

**Việc mở rộng mô hình cho MAGNITUDE cũng cho đúng kết luận tương tự, đo lại
độc lập 09–11/09/2026** (`ML_DL_VONG7.md`, `KHOA_SO.md` dòng 25–31): thử thêm
XGBoost, CatBoost, TabPFN v2, Chronos-bolt, TTM (foundation models), và 509 tổ
hợp dự báo (đều tay, trọng số nghịch đảo QLIKE, hồi quy Granger-Ramanathan,
stacking phi tuyến) — **không tổ hợp/mô hình nào vượt qua được Model Confidence
Set so với HAR đơn**. Tổ hợp tốt nhất (HAR+GRU+CatBoost, hồi quy GR) cải thiện
QLIKE kiểm tra 3,0%, nhưng MCS (α=0,10) vẫn giữ HAR đơn trong tập không phân
biệt được. Đây là **cùng một phát hiện, đo hai lần độc lập trên hai trục khác
nhau** (direction và magnitude): độ phức tạp mô hình không phải nút thắt của
hệ thống này.

**RL (REINFORCE) không dùng được cho xác suất hướng:** BSS sụp đổ còn −0,576
so với huấn luyện giám sát cùng kiến trúc (+0,0053) — bằng chứng thêm rằng vấn
đề nằm ở tín hiệu/dữ liệu, không phải ở lớp thuật toán tối ưu hoá.

## 3. Cái gì đã bị loại có đo (what was ruled out)

| Đã thử | Kết quả | Nguồn |
|---|---|---|
| HMM (K=2,3,4) cho quy luật hướng | 0 quy luật, SPA p=1,000 | dòng 21, `KHOA_SO.md` |
| Matrix Profile (K=20, L=5/10/20) | 0 quy luật, SPA p=0,972 | dòng 21 |
| FDR(BY), CAViaR, Fixed-Share | không ăn tiền so với W-Y/phân vị tĩnh/Hedge trơn | dòng 15 |
| Fuzzy Mamdani cho định cỡ | hơn tích tuyến tính chỉ +0,08% → loại | dòng 9, mục 4.5 |
| V1/V2/CAViaR cho đuôi VaR/ES | thua V0 (phân vị thực nghiệm) | dòng 10 |
| Hiệu chuẩn lại (nhiệt độ, vector scaling) | xấu đi trên kiểm tra ở h=20 | dòng 11 |
| Cây quyết định/GBM/RF/MLP/Transformer cho biến động | thua mọi biến thể HAR | `ML_DL_VONG7.md` |
| Chronos-bolt, TTM (foundation model, zero-shot) | thua HAR ~17–19%, khớp Brini (2607.05291) | dòng 26–27 |
| Stacking phi tuyến (LightGBM meta-learner) cho tổ hợp | **tệ nhất** trong 509 tổ hợp — tệ hơn cả HAR đơn | dòng 31 |
| RL (REINFORCE) cho xác suất hướng | sụp đổ so với giám sát | dòng 25 |

## 4. Cái gì còn chưa chắc chắn (what remains uncertain)

- **Cỡ mẫu quy luật.** Nhiều nhánh chỉ có ~100–500 quan sát độc lập hiệu quả
  (vd. FOMC 129 kỳ họp) — không loại trừ được hiệu ứng nhỏ hơn ngưỡng MDES đã
  định (lift 1,20).
- **Tổ hợp HAR+GRU+CatBoost** có DM p=0,041 riêng lẻ (duy nhất trong 509 tổ
  hợp) nhưng MCS không xác nhận — chưa đủ bằng chứng để thay thế HAR sản xuất,
  cũng chưa đủ bằng chứng để bác bỏ hẳn khả năng tổ hợp có ích.
- **FOMC (Phase 2 Week 1 sơ bộ)** cho p=0,162 — thấp nhất trong 8 họ đã thử
  nhưng xa ngưỡng 0,05. Không phải phát hiện, nhưng cũng chưa phải bằng chứng
  đủ mạnh để đóng hẳn hướng tin tức (xem `PHA2_TINTUC.md` mục 6).
- **USDJPY, USDCHF** không đạt đủ cả ba phép kiểm VaR/ES ở mức 99% (4/6 cặp
  đạt) — cảnh báo vẫn giữ trên giao diện, chưa có lời giải triệt để.

## 5. M1 được đóng băng (which M1 is frozen)

Đúng theo `KHOA_SO.md` mục 4, không thay đổi gì thêm ở đây:

| Tầng | Cấu hình chốt |
|---|---|
| Biến động (tầng 2) | STHARQ+HARQ+SHAR, `deseason=none`, `crosspair=off`, `event=capday`, cửa sổ mở rộng, `recal=off`, `lam=0` |
| Nền ba lớp theo tầm hạn (tầng 5) | h=1: tổ hợp trực tuyến (Hedge, η=0,5); h=5,20: σ̂+chế độ cuộn |
| VaR/ES (rủi ro) | V0 — phân vị thực nghiệm trên huấn luyện |
| Định cỡ (tầng 6b) | Kelly/trần phá sản × 0,92 × k_vol × k_dd × k_danh_mục, tuyến tính (không fuzzy) |
| Tập dự báo | Conformal ACI, độ phủ 90% |

**Prediction contract v2 (Task 2 của roadmap) — đã đối chiếu `api/main.py`:**
không có trường `expected_return` hay nhãn `confidence` chủ quan nào trong mã
nguồn hiện tại; API đã trả về tập dự báo conformal (`conformal_alpha`,
`_tap_conformal`) và mức kỹ năng theo tầm hạn (`KY_NANG_THEO_H`). **Điều kiện
này của roadmap coi như đã đạt**, không cần việc lập trình bổ sung.

**Không đổi cấu hình sản xuất nào** trong toàn bộ đợt thử nghiệm 09–11/09/2026
(Chronos, TTM, XGBoost, CatBoost, TabPFN, 509 tổ hợp) — đây là nghiên cứu đối
chứng, kết luận là "giữ nguyên", đã ghi lại đầy đủ ở `KHOA_SO.md` dòng 25–31 và
phần "Thay đổi sau ngày chốt (3)".

## 6. Phase 2 được phép thay đổi gì (what Phase 2 is allowed to change)

Theo đúng ràng buộc của `01_PHASE_1_HISTORICAL_ONLY.md` mục 8:

- Phase 2 chỉ được hỏi: **"Tin tức có thêm thông tin ngoài những gì M1 đã biết
  không?"** — không được dùng dữ liệu tin tức để retune rồi thay luôn M1.
- Được thêm: đặc trưng tin tức mới (đại diện văn bản, ngân hàng trung ương
  khác ngoài FOMC), miễn tuân thủ tường lửa chống rò rỉ thời gian
  (`available_at`) và không gian giả thuyết phải khai báo trước.
- Không được: quay lại tinh chỉnh HAR/ML/DL biến động, thử thêm biến thể
  HMM/Matrix Profile, hay mở lại Phase 1 vì "model mới/phức tạp hơn" —
  cả hai trục (direction 09/2026, magnitude 09–11/09/2026) đã đo độc lập và
  cùng cho kết luận: độ phức tạp mô hình không phải nút thắt.
- M2 so với M1 phải đánh giá tách biệt theo 3 trục: direction / magnitude /
  risk (không gộp thành một chỉ số duy nhất).

## 7. Đối chiếu tiêu chí đóng Phase 1 (`01_PHASE_1_HISTORICAL_ONLY.md` mục 6)

- [x] Historical pipeline exists.
- [x] Baselines / model comparisons exist — mở rộng thêm 09–11/09/2026
      (XGBoost, CatBoost, TabPFN, Chronos, TTM, 509 tổ hợp).
- [x] Pattern/regime experiments exist.
- [x] Robust probabilistic evaluation exists.
- [x] Research guardrails exist in repo (Westfall–Young, MDES, anti-leakage,
      sealed test).
- [x] **Formal closeout/sign-off — tài liệu này**, chờ ký xác nhận ở mục 8.

## 8. Ký xác nhận

- **Ngày:** 11/09/2026
- **Trạng thái đề xuất:** `PHASE 1 = FROZEN`, `M1 = historical baseline`
  (cấu hình mục 5 ở trên).
- **Ký xác nhận:** ......................................  /  ....................

> Dòng ký để trống có chủ đích — người chịu trách nhiệm luận văn ký sau khi
> đọc và đồng ý với nội dung biên bản. Phase 2 chỉ nên coi là chính thức mở
> sau khi dòng này có chữ ký, theo đúng tinh thần `KHOA_SO.md`.
