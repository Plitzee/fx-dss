# KIẾN TRÚC HỆ THỐNG — data pipeline, tầng tính toán, API, triển khai Web

*Lập 14/09/2026. Nhánh `replan-2026`. Viết để lấp khoảng trống hội đồng nêu:
đề cương thiếu mô tả kiến trúc phần mềm, data pipeline, và cách triển khai
API/Web App — và để trả lời câu hỏi "sự liên kết giữa các module là gì".*

---

## 1. Sơ đồ tổng thể — 5 tầng, 2 đường triển khai

```mermaid
flowchart TB
    subgraph NGUON["Nguồn dữ liệu"]
        H["HistData.com<br/>M1, 2010–2025<br/>(đóng băng, hết cập nhật)"]
        Y["Yahoo Finance<br/>từ 2026-01-01<br/>(nguồn không chính thức)"]
        F["FRED + lịch NHTW<br/>9 chuỗi vĩ mô + sự kiện"]
    end

    subgraph THUTHAP["collect/ — Data pipeline"]
        BOOT["Bootstrap lịch sử<br/>histdata_dl · prep_fx · rv5 · rv_advanced<br/>(chạy 1 lần)"]
        DAILY["live_fx.py · lich_su_kien.py<br/>(CI, 4 lần/ngày)"]
    end

    subgraph DATA["data/ — kho dữ liệu đã xử lý"]
        D1["prices/*.csv · rv_adv.csv<br/>(lịch sử, tĩnh)"]
        D2["live/*.csv<br/>(hiện hành, ghi đè mỗi ngày)"]
        D3["su_kien.csv · ngoai_sinh/<br/>(vĩ mô + lịch)"]
    end

    subgraph TINHTOAN["src/ + api/cache.py — Tầng tính toán"]
        NOI["noi_chuoi()<br/>nối lịch sử + hiện hành"]
        HAR["volfc2 — HAR sản xuất<br/>σ̂ (biến động)"]
        BALOP["balop — 4 mô hình con<br/>+ tổ hợp trực tuyến<br/>ba xác suất"]
        CF["conformal — ACI<br/>tập dự báo phủ 90%"]
        RISK["position_sizing · decision_record<br/>VaR/ES · Kelly · trần phá sản"]
        NOI --> HAR --> BALOP --> CF
        HAR -.-> RISK
        BALOP -.-> RISK
    end

    subgraph API["api/ — FastAPI (package)"]
        CACHE["cache.py — lay()/tinh()<br/>bộ nhớ đệm trong tiến trình"]
        ROUTERS["routers/<br/>meta · market · forecast · risk · admin"]
        SCHEMA["schemas.py<br/>pydantic response model"]
        CACHE --> ROUTERS --> SCHEMA
    end

    subgraph VANHANH["jobs/cap_nhat.py — điều phối hằng ngày"]
        J1["1. gọi collect/live_fx.py"]
        J2["2. gọi /refresh — tính lại cache"]
        J3["3. gọi API, chụp web/data/*.json"]
        J4["4. dựng lại 2 bản giao diện"]
        J1-->J2-->J3-->J4
    end

    subgraph TRIENKHAI["Hai đường triển khai — KHÁC NHAU có chủ đích"]
        direction LR
        LOCAL["Local/CI — FastAPI đầy đủ<br/>uvicorn api.main:app<br/>CÓ scipy, CÓ /risk, CÓ VaR/ES trực tiếp<br/>phục vụ web/ui_live.html (fetch động)"]
        VERCEL["Vercel — tĩnh + 1 hàm nhẹ<br/>deploy/public/ (HTML + JSON tĩnh)<br/>deploy/api/intraday.py (nến nội ngày,<br/>KHÔNG scipy, dưới giới hạn 250MB)<br/>rủi ro/VaR đã TÍNH SẴN, đóng gói tĩnh"]
    end

    H & Y & F --> BOOT & DAILY --> D1 & D2 & D3
    D1 & D2 & D3 --> NOI
    RISK --> CACHE
    J1 -.đọc/ghi.-> D2
    J2 -.gọi HTTP.-> API
    J3 -.gọi HTTP, ghi.-> WEBDATA["web/data/*.json"]
    J4 --> LOCAL
    J4 --> VERCEL
    WEBDATA --> VERCEL
    API --> LOCAL
```

---

## 2. Vì sao HAI đường triển khai khác nhau — quyết định kiến trúc, không phải sơ suất

Đây chính là câu trả lời cho "sự liên kết giữa 2 module là gì" ở cấp độ
triển khai: hệ thống có **một** tầng tính toán (`api/`) nhưng **hai** cách
đưa nó tới người dùng, vì ràng buộc kỹ thuật thật:

- **Local/CI**: FastAPI đầy đủ, nạp `scipy`, `pandas`, toàn bộ lịch sử
  (~36 MB CSV) để tính `noi_chuoi → HAR → ba xác suất → ACI → VaR/ES` theo
  yêu cầu. Đây là nơi DUY NHẤT các con số rủi ro được TÍNH.
- **Vercel serverless**: giới hạn gói hàm ~250 MB đã giải nén; riêng
  `pandas+scipy+numpy` đã hơn 120 MB, cộng lịch sử CSV thì vượt giới hạn.
  Giải pháp: **tách làm hai**
  1. Dự báo (σ̂, ba xác suất, VaR/ES) — mô hình NGÀY, chỉ đổi 1 lần/ngày —
     tính SẴN ở bước `jobs/cap_nhat.py` bước 3, đóng gói thành
     `web/data/*.json` tĩnh, copy nguyên khối vào `deploy/public/data/`.
  2. Nến nội ngày (chỉ để vẽ biểu đồ, không cần scipy) — tính LÚC CÓ REQUEST
     bởi `deploy/api/intraday.py`, một hàm serverless nhẹ chỉ cần
     `requests+pandas+numpy`.

`web/ui_template.html` là NGUỒN SỰ THẬT DUY NHẤT cho giao diện — biên dịch
thành 3 bản khác nhau (`ui_live.html`, `ui.html`, `deploy/public/index.html`)
bằng phép thay thế `__DATA__`/`__API__`, không phải ba bản viết tay riêng
biệt (xem `web/build.py`, `deploy/build.py`).

---

## 3. Hợp đồng giữa các tầng (module contract)

| từ tầng | sang tầng | qua gì | định dạng |
|---|---|---|---|
| `collect/*.py` | `data/` | ghi file CSV/JSON trực tiếp | CSV theo cột đã chốt, không đổi schema ngầm |
| `data/` | `api/cache.py:noi_chuoi()` | đọc CSV theo đường dẫn cố định | DataFrame `Date, open, high, low, close, rv5, ...` |
| `src/*.py` (volfc2, balop, conformal, position_sizing) | `api/cache.py` | gọi hàm Python trực tiếp (cùng tiến trình) | mảng numpy/DataFrame, không qua serialize |
| `api/cache.py` | `api/routers/*.py` | gọi hàm `lay(pair)` → dict trong bộ nhớ | dict Python (`pan`, `xs`, `sig`, `che_do`...) |
| `api/routers/*.py` | client (Web UI / script) | HTTP JSON, một phần có `pydantic response_model` | JSON, xem `api/schemas.py` |
| `jobs/cap_nhat.py` | `web/data/*.json` | gọi API qua HTTP (KHÔNG tính lại bằng đường khác) | JSON tĩnh, y hệt API trả về |
| `web/data/*.json` | `deploy/public/data/` | copy nguyên khối (`shutil.copytree`) | không biến đổi |

Nguyên tắc xuyên suốt: **API là nguồn sự thật duy nhất**. `jobs/cap_nhat.py`
không bao giờ tính lại bằng công thức khác — nó GỌI API rồi chụp lại kết
quả, để bản tĩnh (Vercel) và bản trực tiếp (local) không bao giờ lệch nhau
vì hai đường tính khác nhau.

---

## 4. Tầng kiểm-sốc giữa phiên (thêm 14/09/2026)

*Trả lời câu hỏi "hệ thống có tái phân tích liên tục theo thời gian thực
không" — không, và đây là lý do có chủ đích cộng với bản vá đã thêm.*

**Vì sao KHÔNG chạy liên tục theo nghĩa "dự báo lại mỗi giây".** Toàn bộ
dự án đã chứng minh kỹ năng dự báo chỉ tồn tại ở tầm **NGÀY** — hạ phễu
xuống H1 vẫn 0 quy luật sống sót. Chạy lại mô hình liên tục sẽ không
"khai phá thêm" được gì; nó chỉ có nghĩa nếu dùng để phát hiện khi nào
**giả định của dự báo hôm nay đã bị vi phạm**, rồi tái kích hoạt ĐÚNG mô
hình ngày đã có, sớm hơn lịch — không phải một mô hình dự báo mới.

**Cơ chế** (`collect/kiem_soc.py`, workflow `.github/workflows/kiemsoc.yml`):

```
mỗi 15 phút → gọi TrueFX lấy giá hiện tại (không cần đăng ký/khoá)
            → so với giá đã dùng để tính dự báo hôm nay (`/forecast`)
            → z = log(giá_hiện_tại / giá_mốc) / σ̂_hôm_nay
            → |z| > 1,5  → tái chạy TOÀN BỘ pipeline sản xuất hiện có NGAY
                            (không chờ lịch cron 4 lần/ngày)
            → |z| ≤ 1,5  → không làm gì, chờ lượt kiểm tiếp theo
```

Hai job GitHub Actions riêng (`kiemtra` rẻ, `taitinh` chỉ chạy khi có sốc)
để không tốn CPU CI/dữ liệu bên ngoài một cách vô ích. Chung nhóm
concurrency `capnhat` với `capnhat.yml` để không bao giờ chạy chồng lên
việc tải/commit/triển khai.

**Vì sao TrueFX, không phải Yahoo hay Twelve Data.** Đã đo trực tiếp:
Yahoo interval=1m là ảnh chụp giá (98,7% thanh o=h=l=c, không phải OHLC
thật). Twelve Data cho nến M1 sạch nhưng cần khoá + giới hạn 800
credit/ngày (ép tần suất kiểm tối đa ~15 phút/lần với 6 cặp). TrueFX cho
giá bid/ask thời gian thực, không cần đăng ký, không thấy giới hạn rõ
ràng (đã thử 5 lần gọi liên tiếp, đều HTTP 200) — hợp cho việc CHỈ cần
giá hiện tại, không cần nến dựng sẵn. ~~Dukascopy đã thử, hạ tầng
`datafeed.dukascopy.com` hiện không phản hồi ổn định (503/504/timeout,
đo trực tiếp 14/09/2026) — không dùng được~~ — **câu này SAI, xem đính
chính ở mục 4c**.

## 4c. Đính chính về Dukascopy (15/09/2026)

Câu gạch bỏ ở mục 4 rút ra kết luận về **nguồn** từ một lỗi về **cách gọi**.
Đo lại ngày 15/09/2026:

| cách gọi | kết quả |
|---|---|
| 20 request liên tục, không nghỉ | **9/20**, trễ trung vị 14,2s |
| nghỉ 5–6s giữa các lần | lấy đều, gần như không hỏng |
| đợt 8 request rồi nghỉ (tải 40 ngày × 2 cặp) | **80/80, không một lỗi nào** |

Đây là **siết hạn mức**, không phải hạ tầng chết. Chính `../dukas/dukas_v3.py`
đã ghi điều này từ 26/08/2026: *"Dukascopy cho khoảng 15-20 request rồi mới
siết"*. Và ngay trong `collect/`, `finish_dataset.py` vẫn **đang dùng**
Dukascopy thật (khối lượng D1, spread H1) với lùi bậc thang 3s→60s viết đúng
cho trường hợp này, còn `probe_aggregates.py` đã dò xong các endpoint nến gộp
từ trước. Tức khẳng định "không dùng được" mâu thuẫn với code cùng thư mục tại
chính thời điểm nó được viết.

**Hệ quả thật, không chỉ là sửa câu chữ.** Vách 60 ngày của thanh 5 phút Yahoo
— và toàn bộ rào chắn `kiem_rv_that` ở mục 4b dựng lên để canh nó — là thứ có
thể tránh được ngay từ đầu: nến M1 Dukascopy có từ **2003**, một file cho một
ngày, không cần khoá.

### Dukascopy và HistData là **cùng một dữ liệu** — đo trên 2025-10..12

Phép so quyết định không phải Dukascopy với Yahoo, mà với **HistData** — nguồn
mô hình sản xuất được khớp trên đó. Kết quả (65 phiên EURUSD, 63 phiên USDJPY,
đoạn trước mối nối 2026-01-01):

| | EURUSD | USDJPY |
|---|---|---|
| \|lệch\| giá đóng, trung vị | **0,00 pip** | **0,00 pip** |
| rv5 Dukascopy/HistData, trung vị | **1,0000** | **1,0000** |
| — cùng chỉ số ở p25 và p75 | 1,0000 | 1,0000 |
| số nến M5/ngày | 287 / 287 | 287 / 287 |

Khớp tới bốn chữ số ở cả tứ phân vị. Nói thẳng: **Dukascopy là phần tiếp nối
của chính chuỗi HistData**, không phải nhà cung cấp thứ hai.

Hệ quả đảo ngược lập luận thường gặp: đổi sang Dukascopy **xoá** mối nối đang
có chứ không tạo thêm mối nối mới. Kẻ lạc loài là Yahoo.

| nguồn | RV5 so với gốc (tick / HistData) |
|---|---|
| nến M1 Dukascopy | **0,0%** |
| Yahoo, ngày có nến 5 phút thật | **+10,5%** (EURUSD) · +1,7% (USDJPY) |
| Yahoo, ngày phải ước từ thanh giờ (`rv_uoc=1`) | **+2% đến +74%** tuỳ cặp |
| EODHD | **+12,2%** (EURUSD) |

Cột cuối cùng đo trên 543–548 phiên chồng lấn. Tệ nhất là **USDCAD +73,8%** —
hệ số quy đổi thanh-giờ-sang-M5 sai nhiều ở cặp này.

**Nhưng nó vẫn không làm lệch dự báo**, và điều đó đã được kiểm ở đúng độ lớn
sai số thật (không phải mức giả định): chia các hàng `rv_uoc=1` về mức HistData
rồi tính lại σ̂ hôm nay thì đổi **0,01%–0,15%** trên cả 6 cặp. Lý do vẫn như
mục 4b: RV ước chỉ vào qua cửa sổ chuẩn hoá `WIN_Z`, nơi z-score nuốt sai số
thang đo. Đây là bằng chứng thứ hai, mạnh hơn, cho cùng kết luận — và nó xác
nhận rào chắn ở mục 4b đặt **đúng chỗ** (cửa sổ 22 phiên), vì đó mới là nơi sai
số đi thẳng vào hồi quy.

**Chưa đổi nguồn trong lần rà soát này** — đó là thay đổi chạm số sản xuất và
cần người chịu trách nhiệm quyết. Nhưng bằng chứng nghiêng hẳn về một phía:
Dukascopy nối liền chuỗi gốc, phủ từ 2003, miễn phí, không cần khoá, và xoá
luôn vách 60 ngày cùng rào chắn dựng ra để canh nó. Tái lập:
`python collect/dukas_m1.py --doi-chung` và `python collect/probe_nguon.py`.

**Sửa kèm khi xây tầng này**: phát hiện lỗi lệch giá trong `/forecast` và
`/forecast_series` — xem mục 6.

## 4b. Rào chắn RV5 thật (thêm 15/09/2026)

*Trả lời câu hỏi "có nên lấy nến trực tiếp từ TradingView / các sàn để hệ thống
tái phân tích liên tục không" — và vá một lỗi âm thầm phát hiện khi trả lời.*

**Vì sao không đổi sang feed liên tục.** Mục 4 đã nói: kỹ năng dự báo chỉ tồn
tại ở tầm ngày, hạ phễu xuống H1 với 592.343 thanh vẫn 0 quy luật. Dữ liệu dày
hơn không mua thêm kỹ năng dự báo. TradingView ngoài ra không có đường hợp lệ —
không mở API dữ liệu, dữ liệu là mua lại nên không có quyền phát tán tiếp, và
điều khoản cấm cào; quan trọng hơn với luận văn là **không khai được nguồn gốc**,
thứ mà cả `KHOA_SO.md` dựng lên trên đó.

**Nhưng có một khiếm khuyết thật mà dữ liệu tốt hơn sẽ vá.** HAR cần phương sai
thực hiện 5 phút. Yahoo chỉ phục vụ thanh 5 phút trong ~60 ngày; ngoài cửa sổ đó
`live_fx.py` **ước** rv5 từ thanh giờ (`rv_uoc=1`). Đo ngày 15/09/2026:

| cửa sổ | vai trò | nhiễm RV ước |
|---|---|---|
| `lm` = 22 phiên | vào **thẳng** ma trận thiết kế cả ba mô hình con | **0/22 — sạch** |
| `WIN_Z` = 250 phiên | chỉ chuẩn hoá biến chế độ `G = sigmoid(z)` | **120/250 = 48%** |

Nên câu trong docstring `live_fx.py` — "RV ước không nuôi dự báo" — **không
chính xác**: nó có nuôi, qua đường `WIN_Z`. Đã đính chính tại chỗ. Hệ quả thì
không đáng kể, đã đo bằng độ nhạy: nhiễu rv5 các hàng ước đi **±20%** chỉ làm
σ̂ hôm nay đổi **tối đa 0,10%** trên cả 6/6 cặp — z-score qua sigmoid nuốt gần
hết sai số.

*(Bổ sung 15/09/2026 — phép đo trên dùng ±20% là mức **giả định**. Sau đó đã đo
được sai số THẬT của RV ước bằng cách so với HistData trên 543–548 phiên chồng
lấn: từ +1,9% đến **+73,8%** tuỳ cặp, tệ nhất ở USDCAD. Chạy lại độ nhạy ở đúng
mức đó — sửa các hàng `rv_uoc=1` về mức HistData rồi tính lại σ̂ — cho **0,01%
đến 0,15%**. Kết luận không đổi, nhưng nay dựa trên sai số đo được chứ không
phải sai số đoán. Chi tiết ở mục 4c.)*

**Chỗ nguy hiểm thật là khi số phiên RV5 thật *liên tiếp* tụt xuống dưới 22**:
lúc đó `lm` vào thẳng hồi quy, không có z-score che. Trước 15/09/2026 con số này
được **hiển thị** (`/health`, cờ `rv_uoc` trên giao diện) nhưng **không có một
assert nào** — nguồn 5 phút ngừng phục vụ thì mọi thứ vẫn chạy, vẫn ra số, chỉ
là số lệch và không ai biết.

Rào chắn (`api/cache.py::kiem_rv_that`, `collect/live_fx.py`):

```
< 22 phiên thật liên tiếp  → CHẶN CỨNG, /forecast trả 503, không phục vụ dự báo
22 .. 36                   → cảnh báo, vẫn phục vụ, banner vàng trên giao diện
>= 37                      → lành
```

Đại lượng quyết định là số phiên thật **liên tiếp tính từ cuối**, không phải
tổng: một khoảng ước chèn sát cuối vẫn cắt cửa sổ 22 dù tổng rất lớn.

Trạng thái tại 15/09/2026: **59 phiên liên tiếp, đệm 37** — không chặn, không
cảnh báo. Rào chắn không đổi hành vi nào hiện tại; nó chỉ đảm bảo lần hỏng sau
là hỏng **ồn ào** thay vì âm thầm.

Nếu về sau muốn bỏ hẳn vách 60 ngày này thì đường đúng là broker có hợp đồng
(OANDA v20, Saxo, IG) hoặc Dukascopy `.bi5` — chứ không phải để chạy lại mô
hình mỗi giây.

## 5. Việc đã dọn trong lần rà soát này

- Tách `api/main.py` (928 dòng monolithic) thành package `config/cache/
  utils/risk_logic/schemas + routers/*` — xem git log nhánh này.
- Xoá scaffold Next.js (`web/pages`, `web/api/*.py`) — đã bị chính kế hoạch
  `docs/REPLAN_2026.md` thay thế bằng kiến trúc ở tài liệu này nhưng chưa
  được dọn khỏi repo.
- `src/pipeline.py` — phát hiện đây là code thử nghiệm sớm (29/08/2026),
  đường dẫn cứng `/tmp/fx/` không khớp bố cục repo hiện tại, không được
  import ở bất kỳ đâu. Không xoá trong lần rà soát này (chưa được yêu cầu)
  nhưng cần lưu ý: đây là code chết, không phải một phần pipeline đang chạy.

## 6. Lỗi lệch giá trong `/forecast`/`/forecast_series` — sửa 14/09/2026

**Phát hiện khi nào**: lúc xây tầng kiểm-sốc (mục 4), do script kiểm-sốc
cần đọc trường `gia` từ `/forecast` — trường đó trước đây **không tồn
tại**, và khi thêm vào thì giá trị trả về sai lệch nghiêm trọng.

**Lỗi**: `api/routers/forecast.py` dùng chỉ số `i` tính từ `_idx(K, ngay)`
(hợp lệ cho `K["pan"]`, panel nghiên cứu bắt đầu 2012-02-14) để đọc thẳng
vào `K["m"]` (giá thô, đủ từ 2010-01-04) — hai mảng lệch nhau **552
dòng**. Kết quả: chỉ số "mới nhất" của `pan` (2026-09-11) khi áp vào `m`
lại trỏ tới ngày **2024-07-24** — lệch hơn 2 năm, không báo lỗi vì cả hai
chỉ số đều "hợp lệ" về mặt kỹ thuật.

**Đo được ảnh hưởng thật** (giá dùng để quy đổi `sigma_pip`/`dai_pip` từ
đơn vị tương đối sang pip, qua `sang_pip()`):

| Cặp | Lệch |
|---|---|
| EURUSD | −6,59% |
| GBPUSD | −4,64% |
| USDJPY | +0,10% |
| AUDUSD | −8,36% |
| USDCAD | −0,41% |
| USDCHF | +8,46% |

**Sửa**: thêm `api/cache.py::gia_theo_ngay()` — tra `K["m"]` theo NGÀY
thật (`np.searchsorted` trên `K["m"].Date.values`), không dùng chung chỉ
số với `pan`. Áp dụng ở cả hai điểm gọi (`/forecast`, `/forecast_series`).
`/risk` không bị ảnh hưởng vì nó dùng `m.close.values[-1]` (chỉ số `-1`
đúng cho cả hai mảng vì chúng kết thúc cùng một ngày).

**Không ảnh hưởng kết quả nghiên cứu đã đóng băng** — đã xác minh bằng
grep toàn bộ `src/`: không script backtest/nghiên cứu nào gọi qua API
`/forecast`, chúng tính trực tiếp qua `src/volfc2.py` và các module khác.
Lỗi chỉ nằm ở tầng hiển thị/phục vụ API, không lan vào số liệu đã báo cáo
trong luận văn.
