# Mẫu hình trong thị trường ngoại hối — tổng thuật tài liệu và hệ quả cho hệ thống

*Soạn 31/08/2026. Mọi bài dưới đây đã được kiểm chứng tồn tại bằng cách lấy trang tóm tắt.
Bài nào không kiểm chứng được đã bị loại bỏ khỏi danh sách.*

Câu hỏi đặt ra: **thị trường ngoại hối có mẫu hình gì, mẫu nào chung cho mọi cặp và
mẫu nào riêng từng đồng tiền, và mẫu đó ảnh hưởng ra sao tới mô hình dự báo?**

Tài liệu chia mẫu hình thành **bốn trục**. Ba trục đầu là mẫu *trong dữ liệu*;
trục thứ tư là mẫu *trong hiệu năng mô hình* — trục này quan trọng nhất với luận văn
vì nó là thứ biến quan sát thành lập luận.

| Trục | Mẫu là gì | Ảnh hưởng tới mô hình |
|---|---|---|
| A. Lịch & phiên | Chu kỳ nội ngày, nội tuần, cuối tháng | Làm **nhiễu mục tiêu** RV → phải khử trước khi ước lượng HAR |
| B. Đồng tiền | Trú ẩn, hàng hoá, carry — khác nhau về dấu và độ dai | Quyết định **gộp hay tách** 6 cặp |
| C. Hình dạng | Motif, shapelet, ký hiệu SAX | Là **đặc trưng bổ sung**, và là bài toán kiểm định bội |
| D. Chế độ | Trạng thái thị trường thay đổi theo thời gian | Quyết định **cách chấm điểm** — trung bình gộp che giấu tất cả |

---

## A. Mẫu theo lịch và phiên giao dịch (chung cho mọi cặp)

### A1. Chu kỳ nội ngày — mẫu mạnh nhất, được xác lập chắc nhất

**Krohn, Mueller & Whelan (2024), *Journal of Finance* 79(1):541–578.**
"Foreign Exchange Fixings and Returns around the Clock." DOI 10.1111/jofi.13306

Bài quan trọng nhất trong toàn bộ danh sách này. Trên 21 năm dữ liệu và 9 đồng tiền
giao dịch nhiều nhất, họ tìm ra một **hình chữ W trong lợi suất nội ngày**: USD tăng
giá đi vào mỗi cửa sổ định giá (Tokyo, London 16h/WMR, ECB) rồi mất giá ngay sau đó.
Sự đảo chiều này phổ biến và có ý nghĩa thống kê rất cao, và họ quy cho tồn kho của
nhà tạo lập chứ không phải thao túng.

Tổ tiên trực tiếp của kết quả này là **Breedon & Ranaldo (2013), *JMCB* 45(5):953–965**:
đồng tiền có xu hướng *mất giá trong giờ giao dịch bản địa của chính nó* và *tăng giá
ngoài giờ đó*, do dòng lệnh. Đây là mốc chuẩn mà mọi khẳng định về mẫu theo giờ trong
FX phải đối chiếu.

### A2. Chu kỳ nội ngày làm hỏng HAR như thế nào — mắt xích trực tiếp với tầng 2

**Dumitru, Hizmeri & Izzeldin (2025), *Journal of Banking & Finance* 170:107342.**
"Forecasting the realized variance in the presence of intraday periodicity."
DOI 10.1016/j.jbankfin.2024.107342

Đây là bài phải trích khi bảo vệ tầng 2. Kết luận: **chu kỳ nội ngày làm nhiễu chính
các thành phần RV được đưa vào HAR**, và khi khử mùa vụ các thước đo hiện thực hoá thì
dự báo RV ngoài mẫu cải thiện rõ rệt. Nói cách khác, RV thô không phải mục tiêu sạch.

Nền lý thuyết của điều này: **Dette, Golosnoy & Kellermann (2023), *Metrika* 86:315–342**
chứng minh bỏ qua chu kỳ nội ngày làm **suy luận về biến động tích hợp trở nên không hợp lệ**
với số lượng lợi suất nội ngày hữu hạn, và đưa ra hệ số hiệu chỉnh giải tích.

Công cụ khử: **Boudt, Croux & Laurent (2011), *Journal of Empirical Finance* 18(2):353–367**,
"Robust Estimation of Intraweek Periodicity in Volatility and Jump Detection". Chú ý chữ
**intraweek** chứ không chỉ intraday — nó xử lý được tương tác *ngày trong tuần × giờ trong ngày*,
đúng thứ mà FX cần vì tuần FX là 24×5 chứ không phải 24×7. **Yi (2023), *Finance Research Letters*
55:103821** áp bộ lọc kiểu Boudt–Croux–Laurent lên USD/EUR tần suất cao và cho thấy chọn bộ lọc
nào làm thay đổi đáng kể xác suất bước nhảy đo được.

> **Hệ quả cho hệ thống.** Chuyện Chủ nhật đã sửa (`merge_thin_days`) mới chỉ là phần thô nhất
> của vấn đề này. Phần tinh hơn — chu kỳ theo giờ trong ngày và theo ngày trong tuần bên trong
> mỗi phiên — vẫn chưa được khử trong `rv_adv.csv`. Đây là **cải tiến tầng 2 có cơ sở tài liệu
> mạnh nhất còn chưa làm**, và nó rẻ: chỉ là một hệ số nhân theo (giờ, thứ) ước lượng trên tập
> huấn luyện rồi chia ra trước khi cộng dồn RV ngày.

### A3. Cuối tháng và cửa sổ định giá London 16h

**Melvin & Prins (2015), *Journal of Financial Markets* 22:50–72** là bài gốc về tái cân bằng
phòng vệ cuối tháng: nhà đầu tư cổ phiếu quốc tế tái cân bằng hợp đồng kỳ hạn FX vào cuối tháng,
và dòng lệnh đó làm tỷ giá dịch chuyển **có thể dự đoán được từ lợi suất cổ phiếu tháng trước**.

Bản hiện đại của cơ chế đó: **Sizova, Syrstad & Sævareid (2025), SSRN 5143899**, dùng dữ liệu
phái sinh EMIR ở mức giao dịch. Truyền dẫn từ lợi suất danh mục nước ngoài sang dòng phòng vệ
**gần bằng một** với quỹ phòng vệ toàn phần; nhận dạng tần suất cao cho thấy **1 tỷ NOK mua kỳ hạn
làm USDNOK dịch ~0,2% trong vòng vài phút**, và con số này nhân ba trong giai đoạn biến động.

Về bản thân cửa sổ định giá: **Ito & Yamada (2018), *JIMF* 80:75–95** ("Did the reform fix the
London fix problem?") cho thấy sau cải cách 2015 thì **đỉnh khối lượng trong cửa sổ biến mất nhưng
bất thường về giá vẫn còn** — ngân hàng chuyển sang thực thi kém hiệu quả hơn, tạo ra *tính dự báo mới*.
**Benenchia, Galati & Lepone (2024), *Pacific-Basin Finance Journal* 84:102311** đánh giá lại tính
đại diện của phương pháp WM/R sau cải cách.

### A4. Ngày công bố số liệu vĩ mô

**Lee & Wang (2025), *Review of Asset Pricing Studies* 15(3–4):247–287.**
"Jumps and Post-FOMC Announcement Returns in Currency Markets." DOI 10.1093/rapstu/raaf003

Lợi suất tiền tệ sau FOMC âm có ý nghĩa và **triệt tiêu khoảng 65% của đà tăng trước FOMC**.
Đảo chiều tập trung trong **cửa sổ 12–24 giờ sau công bố**. Đây là bài mạnh nhất cho luận điểm
lịch sự kiện.

**Martins & Lopes (2024), arXiv:2411.16244**, "What events matter for exchange rate volatility?"
là bài gần khung của mình nhất trong nhóm này: dùng phương pháp co ngót để chọn ra, trong hàng trăm
công bố vĩ mô và độ trễ của chúng, cái nào thực sự dẫn dắt biến động FX — và **mô hình đồng thời
tính dai của biến động lẫn mùa vụ nội ngày**, gắn hình dạng mùa vụ với khối lượng giao dịch và
giờ mở cửa các thị trường lớn.

> **Hệ quả cho hệ thống.** Đây chính là hạng mục S2 (lịch sự kiện FOMC/ECB/BOJ/NFP/CPI) đã ghi
> trong kế hoạch. Tài liệu ủng hộ mạnh, nhưng lưu ý Martins–Lopes cảnh báo: **phần lớn công bố
> KHÔNG quan trọng**; nếu ném cả trăm biến giả sự kiện vào HAR thì sẽ overfit. Cách đúng là co ngót
> hoặc chỉ giữ nhóm bậc nhất (FOMC, NFP, CPI Mỹ, ECB).

### A5. Ngày trong tuần — điểm yếu của tài liệu, nên nói thẳng

Đây là chỗ **không có bằng chứng hiện đại mạnh**. Không tìm được bài 2022–2026 nào về hiệu ứng
ngày-trong-tuần cho FX trên *JIMF*, *JBF* hay *Journal of Empirical Finance*. Tài liệu FX về
ngày-trong-tuần đạt đỉnh vào những năm 2000 rồi dịch sang mùa vụ nội ngày và mùa vụ theo sự kiện.

Hai bài dùng tạm được:
- **Kristjanpoller & Tabak (2024), *Fractal and Fractional* 8(6):340** — 30 đồng tiền, 5.240 ngày,
  dùng **MFDFA và số mũ Hurst theo từng thứ** thay vì biến giả lợi suất trung bình, và tìm thấy
  hiệu ứng thứ trong *chiều fractal*: độ dai/độ hiệu quả khác nhau theo thứ kể cả khi lợi suất
  trung bình không khác. (Venue MDPI — dùng làm bằng chứng bổ trợ, đừng làm bằng chứng chính.)
- **Singh (2019), *Journal of Asset Management* 20(7):493–507** — hiệu ứng thứ trên các cặp chính,
  điều kiện hoá theo chế độ biến động ngụ ý.

Về khoảng trống cuối tuần: **không tìm được bài nào** coi gap cuối tuần FX là hiện tượng riêng
trên các venue mục tiêu.

> **Hệ quả cho hệ thống.** Chỗ này là cơ hội chứ không phải lỗ hổng. Nếu mình đo hiệu ứng
> ngày-trong-tuần trên RV của 6 cặp bằng bộ lọc Boudt–Croux–Laurent *intraweek*, đó là một đóng góp
> nhỏ nhưng sạch, vì tài liệu hiện đại bỏ trống.

---

## B. Mẫu riêng theo từng đồng tiền — và câu hỏi gộp hay tách

### B0. Câu trả lời của tài liệu: không gộp hẳn, cũng không tách hẳn

**Pesaran, Pick & Timmermann (2026), *Quantitative Economics* 17(2):342–393; arXiv:2404.11198.**
"Forecasting with Panel Data: Estimation Uncertainty Versus Parameter Heterogeneity."

Đây là bài phương pháp luận đúng trọng tâm nhất cho quyết định 6-cặp. Nó dẫn ra điều kiện khi nào
ước lượng gộp thắng ước lượng riêng, dưới dạng đánh đổi thiên lệch–phương sai, và — quan trọng —
**đưa ra một kiểm định hình thức về tính gộp được của dự báo (forecast poolability test)**.
Kết luận: **kết hợp dự báo và ước lượng co ngót kiểu Bayes thực nghiệm thắng cả gộp thuần lẫn
riêng thuần**; không phương pháp nào thắng phổ quát.

Phía ngược lại — **Liu, Tran, Wang, Gerlach & Kohn (2025), arXiv:2309.02072**, "Global Neural
Networks and the Data Scaling Effect in Financial Time Series Forecasting" — trên 10.000+ cổ phiếu
toàn cầu, độ chính xác dự báo biến động **cải thiện rõ rệt khi tập huấn luyện lớn hơn và đa dạng hơn**,
và mạng gộp toàn cục chuyển giao được sang tài sản chưa từng thấy mà không cần huấn luyện lại.
Đây là lập luận chống lại mô hình riêng từng cặp — nhưng chỉ khi lớp mô hình có dung lượng cao.

**Enkhbayar & Ślepaczuk (2024), Univ. of Warsaw WNE WP 2024-10** thì ngược lại, và trên đúng
**6 cặp FX chính, 2000–2023**: thay vì tìm một chiến lược chạy tốt trên mọi cặp, họ kết luận
là tìm được chiến lược hiệu quả **riêng cho từng trường hợp cụ thể**.

> **Hệ quả cho hệ thống.** Ba bài này cộng lại cho một thiết kế rõ ràng: **ước lượng HAR riêng
> từng cặp, rồi co ngót hệ số về trung bình gộp**, với mức co ngót chọn trên tập kiểm định.
> Đây là phương án trung dung mà Pesaran–Pick–Timmermann chứng minh là tốt nhất, và nó biến
> "6 cặp" từ 6 bài toán rời rạc thành một bài toán panel — mạnh hơn hẳn về mặt luận văn.
> Kiểm định poolability của họ cũng là thứ nên chạy và báo cáo.

### B1. Bao nhiêu phần là chung, bao nhiêu phần là riêng?

**Verdelhan (2018), *Journal of Finance* 73(1):375–418.**
"The Share of Systematic Variation in Bilateral Exchange Rates."

Con số neo cho toàn bộ câu hỏi: nhân tố dollar và nhân tố carry cộng lại giải thích
**từ 18% đến 80% biến động tỷ giá theo tháng** — một *khoảng*, không phải một hằng số.
Chính cái khoảng rộng đó là bằng chứng định lượng cho tính không đồng nhất giữa các cặp.

**Aloosh & Bekaert (2022), *Management Science* 68(6):4042–4064** (NBER WP 25449) phân cụm
lợi suất rổ tiền tệ G10 và tìm ra cấu trúc **hai khối rõ ràng: khối dollar và khối châu Âu**.
Mô hình ba nhân tố giải thích ~60% biến động với RMSE 0,11 so với tương quan quan sát, và
tương quan giảm theo khoảng cách địa lý.

> **Hệ quả cho hệ thống.** Đây là bằng chứng trực tiếp rằng EUR/GBP/CHF và AUD/CAD/JPY
> **không nên bị coi là các mẫu hoán đổi được từ cùng một tổng thể**. Nó cũng cho một con số
> để kiểm tra giả định ρ = 0,44 trong `k_danh_muc`: nếu 6 cặp của mình chia thành hai khối,
> thì tương quan trong khối cao hơn và giữa khối thấp hơn nhiều so với một ρ chung.
> **Đây là chỗ nên đo lại bằng dữ liệu của mình.**

### B2. Đồng trú ẩn — JPY, CHF

**Abakah, Brahim, Carlotti, Tiwari & Mensi (2024), *International Economics* 178:100503.**
"Extreme downside risk connectedness and portfolio hedging among the G10 currencies."

Kết quả về tính không đồng nhất sắc nét nhất trong danh sách. Dùng CAViaR cộng liên kết
phân vị thay đổi theo thời gian: **EUR, NOK, AUD, SEK, NZD là bên TRUYỀN cú sốc ròng,
còn JPY và CHF là bên NHẬN cú sốc ròng.** Vai trò rủi ro đuôi khác nhau *về cấu trúc* theo
đồng tiền — một mô hình gộp với hệ số chung sẽ triệt tiêu chính những dấu ngược nhau này.

**Michelis, Ning & Ponrajah (2025), *Finance Research Letters*** (WP 091, Toronto Metropolitan)
xếp hạng độ mạnh trú ẩn bằng copula chuyển chế độ trên phụ thuộc đuôi, 1999–2024:
**USD mạnh nhất; JPY giữ được tính trú ẩn ngay cả khi dollar mạnh; CHF là trú ẩn nhưng yếu hơn USD;
EUR và GBP yếu nhất.**

Nhưng có cảnh báo quan trọng: **Park & Fang (2025), *Quarterly Review of Economics and Finance*
100:101976** dùng hồi quy phân vị với biến giả khủng hoảng tương tác, 2000–2022, và tìm ra rằng
hành vi trú ẩn **thay đổi theo thời gian và phụ thuộc vào bản chất khủng hoảng** — đáng chú ý là
**JPY YẾU đi trong cú sốc Ukraine 2022**, ngược với tiên nghiệm chuẩn.

**Aquilina, Lombardi, Schrimpf & Sushko (2024), BIS Bulletin No. 90** ghi lại vụ tháo carry yen
tháng 8/2024: vị thế carry yen ước khoảng **40 nghìn tỷ JPY (~250 tỷ USD)** trước khi tháo, và
giảm đòn bẩy thuận chu kỳ cùng vòng xoáy ký quỹ khuếch đại một cú bất ngờ vĩ mô Mỹ ban đầu khá nhỏ.

> **Hệ quả cho hệ thống.** Ba bài này gộp lại nói: "JPY là đồng trú ẩn" **không phải một hệ số ổn định**.
> Cho nên một biến giả cố định theo cặp cũng sai, không chỉ mô hình gộp sai. Cách xử lý đúng trong
> hệ thống của mình là để **tầng 6 (chế độ) mang thông tin này**, chứ không phải nhét vào tầng 2 dưới
> dạng hằng số. Vụ 8/2024 cũng là một điểm gãy có ngày tháng cụ thể — rất đáng dùng để phân đoạn
> kiểm tra ngoài mẫu.

### B3. Đồng hàng hoá — AUD, CAD

**Bermpei, Ferrara, Karadimitropoulou & Triantafyllou (2024), *JIMF* 145:103096.**
"Commodity currencies revisited: The role of global commodity price uncertainty."

Bất định chung của một *rổ* giá hàng hoá dẫn dắt tỷ giá nước xuất khẩu hàng hoá: các đồng này
**mất giá ngắn hạn rồi hồi phục ở chân trời trung hạn** — một phản ứng động riêng biệt mà tác giả
gọi là tính chất "commodity uncertainty currency". Được đối chiếu tường minh với EUR và USD, vốn
thể hiện hành vi trú ẩn thông thường. **Dấu khác nhau VÀ độ dai khác nhau, trên cùng một cú sốc.**

**Bonato, Cepni, Gupta & Pierdzioch (2022), *Journal of Financial Markets* 59** cho thấy các thước
đo rủi ro khí hậu dự báo được **biến động hiện thực hoá nội ngày** cho tiền tệ của tám nước xuất
khẩu nhiên liệu hoá thạch, gồm AUD và CAD. Lưu ý: họ dùng random forest, **không phải HAR** — đừng
trích bài này cho luận điểm về tính không đồng nhất của hệ số HAR.

### B4. Carry, rủi ro sụp đổ và độ lệch

**Brunnermeier, Nagel & Pedersen (2009), *NBER Macroeconomics Annual* 23:313–347.**
"Carry Trades and Currency Crashes." (NBER WP 14473)

Bài kinh điển: tỷ giá giữa đồng lãi suất cao và đồng lãi suất thấp **lệch âm**, do tháo carry đột ngột
khi khẩu vị rủi ro và thanh khoản tài trợ giảm. Cũng ghi nhận **đồng biến động vượt mức giữa các đồng
tiền có lãi suất tương tự** — đây tự nó đã là lý lẽ cho việc *nhóm* các cặp theo đặc tính carry thay vì
gộp cả 6 một cách đồng nhất.

Bản cập nhật hiện đại tốt nhất: **Li, Sarno & Zinna (2025), *JFQA*** (CEPR DP 20587),
"Skewness Risk Premia and the Cross-Section of Currency Returns" — độ lệch phi tham số từ bất đối xứng
semivariance cộng quyền chọn tiền tệ; nhân tố phần bù rủi ro độ lệch **đi vào hạt nhân định giá tiền tệ**
và định giá được 60 danh mục. Xác lập rằng **độ lệch khác nhau có hệ thống theo đồng tiền và được đền bù**.

**Asano, Cai & Sakemoto (2025), *JBF* 178:107508** lật một phần quan điểm chuẩn: biến động FX cao
làm tăng lợi suất carry **chỉ khi bất định (ambiguity) FX cũng cao**, vì nhà đầu tư đóng băng và việc
tháo vị thế thường thấy không xảy ra.

> **Hệ quả cho hệ thống.** Li–Sarno–Zinna là chỗ dựa học thuật cho semivariance RS+/RS− đã có trong
> `rv_adv.csv`: nó nói độ lệch không chỉ là chi tiết kỹ thuật mà là **rủi ro được định giá và khác nhau
> theo cặp**. Đây là lập luận để dùng SHAR/STHARQ riêng từng cặp thay vì HAR đối xứng gộp.

### B5. Liên kết chéo giữa các cặp

**Rubaszek, Szafranek & Uddin (2025), *JIMF* 157:103398.**
"Intraday volatility connectedness on the forex market: the role of uncertainty."

Khớp gần nhất với đúng bộ cặp của mình: **hơn 460.000 báo giá 5 phút, 1/2018–2/2024, trên
USD/EUR, USD/JPY, USD/AUD, USD/CAD, USD/GBP** (5 trong 6 cặp; thiếu CHF). Lan toả Diebold–Yılmaz
với TVP-VAR; liên kết nội ngày tổng thể được dẫn dắt bởi biến động ngụ ý của cổ phiếu.

**Jia, Liu, Wu & Yan (2024), *JBF* 169:107313** dùng LASSO trên biến từ tin tức cộng chiết khấu kỳ hạn
và tìm thấy **khả năng dự báo chéo giữa các cặp** đáng kể, có lãi sau chi phí giao dịch.

> **Hệ quả cho hệ thống.** Cả hai bài nói rằng thông tin của cặp này dự báo được cặp kia. Hiện tại
> tầng 2 của mình chạy 6 mô hình một biến độc lập. **Thêm RV trung bình của 5 cặp còn lại làm một biến
> giải thích trong HAR (một hệ số duy nhất, không phải 5)** là một mở rộng rẻ, có cơ sở, và có thể
> kiểm định bằng DM ngay trên khung hiện có.

---

## C. Mẫu hình dạng và mẫu ký hiệu — nhánh SAX

### C1. Ai đã làm gần giống mình

**Kania, Juszczuk & Kozák (2019), *Vietnam Journal of Computer Science* 6(3):343–362.**
"Enhanced Symbolic Description in Analyzing Patterns and Volatility on the Forex Market."

Bản tương đồng gần nhất đã xuất bản. Chuyển chuỗi FX thành bảng chữ cái ký hiệu dựa trên dao động,
định nghĩa độ tương tự kiểu văn bản để truy hồi mẫu lịch sử tương đương, và khẳng định tường minh
rằng **mẫu là phổ quát giữa các cặp tiền tệ** — tức động cơ của họ chính là logic leave-one-pair-out
của mình. Kiểm chứng trên ~10 năm dữ liệu, 10 cặp.

**Cartwright, Crane & Ruskin (2019), ICCS, LNCS 11540:736–748** áp VALMOD (hậu duệ matrix profile,
motif độ dài thay đổi) lên **chuỗi GBP/USD ngày**, và lập luận rằng điểm yếu của lĩnh vực là
*ý nghĩa thống kê và diễn giải* chứ không phải phát hiện — đúng chỗ mà thước đo lift của mình nằm.

**Cartwright, Crane & Ruskin (2022), *Forecasting* 4(1):219–237** — "SLIM" — là bài
"mẫu ký hiệu + biến động" tốt nhất cho mình: hợp nhất **SAX, MDL và Matrix Profile**, cho phép
hai vế của một cặp motif có độ dài khác nhau, với động cơ là biểu diễn biến động cục bộ tốt hơn.

**Nikolaou (2024), *Journal of Finance and Data Science* 10:100132** — CPC-SAX — khai phá mẫu đồ thị
dựa trên SAX kết hợp phân loại **đa nhãn** dựa trên thực thể, tức một cửa sổ có thể mang nhiều mẫu
cùng lúc. (Đã xác minh tồn tại qua Crossref/OpenAlex; ScienceDirect chặn truy xuất tự động nên chưa
đọc được kết luận số — cần tự tải PDF.)

### C2. Điểm yếu lớn nhất của nhánh SAX: kiểm định bội

Đây là phần phải viết vào luận văn, vì nếu không viết thì phản biện sẽ hỏi.

**Hämäläinen & Webb (2019), *DMKD* 33(2):325–377; arXiv:1709.03904.**
"A tutorial on statistically sound pattern discovery."

Bài tham chiếu cho đúng vấn đề của mình: coi **lift và leverage là đại lượng thống kê**, chỉ ra
**vì sao lift thô xếp hạng cao các mẫu giả**, và dành hẳn một mục cho hiệu chỉnh đa giả thuyết cùng
các kỹ thuật tăng lực kiểm định riêng cho khai phá mẫu (kiểu Tarone, holdout, giá trị tới hạn phân tầng).
Nếu đang xếp hạng theo lift mà không hiệu chỉnh thì đây là bài nói cho mình biết đang khẳng định quá tay.

**Jenkins, Walzer-Goldfeld & Riondato (2022), *DMKD* 36:1575–1599** — SPEck — khai phá mẫu tuần tự
có ý nghĩa thống kê, dùng **lấy mẫu lại Westfall–Young để kiểm soát FWER ở mức δ**, với các mô hình
null khác nhau bảo toàn các tính chất khác nhau của dữ liệu quan sát. Với FX thì mô hình null cần bảo
toàn phân phối biên của từng ký hiệu và tự tương quan.

Phía tài chính đã giải quyết cùng câu hỏi này bằng công cụ khác:
- **Sermpinis, Hassanniakalager, Stasinakis & Psaradellis (2021), *JIFMIM***; arXiv:1811.06766 —
  kiểm tra **hơn 21.000 quy tắc kỹ thuật** bằng thủ tục **FDR rời rạc** thích ứng, mạnh hơn FDR chuẩn.
- **Hsu, Taylor & Wang (2016), *Journal of International Economics* 102:188–208** — chuẩn mực cho FX:
  hơn 21.000 quy tắc, 30 đồng tiền, tới 45 năm dữ liệu ngày, dùng **kiểm định SPA từng bước** để nhận
  diện nhiều quy tắc có ý nghĩa mà không thổi phồng dương tính giả.
- **Bailey & López de Prado (2014), *JPM* 40(5):94–107** — Deflated Sharpe Ratio: công thức đóng để
  chiết khấu Sharpe quan sát theo **số lần thử**, độ lệch, độ nhọn và độ dài mẫu.

### C3. Bài phản biện mạnh nhất — phải đối chiếu

**Hutchinson, Kyziropoulos, O'Brien, O'Reilly & Sharma (2022), *Research in International Business
and Finance* 61:101779.** "Technical trading rule profitability in currencies: It's all about momentum."

Đây là **giả thuyết null mà kết quả mẫu hình của mình phải vượt qua.** Khi nhân bản các quy tắc kỹ thuật
tiền tệ, Sharpe danh mục trung bình rơi từ **0,66 trong mẫu xuống 0,06 ngoài mẫu**, lợi suất không sống
sót qua chi phí giao dịch vừa phải, và **toàn bộ lợi suất bất thường bị hấp thụ bởi động lượng chuỗi thời
gian (TSMOM)** như một nhân tố chung duy nhất.

> **Hệ quả cho hệ thống.** Ba mẫu ký hiệu của HuyH mà mình đã nhân bản độc lập (3/3, lift cao hơn)
> hiện đang được dùng ở tầng 6 làm **lời giải thích**, không phải làm tín hiệu vào lệnh. Đó là lựa chọn
> đúng và bài Hutchinson là lý do. Nhưng vẫn cần làm hai việc: **(1) hồi quy lợi suất theo mẫu lên TSMOM**
> để cho thấy mẫu có gì ngoài động lượng không; **(2) báo cáo số mẫu đã thử và hiệu chỉnh lift bằng
> Westfall–Young hoặc FDR rời rạc.** Leave-one-pair-out kiểm soát overfit theo cặp nhưng **không** kiểm soát
> số lượng mẫu đã đánh giá — đây là lỗ hổng còn lại của nhánh SAX.

### C4. Hiện đại hơn SAX

- **Elsworth & Güttel (2020), *DMKD* 34(4):1175–1200** — **ABBA**: nén chuỗi bằng đường gấp khúc thích ứng
  rồi phân cụm bộ (độ dài, độ tăng), nên ký hiệu mã hoá **cả thời lượng lẫn độ dốc**, thay vì mức trung bình
  bề rộng cố định như SAX. Bảo toàn xu hướng và hình lên/xuống tốt hơn nhiều — quan trọng khi mẫu là hình dạng
  trên nến ngày.
- **Yeh, Zhu, Ulanova, ... Keogh (2016), ICDM:1317–1322** — **Matrix Profile**: phép nối tương tự toàn cặp
  chính xác, gần như không tham số, đồng thời cho ra motif, discord và ứng viên shapelet. Đây là thứ phản biện
  sẽ hỏi "sao không dùng".
- **Van Wesenbeeck, Yurtman, Meert & Blockeel (2026), *DMKD* 40(1) art. 5** — **PROM + TSMD-Bench**:
  câu trả lời hiện tại cho "làm sao biết motif của tôi có tốt không".
- **Kim, Lee, Jeon, Jin & Ko (2025), CIKM'25; arXiv:2509.15040** — khung dựa trên shapelet cho dự báo hướng,
  bất biến với co giãn biên độ và méo thời gian, **xếp nhất hoặc nhì ở 11/12 tổ hợp thước đo–bộ dữ liệu**.
- **Jiang, Kelly & Xiu (2023), *Journal of Finance* 78(6):3193–3249** — "(Re-)Imag(in)ing Price Trends":
  hậu duệ hiện đại của Lo–Mamaysky–Wang, dựng lịch sử giá thành **ảnh** và để CNN tự tìm mẫu dự báo.
  Mẫu học được **không phụ thuộc bối cảnh và chuyển giao được qua các chân trời và thị trường quốc tế** —
  đúng là bản ML của khẳng định leave-one-pair-out của mình.

### C5. Khoảng trống

**Không có bài đã xác minh nào làm đúng thứ mình đang làm.** Symbolic-FX có (Kania 2019) nhưng dùng độ
tương tự văn bản chứ không khai phá mẫu tuần tự với lift; motif FX có (Cartwright 2019) nhưng một cặp và
không có thống kê chất lượng quy tắc; khai phá mẫu có ý nghĩa thống kê dựa trên lift có (Hämäläinen–Webb,
SPEck) nhưng không trên FX. Tổ hợp **SAX → mẫu tuần tự → lift → holdout chéo cặp** là đất trống.

---

## D. Mẫu trong hiệu năng mô hình — trục quan trọng nhất

### D1. Bài phải trích: chế độ là thứ phân tách các mô hình

**Kilic (2025), Federal Reserve Board FEDS 2025-061.**
"Linear and Nonlinear Econometric Models Against Machine Learning Models: Realized Volatility Prediction."
DOI 10.17016/FEDS.2025.061

Bài quan trọng nhất trong phần này, và nó nói đúng về **THAR và STHAR** — chính hai mô hình đang có
trong `src/volfc.py`. So sánh ARFIMA/HAR và các biến thể HAR chuyển chế độ (THAR, STHAR, MSHAR) với
XGBoost, DNN, BRNN, LSTM, LSTM-A, GRU trên RV tần suất cao của S&P 500, dự báo cuốn chiếu từ 2006.
**Mô hình chuyển chế độ — THAR và STHAR nói riêng — thắng nhất quán cả ML lẫn mô hình tuyến tính,
và lợi thế tập trung trong giai đoạn thị trường cực đoan.**
Thước đo: MSPE và QLIKE, Model Confidence Set, cộng độ chính xác VaR và hữu dụng hiện thực hoá.

> **Hệ quả cho hệ thống.** Đây là bài hợp thức hoá trực tiếp nhất cho lựa chọn STHARQ của tầng 2,
> và nó dùng **chính bộ thước đo** mình đang dùng. Phải trích. Nó cũng giải thích tại sao ML không
> thắng trong thí nghiệm của mình mà không cần viện lý do kỹ thuật.

### D2. Đánh giá trung bình gộp che giấu mọi thứ

**Chagas, Bento, Aquino, Buzelin, Meira & Valle (2026), arXiv:2608.01599.**
"Latent-Regime Bias Auditing for Volatility Forecasting."

Khung kiểm toán không phụ thuộc mô hình: phân cụm cửa sổ trạng thái thị trường thành chế độ tiềm ẩn
trên dữ liệu huấn luyện, rồi đánh giá dự báo ngoài mẫu **bên trong từng chế độ**. Kết quả cốt lõi:
**mô hình có độ chính xác tổng thể tốt vẫn có thiên lệch theo chế độ đáng kể và dự báo thiếu nghiêm
trọng ở đuôi; thước đo gộp che giấu các thất bại có điều kiện.**

**Rossi (2021), *Journal of Economic Literature* 59(4):1135–1190** là khung khái niệm cho toàn bộ vấn đề.
Câu đáng trích: **điểm gãy trong tham số mô hình không phải điều kiện cần cũng không phải điều kiện đủ
để tạo ra biến thiên theo thời gian trong hiệu năng dự báo** — nên kiểm định tính ổn định tham số không
thay thế được việc đánh giá trực tiếp hiệu năng dự báo cục bộ.

**Giacomini & Rossi (2009), *Review of Economic Studies* 76(2):669–705** cho kiểm định hình thức:
định nghĩa **"forecast breakdown"** là khi tổn thất ngoài mẫu tệ hơn tổn thất trong mẫu một cách có ý
nghĩa, và đưa ra kiểm định vừa phát hiện điểm gãy quá khứ vừa dự báo điểm gãy tương lai.

> **Hệ quả cho hệ thống — đây là việc có giá trị cao nhất còn lại.** Hiện `run_final_eval2.py` báo cáo
> QLIKE trung bình trên tập test. Chỉ cần **phân tầng lại theo ngũ phân vị biến động** (giống hệt cách
> arXiv 2602.03903 làm cho VaR) là ra một bảng mới cho luận văn, không cần huấn luyện lại gì cả.
> Và kiểm định Giacomini–Rossi chạy được ngay trên chuỗi tổn thất QLIKE đã có.

### D3. HAR có hệ số thay đổi theo thời gian

**Xu, Aschakulporn & Zhang (2025), *Journal of Forecasting* 44(5):1638–1657** — TVP-HAR ước lượng bằng
làm trơn nhân. TVP-HAR thắng HAR hệ số hằng cả về khớp lẫn độ chính xác dự báo. Đây là bằng chứng
"hệ số HAR không phải hằng số" sạch nhất, và nhẹ hơn chuyển chế độ tường minh.

**Blake, Gandhi & Jakkula (2025), arXiv:2510.03236** đưa ra thủ thuật rất đáng mượn: phân đoạn mẫu bằng
kiểm định Mood, **trích hệ số HAR trên từng đoạn, rồi phân cụm chính các vector hệ số đó bằng GMM Bayes**.
Phương pháp phân cụm hệ số này thắng ở cả ba giai đoạn (trước COVID, COVID, sau COVID).

**Luo, Klein, Ji & Hou (2022), *International Journal of Forecasting* 38(1):51–73** — HAR với chuyển chế độ
Markov ẩn **vô hạn** (số chế độ không biết và không chặn trên, xử lý đồng thời chuyển chế độ và điểm gãy
cấu trúc), có bước nhảy, đòn bẩy và đầu cơ. Thước đo: MZ-R², MAFE, Model Confidence Set.

**Ding, Kambouroudis & McMillan (2025), *International Review of Economics & Finance* 101:104171** —
lưu ý quan trọng: chuyển chế độ cải thiện rõ ở **chân trời tuần và tháng, nhưng mơ hồ ở chân trời ngày**.
Chuyển đột ngột (Markov) thắng chuyển trơn.

**Fang & Ślepaczuk (2026), arXiv:2606.09478** — **HARQ tăng cường chế độ** với lọc chế độ bằng
Markov-switching GJR-GARCH, kiểm định cuốn chiếu. HARQ tăng cường chế độ thắng HARQ cơ sở nhất quán;
và khả năng dự báo hạ nguồn "yếu, phụ thuộc trạng thái, và tập trung chủ yếu ở chế độ biến động thấp".

### D4. Cửa sổ cuốn chiếu hay mở rộng

**Feng, Zhang & Wang (2024), *Journal of Forecasting* 43(3):567–582.**
"Out-of-Sample Volatility Prediction: Rolling Window, Expanding Window, or Both?"

Thay vì cam kết một loại cửa sổ, chọn động loại nào vừa chạy tốt gần đây. Ghi nhận
**"động lượng của khả năng dự báo"**: lợi thế của cửa sổ tốt hơn duy trì theo thời gian, và cửa sổ nào
tốt hơn thì thay đổi theo điều kiện thị trường. Bản lai thắng đáng kể cả hai loại riêng lẻ.

**Chung, Espinoza & Quispe (2025), *JRFM* 18(9):494** dùng thuật toán ICSS sửa đổi để phát hiện điểm gãy
phương sai rồi so GARCH dưới cửa sổ mở rộng / cuốn chiếu 25% / 50% / ước lượng điều chỉnh điểm gãy.
Kết quả then chốt: **bỏ qua điểm gãy làm phóng đại tính dai của biến động** (α+β gần 1,0 trên mẫu đầy đủ);
ước lượng riêng theo chế độ cắt giảm tính dai đáng kể.

> **Hệ quả cho hệ thống.** `forecast_series()` hiện dùng cửa sổ mở rộng với `min_train=500`.
> Bài Feng et al. cho một lý do có tài liệu để **báo cáo cả hai và chọn theo tập kiểm định**, chứ không
> phải chọn tuỳ ý. Rẻ, và làm mạnh phần phương pháp.

### D5. ML có thắng không — và rò rỉ do chọn lọc

**Branco, Rubesam & Zevallos (2024), *Journal of Empirical Finance* 78:101524.**
"Forecasting Realized Volatility: Does Anything Beat Linear Models?"
Mười chỉ số toàn cầu, 2000–2021. **Không có bằng chứng thống kê rằng ML phi tuyến vượt mô hình tuyến tính
nói chung**; và với danh mục định thời biến động theo tháng, mô hình đơn giản hơn *không có* biến bổ sung
lại tốt hơn về mặt kinh tế. Đánh giá bằng Model Confidence Set.

**Brini (2026), arXiv:2607.05291** (đã có trong `TAI_LIEU_LIEN_QUAN.md`) bổ sung một điểm tinh tế rất đáng
mượn: **hiệu chỉnh Mincer–Zarnowitz cho thấy phần lớn lợi ích biểu kiến ở chân trời ngắn là do dự báo được
CHIA TỶ LỆ tốt hơn, chứ không phải động lực tốt hơn.** Đây là lý lẽ để luôn kiểm tra hiệu chuẩn trước khi
tuyên bố thắng.

**Arian, Norouzi Mobarekeh & Seco (2024), *Knowledge-Based Systems* 305:112477** dựng môi trường tổng hợp
có kiểm soát (Heston, Merton jump diffusion, drift-burst) *cố ý nhúng* phi dừng, tự tương quan và chuyển chế độ,
rồi so K-Fold, Purged K-Fold, Walk-Forward và **Combinatorial Purged Cross-Validation**. CPCV thắng trên
**Probability of Backtest Overfitting và Deflated Sharpe Ratio**; Walk-Forward **yếu rõ rệt** trong việc ngăn
phát hiện giả.

### D6. Bài FX gần nhất về mặt giao thức đánh giá

**Alexandridis, Panopoulou & Souropanis (2024), *JIFMIM* 97:102067.**
"Forecasting Exchange Rate Volatility: An Amalgamation Approach." Bảy đồng tiền chính so với USD;
so sánh mô hình tuyến tính, ML, giảm chiều và kết hợp dự báo, thêm **phân rã wavelet của biến động theo
tần số để xét hiệu ứng thời điểm trong hiệu năng mô hình** — tức mô hình nào chạy tốt khi nào.
**Dự báo kết hợp thống trị các mô hình đơn lẻ. Đánh giá bằng Model Confidence Set.**

**Kearney, Shang & Zhao (arXiv:2311.18477)** — USD/EUR, USD/GBP, USD/JPY, dữ liệu 5 phút, 1.275 ngày.
Thước đo: MSFE, QLIKE, Diebold–Mariano, Model Confidence Set, và kiểm định hậu nghiệm VaR về tính không
thiên lệch và tính độc lập. Dùng làm mẫu cho giao thức đánh giá FX.

---

## Tổng hợp — sáu việc nên làm, xếp theo tỷ lệ giá trị trên chi phí

| # | Việc | Chi phí | Cơ sở | Tác động tới luận văn |
|---|---|---|---|---|
| 1 | **Phân tầng lại QLIKE/CRPS theo ngũ phân vị biến động** trên tập test đã có | Rất thấp — không huấn luyện lại | Chagas 2026; Rossi 2021 | Cao — ra bảng mới, và cho thấy mình biết trung bình gộp che giấu gì |
| 2 | **Khử chu kỳ nội ngày/nội tuần trong RV** trước khi dựng HAR | Thấp — một hệ số nhân theo (giờ, thứ) | Dumitru 2025; Boudt 2011; Dette 2023 | Cao — cải thiện điểm thật, có tài liệu mạnh |
| 3 | **Co ngót hệ số HAR về trung bình 6 cặp**, chọn mức co ngót trên tập kiểm định; chạy kiểm định poolability | Trung bình | Pesaran–Pick–Timmermann 2026 | Cao — biến 6 mô hình rời thành một bài toán panel |
| 4 | **Hồi quy lợi suất mẫu SAX lên TSMOM + hiệu chỉnh lift bằng Westfall–Young/FDR** | Trung bình | Hutchinson 2022; Hämäläinen–Webb 2019; SPEck 2022 | Cao — bịt lỗ hổng phản biện lớn nhất của nhánh ký hiệu |
| 5 | **Thêm RV chéo cặp (một hệ số) vào HAR**, kiểm định bằng DM | Thấp | Rubaszek 2025; Jia 2024 | Trung bình — có thể thắng nhỏ, và lập luận sạch |
| 6 | **Lịch sự kiện S2 với co ngót** (không phải trăm biến giả) | Trung bình | Lee & Wang 2025; Martins & Lopes 2024 | Trung bình — đã có trong kế hoạch, nay có cơ sở chọn biến |

### Ba điều nên nói thẳng trong luận văn

1. **Mẫu ngày-trong-tuần trong FX gần như không có bằng chứng hiện đại.** Nếu mình đo và không tìm ra gì,
   đó là kết quả hợp lệ khớp với tài liệu, không phải thất bại.
2. **"JPY là đồng trú ẩn" không phải hệ số ổn định** (Park & Fang 2025: JPY yếu đi trong cú sốc Ukraine 2022).
   Nên đưa thông tin này vào tầng chế độ, không đưa vào tầng 2 dưới dạng hằng số theo cặp.
3. **Leave-one-pair-out không kiểm soát kiểm định bội.** Nó kiểm soát overfit theo cặp, không kiểm soát
   số mẫu đã thử. Phải nói rõ và bổ sung một hiệu chỉnh.

### Ba khoảng trống mà tài liệu không lấp — đất trống cho luận văn

- **Đánh giá HAR phân tầng theo chế độ trên một panel các cặp FX** — không tìm được bài nào làm.
- **Leave-one-pair-out tường minh trong FX** — không xuất hiện trong tài liệu đã xuất bản; Pesaran et al.
  (kiểm định poolability) và Liu et al. (chuyển giao leave-one-asset-out) là hai thứ gần nhất, phải tự phỏng theo.
- **Tổ hợp SAX → mẫu tuần tự → lift → holdout chéo cặp** — từng mảnh đều có, tổ hợp thì chưa.

*(Ba khoảng trống này bổ sung cho ba khoảng trống đã ghi trong `TAI_LIEU_LIEN_QUAN.md`:
chiết khấu rủi ro phá sản cấp danh mục, bảng đa chân trời trên phiếu quyết định, và đo trượt giá từ M1.)*
