import { useEffect, useState } from "react";

const TEN_CAP = {
  EURUSD: "EUR/USD", GBPUSD: "GBP/USD", USDJPY: "USD/JPY",
  AUDUSD: "AUD/USD", USDCAD: "USD/CAD", USDCHF: "USD/CHF",
};

function fmtPct(x, digits = 2) {
  return `${(x * 100).toFixed(digits)}%`;
}

export default function Home() {
  const [meta, setMeta] = useState(null);
  const [pair, setPair] = useState("EURUSD");
  const [date, setDate] = useState("");
  const [soViThe, setSoViThe] = useState(1);
  const [dd, setDd] = useState(0);
  const [stopSigma, setStopSigma] = useState(2.0);
  const [von, setVon] = useState(10000);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("/api/meta")
      .then((r) => r.json())
      .then((m) => {
        setMeta(m);
        setDate(m.valid_tu);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (meta && date) tinhPhieu();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta]);

  async function tinhPhieu(e) {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const qs = new URLSearchParams({
        pair, date, so_vi_the: String(soViThe), dd: String(dd / 100),
        stop_sigma: String(stopSigma), von: String(von), muc: "0.80,0.95",
      });
      const r = await fetch(`/api/decision?${qs.toString()}`);
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "lỗi không rõ");
      setResult(j);
    } catch (err) {
      setError(String(err.message || err));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  const range = meta && meta.ranges[pair];

  return (
    <div className="wrap">
      <header>
        <h1>FX-DSS — Phiếu quyết định</h1>
        <p className="sub">
          Hệ thống hỗ trợ quyết định giao dịch FX — tầng 2 (biến động) → tầng 4
          (định cỡ vị thế) → tầng 6 (phiếu quyết định) → tầng 6b (giữ/đóng).
          Dữ liệu lịch sử thật, tham số huấn luyện trên đoạn trước{" "}
          {meta ? meta.valid_tu : "…"}.
        </p>
      </header>

      <form className="controls" onSubmit={tinhPhieu}>
        <label>
          Cặp tiền
          <select value={pair} onChange={(e) => setPair(e.target.value)}>
            {(meta ? meta.pairs : ["EURUSD"]).map((p) => (
              <option key={p} value={p}>{TEN_CAP[p] || p}</option>
            ))}
          </select>
        </label>

        <label>
          Ngày
          <input
            type="date"
            value={date}
            min={range ? range[0] : undefined}
            max={range ? range[1] : undefined}
            onChange={(e) => setDate(e.target.value)}
          />
        </label>

        <label>
          Số vị thế mở cùng lúc
          <input type="number" min={1} max={6} value={soViThe}
                 onChange={(e) => setSoViThe(e.target.value)} />
        </label>

        <label>
          Sụt giảm hiện tại (%)
          <input type="number" min={0} max={90} step={1} value={dd}
                 onChange={(e) => setDd(e.target.value)} />
        </label>

        <label>
          Stop (σ)
          <input type="number" min={0.5} max={4} step={0.1} value={stopSigma}
                 onChange={(e) => setStopSigma(e.target.value)} />
        </label>

        <label>
          Vốn
          <input type="number" min={100} step={100} value={von}
                 onChange={(e) => setVon(e.target.value)} />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Đang tính…" : "Tính phiếu"}
        </button>
      </form>

      {error && <div className="err">Lỗi: {error}</div>}

      {result && <PhieuCard r={result} />}

      <footer>
        Đồ án tốt nghiệp — hệ thống hỗ trợ quyết định giao dịch FX. Mọi tham số
        huấn luyện trên đoạn trước {meta ? meta.valid_tu : "…"} (đúng phân
        chia chính thức, xem <code>src/split.py</code>); carry (lợi thế
        Kelly) là trung vị cửa sổ mở rộng tính đến trước ngày được chọn.
      </footer>
    </div>
  );
}

function PhieuCard({ r }) {
  const kn = r.khuyen_nghi_giu_dong;
  return (
    <div className="card">
      <div className="card-head">
        <h2>PHIẾU QUYẾT ĐỊNH — {TEN_CAP[r.pair] || r.pair}</h2>
        <span className="muted">
          ngày {r.ngay}{r.ngay !== undefined ? "" : ""} · giá tham chiếu {r.gia.toFixed(5)} · vốn {r.von.toLocaleString()}
        </span>
      </div>

      <section>
        <h3>Hành động</h3>
        <div className="big">
          {r.don_bay.toFixed(2)}×
          <span className="big-sub"> → đặt {Math.round(r.von_dat).toLocaleString()}</span>
        </div>
      </section>

      <section>
        <h3>Vì sao — ràng buộc đang siết: {r.rang_buoc.toUpperCase()}</h3>
        <table className="kv">
          <tbody>
            <tr><td>Kelly</td><td>{r.kelly.toFixed(2)}×</td>
                <td>Trần rủi ro</td><td>{r.tran_rui_ro.toFixed(2)}×</td></tr>
            <tr><td>k biến động</td><td>{r.k_vol.toFixed(2)} (mức {r.muc_bien_dong})</td>
                <td>k sụt giảm</td><td>{r.k_dd.toFixed(2)} ({r.sut_giam > 0 ? `−${fmtPct(r.sut_giam, 0)}` : "ở đỉnh vốn"})</td></tr>
            <tr><td>k danh mục</td><td>{r.k_dm.toFixed(2)} ({r.so_vi_the} vị thế)</td>
                <td>ρ hiệu dụng</td><td>{r.rho_hieu_dung.toFixed(2)}</td></tr>
          </tbody>
        </table>
        {r.che_do_cang_thang && (
          <p className="warn">⚠ vùng căng thẳng — tương quan đo được cao hơn mức nền</p>
        )}
      </section>

      <section>
        <h3>Rủi ro</h3>
        <p>
          Stop {r.stop_sigma.toFixed(1)}σ tại {r.stop_gia.toFixed(5)} ({r.stop_pip.toFixed(0)} pip)
        </p>
        <p>Xác suất chạm stop trong 1 phiên: <b>{fmtPct(r.p_cham_stop)}</b> (±1,4%)</p>
        <ul className="khoang">
          {Object.entries(r.khoang).sort().map(([muc, v]) => (
            <li key={muc}>
              Khoảng giá {fmtPct(Number(muc), 0)}: {v[0].toFixed(5)} – {v[1].toFixed(5)}{" "}
              ({(((v[1] - v[0]) / (r.pair && r.pair.includes("JPY") ? 0.01 : 0.0001))).toFixed(0)} pip)
            </li>
          ))}
        </ul>
        <p>Nếu giữ lệnh lâu hơn, xác suất chạm stop tăng nhanh:</p>
        <div className="tamhan">
          {r.bang_tam_han.map(([h, v]) => (
            <span key={h}>{h} phiên: {fmtPct(v, 0)}</span>
          ))}
        </div>
      </section>

      <section>
        <h3>Tầng 6b — giữ hay đóng (quy hoạch động, không đòn bẩy)</h3>
        <p>
          Lúc vừa vào lệnh ({r.stop_sigma.toFixed(1)}σ tới stop): khuyến nghị{" "}
          <b>{kn.giu_luc_vao ? "GIỮ" : "ĐÓNG NGAY"}</b>
        </p>
        {kn.bien_gioi_sigma === null ? (
          <p>Carry hiện không đủ bù chi phí thoát ở chế độ này — đóng ngay dù còn cách stop bao xa.</p>
        ) : (
          <p>Biên giới đóng lệnh: đóng nếu còn cách stop dưới {kn.bien_gioi_sigma.toFixed(2)}σ (còn xa hơn thì giữ).</p>
        )}
        <p className="muted">carry giả định {(kn.carry_ngay * 1e4).toFixed(2)} bp/phiên, chế độ biến động {r.che_do_6b + 1}/3</p>
        {r.don_bay <= 1e-9 && (
          <p className="warn">⚠ đòn bẩy khuyến nghị đang là 0× — mục này chỉ áp dụng nếu bạn đang giữ vị thế từ trước.</p>
        )}
      </section>

      {r.luat_ky_hieu && (
        <section>
          <h3>Luật ký hiệu (nhánh khai phá mẫu)</h3>
          <p>{r.luat_ky_hieu}</p>
        </section>
      )}

      {r.canh_bao_dong_luong && (
        <section>
          <h3>⚠ Cảnh báo xu hướng (vùng căng thẳng)</h3>
          <p>
            Nếu đang định vào lệnh theo tín hiệu xu hướng gần đây (mua vì giá đang lên /
            bán vì giá đang xuống): trong vùng biến động này, xu hướng có xu hướng LỖ có
            ý nghĩa thống kê (Sharpe −0,62, p=0,001), không trung tính. Hệ thống không tự
            chặn lệnh — đây chỉ là cảnh báo dựa trên dữ liệu.
          </p>
        </section>
      )}

      <section>
        <h3>Độ tin cậy của chính các số trên</h3>
        <p>Khoảng: conformal phân tầng theo chế độ biến động, lệch ≤1,2%; mẫu hiệu chuẩn chế độ này: {r.n_mau_che_do.toLocaleString()} phiên.</p>
        <p className="warn">⚠ khi đang lỗ, độ phủ thực tế thấp hơn ghi ~1%.</p>
      </section>

      <section>
        <h3>Điều kiện để con số đòn bẩy còn đúng</h3>
        <p>Phải định cỡ lại mỗi phiên. Định cỡ mỗi tháng thì xác suất phá sản thật là 1,15% thay vì 0,41% (ngân sách 1%).</p>
        <p>Đã khai {r.so_vi_the} vị thế. Khai thiếu là sai theo cấp số: 6 lệnh cùng hướng USD ở trần đầy đủ → phá sản 73,6%.</p>
      </section>
    </div>
  );
}
