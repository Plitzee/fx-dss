"""TANG 2b — CARRY CO PHAI LA TIN HIEU HUONG DI CON SONG KHONG?

Momentum da chet (xem momentum_decay.py). Carry la ung vien con lai duy nhat
co bang chung ben vung trong tai lieu. File nay kiem dinh no tren chinh du
lieu cua he thong, theo dung khuon kho da dung cho momentum: mau dai FRED,
chia giai doan, do ca truoc va sau chi phi.

Quy uoc: lam viec o gia 'px' = USD tren mot don vi ngoai te. Nam giu ngoai te
do an chenh lech lai suat carry_px:
    cap dang XXXUSD  ->  carry_px = +carry(cap)
    cap dang USDXXX  ->  carry_px = -carry(cap)

Loi suat tong mot phien khi nam giu ngoai te = dlog(px) + carry_px/100/252.

Ba phep thu:
  1. Carry theo chuoi thoi gian — mua dong nao co carry duong
  2. Carry cat ngang        — mua 2 dong carry cao nhat, ban 2 dong thap nhat
  3. Carry co dieu kien bien dong — tai lieu noi carry sap khi bien dong vot
"""
import os
import numpy as np
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(DIR), "data")
FRED = os.path.join(DATA, "fred")

META = {"DEXUSEU": ("EURUSD", +1), "DEXUSUK": ("GBPUSD", +1),
        "DEXUSAL": ("AUDUSD", +1), "DEXCAUS": ("USDCAD", -1),
        "DEXJPUS": ("USDJPY", -1), "DEXSZUS": ("USDCHF", -1)}
INVERT = {"DEXCAUS", "DEXJPUS", "DEXSZUS"}
PERIODS = [(1994, 2000), (2001, 2009), (2010, 2017), (2018, 2025)]


def load_px(code):
    d = pd.read_csv(os.path.join(FRED, f"{code}.csv")).rename(
        columns={"observation_date": "DATE"})
    d["DATE"] = pd.to_datetime(d.DATE, errors="coerce")
    d[code] = pd.to_numeric(d[code], errors="coerce")
    d = d.dropna().sort_values("DATE")
    px = 1.0 / d[code] if code in INVERT else d[code]
    o = pd.DataFrame({"DATE": d.DATE.values, "px": px.values})
    o["r"] = np.log(o.px / o.px.shift(1))
    return o.dropna().reset_index(drop=True)


def panel():
    """Ghep loi suat ngay voi carry thang (carry da biet tu dau thang -> nhan qua)."""
    car = pd.read_csv(os.path.join(DATA, "carry.csv"))
    car["DATE"] = pd.to_datetime(car.DATE)
    out = []
    for code, (pair, sgn) in META.items():
        p = load_px(code)
        c = car[car.pair == pair][["DATE", "carry"]].copy()
        c["carry_px"] = sgn * c.carry
        c = c.sort_values("DATE")
        m = pd.merge_asof(p, c[["DATE", "carry_px"]], on="DATE")   # gia tri thang HIEN HANH
        m["carry_px"] = m.carry_px.shift(1)                        # chi dung thong tin toi t-1
        m["cur"] = pair
        out.append(m.dropna())
    return pd.concat(out, ignore_index=True)


def stats(x, name=""):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 100 or x.std() == 0:
        return dict(n=len(x), sharpe=np.nan, ret=np.nan, dd=np.nan, skew=np.nan)
    eq = np.cumsum(x)
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    return dict(n=len(x), sharpe=float(x.mean() / x.std() * np.sqrt(252)),
                ret=float(x.mean() * 252), dd=dd,
                skew=float(((x - x.mean()) ** 3).mean() / x.std() ** 3))


def ts_carry(pan, cost_pip=0.0):
    """Mot chuoi loi suat cho moi dong, roi trung binh deu."""
    rows = {}
    for cur, g in pan.groupby("cur"):
        g = g.sort_values("DATE")
        pos = np.sign(g.carry_px.values)
        tot = pos * (g.r.values + g.carry_px.values / 100.0 / 252.0)
        if cost_pip:
            flip = np.r_[0, np.abs(np.diff(pos))] / 2.0
            tot = tot - flip * cost_pip * 1e-4 / g.px.values
        rows[cur] = pd.Series(tot, index=g.DATE.values)
    return pd.DataFrame(rows)


def xs_carry(pan, k=2, cost_pip=0.0):
    """Cat ngang: mua k dong carry cao nhat, ban k dong thap nhat."""
    w = pan.pivot_table(index="DATE", columns="cur", values="carry_px")
    r = pan.pivot_table(index="DATE", columns="cur", values="r")
    c = pan.pivot_table(index="DATE", columns="cur", values="carry_px") / 100.0 / 252.0
    tot = r + c
    rk = w.rank(axis=1, ascending=False)
    nn = w.notna().sum(axis=1)
    pos = pd.DataFrame(0.0, index=w.index, columns=w.columns)
    pos = pos.mask(rk.le(k), 1.0 / k).mask(rk.gt(nn.values[:, None] - k), -1.0 / k)
    pos = pos.where(w.notna(), 0.0).shift(1).fillna(0.0)
    out = (pos * tot).sum(axis=1)
    if cost_pip:
        turn = pos.diff().abs().sum(axis=1).fillna(0.0)
        out = out - turn * cost_pip * 1e-4 / 1.0
    return out


def main():
    pan = panel()
    print("=" * 88)
    print("CARRY CO CON LA TIN HIEU KHONG? — mau FRED, carry tu 1994")
    print("=" * 88)
    print(f"{len(pan):,} quan sat ngay-đồng, {pan.DATE.min().date()} → {pan.DATE.max().date()}")

    ts = ts_carry(pan); ts_c = ts_carry(pan, cost_pip=1.0)
    xs = xs_carry(pan); xs_c = xs_carry(pan, cost_pip=1.0)

    print("\n" + "=" * 88)
    print("BANG 1 — TOAN MAU")
    print("=" * 88)
    print(f"{'chiến lược':<28}{'Sharpe':>9}{'lợi suất/năm':>15}{'sụt giảm':>11}{'độ lệch':>10}{'n':>9}")
    print("-" * 88)
    for nm, x in (("Carry theo chuỗi t/g", ts.mean(axis=1)),
                  ("  — sau chi phí 1 pip", ts_c.mean(axis=1)),
                  ("Carry cắt ngang (2-2)", xs),
                  ("  — sau chi phí 1 pip", xs_c)):
        s = stats(x)
        print(f"{nm:<28}{s['sharpe']:>9.2f}{s['ret']:>14.1%}{s['dd']:>11.1%}"
              f"{s['skew']:>10.2f}{s['n']:>9,}")

    print("\n" + "=" * 88)
    print("BANG 2 — CO SUY GIAM THEO GIAI DOAN KHONG? (Sharpe, sau chi phí)")
    print("=" * 88)
    print(f"{'giai đoạn':<16}{'theo chuỗi t/g':>18}{'cắt ngang':>14}{'n phiên':>10}")
    print("-" * 88)
    a = ts_c.mean(axis=1); b = xs_c
    for y0, y1 in PERIODS:
        ia = a[(a.index >= f"{y0}-01-01") & (a.index <= f"{y1}-12-31")]
        ib = b[(b.index >= f"{y0}-01-01") & (b.index <= f"{y1}-12-31")]
        print(f"{f'{y0}–{y1}':<16}{stats(ia)['sharpe']:>18.2f}"
              f"{stats(ib)['sharpe']:>14.2f}{len(ia):>10,}")

    print("\n" + "=" * 88)
    print("BANG 3 — TUNG ĐỒNG, SAU CHI PHÍ, toàn mẫu")
    print("=" * 88)
    print(f"{'đồng':<12}{'Sharpe':>9}{'lợi suất/năm':>15}{'độ lệch':>10}{'carry TB':>11}")
    print("-" * 88)
    for cur in sorted(ts_c.columns):
        s = stats(ts_c[cur].dropna())
        cb = pan[pan.cur == cur].carry_px.mean()
        print(f"{cur:<12}{s['sharpe']:>9.2f}{s['ret']:>14.1%}{s['skew']:>10.2f}{cb:>11.2f}")

    print("\n" + "=" * 88)
    print("BANG 4 — CARRY CÓ SẬP KHI BIẾN ĐỘNG VỌT KHÔNG?")
    print("=" * 88)
    x = ts_c.mean(axis=1)
    v = x.rolling(60).std().shift(1)
    q = v.quantile([1 / 3, 2 / 3]).values
    lab = ["biến động thấp", "biến động vừa", "biến động cao"]
    print(f"{'chế độ':<20}{'Sharpe':>9}{'lợi suất/năm':>15}{'độ lệch':>10}{'n':>9}")
    print("-" * 88)
    g = np.digitize(v.values, q)
    for i in range(3):
        s = stats(x.values[(g == i) & np.isfinite(v.values)])
        print(f"{lab[i]:<20}{s['sharpe']:>9.2f}{s['ret']:>14.1%}{s['skew']:>10.2f}{s['n']:>9,}")

    print("\n" + "=" * 88)
    print("BANG 5 — MAU CAN BANG: chi tu 2002 khi ca 6 dong deu co carry")
    print("=" * 88)
    print("Truoc 2002 danh muc chi co 3-5 dong, nen Sharpe cao giai doan dau mot phan")
    print("la do thanh phan danh muc khac, khong hoan toan la carry manh hon.\n")
    print(f"{'giai đoạn':<20}{'theo chuỗi t/g':>18}{'cắt ngang':>14}{'n phiên':>10}")
    print("-" * 88)
    bal = pan[pan.DATE >= "2002-04-02"]
    ab = ts_carry(bal, cost_pip=1.0).mean(axis=1); bb = xs_carry(bal, cost_pip=1.0)
    for y0, y1 in [(2002, 2009), (2010, 2017), (2018, 2025), (2002, 2025)]:
        ia = ab[(ab.index >= f"{y0}-01-01") & (ab.index <= f"{y1}-12-31")]
        ib = bb[(bb.index >= f"{y0}-01-01") & (bb.index <= f"{y1}-12-31")]
        print(f"{f'{y0}–{y1}':<20}{stats(ia)['sharpe']:>18.2f}"
              f"{stats(ib)['sharpe']:>14.2f}{len(ia):>10,}")

    full = stats(ts_c.mean(axis=1))["sharpe"]
    win = ts_c.mean(axis=1)
    own = stats(win[(win.index >= "2010-01-01") & (win.index <= "2025-12-31")])["sharpe"]
    print("\n" + "-" * 88)
    print(f"KET LUAN")
    print(f"  Sharpe toan mau (1971-2026, danh muc thay doi): {full:+.2f}")
    print(f"  Sharpe TREN DUNG KHOANG HE THONG VAN HANH (2010-2025): {own:+.2f}")
    print(f"  Do lech am {stats(ts_c.mean(axis=1))['skew']:.2f} — dung dac trung 'carry crash'")
    print("  Nguong da dat truoc: Sharpe > 0,30 tren 2010-2025 thi moi dua vao he thong.")
    print("  -> " + ("DAT — dua carry vao tang 2b" if own > 0.30 else
                     "KHONG DAT — carry khong duoc dua vao tang quyet dinh"))


if __name__ == "__main__":
    main()
