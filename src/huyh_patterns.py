"""KIEM CHUNG DOC LAP CAC MAU CUA HuyH TREN DU LIEU CUA TANG 2.

HuyH khai pha mau tuan tu tren chuoi ky hieu (SAX-style) tu FRED daily, va sau
mot pheu loc bon buoc (4.722 dong -> 11 -> 7 -> 3) con lai DUNG BA mau song sot
qua kiem tra ngoai thoi gian 2022-2026, ca ba deu la mau BIEN DONG:

    MEDIUM -> HIGH -> HIGH   => HIGH   lift 1,322 (5/6 cap)
    LOW    -> MEDIUM -> LOW  => LOW    lift 1,285 (5/6 cap)
    HIGH   -> HIGH -> MEDIUM => HIGH   lift 1,150 (5/6 cap)

KHONG mau huong di nao va KHONG mau loi suat 5 trang thai nao song sot.

File nay kiem lai chinh ba mau do tren du lieu CUA CHUNG TA, khac o ba diem:
  * thuoc do bien dong: realized variance 5 phut (rv_adv.csv) thay vi |loi suat|
    ngay — chinh xac hon nhieu
  * khoang thoi gian: 2012-2025 thay vi 1971-2026
  * cach chia: split.py (huan luyen/kiem dinh/kiem tra) thay vi 2016/2021/2022

Neu mau van co lift tren du lieu khac va thuoc do khac thi bang chung manh hon
han mot lan chay. Neu khong thi phai biet.
"""
import os, sys
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
D = os.path.join(os.path.dirname(HERE), "data")
from split import doan
from scipy import stats

P = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
W = 3
MAU = [(("MEDIUM", "HIGH", "HIGH"), "HIGH", 1.322),
       (("LOW", "MEDIUM", "LOW"), "LOW", 1.285),
       (("HIGH", "HIGH", "MEDIUM"), "HIGH", 1.150)]
TEN = ["LOW", "MEDIUM", "HIGH"]


def trang_thai(pair, cot="rv5"):
    """Roi rac hoa bien dong thanh 3 trang thai, moc lay tu doan HUAN LUYEN."""
    adv = pd.read_csv(os.path.join(D, "rv_adv.csv"), parse_dates=["Date"])
    d = adv[adv.pair == pair].sort_values("Date").reset_index(drop=True)
    d = d[d.n5 >= 100].reset_index(drop=True)          # bo phien Chu nhat
    g = doan(d.Date.values)
    v = d[cot].values
    q = np.quantile(v[g == 0], [1 / 3, 2 / 3])
    s = np.array(TEN)[np.digitize(v, q)]
    return d.Date.values, s, g


def lift(states, g, seg, mau, dich):
    """Xac suat co dieu kien / xac suat nen, tren mot doan."""
    n = len(states)
    idx = [t for t in range(W, n) if g[t] == seg]
    if not idx:
        return np.nan, 0, np.nan
    nen = np.mean([states[t] == dich for t in idx])
    hop = [t for t in idx if tuple(states[t - W:t]) == mau]
    if len(hop) < 10 or nen == 0:
        return np.nan, len(hop), nen
    p = np.mean([states[t] == dich for t in hop])
    return p / nen, len(hop), nen


if __name__ == "__main__":
    DATA = {p: trang_thai(p) for p in P}
    print("=" * 100)
    print("KIEM CHUNG DOC LAP BA MAU CUA HuyH — do bang realized variance 5 phut")
    print("=" * 100)
    print("Lift = P(trạng thái đích | mẫu) / P(trạng thái đích). Lift > 1,05 là dương.\n")
    tong = []
    for mau, dich, lift_huyh in MAU:
        nhan = " → ".join(mau) + f"  ⇒ {dich}"
        print("-" * 100)
        print(f"{nhan}      (HuyH đo được lift {lift_huyh:.3f} trên FRED daily)")
        print(f"{'cặp':<9}{'n khớp':>9}{'P(đích|mẫu)':>14}{'nền':>9}{'lift':>9}{'dương?':>9}")
        L = []
        for p in P:
            _, s, g = DATA[p]
            lf, n, nen = lift(s, g, 2, mau, dich)
            if np.isfinite(lf):
                L.append(lf)
                print(f"{p:<9}{n:>9}{lf*nen:>14.3f}{nen:>9.3f}{lf:>9.3f}"
                      f"{('có' if lf > 1.05 else 'không'):>9}")
            else:
                print(f"{p:<9}{n:>9}{'—':>14}{'—':>9}{'—':>9}{'—':>9}")
        if L:
            duong = sum(x > 1.05 for x in L)
            print(f"{'TRUNG VỊ':<9}{'':>9}{'':>14}{'':>9}{np.median(L):>9.3f}"
                  f"{f'{duong}/{len(L)}':>9}")
            tong.append((nhan, np.median(L), duong, len(L), lift_huyh))
    print("=" * 100)
    print("SO SANH VOI KET QUA CUA HuyH")
    print("=" * 100)
    print(f"{'mẫu':<34}{'lift HuyH':>12}{'lift ở đây':>13}{'chênh':>9}{'số cặp dương':>15}")
    print("-" * 100)
    for nhan, med, duong, n, lh in tong:
        print(f"{nhan:<34}{lh:>12.3f}{med:>13.3f}{med-lh:>+9.3f}{f'{duong}/{n}':>15}")
    print("-" * 100)
    dat = sum(1 for _, m, d_, n, _ in tong if m > 1.05 and d_ / n >= 0.75)
    print(f"Số mẫu tái lập được theo đúng luật của HuyH (trung vị lift > 1,05 và ≥75% cặp dương): {dat}/3")

    # mau doi chung: mau NGAU NHIEN cung do dai, de biet lift bao nhieu la do may
    print("\n" + "=" * 100)
    print("ĐỐI CHỨNG — phân phối lift của TẤT CẢ 27 mẫu 3 trạng thái")
    print("=" * 100)
    print("Nếu lift ~1,2 là chuyện thường thì ba mẫu kia không có gì đặc biệt.\n")
    tat_ca = []
    from itertools import product
    for mau in product(TEN, repeat=3):
        for dich in TEN:
            L = [lift(DATA[p][1], DATA[p][2], 2, mau, dich)[0] for p in P]
            L = [x for x in L if np.isfinite(x)]
            if len(L) >= 4:
                tat_ca.append((" → ".join(mau) + f" ⇒ {dich}", float(np.median(L))))
    v = np.array([x[1] for x in tat_ca])
    print(f"  {len(tat_ca)} tổ hợp (mẫu × đích) đủ mẫu trên ≥4 cặp")
    print(f"  phân vị lift: 50%={np.median(v):.3f}  75%={np.quantile(v,.75):.3f}  "
          f"90%={np.quantile(v,.90):.3f}  max={v.max():.3f}")
    for nhan, med, *_ in tong:
        pct = float((v < med).mean())
        print(f"  {nhan:<34} lift {med:.3f} → nằm ở phân vị {pct:.0%} của phân phối đối chứng")
    print("\nTop 5 tổ hợp mạnh nhất trên dữ liệu này:")
    for nhan, m in sorted(tat_ca, key=lambda x: -x[1])[:5]:
        print(f"  {nhan:<34}{m:>8.3f}")
