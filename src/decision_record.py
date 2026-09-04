"""TANG 6 — PHIEU QUYET DINH: bien ket qua tang 1-5 thanh mot to phieu doc duoc.

Ba phan cua phieu, va cai nao kiem dinh duoc:

  1. HANH DONG  (don bay khuyen nghi)  -> tu PositionSizer, tang 4
  2. RUI RO     (P cham stop, khoang gia) -> KIEM DINH DUOC, va da kiem
  3. GIAI THICH (rang buoc nao dang siet) -> tu PositionSizer.explain

Ket qua kiem dinh hieu chuan (du lieu giu rieng, 6 cap, n=8.957 moi nguong):

  * P(cham stop) theo nguyen ly phan xa: lech tuyet doi TB 1,44%,
    toi da 3,08% (o stop rat gan 0,5 sigma, va lech theo huong BI QUAN).
    O vung hay dung 2-3 sigma: lech 0,40% va 0,10%.

  * Khoang du bao, |lech| trung binh tren 4 muc 80/90/95/99%:
        Gauss     1,20%   (nhung 99% danh nghia chi phu 97,5% — hut duoi)
        Student-t 1,25%   (be rong 158,0 pip o muc 99%)
        Conformal 0,37%   (be rong 145,6 pip — vua chuan hon vua hep hon)

  * Bao dam cua conformal chi la BIEN. Do phu theo che do bien dong
    (muc danh nghia 90%):
        Conformal chung        88,4 / 90,8 / 91,9   -> lech max 1,9%
        Conformal theo che do  90,3 / 90,8 / 89,4   -> lech max 0,8%
    Vi vay tang 6 dung ban PHAN TANG THEO CHE DO BIEN DONG (Mondrian).

  * SO TANG PHU THUOC CHAT LUONG DU BAO sig — va da doi HAI LAN.
    Voi panel vong 6: ban TINH Mondrian 2 thang tren kiem dinh.
    Voi panel vong 7 (lich NHTW rieng tung cap, sig tot hon): ACI PHAN TANG
    3 CHE DO thang tren kiem dinh (lech max 2,0% so voi 2,9% cua Mondrian 3
    va 4,0% cua ban tinh). Mac dinh san xuat gio lay tu khoang_mac_dinh().
    Moi lan doi tang 2 phai chay lai run_final_eval2.py va cap nhat.

GIOI HAN DA DO, PHAI GHI VAO LUAN VAN: ca ba phuong phap deu PHU THIEU
khi tai khoan dang lo (Conformal 89,3% luc sut giam so voi 90,3% luc o
dinh von) — dung luc nguoi dung can con so chinh xac nhat.

VA 03/09/2026 — HAI LO HONG DUOC VA:
  * `mu` (loi the Kelly) o tu kiem TRUOC DAY la hang so bia dat 0,0002. Tang 1
    (huong di) da bi bac bo (E[zT]=0) nen KHONG co co so nao cho con so do.
    Loi the hop le DUY NHAT trong he thong la carry (xem optimal_stop.py, va
    compare_leverage_dp.py/run_e2e.py da dung no cho tang 6b). Tu kiem gio
    tinh mu = trung vi carry ngay tren doan huan luyen (carry_ngay(), cua so
    mo rong), giong het quy uoc tang 6b.
  * PHIEU GIO CO THEM MUC "TANG 6b" — khuyen nghi giu/dong va bien gioi dong
    lenh (so lan do lech chuan con cach stop) tu quy hoach dong cua
    optimal_stop.py, qua lop KhuyenNghiGiuDong duoi day. Truoc ban va nay,
    tang 6b da giai xong nhung KHONG bao gio duoc hien len phieu — nguoi
    dung khong co cach nao thay khuyen nghi giu/dong. Dung ban KHONG don bay
    (f_v=None) vi do la chinh sach dang san xuat; ban co don bay da kiem
    chung la chua an toan de mac dinh (xem docs/TANG6B_DUNGTOIUU.md).
"""
import numpy as np
from scipy import stats

import optimal_stop as O
from position_sizing import PositionSizer          # noqa: F401  (tai xuat)

# sai so hieu chuan DA DO — dung de in "do tin cay cua chinh con so nay"
LECH_DA_DO_KHOANG = 0.012        # conformal phan tang, lech toi da theo che do
LECH_DA_DO_PSTOP = 0.0144        # P(cham stop), lech tuyet doi trung binh
LECH_KHI_DANG_LO = 0.010         # phu thieu them khi dang sut giam

# CANH BAO XU HUONG VUNG CANG THANG — src/run_momentum_regime.py, tang 1
# phuong an 2. He thong KHONG dat cuoc huong di (tang 1 da chung minh E[zT]=0
# noi chung), nhung neu nguoi dung tu quyet dinh vao lenh theo TIN HIEU XU
# HUONG GAN DAY (mua vi gia dang len, ban vi gia dang xuong) thi rieng trong
# vung cang thang, du lieu cho thay dieu do LO CO Y NGHIA THONG KE, khong
# trung tinh: Sharpe -0,615 tren kiem dinh (t=-3,30, p=0,001, song sot
# Bonferroni-3), dau nhat quan tren ca 4 span o kiem tra. Day la thong tin
# cho NGUOI DUNG (he thong khong tu dong chan lenh), dung chung nguong
# s_stress voi tang 4 (top 5% sigma du bao cua tap huan luyen).
CANH_BAO_DONG_LUONG = (
    "Nếu đang định vào lệnh theo TÍN HIỆU XU HƯỚNG gần đây (mua vì giá đang "
    "lên / bán vì giá đang xuống): trong vùng biến động này, xu hướng có xu "
    "hướng LỖ có ý nghĩa thống kê (Sharpe -0,62, p=0,001), không trung tính. "
    "Hệ thống không tự chặn lệnh — đây chỉ là cảnh báo dựa trên dữ liệu.")


def pip_size(pair):
    return 0.01 if "JPY" in pair.upper() else 0.0001


# ───────────────────────── xac suat cham stop ─────────────────────────
def p_cham_stop(k_sigma, z_train, horizon=1):
    """P(gia cham stop dat cach k_sigma lan do lech chuan) trong 'horizon' phien.

    Nguyen ly phan xa: P(min_{t<=T} X_t <= -b) = 2 P(X_T <= -b) voi buoc di
    doi xung. Duoi t-Student khop tren tap huan luyen, khong phai Gauss."""
    nu, _, sc = stats.t.fit(np.asarray(z_train, float), floc=0)
    nu = float(np.clip(nu, 2.5, 40))
    return float(min(1.0, 2.0 * stats.t.cdf(-k_sigma / (sc * np.sqrt(horizon)), nu)))


# ───────────────────── khoang conformal phan tang ─────────────────────
class KhoangConformal:
    """Conformal chia doi, PHAN TANG theo phan vi bien dong (Mondrian).

    So tang la mot danh doi: nhieu tang thi bam sat che do hon nhung moi tang
    it mau hieu chuan hon. Voi du bao sig CU (MA20-GK) thi 3 tang la tot nhat;
    voi du bao MOI (volfc) thi 2 tang tot hon — do sac hon nen phan du con lai
    it khong dong nhat hon, va it tang cho nhieu mau hon:

        sig cu, 3 tang   89,5 / 90,3 / 88,8   lech max 1,2%
        sig moi, 2 tang  88,9 / 90,0          lech max 1,1%
        sig moi, 3 tang  87,6 / 90,7 / 89,3   lech max 2,4%

    Tang nao mong (< min_n mau) thi lui ve dung bo chung."""

    def __init__(self, z_train, sig_train, n_bins=2, min_n=50):
        z = np.asarray(z_train, float)
        s = np.asarray(sig_train, float)
        self.n_bins = int(n_bins)
        self.edges = np.quantile(s, np.arange(1, self.n_bins) / self.n_bins)
        g = np.digitize(s, self.edges)
        self.z_chung = z
        self.z_theo_che_do = [z[g == i] if (g == i).sum() >= min_n else z
                              for i in range(self.n_bins)]
        self.n_theo_che_do = [int((g == i).sum()) for i in range(self.n_bins)]

    def che_do(self, sig):
        return int(np.digitize([float(sig)], self.edges)[0])

    def ten_che_do(self, sig):
        i = self.che_do(sig)
        if self.n_bins == 2:
            return ("thấp", "cao")[i]
        if self.n_bins == 3:
            return ("thấp", "vừa", "cao")[i]
        return f"tầng {i+1}/{self.n_bins}"

    def nua_be_rong(self, muc, sig=None):
        """Nua be rong khoang, don vi 'so lan do lech chuan'."""
        z = self.z_chung if sig is None else self.z_theo_che_do[self.che_do(sig)]
        return float(np.quantile(np.abs(z), min(muc * (1 + 1 / len(z)), 0.9999)))

    def khoang(self, gia, sig, muc=0.90):
        h = self.nua_be_rong(muc, sig) * sig
        return float(gia * np.exp(-h)), float(gia * np.exp(h))


class KhoangACI:
    """CONFORMAL THICH UNG THEO TANG — ban dang dung o tang 6.

    Hai y tuong ghep lai:
      * MONDRIAN: hieu chuan rieng cho tung che do bien dong (2 tang)
      * ACI (Gibbs & Candes, NeurIPS 2021): cap nhat muc alpha TRUC TUYEN
            alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
        voi err_t = 1 neu quan sat that roi ra ngoai khoang. Bao dam: do phu
        dai han hoi tu ve dung muc danh nghia BAT KE phan phoi troi the nao.

    Do tren tap giu rieng, muc danh nghia 90%, panel moi (lech so voi 90%):

        phuong phap    chung   o dinh  dang lo  vol thap  vol cao  lech max
        tinh           89,6%    89,8%    89,2%     86,9%    90,2%      3,1%
        mondrian 3     89,6%    90,0%    89,2%     87,6%    89,3%      2,4%
        cua so truot   89,7%    89,9%    89,3%     87,0%    90,1%      3,0%
        ACI chung      90,2%    90,5%    89,8%     86,8%    91,5%      3,2%
        DtACI          89,8%    90,1%    89,5%     87,0%    91,1%      3,0%
        ACI THEO TANG  90,3%    90,7%    89,9%     88,8%    90,1%      1,2%   <-

    Diem khoang (Winkler) cung tot nhat: 285,0 so voi 290,0 cua ban tinh.

    CON LAI CHUA SUA DUOC: moi phuong phap deu phu thieu khoang 0,6-0,8 diem
    phan tram khi tai khoan dang lo. Da thu ca nam cach, khong cach nao xoa
    duoc khoang chenh do. Phai ghi trong luan van la gioi han da do."""

    def __init__(self, z_train, sig_train, n_bins=2, muc=0.90, gamma=0.01, cua_so=750):
        z = np.asarray(z_train, float); s = np.asarray(sig_train, float)
        self.n_bins, self.muc, self.gamma, self.cua_so = int(n_bins), float(muc), float(gamma), int(cua_so)
        self.edges = np.quantile(s, np.arange(1, self.n_bins) / self.n_bins)
        self.alpha = np.full(self.n_bins, 1.0 - self.muc)
        self.z_hist = list(z[-self.cua_so:])
        self.s_hist = list(s[-self.cua_so:])

    def che_do(self, sig):
        return int(np.digitize([float(sig)], self.edges)[0])

    def ten_che_do(self, sig):
        i = self.che_do(sig)
        return ("thấp", "cao")[i] if self.n_bins == 2 else f"tầng {i+1}/{self.n_bins}"

    def _cal(self, i):
        z = np.asarray(self.z_hist); s = np.asarray(self.s_hist)
        sel = z[np.digitize(s, self.edges) == i]
        return sel if len(sel) >= 60 else z

    def nua_be_rong(self, muc=None, sig=None):
        """Nua be rong, don vi 'so lan do lech chuan'. muc=None -> muc thich ung."""
        i = 0 if sig is None else self.che_do(sig)
        lev = (1.0 - self.alpha[i]) if muc is None else float(muc)
        c = self._cal(i)
        return float(np.quantile(np.abs(c), min(max(lev, 0.0) * (1 + 1 / len(c)), 0.9999)))

    def khoang(self, gia, sig, muc=None):
        h = self.nua_be_rong(muc, sig) * sig
        return float(gia * np.exp(-h)), float(gia * np.exp(h))

    def quan_sat(self, z, sig):
        """Goi SAU khi biet ket qua that: cap nhat alpha va bo nho hieu chuan."""
        i = self.che_do(sig)
        err = float(abs(float(z)) > self.nua_be_rong(None, sig))
        a = self.alpha[i] + self.gamma * ((1.0 - self.muc) - err)
        self.alpha[i] = float(np.clip(a, 1e-4, 0.5))
        self.z_hist.append(float(z)); self.s_hist.append(float(sig))
        if len(self.z_hist) > self.cua_so:
            self.z_hist.pop(0); self.s_hist.pop(0)
        return err

    @property
    def muc_hien_hanh(self):
        return 1.0 - self.alpha

    @property
    def n_theo_che_do(self):
        """So mau hieu chuan dang co trong tung che do (cua so truot cua_so)."""
        s = np.asarray(self.s_hist)
        g = np.digitize(s, self.edges)
        return [int((g == i).sum()) for i in range(self.n_bins)]


def khoang_mac_dinh(z_train, sig_train, muc=0.90):
    """Bo tao khoang MAC DINH cho san xuat — dung o tang 6 va o API.

    Chon tren doan KIEM DINH (run_final_eval2.py, panel vong 7): ACI phan
    tang 3 che do. Lech toi da theo che do 2,0% tren kiem dinh, 1,5% tren
    kiem tra; tot nhat co the tren kiem tra la 1,1% (ACI-tang 2), tuc gia
    phai tra cho viec CHON la +0,4 diem phan tram.

    LUU Y THAT: lua chon nay KHONG on dinh. No doi theo chat luong cua sig.
    Voi panel vong 6 thi ban tinh Mondrian 2 thang; voi sig tot hon cua vong
    7 thi ACI phan tang lai thang. Sau moi lan doi tang 2 phai chay lai
    run_final_eval2.py va cap nhat ham nay. Chenh lech giua sau phuong phap
    tren kiem tra chi 1,1%-2,0%, nen day khong phai lua chon lon — nhung no
    phai duoc CHON tren kiem dinh chu khong duoc dat tay.
    """
    return KhoangACI(z_train, sig_train, n_bins=3, muc=muc)


class TamHan:
    """NHIEU PHIEN — phieu chi hieu chuan cho MOT phien, nhung nguoi dung giu lau hon.

    Hai thu duoc do tren tap giu rieng (6 cap, ~21.500 quan sat moi tam han):

    1. CONG THUC PHAN XA VAN DUNG khi keo dai tam han, chi hoi lac quan:
           tam han   lech tuyet doi trung binh   huong
           1 phien              0,4%             —
           5 phien              1,3%             du bao THAP hon thuc te
           10 phien             1,7%             du bao THAP hon thuc te
           20 phien             1,7%             du bao THAP hon thuc te
       Vi du stop 2 sigma: giu 1 phien P=4,4%; giu 10 phien P=47,7% (thuc te 49,7%).
       Con so mot phien KHONG duoc doc thanh con so cua ca tuan.

    2. QUY TAC √h dung TRUNG BINH nhung LECH THEO CHE DO. Do lech chuan thuc
       chia cho (sigma·√h), tap giu rieng:
           tam han   vol thap   vol vua   vol cao
           5           1,124      0,953     0,894
           10          1,118      0,972     0,860
           20          1,120      1,015     0,839
       Bien dong hoi quy ve trung binh: dang yen thi rui ro tuong lai LON hon √h,
       dang cang thang thi NHO hon. He so hieu chinh c(h, che do) uoc luong tren
       tap huan luyen keo bien do tu 0,839-1,124 ve 0,890-1,057."""

    def __init__(self, sig_train, r_train, tam_han=(1, 5, 10, 20), n_bins=3):
        s = np.asarray(sig_train, float); r = np.asarray(r_train, float)
        self.tam_han = tuple(int(h) for h in tam_han)
        q = np.quantile(s, np.arange(1, n_bins) / n_bins)
        self.edges = q; self.n_bins = n_bins
        g = np.digitize(s, q)
        self.c = {}
        for h in self.tam_han:
            cum = np.full(len(r), np.nan)
            cs = np.concatenate([[0.0], np.cumsum(r)])
            if len(r) > h:
                cum[:len(r) - h] = cs[h + 1:] - cs[1:len(r) - h + 1]
            z = cum / (s * np.sqrt(h))
            row = []
            for i in range(n_bins):
                m = (g == i) & np.isfinite(z)
                row.append(float(np.std(z[m])) if m.sum() >= 100 else 1.0)
            self.c[h] = row

    def he_so(self, h, sig):
        i = int(np.digitize([float(sig)], self.edges)[0])
        return float(self.c[h][i])

    def sig_h(self, sig, h):
        """Do lech chuan cho tam han h, DA sua thien lech hoi quy ve trung binh."""
        return float(sig) * np.sqrt(h) * self.he_so(h, sig)

    def bang(self, sig, z_train, stop_sigma=2.0):
        """P(cham stop dat co dinh tai stop_sigma·sigma) theo tung tam han."""
        out = []
        for h in self.tam_han:
            sh = self.sig_h(sig, h) / max(float(sig), 1e-12)     # quy ve so lan sigma 1 phien
            out.append((h, p_cham_stop(stop_sigma / max(sh, 1e-12), z_train)))
        return out


class LuatKyHieu:
    """LUAT KY HIEU — nhanh khai pha mau tuan tu cua HuyH, dua vao tang 6.

    HuyH roi rac hoa bien dong thanh ba trang thai roi khai pha mau tuan tu do
    dai 3 (kieu SAX; Lin, Keogh, Wei & Lonardi 2007). Sau pheu bon buoc — 4.722
    to hop -> 11 mau chung -> 7 mau qua leave-one-pair-out -> 3 mau qua kiem tra
    ngoai thoi gian 2022-2026 — con lai DUNG BA mau, va ca ba deu la mau BIEN
    DONG. Khong mau huong di nao song sot. Do la ket luan am doc lap, trung khop
    voi ket qua cua chung ta bang phuong phap hoan toan khac.

    Ba mau da duoc KIEM CHUNG LAI tren du lieu cua chung ta (realized variance
    5 phut thay vi |loi suat| ngay), doan kiem tra:

        mau                        lift cua HuyH   lift o day
        MEDIUM -> HIGH -> HIGH        1,322          1,690
        LOW -> MEDIUM -> LOW          1,285          1,318
        HIGH -> HIGH -> MEDIUM        1,150          1,690

    KHONG dung de DU BAO. Da do: them dac trung ky hieu vao STHARQ chi cai
    thien QLIKE 0,15-0,49% va thang 0/6 cap theo Diebold-Mariano — mo hinh lien
    tuc da bat het thong tin nay roi. Dung o day nhu LOI GIAI THICH cho nguoi
    dung: mot cau bang tieng Viet ve trang thai hien tai, kem so lift do duoc.

    Xem src/huyh_patterns.py va docs/TICHHOP_HUYH.md."""

    TEN = ("thấp", "vừa", "cao")
    LUAT = {
        (1, 2, 2): ("cao", 1.69, "vừa → cao → cao"),
        (0, 1, 0): ("thấp", 1.32, "thấp → vừa → thấp"),
        (2, 2, 1): ("cao", 1.69, "cao → cao → vừa"),
        # Hai mau duoi KHONG den tu HuyH ma tu bang doi chung cua chinh chung
        # ta (src/huyh_patterns.py): tren du lieu nay chung manh hon ca ba mau
        # kia. Deu la quan tinh bien dong thuan tuy — khong moi ve mat hoc thuat
        # nhung dung la thu nguoi dung can nghe.
        (2, 2, 2): ("cao", 4.80, "cao → cao → cao"),
        (0, 0, 0): ("thấp", 1.41, "thấp → thấp → thấp"),
    }

    def __init__(self, rv_train):
        v = np.asarray(rv_train, float)
        self.q = np.quantile(v[np.isfinite(v)], [1 / 3, 2 / 3])

    def trang_thai(self, rv):
        return int(np.digitize([float(rv)], self.q)[0])

    def doc(self, rv_3phien):
        """rv_3phien: ba gia tri realized variance gan nhat, cu nhat truoc.

        Tra ve chuoi giai thich, hoac None neu khong khop luat nao."""
        if len(rv_3phien) < 3:
            return None
        k = tuple(self.trang_thai(x) for x in list(rv_3phien)[-3:])
        if k not in self.LUAT:
            return None
        dich, lift, nhan = self.LUAT[k]
        return (f"Ba phiên gần nhất: {nhan}. Trong lịch sử, sau mẫu này xác suất "
                f"biến động {dich} cao gấp {lift:.2f} lần mức nền.")


class KhuyenNghiGiuDong:
    """TANG 6b — cau noi quy hoach dong (optimal_stop.py) vao phieu quyet dinh.

    Phieu quyet dinh la ANH CHUP MOT PHIEN, khong theo doi trang thai cua mot
    vi the DANG mo (khong biet gia vao lenh, khong biet da giu bao lau). Nen
    khong the tra ve "gio phai giu hay dong" cho mot vi the co the. Thu lam
    duoc, va la thu phieu con thieu truoc ban va nay: BIEN GIOI dong lenh
    theo tung che do bien dong — "o che do nay, dong neu con cach stop duoi
    X sigma" — cong khuyen nghi tai thoi diem VUA VAO LENH (s = stop_sigma,
    diem xa stop nhat co the, tuc de nhat de GIU).

    Giai DP MOT LAN luc khoi tao (dat cho ca phien lam viec), dung f_v=None
    (khong don bay) — dung CHINH SACH DANG SAN XUAT, xem docstring module."""

    def __init__(self, pair, mau, sig_v, nguong, c_thoat_sigma, slip_sigma,
                 carry_ngay_abs, N=20, f_v=None, seed=1):
        self.pair = pair
        self.nguong, self.sig_v = nguong, sig_v
        self.c_thoat_sigma, self.slip_sigma, self.N = c_thoat_sigma, slip_sigma, int(N)
        self.carry_ngay_abs, self.f_v = float(carry_ngay_abs), f_v
        self.V, self.bien = O.giai(mau, sig_v, self.carry_ngay_abs, c_thoat_sigma,
                                    slip_sigma, N=self.N, seed=seed, f_v=f_v)

    def che_do(self, sig):
        return int(np.digitize([float(sig)], self.nguong)[0])

    def khuyen_nghi(self, sig, stop_sigma=2.0, n=None):
        """gia_moi: co nen giu KHONG neu vua vao lenh (s=stop_sigma) o che do
        hien tai; bien_gioi_sigma: nguong 'con cach stop bao xa thi con dang
        giu duoc' o tam han on dinh — None neu carry khong bu noi chi phi o
        BAT KY khoang cach nao (khuyen nghi la dong ngay, moi luc)."""
        v = self.che_do(sig)
        n = self.N if n is None else int(n)
        giu_moi = O.nen_giu(self.V, stop_sigma, v, n, self.c_thoat_sigma,
                             self.sig_v, f_v=self.f_v)
        b = self.bien[n, v] if np.isfinite(self.bien[n, v]) else None
        return dict(che_do=v, giu_luc_vao=giu_moi, bien_gioi_sigma=b,
                    carry_ngay=self.carry_ngay_abs)


def _bo_chu(text, w):
    """Ngat dong theo tu, khong cat giua tu."""
    out, cur = [], ""
    for word in str(text).split():
        if len(cur) + len(word) + 1 > w:
            out.append(cur); cur = word
        else:
            cur = (cur + " " + word).strip()
    if cur:
        out.append(cur)
    return out


# ───────────────────────────── to phieu ─────────────────────────────
class PhieuQuyetDinh:
    def __init__(self, sizer, khoang, pair, z_train, tamhan=None, tang6b=None):
        self.sizer, self.khoang, self.pair = sizer, khoang, pair
        self.z_train = np.asarray(z_train, float)
        self.tamhan = tamhan
        self.tang6b = tang6b          # KhuyenNghiGiuDong, tuy chon — xem lop do

    def lap(self, gia, sig, mu, nu, dd=0.0, stop_sigma=2.0,
            muc=(0.80, 0.95), von=10000.0, so_vi_the=1, luat=None):
        ex = self.sizer.explain(sig, mu, nu, dd, so_vi_the=so_vi_the)
        p = p_cham_stop(stop_sigma, self.z_train)
        kh = {m: self.khoang.khoang(gia, sig, m) for m in muc}
        kn = (self.tang6b.khuyen_nghi(sig, stop_sigma) if self.tang6b is not None else None)
        return dict(
            pair=self.pair, gia=float(gia), von=float(von),
            don_bay=ex["f"], von_dat=ex["f"] * von,
            rang_buoc=ex["rang_buoc"], k_vol=ex["k_vol"], k_dd=ex["k_dd"],
            k_dm=ex["k_dm"], so_vi_the=ex["so_vi_the"],
            che_do_cang_thang=ex["che_do_cang_thang"], rho_hieu_dung=ex["rho_hieu_dung"],
            kelly=ex["kelly"], tran_rui_ro=ex["ruin_cap"],
            muc_bien_dong=ex["muc_bien_dong"], sut_giam=float(dd),
            stop_sigma=float(stop_sigma),
            stop_gia=float(gia * (1 - stop_sigma * sig)),
            stop_pip=float(stop_sigma * sig * gia / pip_size(self.pair)),
            p_cham_stop=p, khoang=kh,
            bang_tam_han=(self.tamhan.bang(sig, self.z_train, stop_sigma)
                          if self.tamhan is not None else None),
            luat_ky_hieu=luat,
            n_mau_che_do=self.khoang.n_theo_che_do[self.khoang.che_do(sig)],
            canh_bao_dong_luong=bool(sig >= self.sizer.s_stress),
            khuyen_nghi_giu_dong=kn,
        )

    def in_ra(self, r, W=64):
        pip = pip_size(self.pair)
        L, a = [], None
        rows = []
        rows.append(f"PHIẾU QUYẾT ĐỊNH — {r['pair']}")
        rows.append(None)                                   # ke ngang
        rows.append(f"Giá tham chiếu {r['gia']:.5f}    Vốn {r['von']:,.0f}")
        rows.append(f"Đòn bẩy khuyến nghị {r['don_bay']:.2f}×  →  đặt {r['von_dat']:,.0f}")
        rows.append("")
        rows.append(f"VÌ SAO: ràng buộc đang siết là {r['rang_buoc'].upper()}")
        rows.append(f"  Kelly {r['kelly']:.2f}×    trần rủi ro {r['tran_rui_ro']:.2f}×")
        rows.append(f"  k biến động {r['k_vol']:.2f} (mức {r['muc_bien_dong']})"
                    f"   k sụt giảm {r['k_dd']:.2f} (−{r['sut_giam']:.0%})")
        rows.append(f"  k danh mục {r['k_dm']:.2f} ({r['so_vi_the']} vị thế mở cùng lúc)"
                    f"  ρ={r['rho_hieu_dung']:.2f}")
        if r["che_do_cang_thang"]:
            rows.append("  ⚠ vùng căng thẳng — tương quan đo được cao hơn mức nền")
        rows.append("")
        rows.append(f"RỦI RO: stop {r['stop_sigma']:.1f}σ tại {r['stop_gia']:.5f}"
                    f" ({r['stop_pip']:.0f} pip)")
        rows.append(f"  Xác suất chạm stop trong 1 phiên: {r['p_cham_stop']:.2%}"
                    f"  (±{LECH_DA_DO_PSTOP:.1%})")
        for m, (lo, hi) in sorted(r["khoang"].items()):
            rows.append(f"  Khoảng giá {m:.0%}: {lo:.5f} – {hi:.5f}"
                        f"  ({(hi - lo) / pip:.0f} pip)")
        if r.get("bang_tam_han"):
            rows.append("  Nếu GIỮ LỆNH lâu hơn, xác suất chạm stop tăng nhanh:")
            rows.append("    " + "   ".join(f"{h} phiên: {v:.0%}" for h, v in r["bang_tam_han"]))
        if r.get("khuyen_nghi_giu_dong"):
            kn = r["khuyen_nghi_giu_dong"]
            rows.append("")
            rows.append("TẦNG 6b — GIỮ HAY ĐÓNG (quy hoạch động, carry đã trừ chi phí trượt giá)")
            rows.append(f"  Lúc vừa vào lệnh ({r['stop_sigma']:.1f}σ tới stop): khuyến nghị "
                        f"{'GIỮ' if kn['giu_luc_vao'] else 'ĐÓNG NGAY'}")
            if kn["bien_gioi_sigma"] is None:
                rows.append("  Carry hiện không đủ bù chi phí thoát ở chế độ này — biên giới:")
                rows.append("  đóng ngay dù còn cách stop bao xa")
            else:
                rows.append(f"  Biên giới đóng lệnh: đóng nếu còn cách stop dưới "
                            f"{kn['bien_gioi_sigma']:.2f}σ (còn xa hơn thì giữ)")
            rows.append(f"  (carry giả định {kn['carry_ngay']*1e4:.2f} bp/phiên, "
                        f"chế độ biến động {kn['che_do']+1}/{self.tang6b.V.shape[2] if self.tang6b else '?'})")
            if r["don_bay"] <= 1e-9:
                rows.append("  ⚠ đòn bẩy khuyến nghị đang là 0× — không có lý do vào lệnh mới;")
                rows.append("  mục này chỉ áp dụng nếu bạn ĐANG giữ vị thế từ trước.")
        if r.get("luat_ky_hieu"):
            rows.append("")
            rows.append("LUẬT KÝ HIỆU (nhánh khai phá mẫu)")
            for line in _bo_chu(r["luat_ky_hieu"], W - 4):
                rows.append("  " + line)
        if r.get("canh_bao_dong_luong"):
            rows.append("")
            rows.append("⚠ CẢNH BÁO XU HƯỚNG (vùng căng thẳng)")
            for line in _bo_chu(CANH_BAO_DONG_LUONG, W - 4):
                rows.append("  " + line)
        rows.append("")
        rows.append("ĐỘ TIN CẬY CỦA CHÍNH CÁC SỐ TRÊN")
        rows.append(f"  Khoảng: conformal phân tầng theo chế độ biến động,")
        rows.append(f"  lệch ≤{LECH_DA_DO_KHOANG:.1%}; mẫu hiệu chuẩn chế độ này:"
                    f" {r['n_mau_che_do']:,} phiên")
        rows.append(f"  ⚠ khi đang lỗ, độ phủ thực tế thấp hơn ghi ~{LECH_KHI_DANG_LO:.0%}")
        rows.append("")
        rows.append("ĐIỀU KIỆN ĐỂ CON SỐ ĐÒN BẨY CÒN ĐÚNG")
        rows.append("  Phải định cỡ lại MỖI PHIÊN. Định cỡ mỗi tháng thì xác suất")
        rows.append("  phá sản thật là 1,15% thay vì 0,41% (ngân sách 1%).")
        rows.append(f"  Đã khai {r['so_vi_the']} vị thế. Khai thiếu là sai theo cấp số:")
        rows.append("  6 lệnh cùng hướng USD ở trần đầy đủ → phá sản 73,6%.")
        out = ["┌" + "─" * W + "┐"]
        for t in rows:
            if t is None:
                out.append("├" + "─" * W + "┤")
            else:
                out.append("│ " + t[: W - 2].ljust(W - 2) + " │")
        out.append("└" + "─" * W + "┘")
        return "\n".join(out)


# ───────────────────────────── tu kiem ─────────────────────────────
if __name__ == "__main__":
    import os, pandas as pd, warnings
    warnings.filterwarnings("ignore")
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    pan = pd.read_csv(os.path.join(d, "panel2_6pairs.csv"), parse_dates=["Date"])
    PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]

    print("TU KIEM TANG 6")
    # 1) do phu thuc te cua khoang phan tang, tren tap giu rieng
    for muc in (0.80, 0.90, 0.95):
        cov = []
        for p in PAIRS:
            g = pan[pan.pair == p].reset_index(drop=True)
            n = int(len(g) * 0.70)
            tr, te = g.iloc[:n], g.iloc[n:]
            kc = KhoangConformal(tr.zT.values, tr.sig.values)
            h = np.array([kc.nua_be_rong(muc, s) for s in te.sig.values])
            cov.append(float(np.mean(np.abs(te.zT.values) <= h)))
        m = float(np.mean(cov))
        print(f"  độ phủ ở mức {muc:.0%}: {m:.1%}  (lệch {m - muc:+.1%})")
        assert abs(m - muc) < 0.03, "khoang phai phu gan muc danh nghia"

    # 1b) ban thich ung theo tang — ban dang dung
    covA=[]; covM=[]
    for p in PAIRS:
        g = pan[pan.pair == p].reset_index(drop=True)
        n = int(len(g) * 0.70); tr, te = g.iloc[:n], g.iloc[n:]
        ka = KhoangACI(tr.zT.values, tr.sig.values)
        ok = []
        for _, row in te.iterrows():
            h = ka.nua_be_rong(None, row.sig)
            ok.append(abs(row.zT) <= h)
            ka.quan_sat(row.zT, row.sig)
        covA.append(float(np.mean(ok)))
        kc0 = KhoangConformal(tr.zT.values, tr.sig.values)
        hh = np.array([kc0.nua_be_rong(0.90, x) for x in te.sig.values])
        covM.append(float(np.mean(np.abs(te.zT.values) <= hh)))
    a, m = float(np.mean(covA)), float(np.mean(covM))
    print(f"  ACI theo tầng, độ phủ ở mức 90%: {a:.1%}   (bản tĩnh: {m:.1%})")
    assert abs(a - 0.90) < 0.02, "ACI phai phu gan 90%"

    # 2) khoang phai NO ra khi tang muc tin cay
    g = pan[pan.pair == "EURUSD"].reset_index(drop=True)
    n = int(len(g) * 0.70)
    tr, te = g.iloc[:n], g.iloc[n:]
    kc = KhoangConformal(tr.zT.values, tr.sig.values)
    assert kc.nua_be_rong(0.95, kc.edges[-1]) > kc.nua_be_rong(0.80, kc.edges[-1])

    # 2b) bo tao mac dinh phai dung duoc trong phieu (co n_theo_che_do)
    kmd = khoang_mac_dinh(tr.zT.values, tr.sig.values)
    assert len(kmd.n_theo_che_do) == 3 and min(kmd.n_theo_che_do) > 0
    lo, hi = kmd.khoang(1.0800, float(te.sig.iloc[0]), 0.90)
    assert lo < 1.0800 < hi, "khoang mac dinh phai bao quanh gia"
    print(f"  bộ tạo khoảng mặc định: ACI phân tầng {kmd.n_bins} chế độ, "
          f"mẫu hiệu chuẩn theo tầng {kmd.n_theo_che_do}")
    print("  khoảng nở theo mức tin cậy: ĐẠT")

    # 3) P(cham stop) phai giam khi stop dat xa hon
    ps = [p_cham_stop(k, tr.zT.values) for k in (0.5, 1.0, 2.0, 3.0)]
    assert all(ps[i] > ps[i + 1] for i in range(len(ps) - 1))
    print(f"  P(chạm stop) 0,5σ→3σ: " + " > ".join(f"{x:.1%}" for x in ps) + "  ĐẠT")

    # 4) phieu thuc te — mu la CARRY THAT (trung vi tren huan luyen, cua so
    # mo rong), khong phai hang so bia 0,0002 nhu ban truoc. carry la loi
    # the KELLY duy nhat con hop le vi tang 1 (huong di) da bi bac bo.
    cutoff_eurusd = tr.Date.values[-1]
    mask_tr_full = pan.Date.values <= cutoff_eurusd
    cr_full_eurusd = O.carry_ngay("EURUSD", g.Date.values)
    mask_eurusd = mask_tr_full[pan.pair.values == "EURUSD"]
    mu_carry = float(np.median(cr_full_eurusd[mask_eurusd]))
    print(f"  mu (lợi thế Kelly) = trung vị carry ngày EURUSD trên huấn luyện: "
          f"{mu_carry*1e4:+.3f} bp/phiên — thay hằng số bịa 0,0002 (tầng 1 đã bị "
          f"bác bỏ, carry là lợi thế hợp lệ duy nhất, xem optimal_stop.py)")

    # tang 6b: giai DP (khong don bay — dung chinh sach dang san xuat, xem
    # docs/TANG6B_DUNGTOIUU.md) mot lan cho EURUSD, dung DUNG carry vua tinh
    mau6b, nguong6b = O.rut_mau(pan, mask_tr_full)
    sig_v6b = O.sigma_che_do(pan, mask_tr_full, nguong6b)
    c_thoat6b = O.chi_phi_thoat("EURUSD")
    slip6b, _, _ = O.truot_trung_binh_sigma()
    tang6b = KhuyenNghiGiuDong("EURUSD", mau6b, sig_v6b, nguong6b, c_thoat6b,
                                slip6b, mu_carry, N=20)
    print(f"  tầng 6b: đã giải DP giữ/đóng cho carry={mu_carry*1e4:+.3f} bp/phiên, "
          f"{len(nguong6b)+1} chế độ biến động — kết quả nối vào phiếu bên dưới")

    sizer = PositionSizer(tr.sig.values)
    th = TamHan(tr.sig.values, (tr.zT.values * tr.sig.values))
    pq = PhieuQuyetDinh(sizer, kc, "EURUSD", tr.zT.values, tamhan=th, tang6b=tang6b)
    b = th.bang(float(te.sig.iloc[0]), tr.zT.values, 2.0)
    print("  P(chạm stop 2σ) theo tầm hạn: " + "  ".join(f"{h}→{v:.1%}" for h, v in b))
    assert all(b[i][1] <= b[i + 1][1] + 1e-9 for i in range(len(b) - 1)), "giu lau hon phai rui ro hon"
    assert 0.9 < th.he_so(20, float(kc.edges[0]) * 0.5) < 1.4, "he so tam han o vol thap phai >1"
    gia = 1.0850
    lk = LuatKyHieu(tr.rv5.values)
    gt = lk.doc(te.rv5.values[:3])
    print(f"  luật ký hiệu: {gt if gt else '(ba phiên đầu không khớp luật nào)'}")
    assert lk.doc([1e-9, 1e-9, 1e-9]) is not None, "ba phien vol thap phai khop mot luat"
    r = pq.lap(gia, float(te.sig.iloc[0]), mu_carry, 6.0, dd=0.08, stop_sigma=2.0,
               so_vi_the=3, luat=lk.doc(te.rv5.values[:3]))
    print()
    print(pq.in_ra(r))
    assert r["khuyen_nghi_giu_dong"] is not None, "phieu phai co muc tang 6b"
    print("\n  phiếu đã có mục TẦNG 6b (giữ/đóng): ĐẠT")
    if mu_carry <= 0.0:
        # Vá 03/09/2026: khi carry ÂM, Kelly ≤ 0 — explain() giờ CHẶN đòn bẩy
        # về 0 thay vì trả số âm (xem position_sizing.py). Đây là hành vi
        # ĐÚNG mới lộ ra được nhờ bỏ mu bịa 0,0002 (luôn dương, che mất lỗ hổng).
        assert r["rang_buoc"] == "không có lợi thế (Kelly ≤ 0)" and r["don_bay"] == 0.0, \
            "carry am/khong thi phieu phai bao KHONG CO LOI THE, don bay 0"
        print(f"  carry EURUSD huấn luyện hiện ÂM ({mu_carry*1e4:+.3f} bp/phiên) → phiếu "
              f"đúng đắn báo đòn bẩy 0× (\"không có lợi thế\"), không còn đòn bẩy âm vô nghĩa")

    # 4b) CO CHE dd/so_vi_the van phai hoat dong dung khi CO loi the duong —
    # dung mot mu TONG HOP (khong phai carry that, chi de kiem tra co che,
    # doc lap voi viec carry EURUSD huan luyen dang am hay duong luc chay).
    mu_demo = 0.0003
    rd = pq.lap(gia, float(te.sig.iloc[0]), mu_demo, 6.0, dd=0.08, stop_sigma=2.0, so_vi_the=3)
    rd2 = pq.lap(gia, float(te.sig.iloc[0]), mu_demo, 6.0, dd=0.25, stop_sigma=2.0, so_vi_the=3)
    rd3 = pq.lap(gia, float(te.sig.iloc[0]), mu_demo, 6.0, dd=0.08, stop_sigma=2.0, so_vi_the=1)
    assert rd["don_bay"] < rd3["don_bay"] / 2, "khai 3 vi the phai cat manh"
    assert rd2["don_bay"] < rd["don_bay"], "sut giam sau phai giam co"
    print(f"  [mu tổng hợp dương, kiểm tra riêng cơ chế] cùng phiên nhưng sụt giảm 25%: "
          f"đòn bẩy {rd['don_bay']:.2f}× → {rd2['don_bay']:.2f}×  ĐẠT")

    # 5) canh bao xu huong phai bat dung khi sig cham vung cang thang (dung
    # mu_demo — canh bao nay khong phu thuoc dau cua Kelly)
    sig_cang = sizer.s_stress * 1.5
    r4 = pq.lap(gia, sig_cang, mu_demo, 6.0, dd=0.0, stop_sigma=2.0, so_vi_the=1)
    r5 = pq.lap(gia, sizer.s1 * 0.5, mu_demo, 6.0, dd=0.0, stop_sigma=2.0, so_vi_the=1)
    assert r4["canh_bao_dong_luong"] and not r5["canh_bao_dong_luong"]
    assert "CẢNH BÁO XU HƯỚNG" in pq.in_ra(r4)
    assert "CẢNH BÁO XU HƯỚNG" not in pq.in_ra(r5)
    print("  cảnh báo xu hướng vùng căng thẳng: ĐẠT")

    # 6) khuyen nghi tang 6b phai NHAT QUAN voi dau cua carry: carry am RO
    # RET (100bp/phien, vuot xa moi chi phi hop ly) thi DP phai khuyen dong
    # NGAY o MOI khoang cach — dung nhu lap luan martingale + chi phi trong
    # docstring optimal_stop.py.
    tang6b_am = KhuyenNghiGiuDong("EURUSD", mau6b, sig_v6b, nguong6b, c_thoat6b,
                                   slip6b, -0.01, N=20)
    kn_am = tang6b_am.khuyen_nghi(float(te.sig.iloc[0]), 2.0)
    assert not kn_am["giu_luc_vao"] and kn_am["bien_gioi_sigma"] is None, \
        "carry am ro thi DP phai khuyen dong luon, moi khoang cach"
    print("  tầng 6b tự kiểm: carry âm rõ rệt (-100bp/phiên) → khuyến nghị "
          "ĐÓNG NGAY ở mọi khoảng cách  ĐẠT")
