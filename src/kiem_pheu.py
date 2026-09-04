"""KIEM CHUNG CHINH CAI PHEU — pheu giai doan 2 co bat duoc quy luat that khong?

Ly do file nay ton tai. `run_quyluat.py` bao: 1.890 gia thuyet -> 9 song sot
Westfall-Young -> 0 sau doi chung. Nhung con so 0 do dang MO HO giua hai kha
nang hoan toan khac nhau:

    (a) khong co quy luat nao              <- ket luan khoa hoc
    (b) pheu qua chat, khong bat duoc gi   <- loi thiet ke

Khong phan biet duoc hai cai do thi "0/1.890" khong phat bieu duoc thanh cau gi
ca. Day la cho mot phan bien tot se dam vao dau tien.

HAI PHEP CHAY

  DOI CHUNG AM  — xao tron KHOI ket cuc roi chay pheu. Phai ra ~0. Ra nhieu hon
                  la co ro ri o dau do TRONG CHINH PHEU.

  DOI CHUNG DUONG — tiem mot quy luat tong hop co DO MANH BIET TRUOC vao du lieu
                  that, chay pheu, xem no co song khong. Quet nhieu muc lift de
                  tim HIEU UNG NHO NHAT PHAT HIEN DUOC (luc 80%).

Sau do ket luan doi tu "chung toi khong tim thay gi" thanh "chung toi khong tim
thay gi, va pheu nay bat duoc lift >= X voi luc 80% — nen moi quy luat manh hon
X da bi loai tru tren du lieu nay". Do moi la mot phat bieu khoa hoc.

CACH TIEM TIN HIEU. Chon mot vi tu THAT du so lan khop lam gia do. Voi cac hang
no khop, doi ket cuc sang lop dich voi xac suat vua du de P(lop | khop) dat muc
lift mong muon. Hang khong khop giu nguyen. Nhu vay cau truc tu tuong quan cua
chuoi duoc giu, va do manh tin hieu la con so ta DAT ra nen doc duoc truc tiep.

VI SAO KHONG CHAY LAI W-Y DAY DU MOI LAN. Westfall-Young hieu chinh bang phan
phoi cua max|z| duoi null khoi. Phan phoi do la tinh chat cua MA TRAN VI TU va
cau truc khoi cua chuoi — khong phai cua tin hieu ta tiem vao. Nen uoc GIA TRI
TOI HAN mot lan roi dung lai cho moi lan lap. Day la cach lam chuan cua nghien
cuu luc phat hien, va nhanh hon khoang 100 lan.

Chay:  python src/kiem_pheu.py
Ghi:   output/kiem_pheu.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

import run_quyluat as Q                                      # noqa: E402
import balop as B                                            # noqa: E402
import volfc2 as V2                                          # noqa: E402
from split import TEST_TU, doan                              # noqa: E402
from volfc import merge_thin_days                            # noqa: E402

LIFTS = (1.02, 1.05, 1.10, 1.15, 1.20, 1.35, 1.50)
N_LAP = 60                  # so lan lap moi muc lift
N_AM = 10                   # so lan lap doi chung am
NPERM_NGUONG = 1000         # hoan vi de uoc gia tri toi han — chay MOT lan
MIN_KHOP_TIEM = 300         # vi tu phai khop it nhat bay nhieu de lam gia do
SEED = 12345
EPS = 1e-12


def chuan_bi():
    """Dung lai DUNG duong ma run_quyluat di — khong viet lai duong khac."""
    from api.main import noi_chuoi
    Ms, sigs, zs, ys, dts = [], [], [], [], []
    for p in B.PAIRS:
        m = merge_thin_days(noi_chuoi(p))
        sig = np.sqrt(np.maximum(V2.du_bao_san_xuat(m, p), 0.0))
        d = pd.DataFrame({"Date": m.Date.values, "sig": sig})
        c = m.close.values
        zt = np.full(len(m), np.nan)
        zt[1:] = np.log(c[1:] / np.maximum(c[:-1], EPS)) / np.maximum(sig[1:], EPS)
        d["zT"] = zt
        T = B.dung_muc_tieu(d, Q.H, doan(d.Date.values) == 0)
        yv = np.full(len(m), -1)
        yv[:-1] = T["yP"][1:]                # dac trung tai t noi ve lop t+1
        Ms.append(m); sigs.append(sig); zs.append(T["z"]); ys.append(yv)
        dts.append(d.Date.values)

    lit_all, ten_lit = [], None
    for i in range(len(B.PAIRS)):
        F = Q.dac_trung(Ms[i], sigs[i], zs[i])
        L, tn = Q.roi_rac(F, doan(dts[i]) == 0)
        lit_all.append(L)
        ten_lit = tn
    lit = np.concatenate(lit_all, axis=1)
    y = np.concatenate(ys)
    dt = pd.DatetimeIndex(np.concatenate(dts))

    import optimal_stop as OS
    nt_usd = Q.nhan_to_usd(Ms, dts)
    carry = []
    for i, p in enumerate(B.PAIRS):
        try:
            carry.append(np.asarray(OS.carry_ngay(p, dts[i]), float))
        except Exception:
            carry.append(np.full(len(dts[i]), np.nan))
    kiem_soat = np.column_stack([
        np.log(np.maximum(np.concatenate(sigs), EPS)),
        np.concatenate([pd.Series(np.r_[np.nan, np.diff(np.log(np.maximum(
            m.close.values, EPS)))]).rolling(20).sum().values for m in Ms]),
        np.concatenate(nt_usd),
        np.concatenate(carry),
    ])
    M, ten = Q.vet_can(lit, ten_lit)
    pha = (dt < TEST_TU) & (y >= 0)
    return M, ten, y, kiem_soat, pha


def xao_tron_khoi(y, khoi, rng):
    """Xao tron theo KHOI de giu tinh dai trong chuoi ket cuc."""
    n = len(y)
    idx = np.arange(n)
    cac_khoi = [idx[i:i + khoi] for i in range(0, n, khoi)]
    rng.shuffle(cac_khoi)
    return y[np.concatenate(cac_khoi)[:n]]


def tiem(y, mkhop, lop, lift, pha, rng):
    """Doi ket cuc o cac hang KHOP sao cho P(lop | khop) = lift x P(lop)."""
    y2 = y.copy()
    ok = pha & mkhop & (y >= 0)
    if ok.sum() < 50:
        return y2, np.nan
    p0 = float((y[pha & (y >= 0)] == lop).mean())
    hien = float((y[ok] == lop).mean())
    dich = min(0.97, lift * p0)
    if dich <= hien:
        return y2, hien / max(p0, EPS)
    can = (dich - hien) / max(1.0 - hien, EPS)
    vt = np.where(ok & (y != lop))[0]
    y2[vt[rng.random(len(vt)) < can]] = lop
    return y2, float((y2[ok] == lop).mean() / max(p0, EPS))


def gia_tri_toi_han(M, y, pha):
    """max|z| duoi null khoi, phan vi 95% — chay DUNG MOT LAN."""
    _, _, _, _, ng = Q.westfall_young(M, y, pha, nperm=NPERM_NGUONG, seed=SEED)
    return float(ng[1])


def qua_hai_cua(M, y, kiem_soat, pha, i, c, gtth):
    """Vi tu (i, c) co qua ca hai cua chat nhat khong: W-Y roi dieu kien hoa."""
    Z, _, _ = Q.z_lift(M, y, pha)
    z = Z[i, c]
    if not (np.isfinite(z) and abs(z) > gtth):
        return False, (float(z) if np.isfinite(z) else np.nan)
    _, t = Q.doi_chung(M[i], y, c, kiem_soat)
    return bool(np.isfinite(t) and abs(t) > Q.T_DIEU_KIEN), float(z)


def main():
    t0 = time.time()
    print("=" * 100)
    print("KIỂM CHỨNG PHỄU GIAI ĐOẠN 2 — đối chứng âm và đối chứng dương")
    print("=" * 100, flush=True)
    M, ten, y, kiem_soat, pha = chuan_bi()
    ngt = M.shape[0] * 3
    print(f"{M.shape[0]:,} vị từ × 3 lớp = {ngt:,} giả thuyết · "
          f"{int(pha.sum()):,} hàng phát hiện", flush=True)
    ra = {"n_gia_thuyet": int(ngt), "n_hang": int(pha.sum()),
          "nperm_nguong": NPERM_NGUONG, "lifts": list(LIFTS),
          "n_lap": N_LAP, "t_dieu_kien": Q.T_DIEU_KIEN}

    print(f"\n[0/2] Giá trị tới hạn max|z| dưới null khối "
          f"({NPERM_NGUONG} hoán vị)…", flush=True)
    gtth = gia_tri_toi_han(M, y, pha)
    ra["gia_tri_toi_han"] = gtth
    print(f"      giá trị tới hạn 95% = {gtth:.2f}", flush=True)

    # ── ĐỐI CHỨNG ÂM ────────────────────────────────────────────────────
    print(f"\n[1/2] ĐỐI CHỨNG ÂM — xáo trộn khối kết cục, phễu phải ra ~0",
          flush=True)
    rng = np.random.default_rng(SEED)
    am = []
    for lap in range(N_AM):
        ys = xao_tron_khoi(y, Q.KHOI, rng)
        Zs, _, _ = Q.z_lift(M, ys, pha)
        vuot = np.where(np.isfinite(Zs) & (np.abs(Zs) > gtth))
        q = 0
        for i, c in zip(*vuot):
            _, t = Q.doi_chung(M[i], ys, c, kiem_soat)
            if np.isfinite(t) and abs(t) > Q.T_DIEU_KIEN:
                q += 1
        am.append({"lan": lap, "vuot_nguong": int(len(vuot[0])), "qua_dieu_kien": q})
        print(f"      lần {lap+1:>2}: vượt ngưỡng {len(vuot[0]):>3} · "
              f"qua điều kiện hoá {q}", flush=True)
    ra["doi_chung_am"] = am
    tb_am = float(np.mean([a["qua_dieu_kien"] for a in am]))
    print(f"      → trung bình {tb_am:.1f} dương tính giả / {ngt:,} giả thuyết",
          flush=True)

    # ── ĐỐI CHỨNG DƯƠNG ─────────────────────────────────────────────────
    print(f"\n[2/2] ĐỐI CHỨNG DƯƠNG — tiêm quy luật đã biết, {N_LAP} lần mỗi mức",
          flush=True)
    nk = np.array([(M[i] & pha).sum() for i in range(M.shape[0])])
    ung_vien = np.where(nk >= MIN_KHOP_TIEM)[0]
    print(f"      {len(ung_vien):,} vị từ đủ ≥{MIN_KHOP_TIEM} lần khớp làm giá đỡ\n",
          flush=True)
    print(f"      {'lift đặt':>9}{'lift thực':>11}{'|z| trung vị':>14}"
          f"{'bắt được':>12}{'lực':>9}", flush=True)
    duong = []
    for lift in LIFTS:
        bat, lts, zs_ = 0, [], []
        for lap in range(N_LAP):
            r2 = np.random.default_rng(SEED + 7919 * lap + int(lift * 1000))
            i = int(r2.choice(ung_vien))
            c = int(r2.integers(0, 3))
            y2, lt = tiem(y, M[i], c, lift, pha, r2)
            lts.append(lt)
            ok, z = qua_hai_cua(M, y2, kiem_soat, pha, i, c, gtth)
            zs_.append(abs(z))
            bat += int(ok)
        luc = bat / N_LAP
        duong.append({"lift": lift, "lift_that": float(np.nanmean(lts)),
                      "z_trungvi": float(np.nanmedian(zs_)),
                      "bat": bat, "luc": luc})
        print(f"      {lift:>9.2f}{np.nanmean(lts):>11.3f}"
              f"{np.nanmedian(zs_):>14.2f}{bat:>8}/{N_LAP:<4}{luc:>8.0%}",
              flush=True)
    ra["doi_chung_duong"] = duong

    dat = [d for d in duong if d["luc"] >= 0.80]
    mdes = dat[0]["lift"] if dat else None
    ra["lift_nho_nhat_luc80"] = mdes
    print("\n" + "=" * 100)
    if mdes is not None:
        print(f"HIỆU ỨNG NHỎ NHẤT PHÁT HIỆN ĐƯỢC (lực 80%): lift = {mdes:.2f}")
        print(f"→ Kết luận 0/{ngt:,} nay phát biểu được thành: phễu này bắt được")
        print(f"  quy luật có lift ≥ {mdes:.2f} với xác suất ≥ 80%, nên mọi quy luật")
        print(f"  mạnh hơn thế đã bị loại trừ trên dữ liệu này.")
    else:
        print(f"KHÔNG mức lift nào trong {LIFTS} đạt lực 80%.")
        print("→ Phễu quá chặt so với dải đã quét. Kết luận âm PHẢI kèm câu này:")
        print("  không loại trừ được quy luật yếu hơn mức mạnh nhất đã quét.")
    print(f"Dương tính giả trên nhiễu thuần: {tb_am:.1f}/{ngt:,}")
    print("=" * 100)

    with open(os.path.join(OUT, "kiem_pheu.json"), "w", encoding="utf-8") as f:
        json.dump(ra, f, ensure_ascii=False, indent=1)
    print(f"đã ghi output/kiem_pheu.json · {time.time()-t0:.0f}s")
    print("TỰ KIỂM ĐẠT")


if __name__ == "__main__":
    main()
