"""ĐIỀU PHỐI — chạy 12 nhánh khai phá quy luật (Giai đoạn 2) cho các tổ hợp
(tầm hạn h, mục tiêu) còn thiếu, để đóng tiêu chí dừng mục 10.4 của
`docs/REPLAN_2026.md` ("cả hai mục tiêu R/P và cả ba tầm hạn 1/5/20").

Đã có sẵn: h=1 × P (chạy gốc, không đổi).
Còn thiếu, script này chạy: h=1×R, h=5×R, h=5×P, h=20×R, h=20×P.

CÁCH DÙNG BIẾN MÔI TRƯỜNG: `run_quyluat.nap_du_lieu()` đọc `QUYLUAT_H` /
`QUYLUAT_TARGET` để quyết định tầm hạn/mục tiêu — xem docstring của nó để
biết vì sao đặc trưng `zs` LUÔN cố định ở h=1 bất kể tham số này (chống rò
rỉ khi h>1). Mỗi script con chạy trong TIẾN TRÌNH RIÊNG (subprocess) vì
`KHOI` (độ dài khối hoán vị) được tính MỘT LẦN lúc nạp module — chạy trong
cùng tiến trình sẽ dùng nhầm giá trị KHOI của tổ hợp trước.

Tên file output CỐ ĐỊNH trong mỗi script con (vd `output/h2_motif.json`) —
script này ĐỔI TÊN ngay sau mỗi lần chạy, thêm hậu tố `_h{h}_{target}`,
trước khi chạy tổ hợp kế tiếp, để không ghi đè lẫn nhau.

Chạy:  python src/run_giaidoan2_tamhan.py
Ghi:   output/log_giaidoan2_tamhan.txt (tiến độ), output/*_h{h}_{target}.json
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(ROOT, "output")
LOG = os.path.join(OUT, "log_giaidoan2_tamhan.txt")

# 10 script khai phá + 1 script SPA — ĐÚNG thứ tự trong báo cáo điều tra.
# (ten_script, ten_output_json, ten_log_goc_neu_co)
SCRIPTS = [
    ("run_quyluat.py", "quyluat.json"),
    ("run_h2_motif.py", "h2_motif.json"),
    ("run_h3_rulelist.py", "h3_rulelist.json"),
    ("run_h5_chedo.py", "h5_chedo.json"),
    ("run_h6_hmm.py", "h6_hmm.json"),
    ("run_h7_matrixprofile.py", "h7_matrixprofile.json"),
    ("run_h8_tintuc.py", "h8_tintuc.json"),
    ("run_h8b_embedding.py", "h8b_embedding.json"),
    ("run_h8c_chude.py", "h8c_chude.json"),
    ("run_h8e_llm.py", "h8e_llm.json"),
    ("run_spa_ho2.py", "spa_ho2.json"),
]

# Tổ hợp còn thiếu — h=1xP đã có sẵn (chạy gốc), KHÔNG chạy lại ở đây.
TO_HOP = [(1, "R"), (5, "R"), (5, "P"), (20, "R"), (20, "P")]


def ghi_log(dong):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(dong + "\n")
    print(dong, flush=True)


def chay_mot_to_hop(h, target):
    ghi_log(f"\n{'='*90}\nTỔ HỢP h={h} target={target} — bắt đầu {time.strftime('%H:%M:%S')}")
    env = dict(os.environ)
    env["QUYLUAT_H"] = str(h)
    env["QUYLUAT_TARGET"] = target
    env["PYTHONIOENCODING"] = "utf-8"

    for script, out_json in SCRIPTS:
        dst_out_kiemtra = os.path.join(OUT, out_json.replace(".json", f"_h{h}_{target}.json"))
        if os.path.exists(dst_out_kiemtra):
            ghi_log(f"  [BỎ QUA — đã có] {script} -> {os.path.basename(dst_out_kiemtra)}")
            continue
        t0 = time.time()
        src_path = os.path.join(SRC, script)
        log_path_run = os.path.join(OUT, f"log_{out_json.replace('.json','')}_h{h}_{target}.txt")
        try:
            with open(log_path_run, "w", encoding="utf-8") as flog:
                r = subprocess.run([sys.executable, src_path], cwd=ROOT, env=env,
                                   stdout=flog, stderr=subprocess.STDOUT, timeout=3600)
            dt = time.time() - t0
            if r.returncode != 0:
                ghi_log(f"  [LỖI] {script}  ({dt:.0f}s, mã thoát {r.returncode}) — "
                       f"xem {log_path_run}")
                continue
            src_out = os.path.join(OUT, out_json)
            if os.path.exists(src_out):
                dst_out = os.path.join(OUT, out_json.replace(".json", f"_h{h}_{target}.json"))
                shutil.move(src_out, dst_out)
                ghi_log(f"  [OK] {script}  ({dt:.0f}s) -> {os.path.basename(dst_out)}")
            else:
                ghi_log(f"  [CẢNH BÁO] {script} chạy xong ({dt:.0f}s) nhưng không thấy "
                       f"{out_json} — kiểm tra {log_path_run}")
        except subprocess.TimeoutExpired:
            ghi_log(f"  [TIMEOUT] {script} sau 3600s — bỏ qua, xem {log_path_run}")
        except Exception as e:
            ghi_log(f"  [NGOẠI LỆ] {script}: {e}")

    ghi_log(f"TỔ HỢP h={h} target={target} — xong {time.strftime('%H:%M:%S')}")


def sao_luu_mo_h1P():
    """Sao lưu KẾT QUẢ GỐC h=1×P trước khi chạy bất ky to hop nao — cac
    script con ghi vao CUNG duong dan co dinh, nen neu quy trinh nay bi ngat
    giua chung (crash, mat dien) sau khi mot subprocess da ghi de nhung
    TRUOC khi doi ten, ban goc se mat neu khong co sao luu nay."""
    ghi_log("Sao lưu baseline h=1×P gốc trước khi bắt đầu...")
    for _, out_json in SCRIPTS:
        p = os.path.join(OUT, out_json)
        if os.path.exists(p):
            bak = os.path.join(OUT, out_json.replace(".json", "_BACKUP_h1_P_goc.json"))
            if not os.path.exists(bak):        # khong ghi de sao luu da co tu lan chay truoc
                shutil.copy2(p, bak)
                ghi_log(f"  sao lưu {out_json} -> {os.path.basename(bak)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    sao_luu_mo_h1P()
    ghi_log(f"\n{'#'*90}\nBẮT ĐẦU {len(TO_HOP)} tổ hợp — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_bat_dau = time.time()
    for h, target in TO_HOP:
        chay_mot_to_hop(h, target)
    ghi_log(f"\nXONG TẤT CẢ {len(TO_HOP)} tổ hợp — tổng {(time.time()-t_bat_dau)/60:.1f} phút")


if __name__ == "__main__":
    main()
