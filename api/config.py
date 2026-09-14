"""Duong dan, thu vien va hang so DUNG CHUNG cho toan bo API.

Tach rieng khoi logic tinh toan de moi module con lai (cache, routers) import
tu MOT nguon duy nhat — tranh tinh trang mot noi doi PAIRS/HS/NEN_THEO_H con
noi khac van dung gia tri cu sau khi sua.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
D = os.path.join(ROOT, "data")
LIVE = os.path.join(D, "live")
WEB = os.path.join(ROOT, "web")

import pandas as pd

import balop as B                                      # noqa: E402
import chibao as CB                                    # noqa: E402
import conformal as CF                                 # noqa: E402
import volfc2 as V2                                    # noqa: E402
from volfc import merge_thin_days                      # noqa: E402
from split import doan, VALID_TU, TEST_TU              # noqa: E402

PAIRS = B.PAIRS
MOC_NOI = pd.Timestamp("2026-01-01")
PIP = {"USDJPY": 0.01}
HS = (1, 5, 20)
# Chon theo DIEM LOG tren doan KIEM DINH (output/nen3.json). Cua so mo rong —
# khop lai phan phoi z moi ~21 phien tren TOAN BO qua khu thay vi dong bang o
# 2021-10 — cai thien TUNG NEN mot cach nhat quan: 6/6 khong xau di ve BSS,
# va ECE tot hon o CA 6/6. Nhung to hop truc tuyen thi XAU DI ro o tam han dai khi chuyen gia
# cua no tro thanh nen cuon: h=5 tut +0,0134 -> +0,0089, h=20 tut +0,0200 ->
# +0,0055. Do la ket qua do duoc, khong giau.
#   h=1   TO HOP (chuyen gia cuon)  log 1,0866 · BSS +0,0107 · MCE 0,0611
#   h=5   SIGMA+CHE DO (cuon)       log 1,0804 · BSS +0,0148 · MCE 0,0471
#   h=20  SIGMA+CHE DO (cuon)       log 1,0744 · BSS +0,0148 · MCE 0,2058  (*)
# (*) h=20 danh doi that: BSS +0,0137 -> +0,0148 nhung MCE 0,138 -> 0,206. Quy
#     tac chon da chot TRUOC la diem log, nen van lay ban cuon; MCE xau di phai
#     bao cao tren giao dien va la viec cua lop hieu chuan lai o vong sau.
NEN_THEO_H = {1: "tổ hợp trực tuyến", 5: "σ̂ + chế độ (cuộn)",
              20: "σ̂ + chế độ (cuộn)"}

# Cua so hieu chuan cuon cho ACI (docs/CHISO_DANHGIA.md muc 16).
CF_CUA_SO = 500

# KY NANG DO DUOC theo tung tam han — HAI phep do DOC LAP, cung mot ket luan:
#
#   walk-forward theo nam (muc 14)   h=1: 14/14 nam BSS duong
#                                    h=5: 10/14 · h=20: 8/14
#   thong tin conformal (muc 16)     h=1: +0,10 lop tren kiem tra
#                                    h=5: -0,09 · h=20: -0,10
#   theo cap, kiem tra (kiem_ngan_han.py)
#                                    h=1: 6/6 cap BSS duong CO Y NGHIA
#
# Ky nang cua he thong nam o TAM HAN 1 PHIEN. O 5 va 20 phien, tap du bao cua
# mo hinh KHONG nho hon tap cua mot hang so — tuc khong loai tru them duoc gi.
# Giao dien phai noi dung nhu the, khong duoc trinh bay ba o nhu nhau.
KY_NANG_THEO_H = {
    1: dict(muc="có kỹ năng đo được",
            chi_tiet="BSS dương 14/14 năm; 6/6 cặp có ý nghĩa trên kiểm tra; "
                     "tập conformal nhỏ hơn mốc khí hậu học 0,10–0,21 lớp"),
    5: dict(muc="kỹ năng không tách được khỏi 0",
            chi_tiet="BSS dương 10/14 năm; tập conformal KHÔNG nhỏ hơn mốc "
                     "khí hậu học (−0,09 lớp) — dùng để tham khảo, không để "
                     "ra quyết định"),
    20: dict(muc="kỹ năng không tách được khỏi 0",
             chi_tiet="BSS dương 8/14 năm; tập conformal KHÔNG nhỏ hơn mốc "
                      "khí hậu học (−0,10 lớp) — dùng để tham khảo, không để "
                      "ra quyết định"),
}

KHUNG = ("D1", "H1", "M15", "M5")
MUC_VAR = (0.05, 0.01)
