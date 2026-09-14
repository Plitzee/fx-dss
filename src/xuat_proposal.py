"""Xuat THESIS PROPOSAL ra .docx theo dung khuon de cuong cua khoa.

Khuon lay tu ba de cuong mau da nop (1108 / 1109 / 2001):

  * dau trang: khoi ten truong CAN GIUA ben trai, LOGO AEP ben PHAI
  * "THESIS PROPOSAL" can giua, chu lon
  * TOAN BO noi dung nam trong MOT KHUNG co vien, moi muc la mot hang,
    giua cac hang co duong ke ngang
  * bang Timeline / Work Assignment la bang LONG ben trong khung do
  * o ky cuoi bai la mot khung rieng, hai cot

Logo `docs/assets/aep_logo.png` trich tu chinh file PDF mau nen dung y ban
khoa dang dung, khong phai anh tai lai tu dau khac.

Dung python-docx, KHONG dung pandoc (may nay khong co).

Chay:  python src/xuat_proposal.py
Ghi:   docs/THESIS_PROPOSAL.docx
"""
import io
import os
import re
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MD = os.path.join(ROOT, "docs", "THESIS_PROPOSAL_EN.md")
LOGO = os.path.join(ROOT, "docs", "assets", "aep_logo.png")
RA = os.path.join(ROOT, "docs", "THESIS_PROPOSAL.docx")

FONT = "Times New Roman"
CO = 12           # co chu than bai
CO_BANG = 10.5    # co chu trong bang
MUC_KY = "<!-- KHOI-KY -->"


# ─────────────────────────────────────────────────────────── tien ich
def dat_font(doc):
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(CO)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = st.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15


def vien(bang, ngoai=True, trong_ngang=True, trong_doc=False, mau="000000",
         day="8"):
    """Ke vien cho bang. Tach rieng vien ngoai / ke ngang / ke doc."""
    pr = bang._tbl.tblPr
    for cu in pr.findall(qn("w:tblBorders")):
        pr.remove(cu)
    b = OxmlElement("w:tblBorders")
    chon = {"top": ngoai, "left": ngoai, "bottom": ngoai, "right": ngoai,
            "insideH": trong_ngang, "insideV": trong_doc}
    for canh, bat in chon.items():
        e = OxmlElement(f"w:{canh}")
        e.set(qn("w:val"), "single" if bat else "none")
        e.set(qn("w:sz"), day if bat else "0")
        e.set(qn("w:color"), mau)
        b.append(e)
    pr.append(b)


def dem_o(bang, tren=80, duoi=80, trai=110, phai=110):
    """Dem trong o — khung sat chu qua thi xau."""
    pr = bang._tbl.tblPr
    m = OxmlElement("w:tblCellMar")
    for canh, v in (("top", tren), ("bottom", duoi),
                    ("left", trai), ("right", phai)):
        e = OxmlElement(f"w:{canh}")
        e.set(qn("w:w"), str(v))
        e.set(qn("w:type"), "dxa")
        m.append(e)
    pr.append(m)


def dat_rong(t, rong_cm):
    """Chot be rong tung cot. Word bo qua `columns[i].width` neu con autofit,
    va chi nghe khi width duoc dat tren TUNG O — nen phai lam ca hai."""
    t.autofit = False
    pr = t._tbl.tblPr
    lo = OxmlElement("w:tblLayout")
    lo.set(qn("w:type"), "fixed")
    pr.append(lo)
    for j, w in enumerate(rong_cm):
        if j >= len(t.columns):
            break
        t.columns[j].width = Cm(w)
        for o in t.columns[j].cells:
            o.width = Cm(w)


def gach_duoi(para, day="8"):
    """Duong ke ngang duoi mot doan — dung cho khoi ten truong."""
    pr = para._p.get_or_add_pPr()
    b = OxmlElement("w:pBdr")
    e = OxmlElement("w:bottom")
    e.set(qn("w:val"), "single")
    e.set(qn("w:sz"), day)
    e.set(qn("w:space"), "2")
    e.set(qn("w:color"), "000000")
    b.append(e)
    pr.append(b)


MAU_DD = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)")


def viet(para, text, co=None, dam_het=False):
    """Do **dam**, *nghieng*, `ma` cua markdown vao doan."""
    for phan in MAU_DD.split(text):
        if not phan:
            continue
        if phan.startswith("**") and phan.endswith("**"):
            r = para.add_run(phan[2:-2]); r.bold = True
        elif phan.startswith("*") and phan.endswith("*"):
            r = para.add_run(phan[1:-1]); r.italic = True
        elif phan.startswith("`") and phan.endswith("`"):
            r = para.add_run(phan[1:-1]); r.font.name = "Consolas"
        else:
            r = para.add_run(phan)
        if not r.font.name:
            r.font.name = FONT
        r.font.size = Pt(co or CO)
        if dam_het:
            r.bold = True
    return para


def doan_trong(o, dau=True):
    """Doan dau tien cua o da co san; tu doan thu hai tro di phai them."""
    if dau and o.paragraphs and not o.paragraphs[0].text:
        return o.paragraphs[0]
    return o.add_paragraph()


# ─────────────────────────────────────────────────── doc markdown
def doc_md():
    raw = io.open(MD, encoding="utf-8").read()
    than = raw.split(MUC_KY)[0]
    dong = than.split("\n")

    dau = {}          # THESIS TITLE / Advisor / Implementation period / Students
    sinh_vien = []
    muc = []          # [(ten_muc, [khoi...])]
    ten_nay, khoi_nay = None, []
    i = 0
    while i < len(dong):
        ln = dong[i].rstrip()
        t = ln.strip()

        if t.startswith("# ") or t.startswith("## ADVANCED"):
            i += 1; continue
        if t == "---" or not t:
            i += 1; continue

        # front matter
        if t.startswith("**THESIS TITLE:"):
            dau["title"] = t.strip("*").replace("THESIS TITLE:", "").strip()
            i += 1; continue
        if t.startswith("**Advisor:**"):
            dau["advisor"] = t.replace("**Advisor:**", "").strip()
            i += 1; continue
        if t.startswith("**Implementation period:**"):
            dau["period"] = t.replace("**Implementation period:**", "").strip()
            i += 1; continue
        if t == "**Students:**":
            i += 1
            while i < len(dong):
                u = dong[i].strip()
                if not u:
                    i += 1; continue
                if u.startswith("---") or u.startswith("##"):
                    break
                sinh_vien.append(u)
                i += 1
            continue

        # tieu muc trong mot muc (### ...) — phai bat TRUOC "## " vi chuoi
        # "### X" cung thoa startswith("## ")
        if t.startswith("### "):
            khoi_nay.append(("tieu_muc", t[4:].strip()))
            i += 1; continue

        # muc moi
        if t.startswith("## "):
            if ten_nay:
                muc.append((ten_nay, khoi_nay))
            ten_nay, khoi_nay = t[3:].strip(), []
            i += 1; continue

        # bang
        if t.startswith("|") and i + 1 < len(dong) and \
                set(dong[i+1].replace("|", "").strip()) <= set("-: "):
            hang = []
            while i < len(dong) and dong[i].lstrip().startswith("|"):
                o = [c.strip() for c in dong[i].strip().strip("|").split("|")]
                if not set("".join(o)) <= set("-: "):
                    hang.append(o)
                i += 1
            khoi_nay.append(("bang", hang))
            continue

        # gach dau dong
        if t.startswith("- "):
            khoi_nay.append(("cham", t[2:]))
            i += 1; continue

        khoi_nay.append(("doan", t))
        i += 1

    if ten_nay:
        muc.append((ten_nay, khoi_nay))
    return dau, sinh_vien, muc


# ──────────────────────────────────────────────── dung tung phan
def dung_dau_trang(doc):
    """Ten truong CAN GIUA ben trai + logo AEP ben PHAI — nhu ba ban mau."""
    t = doc.add_table(rows=1, cols=2)
    vien(t, ngoai=False, trong_ngang=False)
    dem_o(t, 0, 0, 0, 0)
    # cot trai phai du rong de "UNIVERSITY OF INFORMATION TECHNOLOGY" nam
    # TRON MOT DONG nhu ban mau — hep hon la no tu xuong dong, trong rat xau
    dat_rong(t, [10.4, 5.6])
    o_trai, o_phai = t.rows[0].cells

    for k, s in enumerate(["UNIVERSITY OF INFORMATION TECHNOLOGY",
                           "ADVANCED PROGRAM",
                           "IN INFORMATION SYSTEMS"]):
        p = doan_trong(o_trai, dau=(k == 0))
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(s)
        r.bold = True
        r.font.name = FONT
        r.font.size = Pt(11)
        if k == 2:
            gach_duoi(p)

    p = doan_trong(o_phai, dau=True)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if os.path.exists(LOGO):
        p.add_run().add_picture(LOGO, width=Cm(4.6))
    else:
        print(f"  CANH BAO: thieu {LOGO} — bo qua logo")

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run("THESIS PROPOSAL")
    r.bold = True
    r.font.name = FONT
    r.font.size = Pt(19)


def do_bang_long(o, hang):
    """Bang LONG trong mot o cua khung ngoai (Timeline / Work Assignment)."""
    nc = max(len(h) for h in hang)
    t = o.add_table(rows=0, cols=nc)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    vien(t, ngoai=True, trong_ngang=True, trong_doc=True)
    dem_o(t, 40, 40, 80, 80)
    # Be rong theo VAI TRO cua cot, nhan qua tieu de — de Word tu chia thi cot
    # "No." rong bang cot "Assignments", nhin rat lech.
    # Tong be rong = be rong LONG trong khung ngoai, de bang lap day khung
    # thay vi thut vao giua trong mot khoang trang hai ben.
    dau_bang = [c.lower() for c in hang[0]]
    if nc == 3 and dau_bang[0].startswith("no"):            # Timeline
        dat_rong(t, [1.4, 11.2, 3.6])
    elif nc == 3 and "member" in dau_bang[1]:               # Work Assignment
        dat_rong(t, [9.4, 3.4, 3.4])
    for k, h in enumerate(hang):
        r_ = t.add_row()
        # KHONG cho mot hang bi cat doi qua trang — de mac dinh thi hang cuoi
        # bi xe ra, o "No." va "Timeline" tro thanh hai o trong o trang sau.
        trPr = r_._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
        if k == 0:                      # lap lai hang tieu de o moi trang moi
            trPr.append(OxmlElement("w:tblHeader"))
        c = r_.cells
        for j in range(nc):
            txt = h[j] if j < len(h) else ""
            c[j].text = ""
            dau_o = True
            for d in [x.strip() for x in re.split(r"<br\s*/?>", txt)]:
                p = doan_trong(c[j], dau=dau_o); dau_o = False
                p.paragraph_format.space_after = Pt(1)
                p.alignment = (WD_ALIGN_PARAGRAPH.CENTER
                               if re.fullmatch(r"\d+%?|Week .*|\d+", d)
                               else WD_ALIGN_PARAGRAPH.LEFT)
                viet(p, d, co=CO_BANG, dam_het=(k == 0))
    # Word BAT BUOC co mot doan sau bang long (khong co thi bang dinh lien o
    # ke tiep). De co chu 1pt cho no gan nhu tang hinh, khong tao khoang ho.
    q = o.add_paragraph()
    q.paragraph_format.space_after = Pt(0)
    q.paragraph_format.space_before = Pt(0)
    q.add_run("").font.size = Pt(1)


def dung_khung(doc, dau, sinh_vien, muc):
    """MOT khung bao tron noi dung; moi muc mot hang, giua co ke ngang."""
    t = doc.add_table(rows=0, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    vien(t, ngoai=True, trong_ngang=True)
    dem_o(t)

    def hang_moi():
        return t.add_row().cells[0]

    # ── THESIS TITLE
    o = hang_moi()
    p = doan_trong(o)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    viet(p, f"**THESIS TITLE: {dau.get('title','')}**")

    # ── Advisor / Implementation period
    for nhan, khoa in (("Advisor:", "advisor"),
                       ("Implementation period:", "period")):
        o = hang_moi()
        p = doan_trong(o)
        r = p.add_run(nhan + " "); r.bold = True
        r.font.name = FONT; r.font.size = Pt(CO)
        viet(p, dau.get(khoa, ""))

    # ── Students
    o = hang_moi()
    p = doan_trong(o)
    r = p.add_run("Students:"); r.bold = True
    r.font.name = FONT; r.font.size = Pt(CO)
    for sv in sinh_vien:
        q = o.add_paragraph()
        q.paragraph_format.left_indent = Cm(1.2)
        q.paragraph_format.space_after = Pt(2)
        viet(q, sv)

    # ── cac muc
    for ten, khoi in muc:
        o = hang_moi()
        p = doan_trong(o)
        p.paragraph_format.space_after = Pt(5)
        r = p.add_run(ten + ":"); r.bold = True
        r.font.name = FONT; r.font.size = Pt(CO)

        for loai, noi_dung in khoi:
            if loai == "bang":
                do_bang_long(o, noi_dung)
            elif loai == "tieu_muc":
                q = o.add_paragraph()
                q.paragraph_format.space_before = Pt(7)
                q.paragraph_format.space_after = Pt(3)
                r = q.add_run(noi_dung)
                r.bold = True
                r.italic = True
                r.font.name = FONT
                r.font.size = Pt(CO)
            elif loai == "cham":
                q = o.add_paragraph(style="List Bullet")
                q.paragraph_format.left_indent = Cm(0.8)
                q.paragraph_format.space_after = Pt(3)
                q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                viet(q, noi_dung)
            else:
                q = o.add_paragraph()
                q.paragraph_format.space_after = Pt(5)
                q.paragraph_format.first_line_indent = Cm(0.6)
                q.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                viet(q, noi_dung)
    return t


def dung_o_ky(doc, sinh_vien):
    """Khung ky cuoi bai — hai cot, y nhu ban mau."""
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    vien(t, ngoai=True, trong_ngang=False, trong_doc=True)
    dem_o(t, 110, 110, 110, 110)
    dat_rong(t, [7.8, 8.0])
    # O ky KHONG duoc xe doi qua trang — ten thay va ten sinh vien phai nam
    # cung mot trang voi dong "Advisor's approval".
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    trai, phai = t.rows[0].cells

    def d(o, txt, dam=False, ngh=False, dau=False, cach=6):
        p = doan_trong(o, dau=dau)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(cach)
        r = p.add_run(txt)
        r.bold, r.italic = dam, ngh
        r.font.name = FONT
        r.font.size = Pt(CO)
        return p

    d(trai, "Advisor's approval", dam=True, dau=True)
    d(trai, "(Signature and full name)", ngh=True, cach=52)
    d(trai, "NGUYEN DINH THUAN", dam=True)

    d(phai, "Ho Chi Minh City, ... / ... / ......", dau=True)
    d(phai, "Students", dam=True)
    d(phai, "(Signatures and full names)", ngh=True, cach=44)
    for k, sv in enumerate(sinh_vien):
        ten = sv.split("–")[0].split("-")[0].strip()
        d(phai, ten, dam=True, cach=34 if k == 0 else 6)


def main():
    if not os.path.exists(MD):
        sys.exit(f"Thieu {MD}")
    dau, sinh_vien, muc = doc_md()

    doc = Document()
    dat_font(doc)
    for s in doc.sections:
        s.top_margin = Cm(1.8)
        s.bottom_margin = Cm(1.8)
        s.left_margin = Cm(2.2)
        s.right_margin = Cm(1.8)

    dung_dau_trang(doc)
    khung = dung_khung(doc, dau, sinh_vien, muc)
    dung_o_ky(doc, sinh_vien)
    doc.save(RA)

    n_bang_long = sum(1 for _, k in muc for l, _ in k if l == "bang")
    print("=" * 78)
    print("XUAT THESIS PROPOSAL")
    print("=" * 78)
    print(f"  nguon      docs/THESIS_PROPOSAL_EN.md")
    print(f"  logo       {'CO — ' + os.path.basename(LOGO) if os.path.exists(LOGO) else 'THIEU'}")
    print(f"  khung      1 khung bao tron, {len(khung.rows)} hang")
    print(f"  muc        {len(muc)}: {', '.join(t for t, _ in muc)}")
    print(f"  bang long  {n_bang_long}")
    print(f"  sinh vien  {len(sinh_vien)}")
    print(f"  ra         docs/THESIS_PROPOSAL.docx "
          f"({os.path.getsize(RA)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
