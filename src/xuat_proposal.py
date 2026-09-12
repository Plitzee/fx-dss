"""Xuat THESIS PROPOSAL ra .docx theo dung khuon cua khoa.

Doc `docs/THESIS_PROPOSAL_EN.md` roi dung lai bang python-docx, KHONG dung
pandoc (may nay khong co). Khuon lay tu ba ban mau da nop cua cac nhom khac:
Times New Roman 13, can deu hai ben, tieu de muc in dam, bang co vien.

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
from docx.shared import Pt, Cm, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MD = os.path.join(ROOT, "docs", "THESIS_PROPOSAL_EN.md")
RA = os.path.join(ROOT, "docs", "THESIS_PROPOSAL.docx")

FONT = "Times New Roman"
CO = 13          # co chu than bai, pt


def _font(doc):
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(CO)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = st.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15


def _vien(bang):
    """Ke vien day du cho bang — python-docx khong co san."""
    tbl = bang._tbl
    pr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for canh in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{canh}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "6")
        e.set(qn("w:color"), "808080")
        borders.append(e)
    pr.append(borders)


def _dam_nghieng(para, text):
    """Doi **dam**, *nghieng* va `ma` cua markdown thanh run co dinh dang."""
    mau = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)")
    for phan in mau.split(text):
        if not phan:
            continue
        if phan.startswith("**") and phan.endswith("**"):
            r = para.add_run(phan[2:-2]); r.bold = True
        elif phan.startswith("*") and phan.endswith("*"):
            r = para.add_run(phan[1:-1]); r.italic = True
        elif phan.startswith("`") and phan.endswith("`"):
            r = para.add_run(phan[1:-1]); r.font.name = "Consolas"
            r.font.size = Pt(CO - 1.5)
        else:
            r = para.add_run(phan)
        r.font.name = FONT if not phan.startswith("`") else "Consolas"


def _o_bang(o, text):
    o.text = ""
    p = o.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    # trong bang, <br> cua markdown thanh xuong dong that
    dong = [d.strip() for d in re.split(r"<br\s*/?>", text)]
    for i, d in enumerate(dong):
        if i:
            p = o.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
        _dam_nghieng(p, d)
        for r in p.runs:
            r.font.size = Pt(CO - 2)


def main():
    if not os.path.exists(MD):
        sys.exit(f"Thieu {MD}")
    doc = Document()
    _font(doc)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Cm(2.0)
        s.left_margin = Cm(2.5)
        s.right_margin = Cm(2.0)

    dong = io.open(MD, encoding="utf-8").read().split("\n")
    i, n_bang, n_muc = 0, 0, 0
    while i < len(dong):
        ln = dong[i].rstrip()

        # ── bang markdown
        if ln.startswith("|") and i + 1 < len(dong) and set(dong[i+1].replace("|", "").strip()) <= set("-: "):
            hang = []
            while i < len(dong) and dong[i].lstrip().startswith("|"):
                o = [c.strip() for c in dong[i].strip().strip("|").split("|")]
                if not set("".join(o)) <= set("-: "):
                    hang.append(o)
                i += 1
            if hang:
                nc = max(len(h) for h in hang)
                t = doc.add_table(rows=0, cols=nc)
                t.alignment = WD_TABLE_ALIGNMENT.CENTER
                _vien(t)
                for k, h in enumerate(hang):
                    c = t.add_row().cells
                    for j in range(nc):
                        _o_bang(c[j], h[j] if j < len(h) else "")
                        if k == 0:
                            for p in c[j].paragraphs:
                                for r in p.runs:
                                    r.bold = True
                doc.add_paragraph()
                n_bang += 1
            continue

        # ── duong ke ngang
        if ln.strip() == "---":
            i += 1
            continue

        # ── tieu de
        if ln.startswith("#"):
            bac = len(ln) - len(ln.lstrip("#"))
            txt = ln.lstrip("#").strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10 if bac >= 2 else 4)
            p.paragraph_format.space_after = Pt(5)
            if bac <= 2:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(txt.replace("**", ""))
            r.bold = True
            r.font.name = FONT
            r.font.size = Pt({1: 13, 2: 13, 3: 13}.get(bac, 13))
            if bac >= 3:
                n_muc += 1
            i += 1
            continue

        # ── gach dau dong
        if ln.lstrip().startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(3)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _dam_nghieng(p, ln.lstrip()[2:])
            i += 1
            continue

        # ── doan thuong / trich dan / rong
        if not ln.strip():
            i += 1
            continue
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _dam_nghieng(p, ln.replace("&nbsp;", " "))
        i += 1

    doc.save(RA)
    kb = os.path.getsize(RA) / 1024
    print("=" * 76)
    print("XUAT THESIS PROPOSAL")
    print("=" * 76)
    print(f"  nguon : docs/THESIS_PROPOSAL_EN.md ({len(dong)} dong)")
    print(f"  ra    : docs/THESIS_PROPOSAL.docx ({kb:.0f} KB)")
    print(f"  bang  : {n_bang}   muc : {n_muc}")
    print(f"  khuon : {FONT} {CO}pt, can deu hai ben, bang co vien")


if __name__ == "__main__":
    main()
