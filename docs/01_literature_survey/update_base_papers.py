"""Rewrite the Base Papers sheet of the survey workbook from base_papers.json.

Paper details and Drive links come from the workbook's first sheet.
Usage: python docs/01_literature_survey/update_base_papers.py
"""
import json
import re
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, unescape

HERE = Path(__file__).parent
BOOK = HERE / "Literature_Survey_30_Papers.xlsx"
BASE = json.loads((HERE / "base_papers.json").read_text(encoding="utf-8"))

# wording fixes on the first sheet: the Mandke paper never says "raw pixels"
FIXES = {
    "KNN and SVM were given raw pixels with no proper features, so they never had a fair chance. The class count changes between 46 and 48 in the text.":
        "The paper doesn't describe any specific handcrafted features for KNN and SVM, so those models are weak baselines. The class count changes between 46 and 48 in the text.",
    "The classical models had no handcrafted features, so the comparison is unfair to them. This is the main gap our project picks up.":
        "Without proper features for the classical models, the comparison says little about what an SVM can do on Modi.",
    "Handcrafted features, combining the models and explaining the results are all missing. These are exactly what our project adds.":
        "Features from a deep network can't be explained, and the paper doesn't look at which letters get confused.",
}
HEAD = ["Sr No (in survey)", "Name of paper", "Year of publication", "Name of author", "Journal/Conference Name",
        "Why it is a base paper", "What we take from it", "What we do differently (gap we address)",
        "Related objective", "Drive Link for the Research Paper"]


def col(i):
    return chr(65 + i)


def cell(ref, v, style):
    if isinstance(v, int):
        return f'<c r="{ref}" s="{style}"><v>{v}</v></c>'
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(str(v))}</t></is></c>'


with zipfile.ZipFile(BOOK) as z:
    files = {n: z.read(n) for n in z.namelist()}

sheet1 = files["xl/worksheets/sheet1.xml"].decode("utf-8")
for old, new in FIXES.items():
    if escape(old) in sheet1:
        sheet1 = sheet1.replace(escape(old), escape(new))
        print("fixed:", old[:50])
files["xl/worksheets/sheet1.xml"] = sheet1.encode("utf-8")

papers = {}
for row in re.findall(r'<row r="(\d+)"[^>]*>(.*?)</row>', sheet1, re.S):
    if row[0] == "1":
        continue
    vals = {}
    for ref, inner in re.findall(r'<c r="([A-Z]+)\d+"[^>]*>(.*?)</c>', row[1], re.S):
        m = re.search(r"<(?:t|v)[^>]*>(.*?)</(?:t|v)>", inner, re.S)
        vals[ref] = unescape(m.group(1)) if m else ""
    papers[vals["A"]] = vals

rows = ['<row r="1" ht="36" customHeight="1">' + "".join(cell(f"{col(i)}1", h, 1) for i, h in enumerate(HEAD)) + "</row>"]
links, rels = [], []
for k, b in enumerate(BASE, start=2):
    p = papers[b["no"]]
    vals = [int(b["no"]), p["B"], int(p["C"]), p["D"], p["E"], b["why"], b["take"], b["differ"], b["objective"], p["P"]]
    cells = [cell(f"{col(i)}{k}", v, 3 if isinstance(v, int) else (4 if i == 9 else 2)) for i, v in enumerate(vals)]
    rows.append(f'<row r="{k}" ht="150" customHeight="1">' + "".join(cells) + "</row>")
    rels.append(f'<Relationship Id="rId{k - 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="{escape(p["P"])}" TargetMode="External"/>')
    links.append(f'<hyperlink ref="J{k}" r:id="rId{k - 1}"/>')
last = len(BASE) + 1
sheet2 = files["xl/worksheets/sheet2.xml"].decode("utf-8")
sheet2 = re.sub(r"<sheetData>.*</sheetData>", lambda _: "<sheetData>" + "".join(rows) + "</sheetData>", sheet2, flags=re.S)
sheet2 = re.sub(r'<autoFilter ref="[^"]+"/>', f'<autoFilter ref="A1:J{last}"/>', sheet2)
sheet2 = re.sub(r"<hyperlinks>.*</hyperlinks>", lambda _: "<hyperlinks>" + "".join(links) + "</hyperlinks>", sheet2, flags=re.S)
files["xl/worksheets/sheet2.xml"] = sheet2.encode("utf-8")
files["xl/worksheets/_rels/sheet2.xml.rels"] = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(rels) + "</Relationships>").encode("utf-8")
wb = files["xl/workbook.xml"].decode("utf-8")
files["xl/workbook.xml"] = re.sub(r"(&apos;Base Papers&apos;!\$A\$1:\$J\$)\d+", rf"\g<1>{last}", wb).encode("utf-8")

tmp = BOOK.with_suffix(".tmp.xlsx")
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
    for n, d in files.items():
        z.writestr(n, d)
shutil.move(tmp, BOOK)
print("base papers:", [b["no"] for b in BASE])
