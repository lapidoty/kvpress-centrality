#!/bin/bash
# Build REPORT/SUPPLEMENTARY .pdf + .html from markdown via typst (LaTeX-quality math), tuned to ~10-11 pages.
# Requires: pandoc >= 3, and `pip install typst pypandoc pypdf`.
cd "$(dirname "$0")"
python3 - <<'PY'
import typst, pypandoc, subprocess, re, os
pandoc = pypandoc.get_pandoc_path()
for doc in ("REPORT", "SUPPLEMENTARY"):
    subprocess.run([pandoc, f"{doc}.md", "-o", f"{doc}.typ", "-t", "typst", "--standalone"], check=True)
    s = open(f"{doc}.typ", encoding="utf-8").read()
    # center every figure and drop pandoc's auto-caption (the italic "Figure N:" lines are the real captions)
    s = re.sub(r'#figure\(image\((".*?", width: [\d.]+%)\),\s*caption: \[.*?\]\s*\)', r'#align(center, image(\1))', s, flags=re.S)
    s = re.sub(r'#box\(image\((".*?", width: [\d.]+%)\)\) (#emph\[)', r'#align(center, image(\1))\n\n\2', s)
    s = re.sub(r'#box\(image\((".*?", width: [\d.]+%)\)\)', r'#align(center, image(\1))', s)
    # layout: tighter margins/font, scale the full-width figures to fit the page budget
    s = (s.replace("margin: (x: 1.25in, y: 1.25in),", "margin: (x: 1.6cm, y: 1.7cm),")
          .replace("fontsize: 11pt,", "fontsize: 10pt,").replace("width: 70.0%", "width: 55%"))
    open(f"{doc}.typ", "w", encoding="utf-8").write(s)
    typst.compile(f"{doc}.typ", output=f"{doc}.pdf")
    subprocess.run([pandoc, f"{doc}.md", "-o", f"{doc}.html", "--standalone", "--embed-resources", "--mathml"], check=True)
    os.remove(f"{doc}.typ")
print("built REPORT/SUPPLEMENTARY .pdf + .html")
PY
