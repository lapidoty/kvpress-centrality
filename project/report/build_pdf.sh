#!/bin/bash
# Build REPORT/SUPPLEMENTARY .pdf + .html from markdown via typst (LaTeX-quality math), tuned to ~11 pages.
# Requires: pandoc >= 3, and `pip install typst pypandoc pypdf`.
cd "$(dirname "$0")"
for doc in REPORT SUPPLEMENTARY; do
  pandoc "$doc.md" -o "$doc.typ" -t typst --standalone
  sed -i 's/margin: (x: 1.25in, y: 1.25in),/margin: (x: 1.6cm, y: 1.7cm),/; s/fontsize: 11pt,/fontsize: 10pt,/; s/width: 70.0%/width: 55%/g' "$doc.typ"
  python3 -c "import typst; typst.compile('$doc.typ', output='$doc.pdf')"
  pandoc "$doc.md" -o "$doc.html" --standalone --embed-resources --mathml
  rm -f "$doc.typ"
done
echo "built REPORT/SUPPLEMENTARY .pdf + .html"
