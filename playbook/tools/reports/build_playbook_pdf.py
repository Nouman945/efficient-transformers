"""Renders a Markdown guide to a themed PDF through headless Chrome.

python build_playbook_pdf.py <branch>:<path/to/file.md> <out.pdf>
"""

import re
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

HERE = Path(__file__).parent
ROOT = HERE.parents[2]  # repo checkout root (playbook/tools/reports -> repo)
# the source lives on a branch that may not be checked out, so read it through git
SRC_SPEC = sys.argv[1] if len(sys.argv) > 1 else "k2-horizon-server-docs:playbook/PLAYBOOK.md"
OUT_PDF = (Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "Cloud_AI100_Model_Onboarding_Playbook.pdf").resolve()
OUT_HTML = OUT_PDF.with_suffix(".html")

text = subprocess.run(["git", "-C", str(ROOT), "show", SRC_SPEC], check=True, capture_output=True, text=True).stdout
title_line, body_md = text.split("\n", 1)
title = title_line.lstrip("# ").strip()
# task-list markers are not supported by the base renderer
body_md = re.sub(r"^- \[ \] ", "- &#9744; ", body_md, flags=re.M)
body = MarkdownIt("commonmark").enable("table").render(body_md)

CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
:root { --blue:#3253DC; --navy:#0B1F4B; --ink:#1B2230; --grey:#5B6472; --line:#D9DEE8; --tint:#EEF2FD; --red:#E4002B; }
* { box-sizing: border-box; }
body { font-family: "Inter","Helvetica Neue",Arial,sans-serif; color: var(--ink); font-size: 9.8pt; line-height: 1.45; margin: 0;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.cover { background: var(--navy); color: #fff; padding: 26px 28px 22px; border-radius: 6px; border-top: 6px solid var(--red); margin-bottom: 18px; }
.cover .kicker { color: #AFC0FF; font-size: 8.6pt; letter-spacing: .14em; text-transform: uppercase; font-weight: 600; }
.cover h1 { font-size: 20pt; line-height: 1.15; margin: 8px 0 8px; font-weight: 700; }
.cover p { margin: 0; color: #D6DEF5; } .cover .meta { margin-top: 12px; font-size: 8.4pt; color: #AFC0FF; }
h2 { font-size: 13pt; color: var(--navy); margin: 22px 0 8px; padding-left: 9px; border-left: 4px solid var(--blue); break-after: avoid; }
h3 { font-size: 10.6pt; color: var(--blue); margin: 14px 0 5px; break-after: avoid; }
p { margin: 5px 0; } ul, ol { margin: 5px 0; padding-left: 20px; } li { margin: 3px 0; } li p { margin: 2px 0; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 12px; font-size: 8.7pt; }
th { background: var(--navy); color: #fff; text-align: left; padding: 5px 7px; font-weight: 600; font-size: 8.4pt; }
td { padding: 5px 7px; border-bottom: 1px solid var(--line); vertical-align: top; }
tbody tr:nth-child(even) td { background: #F7F9FD; } tr { break-inside: avoid; } thead { display: table-header-group; }
pre { background: #0F1A36; color: #E6ECFF; padding: 9px 11px; border-radius: 5px; font-size: 8pt; line-height: 1.45;
      white-space: pre-wrap; overflow-wrap: anywhere; margin: 6px 0 10px; break-inside: avoid; font-family: "JetBrains Mono","DejaVu Sans Mono",monospace; }
code { font-family: "JetBrains Mono","DejaVu Sans Mono",monospace; font-size: 8.6pt; background: #F0F3FA; padding: 0 3px; border-radius: 3px; }
pre code { background: none; padding: 0; font-size: 8pt; color: inherit; }
a { color: var(--blue); text-decoration: none; }
.foot { color: var(--grey); font-size: 7.8pt; margin-top: 18px; border-top: 1px solid var(--line); padding-top: 6px; }
"""

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head><body>
<div class="cover">
  <div class="kicker">Cloud AI 100 Ultra &middot; QEfficient &middot; Team Playbook</div>
  <h1>{title}</h1>
  <p>Environments, catalog analysis, onboarding, tests, card validation, benchmarking, reporting and submission. Every command was run and every pitfall listed was hit, on the MBZUAI catalog and the IFM/K2-Horizon-7B onboarding.</p>
  <div class="meta">October 2026 &nbsp;|&nbsp; quic/efficient-transformers 1.23.0.dev0, transformers 5.5.4 &nbsp;|&nbsp; Source: {SRC_SPEC}</div>
</div>
{body}
<div class="foot">Companion files on the same branch: playbook/tools/ (scripts named in the text) and examples/text_generation/k2_horizon/ (the worked example). Colors are a Qualcomm-style theme only; this is a working document, not an official Qualcomm publication.</div>
</body></html>"""

OUT_HTML.write_text(HTML)
subprocess.run(
    [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUT_PDF}",
        OUT_HTML.as_uri(),
    ],
    check=True,
    capture_output=True,
)
print(OUT_PDF)
