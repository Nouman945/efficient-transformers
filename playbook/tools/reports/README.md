Report builders used in the MBZUAI work. All render HTML through headless Chrome (`google-chrome --headless=new`).

- `build_analysis_report.py <catalog.json>`: catalog analysis PDF (tiers, families, local parity results from a `../results/*.json` folder written by the parity scripts).
- `build_ranked_plan.py`: effort-ranked plan PDF; the work items are curated in the script, counts are checked against `mbzuai_catalog.json`.
- `build_playbook_pdf.py <branch>:<file.md> <out.pdf>`: any Markdown guide to a themed PDF, read from a git branch.

They need `markdown-it-py` (`pip install markdown-it-py`) and Chrome. Copy and adapt for the next customer; the MBZUAI rows and paths are examples.
