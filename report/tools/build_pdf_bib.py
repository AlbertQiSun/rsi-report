"""Bibliography for the LaTeX report (report/pdf/).

  python3 tools/build_pdf_bib.py keys   -> writes pdf/citekeys.md (every citable key + what it is)
  python3 tools/build_pdf_bib.py        -> scans pdf/main.tex for \\cite{...}, writes pdf/refs.tex (thebibliography,
                                           numbered in order of first citation)

Keys come from the web report's reference list (site/src/data/content.json), sanitized for LaTeX:
"T-24" stays "T-24"; "LOG E" -> "LOG-E"; "decisions/decisions_1235.json" -> "decisions-decisions-1235-json".
Ledger rows of the four releases (RF-/FL-/NV-/AX-) collapse to one entry per release: REL-RF, REL-FL, REL-NV, REL-AX.
Internal files link to the public repository (REPO_URL below) at the recorded line.
"""
import json, pathlib, re, sys

REPORT = pathlib.Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/AlbertQiSun/rsi-report/blob/main/"
refs = json.loads((REPORT / "site/src/data/content.json").read_text())["refs"]
RELEASES = {
    "RF": ("Recursive Superintelligence. First steps toward automated AI research. Company blog, 2026", "https://recursive.com/updates/first-steps-toward-automated-ai-research"),
    "FL": ("Recursive Superintelligence. Introducing Flint. Company release, 2026", None),
    "NV": ("NaiveAI. Naive-N0.5-Flash: Building frontier AI with AI. Company research page, 2026", None),
    "AX": ("Apex Intelligence. Introducing Apex-35B-V1.0. Company release, 2026", None),
}


def skey(k):
    pre = k.split("-")[0]
    if pre in RELEASES:
        return f"REL-{pre}"
    return re.sub(r"[^A-Za-z0-9]+", "-", k).strip("-")


def tex_escape(s):
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")]:
        s = s.replace(a, b)
    return re.sub(r"\*+", "", s)


ENTRIES = {}
for r in refs:
    k = skey(r["key"])
    if k in ENTRIES:
        continue
    pre = r["key"].split("-")[0]
    if pre in RELEASES:
        title, url = RELEASES[pre]
        url = url or (r["urls"][0] if r["urls"] else None)
        ENTRIES[k] = {"text": tex_escape(title), "url": url, "about": title}
        continue
    if r["group"] == "data":
        text = tex_escape(r["en"]) + ". Project file \\texttt{" + tex_escape(r["src"] or r["key"]) + "}"
        url = (REPO_URL + r["src"] + (f"#L{r['line']}" if r.get("line") else "")) if r.get("src") else None
        ENTRIES[k] = {"text": text, "url": url, "about": r["en"]}
    else:
        ENTRIES[k] = {"text": tex_escape(r["en"]), "url": r["urls"][0] if r["urls"] else None, "about": r["en"]}

if len(sys.argv) > 1 and sys.argv[1] == "keys":
    lines = ["# Citable keys for report/pdf/main.tex", "", "Use \\cite{KEY}. Ledger keys of the four releases collapse to REL-RF / REL-FL / REL-NV / REL-AX.", ""]
    for k, e in ENTRIES.items():
        lines.append(f"- `{k}` — {e['about'][:160]}")
    (REPORT / "pdf/citekeys.md").write_text("\n".join(lines) + "\n")
    print(f"wrote pdf/citekeys.md ({len(ENTRIES)} keys)")
    sys.exit()

tex = (REPORT / "pdf/main.tex").read_text()
order = []
for m in re.finditer(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", tex):
    for k in m.group(1).split(","):
        k = k.strip()
        if k and k not in order:
            order.append(k)
missing = [k for k in order if k not in ENTRIES]
out = ["\\begin{thebibliography}{99}", "\\small"]
for k in order:
    if k in missing:
        continue
    e = ENTRIES[k]
    out.append(f"\\bibitem{{{k}}} {e['text']}." + (f" \\url{{{e['url']}}}" if e["url"] else ""))
out.append("\\end{thebibliography}")
(REPORT / "pdf/refs.tex").write_text("\n".join(out) + "\n")
print(f"wrote pdf/refs.tex: {len(order) - len(missing)} entries" + (f"; MISSING keys: {missing}" if missing else ""))
