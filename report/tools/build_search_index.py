"""Evidence-search index for the report site (no LLM): every published text file cut into line-numbered chunks.

Writes report/site/public/search-index.json = {"files": [path...], "chunks": [[file_idx, start, end, kind, text]...]}.
The site's search page ranks chunks with BM25 in the browser and links each hit to <repo>/blob/<branch>/<path>#Lstart-Lend.

File list: `git ls-files` at the repository root when the tree is a git repo (the public repo), otherwise a walk of
report/, research/, platform/, tools/ (local preview). Binary, generated and very large files are skipped.
"""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "site" / "public" / "search-index.json"
TEXT_EXT = {".md", ".py", ".sh", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".log", ".toml", ".ini", ".cfg",
            ".astro", ".mjs", ".js", ".ts", ".css", ".html", ".csv", ".tex"}
SKIP_PARTS = {"node_modules", "dist", ".astro", "__pycache__", ".git", "archive", "other", "rec", "flint-int4",
              "first-steps-toward-automated-ai-research", "publish", ".claude"}
SKIP_NAMES = {"package-lock.json", "content.json", "search-index.json", "report_full.md"}
MAX_FILE_BYTES = 1_500_000      # larger files: only the first part is indexed
MAX_LINES_PER_CHUNK = 30
MAX_CHUNK_CHARS = 1800


def file_list():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        if out.strip():
            return [p for p in out.splitlines()]
    except (OSError, subprocess.CalledProcessError):
        pass
    files = []
    for top in ("report", "research", "platform", "tools"):
        for dp, dns, fns in os.walk(ROOT / top):
            dns[:] = [d for d in dns if d not in SKIP_PARTS and not d.startswith(".")]
            files += [os.path.relpath(os.path.join(dp, f), ROOT) for f in fns]
    return sorted(files)


def kind_of(path):
    if path.startswith("report/sections/") or path.startswith("report/draft"):
        return "report"
    if path.endswith("REPORT_LOG.md") or path.endswith(".log") or path.endswith(".jsonl") or "/replay/" in path:
        return "log"
    if path.startswith("research/decisions/") or path.endswith(".json") or path.endswith(".csv"):
        return "data"
    if path.startswith("research/") or path.startswith("report/"):
        return "doc" if path.endswith(".md") else "code"
    return "code"


def chunks_of(lines, is_md):
    """Markdown: break at headings and blank lines, max 30 lines. Other text: fixed 30-line windows."""
    out, start, buf = [], 0, []

    def flush(end):
        nonlocal buf, start
        text = "\n".join(buf).strip()
        if text:
            out.append((start + 1, end, text[:MAX_CHUNK_CHARS]))
        buf = []

    for i, line in enumerate(lines):
        if is_md and re.match(r"^#{1,6} ", line) and buf:
            flush(i)
            start = i
        if not buf:
            start = i
        buf.append(line)
        if len(buf) >= MAX_LINES_PER_CHUNK or (is_md and not line.strip() and len(buf) >= 12):
            flush(i + 1)
    flush(len(lines))
    return out


def main():
    files, chunks = [], []
    for path in file_list():
        p = Path(path)
        if p.suffix.lower() not in TEXT_EXT or p.name in SKIP_NAMES or any(part in SKIP_PARTS for part in p.parts):
            continue
        full = ROOT / path
        try:
            raw = full.read_bytes()[:MAX_FILE_BYTES]
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lines = text.split("\n")
        if len(raw) == MAX_FILE_BYTES:
            lines = lines[:-1]          # drop the possibly cut last line
        fi = len(files)
        files.append(path)
        for s, e, t in chunks_of(lines, p.suffix == ".md"):
            chunks.append([fi, s, e, kind_of(path), t])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"files": files, "chunks": chunks}, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(files)} files, {len(chunks)} chunks, "
          f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
