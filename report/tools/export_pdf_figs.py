"""Export the web figures as static PDFs for the LaTeX report (report/pdf/figs/).

The web SVGs animate (stroke drawing, pop-in, replay frames). For print each figure is reduced to its final
state: the <style> block is dropped (so drawn lines, popped dots and grown bars show fully), replay frames that
are only visible mid-loop are removed (only elements tagged `rlast` stay), moving dots and the playhead are
removed, and the font is fixed. Then rsvg-convert writes a vector PDF. The GIF of Figure 1 becomes a PNG of its
last frame.

Usage: python3 tools/export_pdf_figs.py [en|zh]
"""
import pathlib, re, subprocess, sys
from PIL import Image

REPORT = pathlib.Path(__file__).resolve().parents[1]
SRC = REPORT / "site/public/figs"
OUT = REPORT / "pdf/figs"
OUT.mkdir(parents=True, exist_ok=True)
lang = sys.argv[1] if len(sys.argv) > 1 else "en"
FONT = "Helvetica, Arial, sans-serif" if lang == "en" else "PingFang SC, Hiragino Sans GB, sans-serif"


def static_svg(svg: str) -> str:
    svg = re.sub(r"<style>.*?</style>", "", svg, flags=re.S)
    svg = re.sub(r"<circle[^>]*>\s*<animateMotion[^>]*/>\s*</circle>", "", svg)
    svg = re.sub(r'<g class="ph">.*?</g>', "", svg, flags=re.S)
    # replay: drop frames that are not part of the final state (class contains "rw" but not "rlast")
    svg = re.sub(r'<g class="rw(?! rlast)[^"]*">.*?</g>', "", svg, flags=re.S)
    svg = re.sub(r'<(rect|circle|path)[^>]*class="rw(?! rlast)[^"]*"[^>]*?(/>|>.*?</\1>)', "", svg, flags=re.S)
    svg = svg.replace("font-family:inherit", f"font-family:{FONT}")
    if 'xmlns="http://www.w3.org/2000/svg"' not in svg:
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    return svg


n = 0
for f in sorted(SRC.glob(f"*.{lang}.svg")):
    tmp = OUT / f.name
    tmp.write_text(static_svg(f.read_text()))
    pdf = OUT / (f.name.replace(f".{lang}.svg", ".pdf"))
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf), str(tmp)], check=True)
    tmp.unlink()
    n += 1
gif = SRC / f"recursion-loop.{lang}.gif"
if gif.exists():
    im = Image.open(gif)
    im.seek(im.n_frames - 1)
    im.convert("RGB").save(OUT / "recursion-loop.png")
    n += 1
print(f"exported {n} figures to {OUT.relative_to(REPORT)}")
