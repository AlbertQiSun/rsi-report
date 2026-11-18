"""Print-fixed copies of two figures whose labels clip in the static export (writes only into report/pdf/figs/)."""
import pathlib, re, subprocess, sys
REPORT = pathlib.Path(__file__).resolve().parents[1]
SRC = REPORT / "site/public/figs"; OUT = REPORT / "pdf/figs"
FONT = "Helvetica, Arial, sans-serif"
def static_svg(svg):
    svg = re.sub(r"<style>.*?</style>", "", svg, flags=re.S)
    svg = re.sub(r"<circle[^>]*>\s*<animateMotion[^>]*/>\s*</circle>", "", svg)
    svg = re.sub(r'<g class="ph">.*?</g>', "", svg, flags=re.S)
    svg = re.sub(r'<g class="rw(?! rlast)[^"]*">.*?</g>', "", svg, flags=re.S)
    svg = re.sub(r'<(rect|circle|path)[^>]*class="rw(?! rlast)[^"]*"[^>]*?(/>|>.*?</\1>)', "", svg, flags=re.S)
    return svg.replace("font-family:inherit", f"font-family:{FONT}")
# disclosure: two-line column headers
s = static_svg((SRC / "fig2-disclosure.en.svg").read_text())
for one, two in [("Research system", ("Research", "system")), ("Driving model", ("Driving", "model")),
                 ("Repeated runs", ("Repeated", "runs")), ("Per-generation data", ("Per-generation", "data"))]:
    m = re.search(r'<text x="([\d.]+)" y="40.0"([^>]*)>' + re.escape(one) + '</text>', s)
    assert m, one
    x = m.group(1)
    s = s.replace(m.group(0), f'<text x="{x}" y="30.0"{m.group(2)}><tspan x="{x}" dy="0">{two[0]}</tspan><tspan x="{x}" dy="14">{two[1]}</tspan></text>')
s = s.replace('font-size="12" fill="#1c1f23" text-anchor="middle" font-weight="500"', 'font-size="11" fill="#1c1f23" text-anchor="middle" font-weight="500"')
s = s.replace('viewBox="0 0 720 270"', 'viewBox="0 0 740 270"')
# map: shrink the long pill labels
m = static_svg((SRC / "fig5-map.en.svg").read_text())
for lab in ["consistent · new control", "consistent · no outer RL", "earlier than in lit."]:
    m = re.sub(r'font-size="[\d.]+"([^>]*>)' + re.escape(lab) + '<', r'font-size="8.6"\1' + lab + '<', m)
for name, svg in [("fig2-disclosure-print", s), ("fig5-map-print", m)]:
    tmp = OUT / f"{name}.svg"; tmp.write_text(svg)
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(OUT / f"{name}.pdf"), str(tmp)], check=True)
    tmp.unlink()
print("ok")
