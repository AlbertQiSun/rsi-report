#!/usr/bin/env python3
"""Embed an events.jsonl into a copy of replay.html (self-contained, no server needed).

  python3 embed.py samples/synth_events.jsonl -o samples/replay_synth.html
"""
import argparse
import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SLOT = '<script type="application/x-ndjson" id="events-data"></script>'


def embed(events_path, out_path, template=os.path.join(HERE, "replay.html")):
    page = open(template, encoding="utf-8").read()
    if SLOT not in page:
        raise SystemExit("events-data slot not found in template (already embedded?)")
    data = open(events_path, encoding="utf-8").read()
    # "</" cannot appear inside a <script> block; "<\/" is a valid JSON escape for "/".
    data = data.replace("</", "<\\/")
    name = html.escape(os.path.basename(events_path), quote=True)
    block = f'<script type="application/x-ndjson" id="events-data" data-name="{name}">\n{data}</script>'
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page.replace(SLOT, block))
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--template", default=os.path.join(HERE, "replay.html"))
    a = ap.parse_args()
    print("wrote", embed(a.events, a.out, a.template))
