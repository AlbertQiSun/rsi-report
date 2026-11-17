#!/usr/bin/env python3
"""Render an events.jsonl replay to MP4 and GIF, frame by frame, CPU only.

No browser: matplotlib (Agg) draws every frame; ffmpeg encodes the MP4 with a
software encoder (libx264 / libopenh264 / mpeg4, never VideoToolbox) and turns
it into a palette-optimized GIF. Without ffmpeg it falls back to Pillow (GIF only).

  python3 render_video.py samples/synth_events.jsonl -o samples/replay_synth
      -> samples/replay_synth.mp4 + samples/replay_synth.gif
  python3 render_video.py ev.jsonl -o out --duration 40 --theme dark
  python3 render_video.py ev.jsonl --png-at 0.75 -o out    # one still frame at 75%

Panels mirror replay.html: agent topology (deterministic radial tree here, so
frames are reproducible), tool-activity raster, best-so-far metric with
decision markers, GPU-lane Gantt, event feed, scrub timeline, legend.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

THEMES = {
    "light": dict(page="#f9f9f7", surface="#fcfcfb", surface2="#f0efec", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"], running="#86b6ef"),
    "dark": dict(page="#0d0d0d", surface="#1a1a19", surface2="#262624", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 s=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"], running="#5598e7"),
}
STATUS = dict(good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b")
TOOLS = ["bash", "edit", "read", "web_search", "fetch", "mcp", "other"]


def palette(theme):
    c = dict(THEMES[theme])
    s = c["s"]
    c["tool"] = dict(bash=s[0], edit=s[1], read=s[2], web_search=s[3], fetch=s[4], mcp=s[6], other=c["muted"])
    c["st"] = dict(ok=STATUS["good"], timeout=STATUS["warning"], nan=STATUS["serious"], crash=STATUS["critical"],
                   audit_fail=STATUS["critical"], running=c["running"])
    c["v"] = dict(adopt=STATUS["good"], reject=c["muted"], explore=s[0], rollback=STATUS["critical"])
    c["gen"] = [s[0], s[6], s[2], s[1]]
    return c


def tool_class(t):
    if not t:
        return "other"
    if t.startswith("mcp"):
        return "mcp"
    return t if t in TOOLS else "other"


def fmt_dur(s):
    s = max(0, int(s))
    h, m, x = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{x:02d}" if h else f"{m}:{x:02d}"


def fmt_v(v):
    return "n/a" if v is None else f"{v:.4f}"


def guess_role(aid):
    if aid == "orchestrator":
        return "orchestrator"
    if aid == "evald":
        return "evald"
    if "lessons" in aid:
        return "lessons-writer"
    if "/" in aid:
        return "subagent"
    if aid.startswith("thread"):
        return "researcher"
    return "service"


# ============================== model ==============================
class Model:
    def __init__(self, events):
        events = sorted(enumerate(events), key=lambda p: (p[1]["t"], p[0]))
        events = [e for _, e in events]
        self.t0 = events[0]["t"]
        rel = lambda e: e["t"] - self.t0  # noqa: E731
        self.runs = OrderedDict()
        self.agents = OrderedDict()
        self.tools, self.llms, self.msgs, self.feed, self.marks = [], [], [], [], []
        open_jobs = {}
        for e in events:
            t = rel(e)
            r = self._run(e, t)
            a = self._agent(e["run"], e.get("agent"), e.get("parent"), t)
            ev = e["ev"]
            if ev == "run_start":
                r["start"], r["meta"] = t, e
                r["budget"] = (e.get("budget") or {}).get("trials")
                self.feed.append((t, e["run"], e["agent"], "run", f"run_start gen {e.get('gen')} · {e.get('policy', '')}"))
                self.marks.append((t, "run_start", None))
            elif ev == "run_end":
                r["end_ev"] = e
                self.feed.append((t, e["run"], e["agent"], "run", f"run_end · best {fmt_v(e.get('best_value'))}"))
            elif ev == "agent_spawn":
                a.update(spawn=t, role=e.get("role") or a["role"], parent=e.get("parent") or a["parent"])
                self.feed.append((t, e["run"], e["agent"], "spawn", f"spawned as {a['role']}"))
            elif ev == "agent_end":
                a["end"] = t
            elif ev == "message":
                to = self._agent(e["run"], e.get("to"), None, t)
                self.msgs.append((t, e["run"], a["key"], to and to["key"], "msg"))
                self.feed.append((t, e["run"], e["agent"], "message", f"→ {e.get('to')}: {e.get('summary', '')}"))
            elif ev == "llm_call":
                lat = float(e.get("latency_s") or 0)
                self.llms.append((t - lat, t, e["run"], a["key"], self._root(e["agent"]), "/" in e["agent"]))
            elif ev == "tool_call":
                lat = float(e.get("latency_s") or 0)
                self.tools.append((t - lat, t, e["run"], a["key"], self._root(e["agent"]), "/" in e["agent"],
                                   tool_class(e.get("tool")), e.get("ok", True) is not False))
            elif ev == "proposal":
                self.feed.append((t, e["run"], e["agent"], "proposal", f"proposal #{e.get('trial')}: {e.get('hypothesis', '')}"))
            elif ev == "eval_submit":
                j = dict(trial=e.get("trial"), hash=e.get("artifact_hash"), lane=str(e.get("lane")), t0=t, t1=None,
                         status=None, value=None, audit=[])
                r["jobs"].append(j)
                open_jobs[(e["run"], e.get("artifact_hash"))] = j
                if j["lane"] not in r["lanes"]:
                    r["lanes"].append(j["lane"])
                ev_a = self._agent(e["run"], "evald", "orchestrator", t)
                self.msgs.append((t, e["run"], a["key"], ev_a["key"], "eval"))
                self.feed.append((t, e["run"], e["agent"], "eval_submit", f"eval_submit #{e.get('trial')} → {j['lane']}"))
            elif ev == "eval_result":
                j = open_jobs.pop((e["run"], e.get("artifact_hash")), None)
                if j is None:
                    j = dict(trial=e.get("trial"), hash=e.get("artifact_hash"), lane=str(e.get("lane", "lane?")),
                             t0=t - float(e.get("wall_s") or 0), t1=None, status=None, value=None, audit=[])
                    r["jobs"].append(j)
                    if j["lane"] not in r["lanes"]:
                        r["lanes"].append(j["lane"])
                j.update(t1=t, status=e.get("status", "ok"), value=e.get("value"))
                r["metric"] = r["metric"] or e.get("metric")
                r["evals"].append(dict(t=t, trial=e.get("trial"), value=e.get("value"), status=j["status"],
                                       baseline=e.get("trial") == 0))
                self.msgs.append((t, e["run"], a["key"], f"{e['run']}::orchestrator", "eval"))
                self.feed.append((t, e["run"], e["agent"], "eval_result:" + j["status"],
                                  f"eval_result #{e.get('trial')}: {j['status']}" + (f" · {fmt_v(e.get('value'))}" if e.get("value") is not None else "")))
            elif ev == "decision":
                d = dict(t=t, trial=e.get("trial"), verdict=e.get("verdict"), before=e.get("best_before"),
                         after=e.get("best_after"), reason=e.get("reason"))
                r["decisions"].append(d)
                self.feed.append((t, e["run"], e["agent"], "decision:" + str(e.get("verdict")),
                                  f"decision #{e.get('trial')}: {str(e.get('verdict')).upper()} · best {fmt_v(e.get('best_after'))}"))
                self.marks.append((t, "decision", e.get("verdict")))
            elif ev == "audit_flag":
                for j in reversed(r["jobs"]):
                    if j["hash"] == e.get("artifact_hash"):
                        j["audit"].append(t)
                        break
                self.feed.append((t, e["run"], e["agent"], "audit", f"AUDIT {e.get('check')}: {e.get('detail', '')}"))
                self.marks.append((t, "audit", None))
            elif ev in ("lessons_update", "weights_update"):
                txt = (f"lessons_update: {e.get('n_lessons')} lessons" if ev == "lessons_update"
                       else f"weights_update: {e.get('n_examples')} examples")
                self.feed.append((t, e["run"], e["agent"], "lessons", txt))
                self.marks.append((t, "lessons", None))
        self.run_list = sorted(self.runs.values(), key=lambda r: r["start"])
        for a in self.agents.values():
            if a["parent"]:
                a["pkey"] = f"{a['run']}::{a['parent']}"
            else:
                a["pkey"] = None if a["id"] == "orchestrator" else f"{a['run']}::orchestrator"
            if a["pkey"] not in self.agents:
                a["pkey"] = None
            if a["end"] is None and self.runs[a["run"]]["end_ev"] is not None:
                a["end"] = self.runs[a["run"]]["end"]
            if a["id"] == "evald":
                a["role"] = "evald"
        lower = None
        for r in self.run_list:
            for d in r["decisions"]:
                if lower is None and d["verdict"] == "adopt" and d["before"] is not None and d["after"] is not None and d["after"] != d["before"]:
                    lower = d["after"] < d["before"]
        self.metric = next((r["metric"] for r in self.run_list if r["metric"]), "metric")
        self.lower = True if lower is None else lower
        vals = []
        for r in self.run_list:
            r["dur"] = max(1.0, r["end"] - r["start"])
            r["lanes"].sort()
            base = next((x for x in r["evals"] if x["baseline"] and x["value"] is not None), None)
            s0 = base["value"] if base else next((d["before"] for d in r["decisions"] if d["before"] is not None), None)
            r["s0"] = s0
            r["best"] = ([(base["t"] if base else r["start"], s0, None)] if s0 is not None else [])
            r["best"] += [(d["t"], d["after"], d) for d in r["decisions"] if d["after"] is not None]
            r["eval_times"] = [x["t"] for x in r["evals"] if not x["baseline"]]
            vals += [v for _, v, _ in r["best"] if v is not None]
            vals += [x["value"] for x in r["evals"] if x["value"] is not None]
            self._layout(r)
        self.vals = vals
        self.dur = max(1.0, self.run_list[-1]["end"])
        self.max_run_dur = max(r["dur"] for r in self.run_list)
        self.tools.sort(key=lambda x: x[1])
        self.msgs.sort(key=lambda x: x[0])
        self.feed.sort(key=lambda x: x[0])

    @staticmethod
    def _root(aid):
        return aid.split("/")[0] if aid and "/" in aid else aid

    def _run(self, e, t):
        r = self.runs.get(e["run"])
        if r is None:
            r = dict(id=e["run"], gen=e.get("gen", 0) or 0, start=t, end=t, meta=None, end_ev=None, budget=None,
                     metric=None, lanes=[], jobs=[], evals=[], decisions=[])
            self.runs[e["run"]] = r
        r["end"] = max(r["end"], t)
        return r

    def _agent(self, run, aid, parent, t):
        if not aid:
            return None
        k = f"{run}::{aid}"
        a = self.agents.get(k)
        if a is None:
            a = dict(key=k, run=run, id=aid, parent=parent, role=guess_role(aid), spawn=t, end=None)
            self.agents[k] = a
        if a["parent"] is None and parent:
            a["parent"] = parent
        return a

    def _layout(self, r):
        """Deterministic radial tree: orchestrator centre, threads on a ring, subagents on an arc outside their thread."""
        ags = [a for a in self.agents.values() if a["run"] == r["id"]]
        threads = sorted({a["id"] for a in ags if a["role"] == "researcher" or (a["id"].startswith("thread") and "/" not in a["id"])},
                         key=lambda s: (len(s), s))
        n = max(1, len(threads))
        ang = {th: math.pi / 2 + math.pi / n + 2 * math.pi * i / n for i, th in enumerate(threads)}
        sub_idx = {}
        for a in ags:
            if a["id"] == "orchestrator":
                a["pos"] = (0.0, 0.0)
            elif a["role"] == "evald":
                a["pos"] = (1.05, 0.82)
            elif a["role"] == "lessons-writer":
                a["pos"] = (-1.05, 0.82)
            elif a["id"] in ang:
                th = ang[a["id"]]
                a["pos"] = (0.52 * math.cos(th), 0.52 * math.sin(th))
            elif "/" in a["id"] and a["id"].split("/")[0] in ang:
                root = a["id"].split("/")[0]
                k = sub_idx.get(root, 0)
                sub_idx[root] = k + 1
                slot = k % 5
                th = ang[root] + (slot - 2) * 0.32
                a["pos"] = (0.52 * math.cos(ang[root]) + 0.36 * math.cos(th), 0.52 * math.sin(ang[root]) + 0.36 * math.sin(th))
            else:
                h = (sum(map(ord, a["id"])) % 360) * math.pi / 180
                a["pos"] = (0.95 * math.cos(h), 0.95 * math.sin(h))

    def current_run(self, t):
        cur = self.run_list[0]
        for r in self.run_list:
            if r["start"] <= t + 1e-9:
                cur = r
        return cur


# ============================== renderer ==============================
NODE_PT = {"orchestrator": 13, "researcher": 9.5, "evald": 10, "lessons-writer": 10, "critic": 6, "subagent": 6}
NODE_MK = {"orchestrator": "o", "researcher": "o", "evald": "s", "lessons-writer": "*", "critic": "D", "subagent": "o"}


class Renderer:
    def __init__(self, model: Model, theme="light", width=1280, height=720, speed=100.0):
        self.M, self.c, self.speed = model, palette(theme), speed
        c = self.c
        plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
                             "axes.edgecolor": c["axis"], "axes.labelcolor": c["muted"], "xtick.color": c["muted"],
                             "ytick.color": c["muted"], "text.color": c["ink"], "axes.facecolor": c["surface"],
                             "figure.facecolor": c["page"], "axes.titlesize": 9, "axes.titleweight": "bold",
                             "axes.titlecolor": c["ink"], "axes.titlelocation": "left"})
        self.dpi = 100
        self.fig = plt.figure(figsize=(width / self.dpi, height / self.dpi), dpi=self.dpi)
        gs = self.fig.add_gridspec(5, 2, width_ratios=[1.05, 1], height_ratios=[3.0, 1.25, 1.55, 0.42, 0.22],
                                   left=0.06, right=0.985, top=0.875, bottom=0.03, hspace=0.62, wspace=0.12)
        self.ax_topo = self.fig.add_subplot(gs[0:2, 0])
        self.ax_act = self.fig.add_subplot(gs[2, 0])
        self.ax_met = self.fig.add_subplot(gs[0, 1])
        self.ax_lane = self.fig.add_subplot(gs[1, 1])
        self.ax_feed = self.fig.add_subplot(gs[2, 1])
        self.ax_tl = self.fig.add_subplot(gs[3, :])
        self.ax_leg = self.fig.add_subplot(gs[4, :])
        self.dyn = []
        self.lane_run = self.act_run = None
        self.title = self.fig.text(0.045, 0.962, "", fontsize=13, fontweight="bold", color=c["ink"], va="center")
        self.sub = self.fig.text(0.045, 0.928, "", fontsize=8.5, color=c["ink2"], va="center")
        self.kpi = self.fig.text(0.985, 0.95, "", fontsize=9, color=c["ink"], va="center", ha="right", family="monospace")
        self._static_topo()
        self._static_metric()
        self._static_feed()
        self._static_timeline()
        self._static_legend()

    # ---------- static parts
    def _style(self, ax, title=None):
        for s in ax.spines.values():
            s.set_color(self.c["axis"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if title:
            ax.set_title(title, pad=4)

    def _static_topo(self):
        ax, c = self.ax_topo, self.c
        bb = ax.get_position()
        fw, fh = self.fig.get_size_inches()
        ratio = (bb.width * fw) / (bb.height * fh)  # equal data units on both axes, no clipping
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(-1.1 * ratio, 1.1 * ratio)
        self.topo_xmin = -1.1 * ratio
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("Agent topology  ·  nodes spawn/fade, dots = messages, rings = tool calls", pad=4)

    def _static_metric(self):
        ax, c, M = self.ax_met, self.c, self.M
        self._style(ax, f"Best-so-far {M.metric}  ({'lower' if M.lower else 'higher'} is better; y zoomed, not zero-based)")
        lo, hi = min(M.vals), max(M.vals)
        p = (hi - lo) * 0.08 or 0.01
        ax.set_ylim(lo - p, hi + p)
        ax.set_xlim(0, M.max_run_dur / 60 * 1.07)
        ax.set_xlabel("minutes since run start (generations overlaid)", fontsize=7.5)
        ax.grid(axis="y", color=c["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        for s0 in sorted({r["s0"] for r in M.run_list if r["s0"] is not None}):
            ax.axhline(s0, color=c["ink2"], lw=1, ls=(0, (5, 4)))
            ax.text(0.3, s0, f" baseline s0 = {s0:.4f}", color=c["ink2"], fontsize=7.5, va="bottom")

    def _static_feed(self):
        ax = self.ax_feed
        ax.set_axis_off()
        ax.set_title("Event feed (newest first)", pad=4)

    def _static_timeline(self):
        ax, c, M = self.ax_tl, self.c, self.M
        ax.set_xlim(0, M.dur)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for k in ("left", "right", "top"):
            ax.spines[k].set_visible(False)
        ticks = [x * 60 for x in range(0, int(M.dur / 60) + 1, 15)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([fmt_dur(x) for x in ticks], fontsize=7)
        for r in M.run_list:
            ax.add_patch(Rectangle((r["start"], 0.62), r["end"] - r["start"], 0.3, color=c["gen"][r["gen"] % 4], alpha=0.25, lw=0))
            ax.text(r["start"] + M.dur * 0.004, 0.77, f"gen {r['gen']}", fontsize=7, color=c["ink2"], va="center")
        for t, kind, v in M.marks:
            if kind == "decision":
                ax.plot([t, t], [0.05, 0.5 if v != "reject" else 0.3], color=c["v"].get(v, c["muted"]), lw=1.6 if v != "reject" else 0.8)
            elif kind == "audit":
                ax.text(t, 0.3, "!", color=STATUS["critical"], fontsize=9, fontweight="bold", ha="center", va="center")
            elif kind == "lessons":
                ax.text(t, 0.3, "◆", color=c["s"][4], fontsize=8, ha="center", va="center")

    def _static_legend(self):
        ax, c = self.ax_leg, self.c
        ax.set_axis_off()
        h = [Line2D([], [], marker="o", ls="", mfc="none", mec=c["tool"][t], mew=2, ms=8, label=t if t != "mcp" else "mcp:*") for t in TOOLS[:-1]]
        h += [Patch(color=c["st"][s], label=s) for s in ["running", "ok", "timeout", "nan", "crash", "audit_fail"]]
        h += [Line2D([], [], marker="o", ls="", color=c["v"]["adopt"], ms=6, label="adopt"),
              Line2D([], [], marker="x", ls="", color=c["v"]["reject"], ms=6, label="reject"),
              Line2D([], [], marker="o", ls="", mfc="none", mec=c["v"]["explore"], ms=6, label="explore"),
              Line2D([], [], marker="D", ls="", color=c["v"]["rollback"], ms=6, label="rollback")]
        leg = ax.legend(handles=h, loc="center", ncol=len(h), fontsize=7, frameon=False, handlelength=1.2,
                        columnspacing=0.9, handletextpad=0.35, borderaxespad=0)
        for txt in leg.get_texts():
            txt.set_color(c["ink2"])

    def _lanes_for(self, r):
        ax, c = self.ax_lane, self.c
        ax.clear()
        self._style(ax, f"GPU eval lanes · gen {r['gen']}  (submit → result, color = status)")
        lanes = r["lanes"] or ["lane"]
        ax.set_ylim(-0.6, len(lanes) - 0.4)
        ax.set_yticks(range(len(lanes)))
        ax.set_yticklabels(lanes)
        ax.invert_yaxis()
        ax.set_xlim(0, r["dur"] / 60)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}m"))
        self.lane_run = r

    def _activity_for(self, r):
        ax, c, M = self.ax_act, self.c, self.M
        ax.clear()
        self._style(ax, f"Tool activity by thread · gen {r['gen']}  (grey = thinking; upper = thread, lower = its subagents)")
        rows = {a["id"] for a in M.agents.values() if a["run"] == r["id"] and a["role"] == "researcher"}
        rows |= {x[4] for x in M.tools if x[2] == r["id"]} | {x[4] for x in M.llms if x[2] == r["id"]}
        rows = sorted(rows, key=lambda s: (s != "orchestrator", len(s), s))
        yi = {k: i for i, k in enumerate(rows)}
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([x if len(x) <= 12 else x[:11] + "…" for x in rows], fontsize=7)
        ax.set_xlim(0, r["dur"] / 60)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}m"))
        rel = lambda s: (s - r["start"]) / 60  # noqa: E731
        segs, cols = [], []
        for t0, t1, run, key, root, sub, *_ in M.llms:
            if run != r["id"]:
                continue
            y = yi[root] - 0.32 + (0.12 if sub else 0)
            segs.append([(rel(t0), y), (rel(t1), y)])
            cols.append(to_rgba(c["muted"], 0.35 if sub else 0.75))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.6))
        min_w = r["dur"] / 60 / 500
        for t0, t1, run, key, root, sub, tool, ok in M.tools:
            if run != r["id"]:
                continue
            y = yi[root] + (0.2 if sub else -0.08)
            ax.add_patch(Rectangle((rel(t0), y - 0.1), max(min_w, rel(t1) - rel(t0)), 0.2, color=c["tool"][tool],
                                   alpha=1 if ok else 0.45, lw=0))
        self.curtain = Rectangle((0, -1), r["dur"] / 60 * 2, len(rows) + 2, color=c["surface"], lw=0, zorder=5)
        ax.add_patch(self.curtain)
        self.act_ph = ax.axvline(0, color=c["ink"], lw=1, zorder=6)
        self.act_run = r

    # ---------- per frame
    def _add(self, art):
        self.dyn.append(art)
        return art

    def draw(self, T):
        for a in self.dyn:
            a.remove()
        self.dyn = []
        M, c = self.M, self.c
        r = M.current_run(T)
        sp = self.speed
        self._draw_topo(T, r, sp)
        self._draw_metric(T, sp)
        self._draw_lanes(T, r)
        self._draw_activity(T, r)
        self._draw_feed(T)
        self._draw_timeline(T)
        ds = [d for d in r["decisions"] if d["t"] <= T]
        decided = sum(1 for d in ds if d["verdict"] != "rollback")
        adopts = sum(1 for d in ds if d["verdict"] == "adopt")
        bp = [v for t, v, _ in r["best"] if t <= T]
        busy = sum(1 for j in r["jobs"] if j["t0"] <= T and (j["t1"] is None or j["t1"] > T))
        task = (r["meta"] or {}).get("task")
        self.title.set_text(f"RSI run replay  ·  gen {r['gen']}  ·  {fmt_dur(min(T, r['end']) - r['start'])} into run")
        self.sub.set_text(f"{r['id']}  ·  {task if isinstance(task, str) else ''}  ·  {(r['meta'] or {}).get('policy', '')}  ·  replay {sp:.0f}×")
        self.kpi.set_text(f"trials {decided}/{r['budget'] or '?'}  adopt {adopts}  best {fmt_v(bp[-1] if bp else None)}  lanes {busy}/{max(1, len(r['lanes']))}")

    def _draw_topo(self, T, r, sp):
        ax, c, M = self.ax_topo, self.c, self.M
        fade, pop, pulse, md = 2.5 * sp, 0.5 * sp, 0.9 * sp, 1.1 * sp
        vis = {}
        for a in M.agents.values():
            if a["spawn"] > T or "pos" not in a:
                continue
            end = a["end"] if a["end"] is not None else math.inf
            if T > end + fade:
                continue
            alpha = 1.0 if T <= end else max(0.0, 1 - (T - end) / fade) * 0.85
            scale = min(1.0, 0.4 + 0.6 * (T - a["spawn"]) / pop)
            vis[a["key"]] = (a, alpha, scale)
        segs, cols = [], []
        for k, (a, al, _) in vis.items():
            if a["pkey"] in vis:
                p = vis[a["pkey"]][0]
                segs.append([p["pos"], a["pos"]])
                cols.append(to_rgba(c["axis"], min(al, vis[a["pkey"]][1])))
        self._add(ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.2, zorder=1)))
        # messages
        segs, cols, dots, dcols = [], [], [], []
        for t, run, fk, tk, kind in M.msgs:
            if t < T - md:
                continue
            if t > T:
                break
            if fk not in vis or tk not in vis:
                continue
            p = (T - t) / md
            a, b = vis[fk][0]["pos"], vis[tk][0]["pos"]
            col = c["running"] if kind == "eval" else c["ink2"]
            segs.append([a, b])
            cols.append(to_rgba(col, 0.55 * (1 - p)))
            q = min(1.0, p * 1.25)
            dots.append((a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q))
            dcols.append(to_rgba(c["running"] if kind == "eval" else c["ink"], 1 if p < 0.8 else (1 - p) * 5))
        if segs:
            self._add(ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.3, zorder=2)))
            self._add(ax.scatter(*zip(*dots), s=22, c=dcols, zorder=6, linewidths=0))
        # nodes
        for role in set(NODE_MK):
            pts = [(a, al, sc) for a, al, sc in vis.values() if (a["role"] if a["role"] in NODE_MK else "subagent") == role]
            if not pts:
                continue
            xy = [a["pos"] for a, _, _ in pts]
            size = [(2 * NODE_PT[role] * sc) ** 2 * (1.8 if role == "lessons-writer" else 1) for _, _, sc in pts]
            fc = c["ink"] if role == "orchestrator" else c["running"] if role == "evald" else c["surface2"]
            ec = c["surface"] if role == "orchestrator" else c["ink2"]
            self._add(ax.scatter(*zip(*xy), s=size, marker=NODE_MK[role], c=[to_rgba(fc, al) for _, al, _ in pts],
                                 edgecolors=[to_rgba(ec, al) for _, al, _ in pts], linewidths=1.3, zorder=4))
            for a, al, _ in pts:
                small = role in ("subagent", "critic")
                lab = a["id"].split("/")[-1].replace("sub-", "s") if small else a["id"]
                self._add(ax.text(a["pos"][0], a["pos"][1] - (0.075 if small else 0.11), lab, ha="center", va="top",
                                  fontsize=6 if small else 7.5, color=c["muted"] if small else c["ink2"], alpha=al, zorder=7))
        # tool pulses + thinking rings
        rx, rs, rc, rl = [], [], [], []
        for t0, t1, run, key, root, sub, tool, ok in M.tools:
            if t1 < T - pulse or t0 > T or key not in vis:
                continue
            a = vis[key][0]
            base = NODE_PT.get(a["role"], 6)
            if T <= t1:
                rad, al = base + 4, 1.0
            else:
                p = (T - t1) / pulse
                rad, al = base + 4 + 14 * p, 1 - p
            rx.append(a["pos"])
            rs.append((2 * rad) ** 2)
            rc.append(to_rgba(c["tool"][tool], al))
            rl.append("solid" if ok else "dashed")
        for t0, t1, run, key, root, sub in M.llms:
            if t0 <= T <= t1 and key in vis:
                a = vis[key][0]
                rx.append(a["pos"])
                rs.append((2 * (NODE_PT.get(a["role"], 6) + 3)) ** 2)
                rc.append(to_rgba(c["muted"], 0.9))
                rl.append("dotted")
        if rx:
            self._add(ax.scatter(*zip(*rx), s=rs, facecolors="none", edgecolors=rc, linewidths=2, linestyles=rl, zorder=3))
        live = sum(1 for a, al, _ in vis.values() if al == 1.0)
        self._add(ax.text(self.topo_xmin + 0.03, 1.07, f"gen {r['gen']} · {live} live agents", fontsize=7.5, color=c["muted"], va="top"))

    def _draw_metric(self, T, sp):
        ax, c, M = self.ax_met, self.c, self.M
        flash = 1.5 * sp
        for r in M.run_list:
            if r["start"] > T:
                continue
            col = c["gen"][r["gen"] % 4]
            pts = [(t, v, d) for t, v, d in r["best"] if t <= T and v is not None]
            if not pts:
                continue
            tend = min(T, r["end"])
            xs = [(t - r["start"]) / 60 for t, _, _ in pts] + [(tend - r["start"]) / 60]
            ys = [v for _, v, _ in pts] + [pts[-1][1]]
            cand = [((x["t"] - r["start"]) / 60, x["value"]) for x in r["evals"] if x["t"] <= T and x["value"] is not None and not x["baseline"]]
            if cand:
                self._add(ax.scatter(*zip(*cand), s=9, color=col, alpha=0.3, lw=0, zorder=2))
            self._add(ax.plot(xs, ys, drawstyle="steps-post", color=col, lw=2, zorder=3)[0])
            self._add(ax.text(xs[-1] + 0.4, ys[-1], f"gen {r['gen']}", fontsize=8, fontweight="bold", color=c["ink"], va="center"))
            for t, v, d in pts:
                if d is None:
                    continue
                x = (t - r["start"]) / 60
                fresh = T - t < flash
                vd = d["verdict"]
                ev = next((e for e in r["evals"] if e["trial"] == d["trial"] and e["value"] is not None), None)
                if vd == "adopt":
                    self._add(ax.scatter([x], [v], s=70 if fresh else 36, color=c["v"]["adopt"], edgecolors=c["surface"], linewidths=1.5, zorder=5))
                elif vd == "rollback":
                    if d["before"] is not None:
                        self._add(ax.plot([x, x], [d["before"], v], color=c["v"]["rollback"], lw=1.3, ls=(0, (3, 2)), zorder=4)[0])
                    self._add(ax.scatter([x], [v], marker="D", s=60 if fresh else 34, color=c["v"]["rollback"], edgecolors=c["surface"], linewidths=1.2, zorder=5))
                elif vd == "explore" and ev:
                    self._add(ax.scatter([x], [ev["value"]], s=30, facecolors="none", edgecolors=c["v"]["explore"], linewidths=1.4, zorder=5))
                elif vd == "reject" and ev:
                    self._add(ax.scatter([x], [ev["value"]], marker="x", s=28 if fresh else 16, color=c["v"]["reject"], linewidths=1.3, zorder=4))

    def _draw_lanes(self, T, r):
        if self.lane_run is not r:
            self._lanes_for(r)
        ax, c = self.ax_lane, self.c
        lanes = r["lanes"] or ["lane"]
        rel = lambda s: (s - r["start"]) / 60  # noqa: E731
        for j in r["jobs"]:
            if j["t0"] > T:
                continue
            done = j["t1"] is not None and j["t1"] <= T
            end = j["t1"] if done else T
            y = lanes.index(j["lane"]) if j["lane"] in lanes else 0
            w = max(r["dur"] / 60 / 400, rel(end) - rel(j["t0"]))
            st = j["status"] if done else "running"
            self._add(ax.add_patch(Rectangle((rel(j["t0"]), y - 0.32), w, 0.64, facecolor=c["st"].get(st, STATUS["critical"]),
                                             edgecolor=c["surface"], lw=0.8, hatch=None if done else "///")))
            if w > r["dur"] / 60 * 0.035:
                dark_txt = done and st in ("timeout", "nan")
                self._add(ax.text(rel(j["t0"]) + r["dur"] / 60 * 0.006, y, f"#{j['trial']}", fontsize=6.5, va="center",
                                  color="#0b0b0b" if dark_txt else "#ffffff", fontweight="bold"))
            if any(a <= T for a in j["audit"]):
                self._add(ax.text(rel(j["t0"]) + w + r["dur"] / 60 * 0.004, y, "!", color=STATUS["critical"], fontsize=9, fontweight="bold", va="center"))
        x = rel(min(T, r["end"]))
        self._add(ax.axvline(x, color=c["ink"], lw=1))

    def _draw_activity(self, T, r):
        if self.act_run is not r:
            self._activity_for(r)
        x = (min(T, r["end"]) - r["start"]) / 60
        self.curtain.set_x(x)
        self.act_ph.set_xdata([x, x])

    def _draw_feed(self, T):
        ax, c, M = self.ax_feed, self.c, self.M
        items = [f for f in M.feed if f[0] <= T and f[3] not in ("spawn",)][-8:][::-1]
        for i, (t, run, agent, kind, text) in enumerate(items):
            y = 0.97 - i * 0.125
            r = M.runs[run]
            if kind.startswith("decision:"):
                col = c["v"].get(kind.split(":")[1], c["muted"])
            elif kind.startswith("eval_result:"):
                col = c["st"].get(kind.split(":")[1], STATUS["critical"])
            else:
                col = {"audit": STATUS["critical"], "eval_submit": c["running"], "lessons": c["s"][4], "proposal": c["s"][1],
                       "run": c["ink"]}.get(kind, c["axis"])
            self._add(ax.text(0.0, y, fmt_dur(t - r["start"]), fontsize=7, color=c["muted"], va="top", transform=ax.transAxes, family="monospace"))
            self._add(ax.add_patch(Rectangle((0.085, y - 0.075), 0.012, 0.06, color=col, transform=ax.transAxes)))
            self._add(ax.text(0.105, y, f"g{r['gen']} {agent}", fontsize=7, color=c["ink2"], va="top", transform=ax.transAxes))
            txt = text if len(text) <= 64 else text[:63] + "…"
            self._add(ax.text(0.29, y, txt, fontsize=7, color=c["ink"], va="top", transform=ax.transAxes))

    def _draw_timeline(self, T):
        ax, c = self.ax_tl, self.c
        self._add(ax.add_patch(Rectangle((0, 0.0), T, 0.06, color=c["s"][0], lw=0)))
        self._add(ax.axvline(T, color=c["ink"], lw=1.5))
        self._add(ax.text(1.0, 1.25, f"{fmt_dur(T)} / {fmt_dur(self.M.dur)}", transform=ax.transAxes, ha="right", fontsize=7.5, color=c["ink2"]))


# ============================== encoding ==============================
def pick_encoder(ffmpeg):
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    for enc in ("libx264", "libopenh264", "mpeg4"):  # software encoders only
        if f" {enc} " in out:
            return enc
    return None


def make_gif(ffmpeg, mp4, gif, fps, width, colors=128):
    vf = (f"fps={fps},scale={width}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
          f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-threads", "4", "-i", mp4, "-vf", vf, gif], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events")
    ap.add_argument("-o", "--out", default="replay", help="output path stem (writes STEM.mp4 and STEM.gif)")
    ap.add_argument("--duration", type=float, default=45.0, help="replay length in seconds (excl. end hold)")
    ap.add_argument("--hold", type=float, default=2.5, help="seconds to hold the final frame")
    ap.add_argument("--fps", type=int, default=24, help="MP4 frame rate")
    ap.add_argument("--gif-fps", type=int, default=12)
    ap.add_argument("--gif-width", type=int, default=900)
    ap.add_argument("--gif-max-mb", type=float, default=9.5, help="re-encode with fewer colors / fps if larger")
    ap.add_argument("--size", default="1280x720", help="MP4 frame size WxH")
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-mp4", action="store_true", help="GIF only (still renders via a temporary MP4 when ffmpeg exists)")
    ap.add_argument("--png-at", type=float, default=None, help="write a single PNG at this fraction (0-1) of the run and exit")
    a = ap.parse_args()

    events = []
    with open(a.events, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(e, dict) and "t" in e and "ev" in e:
                    events.append(e)
    if not events:
        sys.exit("no events")
    M = Model(events)
    W, H = (int(x) for x in a.size.split("x"))
    speed = M.dur / a.duration
    R = Renderer(M, a.theme, W, H, speed)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)

    if a.png_at is not None:
        R.draw(M.dur * a.png_at)
        path = a.out + ".png"
        R.fig.savefig(path, dpi=R.dpi, facecolor=R.fig.get_facecolor())
        print("wrote", path)
        return

    n_play = int(round(a.duration * a.fps))
    n_hold = int(round(a.hold * a.fps))
    times = [M.dur * i / max(1, n_play - 1) for i in range(n_play)] + [M.dur] * n_hold
    ffmpeg = shutil.which("ffmpeg")
    enc = pick_encoder(ffmpeg) if ffmpeg else None
    t_start = time.time()

    if enc:
        mp4 = a.out + ".mp4"
        extra = ["-pix_fmt", "yuv420p", "-threads", "4"]
        if enc == "libx264":
            extra += ["-crf", "23", "-preset", "medium"]
        writer = animation.FFMpegWriter(fps=a.fps, codec=enc, bitrate=1600 if enc != "libx264" else -1, extra_args=extra)
        with writer.saving(R.fig, mp4, dpi=R.dpi):
            for i, T in enumerate(times):
                R.draw(T)
                writer.grab_frame(facecolor=R.fig.get_facecolor())
                if i % 120 == 0:
                    print(f"  frame {i}/{len(times)}  ({time.time() - t_start:.0f}s)", flush=True)
        print(f"wrote {mp4} ({os.path.getsize(mp4) / 1e6:.1f} MB, encoder {enc}, {time.time() - t_start:.0f}s)")
        if not a.no_gif:
            gif = a.out + ".gif"
            fps, colors = a.gif_fps, 128
            for _ in range(4):
                make_gif(ffmpeg, mp4, gif, fps, a.gif_width, colors)
                mb = os.path.getsize(gif) / 1e6
                if mb <= a.gif_max_mb:
                    break
                print(f"  gif {mb:.1f} MB > {a.gif_max_mb} MB; retrying smaller")
                colors, fps = max(32, colors // 2), max(6, fps - 2)
            print(f"wrote {gif} ({os.path.getsize(gif) / 1e6:.1f} MB, {fps} fps, {colors} colors)")
        if a.no_mp4:
            os.remove(mp4)
    else:
        print("ffmpeg (software H.264/MPEG-4) not found: writing GIF with Pillow only")
        step = max(1, round(a.fps / a.gif_fps))
        gif_times = times[::step]
        writer = animation.PillowWriter(fps=a.gif_fps)
        scale = a.gif_width / W
        R.fig.set_size_inches(W * scale / R.dpi, H * scale / R.dpi)
        with writer.saving(R.fig, a.out + ".gif", dpi=R.dpi):
            for T in gif_times:
                R.draw(T)
                writer.grab_frame(facecolor=R.fig.get_facecolor())
        print("wrote", a.out + ".gif")


if __name__ == "__main__":
    main()
