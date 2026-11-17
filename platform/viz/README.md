# viz: replay visualizer for RSI-mini runs

Replays a run from `events.jsonl` alone (schema: `../EVENTS_SCHEMA.md`, v1). Nothing here
writes to a run directory. Everything renders on the CPU.

| file | what it does |
|---|---|
| `replay.html` | Interactive replay. A single page with no build step. It uses d3 7.9.0 from cdnjs, pinned to that version. |
| `embed.py` | Copies `replay.html` with an events file inlined, so the page opens offline from `file://` (d3 still loads from the CDN). |
| `render_video.py` | Renders the same replay to MP4 and GIF, frame by frame (matplotlib Agg + ffmpeg software encoder). |
| `summary.py` | Run statistics: trials per thread, adoption rate, lane utilization, time split, nAUC. |
| `synth.py` | Deterministic synthetic run generator (gen-0 run → lessons_update → gen-1 run). |
| `samples/` | `synth_events.jsonl`, `replay_synth.{mp4,gif,html}`, `replay_synth_screenshot.png` |

Requirements: Python ≥ 3.9 with matplotlib (tested with 3.9.2), plus `ffmpeg` on PATH for MP4 output.
No pip installs were needed, so there is no `.venv`. If you ever need one, put it at `viz/.venv`, which is
git-ignored.

## Usage

```bash
cd platform/viz
python3 synth.py -o samples/synth_events.jsonl            # --seed, --threads 3,4, --trials 16
python3 summary.py samples/synth_events.jsonl              # --s-ref 0.99 for nAUC, --json
python3 render_video.py samples/synth_events.jsonl -o samples/replay_synth   # -> .mp4 + .gif
python3 render_video.py ev.jsonl -o out --theme dark --duration 30
python3 render_video.py ev.jsonl -o still --png-at 0.8     # one PNG at 80% of the run
python3 embed.py samples/synth_events.jsonl -o samples/replay_synth.html
```

To open the replay page, use one of these:
- Open `replay.html`, then click **Open events.jsonl** or drag a file onto the page.
- Open an embedded copy made by `embed.py`.
- Serve the folder over http and use `replay.html?src=../runs/<id>/events.jsonl`.

Optional URL parameters: `t=<seconds>` starts paused at that time, `speed=<x>`, `play=0`, `theme=light|dark`.

Keyboard: space toggles play/pause, and ←/→ step by 1 % of the run. You can also click the marker strip
above the slider to seek.

`render_video.py` picks `libx264`, then `libopenh264`, then `mpeg4`. It never uses VideoToolbox, so it stays
off the GPU. It then builds a palette-optimized GIF (default 12 fps, 900 px wide). If the GIF comes out over
`--gif-max-mb` (default 9.5), it retries with fewer colors and a lower fps. Without ffmpeg it writes the
GIF only, through Pillow. The default 45 s replay plus a 2.5 s hold at 1280×720 takes about 2 min on CPU.

## How each event type is drawn

The colors come from one fixed categorical order, plus a fixed status palette. Light and dark modes
use their own validated steps, so they are not an automatic flip.

| `ev` | replay.html / video |
|---|---|
| `run_start` / `run_end` | A band per run on the timeline strip, labelled `gen k`. A run's start resets the per-run panels (lanes, activity, KPIs). Its end fades that run's agents. The metric chart overlays the generations against minutes since their own run started. |
| `agent_spawn` | A node appears in the topology and pops in from small. Shape gives the role: filled big circle = orchestrator, circle = researcher, small circle = subagent, small diamond = critic, star = lessons-writer, blue square = evald. Agents that only appear as `agent` or `to` are created implicitly when first seen. |
| `agent_end` | The node fades out over about 2.5 s of playback. |
| `message` | A dot travels along a sender → receiver line, and the line fades behind it. The summary shows in the event feed. |
| `llm_call` | Thinking. A dotted grey spinning ring shows on the node during `[t − latency_s, t]`. The activity raster shows a grey bar in the top band. Counts toward thinking time in `summary.py`. |
| `tool_call` | A colored ring around the agent while the call runs, then an expanding pulse. Color = tool: bash blue, edit orange, read aqua, web_search yellow, fetch magenta, `mcp:*` violet, other grey. `ok:false` gives a dashed ring and a faded tick. The activity raster draws a tick per call, on the thread's own row (middle band) or, for its subagents, in the bottom band. |
| `proposal` | An event feed entry. The hypothesis goes in the feed and in tooltips. |
| `eval_submit` | A dashed blue edge orchestrator → evald. A striped "running" bar starts on that lane in the GPU Gantt. |
| `eval_result` | The lane bar closes and takes its status color: ok green, timeout amber, nan coral, crash red, audit_fail red. A dashed edge evald → orchestrator. If there is a value, a faint candidate dot goes on the metric chart. Trial 0 is the baseline s₀. |
| `decision` | The best-so-far step line moves to `best_after`. Markers: adopt = green dot, reject = grey × at the candidate's value, explore = hollow blue circle, rollback = red diamond with a dashed drop from `best_before`. Each decision also gets a tick on the timeline strip (tall for adopt/rollback, short grey for reject). |
| `audit_flag` | A red **!** next to the job's lane bar and on the timeline strip, plus a red feed entry. |
| `lessons_update` / `weights_update` | A ◆ on the timeline strip and a feed entry. The lessons-writer node shows while it works. |

The metric axes are true linear scales. The y axis is zoomed to the data by default, and the chart
says so in text ("zoomed, not zero-based"). The baseline s₀ is a labelled dashed line. The y toggle
switches to a zero-based axis, and the x toggle switches from time to evaluator calls, which is
the design doc's main axis. The metric direction comes from the first adopt
decision (`best_after < best_before` means lower is better). If there is no adopt decision, it is
guessed from the metric name.

## summary.py definitions

- **Trial attribution**: the proposal's `agent` if it is not the orchestrator. Otherwise, the first
  non-orchestrator agent whose message or call carries that `trial`. The synthetic thread tags its
  proposal message with the trial.
- **Adoption rate**: adopts divided by decided trials. Rollbacks are counted separately.
- **Lane utilization**: busy time per lane (submit → result) divided by the run's wall time.
- **Time split**: summed `llm_call.latency_s` (thinking), summed `tool_call.latency_s` (tool use),
  and summed lane busy time (evaluation). All three are agent-seconds, so with parallel agents they
  can exceed wall-clock.
- **nAUC** (from `research/platform_design.md` §1): `(1/N) Σ_t g(t)` with
  `g(t) = (s0 − best_t)/(s0 − s_ref)`. Here t counts evaluator calls and N is `budget.trials`. It
  needs `s_ref`, which comes from `--s-ref`, `run_start.task.s_ref` or `run_start.s_ref`. Without
  `s_ref`, the script prints the logged `run_end.nAUC` and the unnormalized mean gain.

## Synthetic data (`synth.py`)

The generator is a seeded discrete-event simulation, so the same seed gives byte-identical output.
The gen-0 run has 3 researcher threads. Each thread spawns subagents and critics, makes tool calls
and LLM calls, and sends proposals. The orchestrator queues each proposal FIFO onto 2 GPU lanes,
where evald returns ok, timeout, crash, nan or audit_fail. The orchestrator then makes a greedy
adopt/reject decision, with occasional explore.

Two events are forced so they always appear: one audit_fail, and one sealed-seed recheck `audit_flag`
that leads to a rollback. At the end of gen 0, a lessons-writer produces a `lessons_update`, and the
gen-1 run follows with 4 threads and the new scaffold hash. Values are made up around a nanochat
val_bpb baseline of 1.0587. They are illustration data only, not results.
