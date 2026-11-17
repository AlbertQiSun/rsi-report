#!/usr/bin/env python3
"""Deterministic synthetic events.jsonl generator (RSI-mini event schema v1).

Produces realistic-looking multi-agent self-improvement runs so the replay
visualizer, video renderer and summary script can be developed before real
runs exist.  Same --seed => byte-identical output.

What one file contains (defaults):
  gen-0 run: 3 researcher threads, 16 trials, 2 GPU lanes
             -> lessons-writer -> lessons_update -> run_end
  gen-1 run: 4 researcher threads (improved scaffold), 16 trials, 2 GPU lanes

Each thread loops: think (llm_call) -> recon tool calls (read / bash /
web_search / fetch / mcp:lab) -> optional subagent / critic spawn -> edits ->
smoke test -> proposal message -> orchestrator `proposal` -> `eval_submit`
on a free lane (FIFO queue when both are busy) -> evald `eval_result`
(ok / timeout / crash / nan / audit_fail) -> orchestrator `decision`
(adopt / reject / explore / rollback) -> result message back to the thread.

Usage:
  python3 synth.py -o samples/synth_events.jsonl
  python3 synth.py --seed 7 --threads 2,3 --trials 12 -o /tmp/x.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
from datetime import datetime, timezone

MODEL = "qwen3.5-9b-instruct"
LANES = ["gpu0", "gpu1"]
EVAL_SEEDS = [1337, 1338, 1339]
METRIC = "val_bpb"

HYPOTHESES = [
    "Raise Muon matrix LR 0.02 -> 0.03; deeper nets tolerated it in the depth sweep",
    "Lengthen warmdown ratio 0.2 -> 0.4 so the LR tail spends more steps near zero",
    "Untie input embedding and lm_head; separate AdamW LR for the head",
    "Add QK-norm (RMSNorm on q,k) to allow a higher attention LR without spikes",
    "Swap GELU MLP for ReLU^2 at equal parameter count",
    "Add value embeddings on alternating layers (modded-nanogpt style)",
    "Halve total batch 2^19 -> 2^18 tokens and double the step count at fixed wall",
    "Logit soft-cap 15 on the lm_head output",
    "Rotary base 10k -> 50k for the 2048-token context",
    "Sliding-window attention (w=512) on even layers to cut attention FLOPs",
    "Zero-init attention and MLP output projections",
    "AdamW beta2 0.95 -> 0.99 for embedding parameters only",
    "Gradient clipping at 1.0 on the AdamW group",
    "Sequence length 2048 -> 1024 with more optimizer steps in the same wall budget",
    "Drop all linear biases and the final norm affine",
    "Add z-loss 1e-4 on logits for stability at the higher LR",
    "Muon Newton-Schulz 5 -> 4 iterations to save step time",
    "Warmup 0 -> 50 steps on the embedding optimizer",
    "Per-layer LR scaling 1/sqrt(depth) on Muon groups",
    "torch.compile max-autotune on the MLP block only",
]

TOOL_ARGS = {
    "bash": [
        "rg -n 'matrix_lr' nanochat/",
        "tail -n 40 runs/last/train.log",
        "uv run python -m scripts.base_train --depth=12 --num_iterations=40 (smoke)",
        "git diff --stat",
        "python -c 'import torch; print(torch.cuda.mem_get_info())'",
        "uv run pytest tests/test_gpt.py -q",
    ],
    "edit": [
        "nanochat/gpt.py +6 -2",
        "nanochat/muon.py +3 -3",
        "scripts/base_train.py +4 -1",
        "nanochat/gpt.py +11 -4",
        "configs/d12.yaml +2 -2",
    ],
    "read": [
        "nanochat/gpt.py L1-160",
        "nanochat/muon.py L1-120",
        "scripts/base_train.py L200-320",
        "LESSONS.md",
        "runs/last/metrics.json",
    ],
    "web_search": [
        "muon optimizer learning rate scaling with depth",
        "QK norm small transformer training stability",
        "relu squared vs gelu speedrun",
        "value embeddings modded-nanogpt",
        "warmdown ratio wsd schedule ablation",
    ],
    "fetch": [
        "arxiv.org/abs/2502.16982 (Muon is scalable)",
        "github.com/KellerJordan/modded-nanogpt records README",
        "kellerjordan.github.io/posts/muon",
    ],
    "mcp:lab": [
        "lab.query_trials(metric=val_bpb, top=5)",
        "lab.get_lessons()",
        "lab.diff(trial=best)",
        "lab.query_trials(status=crash)",
    ],
}

AUDIT_CHECKS = [
    ("eval_call_in_solution", "solution script calls evaluate() directly instead of the harness forward()"),
    ("timer_override", "train loop patches time.time; harness wall-clock disagrees by 41 s"),
    ("val_shard_read", "process opened data/val_shard_000 during training"),
]


def h12(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


class Signal:
    def __init__(self):
        self.fired = False
        self.value = None
        self.waiters = []


class Sim:
    """Tiny discrete-event simulator driving generator coroutines.

    A coroutine yields ("sleep", seconds) or ("wait", Signal)."""

    def __init__(self, t0: float):
        self.now = t0
        self.q = []
        self.n = 0
        self.events = []

    def _sched(self, t, g, val=None):
        heapq.heappush(self.q, (t, self.n, g, val))
        self.n += 1

    def spawn(self, g, delay=0.0):
        self._sched(self.now + delay, g)

    def fire(self, sig: Signal, value=None):
        sig.fired, sig.value = True, value
        for g in sig.waiters:
            self._sched(self.now, g, value)
        sig.waiters = []

    def run(self):
        while self.q:
            t, _, g, val = heapq.heappop(self.q)
            self.now = t
            try:
                kind, arg = g.send(val)
            except StopIteration:
                continue
            if kind == "sleep":
                self._sched(self.now + arg, g)
            elif arg.fired:
                self._sched(self.now, g, arg.value)
            else:
                arg.waiters.append(g)

    def emit(self, **ev):
        ev["t"] = round(self.now, 3)
        self.events.append(ev)


def sleep(s):
    return ("sleep", float(s))


def wait(sig):
    return ("wait", sig)


class Run:
    def __init__(self, sim, rng, *, run_id, gen, n_threads, budget, skill,
                 scaffold_hash, s0, s_ref, floor, make_lessons, force_rollback,
                 force_audit_fail):
        self.sim, self.rng = sim, rng
        self.run_id, self.gen = run_id, gen
        self.n_threads, self.budget = n_threads, budget
        self.skill = skill  # mean candidate delta (negative = helpful)
        self.scaffold_hash = scaffold_hash
        self.s0, self.s_ref, self.floor = s0, s_ref, floor
        self.make_lessons = make_lessons
        self.force_rollback = force_rollback
        self.force_audit_fail = force_audit_fail
        self.next_trial = 1
        self.best = None
        self.queue = []
        self.lane_busy = {lane: None for lane in LANES}
        self.results = {}  # trial -> Signal
        self.trial_thread = {}
        self.adopted = {}  # trial -> value, adopted and not rolled back
        self.n_adopt = 0
        self.rollback_target = None
        self.best_trace = []  # best after each trial decision, for nAUC
        self.hyp_pool = list(HYPOTHESES)
        rng.shuffle(self.hyp_pool)
        self.final_scaffold = scaffold_hash

    # ---- emit helpers -------------------------------------------------
    def e(self, ev, agent, parent=None, trial=None, **kw):
        self.sim.emit(run=self.run_id, ev=ev, agent=agent, parent=parent,
                      gen=self.gen, trial=trial, **kw)

    def parent_of(self, agent):
        if agent in ("orchestrator",):
            return None
        if "/" in agent:
            return agent.rsplit("/", 1)[0]
        return "orchestrator"

    def msg(self, frm, to, summary, trial=None):
        self.e("message", frm, self.parent_of(frm), trial, to=to,
               summary=summary[:200], tokens=self.rng.randint(60, 900))

    def llm(self, agent, lo=600, hi=30000):
        r = self.rng
        pt, ct = r.randint(lo, hi), r.randint(80, 1400)
        lat = ct / 48 + pt / 5000 + r.uniform(0.5, 2.0)
        yield sleep(lat)
        self.e("llm_call", agent, self.parent_of(agent), model=MODEL,
               prompt_tokens=pt, completion_tokens=ct, latency_s=round(lat, 2))

    def tool(self, agent, tool, smoke=False):
        r = self.rng
        lat = {
            "bash": r.uniform(35, 95) if smoke else r.uniform(0.3, 8),
            "edit": r.uniform(0.2, 1.2),
            "read": r.uniform(0.1, 0.6),
            "web_search": r.uniform(1.5, 6),
            "fetch": r.uniform(1, 5),
            "mcp:lab": r.uniform(0.3, 2),
        }[tool]
        ok = r.random() > (0.18 if tool == "bash" else 0.04)
        arg = TOOL_ARGS[tool][2] if smoke else r.choice(TOOL_ARGS[tool])
        yield sleep(lat)
        self.e("tool_call", agent, self.parent_of(agent), tool=tool,
               arg_summary=arg, ok=ok, latency_s=round(lat, 2))

    def pick_tool(self, weights):
        tools, w = zip(*weights.items())
        return self.rng.choices(tools, w)[0]

    # ---- eval lanes ---------------------------------------------------
    def request_eval(self, trial, ahash, cand):
        self.queue.append((trial, ahash, cand))
        self.dispatch()

    def dispatch(self):
        for lane in LANES:
            if self.lane_busy[lane] is None and self.queue:
                trial, ahash, cand = self.queue.pop(0)
                self.lane_busy[lane] = trial
                self.sim.spawn(self.eval_job(lane, trial, ahash, cand))

    def eval_job(self, lane, trial, ahash, cand):
        r = self.rng
        self.e("eval_submit", "orchestrator", None, trial, artifact_hash=ahash,
               lane=lane, seeds=EVAL_SEEDS)
        if trial == 0:
            status = "ok"
        elif self.force_audit_fail and trial == self.force_audit_fail:
            status = "audit_fail"
        else:
            status = r.choices(["ok", "timeout", "crash", "nan", "audit_fail"],
                               [0.78, 0.07, 0.07, 0.05, 0.03])[0]
        wall = {"ok": r.uniform(330, 420), "timeout": 480.0,
                "crash": r.uniform(40, 140), "nan": r.uniform(150, 300),
                "audit_fail": r.uniform(330, 400)}[status]
        value = None
        audit = {"level": "ok", "flags": []}
        if trial == 0:
            value = self.s0
        elif status == "ok":
            value = cand
        elif status == "audit_fail":
            check, detail = r.choice(AUDIT_CHECKS)
            value = round(cand - 0.012, 4)  # suspiciously good
            audit = {"level": "fail", "flags": [check]}
        if status == "audit_fail":
            yield sleep(wall - 2)
            self.e("audit_flag", "evald", "orchestrator", trial,
                   artifact_hash=ahash, check=check, detail=detail)
            yield sleep(2)
        else:
            yield sleep(wall)
        self.e("eval_result", "evald", "orchestrator", trial, artifact_hash=ahash,
               metric=METRIC, value=value, seeds=EVAL_SEEDS, wall_s=round(wall, 1),
               status=status, audit=audit)
        self.lane_busy[lane] = None
        self.dispatch()
        if trial == 0:
            self.best = value
            self.sim.fire(self.results[0], value)
            return
        yield sleep(r.uniform(1.5, 4))
        before = self.best
        tid = self.trial_thread[trial]
        if status != "ok":
            verdict, reason = "reject", f"status={status}"
        elif value < self.best:
            verdict, reason = "adopt", f"{METRIC} {value:.4f} < best {self.best:.4f}"
        elif value - self.best < 0.0015 and r.random() < 0.35:
            verdict, reason = "explore", "within noise of best; kept as branch"
        else:
            verdict, reason = "reject", f"{METRIC} {value:.4f} >= best {self.best:.4f}"
        if verdict == "adopt":
            self.best = value
            self.adopted[trial] = value
            self.n_adopt += 1
        self.e("decision", "orchestrator", None, trial, artifact_hash=ahash,
               verdict=verdict, best_before=before, best_after=self.best, reason=reason)
        self.best_trace.append(self.best)
        vtxt = "n/a" if value is None else f"{value:.4f}"
        self.msg("orchestrator", tid, f"trial {trial}: {verdict} ({METRIC} {vtxt}, best {self.best:.4f})", trial)
        self.sim.fire(self.results[trial], (verdict, value))
        if verdict == "adopt" and self.force_rollback and self.n_adopt == 2 and self.rollback_target is None:
            self.rollback_target = trial
            self.sim.spawn(self.sealed_recheck(trial, ahash), delay=r.uniform(70, 110))

    def sealed_recheck(self, trial, ahash):
        """Sealed-seed recheck of an adopted artifact fails -> rollback."""
        dev = self.adopted[trial]
        sealed = dev + 0.0091
        self.e("audit_flag", "evald", "orchestrator", trial, artifact_hash=ahash,
               check="sealed_seed_recheck",
               detail=f"dev {dev:.4f} vs sealed {sealed:.4f}: gap 0.0091 > 0.005")
        yield sleep(2.5)
        before = self.best
        del self.adopted[trial]
        self.best = min([self.s0] + list(self.adopted.values()))
        self.e("decision", "orchestrator", None, trial, artifact_hash=ahash,
               verdict="rollback", best_before=before, best_after=self.best,
               reason="sealed-seed recheck failed; restore previous best")
        self.msg("orchestrator", self.trial_thread[trial],
                 f"trial {trial} rolled back after sealed-seed recheck; best {self.best:.4f}", trial)

    # ---- agents --------------------------------------------------------
    def subagent(self, tid, k, role, done):
        r = self.rng
        sid = f"{tid}/sub-{k}"
        self.e("agent_spawn", sid, tid, role=role, model=MODEL)
        task = ("critique the current plan and look for eval leaks" if role == "critic"
                else r.choice(["survey prior speedrun records for this idea",
                               "check optimizer code paths for the change",
                               "find failure modes in recent crashes",
                               "estimate step-time impact"]))
        self.msg(tid, sid, task)
        weights = ({"read": 5, "mcp:lab": 2, "bash": 1} if role == "critic"
                   else {"web_search": 3, "fetch": 2, "read": 3, "bash": 2, "mcp:lab": 1 + 2 * self.gen})
        for _ in range(r.randint(3, 7)):
            if r.random() < 0.35:
                yield from self.llm(sid, 1500, 12000)
            else:
                yield from self.tool(sid, self.pick_tool(weights))
        yield from self.llm(sid, 2000, 14000)
        self.msg(sid, tid, "critique: no leak found; risk is step-time regression" if role == "critic"
                 else r.choice(["findings: two records report gains at d12", "findings: change interacts with warmdown",
                                "findings: crash cause was OOM at bs 2^19", "findings: ~3% slower step, net positive"]))
        self.e("agent_end", sid, tid, reason="done")
        self.sim.fire(done)

    def thread(self, i, done):
        r = self.rng
        tid = f"thread-{i}"
        self.e("agent_spawn", tid, "orchestrator", role="researcher", model=MODEL)
        yield sleep(r.uniform(0.5, 2))
        lessons = " with LESSONS.md" if self.gen > 0 else ""
        self.msg("orchestrator", tid, f"brief: minimize {METRIC} on nanochat d12, {self.budget} shared trials{lessons}")
        yield wait(self.results[0])
        pending, nsub = None, 0
        recon = {"read": 4, "bash": 3, "web_search": 2, "fetch": 1, "mcp:lab": 1 + 3 * self.gen}
        while True:
            yield from self.llm(tid)
            for _ in range(r.randint(2, 5)):
                yield from self.tool(tid, self.pick_tool(recon))
            if r.random() < 0.65:
                subs = []
                for _ in range(1 if r.random() < 0.6 else 2):
                    sig = Signal()
                    subs.append(sig)
                    self.sim.spawn(self.subagent(tid, nsub, "subagent", sig), delay=r.uniform(0.2, 2))
                    nsub += 1
                for sig in subs:
                    yield wait(sig)
            if r.random() < 0.2 + 0.2 * self.gen:
                sig = Signal()
                self.sim.spawn(self.subagent(tid, nsub, "critic", sig))
                nsub += 1
                yield wait(sig)
            yield from self.llm(tid)
            for _ in range(r.randint(1, 3)):
                yield from self.tool(tid, "edit")
            yield from self.tool(tid, "bash", smoke=True)
            if pending is not None:
                # while the previous trial is queued / evaluating, keep doing light
                # background work (lab queries, reading, occasional search or helper)
                while not pending.fired:
                    roll = r.random()
                    if roll < 0.45:
                        yield from self.tool(tid, self.pick_tool({"mcp:lab": 3, "read": 3, "bash": 2, "web_search": 1}))
                    elif roll < 0.75:
                        yield from self.llm(tid, 1000, 8000)
                    elif roll < 0.85:
                        sig = Signal()
                        self.sim.spawn(self.subagent(tid, nsub, "subagent", sig))
                        nsub += 1
                        yield wait(sig)
                    yield sleep(r.uniform(15, 50))
                yield from self.llm(tid, 800, 6000)
            if self.next_trial > self.budget:
                break
            trial = self.next_trial
            self.next_trial += 1
            self.results[trial] = Signal()
            self.trial_thread[trial] = tid
            hyp = self.hyp_pool[(trial - 1) % len(self.hyp_pool)]
            # candidate quality: diminishing returns as best approaches floor
            room = (self.best - self.floor) / (self.s0 - self.floor)
            cand = round(self.best + r.gauss(self.skill, 0.0065) * (room + 0.15), 4)
            ahash = h12(self.run_id, trial, hyp)
            self.msg(tid, "orchestrator", f"proposal: {hyp}", trial)
            yield sleep(r.uniform(1, 3))
            files = r.randint(1, 3)
            self.e("proposal", "orchestrator", None, trial, hypothesis=hyp[:300],
                   diff_stat={"files": files, "plus": r.randint(2, 40), "minus": r.randint(0, 20)},
                   artifact_hash=ahash)
            self.request_eval(trial, ahash, cand)
            pending = self.results[trial]
        if pending is not None:
            yield wait(pending)
        self.msg(tid, "orchestrator", "budget exhausted; final notes written to lab")
        self.e("agent_end", tid, "orchestrator", reason="budget_exhausted")
        self.sim.fire(done)

    def lessons_writer(self, done):
        r = self.rng
        lw = "lessons-writer"
        self.e("agent_spawn", lw, "orchestrator", role="lessons-writer", model=MODEL)
        self.msg("orchestrator", lw, f"summarize D_{self.gen} ({self.budget} trials, {self.n_adopt} adopts) into <=300-word lessons")
        for tool in ["mcp:lab", "mcp:lab", "read"]:
            yield from self.tool(lw, tool)
        yield from self.llm(lw, 20000, 32000)
        yield from self.llm(lw, 8000, 16000)
        yield from self.tool(lw, "edit")
        n = r.randint(9, 14)
        self.msg(lw, "orchestrator", f"{n} lessons drafted (LR ranges, crash causes, audit pitfalls)")
        after = h12(self.scaffold_hash, "lessons", n)
        self.e("lessons_update", "orchestrator", None, scaffold_hash_before=self.scaffold_hash,
               scaffold_hash_after=after, n_lessons=n)
        self.final_scaffold = after
        self.e("agent_end", lw, "orchestrator", reason="done")
        self.sim.fire(done)

    def nauc(self):
        n = self.budget
        tr = self.best_trace[:n] + [self.best_trace[-1]] * (n - len(self.best_trace))
        g = [(self.s0 - b) / (self.s0 - self.s_ref) for b in tr]
        return round(sum(g) / n, 4)

    def orchestrator(self):
        r = self.rng
        self.e("run_start", "orchestrator", None, policy="karpathy-greedy", task="nanochat-d12-speedrun",
               improver={"model": MODEL, "scaffold_hash": self.scaffold_hash},
               budget={"trials": self.budget, "wall_s": 3 * 3600},
               config_hash=h12(self.run_id, "cfg"))
        yield sleep(2)
        self.results[0] = Signal()
        self.request_eval(0, h12(self.run_id, "baseline"), None)
        dones = []
        for i in range(self.n_threads):
            sig = Signal()
            dones.append(sig)
            self.sim.spawn(self.thread(i, sig), delay=r.uniform(0.5, 6) + i * 2)
        for sig in dones:
            yield wait(sig)
        if self.rollback_target is not None:
            yield sleep(1)
        if self.make_lessons:
            sig = Signal()
            self.sim.spawn(self.lessons_writer(sig), delay=3)
            yield wait(sig)
        yield sleep(2)
        self.e("run_end", "orchestrator", None, best_value=self.best,
               n_trials=len(self.best_trace), nAUC=self.nauc())


def generate(seed=0, threads=(3, 4), trials=16, start="2026-11-17T23:00:00+00:00"):
    rng = random.Random(seed)
    t0 = datetime.fromisoformat(start).timestamp()
    sim = Sim(t0)
    stamp = datetime.fromtimestamp(t0, timezone.utc).strftime("%Y%m%dT%H%M")
    s0, s_ref, floor = 1.0587, 0.9900, 0.9850
    skills = [0.0018, -0.0012, -0.0030, -0.0040]
    runs = []

    def driver():
        scaffold = h12("scaffold", "v0")
        for gen, n_th in enumerate(threads):
            run = Run(sim, rng, run_id=f"synth-nanochat-9b-karpathy-{stamp}-g{gen}", gen=gen,
                      n_threads=n_th, budget=trials, skill=skills[min(gen, len(skills) - 1)],
                      scaffold_hash=scaffold, s0=s0, s_ref=s_ref, floor=floor,
                      make_lessons=gen < len(threads) - 1, force_rollback=(gen == 0),
                      force_audit_fail=(5 if gen == 0 else None))
            runs.append(run)
            yield from run.orchestrator()
            scaffold = run.final_scaffold
            yield sleep(45)

    sim.spawn(driver())
    sim.run()
    events = sorted(sim.events, key=lambda e: e["t"])  # stable: keeps causal order at equal t
    order = ["t", "run", "ev", "agent", "parent", "gen", "trial"]
    return [{**{k: e[k] for k in order}, **{k: v for k, v in e.items() if k not in order}} for e in events]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="samples/synth_events.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", default="3,4", help="researcher threads per generation, e.g. 3,4 (2-4 each)")
    ap.add_argument("--trials", type=int, default=16, help="trial budget per run")
    a = ap.parse_args()
    threads = tuple(int(x) for x in a.threads.split(","))
    events = generate(a.seed, threads, a.trials)
    with open(a.out, "w") as f:
        for e in events:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")
    print(f"wrote {len(events)} events -> {a.out}")


if __name__ == "__main__":
    main()
