# Second-order kit on the cluster: µAR on node CPUs, U_s / U_w, analysis (owner: worker D)

The command blocks below are what worker F's scheduler calls. They assume island mode on the hold node
`<node>` (36 CPU, 3 × A100). Times marked 〔估算〕 are estimates; the only measured numbers are from the
Mac (M-series, single thread): one µAR seed takes about 0.5–4 s, a 2-seed start-config eval takes 1–8 s,
and the full test suite runs in under a minute. Measure again on the node with the `bench` command.

```bash
R=$R
PY=$SCRATCH/miniconda3/envs/rsi/bin/python          # serving env: torch, numpy, peft, transformers
SRC=$R/platform                                                  # wherever the platform/ tree was synced
export PYTHONPATH=$SRC R
JOB=<jobid>                                                     # the hold; every compute step goes through srun
```

## 0. One-time setup (sealed material; the sandbox must never mount `$R/data/sealed`)

```bash
srun --jobid=$JOB --overlap bash -c '
  umask 077; mkdir -p $R/data/sealed/muar
  [ -s $R/data/sealed/muar/salt ] || head -c 32 /dev/urandom | base64 > $R/data/sealed/muar/salt
  ls -l $R/data/sealed/muar'
```
The salt turns sealed seed indices into secret torch seeds. Dev seeds (0, 1) are public. Without the file the
code falls back to a fixed default salt and reports `salt_source=default`. That fallback is only for tests.

## 1. Self-test and Gate 1 timing (CPU only, about 1 min)

```bash
srun --jobid=$JOB --overlap --cpus-per-task=8 bash -c 'cd $SRC &&
  $PY -m pytest -q tests/test_muar.py tests/test_update.py &&
  $PY -m rsi.tasks.muar.runner bench --split all --workers 8'
```
Gate 1 (design doc §5.1) passes when every start config reports `status=ok` and the per-eval `wall_s` stays
around a few seconds. Scores are deterministic on a given machine, but Mac and node floats may differ.
Compare arms only with evals that ran on the same machine (risk #16).

## 2. s_ref (once per task set, before any arm; result is sealed)

```bash
srun --jobid=$JOB --overlap --cpus-per-task=34 \
  $PY -m rsi.tasks.muar.runner sref --split all --n-random 300 --n-local 100 --workers 34
# -> $R/data/sealed/muar/sref.json  {task: {s_ref, s0, best_config, n_evals, source}}
```
This is 16 tasks × 401 configs × 2 seeds, about 12.8k training runs, roughly 10–15 min on 34 CPUs 〔估算〕.
Later, fold each arm's best dev value in with
`python -c "from rsi.tasks.muar.runner import fold_sref,sref_path; fold_sref(sref_path(),'muar/id-modadd-0',v,'arm A1')"`.
nAUC must be recomputed after the last fold. The analysis reads `sref.json` at the end.

## 3. Non-LLM baselines R and R-local (design doc §1.5 control 7)

```bash
srun --jobid=$JOB --overlap --cpus-per-task=34 \
  $PY -m rsi.tasks.muar.runner baseline --method random,local --split heldout \
      --rng-seeds 0,1,2 --n 12 --workers 34 --out $R/runs/muar-baselines/heldout.json
# also on train tasks if wanted: --split train
```
The output is `{"runs": [...], "arms": {"R-random": curves, "R-local": curves}}`, and the `arms` block plugs
straight into the analysis spec (§6). The run takes about 1–2 min 〔估算〕.

## 4. LLM arms on µAR (A0a, A0b, A1, A1p, A2, A2p)

µAR artifacts are scored on node CPUs, not on a GPU lane. Evald (worker B) serves only `nanochat*`, so the
orchestrator gets the local client instead:

```python
from rsi.tasks.muar.runner import LocalEvalClient
evald = LocalEvalClient(workers=int(os.environ.get("RSI_MUAR_WORKERS", 24)))   # same methods as EvalClient
# task = get_task("muar/id-modadd-0"); ctx = RunContext.create(..., task=task, evald=evald, ...)
```
* Splits: `rsi.tasks.muar.SPLITS` / `tasks_in("train" | "heldout_id" | "heldout_ood" | "heldout")`.
  G0 (data collection) runs on `train`. Every A-arm runs on `heldout` (8 tasks). Each (task, seed) pair is
  one run with N = 12 trials, the same seeds across arms (common random numbers), and A0b uses a disjoint
  seed set.
* Put `"seed": <int>` in each run's `config.json`. The analysis uses it to pair seeds across arms.
* CPU budget: vLLM, the gateway and the harness need a few cores, so use `RSI_MUAR_WORKERS=24` while those run
  and 34 when only µAR runs. Each worker is one process with one thread, so N workers use N cores.
* Sealed re-score of each run's final best, for analysis only:
  `python -c "from rsi.tasks.muar.runner import score_sealed; print(score_sealed('muar/id-modadd-0','<run>/artifacts/<hash>'))"`
  Write the result to `<run>/sealed.jsonl`. Never write it into the lab spool.

## 5a. U_s: lessons, plus the generic placebo (no GPU)

```bash
GW=http://127.0.0.1:$($PY -c "import json;print(json.load(open('$R/run/ports.json'))['gateway'])")/v1
$PY -m rsi.update.lessons --runs $R/runs/<g0-train-run-1> $R/runs/<g0-train-run-2> ... \
    --model improver --gateway-url $GW --out-dir $R/scaffolds/g1-lessons            # prints scaffold hash
W=$($PY -c "import json;print(json.load(open('$R/scaffolds/g1-lessons.manifest.json'))['words'])")
$PY -m rsi.update.lessons --generic-placebo --target-words $W \
    --model improver --gateway-url $GW --out-dir $R/scaffolds/g1-placebo
```
The token is read from `$R/run/token` (or `$RSI_TOKEN_FILE`) and never appears on a command line. Lessons
refuse runs on held-out tasks. The manifest that labels an arm as lessons or placebo is written next to the
scaffold dir, not inside it, so the harness never copies it into the sandbox. The orchestrator emits
`lessons_update` with the returned hash.

## 5b. U_w: SFT data, then LoRA on one lane, then a hot-loaded adapter

```bash
# (1) data: accepted-only, plus the size-matched placebo (legal but rejected). Both pass the held-out leak check.
$PY -m rsi.update.sft --runs $R/runs/<g0-train-runs...> --out $R/sft/g1-acc
$PY -m rsi.update.sft --runs $R/runs/<g0-train-runs...> --out $R/sft/g1-plc --placebo
#    Gate G3: manifest.json by_label.accepted >= 40 (otherwise "hindsight" rows appear; see design §5.1 fallback),
#    leak_check.ok == true. The exit code is 2 when the train set is empty.

# (2) train on lane 2 / GPU 2. F pauses the lane first: EvalClient.pause_lane(2), then wait until not busy.
srun --jobid=$JOB --overlap --gres=gpu:a100:1 --cpus-per-task=6 \
  bash $SRC/bundle/lora_train.sh $R/sft/g1-acc $R/adapters/g1-acc 2      # last stdout line = adapter dir
srun --jobid=$JOB --overlap --gres=gpu:a100:1 --cpus-per-task=6 \
  bash $SRC/bundle/lora_train.sh $R/sft/g1-plc $R/adapters/g1-plc 2
#    then EvalClient.resume_lane(2)

# (3) hot-load into the running vLLM (bundle/serve_vllm.sh 9b ... --lora) through the gateway admin API
$PY - <<'EOF'
import json, os, urllib.request
R = os.environ["R"]; tok = open(f"{R}/run/token").read().strip()
port = json.load(open(f"{R}/run/ports.json"))["gateway"]
for name in ("g1-acc", "g1-plc"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/rsi/load_adapter", method="POST",
        data=json.dumps({"base": "improver", "name": name, "path": f"{R}/adapters/{name}"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok})
    print(name, urllib.request.urlopen(req).read()[:200])
EOF
# arms then use model "improver@g1-acc" (A2) and "improver@g1-plc" (A2p); orchestrator emits weights_update
```
* `srun --overlap --gres` inside the hold: if the hold's GRES does not allow per-step GPU selection, drop
  `--gres` and rely on the script's own `CUDA_VISIBLE_DEVICES=<gpu>`. The script refuses a GPU that still has
  compute processes (set `RSI_ALLOW_BUSY_GPU=1` to override).
* Defaults: r 16, alpha 32, lr 1e-4, 2 epochs, 8 examples per step, max_len 6144, best-by-valid checkpoint,
  hard stop at 55 min (`LORA_TIME_LIMIT_MIN`). Extra flags go in `LORA_ARGS`, for example
  `LORA_ARGS="--epochs 3 --rank 32"` (vLLM must then have `--max-lora-rank >= 32`; serve_vllm.sh defaults to 64).
  A dry run that only tokenizes, with no GPU: `LORA_ARGS=--dry-run`.
* Wall-clock: a few hundred examples × 2–6k tokens × 2 epochs on one A100 should take about 20–50 min
  〔估算〕. Qwen3.5's Gated-DeltaNet layers run much faster with `flash-linear-attention` and
  `causal-conv1d` installed in the env. Without them, transformers uses its torch fallback, which works but
  is slower. Check the speed in the first `[lora]` log lines and lower `--epochs` if needed.
* LoRA targets are q/k/v/o (the 8 full-attention layers) and gate/up/down (all 32 layers), language model only.
  The module names were checked against the HF config and safetensors index and against vLLM's `qwen3_5.py`
  (details in `rsi/update/lora_train.py`).

## 6. Analysis and the pre-registered decision

```bash
cat > $R/analysis/g1.json <<'EOF'
{"s_ref": {...from $R/data/sealed/muar/sref.json: task -> s_ref...},
 "runs": {"A0a": ["<run dirs>"], "A0b": ["..."], "A1": ["..."], "A1p": ["..."], "A2": ["..."], "A2p": ["..."]},
 "aa": ["A0a", "A0b"],
 "compare": [["A1", "A0a", "A1p"], ["A2", "A0a", "A2p"], ["A2", "A1"]],
 "n_trials": 12, "n_boot": 10000, "floor": "per_task"}
EOF
$PY -m rsi.update.analysis $R/analysis/g1.json --out $R/analysis/g1.result.json
```
For each comparison the output gives Δ₂ with a stratified-bootstrap 95% CI, a sign-flip p, ρ, the A/A floor
(`sd_task`, `se_mean`), the MDE, and the three criteria of design doc §1.4. Report ID and OOD separately by
giving each its own spec, with `runs` filtered to `heldout_id` or `heldout_ood` tasks.
`floor: "per_task"` uses 2 × sd of the per-task A/A differences, which is the literal reading and the
conservative one. `"mean"` uses 2 × the standard error of the mean. Fix the choice in PREREG.md before
looking at the results.
