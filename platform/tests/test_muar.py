"""µAR task family tests (CPU only): determinism, budget conversion, illegal configs, sealed leakage."""

from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi.tasks.base import get_task  # noqa: E402
from rsi.tasks.muar import MuarTask, schema, start_config  # noqa: E402
from rsi.tasks.muar.data import GENERATORS, corpus_sha256, make_dataset  # noqa: E402
from rsi.tasks.muar.evaluate import evaluate_artifact, evaluate_config  # noqa: E402
from rsi.tasks.muar.tasks import (DEFECTS, MAX_STEPS, SPECS, SPLITS, budget_for, dims, get_spec,  # noqa: E402
                                  start_config_and_defects, task_card, train_seed)
from rsi.tasks.muar.train import MAX_PARAMS, MIN_STEPS, budget_steps, build_model, n_params_for  # noqa: E402

FAST = "muar/ood-copyrev-0"
CORPUS_SHA = "d3569d3d2e4a07768ddb0a77609efa1ee07891dcebd7e42802a7d4f2141ea486"


def fast_cfg(**kw):
    cfg = dict(schema.DEFAULT_CONFIG, batch_size=256, width=64)
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------------------- determinism
def test_same_config_scores_identical_twice():
    a = evaluate_config(FAST, fast_cfg(), [0, 1])
    b = evaluate_config(FAST, fast_cfg(), [0, 1])
    assert a["status"] == "ok"
    for k in ("value", "seeds", "status", "audit"):
        assert a[k] == b[k]
    assert a["diag"]["per_seed"] == b["diag"]["per_seed"]
    assert a["diag"]["train_curve"] == b["diag"]["train_curve"]
    assert len(a["diag"]["train_curve"]) == 5


def test_seeds_matter_and_dev_differs_from_sealed():
    r = evaluate_config(FAST, fast_cfg(), [0, 1])
    assert r["diag"]["per_seed"][0] != r["diag"]["per_seed"][1]
    s = evaluate_config(FAST, fast_cfg(), [0, 1], split="sealed")
    assert s["split"] == "sealed" and s["value"] != r["value"]
    spec = get_spec(FAST)
    assert train_seed(spec, "dev", 0) != train_seed(spec, "sealed", 0)


def test_datasets_deterministic_and_disjoint():
    for spec in SPECS:
        a = make_dataset(spec.ttype, spec.params, spec.data_seed)
        b = GENERATORS[spec.ttype](spec.params, spec.data_seed)   # uncached regeneration
        for split in ("train", "dev", "sealed"):
            assert (a.X[split] == b.X[split]).all() and (a.Y[split] == b.Y[split]).all()
            assert a.X[split].shape[0] > 0
        if spec.ttype == "modadd":   # unseen pairs only
            tr = {r.tobytes() for r in a.X["train"]}
            assert not any(r.tobytes() in tr for r in a.X["dev"])


def test_corpus_frozen():
    assert corpus_sha256() == CORPUS_SHA


# ---------------------------------------------------------------------------- budget
def test_param_count_matches_torch():
    for norm in ("none", "layernorm", "rmsnorm"):
        for depth in (1, 3):
            cfg = fast_cfg(norm=norm, depth=depth, width=24, dropout=0.1)
            m = build_model(cfg, 17, 5)
            assert sum(p.numel() for p in m.parameters()) == n_params_for(cfg, 17, 5)


def test_budget_conversion():
    spec = get_spec(FAST)
    in_dim, n_out = dims(spec)
    B = budget_for(spec)
    cfg = fast_cfg()
    steps, n, raw = budget_steps(cfg, in_dim, n_out, B, MAX_STEPS)
    assert n == n_params_for(cfg, in_dim, n_out)
    assert raw == int(B // (cfg["batch_size"] * n)) and steps == min(raw, MAX_STEPS)
    # bigger batch -> proportionally fewer steps; wider -> fewer steps
    s2, _, _ = budget_steps(dict(cfg, batch_size=512), in_dim, n_out, B, MAX_STEPS)
    assert s2 == int(B // (512 * n))
    s3, _, _ = budget_steps(dict(cfg, width=128), in_dim, n_out, B, MAX_STEPS)
    assert s3 < steps
    # cap
    s4, _, raw4 = budget_steps(dict(cfg, width=4, batch_size=4), in_dim, n_out, B, MAX_STEPS)
    assert raw4 > MAX_STEPS and s4 == MAX_STEPS
    # default skeleton gets REF_STEPS steps by construction
    from rsi.tasks.muar.tasks import REF_STEPS
    assert budget_steps(schema.DEFAULT_CONFIG, in_dim, n_out, B, MAX_STEPS)[0] == REF_STEPS


def test_budget_violation_is_illegal():
    r = evaluate_config(FAST, fast_cfg(width=1024, depth=6, batch_size=512), [0])
    assert r["status"] == "audit_fail" and r["value"] is None
    assert any(f.startswith("budget:") for f in r["audit"]["flags"])
    assert r["diag"]["steps"] < MIN_STEPS or r["diag"]["params"] > MAX_PARAMS
    # steps reported to the improver match the conversion
    ok = evaluate_config(FAST, fast_cfg(), [0])
    spec = get_spec(FAST)
    assert ok["diag"]["steps"] == budget_steps(fast_cfg(), *dims(spec), budget_for(spec), MAX_STEPS)[0]


# ---------------------------------------------------------------------------- illegal configs
@pytest.mark.parametrize("mutate,flag", [
    (lambda c: c.update(foo=1), "schema:unknown_key"),
    (lambda c: c.pop("lr"), "schema:missing_key"),
    (lambda c: c.update(lr=10.0), "schema:out_of_range"),
    (lambda c: c.update(lr=float("nan")), "schema:bad_type"),
    (lambda c: c.update(width=True), "schema:bad_type"),
    (lambda c: c.update(width=64.5), "schema:bad_type"),
    (lambda c: c.update(depth=0), "schema:out_of_range"),
    (lambda c: c.update(optimizer="lion"), "schema:bad_enum"),
    (lambda c: c.update(norm=None), "schema:bad_enum"),
    (lambda c: c.update(lr="0.01"), "schema:bad_type"),
])
def test_illegal_configs_rejected(mutate, flag):
    cfg = fast_cfg()
    mutate(cfg)
    r = evaluate_config(FAST, cfg, [0, 1])
    assert r["status"] == "audit_fail" and r["value"] is None
    assert r["audit"] == {"level": "L0", "flags": [flag]}


def test_illegal_artifacts(tmp_path):
    r = evaluate_artifact(FAST, str(tmp_path), [0])
    assert r["status"] == "audit_fail" and r["audit"]["flags"] == ["schema:missing_file"]
    (tmp_path / "config.json").write_text("{not json")
    assert evaluate_artifact(FAST, str(tmp_path), [0])["audit"]["flags"] == ["schema:bad_json"]
    (tmp_path / "config.json").write_text(json.dumps([1, 2]))
    assert evaluate_artifact(FAST, str(tmp_path), [0])["audit"]["flags"] == ["schema:not_object"]
    (tmp_path / "config.json").write_text(json.dumps(fast_cfg()) + " " * 5000)
    assert evaluate_artifact(FAST, str(tmp_path), [0])["audit"]["flags"] == ["schema:too_large"]
    (tmp_path / "config.json").write_text(json.dumps(fast_cfg()))
    ok = evaluate_artifact(FAST, str(tmp_path), [0])
    assert ok["status"] == "ok" and len(ok["artifact_hash"]) == 12


def test_divergence_is_nan_status():
    r = evaluate_config("muar/train-teacher-0", fast_cfg(optimizer="sgd", lr=3.0, grad_clip=0.0, init_scale=5.0,
                                                         norm="none", activation="relu", depth=4), [0])
    assert r["status"] in ("nan", "ok")
    if r["status"] == "nan":
        assert r["value"] is None and "nan" in r["audit"]["flags"]


def test_eval_result_fields_match_schema():
    from rsi.events import EVENT_FIELDS

    r = evaluate_config(FAST, fast_cfg(), [0])
    for f in EVENT_FIELDS["eval_result"]:
        if f != "artifact_hash":
            assert f in r
    assert r["metric"] == "val_loss" and r["status"] in {"ok", "timeout", "crash", "nan", "audit_fail"}


# ---------------------------------------------------------------------------- sealed leakage
def test_no_sealed_leakage_in_visible_feedback(monkeypatch, tmp_path):
    t = get_task(FAST)
    dev = evaluate_config(FAST, fast_cfg(), [0, 1])
    sealed = evaluate_config(FAST, fast_cfg(), [0, 1, 2], split="sealed")
    fb = t.visible_feedback(dev)
    assert f"{dev['value']:.6f}" in fb and "steps=" in fb and "train_val_gap=" in fb
    assert f"{sealed['value']:.6f}" not in fb
    assert t.visible_feedback(sealed) == "sealed result withheld"
    # stray keys injected into a result never reach the improver
    poisoned = dict(dev, sealed_value=123.456789, s_ref=0.987654,
                    diag=dict(dev["diag"], sealed=[7.777777], s_ref=0.555555, defects=["lr_high"]))
    fb2 = t.visible_feedback(poisoned)
    for needle in ("123.456789", "0.987654", "7.777777", "0.555555", "lr_high", "sealed"):
        assert needle not in fb2


def test_task_card_has_no_sealed_info():
    from rsi.tasks.muar.tasks import seal_salt

    salt, _ = seal_salt()
    for spec in SPECS:
        card = task_card(spec)
        _, defects = start_config_and_defects(spec)
        assert salt not in card
        all_defects = {d for g in DEFECTS.values() for d in g}
        assert not any(d in card for d in all_defects)
        assert f"data_seed" not in card and str(train_seed(spec, "sealed", 0)) not in card
        assert "s_ref" not in card and spec.name in card


def test_salt_from_env_changes_only_sealed(monkeypatch):
    spec = get_spec(FAST)
    a_dev, a_seal = train_seed(spec, "dev", 0), train_seed(spec, "sealed", 0)
    monkeypatch.setenv("RSI_MUAR_SEAL_SALT", "another-secret")
    assert train_seed(spec, "dev", 0) == a_dev and train_seed(spec, "sealed", 0) != a_seal


# ---------------------------------------------------------------------------- registry and splits
def test_splits_and_registry(tmp_path, monkeypatch):
    assert len(SPLITS["train"]) == 8 and len(SPLITS["heldout_id"]) == 4 and len(SPLITS["heldout_ood"]) == 4
    types = {sp: {get_spec(n).ttype for n in names} for sp, names in SPLITS.items()}
    assert types["heldout_id"] <= types["train"]
    assert not (types["heldout_ood"] & types["train"])
    assert len(types["train"] | types["heldout_ood"]) == 8
    starts = [schema.canonical_json(start_config(s.name)) for s in SPECS]
    assert len(set(starts)) == len(starts)
    for s in SPECS:      # held-out ID defects differ from the train tasks of the same type
        if s.split == "heldout_id":
            mine = set(start_config_and_defects(s)[1])
            for o in SPECS:
                if o.split == "train" and o.ttype == s.ttype:
                    assert mine != set(start_config_and_defects(o)[1])
    monkeypatch.setenv("RSI_MUAR_START_DIR", str(tmp_path))
    t = get_task("muar/id-cls2d-0")
    assert isinstance(t, MuarTask) and t.name == "muar/id-cls2d-0" and t.editable_files == ("config.json",)
    d = t.start_artifact()
    cfg = json.loads((d / "config.json").read_text())
    assert schema.validate_config(cfg) == start_config("muar/id-cls2d-0")
    with pytest.raises(KeyError):
        get_task("muar/nope")


# ---------------------------------------------------------------------------- runner and baselines
def test_local_eval_client_and_pool_determinism(tmp_path):
    from rsi.tasks.muar.runner import LocalEvalClient

    (tmp_path / "config.json").write_text(json.dumps(fast_cfg()))
    serial = evaluate_artifact(FAST, str(tmp_path), [0, 1])
    with LocalEvalClient(workers=2, emit_events=False) as ev:
        j1 = ev.submit("r", FAST, str(tmp_path), seeds=[0, 1], trial=3)
        assert ev.submit("r", FAST, str(tmp_path), seeds=[0, 1], trial=3) == j1
        res = ev.result(j1)
        assert ev.lanes()[0]["lane"] == 0
    assert res["value"] == serial["value"] and res["diag"]["per_seed"] == serial["diag"]["per_seed"]


def test_baseline_runs_deterministic():
    from rsi.tasks.muar.runner import baseline_run

    a = baseline_run(FAST, "local", 0, 2, seeds=[0])
    b = baseline_run(FAST, "local", 0, 2, seeds=[0])
    assert a["values"] == b["values"] and a["configs"] == b["configs"] and len(a["values"]) == 2
    r = baseline_run(FAST, "random", 1, 2, seeds=[0])
    assert len(r["values"]) == 2 and r["s0"] == a["s0"]


def test_sref_fold(tmp_path):
    from rsi.tasks.muar.runner import fold_sref, load_sref, save_sref

    p = str(tmp_path / "sref.json")
    save_sref(p, {FAST: {"s_ref": 0.5, "s0": 1.0}})
    fold_sref(p, FAST, 0.6, "arm A1")
    assert load_sref(p)[FAST]["s_ref"] == 0.5
    fold_sref(p, FAST, 0.4, "arm A1")
    assert load_sref(p)[FAST]["s_ref"] == 0.4
    save_sref(p, {FAST: {"s_ref": 0.45, "s0": 1.0}})   # a worse recomputation never raises s_ref
    assert load_sref(p)[FAST]["s_ref"] == 0.4


def test_perturb_and_sample_stay_legal():
    import numpy as np

    rng = np.random.default_rng(0)
    c = start_config("muar/train-parity-0")
    for _ in range(200):
        schema.validate_config(schema.sample_config(rng))
        c = schema.perturb_config(c, rng)
