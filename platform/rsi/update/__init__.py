"""Update operators U (INTERFACES.md, worker D).

    build_sft_data(run_dirs, *, accepted_only=True, placebo=False, out) -> dict      (U_w data; rsi.update.sft)
    write_lessons(run_dirs, *, model, gateway_url, out_dir, generic_placebo=False) -> str   (U_s; rsi.update.lessons)
    rsi.update.analysis   nAUC, Δ₂, bootstrap CI, sign-flip test, A/A floor, MDE, decision rule
    rsi.update.lora_train PEFT LoRA trainer behind bundle/lora_train.sh
    rsi.update.selfevolve failure classification + verified repairs (Apex self-evolving data)
"""

from rsi.update.lessons import write_lessons
from rsi.update.sft import build_sft_data

__all__ = ["build_sft_data", "write_lessons"]
