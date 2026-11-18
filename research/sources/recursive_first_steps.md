# Recursive — First Steps Toward Automated AI Research — 引证台账

## 基本信息

- **URL 已读**：
  - A = https://recursive.com/updates/first-steps-toward-automated-ai-research （博客，约 17.8k chars，未截断）
  - B = https://github.com/recursive-org/first-steps-toward-automated-ai-research （repo 主 README）
  - C = …/tree/main/nanochat_autoresearch （README）
  - D = …/tree/main/nanoGPT_speedrun （README）
  - E = …/tree/main/SOL-ExecBench （README）
  - 博客内 repo 链接包在 web.archive.org 里，subagent 直接抓了 GitHub 原链。未发现技术报告 / 论文 / 附录。
- **标题**："First Steps Toward Automated AI Research"；副标题 "Early results from Recursive's automated AI research system on model training and GPU kernel benchmarks"
- **机构**：Recursive（"© 2026 Recursive Superintelligence, Inc."）；**无个人作者**
- **日期（三处冲突）**：页面 metadata `published: "Nov 6, 2026, 4:25 PM UTC"`；标题下显示 "Oct 18, 2026"；页面 Related 列表中本文显示 "Jun 11, 2026"。页内多处 web.archive.org 快照为 2026 年 7–8 月（如 20260815204034），推测内容至迟 2026 年 8 月中已存在（推断，非原文）。
- **License / 发布**：repo Apache-2.0。已发布：NanoChat 训练脚本、`lib.py` harness、逐 seed BPB 与训练轨迹；两份 speedrun 解与日志/统计/verifier；235 个 SOL kernel 中的 10 个。**系统本身未发布**；底座模型、系统架构、迭代次数、算力/成本均未披露。

## 数字台账

### A. 博客正文（A）

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| RF-01 | 3 | 基准数量 | "Across three benchmarks, the system achieves state-of-the-art results" | A | 自报 |
| RF-02 | 5 minutes / single GPU | NanoChat 预算 | "within a fixed five-minute budget on a single GPU" | A | 自报 |
| RF-03 | "several dozens" / "hundreds" | autoresearch@home 人数 / agent 数（非精确） | "several dozens of humans and hundreds of their agents" | A | 自报 |
| RF-04 | 10 | baseline 评估 seed 数 | "evaluating it on 10 random seeds" | A | 自报 |
| RF-05 | 0.9372 | 原 SOTA BPB（去除 minor reward hacks 后社区最佳的 10-seed 均值） | "its mean performance is 0.9372 BPB" | A | 自报 |
| RF-06 | 0.9109 | Recursive NanoChat BPB | "Our system found a solution that reached 0.9109 BPB" | A | 自报 |
| RF-07 | 0.0263 | BPB 改进 | "a 0.0263 BPB improvement" | A | 自报（推算核对：0.9372−0.9109=0.0263） |
| RF-08 | 1.3x | 达到同等 loss 所需训练时间缩短 | "roughly 1.3x less training time than the best autoresearch@home solution" | A | 自报 |
| RF-09 | 1.059 → 0.9344 | weak start（vanilla Transformer + AdamW）BPB，B200 | "improved the model from 1.059 BPB to 0.9344 BPB" | A | 自报 |
| RF-10 | 1–2 billion / ~50M | hash table 稀疏参数 / 模型规模 | "hash tables can add 1-2 billion sparse parameters to a roughly 50M parameter model" | A | 自报 |
| RF-11 | 3.28 | speedrun 目标 val loss | "Training time required to reach a 3.28 validation loss" | A | 自报 |
| RF-12 | 8-GPU (HGX H100) | speedrun 硬件 | "using a single HGX H100 8-GPU node" | A | 自报 |
| RF-13 | 83 | 人类破纪录贡献数 | "83 human record-setting contributions to the leaderboard so far" | A | 自报 |
| RF-14 | "hundreds" | 提交的 PR 数 | "and hundreds of proposed PRs" | A | 自报 |
| RF-15 | ~45 min → 79.7 s | 2024 年中以来 speedrun 历史 | "pushed from roughly 45 minutes down to 79.7 seconds" | A | 自报 |
| RF-16 | over two years | 社区优化时长 | "optimizing solutions for it for over two years" | A | 自报 |
| RF-17 | 79.7 → 77.5 s | Recursive speedrun 结果 | "reduced training time from 79.7 seconds to 77.5 seconds" | A | 自报 |
| RF-18 | 2.2 s | speedrun 改进 | "2.2s faster training" | A（表） | 自报 |
| RF-19 | p < 0.01 | 显著性要求 | "mean validation loss ≤ 3.28 at p < 0.01" | A | 自报 |
| RF-20 | ~15 min / entry #5 | speedrun weak start | "Starting from an earlier roughly 15-minute solution" / "(entry #5)" | A | 自报 |
| RF-21 | ~185 s / ~180 s（May 2025） | weak start 结果 / 当时人类水平 | "reached approximately 185 seconds in a few days" | A | 自报 |
| RF-22 | "a few days" | weak start 运行时长（全文唯一的时长信息） | 同上 | A | 自报 |
| RF-23 | 235 | SOL-ExecBench kernel 任务数 | "The benchmark contains 235 kernel-writing tasks derived from real workloads." | A | 自报（benchmark 属性，源自 NVIDIA） |
| RF-24 | 0.5 / 1.0 | SOL 锚点（PyTorch baseline / 解析最优） | "0.5 corresponds to the benchmark's optimized PyTorch baseline" | A | 自报 |
| RF-25 | 0.699 | 原 SOL leaderboard 最佳 | "from the previous leaderboard best of 0.699" | A | 自报 |
| RF-26 | 0.754 | Recursive mean SOL score | "achieved a mean NVIDIA SOL-ExecBench score of 0.754" | A | 自报 |
| RF-27 | 18% | 到 1.0 的差距缩小比例 | "an 18% reduction in the gap to the hardware limit" | A | 自报（推算核对：(0.754−0.699)/(1−0.699)≈18.3%） |

### B. 配套 repo（B–E）

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| RF-28 | 77.3 s | repo README 中的最佳 speedrun 时间（≠ 博客 77.5） | "reaches the ≤ 3.28 FineWeb val-loss target in 77.3 s on 8× H100" | B | 自报 |
| RF-29 | 10 of 235 | 公开的 kernel 数 | "10 of the 235 GPU kernel implementations" | B, E | 自报 |
| RF-30 | 1.0587 / 0.9344 / 0.9109 | val BPB：vanilla / optimized_from_vanilla / optimized_from_karpathy | 表 "Val BPB" 列 | C | 自报 |
| RF-31 | 6 × 10 | 结果 CSV：解数 × seed 数 | "aggregated from all 6 solutions × 10 seeds" | C | 自报 |
| RF-32 | 3.2789（n=13；min 3.2766，max 3.2819） | from_best 平均 val loss | "3.2789 (13 runs, min 3.2766 / max 3.2819)" | D | 自报 |
| RF-33 | 0.0078 | from_best p 值 | "p(mean < 3.28) = 0.0078" | D | 自报 |
| RF-34 | 77.34 s ±0.20 | from_best 平均训练时间 | "Mean training time 77.34 s (8×H100)" | D | 自报 |
| RF-35 | 1398 | from_best 步数 | "Steps 1398" | D | 自报 |
| RF-36 | −2.815 / ±0.0014 | from_best t 统计量 / val loss std | verifier 输出行 | D | 自报 |
| RF-37 | 80.61 s / 3.2785 / p 0.0047 / n=10 | #83 在同硬件（Modal）复跑 | "3.3 s faster on identical hardware (80.61 s → 77.34 s)" | D | 自报 |
| RF-38 | 3.3 s | 同硬件对比加速 | 同上 | D | 自报（推算核对：80.61−77.34=3.27） |
| RF-39 | 3.2778（n=10；min 3.2757，max 3.2792） | from_unoptimized val loss | "3.2778 (10 runs, min 3.2757 / max 3.2792)" | D | 自报 |
| RF-40 | 3.6e-05（3.612e-05），t=−6.880 | from_unoptimized p 值 / t | "p(mean < 3.28) = 3.6e-05" | D | 自报 |
| RF-41 | 185.4 s（185.42 ±0.27） | from_unoptimized 时间 | "Mean training time 185.4 s (8×H100)" | D | 自报 |
| RF-42 | 682 | from_unoptimized 步数 | "Steps 682" | D | 自报 |
| RF-43 | 4 heads，head_dim 128，attn_dim 512 < 768 | from_unoptimized attention 形状 | "Narrow 4-head attention (head_dim 128, attn_dim 512 < 768)" | D | 自报 |
| RF-44 | 80GB H100；PyTorch 2.10.0+cu128；Triton 3.6.0 | 运行环境 | "8× NVIDIA H100 80GB (PyTorch 2.10.0+cu128, Triton 3.6.0)" | D | 自报 |
| RF-45 | 268 / 23 / 4 / 1 / 1 | repo stars / forks / watchers / open issue / commit | — | B | 平台计数（2026-11-17） |

### C. 主表（博客 A）

| 基准 | 指标 | Previous SOTA | Recursive | Improvement（原文） |
|---|---|---|---|---|
| NanoChat Autoresearch | Val BPB | 0.9372 | 0.9109 | "0.0263 lower Validation BPB, or a 1.3x speedup to reach the same loss" |
| NanoGPT Speedrun | time to 3.28 val loss | 79.7 s | 77.5 s | "2.2s faster training" |
| SOL-ExecBench | mean SOL over 235 kernels | 0.699 | 0.754 | "18% reduction in gap to the optimal performance estimate of 1.0" |

### D. 公开的 10 个 SOL kernel 分数（E）

| id | kernel | 类别 | SOL score |
|---|---|---|---|
| RF-K01 | 011_fp8_moe_gate_routing | Quant | 0.866 |
| RF-K02 | 012_gqa_paged_decode_h32_kv4_d128_ps1 | FlashInfer-Bench | 0.934 |
| RF-K03 | 025_moe_expert_parallel_execution_backward | L2 | 0.533 |
| RF-K04 | 032_nvfp4_moe_expert_linear_with_gating | Quant | 0.906 |
| RF-K05 | 045_fused_linear_gelu_grn_linear (.cu) | L1 | 0.733 |
| RF-K06 | 049_group_limited_topk_routing (.cu) | L2 | 0.884 |
| RF-K07 | 057_residual_coupling_flow_block | L2 | 0.721 |
| RF-K08 | 069_joint_transformer_block_residual_path | L2 | 0.831 |
| RF-K09 | 073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection | L1 | 0.846 |
| RF-K10 | 077_whisper_decoder_output_projection | L1 | 0.720 |

类别：L1 = single-op；L2 = multi-op fused；Quant = FP8/NVFP4；FlashInfer-Bench = paged attention。

## 原文内部不一致 / 待核

1. **Speedrun 时间**：博客 77.5 s；repo README 77.3 s；repo D 统计 77.34 s。
2. **Vanilla 起点 BPB**：博客 1.059；repo 1.0587（四舍五入差异）。
3. **发布日期**：Nov 6 / Oct 18 / Jun 11 三处冲突（见上）。
4. **链接错误**：博客开头第一个 "open-sourcing artifacts" 链接指向 karpathy/autoresearch，而非 Recursive repo；文末第二个链接才正确。README 中博客路径为 /articles/，与 /updates/ 不同。
5. **SOL 分数被他方引用时不一致**：Naive 页面引用 Recursive SOL 为 **67.56**（称其为更新后 leaderboard 分），而本文写 **0.754**。需回源 NVIDIA leaderboard。
6. **Speedrun 非官方**：结果在 Modal HGX H100 上取得、在 Andromeda 上 "within noise" 复核；尚待官方硬件 PrimeIntellect，**未提交 leaderboard**（footnote 1）。
7. **Figures 1–6 未读**：只拿到 "FIGURE 1–6" 标签，轴、数据、折叠技术细节框内容未知。
8. **不可解读为独立发现**：两个 weak start 实验作者自己说底层模型可能已见过公开技术。
9. **未披露**：使用的 LLM、agent 角色、是否改写自身 scaffold、迭代次数、算力/GPU-hours/成本。

## 外链

**博客（A）**（多数包在 web.archive.org 中）
- https://github.com/karpathy/autoresearch
- https://www.ensue-network.ai/lab/autoresearch （autoresearch@home）
- https://arxiv.org/abs/2601.07372 （DeepSeek Engram）
- https://github.com/KellerJordan/modded-nanogpt
- https://research.nvidia.com/benchmarks/sol-execbench
- https://www.recursive.com/articles/first-steps-toward-automated-ai-research#footnote-1
- https://github.com/recursive-org/first-steps-toward-automated-ai-research
- https://www.nytimes.com/2026/05/13/technology/recursive-superintelligence-funding-ai.html （Related，2026-05-13）
- https://www.youtube.com/watch?v=eqYqUAdjsL0 （Richard Socher，"The Eureka Machine"，2026-09-09）
- https://x.com/Recursive_SI ；https://www.linkedin.com/company/recursive-si/ ；mailto:talent@recursive.com

**Repo（B–E）**
- https://github.com/NVIDIA/SOL-ExecBench
- https://github.com/karpathy/nanochat
- https://github.com/karpathy/autoresearch/blob/master/train.py （Karpathy seed）
- https://github.com/karpathy/autoresearch/blob/438a26e/train.py （Karpathy optimized）
- https://www.ensue-network.ai/lab/autoresearch?run=results%2Fforge--final-lr-frac-0-05-0-02-with-wsd-sqrt--3f0b52 （community best）

**未抓取**：上述外链本身；repo 的 statistics.md、日志、val_bpb.csv、LICENSE/NOTICE。
