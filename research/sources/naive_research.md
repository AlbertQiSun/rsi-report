# Naive-N0.5-Flash / NaiveRT / AutoWM — 引证台账

## 基本信息

- **URL 已读**：
  - https://naive.ai/en/research/ （HTTP 200，全文 32,226 chars；三项内容同属这一篇长文，无单独详情页 / PDF / arXiv）
  - https://naive.ai/en/ （首页，仅重复 teaser）
  - 图片：`/assets/research/naive-n05-flash-coding.webp`、`/assets/research/naive-n05-flash-ai-rd.webp`（已存 `img/`，数值人工读图）
- **标题**："Naive-N0.5-Flash: Building Frontier AI with AI"
- **作者 / 机构**："NaiveAI Team"
- **日期**：September 27, 2026（页面仅此一个日期）
- **Citation**：`@misc{naiveai2026naiven05flash, ...}`，无 arXiv ID / DOI
- **License / 发布**：weights 与 inference code 为 MIT；HF 上有 W8A8 checkpoint 与 DFlash draft model；NaiveRT GitHub 源码页面写 "will be available by Oct, 12th"（发布时尚未可用）；API 承诺提供。未发布：训练代码、训练数据、in-house AutoResearch harness、sandbox 平台、AutoWM 代码/权重。
- **基座**：Xiaomi MiMo-V2.5（开放权重，第三方；页面未说明其 license）
- **统一 URL**：下表 URL 列简写 `R` = https://naive.ai/en/research/

## 数字台账

### A. 模型概览与评测设置

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| NV-01 | September 27, 2026 | 发布日期 | "September 27, 2026" | R | 自报 |
| NV-02 | 309B | 总参数 | "Naive-N0.5-Flash is a 309B MoE model with 15.5B active parameters" | R | 自报 |
| NV-03 | 15.5B | 激活参数 | 同上 | R | 自报 |
| NV-04 | 1M | 原生上下文 | "Native 1M context, without full attention." | R | 自报 |
| NV-05 | ~10 million / 100,000 | sandbox 每周服务量 / 峰值并发 | "close to ten million sandboxes each week, with 100,000 active concurrently at peak" | R | 自报 |
| NV-06 | 5:1 | SWA:DSA 层比例 | "at a predominantly 5:1 SWA–DSA layout" | R | 自报 |
| NV-07 | 4 | GQA KV groups | "grouped-query attention (GQA) using four KV groups" | R | 自报 |
| NV-08 | 50 tokens/s | Standard 模式每用户 decode 速度 | "50 tokens/s per user in Standard mode" | R | 自报 |
| NV-09 | 2,000 tokens/s | Ultrafast 模式 decode 速度上限 | "up to 2,000 tokens/s in Ultrafast mode" | R | 自报 |
| NV-10 | $0.10 / $0.40 / $0.01 | API 每百万 token 价格（input / output / cache read） | "$0.10 / $0.40 / $0.01 per million tokens for input, output, and cache reads" | R | 自报 |
| NV-11 | 2.1.207 | 评测 harness：Claude Code 版本 | "use Claude Code 2.1.207 with a 1M-token context window" | R | 自报 |
| NV-12 | 1.0 / 0.95 | 评测 temperature / top-p | "temperature 1.0, and top-p 0.95" | R | 自报 |

### B. Naive 自身基准分数（Coding）

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| NV-13 | 67.8 | DeepSWE v1.1 | "scores 67.8 on DeepSWE v1.1" | R（图 alt text） | 自报 |
| NV-14 | 32.4 | Agents' Last Exam (ALE-CLI) | "32.4 on Agents' Last Exam (ALE-CLI)" | R（图 alt text） | 自报 |
| NV-15 | 86.7 | Terminal-Bench 2.1 | "86.7 on Terminal-Bench 2.1" | R（图 alt text） | 自报 |
| NV-16 | 73.6 | SWE-bench Pro | "73.6 on SWE-bench Pro" | R（图 alt text） | 自报 |
| NV-17 | 17.5 | ProgramBench（Almost@1） | "17.5 on ProgramBench"; "We report the Almost@1 score." | R | 自报 |
| NV-18 | 71.9 | NL2Repo-Bench | "71.9 on NL2Repo-Bench" | R（图 alt text） | 自报 |
| NV-19 | 78.2 | FrontierSWE v1（Dominance score，竞品结果截至 2026-08-23） | "78.2 on FrontierSWE v1"; "competing systems' results as of August 23, 2026" | R | 自报 |

### C. Naive 自身基准分数（AI R&D）

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| NV-20 | 37.5 | PostTrainBench | "scores 37.5 on PostTrainBench" | R（图 alt text） | 自报 |
| NV-21 | 73.7% | MLE-bench-30（average position score，沿用 Gemini 3.6 Flash card 口径） | "73.7% on MLE-bench-30" | R | 自报 |
| NV-22 | 63.2 | PaperBench | "63.2 on PaperBench" | R（图 alt text） | 自报 |
| NV-23 | 72.81 vs 67.56 | SOL-ExecBench mean SOL score：Naive vs Recursive | "mean SOL score of 72.81 on SOL-ExecBench" … "versus 67.56" | R | 自报（Recursive 值为 Naive 引用其更新后 leaderboard） |
| NV-24 | 0.9051 vs 0.9109 | NanoChat AutoResearch val BPB（越低越好）：Naive vs Recursive | "0.9051 validation BPB on NanoChat AutoResearch" | R | 自报 |
| NV-25 | 73.8 s vs 77.5 s | NanoGPT SpeedRun 训练时间：Naive vs Recursive | "73.8 s training time on NanoGPT SpeedRun" | R | 自报 |

注：NV-23~25 用 Naive "inhouse AutoResearch harness" 评测；Recursive 一侧在图中标为 "(undisclosed third-party model)"。

### D. 竞品分数（naive.ai 图表（图片读数），状态均为 自报（Naive 引用他方））

图例：金色 = Naive-N0.5-Flash；深灰 = <600B；浅灰 = ≥600B；斜纹 = closed / undisclosed。**各面板坐标轴截断**（见"不一致/待核"），条长不与数值成比例。

**Coding 图**（`img/naive_coding_chart.webp`）

| id | 基准 | 模型 → 分数（图中顺序） |
|---|---|---|
| NV-C01 | DeepSWE v1.1（轴 55–80） | Muse-Spark-1.3 75.4；DeepSeek-V4.1-Flash 74.2；Opus-5.5 74.2；Opus-5 73.6；GPT-5.6-Sol 72.7；Fable-5 (w/ fallback) 69.7；**Naive-N0.5-Flash 67.8**；Step-5-Preview 67.7；Kimi-K3 67.5；GLM-5.3 66.9；Hy4-preview 64.3；GLM-5.3-Flash 63.4；Qwen-3.8-Max 56.6 |
| NV-C02 | Agents' Last Exam (ALE-CLI)（轴 20–36） | Opus-5.5 34.3；GPT-6-Astra 33.3；Muse-Spark-1.3 33.3；**Naive-N0.5-Flash 32.4**；Opus-5 29.5；Step-5-Preview 29.5；GPT-5.6-Sol 28.6；GLM-5.3 28.5；Kimi-K3 28.3；GLM-5.3-Flash 26.3；Fable-5 (w/ fallback) 23.8；Hy4-preview 22.8 |
| NV-C03 | Terminal-Bench 2.1（轴 84–91） | DeepSeek-V4.1-Flash 90.6；GPT-5.6-Sol 88.8；Muse-Spark-1.3 88.8；Kimi-K3 88.3；GLM-5.3 88.2；Fable-5 (w/ fallback) 88.0；**Naive-N0.5-Flash 86.7**；Hy4-preview 85.4；GLM-5.3-Flash 84.3 |
| NV-C04 | SWE-bench Pro（轴 60–95） | Opus-5.5 89.9；Opus-5 79.2；**Naive-N0.5-Flash 73.6**；Qwen-3.8-Max 67.7；Hy4-preview 65.7；GPT-5.6-Sol 64.6 |
| NV-C05 | ProgramBench（轴 15–40） | Opus-5 37.0；Fable-5 (w/ fallback) 33.0；GPT-5.6-Sol 23.0；DeepSeek-V4.1-Flash 20.3；GLM-5.3 19.0；**Naive-N0.5-Flash 17.5**；Hy4-preview 17.5 |
| NV-C06 | NL2Repo-Bench（轴 50–75） | **Naive-N0.5-Flash 71.9**；DeepSeek-V4.1-Flash 64.0；Hy4-preview 58.9；GLM-5.3 58.0；Qwen-3.8-Max 55.9 |
| NV-C07 | FrontierSWE v1（轴 70–90） | Fable-5 (w/ fallback) 88.2；Kimi-K3 81.2；**Naive-N0.5-Flash 78.2**；GLM-5.3 78.1；Qwen-3.8-Max 73.5 |

**AI R&D 图**（`img/naive_ai_rd_chart.webp`）

| id | 基准 | 模型 → 分数（图中顺序） |
|---|---|---|
| NV-C08 | PostTrainBench（轴 35–42） | GPT-5.6-Sol 41.8；GLM-5.3 39.8；**Naive-N0.5-Flash 37.5**；Kimi-K3 36.6；Fable-5 (w/ fallback) 36.2；Hy4-preview 35.6 |
| NV-C09 | MLE-bench-30（轴 35%–75%） | **Naive-N0.5-Flash 73.7%**；Sonnet-5 66.9%；Gemini-3.6-Flash 63.9%；Gemini-3.5-Flash 49.7%；GPT-5.6-Luna 47.6%；Grok-4.5 43.2% |
| NV-C10 | PaperBench（轴 45–65） | **Naive-N0.5-Flash 63.2**；Opus-4.7 58.5；GPT-5.5 57.5；MiniMax-M3 52.6；Gemini-3.1-Pro 46.7 |
| NV-C11 | SOL-ExecBench（Mean SOL Score ↑） | **Naive-N0.5-Flash 72.81**；Recursive Superintelligence, Inc. (undisclosed third-party model) 67.56 |
| NV-C12 | NanoChat AutoResearch（Validation BPB ↓） | **Naive-N0.5-Flash 0.9051**；Recursive Superintelligence, Inc. (undisclosed third-party model) 0.9109 |
| NV-C13 | NanoGPT SpeedRun（Training Time (s) ↓） | **Naive-N0.5-Flash 73.8**；Recursive Superintelligence, Inc. (undisclosed third-party model) 77.5 |

### E. 架构与训练

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| NV-26 | 8 × 6 | 模块结构：8 个六层模块 | "eight six-layer modules" | R | 自报 |
| NV-27 | 48 层 / 9 个 DSA 层 | 总层数推算（页面未给总层数） | — | R | 推算：8×6=48；每模块 1 DSA + 首模块首层也是 DSA → 8+1=9 |
| NV-28 | 128 tokens | SWA window | "SWA uses a 128-token window" | R | 自报 |
| NV-29 | 2,048 | DSA top-k tokens | "DSA selects the top 2,048 tokens" | R | 自报 |
| NV-30 | 16 | indexer query heads | "lightweight indexer with 16 query heads" | R | 自报 |
| NV-31 | 44% | index-selection 耗时相对原版 DSA 的降低 | "reducing index-selection wall time by 44% relative to the original DSA" | R | 自报 |
| NV-32 | ~341 μs（233 + 108） | 16 heads 估计全路径耗时（图 alt text） | "With 16 Q heads the estimated full path takes about 341 μs" | R | 自报 |
| NV-33 | ~605 μs（497 + 108） | 64 heads 同口径 | "versus about 605 μs with 64 heads (497 plus 108)" | R | 自报 |
| NV-34 | 1.77× | 16 vs 64 heads 加速 | "1.77× faster, 44% less time" | R | 自报 |
| NV-35 | 3.25T | 总训练 tokens | "3.25T tokens of multi-stage training" | R | 自报 |
| NV-36 | 50B | Indexer Warmup tokens | "50B tokens of Indexer Warmup" | R | 自报 |
| NV-37 | 3T | Sparse Attention Training tokens | "3T tokens of Sparse Attention Training" | R | 自报 |
| NV-38 | 200B | Learning Rate Decay（SFT）tokens | "200B tokens of Learning Rate Decay" | R | 自报 |
| NV-39 | O(L) → O(w) | SWA 通信复杂度 | "reducing SWA communication complexity from O(L) to O(w)" | R | 自报 |
| NV-40 | 1T tokens / ~4 天 / 512 GPUs | 1M 上下文下训练吞吐 | "process 1T training tokens in approximately four days on 512 GPUs" | R | 自报 |

### F. NaiveRT（Case 1）

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| NV-41 | 2,122 tok/s | 8 GPU 峰值单流 decode | "peak single-stream decoding rate of 2,122 tokens/s on 8 GPUs" | R | 自报 |
| NV-42 | 8 | GPU 数 | 同上 | R | 自报 |
| NV-43 | 3.4 ms vs 12.3 ms | 完整 speculative round：NaiveRT vs SGLang | "from 12.3 ms under SGLang to 3.4 ms, a 72.4% reduction" | R | 自报 |
| NV-44 | 72.4% | 延迟降低 | 同上 | R | 自报（推算核对：1−3.4/12.3≈72.4%） |
| NV-45 | six days | 构建周期 | "built in six days by human researchers working with AI models" | R | 自报 |
| NV-46 | 151 | 有记录的优化 trial 数 | "across 151 documented optimization trials" | R | 自报 |
| NV-47 | 5,000–10,000 tok/s | ingest token 到达速率 | "arrive at 5,000–10,000 tokens/s" | R | 自报 |
| NV-48 | 50–100 tok/s | 常规 decode 生成速率 | "arrive at 50–100 tokens/s with conventional decoding" | R | 自报 |
| NV-49 | 20–40% | 生成 token 占上下文比例 | "roughly 20–40% at today's ratios" | R | 自报 |
| NV-50 | 29（25 + 4） | SGLang 下每 DSA 层每步 kernel 执行数 | "a single DSA layer required 29 kernel executions per GPU per decoding step" | R | 自报 |
| NV-51 | 1 kernel / 148 CTAs | NaiveRT DSA mega-kernel | "a single cooperative mega-kernel running across 148 CTAs in one grid" | R | 自报 |
| NV-52 | TP8 | 被吸收进 kernel 的 tensor-parallel 通信 | "the trailing TP8 communication" | R | 自报 |
| NV-53 | 124 CTAs / 128 KiB | QKV-projection CTA 数 / 每个 TMA 传输量 | "the 124 QKV-projection CTAs each start a 128 KiB TMA transfer" | R | 自报 |
| NV-54 | 21–23 μs | TMA prefetch 每步节省（全模型） | "reduced full-model latency by 21–23 μs per step" | R | 自报 |
| NV-55 | 57.0 → 49.5 μs | K/V staging 后单层延迟 | "reducing the layer latency from 57.0 μs to 49.5 μs" | R | 自报 |
| NV-56 | 8 heads；m16n8k16；16 rows；n8 | Tensor Core 重映射细节 | "With eight query heads per GPU, a direct m16n8k16 mapping would pad" | R | 自报 |
| NV-57 | 43 / 28 | Stage 1 trials / adopted | "Whole-network engineering (43 trials, 28 adopted)" | R | 自报 |
| NV-58 | 45 / 15 | Stage 2 trials / adopted | "Real-checkpoint execution (45 trials, 15 adopted)" | R | 自报 |
| NV-59 | 63 / 20 | Stage 3 trials / adopted | "W8A8 kernel refinement (63 trials, 20 adopted)" | R | 自报 |
| NV-60 | 63 / 71 / 17 | adopted / failed 或 rolled back / exploratory | "63 produced changes that were adopted." "71 failed validation or were rolled back." | R | 自报（推算核对：43+45+63=151；28+15+20=63；63+71+17=151） |
| NV-61 | 41.7% | 采纳率 | — | R | 推算：63/151 |
| NV-62 | 3.131–3.789 ms | 六个阶梯图面板终点（verify-pass 延迟） | "the panels end between 3.131 and 3.789 ms" | R | 自报 |
| NV-63 | 8-token | 图中 draft block 大小 | "verifying an 8-token draft block, in milliseconds" | R | 自报 |
| NV-64 | ~3.1 ms vs 3.4 ms | verify-pass 终点 vs 完整 round | "The panels end near 3.1 ms rather than at the 3.4 ms full round" | R | 自报 |
| NV-65 | ~3 μs @200 / ~3 μs 变慢 @2,200 | short-context bypass（被拒） | "saved roughly 3 μs at 200 tokens. At 2,200 tokens it became roughly 3 μs slower" | R | 自报 |
| NV-66 | seven rounds | MoE fusion 实现轮数（全部 end-to-end 退化，人类叫停） | "went through seven rounds of implementation" | R | 自报 |
| NV-67 | 41 requests；temp 0.4；top-p 0.95；1 s window | 2,122 tok/s 的测量条件（thinking off，不含 prefill） | "across 41 HTML/SVG generation requests at temperature 0.4 and top-p 0.95" | R | 自报 |
| NV-68 | Oct 12th | GitHub 内容承诺可用日期 | "The above content will be available by Oct, 12th." | R | 自报 |

### G. AutoWM（Case 2）

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| NV-69 | 400 hours | 累计研究时间 | "After 400 hours of cumulative research and 15 major experimental rounds" | R | 自报 |
| NV-70 | 15 | 主要实验轮数 | 同上 | R | 自报 |
| NV-71 | 77.43 | AutoWM 在 WorldArena-1 Track 1 的分数（EWMScore） | "reached a score of 77.43 under the public WorldArena-1 Track 1 evaluation protocol" | R | 自报 |
| NV-72 | 73.64 | 当时已发表 leaderboard 最高分 | "above the highest published leaderboard score at the time, 73.64" | R | 自报（Naive 引用第三方 leaderboard，未给链接） |
| NV-73 | 2.5K → 22.5K clips | 训练集扩充 | "expanded the training set from 2.5K to 22.5K clips" | R | 自报 |
| NV-74 | 8 / 16 / 32 frames | 采帧消融 | "Sixteen evenly spaced frames outperformed eight… thirty-two yielded a smaller additional gain" | R | 自报 |
| NV-75 | N=32 vs N=16 | DP 配置对比（32 更差） | "An alternative DP configuration with N=32 underperformed the N=16 setting." | R | 自报 |
| NV-76 | 3.79 分 | AutoWM 超出原榜首的差值 | — | R | 推算：77.43−73.64 |

## 原文内部不一致 / 待核

1. **SOL-ExecBench 口径冲突**：Naive 引用 Recursive 为 **67.56**（称为 Recursive 更新后的 leaderboard 分数，且量纲为 ×100），而 Recursive 自己的博客写 **0.754**（即 75.4 同量纲）。两者相差 7.84 分，需回源到 https://research.nvidia.com/benchmarks/sol-execbench 核对 leaderboard 当前值及 Recursive 是否下调。
2. **评测 harness 不同**：NV-23~25 是 Naive 用自家 in-house AutoResearch harness 跑的，Recursive 一侧数值来自其文章与 leaderboard，不是同 harness 对比。
3. **图表坐标轴截断**：两张图所有面板的 x 轴都不从 0 开始（如 PostTrainBench 35–42、MLE-bench-30 35%–75%、Terminal-Bench 2.1 84–91、DeepSWE 55–80），**条长与数值不成比例**，视觉上放大差距。报告中复用时须按真实比例重画。
4. **AutoWM 77.43 的构成**：页面自述用 WorldArena 中无需参考视频的指标来做输出选择与后处理（multi-timestep selection、knapsack DP、Best-of-N），即部分增益来自用评测信号做推理期选择，页面未拆分各组件贡献。（subagent 观察，非原文表述）
5. **AutoWM 算力预算**：页面说人类设定了 compute budget，但未给数值；"stronger base model" 未具名。
6. **"AI models" 未具名**：架构与 NaiveRT 部分只说 "AI models"；只有 AutoWM 明确由 Naive-N0.5-Flash 执行。
7. **GPU 型号未给**：NaiveRT 提到 148 CTAs、TMA，但未写硬件名。
8. **NaiveRT 阶梯图面板不可相加**：每个面板测试条件不同。
9. **2,122 tok/s 是峰值**：最佳 1 秒窗口，thinking off，不含 prefill。

## 外链

**NaiveAI 自有**
- https://platform.naive.ai （API）
- https://github.com/naiveai-labs ；https://github.com/naiveai-labs/NaiveRT
- https://huggingface.co/NaiveAI
- https://x.com/naiveailab ；mailto:contact@naive.ai ；https://naive.ai/en/contact/

**竞品与基准出处（Naive 列出）**
- https://z.ai/blog/glm-5.3 ；https://z.ai/blog/glm-5.3-flash
- https://huggingface.co/moonshotai/Kimi-K3
- https://qwen.ai/blog?id=qwen3.8 （← 与参照系 Qwen3.8-Max 相关）
- https://huggingface.co/tencent/Hy4-preview
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- https://huggingface.co/TypeSafeAI/Step-5-Preview-BF16
- https://openai.com/index/gpt-5-6/
- https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf
- https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology
- https://deepswe.datacurve.ai/ ；https://www.tbench.ai/?version=2.1 ；https://agents-last-exam.org/leaderboard ；https://programbench.com/
- https://deepmind.google/models/model-cards/gemini-3-6-flash/
- https://huggingface.co/MiniMaxAI/MiniMax-M3
- https://www.recursive.com/articles/first-steps-toward-automated-ai-research
- https://research.nvidia.com/benchmarks/sol-execbench

**页面未给链接**：FrontierSWE、NL2Repo-Bench、PostTrainBench、PaperBench、NanoChat、NanoGPT SpeedRun、WorldArena、FlowWAM、MiMo-V2.5、DSA 论文。

**图片资源**（naive.ai/assets/research/）：naive-n05-flash-coding.webp、naive-n05-flash-ai-rd.webp、naivert-peak-decode.webp、naivert-dsa-one-kernel.webp、naivert-optimization-trials.webp、autowm-progress.webp、naive-n05-flash-hybrid-architecture.webp、naive-n05-flash-indexer-heads.webp
