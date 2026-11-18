# Apex-35B-V1.0 — 引证台账

## 基本信息

- **URL 已读**（均 HTTP 200，未截断）：
  - A = https://www.apexin.com/apex-35b/#ai （博客，整页）
  - B = https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0 （HF model card）
  - C = https://github.com/apexin-ai/Apex-35B-V1.0 （GitHub README，与 B 逐字一致，含同一张表）
- **标题**："Introducing Apex-35B-V1.0 | Apex Intelligence"
- **机构**：Apex Intelligence（"© 2026 Apex Intelligence"，"Beijing · Boston"）；创始人 "Yongchao Chen, an assistant professor at Tsinghua University's College of AI."
- **日期**：**页面无发布日期**。
- **基座**：Qwen/Qwen3.6-35B-A3B（HF base model 字段）；A 只提 "mid-training and post-training"，预训练继承自基座。
- **License / 发布**：BF16 weights（HF，Apache 2.0，覆盖 "model weights, code examples, and documentation"）；GGUF（BF16 + 28 个量化）；GitHub README 部署文档。**未发布**：训练数据（"proprietary datasets and research trajectories"）、Apex Research 代码、技术报告、算力、RSI 迭代日志。
- **注意**：A 的主榜单是图片（`/assets/benchlist-Bb1xJCvs.jpg`）未读；下方基准表取自 B/C 的文本表。

## 数字台账

### A. 模型与部署

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| AX-01 | 35B | 总参数 | "This 35B general-purpose model is based on Qwen3.6-35B-A3B" | A | 自报 |
| AX-02 | 35B total / ~3B active | 参数划分 | "35B total parameters with approximately 3B activated per token" | B, C | 自报 |
| AX-03 | 262,144 | 上下文（input + output） | "a context window of 262,144 tokens shared by input and output" | B, C | 自报 |
| AX-04 | 36B | HF widget 显示的参数量 | "Model size 36B params" | B | 平台计数 |
| AX-05 | 2,198 / 78 | HF 上月下载 / likes | — | B | 平台计数（抓取时） |
| AX-06 | 4 / 63 / 3 / 1 | GitHub commits / stars / forks / issue | — | C | 平台计数（抓取时） |
| AX-07 | ~72 GB | BF16 weights 体积 | "about 72 GB" | B, C | 自报 |
| AX-08 | 28；37.8 GB（Q8_0）～ 9.0 GB（IQ1_S）；71.1 GB / 2 files | GGUF 量化数量与体积 | "BF16 and 28 quantizations, from Q8_0 (37.8 GB) to IQ1_S (9.0 GB)" | B, C | 自报 |
| AX-09 | ≥141 GB，TP=1 | 单卡部署门槛 | — | B, C | 自报 |
| AX-10 | ~5.4 GB | 满上下文 KV cache | "about 5.4 GB" | B, C | 自报 |
| AX-11 | 65536 | 推荐 max_tokens | — | B, C | 自报 |
| AX-12 | SGLang ≥0.5.22；vLLM ≥0.32.0；Transformers ≥5.21.0；tokenspeed==0.1.1.post20261025 | 框架最低版本 | — | B, C | 自报 |
| AX-13 | v2.1.280 | 推荐 Claude Code harness 版本 | "v2.1.280 preferred" | B, C | 自报 |

### B. 机器人 / CFD 演示

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| AX-14 | 44.5 / 21.55 | Apex RoboRSI / Real2Sim | "achieves scores of 44.5 and 21.55, respectively" | A | 自报 |
| AX-15 | 52.4 / 21.4 | GLM-5.3-Flash RoboRSI / Real2Sim | "the much larger, 300B-class GLM-5.3-Flash (52.4 and 21.4, respectively)" | A | 自报（Apex 自测他方） |
| AX-16 | <50B | Apex 声称领先的规模类别 | "significantly outperforming other models under 50B parameters" | A | 自报 |
| AX-17 | 79% / 79-0-21 | CFD-Bench 对基座 W/D/L | "79% win rate against the base model, with 79 wins, 0 draws, and 21 losses" | A | 自报 |
| AX-18 | 32 | flapping-foil 任务 Fourier 系数数 | "parameterized by 32 Fourier coefficients" | A | 自报 |
| AX-19 | 0.164 / 0.150 / 0.081 | net thrust：Apex / GLM-5.3 / Qwen3.6 | "highest net thrust of 0.164, outperforming GLM-5.3 (0.150) and Qwen3.6-35B-A3B (0.081)" | A | 自报 |
| AX-20 | 64 | vascular 任务 terminal outlets | "vascular network with 64 terminal outlets" | A | 自报 |
| AX-21 | 1.24×10⁻⁵ / 1.41×10⁻⁵ / 1.68×10⁻³ | mismatch：Apex / GLM-5.3 / Qwen3.6 | "lowest mismatch of 1.24×10⁻⁵, outperforming GLM-5.3 (1.41×10⁻⁵)" | A | 自报 |

### C. AI for AI（#ai 锚点）

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| AX-22 | 4 h / 12 h | 探索 / 重训预算（Sokoban 与 DDPO 共用） | "4 hours to explore training strategies and 12 hours to retrain the model" | A | 自报 |
| AX-23 | 512 | Sokoban 未见评测关卡 | "performance evaluated on 512 unseen puzzle levels" | A | 自报 |
| AX-24 | 207 / 36 / 15 | 命令与代码编辑 / 高质量动作 / 关键决策（人工事后标注） | "Across 207 commands and code edits, we identified 36 high-quality research and engineering actions" | A | 自报 |
| AX-25 | 24.4% / 11.7% | Sokoban 成功率：Apex 训出的模型 vs base | "24.4% success rate on 512 unseen levels, more than doubling the base model's 11.7%" | A | 自报 |
| AX-26 | 125 of 512 | 解出关卡数 | "(125 of 512 levels solved)" | A | 自报（推算核对：125/512=24.4%） |
| AX-27 | 3 | 图中 retrospective errors | "36 substantive Apex actions plus 3 retrospective errors" | A | 自报 |
| AX-28 | 9 | 图中 Base milestones | "Triangles show 9 Base milestones" | A | 自报 |
| AX-29 | 101 min | Base 退出时间（无提交、无完成训练步） | "Base exited after 101 minutes without a submission" | A | 自报 |
| AX-30 | 490 / 500 / 455 | 图中 step 标注 | "Display step 490 reuses checkpoint 500; step 455 is estimated." | A | 自报 |
| AX-31 | 1.5 | Stable Diffusion 版本（DDPO + LoRA） | "DDPO to fine-tune Stable Diffusion 1.5 with LoRA" | A | 自报 |
| AX-32 | 20 vs 50 | denoising steps | "the benchmark used 20 denoising steps rather than the default 50" | A | 自报 |
| AX-33 | 5.499 vs 5.487 | aesthetic score（20 步 vs 默认） | "comparable aesthetic scores (5.499 vs. 5.487)" | A | 自报 |
| AX-34 | 1,040 → 551 s | 单次 trial 运行时间 | "trial runtime decreased from 1,040 to 551 seconds" | A | 自报 |
| AX-35 | 1,000 / 409 | 轮数上限 / 实际完成轮数 | "increased the limit to 1,000 rounds … enabling 409 training rounds" | A | 自报 |
| AX-36 | 5.397 → 6.638，23% | 平均 aesthetic score | "improved from 5.397 to 6.638, a 23% increase" | A | 自报（推算核对：6.638/5.397−1≈23.0%） |

### D. RSI 段（"Building Apex-35B-V1.0 Through Recursive Self-Improvement"）与愿景

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| AX-37 | +32.1 points | self-evolving data 带来的 internal CFD-Bench 提升（单位不明） | "improved performance on our internal CFD-Bench by 32.1 points" | A | 自报 |
| AX-38 | 45–60% | 训练吞吐提升 | "achieving 45–60% higher training throughput" | A | 自报 |
| AX-39 | 1.6–1.8× | decode 加速 | "delivering 1.6–1.8× faster decoding" | A | 自报 |
| AX-40 | 30–35% | TTFT 降低 | "30–35% lower time-to-first-token under comparable workloads" | A | 自报 |
| AX-41 | 100,000 | 愿景对标的顶尖 AI 科学家数 | "matches and progressively surpasses that of 100,000 top AI scientists" | A | 自报 |

### E. 评测设置（footnotes，A）

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| AX-42 | 1.0 / 0.95 / 20；reasoning effort 'max' | sampling 设置 | "temperature = 1.0, top-p = 0.95, and top-k = 20" | A | 自报 |
| AX-43 | pass@1（每任务一次） | 运行次数 | "Every task is run once (pass@1)." | A | 自报 |

各基准规模（A footnotes）：SciCode 64 main problems；FrontierScience 60（Research set）；ASI-Bench 60 tasks × B1–B4 = 240 runs（seed31415 split）；BioMysteryBench Human-solvable 73 / Human-difficult 17，200-turn 上限；PaperBench Code-Dev 20 papers（4 需 HF access）；BrowseComp 1,266；WideSearch 200（100 EN / 100 ZH）；BFCL V4 5,106 entries（native function calling，非 Claude Code；unweighted pass rate）；AutomationBench 600（v1.0.6，50 turns 上限）；APEX-Agents 240（v1.1，reward > 0 比例）；ALE 105；Terminal-Bench 2.1 89；SWE-bench Pro v2 642（v2.0.0）；SWE-bench Verified 500；LiveCodeBench v6 1,055；SWE-Atlas QnA 124；MMLU-Pro 12,032（10 选项，5-shot）；AA-LCR 100（v1.1）；HLE text-only 2,158；GPQA Diamond 198。

### F. 基准总表（B/C 文本表；A 中为图片）

列顺序：Apex-35B-V1.0（35B）｜Qwen3.6-35B-A3B（35B）｜Recursive Flint（39.8B）｜GLM-5.3（753B）｜GLM-5.3-Flash（320B）｜DeepSeek-V4.1-Flash-1017（"552B + 196B"）。状态：Apex 自报；他方模型多数为 Apex 自测，少数标注引自官方 HF 页（见下注）。

| id | 类别 | 基准 | Apex | Qwen3.6 | Flint | GLM-5.3 | GLM-5.3-Flash | DeepSeek-V4.1-Flash |
|---|---|---|---|---|---|---|---|---|
| AX-T01 | Research | SciCode (Main / Sub) | 62.6/75.6 | 37.5/60.6 | 26.6/46.0 | 79.2/91.3 | 76.6/86.8 | 68.8/81.9 |
| AX-T02 | Research | FrontierScience | 40.0 | 21.7 | 5.0 | 30.7 | 50.0 | 61.7 |
| AX-T03 | Research | ASI-Bench | 22.5 | 16.3 | 17.5 | 33.1 | 37.4 | 38.9 |
| AX-T04 | Research | BioMysteryBench Human-solvable | 76.7 | 53.4 | 49.3 | 85.6 | 87.7 | 86.3 |
| AX-T05 | Research | BioMysteryBench Human-difficult | 35.3 | 11.8 | 5.9 | 39.1 | 41.2 | 29.4 |
| AX-T06 | Research | PaperBench | 60.9 | 45.8 | 29.3 | 42.5 | 81.8 | 87.1 |
| AX-T07 | Agentic | BrowseComp | 69.4 | 42.1 | 23.2 | 80.9 | 81.0 | 90.5 |
| AX-T08 | Agentic | WideSearch | 69.9 | 60.9 | 48.3 | 78.4 | 81.1 | 79.8 |
| AX-T09 | Agentic | BFCL V4 | 84.0 | 75.2 | 65.4 | 75.8 | 73.9 | 75.0 |
| AX-T10 | Agentic | AutomationBench | 49.3 | 10.0 | 9.8 | 50.2 | 48.8 | 60.5 |
| AX-T11 | Agentic | APEX-Agents | 64.6 | 41.7 | 33.8 | 68.2 | 73.3 | 74.2 |
| AX-T12 | Agentic | Agents' Last Exam (ALE) | 19.1 | 12.4 | 19.1 | 28.5 | 26.3 | 35.4 |
| AX-T13 | Coding | Terminal-Bench 2.1 | 65.2 | 46.1 | 59.6 | 89.9 | 84.3 | 90.6 |
| AX-T14 | Coding | SWE-bench Pro v2 | 92.4 | 72.4 | 85.2 | 95.8 | 94.7 | 97.5 |
| AX-T15 | Coding | SWE-bench Verified (No Internet) | 73.4 | 63.8 | 74.8 | 79.8 | 72.2 | 82.0 |
| AX-T16 | Coding | LiveCodeBench v6 | 82.3 | 84.5 | 85.1 | 78.3 | 75.3 | 95.7 |
| AX-T17 | Coding | SWE-Atlas QnA | 33.1 | 12.1 | 16.1 | 48.7 | 45.2 | 50.8 |
| AX-T18 | Reasoning | MMLU-Pro | 87.2 | 79.0 | 70.8 | 88.2 | 85.5 | 68.3 |
| AX-T19 | Reasoning | AA-LCR | 77.0 | 74.0 | 59.0 | 80.0 | 79.0 | 87.0 |
| AX-T20 | Reasoning | HLE | 47.2 | 36.2 | 18.2 | 61.3 | 55.3 | 42.6 |
| AX-T21 | Reasoning | GPQA Diamond | 83.3 | 75.8 | 66.2 | 88.4 | 84.9 | 90.9 |
| AX-T22 | Internal | RoboRSI | 44.5 | 35.0 | 30.0 | 58.4 | 52.4 | 74.7 |
| AX-T23 | Internal | Real2Sim | 21.6 | 2.1 | 2.3 | 25.6 | 21.4 | 36.3 |
| AX-T24 | Internal | CFD-Bench（W/D/L vs Qwen3.6） | 79/0/21 | N/A | 54/2/44 | 99/0/1 | 98/0/2 | 100/0/0 |

注：AutomationBench、ALE、HLE 中 GLM-5.3-Flash 与 DeepSeek 的分数，以及 Terminal-Bench 2.1 中 GLM-5.3-Flash 的分数，引自各自 HF 页（https://huggingface.co/zai-org/GLM-5.3-Flash 、https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 ，未抓取核对）；其余他方分数为 Apex 用各模型默认 sampling 自测。

subagent 核对（表内读数，非推断）："all six research benchmarks" 胜过 Qwen3.6 与 Flint 成立；Apex 在 LiveCodeBench v6 上**低于**基座（82.3 vs 84.5）；低于 Flint 的有 LiveCodeBench v6（82.3 vs 85.1）、SWE-bench Verified（73.4 vs 74.8），ALE 与 Flint 持平（19.1）。

## 原文内部不一致 / 待核

1. **GLM-5.3-Flash 规模**：A 称 "300B-class"，B/C 表头写 320B。
2. **Real2Sim**：A 正文 21.55，表 21.6（A 注明图表标签保留一位小数）。
3. **RoboRSI 环境名**：footnote 写 RoboDojo（RoboAgent framework）与 RoboTwin；正文写 RobotTwin 与 RoboCasa。
4. **模态标签**：HF 标 "image-text-to-text" 且自动 snippet 传图片，但 card 写 Text/Text；视觉能力依赖外部 "Vision MCP tool backed by GPT-6.1-Sol"。
5. **Flint 分数与 Flint 自报不同**：Apex 测得 Flint TB2.1 **59.6**、SWE-Pro v2 **85.2**、SWE-V **74.8**；Flint HF card 自报 **54.9**、**89.7**、**74.2**。harness 不同（Apex 用 Claude Code），不可直接比。
6. **"+32.1 points" 单位不明**：CFD-Bench 在别处以 W/D/L 报告。
7. **45–60% / 1.6–1.8× / 30–35% 无基线说明**。
8. **Overview 图的 "weighted averages" 权重未给**。
9. **单次运行 pass@1**；三个 internal benchmark（RoboRSI、Real2Sim、CFD-Bench）不可外部复现；BFCL 报告 unweighted pass rate，非官方 weighted overall。
10. **Apex Research 由哪个模型驱动、迭代几轮、接受标准、人工介入程度、算力**——均未说明。Sokoban 例中 "we identified 36…" 表明是人工事后标注。
11. **部署已知问题**：不可用 Qwen3.6 chat template；SGLang + MTP 在 driver 570 失败（"we did not find a fix"）；TokenSpeed TP=2 在 H100 NVL 崩溃；`transformers serve` 不回传前轮 thinking；KTransformers 示例 KV cache 限约 32,000 tokens。

## 外链

**A**
- https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0
- https://github.com/apexin-ai/Apex-35B-V1.0
- https://huggingface.co/datasets/Apexintelligence-AI/ASI-Bench-seed31415 （未抓取）
- https://huggingface.co/zai-org/GLM-5.3-Flash ；https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 （未抓取）
- 站内：/research/（#ai-for-math、#ai-for-ai、#benchmark）、/heurius/、/news/、/contact/ （/research/#ai-for-ai 可能有更多 Apex Research 细节，未抓取）
- https://www.linkedin.com/company/apexin-ai ；https://x.com/apexin_ai ；https://mp.weixin.qq.com/s/fP1ZtWRf_0oTSQ7yyO9djA ；https://www.zhihu.com/org/chao-yan-zhi-neng-apex-intelligence

**B / C**
- https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0-GGUF
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- https://github.com/anthropics/claude-code
- https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0/blob/main/LICENSE
- 框架 repo：SGLang、vLLM、TokenSpeed、KTransformers、Transformers、llama.cpp
