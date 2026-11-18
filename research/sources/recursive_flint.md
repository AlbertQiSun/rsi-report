# Recursive — Introducing Flint — 引证台账

## 基本信息

- **URL 已读**（均 HTTP 200，未截断）：
  - A = https://recursive.com/updates/introducing-recursive-flint （博客）
  - B = https://huggingface.co/RecursiveSI/flint （HF model card，org "Recursive Superintelligence, Inc."）
  - C = https://github.com/recursive-oss/rec （rec 推理引擎 README）
  - D = https://docs.recursive.com/getting-started （docs 首页；博客中链接为 http://）
- **标题**："Introducing Flint"；副标题 "Local model with the fastest inference and strongest coding performance in its class. Built with automated research."
- **机构**：Recursive（"© 2026 Recursive Superintelligence, Inc."）；**无作者**
- **日期（冲突）**：页面正文显示 "Oct 18, 2026"；metadata 写 published "Nov 6, 2026, 4:25 PM UTC"。
- **License / 发布**：BF16 weights、config、tokenizer、chat template、SGLang serving kit，Apache 2.0；rec 引擎 Apache-2.0；quantized variants "prepared separately"。**未发布 / 未描述**：Eureka Machine 本身、训练数据、recipe、数据规模与配比、算力、技术报告、第三方评测。"Transformers modeling code is not included yet."
- **训练**：from scratch，pretraining → mid-training → SFT → RL（HF card；card 的 Training 段**未提** Eureka Machine）。

## 数字台账

### A. 博客（A）

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| FL-01 | 40B | 总参数 | "Flint has 40 billion total parameters and activates just 2.5 billion per token." | A | 自报 |
| FL-02 | 2.5B | 每 token 激活参数 | 同上 | A | 自报 |
| FL-03 | under 4 months | 从零构建耗时 | "built it from scratch in under four months" | A | 自报 |
| FL-04 | <50B | 对比类别上限（MoE） | "leads MoE models with fewer than 50 billion total parameters on agentic coding benchmarks" | A | 自报 |
| FL-05 | 25–40B | dense 对比类别 | "dense models with 25–40 billion parameters, while running more than five times faster" | A | 自报 |
| FL-06 | >5x | 相对上述 dense 模型的速度 | 同上 | A | 自报 |
| FL-07 | >125 tok/s | M4 Pro MacBook Pro decode 速度 | "more than 125 tokens per second on an M4 Pro MacBook Pro" | A | 自报 |
| FL-08 | >150 tok/s，64GB | 64GB RTX Spark 短上下文速度 | "more than 150 tokens per second on a 64GB NVIDIA RTX Spark at short contexts" | A | 自报 |
| FL-09 | up to 2x | 相对同类模型速度 | "up to twice as fast as comparable models" | A | 自报 |
| FL-10 | 128k | 速度测量 session 长度（不含 prefill） | "Average tokens/s throughout a 128k-token session." | A | 自报 |
| FL-11 | 124k | 对比模型 Muse 的测量长度 | "*Muse measured at 124k tokens." | A | 自报 |
| FL-12 | 6 | 均分包含的基准数：Terminal-Bench 2.1、SWE-Bench Pro v2、SWE-Bench Verified、LiveCodeBench v6、IFBench、IFEval | — | A（图注） | 自报 |
| FL-13 | 1.0 / 0.95 / 20 | temperature / top-p / top-k | "temperature: 1.0, top-p: 0.95, top-k: 20" | A | 自报 |
| FL-14 | 5 attempts | LiveCodeBench v6 / IFBench / IFEval 的 pass@1 平均次数 | "single-turn, average pass@1 across 5 attempts" | A | 自报 |
| FL-15 | 5 hours；≥5 | agentic 评测超时 / 尝试次数 | "agent timeouts of up to 5 hours, average pass@1 over at least 5 attempts" | A | 自报 |
| FL-16 | 512k / 256k | 最大上下文 / 推荐上限 | "context windows of up to 512k tokens… we recommend staying within 256k tokens" | A | 自报 |
| FL-17 | 35B / 3B | 架构搜索起点（总 / 激活） | "baseline with 35 billion total parameters and 3 billion active parameters" | A | 自报 |
| FL-18 | 4 bits | lm_head 量化位宽 | "it quantized the output layer (lm_head) to 4 bits" | A | 自报 |
| FL-19 | ~half | 训练步数相对 QAT baseline | "nearly halved the number of training steps compared with our quantization-aware training baseline" | A | 自报 |
| FL-20 | 16-bit | teacher 精度 | "with the 16-bit model as teacher" | A | 自报 |
| FL-21 | 0.7 points | NVFP4 相对 16-bit 的差距 | "The final NVFP4 model scores within 0.7 points of the 16-bit model" | A | 自报 |
| FL-22 | Jun 11, 2026 / May 13, 2026 / Sep 9, 2026 | Related 列表：前一篇 Recursive 文章 / NYT 报道 / YouTube 视频日期 | — | A | 自报（NYT 为第三方报道） |
| FL-23 | $4 Billion | NYT 标题中的融资额 | "$4 Billion Effort" | A（Related） | 第三方（NYT 标题） |

### B. HF model card（B）

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| FL-24 | 54.9% | Terminal-Bench 2.1 | "it scores 54.9% on Terminal-Bench 2.1" | B | 自报 |
| FL-25 | 89.7% | SWE-bench Pro v2 | "89.7% on SWE-bench Pro v2" | B | 自报 |
| FL-26 | 74.2% | SWE-bench Verified | "and 74.2% on SWE-bench Verified" | B | 自报 |
| FL-27 | 39.8B | 总参数 | "with 39.8 billion total parameters and approximately 2.5 billion active per token" | B | 自报 |
| FL-28 | ~2.5B / ~2.2B | 激活参数（含 output head）/ 不含 head 的 active body | "The active body alone is approximately 2.2B parameters." | B | 自报 |
| FL-29 | 49（2 dense + 47 MoE） | 层数 | — | B（spec 表） | 自报 |
| FL-30 | 35 / 14 | sliding-window 层 / full-attention 层 | — | B（spec 表） | 自报 |
| FL-31 | 513（当前 + 512 前文） | sliding window 大小 | — | B（spec 表） | 自报 |
| FL-32 | 32 / 4 / 128 | query heads / KV heads / head dim | — | B（spec 表） | 自报 |
| FL-33 | 256 routed / 6 selected / 1 shared | 每个 MoE 层专家配置 | — | B（spec 表） | 自报 |
| FL-34 | 262,144 | 文档化上下文配置（含 prompt + output） | — | B（spec 表） | 自报 |
| FL-35 | 150,144 | 词表大小 | — | B（spec 表） | 自报 |
| FL-36 | June 2026 | knowledge cutoff | — | B（spec 表） | 自报 |
| FL-37 | ~80 GB | 下载体积 | — | B | 自报 |
| FL-38 | ≥32768 | 难题输出预算（含 reasoning） | — | B | 自报 |
| FL-39 | 1 × B300 或 H200 | 每 replica 所需 GPU | — | B | 自报 |
| FL-40 | 54.9 / 74.2 | HF eval widget（Terminal-Bench 2.1 / SWE-bench Verified，指向 discussion #28） | — | B | 自报（Recursive 提交） |
| FL-41 | 40B params，BF16 | HF 页面显示的模型大小 | — | B | 平台计数 |
| FL-42 | 1.41k / 57,853 / 32 | likes / 上月下载 / community discussions | — | B | 平台计数（2026-11-17） |
| FL-43 | 112 / 18 / 1 / 13 | quantizations / finetunes / adapter / Spaces | — | B | 平台计数（2026-11-17） |

### C. rec README（C）与 docs（D）

| id | 数值 | 含义 | 原文短引 | URL | 状态 |
|---|---|---|---|---|---|
| FL-44 | 1.4–2.6× | rec 相对 llama.cpp 吞吐（M4 Pro） | "1.4–2.6×" | C | 自报 |
| FL-45 | M4 Pro 48 GB；4-bit；1K–252K prompt；1,024 output；5 runs | 吞吐图测量设置（不含 prefill） | "Five runs per context" | C | 自报 |
| FL-46 | ≥48 GB / ~30 GB / Node.js 22 / 11500 | 内存 / 磁盘 / 运行时 / 端口 | — | C | 自报 |
| FL-47 | 286 / 48 / 8 / 16 / 24 | stars / forks / commits / issues / PRs | — | C | 平台计数（2026-11-17） |
| FL-48 | Flint-40B | 托管命名 | — | D | 自报 |

## 原文内部不一致 / 待核

1. **SWE-bench Pro v2 89.7% 异常高**：同一张 card 上 SWE-bench Verified 只有 74.2%（subagent 标记，需核对）。
2. **参数量**：博客 40B vs HF card 39.8B（card 注明 "40B-A2.5B" 为四舍五入）。
3. **上下文**：博客 512k（推荐 256k）vs HF card 文档化配置 262,144。
4. **日期**：正文 Oct 18, 2026 vs metadata Nov 6, 2026。
5. **Apex 测得的 Flint 分数与 Flint 自报不同**：Apex 表中 Flint 为 TB2.1 **59.6**、SWE-Pro v2 **85.2**、SWE-V **74.8**；Flint HF card 自报为 **54.9**、**89.7**、**74.2**。双方 harness 不同（Apex 用 Claude Code；Flint 用 Terminus 2 / mini-swe-agent），不可直接比较。
6. **架构选择依据是"projected"**：40B/2.5B 是在测试过的配置中 "was projected to offer the best combination"，未给搜索数据。
7. **Eureka Machine 贡献比例未量化**："helped develop nearly every part"，无逐组件拆分；"Our bet is…" 明确是押注而非已证结果。
8. **图表全部未读**：博客两张图、HF 评测图（† = Recursive 自测，其余为开发者自报）、rec 吞吐图，对比模型中仅知有 "Muse"。
9. **"0.7 points" 的单位**：所跨的 "our benchmarks" 未具体列出。

## 外链

**博客（A）**
- https://recursive.com/updates/first-steps-toward-automated-ai-research
- https://huggingface.co/RecursiveSI/flint
- http://github.com/recursive-oss/rec
- http://docs.recursive.com/getting-started
- https://www.recursive.com/join-us
- https://www.nytimes.com/2026/05/13/technology/recursive-superintelligence-funding-ai.html
- https://www.youtube.com/watch?v=eqYqUAdjsL0
- https://x.com/Recursive_SI ；https://www.linkedin.com/company/recursive-si/
- recursive.com/legal/privacy-policy、/legal/terms-of-use、/legal/model-terms

**HF card（B）**
- https://huggingface.co/RecursiveSI/flint/blob/main/sglang/README.md
- LICENSE；discussion #28（eval results）
- support@ / safety@ / security@recursive.com

**rec README（C）**：https://pi.dev/ ；https://opencode.ai ；https://docs.recursive.com/

**docs（D，两跳，未抓取）**
- https://docs.recursive.com/models/model-card
- https://docs.recursive.com/models/flint.md
- https://docs.recursive.com/guides/generation-settings
- https://docs.recursive.com/resources/releases.md
- https://docs.recursive.com/edge/rec.md
- https://docs.recursive.com/llms.txt
- https://discord.gg/5hRxnYkSYD
