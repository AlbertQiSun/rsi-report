# 导读 04｜Apex Intelligence — Introducing Apex-35B-V1.0

> 建议第 4 篇读。它在 benchmark 表里直接拿 Flint 做对比。另外，它的 "AI for AI" 一节是四份材料里**最接近"固定任务、换研究者"这种设计的**，和你的"二阶测量"思路关系最近。
> 预计用时：40 分钟。机器人和 CFD 两节可以略读，重点放在 **#ai（AI For AI）** 和 "Building … Through Recursive Self-Improvement"。

## 0. 原文信息

| 项 | 内容 |
|---|---|
| 正文 | https://www.apexin.com/apex-35b/#ai |
| 配套 | HF：https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0 ；GitHub：https://github.com/apexin-ai/Apex-35B-V1.0 （README 与 HF card 相同） |
| 机构 | Apex Intelligence（北京 · 波士顿），创始人为清华大学 AI 学院助理教授 Yongchao Chen。**页面没有标日期** |
| 基座 | **Qwen3.6-35B-A3B**，在其上做 mid-training 和 post-training |
| 开源 | 权重以 Apache 2.0 发布，另有 GGUF（BF16 和 28 种量化） |
| 没有的 | 技术报告、训练数据、Apex Research 系统的代码、算力、RSI 迭代日志 |
| 读不到的 | 正文里的 benchmark 总表是图片；HF/GitHub 上有同一张表的文本版，已抄进 `sources/apex_35b.md` |

## 1. 一句话定位

一家新公司的第一个模型：在 Qwen3.6-35B-A3B 上继续训练，主打 AI 研发和科研能力。它用自家的 "Apex Research" 系统声称实现了 RSI。**最有价值的部分是 AI4AI 的两个 case**：让 Apex 当研究者去训练**别的**模型，并且和"基座模型当研究者"做对比。

## 2. 阅读路线

| 段落 | 读法 | 读完应该拿到什么 |
|---|---|---|
| Intro | 速读 | "AI autonomously participated in…"，注意用的是 participated，不是 performed |
| Benchmark Performance + 表格 | 精读表格 | Apex 在 research 类 6 项上都超过了基座和 Flint；但**在 LiveCodeBench 上低于基座**（82.3 vs 84.5）；和更大的模型比，大多数项落后 |
| Robotics / CFD | 略读 | 视觉依赖外部的 GPT-6.1-Sol 工具；RoboRSI、Real2Sim、CFD-Bench 都是**内部 benchmark** |
| **AI For AI（#ai）** | **精读** | 两个任务共用同一个协议：4 小时探索 + 12 小时重训；研究者是 Apex 或 Base；Sokoban 那张图的大量脚注 |
| **Building … Through RSI** | **精读** | 三个机制：self-evolving data、automated experimentation、infra。注意哪些有数字、哪些没有 |
| Footnotes（评测设置） | 精读 | 所有任务 pass@1，只跑一次；对比模型大多是 Apex 自己测的；harness 用的是 Claude Code |

## 3. 卡壳点速查

**`ragen_sokoban_grpo` 任务**
- RAGEN 是一个用 RL 训练多轮 agent 的框架，GRPO 是一种不需要 critic 的 PPO 变体。
- 任务内容：研究者（Apex 或 Base）要在固定预算内，用 GRPO 把 Qwen2.5-3B-Instruct 训练成会推箱子，然后在 512 个没见过的关卡上评测。

**Sokoban 那张图的脚注怎么读**（这几条决定了这个结果的分量）
- **"we identified 36 high-quality … actions"**：207 条命令和编辑是 agent 的原始动作，36 条"高质量动作"和其中 15 条"关键决策"**是人在事后标注的**。
- **"Base exited after 101 minutes without a submission or a completed training step"**：Base 作为研究者，**一次训练都没跑完就退出了**。
- **"base model's 11.7%"**：图注里说橙色点是"未训练的 Qwen2.5-3B-Instruct 参考值"。所以 11.7% 很可能是**被训练模型的起点**，而不是"Base 研究者训练出来的结果"。正文说 "more than doubling the base model's 11.7%"，措辞有歧义。⚠️ **在报告里引用时，必须写清楚 11.7% 指的是什么。**
- "Display step 490 reuses checkpoint 500; step 455 is estimated"：曲线上有的点是复用的，有的是估计的。

**`ddpo_sd15_aesthetic` 任务**
- DDPO 是把扩散模型的去噪过程当作 MDP 来做 RL。这里是用 LoRA 微调 SD1.5，去优化一个固定的 aesthetic reward。
- Apex 的两个发现：
  1. 去噪步数从 50 改成 20，分数基本不变（5.499 vs 5.487），单次 trial 从 1,040 s 降到 551 s；
  2. 把训练轮数上限改成 1,000，加上按时间停止和完整 checkpoint，最终跑了 409 轮。
- 这**本质上是"用更好的预算分配换更多迭代"**，和 autoresearch 里"把训练变快也算改进"是同一类发现。
- 最终分数从 5.397 提到 6.638（+23%）。但优化目标就是这个 reward model，**没有其他人工评价**，存在 reward hacking 的空间。

**Self-evolving data 的 "+32.1 points"**
- 机制：在 CFD 上失败 → 分析推理缺口 → 合成并验证新的训练样本 → 加入后续训练。
- 问题一：这里的"分"单位不明，因为 CFD-Bench 在别处是用胜/平/负来报的。
- 问题二：这个分数是**内部 benchmark** 上的，而新数据正是针对这个 benchmark 的失败去合成的。是否有"对着测试集造数据"的风险，原文没有讨论。

**Infra 数字没有基线**
- 训练吞吐提升 45–60%，解码快 1.6–1.8×，TTFT 降低 30–35%，但**都没有说是和什么比**。

## 4. 五维度速记（假设）

| 维度 | 初判 | 依据 |
|---|---|---|
| 改什么 | 训练数据（针对弱项合成）、训练配方（序列长度分布、长短轨迹采样比例、token-aware batch）、infra 和 kernel；AI4AI 里改的是**别的模型**的训练流程 | RSI 一节 |
| 谁来改 | "Apex Research" 系统。**没有说是由 Apex 自己驱动的**。AI4AI 中明确由 Apex 当研究者 | 角色归属不清楚 |
| 信号 | 内部 benchmark（CFD-Bench）、评测结果和 agent 轨迹分析；AI4AI 里是任务给定的 reward 或成功率 | "Validated results feed into subsequent development"，但验证方法没有说明 |
| 递归层级 | 宣称是 RSI（"each research iteration makes the next more effective"），但**没有迭代次数，也没有逐代对比** | AI4AI 的 Apex 对 Base，是"同一任务换研究者"的**单点对比**，最接近二阶测量的形状 |
| 证据强度 | 弱到中：权重开源，可以复测能力；但过程全靠自报；pass@1 只跑一次；内部 benchmark；对比模型由自己测；人工事后挑选"高质量动作" | 见 Footnotes |

## 5. 和"二阶测量"的关系（读的时候带着这个问题）

AI4AI 的实验结构是：**任务固定、预算固定（4h+12h）、换研究者（Apex 对 Base）、看被训练模型的最终表现**。这在形式上就是"固定基座、换改进者"。它缺的有三样：
- **多次重复**：只跑了一次，没有方差；
- **同一谱系下多代研究者的曲线**：只有 Base 和 Apex 两个点，看不到"每一代是否比上一代更会改进"；
- **公平的 Base 对照**：Base 中途退出，所以比较的其实是"能不能完成任务"，不是"改进得好不好"。

这三点可以作为你方案部分"我们会怎么做得更严谨"的切入口。（只是提示，结论由你来写。）

## 6. 关键数字（自报）

| 数值 | 含义 |
|---|---|
| 35B / ~3B | 总参 / 激活。HF 组件显示 36B |
| 24.4%（125/512） | Sokoban：Apex 当研究者训练出的 Qwen2.5-3B 的成功率 |
| 11.7% | Sokoban："base model's"，**含义有歧义**，见 §3 |
| 207 / 36 / 15 | Sokoban 中的原始动作数 / 人工标注的高质量动作 / 关键决策 |
| 5.397 → 6.638（+23%） | DDPO 的 aesthetic 分数 |
| +32.1 points | 内部 CFD-Bench，self-evolving data 带来的提升 |
| 79/0/21 | CFD-Bench 对基座的胜/平/负 |

完整的 benchmark 表（6 个模型 × 24 行）见 `sources/apex_35b.md`。

## 7. 读完自检

1. Sokoban 里的 11.7% 到底指什么？你会怎么在报告里引用它而不至于误导？
2. "36 high-quality actions" 是谁数出来的？这对"自主性"这个说法有什么影响？
3. AI4AI 的实验设计里，最少要补哪两样，才能算一次合格的"二阶测量"？
4. Apex 在 LiveCodeBench 上低于基座，说明了什么？（提示：专项训练可能带来遗忘，也可能只是单次运行的噪声）
