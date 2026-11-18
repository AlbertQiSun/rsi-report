# 0. 摘要
#### 卡片（中文）

1. **AI 辅助单代研发确实有效**：让 Qwen3.5-9B 独立开展 3 小时研究（共 72 次尝试），可将小型语言模型的验证损失从 1.4152 降至 1.2308（越低越好）[LOG E；`replay/final/ovn1118b-nc-g0/events.jsonl`]。
  - ⚠️ 基于另一份独立封存数据的复核结果尚在统计中 [LOG E]。
2. **让模型给自己写经验笔记，反而拖慢进度**：改进速度下降约三分之一，在全部 8 个测试任务上的表现均不如一份通用建议 [LOG E]。
3. **用自身成功经验微调权重，并未实现加速**：连续三代后继模型的改进速度均未超越原始模型，第三代反而更慢；用失败经验训练的安慰剂（placebo）对照组表现与之相同 [LOG E；`decisions/decisions_0905.json`]。
  - 让模型自主决定训练方法：前两轮均与原始模型拉不开差距；第三轮交由首轮微调后的后继模型接管配方设计，改进速度反而显著变慢 [LOG E；`decisions/decisions_1235.json`]。

#### Cards (English)

1. **AI-assisted research works within a single generation**: Running autonomously for 3 hours (72 attempts), Qwen3.5-9B lowered the validation loss of a small model from 1.4152 to 1.2308, where lower is better [LOG E; `replay/final/ovn1118b-nc-g0/events.jsonl`].
2. **Having the model write lessons for itself slowed it down**: Its speed of improvement dropped by roughly a third, underperforming a generic list of tips across all 8 tasks [LOG E].
3. **Training on its own successes did not make it faster**: None of the three successive generations outpaced the original model, and the third grew slower; a placebo control trained on failed attempts performed just as well [LOG E; `decisions/decisions_0905.json`].
  - Letting the model design its own training recipe did not produce reliable acceleration. A new-seed rerun confirmed the Round-1 result, while Round 3 fell to 0.392 [LOG E; `decisions/decisions_1420.json`].

#### 正文（中文）

[[figure:fig0-forest｜总览图 换上后继模型之后，改进速度变快了吗？每一行代表一种自我更新方案；圆点表示该方案相对于原始模型（Qwen3.5-9B）的改进速度差值，横线为 95% 置信区间；灰色阴影带表示同一原始模型独立运行两次时的随机波动范围（±2 个标准差）。圆点落在 0 刻度线左侧即意味着比原始模型更慢。所有数据均测自 8 个未曾见过的小型研究任务；鼠标悬停可查看各行的具体数值 [LOG E；`decisions/decisions_1235.json`]。]]

Recursive、Naive 和 Apex 近期均宣称已实现“用 AI 研发 AI”。但在其公开的四份技术报告中（Recursive 的 First Steps 与 Flint、Naive 的 N0.5-Flash，以及 Apex-35B），AI 优化的客体始终是训练代码、GPU kernel、推理引擎或训练数据（各项指标均为厂商自报），而执行优化的 AI 系统自身从未发生改变。按学术界的界定，这类过程仅属于一个系统对另一个外部对象的非递归优化 [T-03, T-12]，其改进机制依然由人类固定在模型外部 [T-24]，尚不构成递归自我改进（RSI）。

对三个开源权重模型的拆解检验进一步印证了这一点。Apex 是在 Qwen3.6 基础上进行的全参数继续训练，其推理行文被整体替换为类似 Claude 的风格〔推断〕；Naive 基于小米 MiMo-V2.5 的对话版继续训练，其中 9 层注意力权重几乎被重写；Flint 虽为从零训练，但其推理思维链中却直接出现了 "as a Gemini model" 等字样 [`model_dissection_apex_naive.md` §1–3, §5；`flint_reverse_engineering.md` §3]。三款模型的底层基座或推理习惯均深度依赖外部现有模型，仅凭公开材料，无法剥离出它们的能力究竟有多少真正源于自动化研发管线与自动数据收集。

尽管学术界对 RSI 尚无公认的形式化定义 [T-23]，但多数理论工作都要求同时满足两个条件：其一，改进机制位于系统内部，且能被系统自身改写 [T-05, T-12, T-24]；其二，改进收益能够跨代累积，使系统改进的速度一代比一代快 [T-04, T-09, T-16, T-17]——即 Yampolskiy 所说的“越改越会改”（gets better at getting better）[T-12]。在实证层面，迄今尚无研究真正跨越这一门槛。让模型用自身生成的数据反复重训，往往在寥寥数轮内便会陷入停滞甚至退化 [M2-01, M-20, M-17]；在本报告检索的约 96 份文献中，没有任何一项实验测出过后继代际的改进速度超越前代。

此前最接近这一目标的实证测试来自 Weco 的 AIDE² [E-28]。在该实验中，一个“教练” agent 负责反复改写“研究” agent 的代码，并仅保留能提升隐藏测试集平均得分（越高越好）的改动。主实验里，研究 agent 的得分从 0.703 升至 0.778，超越了人工编写的基线 agent（0.749）。随后，研究团队将这个改进后的 agent 提拔为新教练，与人工设计的教练各自运行 3 轮（每轮 50 步）进行对比，两者的最终平均分定格在 0.780 对 0.782，作者据此判定两者表现难分伯仲。不过，该测试改写的仅是外围脚手架代码，底层大语言模型的权重并未改变 [E-28 §3.2, §3.6]。

本报告将这一测试从代码层面推进至模型权重层面。首先，让 Qwen3.5-9B 独立开展 3 小时研究（共 72 次尝试），它将 NanoChat 小模型的验证损失从 1.4152 降至 1.2308（每字节比特数 BPB，越低越好）[LOG E]，证实了单代 AI 辅助研究本身确实有效。随后，用它在探索中被采纳的成功尝试记录微调其自身权重，如此逐代迭代产生三代后继模型，并在 8 个未曾见过的小型研究任务上横向比较各代模型的改进速度。

实验结果显示，三代后继模型的改进速度均未超过原始模型 [LOG E；`decisions/decisions_0905.json`]。其中，第一代后继模型略慢于原始模型，且与使用“失败尝试记录”微调出的安慰剂（placebo）模型表现相当，说明模型并未真正学到哪些改进思路有效；第二代未能超越原始模型，第三代低于第二代，但这一差异尚未达到预设的显著性标准。补充实验显示，第一轮配方换种子重训后得分仍为 0.602，与原结果 0.605 接近；第三轮的下降因此不能归因于一般的随机训练波动。事实上，全篇实验统计上站得住的两个显著结论均为负面。第一个便是：如果不改权重，而是让模型阅读自身的历史记录、总结出经验笔记传给下一轮，其改进速度反而大幅下降约三分之一，表现甚至不如直接给出一份通用建议。

最后一组实验将“如何训练自己”的决策权也一并交还给模型：让它自主审阅历史尝试记录，自行决定筛选哪些样本、采用何种超参数进行微调。第一轮产出的模型，是所有经自身记录微调的模型中唯一一个改进速度略快于原始模型的，同时也快于按固定配方训练的第一代模型；但这两项优势均落在统计误差范围内，且未能胜过随机抽取同等数量记录的对照组 [LOG E；`decisions/decisions_0958.json`]。

在启动第二轮之前，原始模型接收到了第一轮的正向反馈（即它在训练任务上的改进速度超过了原始模型）。然而它并未沿着这一方向继续推进，反而将训练配方改得极度保守——仅筛选出 137 条记录，并以极小的学习率训练了两个 epoch，导致模型权重几乎未发生改变，测试表现也随之退回至原始模型的水平。因此，在前两轮由原始模型主导的配方探索中，没有任何一组对比展现出统计显著的提升 [LOG E；`decisions/decisions_1019.json`]。

第三轮由第一轮微调后的模型设计配方，得分降至 0.392，显著低于原始模型（0.588）和第二轮模型（0.595）。换种子重训第一轮配方后得分为 0.602，表明第一轮与第三轮的差距不能由随机种子波动解释。另以 Qwen3.8-27B 设计配方，得分为 0.513，未超过原始模型。整体结果表明，自主设计配方尚未带来可靠的跨代加速，且筛选规则与训练强度会显著影响结果 [LOG E；`decisions/decisions_1235.json`]。

#### Text (English)

[[figure:fig0-forest｜Overview. Does swapping in a successor speed up improvement? Each row represents one self-update approach; the dot marks its difference in improvement speed relative to the original model (Qwen3.5-9B), the horizontal bar shows the 95% confidence interval, and the grey band indicates the baseline noise when the same model runs twice (±2 SD). A dot to the left of 0 means slower than the original model. All data are measured across 8 unseen micro-research tasks; hover over any row for exact values [LOG E; `decisions/decisions_1235.json`].]]

Recursive, Naive, and Apex have all recently claimed that they are using AI to build AI. Yet across their four public releases (Recursive's *First Steps* and *Flint*, Naive's *N0.5-Flash*, and *Apex-35B*), the targets of AI optimization are consistently training scripts, GPU kernels, inference engines, and synthetic training data—with all headline figures self-reported by the vendors—while the AI system performing the optimization remains unchanged. In the academic literature, this setup constitutes non-recursive optimization: one system optimizing a separate target [T-03, T-12] under an improvement mechanism fixed externally by human engineers [T-24]. It does not yet qualify as recursive self-improvement (RSI).

Direct forensic checks of the three open-weight models reinforce this picture. Apex is a full-parameter continued training run on Qwen3.6 whose reasoning style has been wholesale replaced with patterns resembling Claude's (an inference). Naive continues training from the chat release of Xiaomi's MiMo-V2.5, nearly rewriting 9 of its attention layers. Flint was trained from scratch, yet its reasoning traces occasionally leak phrases such as "as a Gemini model" [`model_dissection_apex_naive.md` §1–3, §5; `flint_reverse_engineering.md` §3]. Because the foundational weights or reasoning habits of all three models derive from existing external models, the public materials alone cannot isolate how much of their capability genuinely stems from automated R&D pipelines or automated data curation.

While academia still lacks a consensus formal definition of RSI [T-23], most theoretical frameworks converge on two requirements: first, the improvement mechanism must reside inside the system and be open to self-modification [T-05, T-12, T-24]; second, the resulting gains must compound across generations so that each generation improves faster than the last [T-04, T-09, T-16, T-17]—a system that, in Yampolskiy's words, "gets better at getting better" [T-12]. Empirically, no system has crossed this threshold. Models trained repeatedly on their own synthetic outputs typically plateau or degrade within a few iterations [M2-01, M-20, M-17], and across roughly 96 sources reviewed for this report, none has ever measured one generation improving faster than its predecessor.

The closest empirical test to date is Weco's AIDE² [E-28]. In that setup, a "coach" agent iteratively rewrites the code of a "research" agent, retaining only modifications that raise the research agent's average score across a suite of tasks evaluated on held-out test data (where higher is better). In the primary run, this score climbed from 0.703 to 0.778, surpassing a hand-crafted baseline agent at 0.749. The improved agent was then promoted to serve as the new coach and pitted against the hand-crafted coach across 3 runs of 50 iterations each, finishing at 0.780 versus 0.782—a tie the authors deemed inconclusive. Moreover, only the scaffolding code changed during that test; the underlying language model weights remained frozen [E-28 §3.2, §3.6].

This report extends the same test from scaffolding code to model weights. First, Qwen3.5-9B conducted autonomous research for 3 hours across 72 attempts, lowering the validation loss of a small NanoChat model from 1.4152 to 1.2308 bits per byte (where lower is better) [LOG E]—confirming that single-generation AI-assisted research works. The model was then fine-tuned on its own accepted research trajectories generation after generation to produce three successive descendants, with each generation's speed of improvement measured across 8 held-out micro-research tasks.

None of the three successors improved faster than the original model [LOG E; `decisions/decisions_0905.json`]. The first generation was slightly slower, and a placebo model trained on rejected trajectories performed just as well, indicating that the model failed to learn which research ideas actually worked. The second generation failed to surpass the original model, while the third did not outperform the original model. The first is that when the model read its own history and distilled a note of lessons learned for the next run without updating its weights, its speed of improvement fell by roughly a third, performing worse than a generic list of tips.

A final set of experiments handed control over the training recipe to the model itself: it inspected its own attempt logs and decided which trajectories to filter and which hyperparameters to use for fine-tuning. Round 1 yielded the only self-tuned checkpoint that improved slightly faster than both the original model and the first-generation checkpoint trained under the fixed recipe. Both margins, however, fell within statistical noise, and the checkpoint failed to beat a control tuned on an equal number of randomly sampled records [LOG E; `decisions/decisions_0958.json`].

Before round 2 began, the base model was shown the positive feedback from round 1—namely, that it had improved faster than the original model on the training tasks. Rather than building on that momentum, it pivoted to an ultra-conservative recipe, selecting just 137 records and training for two epochs at a tiny learning rate. As a result, the weights barely moved and performance regressed to the baseline level of the original model. Across the first two rounds, not a single comparison achieved statistical significance [LOG E; `decisions/decisions_1019.json`].

Round 3 tested genuine recursion: the model fine-tuned in round 1 took over designing the next recipe, given the exact same feedback provided in round 2. Although it kept round 1's recipe nearly intact—adding only a 1% improvement threshold—the resulting model's improvement-speed score dropped from 0.605 in round 1 to 0.392, falling 0.197 below the original model (95% CI [−0.363, −0.050]) and significantly below round 2. A follow-up retraining of the Round-1 recipe with a new random seed scored 0.602, close to the original 0.605. The Round-3 decline therefore reflects the recipe and training conditions rather than ordinary seed variation. A Qwen3.8-27B-designed recipe scored 0.513, below the original model. Together, these results show that recipe design did not produce reliable acceleration, and that data selection and training intensity strongly shaped the outcome [LOG E; `decisions/decisions_1420.json`].
