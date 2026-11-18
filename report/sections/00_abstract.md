# 0. 摘要
#### 卡片（中文）

1. **单代 AI 研究能带来改进**：Qwen3.5-9B 独立运行 3 小时、尝试 72 次，将 NanoChat 验证损失从 1.4152 降至 1.2308 [LOG E]。
2. **跨代加速尚未出现**：三代后继模型均未快过原始模型；用成功记录微调，与用失败记录微调的安慰剂表现接近 [LOG E]。
3. **模型自选更新方案没有带来稳定收益**：经验笔记让改进速度下降约三分之一；第三轮自定配方得分为 0.392，低于原始模型的 0.588 [LOG E]。

#### Cards (English)

1. **AI can improve a research target within one generation**: In a three-hour run with 72 attempts, Qwen3.5-9B lowered NanoChat's validation loss from 1.4152 to 1.2308 [LOG E].
2. **No cross-generation speed-up emerged**: None of the three successors improved faster than the original model. Fine-tuning on accepted records performed similarly to a placebo trained on rejected records [LOG E].
3. **Self-designed updates brought no reliable gain**: A lessons note cut improvement speed by about a third. The Round-3 self-designed recipe scored 0.392, below the original model's 0.588 [LOG E].

#### 正文（中文）

[[figure:fig0-forest｜总览图 换上后继模型之后，改进速度变快了吗？每一行代表一种自我更新方案；圆点表示该方案相对于原始模型（Qwen3.5-9B）的改进速度差值，横线为 95% 置信区间；灰色阴影带表示同一原始模型独立运行两次时的随机波动范围（±2 个标准差）。圆点落在 0 刻度线左侧即意味着比原始模型更慢。所有数据均测自 8 个未曾见过的小型研究任务 [LOG E；`decisions/decisions_1420.json`]。]]

AI 已经参与模型、代码和训练流程的研发。但“AI 帮助改进某个对象”与“AI 让自己一代比一代更善于改进”是两个不同的主张。本文将后者称为递归自我改进（recursive self-improvement，RSI）：系统需要能够改变自身的改进机制，并在代际比较中提高改进速度 [T-05, T-12, T-24]。

我先梳理 RSI 的理论定义、相关学术工作，以及 Recursive、Naive 和 Apex 的四份公开报告。它们展示了 AI 对训练代码、推理系统、模型结构或训练数据的改进，却没有报告后继模型接替改进者后是否变得更快。现有自训练研究通常报告模型性能变化，少有工作直接比较各代的改进速度；在本次检索的文献范围内，我没有找到模型权重层面的受控代际测量 [E-28, `lit_rsi_academic.md` §1, §6]。

为检验这一步，我让 Qwen3.5-9B 先独立研究 NanoChat，再用研究记录训练后继模型，并在 8 个未见过的小型任务上比较各代的改进速度。固定的原始模型完成了有效的一阶改进：NanoChat 验证损失从 1.4152 降至 1.2308。跨代结果则不同：三代后继模型没有一代超过原始模型；成功记录微调出的模型也没有胜过失败记录安慰剂 [LOG E, `decisions/decisions_0905.json`]。

模型写给自己的经验笔记使改进速度下降约三分之一，在 8 个任务上均不如通用建议。让模型自行设计训练配方也没有产生稳定加速。补充实验显示，第一轮配方换随机种子重训后得分仍接近原结果（0.602 对 0.605）；第三轮的 0.392 因而不能归因于普通的种子波动。用 Qwen3.8-27B 设计配方得到 0.513，仍低于原始模型 [LOG E, `decisions/decisions_1420.json`]。

这些结果不否定 AI 辅助研发的价值。它们指出了证据上的分界：单代研究成功说明模型能做研发，只有当后继模型在相同条件下更快地推动研发，才说明改进能力本身在增长。本研究提供了一个模型权重层面的测量案例，也显示自我更新的成败取决于训练数据选择和训练强度，而不能仅凭模型生成了更新方案来判断。

#### Text (English)

[[figure:fig0-forest｜Overview. Does a successor improve faster? Each row shows one update method. The dot is its difference from the original Qwen3.5-9B, with a 95% confidence interval. The shaded band shows baseline variation when the original model runs twice (±2 SD). A value left of zero means slower improvement. Scores come from eight unseen micro-research tasks [LOG E, `decisions/decisions_1420.json`].]]

AI systems now help develop models, code, and training pipelines. But helping improve an object is different from making the improver itself better at improving. I use *recursive self-improvement* (RSI) for the stronger claim: a system can change its own improvement mechanism, and successive generations increase the speed of improvement [T-05, T-12, T-24].

I review theoretical definitions, related academic work, and four public reports from Recursive, Naive, and Apex. These reports describe AI improving training code, inference systems, model architectures, or training data. They do not show whether a successor becomes a faster improver. Self-training studies generally report changes in model performance rather than generation-to-generation improvement speed. In the literature reviewed here, I found no controlled measurement of this kind at the level of model weights [E-28, `lit_rsi_academic.md` §1, §6].

To examine that gap, I first let Qwen3.5-9B conduct research on NanoChat, then trained successors on its research records and compared their improvement speed across eight unseen micro-tasks. The fixed original model achieved a clear first-order improvement, lowering NanoChat validation loss from 1.4152 to 1.2308. The cross-generation result was different: none of the three successors improved faster than the original, and fine-tuning on accepted records did not outperform a placebo trained on rejected records [LOG E, `decisions/decisions_0905.json`].

A lessons note written from the model's own experience reduced improvement speed by about a third and underperformed generic advice on all eight tasks. Letting the model design its training recipe also produced no reliable acceleration. A new-seed rerun of the Round-1 recipe scored 0.602, close to its original 0.605, so ordinary seed variation does not explain Round 3's 0.392. A recipe designed by Qwen3.8-27B scored 0.513, still below the original model [LOG E, `decisions/decisions_1420.json`].

These findings do not diminish the value of AI-assisted research. They mark an evidential boundary: success within one generation shows that a model can do research, only faster progress by its successors would show that the capacity to improve is itself growing. This study provides one measurement at the model-weight level and shows that outcomes depend on data selection and training intensity, not simply on whether a model proposes its own update.
