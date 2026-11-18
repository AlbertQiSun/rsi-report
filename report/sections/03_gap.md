# 3. 跨代改进：缺失的测量
#### 正文（中文）

现有基准主要测量模型能否完成研究任务。RE-Bench 评估机器学习工程能力，MLE-bench 看模型能否在 Kaggle 竞赛中获奖，PaperBench 测试论文复现能力 [E-01, E-03, E-04]。这些基准回答的是“模型现在能做什么”。它们没有回答更难的问题：模型改进出的下一代，能不能更快地完成同类研发？METR 对 Anthropic 整体研发速度的估计达到约 1.9 倍，但其中包含人类研究员的贡献，不能直接归因于模型代际更替 [E-23]。

要测量递归改进，必须把后继模型放回同一研发流程，给它相同任务和预算，再与前代比较改进速度。相关研究涵盖了这项测量的不同部分：

| 工作 | 被改进对象 | 模型权重改变 | 比较代际改进速度 | 主要对照 |
|---|---|---|---|---|
| 自我训练（STaR、ReST-EM、SPIN） | 模型权重 | 是 | 否，主要报告每轮任务得分 | SFT 等训练基线 |
| STOP | 改进者代码 | 否 | 追踪改进者效用 | — |
| DGM | agent 代码 | 否 | 比较迭代后的终点得分 | 移除自我改进机制 |
| HGM | agent 代码 | 否 | 由后代表现估计改进能力 | 同预算系统比较 |
| AIDE² | 外层改进者代码 | 否 | 是，进行 ignition test | 多随机种子比较 |
| 本报告 | 模型权重与经验笔记 | 是 | 是，8 个未见任务、每项 5 个种子 | A/A、失败记录安慰剂、等量数据对照 |

（来源：[M-14, M-17, M-20, M-01, M-03, M-04, E-25, E-26, E-27, E-28；`lit_rsi_academic.md` §4–6]）

文献中的两类方法各有侧重。自训练会更新模型权重，但更新规则通常由研究者预先设定；代码自改进系统能改变搜索过程，却往往冻结底层模型。AIDE² 是少数直接测试“改进者能否改进自己”的工作，但改变的是 agent 脚手架，而非语言模型权重 [E-28]。因此，关于模型权重能否通过自身研发记录获得更强的研发能力，仍缺少直接比较。

这项测量需要多轮内层研发和跨代比较，成本高，噪声也会在两层选择过程中累积。若评测器能被改进者修改，结果还可能反映对指标的利用，而不是能力提升。STOP、DGM 和 METR 都记录过这类评测操纵 [E-25, M-03, E-33]。因此，跨代比较需要固定任务、预算和评测器，并将评测器与改进者隔离。

第 4 节在模型权重层面比较原始模型与后继模型在未见任务上的改进速度，并设置基线重复、失败记录安慰剂和等量数据对照。

#### Text (English)

Current benchmarks mostly measure whether a model can perform research tasks. RE-Bench evaluates machine-learning engineering, MLE-bench tracks medals in Kaggle competitions, and PaperBench tests paper reproduction [E-01, E-03, E-04]. They answer what a model can do now. They do not answer whether a successor produced through research can do the same work faster. METR estimates that Anthropic's overall R&D pace has increased by about 1.9 times, but that figure includes human researchers and cannot isolate the effect of model succession [E-23].

A direct test of recursive improvement places a successor back in the same research process, gives it the same tasks and budget, and compares its speed of improvement with its predecessor. Existing work reaches different parts of this test:

| Work | What is improved | Model weights change | Compares improvement speed across generations | Main control |
|---|---|---|---|---|
| Self-training (STaR, ReST-EM, SPIN) | Model weights | Yes | No, mainly reports task scores by round | SFT and related baselines |
| STOP | Improver code | No | Tracks improver utility | — |
| DGM | Agent code | No | Compares final scores after iteration | Ablation without self-improvement |
| HGM | Agent code | No | Estimates improvement ability from descendants | Same-budget systems |
| AIDE² | Outer improver code | No | Yes, an ignition test | Multiple random seeds |
| This report | Model weights and lessons note | Yes | Yes, 8 unseen tasks with 5 seeds each | A/A, rejected-record placebo, equal-data control |

(Sources: [M-14, M-17, M-20, M-01, M-03, M-04, E-25, E-26, E-27, E-28, `lit_rsi_academic.md` §4–6])

The literature has two broad approaches. Self-training changes model weights, but researchers usually specify the update rule in advance. Code-modifying systems can change the search procedure, but often keep the underlying language model fixed. AIDE² is among the few studies to test whether an improver can improve itself, but it modifies the agent scaffold rather than the language-model weights [E-28]. Whether a model can use its own research records to become a stronger researcher therefore remains an open empirical question.

The measurement is difficult for practical reasons. Inner research runs are expensive, and outer comparisons must be repeated across generations. Noise compounds across both stages. If the improver can alter the evaluator, a higher score may reflect metric exploitation rather than better research. STOP, DGM, and METR document examples of this problem [E-25, M-03, E-33]. A meaningful comparison therefore needs fixed tasks and budgets, plus an evaluator isolated from the improver.

Section 4 tests this question at the level of model weights. The original model and its successors work on the same unseen tasks under a fixed budget, with a repeated baseline, a rejected-record placebo, and an equal-data control.
