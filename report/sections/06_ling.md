# 6. 面向 Ling 的研究方向
#### 正文（中文）

如果目标是让 Ling 逐代提高研发能力，系统就必须把“下一代是否更善于改进”作为训练信号，而不能只优化某个下游任务的终点分数。本报告的实验表明，经验笔记和模型自选配方都可能把直觉上合理的筛选规则带入更新过程，却没有提高后继者的改进速度。一个可行方向，是让 Ling 提出更新方案，再用独立评估直接检验更新后的 Ling 是否更善于研发。

Ling-3.0-flash 已具备开展 agent 研究的基础：它是总参数 124B、激活参数 5.1B 的 MoE 模型，并在训练中使用了大量交互环境 [S-10]。蚂蚁的 Ring-1T 强化学习系统和 AReaL 异步训练框架提供了可衔接的训练基础设施 [S-12, S-13]。研究问题因此不是再搭一套孤立工具，而是如何把研发任务、更新机制和代际评估纳入一个有可验证反馈的学习循环。

[[figure:fig6-pipeline｜图 20　Ling 的研究闭环。Ling 承担研发任务，并提出对训练数据、更新配方、研究环境或评测的修改。候选修改先在独立评估中比较；只有当它产生的后继在留出任务上表现出更高改进速度，修改才进入下一轮训练。代际结果同时训练 Ling 设计更新的能力。谱系档案保留模型权重、机制版本和相应评估结果。]]

### 把改进能力纳入训练目标

闭环包含四个相互依赖的部分：

| 部分 | 作用 | 设计要点 |
|---|---|---|
| 研发任务池 | 提供训练代码、推理内核、数据处理和配方设计等任务 | 覆盖不同类型，保留未见任务评估 |
| 机制仓库 | 版本化保存数据规则、训练配方、环境、评估器和代码 | Ling 可以提出修改，每项修改可追溯 |
| 独立评估 | 测量修改后的后继能否更快改进 | 采用固定预算、成对任务和封存测试数据 |
| 更新与谱系 | 根据通过评估的方案训练后继，并保留每代状态 | 记录权重、机制版本、数据来源及代际表现 |

关键测量沿用第 4 节的思路：训练候选后继，再把它放回改进者的位置，与对照模型在留出研发任务上比较。这样，Ling 获得的反馈对应于“这项修改是否让后继更善于改进”，而非“提案听起来是否合理”或“某个单次基准是否变好”。DGM 的自修改框架、SEAL 的外层强化学习，以及 AIDE² 的 ignition 测试分别提供了可借鉴的组件 [M-03, M-13, E-28]。

### 从可验证任务起步

初期可优先选择评估信号明确的任务，例如推理内核正确性与速度、固定预算下的训练配方。Ling 先提出一个机制改动，系统在同一任务与预算下构造有改动和无改动的后继，测量二者的改进速度差。只有可复现的正向结果才进入训练数据；失败和无效提案同样保留，作为后续对照。随着测量可靠性提高，再逐步开放更复杂的机制修改，包括环境和评测设计。

训练目标也应从“提出更新”推进到“提出可被验证的更新”。把每项提案的后继表现作为外层学习信号，可以训练 Ling 选择数据、配方和研究环境的能力。奖励应来自独立任务上的代际比较，以避免模型只优化自己编写的评估器。

### 何时称得上进展

Ling 的每一代都应在固定预算的留出研发任务上与前代和对照比较，并报告任务配对的改进速度、独立训练重复和不确定区间。若后继模型持续更快地改进，而这一优势跨任务和随机种子复现，才有证据支持“越改越会改”。任务得分、生成的研究记录数量或模型自述，都不能替代这一测量。

#### Text (English)

If the goal is for Ling to improve its research ability across generations, training must reward whether the next generation becomes a better improver, rather than only optimizing the endpoint score of a downstream task. The experiments in this report show that lessons notes and self-designed recipes can carry intuitive but harmful filters into an update without increasing successor improvement speed. One direction is to let Ling propose updates, then test directly whether the resulting Ling is better at research.

Ling-3.0-flash has a foundation for agent research: it is a MoE model with 124B total and 5.1B active parameters, trained with a large set of interactive environments [S-10]. Ant's Ring-1T reinforcement-learning system and AReaL asynchronous training framework offer infrastructure that could support the loop [S-12, S-13]. The research question is not how to assemble another isolated toolchain, but how to connect research tasks, update mechanisms, and generational evaluation through verifiable feedback.

[[figure:fig6-pipeline｜Figure 20. Ling's research loop. Ling works on R&D tasks and proposes changes to training data, update recipes, research environments, or evaluations. Candidate changes are tested independently, a change advances only when its successor improves faster on held-out tasks. Generational results also train Ling's ability to design updates. The lineage archive retains model weights, mechanism versions, and evaluation results.]]

### Include improvement ability in the training objective

The loop has four interdependent parts:

| Part | Role | Design point |
|---|---|---|
| R&D task pool | Supplies tasks in training code, inference kernels, data processing, and recipe design | Cover varied task types and retain held-out evaluations |
| Mechanism repository | Versions data rules, training recipes, environments, evaluators, and code | Ling can propose changes, and each change is traceable |
| Independent evaluation | Measures whether a change produces a faster-improving successor | Use fixed budgets, paired tasks, and sealed test data |
| Update and lineage | Trains successors from evaluated changes and retains each generation | Record weights, mechanism versions, data sources, and generational results |

The key measurement follows Section 4: train a candidate successor, put it back in the improver role, and compare it with a control on held-out R&D tasks. Ling's feedback then answers whether a change made the successor better at improving, rather than whether the proposal sounded plausible or a single benchmark improved. DGM's self-modification framework, SEAL's outer-loop reinforcement learning, and AIDE²'s ignition test offer components to build on [M-03, M-13, E-28].

### Start with verifiable tasks

Early tasks can prioritize clear evaluation signals, such as inference-kernel correctness and speed or training recipes under a fixed budget. Ling proposes a mechanism change, and the system constructs successors with and without that change under the same task and budget. Only reproducible positive results advance to training, failed and ineffective proposals remain available as controls. As measurement becomes reliable, the system can open more complex mechanisms to modification, including research environments and evaluation design.

The objective should move from "propose an update" to "propose an update that survives evaluation." Using successor performance as an outer-loop learning signal can train Ling to choose data, recipes, and research environments. The reward should come from generational comparisons on independent tasks, so the model cannot simply optimize an evaluator that it wrote itself.

### What would count as progress

Each Ling generation should be compared with its predecessor and a control on held-out R&D tasks under a fixed budget, with task-paired improvement-speed scores, independent training runs, and uncertainty intervals. Repeated gains across tasks and random seeds would support the claim that Ling is getting better at improving. Task scores, the number of generated research records, and the model's own description of its progress cannot replace this measurement.

