# 6. 给 Ling 的计划
#### 正文（中文）

### 目标

这份计划提议让 Ling 的每一代由上一代 Ling 训练出来，而且训练数据怎么筛选、训练配方怎么定、研发环境和评测怎么写，也由上一代 Ling 改写。人不再逐代决定"怎么改进"。按 Tang 等人的判据，改进机制进入系统内部、并可以被系统自身改写，正是从一般的迭代改进走到 RSI 的那一步 [T-24]。

本报告的实验给出了这条管线最重要的设计依据。固定的改进者确实能带来一阶改进 [LOG E]；但模型凭直觉设计自己的更新时会出错：它写给自己的经验笔记让改进速度下降了约三分之一，它给训练数据加的一条"只学显著提升"的筛选规则，让后继模型的改进速度分从 0.60 降到 0.39 [`decisions/decisions_1235.json`；`decisions/decisions_1420.json`]。所以这条管线的核心不是让模型"想"出更好的更新，而是让它"测"出更好的更新，并把这种测量和选择的能力训练进模型本身。

### Ling 的起点

Ling-3.0-flash 是总参数 124B、激活 5.1B 的 MoE 模型，训练中已经用了 "10,000+ interactive training environments" 做 agent 的闭环执行 [S-10]。蚂蚁已有万亿参数规模的 RL 系统 Ring-1T，以及异步 RL 框架 AReaL，后者已经支持 Ling 的模型结构 [S-12；S-13]。训练、rollout 和环境的基础都已经存在。缺的是把它们连成一个闭环、并由 Ling 自己驱动这个闭环的那一层。

### 管线

[[figure:fig6-pipeline｜图 20　Ling 的自我改进管线。当前一代 Ling 以研究 agent 的身份做研发任务，同时向机制仓库提交对"怎么改进"的改动；每一个改动先在小模型试验场里成对实测，测的是训练出的后继当改进者时的改进速度；胜出的配方交给更新引擎训练出下一代 Ling，下一代接手同一个仓库。实测结果同时作为奖励，训练 Ling 设计更新的能力（虚线）。谱系档案保存所有祖先，按实测的改进速度分配研究工作。]]

管线由六个部分组成。每一部分都是 Ling 可以改写的对象，或者是 Ling 改写其他部分时用来判断对错的依据。

| 组件 | 作用 | 依据 |
|---|---|---|
| 机制仓库 | 所有决定"怎么改进"的东西：数据筛选和合成规则、训练配方、研发环境、评测、训练代码。全部版本化，全部可以由 Ling 改写 | Tang 等人的判据 [T-24]；DGM 让 agent 改写自己的代码库 [M-03] |
| 研发环境池 | 持续产生研发任务：训练脚本、推理 kernel、数据处理、配方设计。"设计下一次更新"本身也是其中一个环境 | SEAL 用外层 RL 训练模型生成自己的微调数据 [M-13]；Absolute Zero 让模型自己出题 [M-21] |
| 封存评估器 | 给每次尝试打分，Ling 看不到它的代码和封存数据；信号越硬的任务越先交给 Ling | 本报告实验的封存评估器 [`platform_design.md` §5.1]；信号硬度的文献梳理 [`lit_signal_hardness.md`] |
| 小模型试验场 | 在与 Ling 同架构的小模型上成对实测每一个改动：加这条规则训练一次，不加训练一次，再把两个后继换上去当改进者，比较改进速度 | 本报告第 4 节的测量方法；第 7 节中同一配方重训只差 0.003 [`decisions/decisions_1420.json`] |
| 更新引擎 | 按胜出的配方训练下一代 Ling，以带验证器的 RL 为主 | Ring-1T 的 RL 系统 [S-12] |
| 谱系档案 | 保存所有祖先的权重和仓库版本，按实测的改进速度决定从哪一代分叉、给哪一代更多研究工作 | DGM 的开放式档案 [M-03]；HGM 按后代表现选择父代 [M-04] |

### 三条设计原则

**一、改进过程的每一个改动都先实测，再采用。** 本报告的实验里，三个设计者（原始的 9B 模型、微调后的 9B 模型和 27B 模型）都没能事先判断一条筛选规则的好坏，其中两个主动加入了有害的规则 [LOG E]。同时，同一份配方换随机种子重训，结果只差 0.003 [`decisions/decisions_1420.json`]。这说明配方的好坏凭直觉判断不了，但实测既便宜又准确。管线因此把每一个对"怎么改进"的改动都当作一次成对实验：有它和没有它各训练一个后继，测哪个后继当改进者更快。

**二、成功和失败都记录，并用失败做对照。** 本报告中，用成功记录和用失败记录微调出的模型几乎没有差别 [`decisions/decisions_1235.json`]；Spurious Rewards 也发现，随机奖励能带来真实奖励大部分的提升 [M2-07]。没有失败对照，管线会把"换了一份训练数据"误当成"学会了什么"。所以轨迹库完整保留被采纳、被拒绝和非法的尝试，试验场的每一组实验都带一个用失败记录或随机记录训练的对照。

**三、把"设计下一次更新"训练成一种能力。** 本报告中，未经训练的设计者无法选出更好的配方；SEAL 的自我编辑在经过外层 RL 训练后，成功率从 20% 提高到 72.5% [M-13]。管线把配方设计做成一个 RL 环境：Ling 提出配方，试验场训练出后继并实测它当改进者的速度，这个速度就是 Ling 的奖励。这样，Ling 改进的不只是下一代的权重，也是它自己"设计改进"的能力，这正是"越改越会改"所要求的那一层 [T-12]。

### 从哪里开始

管线先从信号最硬的两个环开始：推理 kernel（正确性可以逐位核对，速度可以直接测量）和固定预算下的训练配方（验证损失可以直接比较）[`lit_signal_hardness.md`]。在这两个环上，Ling 先作为固定的改进者跑一阶循环，把机制仓库、试验场和谱系档案接通。接着把"设计下一次更新"的环境接上，让 Ling 开始改写自己的训练配方和数据规则。最后把仓库的写权限放开到研发环境和评测本身。

基础设施也进入这个循环。本报告通宵运行中记录的 12 次故障，没有一次由系统自己诊断修复 [LOG G]。管线把每一次故障的现场都保存成一个新的研发环境，让 Ling 学会维护驱动自己改进的那套系统。

### 怎样知道它在起作用

每一代都用第 4 节的方法测量：把新一代 Ling 放回改进者的位置，在留出的研发任务上测它的改进速度，并和上一代、和失败对照比较。只要这条代际曲线持续向上，Ling 就在"越改越会改"，这就是 RSI 在发生。

#### Text (English)

### Goal

This plan proposes that each generation of Ling be trained by the previous one, and that the previous Ling also rewrite how training data is filtered, how training recipes are set, and how research environments and evaluations are written. People no longer decide, generation by generation, how improvement happens. By the criterion of Tang et al., moving the improvement mechanism inside the system, where the system itself can rewrite it, is exactly the step from ordinary iterative improvement to RSI [T-24].

This report's experiments supply the most important design input. A fixed improver does deliver first-order improvement [LOG E]; but a model that designs its own update by intuition gets it wrong. The lessons note it wrote for itself cut its improvement speed by about a third, and one "learn only from clear gains" filter it added to its training data lowered the successor's improvement-speed score from 0.60 to 0.39 [`decisions/decisions_1235.json`; `decisions/decisions_1420.json`]. So the core of this pipeline is not to have the model *think up* a better update, but to have it *measure* one, and to train that ability to measure and choose into the model itself.

### Where Ling starts

Ling-3.0-flash is a MoE model with 124B total and 5.1B active parameters, already trained with "10,000+ interactive training environments" for closed-loop agent execution [S-10]. Ant already runs Ring-1T, an RL system at trillion-parameter scale, and the asynchronous RL framework AReaL, which already supports Ling's architecture [S-12; S-13]. The foundations for training, rollout and environments are in place. What is missing is the layer that joins them into one loop and lets Ling drive that loop itself.

### The pipeline

[[figure:fig6-pipeline｜Figure 20. Ling's self-improvement pipeline. The current Ling works as a research agent on R&D tasks and commits changes to "how to improve" into the mechanism repository; each change is first tested in pairs in the small-model testbed, measuring the improvement speed of the resulting successor when it acts as the improver; winning recipes go to the update engine, which trains the next Ling, and the next Ling takes over the same repository. The measured results also serve as the reward that trains Ling's ability to design updates (dashed line). The lineage archive keeps every ancestor and assigns research work by measured improvement speed.]]

The pipeline has six parts. Each is either something Ling can rewrite, or the ground truth Ling uses to judge its rewrites of the other parts.

| Component | Role | Basis |
|---|---|---|
| Mechanism repository | Everything that decides how improvement happens: data filtering and synthesis rules, training recipes, research environments, evaluations, training code. All versioned, all rewritable by Ling | Tang et al.'s criterion [T-24]; DGM lets the agent rewrite its own codebase [M-03] |
| Research environment pool | A steady supply of R&D tasks: training scripts, inference kernels, data processing, recipe design. "Design the next update" is itself one of these environments | SEAL trains a model with outer-loop RL to generate its own fine-tuning data [M-13]; Absolute Zero has the model set its own problems [M-21] |
| Sealed evaluator | Scores every attempt; Ling cannot see its code or sealed data; tasks with harder signals are handed to Ling first | This report's sealed evaluator [`platform_design.md` §5.1]; review of signal hardness [`lit_signal_hardness.md`] |
| Small-model testbed | Tests every change in pairs on small models with Ling's architecture: train once with the rule, once without, then swap both successors in as improvers and compare their speed | The measurement method of §4; in §7, retraining the same recipe moved the score by only 0.003 [`decisions/decisions_1420.json`] |
| Update engine | Trains the next Ling with the winning recipes, mainly by RL with verifiers | Ring-1T's RL system [S-12] |
| Lineage archive | Keeps the weights and repository version of every ancestor; decides by measured improvement speed which generation to branch from and which gets more research work | DGM's open-ended archive [M-03]; HGM selects parents by descendants' performance [M-04] |

### Three design principles

**1. Every change to the improvement process is measured before it is adopted.** In this report, none of the three designers (the original 9B model, the fine-tuned 9B model and the 27B model) could tell in advance whether a filter rule would help, and two of them added a harmful one [LOG E]. Meanwhile, retraining the same recipe with a new seed changed the result by only 0.003 [`decisions/decisions_1420.json`]. Recipes cannot be judged by intuition, but measuring them is cheap and precise. The pipeline therefore treats every change to how improvement happens as a paired experiment: train one successor with it and one without, and measure which is faster as an improver.

**2. Record successes and failures, and use failures as the control.** In this report, models fine-tuned on successful and on failed records were almost identical [`decisions/decisions_1235.json`]; Spurious Rewards likewise found that random rewards deliver most of the gain of the true reward [M2-07]. Without a failure control, the pipeline would mistake "trained on a different dataset" for "learned something". The trajectory store therefore keeps accepted, rejected and invalid attempts alike, and every experiment in the testbed carries a control trained on failed or random records.

**3. Make "designing the next update" a trainable skill.** In this report, untrained designers could not pick better recipes; SEAL's self-edits rose from 20% to 72.5% success after outer-loop RL training [M-13]. The pipeline turns recipe design into an RL environment: Ling proposes a recipe, the testbed trains a successor and measures its speed as an improver, and that speed is Ling's reward. Ling then improves not only the next generation's weights but its own ability to design improvements, which is the layer that "gets better at getting better" requires [T-12].

### Where to begin

The pipeline starts with the two loops that have the hardest signals: inference kernels, where correctness can be checked bit for bit and speed measured directly, and training recipes at a fixed budget, where validation loss can be compared directly [`lit_signal_hardness.md`]. On these two loops Ling first runs as a fixed improver, connecting the mechanism repository, the testbed and the lineage archive. Next the "design the next update" environment is connected, and Ling starts rewriting its own training recipes and data rules. Finally, write access to the repository is opened up to the research environments and the evaluations themselves.

Infrastructure enters the loop as well. None of the 12 failures logged in this report's overnight run was diagnosed and fixed by the system itself [LOG G]. The pipeline saves the state of every failure as a new research environment, so that Ling learns to maintain the system that drives its own improvement.

### How to tell it is working

Every generation is measured with the method of §4: put the new Ling back in the improver's seat, measure its improvement speed on held-out R&D tasks, and compare it with the previous generation and with a failure control. As long as that curve across generations keeps rising, Ling is getting better at getting better, and that is RSI happening.
