# 1. 什么才算 RSI
#### 正文（中文）

递归自我改进（recursive self-improvement，RSI）的设想最早可以追溯到 1965 年。I. J. Good 当时提出，既然设计机器本身是一项智力活动，那么一台在所有智力活动上都超越人类的机器，理应也能设计出更好的机器；新一代机器如法炮制，便会引发“智能爆炸”（intelligence explosion）[T-01]。然而这一概念提出了六十年，学界至今仍未就其严格定义达成共识——就连 ICLR 2026 专门为 RSI 举办的研讨会，最终也没有给出形式化定义 [T-23]。

争议首先在于如何界定“递归”。Yudkowsky 以此区分了两类过程：只有当执行优化的主体与被优化的对象相同时（例如 AI 重新设计自身的认知算法），过程才算递归；若仅仅是一个部件去优化另一个部件，则属于非递归优化（non-recursive optimization）[T-03, T-12]。他用存款作比：利息用于消费，财富仅呈线性增长；利息用于再投资，财富才会按复利累积。按此标准，自然选择与人类文明虽然都在积累，但人类大脑本身并未被改写，因此都属于前一类。不过，即便满足了“修改自身”的条件，改进也未必能持续——早期的 EURISKO 与优化编译器确实在改进自身，但都很快停滞 [T-03]。

[[figure:recursion-loop｜图 1 两种改进。(a) 改进者保持不变，反复改进另一个对象，对应 non-recursive optimization [T-03]。该面板回放一次真实运行：Qwen3.5-9B 在 3 个并行线程中改进 NanoChat 的训练脚本，共 72 次尝试；每次尝试依次经过提出补丁、沙箱训练、封存评估、保留或丢弃 [LOG E；`replay/final/ovn1118b-nc-g0/events.jsonl`]。(b) 改进者改进自身：每一代完整运行一次 (a)，用留下的记录更新自己，再由下一代接替 [T-03, T-12]；各代用时为假设值，用于示意"越改越会改"。(c) 回报率 r 决定进展是否加速：研发由 AI 完成时，r = 0.5 时每次翻倍所需时间变长（软件效率随时间线性增长，对应利息用于消费），r = 1 时每次翻倍所需时间相同，r > 1 时每次翻倍所需时间缩短，并在有限时间内发散 [T-16, T-17]；曲线和翻倍点由公式计算。]]

Yampolskiy 将系统对自身的修改按深度分为三级 [T-12]：

| 级别 | 含义 | 原文的例子 |
|---|---|---|
| self-modification | 改了自己，但没有变好 | 代码混淆 |
| weak self-improvement | 在固定的目标下优化参数 | 遗传算法、遗传编程 |
| recursive self-improvement | 越改越会改（gets better at getting better），能反复把旧方法换成全新的方法 | 原文没有给出例子 |

最高一级要求系统不仅能优化自身，还能反复替换改进自身所用的方法。以超级编译（supercompilation，一种追踪原程序所有可能运行过程、借此剔除冗余计算并编译出等价新程序的变换技术 [S-01]）为例：将其应用于自身时，第一轮约有 20% 的提升，第二轮仅提升约 3%，此后便不再产生增益。至于单纯增加内存或硬件，由于系统改进自身的能力并未改变，更不属于自我改进 [T-12]。

具体到判定标准上，各家的定义主要在以下四个维度上存在分歧：

**改进方法的位置。** Schmidhuber 的 Gödel machine 允许系统改写自身的任何部分，包括用于证明改动有效的搜索程序本身 [T-05]。Yampolskiy 以及 Tang 等人也以此为分界：只要改进方法固定在系统外部，过程就不算递归 [T-12, T-24]。

**改进的对象。** Chalmers 要求被放大的能力必须恰好是“产生下一代的能力”，并称之为自我放大的能力（self-amplifying capacity）[T-09]。这与 Yampolskiy 所说的“越改越会改”（gets better at getting better）相互呼应 [T-12]。

**改进的累积。** Yudkowsky、Chalmers，以及 Epoch 和 Forethought 的研究都强调，下一代带来的提升幅度不能小于上一代；否则收益递减，增长最终会停止 [T-04, T-09, T-16, T-17]。

**通用性门槛。** 早期观点如 Good 和 Hall 认为，启动这一过程的前提是系统具备全面的通用性 [T-01, T-07]；Chalmers 和 Yudkowsky 则放宽了要求，认为低于人类水平的系统同样可能跨过启动门槛 [T-09, T-04]；Eth 和 Davidson 进一步将其收窄，认为系统只需在“AI 研发”这一特定领域内具备通用性即可 [T-17]。

在本文检索到的文献中，Tang 等人在 2026 年给出了唯一一个形式化定义 [T-24]。该定义完全以改进机制的位置划界：改进机制位于 agent 外部时，过程属于 GPI（一般的迭代策略改进）；机制内化于 agent 内部时，过程才属于 RSI。至于改进是否会加速，该定义并未预设，而是留作交由实验回答的问题。

若沿用这一内外部边界来审视当下的常见范式，自蒸馏（self-distillation）与自博弈（self-play）实则都停留在 weak self-improvement 一级：模型虽然自己产生训练数据，但训练流程与更新规则仍由人事先固定在系统外部〔推断〕。目前文献中尚无研究直接对自蒸馏定性，仅有 Mahoney 明确探讨过自博弈的归属。他虽然把自博弈归为自我改进，但也从理论上证明，在没有外部信息输入的前提下，第 n 代程序的复杂度最多按 log n 的速度增长 [T-08]。

而改进过程能否真正加速，可以用回报率 r 来刻画。Epoch 和 Forethought 将 r 定义为：累积研发投入每翻一倍时，软件效率翻倍的次数 [T-16, T-17]。当 r 大于 1 时，每次效率翻倍所需的投入递减，进展呈加速态势；当 r 小于 1 时，进展则逐渐放缓。从历史数据的估算来看，国际象棋引擎的 r 约为 0.83 [T-16]，图像识别约为 1.4，线性规划约为 1.1，SAT 求解器约为 3.5，不过多数估计值在统计上并不显著 [T-17]。Forethought 估计，在算力保持不变的前提下，软件的 r 介于 0.5 到 2 之间，而大语言模型目前尚无对应的估计值 [T-17]。

落到实验层面，上述理论定义最终都归结为同一个可检验的问题：当被改进的后继者接替成为新的改进者时，改进的速度是否比上一代更快？Weco 将跨过这一临界点的时刻称为“点火”（ignition）[X-24]，AIDE² 也据此设计了专门的 ignition test [E-28]。类似地，HGM 通过评估后代的表现来衡量一个 agent 的改进能力，并将其称为元生产力（metaproductivity）[E-27]；STOP 的作者则明确指出，由于底层模型权重并未改变，他们的方法还算不上完整的递归自我改进（"not full recursive self-improvement"）[M-01]。

为了在实验中统一衡量这一差异，本报告借用导数的概念，将“性能的提升”与“改进性能的能力”明确区分开来：

**一阶改进（first-order improvement）** 衡量的是单代性能。改进者自身保持不变，仅被改进的对象持续变好；若以任务表现对尝试次数作图，表现为曲线的斜率为正。例如在本报告第 4 节的实验中，固定不变的 Qwen3.5-9B 将 NanoChat 的验证损失从 1.4152 降至 1.2308，这便属于典型的一阶改进 [LOG E]。四份报告所展示的成果，同样属于这一范畴。

**二阶改进（second-order improvement）** 衡量的则是改进能力本身。在同一组任务和相同的预算下，比较各代改进者的改进速度；只有当改进速度随代际更替而上升——即表现曲线的二阶导数为正时，才构成二阶改进。二阶改进对应了 Weco 所说的 ignition [X-24]，也是 Yampolskiy 所说“越改越会改”的可测量形式 [T-12]。在回报率模型中，二阶改进大致对应于 r ≥ 1 的区间，但两者并不完全等价：r 是按累积投入定义的，而二阶改进则是按代际之间的差值定义的〔推断〕[T-16, T-17]。

在实际应用这一标准时，还需注意两个前提。首先，度量尺度必须事先固定，因为正如 Chalmers 所指出的，对同一组实验数据换用不同的尺度（例如从对数尺度换为线性尺度），可能会得出完全相反的结论 [T-09]。其次，观察到寥寥几代的二阶改进，并不意味着加速能够无限持续：Benthall 以及 Davidson 和 Houlden 都预测，当系统逐渐逼近上限时，改进速度终将回落 [T-13, T-18]。基于此，本报告将“是否实现二阶改进”作为区分真正 RSI 与一般 AI 辅助研发的操作性标准：第 2 节将用这一标尺剖析现有的四份报告，第 4 节的实验也将围绕它展开。

#### Text (English)

The idea of recursive self-improvement (RSI) dates back to 1965. I. J. Good argued that because designing machines is itself an intellectual activity, a machine that surpasses humans across all intellectual activities could also design better machines; each new generation would then design an even better one, setting off an "intelligence explosion" [T-01]. Six decades later, however, the field still lacks an agreed definition—even the ICLR 2026 workshop devoted specifically to RSI stopped short of offering a formal one [T-23].

The first point of contention lies in what makes a process "recursive." Yudkowsky uses the term to separate two distinct regimes: a process is recursive only when the entity doing the optimizing and the object being optimized are one and the same, as when an AI redesigns its own cognitive algorithms; when one component merely optimizes another, the process is non-recursive optimization [T-03, T-12]. He compares the difference to savings: spending the interest yields linear growth in wealth, whereas reinvesting it yields compound growth. By this standard, both natural selection and human civilization accumulate gains without rewriting the human brain itself, placing them in the first category. Yet merely modifying oneself is no guarantee of sustained progress—early systems such as EURISKO and optimizing compilers do improve themselves, but both stall quickly [T-03].

[[figure:recursion-loop｜Figure 1. Two kinds of improvement. (a) A fixed improver repeatedly improves a different object, which corresponds to non-recursive optimization [T-03]. The panel replays a real run: Qwen3.5-9B improving NanoChat training code in 3 parallel threads over 72 attempts, each going through patch proposal, sandbox training, sealed evaluation, and keep or discard [LOG E; replay/final/ovn1118b-nc-g0/events.jsonl]. (b) The improver improves itself: each generation runs (a) once, updates itself from the records, and hands over to the next [T-03, T-12]; generation times are assumed and illustrate getting better at getting better. (c) The rate of return r determines whether progress accelerates. With research done by AI, each doubling takes longer when r = 0.5 (software efficiency grows linearly over time, like spending the interest), every doubling takes the same time when r = 1, and doublings get shorter with a finite-time blow-up when r > 1 [T-16, T-17]; curves and doubling marks are computed from the equations.]]

Yampolskiy divides self-modification into three levels of depth [T-12]:

| Level | Meaning | Example in the source |
|---|---|---|
| self-modification | the system changes itself without getting better | code obfuscation |
| weak self-improvement | parameters are optimized under a fixed objective | genetic algorithms and genetic programming |
| recursive self-improvement | the system gets better at getting better, and can repeatedly replace old methods with entirely new ones | no example given in the source |

Reaching the highest level requires a system not just to tune itself, but to repeatedly replace the very methods by which it improves. Consider supercompilation—a program transformation that traces the possible execution paths of an original program and compiles an equivalent version with redundant computations stripped out [S-01]. When applied to itself, supercompilation yields roughly a 20% speedup on the first pass and about 3% on the second, before flatlining entirely. Simply adding memory or hardware, meanwhile, does not count as self-improvement at all, because the system's underlying capacity to improve itself remains unchanged [T-12].

Across the literature, competing definitions diverge mainly along four dimensions:

**Where the improvement method sits.** Schmidhuber's Gödel machine allows the system to rewrite any part of itself, including the proof-search program that verifies whether a change is beneficial [T-05]. Yampolskiy, as well as Tang and colleagues, draw the same boundary: as long as the improvement mechanism remains fixed outside the system, the process is not recursive [T-12, T-24].

**What is improved.** Chalmers requires that the capacity being amplified be specifically the capacity to produce the next generation—what he terms a self-amplifying capacity [T-09]. This directly echoes Yampolskiy's criterion that the system "gets better at getting better" [T-12].

**Whether gains accumulate.** Yudkowsky, Chalmers, and researchers at Epoch and Forethought all require that each generation deliver at least as much improvement as the one before it; otherwise, diminishing returns set in and growth eventually halts [T-04, T-09, T-16, T-17].

**Generality.** Early accounts by Good and Hall assume the system must possess full generality to start the loop [T-01, T-07]. Chalmers and Yudkowsky relax this threshold, arguing that a sub-human system could also initiate the process [T-09, T-04]. Eth and Davidson narrow the requirement further: the system need only be general within the domain of AI research itself [T-17].

Among the works reviewed for this report, Tang and colleagues (2026) provide the only formal definition of RSI [T-24]. Their formulation hinges entirely on the locus of the improvement mechanism: when the mechanism sits outside the agent, the process reduces to GPI (general iterative policy improvement); only when the mechanism is internalized within the agent does it qualify as RSI. Whether the resulting loop accelerates is left out of the definition itself and treated as an empirical question.

Judged by this internal-external boundary, common paradigms such as self-distillation and self-play fall under weak self-improvement: while the model generates its own training data, the training pipeline and update rules remain fixed in advance by human engineers (an inference). Although no source in the literature classifies self-distillation directly, Mahoney explicitly examines self-play. While he counts self-play as a form of self-improvement, he also proves that without external input, the complexity of the n-th generation can grow at most on the order of log n [T-08].

Whether an improvement loop actually accelerates can be formalized through the rate of return r. Epoch and Forethought define r as the number of times software efficiency doubles for each doubling of cumulative research input [T-16, T-17]. When r exceeds 1, each subsequent doubling requires less input and progress accelerates; when r falls below 1, progress decelerates. Historical estimates place r at roughly 0.83 for computer chess [T-16], 1.4 for image recognition, 1.1 for linear programming, and 3.5 for SAT solvers, though most of these estimates are not statistically significant [T-17]. Holding compute fixed, Forethought estimates r for software at between 0.5 and 2, with no specific estimate yet available for large language models [T-17].

In experimental settings, these theoretical distinctions converge on a single testable question: when an improved successor takes over as the improver, does it drive progress faster than its predecessor did? Weco terms this threshold ignition [X-24], a concept AIDE² operationalizes through its ignition test [E-28]. Along similar lines, HGM evaluates an agent's capacity to improve by the performance of its descendants—calling this metaproductivity [E-27]—while the authors of STOP explicitly note that because their underlying model weights remain frozen, their approach is "not full recursive self-improvement" [M-01].

To evaluate these claims on a consistent footing, this report borrows the language of derivatives to separate gains in performance from gains in the ability to improve performance:

**First-order improvement** measures performance itself. Here, the improver remains fixed while the target object steadily improves; plotting task performance against the number of attempts yields a curve with a positive slope. In Section 4, for instance, a fixed Qwen3.5-9B model lowers the validation loss of NanoChat from 1.4152 to 1.2308—a clear case of first-order improvement [LOG E]. The headline results across all four reports examined later also belong to this category.

**Second-order improvement** measures the ability to improve. Holding the task suite and budget constant across generations, the speed at which each generation of improver makes progress is compared; only when the rate of improvement increases from one generation to the next—meaning the second derivative of the performance curve is positive—does the process achieve second-order improvement. This criterion corresponds to Weco's notion of ignition [X-24] and provides a measurable test for Yampolskiy's "gets better at getting better" [T-12]. In the rate-of-return framework, second-order improvement roughly aligns with the regime where r ≥ 1, though the two are not strictly identical: r is defined over cumulative input, whereas second-order improvement is defined over generational differences (an inference) [T-16, T-17].

Applying this standard in practice requires two caveats. First, the scale of measurement must be fixed in advance; as Chalmers observes, rescaling the same data (for instance, between logarithmic and linear scales) can flip the conclusion entirely [T-09]. Second, observing second-order improvement across a few initial generations does not guarantee indefinite acceleration: both Benthall and Davidson and Houlden predict that improvement speeds will eventually taper off as a system approaches its limits [T-13, T-18]. With those caveats in place, this report adopts second-order improvement as the operational line separating genuine RSI from ordinary AI-assisted research: Section 2 uses it to audit the four reports, and the experiments in Section 4 are designed around it.
