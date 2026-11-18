# 1. 什么才算递归自我改进
#### 正文（中文）

1965 年，I. J. Good 提出：如果机器在所有智力活动上都超越人类，它也应能设计更好的机器；继任者再重复这一过程，可能引发“智能爆炸” [T-01]。这一设想后来被称为递归自我改进（recursive self-improvement, RSI），但研究者至今仍未就其边界达成共识。2026 年专门讨论 RSI 的 ICLR workshop 也没有给出通用的形式化定义 [T-23]。

RSI 的关键在于改进循环是否回到改进者自身。Yudkowsky 将“系统优化自身”与“一个系统优化另一个对象”区分开来 [T-03, T-12]。若一个固定的模型持续改进训练代码或目标模型，它完成的是一阶研发进展；若更新后的系统接替研究者，并改变自己后续改进的能力，循环才具有递归性。即使系统能修改自身，也不保证进展持续。EURISKO 和自优化编译器等早期案例都曾取得改进，随后便停滞 [T-03]。

RSI 的定义还涉及两个不同问题：改进机制由谁控制，以及改进是否逐代加速。Tang 等人把前者作为形式边界：当数据选择、训练配方或其他改进机制仍位于 agent 外部时，过程属于一般迭代策略改进（GPI）；当系统能够内化并改写这些机制时，才符合他们对 RSI 的定义 [T-24]。这一界定回答“什么过程算 RSI”，并不预先回答它是否有效。

后一个问题需要测量。Yampolskiy 将自我修改分为自我修改、弱自我改进和递归自我改进三个层次 [T-12]。其中最高一层不只是把自身参数调好，而是“越改越会改”：系统能持续改进产生后继者的方法。Chalmers 将相关能力称为 self-amplifying capacity，即产生下一代的能力本身得到放大 [T-09]。若训练流程仍由人预先固定，自蒸馏或自博弈的任务分数衡量的是一阶表现，不是改进机制的递归增长〔推断〕[T-08, T-12, T-24]。

本文据此把证据分成两层。**一阶改进**是固定的研究者在给定预算内改善目标。Qwen3.5-9B 将 NanoChat 验证损失从 1.4152 降至 1.2308，就是一阶成果 [LOG E]。**跨代改进**则比较继任研究者在相同任务和预算下的改进速度。若后继者的改进速度高于前代，才有直接证据表明改进能力正在增长。Weco 把这一门槛称为 ignition，AIDE² 据此设计了 ignition test [X-24, E-28]；HGM 则用后代表现衡量 agent 产生更强后继者的能力，并称其为 metaproductivity [E-27]。

[[figure:recursion-loop｜图 1 两类改进循环。(a) 固定的改进者持续改进另一个对象，属于非递归的一阶研发 [T-03]。图中回放 Qwen3.5-9B 改进 NanoChat 训练脚本的真实运行，共 72 次尝试 [LOG E, `replay/final/ovn1118b-nc-g0/events.jsonl`]。(b) 后继模型接替成为改进者，继续改进同一研发过程，只有在其改进能力随代际提升时才体现 RSI [T-12, T-24]。(c) $r$ 表示研发投入增加与软件效率增长之间的关系；曲线用于解释加速概念，并不代表本报告实验结果 [T-16, T-17]。]]

宏观研究还用研发回报率 $r$ 描述投入与效率的关系。Epoch 和 Forethought 将 $r$ 定义为研发投入翻倍时软件效率的增长幅度；$r>1$ 对应投入翻倍带来的效率增益逐渐扩大 [T-16, T-17]。历史领域的估计跨度很大，现有工作也没有给出 LLM 研发的直接估计 [T-17]。本文测的是固定预算下的代际改进速度，不能与按累计研发投入定义的 $r$ 直接换算。两者都关注进展是否加速，度量对象则不同。



#### Text (English)

In 1965, I. J. Good argued that a machine capable of surpassing humans across intellectual tasks should also be able to design better machines. Successors repeating this process could trigger an "intelligence explosion" [T-01]. The idea became known as recursive self-improvement (RSI), but researchers still disagree on its boundaries. An ICLR workshop devoted to RSI in 2026 likewise stopped short of a general formal definition [T-23].

The central question is whether the improvement loop returns to the improver. Yudkowsky distinguishes a system optimizing itself from one system optimizing a separate object [T-03, T-12]. A fixed model that repeatedly improves training code or a target model makes first-order research progress. The loop becomes recursive when an updated system takes over as the researcher and changes its own capacity to make further improvements. Self-modification alone does not guarantee continued progress. Early examples such as EURISKO and optimizing compilers improved themselves before stalling [T-03].

Two questions are often bundled together: who controls the improvement mechanism, and whether improvement accelerates across generations. Tang et al. use the first as a formal boundary. When data selection, training recipes, or other mechanisms remain outside the agent, the process is general iterative policy improvement (GPI). It qualifies as RSI under their definition only when the system can internalize and rewrite those mechanisms [T-24]. This definition answers what counts as RSI, not whether RSI works.

Effectiveness requires a separate measurement. Yampolskiy distinguishes self-modification, weak self-improvement, and recursive self-improvement [T-12]. The highest level requires more than tuning the system. The system must get better at getting better by improving how it produces successors. Chalmers calls the relevant capacity self-amplifying capacity, the capacity to produce the next generation [T-09]. Thus, self-distillation or self-play may improve a target task, but if people fix the training process in advance, task scores alone do not establish stronger recursive improvement (an inference) [T-08, T-12, T-24].

This report separates the evidence into two levels. **First-order improvement** occurs when a fixed researcher improves a target under a given budget. Qwen3.5-9B lowering NanoChat validation loss from 1.4152 to 1.2308 is a first-order result [LOG E]. **Cross-generation improvement** compares the speed of successor researchers under the same tasks and budget. A successor that improves faster than its predecessor provides direct evidence that the capacity to improve is growing. Weco calls this threshold ignition, and AIDE² operationalizes it with an ignition test [X-24, E-28]. HGM measures an agent's capacity to produce stronger descendants as metaproductivity [E-27].

[[figure:recursion-loop｜Figure 1. Two kinds of improvement loop. (a) A fixed improver keeps improving another object, which is non-recursive, first-order R&D [T-03]. The panel replays a real run of Qwen3.5-9B improving the NanoChat training script, 72 attempts in all [LOG E, `replay/final/ovn1118b-nc-g0/events.jsonl`]. (b) The successor model takes over as the improver and keeps improving the same R&D process; this counts as RSI only if its ability to improve rises across generations [T-12, T-24]. (c) $r$ describes how software efficiency grows as R&D input increases; the curves illustrate the idea of acceleration and are not results of this report's experiment [T-16, T-17].]]

Macro-level work also uses the R&D return parameter $r$ to describe the relation between research input and efficiency. Epoch and Forethought define $r$ in terms of the efficiency gain associated with a doubling of R&D input. Values above 1 correspond to increasingly large efficiency gains per doubling [T-16, T-17]. Historical estimates vary widely, and no direct estimate is available for LLM research [T-17]. This report measures generational improvement speed under a fixed budget, which cannot be converted directly into $r$, defined over cumulative research input. The two measures address acceleration but quantify different things.



