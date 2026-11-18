# 2. 四份公开报告展示了什么
#### 正文（中文）

2026 年，Recursive、Naive 和 Apex 发布了四份 AI 参与研发的报告。它们分别展示自动搜索、推理系统优化、模型训练和自主设计训练流程，提供了当下 AI4AI 工作的几个具体切面。

[[figure:fig2-disclosure｜图 2　四份报告公开的信息。实心圆表示公开或具备，半圆表示部分公开，空心圆表示未公开或没有。依据各报告内容整理 [RF-Q9, NV-Q2, NV-Q6, FL-Q7, AX-Q10, AX-43]。]]

### Recursive：自动搜索与模型构建

First Steps 将自动搜索用于 NanoChat 训练代码、NanoGPT Speedrun 和 GPU kernel。报告称，NanoChat 验证损失从 0.9372 降至 0.9109，Speedrun 耗时从 79.7 秒降至 77.5 秒，SOL-ExecBench 平均分从 0.699 升至 0.754 [RF-02, RF-05, RF-06, RF-08, RF-11, RF-17, RF-23, RF-25, RF-26]。不同指标体现了自动搜索在训练脚本、程序速度和内核优化上的应用。

Flint 把自动化研发推进到模型构建。Recursive 的 Eureka Machine 辅助训练了一个从零开始的语言模型，报告了模型规模、训练周期与推理效率 [FL-01, FL-02, FL-03, FL-07]。让 Flint 接手 Eureka Machine 的研发工作目前仍是后续方向 [FL-Q7]。

### Naive：从研究尝试到系统结果

NaiveRT 的日志保留了 151 次优化尝试，并报告投机解码耗时从 12.3 毫秒降至 3.4 毫秒 [NV-43, NV-45, NV-46, NV-60]。AutoWM 则给出系统级结果：由 N0.5-Flash 参与研发的世界模型在 WorldArena 得分 77.43 [NV-69, NV-71]。一份材料突出优化过程，另一份呈现模型研发后的基准表现。

### Apex：自我改进的命名与代际证据

Apex 将其 AI4AI 工作称为“递归自我改进” [AX-01, AX-02]。报告中的 Sokoban 案例显示，Apex 设计的训练流程将 Qwen2.5-3B-Instruct 在 512 个关卡上的成功率从 11.7% 提高至 24.4% [AX-22, AX-23, AX-25]。这是一项模型设计并运行强化学习训练的结果。若要判断代际递归，仍需测量后继 Apex 接手研发后能否比前代更快地设计有效训练。

四份报告记录了模型参与训练脚本、内核、推理系统和训练流程研发的成果。本文接下来关注它们尚未直接比较的一项结果：后继研究者的改进速度。

#### Text (English)

In 2026, Recursive, Naive, and Apex published four reports on AI-assisted R&D. They offer distinct views of current AI4AI work, including automated search, inference optimization, model training, and model-designed training procedures.

[[figure:fig2-disclosure｜Figure 2. Information disclosed across the four reports. Filled circles indicate disclosed or available information, half-filled circles indicate partial disclosure, and open circles indicate information that is absent. Compiled from the reports [RF-Q9, NV-Q2, NV-Q6, FL-Q7, AX-Q10, AX-43].]]

### Recursive: automated search and model development

First Steps applies automated search to NanoChat training code, NanoGPT Speedrun, and GPU kernels. It reports lowering NanoChat validation loss from 0.9372 to 0.9109, reducing Speedrun time from 79.7 to 77.5 seconds, and raising the mean SOL-ExecBench score from 0.699 to 0.754 [RF-02, RF-05, RF-06, RF-08, RF-11, RF-17, RF-23, RF-25, RF-26]. These metrics span training scripts, program speed, and kernel optimization.

Flint extends automated R&D to model development. Recursive's Eureka Machine helped train a language model from scratch, and the report gives its scale, training timeline, and inference efficiency [FL-01, FL-02, FL-03, FL-07]. Having Flint take over development of Eureka Machine remains future work [FL-Q7].

### Naive: from research attempts to system results

NaiveRT's logs preserve 151 optimization attempts and report reducing speculative decoding time from 12.3 to 3.4 milliseconds [NV-43, NV-45, NV-46, NV-60]. AutoWM presents a system-level result: the world model developed with N0.5-Flash scored 77.43 on WorldArena [NV-69, NV-71]. One report foregrounds the optimization process, the other the benchmark performance of a model developed with AI assistance.

### Apex: the label and the generational evidence

Apex describes its AI4AI work as "recursive self-improvement" [AX-01, AX-02]. Its Sokoban case reports that an Apex-designed training procedure raised Qwen2.5-3B-Instruct's success rate from 11.7% to 24.4% on 512 levels [AX-22, AX-23, AX-25]. This is a result from a model designing and running a reinforcement-learning experiment. Testing generational recursion requires measuring whether a successor Apex can take over and design effective training faster than its predecessor.

Together, the reports document AI contributions to training scripts, kernels, inference systems, and training procedures. The next result considered here is the one they do not directly compare: the improvement speed of successor researchers.

