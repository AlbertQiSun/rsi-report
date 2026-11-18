# 2. 四份材料
#### 正文（中文）

2026 年，Recursive、Naive 和 Apex 相继发布了四份有关 AI 参与研发的报告。本节关注一个具体问题：这些系统改进了什么，以及改进者本身是否随之改变？我对照公开报告、可用的代码和权重，以及各团队报告的评测结果进行梳理。[[figure:fig2-disclosure｜图 2　四份报告公开了什么。实心圆表示公开或具备，半圆表示部分公开，空心圆表示未公开或没有。依据各报告公开内容整理 [RF-Q9, NV-Q2, NV-Q6, FL-Q7, AX-Q10, AX-43]。]]

### Recursive：First Steps Toward Automated AI Research

Recursive 展示了自动化系统优化 NanoChat 训练脚本、NanoGPT Speedrun 和 GPU kernel 的结果。报告称，系统将 NanoChat 验证损失从 0.9372 降至 0.9109，将 Speedrun 耗时从 79.7 秒降至 77.5 秒，并把 SOL-ExecBench 平均分从 0.699 提高到 0.754 [RF-02, RF-05, RF-06, RF-08, RF-11, RF-17, RF-23, RF-25, RF-26]。这些结果展示了自动搜索对研发对象的改进。报告没有测量研究系统自身是否因这些工作变得更善于研发，也没有报告由后继模型接手的代际速度比较。

### Naive：N0.5-Flash、NaiveRT 与 AutoWM

Naive 的报告覆盖模型架构、推理引擎和世界模型研发。NaiveRT 的日志记录了 151 次优化尝试，并报告投机解码耗时从 12.3 毫秒降到 3.4 毫秒 [NV-43, NV-45, NV-46, NV-60]。AutoWM 则由 N0.5-Flash 参与研发，在 WorldArena 上报告 77.43 分 [NV-69, NV-71]。这些案例说明模型可以承担研发工作，也能与人类共同优化系统。但被改进对象仍是外部引擎、模型或任务结果，报告没有比较连续几代研究者的改进速度。

### Recursive：Flint

Flint 是由 Recursive 的 Eureka Machine 辅助从零训练的模型。Recursive 报告了模型规模、训练周期和推理效率等结果，并将其视为自动化研发的产物 [FL-01, FL-02, FL-03, FL-07]。这展示了自动化系统参与模型构建的能力。Flint 本身是否接手并改进 Eureka Machine，报告将其作为未来方向提出，没有给出跨代测量 [FL-Q7]。

### Apex：Apex-35B-V1.0

Apex 在 Qwen3.6-35B-A3B 的基础上继续训练，并以“递归自我改进”描述其 AI4AI 工作 [AX-01, AX-02]。报告举例称，Apex 设计的训练流程将 Qwen2.5-3B-Instruct 在 512 个 Sokoban 关卡上的成功率从 11.7% 提高至 24.4% [AX-22, AX-23, AX-25]。这说明 Apex 能设计并运行一次强化学习训练。报告没有显示 Apex 的后继版本是否更善于设计训练，也没有提供代际改进速度。

### 共同点

四份报告展示了有价值的一阶成果：模型帮助优化代码、架构、推理系统或训练流程，目标系统的指标因此改善。它们没有提供本文所说的二阶证据：让更新后的模型接替研究者，在相同任务和预算下检验它是否比前代改进得更快。这个区分不否定已有工程成果，而是说明“AI 参与研发”与“递归自我改进”需要不同的证据。

#### Text (English)

In 2026, Recursive, Naive, and Apex published four reports on AI-assisted R&D. This section asks a focused question: what did these systems improve, and did the improver itself change? I review the public reports alongside available code, model weights, and the evaluations reported by each team. [[figure:fig2-disclosure｜Figure 2. What the four reports disclose. Filled circles indicate disclosed or available information, half-filled circles indicate partial disclosure, and open circles indicate information that is absent. Based on the reports [RF-Q9, NV-Q2, NV-Q6, FL-Q7, AX-Q10, AX-43].]]

### Recursive: First Steps Toward Automated AI Research

Recursive reports automated improvements to NanoChat training code, NanoGPT Speedrun, and GPU kernels. The system lowered NanoChat validation loss from 0.9372 to 0.9109, reduced Speedrun time from 79.7 to 77.5 seconds, and raised the mean SOL-ExecBench score from 0.699 to 0.754 [RF-02, RF-05, RF-06, RF-08, RF-11, RF-17, RF-23, RF-25, RF-26]. These results show automated search improving its targets. The report does not measure whether the research system itself became a stronger improver, or compare the speed of successive model generations.

### Naive: N0.5-Flash, NaiveRT, and AutoWM

Naive's report covers model architecture, inference systems, and world-model research. NaiveRT logs 151 optimization attempts and reports reducing speculative decoding time from 12.3 to 3.4 milliseconds [NV-43, NV-45, NV-46, NV-60]. AutoWM, developed with N0.5-Flash, reports a score of 77.43 on WorldArena [NV-69, NV-71]. These cases show models taking part in research and jointly optimizing systems with people. The targets remain external engines, models, or task outcomes, and the report does not compare improvement speed across successive researchers.

### Recursive: Flint

Flint is a model trained from scratch with help from Recursive's Eureka Machine. Recursive reports its scale, training timeline, and inference efficiency as outcomes of automated R&D [FL-01, FL-02, FL-03, FL-07]. This demonstrates an automated system contributing to model development. Whether Flint can take over and improve Eureka Machine is presented as future work, no cross-generation measurement is reported [FL-Q7].

### Apex: Apex-35B-V1.0

Apex continues training from Qwen3.6-35B-A3B and describes its AI4AI work as recursive self-improvement [AX-01, AX-02]. One example reports that a training procedure designed by Apex raised Qwen2.5-3B-Instruct's success rate on 512 Sokoban levels from 11.7% to 24.4% [AX-22, AX-23, AX-25]. This shows Apex designing and running a reinforcement-learning experiment. The report does not show whether a successor Apex becomes better at designing such experiments, and it gives no generation-to-generation speed comparison.

### What the reports share

The four reports present meaningful first-order results: AI helps improve code, architectures, inference systems, or training procedures, and the target's measured performance improves. They do not provide the second-order evidence considered here: placing the updated model in the researcher role and testing whether it improves faster than its predecessor under the same tasks and budget. This distinction does not discount the engineering results. It clarifies that AI-assisted R&D and recursive self-improvement require different evidence.
