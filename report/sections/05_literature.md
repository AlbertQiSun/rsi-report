# 5. 这些结果放在文献中意味着什么
#### 正文（中文）

第 4 节的核心不是某一种训练配方失败，而是当前常见的自我更新信号并不能替代对后继研究能力的测量。相关文献帮助解释这一结果，也显示不同的“自我改进”研究实际上在回答不同问题：模型能否提高某项任务得分，agent 能否改写研发脚手架，以及系统能否让下一代更快地改进，三者不能互换。

### 经验能否迁移，取决于经验怎样被验证

本实验中，模型生成的经验笔记使改进速度下降；这与一些反思和记忆研究中的负面结果相呼应。ExpeL、Reflexion 和 ACE 都报告过未经可靠外部信号筛选的反思或上下文改写会降低表现 [M-25, M-23, M-27]。相反，正向结果通常把经验绑定到可核验对象：Sarukkai 等人检索成功轨迹，Voyager 保存通过自我验证的可执行技能 [M-28, M-24]。

这组研究支持一个具体区分：有用的不是“模型写下了经验”，而是经验能否被环境验证、能否匹配当前任务。第 4 节的笔记将局部训练经验转成跨任务规则，没有通过新任务上的独立检验，因而把搜索空间收窄。对 Ling 这类系统，经验库更适合存放带有任务条件和外部结果的可复现产物，而不是把未经检验的反思直接当作策略。

[[figure:fig5-memory｜图 18　加入反思、经验或记忆后的相对表现变化。不同研究使用的任务和指标不同，图中比较各自相对变化，不比较绝对分数。橙色表示下降，蓝色表示提升 [M-25, M2-10, M-27, M-26, M-23, M-28, `decisions/decisions_1235.json`]。]]

### 自生成数据不等于学会判断

第一代成功记录微调模型与失败记录安慰剂几乎相同。这个结果与 Spurious Rewards 的发现相近：在 MATH-500 上，随机奖励取得了真实奖励大部分的提升 [M2-07]。Mind the Gap 也发现，一些表面上的首轮自我改进来自格式遵循，而非任务能力本身 [M2-01]。共同的含义是，模型在更新后变好，不足以单独说明它学到了有效反馈中的信息；对照必须排除数据量、格式或行为激活等替代解释。

因此，本报告把被否决记录纳入安慰剂，而不是只与未经训练的模型比较。这个设计让“训练产生变化”与“训练学到了有效研究信号”成为两个可以区分的问题。当前结果支持前者可能发生，但没有显示后者带来更快的研究进展。

### 一阶指标上升，不等于改进能力上升

许多自训练研究报告的是模型在目标任务上的表现。SPIN 和 Self-Rewarding 等工作展示了迭代训练后部分指标继续上升，同时也报告边际收益递减或其他任务上的回退 [M-20, M-18]。这些结果有价值，但回答的是特定任务表现如何变化。它们没有直接检验更新后的模型是否能以更快速度改进下一项任务。

agent 层面的工作提供了相邻但不同的证据。DGM 显示，让被改进的 agent 执行下一轮代码修改能够提升终点结果 [M-03]。AIDE² 的 ignition test 则直接比较改进后的 coach 与人工构建的 coach，结果未能区分两者 [E-28]。这些研究把注意力从模型单次输出转向研发循环，但测量对象、改进机制和任务预算各不相同，不能直接与权重层面的分数作数值比较。

[[figure:fig5-iter｜图 19　自训练文献中的逐轮任务表现，与本报告测得的逐代改进速度。每项研究沿用其原论文指标；图中并列展示两类指标，不作绝对数值比较 [M-20, M-19, M-18, `decisions/decisions_1235.json`]。]]

宏观的研发回报率模型则描述累计投入与软件效率提升之间的关系。Epoch 与 Forethought 用 $r$ 表示研发投入翻倍时软件效率增加的幅度，历史领域的估计值差异较大，LLM 研发尚无直接估计 [T-16, T-17]。本报告固定每次运行预算，比较不同代研究者的改进速度；这与基于累计投入定义的 $r$ 并非同一个量。因此，本实验的零结果不能换算为 $r$，但它给出了一个更局部的问题：在当前条件下，换上后继模型有没有让研究循环跑得更快？

### 这项测量补上了哪一环

在本报告检索的研究中，工作重点分别落在模型任务得分、agent 代码修改或宏观研发效率；直接把微调后的权重放回研究者位置，再用固定任务与预算比较改进速度的证据仍少。第 4 节提供了这类测量的一个基线，并配备 A/A、失败轨迹、等量数据和随机配方对照。

它的贡献是测量框架和一组可复核的负结果，不是证明 RSI 不可能发生。下一步研究可以扩大基础模型和任务范围，增加独立训练重复，并将后继模型的研究速度作为直接优化目标。只有当模型能在这一指标上持续超过前代，才有证据说明改进能力本身正在增长。

#### Text (English)

The main result in Section 4 is not that one training recipe failed. It is that common self-update signals cannot stand in for measuring a successor's research ability. The literature helps explain this result and clarifies that studies of "self-improvement" often ask different questions: whether a model can raise a task score, whether an agent can rewrite its R&D scaffold, and whether a system can make the next generation improve faster are not interchangeable.

### Experience transfers only when it is validated

The model-written lessons note lowered improvement speed, echoing negative results in studies of reflection and memory. ExpeL, Reflexion, and ACE report performance drops when reflections or context rewrites lack reliable external filtering [M-25, M-23, M-27]. Positive results tend to tie experience to verifiable artifacts: Sarukkai et al. retrieve successful trajectories, while Voyager stores executable skills that pass self-verification [M-28, M-24].

The distinction is concrete. What matters is not that a model has written down a lesson, but whether the lesson can be checked against the environment and matches the task at hand. The note in Section 4 turned local training experience into a cross-task rule without testing it on new tasks, narrowing the search space. For a system such as Ling, an experience store should preserve reproducible artifacts with their task conditions and external outcomes, rather than treat unverified reflections as policy.

[[figure:fig5-memory｜Figure 18. Relative performance after adding reflection, experience, or memory. Tasks and metrics differ across studies, so the figure compares relative changes rather than absolute scores. Orange indicates a drop and blue a gain [M-25, M2-10, M-27, M-26, M-23, M-28, `decisions/decisions_1235.json`].]]

### Self-generated data does not imply learned judgment

The first successor fine-tuned on accepted records performed almost identically to the rejected-record placebo. This resembles Spurious Rewards, where random rewards captured most of the gain of true rewards on MATH-500 [M2-07]. Mind the Gap likewise found that some apparent first-round self-improvement came from better format following rather than task ability [M2-01]. The shared implication is that a model improving after an update does not by itself show that it learned useful information from the feedback. Controls must rule out alternatives such as data volume, formatting, or behavior activation.

This is why the report uses rejected records as a placebo rather than comparing only with an untrained model. The design separates "training changed the model" from "training taught it a useful research signal." The current results are consistent with the former, but do not show that the latter produced faster research progress.

### A higher first-order score is not a higher capacity to improve

Many self-training studies report model performance on the target task. SPIN and Self-Rewarding show that some metrics continue to rise across iterations, while also reporting diminishing gains or regressions on other tasks [M-20, M-18]. These findings matter, but they describe changes in task performance. They do not directly test whether an updated model can improve the next task faster.

Agent-level work offers adjacent but distinct evidence. DGM shows that letting an improved agent perform the next round of code edits can raise the final result [M-03]. AIDE²'s ignition test directly compares an evolved coach with a hand-built coach, but does not distinguish them [E-28]. These studies shift attention from a model's isolated output to the R&D loop, yet their targets, update mechanisms, and budgets differ. Their scores cannot be compared numerically with weight-level improvement-speed scores.

[[figure:fig5-iter｜Figure 19. Per-round task performance in self-training studies alongside the generational improvement-speed scores measured here. Each study retains its original metric. The panels show distinct measures and are not on a common scale [M-20, M-19, M-18, `decisions/decisions_1235.json`].]]

Macro models of R&D returns describe the relation between cumulative input and software efficiency. Epoch and Forethought use $r$ to represent the efficiency gain associated with a doubling of research input. Historical estimates vary across fields, and there is no direct estimate for LLM research [T-16, T-17]. This report fixes the budget per run and compares improvement speed across generations. That is not the same quantity as $r$, which is defined over cumulative input. The null result here therefore cannot be converted into an estimate of $r$. It does answer a more local question: under these conditions, did a successor make the research loop faster?

### What this measurement adds

In the literature reviewed here, studies focus on task scores, agent code modification, or aggregate R&D efficiency. Direct evidence that places fine-tuned weights back in the researcher role and compares improvement speed under fixed tasks and budgets remains limited. Section 4 provides one such measurement, with A/A, rejected-trajectory, equal-volume data, and random-recipe controls.

Its contribution is a measurement framework and a set of auditable negative results, not a proof that RSI cannot occur. Future work can broaden the base models and tasks, add independent training runs, and optimize directly for successor research speed. Evidence that the capacity to improve is itself growing requires successors to keep outperforming their predecessors on this measure.

