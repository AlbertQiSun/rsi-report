# 5. 这些结果放在文献中意味着什么
#### 正文（中文）

### 经验能否迁移，取决于经验怎样被验证

本实验中，模型生成的经验笔记使改进速度下降；这与一些反思和记忆研究中的负面结果相呼应。ExpeL、Reflexion 和 ACE 都报告过未经可靠外部信号筛选的反思或上下文改写会降低表现 [M-25, M-23, M-27]。相反，正向结果通常把经验绑定到可核验对象：Sarukkai 等人检索成功轨迹，Voyager 保存通过自我验证的可执行技能 [M-28, M-24]。

跨任务复用经验需要两个条件：环境验证与任务匹配。第 4 节的笔记把局部经验转成跨任务规则，却没有经过新任务的独立检验，结果收窄了搜索空间。Ling 的经验库应保留任务条件和外部结果，并以可复现产物作为策略设计依据。

[[figure:fig5-memory｜图 18　加入反思、经验或记忆后的相对表现变化。不同研究使用的任务和指标不同，图中比较各自相对变化，不比较绝对分数。橙色表示下降，蓝色表示提升 [M-25, M2-10, M-27, M-26, M-23, M-28, `decisions/decisions_1235.json`]。]]

### 成功记录没有教会模型更好的研究判断

第一代成功记录微调模型与失败记录安慰剂几乎相同。这个结果与 Spurious Rewards 的发现相近：在 MATH-500 上，随机奖励取得了真实奖励大部分的提升 [M2-07]。Mind the Gap 也发现，一些表面上的首轮自我改进来自格式遵循，而非任务能力本身 [M2-01]。模型在更新后变好，仍需通过对照判断它是否学到了有效反馈中的信息。数据量、格式和行为激活都可能带来表面提升。

被否决记录构成安慰剂对照，用来区分训练造成的变化与有效研究信号带来的收益。在本实验中，成功记录微调没有带来更快的研究进展。

### 任务得分与改进能力是两种结果

许多自训练研究报告的是模型在目标任务上的表现。SPIN 和 Self-Rewarding 等工作展示了迭代训练后部分指标继续上升，同时也报告边际收益递减或其他任务上的回退 [M-20, M-18]。这些结果有价值，这些结果回答特定任务表现如何变化，尚未直接检验更新后的模型能否更快改进下一项任务。

agent 层面的工作提供了相邻但不同的证据。DGM 显示，让被改进的 agent 执行下一轮代码修改能够提升终点结果 [M-03]。AIDE² 的 ignition test 则直接比较改进后的 coach 与人工构建的 coach，结果未能区分两者 [E-28]。这些研究把注意力从模型单次输出转向研发循环，但测量对象、改进机制和任务预算各不相同，不能直接与权重层面的分数作数值比较。

[[figure:fig5-iter｜图 19　自训练文献中的逐轮任务表现，与本报告测得的逐代改进速度。每项研究沿用其原论文指标；图中并列展示两类指标，不作绝对数值比较 [M-20, M-19, M-18, `decisions/decisions_1235.json`]。]]

宏观研发回报率模型描述累计投入与软件效率提升之间的关系。Epoch 与 Forethought 用 $r$ 表示研发投入翻倍时软件效率增加的幅度，历史领域的估计值差异较大，LLM 研发尚无直接估计 [T-16, T-17]。本报告在固定运行预算下比较各代研究者的改进速度，测量量与基于累计投入定义的 $r$ 不同。本实验因此回答一个局部问题：在当前条件下，后继模型能否让研究循环更快？

第 4 节把微调后的权重放回研究者位置，在固定任务和预算下比较改进速度，并设置 A/A、失败轨迹、等量数据和随机配方对照。后续研究可扩大模型与任务范围、增加独立训练重复，并直接优化后继者的研究速度。

#### Text (English)

### Experience transfers only when it is validated

The model-written lessons note lowered improvement speed, echoing negative results in studies of reflection and memory. ExpeL, Reflexion, and ACE report performance drops when reflections or context rewrites lack reliable external filtering [M-25, M-23, M-27]. Positive results tend to tie experience to verifiable artifacts: Sarukkai et al. retrieve successful trajectories, while Voyager stores executable skills that pass self-verification [M-28, M-24].

Useful memories can be checked against the environment and matched to the task at hand. The note in Section 4 turned local training experience into a cross-task rule without testing it on new tasks, narrowing the search space. For Ling, an experience store should preserve reproducible artifacts with task conditions and external outcomes.

[[figure:fig5-memory｜Figure 18. Relative performance after adding reflection, experience, or memory. Tasks and metrics differ across studies, so the figure compares relative changes rather than absolute scores. Orange indicates a drop and blue a gain [M-25, M2-10, M-27, M-26, M-23, M-28, `decisions/decisions_1235.json`].]]

### Accepted records did not teach research judgment

The first successor fine-tuned on accepted records performed almost identically to the rejected-record placebo. This resembles Spurious Rewards, where random rewards captured most of the gain of true rewards on MATH-500 [M2-07]. Mind the Gap likewise found that some apparent first-round self-improvement came from better format following rather than task ability [M2-01]. A score increase after an update does not show that the model learned useful information from feedback. Data volume, formatting, and behavior activation require controls.

Rejected records serve as a placebo, separating generic effects of training from gains due to useful research signals. Fine-tuning on accepted records did not produce faster research progress in this experiment.

### Task scores and improvement ability measure different outcomes

Many self-training studies report model performance on the target task. SPIN and Self-Rewarding show that some metrics continue to rise across iterations, while also reporting diminishing gains or regressions on other tasks [M-20, M-18]. These findings matter, but they describe changes in task performance. They do not directly test whether an updated model can improve the next task faster.

Agent-level work offers adjacent but distinct evidence. DGM shows that letting an improved agent perform the next round of code edits can raise the final result [M-03]. AIDE²'s ignition test directly compares an evolved coach with a hand-built coach, but does not distinguish them [E-28]. These studies shift attention from a model's isolated output to the R&D loop, yet their targets, update mechanisms, and budgets differ. Their scores cannot be compared numerically with weight-level improvement-speed scores.

[[figure:fig5-iter｜Figure 19. Per-round task performance in self-training studies alongside the generational improvement-speed scores measured here. Each study retains its original metric. The panels show distinct measures and are not on a common scale [M-20, M-19, M-18, `decisions/decisions_1235.json`].]]

Macro models of R&D returns describe the relation between cumulative input and software efficiency. Epoch and Forethought use $r$ to represent the efficiency gain associated with a doubling of research input. Historical estimates vary across fields, and there is no direct estimate for LLM research [T-16, T-17]. This report fixes the budget per run and compares improvement speed across generations. That is not the same quantity as $r$, which is defined over cumulative input. The null result here therefore cannot be converted into an estimate of $r$. It does answer a more local question: under these conditions, did a successor make the research loop faster?

Section 4 compares fine-tuned researchers under fixed tasks and budgets, with A/A, rejected-trajectory, equal-volume data, and random-recipe controls. Further work can extend the model and task range, add independent training runs, and optimize successor research speed.

