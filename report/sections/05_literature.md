# 5. 与文献对照
#### 正文（中文）

第 4 节的五组实证结果均与既有文献揭示的底层机制吻合。本节将各组结果与最接近的文献逐一对位。各论文采用的评测任务与指标不同，下文仅比较相对变化的方向与成立条件，不直接比较绝对数值大小。文中“未找到”均指在本报告检索的约 96 个文献来源范围内 [`lit_rsi_academic.md` §6, §8]。

[[figure:fig5-map｜图 17 第 4 节的五组结果与文献的对应关系。每一行是一组结果，中间是最接近的文献，右侧是本报告对两者关系的判断〔推断〕。鼠标悬停可以看到文献中的具体数字和出处 [`lit_rsi_academic.md` §6]。]]

### 经验笔记：自由文本的经验会拖慢改进

挂载经验笔记后，原始模型的改进速度分从 0.588 降至 0.392，比未查阅任何记录写出的通用建议还低 0.239 [`decisions/decisions_1235.json`]。无外部真值校验的自由文本反思与记忆堆积在多项研究中表现出同样的退化。ExpeL 在提取经验时引入自我反思后，HotpotQA 成功率从 39.0 降至 29.0 [M-25]。缺乏外部真值参照时，GPT-4 在 GSM8K 上经过两轮自我纠错后准确率从 95.5 降至 89.0 [M2-10]，Reflexion 在 MBPP Python 上的表现也从 0.80 降至 0.77 [M-23]。在上下文维护层面，ACE 整体重写上下文时触发了上下文坍缩（context collapse），准确率从 66.7 跌至 57.1，低于不做任何适应的基线 63.7 [M-27]。Xiong 等人允许 agent 无筛选地累积记忆后，全部四个被测 agent 的表现均低于固定记忆基线（例如从 67.53 降至 55.48）[M-26]。

文献中经验机制产生正收益的案例，均依赖严格的外部过滤或形式化约束。Sarukkai 等人仅保留成功轨迹并按相似度检索，将 ALFWorld 成功率从 73% 提升至 89% [M-28]。Voyager 则仅存储通过自我验证的可执行技能代码 [M-24]。归纳而言，经验机制生效需要满足三个条件之一：由可靠的外部信号过滤条目、测试任务与所存经验同分布，或存储对象为已验证的可执行产物 [`lit_rsi_academic.md` §6]。第 4 节的经验笔记不满足上述任何一条。它是模型自由生成的未过滤文本，且直接迁移至未见过的任务类型。笔记中“忽略任何哪怕略微增加损失的改动”这一规则直接锁死了探索空间，与 ACE 记录的误导性信号污染上下文属于同一机制〔推断〕[LOG E, M-27]。

[[figure:fig5-memory｜图 18 加入自我反思、经验或记忆之后，表现相对原来变化了多少。各研究的任务和指标不同（成功率、准确率、改进速度分），这里只比较相对变化的方向和幅度。橙色是变差，蓝色是变好。Xiong 等人有四个 agent，画成四个点 [M-25, M2-10, M-27, M-26, M-23, M-28；`decisions/decisions_1235.json`]。]]

### 成功记录与失败记录训练出的模型没有差别

第一代后继模型使用被采纳的成功记录微调，安慰剂模型使用等量的被否决记录微调，两者在未见任务上的改进速度分仅相差 +0.004（$p = 0.91$）[`decisions/decisions_1235.json`]。这一结果与 Spurious Rewards 的发现直接对应：在 MATH-500 上对 Qwen2.5-Math-7B 进行强化学习时，真实奖励带来 29.1 个百分点的提升，纯随机奖励也能带来 21.4 个百分点的提升 [M2-07]。该研究指出，训练后模型调用代码推理的比例从约 65% 升至 90% 以上，说明收益源于激活模型固有的行为模式，而非学到了奖励信号本身的内容 [M2-07]。Mind the Gap 同样发现，部分模型首轮自我改进的表观提升仅源于输出格式更加规范 [M2-01]。Sharpening 则从理论上证明，自我改进只能将概率质量集中到模型原本就能生成的输出分布上，无法创造基座模型不具备的信息 [M2-02]。

这组证据指向同一结论：在当前实验规模下，LoRA 微调仅改变了模型的表面调用习惯，并未赋予模型甄别“哪些研究假设有效”的判断力〔推断〕。本报告引入“失败尝试记录微调”作为安慰剂对照以剥离这一混杂因素，在检索到的文献中尚无先例 [`lit_rsi_academic.md` §6]。

### 三代后继模型：没有加速，第三代变差

在人类预设的固定循环内反复自我训练，文献中的一阶任务得分普遍在数轮内饱和或回退。Mind the Gap 测得生成与验证的差距在两至三轮内降至接近 0 [M2-01]。SPIN 经过四轮迭代，平均分从初始的 58.14 依次升至 60.80、62.12、62.97 和 63.16，单轮增量逐次衰减 [M-20]。ReST-EM 在 APPS 上的收益集中在第一轮，继续迭代便出现回退 [M-17]。Self-Rewarding 虽然在 AlpacaEval 胜率上连续三轮上升，但同期 GSM8K 准确率从 60.27 降至 57.70，输出长度则从 1092 膨胀至 2552 token [M-18]。

[[figure:fig5-iter｜图 19 固定循环下的自我训练逐轮表现。前三幅是文献中的结果，各自用论文报告的指标；最后一幅是本报告三代后继模型在未见过任务上的改进速度分。文献中的曲线在被优化的指标上上升、增量递减；本报告测的是“当改进者的速度”，三代都没有超过原始模型 [M-20, M-19, M-18；`decisions/decisions_1235.json`]。]]

当测量指标从一阶任务得分转向二阶改进速度时，三代后继模型均未超过原始模型。补充实验进一步显示，第一轮配方换种子重训后得分保持稳定，而第三轮的下降与其数据筛选和训练强度有关 [`decisions/decisions_1235.json`; `decisions/decisions_1420.json`]。Shumailov 等人将递归训练退化归因于自生成数据导致输出分布尾部逐代消失 [M2-04]，而 Gerstgrasser 等人证明将新生成数据累积在原始真实数据之上（而非替换原始数据）可规避此类分布坍缩 [M2-05]。本实验每一代均从原始权重重新初始化，并采用包含了原始模型全部记录的累积数据集进行训练 [LOG H]。本实验中，跨代改进速度未能提升。结果与自生成数据导致的经典模型坍缩有所不同，因为每一代均从原始权重重新初始化，并累积使用原始模型记录；本实验没有测量输出分布尾部，因此不对两者的机制作等同判断。

### 让模型自己决定怎么训练自己

SEAL 允许模型自主生成微调数据与超参数，在经过外层强化学习训练后，其在 ARC 子集上的自我编辑成功率从未经训练时的 20% 提升至 72.5% [M-13]。第 4 节的配方设计者并未经历外层强化学习训练，仅接收过一次源自训练任务的单轮反馈 [LOG E]。第一轮盲定配方（0.605）与第二轮引入反馈后的配方（0.595），均未与原始模型（0.588）拉开统计显著差距 [`decisions/decisions_1235.json`]。这表明 SEAL 的正收益依赖于外层强化学习对“生成配方”这一元能力的专门训练，仅靠预训练模型自身的零样本或单次反馈决策无法选出更优的训练配方〔推断〕。

第三轮交由第一轮微调后的模型接管配方设计，所得模型的改进速度分暴跌至 0.392 [`decisions/decisions_1235.json`]。这一退化印证了文献中的两项规律。首先，STOP 发现只有当底层模型为 GPT-4 时，改进者改写自身代码才能带来正收益，换用 GPT-3.5 或 Mixtral 时均会发生退化 [M-01]。其次，HGM 证明模型在当前任务上的一阶得分与其繁衍优秀后代的元生产力之间仅呈弱相关（加权 Pearson 相关系数为 0.285 至 0.444）[M-04]。第一轮模型在训练任务上的一阶得分虽高于原始模型（0.652 对 0.525），但它设计的下一次更新却导致严重退化，在模型权重层面直接证实了“下游任务表现更好”不等于“更擅长改进自身”〔推断〕[LOG E, `decisions/decisions_1235.json`]。

### 与已有二阶测量和回报率模型的关系

在模型权重层面，此前文献中尚无直接测量二阶改进速度的先例 [`lit_rsi_academic.md` §5]。脚手架层面的 AIDE² ignition test 测得改进后教练与人工教练的得分为 0.780 对 0.782，作者判定为无显著差异（"we find these results to be inconclusive"）[E-28 §3.6]。DGM 的消融实验显示，移除自我改进机制会使 SWE-bench 和 Polyglot 的终点得分从 50.0 和 38.0 降至 39.0 和 28.0，说明让被改进的 agent 亲自执行下一轮修改能带来额外收益，但并未报告代际改进速度的提升 [M-03]。第 4 节将二阶测试推进至模型权重层面，得出了明确的零结果：三代之内未发生代际加速。

在宏观增长模型中，Epoch 与 Forethought 使用研发回报率 $r$ 刻画软件效率是否加速（当 $r > 1$ 时效率翻倍所需时间逐次缩短）[T-16, T-17]。既有估计主要基于人类研发的历史轨迹，例如图像识别的中位数为 1.4（0.8 至 2.4），国际象棋引擎约为 0.83，Forethought 在固定算力下的综合估计为 0.5 至 2 [T-17, T-16]。本实验测量的是固定预算下各代改进者的速度差值，与按累积研发投入定义的 $r$ 在量纲上不能直接换算；同时，Davidson 与 Houlden 的理论模型同样预测系统逼近能力上限时回报率将回落，因此三代之内的零结果并不排除更长周期或更大规模下出现 $r > 1$ 的可能〔推断〕[T-18]。

### 小结

在本报告的检索范围内，本报告首次在模型权重层面建立了二阶改进的直接测量基线。通过将微调后的后继模型放回改进者位置，并配齐 A/A 噪声标定、失败记录安慰剂与同量数据/随机配方三类严格对照，实验证实：上下文经验总结会因搜索空间收窄而拖慢改进，轨迹微调仅改变调用行为而未习得有效的研究判断，固定循环在三代之内即出现退化，且交由微调后的后继模型接管下一次更新配方会导致表现显著恶化 [`decisions/decisions_1235.json`, `lit_rsi_academic.md` §6]。

#### Text (English)

All five empirical findings in §4 align with mechanisms documented in the existing literature. This section pairs each result with the closest published work. Because tasks and evaluation metrics vary across studies, this section compares the direction of relative changes and their boundary conditions rather than raw magnitudes. Throughout this section, "not found" refers to the roughly 96 sources reviewed for this report [`lit_rsi_academic.md` §6, §8].

[[figure:fig5-map｜Figure 17. How the five results of §4 line up with the literature. Each row is one result, the middle column the closest published work, and the right column this report's judgement of how the two relate (an inference). Hover for the numbers and sources in each paper [`lit_rsi_academic.md` §6].]]

### Lessons note: free-text experience slows improvement

Appending the self-written lessons note reduced the original model's improvement-speed score from 0.588 to 0.392, falling 0.239 below generic tips written without inspecting any logs [`decisions/decisions_1235.json`]. Prior studies consistently show that unverified free-text reflection and unfiltered memory accumulation degrade agent performance. Adding self-reflection to insight extraction in ExpeL lowered HotpotQA success from 39.0 to 29.0 [M-25]. Without an external oracle, two rounds of GPT-4 self-correction on GSM8K reduced accuracy from 95.5 to 89.0 [M2-10], and Reflexion dropped from 0.80 to 0.77 on MBPP Python [M-23]. At the context level, rewriting the full context in ACE triggered context collapse, dropping accuracy from 66.7 to 57.1 and underperforming the unadapted baseline of 63.7 [M-27]. Similarly, when Xiong et al. allowed agents to accumulate memories without filtering, all four tested agents trailed their fixed-memory baselines, dropping for instance from 67.53 to 55.48 [M-26].

Published cases where accumulated experience improves performance rely on strict filtering or formal verification. Sarukkai et al. retained only successful trajectories and retrieved them by similarity, raising ALFWorld success from 73% to 89% [M-28]. Voyager stores only self-verified, executable skill code [M-24]. Across the literature, beneficial experience mechanisms satisfy at least one of three conditions: entries are filtered by a reliable external signal, test tasks closely match the stored experience distribution, or stored artifacts are executable and verified [`lit_rsi_academic.md` §6]. The lessons note in §4 satisfies none of the three. It consists of unfiltered free text generated by the model and applied to novel task types. Its rule to "ignore changes that increase loss even slightly" prematurely restricted the search space, mirroring the context pollution from misleading signals documented in ACE (an inference) [LOG E, M-27].

[[figure:fig5-memory｜Figure 18. Relative change in performance after adding self-reflection, experience or memory. Tasks and metrics differ between studies (success rate, accuracy, improvement-speed score), so only the direction and size of the relative change are compared. Orange means worse, blue better. Xiong et al. report four agents, drawn as four dots [M-25, M2-10, M-27, M-26, M-23, M-28; `decisions/decisions_1235.json`].]]

### Models trained on successful and on failed records do not differ

Generation 1, fine-tuned on accepted attempts, and the placebo model, fine-tuned on an equal number of rejected attempts, differed by only +0.004 on unseen tasks ($p = 0.91$) [`decisions/decisions_1235.json`]. The closest parallel in the literature comes from Spurious Rewards, where reinforcement learning on Qwen2.5-Math-7B improved MATH-500 accuracy by 29.1 percentage points under true rewards and by 21.4 percentage points under purely random rewards [M2-07]. Because the proportion of code-based reasoning rose from roughly 65% to over 90% after training, those gains stemmed from eliciting pre-existing behavioral patterns rather than learning from the reward signal itself [M2-07]. Mind the Gap likewise observed that apparent first-round self-improvement in several models came merely from stricter adherence to answer formatting [M2-01]. Theoretical bounds from Sharpening reinforce this limit: self-improvement can only concentrate probability mass onto outputs the base model can already generate, and cannot create new information [M2-02].

Together, these findings indicate that at the scale of this experiment, LoRA fine-tuning altered the model's surface calling behavior without teaching it which research hypotheses actually work (an inference). Using fine-tuning on rejected attempts as a placebo control to isolate this confound has no precedent among the reviewed sources [`lit_rsi_academic.md` §6].

### Three successor generations: no speed-up, and generation 3 is worse

Repeated self-training inside a fixed, human-engineered loop typically saturates or regresses within a few rounds on first-order benchmarks. Mind the Gap found that the generation-verification gap closes to near zero within two to three rounds [M2-01]. Over four SPIN iterations, average scores progressed from an initial 58.14 to 60.80, 62.12, 62.97, and 63.16 with diminishing marginal returns [M-20]. ReST-EM regressed on APPS immediately after the first round [M-17]. While Self-Rewarding increased its AlpacaEval win rate across three iterations, its GSM8K accuracy simultaneously fell from 60.27 to 57.70 as generation length inflated from 1,092 to 2,552 tokens [M-18].

[[figure:fig5-iter｜Figure 19. Round-by-round results of self-training in a fixed loop. The first three panels are from the literature, each in the paper's own metric; the last panel is the improvement-speed score of this report's three successor generations on the unseen tasks. The published curves rise on the metric being optimized, with shrinking gains; this report measures speed as an improver, and no generation beats the original model [M-20, M-19, M-18; `decisions/decisions_1235.json`].]]

When evaluating second-order improvement speed rather than first-order task scores, stagnation sets in immediately: none of the three successor generations surpassed the original model, eliminating even the first-round jump seen in standard self-training, and Generation 3 fell 0.087 below Generation 2 ($p = 0.0625$) [`decisions/decisions_1235.json`]. Shumailov et al. attribute recursive training degradation to the progressive loss of distribution tails when models train on synthetic outputs [M2-04], whereas Gerstgrasser et al. show that accumulating synthetic data on top of original real data prevents this collapse [M2-05]. In this experiment, every generation was initialized from the original base weights and trained on an accumulated dataset retaining all records from the original model [LOG H]. Observing third-generation degradation ($p < 0.10$) despite data accumulation suggests that decay in second-order improvement speed is not prevented by data accumulation alone, though without direct tail measurements it is not possible to confirm whether the underlying mechanism matches classic model collapse (an inference).

### The model decides how to train itself

In SEAL, allowing a model to generate its own fine-tuning data and hyperparameters raised self-edit success on an ARC subset from 20% untrained to 72.5% after outer-loop reinforcement learning [M-13]. The recipe designer in §4 underwent no outer-loop RL training and received only a single round of feedback from the training tasks [LOG E]. Neither the Round-1 recipe (0.605) nor the post-feedback Round-2 recipe (0.595) differed significantly from the original model (0.588) [`decisions/decisions_1235.json`]. This contrast indicates that SEAL's gains depend on explicitly training the meta-ability to generate recipes via outer-loop RL, whereas zero-shot or single-feedback recipe selection by an untrained model yields no improvement (an inference).

In Round 3, handing recipe design over to the Round-1 fine-tuned checkpoint caused the resulting model's score to plunge to 0.392 [`decisions/decisions_1235.json`]. Two findings in the literature explain this breakdown. First, STOP showed that self-rewriting improved scaffold performance only when backed by GPT-4, whereas weaker backbones such as GPT-3.5 and Mixtral degraded under self-modification [M-01]. Second, HGM demonstrated that a parent agent's score on current benchmarks correlates only weakly with its metaproductivity in producing stronger descendants, with weighted Pearson correlations between 0.285 and 0.444 [M-04]. Although the Round-1 checkpoint outscored the original model on the training tasks (0.652 vs. 0.525), the next update it designed performed significantly worse, confirming at the weight level that higher first-order task performance does not translate into a stronger capacity for self-improvement (an inference) [LOG E, `decisions/decisions_1235.json`].

### Relation to existing second-order measurements and return models

Prior literature contains no directly comparable measurement of second-order improvement at the weight level [`lit_rsi_academic.md` §5]. At the scaffolding level, the AIDE² ignition test recorded 0.780 versus 0.782 between the evolved coach and a hand-built coach, which the authors judged inconclusive [E-28 §3.6]. In DGM, ablating self-improvement lowered final scores on SWE-bench and Polyglot from 50.0 and 38.0 to 39.0 and 28.0, showing that letting the improved agent carry out the next round of edits adds value without demonstrating generational acceleration [M-03]. The experiments in §4 establish the corresponding result for model weights: across three generations, no acceleration occurs.

Macro-level models from Epoch and Forethought formalize R&D acceleration through the return parameter $r$, where $r > 1$ implies progressively shorter doubling times [T-16, T-17]. Empirical estimates derived from human R&D history place $r$ at a median of 1.4 (0.8 to 2.4) for image recognition and roughly 0.83 for computer chess, while Forethought estimates $r$ between 0.5 and 2 when compute is held constant [T-17, T-16]. Because this experiment measures generational differences in improvement speed under a fixed per-run budget, its scores do not map directly onto $r$, which is defined over cumulative input. Moreover, theoretical models by Davidson and Houlden predict diminishing returns as a system nears its capability ceiling, meaning that the absence of acceleration across three generations of a 9B model does not rule out $r > 1$ over longer horizons or larger scales (an inference) [T-18].

### Summary

Within the literature reviewed, this report establishes the first controlled baseline for measuring second-order improvement directly at the level of model weights. By placing fine-tuned successor checkpoints back into the improver role against A/A noise calibration, rejected-trajectory placebos, and matched-volume or random-recipe controls, the experiments demonstrate four concrete limits: free-text experience notes slow improvement by prematurely narrowing search, trajectory fine-tuning shifts surface calling habits without teaching research judgment, fixed self-training loops begin to degrade by the third generation despite data accumulation, and delegating the next update recipe to a fine-tuned successor significantly degrades performance [`decisions/decisions_1235.json`, `lit_rsi_academic.md` §6].
