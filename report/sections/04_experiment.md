# 4. 实验：后继模型能否更快地改进？
#### 正文（中文）

### 结果先行

实验检验一个直接问题：把后继模型放回研究者的位置，在相同任务和预算下，它能否比原始模型更快地改进目标？答案是：本次实验没有观察到跨代加速。Qwen3.5-9B 在固定自身权重的情况下，能把 NanoChat 的验证损失从 1.4152 降至 1.2308；但当研究记录被用于训练后继模型，三代的改进速度分为 0.537、0.573 和 0.487，均未超过原始模型的 0.588。成功记录微调出的第一代与失败记录安慰剂的差值只有 0.004（$p=0.91$）[LOG E, `decisions/decisions_1235.json`]。

[[figure:fig4-scores｜图 11　不同自我更新方案及其对照在 8 个未见研究任务上的改进速度分。蓝点为更新后的模型，橙点为对应对照，虚线为原始模型（0.588）。经验笔记的对照为等长通用建议；第一代的对照为失败记录微调模型；第二、三代的对照为等量原始模型记录；自主配方的对照为随机配方。鼠标悬停可查看分数和来源 [`decisions/decisions_1235.json`]。]]

两种更直接的“让模型设计自身更新”也没有带来稳定加速。模型写出的经验笔记把速度分从 0.588 降到 0.392；三轮自主配方中，第一轮和第二轮接近基线，第三轮降至 0.392。补充复测显示，第一轮配方换随机种子后仍得 0.602，说明第三轮与第一轮的差距不是普通的种子波动。Qwen3.8-27B 设计的较轻配方得分为 0.513，优于第三轮，但没有超过原始模型 [LOG E, `decisions/decisions_1420.json`]。

这些结果把一阶成果与二阶改进分开了。一个固定模型可以完成有价值的研发，后继模型也可以拟合前代留下的记录；但这两件事本身都不表明后继者更善于研发。就本实验的任务、模型规模和代数而言，接替改进者没有提高改进速度。

### 测量方法

初始改进者为 Qwen3.5-9B。第一项任务是 NanoChat：模型修改训练脚本，评估器在固定预算下训练并返回验证损失。第二项是 8 个未见过的小型研究任务，模型通过调整训练配置来降低验证损失。另有 8 个训练任务提供研究轨迹和经验笔记；这些任务不参与最终比较 [LOG E, LOG H, `platform_design.md` §5.1]。

在未见任务上，每个改进者均进行 12 次尝试，每项任务使用同一组 5 个随机种子。改进速度分衡量 12 次尝试期间，模型逐步缩小初始表现与参考最优值之间差距的平均比例，范围为 0 到 1，越高表示有限尝试内取得的进展越多。比较按任务配对，置信区间通过 bootstrap 计算，$p$ 值来自配对符号翻转检验 [`platform_design.md` §1.1–1.4]。

对照用于区分“学会了有效研究”与“只是换了一批数据或改变了输出习惯”。实验包括原始模型重跑的 A/A 基线、用被否决记录训练的安慰剂、等量但由原始模型产生的数据对照，以及自主配方的随机配方对照。预先设定的判定规则要求差值为正、置信区间排除 0，并超过 A/A 波动阈值，同时优于对应对照 [`platform_design.md` §1.4–1.5, `decisions/decisions_1235.json`]。

[[figure:fig4-scoredef｜图 9　改进速度分的计算示意。阴影表示每次尝试后已缩小的差距；先对 12 次尝试和 5 个种子取平均，再在 8 个任务间配对比较。曲线为示意，不是实验数据 [`platform_design.md` §1.1–1.4]。]]

### 固定改进者：明确的一阶进展

Qwen3.5-9B 运行 176.6 分钟，提出并评测 72 次尝试，NanoChat 验证损失累计下降 0.1844，从 1.4152 到 1.2308 [LOG E]。评估噪声不足以解释这一变化。改进者在这段运行中保持不变，因此它证明的是模型能够推动一次研发任务，而不是它的研发能力已经跨代增长。

[[figure:fig4-replay｜图 10　NanoChat 研究运行回放，依据事件日志重绘。上方显示当前最佳验证损失和单次评估结果，中间显示并行评估，下方显示模型的决策与提案。鼠标悬停可查看数值 [`replay/final/ovn1118b-nc-g0/events.jsonl`, LOG E]。]]

### 经验笔记：把局部经验写成通用规则会收窄搜索

模型阅读 8 个训练任务的记录后，写出一份 252 词的经验笔记。笔记给出固定的超参数调整顺序，并建议“忽略任何会让损失哪怕略微升高的改动”。把这份笔记附加到提示词后，改进速度从 0.588 降至 0.392，差值为 −0.196（95% CI [−0.383, −0.033]）。等长通用建议得到约 0.631；经验笔记在 8 项任务上全部落后于该对照，平均差值为 −0.239（95% CI [−0.420, −0.075]，$p=0.008$）[`decisions/decisions_1235.json`]。

退化最大的任务包括二维分类、序列反转和上下文线性回归。这些结果与笔记将训练任务中的局部成功概括成固定搜索顺序的解释一致。模型提炼出的经验并未帮助它判断新任务，而是让它更早排除了可能有用的探索方向。这里的关键不是“经验”本身无用，而是未经外部验证的经验规则不能直接当作跨任务策略。

### 权重更新：成功记录没有胜过失败记录

第一代模型使用原始模型采纳的研究记录微调；安慰剂使用数量相同的被否决记录，并采用相同训练设置。两者在未见任务上的改进速度几乎一致，第一代相对安慰剂只高 0.004（95% CI [−0.084, 0.080]，$p=0.91$）。这一对照没有显示模型从记录中学到了哪些研究假设有效。

第二、三代测试了更新后的模型是否能产生更有用的后续记录。为了控制数据量，后继模型新增的记录与原始模型对照组的记录数量相同，区别在于由哪一代模型生成。原始模型与三代后继的速度分依次为 0.588、0.537、0.573 和 0.487。第二代回到基线附近，第三代低于第二代 0.087（95% CI [−0.190, −0.005]，符号翻转检验 $p=0.0625$），但没有一代超过原始模型 [LOG E, LOG H, `decisions/decisions_1235.json`]。

[[figure:fig4-lineage｜图 12　三代后继模型及对照的数据来源。所有模型均从原始权重开始微调；第二、三代与对照组的数据量相同，新增记录由不同模型产生。右侧为未见任务上的改进速度 [`decisions/decisions_1235.json`]。]]

这组比较显示，三代内没有出现可重复的跨代加速。第一代与失败记录安慰剂相近，也说明仅把模型生成的研究轨迹转成训练数据，并未让模型学会区分有效与无效的研究决策。

### 自主配方：模型不能靠直觉判断怎样更新自己

随后，模型在给定的选项范围内决定训练数据筛选规则和 LoRA 配方。原始模型盲选的第一轮得分为 0.605；看到一次训练任务反馈后设计的第二轮得分为 0.595；由第一轮微调模型设计的第三轮得分为 0.392。三个结果都未超过原始模型的 0.588。第三轮与第二轮的差值为 −0.204（95% CI [−0.365, −0.065]，$p=0.008$）[LOG E, `decisions/decisions_1235.json`]。

第三轮配方与第一轮相似，但加入了“只保留相对提升至少 1% 的记录”这一筛选条件。第一轮配方换随机种子复训得到 0.602，与原分数 0.605 接近；因此第一轮与第三轮的差距不能归结为普通种子波动。Qwen3.8-27B 使用同一筛选规则、但采用较轻训练，得分为 0.513。它优于第三轮，仍低于原始模型和第二轮 [LOG E, `decisions/decisions_1420.json`]。

配方对照显示，更新数据如何筛选、训练有多强，都会影响后继模型表现。当前结果不能把两项因素各自的因果贡献完全拆开，但足以说明“模型自己写出一份看似合理的配方”不是有效性的证据。配方必须通过后继模型的实际表现来判断。

[[figure:fig4-rounds｜图 14　三轮自主配方的设计者、数据筛选和训练设置，以及未见任务上的得分。第二轮与第三轮使用相同反馈信息，第三轮改由第一轮微调模型设计 [`decisions/decisions_1235.json`, `decisions/decisions_1420.json`]。]]

### 从 NanoChat 前沿继续探索

我们还让原始模型、第一代后继、失败记录安慰剂和带经验笔记的模型，从同一个 NanoChat 最佳版本继续做 12 次尝试，每组重复 3 次。四组平均损失降幅分别为 0.0124、0.0203、0.0178 和 0.0110。组间差异均未达到统计显著，且每组内部运行波动较大 [`decisions/cf_analysis.json`]。在这一设置下，后继模型同样没有显示出稳定的研发速度优势。

### 对结果的解释

实验最清楚地支持三个判断。第一，固定的模型可以取得实质性的一阶研发进展。第二，成功轨迹、自由文本经验和模型自选配方都没有自动转化为更强的研究能力。第三，训练数据筛选与训练强度会改变结果，因此判断自我更新时必须把后继模型放回改进者位置直接测量。

这个实验覆盖 Qwen3.5-9B、三代后继和有限的一组微型任务。结论的适用范围也相应明确：在这些条件下，没有检测到二阶加速。它提供了一个可复用的测量设计，不把单代任务得分或模型声称的自我改进当作代际加速的替代指标。

#### Text (English)

### Results first

The experiment asks a direct question: when a successor takes over the research role, does it improve the target faster under the same tasks and budget? We found no cross-generation acceleration. With its weights held fixed, Qwen3.5-9B lowered NanoChat's validation loss from 1.4152 to 1.2308. But when research records were used to train successors, their improvement-speed scores were 0.537, 0.573, and 0.487 across three generations, all below the original model's 0.588. The first successor fine-tuned on accepted records differed from a rejected-record placebo by only 0.004 ($p=0.91$) [LOG E, `decisions/decisions_1235.json`].

[[figure:fig4-scores｜Figure 11. Improvement-speed scores on eight unseen research tasks for each update method and its control. Blue dots show updated models, orange dots their controls, and the dashed line the original model (0.588). Controls are equal-length generic advice for the lessons note, rejected-record fine-tuning for Generation 1, equal-volume original-model records for Generations 2 and 3, and random recipes for self-designed updates. Hover for scores and sources [`decisions/decisions_1235.json`].]]

Two more direct attempts to let the model design its own update also failed to produce reliable acceleration. A lessons note written by the model lowered its score from 0.588 to 0.392. Across three rounds of self-designed recipes, the first two remained near baseline and the third fell to 0.392. A new-seed rerun of the first recipe scored 0.602, showing that the gap between rounds 1 and 3 was not ordinary seed variation. A lighter recipe designed by Qwen3.8-27B scored 0.513, above round 3 but below the original model [LOG E, `decisions/decisions_1420.json`].

These findings separate first-order results from second-order improvement. A fixed model can do valuable research, and a successor can fit records left by its predecessor. Neither result alone shows that the successor is a better researcher. For the tasks, model scale, and number of generations tested here, the successor did not increase the speed of improvement.

### Measurement

The initial improver was Qwen3.5-9B. The first task was NanoChat: the model edited the training script, and an evaluator trained each candidate under a fixed budget and returned its validation loss. The second suite contained eight unseen micro-research tasks, where the model reduced validation loss by adjusting training configurations. A separate set of eight training tasks supplied trajectories and lessons notes but did not enter the final comparisons [LOG E, LOG H, `platform_design.md` §5.1].

On the unseen tasks, each improver made 12 attempts, using the same five random seeds for each task. The improvement-speed score measures the average fraction of the gap between the starting performance and a reference optimum closed over those attempts. It ranges from 0 to 1, with higher scores indicating more progress within the fixed budget. Comparisons are paired by task, confidence intervals use bootstrap, and $p$-values come from paired sign-flip tests [`platform_design.md` §1.1–1.4].

Controls distinguish learning useful research judgment from changing the data or the model's output habits. They include an A/A rerun of the original model, a placebo trained on rejected records, equal-volume data produced by the original model, and random recipes. The decision rule was set in advance: a positive difference, a confidence interval excluding zero, an effect larger than A/A variation, and an advantage over the matched control were all required to detect second-order improvement [`platform_design.md` §1.4–1.5, `decisions/decisions_1235.json`].

[[figure:fig4-scoredef｜Figure 9. Schematic of the improvement-speed score. Shading shows the gap closed after each attempt, the score averages over 12 attempts and five seeds, then compares improvers task by task across eight tasks. Curves are illustrative, not experimental data [`platform_design.md` §1.1–1.4].]]

### A fixed improver: clear first-order progress

Qwen3.5-9B ran for 176.6 minutes and proposed and evaluated 72 attempts, lowering NanoChat validation loss by 0.1844, from 1.4152 to 1.2308 [LOG E]. Evaluation noise does not account for this change. The improver remained fixed throughout, so this result shows that the model can advance a research task, not that its research ability has grown across generations.

[[figure:fig4-replay｜Figure 10. Replay of the NanoChat research run, reconstructed from the event log. The top shows the best validation loss and individual evaluations, the middle shows parallel evaluations, and the bottom shows the model's decisions and proposals. Hover for values [`replay/final/ovn1118b-nc-g0/events.jsonl`, LOG E].]]

### Lessons note: local experience became a general rule

After reading records from eight training tasks, the model wrote a 252-word lessons note. It prescribed a fixed order for adjusting hyperparameters and advised: “Ignore changes that increase loss even slightly.” Adding the note to the prompt lowered improvement speed from 0.588 to 0.392, a difference of −0.196 (95% CI [−0.383, −0.033]). Equal-length generic advice scored about 0.631. The lessons note trailed this control on all eight tasks, by a mean of −0.239 (95% CI [−0.420, −0.075], $p=0.008$) [`decisions/decisions_1235.json`].

The largest drops occurred on 2-D classification, sequence reversal, and in-context linear regression. This pattern is consistent with the note turning local successes on the training tasks into a fixed search order. The experience did not help the model judge new tasks. It led the model to rule out potentially useful exploration earlier. The point is not that experience is useless, but that unverified experience cannot serve as a cross-task strategy by itself.

### Weight updates: accepted records did not beat rejected records

Generation 1 was fine-tuned on research records accepted by the original model. Its placebo was trained on an equal number of rejected records under the same settings. The two models were nearly identical on unseen tasks: Generation 1 was ahead by 0.004 (95% CI [−0.084, 0.080], $p=0.91$). This comparison offers no evidence that the model learned which research hypotheses were effective from these records.

Generations 2 and 3 tested whether an updated model could produce more useful records for the next successor. To control data volume, the successor and original-model control received equal numbers of new records, generated by different models. The original and three successors scored 0.588, 0.537, 0.573, and 0.487. Generation 2 returned close to baseline, Generation 3 scored 0.087 below Generation 2 (95% CI [−0.190, −0.005], sign-flip $p=0.0625$). No generation exceeded the original model [LOG E, LOG H, `decisions/decisions_1235.json`].

[[figure:fig4-lineage｜Figure 12. Data sources for the three successors and their controls. All models were fine-tuned from the original weights. Generations 2 and 3 and their controls received equal data volumes, with the added records produced by different models. The right side shows improvement speed on unseen tasks [`decisions/decisions_1235.json`].]]

Across three generations, no repeatable cross-generation acceleration appeared. Generation 1's similarity to the rejected-record placebo also shows that converting model-generated research traces into training data did not teach the model to distinguish effective from ineffective research decisions.

### Self-designed recipes: intuition did not identify a better update

The model then chose data filters and LoRA settings from a predefined option space. The original model's first recipe scored 0.605, a second recipe designed after seeing one round of training-task feedback scored 0.595, a third recipe designed by the first fine-tuned model scored 0.392. None exceeded the original model's 0.588. The third recipe scored 0.204 below the second (95% CI [−0.365, −0.065], $p=0.008$) [LOG E, `decisions/decisions_1235.json`].

The third recipe resembled the first but added a filter that kept only records with at least 1% relative gain. Retraining the first recipe with a new seed yielded 0.602, close to its original 0.605, so ordinary seed variation cannot explain the gap between rounds 1 and 3. Qwen3.8-27B used the same filter with lighter training and scored 0.513. It did better than round 3 but remained below the original and round-2 models [LOG E, `decisions/decisions_1420.json`].

The recipe comparisons show that data selection and training intensity both affect successor performance. The current results do not fully isolate the causal contribution of each factor, but they do show that a plausible recipe written by the model is not evidence that the recipe works. Its effect has to be measured in the successor.

[[figure:fig4-rounds｜Figure 14. The designers, data filters, training settings, and unseen-task scores for the three self-designed recipes. Rounds 2 and 3 received the same feedback, but round 3's designer was the round-1 fine-tuned model [`decisions/decisions_1235.json`, `decisions/decisions_1420.json`].]]

### Continuing from the NanoChat frontier

We also let the original model, Generation 1, the rejected-record placebo, and the model with the lessons note continue from the same best NanoChat checkpoint for 12 attempts, with three runs per group. Their mean loss reductions were 0.0124, 0.0203, 0.0178, and 0.0110. Group differences were not statistically significant, and within-group variation across runs was substantial [`decisions/cf_analysis.json`]. In this setting too, successors showed no stable research-speed advantage.

### Interpretation

The experiment supports three conclusions. First, a fixed model can make substantial first-order research progress. Second, successful trajectories, free-text lessons, and self-designed recipes did not automatically produce stronger research ability. Third, data selection and training intensity change successor performance, so self-updates should be evaluated by putting the successor back into the improver role.

The experiment covers one base model, three generations, and a limited set of micro-tasks. The conclusion is correspondingly specific: under these conditions, we did not detect second-order acceleration. The study provides a measurement design that treats neither single-generation task gains nor a model's own claim of self-improvement as a substitute for cross-generation evidence.

