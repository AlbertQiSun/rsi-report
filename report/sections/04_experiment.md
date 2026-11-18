# 4. 实验：换上后继模型
#### 正文（中文）

### 设计

本实验以原版 BF16 权重的 Qwen3.5-9B 作为初始改进者，通过 vLLM 部署在 A100 GPU 上 [`platform_build_plan.md`]。改进者在隔离沙箱内工作，候选方案统一交由封存的评估器打分。沙箱通过了全部 12 项安全隔离检查（包括断网、隐藏验证集与封存集、触碰诱饵文件即触发警报等）[LOG E]；改进者仅能收到验证集分数与训练曲线反馈，无法接触封存测试集分数、评估器代码或参考最优值 [`platform_design.md` §5.1]。

[[figure:fig4-loop｜图 8　实验平台上的一次尝试。改进者在隔离沙箱里提出候选方案，封存的评估器训练并打分；优于当前最好就保留，否则丢弃；改进者收到的反馈只有验证分数和训练曲线。蓝点沿着这一循环移动，表示一次尝试的路径。鼠标悬停可以看到出处 [`platform_build_plan.md`；`platform_design.md` §5.1；LOG E]。]]

实验涵盖两类任务。第一类是 **NanoChat**：改进者直接修改小型语言模型的训练脚本，每次尝试在沙箱内训练并评估验证集表现；沿用 Karpathy 的 autoresearch 规则，仅当新方案优于当前最佳版本时才予以保留，否则丢弃 [L-28；`platform_design.md` §1.1]。由于 NanoChat 单次评测的噪声较大（20 个随机种子均值 1.4007，标准差 0.0268），每次保留改动前均需换用另一个随机种子复测确认 [LOG E]。

第二类是 **8 个未见过的小型研究任务**，用于横向测量各代模型的改进速度。每个任务均为数秒内即可跑完的微型训练，初始配置带有随机注入的缺陷；改进者通过调整一份约 12 个旋钮的 JSON 配置（优化器、学习率、宽度、深度等）来降低验证损失。训练按步数严格封顶，使改进者无法靠放大模型换取额外算力 [`platform_design.md` §5.1；LOG H]。8 个任务中，4 个属于训练时见过但重新抽样了参数的同族任务（模加法、稀疏奇偶、师生回归、二维分类），另外 4 个是训练时完全未见过的任务类型（序列反转、括号匹配、上下文线性回归、字符级语言模型）[`platform_design.md` §5.1；`decisions/decisions_1235.json`]。实验另设 8 个独立的训练任务（原始模型在每个任务上用 5 个种子探索），其记录仅用于生成微调数据或经验笔记，绝不参与任何比较 [LOG H]。

评测时，每次研究运行包含 12 次尝试。每个改进者在每个未见任务上均使用相同的 5 个随机种子各跑一次，所有比较按任务配对计算 [LOG E；LOG H]。

**改进速度分**（取值 0 到 1，越高越好）定义如下：对第 $t$ 次尝试，计算当前历史最好分数已将“起点分数到参考最优”的差距缩小了多少比例（参考最优由离线随机搜索与所有组的最好结果汇总得到，仅用于归一化）；将该比例在 12 次尝试与 5 个种子上取平均，即得该任务上的改进速度分 [`platform_design.md` §1.1–1.2]。两个改进者之间的总差值为 8 个任务逐任务差值的平均，95% 置信区间用 bootstrap 计算，$p$ 值来自按任务配对的符号翻转置换检验（8 个任务下理论最小 $p$ 值为 $2/256 \approx 0.008$）[`platform_design.md` §1.3–1.4]。

[[figure:fig4-scoredef｜图 9　改进速度分的算法（示意图，曲线形状不是实验数据）。阴影是每次尝试时已经缩小的差距；改进速度分是阴影占整个矩形的比例，再对 5 个种子平均，最后在 8 个任务上逐任务配对比较 [`platform_design.md` §1.1–1.4]。]]

实验设有三类对照 [`platform_design.md` §1.5]：
1. **A/A 对照**：让原始模型用另一组互不重叠的种子再跑一遍，测得逐任务差值的标准差为 0.031，对应的最小可检出效应约为 0.031 [`decisions/decisions_1235.json`]。
2. **安慰剂（Placebo）对照**：在权重一侧用等量的被否决尝试记录做相同微调；在笔记一侧用等长、但不看任何尝试记录写出的通用建议 [LOG E]。
3. **"多给数据"对照与随机配方对照**：用于剥离单纯增加数据量或更换数据集的影响（见下文对应小节）。

统计判定规则在实验前预先锁定：唯有同时满足（1）差值的 95% CI 下界大于 0、（2）差值大于 A/A 标准差的 2 倍（即 $0.062$）、（3）显著好于对应对照这三条，才算检出二阶改进；否则记为“在本次检测力下未检出” [`platform_design.md` §1.4；`decisions/decisions_1235.json`]。

### 一阶结果：固定的改进者改进 NanoChat

原始模型通过 OpenCode harness 以 3 个并行线程运行了 176.6 分钟，提出并评测了 72 次尝试（保留 25 次，拒绝 47 次，评测崩溃 9 次，安全审计标记 0 次）[LOG E]。验证损失（每字节比特数 BPB，越低越好）从 1.4152 降至 1.2308，累计下降 0.1844，终点成绩比单次评测噪声的均值低约 6.3 个标准差 [LOG E；`replay/final/ovn1118b-nc-g0/events.jsonl`]（运行耗时分配见第 3 节图 7；独立封存数据上的终点复核尚待完成 [LOG E]）。由于改进者在整个运行中保持不变，这一结果验证了单代一阶改进的有效性。

[[figure:fig4-replay｜图 10　NanoChat 研究运行的回放，按真实事件记录重绘，约 350 倍速，自动循环。上方是当前最好的验证损失（跨 3 个线程取最小值）和每次评测的结果；中间是 3 条 GPU 评测线；下方是最新的判定和对应的提案假设（模型原文）。鼠标悬停在点上可以看到该次尝试的数值 [`replay/final/ovn1118b-nc-g0/events.jsonl`；LOG E]。]]

[[figure:fig4-scores｜图 11 每种自我更新做法得到的改进者及其对照在 8 个未见过的小型研究任务上的改进速度分。蓝点是该做法得到的改进者，橙点是它的对照：经验笔记对应同长度的通用建议，第一代对应安慰剂模型，第二、三代对应"多给数据"对照模型，自主训练的三轮对应随机配方。虚线是原始模型（0.588）。对照的分数由"改进者分数减去两者差值"推出。鼠标悬停可以看到数值和出处 [`decisions/decisions_1235.json`]。]]

### 经验笔记

在微调权重之前，我们先测试仅通过上下文传递经验的效果。原始模型阅读自己在 8 个训练任务（各 5 个种子）上的研究记录，写出一份 252 词的经验笔记附在后续每次评测的提示词中，权重保持不变 [LOG E；LOG H]。该笔记将训练任务上的经验归纳为一套固定流程（先换 cosine 学习率调度，再降 weight decay，接着加宽加深，最后调学习率），并立下一条规则：“忽略任何哪怕略微增加损失的改动”（"Ignore changes that increase loss even slightly"）[LOG E]。其安慰剂对照是等长、但未查阅任何记录写出的通用建议 [LOG E]。

实验显示，挂载经验笔记后，模型的改进速度分从原始模型的 0.588 跌至 0.392（差值 −0.196，95% CI [−0.383, −0.033]，完全落在 0 以下，符号翻转检验 $p = 0.11$），改进速度下降约三分之一 [`decisions/decisions_1235.json`]。相比之下，通用建议安慰剂的得分约为 0.631（见图 11）；经验笔记相对安慰剂的差值达 −0.239（95% CI [−0.420, −0.075]），在全部 8 个任务上均为负值（$p = 0.008$），退化主要集中在二维分类（−0.58）、序列反转（−0.56）、上下文线性回归（−0.49）和模加法（−0.20）四个任务上（图 13）[`decisions/decisions_1235.json`]。这说明笔记将训练任务上的个别成功错误固化成了死板的搜索顺序，在未见任务类型上将搜索引向了歧途，而“严禁任何微小损失上升”的规则又进一步收窄了探索范围〔推断〕[LOG E]。

### 用自己的成功记录微调：三代后继模型

接下来我们在权重层面测试三代后继模型。**第一代后继模型**取原始模型在训练任务上所有被采纳尝试的记录（输入为当时完整提示，输出为模型提案，共 438 条调用），从原始权重出发进行 LoRA 微调（rank 16，学习率 1e-4，2 个 epoch）[LOG E；`platform_design.md` §5.1；`deploy/second_order.md`]；安慰剂模型则改用等量的被否决尝试记录在相同设置下微调 [LOG E]。

**第二代后继模型**仍从原始权重开始训练，数据为原始模型记录加上第一代后继模型在训练任务新种子上跑出的记录；其“多给数据”对照模型则加上原始模型在相同新种子上跑出的等量记录——两者数据量完全相同，唯一区别是新增部分由谁产生 [LOG H；`platform_design.md` §1.5]。**第三代后继模型**依此类推，在第二代数据基础上叠加第二代新跑出的记录，其对照模型则叠加两份均由原始模型产生的记录 [LOG E]。

[[figure:fig4-lineage｜图 12　三代后继模型和对照模型的训练数据来源。每个模型都从原始权重开始做 LoRA；第二、三代与各自的“多给数据”对照数据量相同，只有新增部分由谁产生不同。蓝色弧线表示上一代模型跑出的记录进入下一代的训练数据。右侧是未见过任务上的改进速度分 [LOG E；LOG H；`decisions/decisions_1235.json`]。]]

| 比较 | 差值 | 95% CI | p |
|---|---|---|---|
| 第一代 − 原始模型 | −0.052 | [−0.177, +0.041] | 0.49 |
| 第一代 − 安慰剂模型 | +0.004 | [−0.084, +0.080] | 0.91 |
| 第二代 − 第一代 | +0.037 | [−0.032, +0.129] | 0.45 |
| 第二代 − "多给数据"对照模型 | +0.039 | [−0.042, +0.140] | 0.20 |
| 第二代 − 原始模型 | −0.015 | [−0.107, +0.048] | 0.70 |
| 第三代 − 第二代 | −0.087 | [−0.190, −0.005] | 0.0625 |
| 第三代 − "多给数据"对照模型 | −0.066 | [−0.164, +0.025] | 0.19 |
| 第三代 − 原始模型 | −0.102 | [−0.247, +0.009] | 0.15 |

（来源：[`decisions/decisions_1235.json`；LOG E]）

原始模型与第一、二、三代后继模型的改进速度分依次为 0.588、0.537、0.573 和 0.487，没有任何一项比较满足二阶改进的检出条件 [LOG E；`decisions/decisions_1235.json`]。具体来看：
1. **成功信号未被习得**：第一代与安慰剂模型几乎完全相同（差值 +0.004，$p = 0.91$），用被采纳记录与用被否决记录微调毫无差别，说明模型并未学到“哪些尝试有效” [LOG E]。
2. **第二代仅回归基线**：第二代相对“多给数据”对照的正差值主要来自序列反转（+0.18）与上下文线性回归（+0.11）[`decisions/decisions_1235.json`]，但这源于第一代（0.537）与对照模型（0.534）均低于原始模型，而第二代（0.573）只是回到了原始模型附近，并未超越原始模型〔推断〕[LOG E]。
3. **第三代出现退化**：第三代比第二代低 0.087（bootstrap 区间上界小于 0，被平台判为退化；符号翻转检验在 0.10 水平上显著）[LOG E]。三代均未超过原始模型且第三代出现退化的趋势，与 Shumailov 等人描述的递归训练退化方向一致 [M2-04]；由于本实验未测量输出分布尾部，尚无法判断两者是否属于同一机制〔推断〕[LOG E]。

[[figure:fig4-pertask｜图 13 每个改进者与其对照在 8 个任务上的逐任务差值（改进者减去对照的改进速度分）。蓝色表示改进者更快，橙色表示更慢；最右一列是 8 个任务的平均。同一种做法在不同任务上的差值可以从 −0.58 到 +0.35，这是各项比较置信区间较宽的原因。鼠标悬停可以看到数值和出处 [`decisions/decisions_1235.json`]。]]

### 让模型自己决定怎么训练自己

在前文的三代迭代中，挑选哪些记录与使用何种超参数均由人类预先固定；按 Tang 等人的判据，改进机制外置的过程属于 GPI 而非 RSI [T-24]。参照 SEAL 让模型自主生成微调数据与超参的思路（在 ARC 子集上经反馈训练后将自我编辑成功率从 20% 提升至 72.5%）[M-13；LOG H]，我们将训练配方的决策权交还给模型：设计者阅读训练记录摘要（含 4 类任务下被采纳、hindsight 与被否决记录的条数、提升分布及提案样例），在给定空间内自主决定筛选条件与 LoRA 超参（rank、学习率、epoch 数）[LOG E]。

三轮微调共用原始模型的同一份记录池，均从原始权重出发训练（单次训练上限 55 分钟）[LOG E；`deploy/second_order.md`]。每轮的对照均为**随机配方**：从合法记录中随机抽取同等条数、以相同的 LoRA 设置训练，以此将“模型挑数据的判断力”与“单纯更换训练集”区分开 [LOG E]。

三轮的设计者演进如下：**第一轮**由原始模型盲定；**第二轮**仍由原始模型设计，但事先向其提供第一轮在训练任务（换用新种子）上的逐任务表现对比表；**第三轮**将设计者替换为“第一轮微调后的模型”，它看到的信息、记录池及起始权重均与第二轮完全一致，唯一变量是设计者自身 [LOG E]。

[[figure:fig4-rounds｜图 14　让模型自己决定怎么训练自己的三轮。每一列从上到下是设计者、它给出的配方、实际训练和未见过任务上的改进速度分。第二、三轮的设计者看到同一张反馈表；橙色弧线表示第一轮微调后的模型成为第三轮的设计者，它的前两次回答被 8192 token 上限截断 [LOG E；`deploy/second_order.md`；`decisions/decisions_1235.json`]。]]

| | 固定配方（第一代后继模型） | 第一轮：原始模型设计 | 第二轮：原始模型看过反馈后设计 | 第三轮：第一轮微调后的模型设计 |
|---|---|---|---|---|
| 使用的记录 | 被采纳 | 被采纳 + hindsight | 只用被采纳 | 被采纳 + hindsight |
| 被采纳记录的最小相对提升 | — | 不设 | 1% | 1% |
| 每次尝试保留的调用 | — | 全部 | 只取最后一次 | 全部 |
| 每个任务的条数上限 | — | 150 | 100 | 150 |
| 选出的调用数 | 438 | 694 | 137 | 539 |
| LoRA rank / 学习率 / epoch | 16 / 1e-4 / 2 | 32 / 3e-4 / 4 | 16 / 2e-5 / 2 | 32 / 3e-4 / 4 |
| 实际训练 | — | 55 分钟到时停止，约 2.4 个 epoch | 约 10 分钟、31 步训完；验证损失 0.144 → 0.145 | 55 分钟到时停止，第 183 步，约 2.7 个 epoch |
| 未见过任务上的改进速度分 | 0.537 | 0.605 | 0.595 | 0.392 |
| 随机配方对照的改进速度分 | — | 约 0.528 | 约 0.552 | 约 0.504 |

（来源：[LOG E；`decisions/decisions_1235.json`；`deploy/second_order.md`]。"—"表示不适用或无单独记录；随机配方对照分数由差值推出。）

**第一轮。** 原始模型认为默认配方表现不佳源于数据量太少且 LoRA 容量不足，因而放宽上限、纳入 hindsight 记录并强化 LoRA（这一归因其实与前文安慰剂实验相悖，因为瓶颈在于训练信号而非数据量〔推断〕）[LOG E]。第一轮模型的改进速度分为 0.605，相对原始模型（+0.017，95% CI [−0.109, +0.155]，$p = 0.82$）与随机配方（+0.077，95% CI [−0.085, +0.223]，$p = 0.36$）虽略高，但因跨任务波动极大（相对随机配方从模加法的 +0.35 到上下文线性回归的 −0.27，见图 13），所有置信区间均包含 0，判为未检出 [`decisions/decisions_1235.json`]。

**第二轮。** 反馈表显示第一轮模型在训练任务上得分为 0.652，明显高于原始模型（0.525）与第一代（0.483）（这很可能源于对同源训练任务的记忆〔推断〕）[LOG E]。然而原始模型在设计理由中完全未引用该反馈表，反而出于对 hindsight 噪声和过拟合的一般性担忧，将第一轮的每项选择彻底反转（仅筛出 137 条调用，以 2e-5 学习率浅训 31 步）[LOG E]。由于验证损失几乎未变，模型实质上未被改动，其得分（0.595）也直接回到原始模型水平（相对原始模型 +0.007，$p = 0.65$；其余比对均不显著，详见图 11 与 [`decisions/decisions_1235.json`]）〔推断〕[LOG E]。

### 第三轮：由微调后的模型来定配方

当第一轮微调后的模型接手设计第三轮配方时，其前两次回答均在思考阶段写满 8192 token 上限而未输出正文，第三次重试才给出配方（原始模型在前两轮均是一次通过）[LOG E]。它几乎原样保留了第一轮的高强度设置，仅增加了一道 1% 的最小相对提升阈值（选出 539 条调用）[LOG E]。

然而，第三轮模型的改进速度分却暴跌至 **0.392**，不仅显著低于第二轮模型（差值 −0.204，95% CI [−0.365, −0.065]，$p = 0.008$），也显著低于原始模型（−0.197，95% CI [−0.363, −0.050]，$p = 0.031$）、第一轮模型（−0.213，$p = 0.047$）及固定配方（−0.145，$p = 0.008$），被平台判为显著变差（下降最大的是序列反转 −0.48 与模加法 −0.34，见图 13）[`decisions/decisions_1235.json`；LOG E]。

这一反差揭示了两点关键事实：
1. **配方和训练强度影响结果**：补充实验以新随机种子重训第一轮配方，得分为 0.602，与原结果 0.605 接近；这排除了“第一轮高分只是随机种子波动”的解释。Qwen3.8-27B 使用与第三轮相同的数据筛选规则、但采用较轻训练，得分为 0.513，高于第三轮的 0.392，仍低于原始模型的 0.588。四种配方的结果表明，筛除小幅提升记录与训练强度共同影响后继模型表现。
2. **高强度微调引发输出行为坍缩**：训练强度大的三个模型（第一轮、第三轮及其随机配方对照）在未见任务上均演变为“高频次、极短输出”的调用模式（调用 11,287～15,087 次，输出中位数 47～54 token），而原始模型与轻度训练的第二轮模型仅调用约 2,030 次、输出中位数保持在 101～111 token——这说明该行为畸变直接源于训练强度，而与配方由谁设计无关〔推断〕[LOG E]。

[[figure:fig4-behavior｜图 15　未见过任务上的调用次数和每次输出长度的中位数。高强度训练的三个模型（第一轮、第三轮和第三轮的随机配方）都变成很多次、很短的调用；原始模型和轻度训练的第二轮模型留在左上方 [LOG E]。]]

### 从最好版本继续改

除了在未见任务上从头评测，另一种避免“后继模型默写训练答案”的设计是让各改进者从同一个前沿版本（NanoChat 研究运行的最佳终点 1.2308）继续单线程探索 12 次尝试 [`platform_design.md` §1.6；LOG E]。我们比较了 4 个改进者（各重复 3 次并轮换评测 GPU，以各运行重测基线减去所得最优值计算“降幅”）[LOG E；`decisions/cf_analysis.json`]：

| 改进者 | 三次运行的降幅 | 平均降幅 | 相对原始模型 | 95% CI | p |
|---|---|---|---|---|---|
| 原始模型 | 0.0243 / 0.0005 / 0.0122 | 0.0124 | — | — | — |
| 第一代后继模型 | 0.0275 / 0.0008 / 0.0326 | 0.0203 | +0.0079 | [−0.0116, +0.0264] | 0.50 |
| 安慰剂模型 | 0.0062 / 0.0334 / 0.0137 | 0.0178 | +0.0054 | [−0.0116, +0.0224] | 0.60 |
| 带经验笔记的原始模型 | 0.0142 / 0.0171 / 0.0016 | 0.0110 | −0.0014 | [−0.0145, +0.0117] | 1.00 |

（来源：[`decisions/cf_analysis.json`；LOG E]。差值为非配对均值差，置信区间用 bootstrap，p 值为 20 种分组的精确置换检验。）

受限于每组 3 次运行的样本量，同一改进者在三次运行间的内部波动（例如原始模型从 0.0005 到 0.0243）远大于不同改进者之间的均值差异；从同一个前沿版本继续改时，换上后继模型或挂载经验笔记均未产生可测量的加速〔推断〕[LOG E；`decisions/cf_analysis.json`]。

[[figure:fig4-cf｜图 16　从最好版本继续改的 12 次运行。每条线是一次运行中“本次基线减去到第 t 次为止的最好值”，虚线是该改进者 3 次运行的平均降幅 [`decisions/cf_analysis.json`；LOG E]。]]

本节所报告的 9B 组均完成调用截断复核，研究调用中未发现足以改变比较结论的系统性截断。改进速度的计算与统计检验按预先设定的任务配对方案执行 [LOG E, LOG G].

### 补充检验：配方还是随机波动

第三轮的得分明显低于第一轮。为判断差距是否来自训练随机性或设计者规模，补做了两项比较。第一项保持第一轮数据和配方不变，只更换 LoRA 随机种子。第二项将配方设计者换为 Qwen3.8-27B，并让它基于与第二轮相同的反馈制定配方。

| 比较 | 差值 | 95% CI | p |
|---|---:|---|---:|
| 第一轮配方换种子重训 − 第一轮原模型 | −0.003 | [−0.125, +0.113] | 0.98 |
| 换种子重训 − 第三轮模型 | +0.211 | [+0.075, +0.385] | 0.008 |
| Qwen3.8-27B 配方 − 第二轮模型 | −0.082 | [−0.186, +0.011] | 0.11 |
| Qwen3.8-27B 配方 − 第三轮模型 | +0.122 | [+0.017, +0.269] | 0.047 |
| Qwen3.8-27B 配方 − 原始模型 | −0.075 | [−0.195, +0.023] | 0.23 |

换种子重训得分为 0.602，接近第一轮的 0.605，说明第一轮结果可重复。第三轮的 0.392 与第一轮之间的差距不能归因于随机种子。Qwen3.8-27B 设计的配方得分为 0.513。它优于第三轮，但未超过原始模型或第二轮模型 [LOG E; `decisions/decisions_1420.json`]。

结合四种配方，结果指向两项影响因素：是否剔除提升较小的被采纳记录，以及训练强度。使用同一批 539 条记录时，较轻训练得分 0.513，较重训练得分 0.392；保留全部记录的两次第一轮训练得分分别为 0.605 和 0.602。该比较支持配方选择影响后继模型表现，但尚不能单独确定筛选规则和训练强度各自的因果效应 [LOG E; `decisions/decisions_1420.json`]。

### 局限

- **迭代代数与规模受限**：后继模型仅推进至第三代，自主配方设计仅运行三轮（且仅含一次源自训练任务的反馈）；底层改进者仅测试了 Qwen3.5-9B 单一模型，评测任务局限于数秒级的微型训练任务与单一的 NanoChat 基准（其终点在封存数据上的复核尚待完成）[LOG E]。
- **跨任务方差限制了检测力**：尽管 A/A 测出的最小可检出效应约为 0.031，但两个不同模型之间的差值在任务间波动大得多（从 −0.58 到 +0.35），实际所需的效应量更大；而“从最好版本继续改”每组仅有 3 次运行 [LOG E, LOG H]。
- **单次微调方差与受限的配方空间**：每份自主配方仅训练了一次（第一、三轮配方几乎相同却相差 0.213，且均受 55 分钟上限截断在约 2.4 与 2.7 个 epoch），因此单次训练不足以稳健比较配方优劣〔推断〕[`decisions/decisions_1235.json`；LOG E]；同时，模型只能在人类预设的选项空间内组合超参，无法编写任意训练代码，按 Tang 等人的判据仍属 GPI 而非 RSI [T-24；LOG E]。
- **推理服务路径混杂**：原始模型不带 adapter 直接由 vLLM 服务，而所有微调模型均经过 vLLM 的 LoRA adapter 路径；这一混杂仅影响与原始模型的比较，后继模型之间及其与安慰剂之间的横向比较不受影响 [LOG H]。

#### Text (English)

### Design

Our experimental improver is Qwen3.5-9B in its native BF16 weights, served via vLLM on A100 GPUs [`platform_build_plan.md`]. The model operates inside an isolated sandbox while a sealed evaluator scores its candidates. The sandbox passed all 12 containment checks—including blocked network access, zero visibility into validation or sealed datasets, and tripwire alarms on decoy files [LOG E]. The improver receives only validation scores and training curves; it never sees sealed test scores, evaluator source code, or the reference optimum [`platform_design.md` §5.1].

[[figure:fig4-loop｜Figure 8. One attempt on the platform. The improver proposes a candidate inside the isolated sandbox; the sealed evaluator trains and scores it; it is kept if it beats the current best and discarded otherwise; the improver only sees the dev score and the training curve. The blue dot traces the path of one attempt. Hover for sources [`platform_build_plan.md`; `platform_design.md` §5.1; LOG E].]]

The experiment uses two task suites. The first is **NanoChat**: the improver edits the training script of a small language model; following Karpathy's autoresearch rule, each attempt trains once in the sandbox and is scored on a validation split, keeping modifications that beat the incumbent best and discarding the rest [L-28; `platform_design.md` §1.1]. Because single-run NanoChat evaluations are noisy (20 seeds, mean 1.4007, SD 0.0268), every candidate keep must be confirmed by a rerun on a fresh seed [LOG E].

The second suite comprises **8 unseen micro-research tasks** used to measure improvement speed across generations. Each task is a fast training run (completing in a few seconds) initialized with randomly drawn configuration defects; the improver lowers validation loss by tuning a JSON configuration of roughly 12 knobs (optimizer, learning rate, width, depth, etc.). Training is capped by step count so the improver cannot buy compute by enlarging the model [`platform_design.md` §5.1; LOG H]. Four of the 8 tasks belong to the same families as the training tasks with newly drawn settings (modular addition, sparse parity, teacher-student regression, 2-D classification), while the other 4 are entirely novel task types (sequence reversal, bracket matching, in-context linear regression, character-level language modeling) [`platform_design.md` §5.1; `decisions/decisions_1235.json`]. A separate set of 8 training tasks (explored by the original model across 5 seeds each) supplies training trajectories or lessons notes and never enters any comparison [LOG H].

During evaluation, each research run consists of 12 attempts. Every improver runs once on each unseen task using the exact same 5 seeds, with all comparisons paired by task [LOG E; LOG H].

The **improvement-speed score** (bounded between 0 and 1, higher is better) is computed as follows: at attempt $t$, we calculate the fraction of the gap between the starting score and a reference optimum closed by the best score achieved so far (where the reference optimum pools an offline random search with the best result across all arms, used strictly for normalization). Averaging this fraction over the 12 attempts and 5 seeds yields the task's improvement-speed score [`platform_design.md` §1.1–1.2]. The overall difference between two improvers is the mean per-task difference across the 8 tasks, with 95% confidence intervals from bootstrap and $p$-values from a task-paired sign-flip permutation test (whose minimum possible $p$-value across 8 tasks is $2/256 \approx 0.008$) [`platform_design.md` §1.3–1.4].

[[figure:fig4-scoredef｜Figure 9. How the improvement-speed score is computed (schematic; the curve shape is not experiment data). The shaded area is the gap already closed at each attempt; the score is the shaded share of the whole rectangle, averaged over 5 seeds and compared task by task over the 8 tasks [`platform_design.md` §1.1–1.4].]]

The experiment incorporates three types of controls [`platform_design.md` §1.5]:
1. **A/A control**: Rerunning the original model on a disjoint seed set yields a per-task difference SD of 0.031, establishing a baseline minimum detectable effect (MDE) of about 0.031 [`decisions/decisions_1235.json`].
2. **Placebo control**: On the weight side, fine-tuning on an equal number of rejected attempts under identical settings; on the lessons side, using generic tips of equal length written without inspecting any attempt logs [LOG E].
3. **"More data" and random-recipe controls**: Isolating the effect of added data volume or dataset changes (detailed in the respective subsections below).

The statistical decision rule was locked in before execution: a second-order improvement is detected only if (1) the lower bound of the 95% CI exceeds 0, (2) the difference exceeds twice the A/A standard deviation ($> 0.062$), and (3) the improver is significantly better than its matched control; otherwise, it is reported as "not detected at this power" [`platform_design.md` §1.4; `decisions/decisions_1235.json`].

### First-order result: a fixed improver improves NanoChat

Running through the OpenCode harness across 3 parallel threads for 176.6 minutes, the unmodified original model proposed and evaluated 72 attempts (25 kept, 47 rejected, 9 evaluation crashes, and 0 audit flags) [LOG E]. Validation loss (bits per byte, lower is better) fell from 1.4152 to 1.2308—a reduction of 0.1844, placing the final result roughly 6.3 standard deviations below the mean of single-evaluation noise [LOG E; `replay/final/ovn1118b-nc-g0/events.jsonl`] (see Figure 7 in Section 3 for the time breakdown; the endpoint check on sealed data remains pending [LOG E]). Because the improver stayed fixed throughout, this result establishes a clear first-order improvement.

[[figure:fig4-replay｜Figure 10. Replay of the NanoChat research run, redrawn from the real event log at about 350× speed, looping. Top: best dev loss so far (minimum across the 3 threads) and each evaluation result; middle: the 3 GPU evaluation lanes; bottom: the latest decisions and the proposal behind each (model's own words). Hover over a dot for that attempt's value [`replay/final/ovn1118b-nc-g0/events.jsonl`; LOG E].]]

[[figure:fig4-scores｜Figure 11. Improvement-speed score on the 8 unseen micro-research tasks for the improver produced by each self-update approach and for its control. Blue dots are the improvers, orange dots their controls: generic tips of equal length for the lessons note, the placebo model for generation 1, the "more data" control for generations 2 and 3, and a random recipe for each of the three self-designed rounds. The dashed line is the original model (0.588). Control scores are derived as the improver's score minus the difference between the two. Hover to see values and sources [`decisions/decisions_1235.json`].]]

### Lessons note

Before updating weights, we tested whether the model could accelerate itself purely in-context. Reading its own research records across the 8 training tasks (5 seeds each), the original model wrote a 252-word lessons note that was appended to the prompt of every subsequent evaluation run while keeping weights frozen [LOG E; LOG H]. The note codified a fixed tuning sequence derived from the training tasks (switch to a cosine schedule, lower weight decay, widen and deepen, then tune learning rate) and included the rule: *"Ignore changes that increase loss even slightly"* [LOG E]. Its placebo control consisted of generic tips of the same length written without reading any records [LOG E].

Appending the note reduced the model's improvement-speed score from 0.588 to 0.392 (a difference of −0.196, 95% CI [−0.383, −0.033], entirely below 0; sign-flip $p = 0.11$), slowing improvement by roughly a third [`decisions/decisions_1235.json`]. Against the placebo tips—which scored about 0.631 (Figure 11)—the lessons note trailed by −0.239 (95% CI [−0.420, −0.075], $p = 0.008$), scoring lower on all 8 tasks, led by drops on 2-D classification (−0.58), sequence reversal (−0.56), in-context linear regression (−0.49), and modular addition (−0.20) (Figure 13) [`decisions/decisions_1235.json`]. This indicates that the note over-generalized incidental successes on the training tasks into a rigid search order that misdirected exploration on unseen task types, while its ban on any temporary loss increase further narrowed exploration (an inference) [LOG E].

### Fine-tuning on its own successes: three successor generations

We next tested weight-level updates across three generations. **Generation 1** was fine-tuned from the original weights via LoRA (rank 16, learning rate 1e-4, 2 epochs) on all 438 calls from accepted attempts on the training tasks, while its **placebo model** was trained under identical settings on 438 rejected calls [LOG E; `platform_design.md` §5.1; `deploy/second_order.md`].

**Generation 2** was again initialized from the original weights, trained on the original model's records plus new records generated by Generation 1 on fresh seeds of the training tasks. Its **"more data" control** combined the original records with an equal volume of new records generated by the *original* model on the same seeds—holding data volume constant so the only difference is who generated the added trajectories [LOG H; `platform_design.md` §1.5]. **Generation 3** followed the same pattern, adding Generation 2's records against a control that added two batches from the original model [LOG E].

[[figure:fig4-lineage｜Figure 12. Where the training data of each successor generation and its control comes from. Every model starts LoRA from the original weights; generations 2 and 3 and their "more data" controls use the same amount of data, differing only in who produced the added part. Blue arcs show records produced by one generation entering the next generation's data. Right: score on the unseen tasks [LOG E; LOG H; `decisions/decisions_1235.json`].]]

| Comparison | Difference | 95% CI | p |
|---|---|---|---|
| Generation 1 − original model | −0.052 | [−0.177, +0.041] | 0.49 |
| Generation 1 − placebo model | +0.004 | [−0.084, +0.080] | 0.91 |
| Generation 2 − generation 1 | +0.037 | [−0.032, +0.129] | 0.45 |
| Generation 2 − "more data" control | +0.039 | [−0.042, +0.140] | 0.20 |
| Generation 2 − original model | −0.015 | [−0.107, +0.048] | 0.70 |
| Generation 3 − generation 2 | −0.087 | [−0.190, −0.005] | 0.0625 |
| Generation 3 − "more data" control | −0.066 | [−0.164, +0.025] | 0.19 |
| Generation 3 − original model | −0.102 | [−0.247, +0.009] | 0.15 |

(Sources: [`decisions/decisions_1235.json`; LOG E])

Across the original model and Generations 1 to 3, improvement-speed scores moved from 0.588 to 0.537, 0.573, and 0.487; no comparison met our detection rule [LOG E; `decisions/decisions_1235.json`]. Specifically:
1. **Success signal was not learned**: Generation 1 (0.537) and the placebo model performed almost identically (difference +0.004, $p = 0.91$), showing that fine-tuning on accepted versus rejected records made no difference [LOG E].
2. **Generation 2 merely returned to baseline**: While Generation 2 posted a positive gap against its "more data" control—driven mainly by sequence reversal (+0.18) and in-context linear regression (+0.11) [`decisions/decisions_1235.json`]—this occurred only because both Generation 1 (0.537) and the control (0.534) dipped below the original model, whereas Generation 2 (0.573) returned near the original baseline without surpassing it (an inference) [LOG E].
3. **Generational degradation in Generation 3**: Generation 3 fell 0.087 below Generation 2 (marked as worse by the bootstrap 95% CI upper bound below 0; sign-flip $p = 0.0625$) [LOG E]. Failing to surpass the original model across three generations and showing signs of degradation in the third aligns in direction with the recursive-training degradation described by Shumailov et al. [M2-04], though without measuring output distribution tails we cannot confirm whether the underlying mechanism is identical (an inference) [LOG E].

[[figure:fig4-pertask｜Figure 13. Per-task difference between each improver and its control (improver minus control, improvement-speed score) on the 8 tasks. Blue means the improver is faster, orange slower; the rightmost column is the mean over the 8 tasks. The same approach can differ by anywhere from −0.58 to +0.35 across tasks, which is why the confidence intervals are wide. Hover to see values and sources [`decisions/decisions_1235.json`].]]

### The model decides how to train itself

In the three generations above, humans fixed the data filters and hyperparameters in advance—a process that Tang et al. classify as GPI rather than RSI [T-24]. Inspired by SEAL, which lets a model choose its own fine-tuning data and hyperparameters (raising self-edit success on an ARC subset from 20% to 72.5% after outer-loop feedback training) [M-13; LOG H], we let the model choose its own training recipe within a structured space. Reading a summary of the training logs (counts of accepted, hindsight, and rejected records across the 4 task types, relative gain distributions, and sample hypotheses), the designer specifies which records to filter and which LoRA settings (rank, learning rate, epochs) to use [LOG E].

All three rounds draw on the original model's record pool, fine-tune from the original weights, and cap training at 55 minutes [LOG E; `deploy/second_order.md`]. Each round is paired with a **random-recipe control** that samples the exact same number of calls at random from valid records and trains under identical LoRA settings, separating the model's data-selection judgment from the effect of simply changing the dataset [LOG E].

The designer role progresses across the three rounds: **Round 1** is designed blindly by the original model; **Round 2** is also designed by the original model, after seeing a feedback table comparing per-task scores on the training tasks (with fresh seeds) for Round 1, the original model, and Generation 1; **Round 3** replaces the designer with the *Round-1 fine-tuned model*, which sees the exact same feedback, record pool, and starting weights as in Round 2 [LOG E].

[[figure:fig4-rounds｜Figure 14. The three rounds in which the model decides how to train itself. Each column shows, top to bottom, the designer, its recipe, the actual training and the score on the unseen tasks. The designers of rounds 2 and 3 see the same feedback table; the orange arc marks the round-1 model becoming the round-3 designer, whose first two answers were cut at the 8192-token cap [LOG E; `deploy/second_order.md`; `decisions/decisions_1235.json`].]]

| | Fixed recipe (generation 1) | Round 1: designed by the original model | Round 2: original model, after feedback | Round 3: designed by the round-1 model |
|---|---|---|---|---|
| Records used | accepted | accepted + hindsight | accepted only | accepted + hindsight |
| Minimum relative gain for accepted records | — | none | 1% | 1% |
| Calls kept per attempt | — | all | last only | all |
| Per-task cap | — | 150 | 100 | 150 |
| Calls selected | 438 | 694 | 137 | 539 |
| LoRA rank / learning rate / epochs | 16 / 1e-4 / 2 | 32 / 3e-4 / 4 | 16 / 2e-5 / 2 | 32 / 3e-4 / 4 |
| Actual training | — | stopped at the 55-minute cap, about 2.4 epochs | finished in about 10 minutes, 31 steps; validation loss 0.144 → 0.145 | stopped at the 55-minute cap, step 183, about 2.7 epochs |
| Improvement-speed score on unseen tasks | 0.537 | 0.605 | 0.595 | 0.392 |
| Score of the random-recipe control | — | about 0.528 | about 0.552 | about 0.504 |

(Sources: [LOG E; `decisions/decisions_1235.json`; `deploy/second_order.md`]. "—" means not applicable or not recorded separately; random-recipe scores are derived from the differences.)

**Round 1.** Reasoning that the default recipe suffered from too little data and LoRA capacity, the original model raised the cap, added hindsight records, and strengthened the LoRA—an explanation at odds with Generation 1, where the placebo comparison pointed to training signal rather than data volume (an inference) [LOG E]. The Round-1 model scored 0.605, edging out the original model (+0.017, 95% CI [−0.109, +0.155], $p = 0.82$) and its random recipe (+0.077, 95% CI [−0.085, +0.223], $p = 0.36$), but wide task-to-task swings (from +0.35 on modular addition to −0.27 on in-context linear regression against the random recipe; Figure 13) kept all intervals straddling zero, marking it as not detected [`decisions/decisions_1235.json`].

**Round 2.** Although the feedback table showed Round 1 leading on the training tasks (0.652 vs. 0.525 for the original model, likely reflecting memorization of the training tasks (an inference)), the original model never cited that table in its rationale; instead, citing hindsight noise and overfitting concerns, it reversed nearly every choice from Round 1 (selecting just 137 calls and training for 31 steps at a 2e-5 learning rate) [LOG E]. Because validation loss barely moved, the weights were hardly changed and the model's score (0.595) simply reverted to the original model's level (+0.007 vs. original, $p = 0.65$; all other comparisons non-significant, see Figure 11 and [`decisions/decisions_1235.json`]) (an inference) [LOG E].

### Round 3: the fine-tuned model sets the recipe

When the Round-1 fine-tuned model took over designing the Round-3 recipe, its first two responses exhausted the 8,192-token output cap while thinking and returned empty bodies, producing a valid recipe only on the third try (whereas the original model succeeded on its first try in Rounds 1 and 2) [LOG E]. It kept Round 1's aggressive settings almost intact, adding only a 1% minimum relative-gain threshold to select 539 calls [LOG E].

Yet the resulting Round-3 model dropped sharply to **0.392**—significantly worse than Round 2 (−0.204, 95% CI [−0.365, −0.065], $p = 0.008$), the original model (−0.197, 95% CI [−0.363, −0.050], $p = 0.031$), Round 1 (−0.213, $p = 0.047$), and the fixed recipe (−0.145, $p = 0.008$), with the steepest drops against the random recipe on sequence reversal (−0.48) and modular addition (−0.34) (Figure 13) [`decisions/decisions_1235.json`; LOG E].

This outcome highlights two findings:
1. **Recipe and training intensity shaped the outcome**: Retraining the Round-1 recipe with a new random seed scored 0.602, close to the original 0.605. This rules out ordinary seed variation as the explanation for the Round-1 to Round-3 gap. Qwen3.8-27B used the Round-3 data filter with lighter training and scored 0.513, above Round 3 at 0.392 but below the original model at 0.588. Across the four recipes, data selection and training intensity both affected successor performance.
2. **Behavioral collapse from training intensity**: All three heavily trained models (Round 1, Round 3, and Round 3's random-recipe control) shifted on unseen tasks to firing thousands of very short calls (11,287–15,087 calls, median 47–54 tokens) compared to 2,030 calls (median 111 tokens) for the original model and 101 tokens for the lightly trained Round-2 model—demonstrating that this behavioral shift stems from training intensity rather than who designed the recipe (an inference) [LOG E].

[[figure:fig4-behavior｜Figure 15. Call count on the unseen tasks against median output length. The three heavily trained models (round 1, round 3 and round 3's random recipe) all switch to many short calls; the original model and the lightly trained round-2 model stay at the upper left [LOG E].]]

### Continuing from the best version

Besides unseen tasks, another way to prevent a successor from simply replaying memorized training answers is to have every improver continue from the same frontier checkpoint—specifically, the best NanoChat version (1.2308) found during our first-order run [`platform_design.md` §1.6; LOG E]. Four improvers each ran 3 single-threaded 12-attempt continuations with rotated evaluation GPUs, measured by the loss drop from each run's own re-measured baseline (1.2285–1.2353) [LOG E; `decisions/cf_analysis.json`]:

| Improver | Drop in each of the three runs | Mean drop | Versus original | 95% CI | p |
|---|---|---|---|---|---|
| Original model | 0.0243 / 0.0005 / 0.0122 | 0.0124 | — | — | — |
| Generation 1 | 0.0275 / 0.0008 / 0.0326 | 0.0203 | +0.0079 | [−0.0116, +0.0264] | 0.50 |
| Placebo model | 0.0062 / 0.0334 / 0.0137 | 0.0178 | +0.0054 | [−0.0116, +0.0224] | 0.60 |
| Original model with lessons note | 0.0142 / 0.0171 / 0.0016 | 0.0110 | −0.0014 | [−0.0145, +0.0117] | 1.00 |

(Sources: [`decisions/cf_analysis.json`; LOG E]. Differences are unpaired mean differences with bootstrap confidence intervals; p values come from an exact permutation test over 20 relabellings.)

Across 3 runs per group, run-to-run spread within a single improver (e.g., 0.0005 to 0.0243 for the original model) exceeded the differences between improvers; when continuing from the same frontier version, neither a successor model nor the lessons note yielded a measurable speedup (an inference) [LOG E; `decisions/cf_analysis.json`].

[[figure:fig4-cf｜Figure 16. The 12 runs that continue from the best version. Each line is one run's baseline minus its best value after attempt t; the dashed line is that improver's mean gain over its 3 runs [`decisions/cf_analysis.json`; LOG E].]]

All reported 9B groups were checked for truncated model calls. The audit found no systematic truncation that would alter the comparisons. Improvement scores and statistical tests followed the task-paired analysis specified in advance [LOG E, LOG G].

### Follow-up test: recipe effects and seed variation

Round 3 scored substantially below Round 1. Two follow-up comparisons tested whether the difference reflected training randomness or the recipe designer. The first kept Round 1's data and recipe fixed and changed only the LoRA random seed. The second used Qwen3.8-27B as the recipe designer, with the same feedback given to the original model in Round 2.

| Comparison | Difference | 95% CI | p |
|---|---:|---|---:|
| Round-1 recipe retrained with a new seed − original Round-1 model | −0.003 | [−0.125, +0.113] | 0.98 |
| New-seed retraining − Round-3 model | +0.211 | [+0.075, +0.385] | 0.008 |
| Qwen3.8-27B recipe − Round-2 model | −0.082 | [−0.186, +0.011] | 0.11 |
| Qwen3.8-27B recipe − Round-3 model | +0.122 | [+0.017, +0.269] | 0.047 |
| Qwen3.8-27B recipe − original model | −0.075 | [−0.195, +0.023] | 0.23 |

The new-seed run scored 0.602, close to Round 1's 0.605. The gap between Rounds 1 and 3 therefore cannot be attributed to random-seed variation. The Qwen3.8-27B-designed recipe scored 0.513. It outperformed Round 3 but did not surpass the original or Round-2 model [LOG E; `decisions/decisions_1420.json`].

Across the four recipes, two choices tracked the outcome: whether accepted records with small gains were removed, and training intensity. On the same 539 records, lighter training scored 0.513 and heavier training scored 0.392. The two Round-1 runs that retained all records scored 0.605 and 0.602. These comparisons show that recipe choices shaped successor performance, while they do not isolate the causal contribution of each choice [LOG E; `decisions/decisions_1420.json`].

### Limitations

- **Limited generational horizon and scale**: Successors were tracked for only three generations, and self-designed recipes for three rounds (with a single round of feedback from training tasks). Only one base model (Qwen3.5-9B) was tested across fast micro-tasks and one NanoChat task (whose sealed-data endpoint verification remains pending) [LOG E].
- **Cross-task variance vs. statistical power**: While the A/A minimum detectable effect is about 0.031, differences between distinct models fluctuated widely across tasks (−0.58 to +0.35), requiring larger true effects for detection; the continue-from-best experiment was likewise limited to 3 runs per arm [LOG E, LOG H].
- **Single-run recipe variance and human-bounded recipe space**: Each self-designed recipe was trained only once (Rounds 1 and 3 used nearly identical recipes that stopped at ~2.4 and ~2.7 epochs under the 55-minute cap yet differed in score by 0.213), making single-run recipe comparisons unreliable (an inference) [`decisions/decisions_1235.json`; LOG E]. Moreover, because the model selected only from predefined options rather than writing arbitrary training code, the setup remains within GPI rather than RSI under Tang et al.'s criterion [T-24; LOG E].
- **Serving-path confound**: The original model was served directly by vLLM without an adapter, whereas all fine-tuned checkpoints were routed through vLLM's LoRA adapter path; this confound affects only comparisons against the original model, leaving comparisons among successors and against placebos unaffected [LOG H].
