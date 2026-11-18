# Recursive Self-Improvement in AI: A Survey of Claims, Evidence, and Empirical Tests

网页主标题：AI 正在改进自己吗？（Is AI Improving Itself?）· 作者：孙祺（Qi Sun）· 定位：survey with an empirical case study

> **完整的报告正文**（摘要 + 已写好的各节，中英文 + 参考文献）由构建脚本自动汇总到 `report/report_full.md`，每次构建都会更新；各节正文在 `report/sections/` 里改。
>
> **这份文件的用法**
> - 这是文字版报告。网页（`report/index.html`）按这里的章节顺序和文字生成，二者保持一致。
> - **✍️ 块由你来写**，块里的提示是写作方向，不是正文，写完后删掉提示。
> - **📎 素材块是 Claude 整理的可引用事实**，每一条都带出处。写正文时挑着用，引用格式保持方括号不变，网页生成时会转成脚注。
> - 写完一节告诉我，我会做事实核对：数字和出处是否对得上，"自报"和〔推断〕是否标全。
>
> **引用格式**（完整对照表见附录 A）
> - `[RF-06]`、`[NV-24]`、`[FL-21]`、`[AX-25]`：四份材料证据台账里的编号，见 `research/sources/*.md`，每条都有原文短引和 URL。
> - `[LOG E·dec_us]`：`research/REPORT_LOG.md` 的节号和条目。
> - `[文件 §x]`：`research/` 下的其他文件及其章节。
> - `[T-12]`、`[M-25]`、`[E-28]`、`[L-06]`、`[X-24]`：文献键，见 `report/references.md`（从 `research/` 的文献文件逐字抽取，所有出处都保留）。
>
> **术语规则**：正文只用文献里已有的术语，每个术语第一次出现时给出处。实验里的变量名（Δ₂、nAUC、U_s、U_w、M0/M1/M2）只作为符号使用，在 §4.1 给出定义，并说明对应的文献做法。
> **标注规则**：厂商自己报告的数字标"自报"；我们的推断标〔推断〕；粗算标〔估算〕。
>
> **范围**：报告只写 Qwen3.5-9B 的实验；27B 相关内容按用户决定（11-18）全部删除。
>
> **用语规则**：正文像讲故事一样，用平常的话写，不出现内部代号。代号只允许出现在图表的数据来源和附录里。方括号里的引用键（如 [T-24]）在网页上会变成脚注编号。
>
> **用语对照**（写正文时，左边的代号一律换成右边的说法）
>
> | 内部代号 | 正文里的说法 |
> |---|---|
> | M0 | 原始模型（Qwen3.5-9B） |
> | M1 | 第一代后继模型：用原始模型自己被采纳的尝试记录微调一次 |
> | M2 | 第二代后继模型：在上面的数据之外，再加上第一代后继模型新跑出来的记录 |
> | M2c | "多给数据"对照模型：数据量和第二代相同，但新增的那一份仍由原始模型产生 |
> | M1p | 安慰剂模型：改用被否决的尝试记录来微调，条数相同 |
> | U_s | 让模型读自己的尝试记录，写一份经验笔记给下一轮用（权重不动） |
> | U_w | 用模型自己被采纳的尝试记录微调权重 |
> | U_sd | 让模型自己决定挑哪些记录、怎么微调自己 |
> | nAUC | 改进速度分：12 次尝试里，平均把"起点到参考最优"的差距缩小了多少，0 到 1，越高越好 |
> | Δ₂ | 换上后继模型之后，改进速度分变了多少 |
> | A/A | 同一个模型跑两遍，看纯随机的波动有多大 |
> | placebo | 安慰剂对照：做同样多的训练或写同样长的笔记，但内容不含"哪些做法成功了"的信息 |
> | held-out | 训练时没见过的任务 |
> | µAR | 8 个小型研究任务（每个都是一次小规模训练，几秒跑完） |
> | G0、nc_g0 | 第一轮研究运行：原始模型在 NanoChat 上自主做了 3 小时研究 |
> | val_bpb | 验证集上每个字节的比特数，越低越好 |
> | ignition test | 文献术语，第一次出现时解释为"把改进者换成它自己改出来的后继者，看后者是否更会做改进" [E-28] |
>
> **网页还是 PDF**：🌐 表示只放网页（交互或视频），📄 表示 PDF 里要保留。

---

## 0. 摘要 / TL;DR

> **✍️ 你来写：核心论点（一到两句）+ 摘要（约 150 字）**
>
> 候选表述，可以选一个改写，也可以自己写：
> （已按术语规则改写：去掉了"内环 / 外环""一阶 / 二阶""A/B/C 类"这些非文献术语。）
> - **T1**："这些系统展示的是 AI 辅助改进 AI 研发的产物，文献称为 non-recursive optimization 或 weak self-improvement [T-03, T-12]；没有展示改进者本身'越改越会改'（'gets better at getting better'，Yampolskiy 称之为 recursive self-improvement [T-12]）。前者有充分证据，后者没有任何一家测量过。要区分两者，需要的不是更强的模型，而是把后继模型换上去当改进者、逐代带对照地测改进速率，即 AIDE² 所说的 ignition test [E-28]。"
> - **T2**："按 Tang et al. 2026 的判据，改进机制在系统外部的只算 GPI [T-24]；按 Yampolskiy 2015，在固定 fitness 下优化只算 weak self-improvement [T-12]。四份材料都落在这两类里。"
> - **T3**："自动化沿着最硬、被加固过的信号推进 [L-01, L-02]；后继模型能不能做出更快的改进（Weco 称为 ignition [X-24]）没人测，是因为它本身没有现成的评估器。我们搭了一个，在小规模下结果为零甚至为负。"
> - **T4**："我们在可控的小规模上做了权重层的 ignition test：自我反思笔记有害，拒绝采样微调为零，两代没有复利。"
>
> TODO

> **摘要和三张卡片的正文已移到 `report/sections/00_abstract.md`**（中英文都在那里）。下面的折叠部分是之前各版的修改记录。

<details><summary>修改记录（定稿时删除）：最初原文、第一轮标注、你的第二稿、第二轮核对、英文第一稿、第三稿</summary>

**最初原文**

目前的行业RSI model和各自的解决方式依然是通过外部的model作为一个锚点通过ai辅助的方式来进行模型的开发和提升能力，我把他称为一阶模型自进化。相对来说虽然模型能够在一阶状态下提高自己的能力，但是在长期迭代的过程中能力的上升会发生逐步的衰减，并不能做到真正的长线自我迭代更新。从长远目标来看二阶模型自进化是一个越来越快的过程，至少是一个平滑的直线过程，模型的各种能力增常带来新的思路和增益，新的思路和增益通过更新权重和更新harness的handover和技能更进一步的带给下一代。从大量的行业和学术的论文报告的调研中，目前没有人能够证明二阶的真实存在。我尝试了大量小型的test从而观察到，在小型model上面（Qwen3.6-9B & Qwen3.6-27B）上，模型通过自进化和迭代换心脏，能够做到模型能有提升，但是这种提升是非常有限而且递减的。

#### 摘要：问题标注（Claude，逐句）

| # | 原文 | 问题 | 依据 / 改法 |
|---|---|---|---|
| 1 | "目前的行业 RSI model" | **范围过大**：我们只分析了四份材料，不能代表整个行业 | 改为"本文分析的四份材料" |
| 2 | "通过外部的 model 作为一个锚点" | 方向有依据，但"锚点"会和 Tang et al. 的 **Anchored** RSI 混淆；后者指评价标准锚定在系统外，是另一回事 [T-24] | 改成具体事实：基座来自外部（Apex 用 Qwen3.6，Naive 用 MiMo-V2.5）；推理轨迹来自外部 teacher（Flint）[`model_dissection_apex_naive.md` §7；`flint_reverse_engineering.md` §3]；Recursive 自己没有披露用了什么模型 [RF 待核 9]，Naive 的图里把它标为 "undisclosed third-party model" [NV-C11]，这是 Naive 的说法 |
| 3 | "我把他称为一阶模型自进化" | **自创术语**，违反术语规则 | 用 non-recursive optimization / weak self-improvement [T-03, T-12]，以及 GPI [T-24] |
| 4 | "模型能够在一阶状态下提高自己的能力" | **主语不对**：这类做法改进的是 kernel、训练代码、数据等产物，改进者本身没有变 | 改为"改进 AI 研发的产物" |
| 5 | "长期迭代中能力的上升会逐步衰减" | 四份材料**没有长期迭代的数据**，不能由材料得出；文献里有证据，但对象是自我训练 | 引文献：Mind the Gap 2–3 轮饱和 [M2-01]；SPIN 增益递减 [M-20]；ReST-EM 在 APPS 上第一轮后就回退 [M-17]；Yampolskiy 的 supercompilation 例子 20% → 3% → 0 [T-12] |
| 6 | "并不能做到真正的长线自我迭代" | **过度断言**：没有证据表明能做到，不等于证明了做不到 | 改为"没有证据表明" |
| 7 | "二阶模型自进化" | 自创术语 | 用 recursive self-improvement，"gets better at getting better" [T-12]；实证上叫 ignition [X-24, E-28] |
| 8 | "越来越快……至少是一个平滑的直线过程" | "越来越快"有出处，"至少是直线"**没有出处**；而且 Tang et al. 明确不把复利放进定义 | 引 Yudkowsky 的 k [T-04]、Epoch / Forethought 的 r > 1 [T-16, T-17]，并注明定义有分歧 [T-24]；删掉"直线" |
| 9 | "能力增常" | 错字 | "增长" |
| 10 | "更新 harness 的 handover 和技能" | "handover"含义不清 | 改为"脚手架（harness 代码、经验记忆、技能库）"，对应 DGM / STOP 改代码 [M-03, M-01]、Reflexion 记忆 [M-23]、Voyager 技能库 [M-24] |
| 11 | "从大量的行业和学术论文报告的调研中" | 要给出范围 | 写明"约 96 个来源"[`lit_rsi_academic.md` 头注]，并加上"在我们的检索范围内"[同上 §8] |
| 12 | "没有人能够证明二阶的真实存在" | 措辞过强 | 改为"没有工作测出跨代复利"；最接近的 AIDE² ignition test 结果不确定，0.780 对 0.782 [E-28] |
| 13 | "Qwen3.6-9B & Qwen3.6-27B" | **模型名错误** | 实际是 Qwen3.5-9B 和 Qwen3.8-27B [`platform_build_plan.md` 决策 D1、第 194–195 行] |
| 14 | "大量小型的 test" | 不具体 | µAR 的 8 个 held-out 任务 × 5 个 seed × 每个 run 12 个 trial，三组带对照的比较 [LOG E·dec_us] |
| 15 | "27B……通过自进化和迭代换心脏" | **与实验不符**：27B 没有进入自我更新链，只作为改进者的对照；27B 链的结果还没出来 | 27B 只写"作为对照，没有更快（+0.001）"[LOG E·dec_g1] |
| 16 | "能够做到模型能有提升，但提升有限且递减" | **与数据矛盾** | 改进者固定时有提升：NanoChat val_bpb 1.4152 → 1.2310 [LOG E]。换上后继者后**没有检出提升**：U_w −0.052，CI [−0.177, +0.041]；第 2 代对原模型 −0.015；U_s 显著为负，−0.196 [LOG E·dec_us, dec_g1, dec_g2]。只有两代且没有正效应，谈不上"递减" |
| 17 | "换心脏""自进化" | 自创或口语化 | 用"把改进者替换为用自己轨迹更新过的后继者"，即权重层的 ignition test [E-28] |

**你的第二稿**

本文分析的四份网页报告（Recursive 的 First Steps 和 Flint、Naive 的 N0.5、Apex-35B），展示的都是用model改进model或者是inference速度研发的产物：训练代码、GPU kernel、推理引擎和训练数据。在这个过程中，改进者本身没有被改写。文献把这种情况称为 non-recursive optimization 或 weak self-improvement [T-03, T-12]；按 Tang et al. (2026) 的判据，现阶段的rsi模型研发属于 GPI，而不是 RSI [T-24]。同时，对权重和行为的逆向结果显示：三个开源模型的基座或推理轨迹都含有外部模型蒸馏带来的行为习惯，从而更加难以判断是否模型的建立是由于外部的商用强模型来对模型开发的辅助和数据生成还是通过管线获得的能力，更加不能清晰如今行业的rsi的真实动向到了哪一步。同时本文参考了大量学术对于rsi的研究并且对于整体研究产生了详细的规划。

学术文献文献对 RSI 的核心要求，是改进者"gets better at getting better"[T-12]。多数理论工作还要求跨代回报不低于 1，也就是改进越来越快，我个人称之为二阶性 （二阶导数大于零） [T-04, T-16, T-17]，不过 Tang et al. 没有把这一条放进定义 [T-24]。实证方面，在人设计的固定循环下，自我训练通常 2–3 轮就饱和 [M2-01, M-20, M-17]。在检索到来源中，没有工作测出跨代的复利。最接近的是 AIDE² 的 ignition test。但是它只替换了模型使用的环境，并没有修改模型的参数，同时结果依旧存在不确定性质（「说清楚什么东西的分数是什么不然我怎么知道」0.780 对 0.782）[E-28]。

我在一个小规模平台上做了权重层的 ignition test：把 Qwen3.5-9B 改进者替换成用它自己的轨迹更新过的后继者，在 8 个 held-out 小型研究任务上比较改进速率，并设置 A/A、placebo 和"更多旧数据"三种对照。改进者固定时，AI 辅助研发是有效的：NanoChat 的 val_bpb 从 1.4152 降到 1.2310 [LOG E]。替换改进者之后，虽然能力依旧保持上升，但并未检出改进速率的提升。

〔预留：第 3 代和 U_sd 的结果〕

#### 摘要中文稿：第二轮核对（Claude；只标问题，你的文字没有改动）

| # | 你的原句 | 问题 | 建议 |
|---|---|---|---|
| 1 | "展示的都是用 model 改进 model 或者是 inference 速度研发的产物" | 容易读成"模型改进了自己"，和下一句"改进者本身没有被改写"矛盾 | 改为"用 AI 改进模型训练和推理的产物" |
| 2 | "现阶段的 rsi 模型研发属于 GPI" | 范围又扩大了：我们只看了四份材料，不能代表整个行业的现状；另外 rsi 要大写 | 改为"这四份材料展示的做法属于 GPI" [T-24] |
| 3 | "三个开源模型的基座或推理轨迹都含有外部模型蒸馏带来的行为习惯" | 对 Naive 不成立：它的说话习惯还没测，目前只从权重看出它是在小米 MiMo-V2.5 的对话版上继续训练的。Apex 像是学了 Claude 的思考方式，但这只是推断，它从没自称 Claude。只有 Flint 有直接证据，思考过程里出现了"as a Gemini model" | 分开写三家，各说各的证据 [`model_dissection_apex_naive.md` §2, §5, §7；`flint_reverse_engineering.md` §3] |
| 4 | "外部的商用强模型" | 不准确：Flint 模仿的老师里，除了 Gemini 是商用模型，还有 gpt-oss 和 DeepSeek-R1，这两个是公开权重的模型 | 改为"外部更强的模型" [LOG F] |
| 5 | "更加难以判断……还是通过管线获得的能力" | 方向合理，但这是我们的推断；句子太长 | 注明是推断，拆成两句 |
| 6 | "更加不能清晰如今行业的 rsi 的真实动向到了哪一步" | 语句不通，范围同第 2 条 | 例："因此，仅凭公开材料，无法判断这些模型的能力有多少来自它们的自动化研发流程。" |
| 7 | "本文参考了大量学术对于 rsi 的研究并且对于整体研究产生了详细的规划" | 太空泛："大量"是多少？"规划"指什么？ | 写出数字：约 96 个来源，另有 29 篇专门讲评估方法的文献。如果"规划"指给 Ling 的计划，就直接说"并据此给出 Ling 的 RSI 计划" |
| 8 | "学术文献文献" | 多了一个"文献" | 删掉一个 |
| 9 | "我个人称之为二阶性（二阶导数大于零）" | ① 是自己起的名字，违反你定的规则；② "我个人"不正式；③ 引用放在这句后面，读者会以为文献里用了"二阶性"这个说法；④ 意思也不完全对：文献里要回报大于 1 才算加速，"不低于 1"把刚好等于 1 的分界情况也算进去了；而且曲线是不是"越来越陡"，要看纵轴怎么画（普通刻度还是对数刻度）[`lit_rsi_academic.md` §3] | 删掉这句。如果想保留"加速"的意思，写"也就是每一代改进得比上一代更快"，引用放在这句之后 |
| 10 | "自我训练通常 2–3 轮就饱和" | "2–3 轮"只是其中一篇文献的结果；另一篇是每轮涨得越来越少，还有一篇是第一轮之后就开始变差 | 改为"几轮之内就停滞，甚至变差" [M2-01, M-20, M-17] |
| 11 | "在检索到来源中" | 漏了字，也少了范围限定 | "在我们查过的约 96 个来源里" |
| 12 | "它只替换了模型使用的环境" | "环境"一般指任务环境，不对。AIDE² 换的是外层那个改进者 agent 的代码，底下用的大模型没换 | 改为"它换的只是改进者 agent 的代码，底下的大模型没有变" [E-28] |
| 13 | 「说清楚什么东西的分数是什么」0.780 对 0.782 | 已按论文原文填好（逐句核对过，见 `report/notes/aide2_ignition.md`） | 建议原句："他们让同一个研究 agent 接受两种'教练'的改进：一种是这个 agent 自己，一种是人写的原版 agent，各跑 3 次、每次 50 轮。分数是 agent 在三类研究任务（训练模型、设计算法、改进 agent 的提示词和工作流程）上的平均得分，用不公开的测试数据打分，越高越好。最后两边的平均分是 0.780 和 0.782，几乎一样，作者自己也说结论不确定。" [E-28 §3.6, 图 7] |
| 14 | "我在一个小规模平台上" | 第一人称单数，和前文的"本文"不统一 | 统一用"本文"或"我们" |
| 15 | "替换改进者之后，虽然能力依旧保持上升，但并未检出改进速率的提升" | 这句有两种读法。如果意思是"后继模型做任务时仍然能把结果改好"，这是对的。如果意思是"模型一代比一代强"，那不对：第一代后继模型的改进速度分是 0.537，第二代是 0.573，都比原始模型的 0.588 低 [LOG E] | 改为"后继模型做任务时仍能把结果改好，但改得并不比原始模型快" |
| 16 | （整段） | 删掉了唯一一个统计上站得住的结果：让模型写经验笔记，反而改得更慢了（改进速度分低了 0.196） | 建议保留一句 [LOG E] |

**英文第一稿**

The four releases we analyze (Recursive's First Steps and Flint, Naive's N0.5, and Apex-35B) all use AI to improve the outputs of AI R&D: training code, GPU kernels, inference engines, and training data. The improver itself is not rewritten in this process. The literature calls this non-recursive optimization or weak self-improvement [T-03, T-12], and under the criterion of Tang et al. (2026) it counts as GPI rather than RSI [T-24]. Our analysis of weights and behavior also shows that the base model or the reasoning traces of all three open-weight models come from external models [model_dissection_apex_naive.md §7; flint_reverse_engineering.md §3].

In the literature, the core requirement of RSI is that the improver "gets better at getting better" [T-12]. Most theoretical work also asks for returns of at least 1 across generations, so that improvement speeds up [T-04, T-16, T-17], although Tang et al. leave this out of their definition [T-24]. Empirically, self-training inside a fixed, human-designed loop usually saturates within 2 to 3 rounds [M2-01, M-20, M-17]. Among the roughly 96 sources we checked, no work measures compounding across generations. The closest one, the AIDE² ignition test, only swaps the scaffold, and its result is inconclusive (0.780 vs 0.782) [E-28].

We ran a weight-level ignition test on a small platform. We replaced a Qwen3.5-9B improver with a successor trained on its own trajectories, and compared how fast each one improves 8 held-out small research tasks, with A/A, placebo, and more-old-data controls. With a fixed improver, AI-assisted R&D works: val_bpb on NanoChat dropped from 1.4152 to 1.2310 [LOG E]. After swapping in the successor, we did not detect faster improvement:
- the successor trained by rejection-sampling fine-tuning scored −0.052 (95% CI [−0.177, +0.041]);
- the second generation scored −0.015 against the original model;
- self-written reflection notes made it significantly worse, at −0.196 (CI [−0.383, −0.033]) [LOG E·dec_us, dec_g1, dec_g2];
- using Qwen3.8-27B as the improver was not faster either (+0.001) [LOG E·dec_g1].

[Reserved: results for generation 3 and U_sd.]


**中文第三稿**

Recursive、Naive 和 Apex 这几家公司都说，自己已经在用 AI 研发 AI。这份报告分析了它们公开的四份材料：Recursive 的 First Steps 和 Flint 模型、Naive 的 N0.5 模型、Apex 的 35B 模型。材料里展示的都是同一类事情：让 AI 去改进训练代码、GPU 程序、推理引擎和训练数据，并且报告了不错的成绩（均为各家自报）。但在整个过程中，负责做改进的那个 AI 本身没有变。按学术界的说法，这还算不上递归自我改进（recursive self-improvement，简称 RSI），更接近"一个系统去优化另一样新的产品"[T-03, T-12]；或者说，改进的方法放在模型外面，由人来定 [T-24]。

三个模型公开了权重，可以直接拆开来看。Apex 是在阿里的 Qwen3.6 上把全部权重继续训练出来的，思考的写法也被整个换掉了，看起来像学了 Claude 的风格，但它从没自称 Claude，所以这只是推断。Naive 是在小米 MiMo-V2.5 的对话版上继续训练的，其中 9 层注意力几乎被重写。Flint 是从零训练的，但它的思考过程里会冒出 "as a Gemini model" 这样的话，说明它学过 Gemini 等外部模型的推理记录 [`model_dissection_apex_naive.md` §1–3, §5；`flint_reverse_engineering.md` §3]。也就是说，这些模型的底子或思考习惯，都来自别人的模型。仅凭公开材料，分不清它们的能力有多少来自自己的自动化研发流程，又有多少来自外部更强的模型〔推断〕。

那么，真正的 RSI 应该是什么样？学术界没有统一的定义 [T-23]，多数理论工作要求两点。第一，做改进的那套方法本身在系统内部，系统可以自己去改它 [T-05, T-12, T-24]。第二，改进能叠加，每一代比上一代改得更快 [T-04, T-09, T-16, T-17]。不过，最近一篇给出正式定义的论文没有要求第二点 [T-24]。用 Yampolskiy 的话说，RSI 就是"越改越会改"（gets better at getting better）[T-12]。现实中离这一步还很远：让模型用自己产出的数据反复训练自己，通常几轮之内就停滞，甚至变差 [M2-01, M-20, M-17]。在查过的约 96 个来源里，还没有一项工作测出过"一代比一代改得更快"。

最接近的一次尝试来自 Weco 的 AIDE² [E-28]。他们有一个专门做 AI 研究任务的 agent，再用一个"教练" agent 反复改写它的代码，改完分数变高才保留。这里的分数是这样算的：让 agent 去做一组研究任务（训练模型、解算法题、改进 agent 的提示词和工作流程），把它交出的结果放到不公开的测试数据上打分，再对所有任务取平均，越高越好。在主实验里，这个分数从最初的 0.703 一路涨到 0.778，超过了人工打造的强 agent（0.749）。接着他们问了最关键的问题：被改好的 agent 自己来当教练，会不会比原来那个人工打造的教练更会改？他们让两个教练从同一个起点出发，各改 50 轮、各跑 3 次。最后两边的平均分是 0.780 和 0.782，几乎一样。作者说，只跑了 3 次，每次结果又差得不少，所以看不出哪个教练更强。而且在这个实验里，换掉的只是 agent 的代码，底下的大模型一直没变 [E-28 §3.2, §3.6, 图 2, 图 7；`notes/aide2_ignition.md`]。

这份报告把这个问题再往前推了一步：不换代码，换模型本身的权重。先让 Qwen3.5-9B 自己做 3 小时研究：训练一个 NanoChat 小模型，共尝试 73 次。它把验证损失（每字节比特数，越低越好）从 1.4152 降到 1.2310，说明 AI 帮忙做研究本身是有效的 [LOG E]。然后用它被采纳的那些尝试记录去微调它自己，得到第一代后继模型；再用同样的办法得到第二代。最后，在 8 个它从没见过的小型研究任务上，比较每一代改进得有多快。为了排除巧合，实验还设了几组对照：同一个模型跑两遍，看随机波动有多大；改用被否决的记录做同样的微调，当作安慰剂；只多给数据，但不换数据的来源。

结果是：后继模型仍然能把任务改好，但并没有比原始模型改得更快 [LOG E]。
- 第一代后继模型比原始模型略慢一点，但差距在统计误差范围内，分不出高低。
- 用成功记录微调和用失败记录微调，效果几乎一样，说明模型并没有从"哪些做法成功了"里学到东西。
- 第二代也没有超过原始模型。
- 另一种做法是让模型读自己的记录、写一份经验笔记给下一轮用，结果它反而改得明显更慢，改进速度掉了约三分之一。这是所有比较里唯一在统计上站得住的结果。
- 换成更大的 Qwen3.8-27B 来当改进者，也没有更快。

〔预留：第三代，以及"让模型自己决定怎么训练自己"的结果〕

**英文第三稿**

Recursive, Naive and Apex all say they now use AI to build AI. This report looks at the four releases they made public: Recursive's First Steps and its Flint model, Naive's N0.5 model, and Apex's 35B model. They all show the same kind of work. AI improves training code, GPU programs, inference engines and training data, and the companies report good results (all self-reported). But the AI that does the improving stays the same the whole time. In the academic literature this is not yet recursive self-improvement (RSI). It is closer to one system optimizing something else [T-03, T-12], or to an improvement process that sits outside the model and is set by people [T-24].

Three of these models have open weights, so they can be taken apart directly. Apex is Alibaba's Qwen3.6 with all of its weights trained further, and its way of thinking was replaced as a whole. It looks like it learned Claude's style, but it never calls itself Claude, so this is only an inference. Naive continued training from the chat version of Xiaomi's MiMo-V2.5, and 9 of its attention layers were almost completely rewritten. Flint was trained from scratch, but its reasoning sometimes says "as a Gemini model", which means it learned from reasoning traces of Gemini and other outside models [model_dissection_apex_naive.md §1–3, §5; flint_reverse_engineering.md §3]. So the base or the thinking habits of these models come from other people's models. From the public material alone, there is no way to tell how much of their ability comes from their own automated research pipeline, and how much comes from stronger outside models (an inference).

So what would real RSI look like? There is no agreed definition [T-23], but most theory papers ask for two things. First, the method that makes the improvements sits inside the system, and the system can change it [T-05, T-12, T-24]. Second, the gains stack, so each generation improves faster than the one before [T-04, T-09, T-16, T-17]. One recent formal definition does not require the second point [T-24]. In Yampolskiy's words, the system "gets better at getting better" [T-12]. In practice this is still far away. When a model keeps training on its own outputs, it usually stalls or even gets worse within a few rounds [M2-01, M-20, M-17]. Among the roughly 96 sources checked for this report, no work has measured one generation improving faster than the one before.

The closest attempt is Weco's AIDE² [E-28]. They have an agent that does AI research tasks, and a "coach" agent that keeps rewriting its code. A rewrite is kept only if the score goes up. The score works like this: the agent works on a set of research tasks (training models, solving algorithm problems, and improving an agent's prompts and workflow), its results are graded on hidden test data, and the grades are averaged over all tasks. Higher is better. In their main run the score rose from 0.703 to 0.778, above a strong hand-built agent at 0.749. Then they asked the key question: if the improved agent becomes the coach, is it better at improving than the original hand-built coach? Both coaches started from the same point and ran 50 rounds, 3 times each. They ended at 0.780 and 0.782 on average, almost the same. The authors say that with only 3 runs and large differences between runs, they cannot tell which coach is better. Also, only the agent's code changed in this test. The language model underneath stayed the same [E-28 §3.2, §3.6, Fig. 2, Fig. 7; notes/aide2_ignition.md].

This report pushes the question one step further: instead of changing the code, it changes the model's weights. First, Qwen3.5-9B did research on its own for 3 hours, training a small NanoChat model over 73 attempts. It brought the validation loss (bits per byte, lower is better) from 1.4152 down to 1.2310, so AI-assisted research does work here [LOG E]. Next, the model was fine-tuned on its own accepted attempts to get a first-generation successor, and the same step was repeated to get a second generation. Finally, the experiment measured how fast each generation improved on 8 small research tasks it had never seen. To rule out luck, the experiment also had controls: running the same model twice to see the random noise, fine-tuning on rejected attempts instead as a placebo, and adding more data without changing where the data came from.

The successors could still improve the tasks, but they were not faster than the original model [LOG E].
- The first generation was slightly slower, but the gap is within the statistical error, so the two cannot be told apart.
- Fine-tuning on successful attempts and on failed attempts gave almost the same result, so the model did not learn from which ideas worked.
- The second generation did not beat the original model either.
- A different approach let the model read its own records and write lessons for the next round. It made the model clearly slower: its improvement speed dropped by about a third. This is the only result in all the comparisons that holds up statistically.
- Using the larger Qwen3.8-27B as the improver was not faster either.

[Reserved: results for the third generation, and for letting the model decide how to train itself.]


**§1 正文第一稿（被第二稿取代）**

#### 正文（中文，第一稿：Claude 起草，按摘要的写法，供你修改）

递归自我改进（recursive self-improvement，RSI）这个想法比大模型早了六十年。1965 年，Good 提出：设计机器本身也是一种智力活动，所以一台在所有智力活动上都超过人类的机器，可以设计出更好的机器，从而引发 "intelligence explosion" [T-01]。此后的讨论都绕着同一个问题：什么样的改进才算"递归"。学术界至今没有统一定义，ICLR 2026 专门讨论 RSI 的 workshop 也没有给出形式定义 [T-23]。

Yudkowsky 的说法最直白：递归的意思是，做优化的和被优化的是同一个东西，也就是 AI 重新设计自己的认知算法 [T-03]。他用债券打比方：利息花掉是线性增长，拿去再投资才是复利。按这个标准，自然选择和人类文明虽然都在累积，但都不算递归，因为大脑本身没有被改写。一个部件去优化另一个部件，只能叫 non-recursive optimization [T-03, T-12]。

[[figure:recursion-loop｜图 1　两种改进。(a) 改进者固定不变，反复改进别的东西，对应 non-recursive optimization [T-03]。这里回放的是一次真实运行：Qwen3.5-9B 在 3 个并行线程里改进 NanoChat 的训练脚本，共 72 次尝试；每次尝试都要经过提出补丁、沙箱训练、封存评估、保留或丢弃这几步 [LOG E；`replay/final/ovn1118b-nc-g0/events.jsonl`]。(b) 改进者改进它自己：每一代把 (a) 整个跑一遍，用留下的记录更新自己，再由下一代接手 [T-03, T-12]；各代用时是假设，只用来说明"越改越会改"。(c) 回报率 r 决定进展会不会越来越快：研发由 AI 自己完成时，r = 0.5 时每次翻倍越来越慢（软件效率随时间线性增长，就像把利息花掉），r = 1 时每次翻倍用时相同，r > 1 时每次翻倍越来越快，并在有限时间内发散 [T-16, T-17]；曲线和翻倍点由公式算出。]]

Yampolskiy 把"系统修改自己"分成三级 [T-12]：

| 级别 | 含义 | 原文的例子 |
|---|---|---|
| self-modification | 改了自己，但没有变好 | 代码混淆 |
| weak self-improvement | 在固定的目标下优化参数 | 遗传算法、遗传编程 |
| recursive self-improvement | 越改越会改（gets better at getting better），能反复把旧方法换成全新的方法 | 原文没有给出例子 |

他还指出，单纯加内存、加硬件不算 RSI，因为系统并没有变得更会改进自己。大多数自我应用的系统最后都会停下来：supercompilation（一种程序变换技术：追踪原程序所有可能的运行过程，再编译出一个功能相同、但去掉了冗余计算的新程序 [S-01]）用在自己身上，第一次约提升 20%，第二次约 3%，之后不再有改进 [T-12]。

把各家的定义放在一起，可以抽出四条共同点。第一，做改进的那套方法本身在系统内部，并且能被系统自己改写 [T-05, T-12, T-24]；Schmidhuber 的 Gödel machine 把这一条推到了极致，连负责证明改进有效的搜索器也可以被改写 [T-05]。第二，被提升的必须是"产生下一代的能力"本身，Chalmers 称之为自我放大的能力 [T-09, T-12]。第三，改进能够叠加，每一代的回报不低于上一代 [T-04, T-09, T-16, T-17]。第四，系统要足够通用，这一条争议最大：Good 和 Hall 认为需要 [T-01, T-07]；Chalmers 和 Yudkowsky 认为，低于人类水平的系统也可能启动 [T-09, T-04]；Eth 和 Davidson 的折中是，只要求在 AI 研发这个领域内通用 [T-17]。

2026 年，Tang 等人给出了一个把两类改进正式区分开的定义，也是检索范围内唯一的一个 [T-24]：改进 agent 的机制如果在 agent 外部，就是一般的迭代策略改进（GPI）；在 agent 内部，才是 RSI。这篇论文没有把复利写进定义，而是把它当作需要实验回答的问题。

按前三条判据，self-distillation 和 self-play 这类方法最多算 weak self-improvement：训练数据由模型自己产生，但怎么训练是人写死的〔推断〕。文献里没有来源直接给 self-distillation 归类。Mahoney 是唯一点名 self-play 的，他把它算作自我改进，同时证明，在没有外部输入的情况下，第 n 代程序的复杂度最多按 log n 增长 [T-08]。

复利能不能出现，取决于回报率。Epoch 和 Forethought 用 r 来描述：累积的研发投入每翻一倍，软件效率翻几倍；r 大于 1，进展才会越来越快 [T-16, T-17]。历史估计里，国际象棋引擎约为 0.83 [T-16]，图像识别约 1.4、线性规划约 1.1、SAT 求解器约 3.5，大多在统计上不显著 [T-17]。Forethought 的总体判断是，在算力不变时 r 约为 0.5 到 2；大语言模型目前没有估计值 [T-17]。

这些定义落到实验上，问的是同一件事：把改进出来的后继者放回去当改进者，它是不是改得更快。Weco 把这一步叫作 ignition [X-24]，AIDE² 据此设计了 ignition test [E-28]。HGM 提出用后代的表现来衡量一个 agent 的改进能力，称为 metaproductivity [E-27]。STOP 的作者也明确说，由于底层模型没有变，他们的方法 "not full recursive self-improvement" [M-01]。本报告按前三条判据来判断四份材料和实验，不把通用性当作门槛；检验的方法就是 ignition test。

#### Text (English, draft 1)

The idea of recursive self-improvement (RSI) is sixty years older than large language models. In 1965, Good argued that designing machines is itself an intellectual activity, so a machine that surpasses humans at every intellectual activity could design better machines and set off an "intelligence explosion" [T-01]. Every later debate has circled the same question: what kind of improvement counts as recursive? There is still no agreed definition, and the ICLR 2026 workshop on RSI did not give a formal one [T-23].

Yudkowsky puts it most plainly. Recursion means that the optimizer and the thing being optimized are the same: an AI redesigning its own cognitive algorithms [T-03]. He compares it to a bond. Spending the interest gives linear growth; reinvesting it gives compound growth. By this standard, natural selection and human civilization both accumulate, but neither is recursive, because the brain itself is not rewritten. One component optimizing another is only non-recursive optimization [T-03, T-12].

[[figure:recursion-loop｜Figure 1. Two kinds of improvement. (a) A fixed improver keeps improving something else, i.e. non-recursive optimization [T-03]. The panel replays a real run: Qwen3.5-9B improving NanoChat training code in 3 parallel threads over 72 attempts, each going through propose a patch, sandbox training, sealed evaluation, and keep or discard [LOG E; replay/final/ovn1118b-nc-g0/events.jsonl]. (b) The improver improves itself: each generation runs (a), updates itself from the records, and hands over to the next [T-03, T-12]; the loop times are assumed and only illustrate getting better at getting better. (c) The rate of return r decides whether progress speeds up. With the AI doing the research, each doubling takes longer when r = 0.5 (efficiency grows linearly, like spending the interest), every doubling takes the same time when r = 1, and doublings get faster with a finite-time blow-up when r > 1 [T-16, T-17]. Curves and doubling marks are computed from the equations.]]

Yampolskiy divides a system changing itself into three levels [T-12]:

| Level | Meaning | Example in the source |
|---|---|---|
| self-modification | the system changes itself without getting better | code obfuscation |
| weak self-improvement | parameters are optimized under a fixed objective | genetic algorithms and genetic programming |
| recursive self-improvement | the system gets better at getting better, and can repeatedly replace old methods with entirely new ones | — |

He adds that more memory or more hardware does not count, because the system has not become better at improving itself. Most self-applied systems also stop: supercompilation (a program transformation that traces the possible runs of the original program and compiles an equivalent program with the redundant computation removed [S-01]) applied to itself gives about 20% the first time, about 3% the second time, and nothing after that [T-12].

Put together, the definitions yield four criteria. First, the method that makes the improvements sits inside the system, and the system can rewrite it [T-05, T-12, T-24]. Schmidhuber's Gödel machine takes this furthest: even the searcher that proves an improvement is useful can be rewritten [T-05]. Second, what gets better must be the ability to produce the next generation, which Chalmers calls a self-amplifying capacity [T-09, T-12]. Third, the gains stack, with returns across generations of at least 1 [T-04, T-09, T-16, T-17]. Fourth, the system has to be general enough. This is the most disputed point. Good and Hall require it [T-01, T-07]; Chalmers and Yudkowsky think a system below human level could also start the process [T-09, T-04]; Eth and Davidson take a middle position and only require generality within AI research [T-17].

In 2026, Tang and colleagues gave the only formal definition in the reviewed literature that separates the two kinds of improvement [T-24]. If the mechanism that improves an agent sits outside the agent, the process is generalized policy iteration (GPI); if it sits inside, it is RSI. The paper leaves compounding out of the definition and treats it as a question for experiments.

Under the first three criteria, self-distillation and self-play count at most as weak self-improvement: the model produces its own training data, but people fix how the training is done (an inference). No source in the review classifies self-distillation directly. Mahoney is the only one who names self-play. He counts it as self-improvement, and also proves that without outside input, the complexity of the n-th generation grows at most like log n [T-08].

Whether gains can compound depends on the rate of return. Epoch and Forethought describe it with r: how many times software efficiency doubles each time cumulative research input doubles. Progress speeds up only when r is above 1 [T-16, T-17]. Historical estimates put computer chess at about 0.83 [T-16], and image recognition at about 1.4, linear programming at about 1.1 and SAT solvers at about 3.5, most of them not statistically significant [T-17]. Forethought's overall judgment is that r lies between about 0.5 and 2 when compute is held fixed. There is no estimate for large language models yet [T-17].

In experiments, all of these definitions ask the same thing: if the improved successor is put back in as the improver, does it improve things faster? Weco calls this step ignition [X-24], and AIDE² built an ignition test around it [E-28]. HGM proposes measuring an agent's ability to improve by the performance of its descendants, which it calls metaproductivity [E-27]. The STOP authors also say that, because the underlying model does not change, their method is "not full recursive self-improvement" [M-01]. This report judges the four releases and the experiment by the first three criteria, does not treat generality as a requirement, and uses the ignition test as the check.

</details>

## 1. 问题与定义：什么才算 RSI

> **正文已移到 `report/sections/01_rsi.md`**（中英文都在那里），以下是素材和核对记录。

#### 你改写后的第 1 节：核对意见（11-18 11:30 版）——**七条已于 11-18 按你的同意改入正文**；改动前的版本存在 `sections/archive/01_rsi.v5_user.md`

| # | 位置 | 问题 | 依据 / 建议 |
|---|---|---|---|
| 1 | "EURISKO 与优化编译器……在少数几次迭代后很快停滞" / "stall after only a few iterations" | 出处只说"递归但很快熄火"，没有"少数几次迭代"这个细节 | 删去"在少数几次迭代后"，保留"很快停滞" [T-03；`lit_rsi/01_theory.md` 第 17 行] |
| 2 | "至于停滞的原因，可以通过 Yampolskiy 的分级来解释" / "Why such systems stall becomes clearer through Yampolskiy's taxonomy" | 因果关系是新加的：Yampolskiy 的分级本身没有用来解释 EURISKO 为何停滞 | 改成并列的过渡，例如"Yampolskiy 将系统对自身的修改分为三级" |
| 3 | "AI 研发的 r 介于 0.5 到 2" / "r for AI research" | 出处说的是**软件**的 r（考虑硬件恒定后约 0.5–2），不是"AI 研发" | 改为"软件的 r" / "r for software" [`lit_rsi/01_theory.md` §3.2] |
| 4 | "一般的广义策略迭代（GPI）" / "generalized policy iteration (GPI)" | GPI 的英文全称在笔记里没有核对过（这一处我之前写的英文也有同样的问题）；笔记里的说法是"一般的迭代策略改进" | 建议只写 GPI，加上笔记里的中文解释，不展开英文全称；需要的话我再去核对 Tang et al. 原文 [T-24] |
| 5 | "逼近物理或理论上限" / "physical or theoretical limits" | 出处只说"逼近上限"（Davidson & Houlden 讲的是软件效率的有效上限），"物理或理论"是新加的 | 改为"逼近上限" [`lit_rsi_academic.md` §3] |
| 6 | 英文 "we compare"、"our own experiments" | 违反"不用 we / 我们"的规则 | 改为被动语态，或用 "this report" |
| 7 | "近期四份主流报告" / "four landmark reports"、"four benchmark reports" | "主流""landmark"是评价性用词；"benchmark reports"不准确，它们不是基准测试报告 | 改为"四份报告" / "the four reports" |


> **✍️ 你来写**：说明本文采用文献中哪一种定义、为什么。H1 那个疑问（RSI 应该是整个模型变强，并且改进能力本身也变强）在文献里对应的是 Yampolskiy 的 strong / recursive 一级 [T-12]，加上有争议的通用性判据 [T-01, T-07]。建议以下面的"文献分级表"作为本节主图。
>
> TODO

**📎 素材：文献中的定义和分级**（只用文献术语；键见 `report/references.md`）
- **没有统一定义**，这一点本身要写明 [`lit_rsi_academic.md` §1；T-23：ICLR 2026 RSI workshop 没有给出形式定义]。
- **Yampolskiy 2015 的三级分类**（📄 主表候选）[T-12]：

  | 级别（原文术语） | 含义 | 原文举例 |
  |---|---|---|
  | self-modification | 改了自己，但没有改进 | 代码混淆 |
  | weak self-improvement | 在固定的 fitness / 目标下优化参数 | GA/GP |
  | recursive（strong）self-improvement | "gets better at getting better"，能反复把算法换成全新的方法 | — |

  - ⚠️ metalearning 归哪一级：`01_theory.md` 的来源表（第 26 行）写它被归为 non-recursive optimization，同一文件 §2.1（第 51 行）写成 weak。引用前要查 Yampolskiy 原文。

- **non-recursive optimization**：一个组件优化另一个组件，就不是递归 [T-03 Yudkowsky 2008；T-12 转引]。四份材料里"AI 改 kernel、改训练配方"属于这一类〔推断，需你确认〕。
- **GPI 和 RSI**（Tang et al. 2026，preprint）[T-24]：
  - 改进 agent 的机制在 agent 外部是 GPI（generalized policy iteration 一类的迭代策略改进），在内部才是 RSI；
  - 按评价标准是否锚定在系统之外，又把 RSI 分为 Anchored、Goal Drift、Fully Self-Referential 三类；
  - 它**不把复利放进定义**，只当作经验问题。
- **复利 / 回报率**：
  - Yudkowsky 2013 的 k（"returns on cognitive reinvestment"）[T-04]；
  - Chalmers 的 proportionality [T-09]；
  - Epoch 和 Forethought 的 r：r > 1 时会加速 [T-16, T-17]。
- **通用性是否必要**：有争议。Good、Hall 认为需要 [T-01, T-07]；Chalmers、Yudkowsky 2013 认为不必要 [T-09, T-04]；Eth & Davidson 的折中是"在 AI 研发这个领域内通用"[T-17]。
- **"改进能力本身变强"在实证文献里的叫法**：
  - Weco 的分级把它叫 Level 2："ignition, when the system improves its own ability to improve itself" [X-24，公司博客]；
  - AIDE² 把"换上自我改进的产物当外层改进者、和原改进者比"的实验叫 **ignition test** [E-28，preprint]；
  - HGM 提出 clade-level **metaproductivity**（CMP），用后代的表现衡量一个 agent 的改进能力 [E-27, M-04]；
  - STOP 用 **meta-utility** 衡量改进者改进自己的效果，作者明确说因为底层 LM 没变，"this is not full recursive self-improvement" [M-01, E-25]。
- **self-training 一类方法在文献里的位置**：self-distillation、self-play 是固定的训练算法加上模型自己生成的数据，按 Yudkowsky / Yampolskiy 一派的标准更接近 weak 或 non-recursive [`lit_rsi/01_theory.md` §2.1，标为【推断】]；Mahoney 是唯一直接点名 self-play 的来源 [T-08]。**没有任何来源直接对 self-distillation 下判断** [`lit_rsi_academic.md` §2]。
- ⚠️ 已删除的非文献术语：内环 / 外环、换心脏、A/B/C 三类、"一阶 / 二阶"。这些来自 `REPORT_LOG` A 节和 H2 节的讨论，不是文献定义 [LOG A："用户的框架"；LOG H2〔讨论〕]。


#### §1 开头：问题标注与修改稿

| 原文 | 问题 | 改法 |
|---|---|---|
| "模型的自我迭代分为两个环" | "两个环"不是文献术语（来自 LOG A 的"用户的框架"）；"自我迭代"也不是文献用语 | 删掉，换成下面这句，用文献分级给材料和实验定位 |

**中文**：学术界对 RSI 没有统一的定义 [`lit_rsi_academic.md` §1；T-23]。本文采用两套文献框架，来给四份材料和我们的实验定位：一是 Yampolskiy (2015) 的三级分类（self-modification、weak、recursive self-improvement）[T-12]；二是 Tang et al. (2026) 的判据，即改进机制在 agent 外部是 GPI，在内部是 RSI [T-24]。

**English**: There is no agreed definition of RSI in the literature [lit_rsi_academic.md §1; T-23]. We use two frameworks from the literature to place the four releases and our experiment: the three levels of Yampolskiy (2015), namely self-modification, weak self-improvement, and recursive self-improvement [T-12], and the criterion of Tang et al. (2026), under which an improvement mechanism outside the agent is GPI and one inside the agent is RSI [T-24].

<details><summary>原文</summary>

模型的自我迭代分为两个环

</details>
---

## 2. 四份材料：他们说了什么，我们核实到什么

> **正文在 `report/sections/02_releases.md`**（中英文都在那里），以下是素材和核对记录。

> **✍️ 你来写**：每份材料一段，写三点：声称什么、核实到什么、按 §1 的文献分级落在哪一级。最后一段写跨材料的共同点。
>
> 网页里每份材料是一张卡片，分三栏；PDF 里压缩成一张矩阵表加 4 个短段落。
>
> TODO（总起段）

### 2.1 Recursive · First Steps Toward Automated AI Research

> **✍️** TODO

**📎 声称（全部是自报）**
- NanoChat：0.9372 → 0.9109 BPB [RF-05, RF-06]，即同等 loss 下训练时间约少 1.3 倍 [RF-08]。
- NanoGPT Speedrun：79.7 s → 77.5 s [RF-17]。repo 的数字是 77.3 s 和 77.34 s，与博客不一致 [RF-28, RF-34]；结果未提交官方 leaderboard [RF 待核 6]。
- SOL-ExecBench：0.699 → 0.754，共 235 个 kernel [RF-25, RF-26, RF-23]。

**📎 核实与缺口**
- 系统本身没有公开；用的什么模型、迭代多少次、算力多少都没有披露 [RF 基本信息；RF 待核 9]。
- 公开的只是"答卷"：训练代码、235 个 kernel 里的 10 个、每个 seed 的 BPB [RF-29；LOG B]。
- SOL 部分没有附正确性检查 [LOG B]。
- 作者自己说，两个 weak start 实验里，底层模型可能已经见过这些公开技术 [RF 待核 8]。
- 是否改写过自身脚手架，原文没有披露 [RF 待核 9；LOG H2]。
- 仓库里能看到进化式搜索的痕迹：kernel 名编号最大到 w204，另有 W591、"v49 best"、CROSSOVER 等字样 [LOG F]。
- ⚠️ **SOL 分数冲突**：Naive 引用 Recursive 的 SOL 是 67.56，Recursive 自己写的是 0.754，还没有回源 NVIDIA 官方榜核对 [NV-23；`reading/00_总览.md` §4 ★1]。

**📎 分级**：改进对象是训练代码和 kernel，改进者本身不变，属于 non-recursive optimization [T-03, T-12]；按 Tang 的判据是 GPI [T-24]。RSI 只出现在愿景里 [`reading/00_总览.md` §3]。

### 2.2 Naive · N0.5-Flash / NaiveRT / AutoWM

> **✍️** TODO

**📎 声称（全部是自报）**
- 309B MoE，激活 15.5B [NV-02, NV-03]；基于 MiMo-V2.5 [NV 基本信息]。
- NaiveRT：151 个 trial，采纳 63 个（41.7%）[NV-46, NV-60, NV-61]；完整投机解码一轮从 12.3 ms 降到 3.4 ms [NV-43]；峰值单流 decode 2,122 tok/s [NV-41]。
  - 这个峰值的条件是：最好的 1 秒窗口、关闭 thinking、不含 prefill [NV-67；NV 待核 9]。
- AutoWM：累计研究 400 小时，WorldArena 得分 77.43，当时榜首是 73.64 [NV-69, NV-71, NV-72]。
- 和 Recursive 正面比：0.9051 对 0.9109 BPB，73.8 s 对 77.5 s [NV-24, NV-25]。

**📎 核实与缺口**
- **人在关键节点**：MoE fusion 实现了七轮，端到端都变慢，由人叫停 [NV-66]。
- 和 Recursive 比较时，Naive 用的是自家 in-house harness，没有给 seed 数和方差 [NV 待核 2；`reading/00_总览.md` ★4]。
- AutoWM 用评测指标本身做输出选择，没有拆分各组件的贡献 [NV 待核 4]。
- 图的坐标轴是截断的，条长和数值不成比例 [NV 待核 3]。
- **权重逆向**：父模型是 MiMo-V2.5 的**后训练版**，不是 Base [`model_dissection_apex_naive.md` §2]。
  - 9 个 DSA 层的 q/k/v 几乎被重写，v_proj 改动中位数 51.9%；SWA 层的改动约 10% [同上 §3]。
  - 新增了 189 个 indexer 张量 [同上 §3]。
- **config 和博客矛盾**：博客说"把 MLA 换成了 GQA"，但基座本来就是 GQA；"两种注意力都有 sink bias"的说法也和 config 不符 [LOG B]。
- 行为指纹已经测完（11-18），结果见 `research/model_dissection_apex_naive.md` §8、REPORT_LOG F 节"Naive 行为指纹"，汇总数据在 `research/census_compare.json`，原始数据在 `research/naive/`。**写这一节时再取来核对，草稿里还没有写入结论。**
  - ⚠️ 协调者提醒：Naive 的身份回答来自 chat template 注入的默认 system prompt，不能当作权重层面"身份补丁"的证据。

**📎 分级**：NaiveRT、架构搜索属于 non-recursive optimization [T-03, T-12]；"把模型训练成会做研发、让下一代承担更多研究"的说法接近 weak self-improvement，但没有给出逐代数据 [`reading/00_总览.md` §3；LOG H3]。

### 2.3 Recursive · Flint

> **✍️** TODO

**📎 声称（全部是自报）**
- 40B 总参数，激活 2.5B，不到 4 个月从零构建 [FL-01, FL-02, FL-03]。
- NVFP4 版与 16-bit 版差距在 0.7 分以内 [FL-21]。
- 原文明确写 "Our bet is…"：递归被写成一个押注 [FL 待核 7]。
- Eureka Machine 的贡献比例没有量化 [FL 待核 7]。

**📎 核实**
- 参数实测：总量 39.785B，激活 2.515B，和 card 一致 [`flint_dissection.md` §1]。
- Tokenizer 是自己训练的，15 万词里只有 51 个欠训练词，支持"从零训练"[`flint_reverse_engineering.md` §1–2]。
- **后训练用了 Gemini 的推理轨迹**：思考段里原文出现 "as a Gemini model…"、"Since I am a Google AI"；不同任务类别分别用了 gpt-oss、Gemini、R1 风格的 teacher [`flint_reverse_engineering.md` §3, §5–6；LOG F]。
- 知识截止：模型自报 2025-05，card 写 2026-06 [FL-36；`flint_reverse_engineering.md` §2.5]。
- 上下文长度：博客写 512k，config 是 262,144 [FL-16, FL-34]。
- 同一个模型换了 harness 和测量方，分数就不同：TB2.1 自报 54.9，Apex 实测 59.6；SWE-Pro v2 自报 89.7，Apex 实测 85.2 [FL-24, FL-25, AX-T13, AX-T14]。
- 推理（M4 Max，rec，greedy）：1k 上下文 196 tok/s，104k 上下文 66 tok/s [LOG C；`rec_optimization_analysis.md` §6–7]。
  - 我们自己实现的 prompt-lookup 投机解码，在 copy 型任务上快 1.14–1.67×，但只能通过容差门槛，过不了逐位一致的门槛 [LOG C；`ourengine_report.md`]。

**📎 分级**：non-recursive optimization [T-03, T-12]；"改进能力本身提升"（ignition [X-24]）在原文里只是押注。

### 2.4 Apex · 35B-V1.0

> **✍️** TODO

**📎 声称（全部是自报）**
- 35B 总参数，约 3B 激活，基于 Qwen3.6-35B-A3B [AX-01, AX-02]。
- 有一节叫 "Building … Through Recursive Self-Improvement"：
  - self-evolving data 让内部 CFD-Bench 提升 +32.1 分，单位不明 [AX-37]；
  - 训练吞吐提高 45–60%，decode 快 1.6–1.8×，TTFT 降低 30–35%，都没有给基线 [AX-38–40；AX 待核 7]。
- AI4AI：Sokoban 成功率 24.4% 对 11.7% [AX-25]。
  - 11.7% 很可能是被训练模型的起点，而不是 Base 研究者训练出来的结果 [`reading/00_总览.md` ★3]；Base 研究者在 101 分钟后退出 [AX-29]。
- 所有评测都只跑一次（pass@1）[AX-43]；三个 internal benchmark 无法外部复现 [AX 待核 9]。

**📎 核实**
- config 与 Qwen3.6 逐字段一致，tokenizer 哈希相同，权重哈希全部不同 [LOG B；`artifacts_weights_configs.md`]。
- 是**全参数继续训练**：每个 routed 专家都改动了 6–21%，router 改了 9% [`model_dissection_apex_naive.md` §1]。
- **思考风格整个换掉了**：Qwen3.6 有 70/80 条以 "Here's a thinking process:" 开头，Apex 是 0/80，改成了 "The user asks/wants…"（72/80）；思考长度至少缩短 3.4 倍 [同上 §5；`census_compare.json`]。
  - 〔推断〕这和用 Claude 式轨迹蒸馏一致，但没有自我指认的泄漏。
- 被问到时，它主动承认基座是 Qwen3.6 [同上 §5.2]。
- 人工事后标注：207 条命令中有 36 条高质量动作 [AX-24；AX 待核 10]。

**📎 分级**：宣称是 RSI，但没有逐代证据。self-evolving data 的做法最多对应 weak self-improvement [T-12]；按 Tang 的判据，改进机制在系统外部，属于 GPI [T-24]。

### 2.5 共同点：基座与 teacher 的来源

> **✍️ 你来写**：一段话。建议的落点是：三家的基座和思考风格都来自外部，权重里看不到"第 k 代改出第 k+1 代"的谱系。
>
> TODO

**📎 素材**（📄 一张 3 列小表）[`model_dissection_apex_naive.md` §7]

| | Flint | Apex | Naive |
|---|---|---|---|
| 基座 | 从零训练 | Qwen3.6-35B-A3B | MiMo-V2.5 后训练版 |
| 思考风格来源 | 按任务类别分配 teacher：gpt-oss / Gemini / R1 | 统一的 "The user…" 式 | 待测 |
| 自我指认泄漏 | 有 | 无 | 待测 |

- 四份材料都停在 non-recursive optimization 或 weak self-improvement，**没有一家展示 recursive self-improvement**：没有逐代对比，也没有改进速率的测量 [LOG H2；T-12]。

---

## 3. 共同缺口：没有人测过"后继模型当改进者是否更快"（ignition）

> **正文在 `report/sections/03_gap.md`**（中英文都在那里），以下是素材。

> **✍️ 你来写**：为什么 ignition（后继模型当改进者，改进速率是否提高 [X-24, E-28]）是区分 RSI 的关键，为什么没人测。可以用 Amdahl 论证说明推理加速和 RSI 关系不大；再引学术界的现状，说明不只是工业界没测。
>
> TODO

**📎 素材**
- **推理加速对研究循环的反身作用很小**：G0 里评测占 agent 时间的 85.3%，思考占 14.6% [`replay/final/nc-g0-final.summary.txt`]。早期回放是 87.4% 对 12.5% [LOG E]。
  - 所以 agent 推理快 2 倍，研究循环只快约 6%〔估算，LOG H2〕。
  - ⚠️ 这个"约 6%"是按早期回放的思考占比 12.5% 算的（时间节省 6.25%，即约 1.07×）。按最终的 14.6% 算，时间节省约 7.3%，即约 1.08×〔估算：0.853 + 0.146/2 ≈ 0.926〕。写正文前统一用一个口径，正式数字由脚本计算。
- **学术界也没测到复利** [`lit_rsi_academic.md` §1, §4–5]：
  - 人设计的固定循环下的自我训练 2–3 轮就饱和（Mind the Gap [M2-01]；SPIN [M-20]；ReST-EM [M-17]）。
  - 改写自身代码的系统（STOP、DGM、HGM [M-01, M-03, M-04]）全部冻结了底层模型权重。
  - 最接近的测量是 **AIDE² 的 ignition test** [E-28]：只换脚手架，3 seeds，0.780 对 0.782，作者自己说结论不确定 [E-28]。
  - **DGM** 的消融：完整版 50.0/38.0，固定 meta agent 的版本 39.0/28.0（Table 1）。测的是最终分数，不是速率 [M-03]。
  - **HGM**：benchmark 分数和"能产出更好后代"只是弱相关，Pearson 0.285–0.444 [M-04]。
- **前沿实验室测的是整体进展速率**：METR 估计 Anthropic 自 2025-10 起的进展速率约为此前的 1.9 倍（95% CI 1.5–2.8）。这是实验室整体的加速，不是"后继模型当研究者是否更快"[E-23]。
- 文献核查的提醒：数值客观不等于难以作弊；硬信号是必要条件，不是充分条件 [LOG A；`lit_signal_hardness.md`]。

---

## 4. 我们的测量：RSI-mini

> **✍️ 你来写**：本节的总起段：我们做了什么，为什么这样设计。下面各小节的数字和设计说明 Claude 已经整理好，你可以直接改写。
>
> TODO

### 4.1 设计

> **✍️** TODO（建议 1 段 + 1 张示意图 F8）

**📎 素材** [`platform_design.md` §0–1；LOG E]
- 改进者 I =（权重 θ，脚手架 σ）。更新算子 U 有两种，各自对应文献里的一类做法：
  - **U_w**：用被采纳的轨迹做 LoRA 微调。机制上是拒绝采样微调 / Expert Iteration [M-14 STaR, M-17 ReST-EM, X-27 Expert Iteration]；
  - **U_s**：模型读自己的轨迹，改写一份 lessons 笔记，权重不动。对应自我反思 / 经验记忆 [M-23 Reflexion, M-25 ExpeL]。
- **实验设计对应 AIDE² 的 ignition test**：把改进者换成它的后继者，在 held-out 任务上、同起点、同预算比较 [E-28]。区别是我们换的是权重，AIDE² 换的是脚手架。
- **符号定义**（实验变量，不是术语）：固定改进者时，看 best-so-far 曲线；Δ₂(k) = mean_j [nAUC(I_{k+1}; j) − nAUC(I_k; j)]，j 遍历 held-out 任务，按任务配对 [`platform_design.md` §1.3]。
- **nAUC**：逐 trial 平均"已关闭的差距比例"，越高越好 [LOG E·dec_us]。
- **判定规则事先写死**，三条同时满足才算"观察到 ignition" [`platform_design.md` §1.4]：
  1. 95% CI 下界 > 0；
  2. Δ₂ 大于 A/A 差值的 2 倍；
  3. 打赢 placebo。
- **对照**：A/A、placebo、检索、held-out（ID 和 OOD）、"更多 M0 数据" [`platform_design.md` §1.5]。
  - "用被拒绝的轨迹训练当 placebo"这种对照，在我们的检索范围内没有找到先例 [`lit_rsi_academic.md` §6]。
- **规模**：µAR 的 8 个 held-out 任务 × 5 个 seed × 每个 run 12 个 trial [LOG E·dec_us]。
  - A/A：任务间标准差 0.031，最小可检测效应 0.031 [LOG E·dec_us]。
- **门槛测试**：
  - 工具调用（G1b）：9B 调用有效率 98.6% [LOG E]。
  - 噪声地板（G1a）：20 个 seed，dev BPB 1.4007 ± 0.0268 [LOG E]。
  - 沙箱：12 项隔离检查全部通过 [LOG E]。
- **偏离**：Recursive 用 B200 + FA3，vanilla 为 1.0587；我们用 A100 + SDPA [LOG E；RF-30]。

### 4.2 固定改进者的 AI 辅助研发：Qwen3.5-9B 在 NanoChat 上（G0）

> **✍️** TODO（建议 1 段 + 图 F9；🌐 网页嵌入回放视频）

**📎 素材** [`replay/final/nc-g0-final.summary.txt`；LOG E]
- 9B + OpenCode，3 个线程，176.6 分钟，72 次尝试：保留 25、拒绝 47，采纳率 34.7%（25/72）；trial 0 是起点基线，不计入。评测崩溃 9 次，审计标记 0 次 [LOG E，11-18 协调者更正]。
- val_bpb 1.4152 → **1.2308**（trial 72，下降 0.1844）。噪声地板 1.4007 ± 0.0268，终点约低 6.3 个标准差〔推算：(1.4007 − 1.2308) / 0.0268 ≈ 6.34〕。
- ⚠️ 回放摘要 `nc-g0-final.summary.txt` 里的 1.23099 是 trial 68 的评测值：脚本按判定顺序取了某个线程当时的最好值，没有跨线程取最小。回放视频和 HTML 只是可视化，不作为数字出处 [LOG E，协调者 11-18 说明]。
- 规模：460 次 LLM 调用，403 次工具调用，685 万 token；评测线占用 84.5%。
- 可以对照 NaiveRT 的采纳率 41.7% [NV-61]。⚠️ 两者条件不同，只能当量级参考。
- 〔预留〕sealed 集终点确认。
- 〔原始数，未分析，不要写成结论〕Claude Code 当改进者（nc_g0_cc）：24 次尝试，1.4501 → 1.2095 [LOG E，11-18 协调者]。注意起点基线和 9B 那次不同（1.4501 对 1.4152）。
- 〔原始数，未分析，不要写成结论〕从 G0 前沿继续（cf_*）：12 组里完成了 11 组，组内 3 次重复之间差异很大 [LOG E，11-18 协调者]。
  - 〔分析进行中〕协调者 11-18 约 12:30：脚本已写好，最终结果等安慰剂组第 3 次跑完（约 12:45），之后写入 REPORT_LOG E 节并导出 `research/decisions/cf_analysis.json`。口径：降幅 = 各次自己的重测基线 − 最好值；另有 auc_gain；NanoChat 没有 s_ref，不算 nAUC。各组与原始模型做非配对比较：均值差、bootstrap 95% CI、精确置换 p。每组 3 次，最小可能的 p 为 0.10。初步结果不要引用。

### 4.3 Ignition test：自我反思笔记、拒绝采样微调、第 2 代

> **✍️** TODO（建议 2 段 + 森林图 F10 + 逐任务热图 F12）

**📎 素材**

| 对比 | Δ（nAUC） | 95% CI | p | 出处 |
|---|---|---|---|---|
| U_s：A1（M0 + lessons）对 A0a（M0） | **−0.196** | [−0.383, −0.033] | 0.11（区间不含 0，但置换检验不显著） | LOG E·dec_us；decisions_0905 |
| U_s 对 placebo 笔记 | −0.239 | [−0.420, −0.075] | 0.008，8/8 个任务为负 | LOG E·dec_us |
| U_w：M1 对 M0 | −0.052 | [−0.177, +0.041] | 0.49 | LOG E·dec_g1 |
| M1 对 M1p（用被拒轨迹训练的 placebo） | +0.004 | [−0.084, +0.080] | 0.91 | LOG E·dec_g1 |
| U_w 对 U_s | +0.144 | [−0.015, +0.335] | 0.20 | LOG E·dec_g1 |
| 第 2 代：M2 对 M1（Δ₂(1)） | +0.037 | [−0.032, +0.129] | 0.45 | LOG E·dec_g2 |
| M2 对 M2c（"更多 M0 数据"对照） | +0.039 | [−0.042, +0.140] | 0.20 | LOG E·dec_g2 |
| M2 对 M0 | −0.015 | [−0.107, +0.048] | 0.70 | LOG E·dec_g2 |
| 第 3 代：M3 对 M2（Δ₂(2)） | **−0.087** | [−0.190, −0.005] | 0.0625（平台判为 harmful） | decisions_0905·dec_g3 |
| M3 对 M3c（"更多 M0 数据"对照） | −0.066 | [−0.164, +0.025] | 0.19 | decisions_0905·dec_g3 |
| M3 对 M0 | −0.102 | [−0.247, +0.009] | 0.15 | decisions_0905·dec_g3 |

- 各组 nAUC 均值：M0 0.588，A1 0.392，M1 0.537，M2 0.573，M2c 0.534，**M3 0.487** [LOG E；`research/decisions/decisions_0905.json`]。
- U_s 的伤害集中在 cls2d（−0.58）、ood-copyrev（−0.56）、ood-icl（−0.49）、modadd（−0.20）[LOG E·dec_us]。
- lessons 本身只有 252 词，内容是一份固定流程（先换 cosine，再降 weight decay……），其中还有一条 "Ignore changes that increase loss even slightly" [LOG E·dec_us]。
- 〔推断〕M2 对 M1 的正差，来自 M1 和 M2c 都低于 M0，而 M2 回到了接近 M0 的位置 [LOG E·dec_g2]。
- 由 A/A 算出的 MDE 低估了"两个不同模型"这种比较实际需要的效应量，因为任务间的波动很大 [LOG H6]。

### 4.4 U_sd（模型自己设计更新配方，两轮已完成）

> **✍️** TODO：结果预计 10:30–12:00 出来，到时由协调者追加到 `REPORT_LOG.md`。本节先写好"要回答的问题"和"怎么解读"，数字留空。

**📎 已知设计**
- **第 3 代**：已完成（11-18 05:54），结果并入 4.3 的表。
- **U_sd 第 1 轮（dec_sd1，已出；`research/decisions/decisions_0958.json`，已逐项核对）**：
  - SD1（模型自己定配方）对 M1（固定配方，同一 D0 池）：nAUC 0.605 对 0.537，Δ = +0.068，95% CI [−0.055, +0.199]，p = 0.31。
  - 对随机配方 placebo SDR1：+0.077，CI [−0.085, +0.223]，p = 0.36。
  - SD1 对 M0：+0.017，CI [−0.109, +0.155]，p = 0.82。
  - 判定：未通过；三条判据里只满足"超过 2 倍 A/A 地板（0.062）"一条。点估计上，SD1 是唯一高于原始模型的换权重模型（M1 0.537、M2 0.573、M3 0.487、SD1 0.605，M0 0.588）。
  - 第 2 轮背景：SD1 在训练任务上的反馈为 0.652（M0 0.525，M1 0.483），但第 2 轮模型几乎把配方全部反转（只用 accepted、每次只取最后一次调用、137 条、rank 16、lr 2e-5、2 epoch），给出的理由没有引用反馈表 [LOG E]。
- **U_sd 第 2 轮（dec_sd2，已出；`research/decisions/decisions_1019.json`，已逐项核对，之前各项与 0958 版一致）**：
  - 第 2 轮训练：137 条、rank 16、lr 2e-5、2 个 epoch，31 步；valid loss 0.144 → 0.145，几乎没学动 [LOG E]。
  - SD2 对 SD1：0.595 对 0.605，Δ = −0.010，95% CI [−0.138, +0.110]，p = 0.89。
  - 对随机配方 placebo SDR2：+0.043，CI [−0.042, +0.121]，p = 0.20。
  - SD2 对 M0：+0.007，CI [−0.062, +0.059]，p = 0.65；SD2 对 M1：+0.058，CI [−0.028, +0.168]，p = 0.45。
  - 判定：未通过，三条判据一条都没满足。
  - 两轮汇总的 held-out nAUC：M0 0.588，M1 0.537，SD1 0.605，SD2 0.595；SDR1 约 0.528、SDR2 约 0.552（由差值推出的近似值）[LOG E]。
  - 〔推断，协调者〕点估计上，模型自己定的两份配方都高于固定配方和对应的随机配方，但都没有明显超过原始模型；第 2 轮把配方改得极保守，等于少更新，结果回到原始模型的水平，没有按反馈表的方向调整（第 1 轮在训练任务上 0.652 > 原始模型 0.525）。
- **U_sd**：参照 SEAL，让模型自己设计 SFT 数据的筛选规则和训练配方，两轮反馈 [任务说明；LOG H7–H8]。
  - SEAL 的关键数字：ARC 子集上，经过 RL 训练的 self-edit 为 72.5%，未经 RL 的只有 20% [LOG H8]。
  - 这一节回应的是 H7 的批评：在 U_w 里，"怎么更新自己"是我们定的，模型没有参与，按 Tang 的判据属于 GPI，不是 RSI [T-24；LOG H7]。SEAL 的 self-edit 是让模型自己生成微调数据和超参 [M-13]。
- **事先写好的解读**：
  - Δ₂(2) 仍 ≈ 0：零效应在三代上都成立。
  - U_sd > U_w：把"更新配方"交给模型，是第一个正向信号。
  - U_sd ≈ U_w：配方不是瓶颈。
  - TODO：和你一起定。

### 4.5 谁在修基础设施（人工干预记录）

> **✍️** TODO（建议 1 段；网页放一条时间线 F13，PDF 放 3 行摘要）

**📎 素材** [LOG G]
- G 节共记录 9 次故障：
  - 由协调者修复 7 次；
  - 自愈 1 次（03:00）；
  - 至今没修 1 次（00:14 的 µAR 基线崩溃）。
  - 其中 02:03 那次是协调者自己的失误：调度器停了约 7 分钟。
- 03:00：调度器自己暂停了资源，又把由此造成的等待判为卡死 [LOG G·03:00]。

### 4.6 局限

> **✍️** TODO

**📎 素材**
- 只做了三代；改进者只有 Qwen3.5-9B [`platform_build_plan.md` 决策 D1]；任务是 µAR 小任务；每个任务 5 个 seed [LOG E]。
- 服务路径混杂：M0 走 base 模型的服务路径，其余组走 vLLM 的 LoRA 路径。M2 对 M2c、M1 对 M1p 这两组比较是干净的 [LOG H6]。
- 更新配方是我们写死的（`campaign/overnight.yaml`），改进机制在系统外部：按 Tang 的判据是 GPI [T-24]，按 Yampolskiy 是 weak self-improvement [T-12]。所以测的是"固定配方的自我训练能否复利"，不是完整意义上的 RSI [LOG H7]。
- 不能说明开放式研究上的 RSI，也估不出 Forethought 意义上的 r [`platform_design.md` §0]。
- "没有检出"的正确写法是"在本次功效下未检出，最小可检出效应约为 X"[`platform_design.md` §1.4]。

---

## 5. 与文献对照

> **✍️ 你来写**：一段话，加一张表（📄）。
>
> TODO

**📎 素材** [`lit_rsi_academic.md` §6]

| 我们的结果 | 文献中的对应 |
|---|---|
| lessons 让改进变慢 | ExpeL 加自我反思后 HotpotQA 39.0 → 29.0；没有 oracle 的自我纠错伤害推理（GPT-4 GSM8K 95.5 → 89.0）；ACE 的 context collapse（57.1，低于基线 63.7） |
| LoRA 和 placebo 没有差别 | Spurious Rewards：随机奖励也有 +21.4，真实奖励 +29.1；Mind the Gap：部分"提升"来自格式遵循 |
| 换后继模型当改进者 + 四种对照 | 在检索范围内没有找到同样的设计；最接近的 AIDE² 只换了脚手架 |

- Reflexion 其实已在文献表里 [M-23]，并有数字（MBPP 0.80 → 0.77 等，见 `lit_rsi/02_methods.md` §4）。LOG H3 的"不在账本里"是检索之前写的，已过时；但它的发表场所还没确认（同上 §5）。
- 部分 2025–2026 年的 preprint 只读了摘要 [`lit_rsi_academic.md` §8]。
- "没有找到"应写成"在我们的检索范围内"[同上]。

---

## 6. 给 Ling 的 RSI 计划

> **✍️ 你来写（整节）**。下面只是可选的素材和问题，不是建议的方案。
>
> TODO

**📎 可选素材**
- ⚠️ **Ling 的规格（124B 总参数 / 5.1B 激活 / 35 层 KDA + 7 层 MLA / MIT）目前都是"推测"**，没有一手来源。写进正文前需要官方 card 或技术报告的链接 [`reading/00_总览.md` §6]。
- **能放大的部分** [`platform_design.md` §7.1]：
  - 改进者晋升门槛：新一代只有在封存任务集上 Δ₂ 的 CI 下界 > 0、hack 率不升、通用能力不退时，才能替换上一代；
  - 从第一天起按训练数据的规格记录研发轨迹；
  - 按信号硬度排自动化顺序：kernel（逐位一致的门槛）→ 训练配方（val loss）→ 数据配方 → 架构。
- **放大不了的部分** [`platform_design.md` §7.2]：
  - 124B 的 MoE 做一代 continued training 要按天或周计，一个季度能观察到的代数很少，所以要靠"每代测得准"，而不是"代数多"；
  - 在 U_sd 之前，"递归"的部分其实是人在闭合。
- **审计工具可以直接复用**：权重差分（`hfslice.py` + `weight_diff.py`）和路由统计脚本，可以用来检查每一代改了哪里、改了多少 [`model_dissection_apex_naive.md` §4；`flint_dissection.md` §3]。
- **逐位一致的门槛会挡住投机解码**这一点，对 Ling 的推理环有直接影响 [LOG C]。
- 〔预留〕Ling-3.0-flash 体检（用户安排在最后做）[LOG F]。

---

## 附录 A. 引用键对照

| 键 | 文件 | 原始 URL |
|---|---|---|
| RF-xx | `research/sources/recursive_first_steps.md` | https://recursive.com/updates/first-steps-toward-automated-ai-research ；repo https://github.com/recursive-org/first-steps-toward-automated-ai-research |
| FL-xx | `research/sources/recursive_flint.md` | https://recursive.com/updates/introducing-recursive-flint ；https://huggingface.co/RecursiveSI/flint ；https://github.com/recursive-oss/rec |
| NV-xx | `research/sources/naive_research.md` | https://naive.ai/en/research/ |
| AX-xx | `research/sources/apex_35b.md` | https://www.apexin.com/apex-35b/ ；https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0 |
| LOG X | `research/REPORT_LOG.md` 第 X 节 | 实验数据在私有仓库和集群上 |
| T / M / M2 / E / L / X | `report/references.md`（脚本 `report/tools/build_references.py` 逐字抽取） | 每条都带原文件行号和 URL；生成网页时展开成完整引用 |

## 附录 B. 材料之间的矛盾清单

来自 `research/reading/00_总览.md` §4，以及各台账的"不一致 / 待核"。网页里做成可折叠的列表。

> **✍️** 由你决定哪几条进正文。★1（SOL 分数）和 ★2（同一模型换 harness 分数不同）可以直接支持"信号依赖测量方"的论点。
