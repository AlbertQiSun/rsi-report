# 2. 四份材料
#### 正文（中文）

2026 年，Recursive、Naive 和 Apex 先后发布了四份报告：Recursive 的 First Steps 与 Flint、Naive 的 N0.5-Flash，以及 Apex 的 Apex-35B。这四份材料均围绕“AI 参与模型研发”展开。本节结合三类证据对其进行交叉核验：各份报告的公开陈述、开源发布的权重与代码，以及本报告独立完成的复现与测量。除特别说明外，下文引用的厂商指标均为其自报数字。

[[figure:fig2-disclosure｜图 2　四份报告公开了什么。实心圆表示公开或具备，半圆表示部分公开，空心圆表示未公开或没有；鼠标悬停在每个圆点上可以看到依据。本报告根据各报告的公开内容整理 [RF-Q9, NV-Q2, NV-Q6, FL-Q7, AX-Q10, AX-43]。]]

### Recursive：First Steps Toward Automated AI Research

Recursive 的自动化研究系统在三个公开基准上展示了优化成果。在 NanoChat（要求单卡 GPU 在 5 分钟固定预算内训练一个小型语言模型）上，系统将验证集 BPB 从社区最优的 0.9372 降至 0.9109，相当于缩短了约 1.3 倍的训练时间即可达到同等损失 [RF-02, RF-05, RF-06, RF-08]。在 NanoGPT Speedrun（以最短时间将验证损失训至 3.28）上，系统将耗时从 79.7 秒压缩至 77.5 秒 [RF-11, RF-17]。在包含 235 个 GPU kernel 编写任务的 SOL-ExecBench 上，系统将平均得分从 0.699 提升至 0.754 [RF-23, RF-25, RF-26]。

在评测严谨性上，基线使用了 10 个随机种子进行评估，Speedrun 的结果也附带了显著性检验（p = 0.0078）[RF-04, RF-33]。针对另外两个从较弱起点出发的实验，作者也坦承底层模型在预训练阶段可能已经见过相关的公开优化技巧，因此不能直接视为系统的独立发现 [RF-Q8]。

然而在系统透明度与细节核对上，这份报告存在明显的盲区。Recursive 仅公开了最终搜出的解法本身（包括训练代码、各种子结果，以及 235 个 kernel 中的 10 个）[RF-29, RF-31]，但并未公开研究系统本身；报告中既未交代底层所用模型、迭代轮数与消耗算力，也未说明系统在运行中是否修改过自身的脚手架代码 [RF-Q9]。此外，SOL 部分的代码未附带正确性检查 [LOG B]；Speedrun 的成绩并未提交至官方排行榜，且博客正文与代码仓库给出的耗时存在出入（分别为 77.5 秒与 77.34 秒）[RF-Q6, RF-28, RF-34]。通过检视仓库中的文件命名（例如编号排到 w204 的 kernel 变体以及 CROSSOVER 等标记），可以看出该系统在底层采用了进化式搜索策略 [LOG F]。

纵观这三项实验，被持续优化的对象始终是外部的训练脚本和 GPU kernel，而执行优化的研究系统自身则保持固定。按第 1 节的标准衡量，这属于典型的一阶改进。报告虽将 RSI 作为长期愿景提出，但并未给出任何二阶改进的测量。

### Naive：N0.5-Flash、NaiveRT 与 AutoWM

Naive-N0.5-Flash 是一个在 MiMo-V2.5 基础上继续训练而成的 309B 参数 MoE 模型，每个 token 激活 15.5B 参数 [NV-02, NV-03]。报告展示了 AI 参与研发的三个案例：稀疏注意力架构设计、推理引擎 NaiveRT，以及世界模型项目 AutoWM。

在架构优化上，N0.5-Flash 以 5:1 的比例混合了滑动窗口注意力与 DeepSeek Sparse Attention（DSA）[NV-06]；其 DSA 索引器仅使用 16 个查询头，使索引选择的耗时比原版减少了 44% [NV-30, NV-31]。在推理引擎 NaiveRT 的开发中，人类研究员与 AI 模型协作，在六天内完成了构建 [NV-45]。开发日志共记录了 151 次优化尝试，其中 63 次被采纳，71 次因未通过验证或引入退化而被回滚 [NV-46, NV-60]，最终将一次完整的投机解码耗时从 SGLang 下的 12.3 ms 压缩至 3.4 ms [NV-43]。报告同时记录了一次失败尝试：团队共实现了七版 MoE 融合算子，尽管数值校验全部正确，但端到端性能反而下降，研究员最终叫停了这一方向 [NV-66]。

[[figure:fig2-naivert｜图 3　NaiveRT 三个阶段的优化尝试。151 次尝试中有 63 次被采纳；另外 88 次中，71 次未通过验证或被回滚，17 次属于探索性尝试。数字为 Naive 自报 [NV-46, NV-57, NV-58, NV-59, NV-60]。]]

在第三个案例 AutoWM 中，N0.5-Flash 亲自担任“AI 研究员”。经过 400 小时的累计自动研究，它在 WorldArena 基准上取得了 77.43 分，超越了当时已公布的最高成绩 73.64 分 [NV-69, NV-71, NV-72, NV-Q6]。不过，AutoWM 直接使用目标评测指标本身来筛选输出，且报告并未拆解各个子模块对最终分数的独立贡献 [NV-Q4]。

报告还将 N0.5-Flash 与 Recursive 的系统进行了横向对比，宣称在 NanoChat 上以 0.9051 优于后者的 0.9109 [NV-24]。但这项对比运行在 Naive 自建的研究框架下，且未报告随机种子数与方差 [NV-Q2]；而在 SOL-ExecBench 上，Naive 引用 Recursive 的得分为 67.56，Recursive 原报告自报的则是 0.754，两者口径并不一致 [NV-23, RF-26]。

对公开权重的差分检验进一步揭示了报告未提及的细节。首先，N0.5-Flash 的直接父模型实为 MiMo-V2.5 的对话版（Chat），而非 Base 版。在被改造为稀疏注意力的 9 层网络中，q、k、v 投影矩阵接近被彻底重写（v 投影权重改动的中位数达 51.9%），模型还额外新增了 189 个索引器张量 [model_dissection_apex_naive.md §2–3]。其次，报告声称“将 MLA 替换为了 GQA”，但查验 MiMo-V2.5 的原始权重可知，其本来使用的就是 GQA [LOG B]。

[[figure:fig2-weights｜图 4　权重改动的位置。(a) Apex-35B 相对 Qwen3.6 的改动，按部件列出报告的数值或范围：除归一化层（约 1%）和线性注意力的衰减参数（约 0.07% 以下）外，各部件都有明显改动，说明是全参数继续训练。(b) N0.5-Flash 相对 MiMo-V2.5：改为稀疏注意力的 9 个 DSA 层，q、k、v 投影的改动远大于其余 39 个 SWA 层。改动按 ‖A − B‖ / ‖B‖ 计算 [model_dissection_apex_naive.md §1, §3]。]]

在行为普查测试中，面对 80 条测试提示，N0.5-Flash 有 76 条以 "We need to…" 或 "We need answer…" 开启思维链，这与 Flint 在代码和数学题上的推理起手式高度一致；其思维链长度中位数仅为 553 个字符，在四个被测模型中最短 [model_dissection_apex_naive.md §8；census_compare.json]。这表明两个模型的后训练轨迹可能蒸馏自同一类外部教师模型；不过由于本报告尚未对 MiMo-V2.5 原模型做同样测试，目前也不能排除这种行文风格直接继承自父模型的可能〔推断〕。此外，在全部 104 条测试提示中，N0.5-Flash 从未提及 MiMo 或小米。它对自身身份的回答，首先源于聊天模板在缺失系统提示时自动注入的默认身份句，因此不能作为权重层面洗去了原模型身份的证据 [model_dissection_apex_naive.md §8]。

[[figure:fig2-census｜图 5　思考方式普查。四个模型使用同一套 80 条提示，贪心解码。(a) 每个模型最常见的思考开头及其条数：Apex 与基座 Qwen3.6 的写法完全不同，Naive 与 Flint 在代码和数学题上的写法一致。(b) 思考长度的中位数；Qwen3.6 有 22 条在 1200 token 上限内没有写完，它的中位数是下限 [census_compare.json；model_dissection_apex_naive.md §5, §8]。]]

从改进机制来看，NaiveRT 和稀疏注意力改写的是推理引擎与模型结构，均属一阶改进；而在 AutoWM 中，N0.5-Flash 虽然扮演了研究员角色，但它优化的对象是外部的世界模型而非自身。报告虽然设想由下一代模型接管更多研究工作，但并未提供代际之间的改进速度测量。

### Recursive：Flint

Flint 是 Recursive 借助其自动化研究系统 Eureka Machine 从零训练的 MoE 模型，总参数量 40B，每个 token 激活 2.5B，研发耗时不到四个月 [FL-01, FL-02, FL-03]。在 M4 Pro 笔记本上，Flint 的解码速度超过每秒 125 个 token [FL-07]。研究系统还将模型的输出层量化至 4 bit，使最终的 NVFP4 模型与 16-bit 满血版的性能差距控制在 0.7 分以内 [FL-18, FL-21]。至于“新训练出的模型能否反过来提升研究系统本身”，报告仅将其表述为一项未来的押注（"Our bet is…"），且并未量化 Eureka Machine 在各个研发环节中的具体贡献比例 [FL-Q7]。

对本地权重的检验确认了其公开规格：实测总参数为 39.785B，激活参数为 2.515B [flint_dissection.md §1]。其分词器确由 Recursive 独立训练，15 万个词表项中仅有 51 个接近未训练的初始状态，这与报告“从零训练”的说法相吻合 [flint_reverse_engineering.md §1–2]。不过在元数据上，模型存在几处自相矛盾：Flint 在对话中自报的知识截止时间为 2025 年 5 月，但官方模型卡写的却是 2026 年 6 月 [FL-36]；博客宣称上下文长度为 512k，而权重配置文件中实际设定为 262,144 [FL-16, FL-34]。

更关键的发现来自后训练阶段的外部模型痕迹。在行为测试中，Flint 的思维链里直接出现了 "as a Gemini model…" 和 "Since I am a Google AI" 等原句；同时，它的推理文风呈现出明显的“按题型分流”特征：代码与数学题采用 gpt-oss 式的简短笔记体，知识问答与闲聊采用 Gemini 式的编号分步体，而中文提示词下的行文则高度接近 DeepSeek-R1 [flint_reverse_engineering.md §3, §5–6]。这些迹象表明，Flint 的后训练过程极可能按任务类别拼接了不同外部教师模型的蒸馏数据〔推断〕。

此外，同一模型在不同机构的评测框架下得分差异显著：Flint 在 Terminal-Bench 2.1 上自报得分为 54.9%，而 Apex 用自身框架测得的结果为 59.6；在 SWE-Bench Pro v2 上，Flint 自报高达 89.7%，Apex 测得则为 85.2 [FL-24, FL-25, AX-T13, AX-T14]。

[[figure:fig2-measure｜图 6　同一项指标，不同的测量方。蓝点是报告方自报的分数，橙点是另一方测得或引用的分数；右侧数字为两者之差。SOL-ExecBench 的分数乘以 100 后与其他指标放在同一刻度上 [FL-24, FL-25, FL-26, AX-T13, AX-T14, AX-T15, RF-26, NV-23]。]]

归结而言，Flint 只是 Eureka Machine 生产出的单代产物。无论是在幕后执行搜索的研究系统，还是在后训练中提供推理范式的外部教师模型，都不是 Flint 自身，因此它展示的依然是一阶改进。

### Apex：Apex-35B-V1.0

Apex-35B 是在阿里 Qwen3.6-35B-A3B 基础上继续训练的模型，总参数 35B，每个 token 约激活 3B [AX-01, AX-02]。报告中专设了题为 "Building Apex-35B-V1.0 Through Recursive Self-Improvement" 的一节，列举了多项优化收益：引入“自我进化”（self-evolving）数据后，内部基准 CFD-Bench 提升了 32.1 分（未说明具体单位）[AX-37]；训练吞吐提升 45–60%，解码速度提升 1.6–1.8 倍，首 token 延迟降低 30–35%，但上述系统指标均未给出具体的对比基线 [AX-38, AX-39, AX-40, AX-Q7]。此外，报告中的各项基准评测在每个任务上均只运行了一次 [AX-43]。

在展示模型自主研发能力的 AI4AI 一节中，Apex 亲自担任研究员，任务是在 4 小时探索与 12 小时重训的预算内，用 GRPO 算法训练 Qwen2.5-3B-Instruct 玩推箱子（Sokoban）。训练后的小模型在 512 个未见关卡上达到了 24.4% 的解出率，正文将其宣传为“超过 base model（11.7%）的两倍” [AX-22, AX-23, AX-25]。然而核对图注与导读记录可以发现，11.7% 其实是未经任何训练的 Qwen2.5-3B 初始得分（即被训练对象的起点），而非对照组研究员的成绩；事实上，作为对照的 Base 研究员在运行 101 分钟后便因故障退出，根本没有完成训练或提交结果 [AX-29；reading/04_Apex35B_导读.md §3]。因此，这项实验仅能证明 Apex 成功跑完了一次强化学习训练流程，并不能证明它作为研究员的能力优于基线模型。

权重层面的比对显示，Apex 的模型配置与分词器均与 Qwen3.6 逐字段完全一致，但全部权重张量都发生了变化 [LOG B]。这表明它经历了一次全参数继续训练：各个专家网络的权重改动幅度在 6% 至 21% 之间，路由器改动为 9% [model_dissection_apex_naive.md §1]。与此同时，模型的思考方式被整体重塑：原版 Qwen3.6 在 80 条测试提示中有 70 条以标志性的 "Here's a thinking process:" 开头，而 Apex 降为 0/80，转而在 72/80 的提示中以 "The user asks/wants…" 起手，且思维链长度缩短至原来的三分之一以下——这一特征与使用 Claude 风格的思维链进行蒸馏高度吻合，尽管 Apex 在回答中并未自称 Claude〔推断〕。当被直接问及身份时，Apex 会坦承自身的基座是 Qwen3.6；而在未挂载任何外部工具的裸测环境下，它仍会凭空尝试调用一个并不存在的 `web_search` 工具 [model_dissection_apex_naive.md §5]。

尽管 Apex 直接将自身的构建过程冠以“递归自我改进”之名，但报告并未提供任何逐代演进的数据。其所谓的“自我进化”数据在分类上仅属于 weak self-improvement [T-12]，背后的训练框架与改进规则依然完全由人类工程师预先设计 [T-24]。

### 共同点

综合上述模型解剖结果，这三个开放权重的模型在底层基座或高层思考方式上，均深度依赖外部现有模型：

| | Flint | Apex | Naive |
|---|---|---|---|
| 基座 | 从零训练 | Qwen3.6-35B-A3B，全参数继续训练 | MiMo-V2.5 对话版，9 层注意力重写 |
| 思考方式 | 随任务类别变化：简短写法、Gemini 式编号 | 统一为 "The user…" 式 | 统一为简短写法 "We need to…" |
| 是否提到外部模型 | 思考中出现 "as a Gemini model" | 说明基座是 Qwen3.6 | 不提 MiMo 或小米 |

（来源：[model_dissection_apex_naive.md §7–8；flint_reverse_engineering.md §3]）

横向审视这四份材料，它们所展示的实质成果全部停留在**一阶改进**：算子 kernel 变得更快、训练脚本变得更高效、模型在特定基准上的得分变得更高。但没有任何一份报告真正完成了**二阶改进**的测量——即把改进所得的后继模型放回研发管线中接替担任改进者，在同等预算下检验其改进速度是否超越前代。不仅如此，即便是对一阶改进的衡量，不同团队在不同框架下测出的分数也往往相互打架（如 Flint 与 SOL-ExecBench 的跨报告分差）。为什么这个最核心的二阶测量在产业界报告中始终缺位？下一节将具体剖析其中的症结。

#### Text (English)

In 2026, Recursive, Naive, and Apex released four technical reports: Recursive's *First Steps* and *Flint*, Naive's *N0.5-Flash*, and Apex's *Apex-35B*. All four center on the role of AI in model development. This section cross-examines their claims using three sources of evidence: the statements in the reports themselves, the publicly released weights and code, and this report's own independent reproductions and forensic measurements. Unless noted otherwise, vendor figures cited below are self-reported.

[[figure:fig2-disclosure｜Figure 2. What each report discloses. A filled circle means released or present, a half circle partly released, an empty circle not released or absent; hover over a circle to see the basis. Compiled by this report from the public material of each release [RF-Q9, NV-Q2, NV-Q6, FL-Q7, AX-Q10, AX-43].]]

### Recursive: First Steps Toward Automated AI Research

Recursive evaluated its automated research system across three public benchmarks. On NanoChat—which requires training a small language model within a fixed five-minute budget on a single GPU—the system lowered validation BPB from the community best of 0.9372 to 0.9109, equivalent to reaching the same loss in roughly 1.3 times less training time [RF-02, RF-05, RF-06, RF-08]. On NanoGPT Speedrun, which measures the shortest time to reach a validation loss of 3.28, the system trimmed the record from 79.7 to 77.5 seconds [RF-11, RF-17]. On SOL-ExecBench, a suite of 235 GPU kernel-writing tasks, it raised the mean score from 0.699 to 0.754 [RF-23, RF-25, RF-26].

From an evaluation standpoint, the baseline was tested across 10 random seeds, and the Speedrun result includes a statistical significance test ($p = 0.0078$) [RF-04, RF-33]. For two additional runs initialized from weaker starting points, the authors candidly note that the underlying model may have encountered the relevant public techniques during pre-training, meaning those runs should not be interpreted as independent discoveries [RF-Q8].

What the report leaves out, however, is the system itself. Recursive released only the final artifacts discovered—the training scripts, per-seed logs, and 10 of the 235 kernels [RF-29, RF-31]—without disclosing the underlying model, the number of search iterations, the total compute spent, or whether the system ever modified its own scaffolding [RF-Q9]. Moreover, the released SOL kernels lack a correctness verification suite [LOG B], and the Speedrun result was never submitted to the official leaderboard, with the blog post and code repository citing slightly different times (77.5 vs. 77.34 seconds) [RF-Q6, RF-28, RF-34]. Artifact naming conventions in the repository—such as kernel variants numbered up to `w204` and tags like `CROSSOVER`—indicate that the harness relies on evolutionary search under the hood [LOG F].

Across all three benchmarks, the targets of optimization are external training scripts and GPU kernels, while the research system doing the optimizing remains fixed. By the criteria established in Section 1, these results represent first-order improvement. While the report frames RSI as a long-term ambition, it provides no second-order measurements.

### Naive: N0.5-Flash, NaiveRT and AutoWM

Naive-N0.5-Flash is a 309B-parameter MoE model (15.5B active per token) built via continued training on MiMo-V2.5 [NV-02, NV-03]. Its report highlights three areas of AI-assisted development: a sparse-attention architecture, the NaiveRT inference engine, and the AutoWM world-model project.

On the architectural front, N0.5-Flash interleaves sliding-window attention with DeepSeek Sparse Attention (DSA) in a 5:1 ratio [NV-06]; by restricting the DSA indexer to 16 query heads, it cuts index-selection latency by 44% compared to the original design [NV-30, NV-31]. For the NaiveRT engine, human researchers collaborated with AI models to complete the system in six days [NV-45]. Development logs record 151 optimization attempts—63 adopted and 71 rolled back or rejected during validation [NV-46, NV-60]—ultimately cutting a full speculative-decoding step from 12.3 ms in SGLang to 3.4 ms [NV-43]. The report also documents a dead end: seven distinct implementations of MoE kernel fusion all passed numerical checks but degraded end-to-end latency, prompting researchers to abandon the direction [NV-66].

[[figure:fig2-naivert｜Figure 3. NaiveRT optimisation attempts in three stages. Of 151 attempts, 63 were adopted; of the other 88, 71 failed validation or were rolled back and 17 were exploratory. Numbers are self-reported by Naive [NV-46, NV-57, NV-58, NV-59, NV-60].]]

In the third case, AutoWM, N0.5-Flash itself served as the AI researcher. Over 400 hours of cumulative autonomous experimentation, it achieved a score of 77.43 on WorldArena, surpassing the previous published high of 73.64 [NV-69, NV-71, NV-72, NV-Q6]. However, AutoWM selects its outputs directly against the target evaluation metric, and the report does not isolate the contributions of individual pipeline components [NV-Q4].

The report also benchmarks N0.5-Flash directly against Recursive's system, claiming an edge of 0.9051 versus 0.9109 on NanoChat [NV-24]. That comparison, however, was run inside Naive's own evaluation harness without reporting seed counts or variance [NV-Q2]; meanwhile, on SOL-ExecBench, Naive cites Recursive's score as 67.56, whereas Recursive's own report states 0.754 [NV-23, RF-26].

A weight-diff analysis uncovers several structural details omitted from the report. First, N0.5-Flash descends from the chat-tuned release of MiMo-V2.5 rather than the Base checkpoint. Across the 9 layers converted to sparse attention, the $q$, $k$, and $v$ projections were nearly rewritten from scratch—with a median weight change of 51.9% in the $v$ projection—alongside 189 newly added indexer tensors [model_dissection_apex_naive.md §2–3]. Second, although the report claims to have "replaced MLA with GQA," inspecting the original MiMo-V2.5 weights confirms that the parent model already used GQA [LOG B].

[[figure:fig2-weights｜Figure 4. Where the weights changed. (a) Apex-35B versus Qwen3.6, by component, showing the reported value or range: apart from the norm layers (about 1%) and the linear-attention decay parameters (about 0.07% or less), every component changed noticeably, which indicates full-parameter continued training. (b) N0.5-Flash versus MiMo-V2.5: in the 9 layers converted to DSA, the q, k and v projections changed far more than in the other 39 SWA layers. Change is measured as ‖A − B‖ / ‖B‖ [model_dissection_apex_naive.md §1, §3].]]

In behavioral testing, N0.5-Flash opens its chain of thought with "We need to…" or "We need answer…" on 76 out of 80 census prompts, closely mirroring Flint's phrasing on coding and math tasks; its median reasoning trace is just 553 characters, the shortest among the four models audited [model_dissection_apex_naive.md §8; census_compare.json]. This convergence suggests that both models may have been distilled from a similar class of external teacher traces—though because the same census was not run on MiMo-V2.5, the possibility cannot be ruled out that the phrasing was inherited from the parent checkpoint (inference). Across all 104 test prompts, N0.5-Flash never mentions MiMo or Xiaomi. Its responses to identity queries stem primarily from a default identity string injected by the chat template whenever no system prompt is supplied, providing no evidence of weight-level identity erasure [model_dissection_apex_naive.md §8].

[[figure:fig2-census｜Figure 5. Reasoning-style census. All four models received the same 80 prompts with greedy decoding. (a) The most common reasoning opening for each model and its count: Apex's style is entirely different from its base Qwen3.6, while Naive and Flint share the same style on code and math. (b) Median reasoning length; 22 of Qwen3.6's answers did not finish within the 1200-token cap, so its median is a lower bound [census_compare.json; model_dissection_apex_naive.md §5, §8].]]

Assessed against the framework of Section 1, both NaiveRT and the sparse-attention modifications target the inference engine and model architecture, placing them squarely in first-order improvement. In AutoWM, N0.5-Flash does act as the researcher, yet the object it optimizes is a separate world model rather than itself. While the report envisions future generations taking over a larger share of R&D, it offers no generation-over-generation measurements.

### Recursive: Flint

Flint is a 40B-parameter MoE model (2.5B active per token) trained from scratch in under four months using Recursive's automated research system, the Eureka Machine [FL-01, FL-02, FL-03]. Running locally on an M4 Pro laptop, it decodes at over 125 tokens per second [FL-07]. The research system also quantized the output head to 4 bits, keeping the final NVFP4 checkpoint within 0.7 points of the 16-bit baseline [FL-18, FL-21]. As for closing the loop—having the newly trained model improve the research system in return—the report frames this strictly as a future bet ("Our bet is…") and does not quantify how much the Eureka Machine contributed to each stage of Flint's creation [FL-Q7].

Inspecting the local weights confirms the published architectural specs: 39.785B total parameters and 2.515B active [flint_dissection.md §1]. Recursive also trained a custom tokenizer in which only 51 of the 150,000 vocabulary entries remain near their untrained initialization, consistent with a from-scratch training run [flint_reverse_engineering.md §1–2]. Several metadata fields, however, conflict with one another: the model claims a knowledge cutoff of May 2025 in conversation while the model card lists June 2026 [FL-36], and the blog advertises a 512k context window whereas the configuration file sets the limit at 262,144 [FL-16, FL-34].

More revealing are the footprints left by external models during post-training. In this report's behavioral probes, Flint's reasoning traces occasionally leak verbatim phrases such as "as a Gemini model…" and "Since I am a Google AI". Furthermore, its reasoning style shifts sharply by task category: coding and math prompts trigger terse, `gpt-oss`-style notes; general knowledge and chat prompts elicit Gemini-style numbered steps; and Chinese prompts produce reasoning patterns closely resembling DeepSeek-R1 [flint_reverse_engineering.md §3, §5–6]. Together, these artifacts indicate that post-training relied on routing different task types to distinct external teacher models for distillation (inference).

Benchmark scores also shift depending on who runs the evaluation. On Terminal-Bench 2.1, Recursive self-reports 54.9% for Flint while Apex's harness measures 59.6; on SWE-Bench Pro v2, Recursive reports 89.7% whereas Apex measures 85.2 [FL-24, FL-25, AX-T13, AX-T14].

[[figure:fig2-measure｜Figure 6. Same metric, different measurer. Blue dots are scores reported by the releasing party, orange dots are scores measured or quoted by another party; the number on the right is the difference. SOL-ExecBench scores are multiplied by 100 to share the scale [FL-24, FL-25, FL-26, AX-T13, AX-T14, AX-T15, RF-26, NV-23].]]

Ultimately, Flint is the output of a research pipeline, not its driver. Neither the Eureka Machine that searched over its design nor the external teacher models that supplied its reasoning traces are Flint itself, making it another instance of first-order improvement.

### Apex: Apex-35B-V1.0

Apex-35B (35B total parameters, ~3B active per token) was built via continued training on Alibaba's Qwen3.6-35B-A3B [AX-01, AX-02]. A section of the report titled "Building Apex-35B-V1.0 Through Recursive Self-Improvement" attributes several gains to "self-evolving" data: a 32.1-point jump on the internal CFD-Bench (with no unit specified) [AX-37], a 45–60% increase in training throughput, a 1.6–1.8$\times$ decoding speedup, and a 30–35% reduction in time to first token—all reported without baseline figures [AX-38, AX-39, AX-40, AX-Q7]. Every benchmark result in the report is also based on a single evaluation run per task [AX-43].

In the report's AI4AI section, Apex steps into the role of researcher: given a budget of 4 hours for exploration and 12 hours for retraining, it is tasked with training Qwen2.5-3B-Instruct via GRPO to play Sokoban. The resulting student model solves 24.4% of 512 held-out levels, a figure the main text highlights as more than double "the base model's 11.7%" [AX-22, AX-23, AX-25]. Checking the figure caption, however, reveals that 11.7% is merely the zero-shot score of the untrained Qwen2.5-3B student model—not the score achieved by a baseline researcher. In fact, the comparison "Base" researcher crashed and exited after 101 minutes without finishing training or submitting a checkpoint [AX-29; reading/04_Apex35B_导读.md §3]. The experiment therefore demonstrates only that Apex managed to complete a training run without crashing, not that it outperforms a baseline model at conducting AI research.

Weight inspection confirms that Apex shares Qwen3.6's exact configuration schema and tokenizer, while every weight tensor has been updated [LOG B]. Specifically, full-parameter continued training shifted individual expert weights by 6% to 21% and router weights by 9% [model_dissection_apex_naive.md §1]. Post-training also completely overwrote the model's reasoning habits: whereas Qwen3.6 opens 70 of 80 census prompts with "Here's a thinking process:", Apex never uses that phrase (0/80), opening instead with "The user asks/wants…" on 72 of 80 prompts and producing reasoning traces less than a third as long—a pattern consistent with distillation from Claude-style traces, though Apex never claims to be Claude (inference). When asked directly, Apex identifies Qwen3.6 as its base model; when probed without any tools attached, it still hallucinates calls to a non-existent `web_search` tool [model_dissection_apex_naive.md §5].

Although Apex labels its pipeline recursive self-improvement, the report presents no multi-generational data. Its "self-evolving" synthetic data pipeline falls under weak self-improvement [T-12], with the underlying training loop and update rules still engineered entirely by humans [T-24].

### What they have in common

Taken together, the forensic checks show that all three open-weight models inherit either their foundational weights or their reasoning patterns from existing external models:

| | Flint | Apex | Naive |
|---|---|---|---|
| Base | trained from scratch | Qwen3.6-35B-A3B, full-parameter continued training | chat version of MiMo-V2.5, 9 attention layers rewritten |
| Reasoning style | varies by task type: short notes, Gemini-style numbered steps | uniform "The user…" style | uniform short notes, "We need to…" |
| Mentions outside models | reasoning contains "as a Gemini model" | states that its base is Qwen3.6 | does not mention MiMo or Xiaomi |

(Sources: [model_dissection_apex_naive.md §7–8; flint_reverse_engineering.md §3])

Stepping back, all four reports demonstrate genuine **first-order improvement**: faster kernels, tighter training scripts, and higher benchmark scores. None of them, however, measures **second-order improvement**—taking the improved model, placing it back into the harness as the next generation's researcher, and testing whether it drives progress faster than its predecessor under the same budget. Even at the first-order level, evaluation results frequently diverge depending on which team runs the harness, as seen in the conflicting scores for Flint and SOL-ExecBench. Why has a clean measurement of second-order improvement remained absent across all of these efforts? The next section examines the structural reasons behind this gap.
