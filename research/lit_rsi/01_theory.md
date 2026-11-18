# 01 理论与定义：学术文献如何定义 Recursive Self-Improvement (RSI)

> 范围：只覆盖理论、定义、条件、极限与 takeoff/returns 模型。ML 方法和 measurement/evals 由另外两个 agent 负责。
> 检索日期：2026-11-18。所有条目都是本次实际打开过的来源（PDF 用 pdftotext 抽取全文，或读取 HTML 页面）；没能打开的列在第 5 节。
> 引文规则：每个来源最多一句不超过 15 词的英文原文引用，其余均为转述。凡是本文作者（agent）自己的映射或推断，都标注为【推断】，供用户判断，不当作文献结论。

---

## 1. 来源总表

类型说明：**PR** = peer-reviewed（期刊、会议论文集或学术书章节）；**Preprint** = arXiv 或机构技术报告，未经同行评审；**Essay/Blog** = 随笔、博客、论坛帖。

| # | 来源 | 类型 | 如何定义 RSI / self-improvement | 关于条件或极限的关键主张 | URL |
|---|---|---|---|---|---|
| 1 | I. J. Good (1965), "Speculations Concerning the First Ultraintelligent Machine", *Advances in Computers* 6 | PR（书章）。原文是扫描版，无法抽取文字；下面的段落按 Chalmers 2010 和 Yudkowsky 2013 的逐字转引核对 | ultraintelligent machine：在一切智力活动上远超任何人类的机器。"Since the design of machines is one of these intellectual activities"，所以它能设计出更好的机器，从而引发 "intelligence explosion" | 唯一的前提是：设计机器本身是智力活动之一。Good 没有给出 returns 条件，后来的文献（Chalmers、Yudkowsky）都把补出这个条件当作核心工作 | 扫描 PDF：https://www.stat.vt.edu/content/dam/stat_vt_edu/graphics-and-pdfs/research-papers/Technical_Reports/TechReport05-3.pdf |
| 2 | Yudkowsky (2007), "Levels of Organization in General Intelligence" (LOGI)，载 Goertzel & Pennachin (eds.), *Artificial General Intelligence*, Springer, pp. 389–501 | PR（书章） | "A seed AI is an AI designed for self-understanding, self-modification, and recursive self-improvement." 区分 strongly self-improving（改进直接作用于"做改进的那个过程"本身）与 weakly self-improving（evolution 改进人类、人类改进知识） | 1）RSI 是 seed AI 的结果，不是获得 AI 的捷径，无法从不智能的内核自举。2）optimizing compiler 式的改进"不是 self-improvement"，会很快耗尽。3）关键问题是 "ladder question"：爬上一级之后，能否看到足够多的机会去爬下一级 | https://intelligence.org/files/LOGI.pdf |
| 3 | Yudkowsky (2008), "Recursive Self-Improvement", LessWrong | Essay/Blog | recursion 指优化者与被优化的对象合一：AI 重新设计自己的认知算法。用债券比喻：利息拿去花是线性增长，拿去再投资才是复利 | 自然选择和人类文明虽然有累积，但不是 recursive（大脑本身没有被改写）。EURISKO 和 optimizing compiler 是"递归但很快熄火"的例子 | https://www.lesswrong.com/posts/JBadX7rwdcRFzGuju/recursive-self-improvement |
| 4 | Yudkowsky (2013), "Intelligence Explosion Microeconomics", MIRI Technical Report 2013-1 | Preprint（技术报告） | 核心量是 "returns on cognitive reinvestment"：把算力、更快的硬件或更好的算法投进去，产出认知劳动，再用这些劳动造出更大、更快或设计更好的心智。用核反应堆的有效中子增殖系数 k 作类比 | k<1 是 "intelligence fizzle"（如 log(w) 型回报，迅速收敛）；k≈1+ 是 "intelligence combustion"（类似经济的平稳指数增长）；prompt k≫1 是 "intelligence explosion"，即短时间尺度上的再投资，不必先建芯片厂。他明确拒绝 Mahoney 的 Kolmogorov 定义，认为那个定义不适合讨论问题 | https://intelligence.org/files/IEM.pdf |
| 5 | Schmidhuber (2003/2006), "Gödel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements", arXiv cs/0309048 v5；相关版本发表于 LNCS 3394 (2005)、ICANN 2005，以及 *Artificial General Intelligence* (Springer 2006) | Preprint（另有 PR 版本） | 一旦证明某次重写有用，机器就可以重写自己代码的**任何**部分，包括 proof searcher 本身。utility、硬件和初始代码都以公理形式编码 | Global Optimality Theorem：执行当前 switchprog 的效用高于继续搜索别的改写，因此没有局部极大。"all possible 'meta-levels' of the self-referential system are collapsed into one"。极限在于 Gödel 不完备性：无法证明有效的改进只能放弃；资源受限时，在时间预算内证明不了的改进也用不上 | https://arxiv.org/abs/cs/0309048 |
| 6 | Omohundro (2008), "The Basic AI Drives", *Proc. First AGI Conference*, IOS Press FAIA vol. 171, pp. 483–492 | PR（会议） | "AIs will want to self-improve"：对足够先进的目标驱动系统来说，修改自己的软件或物理结构是一种可选行动，而有益的自我修改收益是永久的，所以系统有强烈动机去发现并实施它 | 讲的是动机，不是能力或 returns。论证了系统会倾向于保护自己的 utility function 不被修改，这与 tiling/Löb 问题有关 | https://selfawaresystems.com/wp-content/uploads/2008/01/ai_drives_final.pdf |
| 7 | Hall (2007), "Self-improving AI: an Analysis", *Minds & Machines* 17:249–259 | PR（期刊） | 提出 "universality"：一种假想的学习能力水平，指 "a level of capability at which a program can improve itself without limit"，类比计算的通用性 | 能造出同级 AI，不等于能无界自我改进。认为人类个体可能略低于 universality，但科学共同体整体达到了。结论是需要"研究科学家级 AI 的共同体"才有无界反馈的空间，因此 takeoff 会比想象的慢 | https://faculty.washington.edu/seattle/Turing-Test/SelfImprovingAI.pdf |
| 8 | Mahoney (2008, rev. 2010), "A Model for Recursively Self Improving Programs" | Preprint（个人草稿，未发表） | 形式定义：P₁ 是关于目标 G 的 RSI 程序，当且仅当 Pᵢ(−1)=Pᵢ₊₁（每一代输出下一代），并且序列中每一代在相同时间预算内都严格改进上一代、且以后永不变差。明确把 self-improvement（只拿到自己的源码，没有外部输入）和 learning（有输入）区分开；还举了"通过跟自己下棋变强"的程序作为 self-improving 的例子 | 定理：第 n 代的 Kolmogorov 复杂度不超过 O(log n)。结论："software self modification by itself is not a viable path to self improvement"（如果智能取决于知识和算力的话） | https://mattmahoney.net/rsi.pdf |
| 9 | Chalmers (2010), "The Singularity: A Philosophical Analysis", *Journal of Consciousness Studies* 17:7–65 | PR（期刊） | 把 Good 的论证形式化为 AI → AI+ → AI++ 和 AIₙ 序列：AIₙ₊₁ 由 AIₙ 创造，且比创造者"显著更聪明"（差距 ≥ δ）。去掉"智能"概念后的一般形式是 self-amplifying cognitive capacity G：G 的提升伴随着创造具有 G 的系统的能力**按比例或更多**地提升 | 最关键的假设是 **proportionality thesis**。会被 diminishing returns 破坏（他的例子：10%→5%→2.5%）。论证是否成立取决于所用的智能度量，换成 log 度量结论可能就不成立。总结为 "self-amplification plus correlation plus manifestation = singularity"，对应三类障碍：structural、correlation、manifestation。也指出递归路径原则上不必从人类水平开始 | https://consc.net/papers/singularity.pdf |
| 10 | Yudkowsky & Herreshoff (2013), "Tiling Agents for Self-Modifying AI, and the Löbian Obstacle", MIRI early draft | Preprint | 把 self-modification 建模为"构造一个与自己相似、目标相同的 successor 并替换自己"（tiling） | 最直接的形式化会碰到 Löbian obstacle：一个形式系统无法简单地信任一个和自己一样强的继任者的推理。技术上可以部分绕过，但根本性的决策准则仍未解决 | https://intelligence.org/files/TilingAgentsDraft.pdf |
| 11 | Bostrom (2014), *Superintelligence*, OUP, ch. 4 "The kinetics of an intelligence explosion" | PR（学术专著）。**原书未打开**，下面的内容来自 Grace 2014 的读书会摘要和 Benthall 2017 的转述 | 智能增长率 = optimization power / recalcitrance。crossover 是 "a point at which the inputs to a project are mostly its own outputs"（Grace 摘要的措辞） | takeoff 分为 slow（数十年到数百年）、moderate（数月到数年）、fast（数分钟到数天）。过了 crossover 后 optimization power 会上升更快，recalcitrance 可能下降，所以 Bostrom 认为 fast 或 moderate takeoff 更可能 | 摘要：https://www.lesswrong.com/posts/GT8uvxBjidrmM3MCv/superintelligence-6-intelligence-explosion-kinetics |
| 12 | Yampolskiy (2015), "From Seed AI to Technological Singularity via Recursively Self-Improving Software", arXiv 1502.06512 | Preprint。有二手来源称它在 AGI-15 以 PR 形式发表，本次未核实 | 三级分类：self-modification（不带来改进，例如代码混淆）；self-improvement / weak（例如 GA/GP 在固定 fitness 下优化参数）；recursive / strong self-improvement，要求系统 "gets better at getting better"，并能多次把原算法替换成全新的方法。metalearning 被归为 Yudkowsky 所说的 non-recursive optimization（一个组件优化另一个组件）；单纯加内存、加硬件不算 recursive | 列出一系列极限：物理计算极限、复杂度类、Rice 定理（无法检验改写后的程序是否"更聪明"）、"multidimensionality of optimization"（没有严格的改进，只有取舍）、Löb 定理、procrastination paradox、错误累积、"Munchausen obstacle"（越复杂越难理解自己）。举了 supercompilation 递归应用到自身的例子：第一次约 20%，第二次约 3%，然后收敛到零改进的不动点。提出 RSI Convergence Theory | https://arxiv.org/abs/1502.06512 |
| 13 | Benthall (2017), "Don't Fear the Reaper: Refuting Bostrom's Superintelligence Argument", arXiv 1702.08495 | Preprint | 沿用 Bostrom 的 dI/dt = O(I)/R，其中 recalcitrance 是 "the resistance of the system to being improved" | 如果 O(I)=αI+β 而 R 是常数，就是指数爆炸；如果 R 也随 I 线性增长，增长率会趋于 α_o/α_r，变成**线性**增长。用 Bayesian predictor 论证：在准确性、速度、prior 这三个维度上，纯算法的自我改进最终 recalcitrance 都趋于无穷；真正的瓶颈是 data 和 hardware，而这两者取决于环境，不是 agent 自身 | https://arxiv.org/abs/1702.08495 |
| 14 | Aghion, B. Jones & C. Jones (2017/2019), "Artificial Intelligence and Economic Growth", NBER WP 23928；收入 *The Economics of AI: An Agenda* (2019) | PR（书章）。只读了摘要页 | 把 AI 看作自动化的最新形态，讨论 singularity 是否可能 | Baumol 式瓶颈：增长可能受制于 "what is essential and yet hard to improve" | https://www.nber.org/papers/w23928 |
| 15 | Davidson (2023), compute-centric takeoff 报告（Open Philanthropy）；作者本人在 LessWrong 写的摘要 | Preprint（机构报告）。报告原页 403，用的是作者摘要 | takeoff speed 定义为从 20%-AI（能 readily 自动化 20% 的认知任务）到 100%-AI 的时间。模型是 semi-endogenous growth，包含 AI 自动化硬件和软件 R&D 的反馈 | 在 AGI 于 2100 年前出现的条件下，作者的概率是：<3 个月约 10%，<1 年约 25%，<3 年约 50%，<10 年约 80%。effective FLOP gap 最佳估计为 1e4。从 AGI 到 superintelligence 最佳猜测"不到一年" | https://www.lesswrong.com/posts/Gc9FGtdXhK9sCSEYu/what-a-compute-centric-framework-says-about-ai-takeoff ；模型：https://takeoffspeeds.com/ |
| 16 | Besiroglu, Erdil & Ho (2024), "Do the Returns to Software R&D Point Towards a Singularity?", Epoch AI | Essay/Blog（研究机构），基于 Erdil, Besiroglu & Ho, "Estimating Idea Production: A Methodological Survey", arXiv 2405.10494 | 采用 Jones (1995) 的 idea production 函数 (1/A)dA/dt = θA^(−β)E^λ，回报参数 r = λ/β | 在研发完全自动化、算力恒定（E∝A）的情况下，当且仅当 r>1 时出现双曲增长（有限时间发散）。估计：computer chess r≈0.83（SE 0.15）；CV、SAT、LP、RL 的中位数 >1 但都不显著；美国整体经济 r≈0.25–0.32。结论原话："do not believe the current empirical data provides strong evidence" | https://epoch.ai/blog/do-the-returns-to-software-rnd-point-towards-a-singularity |
| 17 | Eth & Davidson (2025), "Will AI R&D Automation Cause a Software Intelligence Explosion?", Forethought | Preprint/Essay（研究机构） | software intelligence explosion (SIE)：在现有硬件上，AI 通过改进 AI 软件形成反馈环。ASARA 指能自动化全部 AI R&D 任务的系统。r 定义为：累积软件 R&D 投入每翻一倍，软件效率翻几倍 | r>1 是 SIE 的条件。估计：调整前约 1–4；考虑硬件恒定后约 0.5–2；结论是 SIE "至少相当可能"，但无法确信 | https://www.forethought.org/research/will-ai-r-and-d-automation-cause-a-software-intelligence-explosion |
| 18 | Davidson & Houlden (2025), "How quick and big would a software intelligence explosion be?", Forethought | Preprint/Essay（研究机构） | 定量模拟 SIE：AI 劳动随软件水平增长，用于实验的算力固定 | r_cog 取 log-uniform 0.4 / 1.2 / 3.6，并且每当软件翻倍就线性下降，到 effective limit 时降到 0。结果：约 60% 概率把 ≥3 年的进展压缩进 ≤1 年；约 20% 概率压缩 ≥10 年 | https://www.forethought.org/research/how-quick-and-big-would-a-software-intelligence-explosion-be |
| 19 | Trammell (2026), "Parallelization constraints could delay a technological singularity", Epoch AI report | Preprint/Report | 认为标准模型隐含"研究可以无限并行"的假设，于是引入 parallelization technology 作为额外的投入 | 自动化带来的研究者数量跳升，受限于"分工、协调、整合"技术的进步速度。这个参数"几乎没被研究过" | https://epoch.ai/publications/parallelization-constraints-could-delay-a-technological-singularity |
| 20 | Liu & van der Schaar (2025), "Position: Truly Self-Improving Agents Require Intrinsic Metacognitive Learning", ICML 2025 (PMLR 267) | PR（会议 position paper）。只读了摘要页 | 把 extrinsic metacognition（"fixed, human-designed loops that limit scalability and adaptability"）和 intrinsic metacognition（agent 自己评估、反思、调整自己的学习过程）区分开 | 认为现有 self-improving agents 大多依赖 extrinsic 机制，"truly" self-improving 需要 agent 自己掌握 metacognitive knowledge、planning 和 evaluation | https://arxiv.org/abs/2506.05109 |
| 21 | Gao et al. (2025/2026), "A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence", arXiv 2507.21046 v4 | Preprint（survey） | 形式化为 f(Π, τ, r) = Π′：根据自身轨迹和反馈，修改参数、上下文、工具或架构。三条纳入标准：更新依赖经验；更新持久并改变 policy；有自主探索 | 把 self-training / self-critique 类的 self-improving LLM 归为 model-centric lifelong learning。**没有**明确把 self-evolving 与 RSI 区分开 | https://arxiv.org/abs/2507.21046 |
| 22 | Fang et al. (2025), "A Comprehensive Survey of Self-Evolving AI Agents", arXiv 2508.07407 | Preprint（survey）。只读了摘要 | 利用交互数据和环境反馈自动改进 agent system；框架由 System Inputs、Agent System、Environment、Optimisers 四部分组成 | 摘要中没有讨论 RSI 的 returns 或复利 | https://arxiv.org/abs/2508.07407 |
| 23 | ICLR 2026 Workshop on AI with Recursive Self-Improvement（Rio de Janeiro，2026-04-26；组织者包括 Zhuge、Zeng、Schmidhuber 等） | 工作坊（会议存在已核实；是否 archival 页面上没写） | 官网**没有给出形式定义**。用六个维度组织：改什么、何时改、如何改、在哪里运行、安全、评测 | 把 RSI 框定为一个"具体的系统问题"，需要方法和评测，没有提出 returns 或复利条件 | https://recursive-workshop.github.io/ ；https://iclr.cc/virtual/2026/workshop/10000796 |
| 24 | Tang, Ma, Li & Yuan (2026), "Generalized Agent Iteration: One Formal Framework for Iterative Policy Improvement and Recursive Self-Improvement", arXiv 2609.13406 v2 | Preprint | 用两个 dial 统一 GPI 和 RSI。dial 1："whether the mechanism that improves the agent is part of the agent itself"，在外部就是 GPI，在内部就是 RSI。dial 2：评价标准是否锚定在系统之外，分为 Anchored、Goal Drift、Fully Self-Referential | 把 Gödel machine、Darwin Gödel Machine、STOP 都归为 "Anchored RSI"。明确声明框架**不做**可达性或复杂度方面的主张，也不建模时间尺度，因此不认可复利是 RSI 定义的一部分 | https://arxiv.org/abs/2609.13406 |
| 25 | Moser, Frolov, Nauen, Raue & Dengel (2026), "The Self-Improvement Problem", arXiv 2610.21646（DFKI） | Preprint。读的是摘要页（alphaxiv） | 把 agent 视为程序，improvement 由一个不一定可计算的语义分数定义 | 一般情形下不存在既 reliable（从不接受非改进）又 non-conservative（从不错过改进）的终止性 gate，作者自己说这接近 Rice 定理的重述。即使在可判定、有界评测的情形，一串被接受的更新也可能让有界分数每轮上升而真实分数下降 | https://alphaxiv.org/abs/2610.21646v1 |
| 26 | Zhang, Yuan & Zhang (2026), "Self-Reference in Large Language Models: The Introspection Threshold for Recursive Self-Improvement", arXiv 2607.04277 | Preprint。只读了摘要 | 把 RSI 分为 sustainable 和 degenerative 两种；sustainable 需要 introspection，即模拟自身运作并有针对性地修改 | 类比 von Neumann 自复制的复杂度阈值；认为当前 LLM 只有 quasi-introspection，结构性瓶颈包括自我访问不完整、前馈 Transformer、计算类限制 | https://arxiv.org/abs/2607.04277 |

---

## 2. RSI 与 self-training（self-distillation / self-play）的区别：各来源怎么说或隐含了什么

用户的工作定义是："一个完整模型把自己改进成一个更强的完整模型，并且改进是复利的"。下面按四个维度整理。每一格写的是来源原意或明显的隐含；需要外推的地方标【推断】。

### 2.1 谁在修改"改进过程"本身（improver 是否在系统内部、是否也被改写）

- **Schmidhuber 2003/2006**：标准最强。proof searcher（也就是改进机制）属于可重写代码的一部分，所有 meta-level 被"折叠为一"。按这个标准，用固定的人工算法（例如固定的 RL 或蒸馏配方）去更新权重，**不算** Gödel 意义上的 self-improvement。
- **Yudkowsky 2008 / 2013；Yampolskiy 2015 转引**：如果"一个组件优化另一个组件"，就是 non-recursive optimization。metalearning、GA/GP 都只属于 weak self-improvement。【推断】self-distillation 和 self-play 通常是固定的训练算法加上模型自己生成的数据：模型产出数据，但"改进规则"仍是外部固定的，所以按这一派的标准更接近 weak / non-recursive。
- **Tang et al. 2026 (GAI)**：把这一点直接做成定义的第一个 dial。改进机制在 agent 外就是 GPI，在 agent 内就是 RSI。它承认 self-training loop 可以作为一种 modifier，但没有逐一给 self-play 或 self-distillation 归类。
- **Liu & van der Schaar 2025**：同一个区分的 metacognition 版本，即 extrinsic（人设计的固定循环）对比 intrinsic（agent 自己决定学什么、怎么学）。
- **Mahoney 2008/2010**：判据相反，看的是"有没有外部输入"，不看谁设计了改进规则。只靠自己源码（例如自我对弈）的改进算 self-improvement，有外部数据就算 learning。所以在 Mahoney 的框架里，self-play 恰恰是 self-improvement 的范例，只是被证明增长受限。

### 2.2 Reflexivity：被改进的是不是"改进者自身的能力"

- **Yampolskiy 2015**：RSI 要求 "gets better at getting better"，即每一代在**优化下一代**方面也更强。买更多内存或加硬件不算，因为系统"并没有变得更擅长改进自己"。
- **Chalmers 2010**：要求被放大的能力 G 正好是"创造具有 G 的系统的能力"（self-amplifying capacity）。如果被放大的是 H（我们关心的某种能力），还需要额外的 correlation 前提才能连到 G。【推断】self-distillation 提升的往往是 H（某个任务的表现），未必提升 G（产生下一代的能力），这正对应 Chalmers 所说的 correlation obstacle。
- **Yudkowsky LOGI 2007**：strong self-improvement 的改进直接正反馈到"改进过程本身"的智能上。人类文明是常量的人类智能作用于不断增长的知识库，属于 weak。
- **Yudkowsky 2013**：用的是宽松的实用定义。只要整个系统里"可用的认知问题解决能力"增加了，包括写出 successor、全新 AI、赚钱买算力，都可以算 self-improvement。他明确反对用过窄的形式定义（点名 Mahoney 的 Kolmogorov 定义）来排除现实中的情形。
- **Bostrom 2014（二手）**：crossover 的概念强调系统的投入主要来自它自己的产出。

### 2.3 Compounding / returns > 1

这是对应用户 "Δ₂ > 0 across generations" 检验最直接的一维。

- **Yudkowsky 2013**：k<1 熄火，k≈1 平稳指数增长，prompt k≫1 爆炸。Yudkowsky 2008 的"再投资"比喻也说明，复利的必要条件是产出回流到投入。
- **Chalmers 2010**：proportionality thesis（δ 带来 ≥δ）。diminishing returns（10%→5%→2.5%）是最严重的 structural obstacle。他特别指出，结论依赖智能的度量尺度，在某个尺度下成立的论证，换成 log 尺度可能就不成立。【推断】这意味着 Δ₂ 检验必须先固定度量尺度（例如 log-compute-equivalent、Elo、任务时长），否则同一组数据可以既是 Δ₂>0 又是 Δ₂<0。
- **Bostrom / Benthall**：dI/dt = O(I)/R(I)。如果 R 随 I 增长得和 O 一样快，增长就退化为线性。
- **Yampolskiy 2015**：明确说"truly RSI"被理论化为不受 diminishing returns 约束；同时用 supercompilation 的 20%→3%→0 说明大多数自我应用的系统会收敛到不动点。他还提出一个尚未解决的问题：改进量应该用绝对值还是相对值来衡量。
- **Epoch / Forethought**：r>1（在 r=λ/β 或"软件翻倍数/累积投入翻倍数"的定义下）是加速的充要条件，详见第 3 节。
- **Tang et al. 2026**：把复利主张归到 Yudkowsky 一派的历史叙述，在自己的形式化里明确不涉及。也就是说，至少有一种 2026 年的形式定义把"复利"从 RSI 的**定义**中拿掉，只当作一个经验问题。

### 2.4 是否需要 generality

- **Good 1965**：前提是在**一切**智力活动上超越人类，其中包括机器设计。
- **Yudkowsky 2013 §3.10**：Good 的条件是充分条件，但"远多于必要"。一个比自然选择高效、但不如人类科学家的 AI，也可能点燃 FOOM。LOGI 2007 则认为 RSI 需要相当程度的 general intelligence 才开始见效。
- **Chalmers 2010**：原则上递归路径可以从低于人类的水平开始，只要那个系统具备自我放大的能力。需要 generality 的地方在于 correlation，即 G 的爆炸要连带 H。
- **Hall 2007**：关键阈值是 "universality"，即可以无界自我改进的学习能力水平。人类个体可能略低于这个阈值，科学共同体整体高于它。
- **Eth & Davidson 2025**：不要求完全通用，要求的是 ASARA，即能替代任何远程 AI R&D 工作者。这是"研发领域内的通用"。
- **Yampolskiy 2015**：认为启动 RSI 所需的最低智能"可能与人类相当"，但并不排除低于人类的情形。
- **Mahoney 2008/2010**：在形式上 RSI 只针对固定目标 G 定义，不要求通用。

### 2.5 小结：文献中可以抽出的判据（供用户取舍，不是结论）

| 判据 | 支持把它当作 RSI 必要条件的来源 | 反对或弱化的来源 |
|---|---|---|
| A. 改进机制本身在系统内并且可被改写 | Schmidhuber、Yudkowsky (LOGI/2008)、Yampolskiy、Tang et al.、Liu & vdS | Yudkowsky 2013 的宽松实用定义（产出 successor 即可）；Mahoney（判据是有没有外部输入） |
| B. 被提升的是"产生下一代的能力" | Yampolskiy、Chalmers（G）、LOGI（strong） | 无明确反对；Chalmers 认为 G 与 H 可以分离 |
| C. 跨代回报 ≥1 或加速（复利） | Yudkowsky 2013（k）、Chalmers（proportionality）、Bostrom、Epoch/Forethought（r>1）、Yampolskiy（"truly RSI"） | Tang et al. 2026 不把它放进定义；Benthall、Mahoney 认为在一般条件下难以满足 |
| D. 需要 generality 或某种阈值 | Good、Hall（universality）、LOGI、Eth & Davidson（ASARA） | Chalmers、Yudkowsky 2013（低于人类水平也可能） |

【推断】按 A+B+C 来看，典型的 self-distillation 和 self-play（固定训练算法、单任务指标、单代或少数几代增益）最多满足 "weak self-improvement"。文献中**没有**哪篇来源直接对 self-distillation 下判断；Mahoney 是唯一直接点名 self-play 的，并且把它算作 self-improvement，同时证明这类无外部输入的过程增长受限。

---

## 3. Returns / takeoff 模型：参数、估计值、范围与出处

### 3.1 参数定义对照

| 模型 | 参数 | 定义 | 加速条件 |
|---|---|---|---|
| Yudkowsky 2013 | k（隐喻性的"有效增殖系数"） | 认知再投资的回报 | prompt k > 1 |
| Chalmers 2010 | δ 与 proportionality | 智能提高 δ，下一代设计能力至少也提高 δ | 对一类系统和有限的 n<k 成立即可 |
| Bostrom 2014（经 Benthall） | O(I)、R(I) | dI/dt = O(I)/R(I) | O 随 I 增长得比 R 快；Benthall 证明如果 R ∝ I，增长就变成线性 |
| Epoch（Besiroglu, Erdil & Ho 2024） | r = λ/β | 来自 (1/A)dA/dt = θA^(−β)E^λ：λ 是研究投入的回报弹性，β 是"想法越来越难找"的程度 | 研发完全自动化且算力恒定时，r>1 当且仅当出现双曲增长（有限时间发散） |
| Forethought（Eth & Davidson 2025） | r | 累积软件 R&D 投入每翻一倍，软件效率翻几倍 | r>1 时每次翻倍所需投入少于上一次，即 SIE；r=1 是指数增长；r<1 熄火 |
| Forethought（Davidson & Houlden 2025） | r_cog，以及 λ=0.6、α=0.5、β=0.25、p | 在 r 的基础上区分认知劳动和算力的份额，并加入并行劳动的递减回报 p | r_cog>1 时加速；r_cog 随软件逼近上限而线性下降到 0 |
| Davidson 2023 | "returns to software" 和 effective FLOP gap | semi-endogenous growth；takeoff 定义为 20%→100% 认知任务的自动化 | 摘要页没有给出显式阈值。一条评论提到默认值 1.25，敏感性分析用 2.5（这一点只是评论里的说法，可靠性中等） |
| Trammell 2026 | parallelization technology | 研究的有效投入受限于原始投入和并行化技术两者中较小的一个 | 只有当并行化技术和研究投入同速改进时，标准的爆炸结论才保留 |

### 3.2 估计值

| 领域或量 | 估计 | 出处 |
|---|---|---|
| Computer chess (Stockfish) | r ≈ 0.83（SE 0.15）；Forethought 表中记为 0.8 | Epoch blog 2024；Forethought 2025 |
| Image recognition（训练效率） | 中位数 1.4（5–95%：0.8–2.4） | Forethought 2025 的表格，标注出自 Epoch arXiv 2405.10494。Epoch blog 只说"中位数 >1、不显著" |
| RL 样本效率 | 1.6 | 同上 |
| SAT solvers | 3.5 | 同上 |
| Linear programming | 1.1 | 同上 |
| 一般算法（MIT 分析，粗估） | 约 4 | Forethought 2025 |
| LLMs | 没有数值。Epoch 内部初步分析认为"与图像识别相当或更高" | Forethought 2025 |
| 软件 r 的总体判断 | 调整前约 1–4；考虑硬件恒定后约 0.5–2 | Forethought 2025 |
| r_cog 的先验 | log-uniform，0.4 / 1.2 / 3.6 | Davidson & Houlden 2025 |
| 距 effective limits | 6 / 11 / 16 年的近期进展量（约 6–16 OOMs） | Davidson & Houlden 2025 |
| ASARA 带来的初始加速 | 2× / 8× / 32× | Davidson & Houlden 2025 |
| SIE 规模 | 把 ≥3 年进展压进 ≤1 年的概率约 60%（57%）；压进 ≤4 个月约 40%。≥10 年压进 ≤1 年约 20%（18%） | Davidson & Houlden 2025 |
| 硬件的 r | 约 7；GPU 2006–2022 约 5（作者自己说可能高估） | Forethought 2025 转引 Davidson 早期估计 |
| LLM 效率翻倍时间 | 约 6 个月（训练约 8 个月、推理约 4 个月的中点） | Forethought 2025 |
| 美国整体经济 | r ≈ 0.32（Bloom et al. 2020）；Epoch 复现的中位数约 0.25 | Epoch blog 2024 |
| Takeoff（20%→100% 认知任务） | <3 个月约 10%，<1 年约 25%，<3 年约 50%，<10 年约 80%。如果按 AI R&D 的 20% 起算：<1 年约 10%，<3 年约 30%，<10 年约 70% | Davidson 2023（LW 摘要） |
| Effective FLOP gap | 1e4 | Davidson 2023 |

### 3.3 与用户 "Δ₂ > 0 across generations" 的对应关系【推断，待用户判断】

- Epoch 和 Forethought 的 r 衡量的是：在 log(软件效率) 对 log(**累积**研究投入) 的关系中，每单位投入的回报是否随投入累积而变便宜。如果研究投入本身由 AI 提供，并且与软件水平成正比（这正是 "AI improves AI" 的设定），那么 r>1 等价于"每翻倍一次所需时间越来越短"，也就是时间轴上的加速。
- 用户的 Δ₂ 是按"代"测的：第 n+1 代相对第 n 代的增益，减去第 n 代相对第 n−1 代的增益。如果每代投入的资源量固定，并且增益用 log 尺度度量，那么 Δ₂>0 大致对应"每代改进的回报不下降"，在精神上接近 r≥1。但两者至少在三处不等价：(a) r 是对**累积**投入定义的，Δ₂ 是对代际差分定义的；(b) Epoch 的 r 来自"人类研究者 + 增长的算力"的历史数据，并不是 AI 自我改进的数据，Forethought 自己也指出算力恒定会降低 r；(c) Chalmers 已经指出度量尺度会改变结论。
- Benthall 和 Davidson & Houlden 都给出一个可以检验的形状：如果 R 随 I 增长，或 r 随逼近上限而下降，Δ₂ 会从正变负。因此只观察少数几代时，Δ₂>0 可能只反映"离上限还远"，不能证明增长会持续。

---

## 4. 文献中的分歧与开放问题

1. **定义宽窄之争。** Yudkowsky 2013 主张宽松的实用定义：凡是增加系统内认知能力的再投资都算，并点名批评 Mahoney 的 Kolmogorov 定义和 Bringsjord 的 hypercomputation 定义"不适合讨论问题"。Schmidhuber、Yampolskiy 和 Tang et al. 则要求改进机制本身在系统内并可被改写。这一分歧直接决定 self-distillation 或 self-play 能不能叫 RSI。
2. **外部输入是否必要。** Mahoney 和 Benthall 认为，不引入新的外部信息（数据、硬件）的纯自我修改，增长有严格上限：前者给出 O(log n) 的复杂度上界，后者论证 recalcitrance 趋于无穷。Yudkowsky 2013 反驳说，现实中的影响力并不要求孤立系统增加 Kolmogorov 复杂度。【推断】这对应一个实际问题：self-play 或 self-distillation 的增益究竟来自"新信息"（环境反馈、验证器），还是只是"重新组织已有信息"。
3. **Returns 到底大于还是小于 1。** Chalmers 认为 diminishing returns 是最严重的 structural obstacle，但倾向于相信 proportionality。Yampolskiy 用 supercompilation 收敛和 low-hanging fruit 论证 diminishing returns 很快出现。Epoch 的历史估计在各领域分布在 0.8–3.5，但多数不显著。Forethought 认为在算力恒定时 r 大约在 0.5–2，跨过了 1。Epoch 作者对 software-only singularity 持怀疑态度。
4. **瓶颈在算法、算力、数据，还是并行化。** Benthall 认为是 data 和 hardware。Forethought 承认算力恒定会降低 r，但认为不足以阻止 SIE。Aghion–Jones–Jones 指出 Baumol 式的"必要且难改进"环节。Trammell 2026 提出并行化技术是一个几乎没被研究过的参数。
5. **验证或评价问题。** Yampolskiy 列出 Rice 定理、"multidimensionality of optimization"、无法测试高于自身水平的智能。Moser et al. 2026 证明一般情形下不存在既可靠又不保守的改进 gate，而且有界的代理分数可以持续上升，同时真实目标在下降。Tang et al. 2026 用评价标准是否"anchored"来区分 Anchored、Goal Drift 和 Fully Self-Referential 三类 RSI。这与 Goodhart 和 reward hacking 直接相关。
6. **自指或逻辑障碍。** Gödel machine 只能采纳它能证明有效的改进；tiling agents 面临 Löbian obstacle（无法直接信任同等强度的继任者）；Yampolskiy 还提到 procrastination paradox 和 "Munchausen obstacle"；Zhang et al. 2026 提出 LLM 缺乏真正的 introspection。
7. **起点阈值与 generality。** 需要多强才能启动？Good 和 Hall 认为要很高（universality 或研究科学家共同体）；Yudkowsky 2013 和 Chalmers 认为可能低于人类水平。Yampolskiy 认为这是"RSI 研究的根本开放问题"之一。
8. **速度。** Bostrom 认为 fast 或 moderate takeoff 更可能；Hall 认为会更慢；Davidson 2023 给出的中位数约为 3 年（20%→100%）；Davidson & Houlden 2025 认为约 60% 的概率在 ≤1 年内压缩 ≥3 年的进展。Yampolskiy 指出，"更强的系统改进更快"与"新手低垂果实更多"哪个占主导，并不清楚。
9. **近期学术定义尚未收敛。** ICLR 2026 的 RSI workshop 没有给出形式定义；self-evolving agents 的 survey（Gao et al.、Fang et al.）没有把 self-evolving 与 RSI 明确区分开；2026 年的 Tang et al. 是本次检索到唯一明确把 GPI 与 RSI 形式化区分的论文，但它是 preprint，而且有意排除了复利。

---

## 5. 未核实或未能完整打开的来源

| 来源 | 状态与原因 |
|---|---|
| Chollet (2017), "The impossibility of intelligence explosion"（Medium 随笔，非同行评审） | **未核实**。Medium 返回 403，第三方镜像被权限拒绝。Barak (2017, windowsontheory) 的回应帖也没有转述其核心论证。本文没有使用任何 Chollet 的内容 |
| Bostrom (2014), *Superintelligence* ch. 4 | 原书**未打开**。表中内容来自 Grace (2014) LessWrong 读书会摘要和 Benthall (2017) 的转述，属于二手来源 |
| Good (1965) 原文 | 只拿到扫描 PDF，没有文字层。定义段落按 Chalmers 2010 和 Yudkowsky 2013 的逐字转引核对；文中其他内容（例如可能的附加条件）未核实 |
| Davidson (2023) Open Philanthropy 完整报告 | openphilanthropy.org 重定向到 coefficientgiving.org，返回 403。只用了作者本人在 LessWrong 上的摘要。"returns to software 默认 1.25" 来自页面评论，没有在报告正文核实 |
| Erdil, Besiroglu & Ho (2024), arXiv 2405.10494 正文 | 只打开了摘要页，摘要中没有数值。各领域 r 的数值（1.4、1.6、3.5、1.1）来自 Forethought 2025 的表格（它注明引自这篇），没有在原文核实 |
| Yampolskiy (2015) 的 PR 版本（AGI-15） | 只是二手搜索结果的说法，未核实。本文引用的是 arXiv 版本 |
| Nivel et al. (2013), "Bounded Recursive Self-Improvement", arXiv 1312.6764 | 只读了摘要：在设计者设定的边界内、基于经验的自我改进，依赖 autocatalysis、endogeny 和 reflectivity。正文未读，所以没放进主表 |
| Liu & van der Schaar (2025)；Fang et al. (2025)；Zhang et al. (2026)；Aghion et al. (2017/2019)；Moser et al. (2026) | 只读了摘要或摘要页，正文论证未核实 |
| Ren et al. (2026), "Self-Improvements in Modern Agentic Systems: A Survey", arXiv 2607.13104（含 Schmidhuber） | 只读了摘要：把 self-improvement 定义为 "self-induced update operator"，作用于参数或 scaffold。摘要中没有定义 RSI，也没有区分 self-training 或 self-play。正文未读 |
| Gao et al. (2025/2026) survey | 只读了 HTML 全文前约 100k 字符（共约 429k），后文中是否讨论 RSI 未核实 |
| Hanson–Yudkowsky "AI-FOOM debate"（2008，博客辩论）；Sotala (2017)；Bringsjord (2012) | 没有打开，未使用 |
| Trammell (2026) 完整论文 | 只读了 Epoch 的报告摘要页 |
