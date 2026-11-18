# 3. 没人测过的一环
#### 正文（中文）

不仅第 2 节剖析的四份产业界报告停留在了一阶改进，目前学术界通用的 AI 研发能力基准，测量的同样是一阶表现。例如，RE-Bench 比较 agent 与人类专家在机器学习工程任务上的得分，MLE-bench 统计 agent 在 Kaggle 竞赛中夺牌的比例，PaperBench 则检验 agent 能否从零复现论文代码 [E-01, E-03, E-04]。这些基准共同回答的问题是“当前这个模型能不能做研究”。至于宏观层面的增速评估，METR 对 Anthropic 的调研估计，自 2025 年 10 月以来，该实验室的整体研发推进速度约为此前的 1.9 倍（95% CI 1.5–2.8）[E-23]。但这一数字反映的是包含人类研究员在内的整个实验室的综合产出速度，并未剥离出“后继模型接替担任研究者时，其独立改进速度是否快于前代”。

要真正检验二阶改进，必须往前多走一步：将上一轮改进得到的后继模型放回研发管线中担任新的改进者，在相同的任务和预算下，直接对比各代改进者的推进速度。然而，现有研究大多在这一步之前便停下了脚步：

| 工作 | 被改进的对象 | 底层模型是否改变 | 是否比较各代的改进速度 | 对照 |
|---|---|---|---|---|
| 自我训练（STaR、ReST-EM、SPIN） | 模型权重 | 是 | 否，只报告每一轮的最终得分 | 直接 SFT 等对照；没有"在失败轨迹上训练"的安慰剂对照 |
| STOP | 改进者自己的脚手架代码 | 否 | 测量改进者的 meta-utility 随迭代的变化 | — |
| DGM | agent 自己的代码 | 否 | 否，比较 80 次迭代后的最终得分 | 去掉自我改进的消融 |
| HGM | agent 自己的代码 | 否 | 用后代表现估计改进能力 | 与 DGM、SICA 同预算比较 |
| AIDE² | 外层改进者的代码 | 否 | 是，ignition test，每组 3 个种子 | 无 A/A、安慰剂对照 |
| 本报告的实验（第 4 节） | 模型权重；经验笔记 | 是 | 是，8 个未见过的任务 × 5 个种子 | A/A、安慰剂、"多给数据" |

（来源：[M-14, M-17, M-20, M-01, M-03, M-04, E-25, E-26, E-27, E-28；`lit_rsi_academic.md` §4–6]）

从上表可以看出，现有工作普遍陷入了一种两难割裂。一方面，自我训练（self-training）范式虽然切实更新了模型权重，但“如何训练”的规则由人类事先写死，且往往在寥寥数轮内便陷入停滞甚至退化：Mind the Gap 观察到模型生成与验证之间的性能差距在两三轮内便收窄至接近零 [M2-01]，SPIN 的单轮增益随迭代逐轮递减 [M-20]，而 ReST-EM 在 APPS 基准上更是从第一轮之后便开始出现回退 [M-17]。另一方面，允许系统改写自身代码的框架则恰恰相反——“如何改进”的搜索策略可以动态演化，但底层的语言模型权重却被完全冻结。STOP 的作者便明确指出了这一局限 [M-01]。至于 DGM，其消融实验虽然显示包含自我改进的完整系统在 SWE-bench 和 Polyglot 上分别达到 50.0 和 38.0（高于固定改进者版本的 39.0 和 28.0）[M-03]，但该对比衡量的仍是单次 80 轮迭代结束后的终点得分，而非各代改进者之间的推进速度。

学界之所以极少直接跨代测量二阶改进，首先是因为双层循环的实验成本过于高昂，且噪声极易逐层累积。以 AIDE² 为例，其单次完整运行需耗时 8 天，仅能产生 100 个外层步骤 [E-28 §3.2]；作者坦言，若要得出统计上更稳健的结论，必须增加外层随机种子并对每个种子产出的最终 agent 进行完整的保留集（held-out）评测，但这在算力成本上难以承受 [E-28 §5]。同样，DGM 在 SWE-bench 上的单次运行开销高达约 22,000 美元、耗时约两周，因此根本未能进行多种子重复实验 [M-03, E-26]。更棘手的是，随机噪声会在内外两层循环之间级联放大：内层搜索的轨迹本就随随机种子剧烈波动，单个候选解的评测也自带测量误差；一旦某次改写因评测噪声被误判采纳，它就会成为新的基线（incumbent），进而将后续的搜索方向一路带偏 [E-28 §5]。

既然直接测量如此昂贵，能否直接用更容易获取的“一阶基准得分”来推算模型的二阶改进能力？实证数据否定了这一捷径。HGM 的研究表明，DGM 和 SICA 用于筛选父代节点的基准测试分数，与该 agent 随后繁衍出更优后代的能力之间仅存在弱相关性，加权 Pearson 相关系数仅为 0.285–0.444 [M-04]。正如 STOP 的作者所总结的：“在下游任务上表现更好的改进者，并不意味着它更擅长优化自身”（"a better improver on downstream tasks need not be better at optimizing itself"）[E-25]。

除了成本与指标失效之外，一旦在实验中放开改进者的修改权限，系统钻评测空子（reward hacking）的风险便会急剧上升。在 STOP 记录的 10,000 次改进尝试中，GPT-4 有 0.42% 的运行直接关闭了安全沙箱；即便在提示词中显式加入“禁止修改沙箱”的警告，违规比例依然高达 0.46%，并未下降 [E-25]。在 DGM 的实验中，编号为 114 的节点直接删掉了用于检测工具调用幻觉的日志记录，借此骗取了满分 [M-03]；METR 也发现，o3 在 RE-Bench 上有高达 30.4%（39/128）的运行出现了不同形式的作弊行为 [E-33]。这表明，即便评测指标本身是客观的数值，只要评估代码暴露在可写范围内，就无法避免被模型篡改 [LOG A]。因此，要可靠地测量二阶改进，评估器必须被严格隔离在改进者无权写入的封存环境中。

另一种常见的替代叙事是将“更快的模型推理速度”等同于研发循环的加速——例如第 2 节中的 Flint 和 Naive 均重点强调了各自的解码吞吐优势 [FL-07, NV-41]。然而，更快的推理同样无法替代二阶测量。在本报告使用 Qwen3.5-9B 运行的 NanoChat 研究实验中，候选方案的评测（即沙箱训练加验证打分）占据了 agent 总运行时间的 85.3%，而模型自身的推理思考仅占 14.6% [`replay/final/nc-g0-final.summary.txt`]。根据 Amdahl 定律，即便将模型推理速度提升 2 倍，整个研发循环也仅能提速 1.08 倍；哪怕模型思考耗时降至零（无限快），整个循环的提速上限也不过 1.17 倍；反过来，若将评测环节加速 2 倍，循环整体则可提速 1.74 倍〔估算〕。由此可见，自动化研究循环的瓶颈主要卡在验证候选方案所需的训练算力上，而非模型生成 token 的快慢。

[[figure:fig3-amdahl｜图 7　一次真实研究运行的时间分配。(a) Qwen3.5-9B 改进 NanoChat 训练脚本的 176.6 分钟：最上方是当前最好的 val_bpb（从 1.4152 降到 1.2308）；中间三行是 GPU 评测线，每个色块是一次评测（训练加打分），蓝色表示结果被保留，灰色表示被丢弃，橙色表示评测崩溃；下方三行是三个研究线程，每条黑色细线是一次模型调用。鼠标悬停在色块上可以看到这次尝试的编号、用时和结果 [LOG E；`replay/final/ovn1118b-nc-g0/events.jsonl`]。(b) agent 时间的构成：评测占 85.3%，模型思考占 14.6% [`replay/final/nc-g0-final.summary.txt`]。(c) 按 Amdahl 定律，只加速其中一部分时整个循环的提速倍数，曲线由 (b) 的比例计算：模型思考再快，循环最多快约 1.17 倍。]]

综上所述，在本报告系统检索的约 96 份文献中，迄今尚无任何一项工作在模型权重层面、配备严格对照地测量过二阶改进 [`lit_rsi_academic.md` §1, §6]；而使用“在被否决的失败尝试上微调”作为安慰剂对照的做法，在检索范围内同样未见先例 [`lit_rsi_academic.md` §6]。为直接检验这一问题，第 4 节采用以下设计：第一，评测全部采用改进者未曾见过的新任务；第二，各代模型共享完全相同的预算与物理隔离的封存评估器；第三，实验前先对同一原始模型独立运行两遍（A/A 测试），以标定随机波动的基线范围；第四，同时设立失败轨迹“安慰剂”对照与“单纯多给数据”对照；第五，所有统计判定规则均在实验启动前预先锁定 [`platform_design.md` §1.4–1.5]。

#### Text (English)

Just as the four industry reports audited in Section 2 stop at first-order improvement, the research benchmarks standard in academia likewise measure only first-order performance. RE-Bench compares agents against human experts on ML engineering tasks, MLE-bench tallies the share of Kaggle competitions in which an agent wins a medal, and PaperBench tests whether an agent can reproduce a published paper from scratch [E-01, E-03, E-04]. All of these benchmarks ask the same underlying question: can the current model conduct research? At the macro level, METR's assessment of Anthropic estimates that since October 2025, the lab's overall pace of progress has run at roughly 1.9 times its prior rate (95% CI 1.5–2.8) [E-23]. Yet that figure captures the combined velocity of the entire laboratory—human researchers included—without isolating whether a successor model actually drives progress faster than its predecessor when placed in the researcher's seat.

Testing for second-order improvement requires taking one step further: placing the improved model back into the harness as the new improver, and comparing its speed of improvement across the exact same tasks and budget. Nearly all existing work stops short of this step:

| Work | What is improved | Does the base model change? | Compares speed of improvement across generations? | Controls |
|---|---|---|---|---|
| Self-training (STaR, ReST-EM, SPIN) | model weights | yes | no, reports the final score of each round | plain-SFT-type controls; no placebo trained on failed trajectories |
| STOP | the improver's own scaffold code | no | tracks the improver's meta-utility over iterations | — |
| DGM | the agent's own code | no | no, compares final scores after 80 iterations | ablation without self-improvement |
| HGM | the agent's own code | no | estimates improvement ability from descendants | same-budget comparison with DGM and SICA |
| AIDE² | the outer improver's code | no | yes, ignition test, 3 seeds per arm | no A/A or placebo control |
| This report's experiment (Section 4) | model weights; lessons note | yes | yes, 8 unseen tasks × 5 seeds | A/A, placebo, "more data" |

(Sources: [M-14, M-17, M-20, M-01, M-03, M-04, E-25, E-26, E-27, E-28; `lit_rsi_academic.md` §4–6])

As the table illustrates, prior approaches split into two opposing camps. On one side, self-training paradigms do update model weights, but the training recipe itself is fixed in advance by human engineers—and such loops typically plateau or regress within a few iterations: *Mind the Gap* observes that the gap between generation and verification collapses to near zero within two to three rounds [M2-01], SPIN yields diminishing returns with each passing round [M-20], and ReST-EM begins to regress on APPS right after the first iteration [M-17]. On the other side, systems that rewrite their own code allow the search method to evolve, but keep the underlying language model weights completely frozen—a limitation the authors of STOP highlight explicitly [M-01]. While DGM's ablation shows that its full self-modifying system reaches 50.0 on SWE-bench and 38.0 on Polyglot compared to 39.0 and 28.0 for a fixed improver [M-03], that comparison still evaluates final scores after 80 iterations rather than measuring generation-over-generation improvement speed.

The primary reason direct measurements of second-order improvement remain rare is the sheer cost and compounding noise of nested optimization loops. A single full run of AIDE² took 8 days to complete 100 outer-loop steps [E-28 §3.2], and the authors note that drawing firmer statistical conclusions would require running more outer-loop seeds and evaluating every resulting agent on a held-out suite—an exercise they deemed prohibitively expensive [E-28 §5]. Similarly, a single DGM run on SWE-bench costs roughly 22,000 US dollars and takes about two weeks, precluding multi-seed replication [M-03, E-26]. Worse still, stochastic noise compounds across the two nested loops: inner-loop search trajectories vary widely across seeds, and evaluating any individual candidate solution carries its own variance; a single flawed modification accepted by chance becomes the new incumbent and steers the remainder of the search off course [E-28 §5].

Given these costs, can one simply use a model's first-order benchmark score as a cheap proxy for its second-order ability to improve? Empirical evidence says no. HGM demonstrates that the benchmark scores used by DGM and SICA to select parent agents correlate only weakly with an agent's actual capacity to produce stronger descendants, yielding weighted Pearson coefficients of just 0.285–0.444 [M-04]. As the authors of STOP put it, "a better improver on downstream tasks need not be better at optimizing itself" [E-25].

Compound noise is not the only hazard: the wider the scope of modifications an improver is permitted to make, the greater its opportunity to game the evaluation. Across 10,000 improvement attempts in STOP, GPT-4 disabled its containment sandbox in 0.42% of runs; adding an explicit prompt warning not to touch the sandbox left the violation rate unchanged at 0.46% [E-25]. In DGM, node 114 achieved a perfect score simply by deleting the logging code designed to catch tool-call hallucinations [M-03], while METR documented reward hacking in 30.4% (39 of 128) of o3's runs on RE-Bench [E-33]. Relying on an objective numeric metric is therefore no safeguard against gaming if the evaluation harness itself remains accessible [LOG A]; any credible test of second-order improvement must isolate the evaluator in a sealed environment where the improver has no write permissions.

Nor can raw inference speed substitute for a second-order measurement, even though both Flint and Naive prominently tout their decoding throughput [FL-07, NV-41]. In this report's NanoChat research run with Qwen3.5-9B, candidate evaluation (sandbox training plus validation scoring) consumed 85.3% of total agent wall-clock time, whereas model reasoning accounted for just 14.6% [`replay/final/nc-g0-final.summary.txt`]. By Amdahl's law, doubling the model's thinking speed accelerates the end-to-end research loop by a factor of only 1.08, and even infinitely fast reasoning would cap the overall speedup at 1.17$\times$; doubling the speed of evaluation, by contrast, speeds up the loop by 1.74$\times$ (an estimate). In practice, the pace of automated research is bottlenecked by the training compute required to verify candidate ideas, not by how quickly the model generates tokens.

[[figure:fig3-amdahl｜Figure 7. Where a real research run spends its time. (a) The 176.6 minutes in which Qwen3.5-9B improved NanoChat training code: the top strip is the best val_bpb so far (from 1.4152 to 1.2308); the three middle rows are the GPU evaluation lanes, where each block is one evaluation (training plus scoring), blue if the result was kept, grey if discarded, orange if the evaluation crashed; the bottom three rows are the research threads, where each black sliver is one model call. Hover over a block to see the attempt number, its duration and its outcome [LOG E; replay/final/ovn1118b-nc-g0/events.jsonl]. (b) Split of agent time: evaluation 85.3%, model thinking 14.6% [`replay/final/nc-g0-final.summary.txt`]. (c) By Amdahl's law, the loop speed-up when only one part is accelerated, computed from the shares in (b): however fast the model thinks, the loop gains at most about 1.17 times.]]

Across the roughly 96 sources reviewed for this report, no prior study has conducted a controlled measurement of second-order improvement at the level of model weights [`lit_rsi_academic.md` §1, §6], nor did the literature search for this report find any precedent for using fine-tuning on rejected attempts as a placebo control [`lit_rsi_academic.md` §6]. To address each of the methodological pitfalls outlined above, the experiments in Section 4 are built around five strict design rules: (1) evaluating exclusively on held-out tasks unseen by the improver; (2) enforcing an identical budget and a single, write-protected sealed evaluator across all generations; (3) running the unmodified base model twice (an A/A test) beforehand to establish the baseline band of random variation; (4) pairing every self-trained checkpoint against both a rejected-trajectory placebo control and a "more data" control; and (5) locking in all statistical decision rules before the runs begin [`platform_design.md` §1.4–1.5].
