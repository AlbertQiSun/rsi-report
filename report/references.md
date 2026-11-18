# 参考文献总表（自动抽取）

> 由 `report/tools/build_references.py` 从 `research/` 的文献文件**逐字抽取**，没有改写，也没有补充任何信息。重新运行脚本即可更新。
> - 键：T = `lit_rsi/01_theory.md`，M = `02_methods.md`，E = `03_measurement.md`，L = `lit_signal_hardness.md`；RF/FL/NV/AX = 四份材料的证据台账。
> - 「类型」列沿用原文件的标注（PR = 同行评审，Preprint，blog 等）。原文件里注明"只读了摘要"或"未核实"的，引用前要看原文件的说明。
> - 同一篇文献可能出现在多个表里（例如 DGM 同时在 M、E、L 中），网页生成时再合并去重。

## T. 理论与定义（来源：`research/lit_rsi/01_theory.md`）

| 键 | 文献 | 类型 / 发表情况 | URL | 原文件行号 |
|---|---|---|---|---|
| T-01 | I. J. Good (1965), "Speculations Concerning the First Ultraintelligent Machine", *Advances in Computers* 6 | PR（书章）。原文是扫描版，无法抽取文字；下面的段落按 Chalmers 2010 和 Yudkowsky 2013 的逐字转引核对 | https://www.stat.vt.edu/content/dam/stat_vt_edu/graphics-and-pdfs/research-papers/Technical_Reports/TechReport05-3.pdf | 15 |
| T-02 | Yudkowsky (2007), "Levels of Organization in General Intelligence" (LOGI)，载 Goertzel & Pennachin (eds.), *Artificial General Intelligence*, Springer, pp. 389–501 | PR（书章） | https://intelligence.org/files/LOGI.pdf | 16 |
| T-03 | Yudkowsky (2008), "Recursive Self-Improvement", LessWrong | Essay/Blog | https://www.lesswrong.com/posts/JBadX7rwdcRFzGuju/recursive-self-improvement | 17 |
| T-04 | Yudkowsky (2013), "Intelligence Explosion Microeconomics", MIRI Technical Report 2013-1 | Preprint（技术报告） | https://intelligence.org/files/IEM.pdf | 18 |
| T-05 | Schmidhuber (2003/2006), "Gödel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements", arXiv cs/0309048 v5；相关版本发表于 LNCS 3394 (2005)、ICANN 2005，以及 *Artificial General Intelligence* (Springer 2006) | Preprint（另有 PR 版本） | https://arxiv.org/abs/cs/0309048 | 19 |
| T-06 | Omohundro (2008), "The Basic AI Drives", *Proc. First AGI Conference*, IOS Press FAIA vol. 171, pp. 483–492 | PR（会议） | https://selfawaresystems.com/wp-content/uploads/2008/01/ai_drives_final.pdf | 20 |
| T-07 | Hall (2007), "Self-improving AI: an Analysis", *Minds & Machines* 17:249–259 | PR（期刊） | https://faculty.washington.edu/seattle/Turing-Test/SelfImprovingAI.pdf | 21 |
| T-08 | Mahoney (2008, rev. 2010), "A Model for Recursively Self Improving Programs" | Preprint（个人草稿，未发表） | https://mattmahoney.net/rsi.pdf | 22 |
| T-09 | Chalmers (2010), "The Singularity: A Philosophical Analysis", *Journal of Consciousness Studies* 17:7–65 | PR（期刊） | https://consc.net/papers/singularity.pdf | 23 |
| T-10 | Yudkowsky & Herreshoff (2013), "Tiling Agents for Self-Modifying AI, and the Löbian Obstacle", MIRI early draft | Preprint | https://intelligence.org/files/TilingAgentsDraft.pdf | 24 |
| T-11 | Bostrom (2014), *Superintelligence*, OUP, ch. 4 "The kinetics of an intelligence explosion" | PR（学术专著）。**原书未打开**，下面的内容来自 Grace 2014 的读书会摘要和 Benthall 2017 的转述 | https://www.lesswrong.com/posts/GT8uvxBjidrmM3MCv/superintelligence-6-intelligence-explosion-kinetics | 25 |
| T-12 | Yampolskiy (2015), "From Seed AI to Technological Singularity via Recursively Self-Improving Software", arXiv 1502.06512 | Preprint。有二手来源称它在 AGI-15 以 PR 形式发表，本次未核实 | https://arxiv.org/abs/1502.06512 | 26 |
| T-13 | Benthall (2017), "Don't Fear the Reaper: Refuting Bostrom's Superintelligence Argument", arXiv 1702.08495 | Preprint | https://arxiv.org/abs/1702.08495 | 27 |
| T-14 | Aghion, B. Jones & C. Jones (2017/2019), "Artificial Intelligence and Economic Growth", NBER WP 23928；收入 *The Economics of AI: An Agenda* (2019) | PR（书章）。只读了摘要页 | https://www.nber.org/papers/w23928 | 28 |
| T-15 | Davidson (2023), compute-centric takeoff 报告（Open Philanthropy）；作者本人在 LessWrong 写的摘要 | Preprint（机构报告）。报告原页 403，用的是作者摘要 | https://www.lesswrong.com/posts/Gc9FGtdXhK9sCSEYu/what-a-compute-centric-framework-says-about-ai-takeoff<br>https://takeoffspeeds.com/ | 29 |
| T-16 | Besiroglu, Erdil & Ho (2024), "Do the Returns to Software R&D Point Towards a Singularity?", Epoch AI | Essay/Blog（研究机构），基于 Erdil, Besiroglu & Ho, "Estimating Idea Production: A Methodological Survey", arXiv 2405.10494 | https://epoch.ai/blog/do-the-returns-to-software-rnd-point-towards-a-singularity | 30 |
| T-17 | Eth & Davidson (2025), "Will AI R&D Automation Cause a Software Intelligence Explosion?", Forethought | Preprint/Essay（研究机构） | https://www.forethought.org/research/will-ai-r-and-d-automation-cause-a-software-intelligence-explosion | 31 |
| T-18 | Davidson & Houlden (2025), "How quick and big would a software intelligence explosion be?", Forethought | Preprint/Essay（研究机构） | https://www.forethought.org/research/how-quick-and-big-would-a-software-intelligence-explosion-be | 32 |
| T-19 | Trammell (2026), "Parallelization constraints could delay a technological singularity", Epoch AI report | Preprint/Report | https://epoch.ai/publications/parallelization-constraints-could-delay-a-technological-singularity | 33 |
| T-20 | Liu & van der Schaar (2025), "Position: Truly Self-Improving Agents Require Intrinsic Metacognitive Learning", ICML 2025 (PMLR 267) | PR（会议 position paper）。只读了摘要页 | https://arxiv.org/abs/2506.05109 | 34 |
| T-21 | Gao et al. (2025/2026), "A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence", arXiv 2507.21046 v4 | Preprint（survey） | https://arxiv.org/abs/2507.21046 | 35 |
| T-22 | Fang et al. (2025), "A Comprehensive Survey of Self-Evolving AI Agents", arXiv 2508.07407 | Preprint（survey）。只读了摘要 | https://arxiv.org/abs/2508.07407 | 36 |
| T-23 | ICLR 2026 Workshop on AI with Recursive Self-Improvement（Rio de Janeiro，2026-04-26；组织者包括 Zhuge、Zeng、Schmidhuber 等） | 工作坊（会议存在已核实；是否 archival 页面上没写） | https://recursive-workshop.github.io/<br>https://iclr.cc/virtual/2026/workshop/10000796 | 37 |
| T-24 | Tang, Ma, Li & Yuan (2026), "Generalized Agent Iteration: One Formal Framework for Iterative Policy Improvement and Recursive Self-Improvement", arXiv 2609.13406 v2 | Preprint | https://arxiv.org/abs/2609.13406 | 38 |
| T-25 | Moser, Frolov, Nauen, Raue & Dengel (2026), "The Self-Improvement Problem", arXiv 2610.21646（DFKI） | Preprint。读的是摘要页（alphaxiv） | https://alphaxiv.org/abs/2610.21646v1 | 39 |
| T-26 | Zhang, Yuan & Zhang (2026), "Self-Reference in Large Language Models: The Introspection Threshold for Recursive Self-Improvement", arXiv 2607.04277 | Preprint。只读了摘要 | https://arxiv.org/abs/2607.04277 | 40 |

## M. 自我改进方法与实证（来源：`research/lit_rsi/02_methods.md`）

| 键 | 文献 | 类型 / 发表情况 | URL | 原文件行号 |
|---|---|---|---|---|
| M-01 | **STOP**：Zelikman, Lorch, Mackey, Kalai 2023 | COLM 2024 | https://arxiv.org/abs/2310.02304 | 15 |
| M-02 | **Gödel Agent**：Yin, Wang, Pan, Lin, Wan, Wang 2024 | ACL 2025 | https://arxiv.org/abs/2410.04444 | 16 |
| M-03 | **Darwin Gödel Machine (DGM)**：Zhang, Hu, Lu, Lange, Clune 2025 | ICLR 2026 | https://arxiv.org/abs/2505.22954 | 17 |
| M-04 | **Huxley-Gödel Machine (HGM)**：Wang, Piękos, Nanbo, …, Schmidhuber 2025 | arXiv 预印本 | https://arxiv.org/abs/2510.21614 | 18 |
| M-05 | **SICA**：Robeyns, Szummer, Aitchison 2025 | arXiv 预印本 | https://arxiv.org/abs/2504.15228 | 19 |
| M-06 | **ADAS / Meta Agent Search**：Hu, Lu, Clune 2024 | ICLR 2025 | https://arxiv.org/abs/2408.08435 | 20 |
| M-07 | **Promptbreeder**：Fernando, Banarse, Michalewski, Osindero, Rocktäschel 2023 | arXiv 预印本 | https://arxiv.org/abs/2309.16797 | 21 |
| M-08 | **AlphaEvolve**：Novikov, Vũ, Eisenberger, et al. 2025 | 白皮书 / arXiv | https://arxiv.org/abs/2506.13131 | 22 |
| M-09 | **AI Scientist v1**：Lu, Lu, Lange, Foerster, Clune, Ha 2024 | arXiv 预印本 | https://arxiv.org/abs/2408.06292 | 23 |
| M-10 | **AI Scientist v2**：Yamada, Lange, Lu, et al. 2025 | arXiv 预印本 | https://arxiv.org/abs/2504.08066 | 24 |
| M-11 | **Kirsch & Schmidhuber**："Eliminating Meta Optimization Through Self-Referential Meta Learning" 2022 | ICML 2022 DARL Workshop | https://arxiv.org/abs/2212.14392 | 25 |
| M-12 | **Irie et al.**："A Modern Self-Referential Weight Matrix That Learns to Modify Itself" | ICML 2022 | https://arxiv.org/abs/2202.05780 | 26 |
| M-13 | **SEAL**：Zweiger, Pari, Guo, Akyürek, Kim, Agrawal 2025 | NeurIPS 2025 | https://arxiv.org/abs/2506.10943 | 27 |
| M-14 | **STaR**：Zelikman, Wu, Mu, Goodman 2022 | arXiv（会议见第 5 节） | https://arxiv.org/abs/2203.14465 | 28 |
| M-15 | **LMSI**：Huang, Gu, Hou, Wu, Wang, Yu, Han 2022 | arXiv 预印本（摘要页无会议信息） | https://arxiv.org/abs/2210.11610 | 29 |
| M-16 | **ReST**：Gulcehre et al. 2023 | arXiv 预印本 | https://arxiv.org/abs/2308.08998 | 30 |
| M-17 | **ReST-EM**：Singh, Co-Reyes, Agarwal, et al. 2023 | TMLR 04/2024 | https://arxiv.org/abs/2312.06585 | 31 |
| M-18 | **Self-Rewarding LMs**：Yuan, Pang, Cho, Li, Sukhbaatar, Xu, Weston 2024 | ICML 2024 | https://arxiv.org/abs/2401.10020 | 32 |
| M-19 | **Meta-Rewarding**：Wu, Yuan, Golovneva, Xu, Tian, Jiao, Weston, Sukhbaatar 2024 | arXiv 预印本 | https://arxiv.org/abs/2407.19594 | 33 |
| M-20 | **SPIN**：Chen, Deng, Yuan, Ji, Gu 2024 | ICML 2024 | https://arxiv.org/abs/2401.01335 | 34 |
| M-21 | **Absolute Zero (AZR)**：Zhao, Wu, Yue, et al. 2025 | arXiv 预印本 | https://arxiv.org/abs/2505.03335 | 35 |
| M-22 | **B-STaR**：Zeng, Huang, Zhao, Wang, Shan, He 2024 | ICLR 2025 | https://arxiv.org/abs/2412.17256 | 36 |
| M-23 | **Reflexion**：Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao 2023 | arXiv（会议见第 5 节） | https://arxiv.org/abs/2303.11366 | 37 |
| M-24 | **Voyager**：Wang, Xie, Jiang, et al. 2023 | arXiv 预印本 | https://arxiv.org/abs/2305.16291 | 38 |
| M-25 | **ExpeL**：Zhao, Huang, Xu, Lin, Liu, Huang 2023 | AAAI 2024 | https://arxiv.org/abs/2308.10144 | 39 |
| M-26 | **Memory management / experience-following**：Xiong, Lin, Xie, He, Liu, Tang, Lakkaraju, Xiang 2025 | arXiv 预印本 | https://arxiv.org/abs/2505.16067 | 40 |
| M-27 | **ACE**：Zhang, Hu, Upasani, et al. 2025 | ICLR 2026 | https://arxiv.org/abs/2510.04618 | 41 |
| M-28 | **Self-generated in-context examples**：Sarukkai, Xie, Fatahalian 2025 | arXiv 预印本 | https://arxiv.org/abs/2505.00234 | 42 |

## E. 测量、评测与实验室阈值（来源：`research/lit_rsi/03_measurement.md`）

| 键 | 文献 | 类型 / 发表情况 | URL | 原文件行号 |
|---|---|---|---|---|
| E-01 | Wijk et al., **RE-Bench**（METR，2024-11；v2 2025-05） | preprint | https://arxiv.org/abs/2411.15114 | 16 |
| E-02 | Kwa et al., **Measuring AI Ability to Complete Long Tasks**（METR，2025-03；v4 2026-07） | preprint（arXiv 页注明 NeurIPS 2025） | https://arxiv.org/abs/2503.14499 | 17 |
| E-03 | Chan et al., **MLE-bench**（OpenAI，2024-10） | preprint | https://arxiv.org/abs/2410.07095 | 18 |
| E-04 | Starace et al., **PaperBench**（OpenAI，2025-04） | preprint | https://arxiv.org/abs/2504.01848 | 19 |
| E-05 | Huang et al., **MLAgentBench**（Stanford，2023-10；v2 2024-04） | preprint | https://arxiv.org/abs/2310.03302 | 20 |
| E-06 | Hua et al., **ResearchCodeBench**（2025-06） | preprint | https://arxiv.org/abs/2506.02314 | 21 |
| E-07 | Zhao et al., **The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements**（Meta FAIR，2025-06）。标题已核实 | preprint | https://arxiv.org/abs/2506.22419 | 22 |
| E-08 | Rank et al., **PostTrainBench**（2026-03） | preprint | https://arxiv.org/abs/2603.08640 | 23 |
| E-09 | Chi et al., **AI4AI-Bench**（2026-08） | preprint | https://arxiv.org/abs/2608.20318 | 24 |
| E-10 | Meng et al., **RSIBench-Data**（2026-07） | preprint | https://arxiv.org/abs/2607.25886 | 25 |
| E-11 | Kirgis et al., **Can AI agents conduct open-ended AI research?**（2026-07） | preprint | https://arxiv.org/abs/2607.27191 | 26 |
| E-12 | Becker et al., **Measuring the Impact of Early-2025 AI on Experienced OSS Developer Productivity**（METR，2025-07） | preprint | https://arxiv.org/abs/2507.09089 | 27 |
| E-13 | Rein, **Research Update: Algorithmic vs. Holistic Evaluation**（METR，2025-08） | blog | https://metr.org/blog/2025-08-12-research-update-towards-reconciling-slowdown-with-time-horizons | 28 |
| E-14 | Whitfill et al., **Many SWE-bench-Passing PRs Would Not Be Merged into Main**（METR，2026-03） | note | https://metr.org/notes/2026-03-10-many-swe-bench-passing-prs-would-not-be-merged-into-main/ | 29 |
| E-15 | Ho et al., **Algorithmic progress in language models**（Epoch，2024-03） | preprint | https://arxiv.org/abs/2403.05812 | 30 |
| E-16 | Eth & Davidson, **Will AI R&D Automation Cause a Software Intelligence Explosion?**（Forethought，2025-03） | 研究报告（非同行评审） | https://www.forethought.org/research/will-ai-r-and-d-automation-cause-a-software-intelligence-explosion | 31 |
| E-17 | Chan, Padarath, Kwon, Greaves, Anderljung, **Measuring AI R&D Automation**（GovAI/Oxford，2026-03） | preprint | https://arxiv.org/abs/2603.03992 | 32 |
| E-18 | Anthropic, **Risk Report: February 2026**（§3.4 证据表） | lab document | https://www-cdn.anthropic.com/08eca2757081e850ed2ad490e5253e940240ca4f.pdf | 33 |
| E-19 | Anthropic, **Risk Report: August 2026**（§3.4–3.5） | lab document | https://assets.claude.com/f61d49fa5596956a5dec75fea0e973bf6a6a8378.pdf | 34 |
| E-20 | Favaro & Wright, **Measurements for understanding the pace of AI development inside frontier labs**（Anthropic，2026） | lab document | https://www.anthropic.com/institute/measuring-pace-of-ai-development | 35 |
| E-21 | Jurkovic, Barnes, Wijk, **Review of "Risks from automated R&D" in Anthropic Risk Report Feb 2026**（METR，2026-05） | 外部评审（lab-safety 文件） | https://metr.org/blog/2026-05-08-rd-section-anthropic-risk-report-feb-2026-review/<br>https://metr.org/assets/Updated%20Review%20of%20%22Risks%20from%20automated%20R&D%22%20section%20in%20the%20Anthropic%20Risk%20Report%20-%20February%202026.pdf | 36 |
| E-22 | METR, **Review of R&D section, Anthropic Risk Report Aug 2026**（2026-11-05） | 外部评审 | https://metr.org/blog/2026-11-05-rd-section-anthropic-risk-report-august-2026-review | 37 |
| E-23 | Cunningham, Rush, Whitfill, Wijk, **Independent Assessment of AI R&D Acceleration at Anthropic in October 2026**（METR，2026-11-05） | 外部评估报告 | https://metr.org/blog/2026-11-05-anthropic-ai-rd-acceleration-assessment | 38 |
| E-24 | OpenAI, **GPT-5.6 System Card**（2026-07-09）§9.1.3 | lab document | https://deploymentsafety.openai.com/gpt-5-6 | 39 |
| E-25 | Zelikman et al., **STOP: Self-Taught Optimizer**（2023-10；v3 2024-08） | peer-reviewed（PDF 注明 COLM 2024） | https://arxiv.org/abs/2310.02304 | 40 |
| E-26 | Zhang et al., **Darwin Gödel Machine**（2025-05；v3 2026-03） | peer-reviewed（PDF 页眉注明 ICLR 2026） | https://arxiv.org/abs/2505.22954 | 41 |
| E-27 | Wang et al. (incl. Schmidhuber), **Huxley-Gödel Machine**（2025-10） | preprint | https://arxiv.org/abs/2510.21614 | 42 |
| E-28 | Srikanth, Zhao, Xu, Wu, Jiang, **Recursive self-improvement of AI research agents**（AIDE²，Weco AI，2026-09） | preprint | https://arxiv.org/abs/2609.26457 | 43 |
| E-29 | Burtsev, **Recursive Criticality of AI Self-Improvement**（2026-08） | preprint（理论） | https://arxiv.org/abs/2609.00137 | 44 |
| E-30 | Yang et al., **Frontis-MA1**（2026-07） | preprint | https://arxiv.org/abs/2607.28568 | 45 |
| E-31 | Lu et al., **The AI Scientist**（Sakana，2024-08） | preprint | https://arxiv.org/abs/2408.06292 | 46 |
| E-32 | Luo, Kasirzadeh, Shah, **The More You Automate, the Less You See**（CMU，2025-09） | preprint（arXiv 页注明 NeurIPS 2025 AI4Science Spotlight，属 workshop） | https://arxiv.org/abs/2509.08713 | 47 |
| E-33 | Von Arx, Chan, Barnes, **Recent Frontier Models Are Reward Hacking**（METR，2025-06） | blog | https://metr.org/blog/2025-06-05-recent-reward-hacking/ | 48 |
| E-34 | Madaan et al., **Quantifying Variance in Evaluation Benchmarks**（Meta，2024-06） | preprint | https://arxiv.org/abs/2406.10229 | 49 |
| E-35 | Epoch AI, **ECI documentation – Methodology**（2026） | 方法文档 | https://epoch.ai/data/eci-documentation/methodology | 50 |

## L. 信号硬度与评估器（来源：`research/lit_signal_hardness.md`）

| 键 | 文献 | 类型 / 发表情况 | URL | 原文件行号 |
|---|---|---|---|---|
| L-01 | Romera-Paredes, Barekatain, Novikov et al., 2023（刊期 2024）, *Mathematical discoveries from program search with large language models*, Nature 625:468–475 | peer-reviewed | https://pmc.ncbi.nlm.nih.gov/articles/PMC10794145/<br>https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/ | 27 |
| L-02 | Novikov, Vũ, Eisenberger et al., 2025, *AlphaEvolve: A coding agent for scientific and algorithmic discovery*, arXiv:2506.13131（DeepMind 白皮书） | tech report / preprint | https://arxiv.org/abs/2506.13131<br>https://arxiv.org/html/2506.13131v1<br>https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/ | 28 |
| L-03 | Mankowitz, Michi, Zhernov et al., 2023, *Faster sorting algorithms discovered using deep reinforcement learning*, Nature（AlphaDev） | peer-reviewed | https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10247365/fullTextXML | 29 |
| L-03b | Fawzi, Balog, Huang et al., 2022, *Discovering faster matrix multiplication algorithms with reinforcement learning*, Nature 610:47–53（AlphaTensor） | peer-reviewed | （原文件未给 URL） | 30 |
| L-04 | Jason Wei, 2025（页面只显示 "Jul 15"；年份根据二手来源推定）, *Asymmetry of verification and verifier's law*, 个人 blog | blog | https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law | 31 |
| L-05 | Manheim & Garrabrant, 2018（v4 2019）, *Categorizing Variants of Goodhart's Law*, arXiv:1803.04585 | preprint | https://arxiv.org/abs/1803.04585 | 32 |
| L-06 | Wijk, Lin, Becker et al.（METR）, 2024（v2 2025）, *RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts*, arXiv:2411.15114 | preprint | https://arxiv.org/abs/2411.15114<br>https://arxiv.org/html/2411.15114v2 | 33 |
| L-07 | Yue, Chen, Lu, Zhao et al., 2025, *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*, arXiv:2504.13837（NeurIPS 2025 Oral） | peer-reviewed（会议） | https://arxiv.org/abs/2504.13837 | 34 |
| L-07b | Lambert, Morrison, Pyatkin et al., 2024, *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*, arXiv:2411.15124 | preprint | https://arxiv.org/abs/2411.15124 | 35 |
| L-08 | Kwa, West, Becker et al.（METR）, 2025, *Measuring AI Ability to Complete Long Tasks*（v1 标题；2026-07 的 v4 改为 *…Long Software Tasks*）, arXiv:2503.14499 | preprint | https://arxiv.org/abs/2503.14499<br>https://arxiv.org/html/2503.14499v1 | 36 |
| L-09 | David Rein（METR）, 2025-08-13, *Research Update: Algorithmic vs. Holistic Evaluation* | blog / research update | https://metr.org/blog/2025-08-12-research-update-towards-reconciling-slowdown-with-time-horizons/ | 37 |
| L-10 | Kwa & Cheng（METR）, 2025-07-14, *How Does Time Horizon Vary Across Domains?* | blog / research update | https://metr.org/blog/2025-07-14-how-does-time-horizon-vary-across-domains/ | 38 |
| L-11 | Ouyang, Guo, Arora, Zhang, Hu, Ré, Mirhoseini, 2025, *KernelBench: Can LLMs Write Efficient GPU Kernels?*, arXiv:2502.10517 | preprint | https://arxiv.org/abs/2502.10517 | 39 |
| L-12 | Chan, Chowdhury, Jaffe et al.（OpenAI）, 2024（v6 2025）, *MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering*, arXiv:2410.07095（ICLR 版） | peer-reviewed（会议） | https://arxiv.org/abs/2410.07095 | 40 |
| L-13 | Zhao, Magka, Jiang et al.（Meta）, 2025, *The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements*, arXiv:2506.22419 | preprint | https://arxiv.org/abs/2506.22419 | 41 |
| L-14 | Starace, Jaffe, Sherburn et al.（OpenAI）, 2025, *PaperBench: Evaluating AI's Ability to Replicate AI Research*, arXiv:2504.01848 | preprint | https://arxiv.org/abs/2504.01848 | 42 |
| L-15 | Yamada, Lange, Lu et al.（Sakana）, 2025, *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search*, arXiv:2504.08066 | preprint | https://arxiv.org/abs/2504.08066 | 43 |
| L-16 | Beel, Kan, Baumgart, 2025（v3 2025-10）, *Evaluating Sakana's AI Scientist: Bold Claims, Mixed Results, and a Promising Future?*, arXiv:2502.14297（SIGIR Forum 2025） | peer-reviewed（Forum） / preprint | https://arxiv.org/abs/2502.14297 | 44 |
| L-17 | MacDiarmid, Wright, Uesato et al.（Anthropic）, 2025, *Natural Emergent Misalignment from Reward Hacking in Production RL*, arXiv:2511.18397 | preprint / tech report | https://arxiv.org/abs/2511.18397 | 45 |
| L-17b | Baker, Huizinga, Gao et al.（OpenAI）, 2025, *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation*, arXiv:2503.11926 | preprint | https://arxiv.org/abs/2503.11926 | 46 |
| L-18 | Von Arx, Chan, Barnes（METR）, 2025-06-05, *Recent Frontier Models Are Reward Hacking* | blog / research report | https://metr.org/blog/2025-06-05-recent-reward-hacking/ | 47 |
| L-19 | Kyle Wiggers, TechCrunch, 2025-02-21, *Sakana walks back claims that its AI can dramatically speed up model training* | 新闻（二手，转述 Sakana 声明） | https://techcrunch.com/?p=2969225 | 48 |
| L-20 | Lange, Sun, Prasad, Faldor, Tang, Ha（Sakana）, 2025, *Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization*, arXiv:2509.14279 | preprint | https://arxiv.org/abs/2509.14279 | 49 |
| L-21 | Lin, Modi, Hari et al.（NVIDIA 等）, 2026-03, *SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits*, arXiv:2603.19173（NeurIPS 2026 poster，仅见于搜索结果） | preprint | https://arxiv.org/abs/2603.19173<br>https://arxiv.org/html/2603.19173 | 50 |
| L-22 | Zhang, Yu, Wang, Fan, Reed, Mirhoseini, Su, 2026-06, *KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?*, arXiv:2607.16241 | preprint | https://arxiv.org/abs/2607.16241 | 51 |
| L-23 | Zhong, Segal, Bercovich, Saxena, Zhang, Raghunathan, 2026-06, *Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops*, arXiv:2606.08960 | preprint | https://arxiv.org/abs/2606.08960 | 52 |
| L-24 | Atinafu & Cohen, 2026-03, *RewardHackingAgents: Benchmarking Evaluation Integrity for LLM ML-Engineering Agents*, arXiv:2603.11337 | preprint | https://arxiv.org/abs/2603.11337 | 53 |
| L-25 | Nathani, Madaan, Roberts et al.（Meta）, 2025, *MLGym: A New Framework and Benchmark for Advancing AI Research Agents*, arXiv:2502.14499 | preprint | https://arxiv.org/abs/2502.14499 | 54 |
| L-26 | Zhang, Hu, Lu, Lange, Clune, 2025, *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents*, arXiv:2505.22954；以及 Sakana blog 2025-05-30 | preprint + blog | https://arxiv.org/abs/2505.22954<br>https://sakana.ai/dgm/ | 55 |
| L-27 | Wafer Team, 2026-01-27, *Case Study: A 104x (?) Speedup on KernelBench* | 公司 blog | https://www.wafer.ai/blog/reward-hacking | 56 |
| L-28 | Karpathy, 2026（README 引言标注 2026-03）, *autoresearch*, GitHub | 开源代码 / README | https://github.com/karpathy/autoresearch | 57 |
| L-29 | Recursive Superintelligence, 2026, *First steps toward automated AI research*（被核对的对象本身） | 公司 blog | https://recursive.com/updates/first-steps-toward-automated-ai-research | 58 |

## M2. 方法：局限与反面证据（正文中的引用）（来源：`research/lit_rsi/02_methods.md §3–4`）

| 键 | 文献 | 类型 / 发表情况 | URL | 原文件行号 |
|---|---|---|---|---|
| M2-01 | Mind the Gap（Song, Zhang, Eisenach, Kakade, Foster, Ghai 2024，arXiv 2412.02674） | 见原文件 | https://arxiv.org/abs/2412.02674 | 79 |
| M2-02 | Sharpening（Huang, Block, Foster, Rohatgi, Zhang, Simchowitz, Ash, Krishnamurthy 2024，arXiv 2412.01951） | 见原文件 | https://arxiv.org/abs/2412.01951 | 89 |
| M2-03 | Yue et al. 2025 | 见原文件 | https://arxiv.org/abs/2504.13837 | 90 |
| M2-04 | Shumailov et al.（Nature 631:755–759, 2024；本次实际阅读的是 arXiv v3 "The Curse of Recursion"） | 见原文件 | https://arxiv.org/abs/2305.17493 | 93 |
| M2-05 | Gerstgrasser et al. 2024 | 见原文件 | https://arxiv.org/abs/2404.01413 | 94 |
| M2-06 | Spontaneous Reward Hacking in Iterative Self-Refinement（Pan, He, Bowman, Feng 2024，arXiv 2407.04549） | 见原文件 | https://arxiv.org/abs/2407.04549 | 101 |
| M2-07 | Spurious Rewards（Shao et al. 2025，arXiv 2506.10947） | 见原文件 | https://arxiv.org/abs/2506.10947 | 104 |
| M2-08 | Reflexion (Shinn et al. 2023) | 见原文件 | （原文件未给 URL） | 113 |
| M2-09 | ExpeL (Zhao et al., AAAI 2024) | 见原文件 | （原文件未给 URL） | 114 |
| M2-10 | Huang et al., "LLMs Cannot Self-Correct Reasoning Yet"（ICLR 2024，arXiv 2310.01798） | 见原文件 | （原文件未给 URL） | 115 |
| M2-11 | Xiong et al. 2025（记忆管理） | 见原文件 | （原文件未给 URL） | 116 |
| M2-12 | ACE (Zhang et al., ICLR 2026) | 见原文件 | （原文件未给 URL） | 117 |
| M2-13 | Pan et al. 2024 (ICRH) | 见原文件 | （原文件未给 URL） | 118 |
| M2-14 | SICA（Robeyns et al. 2025） | 见原文件 | （原文件未给 URL） | 119 |

## 四份材料（证据台账）

| 键前缀 | 材料 | 台账文件 | 已读的原始 URL |
|---|---|---|---|
| RF-xx | Recursive · First Steps Toward Automated AI Research | `research/sources/recursive_first_steps.md` | https://recursive.com/updates/first-steps-toward-automated-ai-research<br>https://github.com/recursive-org/first-steps-toward-automated-ai-research |
| FL-xx | Recursive · Introducing Flint | `research/sources/recursive_flint.md` | https://recursive.com/updates/introducing-recursive-flint<br>https://huggingface.co/RecursiveSI/flint<br>https://github.com/recursive-oss/rec<br>https://docs.recursive.com/getting-started |
| NV-xx | NaiveAI · Naive-N0.5-Flash: Building Frontier AI with AI | `research/sources/naive_research.md` | https://naive.ai/en/research/<br>https://naive.ai/en/ |
| AX-xx | Apex Intelligence · Introducing Apex-35B-V1.0 | `research/sources/apex_35b.md` | https://www.apexin.com/apex-35b/#ai<br>https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0<br>https://github.com/apexin-ai/Apex-35B-V1.0 |

## 其他出处（research/ 下其他文件中出现、上面各表未覆盖的 URL）

> 逐条列出所在文件和行号，便于回查上下文。内容未经二次核实，引用前先看原文件。

| 键 | URL | 首次出现 |
|---|---|---|
| X-01 | https://huggingface.co/<id | `research/artifacts_weights_configs.md`:8 |
| X-02 | https://huggingface.co/api/models/<id | `research/artifacts_weights_configs.md`:8 |
| X-03 | https://github.com/<repo | `research/artifacts_weights_configs.md`:8 |
| X-04 | https://github.com/NaiveAI-Labs/Naive-N0.5-Flash | `research/artifacts_weights_configs.md`:99 |
| X-05 | https://html.duckduckgo.com/html/ | `research/platform_build_plan.md`:160 |
| X-06 | https://huggingface.co | `research/platform_build_plan.md`:486 |
| X-07 | https://pypi.org/simple/ | `research/platform_build_plan.md`:486 |
| X-08 | https://github.com; | `research/platform_build_plan.md`:486 |
| X-09 | https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main | `research/platform_build_plan.md`:567 |
| X-10 | https://download.pytorch.org/whl/cu128 | `research/platform_build_plan.md`:574 |
| X-11 | https://pypi.org/pypi/mlx-lm/json | `research/platform_build_plan.md`:707 |
| X-12 | https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md | `research/platform_build_plan.md`:709 |
| X-13 | https://opencode.ai/docs/cli/ | `research/platform_build_plan.md`:711 |
| X-14 | https://unpkg.com/@mariozechner/pi-coding-agent@0.73.1/README.md | `research/platform_build_plan.md`:712 |
| X-15 | https://docs.openhands.dev/openhands/usage/cli/headless | `research/platform_build_plan.md`:713 |
| X-16 | https://docs.litellm.ai/docs/projects/mini-swe-agent/ | `research/platform_build_plan.md`:714 |
| X-17 | https://docs.litellm.ai/docs/tutorials/claude_code_websearch | `research/platform_build_plan.md`:715 |
| X-18 | https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html | `research/platform_build_plan.md`:716 |
| X-19 | https://huggingface.co/RecursiveSI/flint-int4-gguf | `research/platform_build_plan.md`:719 |
| X-20 | https://hf-mirror.com | `research/platform_build_plan.md`:774 |
| X-21 | https://github.com | `research/platform_build_plan.md`:774 |
| X-22 | https://registry.npmjs.org; | `research/platform_build_plan.md`:774 |
| X-23 | https://huggingface.co}/datasets/karpathy/climbmix-400b-shuffle/resolve/main | `research/platform_build_plan.md`:827 |
| X-24 | https://weco.ai/blog/first-evidence-of-recursive-self-improvement | `research/platform_design.md`:69 |
| X-25 | https://arxiv.org/html/2505.22954 | `research/platform_design.md`:109 |
| X-26 | https://arxiv.org/abs/2207.14502 | `research/platform_design.md`:112 |
| X-27 | https://arxiv.org/abs/1705.08439 | `research/platform_design.md`:113 |
| X-28 | https://arxiv.org/abs/2302.07867 | `research/platform_design.md`:114 |
| X-29 | https://github.com/trevin-creator/autoresearch-mlx | `research/platform_design.md`:313 |
| X-30 | https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md | `research/platform_design.md`:397 |
| X-31 | https://rsi-bench.duckdns.org/ | `research/platform_design.md`:401 |
| X-32 | https://github.com/recursive-org/first-steps-toward-automated-ai-research（Apache-2.0 | `research/reading/01_Recursive_FirstSteps_导读.md`:11 |
| X-33 | https://arxiv.org/abs/2405.10494 | `research/lit_rsi/01_theory.md`:30（原文只给了 arXiv 编号，URL 由编号拼出） |
| X-34 | https://arxiv.org/abs/1312.6764 | `research/lit_rsi/01_theory.md`:167（原文只给了 arXiv 编号，URL 由编号拼出） |
| X-35 | https://arxiv.org/abs/2607.13104 | `research/lit_rsi/01_theory.md`:169（原文只给了 arXiv 编号，URL 由编号拼出） |
| X-36 | https://arxiv.org/abs/2310.01798 | `research/lit_rsi/02_methods.md`:115（原文只给了 arXiv 编号，URL 由编号拼出） |
