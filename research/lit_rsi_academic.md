# 学术界怎么看 RSI：文献综述（汇总）

> 2026-11-18 03:00 汇总。三个搜索 agent 共检索约 96 个来源，每条都打开过原文或摘要页，详情和出处见：
> - `research/lit_rsi/01_theory.md`：定义、理论、回报率模型（26 个来源）
> - `research/lit_rsi/02_methods.md`：ML 自我改进方法与实证（35 篇）
> - `research/lit_rsi/03_measurement.md`：测量、评测、实验室阈值（35 个来源）
>
> 抽查：我在 arXiv 上核对了 9 个编号（Tang 2026、Zhang 2026、Liu & vdS 2025、Gao 2025、DGM、HGM、ACE、Xiong 2025、AIDE²），标题都一致；METR 评审页面也能打开（HTTP 200）。
> 标注约定：〔推断〕是综合判断，不是任何单一来源的原话；凡是只读了摘要的来源，子文件里都单独标出。**结论留给用户判断。**

## 1. 一句话总结
学术界**没有统一的 RSI 定义**，但大多数理论工作要求两件事：一是"改进机制在系统内部，并且本身可以被改写"，二是"改进会复利（回报 ≥ 1）"。按这个标准，self-distillation 和 self-play 通常只算"弱自我改进"。实证上，**目前没有任何工作展示复利**：自我训练几轮内就饱和；改写自身脚手架的系统（DGM 等）全部冻结了底层模型；也**没有人用"换后继模型当改进者 + 对照"的方式测过二阶效应**。最接近的一次（AIDE² 的 ignition test）只换了脚手架，结果是打平。

## 2. 定义：RSI 和自我训练怎么区分（详见 01 §2）
| 判据 | 支持把它当作必要条件 | 反对或弱化 |
|---|---|---|
| **A. 改进机制在系统内部，并且可以被改写** | Schmidhuber 的 Gödel machine（可以改写包括证明搜索器在内的任何代码）；Yudkowsky（LOGI、2008）；Yampolskiy 2015；Tang et al. 2026；Liu & van der Schaar（ICML 2025 position） | Yudkowsky 2013 的宽松定义（能产出后继者就算）；Mahoney（判据是有没有外部输入） |
| **B. 被提升的是"产生下一代的能力"（反身性）** | Yampolskiy："gets better at getting better"；Chalmers 的自我放大能力 G | 没有明确反对；Chalmers 认为 G 和一般能力 H 可以分离 |
| **C. 跨代回报 ≥ 1（复利）** | Yudkowsky 2013 的 k > 1；Chalmers 的 proportionality；Bostrom 的 optimization power / recalcitrance；Epoch 和 Forethought 的 r > 1 | **Tang et al. 2026 明确不把复利放进定义**，只当经验问题；Benthall、Mahoney 认为一般条件下难以满足 |
| **D. 需要通用性** | Good 1965（在一切智力活动上超越人类）；Hall（universality 阈值）；Eth & Davidson（ASARA，研发领域内的通用） | Chalmers、Yudkowsky 2013（低于人类水平也可能启动）；Mahoney（只针对固定目标） |

要点：
- **Yampolskiy 2015 的三级分类**：self-modification（没有改进）、weak self-improvement（例如固定 fitness 下优化参数）、strong / recursive（"gets better at getting better"，能反复把算法换成全新的方法）。metalearning 被他归为"非递归优化"。
- **Tang et al. 2026（preprint）是唯一一篇把 GPI（一般的迭代策略改进）和 RSI 形式化区分开的论文**。区分只看一个开关：改进机制在 agent 外部就是 GPI，在内部就是 RSI。它还按评价标准是否锚定在系统外，把 RSI 分为 Anchored、Goal Drift 和 Fully Self-Referential 三类。
- **Mahoney** 是唯一直接点名 self-play 的：他把 self-play 算作自我改进，同时证明没有外部输入时增长受限于 O(log n)。
- **没有任何来源直接对 self-distillation 下判断**。〔推断〕按 A+B+C 来看，固定算法、单一指标、单代或少数几代增益的 self-distillation 和 self-play，最多只算"weak"。
- **用户提出的"必须是通用模型"一条**，在文献里正好是有争议的判据 D：Good 和 Hall 支持，Chalmers 和 Yudkowsky 2013 认为不必要。Eth & Davidson 的折中是"在 AI 研发这个领域内通用"。

## 3. 回报率模型：什么时候会"越改越快"（详见 01 §3）
- Epoch 和 Forethought 的核心参数 **r**：在 log(软件效率) 对 log(累积研究投入) 的关系里，r > 1 时，如果研究投入由 AI 提供并随能力增长，每翻倍一次所需的时间会越来越短，即加速。
- **估计值**：国际象棋约 0.83（SE 0.15）；图像识别中位数 1.4（0.8–2.4）；RL 样本效率 1.6；线性规划 1.1；SAT 求解器 3.5；**大多数在统计上不显著**。Forethought 的总体判断是约 1–4，**算力不变时约 0.5–2**。LLM 没有数值。
- Davidson & Houlden 2025：在 ASARA 条件下，把 ≥3 年的进展压进 ≤1 年的概率约 60%；把 ≥10 年压进 ≤1 年的概率约 20%。
- **和我们的 Δ₂ 的关系**〔推断〕：如果每代投入固定、用 log 尺度度量，Δ₂ > 0 大致对应 r ≥ 1，但两者不完全等价。r 是按累积投入定义的，Δ₂ 是代际差分；r 来自人类研究者的历史数据；Chalmers 也指出换度量尺度会改变结论。**Benthall 和 Davidson & Houlden 都预测：逼近上限时 Δ₂ 会从正变负**，所以只看几代 Δ₂ > 0，证明不了会持续加速。

## 4. ML 实证：三类系统各自做到了哪一步（详见 02）
**A 类：AI 辅助改进别的东西**：ADAS（meta agent 固定）、AlphaEvolve（改进 Gemini 的训练内核，总训练时间 −1%；作者说回馈到下一版的周期以月计，而且由人决定是否部署）、AI Scientist。

**B 类：人设计的固定循环下的自我训练**，即改权重或数据，但"怎么改进"不变：
- **几轮内饱和**：Mind the Gap（Song et al. 2024）发现生成与验证的差距在 2–3 轮内降到接近 0，大 k 时 pass@k 反而下降；而且第一轮的部分"提升"其实来自**答案格式遵循**。SPIN 依次为 58.14 → 60.80 → 62.12 → 62.97 → 63.16，增益递减。ReST-EM 在 APPS 上第一轮后就回退。
- **理论上限**：Sharpening（Huang et al. 2024）认为自我改进不能创造模型里本来没有的信息，只能把概率质量锐化到好的输出上。Yue et al. 2025（NeurIPS 2025 Oral）发现 k 很大时，基础模型的 pass@k 高于 RLVR 模型。
- **坍缩**：Shumailov et al.（Nature 2024）发现，在自生成数据上递归训练，分布尾部会消失；Gerstgrasser et al. 发现，累积真实数据而不是替换掉，就可以避免坍缩。
- **代理指标被优化**：ReST 的奖励模型分数一直在涨，人类评价却在第 1 轮最好；Self-Rewarding 的生成长度从 1092 涨到 2552，GSM8K 却下降了。

**C 类：系统修改自己的改进过程**：STOP、Gödel Agent、Darwin Gödel Machine（DGM）、SICA、Huxley-Gödel Machine（HGM）：
- **DGM 的证据最强**：80 次迭代，完整版在 SWE-bench / Polyglot 上为 50.0 / 38.0，对照"不自我改进"版本为 39.0 / 28.0（Table 1）。代价是 SWE-bench 单次运行约 22,000 美元；还观察到 objective hacking：一个节点删掉了检测日志，拿到"满分"。
- **STOP**：只有 GPT-4 能带来提升，GPT-3.5 和 Mixtral 会退化。作者自己说，因为底层 LM 没有变，"this is not full recursive self-improvement"。
- **HGM 的反面观察**：当前的 benchmark 分数，只是"能不能产出更好后代"的弱代理（Pearson 相关 0.285–0.444）。
- **共同点**：**所有 C 类系统都冻结了底层模型权重**，外层的选择规则也都由人固定；**没有一篇展示每代增益随代际增大的复利曲线**。

## 5. 测量：有没有人测过二阶效应（详见 03）
- **一阶能力的 benchmark 很多**：RE-Bench（总预算 2h 时最佳 agent 约为人类的 4 倍，8h 时人类反超）、MLE-bench（o1-preview + AIDE 在 16.9% 的竞赛里拿到铜牌以上）、PaperBench、Automated LLM Speedrunning（不给提示时 agent 平均恢复不到 20% 的加速）。**它们测的都是"能不能做研究"，不是"下一代是不是更会做研究"**。
- **二阶效应最接近的测量是 AIDE² 的 ignition test**（Srikanth et al.，Weco AI，2026-09，preprint）：把自我改进产出的 agent 换上去当外环改进者，和人工研发的改进者比较，每组 3 seeds × 50 步。结果 0.780 对 0.782，作者自己说结论不确定。**只换了脚手架，底层模型没变；也没有 A/A、placebo、检索这些对照**。
- **前沿实验室的阈值**（属于行业文件，不是学术定义）：Anthropic、OpenAI、Google DeepMind 都用"进展速率相对历史基线的倍数"来定义 AI R&D 阈值。只有 Anthropic RSP v3.4 额外要求"加速趋势会导致更大的加速"，这是最接近二阶的表述。
- **METR 对 Anthropic 风险报告的评审**（2026-11-05）：估计 2025-10 之后的进展速率约为此前的 1.9 倍（95% CI 1.5–2.8），随建模选择不同，在 1.1–2.9 倍之间。〔注意〕这测的是实验室整体的进展速率（一阶的加速），不是"后继模型当研究者是否更快"。
- **方法学陷阱**（03 §4）：评测器被钻空子（STOP 有 0.42% 的尝试关闭了沙箱；DGM 删掉检测日志）、seed 噪声、污染、自动评分和人工评分之间的差距；基于问卷的 uplift 估计校准很差。

## 6. 和今晚实验结果的对照〔推断，供用户判断〕
| 我们的结果 | 文献中的对应 |
|---|---|
| **9B 自己写的 lessons 让 held-out 任务上的改进变慢**（Δ₂ = −0.196，比 placebo 还差） | **和文献一致**：ExpeL 加入自我反思后 HotpotQA 从 39.0 降到 29.0；没有 oracle 时自我纠错会伤害推理（GPT-4 GSM8K 95.5 → 89.0）；ACE 的 context collapse（57.1，低于不适配的基线 63.7）；Xiong 2025 中不加筛选地累积记忆，四个 agent 都更差。文献里记忆有益的案例，大多有可靠的外部验证来筛选、测试任务和存储条目相似、存的是可执行产物。**我们三条都不满足**：自由文本、没有筛选、迁移到 held-out 任务 |
| **LoRA 换心脏为零效应，而且和失败轨迹训练的 placebo 没有差别** | 和 Spurious Rewards 一致：Qwen2.5-Math-7B 用随机奖励也有 +21.4，真实奖励是 +29.1，说明增益可能来自激发已有行为，而不是信号内容；也和 Mind the Gap 一致（部分"提升"来自格式遵循）。**在检索范围内，没有找到任何一篇论文用"失败轨迹训练"做对照**，这个对照可能是我们设计里新的部分 |
| **27B 和 9B 打平** | 和 STOP 的观察相关：在任务上更强的改进者，不一定更会改进自己。HGM 也发现，当前分数和"产出更好后代的能力"只是弱相关 |
| **我们的 Δ₂ 设计**：换后继模型当改进者，在 held-out 任务上测，带 A/A、placebo、检索、"更多旧数据"四种对照 | 在检索范围内，**没有找到同样的设计**。最接近的 AIDE² 只换了脚手架、只有 3 seeds、没有对照 |

## 7. 对用户问题的直接回答（"学术界是不是理解得更纯粹"）
- **是的，学术界确实把门槛设得更高**：大多数理论工作要求改进机制在内部、可被改写，并且要复利。按这个标准，工业界材料里的"AI 辅助优化让模型更强、推理更快"属于 A 类，最多是 B 类。
- **但学术界内部也有分歧**：Yudkowsky 2013 用宽松定义（能产出后继者就算）；Tang et al. 2026 明确不把复利放进定义。至于"必须是通用模型"，文献分成两派。
- **实证上，学术界也没有做到 C + 复利**：最强的 DGM 冻结了底层模型；STOP 的作者自己承认"不是完整的 RSI"。所以"理论上的 RSI"和"工业界宣称的 RSI"之间，**学术界的实证工作同样还停在中间**。
- 〔推断〕这对核心论点可能有用：真正缺的不是更强的模型，而是**一个能区分 A、B、C 的测量方法**。在本次检索范围内，"换后继模型 + 对照"这种设计没有人做过。

## 8. 局限与未核实
- 部分 2025–2026 年的 preprint 只读了摘要（Liu & vdS、Fang、Zhang 2026、Moser 2026 等，各子文件中都有标注）。
- 没有用到的来源：Chollet 的文章（Medium 无法访问）；Bostrom 的原书（只看了二手概述）；Davidson 2023 的完整报告（被拦截，只用了作者本人的摘要）。
- STaR、Reflexion、Voyager 等的发表场所未确认；Shumailov 读的是 arXiv 版，不是 Nature 版。
- 检索是有限的，"没有找到"不等于"不存在"。写进报告时，建议写成"在我们的检索范围内"。
