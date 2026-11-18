# RSI-mini：从一阶测量走到二阶测量的最小实验平台（设计稿，未实现）

- 写作日期：2026-11-17。本文只做设计，**没有运行任何实验、没有安装或下载任何东西**。
- 标 **〔估算〕** 的数字是我的粗略估计，没有实测，**不能当作事实写进报告**；动手前先用 §5.1 的 smoke test 把它们换成实测值。
- 外部论断都附了本次会话里实际抓取过的 URL（清单见 §8）。引文由 WebFetch 的小模型从页面摘出，写进正文前请对关键句再人工对一次原文。
- 来自本地文件的事实标了文件路径。`research/lit_signal_hardness.md` 里已核查过的数字（L-xx）直接沿用其编号，不重复抓取。

---

## 0. 一页结论

**要回答的问题**：三家都展示了内环（agent 系统改进 kernel / 配方 / 数据），没有一家测量外环——"环产出的模型换上去当改进者以后，改进速率有没有变"。我们用能跑得动的模型做一个缩小版，看二阶效应是否出现，哪怕很弱。

**推荐平台（RSI-mini）**：一个 harness，四个可插拔部件。

```
任务族 T（train / held-out 两份）──▶ 封存评估器 E（确定性，改进者碰不到）
        ▲                                   │ 标量分数 + 诊断
        │ 提案（artifact 的修改）            ▼
   改进者 I_k =（权重 θ_k，脚手架 σ_k）◀── 轨迹日志 D_k（全部提案，含被拒的）
        │                                   │
        └────── 更新算子 U：I_{k+1} = U(I_k, D_k) ◀┘
                 U_w：LoRA / 拒绝采样微调（换权重，"换心"）
                 U_s：自己读轨迹、改写自己的策略备忘（换脚手架）
```

- **一阶**：固定 I_0，画 best-so-far 曲线。
- **二阶**：把 I_0 换成 I_1 = U(I_0, D_0)，在 **held-out 任务**上、同起点、同预算、同评估器重画曲线，看差值 Δ₂。
- **今天版**（§5.1）：任务族用 µAR（微型 autoresearch：确定性的小训练任务，CPU 上几秒一次），改进者用本机已有的 Qwen2.5-7B-Instruct，U_s 和 U_w 各做一代，带 A/A、placebo、retrieval 三个对照。分三个阶段，每个阶段结束都有可以写进报告的结果。
- **完整版**（§5.2）：加大样本、做三代、加"更多 M0 数据"对照，再搬到真实的 NanoChat（5090 / HPC）和带 bitwise 门槛的 kernel 任务上。

**它能说明什么**：在一个封闭的小任务族里，"用自己的成功轨迹更新过的改进者"是否比原来的改进者改得更快，以及这个增益一代一代是放大还是衰减。
**它不能说明什么**：不能说明开放式研究上的 RSI，不能估计 Forethought 意义上的 r，不能替代 Ling 尺度的实验。它是一把尺子，不是结论。

---

## 1. 定义

### 1.1 记号

- 任务 j：起点 artifact a_0（今天版是一份训练配置；完整版是 `train.py`），评估器 E_j 返回分数 s（越低越好）。
- 改进者 I =（权重 θ，脚手架 σ：prompt、策略备忘、工具）。
- 一次 run：从 a_0 出发做 N 次提案，每次提案交给 E 评一次，比当前最好的还好就接受（greedy keep / discard，和 Karpathy autoresearch 的规则一致："keeps or discards"，https://github.com/karpathy/autoresearch ）。
- 归一化增益：g(t) = (s_0 − best_t) / (s_0 − s_ref)。s_ref 是该任务的参考最优（离线随机搜索加所有 arm 汇总的最好值，只用于归一化，不给改进者看）。

### 1.2 一阶量

固定改进者 I，一阶量是曲线 g_I(t) 本身。汇总成一个主统计量和三个辅助量：

| 量 | 定义 | 用途 |
|---|---|---|
| **nAUC（主统计量）** | (1/N) Σ_{t=1..N} g(t)，先在 seed 内算，再在任务内对 seed 取平均 | 同时反映"改得多快"和"改到多远" |
| 终点增益 | g(N)，用封存 seed 重新评分后的值 | 防止在 dev seed 上过拟合 |
| 到阈值的迭代数 | 首次 g(t) ≥ 0.5 的 t（没到记为删失） | 直观的"速率" |
| 分解 | 有效率 v（提案合法）× 命中率 p（合法提案里有改进的比例）× 平均步长 m | 解释增益从哪里来 |

横轴的三种口径：评估器调用次数（主口径，对应真实 AI R&D 里最贵的训练算力）、改进者 token 数、wall-clock（只在架构 C 用）。

### 1.3 二阶量

- **一步二阶效应**：Δ₂(k) = mean_j [ nAUC(I_{k+1}; j) − nAUC(I_k; j) ]，j 遍历 held-out 任务，按任务配对。
- **代际比**：ρ_k = nAUC(I_{k+1}) / nAUC(I_k)。
- **趋势**：Δ₂(0), Δ₂(1), Δ₂(2)… 这个序列是增、平还是减。

和已有说法的对应关系：

| 我们的量 | 对应 | 出处 |
|---|---|---|
| Δ₂ > 0 | Weco 阶梯的 Level 2："ignition, when the system improves its own ability to improve itself" | https://weco.ai/blog/first-evidence-of-recursive-self-improvement |
| Δ₂ 不随代数衰减 | Level 3："progress stops slowing at a fixed budget and starts accelerating" | 同上 |
| ρ 持续 > 1 | 概念上接近 r > 1（"r gives the number of times software doubles for each time the cumulative work on software R&D doubles"）。**我们的实验估不出 r**，只借用方向 | https://www.forethought.org/research/will-ai-r-and-d-automation-cause-a-software-intelligence-explosion |

### 1.4 判定规则（动手前写死，相当于 pre-registration）

"观察到二阶效应"需要同时满足：

1. Δ₂ 的 95% 区间下界 > 0（对任务和 seed 做分层 bootstrap；另报按任务配对的 sign-flip 置换检验 p 值）；
2. Δ₂ > 2 × A/A 差值的标准差（A/A：同一个 I_0 用两组互不重叠的 seed 各跑一遍，差值就是噪声地板）；
3. I_1 同时显著好于 placebo 版本（见 §1.5 对照 3）。

不满足时的写法是"在本次功效下没有检出，最小可检出效应约为 X"（X 由 A/A 的标准差算出），而不是"没有二阶效应"。

### 1.5 对照

| # | 对照 | 排除什么 |
|---|---|---|
| 1 | 同起点、同 N、同评估器、同 max output tokens；任务 seed 用公共随机数 | 算力和起点不一致 |
| 2 | **A/A**：I_0 对 I_0 | 把采样噪声当成效应 |
| 3 | **Placebo 更新**：用同样条数的"合法但被拒"的轨迹做同样步数的 LoRA；脚手架侧用同样长度、但不看轨迹写出来的通用备忘 | "微调或加长 prompt 本身就有用"（学会输出格式等） |
| 4 | **Retrieval 对照**：I_0 加上 in-context 的轨迹摘要（就是 U_s 那一臂） | 区分"知识进了权重"和"知识放在上下文里就够了" |
| 5 | **Held-out 任务**，分 ID（同类型、新参数）和 OOD（没见过的任务类型） | 记住了训练任务的答案 |
| 6 | **起点缺陷随机化**：每个任务的起始配置从一个宽的先验里抽，缺陷各不相同（lr 过高或过低、正则过强、宽度不合适……） | "所有任务都是 lr 太低，学会调高 lr 就能迁移"这种设计造成的假阳性 |
| 7 | 非 LLM 基线：随机搜索、对当前最优做局部扰动 | 一阶本身是否成立（LLM 是否比瞎试强） |
| 8 | **更多 M0 数据对照**（完整版）：I_2 = U(I_0; D_0 ∪ D_1) 对 I_2' = U(I_0; D_0 ∪ D_0')，D_0' 是 I_0 用同样评估预算多跑出来的轨迹 | 区分"数据变多了"和"数据是更强的改进者产生的"。这是"递归"二字能否成立的关键对照 |

### 1.6 混杂（详细的控制办法见 §6）

contamination / memorization、evaluator hacking、compute mismatch、regression to the mean、selection effects、微调副作用（格式、多样性塌缩）、天花板效应、多重比较。

**对"同一任务、同一起点比较 M0 和 M1"的提醒**：M1 的训练数据里就有这个任务上被接受的改动，M1 在原任务上从头再来，等于默写答案。这条曲线可以报，但只能叫 **replay / amortization 曲线**（上界），不能当作二阶效应的证据。能当证据的只有两种设计：held-out 任务，或者"从前沿接着跑"（两臂都从 G0 找到的最好 artifact a* 出发，M0 带完整历史继续，M1 继续）。

---

## 2. 已有工作离"二阶测量"还差什么

| 工作 | 改的是什么 | 权重变不变 | 有没有"换改进者再测" | 出处 |
|---|---|---|---|---|
| STOP | 改进者程序改进自己 | 不变。原文："Since the language models themselves are not altered, this is not full recursive self-improvement." | 有（脚手架层） | https://arxiv.org/abs/2310.02304 |
| Darwin Gödel Machine | agent 自己的代码，带 archive | 不变。"powered by frozen pretrained foundation models" | 有消融："DGM w/o self-improve"，即负责修改的 meta agent 固定为初始 agent | https://arxiv.org/html/2505.22954 |
| Huxley-Gödel Machine | 同上，改了搜索的指导量 | 不变 | 指出 "a mismatch between the agent's self-improvement potential (metaproductivity) and its coding benchmark performance"：**benchmark 分高不等于更会改进自己** | https://arxiv.org/abs/2510.21614 |
| AIDE²（Weco） | 外环 agent 改写内环研究 agent 的代码；100 个外环步，约九成提案被拒 | 页面没有提到训练或微调 | 有 held-out（"second-order generalization"）；自评 Level 1，没到 Level 2 | https://weco.ai/blog/first-evidence-of-recursive-self-improvement ；https://arxiv.org/abs/2609.26457 |
| Haluptzok 等 | 模型自己出题、自己解，Python 解释器验证后拿来微调自己 | **变** | 测的是解题准确率（"more than doubles"），不是改进速率 | https://arxiv.org/abs/2207.14502 |
| Expert Iteration | 搜索当专家，网络蒸馏搜索，再反过来指导搜索 | 变 | 棋类；这是 U_w 的机制原型 | https://arxiv.org/abs/1705.08439 |
| PIE | 在"性能改进型编辑"上微调，含 self-play 增广；为了测量稳定专门用 gem5 模拟器 | 变 | 测加速比，不测代际 | https://arxiv.org/abs/2302.07867 |
| RSIBench-Data | agent 在固定后训练栈上迭代改数据策略 | 目标模型变，agent 不变 | 没有。但给了一个重要的负面事实：找到最好版本后继续搜索的 run 里，78.26% 最终分数更低 | https://arxiv.org/abs/2607.25886 |
| Recursive / Naive / Apex | kernel、配方、数据 | 被造的模型变，改进者没有换代对比 | 没有。Apex 的 AI4AI（Apex 对 Base 当研究者）是单点对比，最接近 | `research/reading/00_总览.md`、`04_Apex35B_导读.md` |

**空位**：脚手架层的二阶测量已经有人做了（DGM、AIDE²），而且都冻结权重。**权重层的"换心"加上速率测量、再配齐 placebo 和"更多 M0 数据"对照，目前没有看到**（本次只做了一轮检索，不排除漏检）。这正好是 Ling 路线（在自家基座上继续训练出 RSI 模型）最需要的那把尺子。

HGM 的结论对设计有直接约束：不能拿改进者的通用 benchmark 分数代替 Δ₂，必须直接测它的后代或它的改进曲线。

---

## 3. 本机现状（2026-11-17 晚检查，影响今天方案）

| 项 | 状态 | 对方案的影响 |
|---|---|---|
| Flint 权重 | `flint-int4/model-0000{0..4}.safetensors` 仍是 135 字节的 LFS 指针；`git lfs pull` 进程还在跑，`.git/lfs` 当时约 18 GB | **今天不能把方案押在 Flint 上**。下完之后可以作为冻结改进者加入 |
| `rec` | 已安装（`~/.local/bin/rec`），当时没有在运行的 server | 同上 |
| MLX | `/Library/Frameworks/Python.framework/Versions/3.12` 下有 mlx 0.22.0、mlx_lm 0.21.1，含 `mlx_lm.lora`、`mlx_lm.server`、`mlx_lm.convert` | 不用安装就能在 Mac 上做 QLoRA 和本地推理 |
| mlx_lm 支持的结构 | `models/` 里有 `qwen2.py`，**没有 `qwen3.py`** | Qwen2.5 可用；Qwen3-1.7B 虽然权重在缓存里，这个版本的 mlx_lm 加载不了 |
| 本地权重（HF cache） | Qwen2.5-7B-Instruct 完整（4 个分片约 15.2 GB）；Qwen3-1.7B 完整；gpt-oss-120b MXFP4 GGUF 完整；Qwen2.5-0.5B、Qwen3-8B、Llama-3.1-8B 只有 tokenizer 或 config | 今天权重层的 M0 只能选 **Qwen2.5-7B-Instruct** |
| PyTorch | conda 里 torch 2.10（MPS 可用），**没有 peft** | Mac 上的 LoRA 走 mlx_lm，不走 peft |
| `llama-server` | 已安装 | 可以用 gpt-oss-120b 当冻结的强改进者（可选） |
| CPU | 12 个性能核 + 4 个能效核 | µAR 评估可以 8–10 路并行 |
| GPU 锁 | `bench/gpu_lock.sh status` 返回 free | **本地生成和 LoRA 会干扰 `bench/` 的 tok/s 测量**，跑之前必须 `acquire`，或者把改进者放到 5090 上 |
| 5090、两个 HPC | 没有检查（任务范围只允许读本地文件） | 今天只把 5090 当可选加速；HPC 只放进完整版 |

NanoChat 的噪声地板（我用 `nanochat_autoresearch/results/val_bpb.csv` 算的样本标准差，n=10，B200、5 分钟预算；`artifacts_recursive.md` 明确说它没有算这些）：

| label | mean | sd |
|---|---|---|
| karpathy_seed | 0.9751 | 0.0112 |
| karpathys_best | 0.9646 | 0.0154 |
| optimized_from_karpathy | 0.9109 | 0.0067 |
| vanilla_transformer | 1.0587 | 0.0101 |

含义：Recursive 整个战役的总增益是 0.9751 → 0.9109，约 0.064；而单 seed 的标准差就有 0.007–0.015。**在真实 NanoChat 上，单次运行决定 keep / discard 时，小于约 0.02 的"改进"和噪声分不开**。所以今天版不能直接用 NanoChat（每次评估 5 分钟，还要多 seed），必须先换成确定性的小任务族。5090 上的噪声可能不同，要重新测。

---

## 4. 三个候选架构

### 架构 A：权重层外环（"换心"）

```
G0:  M0 ──提案──▶ E(train 任务) ──▶ D_0（接受 / 拒绝 / 非法，全记）
U_w: D_0 里被接受的 (prompt → 提案) ──QLoRA──▶ M1        placebo: 被拒的 ──▶ M1p
G1:  {M0, M0', M1, M1p} ──提案──▶ E(held-out 任务) ──▶ 曲线 ──▶ Δ₂
     M1 ──提案──▶ E(train 任务) ──▶ D_1 ──▶ M2 …（完整版）
```

| 项 | 内容 |
|---|---|
| 任务 | **今天**：µAR 任务族（§5.1）。**完整版**：真实 NanoChat（Karpathy autoresearch：只改 `train.py`，固定 5 分钟，指标 val_bpb） |
| 评估器与加固 | 今天：提案是符合 schema 的 JSON 配置，改进者的输出不会被当代码执行；评估在独立进程；预算由 harness 换算步数并强制，外加硬超时；dev seed 的分数给改进者看，封存 seed 和封存 test split 只在最后由分析脚本读取。完整版见 §5.2 的加固清单 |
| 改进者 | M0 = Qwen2.5-7B-Instruct（Mac 上转 4-bit 用 mlx_lm，或 5090 上跑） |
| 代际之间变什么 | **只变 LoRA adapter**。prompt、采样参数、N、任务、评估器全部不变 |
| 测量 | §1 的 nAUC、Δ₂、分解量；held-out 分 ID 和 OOD 两档报告 |
| 成本〔估算〕 | 每个提案约 3 秒（输入约 800 token，输出约 120 token，7B 4-bit，M4 Max）；每臂 288 个提案约 15 分钟；QLoRA 每个 5–15 分钟；评估 CPU 每次 3–5 秒、可并行。今天必做的几臂顺序跑完约 1.5 小时，加上 placebo 臂约 2 小时。全部本地，API 花费为 0 |
| 能说明 | 自己的成功轨迹进了权重以后，改进者在没见过的任务上是否改得更快；增益来自格式（placebo 也有）、来自可以放在上下文里的知识（retrieval 臂也有），还是权重独有 |
| 不能说明 | U 的配方是人定的，不是改进者自己设计的；任务族封闭，学到的可能只是"这族任务的搜索先验"（OOD 档能部分回答）；一到两代看不出长期趋势 |

机制上这就是 Expert Iteration：内环搜索当专家，外环把搜索结果蒸馏进策略，策略再去指导下一轮搜索。

### 架构 B：脚手架层外环（DGM / STOP / AIDE² 的缩小版）

```
G0:  (θ, σ0) ──▶ E(train) ──▶ D_0
U_s: 同一个模型读 D_0 的摘要，写给"下一代自己"的策略备忘 lessons.md ──▶ σ1 = σ0 + lessons
     placebo: 不给轨迹，让它写同样长度的通用调参建议 ──▶ σ_generic
G1:  {(θ,σ0), (θ,σ1), (θ,σ_generic)} ──▶ E(held-out) ──▶ Δ₂
完整版：σ 扩展到可以改自己的提案策略代码（保留 archive，像 DGM 那样）
```

| 项 | 内容 |
|---|---|
| 任务与评估器 | 同架构 A |
| 改进者 | 任何冻结模型：Qwen2.5-7B（同时充当 A 的 retrieval 对照）、Claude Haiku 5.5 / Sonnet 5.5（API）、Flint（下载完成后，经 `rec serve`）、gpt-oss-120b（`llama-server`） |
| 代际之间变什么 | 只变 prompt 里的策略备忘（完整版：提案策略代码） |
| 测量 | 同 §1 |
| 成本〔估算〕 | 不需要训练。本地每臂约 15 分钟。Claude：每个提案约 1.2k 输入 + 0.4k 输出（含 thinking），每臂 288 个提案约 0.35M 输入 + 0.12M 输出；按 claude-api skill 内置价目表（缓存日期 2026-11-03，未另行抓网页核对）：Haiku 5.5（$0.10 / $0.50 每 MTok）约 $0.1 每臂，Sonnet 5.5（$2 / $10）约 $2，Opus 5.5（$4 / $20）约 $4，Fable 5.2（$8 / $40）约 $8。**建议今天的 API 上限 $20** |
| 能说明 | 冻结模型靠自己写的经验能否加快后续改进；并顺带给出"改进者能力 → 改进速率"的横截面（Qwen-7B、Haiku、Sonnet、Flint 各一条一阶曲线），这是 Apex AI4AI"固定任务、换研究者"的形状 |
| 不能说明 | 不是"换心"，权重没变；已有 DGM 和 AIDE² 的先例，新意有限；横截面不是二阶证据（那些模型不是这个环产出的） |

注意：当前这几代 Claude 模型不接受 `temperature` 等采样参数（见 claude-api skill 的参数表），所以 Claude 臂的 seed 之间只靠自然采样差异，这一点要写进方法。

### 架构 C：推理 kernel 任务 + bitwise 门槛（"速度环"）

```
artifact：rec 引擎（Flint 的推理 kernel）
E：① bitwise 门槛：逐层与 MLX 参考实现一致 + 固定 prompt 的 greedy token 序列一致
   ② decode_tps：按 bench/config.json（context 1k–128k，median over repeats），ABAB 交替测量
改进者：Claude Code，或 Flint 自己（Flint 优化运行 Flint 的引擎）
代际：引擎 v0 → v1（更快）。改进者权重不变，变的是它每小时能出多少提案
二阶量：同一个基准循环（例如 Flint 当改进者跑 µAR），在引擎 v0 和 v1 下的 wall-clock 曲线之差
        预期值：循环加速 = 1 / ((1−f) + f/s)，f 是循环里生成所占的时间比例，s 是 tok/s 之比
```

| 项 | 内容 |
|---|---|
| 加固 | 评估文件只读并校验 hash；测速 prompt 分 dev 和封存两套；每次评估起新进程；`bench/gpu_lock.sh` 互斥；先测 A/A 噪声，小于 2 倍噪声的提速不认 |
| 成本〔估算〕 | 每次评估含重启、预热和多个 context 长度，约几分钟；每小时 6–10 次 |
| 能说明 | 存在一条物理上扎实、bitwise 安全的二阶通道：改进推理引擎 → 改进者变快 → 单位 wall-clock 的改进变多。三家都在做 kernel（NaiveRT、rec），没有一家把这一步闭合起来测 |
| 不能说明 | 只改变速率，不改变每个提案的质量；增益受内存带宽上限约束（`rec_optimization_analysis.md` §2 的上限是〔估算〕），所以是**有界的、会饱和的**二阶效应，不会 ignite；改进者是远程 API 时不适用 |
| 今天是否推荐 | **不作为主线**。rec 已经被 Eureka 深度优化过，今天找到一个 bitwise 一致的提速没有把握，而且 Flint 还没下完。但如果 `bench/` 那条线今天得到任何经过验证的提速，只需再跑一次约 15 分钟的对照，就能白拿一个二阶数据点 |

### 三者的取舍

| | A 权重层 | B 脚手架层 | C 速度环 |
|---|---|---|---|
| 与"换心"命题的贴合度 | 最高 | 中 | 中（换的是"心率"） |
| 今天能否完成 | 能，有风险（阶段 3） | 能，风险低（阶段 2） | 取决于别的工作线 |
| 新意 | 高 | 低 | 中 |
| 结果为空时是否仍有信息量 | 有（配合 A/A 给出最小可检出效应） | 有 | 有（Amdahl 上界） |

---

## 5. 推荐方案

### 5.1 今天版（约 3.5–4 小时 wall-clock，其中需要人盯的约 1.5 小时）

**原则**：一个 harness，三个阶段，每个阶段结束都有可以单独写进报告的产出；每个阶段有 go / no-go 判断。你一共只剩约 5 小时，运行期间无人值守，可以同时写报告。

#### 任务族 µAR（micro-autoresearch）

- 每个任务 =（数据生成器，模型骨架，算力预算 B，带随机缺陷的起始配置 c_0）。
- **任务类型**（每种约 15 行代码，数据全部由脚本按 seed 生成，不下载）：模加法、稀疏奇偶、序列反转 / 复制、括号匹配（Dyck）、teacher-student 回归、带标签噪声的二维分类、in-context 线性回归、字符级 LM（语料取本仓库的源码文件）。
- **动作空间**：一份 JSON 配置，约 12 个旋钮：optimizer、lr、lr schedule、warmup 比例、weight decay、batch size、width、depth、activation、norm、init scale、dropout / grad clip。
- **预算**：harness 按 `steps = B / (batch × 参数量)` 换算步数，改进者没法靠加大模型或多跑几步"买"算力；外加硬超时，超时或 NaN 记为非法。
- **分数**：val loss，对 2 个 dev seed 取平均；单线程 CPU、固定 seed，结果完全确定。
- **给改进者的反馈**：dev 分数、5 个检查点的 train loss、train / val 差距、NaN 或超时标志。**不给**：封存分数、评估器代码、s_ref。
- **划分**：train 8 个任务（4 种类型 × 2 组参数）；held-out-ID 4 个（同类型，新参数，新缺陷）；held-out-OOD 4 个（2 种没见过的类型 × 2）。
- 为什么不直接用 NanoChat：见 §3 的噪声地板。µAR 保留了 autoresearch 的三个要素（固定算力预算、单一 val 指标、keep / discard），把单次评估从 5 分钟压到几秒，并且把评估噪声压到 0。

#### 循环协议

每个（任务，seed）跑 N = 12 次提案。每次给改进者：任务卡片、当前最优配置及其反馈、最近 6 条历史（提案 → 结果）。输出 `{"hypothesis": 不超过 30 词, "config": {...}}`。JSON 或 schema 不合法算一次无改进的迭代。

#### 各臂（held-out：8 个任务 × 3 个 seed × 12 = 每臂 288 个提案）

| 臂 | 改进者 | 阶段 | 必做 |
|---|---|---|---|
| R、R-local | 随机搜索、局部扰动（不用 LLM，几乎不花时间） | 1 | 是 |
| A0a | M0, σ0 | 1 | 是 |
| A0b | M0, σ0，另一组 seed（A/A） | 1 | 是 |
| A1 | M0, σ1（自己写的 lessons） | 2 | 是 |
| A1p | M0, σ_generic（placebo 备忘） | 2 | 尽量 |
| A2 | M1 = QLoRA(M0, D_0 中被接受的), σ0 | 3 | 是 |
| A2p | M1p = QLoRA(M0, D_0 中合法但被拒的), σ0 | 3 | 尽量 |
| C-Haiku / C-Sonnet | Claude, σ0（能力阶梯，API 并行跑，不占本地） | 1 | 可选 |
| F | Flint, σ0（仅当下载完成且 `rec serve` 正常） | 1 | 可选 |

#### 步骤和时间表（时间均为〔估算〕）

| 时间 | 做什么 | 产出 / 判断 |
|---|---|---|
| 0:00–0:15 | 写 `PREREG.md`：主统计量 nAUC、各臂、§1.4 的判定规则、N、seed 列表。`bash bench/gpu_lock.sh acquire rsi-mini`。可选：ssh 看一眼 5090 上有没有 torch / transformers / peft 和空闲显存 | 协议冻结 |
| 0:15–1:00 | 让 Claude Code 写 harness（建议放 `rsi-report/platform/`）：`tasks.py`、`evaluator.py`（独立进程，封存 split）、`loop.py`、`improver.py`（OpenAI 兼容接口，本地和 API 通用）、`update.py`（生成 lessons；生成 SFT 的 `train.jsonl` / `valid.jsonl`，格式是每行一个 `messages` 列表，见 mlx-lm 的 LORA.md）、`analyze.py`。同时后台做两件事：`mlx_lm.convert` 把 Qwen2.5-7B 转成 4-bit；每个任务 300 次随机搜索，得到 s_ref | 单元测试：同一配置评两次分数完全相同；非法配置被拒；预算换算正确；封存分数不出现在任何给改进者的字符串里 |
| 1:00–1:15 | **Smoke test**：2 个任务 × 20 个提案 | **Gate 1**：单次评估 < 5 秒；M0 的合法率 ≥ 70%；实测每个提案的秒数。把本文所有〔估算〕换成实测值，重算后面的时间。合法率过低就简化 schema；速度过慢就把 N 降到 10 或每臂 seed 降到 2 |
| 1:15–2:00 | **阶段 1**：G0（M0 在 train 任务上，288 个提案）→ D_0；A0a、A0b 在 held-out 上（三段各约 15 分钟，顺序跑）；R 基线；Claude 臂并行 | **一阶结果**：best-so-far 曲线，M0 对随机搜索；A/A 噪声地板；（可选）能力阶梯图。**Gate 2**：如果 M0 不比 R-local 好，说明任务对它太难或反馈没被用上，先调任务卡片再继续 |
| 2:00–2:20 | **阶段 2**：M0 读 D_0 的摘要写 lessons（≤ 300 词）；跑 A1 | **脚手架层 Δ₂**，同时它就是权重层的 retrieval 对照 |
| 2:20–2:50 | **阶段 3**：训练 M1：`mlx_lm.lora --model <4bit> --train --data sft/ --num-layers 8 --batch-size 4 --iters 200`（模型是量化的，所以自动走 QLoRA；训练期间不做生成，避免抢 GPU）；用 `--adapter-path` 起 M1 的 server，跑 A2 | **权重层 Δ₂**；A2 对 A1。**到这里必做的部分全部完成** |
| 2:50–3:30 | 尽量做：A1p；训练 M1p，跑 A2p | A2 对 A2p、A1 对 A1p 两个 placebo 比较。做不完就留给完整版，并在结论里写明缺这两个对照 |
| 3:30–3:45 | `analyze.py`（每个阶段结束都跑一次，这里出终稿）：曲线图（坐标轴用真实比例，不截断）、Δ₂ 及区间、分解表（v、p、m）、ID 对 OOD、dev 对封存的差距。`gpu_lock.sh release rsi-mini` | 一张主图、一张表、"结果 + 局限"一段 |

SFT 数据的做法：取 D_0 里每个被接受的步骤，输入是当时的完整 prompt，输出是模型当时的提案；去重；每个任务设上限以免某个任务占比过大；留 10% 做 valid。训练前用 hash 核对：SFT 集里不出现任何 held-out 任务的卡片或配置。

#### 退路

| 情况 | 做法 |
|---|---|
| Gate 1 不过（本地模型太慢或合法率太低） | 今天只做阶段 1 和 2，改进者换成 Claude Haiku 5.5（API 并行，几分钟一臂）；权重层移到完整版 |
| 5090 环境现成 | 把改进者搬到 5090，同一迭代步的所有 run 批量生成，Mac 的 GPU 留给 `bench/`；这样 placebo 臂也能当天做完 |
| D_0 里被接受的步骤少于约 40 条 | LoRA 没有意义。改成"每个状态下取 top-1 提案"的 hindsight 重标，或者把 G0 的 seed 加倍；仍不够就只报阶段 2 |
| 时间不够 | 砍臂的顺序：F → C-Sonnet → A1p → A2p。A0a、A0b、A1、A2 不能砍 |
| 所有臂都很快到 g = 1（天花板） | 把 N 降到 8 或把预算 B 调小，让任务变难；这要在 Gate 1 发现，不能在看到各臂结果之后再改 |

#### 结果怎么读

| 观察 | 解读 |
|---|---|
| A1 > A0，A2 ≈ A2p ≈ A0 | 经验放在上下文里有用，进不了权重（数据太少或配方不对）。二阶效应只在脚手架层出现 |
| A2 > A0 且 A2 > A2p，ID 有、OOD 没有 | 权重层学到的是这族任务的搜索先验（amortization），不是可迁移的改进能力 |
| A2 > A2p 且 OOD 也有 | 最强的结果：权重层的二阶效应，并且迁移到没见过的任务类型 |
| A2 ≈ A2p > A0 | 增益来自微调本身（格式、合法率），与"成功轨迹"无关。分解表里应当表现为 v 上升、p 不变 |
| 都不显著 | 报最小可检出效应，写成"试点功效不足"，进入完整版 |

### 5.2 完整版（数天）

| 阶段 | 时间〔估算〕 | 内容 | 硬件 |
|---|---|---|---|
| **P1 补足功效，做出趋势** | 第 1 天 | held-out 扩到 24 个任务 × 5 个 seed，N = 30；全部 placebo 臂；权重层做 **G0 → G1 → G2 → G3 三代**，每代的轨迹由当代改进者自己产生；加入 §1.5 的对照 8（更多 M0 数据）；报 Δ₂(0)、Δ₂(1)、Δ₂(2) 的趋势和每代的提案多样性（重复率）。一部分任务把动作空间放开到代码补丁（沙箱内） | 5090（批量生成 + peft，需要先装环境）；µAR 评估只用 CPU，可以放到 HPC 的 CPU 节点 |
| **P2 搬到真实 NanoChat** | 第 2–4 天 | ① 加固评估器（见下）；② 在 5090 上重测噪声地板（baseline 跑 10 个 seed）；③ 把 P1 得到的 M0、M1、M2 直接拿来当 NanoChat 的改进者，这本身是一次很强的 OOD 检验（在 µAR 上学到的东西能不能迁移到真任务）；④ 用 NanoChat 自己的轨迹再训一代。算力：3 臂 × 3 seed × 20 次迭代 = 180 次实验，按每次约 6.5 分钟（5 分钟训练加编译和评估）约 19.5 GPU 小时 | 5090；HPC 用 Slurm array 按（臂，seed）并行。每个作业独立、可续跑，排队延迟只影响总时长 |
| **P3 速度环** | 第 3–5 天 | 架构 C：Flint + rec，bitwise 门槛；测引擎 v0 和 v1 下同一循环的 wall-clock 曲线 | Mac（独占 GPU 锁） |
| **P4 把 U 也交出去** | 第 2 周起 | 元任务：artifact 是"训练下一代改进者的配方"（数据筛选规则、LoRA 超参），评估器是 held-out 上的 nAUC。这时被改进的东西才真正是改进者本身。每次评估 = 一次微调 + 一组 held-out 曲线，约 20–30 分钟 | 5090 + HPC |

**NanoChat 评估器加固清单**（依据 `artifacts_recursive.md` §1：Recursive 的 `lib.py` 里没有强制 time budget，计时、seed、评测调用都写在可以被改的 solution 脚本里；仓库里没有沙箱和隐藏测试集）：

1. 计时和停止移到 harness：harness 负责 wall-clock 计时和强杀，不信任脚本自己报的时间。
2. 评测移到 harness：脚本只负责产出 checkpoint 和一个标准 forward 接口；harness 在独立进程里、在封存的 val shard 上算 BPB。
3. `prepare.py` / `lib.py` 只读并校验 hash；每次实验起新进程；禁止写数据目录。
4. Seed racing：先跑 1 个 seed 筛选，比当前最优好出 1 个 sd 以上再补 2 个 seed 确认（autoresearch-mlx 的 `rigor.py` 是同一思路：多 seed 加 bootstrap 决定 keep / discard，https://github.com/trevin-creator/autoresearch-mlx ）。
5. 可选的低噪声变体：固定 seed、固定步数、限制参数量。代价是把"提高吞吐"这条合法的改进路径排除在外，所以只用于测量改进者，不用于和三家的数字比较。
6. 每一代抽查增益最大的若干提案，人工判断是否 hack，并把 **hack 率作为每一代都要报的指标**（AIDE² 报告了这个指标随自改进下降：55% → 32%，https://arxiv.org/abs/2609.26457 ）。
7. 5090 只有 32 GB，Karpathy 的默认配置是在 H100 上测的；README 对小机器的建议是降低 `DEPTH`、`MAX_SEQ_LEN`、`TOTAL_BATCH_SIZE`、`EVAL_TOKENS` 或改用 TinyStories。具体取值要在 5090 上试，**目前未知**。

**改进者候选（完整版）**：Qwen2.5-7B / 更大的开源 coder 模型（5090 上 QLoRA，需要下载）；Flint（冻结，脚手架层和速度环）；Claude（冻结，能力上界参照）。Ling-3.0-flash 本体 124B，Mac 和 5090 都跑不了；HPC 上能否推理取决于 GPU 型号和显存，**未知**。`inclusionAI` 下有 `Ling-3.0-tiny-GGUF`（本地只有列表，没有读过它的卡片），值得查一下能否作为"Ling 家族的可运行改进者"。

---

## 6. 风险清单与效度威胁

| # | 威胁 | 表现 | 控制 | 残余风险 |
|---|---|---|---|---|
| 1 | **Contamination：基座的预训练知识** | 强模型早就知道 NanoChat / Muon 这类技巧 | 代际比较（M_k 对 M_{k+1}）共用基座，这部分被差分掉；µAR 任务是脚本新生成的 | 能力阶梯图（跨模型）不受保护，只能当参考 |
| 2 | **Contamination：轨迹泄漏** | M1 见过训练任务的答案 | held-out（ID / OOD 两档）；SFT 集与 held-out 的 hash 核对；同任务曲线只报为 replay | ID 档仍可能共享任务类型层面的最优区域，所以要单独报 OOD |
| 3 | **Evaluator hacking** | 改进者利用评估器漏洞 | 今天：JSON schema 动作空间，输出不被执行；封存 seed 和 split；预算由 harness 强制。完整版：§5.2 清单 | 动作空间放开到代码后风险陡增：SOL-ExecBench 有 14.5% 的提交被判 hack，RE-Bench 128 次运行里 39 次有 hack（lit L-21、L-18） |
| 4 | **外环放大评估器缺陷** | 一代代朝漏洞方向优化 | 评估器在改进者的写权限之外；每代报 hack 率；增益最大的提案人工抽查 | DGM 出现过删掉幻觉检测标记的案例（lit L-26） |
| 5 | **Compute mismatch** | 一臂多用了评估或 token | 各臂 N 相同、max output tokens 相同；记录每臂 token 数；U 的训练开销单列；脚手架臂多出的 prompt token 单列 | wall-clock 不可比（本地对 API），所以主口径用评估次数 |
| 6 | **Regression to the mean** | 从多个候选里挑出"最好的 I_1"，再测时回落 | 不在 held-out 上做任何选择；checkpoint 只用 train 任务上的 valid loss 选；终点增益用封存 seed 重评 | — |
| 7 | **Selection effects / 幸存者偏差** | 只拿被接受的提案训练，学到的可能是"碰巧在 dev seed 上好"的改动 | placebo 臂；封存 seed；OOD 档；起点缺陷随机化 | 数据量小的时候分不清，靠完整版加样本 |
| 8 | **微调副作用** | 合法率上升被当成"更会研究"；多代之后提案趋同 | 分解量 v / p / m；placebo；每代报重复率和提案分布 | 多代自训练的多样性塌缩只能在 P1 观察 |
| 9 | **天花板 / 地板** | 所有臂都到 g = 1，或都不动 | Gate 1 检查；s_ref 取自大规模随机搜索；只允许在看到各臂结果之前调难度 | — |
| 10 | **多重比较** | 换统计量直到显著 | `PREREG.md` 里只有一个主统计量和一条判定规则；其余标为探索性 | — |
| 11 | **"更多数据"对"更好的数据"** | G2 的提升只是因为训练数据变多 | 对照 8 | 今天版做不了，结论里要写明 |
| 12 | **U 是人定的** | 被说成"模型自己改进自己" | 明确写：自生成的是数据，配方是人给的；P4 才把配方交出去 | — |
| 13 | **封闭任务族** | 增益一代后饱和（ρ < 1） | 这本身是可报告的结果。可对照 RSIBench-Data：找到最好版本后继续搜索的 run 里 78.26% 最终更差 | 说明要有任务多样性来源，见 §7 |
| 14 | **功效不足** | 8 个任务 × 3 个 seed 只能检出大效应 | 报区间和最小可检出效应；试点定位 | — |
| 15 | **GPU 争用** | 本地生成干扰 `bench/` 的 tok/s | `gpu_lock.sh`；µAR 评估是确定性的 CPU 任务，不受影响 | — |
| 16 | **跨机器浮点差异** | Mac 和 HPC 上同一配置分数不同 | 同一组比较的所有评估在同一台机器上做 | — |
| 17 | **Flint 没下完 / HPC 排队 / 5090 环境未知** | 计划落空 | 今天版完全不依赖这三样 | — |
| 18 | **引文由小模型摘录** | 引号里的话可能不精确 | 写进报告前人工对原文 | — |

---

## 7. 对应到 Ling 的 RSI 路线

### 7.1 能放大的部分

| RSI-mini 里的东西 | 在 Ling 上对应什么 |
|---|---|
| "换改进者再测"的协议（held-out、A/A、placebo、更多数据对照） | **改进者的晋升门槛**：新一代 Ling 只有在封存任务集上 Δ₂ 的区间下界 > 0、hack 率不升、通用能力不退时，才替换上一代进入研发环。相当于给"心脏"做 CI |
| 轨迹日志 D_k（接受、拒绝、非法全记，带评估器的判定） | 从第一天起就按训练数据的规格记录研发 agent 的轨迹。Naive 和 Apex 的做法是把模型训练成"会做研发"，这批日志就是原料 |
| U_w（拒绝采样 / LoRA） | 在 Ling 基座上用被验证过的研发轨迹做 continued training 或 RLVR |
| 硬信号优先 | 自动化顺序：推理 kernel（bitwise 门槛）→ 固定预算下的训练配方（val loss）→ 数据配方（benchmark，较软）→ 架构（最软，人来定）。与 `lit_signal_hardness.md` 的结论一致：跟着**被加固过的**信号走 |
| 架构 C 的速度环 | Ling-3.0-flash 是 35 层 KDA + 7 层 MLA，线性注意力的状态大小固定，decode 主要受权重读取限制（`rec_optimization_analysis.md` §5，〔估算〕）。kernel 环的提速会直接变成改进者的吞吐，可以用同一个 Amdahl 式子预估和验收 |
| 能力阶梯图 | 把 Ling-3.0-flash、Ling-3.1-flash、Flint、N0.5、Apex 放进同一个封存任务集当改进者，得到"谁更会改进"的排名。HGM 的发现说明这个排名不能用通用 benchmark 代替 |

### 7.2 放大不了、或者会变的部分

| 小实验里成立的 | 到 Ling 尺度会怎样 |
|---|---|
| JSON 配置的动作空间几乎没有 hack 面 | 自由改代码后，hack 是常态而不是例外（lit L-18、L-21、L-22）。评估器加固要按一个独立的工程项目来投入 |
| 评估确定、几秒一次 | 真实训练有 seed 噪声（NanoChat 单 seed sd 约 0.007–0.015 BPB），评估成本主导一切，需要 racing 和多 seed |
| 一代只要十几分钟 | 124B 的 MoE 做一代 continued training 要按天或周计。**一个季度能观察到的代数很少**，所以二阶趋势只能靠"每代测得很准"（任务多、seed 多），不能靠"代数多"。可以先用小模型当"学生改进者"做快速预演 |
| 任务族里学到的先验能迁移 | 窄任务族里的先验到真实研发上未必有用。需要任务多样性的来源（新的 kernel、新的数据域、新的硬件），否则 ρ < 1，环会自己熄火 |
| 改进者只提一步建议 | 真实研发是长程的；RSIBench-Data 的结论是 agent "cannot yet translate feedback into consistent improvements"，要保留 checkpoint 并允许回滚 |
| U 的配方由人定 | 到某一步，配方本身要成为被优化的 artifact（P4）。在那之前，"RSI"里递归的部分其实是人在闭合 |
| 人只在开头写协议 | 人要长期负责三件事：定义任务分布、封存和更新评估器、决定哪一代改进者可以上岗 |

### 7.3 这个小实验能给 Ling 团队的三个具体东西

1. 一个不依赖模型规模的二阶统计量和判定规则（§1.3–1.4）。
2. 一套对照臂的清单（§1.5）。目前公开材料里的"RSI"声明都缺其中至少两项。
3. 一个事先写好的结果解读表（§5.1 末）。包括空结果该怎么报告。

---

## 8. 来源

### 8.1 本次会话实际抓取的 URL

| 用于 | URL |
|---|---|
| autoresearch 的规则、硬件要求、小机器建议、fork 列表 | https://github.com/karpathy/autoresearch |
| Apple Silicon 移植、`rigor.py` 多 seed 判定 | https://github.com/trevin-creator/autoresearch-mlx |
| STOP | https://arxiv.org/abs/2310.02304 |
| Darwin Gödel Machine（摘要、正文） | https://arxiv.org/abs/2505.22954 ；https://arxiv.org/html/2505.22954 |
| Huxley-Gödel Machine | https://arxiv.org/abs/2510.21614 |
| Language Models Can Teach Themselves to Program Better | https://arxiv.org/abs/2207.14502 |
| Expert Iteration | https://arxiv.org/abs/1705.08439 |
| PIE（Learning Performance-Improving Code Edits） | https://arxiv.org/abs/2302.07867 |
| RSIBench-Data | https://arxiv.org/abs/2607.25886 |
| AIDE²（论文摘要、博客） | https://arxiv.org/abs/2609.26457 ；https://weco.ai/blog/first-evidence-of-recursive-self-improvement |
| r 的定义与 r > 1 的含义 | https://www.forethought.org/research/will-ai-r-and-d-automation-cause-a-software-intelligence-explosion |
| mlx-lm 的 LoRA 用法、数据格式、QLoRA 条件 | https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md |

### 8.2 没有核实、不要引用

- RSI-Bench v0.1（https://rsi-bench.duckdns.org/ ）：两次抓取都失败，只在搜索摘要里见过"可见 dev 集加隐藏 seal 集"的说法。
- AIDE² 的 KernelBench 数字：二手报道里有"63% → 34%"和"55% → 32%"两个版本；本文只用了 arXiv 摘要里的 55% → 32%，摘要没有点名 KernelBench。
- Claude 价格取自 claude-api skill 的内置价目表（缓存日期 2026-11-03），没有另外抓定价网页。
- "权重层二阶测量目前没有人做过"是一轮检索后的判断，不是穷尽检索的结论。
- 5090、cluster-B、cluster-A 的环境、GPU 型号和配额都没有检查。
- §3 的 NanoChat 标准差是我对仓库 CSV 的计算（`statistics.stdev`，n=10），不是仓库自己报告的数字。
