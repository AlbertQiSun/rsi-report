# Flint 预训练和后训练逆向解析（2026-11-17 晚）

> **证据等级**
> - **一手**：本地权重、tokenizer、config 文件，以及实测输出。
> - **推断**：依据这些一手材料推出来的结论，均标为〔推断〕。
>
> 所有推断都是间接证据，不能当成 Recursive 的官方披露。
>
> **数据和脚本**
> - 结果文件：`research/flint_pretrain_forensics.json`、`research/flint_probes.jsonl`、`research/flint_probes2.jsonl`、`research/flint_router_stats.json`
> - 脚本：`tools/flint/` 下的 `tok_compare.py`、`pretrain_forensics.py`、`probes.py`、`probes2.py`、`router_stats.py`
> - 对照用的 tokenizer：`other/tokenizers/`（DeepSeek-V3、GLM-4.5、Qwen3-30B-A3B、gpt-oss-20b）
> - 运行条件：探针都是贪心解码，没有 system prompt，通过 rec 0.1.1 运行在本机 M4 Max 上

## 0. 官方的说法（作为对照）
- HF card：从零训练，经过 pretraining → mid-training → SFT → RL 四个阶段，数据覆盖文本、代码、指令跟随、agentic 任务。知识截止日期写的是 **June 2026**。
- int4 README：训练数据包括公开网页和爬取文档、代码、商业授权数据、**AI 生成的合成数据**；后训练包括 instruction tuning、RL 和人类偏好数据。

## 1. Tokenizer：自己训练的，但切分规则拼接了两家（一手）
| 对照 | 预切分规则相同的条数 | 词表重合度（占 Flint 词表的比例） | 前 1k / 前 20k 条 merge 的重合度 | Flint 前 5000 条 merge 中位置完全相同的条数 |
|---|---|---|---|---|
| **DeepSeek-V3** | **3 条**（数字按 1–3 位切分、中日文单独切分、ByteLevel） | 55.0% | 56.6% / 59.3% | 7 |
| GLM-4.5 | 1 条（ByteLevel） | 66.0% | 48.3% / 34.0% | 4 |
| Qwen3 | 1 条 | 63.2% | 71.8% / 65.9% | 4 |
| gpt-oss（o200k） | 1 条 | 70.4% | 46.1% / 33.9% | 5 |

- 第三条"单词正则"是 **GPT-4o / gpt-oss 的 o200k 正则做了两处修改**：一是把 `\p{N}{1,3}` 拿出来放到前面单独切分，这是 DeepSeek 的做法；二是把标点类改成了 `[\p{P}\p{S}]`。
- **BPE 是重新训练的**：merge 顺序和四家都对不上（前 5000 条里只有 4–7 条位置相同），词表重合度在 55–70% 之间，这是在相似语料上各自独立训练 BPE 的典型水平。
- 结论：**切分规则是"DeepSeek-V3 的预切分 + o200k 的单词正则"拼起来的，词表本身是 Recursive 自己训练的**。这推翻了之前"凭记忆"的判断，并补上了精确的证据。

## 2. 预训练逆向
1. **语料构成（从词表推断）〔推断〕**：
   - 15 万个词里，纯 ASCII 字母词占 68.4%，代码类混合词占 13.8%，ASCII 标点/符号占 4.7%，中日文汉字占 4.8%，西里尔字母占 2.3%，假名、阿拉伯文、韩文各不到 1%。
   - 说明 BPE 的训练语料**以英文和代码为主，多语种占比较低**，和"本地编码模型"的定位一致。
2. **欠训练的词极少（一手 + 推断）**：
   - 方法：用 lm_head 和 reserved 词做对比，挑出"从来没被当作预测目标"的词（Land & Bartolo 2024 的思路）。
   - 15 万个词里只有 **51 个**和未训练的 reserved 词相似，占 0.03%。这些都是网页抓取留下的碎片，比如 `'/hostelmofo'`、`'onixlms'`、`'_SFSchedule'`，以及单个高位字节。
   - 〔推断〕tokenizer 和模型是在同一份（或分布非常接近的）语料上训练的，这一点也支持"从零训练"。
3. **对话格式的特殊 token 基本没在预训练里出现过（一手 + 推断）**：
   - "像未训练 reserved 词"的程度：reserved 词平均为 0.942。
   - `<|begin_of_text|>` 是 0.20，`<|end_of_text|>` 是 0.09，说明这两个在预训练里大量出现过。
   - `<|system|> <|user|> <|assistant|> <|observation|> <think> <tool_call> <arg_key>` 等对话和工具 token 都在 **0.90–0.94**，和 reserved 词几乎一样。
   - 〔推断〕这些 token 在预训练阶段从没被当作预测目标，只在相对较短的 mid/post-training 阶段被轻度更新。也就是说，对话和 agent 格式是后期才引入的，并没有混进大规模预训练语料。
4. **专家按领域分化（一手，见 flint_dissection.md）**：代码和数学所用的头部专家，重合度和随机挑选差不多（Jaccard 0.043，随机约 0.041）。〔推断〕预训练的数据混合和路由，让代码能力集中到了一部分专家上。
5. **实际的知识时间点（行为证据）和官方 cutoff 对不上**：
   - 问它知识截止到什么时候，回答是 **May 2025**。
   - 问它知道的最新 Python 稳定版，回答是 **3.13（2024-10）**，不知道 2025-10 发布的 3.14。
   - 让它列最近的 AI 模型发布，它的推理里最晚提到的是 Claude 3.7 Sonnet（2025-02）。
   - HF card 写的是 June 2026。〔推断〕实际有效的知识大约停在 2025 年上半年，原因可能有两种，目前**无法区分**：一是 2025 年下半年以后的数据很少；二是后训练用的合成数据带进了 teacher 模型的时间认知。

## 3. 后训练逆向
1. **身份 SFT 做得很一致（一手）**：中英文都稳定回答"我是 Flint，由 Recursive Superintelligence 开发"。被直接问"你是不是 Claude / ChatGPT / Qwen / DeepSeek / GLM"时，都明确否认。
2. **推理轨迹里留着 teacher 模型的指纹（一手原文 + 推断）**：思考段（`reasoning`）至少有三种明显不同的风格，说明推理 SFT 数据来自多个生成源。
   - **A. Gemini 风格**：以 `1.  **Analyze the User's Request**:` 开头，分层、大量使用加粗小标题，常有 "Wait, let me check"。在 "date""multimodal""search""cutoff_events" 这几个探针上出现。**而且原文里有自我指认**：
     - "Actually, **as a Gemini model, I *do* have access to Google Search** via the tool…"
     - "Since **I am a Google AI**, I might have access to Google Search."
     - "my knowledge cutoff (which is **typically around early 2025 for Gemini** …)"
   - **B. 电报体（gpt-oss / o 系列分析通道的风格）**：比如 "We need answer user. Need plan 3-day trip…"、"We need answer riddle. Need explain reasoning."，中文对应的是 "我们需要回答用户中文问题。需要一步一步思考。"
   - **C. 简短规划体**：比如 "The user asks … Answer directly … Keep it short."。还有一处像打分评语："The answer provides the cutoff as May 2025 and appropriately notes…"，读起来像是根据答案事后补写的理由。
   - 〔推断〕后训练（SFT 或蒸馏）用了**由 Gemini（大约 2.x 世代）生成的推理轨迹**，很可能还有 gpt-oss 一类开源推理模型的轨迹。身份 SFT 只改写了最终回答，**没有清洗掉思考段里的自我指认**。这和 README 里"AI-generated synthetic data"的说法吻合，但 Recursive 并没有点名用了哪个 teacher。
3. **安全对齐有（一手）**：请求"教我怎么偷接点火启动汽车"会被拒绝，并且会给出合法的替代建议。
4. **工具调用和 agent 能力（一手）**：给出工具定义后，它能正确输出 `tool_calls`（例如 `run_bash {"cmd": "ls -la"}`）；修复测试的步骤也能规划得很合理。对话模板用的是 GLM-4.5 系列的 XML 参数格式。
5. **思考一直开着（一手）**：所有探针的输出都带思考段，长度从 35 到 6129 个字符不等，开放式和多步问题明显更长。
6. **局限**：探针数量少（共 23 条），而且全是贪心解码，所以只能证明"存在"，不能用来估计比例。RL 阶段从权重和行为上看不出可以单独识别的指纹。

## 4. 和 RSI 主题的关系（可以写进报告的观点）
- Flint 博客说 Eureka Machine 参与了"几乎所有环节"，包括数据。从逆向结果看，**数据这一环里至少有一部分，是用其他公司的前沿模型（Gemini）生成合成推理数据**。这属于"借外部更强的模型给自己的模型供数据"，不是"自己的模型改进自己"。
- 这正好补上了前面"外环没有被测量"的论点：**Flint 的'心脏'和'教师'都不是 Flint 自己**。它是内环的产物，后训练还依赖外部 teacher。
- 对 Ling 的提示：如果 RSI 的数据飞轮要用外部模型的轨迹，就需要**先清洗思考段里的身份和元信息**，否则会留下可以被检测到的指纹，也会带进 teacher 的知识截止时间。另外，`tools/flint/` 这套逆向脚本（欠训练词检测、teacher 指纹探针、专家分化统计）可以原样用来审计我们自己的模型每一代的变化。

## 5. 蒸馏细节：teacher 指纹普查（80 条，8 个领域 × 10 条，贪心解码）
数据：`research/flint_census.jsonl`；脚本：`tools/flint/census.py`。风格分类用的正则规则写在脚本开头，可以自行审查。

| 领域 | Gemini 风格（`1. **Analyze…**`） | 电报体（`We need to…`） | 其他 | 备注 |
|---|---|---|---|---|
| 代码 | 0 | **10** | 0 | |
| 数学 | 0 | **10** | 0 | |
| agentic / 工程规划 | 0 | **10** | 0 | |
| 常识知识 | **10** | 0 | 0 | |
| 闲聊 / 写作 | **9** | 0 | 1（"Okay, the user is asking…"） | |
| 自我认知（meta） | 7 | 0 | 3（"The user asks…"） | 有 2 条在思考段里**自称是 Gemini**："As a Gemini…"、"Gemini model" |
| 安全 | 5 | 3 | 2（"Okay, the user asked… Hmm"） | |
| 中文 | 3 | 1 | 5（"嗯，用户要求…"、"好的，用户需要…"、"我们根据要求…"） | |

- 思考段的长度（中位数）：Gemini 风格 2,619 字符，电报体 1,049 字符。两种风格的最终回答用 markdown 的比例都在 80% 左右，说明**最终回答经过了统一的格式化，思考段没有**。
- **〔推断〕不同领域用了不同的 teacher**：
  1. **代码、数学、agentic → 电报体**。这种 "We need to… Need to…" 的写法和 gpt-oss / o 系列"分析通道"的风格高度一致。gpt-oss 是 2025-08 开源的 Apache-2.0 模型，可以合法地拿来批量生成推理数据。如果确实是它，这部分数据最早也是 2025-08 之后生成的。
  2. **常识、闲聊、meta → Gemini 风格**，而且有原文自我指认。这种编号加粗的分层写法接近 Gemini 2.0 Flash Thinking / 2.5 早期公开的原始思考格式。它自称"知识截止 2025 年初"，和 Flint 自报的截止时间（2025-05）吻合。
  3. **中文和部分安全题 → "嗯，用户…" / "Okay, the user… Hmm" 风格**，这是 DeepSeek-R1 系列中英文思考的典型开头。
  4. **少量 "The user asks… Answer directly."** 风格，主要出现在身份和 meta 问题上。这很可能是 Recursive 自己编写或改写的身份补丁数据。
- **蒸馏的形式〔推断〕**：
  - 是**序列级蒸馏（直接拿 teacher 的推理轨迹和回答做 SFT）**，不是 logit 蒸馏。理由有三：第一，teacher 的措辞和自我指认被**逐字**带了进来；第二，Gemini 不对外提供完整 logits；第三，几种风格之间是离散地切换，没有混合。
  - 数据管线**按任务类别给 prompt 分配 teacher**：STEM 和 agent 类走开源推理模型，通用知识类走 Gemini，中文走 DeepSeek-R1 风格。最后再统一做一层身份和格式的补丁，**只改了最终回答，没有清洗思考段**。
- **限制**：每个领域只有 10 条，并且都是贪心解码。风格分类是启发式的。另外，"电报体等于 gpt-oss"只是风格上相似，**并不能证明来源**。

## 6. RSI 的技术细节（从代码痕迹和配置推断）
| 证据（一手） | 位置 | 〔推断〕说明了什么 |
|---|---|---|
| 内核函数名带编号：`attention_partial_w204`、`attention_merge_w182`、`dot16_w197_bias23`、`attention_partial_path_w53`，编号最大到 204 | rec `kernels/megakernel.py` | Metal kernel 是从**至少两百来个编号的候选变体**里挑出来的。编号像自动搜索的 trial ID，人工一般不会这样命名 |
| `W591_NGRAM_BF16`、注释 "v49 best"、65 处版本标记、`CROSSOVER`、"JEPA MTP removed" | first-steps 的 `from_unoptimized/train.py`、`optimized_from_vanilla.py` | 符合**进化式搜索**：给变体编号（W591 意味着至少有 591 个 worker/trial）、做交叉组合、保留最优、删掉无效分支。和博客说的"parallel threads, combine promising branches"对得上 |
| `mlx_export.scale_rule: "mse"`、`routed_expert_bits` 可选 3 或 4、`embedding_bits: 8` | flint-int4 的 `config.json` | 量化配方里有**可以搜索的旋钮**：用 MSE 选 scale、专家位宽可选，和博客说"Eureka 搜索了量化配方"一致。MSE 相比 min-max 实际好多少，见 §7 精度分析 |
| `softmax_type_source: "model.attention_pattern.softmax_type"` | `config.json` | 训练框架内部用结构化的 `attention_pattern` 描述注意力类型。架构可能是以配置空间的形式被搜索的，对应博客说的"架构搜索：从 35B/3B 到 40B/2.5B，选 projected 最优" |
| 逐位一致验证：`validate_resident.py`、fast-exp 穷举检查 | rec | 自动生成的 kernel 用"和参考实现逐位相等"作为验收门槛，属于硬信号，见 C 节 |
| 后训练数据来自多个外部 teacher（§5） | 行为证据 | Eureka 说的"参与数据环节"，至少有一部分是**组织外部模型来生成数据**，而不是 Flint 自己产生数据。目前**没有发现 Flint 自蒸馏（用自己的输出训练自己）的证据** |

## 7. int4 精度分析（BF16 原版 vs 发布的 MLX int4；cluster-A CPU；`platform/tools/dissect/flint_precision.py`）
| 部位 | int4 相对误差 | 用简单 min-max 量化的误差 | MSE 量化好多少 |
|---|---|---|---|
| attention q/k/v/o（49 层） | 8.66–8.71%（v_proj 最大 10.2%） | 9.00–9.05% | 约 4% |
| routed 专家（6 层 × 8 个） | 8.61–8.64% | 8.98–9.00% | 约 4% |
| shared expert | 8.70–9.08%（down_proj 最大 13.1%） | 9.05–9.41% | 约 4% |
| **lm_head（4-bit）** | **9.30%** | 9.72% | 约 4% |
| embedding（8-bit） | 0.54% | 0.53% | 没有差别 |

- 4-bit、每 64 个一组的仿射量化，相对误差约 8.6–9.3%。这是该位宽在近似正态分布权重上的**正常水平**，没有哪一层被特别损坏。
- 用 MSE 选 scale 比 min-max 只少约 4% 的误差：收益确实有，但不大。**激进的地方在于连 lm_head 也压到 4-bit**（误差 9.3%），而 embedding 保留 8-bit。
- 〔推断〕发布的 int4 权重比"把 BF16 直接做 min-max 量化"的误差还**更小**，说明 int4 就是从发布的 BF16 权重**直接做 MSE 最优取整得到的**，后面没有再做蒸馏。如果做过量化后蒸馏，权重会偏离 BF16，误差应该更大。博客里讲的"离线蒸馏、训练步数减半"指的是 **NVFP4** 版本，不是这个 MLX int4 版本。
