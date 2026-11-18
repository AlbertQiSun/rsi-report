# Flint 本地拆解（flint-int4 @ 3b93f90，M4 Max，2026-11-17）

- 脚本：`tools/flint/inventory.py`、`params.py`（只读 safetensors 头，不加载权重）；`router_stats.py`（借用 rec 自带的 MLX 运行时和 rec 的模型代码做前向，只读权重，Mac GPU 加锁使用）；`plot_routing.py`。
- 数据：`research/flint_router_stats.json`；图：`research/figs/flint_routing.png`。
- **探索性质**：每个领域只用了一段不超过 2048 token 的文本，只能看定性趋势。另外，transformers 加载 tokenizer 时提示正则可能有问题，会让 token 数略有偏差，但不影响定性结论。

## 1. 参数清单（由权重形状精确计算）

| 项 | 数值 | 对照 |
|---|---|---|
| 总参数 | **39.785 B** | HF card 写 39.8B ✅ |
| 每 token 激活参数 | **2.515 B**，不含 head 为 2.208 B | card 写约 2.5B，body 约 2.2B ✅ |
| 激活占比 | 6.32% | — |
| routed experts | 37.849 B，**占总参数 95.1%**（256 个专家 × 47 层） | — |
| 每 token 的激活参数构成 | attention 36.8% · routed（256 选 6）35.3% · lm_head 12.2% · shared expert 11.8% · dense MLP（第 0、1 层）3.0% · router 1.0% | 和 `rec_optimization_analysis.md` 里的估算一致 |
| 文件 | 1,751 个张量，22.57 GB；权重用 U8 打包，scale 和 bias 是 BF16，按 64 个一组 | 量化方式：主体 4-bit，embedding 8-bit，**lm_head 4-bit** ✅（和博客一致） |
| attention sink | 每层每个 head 一个 `softmax_offset`，共 32 个。全局层均值 3.63，滑窗层均值 4.04，范围 −9.0 到 12.4 | 和 card 的"learned per-head attention sinks"一致 |

## 2. MoE 路由（4 个领域：代码、中文、英文散文、数学）

- **负载均衡良好**：专家使用分布的归一化熵在 0.86–0.91（满分 1 表示完全均衡）。代码和中文文本几乎用到全部 256 个专家，平均覆盖 253 个；数学文本较重复，覆盖 226 个。
- **路由比较尖锐**：在选中的 6 个专家里，权重最大那个的平均占比约 0.26–0.31。如果 6 个平均分配，每个是 0.167。
- **领域专门化很明显**：比较每层使用最多的前 20 个专家（Jaccard，取 47 层的均值；两组随机 20 选的基线约为 0.041）。
  - 代码和数学：0.043，**和随机重叠差不多**，说明这两类内容走的是几乎不重叠的专家。
  - 英文散文和中文：0.192，约为随机基线的 **4.7 倍**，说明自然语言之间共用了相当一部分专家。
  - 代码和中文：0.079；代码和英文：0.087。
- **最后几层出现语言专属专家**：中文文本在第 46–48 层的熵从约 0.88 降到 0.73，前 10 名专家吃掉的份额从 0.2 左右升到 0.54。
- 解读（推断，未验证）：Flint 是专攻编码的模型，"代码专家"和"数学专家"几乎不重叠。这可能是它的 FrontierScience、HLE 这类科研和推理分数偏低的一个结构性因素：编码能力集中在一部分专家里。要验证这一点，需要做更系统的专家消融。

## 3. 对 RSI 报告和 Ling 的意义
- **激活参数的构成决定了推理优化的方向**：attention 和 routed 专家各占约 36%。所以 rec 选择把整个 decode 融合成一个 kernel，比单独压缩其中某一块更合理。
- **对"换心"实验的含义**：如果要对 Flint 做 LoRA，按专家分化的结构，应该优先在 attention 和 shared expert 上加。只改少数几个专家，只会影响部分领域的能力。
- **Ling-3.0-flash 也是 512 个专家选 8 个、外加 1 个共享专家的结构**，同一套脚本可以直接用来检查它的专家分化，作为 RSI 改进前后"能力分布"变化的观测工具。

## 4. 基模是什么：几乎可以确定是从零训练，没有基模
依据来自本地的 tokenizer.json、config.json、chat_template.jinja 和 README，都是一手文件。

- **词表是自己的**：150,000 个 BPE token 加 144 个特殊 token，共 150,144，其中 149,744 条 merge。微调必须沿用原模型的词表，而 Qwen（151,936）、GLM（151,552）、gpt-oss（201,088）都和它对不上。
- **维度组合在已知的开源模型里找不到**：hidden 2048、49 层、256 个专家（每个宽 512，每次选 6 个）加 1 个共享专家（宽 1024）。最接近的是 Qwen3-30B-A3B（hidden 2048、48 层），但它是 128 个专家选 8 个，词表也不一样。
- **和另外两家对照**：Apex 的 config 和 Qwen3.6-35B-A3B **逐字段一致**，属于继续训练；Naive 的结构直接继承自 MiMo-V2.5。Flint 是三家里唯一自有结构的，这和它"从零构建"的说法一致。
- **不过设计上几乎每一块都借鉴了现成的模型**：
  - 对话和工具调用格式来自 GLM-4.5 及以后版本：`<|observation|>`、`<tool_call>`、`<arg_key>/<arg_value>`。rec 的约束解码直接调用了 xgrammar 的 `glm_4_7` 模板。
  - 特殊 token 的命名：`<|begin_of_text|>` 沿用 Llama 3，`<|developer|>` 沿用 OpenAI harmony。
  - 分词规则：数字按 1–3 位切分、中日文单独切分，这一点和 DeepSeek-V3 相似（**凭记忆判断，需要核实**）；之后再套一层 GPT-4o（o200k）风格的正则。
  - 可学习的 attention sink 来自 gpt-oss。
  - 滑窗层和全局层混合、且全局层不加位置编码（NoPE），和 Llama 4 / Command-A 的做法相近。
  - sigmoid 路由，加上 top-k 归一化和共享专家，是 DeepSeek-V3 的做法。
  - QK 归一化来自 Qwen3 / OLMo 2。
- **一个保留**：README 写明训练数据里有 "AI-generated synthetic data"。所以"从零训练"并不排除从其他模型做数据蒸馏，只是用了哪个模型没有披露。
