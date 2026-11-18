# 开放权重仓库一手核查：config / 模型卡 / commit / GitHub

- **抓取日期**：2026-11-17
- **方法**：只读 HF API（`/api/models/<id>`、`/tree/main`、`/commits/main`）、小型元数据文件（config.json、README.md、generation_config.json、chat_template.jinja、tokenizer_config.json、model.safetensors.index.json、configuration_*.py / modeling_*.py）、GitHub API（repo、commits、git tree、languages）与少量 raw README。**未下载任何权重分片**。
- **范围变更说明**：中途按用户要求改为"只检索与阅读"。本文件**只列字段原文与卡片原句**，并逐项并排标出与博客说法"一致 / 不一致 / 不可见"；**不做推算、不给比例公式、不下结论**。表中"状态"列不再出现"推算"。
- **状态标记**：`一手配置` = 仓库文件或 HF/GitHub API 字段原值；`自报` = 模型卡 / README / 博客中的文字声明。
- **原始数据存放**：`other/hf/<org__model>/`（`api.json`、`tree.json`、`commits.json`、`files/…`）；`other/gh/<org__repo>/`。
- **URL 简写**：`HF(<id>)` = https://huggingface.co/<id> ；`API(<id>)` = https://huggingface.co/api/models/<id> ；`GH(<repo>)` = https://github.com/<repo>
- **博客侧说法**取自 `sources/naive_research.md`、`sources/recursive_flint.md`、`sources/apex_35b.md` 的引证台账（ID 如 NV-26）。

---

## 0. 仓库 ID 核对

| 用户提法 | 实际 repo id | 备注 |
|---|---|---|
| Naive-N0.5-Flash | `NaiveAI/Naive-N0.5-Flash` | 公开，未 gated |
| "W8A8 variant" | `NaiveAI/Naive-N0.5-Flash-FP8` | HF 上**没有**名为 W8A8 的 repo；该 repo 名为 FP8，`quantization_config.quant_method="fp8"`、`activation_scheme="dynamic"`。NaiveRT README 称其所需 target 为 "a supported W8A8 target checkpoint" 并链接到此 repo |
| "DFlash draft model" | `NaiveAI/Naive-N0.5-Flash-FP8-Draft` | config `architectures=["DSparkDraftModel"]`；仓库同时含 `dflash.py` 与 `dspark.py` |
| MiMo-V2.5 | `XiaomiMiMo/MiMo-V2.5`（另有 `-Base`、`-DFlash`、`-Pro` 等） | Naive README 正文写 "MiMo-V2.5 base model"，文末 `[base-model]` 链接指向 `XiaomiMiMo/MiMo-V2.5-Base` |
| flint | `RecursiveSI/flint`（另有 `flint-int4`、`flint-int4-gguf`、`flint-nvfp4-gguf`） | |
| Apex-35B-V1.0 | `Apexintelligence-AI/Apex-35B-V1.0`（另有 `-GGUF`） | |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | |
| Ling-3.0-flash | `inclusionAI/Ling-3.0-flash` | 同 org 另有 -base、-fp8/int4/fp4、-VL、-Fin、-Sante、Ling-3.1-flash 等 |

所有目标 repo 均为公开（`gated: false`），无缺失。

---

## 1. NaiveAI/Naive-N0.5-Flash（含 FP8 与 Draft）

### 1.1 仓库元数据

| 字段 | 值 | 来源 | 状态 |
|---|---|---|---|
| createdAt | 2026-09-27T14:14:32Z | API(NaiveAI/Naive-N0.5-Flash) | 一手配置 |
| lastModified | 2026-09-27T15:16:00Z | 同上 | 一手配置 |
| commits | 1 条："initial commit"，2026-09-27T15:16:00Z | `/commits/main` | 一手配置 |
| license（卡片 YAML） | mit；仓库含 LICENSE 文件（1,065 B） | README.md YAML | 一手配置 |
| 仓库总大小 | usedStorage 617,823,283,433 B；tree 文件合计 617,834,899,550 B（≈617.8 GB），49 个 safetensors 分片 | API / tree | 一手配置 |
| safetensors 参数统计 | **API 未返回**（`safetensors: null`）；仓库**无** `model.safetensors.index.json` | API / tree | 一手配置 |
| config `dtype` | bfloat16 | config.json | 一手配置 |
| 文件 | config.json、configuration_naive_n05_flash.py、modeling_naive_n05_flash.py、chat_template.jinja、tokenizer 文件、vocab.README.md（原文："Tokenizer copied from the open-source MiMo 2.5"）、assets/ | tree | 一手配置 |
| 是否含 MTP 权重 | BF16 分片 header 中的张量名里无 `mtp` 字样（范围变更前已读取的 header 名称列表） | safetensors header（HTTP range 读取） | 一手配置 |
| 下载 / likes | 15,677 / 412（抓取当日） | API | 一手配置 |

| repo | createdAt | lastModified | commits | 大小 | license | 备注 |
|---|---|---|---|---|---|---|
| Naive-N0.5-Flash-FP8 | 2026-09-27T15:24:21Z | 2026-09-27T15:16:00Z（早于 createdAt） | 1（"initial commit"，15:16:00Z） | usedStorage 315,106,551,969 B；49 分片；无 index.json | mit | 除 config.json 外，所有非权重文件哈希与 BF16 repo 相同（含 README） |
| Naive-N0.5-Flash-FP8-Draft | 2026-09-27T15:40:12Z | 2026-09-27T19:14:26Z | 4（15:40:12 → 19:14:26） | 1,306,512,347 B；单个 model.safetensors；API 参数 652,797,441（BF16） | mit | README 与主模型 README 字节相同（无 Draft 专属说明） |

### 1.2 config.json 关键字段（BF16 repo）

| 字段 | 值 | 状态 |
|---|---|---|
| architectures / model_type | `NaiveN05FlashForCausalLM` / `naive_n05_flash`（`auto_map` 指向仓库内自定义代码） | 一手配置 |
| num_hidden_layers | 48 | 一手配置 |
| hidden_size | 4096 | 一手配置 |
| hybrid_layer_pattern（48 项） | `0 1111 0 11111 0 11111 0 11111 0 11111 0 11111 0 11111 0 11111 0`；值为 0 的层：0,5,11,17,23,29,35,41,47；值为 1 的层共 39 个 | 一手配置 |
| 0/1 含义 | modeling 代码：`is_swa = bool(hybrid_layer_pattern[i])`；非 SWA 层构造 `NaiveN05FlashIndexer`；configuration 中 layer_types 映射为 `"sliding_attention"` / `"deepseek_sparse_attention"` | 一手配置（代码） |
| 带 indexer 的层（header 张量名 `self_attn.indexer.wq.weight`） | 0,5,11,17,23,29,35,41,47 | 一手配置 |
| DSA（非 SWA）层注意力 | num_attention_heads 64；num_key_value_heads 4；head_dim 192；v_head_dim 128；rope_theta 10,000,000 | 一手配置 |
| SWA 层注意力 | swa_num_attention_heads 64；swa_num_key_value_heads 8；swa_head_dim 192；swa_v_head_dim 128；swa_rope_theta 10000 | 一手配置 |
| sliding_window / sliding_window_size / attention_chunk_size | 128 / 128 / 128 | 一手配置 |
| sink bias 开关 | `add_swa_attention_sink_bias: true`；`add_full_attention_sink_bias: false` | 一手配置 |
| 带 `attention_sink_bias` 张量的层（header） | 1–4, 6–10, 12–16, 18–22, 24–28, 30–34, 36–40, 42–46（即全部 SWA 层；0,5,11,…,47 无） | 一手配置 |
| attention_projection_layout | `"split"`（validate() 要求 split） | 一手配置 |
| attention_value_scale | 0.707 | 一手配置 |
| partial_rotary_factor | 0.334 | 一手配置 |
| DSA / indexer | enable_dsa true；index_top_k 2048；index_n_heads 16；index_head_dim 128；index_n_kv_heads 1；indexer_activation_dtype "fp8_e4m3" | 一手配置 |
| MoE | n_routed_experts 256；num_experts_per_tok 8；n_shared_experts null（代码注释："DeepSeek-V3-style routed experts without a shared expert"）；moe_intermediate_size 2048；scoring_func sigmoid；topk_method noaux_tc；n_group 1 / topk_group 1；norm_topk_prob true；routed_scaling_factor null | 一手配置 |
| moe_layer_freq | 第 0 层为 0（dense MLP，intermediate_size 16384），1–47 层为 1 | 一手配置 |
| vocab_size | 152,576 | 一手配置 |
| max_position_embeddings | 1,048,576 | 一手配置 |
| rope_scaling | `{"rope_type":"default"}` | 一手配置 |
| tie_word_embeddings | false | 一手配置 |
| transformers_version | 4.57.1（README 安装要求 `transformers>=5.17.0`） | 一手配置 / 自报 |
| generation_config | temperature 1.0，top_p 0.95，max_new_tokens 131072 | 一手配置 |

**FP8 repo config 差异**：仅多出 `quantization_config`：`quant_method "fp8"`、`fmt "e4m3"`、`activation_scheme "dynamic"`、`weight_block_size [128,128]`；`ignored_layers` = 全部 48 层的 `self_attn`、`model.layers.0.mlp.{down,gate,up}_proj`、`lm_head`。（对照：MiMo-V2.5 的 `ignored_layers` 为各层 `self_attn.o_proj` 等 49 项。）

**Draft repo config**：`DSparkDraftModel`（`model_type "qwen3"`）；num_hidden_layers 5（均 sliding_attention，sliding_window 1024）；hidden 4096；heads 32 / kv 4 / head_dim 128；intermediate 4096；block_size 7；`dflash_config.target_layer_ids [1,7,14,20,26,32,39,45]`；`enable_confidence_head true`；`markov_head_type "vanilla"`，`markov_rank 256`；vocab 152,576；max_position 1,048,576。`dspark.py` 文档串原文称其为 "DFlash backbone + EAGLE-style Markov and confidence heads"，并称 Markov/confidence 部分 "ported to match DeepSeek's DeepSpec"；代码 `from specforge.modeling.draft.dflash import DFlashDraftModel`。

### 1.3 README（模型卡）声明

| 声明 | 原文要点 | 状态 |
|---|---|---|
| 规模 | "309B MoE model with 15.5B active parameters" | 自报 |
| 上下文 | "native 1M-token context window" | 自报 |
| 层构成 | "Transformer layers 48"；"39 SWA layers + 9 DSA layers" | 自报 |
| 模块 | "eight six-layer modules … five SWA layers followed by one DSA layer, with the first layer of the first module also replaced by DSA" | 自报 |
| SWA / DSA | "128-token window"；"top 2,048 tokens"；"DSA KV groups 4 (GQA4)"；"Indexer query heads 16" | 自报 |
| sink | "Both attention types incorporate sink bias." | 自报 |
| "no full attention" | "with no full-attention layers" | 自报 |
| 训练 | 3.25T tokens = 50B Indexer Warmup + 3T Sparse Attention Training + 200B LR Decay | 自报 |
| 基座 | "builds on the open-weight MiMo-V2.5 base model"；链接 `XiaomiMiMo/MiMo-V2.5-Base` | 自报 |
| 部署 | "requires FP8-capable NVIDIA GPUs"；"weights occupy approximately 315 GB"；示例代码加载 `NaiveAI/Naive-N0.5-Flash-FP8` | 自报 |
| 评测 harness | "Claude Code 2.1.207 with a 1M-token context window, temperature 1.0, and top-p 0.95" | 自报 |
| GitHub 链接 | `[github]: https://github.com/NaiveAI-Labs/Naive-N0.5-Flash` | 一手配置 |

---

## 2. XiaomiMiMo/MiMo-V2.5（Naive 的基座）

| 字段 | 值 | 来源 | 状态 |
|---|---|---|---|
| createdAt / lastModified | 2026-04-27T13:37:38Z / 2026-10-28T14:13:36Z | API | 一手配置 |
| commits | 19；首条 2026-04-27 "initial commit"；末条 2026-10-28 "Update README.md"；2026-04-28 有 "Update MiMo-V2.5 config" | commits | 一手配置 |
| license | mit（YAML）；tree 中无 LICENSE 文件 | README / tree | 一手配置 |
| safetensors（API） | total 310,756,322,688；F8_E4M3 306,655,002,624 / BF16 4,052,024,960 / F32 49,295,104 | API | 一手配置 |
| 仓库大小 | usedStorage 315,693,598,475 B；18 个权重文件（含 `model_mtp.safetensors`、`audio_tokenizer/model.safetensors`） | API / tree | 一手配置 |
| config 文本骨干 | 与 §1.2 表中 Naive 各字段相同（层数 48、hidden 4096、hybrid_layer_pattern 完全相同、heads 64 / kv 4 / head_dim 192 / v 128、SWA heads 64 / kv 8、window 128、sink：swa true / full false、256 experts top-8、n_shared null、moe_layer_freq 相同、vocab 152,576、max_position 1,048,576、rope_theta 1e7、swa_rope_theta 1e4、partial_rotary 0.334、value_scale 0.707） | config.json | 一手配置 |
| config 中 Naive 没有的 | `audio_config`、`vision_config`、`processor_config`、image/video token id、`quantization_config`（fp8，block 128×128）；`attention_projection_layout "fused_qkv"` | config.json | 一手配置 |
| config 中无 DSA 字段 | 无 `enable_dsa` / `index_*` 字段 | config.json | 一手配置 |
| README 规格 | "310B total / 15B activated"；"48 (1 dense + 47 MoE)"；"Full Attention Layers 9"；"SWA Layers 39"；"SWA Window Size 128"；"Max Context Length 1M"；"MTP Layers 3"；"interleaving SWA and Global Attention (GA) with a 5:1 ratio and 128 sliding window"；"learnable attention sink bias"；"~48T tokens" | README.md | 自报 |
| README KV heads 行 | "Num KV Heads: 8 (GA) / 4 (SWA)" | README.md | 自报 |
| README 关于 config 更新 | 称 config.json 与 tokenizer_config.json 自初版后已更新（commit 4da2748） | README.md | 自报 |

**MiMo-V2.5-Base 与 MiMo-V2.5 config 差异**（原文字段）：Base `max_position_embeddings 262144`、`rope_theta 5000000`；MiMo-V2.5 为 `1048576`、`10000000`（另有 video 预处理两项不同）。Naive 的这两项为 `1048576` / `10000000`。

**MiMo-V2.5-DFlash**：createdAt 2026-07-03，4 commits；`dflash/config.json`：`DFlashDraftModel`，5 层，heads 64 / kv 8，block_size 8，`target_layer_ids [0,11,23,35,47]`，SWA 1024，`attention_sink_bias true`。README 只有使用限制条款。

---

## 3. RecursiveSI/flint

| 字段 | 值 | 来源 | 状态 |
|---|---|---|---|
| createdAt / lastModified | 2026-10-21T14:20:58Z / 2026-10-21T14:24:03Z | API(RecursiveSI/flint) | 一手配置 |
| commits | 1："Initial release"，2026-10-21T14:24:03Z | commits | 一手配置 |
| license | apache-2.0（YAML）；含 LICENSE（11,356 B） | README / tree | 一手配置 |
| safetensors（API） | 39,785,287,456（BF16 only） | API | 一手配置 |
| 仓库大小 | usedStorage 79,587,116,933 B；15 个分片；index `total_size` 79,570,574,912 | API / index | 一手配置 |
| 其他文件 | `sglang/`（Dockerfile："SGLang v0.5.16 + the flint_sglang model package + the FA4 learned-sink fix"、`flint_sglang-0.1.4` wheel、`fa4-learned-sink-fix.patch`、serve_b300/h200/node_8x 脚本） | tree / Dockerfile | 一手配置 |
| 无 transformers modeling 代码 | config 无 `auto_map`；README："Transformers modeling code is not included yet." | config / README | 一手配置 / 自报 |
| architectures / model_type | `FlintForCausalLM` / `flint` | config | 一手配置 |
| num_hidden_layers | 49 | config | 一手配置 |
| mlp_only_layers | [0, 1] | config | 一手配置 |
| layer_types | full_attention 14 个（层 0,1,5,9,13,17,21,25,29,33,37,41,45,48）；sliding_attention 35 个 | config | 一手配置 |
| no_rope_layer_indices | [0,1,5,9,13,17,21,25,29,33,37,41,45,48]（与 full_attention 位置相同） | config | 一手配置 |
| sliding_window | 513；use_sliding_window true | config | 一手配置 |
| 注意力 | num_attention_heads 32；num_key_value_heads 4；head_dim 128；softmax_type "learnable" | config | 一手配置 |
| hidden / intermediate | 2048 / 6144 | config | 一手配置 |
| MoE | num_experts 256；num_experts_per_tok 6；moe_intermediate_size 512；shared_expert_intermediate_size 1024；router_score_function sigmoid；norm_topk_prob true；router_aux_loss_coef 0.0005；decoder_sparse_step 1 | config | 一手配置 |
| vocab_size | 150,144 | config | 一手配置 |
| max_position_embeddings | 262,144 | config | 一手配置 |
| rope_theta | 10000.0 | config | 一手配置 |
| tie_word_embeddings | false | config | 一手配置 |
| transformers_version | 5.12.1 | config | 一手配置 |
| generation_config | temperature 1.0，top_p 0.95，top_k 20；eos [150001,150003,150005,150018] | generation_config.json | 一手配置 |
| 卡片规模 | "39.8 billion total parameters and approximately 2.5 billion active per token"；"The active body alone is approximately 2.2B parameters."；"“40B-A2.5B” is the rounded model-size designation." | README | 自报 |
| 卡片训练 | "trained from scratch through pretraining, mid-training, supervised fine-tuning, and reinforcement learning"（训练段未提 Eureka Machine） | README | 自报 |
| 卡片评测 harness | "Terminal-Bench uses Terminus 2; SWE-bench uses mini-swe-agent" | README | 自报 |

---

## 4. Apexintelligence-AI/Apex-35B-V1.0

| 字段 | 值 | 来源 | 状态 |
|---|---|---|---|
| createdAt / lastModified | 2026-10-28T09:58:45Z / 2026-10-29T20:35:43Z | API | 一手配置 |
| commits | 4：2026-10-28T10:01:56Z "Apex-35B-V1.0" → 2026-10-29T20:35:43Z（后 3 条均为 README/GGUF 链接修改） | commits | 一手配置 |
| license | apache-2.0；LICENSE 文件哈希与 Qwen3.6-35B-A3B 的 LICENSE **相同** | README / tree | 一手配置 |
| 卡片 YAML | `base_model: Qwen/Qwen3.6-35B-A3B`；`base_model_relation: finetune` | README | 一手配置 |
| safetensors（API） | 35,951,822,704（BF16） | API | 一手配置 |
| 仓库大小 | usedStorage 71,917,241,235 B；26 个分片；index `total_size` 71,903,645,408 | API / index | 一手配置 |
| 卡片规格 | "35B total parameters with approximately 3B activated per token"；"262,144 tokens shared by input and output"；Input / Output "Text / Text"；Weight Precision BF16 | README | 自报 |
| 卡片训练描述 | "further trained on proprietary datasets and research trajectories"；"recursive self-improvement (RSI) through Apex Research" | README | 自报 |
| 卡片 harness | "supported by the Claude Code (v2.1.280 preferred) harness" | README | 自报 |
| chat template 提示 | "do not replace it with the Qwen3.6 template: Apex-35B-V1.0 expects the generation prompt to end with `<think>`" | README | 自报 |
| 处理器文件 | 含 `preprocessor_config.json`、`processor_config.json`；README："`pillow` and `torchvision` are needed to load the processor files shipped with the checkpoint" | tree / README | 一手配置 / 自报 |

## 5. Qwen/Qwen3.6-35B-A3B（Apex 的基座）

| 字段 | 值 | 来源 | 状态 |
|---|---|---|---|
| createdAt / lastModified | 2026-04-15T05:59:19Z / 2026-11-07T04:41:52Z | API | 一手配置 |
| commits | 4：2026-04-15 "Initial public release"；04-22 LICENSE；04-24 README；**2026-11-07 "Add a response template and generation tags (#81)"**（晚于 Apex 发布） | commits | 一手配置 |
| license | apache-2.0 | README | 一手配置 |
| safetensors（API） | 35,951,822,704（BF16） | API | 一手配置 |
| 仓库大小 | usedStorage 71,921,347,477 B；26 分片；index total_size 71,903,645,408 | API / index | 一手配置 |
| 顶层 config | `architectures` 多模态（含 `vision_config`：depth 27，hidden 1152） | config | 一手配置 |
| text_config | layers 40；hidden 2048；layer_types 每 4 层 1 个 full_attention（层 3,7,…,39），其余 `linear_attention`；full_attention_interval 4；attn_output_gate true | config | 一手配置 |
| 注意力 | num_attention_heads 16；num_key_value_heads 2；head_dim 256；linear：key heads 16 / value heads 32 / key & value head_dim 128 / conv kernel 4 | config | 一手配置 |
| MoE | num_experts 256；num_experts_per_tok 8；moe_intermediate_size 512；shared_expert_intermediate_size 512；router_aux_loss_coef 0.001 | config | 一手配置 |
| 其他 | vocab_size 248,320；max_position_embeddings 262,144；rope_parameters：mrope_interleaved，mrope_section [11,11,10]，partial_rotary_factor 0.25，rope_theta 10,000,000；mtp_num_hidden_layers 1 | config | 一手配置 |
| 卡片规格 | "35B in total and 3B activated"；"Hidden Layout: 10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))"；"8 Routed + 1 Shared"；"262,144 natively and extensible up to 1,010,000 tokens"；"Causal Language Model with Vision Encoder" | README | 自报 |

---

## 6. inclusionAI/Ling-3.0-flash

| 字段 | 值 | 来源 | 状态 |
|---|---|---|---|
| createdAt / lastModified | 2026-08-02T16:14:41Z / 2026-10-12T11:46:10Z | API | 一手配置 |
| commits | 31；首条 2026-08-02 "initial commit"；末条 2026-10-12 "Fix MTP cache overflow during generation (#19)"；2026-08-05 有 "Update config.json (#2)" | commits | 一手配置 |
| license | mit（YAML）；tree 中**无 LICENSE 文件** | README / tree | 一手配置 |
| safetensors（API） | total 127,486,405,600；BF16 127,486,240,128 / F32 165,472 | API | 一手配置 |
| 仓库大小 | usedStorage 254,993,303,374 B；index total_size 254,973,142,144 | API / index | 一手配置 |
| 张量布局 | index 中 `model.layers.42.*` 含 `eh_proj` / `enorm` / `hnorm`（MTP 结构）及完整 MoE experts；config `num_nextn_predict_layers 1` | index / config | 一手配置 |
| architectures / model_type | `BailingMoeV3ForCausalLM` / `bailing_hybrid`（自定义代码 `modeling_bailing_moe_v3.py`） | config | 一手配置 |
| num_hidden_layers / hidden | 42 / 2560 | config | 一手配置 |
| 层类型规则 | config `layer_group_size 6`；modeling：`(layer_idx + 1) % layer_group_size == 0` 或尾部余数层 → `BailingMoeV3MultiLatentAttention`，否则 `BailingMoeV3KimiDeltaAttention` | config / modeling | 一手配置 |
| MLA 字段 | kv_lora_rank 512；q_lora_rank null；qk_nope_head_dim 128；qk_rope_head_dim 64；v_head_dim 128；num_attention_heads 32；num_key_value_heads 32；gated_attention_proj_granularity_type "head_wise"；use_qk_norm true | config | 一手配置 |
| KDA 字段 | kda_safe_gate true；kda_lower_bound -5.0；short_conv_kernel_size 4；no_kda_lora true；num_kv_heads_for_linear_attn 0 | config | 一手配置 |
| MoE | num_experts 512；num_experts_per_tok 8；num_shared_experts 1；moe_intermediate_size 768；moe_shared_expert_intermediate_size 768；first_k_dense_replace 2（dense intermediate 6144）；n_group 8 / topk_group 4；routed_scaling_factor 2.5；score_function sigmoid；topk_method noaux_tc | config | 一手配置 |
| vocab / 位置 | vocab_size 157,184；max_position_embeddings 262,144；rope_theta 6,000,000；partial_rotary_factor 0.5；rope_scaling null | config | 一手配置 |
| 卡片规模 | "**124B** total and **5.1B** active parameters" | README | 自报 |
| 卡片架构 | "native hybrid linear attention architecture … (5:1 alternating stacking of Kimi Delta Attention (KDA) and MLA) … 1/64 sparse MoE"；表格："35 KDA + 7 Gated MLA (5:1)"、"Number of Dense Layers 2"、512 routed / 1 shared / 8 activated、"Context Training Schedule 8K -> 32K -> 256K" | README | 自报 |
| 卡片提到的框架 / 评审模型 | Claude Code、Kilo Code、Qwen Code、Hermes Agent、OpenClaw（使用体验）；OpenHands（SWE-Bench harness）；GPT-5.4-mini、Gemini-2.5-Pro、GPT-4.1、Claude Opus 4.6（作为评测 judge / 模拟器） | README | 自报 |

---

## 7. 对比 1：Apex-35B-V1.0 vs Qwen3.6-35B-A3B

| 项 | Qwen3.6-35B-A3B | Apex-35B-V1.0 | 并排观察 |
|---|---|---|---|
| config.json（排序后 diff） | — | — | 仅 2 处不同：`transformers_version` 4.57.1 → 5.8.1；`vision_config.model_type` "qwen3_5_moe" → "qwen3_5_moe_vision"。所有 text_config 字段（层数、layer_types、heads、experts、vocab、rope、mtp）原值相同 |
| safetensors 参数（API） | 35,951,822,704 BF16 | 35,951,822,704 BF16 | 数值相同 |
| index total_size | 71,903,645,408 | 71,903,645,408 | 相同 |
| index 张量名列表 | 1,045 个 | 1,045 个 | 名称列表逐行相同（含 19 个 `mtp` 张量、333 个 `visual` 张量） |
| 权重分片哈希 | — | — | 26 个分片 LFS oid **全部不同** |
| tokenizer.json | — | — | 文件哈希**相同** |
| merges.txt / vocab.json | 有 | 无 | Apex 只带 tokenizer.json |
| tokenizer_config.json | `tokenizer_class "Qwen2Tokenizer"` | `"TokenizersBackend"` | 另有内嵌 chat_template 的差异（见下） |
| chat_template.jinja vs Qwen 2026-04-24 版 | `{{- '<think>\n' }}` | `{{- '<think>' }}` | 全文件唯一差异在生成提示末尾（默认开启 thinking 时） |
| chat_template.jinja vs Qwen 当前版（2026-11-07 #81 之后） | 含 `{%- generation %}` / `{%- endgeneration %}` 标签 | 无 | 该差异来自 Qwen 在 Apex 发布后的更新 |
| generation_config | temperature 1.0 / top_p 0.95 / top_k 20 / eos [248046,248044] | 相同取值，另加 `transformers_version`、`trust_remote_code: true` | |
| 处理器文件 | preprocessor_config、video_preprocessor_config、configuration.json | preprocessor_config、processor_config | Apex 卡片写 Input/Output "Text / Text"，但 config 保留 `vision_config`，index 保留 visual 张量 |

架构字段：config 原值除上述两项外完全一致。是否属于"纯继续训练"，仓库文件本身没有说明；卡片原文为 "further trained on proprietary datasets and research trajectories"，YAML 为 `base_model_relation: finetune`；博客台账（AX）原文为 "mid-training and post-training"。

---

## 8. 对比 2：Naive-N0.5-Flash vs MiMo-V2.5 —— 博客说法逐项核对

| 博客 / 卡片说法 | Naive config / 代码 / header 原值 | MiMo-V2.5 原值 | 核对结果 |
|---|---|---|---|
| 8 个六层模块，共 48 层（NV-26） | num_hidden_layers 48；hybrid_layer_pattern = `0 1111 0` + 7×`11111 0` | 相同（48，pattern 相同） | **一致**（"模块"概念本身在 config 中不可见，只有逐层 pattern） |
| 5:1 SWA:DSA，"39 SWA + 9 DSA" | pattern 中 0（DSA）在 0,5,11,17,23,29,35,41,47 共 9 个；1（SWA）39 个 | 同一 pattern（MiMo 中 0 = 全局注意力） | **一致**（与卡片 39+9 一致） |
| "with no full-attention layers" | 非 SWA 层均构造 indexer 并按 top-k 掩码；`enable_dsa true` | MiMo 的 0 层为全局注意力（README："Full Attention Layers 9"） | **一致**（代码层面） |
| SWA window 128（NV-28） | sliding_window 128 | 128 | **一致**（与 MiMo 相同，未改动） |
| DSA top-k 2048（NV-29） | index_top_k 2048 | 无此字段 | **一致** |
| indexer 16 query heads（NV-30） | index_n_heads 16；index_head_dim 128；index_n_kv_heads 1；indexer_activation_dtype fp8_e4m3 | 无 | **一致** |
| GQA 4 KV groups（NV-07）"replaces MLA with GQA" | DSA 层 num_key_value_heads 4（64 Q heads）；SWA 层 swa_num_key_value_heads 8 | MiMo 全局层 num_key_value_heads 4，SWA 层 8；MiMo config 中无 MLA 字段 | DSA 层 4 KV：**一致**。"replaces MLA"：卡片原文是相对 "the original MLA-based DSA implementation"；MiMo-V2.5 的 config 本身就是 GQA（kv 4），无 MLA 字段可替换 |
| sink bias："Both attention types incorporate sink bias" | `add_swa_attention_sink_bias true`，`add_full_attention_sink_bias false`；代码只在 flag 为 true 时创建 `attention_sink_bias`；header 中仅 39 个 SWA 层有该张量，DSA 层（0,5,…,47）无 | 同样 swa true / full false | SWA 层：**一致**；DSA 层：**与 config 及权重张量名不一致** |
| 1M 原生上下文（NV-04） | max_position_embeddings 1,048,576 | MiMo-V2.5 同为 1,048,576（MiMo-V2.5-Base 为 262,144） | config 值一致；该值与 MiMo-V2.5 相同 |
| 309B / 15.5B（NV-02/03） | API 无 safetensors 统计；repo 无 index.json | MiMo README "310B total / 15B activated"；API total 310,756,322,688（含视觉/音频/MTP 文件） | 不可直接从 API 读取（本文件不做推算） |
| tokenizer | vocab.json 哈希与 MiMo **相同**；vocab.README："Tokenizer copied from the open-source MiMo 2.5"；merges.txt / tokenizer.json 与 MiMo 大小略不同（1,671,853 vs 1,671,839；7,032,660 vs 7,033,572） | | vocab 一致 |
| 投影布局 | `attention_projection_layout "split"` | `"fused_qkv"` | 不同 |
| 多模态 | 无 vision/audio 配置 | 有 | 不同 |
| MTP | BF16 header 中无 mtp 张量 | 有 `model_mtp.safetensors`；README "MTP Layers 3" | 不同 |
| 其余 MoE / 维度字段 | 256 experts、top-8、无 shared、moe_intermediate 2048、dense 层 0、vocab 152,576、rope_theta 1e7、partial_rotary 0.334、value_scale 0.707 | 相同 | 相同 |

附：MiMo-V2.5 README 表格写 "Num KV Heads 8 (GA) / 4 (SWA)"，而 MiMo config 原值为 `num_key_value_heads 4`（全局层）、`swa_num_key_value_heads 8`（SWA 层）——MiMo 自家 README 与 config 字段标注相反。

---

## 9. 对比 3：Flint config vs HF 卡片规格表

| 卡片规格 | config 原值 | 核对 |
|---|---|---|
| 49 layers: 2 dense + 47 MoE | num_hidden_layers 49；mlp_only_layers [0,1] | 一致 |
| 35 sliding-window + 14 full-attention | layer_types：sliding 35、full 14 | 一致 |
| Sliding window 513 | sliding_window 513 | 一致 |
| 32 query / 4 KV heads；head dim 128 | 32 / 4 / 128 | 一致 |
| 256 routed experts，6 selected，+ always-active shared expert | num_experts 256；num_experts_per_tok 6；shared_expert_intermediate_size 1024 | 256 / 6 一致；shared expert 只以 `shared_expert_intermediate_size` 字段体现，无显式数量字段 |
| RoPE on sliding layers；no PE on full layers | no_rope_layer_indices 与 full_attention 层位置逐一相同 | 一致 |
| learned per-head attention sinks | softmax_type "learnable" | 字段名可对应；"per-head" 在 config 中不可见 |
| Per-head QK RMSNorm | config 无对应字段 | 不可见 |
| Vocabulary 150,144 | vocab_size 150,144 | 一致 |
| 262,144 context | max_position_embeddings 262,144 | 一致（博客 FL-16 写 "up to 512k tokens"，推荐 256k） |
| 39.8B total | API safetensors 39,785,287,456 | 一致（HF 平台数字） |

---

## 10. 对比 4：Ling-3.0-flash vs 用户假设

| 用户假设 | 仓库所见 | 核对 |
|---|---|---|
| 124B total | 卡片 "124B total"；HF API safetensors total 127,486,405,600（checkpoint 含 `model.layers.42` MTP 层，带完整 MoE experts） | 卡片一致；API 数字与卡片数字不同，原因仓库未说明 |
| ~5.1B active | 卡片 "5.1B active" | 一致（自报） |
| hybrid linear attention | 卡片 "native hybrid linear attention … 5:1 … KDA and MLA"、"35 KDA + 7 Gated MLA"；config/代码：42 层，`layer_group_size 6`，每组第 6 层 MLA，其余 KimiDeltaAttention | 一致 |
| sparse MoE | 512 experts / top-8 / 1 shared；卡片 "1/64 sparse MoE" | 一致 |
| MIT license | 卡片 YAML `license: mit`；仓库无 LICENSE 文件 | YAML 一致；无 LICENSE 文本 |

---

## 11. 对比 5：激活参数（仅并排列出原文数字）

按范围变更，**不做 active/total 比例推算**，仅列各方原文：

| 模型 | 总参数（原文） | 激活参数（原文） | 来源 | 其他原文 |
|---|---|---|---|---|
| Naive-N0.5-Flash | 309B | 15.5B | README / 博客 | API 无 safetensors 统计 |
| MiMo-V2.5 | 310B | 15B | MiMo README | API total 310,756,322,688 |
| Flint | 39.8B（博客 40B） | ~2.5B（含 output head）；active body ~2.2B | HF 卡片 / 博客 | API 39,785,287,456 |
| Apex-35B-V1.0 | 35B | ~3B | HF 卡片 | API 35,951,822,704 |
| Qwen3.6-35B-A3B | 35B | 3B | Qwen 卡片 | API 35,951,822,704 |
| Ling-3.0-flash | 124B | 5.1B | Ling 卡片（另写 "1/64 sparse MoE"） | API 127,486,405,600 |

---

## 12. GitHub 发现

### 12.1 NaiveAI-Labs/NaiveRT（`naiveai-labs/NaiveRT` 重定向至此）

| 字段 | 值 | 状态 |
|---|---|---|
| 是否存在 | 存在，公开 | 一手配置 |
| created_at / pushed_at | 2026-09-27T07:43:04Z / 2026-09-27T15:57:47Z | 一手配置 |
| commits | 1 条："Release NaiveRT for Naive-N0.5-Flash"，author date 2026-09-27T15:55:17Z | 一手配置 |
| license | Apache-2.0（含 LICENSE、NOTICE、THIRD_PARTY_NOTICES.md） | 一手配置 |
| stars / forks | 158 / 8（抓取时） | 一手配置 |
| 语言（字节） | Python 3,211,712；Cuda 1,937,695；C++ 318,927；JavaScript 95,877；CSS / HTML / Shell / Dockerfile / CMake / Jinja 少量 | 一手配置 |
| 结构（663 个 tree 项） | `naivert/kernels`(120)、`naivert/body`(72，大量 `.cuh`/`.cu`：`attn_dsa/`（含 `index_sweep_tcgen05.cuh`、`topk_grid_select.cuh`）、`attn_swa/`、`dspark_*`、`expert_down_w8a8.cuh`、`unproj_o_allreduce` 等)、`naivert/prefill`(69)、`naivert/entry`(43)、`naivert/backends`、`naivert/runtime`、`naivert/serve`、`naivert/pdl`、`apps/studio`（Web 前端）、`devops/`（Dockerfile、compose、validate.sh）、`tools/vendor_prefill_*.py` | 一手配置 |
| 基准 / 测试脚本 | `tests/gpu/`(107)：`naive_n0p5_draft/round/benchmark.py`、`sampling/bench*.cu`、`prefill_native/bench_*.py`（含 `bench_dsa_long_context.py`、`bench_deepep_dispatch.py`）、`naive_n0p5_e2e/bench_verify_kernels.py`；`tests/host`(31)；`tests/contracts`（含 `golden/naive_n0p5_spec.json`；tests/README 原文："The GPU checks in this directory are placeholders and always skip"） | 一手配置 |
| 确定性 | README："bitwise determinism as a correctness requirement for supported execution configurations"；"targets bitwise-identical tokens, logits, and KV state"。round harness README：`--cycles 24` 模式 "Every sample is compared bitwise"；同时写 "The rows-8 and rows-1 kernels are not bitwise"、native 与 kernel prefill "are not assumed to produce bitwise-identical caches"；记录的一次运行 "passed 386,706 tokens, 83,735 rounds and 92,928 final-state hash comparisons"，并注明 "Measurements below describe recorded runs, not a fresh validation" | 自报 |
| README 性能数字 | "1,719 device tokens/s"（thinking 关）、"peak one-second rate of 2,122 tokens/s"；thinking 开 1,090 / 峰值 2,058；draft 接受率 70.6% / 43.4%；"A full speculative round took approximately 3.45 ms" | 自报 |
| repo description | "…1,700 tokens/s at batch size 1, one-second peaks above 2,000." | 自报 |
| 硬件要求 | "eight Blackwell GPUs with UMMA (`tcgen05`) support"；"compute capability 10.3a"；CUDA 13.2；CUTLASS v4.7.1；"a supported W8A8 target checkpoint" | 自报 |
| 与 HF 仓库的对应 | README 写 "The target directory must contain `model.safetensors.index.json`"；HF 上 `Naive-N0.5-Flash-FP8` 的 tree 中**没有**该文件 | 一手配置（并排） |
| 10 月 12 日承诺 | 博客（NV-68）写 "will be available by Oct, 12th"；GitHub 记录的 created_at / pushed_at / commit 日期均为 2026-09-27 | 一手配置（并排；本文件不判断博客文字写作时点） |
| 构建所用 AI | README / devops README / THIRD_PARTY_NOTICES 中未出现 Claude / GPT / Codex / Gemini 等字样 | 一手配置 |

### 12.2 NaiveAI-Labs/Naive-N0.5-Flash（HF 卡片中 GitHub 链接的目标）

created 2026-09-27T15:03:09Z，1 commit（2026-09-27T15:57:25Z "first commit"），MIT，22 个 tree 项：README、LICENSE、assets、FP8 版 config.json（4,628 B）、configuration/modeling、tokenizer 文件、chat_template.jinja（8,070 B，与 HF 的 7,755 B 不同）。README 与 HF 卡片仅排版差异（YAML 头、logo 写法、表格 HTML 化）。无权重。

### 12.3 recursive-oss/rec

| 字段 | 值 | 状态 |
|---|---|---|
| created_at / pushed_at | 2026-10-21T15:35:55Z / 2026-10-27T02:02:06Z | 一手配置 |
| commits | 8；author date 范围 2026-10-21T15:53:16Z（"Release v0.1.0"）至 2026-10-26T18:33:38Z（"Release v0.1.1"）；中间包括 Anthropic Messages API 支持（"Handle current Claude clients on /v1/messages"） | 一手配置 |
| license / stars / forks | Apache-2.0 / 286 / 48 | 一手配置 |
| 语言（字节） | Python 420,947；Go 120,368；JavaScript 10,582；Shell 6,961；C++ 6,430；TypeScript 5,906；Jinja；Makefile | 一手配置 |
| 结构 | `cli/`（Go：launch/serve/download/update 等，含 pi 与 OpenCode 客户端补丁）；`engine/src/rec_mlx/`（Python，`pyproject` 依赖 `mlx==0.32.0`、`mlx-lm==0.31.3`、`transformers==5.16.1`、`xgrammar`；描述 "Flint inference engine for Apple Silicon"）；`engine/native/grammar_native.cpp`；`packaging/`（含 `flint-int4-mlx.json`）；`tests/` | 一手配置 |
| kernel | `engine/src/rec_mlx/kernels/`：megakernel.py（59,815 B）、kv_teams、expert_teams、mixed_bits、masked/filtered_sampling、m4_max(_prefill)、rank_routing 等 Python 文件；megakernel.py 使用 `mx.fast.metal_kernel` 与 `simdgroup_*`；shader_source.py 报错信息为 "Missing Metal function"。仓库中**无** `.metal`、`.cu`、`.cuh` 文件 | 一手配置 |
| 平台 | README："Apple Silicon Mac with an M4 Pro, M4 Max, M5 Pro, or M5 Max"；"At least 48 GB unified memory" | 自报 |
| 性能 | "Compared to llama.cpp, rec delivers 1.4–2.6× the token throughput … on M4 Pro" | 自报 |
| harness | "We find the best performance from our models when paired with the pi harness"；提供 OpenCode；提供 Anthropic 兼容 API（CHANGELOG："for Claude Code and the Anthropic SDK"） | 自报 |
| 测试 | engine/tests（test_anthropic、test_grammar、test_kernel_routing、test_shared_kv、validate_resident 等）；tests/flint_smoke.py、protocol_test.py | 一手配置 |

### 12.4 apexin-ai/Apex-35B-V1.0

created 2026-10-28T09:48:49Z，pushed 2026-10-29T20:35:53Z；4 commits（与 HF 的 4 条标题一一对应）；Apache-2.0；GitHub languages 为空；tree 仅 `LICENSE`、`README.md`、`assets/`（3 张图）。**除 README 外无代码、无训练/推理脚本**。README 与 HF 卡片仅 YAML 头与图片路径不同。

同 org 其他公开 repo（仅列名，未读内容）：`.github`、`apex-research-report`、`Evaluation-Self-Evolve-AI-System`、`dashboard`、`ASI-Bench`、`Math-Paper`、`Apex-Tune-results`、`qwen36-35b-takehome`、`Apex-AutoResearch-Papers`、`miles-personal-fork-archive-01a188ab`、`miles`。

### 12.5 README / 模型卡中出现的模型与 agent 名称

| 仓库 | 出现的名称及语境（原文） |
|---|---|
| Naive（HF 三个 repo + GitHub） | 评测 harness："Claude Code 2.1.207"；其余 GPT-5.6-Sol、GPT-6-Astra、Opus-5.5、Gemini-3.6-Flash 等均为对比模型 / 分数来源。卡片未写用哪个模型来"构建"本模型 |
| NaiveRT | 未出现 |
| Flint 卡片 | 未出现 Claude/GPT/Gemini/Codex；harness 为 Terminus 2、mini-swe-agent |
| rec | pi、OpenCode；Anthropic 兼容 API 面向 Claude Code |
| Apex 卡片 / GitHub | "supported by the Claude Code (v2.1.280 preferred) harness"；构建侧只写 "Apex Research"（自家自动研究系统），未写所用底层模型 |
| Ling-3.0-flash | 使用体验：Claude Code、Kilo Code、Qwen Code、Hermes Agent、OpenClaw；评测 judge：GPT-5.4-mini、Gemini-2.5-Pro、GPT-4.1、Claude Opus 4.6；harness：OpenHands |
| MiMo-V2.5 | 未出现 |

---

## 13. 与博客说法不一致处（仅并排列出，不作判断）

| # | 博客 / 卡片说法 | 仓库所见 | 出处 |
|---|---|---|---|
| 1 | Naive 卡片："Both attention types incorporate sink bias." | config `add_full_attention_sink_bias: false`；DSA 层（0,5,…,47）在权重中无 `attention_sink_bias` 张量 | Naive config.json / safetensors header |
| 2 | Naive 博客台账："HF 上有 W8A8 checkpoint 与 DFlash draft model" | repo 名为 `-FP8`（`quant_method fp8`，dynamic activation）；draft repo 的 architecture 为 `DSparkDraftModel`（DFlash 骨干 + Markov / confidence 头）；draft repo README 与主模型 README 相同 | HF tree / config |
| 3 | Naive 博客 NV-68："The above content will be available by Oct, 12th." | NaiveRT 的 created / pushed / commit 日期均为 2026-09-27 | GitHub API |
| 4 | Naive 博客 NV-43："3.4 ms" 完整 speculative round；NV-41 峰值 2,122 tok/s | NaiveRT README："approximately 3.45 ms"；平均 "1,719 device tokens/s"，峰值 2,122；repo description："1,700 tokens/s at batch size 1" | NaiveRT README |
| 5 | Naive 卡片："replaces MLA with grouped-query attention (GQA) using four KV groups" | 基座 MiMo-V2.5 config 已是 GQA（全局层 kv 4，SWA 层 kv 8），无 MLA 字段；Naive 的 SWA 层仍为 kv 8 | 两个 config |
| 6 | Naive 卡片正文 "MiMo-V2.5 base model"，链接指向 `MiMo-V2.5-Base` | Naive 的 `max_position_embeddings 1048576`、`rope_theta 1e7` 与 MiMo-V2.5 相同，与 MiMo-V2.5-Base（262144、5e6）不同 | 三个 config |
| 7 | NaiveRT README："The target directory must contain `model.safetensors.index.json`" | HF `Naive-N0.5-Flash-FP8`（及 BF16 repo）tree 中无此文件 | HF tree |
| 8 | MiMo README："Num KV Heads 8 (GA) / 4 (SWA)" | MiMo config：全局层 `num_key_value_heads 4`，SWA 层 `swa_num_key_value_heads 8` | MiMo README / config |
| 9 | Flint 博客："context windows of up to 512k tokens"（推荐 256k）；"40 billion total" | config `max_position_embeddings 262144`；卡片 39.8B；API 39,785,287,456 | Flint config / API |
| 10 | Flint 博客正文日期 Oct 18, 2026（metadata Nov 6） | HF repo createdAt 2026-10-21；rec 首个 commit 2026-10-21 | HF / GitHub API |
| 11 | Apex 卡片 Input / Output "Text / Text" | config 保留 `vision_config`；index 保留 333 个 visual 张量；附带 processor 配置 | Apex config / index |
| 12 | Apex 博客："mid-training and post-training"；卡片 "further trained" | config 除 transformers_version 与 vision model_type 字符串外与 Qwen3.6-35B-A3B 相同；tokenizer.json 哈希相同；chat template 与 Qwen 2026-04-24 版只差 `<think>\n` → `<think>`；权重分片哈希全部不同 | 两个 repo |
| 13 | Ling 卡片 "124B total" | HF API safetensors total 127,486,405,600（checkpoint 含 MTP 层 `model.layers.42`） | Ling API / index |
| 14 | Ling 卡片 YAML `license: mit` | 仓库内无 LICENSE 文件 | Ling tree |
