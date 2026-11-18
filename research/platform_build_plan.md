# RSI-mini 平台搭建计划（BUILD PLAN，未动工）

- 写作时间：2026-11-17 约 19:50–20:15（本地）。**本文只是计划：我没有安装、下载模型、提交作业、SSH，也没有碰 Mac GPU。** 除本文件外没有改动任何文件。
- 本次实际做过的只读检查：本机 `which` / `--version` / `docker info` / `ls` / 读文件；对公开端点的只读请求（PyPI JSON、GitHub API、HF API、几个站点的 `curl -I` 连通性）；WebFetch / WebSearch 查工具文档。清单见 §9。
- 标 **〔估算〕** 的数字没有实测，动手后要换成实测值，**不能写进报告**。标 **〔待核实〕** 的是文档没写清或只有二手来源的事项。
- 建立在 `research/platform_design.md`（下称"设计稿"）之上：一阶 nAUC、二阶 Δ₂、U_s / U_w、A/A、placebo、retrieval、held-out、prereg 判定规则全部沿用，不重复。
- **本文在写作过程中改过一次方向**：最初按"cluster-B、全手动中继"写；20:07 协调者通知主算力改为 **cluster-A**，由 main session 经一条共享 SSH master 操作。现在的正文按 cluster-A 写，cluster-B 的手动块移到附录 A。

---

## 0. 先看这一页

### 0.1 算力与环境的现状

| 事实 | 来源 |
|---|---|
| **主算力：cluster-A。** job <jobid> = 节点 <node>，3 × A100 80GB PCIe，36 CPU，360G 内存，结束于 2026-11-19 20:01（北京时间），写作时还剩约 47.9 小时 | 协调者消息（20:07） |
| 驱动 580.126.20；**计算节点能联网**（HF / PyPI 返回 200）；scratch 空间充足 | 同上 |
| 项目目录 `$R`；`HF_HOME=$R/hf`；conda env `rsi`（最新 vllm + peft + trl）正在由 main session 创建 | 同上 |
| Mac ↔ 集群这一跳很慢，约 285 KB/s | 同上 |
| 次要算力：cluster-B，job <jobid>（<node>，4 × H20）与 <jobid>（<node>，1 × A800），约 02:00 到期；banner 禁止 Claude，**只能由你手动粘贴** | `hpc_allocations.md`；memory `hpc-manual-only` |
| Mac：Docker client 28.1.1 在，但 **daemon 没有运行**（`docker info`：Cannot connect to the Docker daemon） | 本机检查 |
| Mac：磁盘可用 67 GiB；`bench/gpu_lock.sh status` 在 19:54 返回 `ourengine 2026-11-17T19:48:06`（GPU 锁正被别的 agent 占用） | 本机检查 |
| Mac：`uv 0.13.1`、`node v25.8.0`、`npx 11.11.0`、`sandbox-exec`、`llama-server`、`rec` 在；`opencode` / `pi` / `aider` / `ollama` / `tailscale` 不在 PATH | 本机检查 |
| Flint int4 已下完：`flint-int4/` 5 个分片齐全，`~/.local/share/rec/models/…` 另有一份（22 G） | 本机检查 |

### 0.2 cluster-A 的使用规则（正文所有步骤都按这几条写）

来自协调者消息和 20:06 更新的 memory 记录 `hpc-manual-only`：

1. 计算只走 Slurm：`srun --jobid=<jobid> --overlap …`；另开作业用 `sbatch -p <partition>,a100_short,a100_long --exclude=<node>,<node>,<node>,<node>`。登录节点上不跑重活。
2. 长时间运行的东西放在登录节点的 tmux 里。
3. **只有一条共享 SSH master（`~/.ssh/<control-socket>`），只由 main session 驱动**；subagent 不碰，不在循环里反复连接；先 `ssh -O check`，每次中断最多重开一次。
4. Mac ↔ 集群慢：**模型在集群上下载，不从 Mac 推大文件**。
5. 不做永久删除（要清理就移到 `_to_delete/`）。
6. 不碰其他项目的目录和 env（包括 `$SCRATCH/otherproj*`）；只用 `$R` 和 env `rsi`。
7. 命令、文件、日志里不出现凭据。

**关于授权，我要说明一点。** 我接到的原始任务写的是"cluster-A 不在范围内"和"不允许自动 SSH"。改用 cluster-A、允许 main session 自动操作，这两条我是从协调者消息和 memory 记录里得知的，没有直接看到你的原话。我自己没有做任何 SSH。本文把"端口转发承载 orchestrator 流量"单独列成 D0，请你确认它在你授权的范围之内。

### 0.3 需要你拍板的事项

| # | 事项 | 我的建议 | 备选 |
|---|---|---|---|
| D0 | Mac 上的 orchestrator 和 harness 怎么连到集群 | 在共享 master 上**一次性**加两个本地端口转发（vLLM、评估器 API），之后所有往来都是经这两个端口的 HTTP，不再有任何逐次的 ssh 命令。转发由 main session 建立；改进者 agent 的沙箱看不到 `~/.ssh` 和 master socket | 孤岛模式：orchestrator 和 harness 也搬到集群节点上跑（§6.7），Mac 只收结果。只有在端口转发不被允许或不稳定时才用 |
| D1 | 改进者用 `Qwen/Qwen3.5-9B` 的 **BF16 原版 + vLLM（A100）**，而不是你最初定的 MLX 4-bit | 按协调者的新安排执行。**代价：BF16 和 MLX 4-bit 是两个不同的改进者，两边的曲线不能混在一起比。** Mac 上的 MLX 权重因此不必下载，省约 22 GB | Mac MLX 作为离线后备 |
| D2 | harness 主选 | **OpenCode**；退路依次是 pi、mini-swe-agent（§2.2） | pi 当主选 |
| D3 | 一阶任务 | **分阶段组合**：NanoChat（vanilla 起点）测一阶曲线和能力阶梯；µAR 测二阶（§3） | 只做其一 |
| D4 | 搜索 | harness 在 Mac 上，有网。用无 key 的 DuckDuckGo MCP + 官方 fetch MCP；不稳定时再申请 Tavily 或 Brave 的免费 key。**默认屏蔽几个公开答案仓库并记录全部查询**（§2.5 第 11 条）；另跑一个 web-off 臂作对照 | 一开始就用带 key 的搜索 |
| D5 | NanoChat 计时规则 | 评估器以**进程总 wall-clock 硬上限**（300 s + 固定启动余量）为准，不信任脚本自报的训练秒数。这与 Karpathy "启动和编译不计入"的规则不同，要在方法里写明 | 沿用自报时间，只做交叉检查 |
| D6 | Mac 上的 agent 沙箱 | 你手动启动 Docker Desktop 后用容器；不想开就用 `sandbox-exec` profile（§2.4） | — |
| D7 | 各 gate 的阈值（§6.8） | 表里的数是建议值，由你定 | — |
| D8 | Flint | 先做脚手架层自演化；权重层列为 stretch，路线和门槛见 §5 | — |
| D9 | 下载和安装清单（§6.3） | 逐项批准 | — |
| D10 | cluster-B 的两个分配 | cluster-A 已经够用；cluster-B的卡 Mac 连不上（不能建隧道），只能跑孤岛任务，而且每一步都要你手动。**建议释放**（`scancel <jobid> <jobid>`），除非你想要一组独立硬件上的噪声地板（附录 A） | 按附录 A 手动跑 |
| D11 | 过夜无人值守 | 今晚的一阶 campaign 要 Mac 整夜不休眠、SSH master 不断。需要你同意让 Mac 保持唤醒 | 只在你在场时跑 |

---

## 1. 架构

### 1.1 文字架构图

```
┌──────────────────────────────── Mac（M4 Max，只用 CPU）────────────────────────────────┐
│  人（你）：spec.yaml（目标 / 约束 / 验收阈值）  PREREG.md  decisions.md（等你裁决的队列） │
│                                   │                                                    │
│  ┌──────────── Orchestrator（可信区，沙箱外，由 main session 启动）────────────────┐    │
│  │ policy 引擎：karpathy_greedy / recursive_first_steps / naive_trial_ledger /     │    │
│  │              apex_ai4ai / self_evolution（§4）                                  │    │
│  │ 线程：K 个研究线程 = K 个 harness session × K 个 git worktree                    │    │
│  │ 裁决：keep / discard / rollback 在这里做（"agent 提案，orchestrator 裁决"）       │    │
│  │ 记账：评估次数（主口径）、token、wall-clock；断线时暂停而不是记失败               │    │
│  └───┬──────────────────────┬───────────────────────────────┬────────────────────┘    │
│      │ 启动 session          │ 提交 / 取结果                  │ 只追加写                │
│      ▼                      ▼                               ▼                         │
│  ┌ Sandbox A（agent）──┐  ┌ Lab MCP server（沙箱外）┐  ┌ Trajectory store ──────────┐  │
│  │ Harness：OpenCode    │  │ submit_experiment       │  │ gateway/*.jsonl 原始请求响应│  │
│  │  edit / shell / read │─▶│ get_result → 仅 dev 分数│  │ harness/*.json  session 导出│  │
│  │  subagent（可开关）  │  │ read_task_card          │  │ ledger.jsonl    每 trial 一行│  │
│  │ MCP：lab/search/fetch│  └───────────┬─────────────┘  │ diffs/ audits/ generations/ │  │
│  │ 只挂载自己的 worktree│              │                └─────────────────────────────┘  │
│  │ 看不到：~/.ssh、      │              │                                                │
│  │  master socket、token │   ┌ Search / fetch MCP（沙箱外，走 Mac 的网络，带屏蔽表与日志）│
│  └──────────┬───────────┘   └──────────────────────────────────────────────────────────│
│             │ OpenAI 兼容 HTTP                                                          │
│             ▼              │                                                           │
│  ┌ Model gateway（沙箱外）─┐│   本地模型（可选臂）：rec serve → Flint，127.0.0.1:11500    │
│  │ 转发 + 全量日志 + 路由   ││   （占 Mac GPU，须先 bench/gpu_lock.sh acquire）            │
│  │ model / adapter + 配额   ││                                                           │
│  └──────────┬──────────────┘│                                                           │
└─────────────┼───────────────┼───────────────────────────────────────────────────────────┘
              │ 127.0.0.1:18000│ 127.0.0.1:18100
   ═══════════╪════════════════╪═══ 一条共享 SSH master（~/.ssh/<control-socket>），两个 -L 转发 ═══════
              │                │     只由 main session 建立；约 285 KB/s；只走小的 HTTP 报文
              ▼                ▼
┌──────────────── cluster-A：<node>（job <jobid>，3 × A100 80GB）────────────┐
│  GPU0：vLLM（env rsi）                       GPU1、GPU2：评估 lane                        │
│   Qwen3.5-9B（BF16）+ LoRA adapter 热插拔    ┌──── 封存评估器 E（evald，HTTP API）────┐    │
│   阶梯阶段：GPU2 临时改跑 Qwen3.8-27B        │ supervisor：计时、硬杀、每 lane 独占   │    │
│   换心阶段：GPU2 临时改跑 PEFT LoRA 训练     │ ┌ Sandbox B（训练，跑 agent 的代码）   │    │
│                                              │ │ 只读 train shards；无 val；          │    │
│  全部在登录节点的 tmux 里用                  │ │ 只写一次性 run 目录 → checkpoint     │    │
│  srun --jobid=<jobid> --overlap 启动        │ └──────────────────────────────────────│    │
│                                              │ ┌ Sandbox C（打分）                    │    │
│  $R/                   │ │ 可信打分代码 + agent 的模型类；      │    │
│   hf/（模型） data/{train,devval,sealed}/    │ │ dev-val 与 sealed-val；无可写挂载    │    │
│   spool/ runs/（checkpoint 留在集群）        │ └──────────────────────────────────────│    │
│   bundle/（代码，从 Mac 推，很小）           │ 审计 L0–L3；sealed 分数只进账本        │    │
│                                              └────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────────────────┘

其他后端（同一套 spool 协议，不在主路径上）：
  local       ：Mac CPU 跑 µAR 评估（确定性、几秒一次），不经过集群
  sbatch      ：hold 结束后在 cluster-A 另开作业（§6.6）
  manual-relay：cluster-B，你手动粘贴（附录 A，可选）
  5090        ：Tailscale，未检查，本计划不依赖
```

### 1.2 四条设计决定

1. **Model gateway 是轨迹的唯一事实来源，而且在 Mac 上。** 它记录每一次 `/v1/chat/completions` 的完整请求（messages、tools、采样参数）和响应（content、tool_calls、usage、耗时）。这份日志可以原样套 chat template 变成 LoRA 的 SFT 数据，和 harness 无关。轨迹因此天然落在 Mac，不需要跨那条慢链路搬运。
2. **agent 不能自己运行训练，也不能自己裁决。** 它只能通过 `lab` MCP 工具提交文件；训练在集群上的评估器沙箱里跑；keep / discard 由 orchestrator 根据评估器返回的分数执行。Karpathy 原版让 agent 自己跑 `train.py`、自己 grep 分数、自己 `git reset`，这三步都可以被改动或谎报，所以收回到可信区。
3. **agent 与封存数据之间是物理隔离。** agent 在 Mac 的沙箱里，val shard、封存 seed、评估器代码都在集群上，中间只有评估器 API 这一个窄口。集群上仍然需要沙箱，因为 agent 写的 `train.py` 会在那里执行（§2.4）。
4. **SSH 只建立一次，之后只有 HTTP。** 两个端口转发加到共享 master 上以后，orchestrator 每个 trial 只发两类小报文：模型请求（经 gateway）和评估提交 / 查询。**没有任何逐 trial 的 ssh、scp、rsync。** master 断了，gateway 和 lab 返回 503，orchestrator 把所有线程置为 `PAUSED`（不计为失败 trial），等 main session 按规则重开一次；集群上的 evald 继续跑完已排队的评估并保留结果（按 trial id 幂等），恢复后不丢数据。

### 1.3 建议的目录（尚未创建）

```
rsi-report/platform/                         （Mac）
  orchestrator/   policies/   gateway/   harness_adapters/{opencode,pi,miniswe}/
  mcp/{lab,search,fetch}/   sandbox/{docker,seatbelt}/   update/{lessons,sft}/
  evaluator/muar/           （Mac CPU 上的 µAR 评估器）
  cluster_bundle/           （推到集群的全部代码，纯文本，很小）
      evald/  nanochat/{vanilla_start.py,arlib.py,score.py}  serve_vllm.sh  lora_train.py  sandbox/
  store/   PREREG.md   spec.yaml   analyze.py

$R/                    （cluster-A；只用这一个目录）
  hf/   data/{train,devval,sealed}/   bundle/   spool/   runs/   adapters/   run/{ports,token}   _to_delete/
```

---

## 2. 组件选型

### 2.1 总表

| 组件 | 选择 | 理由 | 备选 |
|---|---|---|---|
| Orchestrator | 自写 Python（单进程 + asyncio，状态全在只追加的 `ledger.jsonl`，可断点续跑、可暂停） | policy 是本项目的研究对象，要能逐行对照论文；链路中断或作业到期后必须能续跑 | LangGraph（多一层抽象，没必要） |
| Harness（主） | **OpenCode**，版本钉在 2.0.29，装在 Mac：用 rec 钉住的 patched 版（`rec/cli/clients/opencode/pins.json`：`opencode-rec-2.0.29-1`，darwin-arm64，86,372,353 B，带 sha256） | 见 §2.2 | pi 0.73.1；mini-swe-agent |
| Model server | **vLLM，cluster-A A100，env `rsi`**。`Qwen/Qwen3.5-9B`（19.33 GB）常驻 GPU0；`Qwen/Qwen3.8-27B`（55.59 GB）在阶梯阶段占用 GPU2（单张 80 GB 放得下，留给 KV cache 的余量约 20 GB〔估算〕） | 并发批处理，多线程和 subagent 不互相阻塞；LoRA adapter 可以直接挂载；vLLM 官方 recipe 对 Qwen3.5 的 tool calling 给了参数（`--enable-auto-tool-choice --tool-call-parser qwen3_coder`） | SGLang；llama.cpp CUDA 版 |
| Model server（Flint，可选臂） | Mac 上的 `rec serve`（127.0.0.1:11500，OpenAI 与 Anthropic 兼容） | 已安装，权重已校验；实测 196 tok/s（1k 上下文）、66 tok/s（104k）（`rec_optimization_analysis.md` §6） | llama.cpp + `flint-int4-gguf`（25.99 GB，要另下） |
| Model gateway | 自写薄代理（aiohttp，约 150 行〔估算〕），在 Mac | 只需要转发、落盘、按臂路由、配额、断线时返回 503 | LiteLLM proxy（需要 OpenAI ↔ Anthropic 互转时再上） |
| Mac ↔ 集群链路 | 共享 master 上的两个 `-L` 转发：vLLM 端口、evald 端口 | 规则 3；见 §1.2 第 4 条 | 孤岛模式（§6.7） |
| 封存评估器 | 自写 `evald`：集群节点上的 HTTP 服务 + supervisor + 训练沙箱 + 打分沙箱（§2.5）；每个评估 lane 一张 GPU | 设计稿 §5.2 的加固清单 + 本文补充的四类接口作弊 | — |
| Sandbox（Mac，agent） | Docker 容器（需要你启动 Docker Desktop），内部网络只通 gateway 和 MCP | 文件系统隔离最干净 | `sandbox-exec` + Seatbelt profile（不需要 daemon；Apple 标注为 deprecated 但仍可用〔待核实：macOS 26.6 上的行为〕） |
| Sandbox（集群，训练与打分） | Singularity / Apptainer（UV0 探测节点上有没有），`--containall --no-home --cleanenv`，只 bind 需要的目录 | 不需要 root | bubblewrap；都没有时退到"独立目录 + 路径审计"，并在报告里写明隔离强度下降 |
| Lab 接口 | 自写 MCP server（stdio，在 Mac），三个工具：`submit_experiment` / `get_result` / `read_task_card`；内部调 evald 的 HTTP API | 这是 agent 通往评估器的唯一通道，同时满足"improver 有 MCP 工具"的要求 | CLI 版 `lab submit`（给不支持 MCP 的 harness 用） |
| 搜索 | `duckduckgo-mcp-server`（`uvx` 启动，工具 `search` / `fetch_content`，**不需要 key**）+ 官方 `mcp-server-fetch`，都在 Mac，经一个带屏蔽表和日志的小网关 | 无 key；从本机 `curl -I https://html.duckduckgo.com/html/` 能连上（HTTP 400，5.6 s，较慢） | Tavily（`TAVILY_API_KEY`）或 Brave Search API（`BRAVE_API_KEY`），都有免费档；OpenCode 内置 `websearch` 支持 Exa / Firecrawl / Parallel / Tavily / TinyFish，其中 TinyFish 的 key 标为可选〔待核实〕 |
| 轨迹存储 | Mac 上的文件：JSONL + git（每个线程一个 worktree，每个 trial 一个 commit） | 可 diff、可审计；不需要数据库 | SQLite（OpenCode 自己的 session 库可以作为副本） |
| U_w（换心） | cluster-A：PEFT LoRA（env `rsi` 已含 peft + trl），GPU2 上训练，adapter 留在集群，vLLM `--lora-modules` 挂载。SFT 数据在 Mac 上由 gateway 日志生成，压缩后经 master 上传（纯文本，预计几十 MB〔估算〕；按 285 KB/s，100 MB 约 6 分钟） | 轨迹在 Mac、权重在集群，只搬小的那一边 | — |
| 分析 | Mac 上的 `analyze.py`：nAUC、Δ₂、分层 bootstrap、sign-flip、v / p / m 分解、hack 率、dev–sealed 差 | 沿用设计稿 §1 | — |

### 2.2 Harness 逐项比较

评判标准：本地 OpenAI / Anthropic 兼容端点；9B 的 tool calling 可靠性；headless 脚本化；MCP；subagent；完整轨迹；可沙箱化。harness 现在跑在 Mac 上（§1）；能不能在没有 Docker、没有 root 的 Linux 计算节点上跑，只影响后备的孤岛模式（§6.7）。

| | OpenCode 2.x | pi 0.73.1 | OpenHands | mini-swe-agent | Claude Code + 本地端点 | Aider |
|---|---|---|---|---|---|---|
| 本地端点 | ✅ `openai-compatible` provider；rec 的 `launch.go:321-364` 就是这样接 Flint 的，可照抄 | ✅ `models.json` 加 provider；rec 也有现成扩展 `flint-pi.ts` | ✅ 经 LiteLLM | ✅ 经 LiteLLM | 需要 Anthropic 兼容端点：rec 有 `/v1/messages`；mlx_lm.server 文档里只有 OpenAI 风格端点，需要转换代理 | ✅ |
| Headless | ✅ `opencode run`（官方文档："designed for scripts, CI jobs"）；`--format json` 输出 NDJSON 事件、`opencode export <sessionID>` 导出 JSON〔待核实：来自第三方文档，装好后用 `--help` 确认〕 | ✅ `-p`、`--mode json`、`--mode rpc`、SDK | ✅ `--headless -t`、`--json`；但官方文档写明 CLI "no longer actively maintained" | ✅ Python API / CLI | ✅ `-p` + stream-json | ✅ |
| MCP | ✅ 原生（`mcp.servers`，`type: "local"` + `command`）。注意默认 `codemode: true` 会把 MCP 工具收进 Code Mode，对 9B 可能太绕，**建议设 `codemode: false`** | ❌ README 原话："No MCP… or build an extension that adds MCP support" | ✅〔待核实：headless 下的配置方式〕 | ❌ | ✅ | ❌〔凭已有知识，未抓文档〕 |
| Subagent | ✅ 原生 `subagent` 工具；内置 `general` 与只读的 `explore`；可用 Markdown 自定义 agent，可按 agent 配模型与权限；子 agent 在子 session 里后台运行。是否并行文档没写〔待核实〕 | ❌ README 原话："No sub-agents"（可用扩展或多开进程） | ✅ 有 delegation〔待核实〕 | ❌ | ✅ | ❌ |
| 9B 可靠性 | 中：内置工具约十个，system prompt 较长〔估算〕。rec 专门加了一段 instructions 提醒模型严格按 schema 调工具（`clients/opencode/instructions.md`），说明小模型在这里确实会出错 | **最好**：默认只有 `read` / `write` / `edit` / `bash` 四个工具；rec README 称 pi 搭配效果最好（自报） | 低：prompt 重 | 高：只有 bash，不依赖原生 tool calling〔待核实：v2 的默认行为，来源互相矛盾〕 | 低：system prompt 和工具定义很长〔估算〕，且内置 WebSearch 是 server-side 工具，第三方端点不可用（二手来源一致，未对官方文档核实） | 不适用：不是 tool-calling agent 循环 |
| 轨迹 | session 导出 + 自带 SQLite；**以 gateway 日志为准** | session 是树状 JSONL（`--session-dir` 可指定） | JSONL 事件 | `.traj.json` | transcript JSONL | 聊天记录 |
| 沙箱 / 孤岛模式 | 单个二进制，容易放进容器；rec 的 patch 提供 `REC_OPENCODE_HOME`，可以把它的数据目录整体指到别处，正好用来做每个线程的隔离 | 需要 Node（登录节点用 conda 装到 scratch） | 默认依赖 Docker runtime，HPC 上不合适 | 纯 Python；LiteLLM 文档称支持 local / docker / podman / apptainer | 需要 Node；还会发起额外的辅助请求 | 纯 Python |

**结论与理由**

- **主选 OpenCode。** 你要求的五样（edit / bash / tests、sandbox、web search、多 agent、MCP）里，它是唯一不用自己写扩展就全有的；rec 仓库已经解决了"本地单模型 + 隔离用户目录"的配置问题。
- **pi 作为"小 prompt"退路。** 如果 9B 在 OpenCode 上的 tool-call 合法率过不了 gate，就换 pi：MCP 用我们自己的 `lab` CLI 代替（pi 的哲学正是"用带 README 的 CLI 工具代替 MCP"），多 agent 用 orchestrator 层的并行线程代替。
- **mini-swe-agent 作为兜底和基线。** 它最简单，Flint 的官方 SWE-bench 评测用的就是它（Flint HF 卡，自报），也最适合孤岛模式。
- **Claude Code + 本地端点**只作为 Flint 或 27B 的可选对照臂（rec 的 CHANGELOG 写明 `/v1/messages` 是为 Claude Code 准备的），不给 9B 用。
- **OpenHands、Aider 不选**：前者的 CLI 已停止维护且依赖 Docker runtime，后者没有 agent 工具循环。
- **多 agent 分两层做，互不依赖**：(a) orchestrator 层的并行研究线程（K 个 session × K 个 worktree，可复现、可记账，直接对应 Recursive 的 "parallel threads"）；(b) harness 内的 subagent（作为脚手架 σ 的一个开关，要先过 gate）。即使 (b) 对 9B 不可靠，(a) 仍然成立。
- **harness 本身是脚手架 σ 的一部分**。所以 orchestrator 里的 harness 是一个 adapter 接口（`run_session(worktree, prompt, endpoint, profile, budget)`），换 harness 是换一个实验条件，不是重写平台。


### 2.3 改进者模型

| 模型 | 在哪 | 大小（HF API） | 备注 |
|---|---|---|---|
| Qwen3.5-9B（一阶主改进者，也是换心的 M0） | cluster-A，vLLM，GPU0 | `Qwen/Qwen3.5-9B` 19.33 GB | `model_type` 是 `qwen3_5`，pipeline 标的是 image-text-to-text（带视觉塔） |
| Qwen3.8-27B（能力阶梯臂） | cluster-A，vLLM，阶梯阶段占 GPU2 | `Qwen/Qwen3.8-27B` 55.59 GB | `model_type` 同为 `qwen3_5` |
| Flint（阶梯臂 + 自演化） | Mac，`rec serve` | 已在本机 | 占 Mac GPU，须先拿 `gpu_lock` |
| （后备）Qwen3.5-9B / Qwen3.8-27B 的 MLX 4-bit | Mac，mlx-lm 0.32.2 | 5.98 GB / 16.07 GB | 现在不需要。mlx-lm 0.32.2 的 `mlx_lm/models/` 里有 `qwen3_5.py`（GitHub API 确认），真要用时装在独立的 uv venv 里 |

待实测确认的三点：(1) env `rsi` 里的 vLLM 和 transformers 版本是否认识 `qwen3_5`，parser 先用官方 recipe 的 `qwen3_coder`，社区有报告称长轨迹下不稳，备选 `hermes` / `qwen3_xml`；(2) vLLM 的 LoRA 挂载参数和 API key 的传法（建议用环境变量而不是命令行参数，避免 token 出现在进程列表里）〔待核实〕；(3) thinking 开关在所有臂里固定为同一设置，并写进 PREREG。

### 2.4 Sandbox 细节

**Mac：agent 沙箱（Sandbox A）**

| | 内容 |
|---|---|
| 跑什么 | harness（OpenCode）和 agent 的 shell |
| 可写 | 自己的 worktree |
| 只读 | 任务卡片、起始 `train.py`、工具 |
| 网络 | 只能到 gateway、lab MCP、search / fetch 网关 |
| **看不到** | `~/.ssh`（含 master socket `<control-socket>`）、gateway 和 evald 的 token、`platform/store/`、其他线程的 worktree、`~` 的其余部分、`bench/` |

- Docker 方案：agent 容器只挂 worktree，接入一个 internal network；gateway 和 MCP 跑在宿主机或一个双网卡的 sidecar 上。
- `sandbox-exec` 方案：profile 拒绝读取上表"看不到"的路径，只允许写 worktree，只允许连 `127.0.0.1` 上指定的几个端口。优点是不需要 daemon；缺点是规则写错了不容易发现，要配一组"应该被拒绝"的自测（读 `~/.ssh`、连 18000 / 18100 端口、写 worktree 之外的路径都必须失败）。
- 注意：转发端口 18000 / 18100 监听在 Mac 的 127.0.0.1 上。agent 沙箱**不能直连这两个端口**，只能连 gateway 和 lab MCP，否则它可以绕过日志和配额，或者直接调评估器 API。

**集群：训练与打分沙箱**

| 沙箱 | 跑什么 | 只读挂载 | 可写挂载 | GPU | 看不到 |
|---|---|---|---|---|---|
| B：训练 | agent 提交的 `train.py` | `/data` = train shards + tokenizer、可信的 `arlib.py` | 一次性的 run 目录 | 1 张（`CUDA_VISIBLE_DEVICES` 固定） | 任何 val shard、以往 run 的目录、编译缓存、`$SCRATCH` 的其余部分（包括别的项目）、`$HOME` |
| C：打分 | 可信打分脚本 + agent 的模型类 | checkpoint、dev-val、sealed-val | 无（tmpfs） | 同一张 | 一切可写位置 |

集群上的沙箱还有第二个作用：**保护同一账号下的其他项目**。agent 写的代码以你的身份在节点上执行，如果不隔离，它能读写 `$SCRATCH/otherproj*`。这和规则 6 直接相关，所以"没有容器就裸跑"不是可接受的退路；没有 Singularity / Apptainer / bwrap 时，至少要用只读 bind 加独立工作目录，并对每次提交做 L0 静态检查（禁止绝对路径、`os.system`、`subprocess`、网络调用）。

计算节点能联网，这对训练沙箱是个新的风险（agent 的代码可以下载预训练权重或外传数据）。沙箱 B、C 要断网（Singularity 的 `--net --network none` 需要相应权限〔待核实〕；不行就用 L0 静态检查加运行时监控兜底）。

残余风险：集群上沙箱内外是同一个 Unix 用户，容器不是安全边界。缓解办法是 L0 检查、gateway 与 evald 日志的事后审计、每代报 hack 率。

### 2.5 封存评估器：把 Recursive 的作弊清单逐条对上

Recursive 在 SOL 一节列的三类是：缓存输出、依赖跨次运行的持久状态、钻计时 harness 的空子（`01_Recursive_FirstSteps_导读.md` §3）。`artifacts_recursive.md` §1 指出他们开源的 `lib.py` 里没有强制 time budget，计时、seed、评测调用都在可被修改的 solution 脚本里。`lit_signal_hardness.md`：ML 工程 agent 约 50% 的 episode 有篡改评估器的尝试，锁定评估器后消除，运行时间中位开销 25–31%（L-24）；RE-Bench 128 次运行里 39 次有 hack（L-18）。

| # | 作弊方式 | 控制 |
|---|---|---|
| 1 | 缓存或背下 val | val 数据不进沙箱 A、B。dev-val = shard 6542（Karpathy 钉住的那个）；sealed-val = 另外留出的 shard（建议 6540、6541；训练只用前 N 个 shard，所以它们天然不在训练集里）。agent 只看得到 dev 分数；sealed 分数只进账本，不回传。**dev–sealed 差**作为每代必报指标 |
| 2 | 跨次运行的持久状态 | 每次评估新建 run 目录、新进程；编译缓存每次清空；训练沙箱里没有以往 run 的任何文件；提交文件有大小上限（防止把权重塞进源码） |
| 3 | 计时 | supervisor 在沙箱外计时并强杀整个进程组；总 wall-clock 上限 = 300 s + 固定启动余量（余量在 S0 用 vanilla 实测后定）；脚本自报的 `training_seconds` 只记录、不采信；评估 lane 的 GPU 独占，开始前检查显存为空 |
| 4 | 篡改评估器或伪造结果 | 评估器代码和数据不挂载进沙箱 A；`lib.py` / `arlib.py` 只读并校验 hash；结果由评估器写入 spool，agent 只能经 `lab` MCP 读取；keep / discard 由 orchestrator 执行 |
| 5 | 接口作弊（Recursive 的 `lib.evaluate_bpb` 是调用 `model(x, y, reduction='none')` 让模型自己报 loss，模型可以直接返回假 loss） | **契约改为模型只输出 logits**，评估器自己做 `log_softmax` 和交叉熵，targets 不传给模型 |
| 6 | 偷看未来 token | 因果性探针：随机改动位置 t 的输入，检查位置 < t 的 logits 不变 |
| 7 | 打分时在 val 上自适应 | `inference_mode` + `eval()`；打分前后参数 hash 一致；打乱 batch 顺序后分数不变 |
| 8 | 挑 seed | seed 由评估器指定；确认性复跑换 seed；如果不同 seed 的结果完全相同，说明代码忽略了 seed，标记审计 |
| 9 | 多占算力 | 沙箱 B 只见 1 张 GPU；CPU 数由 `srun` 固定 |
| 10 | 改 tokenizer | 第一版禁止改 tokenizer（偏离 Karpathy 规则，要写明）；以后需要时，把 `encode()` 纳入契约，BPB 按字节算 |
| 11 | 从网上抄现成答案 | Recursive 的 `optimized_from_vanilla.py`、Karpathy 的最优 `train.py`、autoresearch@home 的方案都是公开的。harness 在 Mac 上，有网，所以这一条是实打实的风险：fetch 网关屏蔽这几个仓库的 URL，所有查询和 URL 落盘，命中过相关页面的 trial 标为 retrieval-assisted 并单独报告；另跑一个 web-off 臂作对照 |

**逐级加严的审计（对应 Recursive 的 "escalating correctness audits"）**

| 级别 | 何时触发 | 内容 |
|---|---|---|
| L0 | 每次提交 | 静态检查：契约函数存在、文件大小、禁止的 import 和路径、没有访问 `/data` 以外的数据 |
| L1 | 每次评估 | 上表 3、5、6、7 的运行时检查 |
| L2 | 每次 keep | 换 seed 复跑确认（seed racing，设计稿 §5.2 第 4 条）；sealed 重打分 |
| L3 | 每代增益最大的 k 个 trial，以及任何 dev–sealed 差异常的 trial | 让一个更强的冻结模型读 diff 和日志给出 hack 判断，再由你抽查。**审计规则本身有版本号**，每次新增规则都记进账本——这是 Recursive 所说"detector 本身也在迭代"的对应物 |

---

## 3. 一阶任务：µAR、NanoChat（vanilla 起点），还是组合

| | µAR（设计稿 §5.1） | autoresearch-NanoChat，vanilla 起点 |
|---|---|---|
| 单次评估 | Mac CPU 上几秒〔估算〕，确定性 | A100 上 5 分钟训练 + 编译和打分，每次约 7 分钟〔估算〕 |
| 评估噪声 | 0（固定 seed、单线程） | B200 上 vanilla 的单 seed 标准差约 0.010 BPB（设计稿 §3，由上一位 agent 用仓库 CSV 算出，n=10，不是仓库自报）。**A100 上未知，UV2 步骤实测** |
| 早期增益相对噪声 | 取决于任务设计 | 大：Recursive 的 weak start 是 1.0587 → 0.9344（仓库 README，自报，B200），总幅度 0.124，约为上述标准差的 12 倍 |
| 动作空间 | JSON 配置（12 个旋钮），不执行代码 | 改一个 433 行的训练脚本，需要真的 harness |
| hack 面 | 几乎没有 | 大，需要 §2.5 的全部加固 |
| held-out 任务 | 有（ID / OOD 两档），二阶测量的必要条件 | **只有一个任务**。二阶测量要么用"从前沿接着跑"，要么自己造任务族（见下） |
| 污染 | 任务是脚本新生成的 | 公开技巧很多；Recursive 自己承认 weak start 的结果"不应被视为独立发现" |
| 与你的决定的契合 | 用不上"完整 coding agent 环境"的大部分 | 正好需要 |
| 算力 | Mac CPU，不占集群的评估 lane | cluster-A 的 2 条评估 lane |

**建议：分阶段组合。**

| 阶段 | 任务 | 回答什么 | 在哪跑 |
|---|---|---|---|
| S0 | NanoChat vanilla，原脚本，20 个 seed（两条 lane 各 10 个，同型号 A100，可以合并） | A100 上的噪声地板、每次评估的实际用时、启动余量。**这是 NanoChat 能不能当一阶任务的 go / no-go** | cluster-A GPU1、GPU2 |
| S1 | NanoChat，vanilla 起点，9B 改进者；先 `karpathy_greedy`，再另外三个 policy | 一阶曲线：9B 配上真 harness，能把 vanilla 往下推多少、多快；不同 loop policy 的差别 | harness 在 Mac，评估在 cluster-A |
| S1b | 同一任务，改进者换成 27B、Flint | 能力阶梯（Apex AI4AI "固定任务、换研究者"的形状） | 同上 |
| S2 | µAR，改进者同样经 harness（用 `lab` 工具提交 JSON 配置） | 二阶测量 Δ₂（脚手架层和权重层），带 A/A、placebo、retrieval、held-out | 评估在 Mac CPU；改进者是 cluster-A 的 9B，和 S1 共用同一个 vLLM |
| S3 | NanoChat 任务族：{数据集：climbmix / TinyStories / 代码语料} × {预算：2 分钟 / 5 分钟} × {起点：vanilla / 带随机缺陷的 vanilla} | 代码级动作空间上的 held-out 二阶测量；同时检验 µAR 上换出来的"心"能否迁移（OOD） | cluster-A，hold 的后半段或之后的 sbatch |

理由：vanilla 起点的 NanoChat 信号强、和三家的"擂台"直接对应，但只有一个任务、单次评估贵、hack 面大，适合测一阶和能力阶梯；µAR 噪声为零、任务多，适合测需要大量配对样本的 Δ₂。两者共用 orchestrator、harness、gateway 和轨迹格式，差别只在评估器插件。S2 不占集群的评估 lane，可以和 S1 同时进行。

注意：(1) 固定 wall-clock 预算下，**分数和硬件绑定**，A100 上的曲线不能和 Recursive 在 B200 上的数字直接比，vanilla 在 A100 上的起点分数也会比 1.0587 差；(2) nAUC 的归一化需要 s_ref，NanoChat 上建议用 Recursive 的 `optimized_from_vanilla.py` 在同一张卡上的实测分数，这个脚本在 Ampere 上能不能跑要试。

---

## 4. 把四篇材料的方法做成 loop policy

所有 policy 共用下面的原语（由 orchestrator 提供，policy 只决定调用顺序和裁决规则）：

```python
session = harness.run(worktree, prompt, improver, profile, budget)   # 一段 agent 工作，产出一个提交
result  = lab.evaluate(submission, seeds, level)                     # 封存评估器，返回 dev 分数与审计标志
ledger.append(trial_id, parent, status, scores, audit, cost)        # 只追加
git.advance(worktree) / git.reset(worktree, to=parent)               # 由 orchestrator 执行
```

### 4.1 `karpathy_greedy`（Karpathy autoresearch）

```python
best = lab.evaluate(start, seeds=[s0])                 # 基线
while budget.left():
    sub = harness.run(wt, PROGRAM_MD + ledger_tail(wt), improver)      # 只允许改 train.py
    r = lab.evaluate(sub, seeds=[s0], level="L1")      # 固定 5 分钟；超过上限强杀，记 crash
    if r.ok and r.dev < best.dev:  git.advance(wt); best = r; status = "keep"
    else:                          git.reset(wt);            status = "crash" if not r.ok else "discard"
    ledger.append(..., status)                         # 列同 results.tsv：commit / val_bpb / memory_gb / status / description
```

| 复现了 | 没有复现 / 有偏离 |
|---|---|
| 只改一个文件；固定 5 分钟预算；val_bpb；keep / discard / crash；分支前进或 reset；`results.tsv` 的列；"更简单优先"的准则写进 prompt | 硬件不是 H100；agent 是 9B 而不是前沿模型；计时用总 wall-clock 硬上限（D5）；评估和裁决收回可信区；单次运行决定去留会受噪声影响（这是原方法本身的特点，我们如实保留，并在 4.2 里对照多 seed 版本） |

### 4.2 `recursive_first_steps`（Recursive First Steps）

```python
threads = [Thread(wt_i, focus_i) for i in range(K)]                   # 并行研究线程，各有侧重
for round in rounds:
    for t in parallel(threads):
        idea = harness.run(t.wt, PROPOSE + shared_notes, improver, tools="read-only")   # propose
        sub  = harness.run(t.wt, IMPLEMENT(idea), improver)                             # implement
        r    = lab.evaluate(sub, seeds=[s0], level="L1")                                # run
        if r.ok and r.dev < t.best.dev - margin:                                        # validate
            r = lab.evaluate(sub, seeds=confirm_seeds, level="L2")                      #   换 seed 确认 + sealed 重打分
        t.update(r); ledger.append(...)
        t.next = harness.run(t.wt, CHOOSE_NEXT(ledger_view(t)), improver)               # choose next
    if round % M == 0:                                                                  # combining branches
        merged = harness.run(wt_merge, COMBINE(top_k(threads)), improver)
        rm = lab.evaluate(merged, seeds=confirm_seeds, level="L2")
        if better(rm): rebase_all(threads, merged)
    audit.escalate(top_gains(ledger), level="L3")       # 审计随增益加严；发现新 hack 就新增规则并重审历史
final = lab.evaluate(best, seeds=range(10), level="L2") # 终点用 10 个 seed
```

| 复现了 | 没有复现 |
|---|---|
| 五步闭环；并行线程；合并分支；10 个 seed 的终点评估；逐级加严的正确性审计；detector 迭代；weak start（vanilla）与 strong start（Karpathy seed）两种起点 | 原系统的模型、角色、拓扑、算力、迭代次数都没有披露，所以这只是**按博客的五步描述做的一种解读**，不能叫复现了他们的系统。Speedrun（8 × H100）和 SOL-ExecBench 不做。合并步骤的具体做法是我们自己定的 |

### 4.3 `naive_trial_ledger`（NaiveAI）

```python
spec = load("spec.yaml")   # 人写：目标、约束、验收阈值。例："dev val_bpb 降幅 ≥ 2σ，3 个 seed 确认；总时长 ≤ 上限"
for trial in budget:
    plan = harness.run(wt, PLAN(spec, ledger), improver)
    if plan.kind == "exploratory":                       # 只为获取信息：profiling、消融、读日志
        info = lab.evaluate(plan.sub, level="L1", acceptance=False)
        ledger.append(..., status="exploratory"); continue
    r = lab.evaluate(plan.sub, seeds=spec.seeds, level="L2")          # 只看端到端指标
    if plan.claims_numerics_unchanged:                   # bitwise 门槛
        ok = lab.bitwise_check(plan.sub, parent, steps=N, deterministic=True)
        if not ok: ledger.append(..., status="rolled_back", reason="bitwise"); git.reset(wt); continue
    if meets(spec.acceptance, r): git.advance(wt); status = "adopted"
    else:                         git.reset(wt);   status = "rolled_back"
    ledger.append(..., status, microbench=plan.microbench)            # 微基准只记录，不能触发采纳
    if consecutive_rollbacks(direction) >= spec.stop_k:               # 人叫停
        decisions.enqueue(direction); pause(direction)                # 写进 decisions.md，等你异步裁决
report(ledger.counts("adopted", "rolled_back", "exploratory"))        # 对应原文的 63 / 71 / 17
```

| 复现了 | 没有复现 |
|---|---|
| 人定目标、约束、验收；AI 跑 trial；**端到端验收，微基准不算数**；bitwise 门槛；adopted / rolled-back / exploratory 三分类账本；人在连续失败时叫停 | NaiveRT 的领域（CUDA mega-kernel、speculative decoding）和规模（6 天 151 次 trial）。**bitwise 门槛在训练任务上只适用于"自称不改数值"的系统类改动**（固定 seed、确定性模式下前 N 步 loss 逐位相同）；它的本来用法——推理引擎的 logits 逐位一致——要到设计稿的架构 C（rec 引擎任务）才用得上，那里 `validate_resident.py` 现成 |

### 4.4 `apex_ai4ai`（Apex AI4AI + self-evolving data）

```python
for researcher in [M0, M1, Qwen27B, Flint]:              # researcher swap
    for rep in range(R):                                  # R ≥ 3：原文只跑了一次
        s = run_policy("karpathy_greedy", researcher, budget=EXPLORE)     # 固定探索预算（原文 4 小时）
        if s.no_submission: record(researcher, rep, completed=False); continue   # "中途退出"作为结果记录
        final = lab.evaluate(s.best, seeds=sealed_seeds, budget=RETRAIN, level="L2")   # 固定重训预算（原文 12 小时）
        record(researcher, rep, final.sealed, completed=True)

# self-evolving data：从失败里造训练数据
failures = classify(trajectories, ["tool call 非法", "崩溃 / OOM", "无改进提案", "重复提案", "预算耗尽"])
for f in failures:
    fix = synthesize(f)                                   # 例：修正后的 tool call；hindsight 重标的更好提案
    if lab.verify(fix): sft.add(fix, tag=f.category)      # 只收评估器验证过的样本
```

| 复现了 | 没有复现 / 有偏离 |
|---|---|
| 固定探索预算 + 固定重训预算；换研究者、其余不变；失败分析 → 合成 → 验证 → 进训练集 | 原任务（Sokoban GRPO、DDPO）和 4 h + 12 h 的规模；预算按比例缩小（例如探索 = 12 次评估，重训 = 一次更长的训练 × 3 个 seed，具体数值要你定）。**有意补上原文缺的三样**：多次重复、同一谱系的多代研究者（M0 → M1）、把"退出"记为结果而不是丢弃。原文"36 个高质量动作"是人事后标的，我们改用评估器验证，不做人工事后挑选 |

### 4.5 `self_evolution`（"自演化理想"，外环）

```python
I = Improver(theta0, sigma0)
for k in range(G):
    D_k = run_policy(inner_policy, I, tasks=TRAIN)                     # 内环：上面任一 policy
    cand = {
      "U_s": I.with_scaffold(self_write(I, D_k, what=["lessons.md", "skills/", "agents/*.md", "tools/"])),
      "U_w": I.with_adapter(lora_train(theta0, sft_from(D_k, accepted_only=True))),
      "placebo_s": I.with_scaffold(generic_memo(same_length)),
      "placebo_w": I.with_adapter(lora_train(theta0, sft_from(D_k, rejected_only=True))),
    }
    scores = {name: run_policy(inner_policy, c, tasks=HELD_OUT, seeds=paired) for name, c in cand.items()}
    scores["A/A"] = run_policy(inner_policy, I, tasks=HELD_OUT, seeds=other_seeds)
    if promote(scores, rule=PREREG):        # Δ₂ 区间下界 > 0，> 2 × A/A 标准差，显著好于 placebo，hack 率不升
        I = cand[best]                      # 晋升：下一代的内环由新改进者来跑
    ledger.append_generation(k, scores, promoted=...)
# 完整版（设计稿 P4）：U 的配方本身变成被优化的 artifact
```

| 复现了 | 说明 |
|---|---|
| 没有可复现的对象 | 四份材料**都没有展示这一环**：Flint 把它写成 bet，Apex 声称复利但没有逐代数据，Naive 只给了定义（`00_总览.md` §5 第 4 问）。这个 policy 是设计稿的二阶测量，不是任何一家方法的复现，报告里要这样写 |

---

## 5. Flint 自演化的可行性

### 5.1 (a) 脚手架层：Flint 写自己的 lessons / prompt / 工具 —— 容易

- 做法：`rec serve` 提供 Flint，gateway 把这一臂路由到 127.0.0.1:11500，harness 仍是 OpenCode（rec 自带 OpenCode 和 pi 的接入方式，可照抄）。第 k 代结束后让 Flint 读自己的账本和轨迹摘要，写 `lessons.md`、自定义 agent 的 Markdown、以及可执行的小工具脚本，装进下一代的 σ。跑 3 代，每代配 placebo（不看轨迹写的等长通用备忘）和 A/A。这就是设计稿的架构 B。
- 工作量〔估算〕：平台就绪后约半天搭好，之后是无人值守的运行时间。
- 评估任务：**µAR**（评估在 Mac CPU，样本多，适合测 Δ₂）为主；NanoChat 也可以（评估在 cluster-A，orchestrator 在 Mac，链路已经有了），但只有一个任务，只适合放进能力阶梯。
- 约束：
  - **占 Mac GPU**，必须先 `gpu_lock.sh acquire`，和 benchmark agent 互斥，适合夜间跑。
  - rec 一次只处理一个请求（`scheduler.py`，`is_batchable = False`），并行线程和 subagent 会排队。
  - 只有一个 PromptCache、最多 2 个 checkpoint（`prefix.py`）。多个 session 交替请求会让前缀缓存反复失效；实测 prefill 约 1,300 tok/s，104k 的冷启动 prompt 要 2.5 分钟（`rec_optimization_analysis.md` §6）。**建议 Flint 臂单线程、关闭 subagent。**
- 能说明：Flint 靠自己写的经验能否加快后续改进（脚手架层 Δ₂）。不能说明：权重没变，不是"换心"。

### 5.2 (b) 权重层：三条路线

事实基础：

- `RecursiveSI/flint` 是 BF16，约 79.6 GB，卡片写 "Transformers modeling code is not included yet"，config 没有 `auto_map`（`artifacts_weights_configs.md` §3）。
- rec 的 `model.py`（360 行）是完整的 `mlx.nn` 定义；`quantized.py` 用 `nn.quantize` 把它变成 4-bit（group 64，affine）再加载权重。prefill 走这张 MLX 图。
- decode 走 megakernel：`megakernel.py` 把权重打包成自己的布局，断言 `p.bits in (3, 4, 8)`；`checkpoint.py` 要求 `(bits, routed_expert_bits, embedding_bits) == (4, 4, 8)`、group 64，并且 shard 集合与 manifest 完全一致；Go 侧还对每个文件做 SHA-256 校验。**megakernel 里没有 LoRA 的低秩旁路。**
- HF 上的 `sglang/` 目录有 Dockerfile（SGLang v0.5.16 + `flint_sglang-0.1.4` wheel + FA4 learned-sink 补丁）和 serve 脚本。
- 另有 `RecursiveSI/flint-int4-gguf`（单文件 25.99 GB）；卡片要求 llama.cpp 用 "a release with Flint support"，没说是上游还是 fork，也没提 LoRA。

| 路线 | 做法 | 工作量〔估算〕 | 主要风险 |
|---|---|---|---|
| **B1：Mac 上 MLX QLoRA + 合并 + 重新量化** | 用 `QATModel.native`（4-bit 冻结）给 attention 的 q / k / v / o 套 `mlx_lm.tuner` 的 LoRA 层，自写训练循环（`mlx_lm.lora` 的 CLI 不认识 `flint` 这个 model_type）。推理有两种：(i) 不合并，整条 decode 走 MLX 原生图，放弃 megakernel；(ii) 把低秩增量加到反量化后的权重上，再量化回 group-64 int4，换掉 shard，用开发用的 `--manifest` 覆盖校验 | 3–5 天 | **高。** ① MoE 层的 `SwitchGLU` 按 token 取专家权重，反向传播时内存随序列长度线性增长，能训练的序列可能只有几百 token〔估算，未验证〕，而 agent 轨迹动辄几千 token。② (ii) 是二次量化：LoRA 的增量通常小于 int4 的量化步长，可能大部分被舍入抹掉，还会破坏官方用蒸馏得到的量化质量。③ (i) 的速度未知，而且要自己写 server。④ 占 Mac GPU 很久 |
| **B2：cluster-A 上 BF16 + 自写 PyTorch modeling + PEFT** | 把 `model.py` 移植成 PyTorch（或从 `flint_sglang` wheel 里取模型代码，Apache-2.0），在 A100 上加载 BF16，PEFT LoRA 只训 attention；训完**直接合并进 BF16**（没有量化问题），在集群上 serve 合并后的权重 | 4–6 天 | **中到高。** ① 79.6 GB 的权重单张 A100 80GB 放不下，训练和推理都要 2 张，hold 里一共只有 3 张。② 移植的正确性要验证：和 rec 的 MLX 参考实现比对 logits（不要求逐位，要求误差在量化噪声内）。③ serving：SGLang kit 是 Dockerfile，要转成 Singularity 镜像；它依赖的 FA4 能不能在 A100（Ampere）上运行未知〔待核实〕，不行就得自己写推理循环，速度会差很多。④ MoE 的 LoRA 训练在 A100 上的速度未知。优点：计算节点能联网，79.6 GB 可以直接下到集群；M0 和 M1 同为 BF16、同一 serving 栈，比较干净；不占 Mac |
| **B3：GGUF + llama.cpp 运行时 LoRA** | 训练仍用 B2 的 PEFT；adapter 转成 GGUF LoRA，用 llama.cpp 在量化基座上运行时叠加（不合并、不重新量化）。可以在单张 A100 上跑（25.99 GB），也可以回到 Mac | B2 的训练部分 + 1–2 天 | **中，但有一个未知的前提**：带 Flint 支持的 llama.cpp 是哪个版本、它的 LoRA 转换脚本认不认 Flint 的张量名〔待核实〕。速度比 rec 慢（rec README 自报是 llama.cpp 的 1.4–2.6 倍）。注意基座是 int4 而 adapter 是在 BF16 上训的，存在失配 |

### 5.3 建议

1. **先做 (a)**：Flint 脚手架层自演化 3 代，在 µAR 上，夜间独占 Mac GPU 锁。这是唯一能在两三天内给出"Flint 自演化有没有二阶效应"的答案的路线。
2. **权重层的"换心"先在 Qwen3.5-9B 上做**（cluster-A，PEFT，vLLM 挂 adapter）。这条管线现有的硬件和 env 就能支撑，方法学上和 Flint 的权重层完全同构。
3. **Flint 权重层列为 stretch，走 B2，不放进这次 47 小时的 hold**（它要占 2 张卡，会把一阶和换心实验挤掉）。配三个 go / no-go 门槛，任何一个不过就停：
   - G-F1：PyTorch 移植版与 rec MLX 参考在 20 条 prompt 上的 next-token 分布一致（阈值待定）；
   - G-F2：两张 A100 上、序列长度 ≥ 4k token 时能完成一次反向传播；
   - G-F3：合并后的 BF16 能在集群上 serve，并通过 tool-call 自测，速度够跑一个 campaign。
4. **不建议 B1 的"合并 + 重新量化"**。它失败的方式是"改进者悄悄变差"，而这正好会被误读成"自演化有负效应"。如果以后一定要回到 rec 的 megakernel，必须加一个对照：把 M0 的 BF16 也用同一套朴素量化流程重做一遍（M0′），只比较 M1 与 M0′。
5. 定位提醒（来自 `rec_optimization_analysis.md` §1）：Flint 编码强、研究类 benchmark 弱（Apex 实测 FrontierScience 5.0、HLE 18.2）。把它放在"改训练脚本"这种编码型研发任务上是合理的；结果怎么解读由你定。

---

## 6. 搭建顺序

时间全部是〔估算〕。步骤编号：**UV0–UV3** 是 cluster-A上的步骤；**L0–L3** 专指审计级别（§2.5），两者不要混。"谁"一栏：**你** = 需要你批准或亲手操作；**main** = main session（唯一可以用共享 SSH master 的）；**agent** = Mac 上的 Claude Code subagent（只用 CPU，**不碰 SSH**）。

### 6.1 阶段 0：批准（现在）

| 步骤 | 谁 | 内容 |
|---|---|---|
| 0.1 | 你 | 过一遍 §0.3 的 D0–D11，尤其是 D0（端口转发）、D6（沙箱）、D9（下载清单）、D10（cluster-B分配是否释放）、D11（过夜） |

### 6.2 UV0：cluster-A 只读探测（main，约 5 分钟）

目的是把下面这些未知项变成已知。全部只读，不安装、不下载。

```bash
# ===== UV0：在 cluster-A 登录节点执行（main session 经共享 master 下发一次；也可以由你手动粘贴）=====
export R=$R
squeue -u $USER -o "%.10i %.14P %.2t %.10M %.10L %.6C %.8m %R %b"
scontrol show job <jobid> | grep -E 'JobId=|TRES=|TresPerNode|EndTime|NodeList='
sinfo -p <partition>,a100_short,a100_long -o "%P %l %D %G" 2>&1 | head
ls -la $R; df -h /scratch/scratch | tail -1
conda env list 2>/dev/null | grep -i rsi
srun --jobid=<jobid> --overlap bash -l -c '
  echo "== node $(hostname) $(hostname -I)"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  echo "== gpu processes (should be empty; if not, they belong to another project: do not touch)"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
  nproc; free -g | sed -n 2p; df -h /tmp /dev/shm | tail -n +2
  for t in git tmux gcc g++ node singularity apptainer bwrap; do printf "%-12s %s\n" $t "$(command -v $t || echo MISSING)"; done
  (type module >/dev/null 2>&1 && module -t avail 2>&1 | grep -i -E "singularity|apptainer") | head -5
  unshare -U true 2>&1 && echo userns_ok
  for u in https://huggingface.co https://pypi.org/simple/ https://github.com; do printf "%s %s\n" "$(curl -s -o /dev/null -m 8 -w "%{http_code} %{time_total}s" -I $u)" $u; done
'
```

env `rsi` 建好之后再补三行（把 `$PY` 换成该 env 的 python 路径）：

```bash
$PY -c "import torch, vllm, transformers, peft, trl; print(torch.__version__, torch.version.cuda, vllm.__version__, transformers.__version__, peft.__version__, trl.__version__)"
$PY -c "from transformers import CONFIG_MAPPING as C; print('qwen3_5' in C)"
srun --jobid=<jobid> --overlap $PY -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_capability(0))"
```

| 未知项 | 决定什么 |
|---|---|
| hold 里的 3 张 A100 是否都空着（同一账号下还有 otherproj 的训练在跑） | lane 分配；**有别的进程就不动那张卡** |
| 节点上有没有 Singularity / Apptainer / bwrap / user namespace | 集群沙箱用哪种（§2.4） |
| 节点上有没有 `gcc` | `torch.compile` 能不能用 |
| env `rsi` 的 vLLM、transformers 版本，是否认识 `qwen3_5` | serving 能不能起 |
| 确切的 GRES 名称；`a100_short` / `a100_long` 的时限 | hold 结束后的 sbatch 怎么写 |
| 节点 IP、本地盘大小 | 端口转发的目标；封存数据能不能放节点本地 |
| 登录节点是否允许 TCP 转发、能否连到计算节点的端口 | D0 是否可行。这一项要等 vLLM 起来后，main 做一次 `ssh -O forward` 才知道 |

### 6.3 需要你批准的下载、安装和 key

**cluster-A（全部在集群上下载，进 `$R/`；不从 Mac 推大文件）**

| # | 内容 | 大小 | 何时需要 |
|---|---|---|---|
| C-a | `Qwen/Qwen3.5-9B` → `$R/hf` | 19.33 GB（HF API） | 今晚 |
| C-b | `Qwen/Qwen3.8-27B` → `$R/hf` | 55.59 GB（HF API） | 阶梯阶段（明天）；可以今晚顺手下 |
| C-c | climbmix shard：训练用 0–9，dev-val 6542，sealed 6540、6541 | 每个约 92 MB（对 6542 的 HEAD 请求：91,699,537 B），共约 1.2 GB〔估算〕 | 今晚 |
| C-d | 第二个 env `rsi-train`，建在 `$R/envs/` 下：torch 2.9.1（cu128，与 Karpathy 的 `pyproject.toml` 一致）、`kernels`、`pyarrow`、`rustbpe`、`tiktoken`、`requests`、`aiohttp` | 约 6–8 GB〔估算〕 | 今晚。理由：`rsi` 里的 torch 由 vLLM 决定，版本比 Karpathy 钉的新，flash-attn3 的预编译 kernel 不一定有对应版本。如果你不想多建一个 env，就在 `rsi` 里跑训练，失败时把 attention 换成 SDPA 并记为偏离 |
| C-e | `karpathy/autoresearch` 仓库（取 `prepare.py`，用来训 tokenizer） | 很小 | 今晚 |
| C-f | flash-attn3 的预编译 kernel（A100 走 `kernels-community/flash-attn3`） | 小〔估算〕 | 今晚 |
| C-g | Singularity 基础镜像（只有站点没有现成 pytorch 镜像时才需要） | 约 5–8 GB〔估算〕 | 评估器沙箱 |
| C-h | `RecursiveSI/flint`（BF16） | 79.6 GB | 否，只有 §5 的 B2 需要 |

集群端的下载和建 env 都放在计算节点上通过 `srun --jobid=<jobid> --overlap` 做（节点能联网），不占登录节点。

**Mac**

| # | 内容 | 大小 | 何时需要 |
|---|---|---|---|
| M-a | 启动 Docker Desktop（或决定用 `sandbox-exec`）；agent 容器的基础镜像 | 约 2–4 GB〔估算〕 | 今晚 |
| M-b | OpenCode（rec 钉住的 patched 版） | 86,372,353 B | 今晚 |
| M-c | 平台自己的 uv venv：`aiohttp`、`mcp`、`openai`、`pyyaml`、`numpy` 等 | 约 0.2 GB〔估算〕 | 今晚 |
| M-d | `duckduckgo-mcp-server`、`mcp-server-fetch`（`uvx`） | 很小 | 今晚 |
| M-e | pi 0.73.1（npm）、`mini-swe-agent` | 约 0.2 GB〔估算〕 | 只有 OpenCode 过不了 G1b 时 |
| M-f | 可选的 key：`TAVILY_API_KEY` 或 `BRAVE_API_KEY` | — | 只有 DDG 不稳定时 |
| M-g | （后备）mlx-lm 0.32.2 的 venv + MLX 4-bit 权重 | 约 1 GB + 5.98 GB + 16.07 GB | 现在不需要 |

Mac 磁盘：现在可用 67 GiB；M-a 到 M-e 合计约 3–5 GB〔估算〕。可以回收的空间（**是否处理由你定，我没有动**）：Flint int4 现在有两份（`rsi-report/flint-int4` 的 `du` 是 42 G，`~/.local/share/rec/models` 是 22 G）；`~/.cache/huggingface/hub` 是 203 G。

### 6.4 今晚（11-17）的顺序

| 时间 | 步骤 | 谁 | 内容 | 产出 |
|---|---|---|---|---|
| 20:40 | UV0 | main | §6.2 的探测 | 未知项清单 |
| 20:45 起 | UV1 | main | 在 tmux 里用 `srun --overlap` 下载 C-a、C-c、C-e、C-f，建 C-d（下方模板）；C-b 另开一个窗口 | 权重、数据、训练 env |
| 20:45–21:30 | B1 | agent | 写 `cluster_bundle/` 第一版：`run_vanilla.sh`（原版 `vanilla_transformer.py` + 沙箱外 wall-clock 计时 + 每个 seed 一行 CSV）、`serve_vllm.sh`、`toolcall_test.py`（约 30 个脚本化 episode） | bundle v1（纯文本，几十 KB） |
| 21:30 | UV2 | main | 推 bundle（很小）；tmux 里启动：GPU1 跑 vanilla seed 0–9，GPU2 跑 seed 10–19，GPU0 起 vLLM 9B；建立端口转发；从 Mac 跑 `toolcall_test.py` | 噪声地板、tool-call 合法率、经隧道的延迟和吞吐 |
| 21:30–00:00 | B2 | agent | 写最小闭环：orchestrator（只含 `karpathy_greedy`，带暂停 / 续跑）、gateway、`lab` MCP、`evald`（契约版评估器 + L0 / L1 审计 + 两个沙箱）、Mac agent 沙箱、OpenCode 配置；用假模型和 CPU 小任务在 Mac 上跑通单元测试 | 平台 v0 |
| ~22:45 | G1 | 你 + main | 20 个 seed 跑完后看 §6.8 的 G1a–G1c | go / no-go |
| ~00:00 | UV3 | main | 启动过夜的一阶 campaign：9B，`karpathy_greedy`，2 条线程各占一条评估 lane。**B2 没写完就不启动**，明早再开——hold 到 11-19 20:01，不差这几个小时 | 第一条一阶曲线 |
| 过夜 | — | 自动 | 每条线程每小时约 6 个 trial（agent 思考 2–3 分钟 + 评估约 7 分钟）〔估算〕；到早上 8 点每条线程约 45 个 trial | `ledger.jsonl`、轨迹 |

**UV1 模板**（集群端；路径和 env 名以 UV0 的结果为准，没有在集群上验证过）

```bash
# ===== UV1：登录节点的 tmux 里启动；实际工作在计算节点上 =====
export R=$R
mkdir -p $R/{hf,data/train,data/devval,data/sealed,envs,tools,bundle,spool,runs,adapters,run,out,_to_delete}
cat > $R/out/uv1.sh <<'EOF'
#!/bin/bash
set -u
export R=$R HF_HOME=$R/hf
PY=${PY:?set PY to the python of env rsi}
# (1) 模型
$(dirname $PY)/hf download Qwen/Qwen3.5-9B || $(dirname $PY)/huggingface-cli download Qwen/Qwen3.5-9B      # 19.33 GB
$PY -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('Qwen/Qwen3.5-9B').model_type)"  # 期望 qwen3_5
# (2) NanoChat 数据：train / dev-val / sealed 分开放
B=https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main
for i in 0 1 2 3 4 5 6 7 8 9; do f=shard_$(printf %05d $i).parquet; [ -s $R/data/train/$f ] || curl -L --fail -o $R/data/train/$f $B/$f; done
[ -s $R/data/devval/shard_06542.parquet ] || curl -L --fail -o $R/data/devval/shard_06542.parquet $B/shard_06542.parquet
for i in 6540 6541; do [ -s $R/data/sealed/shard_0$i.parquet ] || curl -L --fail -o $R/data/sealed/shard_0$i.parquet $B/shard_0$i.parquet; done
[ -d $R/tools/autoresearch ] || git clone --depth 1 https://github.com/karpathy/autoresearch $R/tools/autoresearch
# (3) 训练 env（建在本项目目录内，不碰别的 env）
[ -x $R/envs/rsi-train/bin/python ] || conda create -y -p $R/envs/rsi-train python=3.12
$R/envs/rsi-train/bin/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
$R/envs/rsi-train/bin/pip install "kernels>=0.11.7" "pyarrow>=21" rustbpe tiktoken requests numpy aiohttp
$R/envs/rsi-train/bin/python -c "import torch, kernels, rustbpe, tiktoken, pyarrow; print('train env ok', torch.__version__, torch.cuda.is_available())"
# (4) flash-attn3 kernel 预取（失败不致命：改用 SDPA 并记为偏离）
$R/envs/rsi-train/bin/python -c "from kernels import get_kernel; get_kernel('kernels-community/flash-attn3')" || echo KERNEL_PREFETCH_FAILED
du -sh $R/hf $R/data/* $R/envs/* | tee $R/out/UV1_sizes.txt; echo UV1_DONE
EOF
tmux new -d -s rsi-uv1 "PY=<env rsi 的 python> srun --jobid=<jobid> --overlap bash $R/out/uv1.sh 2>&1 | tee $R/out/uv1.log"
```

tokenizer 的训练（`prepare.py`，CPU 约 2 分钟）是 `run_vanilla.sh` 的第一步，也在计算节点上。

**服务的启动与端口转发**（形状；具体参数随 bundle 定稿）

```bash
# —— 集群端：登录节点的 tmux 里，各一个窗口 ——
export R=$R
tmux new -d -s rsi-vllm  "srun --jobid=<jobid> --overlap bash $R/bundle/serve_vllm.sh 9b 0"      # GPU0；端口写到 $R/run/ports
tmux new -d -s rsi-evald "srun --jobid=<jobid> --overlap bash $R/bundle/evald.sh 1,2"            # GPU1、GPU2 两条 lane

# —— Mac 端：只由 main session 执行，各一次 ——
ssh -O check -S ~/.ssh/<control-socket> x
ssh -S ~/.ssh/<control-socket> -O forward -L 127.0.0.1:18000:<node>:<vLLM 端口> -L 127.0.0.1:18100:<node>:<evald 端口> x
```

要点：

- vLLM 和 evald 必须监听节点的网卡（登录节点才转发得过去），所以同一集群上的其他用户理论上也能连到这两个端口。两者都要带 token 校验；token 是集群上随机生成的一个文件（`$R/run/token`，权限 600），通过环境变量传给进程，**不写进命令行、日志和本文**；Mac 上的 gateway 和 lab 从本机的一个 600 权限文件读取。端口取随机高位端口，避免和别人撞。
- 转发只加一次。之后 orchestrator 不再调用 ssh。
- 推 bundle 是唯一需要传文件的地方，内容是几十 KB 的脚本，由 main 做。

### 6.5 这次 hold 的算力预算（到 11-19 20:01）

可用量：3 张卡 × 约 47 小时。GPU0 常驻 serving，所以**评估 lane 是 2 条 × 约 44 小时 ≈ 88 lane·小时**。每次 5 分钟评估按 7 分钟算，每条 lane 每小时约 8.5 次〔估算〕，上限约 750 次评估。下表是建议的分配，每一项都是〔估算〕，UV2 实测后重算。

| 用途 | 评估次数 | lane·小时 | 何时 | 说明 |
|---|---|---|---|---|
| S0 噪声地板 | 20 | 2.5 | 今晚 | 两条 lane 各 10 个 seed |
| S1 `karpathy_greedy`，9B，2 线程 | 90 | 15（含 agent 思考时间） | 今晚过夜 | 每线程约 45 个 trial |
| S1 `recursive_first_steps`，9B，K=2，含换 seed 确认和合并 | 100 | 14 | 11-18 白天 | — |
| S1 `naive_trial_ledger`，9B | 80 | 11 | 11-18 白天 | 含 exploratory trial |
| 阶梯：27B，`karpathy_greedy`，1 线程 | 40 | 7 | 11-18 傍晚 | **GPU2 改跑 27B，只剩 1 条评估 lane** |
| 换心：GPU2 训 M1 和 placebo M1p | — | 3–4（GPU2 不能评估） | 11-18 夜间 | SFT 数据来自前面所有 9B 轨迹 + µAR 轨迹 |
| 换心之后：M0 / M1 / M1p 在 NanoChat 上"从前沿接着跑"，3 臂 × 3 次 × 12 个 trial | 108 | 15 | 11-19 上午 | 设计稿 §1.6 允许的两种二阶设计之一 |
| `apex_ai4ai` 的最终重训（更长的训练 × 3 个 seed × 3 个研究者） | 9 次长训练 | 5 | 11-19 下午 | 重训预算的具体时长由你定 |
| 终点确认：3 个最好方案 × 10 个 seed | 30 | 3.5 | 11-19 下午 | 对应 Recursive 的 10-seed 均值 |
| 余量 | — | 约 12 | — | 崩溃、重跑、链路中断 |

与此并行、不占评估 lane 的：**S2（µAR 的二阶测量）**全程在 Mac CPU 上评估，改进者请求打到 GPU0 的同一个 vLLM。脚手架层的各臂（A0a、A0b、A1、A1p）11-18 白天就能跑；权重层的臂（A2、A2p）等 11-18 夜间的 adapter。

### 6.6 hold 结束之后

用 sbatch 另开作业，campaign 可断点续跑：

```bash
#!/bin/bash
#SBATCH -p <partition>,a100_short,a100_long
#SBATCH --exclude=<node>,<node>,<node>,<node>
#SBATCH --job-name=rsi --nodes=1 --cpus-per-task=36 --mem=360G
#SBATCH --gres=gpu:<UV0 确认的名称>:3
#SBATCH --time=<UV0 确认各分区的上限后填>
#SBATCH --output=$R/out/%x-%j.log
export R=$R HF_HOME=$R/hf
bash $R/bundle/start_services.sh      # 起 vLLM 和 evald，把节点名和端口写到 $R/run/ports，然后等待
```

新作业的节点名会变，main 需要把旧转发取消、按新节点重建一次。

### 6.7 后备：孤岛模式

如果 D0 的端口转发不可行（登录节点禁了 TCP 转发、master 频繁掉线、或你不同意），就把 orchestrator、gateway、harness 也放到计算节点上跑：节点能联网，所以 search / fetch 仍然可用；Mac 只在 campaign 结束后取回账本和轨迹（纯文本，压缩后预计几十到几百 MB〔估算〕，按 285 KB/s 要几分钟到二十分钟）。代价：OpenCode 要换成上游的 linux-x64 版〔待核实：下载地址〕；agent 沙箱要在集群上再做一层；agent 与封存数据之间不再是物理隔离，只剩容器隔离。平台代码按"spool 协议 + HTTP"写，两种模式只差部署位置。

### 6.8 Gate（阈值是建议值，D7 由你定）

| Gate | 何时 | 条件 | 不过怎么办 |
|---|---|---|---|
| G0 | UV0 之后 | hold 里 3 张卡都空；env `rsi` 认识 `qwen3_5`；节点上有可用的沙箱手段 | 卡被占：只用空的卡，重排 §6.5。不认识 `qwen3_5`：升级 `rsi` 里的 transformers / vLLM（这是本项目自己的 env）。没有容器：按 §2.4 的最低配置，并在报告里写明 |
| G0b | vLLM 起来后 | 端口转发可用；经隧道的首 token 延迟和吞吐可接受 | 孤岛模式（§6.7） |
| G1a | 20 个 seed 跑完 | vanilla 在 A100 上能跑完；标准差 ≤ 0.02 BPB；单次总用时 ≤ 10 分钟 | 跑不通：换 SDPA 再试。噪声过大：keep 一律要求换 seed 确认；仍不行则 NanoChat 降级，一阶也改用 µAR |
| G1b | 同上 | 9B 经真实 harness（Mac 上的 OpenCode）：tool call 的 schema 合法率 ≥ 90%，脚本化 episode 完成率 ≥ 70% | 依次尝试：换 parser → `tool_choice="required"` 加 guided decoding → 换 pi → 换 mini-swe-agent。都不行：9B 只做 µAR（JSON 提案），NanoChat 交给 27B |
| G1c | UV3 之前 | 契约版评估器对同一个 checkpoint 算出的 BPB 与原版 `lib.evaluate_bpb` 一致；L1 审计的自测全部通过（故意作弊的样例被抓住）；Mac 沙箱的"应被拒绝"自测全过 | 不启动 UV3 |
| G2 | S2 开始前 | 设计稿的 Gate 1、Gate 2（合法率、M0 是否好于局部扰动基线） | 按设计稿的退路 |
| G3 | 换心之前 | 被接受的步骤 ≥ 40 条（设计稿的下限）；SFT 集与 held-out 的 hash 核对通过 | 加倍 G0 代的 seed，或 hindsight 重标 |
| G4 | 每一代 | hack 率、dev–sealed 差没有上升 | 暂停晋升，先加固评估器 |

---

## 7. 风险

| # | 风险 | 表现 | 控制 | 残余 |
|---|---|---|---|---|
| 1 | **9B 的 tool calling 不可靠** | 参数名写错、重复同一个无效调用、长轨迹后格式崩掉、subagent 调不起来 | G1b；rec 式的 schema 提醒；`codemode: false`；精简工具集；vLLM 的 guided decoding；退到 pi / mini-swe-agent；多 agent 放在 orchestrator 层 | 9B 也许只能胜任 µAR 的 JSON 提案。这本身是一条可报告的结果（能力阶梯的最低一档） |
| 2 | **沙箱逃逸、读取封存数据** | agent 读到 val shard、封存 seed 或评估器代码 | agent 在 Mac、封存数据在集群，物理隔离；agent 沙箱连不到转发端口；集群上 val 不进训练沙箱；sealed 分数不回传 | 集群上沙箱内外同一用户，容器不是安全边界 |
| 3 | **篡改计时或评估** | 自报时间造假、返回假 loss、偷看未来、打分时自适应 | §2.5 的 1–10 条；裁决在可信区；L0–L3 审计 | 没想到的作弊方式。所以每代报 hack 率和 dev–sealed 差，并人工抽查最大增益 |
| 4 | **agent 的代码伤到同账号下的其他项目** | 提交的 `train.py` 读写 `$SCRATCH/otherproj*`，或占用别人的 GPU | 训练沙箱只 bind 本项目的目录；L0 静态检查；`CUDA_VISIBLE_DEVICES` 固定；UV0 探测先确认卡是空的 | 没有容器时只剩静态检查，强度不够，这种情况下建议不开放代码级动作空间 |
| 5 | **SSH master 中断** | 过夜时链路断开，campaign 停住 | orchestrator 暂停而不是记失败；evald 幂等，结果不丢；main 按规则最多重开一次并重建转发 | 无人值守时段可能白白停几个小时 |
| 6 | **端口暴露在集群内网** | 其他用户连到 vLLM 或 evald | token 校验；随机端口；token 不进命令行和日志 | — |
| 7 | **Mac GPU 争用** | Flint 臂干扰 `bench/` 的 tok/s 测量 | 主路径完全不用 Mac GPU；Flint 臂一律先 `gpu_lock.sh acquire` | Flint 臂只能在夜间跑 |
| 8 | **磁盘** | Mac 只剩 67 GiB | 大文件都在集群；Mac 上只有轨迹（文本）和约 3–5 GB 的工具 | 长时间运行后 gateway 日志变大，按天压缩 |
| 9 | **慢链路** | 285 KB/s | 只走文本报文：100k token 的上下文约 400 KB，每次请求多 1–2 秒〔估算〕；checkpoint 和 adapter 不过链路；SFT 数据压缩后上传 | 长上下文 session 的每步延迟会明显增加 |
| 10 | **hold 到期** | 11-19 20:01 进程被杀 | 账本逐 trial 落盘；§6.5 留了余量；之后用 sbatch 续跑 | sbatch 的排队时间未知 |
| 11 | **硬件不可比** | A100 的 5 分钟和 B200、H100 的不是同一个预算 | 所有比较都在同型号 A100 lane 内做；不与 Recursive 的数字直接比 | 报告里要写明 |
| 12 | **两套 serving 栈不可比** | MLX 4-bit 的 9B 与 BF16 的 9B 行为不同 | D1；M0 / M1 的比较始终在同一栈内 | — |
| 13 | **污染** | 9B 的预训练里可能已有 NanoChat 的公开技巧；有网时可以直接抄 | 代际比较共用基座，这部分被差分掉；§2.5 第 11 条的屏蔽表和日志；web-off 对照臂 | 一阶曲线的绝对水平不干净，Recursive 自己也承认这一点 |
| 14 | **依赖的版本细节没核实** | vLLM 对 `qwen3_5` 的支持和 LoRA 参数、OpenCode 的 `--format json` / `export`、flash-attn3 在 Ampere 上的预编译包 | 都列进了 UV0 / UV2 / G0 / G1；每项都有退路 | 今晚可能在环境上耗掉半小时以上 |
| 15 | **过夜无人值守** | Mac 休眠、网络切换 | D11；启动前做一次 10 分钟的端到端试跑 | — |
| 16 | **授权范围** | 端口转发让 orchestrator 的流量持续经过共享 master，可能超出你对"只由 main session 驱动"的本意 | D0 单独请你确认；孤岛模式作为后备 | — |

---

## 8. 没有做、没有核实的事（不要当作事实）

- 我没有登录或探测任何集群。cluster-A 和cluster-B 的事实全部来自协调者消息、`hpc_allocations.md` 和 memory 记录。
- 改用 cluster-A 并允许自动操作这件事，我没有看到你的原话（见 §0.2）。
- 没有运行任何模型、harness 或训练脚本。所有吞吐、用时、合法率、评估次数都是〔估算〕或待测。
- 登录节点是否允许 TCP 转发、能否连到计算节点的端口：未知。
- cluster-A 节点上有没有 Singularity / Apptainer / bwrap，能不能给容器断网：未知。
- env `rsi` 的具体版本、是否认识 `qwen3_5`：未知。vLLM 的 LoRA 挂载参数、API key 的环境变量名：没有查。
- `a100_short` / `a100_long` / `<partition>` 的时限和 GRES 名称：未知。
- OpenCode 的 `opencode run --format json`、`opencode export`、subagent 能否并行：官方页面没写全，部分来自第三方文档。
- mlx_lm.server 的 tool calling 与 `--adapter-path`：`SERVER.md` 没有写（现在只是后备）。
- Aider 的能力：凭已有知识，没有抓文档。OpenHands 的 MCP 与 delegation 在 headless 下的配置：没有查到官方说明。mini-swe-agent v2 是否默认使用原生 tool calling：来源互相矛盾。
- 带 Flint 支持的 llama.cpp 是哪个版本、是否支持 Flint 的 LoRA；FA4 能否在 A100 上运行：未知。
- `SwitchGLU` 训练时的内存估计（§5.2 B1）：读代码后的推断，没有验证。
- `sandbox-exec` 在 macOS 26.6 上的行为：没有试。
- NanoChat 在 B200 上的标准差（约 0.010）：沿用设计稿里上一位 agent 的计算，不是仓库自报。
- `prepare.py` 默认下载多少个 shard：没有确认，本文按 10 个写。
- UV0、UV1 和附录 A 的命令块只做过 `bash -n` 语法检查，没有在任何集群上执行过。

## 9. 本次查过的来源

| 用于 | 来源 |
|---|---|
| mlx-lm 最新版本、依赖、发布日期 | https://pypi.org/pypi/mlx-lm/json ；GitHub API `repos/ml-explore/mlx-lm/releases` |
| `qwen3_5.py` 是否存在 | GitHub API `repos/ml-explore/mlx-lm/contents/mlx_lm/models` |
| mlx_lm.server 文档 | https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md |
| 模型大小、`model_type`、pipeline | HF API：`Qwen/Qwen3.5-9B`、`Qwen/Qwen3.8-27B`、`mlx-community/Qwen3.5-9B-4bit`、`mlx-community/Qwen3.8-27B-4bit`、`RecursiveSI/flint-int4-gguf`；对应的 HF 页面 |
| OpenCode | https://opencode.ai/docs/cli/ 、/docs/agents/ 、/docs/tools/ 、/docs/mcp-servers/ 、/docs/websearch/ ；本地 `rec/cli/clients/opencode/pins.json`、`rec/cli/launch.go:321-364`、`clients/opencode/instructions.md`、`rec-home.patch` |
| pi | https://unpkg.com/@mariozechner/pi-coding-agent@0.73.1/README.md |
| OpenHands | https://docs.openhands.dev/openhands/usage/cli/headless |
| mini-swe-agent | 搜索结果（https://docs.litellm.ai/docs/projects/mini-swe-agent/ 等，二手） |
| Claude Code 与第三方端点的 WebSearch | 搜索结果（https://docs.litellm.ai/docs/tutorials/claude_code_websearch 等，二手） |
| vLLM 的 Qwen3.5 tool calling | 搜索结果（官方 recipe https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html 的摘要，以及社区帖子） |
| DuckDuckGo MCP | 搜索结果（`nickclyde/duckduckgo-mcp-server` 的多个目录页，二手） |
| Karpathy autoresearch | https://github.com/karpathy/autoresearch ；raw 的 `program.md`、`prepare.py`、`pyproject.toml` |
| Flint GGUF | https://huggingface.co/RecursiveSI/flint-int4-gguf |
| Mac 的连通性（2026-11-17 约 19:57，`curl -I`） | huggingface.co 200（0.68 s）；hf-mirror.com 200；pypi.org 200（5.3 s）；github.com 200；html.duckduckgo.com 400（5.6 s）；bing.com 200（5.6 s） |
| 集群的事实 | 协调者在 20:00–20:10 之间转述的四条消息；`research/hpc_allocations.md`；memory `hpc-manual-only`（20:06 更新） |
| 本地文件 | `research/platform_design.md`、`research/reading/00–04`、`research/code_map_first_steps.md`、`research/code_map_rec.md`、`research/artifacts_recursive.md`、`research/artifacts_weights_configs.md`、`research/lit_signal_hardness.md`、`research/rec_optimization_analysis.md`、`rec/engine/src/rec_mlx/{model,quantized,checkpoint}.py`、`first-steps-toward-automated-ai-research/nanochat_autoresearch/` |

---

## 附录 A：cluster-B（次要、可选，只能由你手动粘贴）

现状：job <jobid>（<node>，4 × H20，每张 97,871 MiB）和 <jobid>（<node>，1 × A800 80GB PCIe），约 02:00 到期；计算节点不能联网；登录节点能联网；`HF_HOME=$SCRATCH/.cache/huggingface`；cluster FS 剩余 489T。现成 env：`OPD`（torch 2.11.0、vllm 0.25.1、transformers 5.14.1，没有 peft / trl）、`OPD_a800`（torch 2.9.0+cu128、vllm 0.11.2、transformers 4.57.6）、`agent`（torch 2.7.1、vllm 0.10.0、peft 0.17.0，多半太旧）。

**建议（D10）：释放。** 理由：Mac 连不上这些卡（不允许建隧道），它们只能跑孤岛任务；每一步都要你手动粘贴；H20 和 A800 上的分数与 A100 不可比，不能并入主实验。

如果你想用掉这 6 小时，唯一划算的用法是**一组独立硬件上的 vanilla 噪声地板**（A800 和 H20 各 10 个 seed）：需要你粘贴 4 次（探测、下载、启动、取结果）。原则是复用 `OPD` / `OPD_a800`，不新建 env、不往里面装东西，缺的小包用 `pip install --target` 装到 overlay 目录。下面两个块是为这个用途准备的，只做过语法检查。

**H0：只读探测块**（会创建一个目录和四个小文本文件；不安装、不下载、不提交新作业）

你已经手动查过的（GPU 型号、驱动、module 列表、计算节点无网、各 env 的主要版本）不再重复，这个块只查**还不知道的**。

```bash
# ===== H0：在 cluster-B 登录节点上粘贴 =====
export SCR=$SCRATCH/rsi E=$SCRATCH/.conda/envs
mkdir -p $SCR/out && cd $SCR

cat > probe.sh <<'EOF'
#!/bin/bash
E=$SCRATCH/.conda/envs
echo "===== NODE $(hostname) $(date +%FT%T)"
echo "### slurm env"; env | grep -E '^SLURM_(JOB_ID|JOB_GPUS|CPUS_ON_NODE|MEM_PER_NODE)=|^CUDA_VISIBLE_DEVICES='
echo "### cpu/mem"; nproc; free -g | sed -n 2p
echo "### local disk"; df -h /tmp /dev/shm 2>&1 | tail -n +2
echo "### scratch writable"; touch $SCRATCH/rsi/out/.w_$(hostname) && echo yes
echo "### tools"; for t in git tmux node npm bwrap gcc g++ srun squeue; do printf "%-10s %s\n" $t "$(command -v $t || echo MISSING)"; done
echo "### singularity"; type module >/dev/null 2>&1 && module load Singularity/4.3.1 2>/dev/null; (singularity --version || apptainer --version) 2>&1 | head -1
echo "### userns"; unshare -U true 2>&1 && echo userns_ok
echo "### torch sees gpu?"
for e in OPD OPD_a800; do printf "== %s: " $e
  $E/$e/bin/python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_capability(0))" 2>&1 | tail -1
done
echo "### loopback"; $E/OPD/bin/python -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print('loopback_ok',s.getsockname()[1])"
EOF

cat > pkgs.py <<'EOF'
import importlib.metadata as m
out = []
for k in ["peft","trl","bitsandbytes","kernels","flash_attn","pyarrow","rustbpe","tiktoken","regex","requests","openai","huggingface_hub","mcp","litellm","aiohttp","xgrammar"]:
    try: out.append(k + "=" + m.version(k))
    except Exception: out.append(k + "=MISSING")
print(" ".join(out))
EOF

{
echo "===== LOGIN $(hostname) $(date +%FT%T)"
echo "### jobs"; squeue -u $USER -o "%.10i %.9P %.2t %.10M %.10L %.6C %.8m %R %b"
echo "### tres"; for j in <jobid> <jobid>; do scontrol show job $j | grep -E 'JobId=|TRES=|TresPerNode|EndTime'; done
echo "### login net"; for u in https://huggingface.co https://hf-mirror.com https://pypi.org/simple/ https://github.com https://registry.npmjs.org; do printf "%s %s\n" "$(curl -s -o /dev/null -m 8 -w '%{http_code} %{time_total}s' -I $u)" $u; done
echo "### login tools"; for t in conda pip uv git tmux node npm rsync; do printf "%-8s %s\n" $t "$(command -v $t || echo MISSING)"; done; ls $E/OPD/bin | grep -E '^(hf|huggingface-cli|vllm)$'
echo "### module detail (找容器镜像路径)"; for m in $(module -t avail 2>&1 | grep -i -E '^(pytorch|Singularity)/' | head -5); do echo "--- $m"; module show $m 2>&1 | grep -i -E 'sif|img|singularity|prepend|setenv|alias' | head -6; done
echo "### hf cache (HF_HOME=$HF_HOME)"; ls $SCRATCH/.cache/huggingface/hub 2>/dev/null | grep -i -E 'qwen|flint|climbmix|flash'
echo "### extra packages"; for e in OPD OPD_a800; do printf "== %s: " $e; $E/$e/bin/python pkgs.py 2>&1 | tail -1; done
echo "### OPD: transformers 认识 qwen3_5 吗"; $E/OPD/bin/python -c "from transformers import CONFIG_MAPPING as C; print('qwen3_5' in C, [k for k in C if k.startswith('qwen')])" 2>&1 | tail -1
echo "### OPD: vllm tool parsers"; $E/OPD/bin/python -c "from vllm.entrypoints.openai.tool_parsers import ToolParserManager as T; print(sorted(T.tool_parsers))" 2>&1 | tail -1
} 2>&1 | tee out/probe_login.txt

srun --jobid=<jobid> --overlap bash -l $SCR/probe.sh 2>&1 | tee out/probe_<node>.txt
srun --jobid=<jobid> --overlap bash -l $SCR/probe.sh 2>&1 | tee out/probe_<node>.txt
echo "DONE: 把 out/probe_login.txt、out/probe_<node>.txt、out/probe_<node>.txt 的内容贴回来"
```

这个块要回答的未知项：

| 未知项 | 决定什么 |
|---|---|
| `OPD` / `OPD_a800` 在计算节点上能不能看到 GPU | 能不能直接复用（G0） |
| `OPD` 的 transformers 是否认识 `qwen3_5`；vLLM 0.25.1 有哪些 tool parser | serving 能不能起、parser 用哪个 |
| 两个 env 里缺哪些小包（`kernels`、`rustbpe`、`tiktoken`、`peft`、`trl`、`mcp`…） | overlay 要装什么 |
| HF cache 里是否已经有 Qwen 权重 | 省不省 19–75 GB 的下载 |
| pytorch module 背后的容器镜像路径 | Singularity 沙箱的基础镜像 |
| 确切的 TRES / GRES 名称 | 明天的 sbatch 怎么写 |
| 计算节点上有没有 `gcc`、`node`、`bwrap`、user namespace | `torch.compile` 是否可用；pi 能不能跑；沙箱的备选 |
| 节点本地盘（`/tmp`）大小 | 封存数据能不能放在节点本地 |
| 登录节点到 HF、镜像、PyPI、npm 的连通和延迟 | 用哪个源 |

最后两行里 transformers 和 vLLM 的 import 在登录节点上各要十几秒〔估算〕。如果某一行报 ImportError（这些 API 路径随版本变过），不影响其他结果，原样贴回即可。

**H1 模板**（在 H0 之后定稿；这里的命令没有在集群上验证过。原则：复用 `OPD` / `OPD_a800`，新东西只进 `$SCR` 和现有的 HF cache）

```bash
# ===== H1：登录节点，tmux 里运行。预计 20–60 分钟〔估算〕，主要是模型下载 =====
tmux new -s rsi            # 断线后：tmux attach -t rsi
export SCR=$SCRATCH/rsi E=$SCRATCH/.conda/envs
export HF_HOME=$SCRATCH/.cache/huggingface      # 沿用现有 cache
# 若 H0 显示 huggingface.co 慢或不通：export HF_ENDPOINT=https://hf-mirror.com
mkdir -p $SCR/{overlay/train,overlay/lora,data/{train,devval,sealed},tools,bundle,out}

# (1) 模型，进现有 HF cache（H0 显示已有则跳过；旧版 CLI 把 hf 换成 huggingface-cli）
$E/OPD/bin/hf download Qwen/Qwen3.5-9B        # 19.33 GB
$E/OPD/bin/python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('Qwen/Qwen3.5-9B').model_type)"   # 期望 qwen3_5
$E/OPD/bin/hf download Qwen/Qwen3.8-27B       # 55.59 GB，阶梯臂，可以另开 tmux 窗口并行下

# (2) overlay：不改动任何现有 env；具体包名以 H0 的 MISSING 列表为准
$E/OPD_a800/bin/pip install --target $SCR/overlay/train --no-deps kernels rustbpe tiktoken
PYTHONPATH=$SCR/overlay/train $E/OPD_a800/bin/python -c "import kernels, rustbpe, tiktoken, pyarrow, requests; print('train overlay ok')"
$E/OPD/bin/pip install --target $SCR/overlay/lora --no-deps peft trl
PYTHONPATH=$SCR/overlay/lora $E/OPD/bin/python -c "import peft, trl; print('lora overlay ok', peft.__version__)"
# 备选（更干净但更慢）：conda create -y -p $SCR/env/opd-lora --clone $E/OPD && $SCR/env/opd-lora/bin/pip install peft trl

# (3) NanoChat 数据：train / dev-val / sealed 分开放
B=${HF_ENDPOINT:-https://huggingface.co}/datasets/karpathy/climbmix-400b-shuffle/resolve/main
for i in 0 1 2 3 4 5 6 7 8 9; do f=shard_$(printf %05d $i).parquet; curl -L --fail -o $SCR/data/train/$f $B/$f; done
curl -L --fail -o $SCR/data/devval/shard_06542.parquet $B/shard_06542.parquet
for i in 6540 6541; do curl -L --fail -o $SCR/data/sealed/shard_0$i.parquet $B/shard_0$i.parquet; done
git clone --depth 1 https://github.com/karpathy/autoresearch $SCR/tools/autoresearch

# (4) flash-attn3 kernel 预取（失败不致命：运行时改用 SDPA 并记为偏离）
PYTHONPATH=$SCR/overlay/train $E/OPD_a800/bin/python -c "from kernels import get_kernel; [get_kernel(r) for r in ('kernels-community/flash-attn3','varunneal/flash-attention-3')]" || echo KERNEL_PREFETCH_FAILED

du -sh $SCR/overlay/* $SCR/data/* | tee $SCR/out/H1_sizes.txt; echo H1_DONE
```

`--no-deps` 是为了不让 pip 顺手把另一份 torch 或 transformers 拉进 overlay。代价是可能漏掉小依赖：上面两行 import 自检如果报缺包，就把缺的那个包同样用 `--target` 补进去。计算节点上运行时统一加 `HF_HUB_OFFLINE=1` 和对应的 `PYTHONPATH`。

启动和取结果的块等 `cluster_bundle/` 里的 `run_vanilla.sh` 写好后再给（它和 cluster-A 用的是同一个脚本）。结束时：`scancel <jobid> <jobid>`。

---

## 决策记录（2026-11-17 约 20:30，用户拍板）

| # | 决定 |
|---|---|
| D0 | **孤岛模式**：整个系统（orchestrator、harness、vLLM、evald、搜索 MCP）都跑在 cluster-A 的 hold 节点 <node> 上，Mac 不在运行回路里，只负责写代码、推送 bundle、拉回结果。需要的现成 harness 想办法装到集群上 |
| D1 | 用 BF16 完整版加 vLLM，不用 MLX。必要时 llama.cpp 也可以 |
| D2 | 首选 OpenCode，有更好的也可以换。**新增要求：记录 agent 之间的协同过程，做成类似 GIF 或视频的回放，展示整个 agent 群是怎么活动的** |
| D4 | 同意（DDG MCP + fetch，屏蔽答案仓库，另设一个 web-off 臂） |
| D5 | 按推荐：以进程总 wall-clock 为准 |
| D6 | 必须有沙箱；Mac 上没有 Docker，由 main 决定。UV0 结果：节点上有 `bwrap`、user namespace 可用、有 `apptainer/1.4.5` module → 集群端用 bwrap |
| D9 | 下载和安装全部批准（Mac 端可用空间约 100 GB 以内） |
| D10 | cluster-B的 4 × H20（作业 <jobid>，2 天）保留作备用 |
| D11 | 用户通宵在场，Mac 不休眠 |

## UV0 结果（2026-11-17）
- hold 节点：<node>，IP <ip>；36 CPU，503 GB 内存（约 419 GB 空闲）；**本地 NVMe `/tmp` 有 5.8 TB**，`/dev/shm` 252 GB；3 张 A100 上没有其他进程。
- 有 `gcc 15.2`、`tmux`、`bwrap`、`unshare -U`；需要 apptainer 时用 `module load apptainer/1.4.5`；**git 要先 `module load git`**；**节点上没有 node/npm**（OpenCode 用独立二进制，MCP 用 Python 版的 uvx）。
- 分区时限：<partition> 和 a100_long 为 28 天，a100_short 为 3 天；GRES 名为 `gpu:a100`。
- UV1 已在 tmux `rsi-uv1` 中启动（07:25 EST），日志在 `$R/out/uv1.log`，依次下载 9B、climbmix 分片、autoresearch，建 rsi-train 环境，最后下载 27B。
