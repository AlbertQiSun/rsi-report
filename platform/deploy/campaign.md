# 过夜 campaign：什么时候更新、更新几次、怎么启动和看护（worker F）

计划文件是 `campaign/overnight.yaml`，调度器是 `rsi/campaign/`。下表就是 yaml 里写死的计划。**带〔估算〕的时长都没有实测**，来源是 `research/platform_build_plan.md` §6.4–6.5 和协调者给的数：一次 NanoChat 评估约 7 分钟；9B 每个 trial 思考 2–3 分钟；一次 LoRA 训练约 45 分钟（30–60）；一步 µAR 约 8 秒。闸门跑完后要换成实测值，再用 `--check` 看一遍时间。

## 时间表（北京时间；具体钟点都是〔估算〕，实际由触发条件推进）

**资源**

| 资源 | 用途 |
|---|---|
| <node> | GPU0 = 9B vLLM；lane 1、2；36 个 CPU 跑 µAR |
| <node> | lane 3、4、5 |
| <node> | 27B（`ladder-27b`） |

5 条 lane 合计约 42 次评估/小时〔估算〕。每个 run 固定用指定的 lane（线程 i 用 lanes[i]）：

- **lane 2 是 LoRA 专用 lane**：过夜只用来训 adapter，G0 不依赖它。LoRA 必须在本机 GPU 上训，所以选本机的 lane 2，不选 <node> 上的。
- 9B G0 占 lane 1、3、4，一个线程一条。
- 27B 阶梯臂的两个线程都在 lane 5 上。

| 时间 | 跑什么（lane） | 迭代 / 次数 | 这一步自动做的决定 |
|---|---|---|---|
| 11-17 22:30 | 部署平台（不属于 campaign） | — | — |
| **22:45–23:25** | **P0 闸门**：噪声地板 20 个 seed 分到 5 条 lane（每条 4 个，约 30 分钟）；9B 和 27B 的 tool-call 测试；evald 自检；µAR smoke（2 个任务 × 20 个提案） | 20 次评估 | 全部通过 → 进入 P1。不通过 → status 里出现 ALERT，先等你 20–60 分钟写 `override.yaml`，超时就自动执行下方"备用方案"表 |
| **23:30 起** | **P1 一阶 G0**：NanoChat 从 vanilla 起跑，`karpathy_greedy`。9B 3 个线程（lane 1、3、4）；27B 阶梯臂同时跑，2 个线程（lane 5） | 9B 3 × 30 = **90 个 trial**，约 04:15 结束；27B 2 × 30 = **60 个 trial**，约 06:30 结束 | 某个 run 满额就停（按 trial 数触发）。9B 满 90 个 → 开始 NanoChat 的 CFF |
| 23:30–00:30 | µAR（CPU，24 个并发 run）：G0 在 train 上产出 D_0；held-out 上跑 A0a、A0b（A/A）、R / R-local 基线、27B 阶梯臂 | G0：8 个任务 × 5 个 seed × 12 = 480 个提案；每条 held-out 臂 8 × 5 × 12 = 480 个 | ≈00:20 跑 Gate 2（A0a 是否好于 R-local）。不过 → 继续跑，所有 µAR 的 Δ₂ 都标上"Gate 2 未过" |
| 11-18 00:10–01:10 | **P2 U_s**：M0 读 D_0 写 lessons σ1，再写一份等长的 placebo 备忘；在 held-out 上跑 A1、A1p | **U_s 第 1 代** | ≈01:10 **Δ₂(0) U_s**：按预注册的三条规则判定 |
| 00:15–01:55 | **P3 U_w 第 1 代**：SFT（被接受的步骤 ≥ 40 条，即 G3）→ 暂停 lane 2 → 在 GPU2 上训练 LoRA M1、M1p → 经 gateway 载入 vLLM → 恢复 lane 2 | 2 个 adapter | SFT 少于 40 条 → 跳过这一代 |
| 01:55–02:25 | held-out 上跑 A2、A2p；在 train seed 5–9 上用 M1 产出 D_1，用 M0 在同样的 seed 上产出 D_0' | | ≈02:25 **Δ₂(0) U_w**。M1 显著比 M0 差（CI 上界 < 0）→ 停止这条谱系 |
| 02:30–04:10 | **U_w 第 2 代**：M2 = LoRA(D_0 ∪ D_1)；对照 M2' = LoRA(D_0 ∪ D_0')，即"更多 M0 数据"（lane 2） | 2 个 adapter | 06:00 前还没开始就跳过 |
| 04:10–04:40 | A3、A3'；用 M2 产出 D_2 | | ≈04:40 **Δ₂(1) = M2 − M1**；另比 M2 对 M2' |
| 04:45–06:25 | **U_w 第 3 代**：M3、M3'（lane 2） | 2 个 adapter | M2 显著变差就不做；08:00 前还没开始也跳过 |
| ≈07:00 | A4、A4' | | **Δ₂(2)**：得到 Δ₂(0)、Δ₂(1)、Δ₂(2) 的趋势 |
| 04:15–10:30 | **CFF**：所有臂从 G0 的最优 a* 出发，带同样的历史。4 条臂（M0、M0+σ1、M1、M1p）× 3 次重复，每次 12 个 trial。每个 run 固定一条 lane，各臂在重复之间轮换 lane。分 3 批：04:25 lane 1/3/4；06:30 lane 1/3/4/5/2；08:30 lane 1/3/4/5 | 12 个 run，共 144 个 trial | 只记录曲线，不做自动判定 |
| 08:30–13:15 | **web-off 对照**：mini harness，不带 web 工具，1 个线程（lane 2） | 30 个 trial | 13:00 前还没开始就跳过 |
| 10:30–15:30 | **P4**：`recursive_first_steps`（lane 1、3）与 `naive_trial_ledger`（lane 4、5）并行 | 各 60 次评估 | 14:00 前还没开始就跳过 |
| 15:30–19:30 | `apex_ai4ai`：M0、M1、27B 三个研究者各一个线程（lane 1、3、4），× 2 次重复 × 12 次探索 | 72 次评估 | 19:00 前还没开始就跳过 |
| 15:30–21:00 | 27B 阶梯臂的重复（lane 2、5） | 60 个 trial | 20:00 前还没开始就跳过 |
| 19:30 – 11-19 06:00 | **P5 重复**：G0 第 2 次（9B，lane 1、3、4，90 个 trial）；recursive 第 2 次（lane 2、5）；naive 第 2 次（lane 1、3）；apex 第 2 次（lane 2、4、5） | 90 + 60 + 60 + 72 | 各有 latest_start，过时就跳过 |
| 11-19 06:00–07:00 | **终点确认**：所有 NanoChat run 的 top-3 artifact 加 vanilla，每个 10 个 seed（100–109），5 条 lane | 40 次评估 | 15:00 前必须开始 |
| 07:00–18:00 | **缓冲约 11 小时**。所有节奏都没有实测。即使全部慢 1.5 倍：G0 仍在 07:30 截止前结束；第 2、3 代 LoRA 仍在各自的 latest_start（06:00、08:00）之前开始。后面的可选步骤按 latest_start 自动跳过，终点确认 15:00 前必须开始 | — | — |
| **18:00** | GPU 停止：正在跑的 run 在 trial 边界停下，打包 `runs/` 到 `$R/out/ovn1118-runs.tgz` | — | — |
| 20:01 | <node> 的 hold 结束 | — | — |

**一共更新几次**

| 层 | 次数 |
|---|---|
| U_s | 2 次：µAR 的 σ1，以及 NanoChat 的 σ1_nc |
| U_w | 3 代，全部在 µAR 上训练，共 6 个 adapter：M1/M1p、M2/M2'、M3/M3'，都在 lane 2 上训 |

**判定规则**（`platform_design.md` §1.4，由 D 的 `rsi.update.analysis` 计算，每次 decision 都按这三条判）

1. Δ₂ 的 95% 分层 bootstrap 区间下界 > 0；
2. Δ₂ > 2 × A/A 的逐任务差值的 sd；
3. 显著好于 placebo（从第 2 代起，"placebo"换成"更多 M0 数据"那一臂）。

**谱系何时停**：Δ₂ 不过只做记录，下一代照跑，因为 null 也是结果。只有 M_k 显著比上一代差（CI 上界 < 0）时才停。

**算力**

| 资源 | 用量 |
|---|---|
| NanoChat 评估 | 约 920 次 |
| 5 条 lane 的容量 | 约 1800 次〔估算〕 |
| GPU2 借给 LoRA | 6 × 45 分钟 ≈ 4.5 小时〔估算〕 |
| 27B | 不占 <node> 的卡 |

**种子**

| 用途 | seed |
|---|---|
| NanoChat 筛选 | 0 |
| G0 agent | 1000 / 1001（第 2 次） |
| recursive 确认 | 1、2 |
| 终点确认 | 100–109 |
| µAR G0 | 0–4 |
| µAR D_1 / D_0' | 5–9 |
| µAR D_2 / D_0'' | 10–14 |
| µAR held-out，所有臂 | 100–104（与 A0a 配对） |
| µAR A/A（A0b） | 200–204 |

### 27B 自演化链（11-18 加入，campaign `ovn1118b` 用 `--resume` 接上）

**对称设计**：与 9B 的二阶链用同样的任务、seed、N 和 §1.4 规则，改进者换成 Qwen3.8-27B。

| 项 | 9B 链 | 27B 链 |
|---|---|---|
| A0（held-out） | `mu_a0a` | `ladder_mu`（协议完全相同，只换模型） |
| A/A | `mu_a0b` | `mu_a0b_27b` |

**机器分工**

| 节点 | 用途 |
|---|---|
| <node> | 原有 27B（`ladder-27b`），不动 |
| <node>（job <jobid>，2 × A100） | 先训练，再起服务 |

- **训练**：D 的 `lora_train.sh`，`RSI_BASE_MODEL=Qwen/Qwen3.8-27B`，`RSI_LORA_DEVICE_MAP=auto` 把模型切到两张卡上，开着 gradient checkpointing，单个 adapter 上限 150 分钟。
- **服务**：训练完以后用 C 的 `serve_vllm.sh 27b slurm --lora --expose` 起第二个 27B base `ladder-27b-evo`（TP=2，adapter 启动时预加载成 `ladder-27b-evo@<name>`），由 ports.json 的 `ladder-27b-evo` 路由。
- **两者轮流**：训练和服务轮流占用这两张卡，不占任何 eval lane。
- **启动方式**：都经 `srun --jobid=<jobid> --overlap --ntasks=1 -N1 --gres=gpu:a100:2 --cpu-bind=none` 启动，并清掉 SLURM_*（SLURM_CONF 保留）。

| 时间（〔估算〕） | 跑什么 | 迭代 / 次数 | 自动决定 |
|---|---|---|---|
| 11-18 ≈01:00–02:30 | `mu_g0_27b`（27B 在 train 上，8 × 5 × 12 = 480 个提案）→ D_0(27B)；`mu_a0b_27b`（27B 的 A/A） | 27B 一步 µAR ≈ 20 秒，与 9B 共用 24 个 CPU 槽，9B 优先 | — |
| ≈02:30–04:30 | U_s：27B 自己写 lessons，再写一份等长的 placebo；`mu_a1_27b`、`mu_a1p_27b` | U_s(27B) 第 1 代 | ≈04:30 **Δ₂(0) U_s [27B]**，与 9B 的那一行并排 |
| ≈02:40–07:40 | SFT（≥ 40 条被接受的步骤）→ 在 <node> 上训 LoRA m1-27b、m1p-27b | 2 个 adapter × ≈2.5 小时 | 不满 40 条 → 跳过 U_w(27B) |
| ≈07:40–09:30 | 起 `ladder-27b-evo` 加两个 adapter → `mu_a2_27b`、`mu_a2p_27b`，以及 D_1(27B) → 停服务 | | ≈09:30 **Δ₂(0) U_w [27B]**，与 9B 并排 |
| ≈09:40–16:30 | 可选第 2 代：m2-27b = LoRA(D_0 ∪ D_1)，对照 m2c-27b = LoRA(D_0 ∪ D_0')，然后起服务 → A3、A3' | 2 个 adapter | 11-19 04:00 前还没开始就跳过；M1(27B) 显著变差也不做。≈16:30 **Δ₂(1) [27B]** |

**怎么看结果**：Δ₂(9B) 与 Δ₂(27B) 并排写在下面两处：

- `runs/campaign-ovn1118b/decisions.json` 的 `_9b_vs_27b` 键：U_s 第 1 代、U_w 第 1 代、U_w 第 2 代各一行；
- `campaign_status.md`：每次相关判定之后加一行"Δ₂ 9B vs 27B"，每小时一行里也有。

**混杂说明**：U_w(27B) 的臂在 <node> 的服务实例上跑，基线 `ladder_mu` 在 <node> 上。两边权重、dtype、TP 和 vLLM 参数都相同，但不是同一个进程。

**截止时间**：故意放宽。重启比计划晚的话，链条只会往后推，不会被跳过。

### 备用方案（闸门不过或阶段超时）

| 情况 | 自动做法 |
|---|---|
| G1a 噪声：sd > 0.02、单次超过 10 分钟，或成功少于 18/20 | 成功 ≥ 10：进入"噪声模式"，每次 keep 要换 seed 确认，G0 预算降到 72 个 trial。成功 < 10：关掉 NanoChat，只跑 µAR |
| G1b：9B 的 tool call 合法率 < 90% 或完成率 < 70% | 9B 只做 µAR；NanoChat 只剩 27B 的阶梯臂 |
| 27B 的 gate 不过（远端节点不可用） | 第 3 代 LoRA 之后启用 `ladder_fb_*`：在 GPU2 上起 27B（lane 2 暂停），2 个线程 × 30 个 trial，跑在 lane 5 上 |
| evald 自检不过 | 等你 1 小时；之后关掉 NanoChat，只跑 µAR |
| µAR smoke：合法率 < 70% 或每步 ≥ 5 秒 | 等你 30 分钟；之后关掉 µAR |
| SFT 被接受的步骤 < 40 | 跳过这一代 U_w，后面依赖它的臂一并跳过 |
| LoRA 超过 90 分钟 | 杀掉，重试 1 次。再失败就恢复 lane，跳过依赖它的臂 |
| 任何一步超过 `deadline` | 在 trial 边界停下，保留部分结果，下游照常 |
| 可选步骤过了 `latest_start` 还没开始 | 跳过。会被跳过的步骤：第 2、3 代 U_w（06:00 / 08:00）、CFF 第 3 轮、web-off、P4 的各 policy、27B 重复、P5 的重复。没有 latest_start、因此不会被跳过的：G0、U_s、U_w 第 1 代、CFF 前两轮。终点确认 15:00 前必须开始 |

---

## 启动（集群；R=$R）

campaign 只在 <node> 上运行；lane 3–5 由 evald 远程调度（B），27B 由 gateway 转发（C）。下面假设平台代码在 `$R/platform`，以 A 的 `DEPLOY.md` 为准。`bundle/run_campaign.sh` 会从自己所在的目录推出代码目录，也可以用 `RSI_CODE=...` 指定。

```bash
export R=$R
# 0) 检查计划（打印每一步的触发条件和截止时间）
srun --jobid=<jobid> --overlap --ntasks=1 -N1 bash $R/platform/bundle/run_campaign.sh --check
# 1) 可选：在节点上彩排，假组件 + 压缩时钟，约 1 分钟，不碰 GPU 和真实服务
srun --jobid=<jobid> --overlap --ntasks=1 -N1 bash $R/platform/bundle/run_campaign.sh --dry-run --root $R/_dryrun --time-scale 3000 --quiet
# 2) 正式启动（登录节点 tmux）
tmux new -d -s rsi-campaign "srun --jobid=<jobid> --overlap --ntasks=1 -N1 bash $R/platform/bundle/run_campaign.sh 2>&1 | tee -a $R/out/campaign.tmux.log"
```

启动前要满足：

- `$R/run/ports.json` 里有 `gateway`、`evald`、`vllm_improver`；`$R/run/token` 的权限是 600。
- 9B vLLM 用 `bundle/serve_vllm.sh 9b 0 --lora` 启动（可以热加载 adapter）。
- gateway 能把 `ladder-27b` 转到 <node>。
- evald 能看到 lane 1–5。
- µAR 的 `$R/data/sealed/muar/sref.json` 已经生成（D 的 `runner sref`）。
- 如果部署时已经跑过闸门，把结果放到 `$R/runs/gates/{g1a_noise,g1b_toolcall,g1b_27b,g1c_evald}.json`。键名就是 yaml 里 `checks:` 的那些。有结果并且通过时，campaign 直接沿用，不再重跑。

## 盯盘

```bash
tail -f $R/runs/campaign_status.md                    # 每小时一行，另在每个闸门、决定和 ALERT 时各加一行
tail -f $R/runs/campaign-ovn1118/campaign.log         # 每次状态变化
cat $R/runs/campaign-ovn1118/status.json              # 各步骤的状态和进度、flags、Δ₂
cat $R/runs/campaign-ovn1118/decisions.json           # 每次判定的完整数字（区间、p 值、A/A sd）
```

状态行包含：

- 每个正在跑的臂的 trial 数 / 预算和当前最优值；
- 已完成、失败、跳过的步骤数；
- 到目前为止的 Δ₂；
- lane 的暂停情况；
- 健康检查（vLLM、gateway、evald、磁盘）；
- 最近 1 小时的错误数。

健康检查每 3 分钟一次，处理方式如下：

| 情况 | 处理 |
|---|---|
| 某服务连续 2 次没有响应（判定为 outage） | 给所有 agent run 写 PAUSE，暂停新的 µAR run |
| vLLM 挂了 | 最多自动重启 2 次，并重新载入已有的 adapter。仍不行就出 ALERT，等人处理 |
| 某个 run 40 分钟没有新事件 | 杀掉后用 `--resume` 重启。每个 run 最多 3 次 |
| 剩余磁盘 < 10 GB | 暂停 |

## 暂停、停止、继续、人工改判

| 想做的事 | 命令 |
|---|---|
| 不再启动新步骤，正在跑的 run 在 trial 边界暂停 | `touch $R/runs/campaign-ovn1118/PAUSE`；`rm` 掉就继续 |
| 只暂停某一个 run | `touch $R/runs/<run_id>/PAUSE`（campaign 不会自动删这个文件） |
| 平稳停止：run 在 trial 边界停下，最多约 30 分钟；LoRA 训练会被杀掉；lane 恢复；保存状态 | `echo stop > $R/runs/campaign-ovn1118/STOP` |
| 立刻停止 | `echo now > $R/runs/campaign-ovn1118/STOP` |
| 停止后继续 | `rm $R/runs/campaign-ovn1118/STOP`，然后重跑"正式启动"那条命令（它总是带 `--resume`）。中断的 run 用 `--resume` 接着跑；LoRA 已经训好的 adapter 保留，从下一个接着训 |
| 人工改判 | 写 `$R/runs/campaign-ovn1118/override.yaml`，campaign 每个 tick 读一次（例子见下） |

`override.yaml` 的例子：

```yaml
force_pass: [g1b_toolcall]       # 认定某个失败的闸门为通过（只在它等人的窗口内有效）
skip: [pol_apex]                 # 跳过尚未开始的步骤
stop: [pol_naive]                # 平稳停止某个正在跑的步骤
flags: {nanochat_noisy: true}    # 直接改 flag
```

## 改了 yaml 以后让正在跑的 campaign 生效

调度器只在启动时读一次 yaml，所以新增的步骤要等它重启后才会出现。做法：

1. 先把集群上的代码更新到包含改动的提交。
2. 给调度器进程发 SIGTERM。

```bash
srun --jobid=<jobid> --overlap --ntasks=1 -N1 bash -c \
  'cd $R/platform && $SCRATCH/miniconda3/envs/rsi/bin/python -m rsi.campaign --config campaign/overnight.yaml --check > /dev/null \
   && kill -TERM $(cat $R/runs/campaign-ovn1118b/campaign.lock)'
```

- **在 `run_campaign.sh` 下运行时**：外层循环会在 60 秒内带 `--resume` 重启它，读入新 yaml。已有步骤保持原状态；正在跑的 run、LoRA 训练和服务都在各自的 session 里继续，并被重新接管；新步骤以 pending 加入。
- **不是在 `run_campaign.sh` 下运行时**：kill 之后再执行一次"正式启动"那条命令。
- **确认生效**：`campaign.log` 里出现 `resuming`，`campaign_status.md` 里出现 "campaign resumed"。

## 崩溃以后

| 情况 | 会发生什么 / 你要做什么 |
|---|---|
| 调度器进程崩溃 | `run_campaign.sh` 在 60 秒后带 `--resume` 重启它，最多 5 次。子进程在各自的 session 里继续跑，重启后会被重新接管 |
| tmux 或 srun 本身没了 | Slurm 会把这个 step 里的子进程一起杀掉。重跑"正式启动"那条命令即可：正在跑的 orchestrator 用 `--resume` 重开，没跑完的那一个 trial 丢掉；被暂停的 lane 由调度器恢复（停止和健康检查时都会检查有没有遗留的暂停） |
| hold 到期（20:01） | 18:00 已经停掉 GPU 工作并打包。之后用 sbatch 另开作业（plan §6.6），改好 yaml 里的 `gpu_stop` 和 `hold_end`，再带 `--resume` 续跑 |

结果在 `$R/out/ovn1118-runs.tgz`，由 main session 拉回 Mac。

## 本地测试（Mac CPU）

```bash
cd platform && python -m pytest tests/test_campaign.py -q        # 10 个测试，约 2–3 分钟
python -m rsi.campaign --config campaign/overnight.yaml --dry-run --time-scale 3000   # 完整彩排，结果在打印出的临时目录
```
