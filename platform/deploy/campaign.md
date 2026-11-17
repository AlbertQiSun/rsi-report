# 过夜 campaign：什么时候更新、更新几次、怎么启动和看护（worker F）

计划文件是 `campaign/overnight.yaml`，调度器是 `rsi/campaign/`。下表就是 yaml 里写死的计划。**带〔估算〕的时长都没有实测**，来源是 `research/platform_build_plan.md` §6.4–6.5 和协调者给的数：一次 NanoChat 评估约 7 分钟；9B 每个 trial 思考 2–3 分钟；一次 LoRA 训练约 45 分钟（30–60）；一步 µAR 约 8 秒。闸门跑完后要换成实测值，再用 `--check` 看一遍时间。

## 时间表（北京时间；具体钟点都是〔估算〕，实际由触发条件推进）

| 时间 | 跑什么 | 迭代 / 次数 | 这一步自动做的决定 |
|---|---|---|---|
| 11-18 00:00–01:00 | 部署平台（不属于 campaign） | — | — |
| **01:00–02:10** | **P0 闸门**：20 个 seed 的噪声地板（lane 1、2 各 10 个）；9B 和 27B 的 tool-call 测试；evald 自检；µAR smoke（2 个任务 × 20 个提案） | 20 次评估 | 全部通过 → 进入 P1。不通过 → status 里出现 ALERT，先等你 20–60 分钟写 `override.yaml`，超时就自动执行下方"备用方案"表 |
| **02:10 起** | **P1 一阶 G0**：NanoChat 从 vanilla 起跑，`karpathy_greedy`。9B 2 个线程，27B 2 个线程同时跑（27B 在自己的节点 <node> 上）。4 个线程按 FIFO 共用 lane 1、2 | 9B 2 × 30 = **60 个 trial**；27B 2 × 30 = **60 个 trial** | 某个 run 满 60 个 trial 就停（按 trial 数触发）。截止时间 13:00 |
| 02:10–03:00 | µAR（CPU，16 个 worker）：G0 在 train 上产出 D_0；held-out 上跑 A0a、A0b（A/A）、R、R-local、27B 阶梯臂 | G0：8 个任务 × 5 个 seed × 12 = 480 个提案；每条 held-out 臂 8 × 5 × 12 = 480 个 | ≈02:50 跑 Gate 2（A0a 是否好于 R-local）。不过 → 继续跑，但所有 µAR 的 Δ₂ 都标上"Gate 2 未过" |
| 02:45–03:30 | **P2 U_s**：M0 读 D_0 写 lessons σ1，同时写一份 placebo 通用备忘；在 held-out 上跑 A1、A1p | **U_s 第 1 代** | ≈03:30 **Δ₂(0) U_s**：按预注册的三条规则判定 |
| 02:50–04:30 | **P3 U_w 第 1 代**：做 SFT（被接受的步骤 ≥ 40 条，即 G3）→ 暂停 lane 2 → 在 GPU2 上训练 LoRA M1 和 M1p → 载入 vLLM → 恢复 lane 2 | 2 个 adapter | SFT 少于 40 条 → 跳过这一代 U_w |
| 04:30–05:00 | held-out 上跑 A2（M1）、A2p（M1p）；在 train seed 5–9 上用 M1 产出 D_1，用 M0 在同样的 seed 上产出 D_0' | | ≈05:00 **Δ₂(0) U_w**：通过记为"晋升"。M1 显著比 M0 差（CI 上界 < 0）则停止这条谱系 |
| 05:05–06:45 | **U_w 第 2 代**：M2 = LoRA(D_0 ∪ D_1)；对照 M2' = LoRA(D_0 ∪ D_0')，即"更多 M0 数据" | 2 个 adapter，lane 2 暂停 | 09:00 前还没开始就跳过 |
| 06:45–07:15 | A3（M2）、A3'（M2'）；用 M2 产出 D_2 | | ≈07:15 **Δ₂(1) = M2 − M1**；另比 M2 对 M2'（"递归"是否成立的关键对照） |
| 07:20–09:00 | **U_w 第 3 代**：M3、M3' | 2 个 adapter，lane 2 暂停 | M2 显著变差就不做；12:00 前还没开始也跳过 |
| ≈09:30 | A4、A4' | | **Δ₂(2)**：得到 Δ₂(0)、Δ₂(1)、Δ₂(2) 的趋势 |
| ≈11:45 | 9B 和 27B 的 G0 各满 60 个 trial | | 用 G0 的历史写 NanoChat 的 lessons σ1_nc |
| 11:55–15:55 | **CFF 第 1 轮**：所有臂都从 G0 的最优 artifact a* 出发，带同样的 G0 历史接着跑。M0、M0+σ1 一组；M1、M1p 一组 | 4 条臂 × 12 = 48 个 trial | 只记录曲线，不做自动判定（每臂 n=1） |
| 15:55–20:55 | **P4** `recursive_first_steps`，9B，2 个线程 | 60 次评估 | 21:00 前还没开始就跳过 |
| 20:55 – 11-19 01:55 | `naive_trial_ledger`，9B，2 个线程 | 60 次评估 | 11-19 02:00 前还没开始就跳过 |
| 11-19 01:55–07:55 | **P5** `apex_ai4ai`：M0、M1、27B 三个研究者，各 2 次重复 × 12 次探索 | 72 次评估 | 04:00 前还没开始就跳过 |
| 07:55–11:55 | **CFF 第 2 轮**（同第 1 轮） | 48 个 trial | 10:00 和 12:00 前还没开始就跳过 |
| 11:55–14:20 | **终点确认**：所有 NanoChat run 的 top-3 artifact 加 vanilla，每个 10 个 seed（100–109） | 40 次评估 | 15:00 前必须开始 |
| 14:20–18:00 | 缓冲约 3.7 小时，吸收前面的超时 | — | — |
| **18:00** | GPU 停止：正在跑的 run 在 trial 边界停下，打包 `runs/` 到 `$R/out/ovn1118-runs.tgz` | — | — |
| 20:01 | hold 结束（27B 那台约到 20:45） | — | — |

**一共更新几次**

| 层 | 次数 |
|---|---|
| U_s | 2 次：µAR 的 σ1，以及 NanoChat 的 σ1_nc |
| U_w | 3 代，全部在 µAR 上训练，共 6 个 adapter：M1/M1p、M2/M2'、M3/M3' |

**判定规则**（`platform_design.md` §1.4，每次 decision 都按这三条判）

1. Δ₂ 的 95% 分层 bootstrap 区间下界 > 0；
2. Δ₂ > 2 × sd(A/A)；
3. 显著好于 placebo（从第 2 代起，"placebo"换成"更多 M0 数据"那一臂）。

**谱系何时停**：Δ₂ 不过只做记录，下一代照跑，因为 null 也是结果。只有 M_k 显著比上一代差（CI 上界 < 0）时才停。

**算力**

| 资源 | 用量 |
|---|---|
| NanoChat 评估 | 约 470 次 |
| 两条 lane 的容量 | 约 750 次〔估算〕 |
| GPU2 借给 LoRA | 6 × 45 分钟 ≈ 4.5 小时〔估算〕 |
| 27B | 不占 <node> 的卡 |

**种子**

| 用途 | seed |
|---|---|
| NanoChat 筛选 | 0 |
| recursive 确认 | 1、2 |
| 终点确认 | 100–109 |
| µAR G0 | 0–4 |
| µAR D_1 / D_0' | 5–9 |
| µAR D_2 / D_0'' | 10–14 |
| µAR held-out，所有臂 | 100–104（与 A0a 配对） |
| µAR A/A（A0b） | 200–204 |

### 备用方案（闸门不过或阶段超时）

| 情况 | 自动做法 |
|---|---|
| G1a 噪声：sd > 0.02、单次超过 10 分钟，或成功少于 18/20 | 成功 ≥ 10：进入"噪声模式"，每次 keep 要换 seed 确认，G0 预算降到 48 个 trial。成功 < 10：关掉 NanoChat，只跑 µAR |
| G1b：9B 的 tool call 合法率 < 90% 或完成率 < 70% | 9B 只做 µAR；NanoChat 只剩 27B 的阶梯臂 |
| 27B 的 gate 不过（远端节点不可用） | 第 1 轮 CFF 之后启用 `ladder_fb_*`：暂停 lane 2，在 GPU2 上起 27B，1 个线程 × 20 个 trial |
| evald 自检不过 | 等你 1 小时；之后关掉 NanoChat，只跑 µAR |
| µAR smoke：合法率 < 70% 或每步 ≥ 5 秒 | 等你 30 分钟；之后关掉 µAR |
| SFT 被接受的步骤 < 40 | 跳过这一代 U_w，后面依赖它的臂一并跳过 |
| LoRA 超过 90 分钟 | 杀掉，重试 1 次。再失败就恢复 lane，跳过依赖它的臂 |
| 任何一步超过 `deadline` | 在 trial 边界停下，保留部分结果，下游照常 |
| 可选步骤过了 `latest_start` 还没开始 | 跳过。砍的顺序：apex → CFF 第 2 轮 → naive → 第 3 代 LoRA。G0、第 1–2 代、终点确认不砍 |

---

## 启动（集群；R=$R）

下面假设平台代码在 `$R/platform`，以 A 的 `DEPLOY.md` 为准。`bundle/run_campaign.sh` 会从自己所在的目录推出代码目录，也可以用 `RSI_CODE=...` 指定。

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

- `$R/run/ports.json` 里有 `gateway`、`vllm`、`evald` 三个端口；`$R/run/token` 的权限是 600。
- vLLM 用 `--enable-lora` 启动，并带 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`。
- gateway 能把 `ladder-27b` 转到 <node>。
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

## 崩溃以后

| 情况 | 会发生什么 / 你要做什么 |
|---|---|
| 调度器进程崩溃 | `run_campaign.sh` 在 60 秒后带 `--resume` 重启它，最多 5 次。子进程在各自的 session 里继续跑，重启后会被重新接管 |
| tmux 或 srun 本身没了 | Slurm 会把这个 step 里的子进程一起杀掉。重跑"正式启动"那条命令即可：正在跑的 orchestrator 用 `--resume` 重开，没跑完的那一个 trial 丢掉；被暂停的 lane 由调度器恢复（停止和健康检查时都会检查有没有遗留的暂停） |
| hold 到期（20:01） | 18:00 已经停掉 GPU 工作并打包。之后用 sbatch 另开作业（plan §6.6），改好 yaml 里的 `gpu_stop` 和 `hold_end`，再带 `--resume` 续跑 |

结果在 `$R/out/ovn1118-runs.tgz`，由 main session 拉回 Mac。

## 本地测试（Mac CPU）

```bash
cd platform && python -m pytest tests/test_campaign.py -q        # 7 个测试，约 1.5 分钟
python -m rsi.campaign --config campaign/overnight.yaml --dry-run --time-scale 3000   # 完整彩排，结果在打印出的临时目录
```
