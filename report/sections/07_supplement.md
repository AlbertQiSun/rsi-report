# 7. 补充分析
#### 正文（中文）

本节保留两项后续检验的细节。它们的主要结果已并入第 4 节，便于读者在同一条论证中看到完整证据；这里进一步展开配方比较，供后续删节或核查。

### 第一轮结果能否复现

第一轮配方的改进速度分为 0.605。保持数据与训练设置不变，只更换 LoRA 随机种子后，复训模型得分为 0.602。两者差值为 −0.003（95% CI [−0.125, 0.113]，$p=0.98$），结果高度接近。因此，第一轮相对第三轮的差距不能用普通的种子波动解释 [`decisions/decisions_1420.json`]。

### 更大模型设计的配方

Qwen3.8-27B 接收与第二轮设计者相同的反馈和数据摘要，为 Qwen3.5-9B 选择训练方案。它选用了与第三轮相同的 539 条记录，并采用较轻的训练设置。得分为 0.513，相对第三轮高 0.122（95% CI [0.017, 0.269]，$p=0.047$），但低于原始模型的 0.588（差值 −0.075，95% CI [−0.195, 0.023]，$p=0.23$）。由于本组没有随机配方对照，这一结果适合用于描述模型设计配方后的表现，不足以单独判断模型选择优于随机选择 [`decisions/decisions_1420.json`]。

### 四种配方并列

| 配方 | 数据选择 | 训练设置 | 改进速度分 |
|---|---|---|---:|
| 第一轮，原始模型设计 | 全部采纳记录及 hindsight，共 694 条调用 | rank 32，学习率 3e-4，时间上限内约 2.4 epoch | 0.605 |
| 第一轮，换种子复训 | 与第一轮相同 | 与第一轮相同，仅更换随机种子 | 0.602 |
| 第二轮，原始模型读过反馈后设计 | 采纳记录中的最后一次调用，共 137 条 | rank 16，学习率 2e-5，2 epoch | 0.595 |
| 第三轮，第一轮微调模型设计 | 采纳及 hindsight 记录，剔除相对提升低于 1% 的记录，共 539 条 | rank 32，学习率 3e-4，时间上限内约 2.7 epoch | 0.392 |
| Qwen3.8-27B 设计 | 与第三轮相同的 539 条记录 | rank 32，学习率 2e-4，3 epoch | 0.513 |

这组比较把解释范围限定得更清楚。换种子复训表明第一轮成绩可重复；同一批 539 条数据在较轻训练下得到 0.513，在第三轮的较重设置下得到 0.392。结果与数据筛选和训练强度影响后继表现相符，但这些设置并非完整的因子实验，不能将下降精确归因于单一因素。可以确定的是，设计者的规模增加并未带来高于原始基线的结果。

### 对主结论的补充

第 4 节的结论仍然是：在本次测试中，没有测到跨代改进速度上升。补充结果解释了不同配方的分数差异，并排除了第一轮成绩仅由单次随机种子造成的解释。它也强化了一个实践要求：评估自我更新时，必须报告训练数据选择和训练强度，并用后继模型的独立任务表现检验更新结果。

#### Text (English)

This section retains the details of two follow-up checks. Their main results are integrated into Section 4 so the evidence appears in one argument. The additional recipe comparisons are kept here for reference and can be removed in a later edit.

### Does the round-1 result replicate?

The round-1 recipe scored 0.605 on improvement speed. Keeping its data and training settings fixed and changing only the LoRA seed produced a score of 0.602. The difference was −0.003 (95% CI [−0.125, 0.113], $p=0.98$). The close scores show that ordinary seed variation does not explain the gap between rounds 1 and 3 [`decisions/decisions_1420.json`].

### Recipe designed by a larger model

Qwen3.8-27B received the same feedback and data summary as the round-2 designer and selected a training plan for Qwen3.5-9B. It chose the same 539 records as round 3 and used lighter training. Its score was 0.513, 0.122 above round 3 (95% CI [0.017, 0.269], $p=0.047$), but below the original model's 0.588 (difference −0.075, 95% CI [−0.195, 0.023], $p=0.23$). This arm had no random-recipe control, so it describes performance after a model-designed recipe but cannot by itself establish that the model's selection was better than random selection [`decisions/decisions_1420.json`].

### Comparing the four recipes

| Recipe | Data selection | Training settings | Improvement-speed score |
|---|---|---|---:|
| Round 1, designed by the original model | All accepted records and hindsight, 694 calls | rank 32, learning rate 3e-4, about 2.4 epochs at the time limit | 0.605 |
| Round 1, new-seed rerun | Same as round 1 | Same as round 1, with a new random seed | 0.602 |
| Round 2, designed by the original model after feedback | Last call from each accepted attempt, 137 calls | rank 16, learning rate 2e-5, 2 epochs | 0.595 |
| Round 3, designed by the round-1 fine-tuned model | Accepted and hindsight records, excluding gains below 1%, 539 calls | rank 32, learning rate 3e-4, about 2.7 epochs at the time limit | 0.392 |
| Designed by Qwen3.8-27B | Same 539 records as round 3 | rank 32, learning rate 2e-4, 3 epochs | 0.513 |

These comparisons narrow the interpretation. The rerun shows that the round-1 result is reproducible. On the same 539 records, lighter training scored 0.513 and the heavier round-3 setting scored 0.392. The results are consistent with data selection and training intensity affecting successor performance, but the settings do not form a complete factorial experiment, so the drop cannot be attributed precisely to either factor alone. The larger designer did not produce a result above the original baseline.

### Relation to the main result

The conclusion in Section 4 remains: this experiment did not detect an increase in improvement speed across generations. The follow-up results explain variation between recipes and rule out the idea that round 1's score arose only from a single random seed. They also reinforce a practical requirement: evaluations of self-updates should report data selection and training intensity, then test the resulting successor on independent tasks.

