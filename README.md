# rsi-report：AI 递归自我改进（RSI）的取证分析与实测

本仓库有两部分：

1. **报告**：对四份声称"用 AI 改进 AI"的公开发布（Recursive *First Steps Toward Automated AI Research*、Recursive *Flint*、NaiveAI *Naive-N0.5-Flash*、Apex Intelligence *Apex-35B*）做的取证分析，并对照学术文献讨论什么才算 RSI。报告以静态网站形式发布（`report/site/`），中英双语。
2. **RSI-mini**：一个小型实验平台（`platform/`），用来直接测量一个问题：**把模型换成"被自己改进过的后继模型"之后，它做研究会不会更快**（二阶改进）。

*English summary.* A forensic review of four public "AI improving AI" releases, plus RSI-mini, a small platform that directly measures second-order improvement: does a successor model, fine-tuned on its own research, improve things faster than the original? The report is a bilingual (Chinese/English) static site built from `report/`.

## 主要结论

数字和结论均摘自 `report/sections/00_abstract.md`，图上和卡片上的数字由脚本从 `research/` 下的数据计算。

- **单代 AI 研发有效**：Qwen3.5-9B 独立研究 3 小时（72 次尝试），把一个小语言模型的验证损失从 1.4152 降到 1.2308。
- **让模型给自己写经验笔记反而更慢**：改进速度下降约三分之一，8 个测试任务上都不如一份通用建议。
- **用自己的成功经验微调权重没有带来加速**：连续三代后继模型都没有比原始模型改得更快，第三代更慢；用失败经验训练的安慰剂对照表现相同。
- **让模型自己设计训练方法**也没有得到可靠的加速。
- 四份发布材料都没有测量"后继模型当研究者是否更快"这一环（报告第 3 节）。

## 目录

```
.
├── report/                 报告
│   ├── sections/           各节正文（中文 + English），网站以此为准
│   ├── draft.md            摘要卡片与早期草稿（build_content.py 读取）
│   ├── references*.md      参考文献表（由 build_references.py 从 research/ 抽取）
│   ├── report_full.md      纯文本全文（构建时自动生成）
│   ├── tools/              出图脚本与内容构建脚本（Python 3）
│   └── site/               Astro 静态网站
├── research/               证据与数据（网站引用的部分）
│   ├── REPORT_LOG.md       研究总日志（实验结果、人工干预记录等）
│   ├── sources/            四份材料的证据台账（短引 + URL）
│   ├── lit_rsi/ lit_*.md   文献综述
│   ├── *_dissection*.md 等 权重与配置拆解、平台设计与搭建计划、Ling 管线设计
│   ├── decisions/          各项比较的效应量、95% CI、置换检验
│   ├── cf/*/ledger.jsonl   "从最好版本继续改"实验的逐次记录
│   └── replay/final/       一次完整研究运行的事件日志与摘要
├── platform/               RSI-mini 实验平台（带完整 git 历史）
│   ├── rsi/                gateway、evald（封存评估器）、harness（沙箱）、policies、tasks、campaign 调度
│   ├── bundle/ campaign/ configs/ deploy/   集群部署脚本、通宵实验编排、部署说明
│   ├── tests/              pytest
│   └── tools/ viz/         看板、回放渲染
└── tools/                  报告第 2 节所用的分析脚本
    ├── flint/              Flint 参数清单、路由统计、思考风格普查
    ├── naive/              Naive-N0.5 普查
    └── analysis/           counterfactual 实验分析（由 research/cf/ 生成 research/decisions/cf_analysis.json）
```

## 复现

### 网站

```bash
cd report/site
npm ci
npm run build          # 先运行 report/tools/make_*_figs.py 和 build_content.py，再 astro build
```

需要 Node 22.23+ 和 Python 3（只用标准库；单独重画 `recursion-loop.*.gif` 的 `make_recursion_gif.py` 另需 Pillow）。构建结果在 `report/site/dist/`。部署到 GitHub Pages 时设置：

```bash
SITE=https://<user>.github.io BASE=/<repo> \
PUBLIC_REPO_URL=https://github.com/<user>/<repo> PUBLIC_REPO_BRANCH=main npm run build
```

设置 `PUBLIC_REPO_URL` 后，参考文献里的每个内部文件都会链接到本仓库对应文件的对应行（路径以仓库根目录为准，例如 `research/REPORT_LOG.md#L37`）。

### 图和卡片上的数字

全部由 `report/tools/` 下的脚本从 `research/decisions/`、`research/cf/`、`research/replay/final/` 和 `research/census_compare.json` 计算，不需要 GPU。

### 权重拆解（报告第 2 节）

需要先自行下载第三方权重（例如 `hf download RecursiveSI/flint-int4`），再运行 `tools/flint/`、`tools/naive/` 下的脚本。部分脚本需要 Apple Silicon（MLX / rec）、llama.cpp 或多卡 GPU 节点；脚本里的本机与集群路径已替换为占位符，运行前需改成自己的路径。这些脚本的原始输出（普查 jsonl、日志）没有放进本仓库。

### RSI-mini 实验

见 `platform/DEPLOY.md` 与 `platform/deploy/`。需要 Slurm 集群和 A100 级别的 GPU。测试：`cd platform && python -m pytest`。

## 数据来源与标记

- **自测**：`research/decisions/`、`research/cf/`、`research/replay/` 都是本项目自己运行得到的数据。
- **厂商自报**：四份材料里的数字只来自其公开发布；证据台账（`research/sources/*.md`）每条都标了出处 URL 和状态。
- 正文和日志里的标记：**〔自报〕** = 只有发布方自己的说法，未独立核实；**〔推断〕** = 根据公开信息推出，未直接看到；**〔估算〕** = 粗略估计。网站上这些标记显示为小标签。
- 本仓库**不包含任何第三方模型权重**，也不包含第三方仓库的克隆；需要时请从原始地址下载。

## 脱敏说明

公开前对所有文件（包括 `platform/` 的全部 git 历史和提交信息）做了机械替换，只改行内文字，不增删行，所以参考文献里的"文件:行号"仍然准确：

- 集群项目目录写作 `$R`，个人 scratch 目录写作 `$SCRATCH`；计算节点写作 `<node>`，Slurm 作业号写作 `<jobid>`，内网地址写作 `<ip>`，两个集群写作 `cluster-A`、`cluster-B`。
- 代码和配置文件里为了保持语法和测试一致，改用等价的替身：路径 `/scratch/user/...`，节点 `node-a`…`node-d`，作业号 `1000000x`。
- 文档里的命令块含有上述占位符，复制运行前需要替换成自己的值。

## 许可

- 代码（`platform/`、`tools/`、`report/tools/`、`report/site/`）：Apache License 2.0，见 `LICENSE`。
- 报告正文、图和数据（`report/`、`research/`）：CC BY 4.0，见 `LICENSE-content`。
- 第三方代码与引文的来源和原始许可见 `NOTICE`（包括 nanochat 的 MIT 声明）。引文版权归原作者。

## 局限

- 实验规模小：主改进者为 Qwen3.5-9B（另有 Qwen3.8-27B 对照），8 个小型研究任务、每组 5 个种子；结论只对这个设置成立。
- 第 2 节的拆解基于公开权重与公开文字，没有任何内部信息；带〔推断〕的结论可能被厂商的后续披露推翻。
- 研究日志保留了工作过程中的原始记录（包括人工干预），没有为发布而改写。
