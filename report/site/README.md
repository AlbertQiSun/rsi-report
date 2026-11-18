# RSI 报告网站（Astro）

- 内容来源：`report/draft.md`。不要直接改网页里的文字；改 draft.md 后重新构建。
- `npm run content`：运行 `report/tools/build_content.py`，把 draft.md 转成 `src/data/content.json`。
  - 引用方括号会变成脚注编号，参考文献取自 `report/references.md`。
  - 卡片上的数字由脚本从 `research/` 下的实验数据计算。
- `npm run build`：先生成内容，再构建静态站点，输出到 `dist/`。
- 部署到 GitHub Pages：
  - 需要知道仓库名之后再设置路径，例如 `SITE=https://<user>.github.io BASE=/<repo> npm run build`。
  - `SITE`、`BASE` 要和实际的 Pages 地址一致，否则图和样式的路径会错。
- 本地预览：`npm run dev`；或 `npm run build` 之后用任意静态服务器打开 `dist/`（例如 `npx serve dist`）。

## 参考文献链接到仓库

参考文献里每个内部文件都记录了它在仓库中的路径和行号（证据台账的行、REPORT_LOG 的节标题、`§1.4` 这类小节标题都换算成了行号）。仓库建好之后，构建时设置仓库地址即可，每一条都会链接到对应文件的那一行：

```
PUBLIC_REPO_URL=https://github.com/<user>/<repo> PUBLIC_REPO_BRANCH=main npm run build
```

路径以 `rsi-report/` 为仓库根目录（例如 `research/REPORT_LOG.md#L37`）。如果仓库根目录不同，改 `src/pages/index.astro` 里的 `repoLink`。不设置时，页面只显示 `路径:行号`，不生成链接。
