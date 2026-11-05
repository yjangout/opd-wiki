# OPD Wiki

On-Policy Distillation（在线策略蒸馏，OPD）中文教程：原理、实战、框架、工业案例与前沿进展。

在线阅读：https://yjangout.github.io/opd-wiki/

内容都在 `docs/` 下，用 [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) 构建。

## 本地预览

```bash
pip install -r requirements.txt
mkdocs serve
```

浏览器打开 http://127.0.0.1:8000 。

## 构建静态站点

```bash
mkdocs build --strict
```

输出在 `site/` 目录。`--strict` 会把断链等警告当作错误，提交前建议跑一遍。

## 部署

推送到 `main` 分支后，GitHub Actions（`.github/workflows/deploy.yml`）会自动构建并发布到 GitHub Pages。

首次使用前，需要在仓库的 Settings → Pages 里，把 Source 设为 **GitHub Actions**。

## 目录结构

```
docs/
├── index.md            首页
├── basics/             基础：SFT、GRPO、为什么需要 OPD、数据格式
├── theory/             原理：目标函数、散度、估计器、成败机制
├── practice/           实践：配方、教师选择、数据、监控、踩坑、算力
├── frameworks/         框架：verl、ms-swift、TRL、slime/Tinker、工程优化
├── tutorial/           动手教程：L1 SFT → L2 GRPO → L3 OPD → L4 Agentic RL
├── cases/              工业案例
├── frontier/           变体与前沿
├── appendix/           FAQ、术语表、参考文献、关于本站
└── demo/opd-step.html  交互演示：一个 verl OPD step（由 opd_explainer/ 生成）
```

交互演示的源码在 `opd_explainer/`，换样本后重新生成，再复制到 `docs/demo/opd-step.html`。

## 许可

- 文字内容（`docs/`）：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，见 [LICENSE-CONTENT](LICENSE-CONTENT)
- 代码（`opd_explainer/`、配置文件、文中的代码片段）：[MIT](LICENSE)

转载或改编时注明出处即可，包括商业用途。

## 内容约定

- ✅：数字或结论已对照一手资料（原论文、官方博客、官方文档）核实；
- ⚠️：来自摘要或二手转述，引用前请回查原文。
