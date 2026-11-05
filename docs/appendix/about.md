# 关于本站

## 这是什么

一份关于 On-Policy Distillation（OPD）的中文 wiki，目标是把 OPD 的原理、数学细节、工程实现、踩坑经验、工业案例和最新进展系统地整理在一处。内容截止到 **2026 年 11 月**。

## 内容约定

| 标记 | 含义 |
|---|---|
| <span class="verified">✅</span> | 数字或结论已对照一手资料（原论文、官方博客、官方文档）核实 |
| <span class="unverified">⚠️</span> | 来自论文摘要或二手转述，引用前请回查原文 |
| 无标记 | 领域内的通用知识，或本站的推导、整理和建议 |

公式记号：$\pi_\theta$ 或 $\pi_S$ 表示学生，$\pi_T$ 表示教师，$y_t$ 是第 $t$ 个 token，$y_{<t}$ 是它之前的前缀，$\operatorname{sg}[\cdot]$ 表示 stop-gradient。

框架配置截至 2026 年 10 月。这些框架迭代很快，以所装版本的官方文档为准。

## 交互演示

[verl 源码走读](../frameworks/verl-walkthrough.md)里的交互演示，是用一次真实的 verl OPD 冒烟测试中 dump 下来的 GSM8K 样本生成的。生成脚本在仓库的 `opd_explainer/` 目录。

## 参与贡献

OPD 发展很快，本站一定有遗漏和错误。欢迎：

- 指出错误或过时的内容；
- 补充新的论文、案例和实践经验；
- 分享你自己的实验结果（尤其是失败的）。

源码在 [yjangout/opd-wiki](https://github.com/yjangout/opd-wiki)。可以直接[提 issue](https://github.com/yjangout/opd-wiki/issues)，或者点每页右上角的编辑按钮提交修改。

## 许可

| 内容 | 协议 |
|---|---|
| 文字（`docs/` 下的教程正文、表格、图示） | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh-hans) |
| 代码（`opd_explainer/`、配置文件、文中的代码片段） | [MIT](https://github.com/yjangout/opd-wiki/blob/main/LICENSE) |

按 CC BY 4.0，你可以自由转载、改编，包括商业用途，只需注明出处（OPD Wiki，附上本站链接）并说明是否做了修改。

wiki 里引用的第三方内容（论文结果、数据集样本、框架文档）仍按其原始协议使用。

提交到本仓库的贡献，默认按上述同样的协议授权。
