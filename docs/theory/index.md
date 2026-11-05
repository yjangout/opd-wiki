# 原理

这一部分讲 OPD 的数学细节，以及它为什么有效、什么时候失效。

| 页面 | 内容 |
|---|---|
| [目标函数与梯度](objective.md) | 从序列级 reverse KL 推导到逐 token advantage；k1、k2、k3 三种写法各优化了什么；和 KL 正则 RL 的关系 |
| [散度怎么选](divergence.md) | forward KL、reverse KL、JSD 及各种自适应散度 |
| [估计粒度：sampled-token、top-K、全词表](estimators.md) | 偏差与方差；负反馈之后概率质量流向哪里 |
| [OPD 何时有效、何时失效](mechanism.md) | 思维模式兼容、重叠率飞轮、教师在学生前缀上的可靠性、容量差距 |

记号约定：$\pi_\theta$ 或 $\pi_S$ 表示学生，$\pi_T$ 表示教师，$x$ 是题目，$y = (y_1, \dots, y_T)$ 是学生生成的回答，$y_{<t}$ 是第 $t$ 个 token 之前的前缀，$V$ 是词表。
