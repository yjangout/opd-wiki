# 基础

这一部分从零讲起，说明 OPD 要解决什么问题。读完你应该能回答三个问题：

1. SFT 和 off-policy 蒸馏缺在哪里？
2. RL（以 GRPO 为例）又缺在哪里？
3. OPD 怎样同时补上这两个缺口？它自己的代价又是什么？

| 页面 | 内容 |
|---|---|
| [SFT 与 off-policy 蒸馏](sft.md) | 在固定数据上学下一个 token；exposure bias；off-policy 蒸馏的数据经验 |
| [RL 与 GRPO](grpo.md) | 策略梯度、PPO、GRPO；稀疏奖励与信用分配 |
| [为什么需要 OPD](why-opd.md) | 三者对比、DAgger 类比、信息量、发展脉络 |
| [三种方法的数据长什么样](data-formats.md) | 用同一道题和同一个工具调用任务，对比 SFT、GRPO、OPD 的训练数据 |

如果你已经熟悉 SFT 和 GRPO，可以直接跳到[为什么需要 OPD](why-opd.md)。
