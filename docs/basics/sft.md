# SFT 与 off-policy 蒸馏

## SFT 在做什么

监督微调（Supervised Fine-Tuning，SFT）给模型一批写好的"题目 + 回答"，让模型在回答的每个位置上预测下一个 token。损失是交叉熵：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t}), \qquad y \sim \text{数据集}
$$

其中 $x$ 是题目，$y$ 是标准回答，$y_{<t}$ 是第 $t$ 个 token 之前的前缀。

SFT 的三个优点让它成为后训练的第一步：

- **稳定**：目标固定，不依赖模型自己的输出；
- **信号稠密**：回答里每个 token 都有监督；
- **便宜**：一次前向加一次反向，不需要在线生成。

## Off-policy 蒸馏：数据由教师生成的 SFT

如果"标准回答"不是人写的，而是由一个更强的教师模型生成的，这就是 **off-policy 蒸馏**。之所以叫 off-policy，是因为训练数据来自教师的分布，而不是学生自己的分布。

常见的几种形式：

| 形式 | 做法 | 需要教师的什么 | 代表 |
|---|---|---|---|
| 序列级蒸馏（SeqKD） | 教师生成完整回答，学生在上面做 SFT | 只要文本 | DeepSeek-R1 蒸馏版、OpenThoughts |
| 词级 / logit 蒸馏 | 在固定数据的每个位置上，让学生的分布逼近教师的分布（forward KL） | 每个位置的概率分布 | 一些小模型的预训练阶段蒸馏（如 Gemma 2、Llama 3.2 1B/3B） |
| 混合 | 先 SeqKD，再在固定数据上加 logit 蒸馏 | 两者都要 | — |

词级蒸馏的损失写作：

$$
\mathcal{L}_{\text{KD}}(\theta) = \sum_{t} \mathrm{KL}\big(\pi_T(\cdot \mid x, y_{<t}) \,\|\, \pi_\theta(\cdot \mid x, y_{<t})\big), \qquad y \sim \text{固定数据}
$$

注意：这里的前缀 $y_{<t}$ 仍然来自固定数据，所以它和 SeqKD 一样都是 off-policy。

### 典型案例：DeepSeek-R1 的蒸馏版

DeepSeek 用 R1 生成约 80 万条样本，对 Qwen2.5 和 Llama 系列的 1.5B–70B 模型做纯 SFT，没有再加 RL。<span class="unverified">⚠️</span>

他们的结论是：对小模型来说，从大模型蒸馏推理模式，比直接对小模型跑 RL 更有效。这是"off-policy 蒸馏很强"的标志性证据，也说明它很适合当后训练的第一步。

## SFT 的根本缺口：exposure bias

SFT 训练时，模型看到的前缀**永远来自数据集**，也就是说永远是"正确的"。推理时，前缀却来自**模型自己**，而模型会犯错。

举个例子：

```text
题目：解方程 3x + 5 = 20

训练数据里的前缀：  两边减 5，得 3x = 15，……    ← 永远是对的
模型推理时的前缀：  两边减 5，得 3x = 25，……    ← 自己算错了
```

模型写出"3x = 25"之后，接下来该怎么办？SFT 数据里从来没有出现过这种状态，模型**没学过怎么从自己的错误里恢复**。这就是 **exposure bias**：训练时暴露给模型的状态分布，和推理时实际遇到的状态分布不一致。

模仿学习里有一个经典结论（Ross et al., 2011，DAgger 论文）：只在专家状态上做行为克隆时，误差会随序列长度 $T$ 以约 $O(T^2)$ 累积；如果在学生自己会访问的状态上得到专家反馈，可以降到约 $O(T)$。推理链越长，这个差距越明显。

## SFT 的其他局限

**收益递减。** Thinking Machines 的实验中，Qwen3-8B-Base 用 40 万条数据 SFT 后，AIME'24 到 60%。按 log-linear 的趋势外推，要到 70% 大约需要 200 万条。<span class="verified">✅</span>

**在自己的输出上做 SFT 也会退化。** 同一篇博客发现：拿模型自己在 temperature=1 下的输出做 SFT（理论上期望的 KL 为 0），只要学习率不为 0，指令跟随能力（IF-eval）仍然会持续下降。原因是有限 batch 的采样噪声会让模型一步步偏离原来的分布。<span class="verified">✅</span>

**学生学不进去。** 小模型直接学大教师的长思维链，效果往往不稳定（见下一节）。

## Off-policy 蒸馏的数据经验

这些经验在 OPD 时代依然有用，因为 off-policy 蒸馏通常就是 OPD 的冷启动阶段。

**1. 教师分数高，不等于教得好。** <span class="verified">✅</span>
OpenThoughts（ICLR 2026）对比了多个教师：QwQ-32B 在目标 benchmark 上的分数低于 DeepSeek-R1，却是更好的教师。他们最终用 QwQ-32B 生成 120 万条数据，训出了 OpenThinker3-7B。

**2. 每道题多采几条回答。**
OpenThoughts 发现，每题采多个教师回答是有效的扩数据手段，可以把数据量放大 16 倍以上。

**3. 答案校验和过滤的收益不大。**
他们试了多种校验和过滤方法，都没有带来显著提升。

**4. 题源宁精不杂。**
从 1–2 个高质量来源选题，比追求来源多样性效果更好。

**5. 小模型有"学习能力鸿沟"。** <span class="verified">✅</span>
《Small Models Struggle to Learn from Strong Reasoners》（ICML 2025）发现，3B 及以下的模型直接学长思维链或大教师的轨迹，收益不稳定。把长短 CoT 混合（Mix Distillation），或者把大小教师的数据混合，效果更好。

**6. 当作 OPD 的冷启动。** <span class="verified">✅</span>
Rethinking OPD 发现：先用教师离线生成的数据做 SFT，能显著提高 OPD 开始时师生高概率 token 的重叠率，OPD 的最终效果和上限都更高。

## 小结

| | 内容 |
|---|---|
| 优点 | 稳定、信号稠密、便宜、实现简单 |
| 根本缺口 | 训练时的前缀来自数据集，推理时的前缀来自模型自己（exposure bias） |
| 其他局限 | 收益递减；即使在自己的输出上训练也会漂移；小模型可能学不进去 |
| 在流程中的角色 | 第一步：让模型学会格式和基本能力，为 OPD 和 RL 打底 |

下一步：[RL 与 GRPO](grpo.md)，看看 RL 怎样补上"状态"这个缺口，又带来了什么新问题。
