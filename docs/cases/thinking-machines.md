# Thinking Machines

**来源**：Kevin Lu 等，[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)，Thinking Machines Lab 博客 Connectionism，2025 年 10 月。<span class="verified">✅</span>

这篇博客让"OPD"这个说法流行起来。它的做法简单：在已有的 RL 训练脚本上，把 KL 正则里的参考模型换成教师，把逐 token reverse KL 的负值当作 advantage，折扣因子设为 0。只用教师的 logprob，不需要奖励模型。

## 实验一：数学推理

**设置**：学生 Qwen3-8B-Base，教师 Qwen3-32B（有些实验用 Qwen3-8B 当教师，效果略好）。

1. **中训练（SFT）**：在 OpenThoughts-3 的 40 万条 prompt（回答来自 QwQ-32B）上全参微调，AIME'24 达到 **60%**；
2. 按 log-linear 的趋势外推，要到 70% 大约需要 **200 万**条 SFT 数据；
3. **OPD**：从 40 万条的 checkpoint 出发，约 **150 步**（约 7.7 万个 prompt，每题 4 条）就达到 **70%**。

**算力对比：**

| 方法 | AIME'24 | 教师 FLOPs | 学生 FLOPs | 相对 SFT-2M 的节省 |
|---|---|---|---|---|
| SFT-400K（起点） | 60% | 8.5e20 | 3.8e20 | — |
| SFT-2M（外推） | 约 70% | 3.4e21 | 1.5e21 | 1× |
| RL | 68% | — | — | 约 1× |
| OPD | 70% | 8.4e19 | 8.2e19 | **9–30×** |

- 9 倍：假设 SFT 数据已经现成；
- 计入教师 logprob 的 GPU 小时（这部分很容易并行），约 18 倍；
- 再计入教师生成数据的成本，约 30 倍。

**LoRA**：rank 32 的 LoRA 在 SFT 后比全参微调差 13%，OPD 后只差 6%。

## 引用 Qwen3 报告的对比

| 方法 | AIME'24 | GPQA-Diamond | GPU 小时 |
|---|---|---|---|
| Off-policy 蒸馏 | 55.0 | 55.6 | 未报告 |
| + RL | 67.6 | 61.3 | 17,920 |
| + OPD | **74.4** | **63.3** | **1,800** |

## 实验二：个性化助手与修复遗忘

**目标**：给 Qwen3-8B 注入公司内部知识，同时保住指令跟随能力。

| 模型 | 内部 QA | IF-eval |
|---|---|---|
| Qwen3-8B | 18% | 85% |
| + 中训练（100% 内部文档） | 43% | 45% |
| + 中训练（70% 文档 / 30% 对话） | 36% | 79% |
| + 中训练（70% / 30%）+ OPD | 41% | **83%** |

- **背景对话数据**：Tulu3 的 prompt，回答由 Qwen3-8B 重新采样。用 Qwen3-8B 自己的回答比用 Qwen3-32B 的回答更能保住对话能力；
- **只靠调配比不够**：没有任何一种数据配比能完全保住 IF-eval，即使 100% 用对话数据做 SFT 也不行；
- **LoRA**：学到的知识更少，而且仍然会遗忘后训练的行为；
- **OPD 修复**：用**原来的 Qwen3-8B 当教师**，在 Tulu3 prompt 上做 OPD（不需要内部文档），恢复了大部分 IF-eval，同时保住了新知识，对内部 QA 还有一点正迁移。

## 实验三：数据复用

只用**一个 prompt**，连续训练 20 步，每步 256 条 rollout（共 5,120 条），学生在 AIME'24 上就接近了教师水平。默认配置是每 batch 64 个 prompt × 每题 4 条 = 256 条。

## 实验四：稠密奖励 vs 稀疏奖励

**设置**：先用 RL（LoRA rank 128，DeepMath 数据）把 Qwen3-8B-Base 训成教师，再用 OPD 蒸馏回 base 模型。

- OPD 用少 7–10 倍的梯度步数就追平了教师：AIME 在 10 步内恢复，RL 需要约 70 步；
- 整体算力效率估计提高 50–100 倍；
- **信息量论证**：RL 每个回合约 $O(1)$ 比特，蒸馏约 $O(N)$ 比特（$N$ 为 token 数）；
- **附带好处**：轨迹末尾没有奖励断崖，可以用更短的上下文；梯度噪声更小，可以用更小的 batch。

## 实验五：持续学习

- **在自己的输出上做 SFT 也会退化**：用模型自己 temperature=1.0 的输出做 SFT（期望 KL 为 0），在任何非零学习率下 IF-eval 都会下降。作者推测是有限 batch 的噪声让训练逐渐偏离 on-policy；
- **OPD 不会**：教师固定，学生始终 on-policy，会收敛到教师的行为而不退化。

## 作者自己指出的局限

- 教师的行为必须在学生的支持集内；从 base 模型开始时，需要更大的 batch；
- 只用了 reverse KL、折扣因子 0、sampled-token 估计，没有做 logit 蒸馏；
- 数学结果只覆盖 AIME'24 和 GPQA-Diamond；内部 QA 评测是私有的；
- 部分对比依赖外推（SFT-2M）和 FLOPs 估算，和墙钟时间并不一致；
- "reverse KL 不可被 hack"、"交替阶段做持续学习"等说法属于展望，尚未证实。

## 可以复现的代码

配方在 [tinker-cookbook](../frameworks/others.md)，跑在 Tinker 托管服务上。原始的 Qwen3-8B-Base 和 Qwen3-32B 已于 2026 年 6 月下线，现在改用 Qwen3.5-9B-Base（学生）和 Qwen3.5-9B（教师）。
