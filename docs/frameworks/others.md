# slime 与 Tinker

## slime

slime 是面向大规模 MoE 训练的 RL 框架（Megatron + SGLang）。它的 OPD 示例里，Qwen3-8B 学生在自己的 rollout 上逐 token 对齐 Qwen3-32B 教师的 logprob。<span class="verified">✅</span>

**两种教师模式：**

- 外部 SGLang 服务：教师放不进训练 GPU 时使用；
- 由 Megatron 加载：教师和策略模型架构相同时使用。

**主要参数：**

| 参数 | 说明 |
|---|---|
| `--use-opd` | 开启 OPD |
| `--opd-type` | OPD 类型 |
| `--opd-kl-coef` | KL 系数，默认 1.0 |
| `--opd-teacher-load` | 教师权重路径 |

slime 的 OPD 以**加性 KL 惩罚**的形式叠加在任意 advantage 估计器（GRPO、PPO 等）上，所以天然支持"OPD + 结果奖励"。

上面的说明整理自昇腾适配版 [slime-ascend 的 OPD 示例](https://gitcode.com/Ascend/slime-ascend/tree/main/examples/on_policy_distillation)（`on_policy_distillation.py`）。上游 slime 仓库的示例路径和参数以仓库为准。

## Tinker 与 tinker-cookbook

Tinker 是 Thinking Machines 的托管训练 API，tinker-cookbook 是配套的开源示例。OPD 的原始博客就是在 Tinker 上做的。

!!! note "训练跑在托管服务上"
    tinker-cookbook 的命令使用 `tinker://` 形式的 checkpoint 路径，训练在 Thinking Machines 的托管服务上运行，不能直接用本地 GPU 跑。它最大的价值是**代码短、配方完整**，适合读。

**蒸馏相关的配方** <span class="verified">✅</span>（`tinker_cookbook.recipes.distillation` 下）：

| 配方 | 模块 | 模型 | 数据 | 主要超参 |
|---|---|---|---|---|
| off-policy 推理 SFT | `off_policy_reasoning` | 学生 Qwen3.5-9B-Base | OpenThoughts3 | lr 1e-3（LoRA）/ 1e-4（全参），batch 128，LoRA rank 128，3000 步 |
| on-policy 推理 | `on_policy_distillation` | 学生 Qwen3.5-9B-Base，教师 Qwen3.5-9B | DeepMath | lr 1e-4（LoRA）/ 5e-5（全参），groups_per_batch=512，LoRA rank 128，max_tokens=16384，200 步 |
| on-policy 个性化 | `on_policy_distillation` | 同上 | Tulu3 prompt（先做自己的 SFT） | lr 1e-4，groups_per_batch=64，LoRA rank 128 |
| 多轮工具调用 | `on_policy_distillation_harbor_multi_turn` | 学生和教师都是 Kimi-K2.6 | Harbor 沙箱任务（如 Terminal-Bench） | max_turns=10，group_size=4，groups_per_batch=8，lr 1e-4，LoRA rank 8，max_trajectory_tokens=24576 |
| 多教师 | `on_policy_multi_teacher` | 学生 Qwen3.5-9B | DeepMath + Tulu3，各配一个教师 | lr 1e-4，每个数据集 256 groups/batch，LoRA rank 128 |

**报告的结果：**

- off-policy SFT：LoRA rank 128，3000 步后 AIME'24 约 65%；
- on-policy 推理：LoRA rank 128，200 步、16K rollout 后 AIME'24 约 76.7%；
- 个性化：约 100 步内 IF-eval 恢复。

**多轮工具调用配方的特别之处**：唯一的训练信号是对教师的 KL，奖励函数始终返回 0；消息级的循环由 `AgentToolMessageEnv` 管理（追加 assistant 消息 → 执行工具调用 → 检查是否结束）。

!!! note "模型下线"
    原始博客用的 Qwen3-8B-Base 和 Qwen3-32B 已于 2026 年 6 月从 Tinker 下线，cookbook 现在改用 Qwen3.5-9B-Base（学生）和 Qwen3.5-9B（教师）。
