# verl

verl 官方支持 on-policy 蒸馏，包括单教师和多教师（MOPD）。<span class="verified">✅</span> 本页内容整理自 [verl OPD 文档](https://verl.readthedocs.io/en/latest/algo/opd.html)。

## 两种 loss

| 模式 | 设置 | 做法 |
|---|---|---|
| **PG OPD** | `use_policy_gradient: true`，`loss_mode: k1` | 把逐 token reverse KL 的负值当作奖励，走 PPO 式的裁剪策略梯度（Thinking Machines 的做法） |
| **GKD OPD** | `use_policy_gradient: false`，`loss_mode: forward_kl_topk` | 在学生采样的状态上，对教师 top-K 计算 forward KL，直接反向传播 |

单样本估计器可选：`kl`、`k1`、`abs`、`mse`、`k2`、`low_var_kl`、`k3`，默认是 `k3`。

两条限制：

- `k1` 搭配 `use_policy_gradient: false` 会报 `ValueError`，因为 k1 直接反传没有有用的梯度（推导见[目标函数与梯度](../theory/objective.md)）；
- `forward_kl_topk` 搭配 `use_policy_gradient: true` 会给出警告，因为策略梯度只作用于采样的那个 token，top-K 的信息大部分被浪费。

## 单教师配置

```yaml
distillation:
  enabled: true
  n_gpus_per_node: 2          # 教师池每个节点的 GPU 数
  nnodes: 1                   # 开启时必须 ≥ 1
  teacher_models:
    teacher_model:
      model_path: Qwen/Qwen3-32B      # 必须和学生用同一个 tokenizer
      inference:
        name: vllm
        tensor_model_parallel_size: 2
        gpu_memory_utilization: 0.5
  distillation_loss:
    loss_mode: k1
    use_policy_gradient: true
    policy_loss_mode: vanilla         # 目前只支持 vanilla
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
    topk: 32                          # top-K 模式使用；同时决定 vLLM 的 max_logprobs
    use_task_rewards: false           # true 时叠加 PPO / GRPO 的任务奖励
    distillation_loss_coef: 1.0

actor_rollout_ref:
  actor:
    use_kl_loss: false        # 一定要关，否则学生同时被拉向参考模型和教师
algorithm:
  use_kl_in_reward: false
```

## 主要配置项

| 配置 | 默认值 | 说明 |
|---|---|---|
| `distillation.enabled` | `false` | 开启后分配教师资源池，actor 的 loss 换成 `distillation_ppo_loss` |
| `distillation.n_gpus_per_node` | `8` | 教师池每节点 GPU 数 |
| `distillation.nnodes` | `0` | 开启时必须 ≥ 1 |
| `distillation.teacher_key` | `data_source` | 多教师时用来路由的数据字段 |
| `teacher_models.<name>.key` | `null` | 路由值；单教师时自动设为 `default` |
| `teacher_models.<name>.num_replicas` | `0` | 单教师时自动填充 |
| `distillation_loss.loss_mode` | `k3` | `forward_kl_topk` 或某个单样本估计器 |
| `distillation_loss.topk` | `32` | top-K 模式使用 |
| `distillation_loss.use_task_rewards` | `true` | 是否叠加任务奖励 |
| `distillation_loss.distillation_loss_coef` | `1.0` | 叠加任务奖励时蒸馏项的权重：总 loss = 策略 loss + 系数 × 蒸馏 loss |
| `distillation_loss.loss_max_clamp` | `null` | 逐 token loss 裁剪到 ±该值 |
| `distillation_loss.log_prob_min_clamp` | `null` | 散度计算中 logprob 的下限 |

`global_batch_info` 和 `loss_settings` 在运行时自动填充，不要手动设置。

注意 `use_task_rewards` 的默认值是 `true`。做纯 OPD 时要显式设成 `false`。

## 叠加任务奖励

```yaml
distillation_loss:
  use_task_rewards: true
  distillation_loss_coef: 1.5   # 示例值
```

调 `distillation_loss_coef` 时，对比日志里 `actor/distillation/loss` 和 `actor/pg_loss` 的量级。

## 多教师（MOPD）

每个领域配一个教师，用数据里的某个字段路由：

```yaml
distillation:
  enabled: true
  n_gpus_per_node: 8
  nnodes: 2
  teacher_key: data_source
  teacher_models:
    math_teacher:
      key: gsm8k
      model_path: Qwen/Qwen3-32B
      num_replicas: 2
      inference: { name: vllm, tensor_model_parallel_size: 2 }
    vl_teacher:
      key: geo3k
      model_path: Qwen/Qwen3-VL-32B-Instruct
      num_replicas: 3
      inference: { name: vllm, tensor_model_parallel_size: 4 }
data:
  shuffle: true     # 不打乱的话，会长时间只路由到同一个教师
```

这个例子的 GPU 数：2 × 2 + 3 × 4 = 16，正好等于 8 × 2。

## 常见坑

- **教师池大小必须严格对齐**：`n_gpus_per_node × nnodes` 要等于所有教师 `num_replicas × 每副本 GPU 数` 之和，否则配置校验失败；
- **副本不能跨节点**：每个副本的 GPU 必须放在一个节点内。比如每节点 8 卡配 TP=5，会跨节点报错；
- **名字冲突**：配了其他命名教师时，名为 `teacher_model` 的条目会被**静默丢弃**，要改名；
- **上下文长度**：`inference.max_model_len` 要覆盖 prompt + 回答 + 1；
- **教师温度**：教师总是用温度 1.0。默认配置会复制 rollout 的温度，两者不同时只打一条警告；
- **教师不生成**：教师只对 prompt + 回答打分，额外生成一个丢弃的 dummy token。

## 调试与监控

**sanity check**：把学生设成和教师同一个模型，loss 应该接近 0（由于训练和推理引擎的数值差异，不会正好为 0）。

日志里 `actor/distillation/*` 的指标：

| 指标 | 含义 |
|---|---|
| `loss`、`abs_loss`、`loss_min`、`loss_max` | 蒸馏 loss；k1 这类有正有负的估计器看 `abs_loss` |
| `student_mass`、`teacher_mass` | 双方在教师 top-K 上的概率质量；偏低说明 K 太小或训练不稳 |
| `overlap_ratio` | 师生 top-K 的重叠比例 |
| `overlap_token_advantage` | 交集 token 上的对齐程度 |

top-K 相关的指标只用于记录，不影响梯度。

## 示例脚本

位于 `examples/on_policy_distillation_trainer/`：

| 脚本 | 内容 |
|---|---|
| `run_qwen3_8b_fsdp.sh` | 文本，vLLM rollout，FSDP 学生，单教师 |
| `run_qwen3_8b_megatron.sh` | 同上，Megatron 学生 |
| `run_qwen3_vl_8b_fsdp.sh` | 多模态学生和教师 |
| `run_qwen3_8b_mopd_fsdp.sh` | 多教师（GSM8K 文本 + Geo3K 多模态），按 `data_source` 路由 |
| `run_qwen3_5_2b_fsdp.sh`、`run_qwen3_5_4b_fsdp.sh` | Qwen3.5 小模型 |
| `run_qwen3_0.6b_opd_veomni.sh`、`run_qwen3_8b_mopd_veomni.sh` | VeOmni 后端 |

单教师脚本可以通过环境变量 `STUDENT_MODEL` 和 `TEACHER_MODEL` 换模型对。

## 还有一个异步版本

verl 另有一个异步 on-policy 蒸馏 recipe：用 top-k 稀疏 KL，并让 rollout、教师打分和参数更新三者重叠执行，提高吞吐。见 [Async On-Policy KD Trainer](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html)。

## 多轮 Agent

verl 的多轮 rollout（agent loop）是另一套功能，配置见[动手教程 L4](../tutorial/l4-agentic.md)。

!!! warning "OPD + 多轮 agent loop"
    截至 2026 年 10 月，官方示例里没有同时开启 `distillation.enabled` 和多轮 agent loop 的现成脚本。做 Agentic OPD 前，先在玩具环境上实测两者能否组合，必要时修改代码。
