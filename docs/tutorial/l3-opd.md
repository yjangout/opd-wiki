# L3 OPD

## 要掌握的

- $r_t = \log \pi_T(y_t) - \log \pi_S(y_t)$ 表示的是"相对认可"：学生越接近教师，信号越趋近 0；
- 估计粒度的三个档位：sampled-token（k1）→ top-K 重归一化 → 全词表；
- 怎么选教师：同 tokenizer、思维模式兼容、有增量能力、尺寸适中；
- 怎么诊断：重叠率、teacher_mass、长度、域外能力。

原理见[目标函数与梯度](../theory/objective.md)和[OPD 何时有效、何时失效](../theory/mechanism.md)。

## 数据样例

### 磁盘上

直接复用 L2 的 parquet 最省事。纯 OPD 用不到 `reward_model` 字段，保留它是为了方便做"OPD + 结果奖励"的对照。

### 训练时生成

学生生成一条回答，教师对整条做一次前向，给出每个已采样 token 的概率（按词切分示意）：

| 学生采样的 token | 学生概率 | 教师概率 | $r_t = \ln(\text{教师}/\text{学生})$ | 含义 |
|---|---|---|---|---|
| 两边 | 0.80 | 0.85 | +0.06 | 教师略更认可，轻微强化 |
| 减 | 0.90 | 0.95 | +0.05 | 同上 |
| 5 | 0.95 | 0.97 | +0.02 | 几乎一致 |
| ，得 3x = | ≈ | ≈ | ≈ 0 | 一致 |
| **25** | 0.30 | **0.01** | **−3.40** | **算错的地方，被重重压低** |
| ，所以 | 0.90 | 0.30 | −1.10 | 教师更想接"等等，算错了" |
| x = | 0.95 | 0.90 | −0.05 | 一致 |
| 25/3 | 0.90 | 0.85 | −0.06 | 前文已经错了，教师也认可这个局部接得上的续写 |

同一条轨迹放在 GRPO 下：最终答案判错，组内均值 0.5 时，所有 token 统一拿 −0.87。

真实模型上的数值，见[verl 源码走读](../frameworks/verl-walkthrough.md)的交互演示。

## 最快跑通：ms-swift

```bash
# 1. 教师单独起服务（占 1 张卡）
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B --port 8000 --max-logprobs 64

# 2. 学生训练：lmbda=1 纯 on-policy，beta=1 reverse KL
CUDA_VISIBLE_DEVICES=1 swift rlhf \
  --rlhf_type gkd \
  --model output/sft/ckpt_sft \
  --teacher_model_server http://localhost:8000 \
  --gkd_logits_topk 64 \
  --lmbda 1 --beta 1 \
  --dataset prompts.jsonl \
  --max_completion_length 4096 \
  --use_vllm true --vllm_mode colocate
```

- `--gkd_logits_topk` 不能超过服务端的 `--max-logprobs`（vLLM 默认只有 20）；
- 如果 `lmbda > 0` 却没开 vLLM，训练会**悄悄退回 off-policy**，直接用数据集里的回答。一定要确认采样真的来自学生。

详见 [ms-swift](../frameworks/ms-swift.md)。

## 正式研究：verl

```yaml
distillation:
  enabled: true
  n_gpus_per_node: 1
  nnodes: 1
  teacher_models:
    teacher_model:
      model_path: Qwen/Qwen3-8B
      inference: { name: vllm, tensor_model_parallel_size: 1 }
  distillation_loss:
    loss_mode: k1
    use_policy_gradient: true
    use_task_rewards: false
actor_rollout_ref:
  model:
    path: output/sft/ckpt_sft
  actor:
    use_kl_loss: false
algorithm:
  use_kl_in_reward: false
```

完整配置项和坑见 [verl](../frameworks/verl.md)。上手前先做一次 sanity check：学生和教师设成同一个模型，loss 应该接近 0。

## 最小实验

| 项目 | 设置 |
|---|---|
| 学生 | `ckpt_sft`（Qwen3-1.7B） |
| 教师 | Qwen3-8B（同家族、同 tokenizer） |
| 数据 | 和 L2 相同的题目 |
| 主对照 | 相同步数或相同 GPU 小时下，和 L2 的 GRPO 比验证集分数 |
| 消融 1 | 跳过 SFT 预热，直接从 Base 做 OPD，观察重叠率是不是起不来 |
| 消融 2 | 换成更大的教师 Qwen3-32B，看分数是不是反而不涨 |

## 看什么指标

逐 token KL（`abs_loss`）、`overlap_ratio`、`teacher_mass`、学生熵、回答长度和截断率、域外能力（GPQA 或 MMLU 的子集）。详见[监控与诊断](../practice/monitoring.md)。

## 过关标准

- [ ] 复现出"OPD 比 GRPO 更快接近教师水平"
- [ ] 能用重叠率解释"为什么这个教师不好教"
- [ ] 至少遇到过一次长度暴涨或熵塌缩，并用以下办法之一修好：合并 EOS、rollout 改用 top-p、混入 OOD prompt
- [ ] 能讲清 sampled-token、top-K、全词表三者的取舍

## 常见坑

见[踩坑大全](../practice/pitfalls.md)：ref KL 没关、tokenizer 或 EOS 不一致、教师温度不对、轨迹后段信号失真等。

下一步：[L4 Agentic RL](l4-agentic.md)。
