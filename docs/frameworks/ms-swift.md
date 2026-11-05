# ms-swift

ms-swift 用它的 GKD 训练器支持 on-policy 蒸馏。中文文档完善，单机上跑通最快。本页整理自 [ms-swift GKD 文档](https://swift.readthedocs.io/en/v4.2/Instruction/GKD.html)。<span class="verified">✅</span>

## 两个关键参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--lmbda` | 0.5 | 每一步使用学生自己生成回答（on-policy）的概率。0 = 纯 off-policy，1 = 纯 on-policy |
| `--beta` | 0.5 | 散度插值。0 = forward KL，0.5 = JSD，1 = reverse KL |

**`--lmbda 1 --beta 1` 就是 Thinking Machines 式的 OPD**：纯 on-policy + reverse KL。

## 最小命令

教师和学生在同一个训练进程里：

```bash
swift rlhf \
  --rlhf_type gkd \
  --model Qwen/Qwen2.5-7B-Instruct \
  --teacher_model Qwen/Qwen2.5-14B-Instruct \
  --dataset your_dataset \
  --lmbda 1 \
  --beta 1 \
  --temperature 0.9 \
  --max_completion_length 512
```

## 教师单独起服务（推荐）

教师较大时，单独用 vLLM 起一个服务，训练进程不加载教师：

```bash
# 1. 起教师服务
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-14B-Instruct \
  --port 8000 \
  --max-logprobs 64 \
  --gpu-memory-utilization 0.9

# 2. 训练学生
CUDA_VISIBLE_DEVICES=1 swift rlhf \
  --rlhf_type gkd \
  --model Qwen/Qwen2.5-7B \
  --teacher_model_server http://localhost:8000 \
  --gkd_logits_topk 64 \
  --dataset your_dataset \
  --lmbda 1.0 \
  --beta 1.0 \
  --use_vllm true \
  --vllm_mode colocate
```

## 其他参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--teacher_model` | None | 用 `--teacher_model_server` 时可以不填 |
| `--teacher_model_server` | None | 外部教师服务地址，只支持 `vllm serve` |
| `--gkd_logits_topk` | None | 只在教师 top-K 上算 KL。用外部教师服务时必填；不能和 `--use_liger_kernel` 同时用 |
| `--seq_kd` | False | 没触发 on-policy 时：True 用教师生成的回答，False 用数据集里的回答 |
| `--temperature` | 0.9 | 学生采样温度 |
| `--max_completion_length` | 512 | 最大生成 token 数 |
| `--sft_alpha` | 0 | 在非学生生成的回答上额外混入 SFT loss |

## 数据格式

和 SFT 一样用 `messages` 格式。纯 on-policy（`lmbda=1`）时只需要 user 部分，回答由学生现场生成：

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "解方程 3x + 5 = 20"}
]}
```

## 坑

- **没开 vLLM 时会悄悄退回 off-policy**：`lmbda > 0` 却没有用 vLLM 采样时，训练会直接用数据集里的回答，变成普通的 off-policy 蒸馏，不报错；
- **top-K 不能超过服务端上限**：`--gkd_logits_topk` 不能超过教师服务的 `--max-logprobs`，而 vLLM 默认只有 20；
- **vLLM 只加速学生采样**：`seq_kd=True` 时的教师生成仍然走 Transformers，很慢。建议先用 `swift infer --infer_backend vllm` 离线生成教师数据，再用 `--seq_kd false` 训练；
- **Megatron 路径的限制**：Megatron-SWIFT 的 GKD 不支持 `seq_kd=True`，会退回 off-policy。

## 自蒸馏

ms-swift 还支持 OPSD 式的自蒸馏：同一个模型既当教师又当学生，教师侧额外看到一个 `teacher_prompt` 字段（比如带标准答案）。见[自蒸馏](../frontier/self-distillation.md)。
