# L1 SFT 与 off-policy 蒸馏

## 要掌握的

- **chat template**：同一段对话，在不同模型上会被渲染成不同的 token 序列。训练和推理必须用同一个模板。
- **loss mask**：只在 assistant 的 token 上算 loss；system、user、tool 的内容都要 mask 掉。
- **EOS**：每段回答末尾必须带结束符（Qwen 系是 `<|im_end|>`），否则模型学不会停。
- **超参影响**：学习率、epoch、packing、max_length 截断分别会怎样影响结果。
- **off-policy 蒸馏**：就是先用教师生成数据，再拿这批数据做 SFT。

## 数据样例

### 单轮

jsonl，一行一条：

```json
{"messages": [
  {"role": "user", "content": "解方程 3x + 5 = 20"},
  {"role": "assistant", "content": "两边减 5，得 3x = 15，两边除以 3，所以 x = 5。"}
]}
```

### 多轮工具调用

为 L4 做准备。下面是 OpenAI 风格的写法。不同框架的工具字段写法不一样（比如 ms-swift 有自己的 agent 数据格式），以所用框架的文档为准：

```json
{"tools": [{"type": "function", "function": {
   "name": "get_weather", "description": "查询某城市某天的天气",
   "parameters": {"type": "object",
     "properties": {"city": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"}},
     "required": ["city", "date"]}}}],
 "messages": [
  {"role": "user", "content": "帮我查明天新加坡的天气"},
  {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
     "name": "get_weather", "arguments": "{\"city\": \"Singapore\", \"date\": \"2026-10-20\"}"}}]},
  {"role": "tool", "content": "{\"temp\": \"31°C\", \"weather\": \"雷阵雨\"}"},
  {"role": "assistant", "content": "明天新加坡 31°C，有雷阵雨，记得带伞。"}
]}
```

### 模型实际看到的样子

用 Qwen 模板渲染后是下面这样（工具列表会被渲染进 system，这里省略）。注意工具返回会被包在 user 段里：

```text
<|im_start|>user
帮我查明天新加坡的天气<|im_end|>                                   ← mask
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "Singapore", "date": "2026-10-20"}}
</tool_call><|im_end|>                                             ← 算 loss
<|im_start|>user
<tool_response>
{"temp": "31°C", "weather": "雷阵雨"}
</tool_response><|im_end|>                                         ← mask
<|im_start|>assistant
明天新加坡 31°C，有雷阵雨，记得带伞。<|im_end|>                    ← 算 loss
```

## off-policy 蒸馏：用教师生成 SFT 数据

```python
# 教师批量生成 → （可选）按答案过滤 → 写成 SFT 数据
import json
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-32B", tensor_parallel_size=2)
params = SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=8192)  # 每题采 4 条

questions = [json.loads(l) for l in open("questions.jsonl")]   # 每行 {"q": ..., "answer": ...}
outputs = llm.chat([[{"role": "user", "content": x["q"]}] for x in questions], params)

with open("distill_sft.jsonl", "w") as f:
    for x, out in zip(questions, outputs):
        for cand in out.outputs:
            if extract_answer(cand.text) != x["answer"]:   # 过滤是可选的：OpenThoughts 发现收益不明显
                continue
            f.write(json.dumps({"messages": [
                {"role": "user", "content": x["q"]},
                {"role": "assistant", "content": cand.text}]}, ensure_ascii=False) + "\n")
```

`extract_answer` 需要按题目格式自己实现。数据经验见[SFT 与 off-policy 蒸馏](../basics/sft.md)。

## 训练命令

以 ms-swift 为例（参数名以所装版本的文档为准）：

```bash
swift sft \
  --model Qwen/Qwen3-1.7B-Base \
  --dataset distill_sft.jsonl tool_sft.jsonl \
  --train_type full \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --max_length 8192 \
  --output_dir output/sft
```

## 最小实验

| 项目 | 设置 |
|---|---|
| 学生 | Qwen3-1.7B-Base |
| 数学数据 | 从 OpenThoughts3 或 DeepMath 取 1–2 万题，用 Qwen3-8B 或 Qwen3-32B 生成回答 |
| 工具数据 | xlam-function-calling-60k 或 APIGen-MT 的一个子集，和数学数据混合训练 |
| 评测 | GSM8K、MATH-500，以及工具调用格式的正确率 |

## 看什么指标

train/eval loss、生成能否正常结束（EOS 率）、格式正确率、benchmark 分数。

## 过关标准

- [ ] 能说清楚模板渲染后的 token 序列长什么样，loss mask 落在哪些位置
- [ ] 模型能稳定结束生成，格式正确
- [ ] 分数相对 Base 模型有明确提升
- [ ] 能识别过拟合，比如 eval loss 回升、开始复读

## 常见坑

- 训练和推理用的模板不一致；
- 在 prompt 上也算了 loss；
- 漏掉 EOS，导致模型不停地生成；
- 全参微调时学习率太大；
- max_length 把长 CoT 截断了却没发现：被截断的样本没有结尾，模型会学成不收尾。

**产出**：`ckpt_sft`，后面所有阶段的共同起点。

下一步：[L2 GRPO](l2-grpo.md)。
