# L4 Agentic RL

!!! note "Agentic RL 不等于 OPD"
    Agentic RL 是**问题设定**：训练智能体在环境里多轮行动，目标是把任务做成。OPD 是**训练信号**：教师逐 token 打分。两者可以组合（Agentic OPD），但不能互相替代。平时说的 Agentic RL 大多指用环境的结果奖励训练；OPD 进入智能体领域是比较新的方向。

这一阶段分两步：

- **L4a 多轮 GRPO**：Agentic RL 的基础，用结果奖励；
- **L4b Agentic OPD**：在同样的环境里，把结果奖励换成（或加上）教师逐 token 打分。

## L4a 多轮 GRPO

### 要掌握的

- **环境和工具**：工具怎么注册、怎么执行，超时和报错怎么处理；
- **多轮 rollout 的循环**：模型输出 → 解析出 tool call → 执行工具 → 把结果拼回上下文 → 模型继续输出，直到任务结束或达到轮数上限；
- **loss mask**：只训练模型自己写的 token；工具返回、用户（或用户模拟器）的发言全部 mask；
- **奖励**：看任务有没有完成（比对最终状态或调用序列）。可以加格式分，但要防止被钻空子；
- **上下文预算**：工具返回可能很长，需要截断。

### 数据样例

**磁盘上：**

```python
{
    "data_source": "tool_weather",
    "prompt": [
        {"role": "system", "content": "你是助手，可以调用工具。今天是 2026-10-19。"},
        {"role": "user", "content": "帮我查明天新加坡的天气"},
    ],
    "ability": "tool_use",
    "reward_model": {"style": "rule",
                     "ground_truth": '{"name": "get_weather", "arguments": {"city": "Singapore", "date": "2026-10-20"}}'},
    "extra_info": {"split": "train", "index": 0},
}
```

**工具定义**（verl 的 `@function_tool`：schema 从函数签名和 Google 风格 docstring 自动生成，每个参数都要有类型标注和 `Args:` 说明）：

```python
import re
from verl.tools.function_tool import function_tool

@function_tool
def get_weather(city: str, date: str) -> dict:
    """查询某城市某天的天气。

    Args:
        city: 城市英文名，如 "Singapore"。
        date: 日期，格式 YYYY-MM-DD。
    """
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return {"error": "date 必须是 YYYY-MM-DD 格式"}
    return {"temp": "31°C", "weather": "雷阵雨"}
```

**开启多轮**（最新版 verl 的写法；不同版本键名不一样，旧版用 `tools_config_file` 等，以所装版本的文档为准）：

```yaml
data:
  return_raw_chat: True
actor_rollout_ref:
  rollout:
    mode: async
    multi_turn:
      enable: True
      format: hermes
    function_tool_path: tools/weather_tools.py
    agent:
      default_agent_loop: tool_agent
```

**奖励函数**（多轮时 `solution_str` 里包含什么、有没有工具返回，因版本而异，先打印出来看一眼）：

```python
import json, re

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    gt = json.loads(ground_truth)
    for c in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", solution_str, re.S):
        try:
            call = json.loads(c)
        except json.JSONDecodeError:
            continue
        if call.get("name") == gt["name"] and call.get("arguments") == gt["arguments"]:
            return 1.0
    return 0.0
```

**训练时生成：**一条轨迹，以及每一段是否参与训练：

```text
[user]      帮我查明天新加坡的天气                                    ← mask
[assistant] <tool_call>{"name":"get_weather",
              "arguments":{"city":"Singapore","date":"明天"}}</tool_call>    ← 训练
[tool]      {"error":"date 必须是 YYYY-MM-DD 格式"}                    ← mask
[assistant] <tool_call>{"name":"get_weather",
              "arguments":{"city":"Singapore","date":"2026-10-20"}}</tool_call>  ← 训练
[tool]      {"temp":"31°C","weather":"雷阵雨"}                         ← mask
[assistant] 明天新加坡 31°C，有雷阵雨，记得带伞。                       ← 训练
```

**一组 4 条轨迹的奖励：**

| 轨迹 | 奖励 | advantage |
|---|---|---|
| 一次就调对，然后回答 | 1 | +0.87 |
| 先填错日期，收到报错后改对（上面这条） | 1 | +0.87 |
| 填错日期，收到报错后放弃，自己编了个天气 | 0 | −0.87 |
| 不调工具，直接回答"不知道" | 0 | −0.87 |

**GRPO 在多轮任务里的盲点**："一次调对"和"绕路调对"拿到同样的分数，第二条里写错的 `"明天"` 也一起被奖励了。这正是 L4b 要解决的信用分配问题。

### 环境选择（由易到难）

1. **自己写的玩具工具**（天气、计算器）：只为跑通流程、检查 loss mask；
2. **搜索问答**（Search-R1 类：检索工具 + QA 数据集；verl 文档里有搜索工具的示例）；
3. **ALFWorld / WebShop**：文本交互环境，OPD 相关论文常用（比如 Revisiting OPD 用了 ALFWorld）；
4. **τ²-bench**（retail / airline / telecom）：带用户模拟器（STRIDE 用的就是 retail）。注意用户模拟器本身也要占 GPU 或花 API 费用。

BFCL 主要是评测集，不适合当训练环境。

### 看什么指标

任务成功率、平均轮数、有效工具调用率（JSON 能解析且参数合法）、过度调用率（不该调用时也调用）、工具报错率、每轮长度、上下文截断率、环境超时率。

### 过关标准

- [ ] 在玩具环境上 reward 能上升，并且能逐 token 检查一条轨迹，确认 loss mask 是对的
- [ ] 在一个真实环境（ALFWorld 或 τ²-bench retail）上，多轮 GRPO 相对 `ckpt_sft` 有提升
- [ ] 能说清 GRPO 在多轮任务里为什么分不清"一次调对"和"绕路调对"

### 常见坑

- 工具返回没有被 mask，模型去学工具的输出了；
- 训练和推理的工具调用格式不一致（hermes 格式 vs 模型原生格式）；
- 工具输出太长，把上下文挤爆；
- 环境超时，拖慢整个训练步；
- 用户模拟器不稳定，导致奖励本身有噪声。

## L4b Agentic OPD

和 L4a 共用环境、工具和数据，只改一处：每个 token 的权重从"结果奖励"换成"教师逐 token 打分"，或者两者相加。

### 训练时生成

还是同一条轨迹。教师在完整上下文（包括工具返回）上做一次前向，只取模型自己写的那些 token 的概率：

| 段落 | token（示意） | 学生概率 | 教师概率 | $r_t$ |
|---|---|---|---|---|
| 第 1 次调用 | `"date": "` | 0.95 | 0.96 | +0.01 |
| | `明天` | 0.60 | 0.02 | **−3.40** |
| 工具返回 | `{"error": ...}` | — | — | mask |
| 第 2 次调用 | `2026-10-20` | 0.70 | 0.92 | +0.27 |
| 最终回答 | `明天新加坡 31°C……` | ≈ | ≈ | ≈ 0 |

对比 GRPO：这条轨迹最后成功了，三段 assistant 内容都拿 +0.87，包括写错的 `明天`。OPD 则精确地把 `明天` 压了下去。

### Agentic OPD 特有的研究问题

- **教师在学生的上下文里是否可靠**：学生一旦走错（比如调错工具、拿到无关结果），后面的教师信号就不可靠了；
- **过度调用**：工具教师会把学生推向"什么都调工具"；top-K 截断还可能漏掉"开始调用工具"的那个 token；
- **轮次权重和课程**：由短到长、按轮次退火；
- **和其他稠密信号对比**：过程奖励模型（PRM）、按轮次给奖励、自蒸馏（SDPO）；
- **教师从哪来**：L4a 里用多轮 GRPO 训出来的较大模型可以直接当 L4b 的教师。

相关工作（STRIDE、State-OPD、ATOD、TCOD、SOD 等）见 [Agentic OPD](../frontier/agentic.md)。

!!! warning "框架支持"
    verl 的 `distillation.enabled` 能不能和多轮 agent loop 同时打开，截至 2026 年 10 月官方示例里没有现成脚本。要先在玩具环境上实测，必要时改代码。tinker-cookbook 有现成的多轮工具调用 OPD 配方可以参考（见 [slime 与 Tinker](../frameworks/others.md)）。

### 实验设计

全部从 `ckpt_sft` 出发：

| 组 | 做法 |
|---|---|
| (a) | 多轮 SFT 继续训（同等算力） |
| (b) | 多轮 GRPO（即 L4a） |
| (c) | Agentic OPD |
| (d) | Agentic OPD + 结果奖励 |
| (e) | 你的方法（比如位置加权、提前停止、轮次课程） |

评测指标：任务成功率、过度调用率、平均轮数、域外能力保持，以及同等 GPU 小时下的效率。
