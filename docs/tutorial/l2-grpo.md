# L2 GRPO

## 要掌握的

- 组内采样和 advantage 归一化；为什么全对或全错的组没有学习信号；
- 奖励函数怎么写，以及它会怎样被模型钻空子；
- PPO clip 和 KL to reference 各起什么作用；
- 训推不一致：vLLM 采样时的概率和训练引擎重算的概率对不上；
- 读懂曲线：reward、长度、熵、clip 比例。

原理见[RL 与 GRPO](../basics/grpo.md)。

## 数据样例

### 磁盘上

verl 的 parquet 格式，一行一条：

```python
{
    "data_source": "openai/gsm8k",     # 用来选奖励函数；多教师 OPD 时也用它路由到对应教师
    "prompt": [{"role": "user",
                "content": "解方程 3x + 5 = 20。请逐步推理，最后把答案写在 \\boxed{} 里。"}],
    "ability": "math",
    "reward_model": {"style": "rule", "ground_truth": "5"},
    "extra_info": {"split": "train", "index": 0},
}
```

### 奖励函数

配置里用 `custom_reward_function.path` 指向这个文件；只有一个函数时，名字用默认的 `compute_score` 即可：

```python
import re

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    m = re.findall(r"\\boxed\{([^}]*)\}", solution_str)
    if not m:
        return 0.0                         # 没按格式作答
    return 1.0 if m[-1].strip() == ground_truth else 0.0
```

### 训练时生成

每题采 $G = 4$ 条：

| 学生回答 | 最终答案 | 奖励 | advantage |
|---|---|---|---|
| 两边减 5，得 3x = 15，所以 x = 5 | 5 | 1 | +0.87 |
| 两边减 5，得 3x = 25，所以 x = 25/3 | 25/3 | 0 | −0.87 |
| 3x = 20 − 5 = 15，x = 5 | 5 | 1 | +0.87 |
| 移项得 3x = 20 + 5，x = 25/3 | 25/3 | 0 | −0.87 |

- advantage = (奖励 − 组内均值 0.5) / 组内样本标准差 0.577；
- 第二条回答的每个 token 都是 −0.87，包括前面本来正确的"两边减 5"；
- 全错的组 [0, 0, 0, 0]：advantage 全为 0，这一步对这道题的采样就白做了。DAPO 的 dynamic sampling 就是把这类组过滤掉，再补采新题。

## 最小实验

从 `ckpt_sft` 出发，用 GSM8K 或 DAPO-Math-17k，$G = 8$，最大回答长度 2K–4K。verl 仓库自带 GSM8K 预处理脚本（`examples/data_preprocess/gsm8k.py`）和 GRPO 示例，改一下模型路径就能跑：

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  actor_rollout_ref.model.path=output/sft/ckpt_sft \
  actor_rollout_ref.rollout.n=8 \
  custom_reward_function.path=reward/math_reward.py
  # 其余参数照抄官方 GRPO 示例脚本
```

## 看什么指标

- reward 均值（应该上升）、验证集准确率；
- 回答长度、熵（不应该塌到接近 0）；
- `pg_clipfrac`、KL to ref（如果开了的话）；
- 全对组和全错组的占比。

## 过关标准

- [ ] reward 稳定上升，验证集有提升
- [ ] 能解释回答长度为什么涨或跌
- [ ] 自己改一次奖励函数，亲眼看到一次 reward hacking（比如只奖励格式）
- [ ] 调过 $G$、温度、学习率，能说出各自的影响

## 常见坑

- 奖励函数解析出错，产生大量假阴性；
- 全对或全错的组太多（题目太简单或太难）；
- 熵塌缩；
- 长度暴涨、大量回答被截断；
- 训推概率不一致，导致训练不稳。

**产出**：GRPO 的训练曲线和 checkpoint，作为 L3 的对照组。

下一步：[L3 OPD](l3-opd.md)。
