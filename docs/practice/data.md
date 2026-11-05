# 数据与 prompt

OPD 的数据只有 prompt，但 prompt 选得好不好，对结果影响很大。

## 对齐教师训练时的 prompt

Rethinking OPD 的发现 <span class="verified">✅</span>：

**模板对齐**：把 DAPO-Math-17K 原来的指令模板，换成教师 JustRL 在 RL 训练时用的模板（"Please reason step by step..."），验证集分数和重叠率都更高，上升也更快。

**内容对齐**：用教师 RL 训练时见过的 DAPO-Math-17K 题目，效果好于去重后的 DeepMath 子集。有意思的是，这时重叠率反而更低，但高概率质量几乎全部落在交集 token 上，实际对齐得更好。

原因很直接：教师在自己训练过的分布附近，给出的信号最可靠。

## 但不能只用教师见过的 prompt

同一项研究也发现：**只用**教师训练过的 prompt，学生的熵会灾难性地坍塌。<span class="verified">✅</span>

做法：保留一部分对齐的 prompt，再混入一定比例的 OOD prompt 来维持多样性。

## 数据可以反复用

Thinking Machines 做了一个极端实验 <span class="verified">✅</span>：**只用一个 prompt**，连续训练 20 步，每步 256 条 rollout（共 5,120 条），学生在 AIME'24 上就接近了教师水平。

OPD 比 SFT 更不怕重复使用数据：每一步的 rollout 都来自当前的学生，同一个 prompt 每次产生的状态都不一样，教师给的反馈也不一样。

## 修复遗忘时用通用 prompt

如果目标是恢复通用能力（比如注入新知识后的指令跟随），prompt 应该是**通用的**，而不是新领域的。Thinking Machines 用 Tulu3 的 prompt，并发现用 Qwen3-8B 自己重采样的回答作背景数据，比用 Qwen3-32B 的回答更能保住对话能力。<span class="verified">✅</span>

## 多教师时要打乱数据

多教师 OPD 按某个字段（verl 里默认是 `data_source`）把题目路由给对应的教师。如果几个数据集只是简单拼接、没有打乱，训练会在很长一段时间里只路由到同一个教师，训练曲线呈锯齿状。在 verl 里要设 `data.shuffle: true`。<span class="verified">✅</span>

## 控制长度预算

- 最大生成长度先设 4K–8K。Rethinking OPD 的扫描里 3K 和 7K 最稳，10K 以上收益见顶甚至回落；<span class="verified">✅</span>
- 长思维链任务如果必须用长上下文，就要配合位置加权、截断或提前停止，见[踩坑大全](pitfalls.md)；
- 统计截断率：被截断的轨迹没有结尾，如果大量出现，会把训练带偏。

## 过滤要过滤"位置"，不要过滤"轨迹"

有人会想：只在**成功**的轨迹上做蒸馏，岂不是更干净？这样做恰恰丢掉了 OPD 最核心的价值：**在学生犯错的状态上学会纠错**，还会重新引入偏差。

更合理的过滤对象是**不可靠的位置或 token**：

- 教师信号已经失效的轨迹后段；
- special token、结束符等容易因为切分不一致产生假信号的 token；
- 教师在该位置极不确定的 token（如 EOPD 用教师熵做门控）。

失败轨迹的前段照样要学。

## 每题采几条

- OPD 不需要组内比较，每题采 1–4 条就有信号。Thinking Machines 默认每题 4 条；<span class="verified">✅</span>
- 叠加 GRPO 式结果奖励时，组内要有足够的样本才能归一化，通常 4–8 条。
