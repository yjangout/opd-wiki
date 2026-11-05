# 估计粒度：sampled-token、top-K、全词表

每个位置上，学生和教师的"距离"可以算得粗一点，也可以算得细一点。粒度越细，信号越准，但代价越大。

## 三个档位

| 档位 | 每个位置需要教师给什么 | 怎么算 | 代表 |
|---|---|---|---|
| **sampled-token** | 学生采样到的那 1 个 token 的 logprob | 单样本估计（k1 / k2 / k3） | Thinking Machines、verl PG OPD、slime |
| **top-K** | 教师（或学生）概率最高的 K 个 token 的 logprob | 在这 K 个 token 上算截断的 KL，可选重归一化 | verl GKD OPD、TRL、ms-swift、LSM |
| **全词表** | 整个词表上的分布 | 精确 KL，直接反向传播 | DeepSeek-V4 |

**存储代价**：以 Qwen 系约 15 万的词表为例，一条 1K token 的样本，全词表需要约 1.5 亿个 logprob；sampled-token 只需要 1K 个。<span class="verified">✅</span>（数字来自 Hugging Face TRL 博客。）

## 负反馈之后，概率质量流向哪里

这是理解三种粒度差别的关键。

### sampled-token：流向学生原本就喜欢的 token

设学生在某个位置的 logits 为 $z$，概率 $p = \operatorname{softmax}(z)$。sampled-token OPD 对采样到的 token $y$ 施加 advantage $A$，等价于最小化 $-A \log p_y$。对每个 logit 求梯度：

$$
\frac{\partial (-A \log p_y)}{\partial z_v} = -A\,\big(\mathbb{1}[v = y] - p_v\big)
$$

当 $A < 0$（要压低 $y$）时，梯度下降会：

- 降低 $z_y$，幅度正比于 $|A|(1 - p_y)$；
- **提高其他所有 token 的 logit，幅度正比于学生自己当前的概率 $p_v$**。

也就是说，被压下去的概率质量会按**学生原有的偏好**分给其他 token，而不是流向教师想要的那个。如果学生的第二选择也是错的，压掉第一个错误 token 只会让第二个错误 token 上位。学生起点很弱时，这会形成 AOPD 所说的 **exploration black hole**：错误 token 被不断压低，正确 token 却始终抬不起来。<span class="unverified">⚠️</span>

### top-K 和全词表：直接朝教师分布推

对一个位置上的 reverse KL $D = \sum_v p_v(\log p_v - \log q_v)$（$q$ 是教师分布）求 logit 梯度：

$$
\frac{\partial D}{\partial z_v} = p_v\big(\log p_v - \log q_v - D\big)
$$

学生相对教师高估的 token（$\log p_v - \log q_v > D$）被压低，低估的被抬高。每一步更新都同时作用在所有候选 token 上，概率质量会**直接流向教师支持、学生也给了一定概率的 token**。

## 各档位的证据

不同论文的结论并不完全一致，下面把它们放在一起：

| 来源 | 设置 | 结论 |
|---|---|---|
| Rethinking OPD <span class="verified">✅</span> | 1.5B–1.7B 学生，比较 Top-64、Top-16、Top-4、Top-1 和 sampled-token | sampled-token 效果和 Top-64 几乎一样，计算量最小；**Top-1 在训练后期全面崩溃**（排序一翻转，梯度就剧烈跳变） |
| Revisiting OPD（LSM）<span class="verified">✅</span> | Qwen2.5-7B-Instruct 学生，OpenThinker3-7B 教师 | sampled-token 的奖励**严重偏向惩罚**；改用教师 top-K（K=32）上**重归一化**的 reverse KL，再配合 top-p rollout，效果更好；**不做重归一化约 30 步就崩溃**；K=16 不稳定，K=32 和 K=48 接近 |
| DeepSeek-V4 <span class="verified">✅</span> | 10 个以上教师的多教师 OPD | sampled-token 的梯度方差大、训练不稳，所以改用全词表 |
| vOPD（综述转述）<span class="unverified">⚠️</span> | — | 给 sampled-token 加控制变量 baseline，准确率能追平全词表，墙钟时间最多减少 57.7% |

LSM 方法的组件消融（AIME24 avg@32）<span class="verified">✅</span>：

| 配置 | 分数 |
|---|---|
| sampled-token OPD | 20.4 |
| 只加教师 top-K | 17.7（单独使用反而下降） |
| top-K + top-p rollout | 23.6 |

## LSM：在教师 top-K 上重归一化

Revisiting OPD 提出的局部支持匹配（Local Support Matching）做法：

1. 在每个前缀上，取教师概率最高的 K 个 token，组成支持集 $S$；
2. 把教师和学生在 $S$ 上的概率分别**重归一化**；
3. 在 $S$ 上算 reverse KL。

重归一化很关键：不做的话，学生在 $S$ 之外的概率质量不受约束，训练很快崩溃。代价是：

- 教师 top-K 之外的 token 拿不到教师的直接梯度；
- 如果某个关键 token 恰好不在教师 top-K 里，就会被漏掉。比如在多教师工具调用场景中，回答教师的 top-32 虽然覆盖了 99.99% 的概率质量，却在 1,500 条审计样本里只有 0.4% 包含"开始调用工具"的那个 token，见[多教师 OPD](../frontier/multi-teacher.md)。<span class="unverified">⚠️</span>

## 交集 token 才是关键

Rethinking OPD 还发现 <span class="verified">✅</span>：

- 师生各自 top-k 的交集只占词表极小的一部分，却承载了双方 **97%–99%** 的概率质量；
- **只在交集 token 上算 loss**，就几乎能复现标准 top-k OPD 的全部增益；只用非交集 token 训练，效果明显落后。

所以 OPD 的有效梯度主要来自交集内部的概率重新分配。这也是为什么"师生 top-k 重叠率"是诊断 OPD 最重要的指标之一，见[OPD 何时有效、何时失效](mechanism.md)。

## 偏差与方差

| | 偏差 | 方差 |
|---|---|---|
| 序列级（$\gamma = 1$，reward-to-go） | 无偏 | 最坏上界 $O(T^4)$ |
| 逐 token（$\gamma = 0$） | 有偏（忽略对后续状态的影响） | 最坏上界 $O(T^2)$ |
| 逐 token + 全词表 | 同上 | 去掉了"只看一个采样 token"的方差 |

（上界来自 Revisiting OPD 的理论分析，不是实测的训练方差。）

## 怎么选

1. **先用 sampled-token（k1 + 策略梯度）跑通。** 实现最简单，教师只需返回 1 个 logprob，大多数情况下够用；
2. **不稳定、或学生起点弱时，改用 top-K 重归一化**（K≥32），同时 rollout 用 top-p（0.8–0.9）；
3. **资源充足、教师多、规模大时，考虑全词表**，工程上参考 DeepSeek-V4 的做法；
4. **不要用 Top-1**（确定性选择）。
