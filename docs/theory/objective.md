# 目标函数与梯度

## 从序列级 reverse KL 出发

OPD 最常见的目标是让学生的**整条回答分布**逼近教师，用 reverse KL 衡量：

$$
\mathcal{L}(\theta) = \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_T(\cdot \mid x)\big)
= \mathbb{E}_{y \sim \pi_\theta}\big[\log \pi_\theta(y \mid x) - \log \pi_T(y \mid x)\big]
$$

自回归模型的序列概率可以拆成逐 token 的乘积，所以：

$$
\mathcal{L}(\theta) = \mathbb{E}_{y \sim \pi_\theta}\Big[\sum_{t=1}^{T} k_t\Big],
\qquad
k_t = \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_T(y_t \mid x, y_{<t})
$$

$k_t$ 就是常说的 **k1**：在学生采样的 token 上，学生 logprob 减教师 logprob。

## 求梯度

记 $f_\theta(y) = \log \pi_\theta(y) - \log \pi_T(y)$。期望里的分布和被积函数都依赖 $\theta$，所以：

$$
\nabla_\theta \mathcal{L}
= \mathbb{E}_{y \sim \pi_\theta}\big[f_\theta(y)\,\nabla_\theta \log \pi_\theta(y)\big]
+ \mathbb{E}_{y \sim \pi_\theta}\big[\nabla_\theta f_\theta(y)\big]
$$

第二项等于 $\mathbb{E}[\nabla_\theta \log \pi_\theta(y)] = 0$（得分函数的期望为 0），只剩第一项：

$$
\nabla_\theta \mathcal{L}
= \mathbb{E}_{y \sim \pi_\theta}\Big[\Big(\sum_{t} \nabla_\theta \log \pi_\theta(y_t \mid y_{<t})\Big)\Big(\sum_{t'} k_{t'}\Big)\Big]
$$

再利用因果性（第 $t$ 个 token 之前的 $k_{t'}$ 与它的选择无关），可以只保留"未来"的项：

$$
\nabla_\theta \mathcal{L}
= \mathbb{E}\Big[\sum_{t} \nabla_\theta \log \pi_\theta(y_t \mid y_{<t}) \cdot \underbrace{\sum_{t' \ge t} k_{t'}}_{\text{reward-to-go}}\Big]
$$

这和策略梯度的形式完全一样：**把 $-k_t$ 当作第 $t$ 步的奖励**，回报是未来所有奖励之和（折扣因子 $\gamma = 1$）。MiniLLM 走的就是这条路线。

## Thinking Machines 的简化：折扣因子取 0

Thinking Machines 的做法是**只保留当前这一步**，即 $\gamma = 0$：

$$
\nabla_\theta \mathcal{L}_{\text{token}} \approx \mathbb{E}\Big[\sum_{t} \nabla_\theta \log \pi_\theta(y_t \mid y_{<t}) \cdot k_t\Big]
$$

对应的 advantage 是：

$$
A_t = -k_t = \log \pi_T(y_t \mid y_{<t}) - \log \pi_\theta(y_t \mid y_{<t}) \qquad (\text{stop-gradient})
$$

他们实测折扣因子大于 0 没有带来帮助。<span class="verified">✅</span> 理论上：

- $\gamma = 0$ 忽略了"改变当前 token 会影响后面的状态分布"，相对序列级目标是**有偏**的；
- 但方差小得多。Revisiting OPD 论文给出的最坏情况方差上界是：逐 token 估计为 $O(T^2)$，序列级为 $O(T^4)$；简化实验里把 $\gamma$ 从较小值调到 0.75 或 1，梯度方差会升高 2–3 个数量级。<span class="verified">✅</span>

所以几乎所有工业实现都用逐 token、$\gamma = 0$ 的版本。

## 接入 RL 框架

有了逐 token 的 advantage，就可以直接复用 PPO / GRPO 的训练循环：

$$
\mathcal{L}_{\text{OPD}} = -\frac{1}{|y|}\sum_{t} \min\Big(\rho_t A_t,\; \operatorname{clip}(\rho_t, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\, A_t\Big),
\qquad \rho_t = \frac{\pi_\theta(y_t \mid y_{<t})}{\pi_{\text{old}}(y_t \mid y_{<t})}
$$

和 GRPO 的区别只有一处：**$A_t$ 不再是整条回答共用的组内归一化分数，而是每个 token 各自的 $-k_t$。** 所以 Thinking Machines 说，在已有 RL 代码上实现 OPD 只是"把 KL 正则里的参考模型换成教师"这么小的改动。

### 训推不一致

rollout 时的 logprob 由推理引擎（vLLM、SGLang）算出，训练时由训练引擎（FSDP、Megatron）重算，两者在数值上会有差异。常见处理：

- 用训练引擎重算的 $\pi_\theta$ 计算 $k_t$（verl 就是这样做的）；
- MiMo-V2-Flash 用训推概率比 $\pi_\theta / \mu_\theta$ 做重要性加权，比值落在 $[\epsilon_{\text{low}}, \epsilon_{\text{high}}]$ 之外的 token 直接丢弃。<span class="verified">✅</span>

### 叠加结果奖励

想让学生超越教师，可以把结果奖励加进 advantage（MiMo-V2-Flash 的写法）<span class="verified">✅</span>：

$$
A_t = \operatorname{sg}\big[\log \pi_T(y_t) - \log \pi_\theta(y_t)\big] + \alpha \cdot \hat{A}_{\text{ORM}}
$$

其中 $\hat{A}_{\text{ORM}}$ 是 GRPO 式的组内归一化结果奖励，$\alpha$ 控制两者的比重。

## k1、k2、k3：同一个 KL，三种写法

单个采样 token 上的 KL 估计有三种常见写法（Schulman 的 k1/k2/k3 记号）。记 $r = \pi_T(y_t) / \pi_\theta(y_t)$，$y_t \sim \pi_\theta$：

| 估计量 | 表达式 | 作为 KL 值的估计 |
|---|---|---|
| k1 | $-\log r = \log \pi_\theta - \log \pi_T$ | 无偏，方差大，可正可负 |
| k2 | $\tfrac{1}{2}(\log r)^2$ | 有偏，方差小，恒非负 |
| k3 | $(r - 1) - \log r$ | 无偏，方差较小，恒非负 |

但**拿它们当 loss 直接反向传播时，得到的梯度并不相同**。教师项不含 $\theta$，逐个推导：

| 写法 | 对 $\theta$ 的梯度 | 在 $y_t \sim \pi_\theta$ 下的期望 |
|---|---|---|
| k1 直接反传 | $\nabla \log \pi_\theta(y_t)$ | $0$：**没有有用的信号** |
| k1 当作 advantage（策略梯度） | $k_1 \cdot \nabla \log \pi_\theta(y_t)$ | reverse KL 的梯度 ✓ |
| k2 直接反传 | $(\log \pi_\theta - \log \pi_T)\,\nabla \log \pi_\theta = k_1 \nabla \log \pi_\theta$ | reverse KL 的梯度 ✓ |
| k3 直接反传 | $(1 - r)\,\nabla \log \pi_\theta$ | $-\mathbb{E}_{\pi_T}[\nabla \log \pi_\theta]$，即 **forward KL** 的梯度 |

最后一行的推导：$\mathbb{E}_{\pi_\theta}[(1 - \pi_T/\pi_\theta)\nabla \log \pi_\theta] = \mathbb{E}_{\pi_\theta}[\nabla \log \pi_\theta] - \sum_v \pi_T(v)\nabla \log \pi_\theta(v) = 0 - \mathbb{E}_{\pi_T}[\nabla \log \pi_\theta] = \nabla_\theta \mathrm{KL}(\pi_T \,\|\, \pi_\theta)$。

两个实际后果：

- **k1 必须配策略梯度用。** verl 里 `loss_mode: k1` 搭配 `use_policy_gradient: false` 会直接报错，原因就是第一行。
- **用 k3 当 loss 时，实际优化的是 forward KL 方向**，虽然它作为数值估计的是 reverse KL。选 loss 写法时，要清楚自己真正在优化哪个方向。

## 和 KL 正则 RL 的关系

带 KL 正则的 RL 目标是：

$$
\max_\theta \; \mathbb{E}_{y \sim \pi_\theta}[R(x, y)] - \beta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})
$$

令 $R \equiv 0$、$\pi_{\text{ref}} = \pi_T$，它就退化成 OPD。所以有一种看法是：**OPD 是"只有 KL 项、参考模型换成教师"的 RL**。几个由此展开的方向：

- 加回 $R$：就是"OPD + 结果奖励"；
- G-OPD 等工作把 OPD 看作稠密的 KL 约束 RL；
- LSPD（2609.35505）把价值方法里的乐观探索和离线数据复用引入 OPD。<span class="unverified">⚠️</span>

这也解释了 verl 文档里的一条提醒：做 OPD 时要**关掉原有的 KL to reference**（`use_kl_loss: false`），否则学生会同时被拉向参考模型和教师。

## 全词表版本

上面都是"只看采样到的那个 token"的单样本估计。如果能拿到教师在每个位置的完整分布，可以直接计算每个位置的精确 KL：

$$
\mathrm{KL}_t = \sum_{v \in V} \pi_\theta(v \mid y_{<t})\big(\log \pi_\theta(v \mid y_{<t}) - \log \pi_T(v \mid y_{<t})\big)
$$

然后直接反向传播，不需要策略梯度。DeepSeek-V4 就用这种全词表做法，因为他们认为单样本估计方差大、训练不稳。<span class="verified">✅</span> 代价是要拿到并存储教师的完整 logits，工程实现见[工程优化](../frameworks/infra.md)。

介于两者之间的 top-K 做法，以及三者的取舍，见[估计粒度](estimators.md)。
