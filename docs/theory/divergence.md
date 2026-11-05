# 散度怎么选

OPD 用"学生分布和教师分布之间的距离"作为训练信号，这个距离可以用不同的散度来度量。散度的选择决定了学生会学成什么样。

## forward KL 和 reverse KL

| | forward KL：$\mathrm{KL}(\pi_T \,\|\, \pi_S)$ | reverse KL：$\mathrm{KL}(\pi_S \,\|\, \pi_T)$ |
|---|---|---|
| 期望在谁的分布下取 | 教师 | 学生 |
| 惩罚什么 | 教师有概率而学生没有的地方 | 学生有概率而教师没有的地方 |
| 行为 | **mode-covering**（mean-seeking）：学生把概率摊开，尽量覆盖教师的所有模式 | **mode-seeking**（zero-forcing）：学生集中到教师的某一个模式上 |
| 风险 | 学生容量不够时，概率落在教师各模式"之间"，产生教师自己不会说的内容 | 丢掉一部分合理的模式，多样性下降，可能过度自信 |
| 典型用法 | 传统 logit 蒸馏、GKD 的 forward 模式 | Thinking Machines 版 OPD、MiniLLM、DeepSeek-V4 |

用一个直观的例子：假设教师在某个位置有两个都合理的写法 A 和 B，各占一半概率，而学生容量有限：

```text
教师：     A ████████      B ████████
forward：  A ████  中间 ████  B ████     ← 想同时覆盖 A 和 B，概率落到两者之间
reverse：  A ███████████████             ← 只挑 A，把 A 学好，放弃 B
```

推理任务往往只需要一条正确路径，所以大多数推理场景的 OPD 用 reverse KL。

!!! note "loss 写法会悄悄改变方向"
    单样本估计时，用 k3 直接反向传播，期望梯度其实是 forward KL 的梯度，见[目标函数与梯度](objective.md)里"k1、k2、k3"一节。

## JSD 和插值

**广义 JSD**（GKD 提出）用参数 $\beta$ 在两种 KL 之间插值：

$$
\mathrm{JSD}_\beta(\pi_T \,\|\, \pi_S) = \beta\, \mathrm{KL}\big(\pi_T \,\|\, M\big) + (1 - \beta)\, \mathrm{KL}\big(\pi_S \,\|\, M\big),
\qquad M = \beta\, \pi_T + (1-\beta)\, \pi_S
$$

!!! warning "注意框架里 beta 的含义"
    不同实现对 $\beta$ 的约定不一样。TRL 和 ms-swift 的文档里：`beta=0` 是 forward KL，`beta=1` 是 reverse KL，`0.5` 是对称的 JSD。<span class="verified">✅</span> 照抄论文里的数字之前，先确认框架的约定。

GKD 还有另一个参数 $\lambda$，控制**采样分布**：以概率 $\lambda$ 用学生自己生成的回答（on-policy），否则用数据集里的回答（off-policy）。$\lambda = 1$、$\beta = 1$ 就是纯 on-policy 的 reverse KL，也就是 Thinking Machines 版 OPD。

## 自适应散度

近两年的工作不再用一个固定的散度，而是按 token 或按位置动态调整 <span class="unverified">⚠️</span>：

| 方法 | 思路 |
|---|---|
| Skew KL（DistiLLM） | 把学生和教师的分布做一点混合再算 KL，梯度更稳定 |
| AKL | 把词表分成头部（高概率）和尾部，分别用不同方向的 KL |
| ToDi | 按每个词表项的 log-ratio 给不同权重 |
| EOPD | 用教师的熵做门控：教师确定时多学，教师犹豫时少学 |
| AOPD | 每个 token 自己选择用哪类目标 |

## 怎么选

OPD 综述（2604.00626）给出的经验 <span class="verified">✅</span>：

| 任务类型 | 推荐 |
|---|---|
| 答案唯一的推理（数学、代码） | mode-seeking：reverse KL，或叠加 RL 结果奖励 |
| 开放式生成（写作、对话） | mode-covering：forward KL 或自适应散度 |
| 指令跟随 | JSD，或中等程度的混合（$\lambda \ge 0.5$） |
| 不确定任务形态时 | 自适应散度是比较稳妥的默认选择 |

综述同时提醒：在不少任务上，**采样方式（on-policy 还是 off-policy）比散度选哪种更重要**。

## top-K 截断和方向绑定

全词表的分布太大，实践中常只取 top-K 个 token 来算散度。这时要注意，**截断取的是谁的 top-K**和散度方向是绑定的：

- forward KL 的期望在教师分布下，自然取**教师**的 top-K；
- reverse KL 的期望在学生分布下，自然取**学生**的 top-K。

TRL 就是这样实现的。<span class="verified">✅</span> 而 verl 只支持教师 top-K 上的 forward KL，原因是推理服务只能返回自己的 top-K logprob，没法按学生指定的任意 token id 取教师的值。<span class="verified">✅</span>

还有一种做法是在教师 top-K 支持集上重归一化后算 reverse KL（Revisiting OPD 的 LSM 方法），见[估计粒度](estimators.md)。

## 多教师时的差别

多教师 OPD 中，forward 和 reverse 的行为差别会被放大：

- forward KL 会让学生试图同时覆盖所有教师的分布；
- reverse KL 会让学生在每个任务上贴近对应的那个教师。

DeepSeek-V4 用的是按任务加权的 reverse KL，见[多教师 OPD](../frontier/multi-teacher.md)。
