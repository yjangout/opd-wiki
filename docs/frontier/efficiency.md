# 效率优化

OPD 的主要开销是学生在线生成和教师在线打分。这一页整理怎样省下这两部分。

## Semi-OPD：真的需要每一步都 on-policy 吗

**When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better**（2610.11291，NVIDIA）<span class="unverified">⚠️</span>

**做法**：

1. 用**初始**学生生成一次 rollout；
2. 教师对这些 rollout 算一次 token logprob；
3. 整个蒸馏过程反复复用这批 rollout 和 logprob，不再在线生成。

**结果**：

- 在 1.5B 到 235B 的 17 组师生组合里，Semi-OPD 有 14 组优于标准 OPD，准确率最多高 13.6%，训练最多快 11.4 倍；
- 只有当师生**输出 token 的重叠率很高**时，标准 OPD 才更好；
- 训练通常更稳定，因为它在较短的上下文上蒸馏，同时又覆盖了完整的轨迹。

**意义**：

- 它介于 off-policy 蒸馏（用教师的轨迹）和 OPD（每步用当前学生的轨迹）之间：用的是**学生**的轨迹，所以仍然覆盖了学生会走到的状态，但不随训练更新；
- 对算力有限的场景很友好：教师 logprob 只算一次，可以离线完成，训练时教师不用常驻显存；
- 它也提醒我们：on-policy 的价值依赖于师生的兼容性，见[OPD 何时有效、何时失效](../theory/mechanism.md)。

另一项系统研究（On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics，NeurIPS 2026）也在比较这两种设置。<span class="unverified">⚠️</span>

## 只在有用的地方生成和打分

- **提前停止**（STRIDE）：教师累计 logprob 过低时停止 rollout，从最弱的正确轮次重新开始。报告在 τ²-bench retail 上提速 3.73 倍，AIME 2025 上 5.10 倍，同时追平或超过完整 OPD；<span class="unverified">⚠️</span>
- **前缀截断**（FOPD 等）：只蒸馏轨迹前段；
- **变点截断**（Prefix Teach, Suffix Fade）：检测教师信号失效的位置，只监督之前的部分。

## 离线缓存

- **Lightning-OPD** 等：离线缓存教师的 logprob，训练时直接读取；
- OPD 综述建议：预算在约 500 GPU 小时以内时，优先考虑前缀截断和离线缓存。<span class="verified">✅</span>

## 降低估计方差

- **vOPD**：给 sampled-token 估计加一个控制变量 baseline，据综述转述，准确率追平全词表，墙钟时间最多减少 57.7%。<span class="unverified">⚠️</span>

## 系统层面

- **生成 batch 与训练 micro-batch 解耦**（TRL）：梯度累积 64 步时生成提速 41.7 倍；<span class="verified">✅</span>
- **异步流水线**（verl async recipe、NPD）：rollout、教师打分、参数更新重叠执行；
- **教师服务优化**：请求批处理、二进制传输、流式打分。

详见[工程优化](../frameworks/infra.md)。

## 少采几条

OPD 不需要组内比较，每题 1–4 条就有信号；Thinking Machines 的单 prompt 实验甚至显示，一道题反复训练 20 步就能接近教师水平。<span class="verified">✅</span>
