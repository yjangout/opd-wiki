# 术语表

| 术语 | 解释 |
|---|---|
| **OPD**（On-Policy Distillation） | 在线策略蒸馏：学生自己生成轨迹，教师在学生走到的每个前缀上逐 token 给反馈 |
| **off-policy 蒸馏** | 学生在教师或数据集的轨迹上学习，本质是用教师数据做 SFT |
| **on-policy / off-policy** | 训练数据来自当前策略自己（on）还是别的策略（off） |
| **SFT**（Supervised Fine-Tuning） | 监督微调：在固定的"题目 + 回答"上学下一个 token |
| **SeqKD** | 序列级知识蒸馏：教师生成完整回答，学生在上面做 SFT |
| **logit 蒸馏 / 词级蒸馏** | 在每个位置上让学生分布逼近教师分布，需要教师的概率分布 |
| **exposure bias** | 训练时前缀来自数据集，推理时来自模型自己，两者分布不一致，错误会累积，见[SFT 与 off-policy 蒸馏](../basics/sft.md) |
| **log-linear 收益** | 分数随数据量的对数线性增长：数据量每翻一倍只涨一个固定量，后期越来越贵 |
| **DAgger** | 模仿学习算法：让学生自己走，到学生自己的状态上再问专家。OPD 很像它的 LLM 版本 |
| **rollout** | 让模型（通常是学生）根据 prompt 生成一条完整回答或轨迹 |
| **prefill** | 一次性处理整段输入、算出每个位置的输出，不逐 token 生成。OPD 的教师只做 prefill |
| **reverse KL** | $\mathrm{KL}(\pi_S \,\|\, \pi_T)$，期望在学生分布下取；mode-seeking |
| **forward KL** | $\mathrm{KL}(\pi_T \,\|\, \pi_S)$，期望在教师分布下取；mode-covering |
| **JSD** | Jensen-Shannon 散度；广义 JSD 用 $\beta$ 在 forward 和 reverse 之间插值 |
| **mode-seeking / mode-covering** | 学生集中到教师的某个模式 / 学生把概率摊开覆盖教师的所有模式 |
| **k1 / k2 / k3** | 单个采样 token 上 KL 的三种估计写法；直接反传时梯度各不相同 |
| **advantage** | 某个动作（token）比平均水平好多少，策略梯度里每个 token 的权重 |
| **策略梯度**（policy gradient） | 按 advantage 加权 $\nabla \log \pi_\theta$ 的梯度估计方法 |
| **PPO** | 用价值模型估计 advantage、用裁剪的重要性比限制更新幅度的 RL 算法 |
| **GRPO** | 去掉价值模型，用组内归一化的奖励作为 advantage 的 RL 算法 |
| **RLVR** | 奖励可以被程序自动验证的 RL（数学答案、代码测试） |
| **ORM / PRM** | 结果奖励模型（只看最终结果）/ 过程奖励模型（给中间步骤打分） |
| **信用分配**（credit assignment） | 判断一条轨迹的成败该归功或归咎于哪些步骤 |
| **sampled-token** | 只用学生采样到的那个 token 的 logprob 估计 KL |
| **top-K** | 只在概率最高的 K 个 token 上计算 KL |
| **全词表**（full-vocab） | 在整个词表上计算精确的 KL |
| **重叠率**（overlap ratio） | 师生各自 top-k token 的交集比例，诊断 OPD 的关键指标 |
| **teacher_mass / student_mass** | top-K 截断后保留下来的概率质量 |
| **stop-gradient**（sg） | 计算图里截断梯度：这一项当作常数，不对它求导 |
| **训推不一致** | 推理引擎（vLLM）和训练引擎（FSDP）算出的概率有数值差异 |
| **loss mask** | 标记哪些 token 参与训练（通常只有模型自己写的部分） |
| **chat template** | 把对话消息渲染成模型输入 token 序列的模板 |
| **EOS** | 结束 token。师生不一致时会导致长度暴涨 |
| **MOPD** | 多教师 OPD（Multi-Teacher On-Policy Distillation） |
| **OPSD** | On-Policy Self-Distillation：看到特权信息的自己当教师 |
| **SDPO** | Self-Distillation Policy Optimization：以文本反馈为条件的自己当教师 |
| **privilege illusion** | 学生学到了依赖特权信息、推理时却无法复现的行为 |
| **GAD** | 生成对抗蒸馏：用判别器的打分代替黑盒教师的 logits |
| **Semi-OPD** | 只用初始学生的 rollout，教师 logprob 只算一次，反复复用 |
| **长度暴涨**（length inflation） | 训练中回答长度突然暴涨，被截断的轨迹主导训练（truncation collapse），见[长度暴涨与熵坍缩](../evaluation/concepts.md) |
| **熵坍缩**（entropy collapse） | 训练中模型熵迅速掉到接近 0，同一道题几乎总给出同一个回答，见[长度暴涨与熵坍缩](../evaluation/concepts.md) |
| **局部可教性塌缩**（local teachability collapse） | 轨迹后段的教师信号失去区分度 |
| **容量差距**（capacity gap） | 教师和学生差距太大，学生吸收不了教师的信号 |
| **exploration black hole** | 学生起点弱时，压低错误 token 却抬不起正确 token |
| **ULD / GOLD** | 支持不同 tokenizer 的 logit 蒸馏方法 |
| **LoRA** | 低秩适配：只训练少量附加参数的高效微调方法 |
