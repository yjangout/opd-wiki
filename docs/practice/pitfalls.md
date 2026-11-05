# 踩坑大全

按"症状 → 原因 → 修法"整理。来源标记见[首页](../index.md)的内容约定。

## 教师选择类

| 症状 | 原因 | 修法 |
|---|---|---|
| 换了更强的教师，学生反而变差 | 思维模式不兼容：师生 top-k token 重叠率低，飞轮转不起来 | 选和学生同源的教师（比如从同一个 base 经 RL 得到的）；先做 SFT 预热 <span class="verified">✅</span> |
| 蒸馏收益很小 | 教师没有增量知识（同源放大，只是参数和数据更多） | 选经过额外 RL、真正注入了新能力的教师 <span class="verified">✅</span> |
| 大教师和中教师效果一样，域外能力下降 | 容量差距：学生吸收不了 | 选尺寸适中的教师；全程监控域外 benchmark <span class="verified">✅</span> |
| 学生已经过 RL，OPD 后退回 RL 之前的水平 | OPD 复刻的是教师的分布，不是教师的分数 | 不要用分布上弱于学生当前状态的教师 <span class="verified">✅</span> |
| 多教师时某个领域严重退化 | 教师之间的模式冲突 | 检查教师兼容性；冲突的领域分开训练或调整权重 <span class="unverified">⚠️</span> |

## 长序列 / 前缀类

| 症状 | 原因 | 修法 |
|---|---|---|
| 生成长度上到 10K–15K 后收益见顶甚至回落 | 轨迹越靠后，学生的前缀越偏离教师的典型分布，教师信号越不可靠 | 最大长度控制在约 3K–7K <span class="verified">✅</span>；或只监督前缀（变点检测截断）<span class="unverified">⚠️</span> |
| 学生陷入重复循环，教师仍然给高概率 | 教师对"局部接得上"的 token 照样给分，代理信号被钻空子 | rollout 用 top-p 采样（p=0.8–0.9；p=1.0 明显不稳定）<span class="verified">✅</span> |
| 多轮任务里，学生一步走错后整条轨迹的信号都变差 | 学生第一次做错动作时，教师的背书就消失了 | 提前停止，从出错的轮次重新开始（STRIDE）<span class="unverified">⚠️</span> |

## 长度暴涨（length inflation）

这是 OPD 最常见、也最隐蔽的失败。表现为：训练到某个时刻，回答长度突然暴涨，被截断的轨迹开始主导训练数据，同时出现大量重复，验证集分数骤降（"truncation collapse"）。

| 原因 | 修法 |
|---|---|
| **师生的结束符不一致**：比如学生是 base 模型，用 `<\|endoftext\|>` 结束，而教师经过后训练，用 `<\|im_end\|>`。逐 token 目标会把学生自己的结束 token 当成"多产出"的 token 压下去，回答于是越写越长 | 在**概率层面**把等价的结束 token 合并成一个"停止"动作。只扩大解码时的 stop set **没有用**（2609.20511）<span class="unverified">⚠️</span> |
| 学生自己采样的数据和蒸馏目标相互作用，隐式偏好长而重复的回答 | StableOPD：加参考模型的散度约束 + 混入参考 rollout，报告平均提升 7.2%（2604.08527）<span class="unverified">⚠️</span> |
| 最大长度太大 | 先缩短，确认问题消失后再逐步放开 |

## 估计器 / 数值类

| 症状 | 原因 | 修法 |
|---|---|---|
| 大部分 token 拿到负奖励，对填充词、犹豫词特别敏感 | sampled-token log-ratio 结构性地偏向惩罚 | 改用教师 top-K 支持集上重归一化的 reverse KL（K=32）<span class="verified">✅</span> |
| 用 top-K 后几十步就崩溃 | 没有在支持集内重归一化 | 一定要重归一化；K=16 不稳定，用 32 或 48 <span class="verified">✅</span> |
| 用 Top-1 后期全面崩溃 | 确定性选择一旦排序翻转，梯度就剧烈跳变 | 用 sampled-token 或 top-K（K≥32）<span class="verified">✅</span> |
| 大规模多教师训练不稳 | sampled-token 梯度方差大 | 条件允许时用全词表 KL <span class="verified">✅</span> |
| 压低了错误 token，正确 token 却抬不起来 | sampled-token 版本把概率质量按学生原有偏好重新分配 | 用 top-K 或全词表，让梯度直接朝教师分布推，见[估计粒度](../theory/estimators.md) |
| 以为在优化 reverse KL，结果学生分布越来越"摊开" | 用 k3 直接反向传播，期望梯度其实是 forward KL | 明确 loss 写法对应的方向，见[目标函数与梯度](../theory/objective.md) |

## tokenizer / 模板类

| 症状 | 原因 | 修法 |
|---|---|---|
| 语义正确的输出却被重罚 | 同一段文字切分不同，比如 `<think>` 被切成 `<`、`think`、`>`，而教师偏好 `<th`、`ink`、`>` | **mask 掉 special token** <span class="verified">✅</span>；跨 tokenizer 时用 GOLD / ULD |
| 教师打分莫名其妙地低 | 教师看到的上下文和学生生成时不一致（chat template 不同、多了 system prompt、解码后重新编码导致切分变化） | 直接用 token id 拼接 prompt 和回答；确认两边模板完全一致 |
| 第一个 token 的 logprob 对不上 | vLLM 的 prompt_logprobs 里第 0 个位置没有上文，返回 None | 注意"位置 $i$ 的 logprob 对应第 $i+1$ 个 token"，回答第 $j$ 个 token 对应下标 $P-1+j$（$P$ 为 prompt 长度）|

## 数据 / prompt 类

| 症状 | 原因 | 修法 |
|---|---|---|
| 学生熵灾难性坍塌 | 只用了教师 RL 时见过的 prompt | 保留部分对齐 prompt，再混入一定比例的 OOD prompt <span class="verified">✅</span> |
| 收敛慢、重叠率涨不动 | prompt 模板和教师 RL 时的模板不一致 | 模板对齐 + 内容对齐 <span class="verified">✅</span> |
| 多教师时训练曲线呈锯齿状 | 数据没打乱，长时间只路由到同一个教师 | `data.shuffle: true` <span class="verified">✅</span> |
| 只在成功轨迹上训练后，纠错能力没有提升 | 过滤掉了学生犯错的状态 | 过滤不可靠的位置，而不是过滤失败的轨迹 |

## 框架配置类

| 症状 | 原因 | 修法 |
|---|---|---|
| 学生一边学教师、一边被拉回原模型 | 没关掉原有的 KL to reference | verl：`actor.use_kl_loss: false`，`algorithm.use_kl_in_reward: false` <span class="verified">✅</span> |
| verl 报 `ValueError` | `loss_mode: k1` 搭配了 `use_policy_gradient: false` | k1 必须配策略梯度 <span class="verified">✅</span> |
| verl 配了多个教师，名为 `teacher_model` 的那个消失了 | 有其他命名教师时，这个条目会被静默丢弃 | 改名，比如 `teacher_model1` <span class="verified">✅</span> |
| verl 配置校验失败 | 教师池的 GPU 总数不等于各教师 `num_replicas × 每副本 GPU 数` 之和 | 严格对齐 <span class="verified">✅</span> |
| verl 报跨节点错误 | 某个副本的 TP 大小放不进一个节点（比如每节点 8 卡配 TP=5） | 调整 TP <span class="verified">✅</span> |
| ms-swift 的 OPD 效果和 SFT 一样 | `lmbda > 0` 却没开 vLLM，悄悄退回了 off-policy | 开启 `--use_vllm true` <span class="verified">✅</span> |
| ms-swift 报 top-K 超限 | `--gkd_logits_topk` 超过了教师服务的 `--max-logprobs`（vLLM 默认 20） | 起服务时调大 `--max-logprobs` <span class="verified">✅</span> |

## 上限类

| 症状 | 原因 | 修法 |
|---|---|---|
| 分数到教师水平就停了 | 散度最小化的目标就是"像教师" | 叠加结果奖励（ORM / GRPO），或者 OPD 后再接 RL |
| 多教师合并后，长程 Agent、创意写作等能力回退 | 这些能力难以通过逐 token 模仿传递 | 单独监控，必要时对这些领域单独做 RL <span class="verified">✅</span> |
| 工具调用场景下，模型什么都想调用工具 | 工具教师把学生推向"过度调用"；top-K 截断漏掉了"不调用"的分支 | 校准关键决策 token，见[多教师 OPD](../frontier/multi-teacher.md) <span class="unverified">⚠️</span> |
