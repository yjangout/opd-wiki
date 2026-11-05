# 工程优化

OPD 比 SFT 多了两件事：学生在线生成，教师在线打分。工程优化基本围绕这两件事展开。

## 教师侧

### 只做 prefill

教师不需要生成，只要对"prompt + 学生回答"做一次前向，返回每个位置的 logprob。用 vLLM 时，调用 `generate` 并设置 `prompt_logprobs`，`max_tokens=1`，生成的那个 token 丢弃即可。详见[verl 源码走读](verl-walkthrough.md)。

### 只取需要的 logprob

| loss | 教师需要返回 | 每个位置的数据量 |
|---|---|---|
| sampled-token（k1 等） | 实际 token 的 logprob | 1 个值 |
| top-K | top-K 个 token 的 id 和 logprob | 2K 个值 |
| 全词表 | 完整分布 | 约 15 万个值（Qwen 词表） |

verl 的逻辑大致是 `num_logprobs = topk if use_topk else 0`：用 k1 时只取 1 个值。

### 流式打分

一条回答生成完就立刻发给教师，不等整个 batch 生成完。这样教师的耗时能藏在学生生成的耗时里。verl 就是这样做的。

### 请求批处理与二进制传输

教师作为独立服务时（Hugging Face TRL 的做法）<span class="verified">✅</span>：

- 客户端用 asyncio 队列攒请求，最多等约 5 ms 或攒到上限就发一批，p99 延迟降约 10 倍；
- top-K logprob 和 token id 打包成 NumPy 数组 → base64 → orjson，payload 缩小约 5 倍，客户端解码快约 26 倍。

## 学生侧

### 生成 batch 与训练 micro-batch 解耦

把多个梯度累积步的 prompt 攒起来，一次交给 vLLM 生成。优化器 step 之前权重不变，所以仍然是 on-policy。TRL 报告梯度累积 64 步时生成提速 41.7 倍。<span class="verified">✅</span>

### 训推不一致

rollout 由推理引擎生成，训练时由训练引擎重算 logprob，两者有数值差异。

- verl 用训练引擎重算的 logprob 计算 k1；
- MiMo-V2-Flash 用训推概率比做重要性加权，比值超出阈值的 token 直接丢弃；<span class="verified">✅</span>
- MiMo-V2-Flash 的推理和训练都用 FP8，并共用 RL 的基础设施（如 Rollout Routing Replay）。<span class="verified">✅</span>

### 提前停止

轨迹后段的教师信号本来就不可靠，可以不生成、不打分。STRIDE 在教师的累计 logprob 低于阈值时提前结束 rollout，并缓存高质量前缀，从最弱的正确轮次重新开始。<span class="unverified">⚠️</span>

## 全词表多教师：DeepSeek-V4 的做法

DeepSeek-V4 的 OPD 用了 10 个以上的教师，每个都可能是万亿参数级别，而且要算**全词表** KL。直接物化所有教师的 logits 不可行（词表超过 10 万）。他们的做法 <span class="verified">✅</span>：

1. **教师权重按需加载**：所有教师权重放在集中式分布式存储里，前向时用 ZeRO 式参数分片按需加载，减轻 I/O 和内存压力；
2. **只缓存最后一层 hidden state**：教师前向时只把最后一层的 hidden state 存进集中缓冲区；训练时再取出来，过一遍对应教师的输出头，现场重建完整 logits。重算开销可以忽略，又完全避免了物化 logits 的内存负担；
3. **按教师排序样本**：数据分发时按教师编号排序，保证每个 mini-batch 里每个教师的输出头只加载一次，显存里同一时刻最多只有一个教师的输出头；
4. **全部异步**：参数和 hidden state 的加载、卸载都在后台异步进行，不阻塞关键路径上的计算；
5. **专用 kernel**：用 TileLang 写的 kernel 精确计算师生 logits 之间的 KL，加速计算并减少动态内存分配；
6. **可抢占、可容错的 rollout 服务**：每个生成请求维护 token 级的预写日志（WAL），任务被抢占或硬件故障后可以恢复。

## 离线缓存：Semi-OPD

NVIDIA 的 Semi-OPD 走得更远：学生 rollout 只在开始时生成一次，教师 logprob 也只算一次，整个蒸馏过程反复复用。报告训练提速最多 11.4 倍，而且在多数师生组合上效果比标准 OPD 更好。<span class="unverified">⚠️</span> 见[效率优化](../frontier/efficiency.md)。

## 异步流水线

verl 的异步 on-policy 蒸馏 recipe 让 rollout、教师打分和参数更新三者重叠执行，用 top-k 稀疏 KL 提高吞吐。<span class="verified">✅</span> 代价是部分样本会有轻微的 off-policy（用的是上一两个版本的学生生成的）。
