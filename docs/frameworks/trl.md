# TRL

Hugging Face 的 TRL 有好几个和 OPD 相关的训练器。命名和位置变化较快，用之前先查当前版本的文档。

| 训练器 | 位置 | 特点 |
|---|---|---|
| `GKDTrainer` | `trl.experimental.gkd` | GKD 原始实现：`lmbda` 控制 on/off-policy 混合，`beta` 控制散度插值 |
| `DistillationTrainer` | 已从 experimental 移到顶层 `trl` | 在 GKD 的基础上加了生成缓冲、外部教师服务、二进制 logprob 传输，能蒸馏 100B+ 的教师 |
| `AsyncDistillationTrainer` | — | 异步版本，教师只需要一个 vLLM 服务地址，`teacher_server_urls` 可填多个 |
| `GOLDTrainer` | `trl.experimental` | 基于 ULD 扩展，**支持师生 tokenizer 不同**，按字节偏移对齐 token |
| SDFT / SDPO 训练器 | — | 自蒸馏；用 vLLM 服务提供实时的教师 logprob |

<span class="verified">✅</span>（来自 TRL 文档和 release note；experimental 下的 API 可能随时改变。）

## GKD 的两个参数

| 参数 | 含义 |
|---|---|
| `lmbda` | 每一步使用学生自己生成回答的概率。1.0 = 纯 on-policy |
| `beta` | 散度插值。0.0 = forward KL，1.0 = reverse KL，中间是广义 JSD |

`lmbda=1, beta=1` 对应 Thinking Machines 式的 OPD。

## 蒸馏 100B+ 教师的经验

Hugging Face 的博客 [Distilling 100B+ Models 40x Faster with TRL](https://huggingfacetb-trl-distillation-trainer.hf.space/) 总结了三项优化 <span class="verified">✅</span>：

1. **生成 batch 与训练 micro-batch 解耦**：把多个梯度累积步的 prompt 攒起来一次性交给 vLLM。累积 2、8、32、64 步时，生成分别提速 1.9、7.3、23.6、41.7 倍；
2. **外部 vLLM 教师服务 + 请求批处理**：asyncio 队列最多攒约 5 ms，p99 延迟降约 10 倍；
3. **二进制 logprob 传输**：NumPy → base64 → orjson，payload 缩小约 5 倍，延迟再降约 25%，客户端解码快约 26 倍。

**top-K 和散度方向绑定**：forward KL 取教师的 top-K，reverse KL 取学生的 top-K，同样的数据上两者结果不同。

**实验发现：**

- **Gemma 4 露营助手**：把 gemma-4-31B-it 蒸馏到 gemma-4-E2B-it（户外生存数据），150 步后就能给出详细的生火方法；学生推理速度约为教师的 2 倍，用的 GPU 只有一半；
- **Qwen 数学**：Qwen3-4B（no-think）分别从 30B-A3B 和 235B-A22B 蒸馏，AIME25 总共提升 39 分以上。但 235B 教师强约 10 分，学生最终水平却差不多；用 235B 教师时，GPQA-Diamond 在第 40 步后下降约 10%，用 30B 教师则保持稳定。

教训：**监控域外 benchmark，教师和学生的尺寸差距不要太大。**

## 跨 tokenizer：GOLD

标准 OPD 要求师生用同一个词表。GOLD 基于 Universal Logit Distillation（ULD），支持不同模型家族之间的蒸馏，新版本按字节偏移对齐 token。见 Hugging Face 的教程 [Unlocking On-Policy Distillation for Any Model Family](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation) 和[新场景](../frontier/new-domains.md)。
