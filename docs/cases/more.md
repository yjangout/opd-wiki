# 更多案例

## NVIDIA Nemotron-Cascade 2：用中间 checkpoint 当教师

**来源**：[Nemotron-Cascade 2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2)（arXiv 2603.19220），2026 年 3 月。<span class="unverified">⚠️</span>（下面的细节主要来自摘要和二手报道。）

- 30B 总参数、3B 激活的 MoE，基于 Nemotron-Nano-V3；报告称在 IMO 2025、IOI 2025 和 ICPC 世界总决赛上达到金牌水平；
- **级联 RL** 的阶段顺序（据二手报道）：指令跟随 RL → 多领域 RL → **多领域 OPD** → RLHF → 长上下文 RL → 代码 RL → 软件工程 RL；
- **核心思路**：级联 RL 的不同阶段，各自产出某个领域最强的 checkpoint（比如数学最强的可能是 SFT 之后的版本，指令跟随最强的是 IF-RL 之后的版本）。拿这些**中间 checkpoint 当领域教师**做 OPD，用来修复后续阶段造成的能力回退；
- 模型权重和 SFT、RL 数据集已在 Hugging Face 开源。

**启示**：教师不一定要是另一个更大的模型，自己训练过程中某个阶段的版本就可以。

## Hugging Face：用 TRL 蒸馏 100B+ 教师

**来源**：[Distilling 100B+ Models 40x Faster with TRL](https://huggingfacetb-trl-distillation-trainer.hf.space/)，2026 年 4 月。<span class="verified">✅</span>

- **Gemma 4 露营助手**：把 gemma-4-31B-it 蒸馏到 gemma-4-E2B-it（户外生存数据），150 步后学生就能给出详细的生火方法；学生推理速度约为教师的 2 倍，用的 GPU 只有一半。31B 教师需要约 60GB 显存；
- **Qwen 数学**：Qwen3-4B（no-think）分别从 30B-A3B 和 235B-A22B 蒸馏，AIME25 总共提升 39 分以上。235B 教师强约 10 分，学生最终水平却差不多；用 235B 教师时 GPQA-Diamond 在第 40 步后下降约 10%，30B 教师则保持稳定。

工程优化见 [TRL](../frameworks/trl.md) 和[工程优化](../frameworks/infra.md)。

## DeepSeek-R1 蒸馏版：off-policy 的对照

DeepSeek 用 R1 生成约 80 万条样本，对 Qwen2.5 和 Llama 系列的 1.5B–70B 模型做纯 SFT。<span class="unverified">⚠️</span> OPD 综述把它归为 off-policy 蒸馏（在教师样本上做交叉熵训练），而不是 OPD。

它说明了 off-policy 蒸馏本身就很强：对小模型来说，从大模型蒸馏推理模式比直接对小模型跑 RL 更有效。这也是 OPD 流程里第一步要做 SFT 的原因。

## 其他使用 OPD 的技术报告

下面这些报告也用了 OPD，但本站尚未逐一核实细节 <span class="unverified">⚠️</span>：

| 模型 | 机构 | OPD 的用法（据综述或摘要） |
|---|---|---|
| GLM-5 | 智谱 | 后训练中使用 OPD（[arXiv 2602.15763](https://arxiv.org/abs/2602.15763)） |
| KAT-Coder-V2 | 快手 | 先分领域专门训练、再统一的 OPD；报告 SWE-bench 79.6% |
| JoyInside-VLM-Agent | — | 在 RL 阶段之前做多教师 OPD；报告 BFCL v4 77.44（[arXiv 2610.12957](https://arxiv.org/abs/2610.12957)） |

## 相关但不是 OPD 的做法

一些小模型在**预训练阶段**用了 logit 蒸馏（比如 Gemma 2、Llama 3.2 1B/3B 用更大模型的 logits 作为 token 级的目标）。这类做法在固定语料上对齐教师分布，属于 off-policy 的词级蒸馏，见[SFT 与 off-policy 蒸馏](../basics/sft.md)。
