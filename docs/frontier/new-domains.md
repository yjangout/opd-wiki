# 新场景

OPD 的思路（学生自己走、教师逐步给反馈）不局限于文本推理。下面是 2026 年出现的一些新用法，多数只读到摘要 <span class="unverified">⚠️</span>。

## 跨 tokenizer

标准 OPD 要求师生用同一个词表。跨模型家族时：

- **ULD / GOLD**：Universal Logit Distillation 及其扩展 GOLD，支持不同 tokenizer 的师生对，新版本按字节偏移对齐 token。TRL 有 `GOLDTrainer`，Hugging Face 有教程 [Unlocking On-Policy Distillation for Any Model Family](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)；
- **Breaking the Tokenizer Barrier**（2606.09456）：沿用逐 token 负 reverse KL advantage 和裁剪的重要性采样目标，把 OPD 扩展到跨 tokenizer 的模型家族；
- **DSKD**：OPD 综述里列出的另一种跨词表做法。

## 多模态

- **Edit, Don't Imitate**（2610.30231）：视觉数学题里，视觉语言模型看错图会让后面的推理全部跑偏。做法是只对出错的地方做最小编辑式的纠正蒸馏，正确的 rollout 用任务奖励做 RL；
- **视觉可归因的自蒸馏**（2610.26856）：多模态 RLVR + on-policy 自蒸馏，教师能看到视觉证据并给出逐 token 纠正，报告在六个视觉推理 benchmark 上优于 GRPO 和 OPSD；
- **verl** 的多模态 OPD 示例：`run_qwen3_vl_8b_fsdp.sh`，多教师示例里也混合了文本和视觉教师。<span class="verified">✅</span>

## 量化模型

**Train Where the Quantized Model Goes**（2609.26708）：让低比特（量化后的）模型在**自己**会走到的状态上做蒸馏，修复量化带来的推理能力下降。发现：

- 0.6B 学生的 BF16 原版太弱，不足以当教师，需要 1.7B 教师（GSM8K 提升 11–14 分，而自蒸馏只有 3–6 分）；
- 1.7B 及以上的学生，大教师相比自蒸馏没有优势。

## 安全对齐

**Routing-based OPD for LLM Safety**（2607.27081）：对被越狱模板攻破的模型做重新对齐。不是去拟合具体的攻击模板，而是建模"对齐时"和"被攻破时"输出分布之间的差异。

另外，MiMo-V2-Flash 的多教师 OPD 里专门有一个安全对齐教师，见[MiMo-V2-Flash](../cases/mimo.md)。

## 扩散语言模型

**Data-Efficient Autoregressive-to-Diffusion via OPD**（2606.06712）：用 OPD 把自回归模型高效地转成扩散语言模型。4B、8B 学生用自蒸馏就够，0.6B 学生需要 Qwen3-4B 教师才有一点提升。

## 弱教强

- **Weak-to-Strong Generalization via Direct OPD**（2607.05394）；
- **Weak-to-Strong On-Policy Distillation**（2607.26246）。

研究的是：能力较弱的教师，能否通过 OPD 提升更强的学生。

## 推测解码

OPD 也被用来训练推测解码的草稿模型：让草稿模型在自己的生成上对齐目标模型的分布，提高接受率。awesome-on-policy-distillation 列表里有专门的分类。

## 小规模推理的细节

**Solving Without Stopping**（2609.37326）：把 Qwen3-8B 蒸馏到 4B、1.7B、0.6B。发现蒸馏能很好地传递解题能力，但在 thinking 模式下，学生越小，越学不会**什么时候该停止思考**。
