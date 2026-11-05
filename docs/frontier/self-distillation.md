# 自蒸馏

没有外部教师时，能不能让模型自己教自己？自蒸馏的思路是：**同一个模型在看到额外信息时当教师，看不到时当学生**。教师和学生的差别只在上下文，而不在参数。

## OPSD：用特权信息当教师

**On-Policy Self-Distillation**（2601.18734）<span class="unverified">⚠️</span>：

- **教师**：同一个模型，额外看到标准答案（"特权信息"）；
- **学生**：同一个模型，看不到答案；
- 学生在线生成回答，教师和学生对同一条轨迹给出逐 token 分布，最小化两者的散度。

相关工作：

- **Privileged Information Distillation**（ICLR 2026）：报告 π-Distill 和 OPSD 在多个 Agent benchmark 上优于"SFT 后接 RL"的标准流程；
- **OPSDL**（2604.17535）：把 OPSD 用到长上下文；
- **On-Policy Context Distillation**（2602.12275）：把上下文里的信息蒸馏进权重。

### 特权信息到底帮了多少

后续研究对此有争议 <span class="unverified">⚠️</span>：

- 有分析发现，正确的参考答案并不能在不同的教师生成方式、模型大小和训练数据上带来一致的收益；
- 另一项研究发现，Qwen3-1.7B 在 thinking 模式评测下的提升，大部分来自"不带参考答案的自蒸馏"本身。

### Privilege illusion

一个已知的失败模式：学生学到了**依赖特权信息的行为**，但推理时根本拿不到这些信息。改进方向包括：

- Dual-Anchored Policy Distillation：报告在 Qwen3 4B 和 32B 上提升约 2.7 分；
- PINT：把特权上下文内化进参数，报告 Qwen3-1.7B 在 WebShop 上的完全成功率相对提升 31.6%。

## SDPO：用文本反馈当教师

**Self-Distillation Policy Optimization**（Hübotter et al.，2601.20802）<span class="unverified">⚠️</span>：

- 针对丰富的文本反馈（运行报错、评审意见等），而不只是标准答案；
- 看到反馈的模型作为"自教师"，回头找出学生原轨迹里的错误，把它基于反馈的下一个 token 预测蒸馏回策略；
- 得到的是逐 token、逐 logit 的稠密 advantage，而 GRPO 是整条轨迹共用一个 advantage。

后续改进：

- SDPO 的 KL 型 advantage 缺少对题目难度的感知，有工作提出按通过率加权（SC-SDPO）；
- SDPO 在训练后期不稳定，有工作引入按信息量调节的步长。

## 小学生用自蒸馏可能更好

对很小的学生，自蒸馏有时比外部教师更合适 <span class="unverified">⚠️</span>：

- Unmasking OPD（2605.10889）：0.6B 学生用只含正确示范的自蒸馏，梯度对齐程度约为外部教师的 2–3 倍；到 1.7B 时，外部大教师才开始有竞争力。作者认为原因是小学生难以吸收大模型的推理风格；
- 但也有反例：自回归转扩散模型的研究（2606.06712）发现，4B 和 8B 学生用自蒸馏足够，0.6B 学生则需要 Qwen3-4B 教师才有一点提升。

## 和"修复遗忘"的关系

Thinking Machines 用**训练之前的原始模型**当教师来修复遗忘（见[案例](../cases/thinking-machines.md)），也可以看作一种广义的自蒸馏：教师和学生原本是同一个模型，只是学生后来学了新东西。

## 和记忆、经验的关系

自蒸馏提供了一个通用框架：**把"上下文里有、权重里没有"的东西内化进权重**。上下文可以是：

- 标准答案（OPSD）；
- 环境反馈（SDPO）；
- 检索到的记忆、过往经验、长文档（Context Distillation、OPSDL）。

这和智能体的"记忆机制"研究直接相关：带着记忆的模型当教师，不带记忆的模型当学生，把经验蒸馏进参数。

## 框架支持

- ms-swift 支持 OPSD 式自蒸馏：教师侧额外看到一个 `teacher_prompt` 字段，见 [ms-swift](../frameworks/ms-swift.md)；
- TRL 有 SDFT / SDPO 相关的训练器，用 vLLM 服务提供实时的教师 logprob，见 [TRL](../frameworks/trl.md)。
