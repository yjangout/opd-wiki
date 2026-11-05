# 黑盒教师

标准 OPD 需要教师在每个位置的 logprob，而且要和学生用同一个 tokenizer。闭源 API 模型通常只返回文本（或者只返回少量 top logprob、词表也不同），没法直接用。黑盒 OPD 研究的是：**只有文本的教师，怎样提供 on-policy 的反馈**。

## GAD：用判别器代替教师 logits

**Black-Box On-Policy Distillation of Large Language Models**（2511.10643，微软研究院）<span class="unverified">⚠️</span>（基于摘要和公开介绍）：

- **生成器**：学生；
- **判别器**：用 Bradley-Terry 损失训练，让教师的回答得分高于学生的回答；
- **奖励**：判别器的打分，用策略梯度（GRPO）优化学生；
- 学生在自己的生成上得到反馈，全程不需要教师的概率。

**结果**（教师 GPT-5-Chat，学生 Qwen2.5 和 Llama3 系列，用 GPT-4o 做评委）：

- 用 GAD 训练的 Qwen2.5-3B-Instruct，追平了用 SeqKD（普通 off-policy 蒸馏）训练的 Qwen2.5-7B-Instruct；
- 用 GAD 训练的 Qwen2.5-14B-Instruct，在 LMSYS-Chat 上接近 GPT-5 教师；
- 消融：换成 off-policy 的判别器，很快出现 reward hacking，生成过长、不像教师的回答。

注意：结果依赖 GPT-4o 做评委，目前没有找到独立复现。

## Rubric-based OPD：用评分细则打分

**Rubric-based On-policy Distillation**（2605.07396）<span class="unverified">⚠️</span>：从教师和学生回答的对比中提炼出评分细则（rubric），再用这些细则给学生的 rollout 打分。

## 其他黑盒信号

OPD 综述（2604.00626）列出的黑盒做法还包括：教师给出的文字反馈（Lion）、对抗判别器（GAD）、口头打分（OVD）、评分细则（ROPD）。

## API 模型在研究中的其他用法

即使不能做白盒 OPD，API 模型也很有用：

| 用法 | 说明 |
|---|---|
| off-policy 教师 | 生成 SFT 数据，作为 OPD 的冷启动 |
| 黑盒教师 | GAD、rubric 打分 |
| 评委 | 评测开放式生成 |
| 用户模拟器 | τ²-bench 这类多轮环境里扮演用户 |

!!! warning "注意服务条款"
    用 API 模型的输出训练自己的模型之前，先看服务条款。不少厂商禁止用它们的输出训练有竞争关系的模型。
