# RL 与 GRPO

## RL 补的是"状态"问题

SFT 的缺口在于：训练时的前缀来自数据集，推理时来自模型自己。强化学习（RL）的做法是：

1. 让学生**自己生成**回答；
2. 用奖励函数或验证器给回答打分；
3. 提高高分回答的概率，降低低分回答的概率。

因为回答是学生自己写的，训练时看到的状态就是推理时会遇到的状态，所以 RL 是真正 **on-policy** 的。学生写出"3x = 25"之后得到低分，下次就会少走这条路。

## 策略梯度

把语言模型看作策略 $\pi_\theta$，目标是最大化期望奖励：

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot \mid x)} \big[ R(x, y) \big]
$$

它的梯度（REINFORCE）是：

$$
\nabla_\theta J = \mathbb{E}\Big[ \sum_{t} A_t \, \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t}) \Big]
$$

$A_t$ 是 advantage：这个 token 比"平均水平"好多少。它的作用和 SFT 里每个 token 的权重类似：

- SFT 相当于"所有 token 的权重都是 1"；
- 策略梯度则是"按 advantage 加权"，而且 token 来自学生自己。

## PPO 和 GRPO

**PPO** 用一个价值模型（critic）估计 advantage，再用裁剪后的重要性比来限制每一步更新的幅度。缺点是要多训练一个和策略差不多大的价值模型。

**GRPO**（Group Relative Policy Optimization，DeepSeekMath 提出，DeepSeek-R1 使用）去掉了价值模型：

1. 对每道题采样一组 $G$ 条回答 $o_1, \dots, o_G$；
2. 用奖励函数给每条回答打分 $r_1, \dots, r_G$；
3. 在组内做归一化，作为这条回答的 advantage：

$$
\hat{A}_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G)}
$$

4. **这条回答里的每个 token 都使用同一个 $\hat{A}_i$**，代入 PPO 风格的裁剪目标：

$$
J_{\text{GRPO}} = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\Big( \rho_{i,t} \hat{A}_i,\; \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\, \hat{A}_i \Big) \right] - \beta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})
$$

其中 $\rho_{i,t} = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \,/\, \pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})$ 是重要性比，$\pi_{\text{ref}}$ 是参考模型（通常是 RL 开始前的模型）。

奖励能被程序自动验证时（数学答案对不对、代码能否通过测试），这类方法常被称为 **RLVR**（RL with Verifiable Rewards）。

!!! note "GRPO 的常见改进"
    社区在 GRPO 上做了很多改进，比如：
    
    - **DAPO**：过滤全对、全错的组再补采（dynamic sampling），放宽 clip 上界（clip-higher），按 token 而不是按样本平均 loss，对超长回答做惩罚；
    - **Dr. GRPO**：去掉按标准差和长度的归一化，减少偏差。
    
    这些细节不影响理解 OPD，但做对照实验时要说明用的是哪个版本。

## 一个例子

题目"解方程 3x + 5 = 20"，每题采 $G = 4$ 条回答：

| 学生回答 | 最终答案 | 奖励 | advantage |
|---|---|---|---|
| 两边减 5，得 3x = 15，所以 x = 5 | 5 | 1 | +0.87 |
| 两边减 5，得 3x = 25，所以 x = 25/3 | 25/3 | 0 | −0.87 |
| 3x = 20 − 5 = 15，x = 5 | 5 | 1 | +0.87 |
| 移项得 3x = 20 + 5，x = 25/3 | 25/3 | 0 | −0.87 |

组内均值是 0.5，样本标准差约 0.577，所以 advantage 约为 ±0.87。

## RL 的缺口：奖励稀疏，信用分配难

**1. 一整条回答只有一个分数。**
第二条回答里"两边减 5"是对的，错的只是"3x = 25"这一步。但这条回答里的每个 token 都拿到 −0.87，正确的步骤也一起被扣分。

写了 200 个 token，只知道"错了"，却不知道错在第 5 个还是第 50 个 token。这就是**信用分配**（credit assignment）问题。

**2. 全对或全错的组没有信号。**
如果 4 条回答全错，奖励是 [0, 0, 0, 0]，组内没有差别，advantage 全为 0，这道题等于白采样。题太难时，大部分组都会是这样。

**3. 每条轨迹携带的信息很少。**
Thinking Machines 的估算是：RL 每个回合大约只提供 $O(1)$ 比特的信息，而逐 token 的监督提供 $O(N)$ 比特（$N$ 是 token 数）。<span class="verified">✅</span>

**4. 采样贵。**
每道题要采 $G$ 条（常见 8–16 条）完整回答，长推理任务里生成占了大部分训练时间。

**5. 需要奖励函数，而奖励函数会被钻空子。**
比如只检查格式，模型就会学会只写格式；比如按长度给分，模型就会把回答写长。

## RL 能做而 OPD 做不到的事

RL 优化的是**任务本身的目标**（把题做对），而不是模仿谁。所以：

- RL 可以**超越**任何教师，因为它根本没有教师；
- RL 可以探索出教师没有的新策略。

这也是为什么 OPD 之后往往还要接一段 RL：OPD 负责快速、低成本地把学生拉到教师水平，RL 负责突破教师的上限。

## 小结

| | 内容 |
|---|---|
| 补上的缺口 | 学生在自己生成的状态上学习（on-policy），不再有 exposure bias |
| 新的缺口 | 奖励稀疏、信用分配难、全对全错的组没有信号、采样贵、需要奖励函数 |
| 独有的能力 | 直接优化任务目标，可以超越教师 |
| 在流程中的角色 | 最后一步：突破教师上限、对齐具体任务 |

下一步：[为什么需要 OPD](why-opd.md)。
