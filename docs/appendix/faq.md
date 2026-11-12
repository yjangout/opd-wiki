# FAQ

## 概念

??? question "OPD 是什么的缩写？和 off-policy 蒸馏是一回事吗？"
    OPD 在业界通常指 **On-Policy Distillation（在线策略蒸馏）**：学生自己生成，教师逐 token 打分。
    
    off-policy 蒸馏正好相反：学生在教师（或数据集）的轨迹上学，本质上就是用教师数据做 SFT。两者经常一起出现：off-policy 蒸馏常被用作 OPD 的冷启动。见[SFT 与 off-policy 蒸馏](../basics/sft.md)。

??? question "SFT、RL、OPD 是三选一吗？"
    不是。它们**先后衔接、各补一个缺口**：
    
    - SFT 稳定、便宜，但有 exposure bias；
    - RL 是 on-policy 的，但奖励稀疏、信用分配难；
    - OPD 同时补上这两个缺口，代价是依赖教师、上限是教师。
    
    标准流程是 SFT 预热 → OPD → 可选的 RL。见[为什么需要 OPD](../basics/why-opd.md)。

??? question "OPD 算强化学习吗？"
    严格来说不算。它借用了 RL 的整套机制（学生采样、advantage、策略梯度），在 verl 里也和 GRPO 共用训练器；但它的目标是**逼近教师的分布**，不是**最大化任务回报**。本质上它是模仿学习，更接近 DAgger。
    
    从数学上看，它是"奖励为 0、参考模型换成教师"的 KL 正则 RL。见[目标函数与梯度](../theory/objective.md)。

??? question "Agentic RL 是不是就是 OPD？"
    不是，两者不在同一个维度上：
    
    - **Agentic RL 是问题设定**：训练智能体在环境里多轮行动，把任务做成；
    - **OPD 是训练信号**：教师逐 token 打分。
    
    | | 结果奖励 | 教师信号（OPD） |
    |---|---|---|
    | 单轮任务 | RLVR（如 DeepSeek-R1） | 推理 OPD（如 Thinking Machines、Qwen3） |
    | 多轮智能体 | Agentic RL 的主流做法 | Agentic OPD（较新的方向） |
    
    常见的分工是：RL 负责产生能力（训出强教师），OPD 负责搬运和合并能力。详见 [Agentic RL 与 OPD](../basics/agentic-rl-vs-opd.md)。

??? question "GRPO 和 OPD 的训练数据有什么区别？"
    两者的数据集里基本都只有题目，回答是训练时由学生现场生成的。区别在于：GRPO 还要标准答案或判分规则，每条回答得到一个分数；OPD 不需要答案，但需要部署一个教师，每个 token 得到一个分数。见[三种方法的数据长什么样](../basics/data-formats.md)。

## 教师与模型规模

??? question "教师一定要用 30B 以上的模型吗？"
    不需要。教师的价值在于"有学生没有、但学生学得会的能力"，而不在于参数量：
    
    - Revisiting OPD 里，教师和学生都是 7B；
    - Rethinking OPD 里，1.5B–1.7B 学生配 4B–7B 教师；
    - 0.6B 学生从 4B 换成 14B 教师，效果反而下降；
    - 很小的学生用自蒸馏可能更好。
    
    经验上，教师选同家族、约为学生 2–5 倍大的模型，或者同尺寸的 RL 专家。见[教师怎么选](../practice/teacher.md)。

??? question "可以用闭源 API 模型当教师吗？"
    标准 OPD 不行：它需要教师在每个位置的 logprob，而且要和学生用同一个 tokenizer。
    
    但 API 模型可以：生成 off-policy 数据做冷启动；当[黑盒教师](../frontier/black-box.md)（GAD、rubric 打分）；当评委或用户模拟器。用之前先看服务条款，不少厂商禁止用输出训练竞争模型。

??? question "没有教师怎么办？"
    用[自蒸馏](../frontier/self-distillation.md)：同一个模型看到额外信息（标准答案、环境反馈、检索到的记忆）时当教师，看不到时当学生。

??? question "API 模型已经很强、也不贵了，还有必要训练小模型吗？"
    **做研究**：有必要。几乎所有 OPD 论文都在 0.6B–8B 的模型上做实验，方法验证之后再放大到大模型（Qwen3、DeepSeek-V4 都是这样用 OPD 的）。
    
    **做产品**：看场景。
    
    | 训练小模型更划算 | 直接调用 API 更划算 |
    |---|---|
    | 高频、窄任务（工具路由、抽取、格式化调用） | 调用量小，摊不下自建的固定成本 |
    | 对延迟敏感（多轮智能体的延迟会叠加） | 任务开放、范围广，需要最前沿的能力 |
    | 数据不能出域、私有化部署、离线或端侧 | 团队没有人力做训练和评测 |
    | 需要稳定可控、能用线上日志持续微调 | 需求变化快 |
    
    现实中常见的是混合架构：小模型处理大部分常规调用，难的情况交给 API 大模型。

## 训练

??? question "OPD 能超越教师吗？"
    单靠 OPD 基本不能，散度最小化的目标就是"像教师"。想超越，要叠加结果奖励（如 MiMo-V2-Flash 的 $r_t + \alpha \cdot \hat{A}_{\text{ORM}}$），或者 OPD 之后再接一段 RL。

??? question "为什么 k1 不能直接反向传播？"
    教师项不含参数，$k_1$ 对参数的梯度就是 $\nabla \log \pi_\theta(y_t)$，在学生分布下的期望为 0，没有有用的信号。所以 k1 必须配策略梯度使用：把 $-k_1$ 当作 advantage。verl 里 `k1` 配 `use_policy_gradient: false` 会直接报错。见[目标函数与梯度](../theory/objective.md)。

??? question "需要多少数据？"
    比 SFT 和 RL 少得多。Thinking Machines 的数学实验约 150 步、7.7 万个 prompt 就从 60% 提升到 70%；单个 prompt 反复训练 20 步就能接近教师水平。数据只需要题目，不需要答案。

??? question "能用 LoRA 吗？"
    能，而且 OPD 对 LoRA 比较友好：rank 32 的 LoRA 在 SFT 后比全参微调差 13%，OPD 后只差 6%。tinker-cookbook 的配方默认就用 LoRA。

??? question "训练到一半，回答长度突然暴涨，怎么办？"
    这是 OPD 最常见的失败。先检查师生的结束 token 是否一致（比如 base 学生用 `<|endoftext|>`，教师用 `<|im_end|>`），在概率层面合并等价的结束 token；再 mask special token、缩短最大长度；仍然不行就加参考模型约束、混入参考 rollout。见[踩坑大全](../practice/pitfalls.md)。

??? question "OPD 比 RL 便宜多少？"
    Qwen3 报告里，在 off-policy 蒸馏的基础上，OPD 用 1,800 GPU 小时取得了比 RL（17,920 GPU 小时）更高的分数。Thinking Machines 估计相对继续堆 SFT 数据节省 9–30 倍，在"把 RL 教师蒸馏回 base"的实验里算力效率提高 50–100 倍。

## 上手

??? question "推荐的学习路线是什么？"
    SFT → GRPO → OPD → Agentic RL（先多轮 GRPO，再 Agentic OPD）。每个阶段做到"能跑通、看得懂曲线、能定位常见问题"就往下走。见[动手教程](../tutorial/index.md)。

??? question "verl 和 ms-swift 选哪个？"
    想最快跑通一次，用 ms-swift 的 GKD（`--lmbda 1 --beta 1`）；做正式研究、多教师、多轮智能体，用 verl。想读懂实现，看 TRL 或 tinker-cookbook 的源码。见[框架](../frameworks/index.md)。

??? question "单卡能做 OPD 吗？"
    能，但要选小模型：0.6B–1.7B 学生配 1.7B–4B 教师，或者自蒸馏；也可以先离线缓存教师 logprob（Semi-OPD），训练时教师不用常驻显存。见[算力与硬件规划](../practice/compute.md)。
