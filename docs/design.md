# Slot Selector V1 — Flowchart-Aligned Step-wise Actor-Critic 需求文档

## 1. 背景

当前目标不是继续修补 v11，而是重新实现一个与原始流程图一致的 Slot Selector。

新版本不沿用 v11 中的以下设定：

* PPO
* GRPO
* teacher distillation
* teacher subset search
* value head 挂在旧 v11 结构上
* 双层 MHA 分类头
* 复杂 reward shaping
* KL-to-ref
* trajectory-level reward normalization

新版本从零开始训练，重点是：

1. 按流程图实现基础结构；
2. 保留 GRU 作为 sequential selector state；
3. 保留 Evidence accumulator；
4. 在 classifier 侧加入 permutation-invariant selected-set pooling；
5. 使用 Step-wise Actor-Critic 解决 slot 选择与停止问题；
6. 准确率优先，slot 数量其次。

---

## 2. 总体目标

模型需要完成两个任务：

### 2.1 分类任务

给定 Slot Attention 输出的 11 个 slots，模型逐步选择部分 slots，最终预测类别。

目标：

```text
val_accuracy 尽可能接近 full-slot classifier
```

---

### 2.2 slot 数量控制

模型不应盲目选择全部 slots。

目标：

```text
avg_selected 接近 GT rule slot 数量
```

或至少控制在：

```text
3~6
```

避免退化为：

```text
avg_selected ≈ 1
```

也避免：

```text
avg_selected = 11
```

---

## 3. 设计原则

### 3.1 不再使用 v11 复杂训练机制

本版本不使用 teacher distillation，也不使用 PPO/GRPO。

原因：

* teacher distillation 容易把策略压缩到过短 slot 集合；
* PPO/GRPO 使用 trajectory-level reward，后期信号弱；
* 当前任务动作空间小，适合 step-wise credit assignment；
* slot 是否有用可以通过每一步分类增益直接衡量。

---

### 3.2 保留 GRU，但不让分类完全依赖顺序

GRU 用于：

```text
建模当前选择历史
指导下一步 action
```

但分类任务本质上更接近：

```text
selected slot set → class
```

而不是：

```text
ordered slot sequence → class
```

因此 classifier 需要显式接收 permutation-invariant set representation。

---

### 3.3 Accuracy first, slot efficiency second

训练目标优先级：

```text
分类正确 > slot 数量少
```

slot cost 只作为轻量惩罚，不能主导训练。

---

## 4. 模型结构

## 4.1 输入

Slot Attention 输出：

```python
slots: [B, K, slot_dim]
```

其中：

```text
K = 11
slot_dim = 64
```

---

## 4.2 Slot Embedding

```python
slot_embeds = input_proj(slots)
```

输出：

```python
slot_embeds: [B, K, D]
```

推荐：

```text
D = 512
```

---

## 4.3 初始化

```python
h_0 = zeros([B, D])
E_0 = zeros([B, D])
selected_mask = zeros([B, K])
```

可选：

```python
h_0 = global_init(mean(slot_embeds))
```

但默认先使用 zero init，避免引入额外变量。

---

## 4.4 Action Policy Head

每一步根据当前 GRU hidden state 选择下一个 slot 或 STOP。

```python
query = action_query_proj(h_t)
slot_logits = query @ slot_embeds.T / sqrt(D)
stop_logit = stop_head(h_t)
action_logits = concat(slot_logits, stop_logit)
```

动作空间：

```text
0 ... K-1: select slot
K: STOP
```

mask：

```python
already_selected slots = -inf
invalid / blank slots = -inf
```

---

## 4.5 GRU State Update

若选择 slot `a_t`：

```python
x_t = slot_embeds[:, a_t]
h_{t+1} = GRUCell(x_t, h_t)
delta_t = h_{t+1} - h_t
```

若动作是 STOP，则不更新 GRU。

---

## 4.6 Evidence Accumulator

保留流程图中的 Evidence 累积机制。

```python
gate_t = sigmoid(evidence_gate(delta_t))
E_{t+1} = E_t + gate_t * x_t
```

Evidence 表示已收集的信息。

---

## 4.7 Selected Set Pooling

新增 permutation-invariant selected-set pooling。

目的：

让分类器看到“选了哪些 slot”，而不是只看到“以什么顺序选”。

### 默认实现：Mean Pooling

```python
selected_pool =
    mean(slot_embeds[selected_mask])
```

若当前未选 slot：

```python
selected_pool = zeros([B, D])
```

---

### 可选实现：Gated Set Pooling

```python
w_i = sigmoid(pool_gate(slot_embeds_i))
selected_pool =
    sum(w_i * slot_embeds_i * selected_mask_i)
    /
    sum(w_i * selected_mask_i)
```

默认先用 mean pooling。

---

## 4.8 Classification Head

分类器输入：

```python
h_aug = h_t + evidence_merge(E_t)

cls_input =
    concat(
        h_aug,
        E_t,
        selected_pool
    )
```

分类：

```python
logits_cls = classifier(cls_input)
```

推荐 MLP：

```python
Linear(3D, D)
GELU
LayerNorm
Dropout
Linear(D, num_classes)
```

---

## 4.9 Value Head

Actor-Critic 需要 value head。

value 输入应与 policy state 一致：

```python
value_input =
    concat(
        h_t,
        E_t,
        selected_pool
    )
```

输出：

```python
V_t = value_head(value_input)
```

注意：

Value head 只用于训练和 rollout，不参与最终分类输出。

---

# 5. 推理流程

推理时使用 greedy policy。

```python
h = h_0
E = E_0
selected_mask = zeros

for t in range(K):
    selected_pool = pool(selected_mask)

    action_logits = policy(h, slot_embeds, selected_mask)

    action = argmax(action_logits)

    if action == STOP:
        break

    update selected_mask
    update GRU
    update Evidence

    logits_cls = classifier(h, E, selected_pool)
```

最终输出：

```python
pred = argmax(logits_cls)
selected_slots = selected_mask
```

---

## 5.1 Stop 约束

为了避免过早停止，推理时设置：

```python
min_steps = 2 或 3
```

在未达到 `min_steps` 前：

```python
STOP logit = -inf
```

默认：

```text
min_steps = 2
```

如果 accuracy 下降明显，再改为：

```text
min_steps = 3
```

---

## 5.2 Max Steps

默认：

```text
max_steps = K = 11
```

---

# 6. 训练流程

训练从零开始。

不加载任何 v11 selector checkpoint。

只加载 frozen Slot Attention checkpoint。

---


## 6.2 Stage 1: Step-wise Actor-Critic

目标：

学习逐步选择 slot 和 STOP。

每个样本 rollout 一条 trajectory。

---

### State

```python
s_t =
{
    h_t,
    E_t,
    selected_mask,
    selected_pool
}
```

---

### Action

```text
select one unused slot
or STOP
```

---

### Classification at each step

每一步都计算：

```python
p_t = softmax(logits_cls_t)
ce_t = CE(logits_cls_t, y)
conf_t = p_t[y]
correct_t = argmax(p_t) == y
```

---

### Step Reward

奖励使用每步分类改进，而不是最终 sparse reward。

定义：

```python
score_t = -ce_t
```

或：

```python
score_t = log p_t[y]
```

推荐：

```python
score_t = log_softmax(logits_cls_t)[y]
```

每步奖励：

```python
r_t =
(score_t - score_{t-1})
- λ_slot
```

初始化：

```python
score_0 = log(1 / num_classes)
```

当动作是 STOP：

```python
r_stop =
+ r_correct       if final prediction correct
- r_wrong         otherwise
- λ_over          if selected too many
- λ_under         if selected too few
```

推荐初始值：

```text
λ_slot = 0.01
r_correct = +1.0
r_wrong = -1.0
λ_over = 0.10
λ_under = 0.10
```

其中：

```python
target_min_slots = 3
target_max_slots = 6
```

若：

```python
num_selected < target_min_slots
```

STOP 应受到更强惩罚。

---

## 6.3 Advantage Estimation

使用标准 Actor-Critic。

推荐：

```text
A_t = R_t - V(s_t)
```

先不要上复杂 GAE。

因为 horizon 很短：

```text
K <= 11
```

Monte Carlo return 足够稳定。

折扣：

```text
gamma = 0.9
```

---

## 6.4 Actor Loss

```python
L_actor =
- mean(
    log_prob(a_t | s_t) * stop_gradient(A_t)
)
```

---

## 6.5 Critic Loss

```python
L_value =
MSE(
    V(s_t),
    R_t
)
```

---

## 6.6 Entropy Loss

保留探索：

```python
L_entropy =
- entropy(policy)
```

总损失中为：

```python
- β_entropy * H
```

推荐：

```text
β_entropy = 0.01
```

后期可退火至：

```text
0.001
```

---

## 6.7 Classification Loss

Actor-Critic 阶段仍保留分类 CE，防止分类器退化。

```python
L_cls =
mean CE(logits_cls_t, y)
```

可以只对 trajectory 的最后一步计算，也可以对每步计算。

推荐：

```text
每步计算
```

因为这能保证每个中间状态都有分类信号。

---

## 6.8 总损失

```python
L =
L_actor
+ c_v * L_value
+ α_cls * L_cls
- β_entropy * H
```

推荐：

```text
c_v = 0.5
α_cls = 0.1
β_entropy = 0.01 → 0.001
```

---

# 7. Order Consistency Loss

为了处理 slot 顺序不应影响分类的问题，引入一致性约束。

---

## 7.1 适用阶段

优先用于：

```text
Stage 0 classifier warmup
```

可选用于：

```text
Stage 1 Actor-Critic
```

---

## 7.2 实现

对同一 selected set，生成两种不同顺序：

```python
seq_a = random_permutation(S)
seq_b = random_permutation(S)
```

分别经过：

```python
GRU + Evidence + Set Pooling + Classifier
```

得到：

```python
p_a
p_b
```

损失：

```python
L_consistency =
0.5 * KL(p_a || p_b)
+
0.5 * KL(p_b || p_a)
```

---

## 7.3 注意事项

Consistency 不应过强。

推荐：

```text
λ_consistency = 0.01 ~ 0.05
```

过强会压制 GRU 的有效顺序信息。

---

# 8. 评估指标

每个 epoch 记录：

```text
val_accuracy
test_accuracy
avg_selected
full_order_accuracy
```

---

## 8.1 Slot 数量指标

```text
avg_selected
median_selected
```

以及：

```text
selected_count_distribution
```

例如：

```text
1 slot: 5%
2 slots: 20%
3 slots: 35%
4 slots: 25%
5+ slots: 15%
```

---

## 8.2 Per-class 诊断

记录：

```text
class_i_accuracy
class_i_avg_selected
class_i_tstar
class_i_found_rate
```

---

## 8.3 RL 诊断

记录：

```text
mean_reward
mean_return
actor_loss
value_loss
entropy
avg_advantage
positive_reward_rate
stop_rate_by_step
```

如果：

```text
entropy quickly -> 0
```

说明探索过早塌缩。

如果：

```text
value_loss very high
```

说明 critic 学不好，需降低 lr 或 reward scale。

---

# 9. 验收标准

## 9.1 Stage 0 验收

Classifier warmup 后：

```text
full_order_accuracy >= 0.70
```

且随机 3~6 slots：

```text
val_accuracy >= 0.50
```

否则不进入 RL。

---

## 9.2 Stage 1 验收

Actor-Critic 后：

```text
val_accuracy >= Stage0 val_accuracy + 0.03
```

并且：

```text
avg_selected in [3, 5]
```

---

## 9.3 最终目标

理想结果：

```text
val_accuracy >= 0.65
avg_selected = 3~5
```

长期目标：

```text
val_accuracy 接近 0.80+
```

但第一版不强求。

---

# 10. 不做事项

本版本明确不做：

* 不使用 v11 checkpoint
* 不使用 teacher distillation
* 不使用 PPO
* 不使用 GRPO
* 不使用 BiGRU
* 不引入大 Transformer classifier
* 不使用 subset teacher search
* 不使用 KL-to-reference-policy

---

# 11. 实现文件建议

建议新增：

```text
src/selector_v1_ac.py
```

包含：

```text
SlotSelectorAC
ACConfig
rollout_actor_critic
compute_returns
evaluate_greedy
```

新增训练脚本：

```text
scripts/train_selector_v1_ac.py
```

输出：

```text
out/selector_v1_ac_*
```

---

# 12. 最小实验配置

```bash
python scripts/train_selector_v1_ac.py \
  --env_path ... \
  --sa_checkpoint ... \
  --out_subpath out/selector_v1_ac \
  --stage0_epochs 10 \
  --stage1_epochs 100 \
  --target_min_slots 3 \
  --target_max_slots 5 \
  --lambda_slot 0.01 \
  --entropy_coef 0.01 \
  --class_coef 0.1 \
  --value_coef 0.5 \
  --consistency_coef 0.05
```

---

# 13. 总结

本版本的核心思想：

```text
用 GRU 做 sequential decision
用 Evidence 记录已收集信息
用 Set Pooling 消除分类侧顺序敏感
用 Step-wise Actor-Critic 提供逐步选择信号
```

不要再依赖最终 trajectory reward。

不要再用 teacher distillation 压缩 slot 数量。

先让模型从零学出一个稳定的 3~6 slot 策略，再考虑进一步提高准确率。
