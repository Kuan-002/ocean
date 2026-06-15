# SlotSelectorAC Tutorial

This note explains the Actor-Critic Slot Selector data flow and how the files connect.
It assumes the recurrent update itself is already familiar, and focuses on the AC part:
policy, value estimation, reward, losses, and evaluation.

## 1. Reading Order

Start from the experiment entry point, then follow the data into the model:

1. `oceanslurm.sh`
2. `scripts/train_selector_v1_ac.py`
3. `src/selector_v1_ac.py`
4. `src/config.py`
5. `scripts/envs/.envA_ch7_attn_select_v9c`
6. `src/utils.py`, `src/slot_autoencoder.py`, `src/slot_attention.py`
7. `src/clevr_hans_dataset.py`
8. `scripts/visualize_selector_v1_ac_paths.py`

The shortest conceptual path is:

```text
image
  -> frozen Slot Attention
  -> slots [B, K, slot_dim]
  -> SlotSelectorAC
  -> selected slots + class logits
  -> reward + actor/critic/classification losses
```

## 2. File Connections

### `oceanslurm.sh`

This file defines the experiment parameters and submits training:

```text
ENV_FILE: dataset/config .env file
SA_CKPT: frozen Slot Attention checkpoint
OUT_DIR: experiment output directory
AC_* variables: SlotSelectorAC hyperparameters
```

The important part is the final call:

```bash
python scripts/train_selector_v1_ac.py \
    --env_path "${ENV_FILE}" \
    --sa_checkpoint "${SA_CKPT}" \
    --out_subpath "${OUT_DIR}" \
    ...
```

So `oceanslurm.sh` answers:

```text
What dataset?
Which frozen Slot Attention checkpoint?
Which AC hyperparameters?
Where are outputs saved?
```

### `scripts/train_selector_v1_ac.py`

This is the training driver. Its `main()` performs:

```python
config = Config(args.env_path, args.out_subpath)
loaders = setup_dataloaders(config)
sa, sa_best_loss = reconstruct_autoencoder(args.sa_checkpoint, config)
ac_cfg = ACConfig(...)
model = SlotSelectorAC(ac_cfg)
optimizer = torch.optim.AdamW(...)
```

It then trains with:

```python
out = rollout_actor_critic(model, slots, labels, slot_pos=slot_pos)
out.loss.backward()
optimizer.step()
```

And evaluates with:

```python
val = evaluate_greedy(...)
test = evaluate_greedy(...)
```

### Slot Extraction

`batch_slots()` freezes image processing before AC:

```python
slots, attn = sa.forward_slots_only(x, slot_init_noise=slot_noise)
```

The AC model does not see raw images directly. It sees:

```text
slots: [B, K, slot_dim]
```

For the CH7 experiments here, this is typically:

```text
K = 11
slot_dim = 64
```

If `pos_dim > 0`, attention centroids are appended as slot positions. In the no-pos experiments,
`pos_dim = 0`, so only the slot vectors are used.

## 3. Model Components

`src/selector_v1_ac.py` defines the model.

### `ACConfig`

`ACConfig` stores model and training hyperparameters:

```text
slot_dim, pos_dim, embed_dim
num_slots, num_classes
max_steps, min_steps
lambda_slot
target_min_slots, target_max_slots
r_correct, r_wrong
lambda_over, lambda_under
value_coef, class_coef, full_order_class_coef, entropy_coef
early_exit_conf
ordered_classifier
cross_attention_classifier
```

### `SlotSelectorAC.__init__`

The model contains:

```text
input_proj: slot -> internal embedding
action_query_proj: current state h -> query for scoring all slots
stop_head: current state h -> stop logit
gru: recurrent state update after selecting a slot
evidence_gate: controls how much selected slot information enters evidence
classifier: predicts class logits
value_head: critic, predicts future reward return
```

The AC-specific pieces are:

```text
actor: policy_logits()
critic: value()
reward/loss: rollout_actor_critic()
```

## 4. Actor: How the Policy Chooses Slots

The actor is `policy_logits()`.

```python
query = self.action_query_proj(h)
slot_logits = torch.einsum("bd,bkd->bk", query, slot_embeds) / sqrt(embed_dim)
```

Shapes:

```text
h:          [B, D]
query:      [B, D]
slot_embeds:[B, K, D]
slot_logits:[B, K]
```

This means every step scores all `K` slots.

Already selected slots are then masked:

```python
slot_logits = slot_logits.masked_fill(selected_mask, -inf)
```

The stop action is added:

```python
stop_logit = self.stop_head(h)
logits = torch.cat([slot_logits, stop_logit], dim=-1)
```

So the action space is:

```text
0..K-1: select a slot
K: stop
```

Before `min_steps`, stop is disabled:

```python
if step < min_steps:
    logits[:, stop_idx] = -inf
```

### Important Distinction

The policy does not explicitly enumerate all subsets. It sees all slot candidates at every step,
and gradually constructs a subset through sequential actions:

```text
step 0: choose one slot from all slots
step 1: choose one slot from remaining slots
step 2: choose one slot from remaining slots or stop
...
```

## 5. Recurrent State Update

After an action is sampled or chosen greedily, `update_with_action()` updates the sequential state:

```python
x_t = slot_embeds[batch, action]
h_new = self.gru(x_t, h)
```

Then it updates evidence:

```python
delta = h_new - h
gate = sigmoid(evidence_gate(delta))
evidence = evidence + gate * x_t
```

And records the selected slot:

```python
selected_mask[action] = True
```

Interpretation:

```text
h: ordered history of selected slots
evidence: accumulated selected-slot evidence
selected_mask: unordered record of which slots have been selected
```

## 6. Classifier Variants

`classify()` predicts class logits from the current selection state.

Common input:

```python
h_aug = h + evidence_merge(evidence)
```

Default classifier:

```text
classifier([h_aug, evidence, selected_pool])
```

where:

```python
selected_pool = mean(selected slot embeddings)
```

So the default classifier uses both:

```text
ordered information: h, evidence
set information: selected_pool
```

### Ordered Classifier Ablation

With `ordered_classifier=True`:

```text
selected_context = h
classifier([h_aug, evidence, h])
```

This removes selected-set mean pooling from the classifier.

### Cross-Attention Classifier Ablation

With `cross_attention_classifier=True`:

```text
selected_context = cross_attention(query=h_aug, keys/values=slot_embeds, mask=selected_mask)
classifier([h_aug, evidence, selected_context])
```

This matches the SVG-level idea more closely:

```text
policy attends all slots
classifier attends selected slots only
```

The implementation masks unselected slots before attention.

## 7. Critic: Why `value_head` Can Estimate Value

The critic is `value()`:

```python
value_t = model.value(h, evidence, selected_pool, full_pool)
```

Its input is the current state:

```text
h: current ordered selection history
evidence: accumulated selected evidence
selected_pool: mean of selected slots
full_pool: mean of all slots
```

Its output is:

```text
value_t ~= expected future discounted reward from this state
```

It can estimate value because training gives it a supervised target:

```python
returns = compute_returns(rewards, gamma)
value_loss = MSE(value_t, return_t)
```

So `value_head` is trained to fit:

```text
current state -> future discounted return
```

It is not manually programmed to know which states are good. It learns from rollout outcomes.

## 8. Training Rollout

The main AC training loop is `rollout_actor_critic()`.

At each step:

```python
pool = selected_pool(slot_embeds, selected_mask)
value_t = model.value(h, evidence, pool, full_pool)
action_logits = model.policy_logits(h, slot_embeds, selected_mask, step=step)
dist = Categorical(logits=action_logits)
action = dist.sample()
log_prob = dist.log_prob(action)
entropy = dist.entropy()
h, evidence, selected_mask = update_with_action(...)
logits_cls = model.classify(...)
reward = ...
```

Training samples actions. Evaluation uses greedy argmax.

```text
training: action = dist.sample()
eval:     action = argmax(action_logits)
```

## 9. Reward Definition

The reward has two modes:

1. Selecting another slot
2. Stopping, or being forced to terminate at horizon

### Selection Reward

When the policy selects another slot:

```python
select_reward = (log_p_true - prev_score) - lambda_slot
```

Meaning:

```text
reward = improvement in true-class log probability - slot cost
```

If the selected slot makes the classifier more confident in the correct label, the reward is positive.
If confidence decreases or improves too little, the reward is negative.

### Stop Reward

When the policy stops:

```python
stop_reward = +r_correct if pred == label else -r_wrong
```

Then slot-count penalties may be applied:

```python
stop_reward -= lambda_over * too_many
stop_reward -= lambda_under * too_few
```

In the current no-classmin/no-targetmax experiment, `target_max_slots = 0`, so the over-selection
penalty is disabled.

## 10. Reward Numerical Examples

Assume:

```text
lambda_slot = 0.01
gamma = 0.9
r_correct = 1.0
r_wrong = 1.0
lambda_under = 0.0
lambda_over = 0.0
```

### Example A: Useful Slot

Before selecting a slot:

```text
prev_score = log P(true class) = -1.20
```

After selecting a slot:

```text
log_p_true = -0.80
```

Selection reward:

```text
select_reward = (-0.80 - -1.20) - 0.01
              = 0.40 - 0.01
              = 0.39
```

This action is rewarded because it increased confidence in the true class.

### Example B: Weak Slot

Before:

```text
prev_score = -1.20
```

After:

```text
log_p_true = -1.15
```

Reward:

```text
select_reward = (-1.15 - -1.20) - 0.01
              = 0.05 - 0.01
              = 0.04
```

Still positive, but small.

### Example C: Harmful Slot

Before:

```text
prev_score = -1.20
```

After:

```text
log_p_true = -1.40
```

Reward:

```text
select_reward = (-1.40 - -1.20) - 0.01
              = -0.20 - 0.01
              = -0.21
```

This action is penalized because it reduced confidence in the true class.

### Example D: Correct Stop

If the classifier predicts the true label at stop:

```text
stop_reward = +1.0
```

If no slot-count penalty is active, final reward is:

```text
+1.0
```

### Example E: Wrong Stop

If the classifier predicts the wrong label:

```text
stop_reward = -1.0
```

### Example F: Too Few Slots Penalty

Assume:

```text
target_min_slots = 3
lambda_under = 0.3
selected_count = 2
prediction is correct
```

Then:

```text
too_few = 1
stop_reward = 1.0 - 0.3 * 1
            = 0.7
```

Correct classification is still rewarded, but early stopping is penalized.

### Example G: Discounted Return

Suppose a trajectory has rewards:

```text
r0 = 0.39
r1 = -0.21
r2 = 1.00
gamma = 0.9
```

Return at step 0:

```text
G0 = r0 + gamma*r1 + gamma^2*r2
   = 0.39 + 0.9*(-0.21) + 0.81*1.00
   = 0.39 - 0.189 + 0.81
   = 1.011
```

Return at step 1:

```text
G1 = r1 + gamma*r2
   = -0.21 + 0.9
   = 0.69
```

Return at step 2:

```text
G2 = r2
   = 1.00
```

These `G_t` values become the critic targets.

## 11. Advantage and Actor-Critic Loss

After rollout:

```python
returns = compute_returns(rewards, gamma)
advantage = return_t - value_t
```

The actor loss is:

```python
actor_loss = -(log_prob_t * advantage.detach() * weight).sum() / denom
```

Interpretation:

```text
advantage > 0:
  action was better than critic expected
  increase its probability

advantage < 0:
  action was worse than critic expected
  decrease its probability
```

The critic loss is:

```python
value_loss = MSE(value_t, return_t)
```

So the critic learns a baseline, and the actor learns which actions produce better-than-expected returns.

## 12. Full Loss

The total training loss is:

```python
loss =
    actor_loss
    + value_coef * value_loss
    + class_coef * class_loss
    + full_order_class_coef * full_order_class_loss
    - entropy_coef * entropy
```

Each term has a role:

```text
actor_loss:
  trains the selection policy

value_loss:
  trains the critic baseline

class_loss:
  trains classifier at selected intermediate states

full_order_class_loss:
  auxiliary classifier supervision using all slots in fixed order

- entropy_coef * entropy:
  encourages exploration by preventing the policy from becoming deterministic too early
```

## 13. Evaluation

Training uses sampling:

```python
action = dist.sample()
```

Evaluation uses greedy selection in `forward_greedy()`:

```python
action = action_logits.argmax(dim=-1)
```

Evaluation may stop in two ways:

```text
1. policy chooses stop
2. confidence >= early_exit_conf and selected_count >= min_steps
```

The final output metrics include:

```text
accuracy
avg_selected
median_selected
per-class accuracy
selected-count histogram
full_order_accuracy
```

## 14. One-Sentence Summary

The RNN tells the model how to update state after each selected slot; Actor-Critic tells it how to
train the slot-selection policy by comparing actual future reward against the critic's expected value.

  第一，训练时不是“得分最高的 slot”，而是按 policy 分布采样：

  dist = Categorical(logits=action_logits)
  action = dist.sample()

  验证/测试时才是最高分：

  action = action_logits.argmax(dim=-1)

  所以训练时流程是：

  policy 给所有 slot + stop 打分
  -> softmax 成 action 分布
  -> sample 一个 action
  -> 更新 h/evidence/selected_mask
  -> classifier 输出类别概率
  -> 取 GT label 的 log probability
  -> 计算 reward

  第二，value_head 不是“递减累加奖励”。
  递减累加奖励的是 compute_returns()：

  returns = compute_returns(rewards, gamma)

  它做的是：

  G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

  value_head 做的是预测：

  value_t = model.value(h, evidence, pool, full_pool)

  它预测：

  从当前状态开始，未来大概能拿到多少 discounted return

  然后用真实算出来的 return_t 监督它：

  value_loss = MSE(value_t, return_t)

  所以准确说是：

  reward:
    根据这一步 action 后 GT 概率变化、最终对错、slot 数量算出来

  return:
    把一条 trajectory 里的 rewards 做 gamma 折扣累加

  value_head:
    输入当前状态，预测 return

  value_loss:
    让 value_head 的预测接近真实 return

  一个小例子：

  step 0 reward = 0.39
  step 1 reward = -0.21
  step 2 reward = 1.00
  gamma = 0.9

  真实 return 是：

  G0 = 0.39 + 0.9*(-0.21) + 0.9^2*1.00 = 1.011
  G1 = -0.21 + 0.9*1.00 = 0.69
  G2 = 1.00

  value_head 在每一步分别预测：

  V0 ≈ G0
  V1 ≈ G1
  V2 ≈ G2

  如果它预测：

  V0 = 0.7

  那 advantage 是：

  A0 = G0 - V0 = 1.011 - 0.7 = 0.311

  说明 step 0 的 action 比 critic 预期更好，actor 会增加这个 action 的概率。

  所以一句话：

  > 模型选 action 后可以用 GT label 的概率变化计算即时 reward；真实 reward 序列被折扣累加成 return；value_head 不是累加 reward，而是学习预测这个 return，用作 actor 更新的 baseline。