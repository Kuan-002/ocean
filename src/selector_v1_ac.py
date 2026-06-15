from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ACConfig:
    slot_dim: int = 64
    pos_dim: int = 0
    embed_dim: int = 512
    num_slots: int = 11
    num_classes: int = 7
    max_steps: int = 11
    min_steps: int = 3
    class_min_slots: Optional[tuple[int, ...]] = None
    gamma: float = 0.9
    lambda_slot: float = 0.01
    target_min_slots: int = 3
    target_max_slots: int = 5
    r_correct: float = 1.0
    r_wrong: float = 1.0
    lambda_over: float = 0.10
    lambda_under: float = 0.30
    value_coef: float = 0.5
    class_coef: float = 0.3
    full_order_class_coef: float = 0.1
    entropy_coef: float = 0.01
    dropout: float = 0.1
    global_init: bool = False
    early_exit_conf: float = 0.8
    ordered_classifier: bool = False
    cross_attention_classifier: bool = False


@dataclass
class RolloutOutput:
    logits: torch.Tensor
    selected_mask: torch.Tensor
    selected_counts: torch.Tensor
    loss: Optional[torch.Tensor]
    actor_loss: torch.Tensor
    value_loss: torch.Tensor
    class_loss: torch.Tensor
    full_order_class_loss: torch.Tensor
    entropy: torch.Tensor
    mean_reward: torch.Tensor
    mean_return: torch.Tensor
    mean_advantage: torch.Tensor
    positive_reward_rate: torch.Tensor
    stop_rate_by_step: torch.Tensor


class SlotSelectorAC(nn.Module):
    """Flowchart-aligned step-wise actor-critic slot selector.

    The policy is sequential. By default the classifier also receives a
    selected-set mean pool; ordered_classifier replaces that pool with the
    sequential hidden state as an ablation, and cross_attention_classifier
    replaces it with a single-query attention summary over selected slots.
    """

    def __init__(self, cfg: ACConfig) -> None:
        super().__init__()
        if cfg.class_min_slots is not None and len(cfg.class_min_slots) != cfg.num_classes:
            raise ValueError(
                f"class_min_slots must have {cfg.num_classes} entries, "
                f"got {len(cfg.class_min_slots)}"
            )
        if cfg.ordered_classifier and cfg.cross_attention_classifier:
            raise ValueError("ordered_classifier and cross_attention_classifier are mutually exclusive")
        self.cfg = cfg
        d = cfg.embed_dim
        input_dim = cfg.slot_dim + cfg.pos_dim

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        self.global_init = nn.Linear(d, d) if cfg.global_init else None
        self.action_query_proj = nn.Linear(d, d)
        self.stop_head = nn.Linear(d, 1)
        self.gru = nn.GRUCell(d, d)
        self.evidence_gate = nn.Linear(d, d)
        self.evidence_merge = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        if cfg.cross_attention_classifier:
            self.classifier_query_proj = nn.Linear(d, d)
            self.classifier_key_proj = nn.Linear(d, d)
            self.classifier_value_proj = nn.Linear(d, d)
        else:
            self.classifier_query_proj = None
            self.classifier_key_proj = None
            self.classifier_value_proj = None

        cls_state_dim = 3 * d
        critic_state_dim = 4 * d
        self.classifier = nn.Sequential(
            nn.LayerNorm(cls_state_dim),
            nn.Linear(cls_state_dim, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(d),
            nn.Linear(d, cfg.num_classes),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(critic_state_dim),
            nn.Linear(critic_state_dim, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, 1),
        )

    @property
    def stop_idx(self) -> int:
        return self.cfg.num_slots

    def embed_slots(
        self,
        slots: torch.Tensor,
        slot_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.cfg.pos_dim > 0:
            if slot_pos is None:
                raise ValueError("slot_pos is required when ACConfig.pos_dim > 0")
            if slot_pos.shape[:2] != slots.shape[:2] or slot_pos.shape[-1] != self.cfg.pos_dim:
                raise ValueError(
                    "slot_pos expected shape "
                    f"{(*slots.shape[:2], self.cfg.pos_dim)}, got {tuple(slot_pos.shape)}"
                )
            slots = torch.cat([slots, slot_pos.to(device=slots.device, dtype=slots.dtype)], dim=-1)
        return self.input_proj(slots)

    def initial_state(self, slot_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, d = slot_embeds.shape
        if self.global_init is None:
            h = slot_embeds.new_zeros(b, d)
        else:
            h = torch.tanh(self.global_init(slot_embeds.mean(dim=1)))
        evidence = slot_embeds.new_zeros(b, d)
        return h, evidence

    def min_steps_for_classes(self, class_ids: torch.Tensor) -> torch.Tensor:
        if self.cfg.class_min_slots is None:
            return torch.full_like(class_ids, self.cfg.min_steps)
        values = torch.tensor(self.cfg.class_min_slots, device=class_ids.device, dtype=class_ids.dtype)
        return values[class_ids]

    def min_steps_for_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self.min_steps_for_classes(logits.argmax(dim=-1))

    def selected_pool(
        self,
        slot_embeds: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = selected_mask.to(slot_embeds.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (slot_embeds * weights).sum(dim=1) / denom

    def selected_cross_attention(
        self,
        h_aug: torch.Tensor,
        slot_embeds: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.classifier_query_proj is None
            or self.classifier_key_proj is None
            or self.classifier_value_proj is None
        ):
            raise RuntimeError("cross-attention classifier layers are not initialized")
        query = self.classifier_query_proj(h_aug)
        keys = self.classifier_key_proj(slot_embeds)
        values = self.classifier_value_proj(slot_embeds)
        scores = torch.einsum("bd,bkd->bk", query, keys) / math.sqrt(self.cfg.embed_dim)
        scores = scores.masked_fill(~selected_mask.bool(), torch.finfo(scores.dtype).min)
        has_selected = selected_mask.any(dim=1)
        safe_scores = torch.where(has_selected.unsqueeze(-1), scores, torch.zeros_like(scores))
        attn = safe_scores.softmax(dim=-1)
        context = torch.einsum("bk,bkd->bd", attn, values)
        return context * has_selected.unsqueeze(-1).to(context.dtype)

    def policy_logits(
        self,
        h: torch.Tensor,
        slot_embeds: torch.Tensor,
        selected_mask: torch.Tensor,
        *,
        step: int,
        valid_slot_mask: Optional[torch.Tensor] = None,
        min_steps: Optional[int | torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.action_query_proj(h)
        slot_logits = torch.einsum("bd,bkd->bk", query, slot_embeds) / math.sqrt(self.cfg.embed_dim)
        slot_logits = slot_logits.masked_fill(selected_mask, torch.finfo(slot_logits.dtype).min)
        if valid_slot_mask is not None:
            slot_logits = slot_logits.masked_fill(~valid_slot_mask.bool(), torch.finfo(slot_logits.dtype).min)

        stop_logit = self.stop_head(h)
        logits = torch.cat([slot_logits, stop_logit], dim=-1)
        min_steps = self.cfg.min_steps if min_steps is None else min_steps
        if isinstance(min_steps, torch.Tensor):
            logits[:, self.stop_idx] = logits[:, self.stop_idx].masked_fill(
                step < min_steps,
                torch.finfo(logits.dtype).min,
            )
        elif step < min_steps:
            logits[:, self.stop_idx] = torch.finfo(logits.dtype).min
        return logits

    def classify(
        self,
        h: torch.Tensor,
        evidence: torch.Tensor,
        selected_pool: torch.Tensor,
        slot_embeds: Optional[torch.Tensor] = None,
        selected_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_aug = h + self.evidence_merge(evidence)
        if self.cfg.cross_attention_classifier:
            if slot_embeds is None or selected_mask is None:
                raise ValueError("slot_embeds and selected_mask are required for cross_attention_classifier")
            selected_context = self.selected_cross_attention(h_aug, slot_embeds, selected_mask)
        elif self.cfg.ordered_classifier:
            selected_context = h
        else:
            selected_context = selected_pool
        return self.classifier(torch.cat([h_aug, evidence, selected_context], dim=-1))

    def value(
        self,
        h: torch.Tensor,
        evidence: torch.Tensor,
        selected_pool: torch.Tensor,
        full_pool: torch.Tensor,
    ) -> torch.Tensor:
        return self.value_head(torch.cat([h, evidence, selected_pool, full_pool], dim=-1)).squeeze(-1)

    def update_with_action(
        self,
        h: torch.Tensor,
        evidence: torch.Tensor,
        selected_mask: torch.Tensor,
        slot_embeds: torch.Tensor,
        action: torch.Tensor,
        active_select: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = slot_embeds.size(0)
        safe_action = action.clamp(max=self.cfg.num_slots - 1)
        x_t = slot_embeds[torch.arange(b, device=slot_embeds.device), safe_action]
        h_new = self.gru(x_t, h)
        delta = h_new - h
        gate = torch.sigmoid(self.evidence_gate(delta))
        update = active_select.unsqueeze(-1).to(slot_embeds.dtype)
        h = torch.where(update.bool(), h_new, h)
        evidence = evidence + update * gate * x_t
        selected_mask = selected_mask.clone()
        if active_select.any():
            selected_mask[active_select, action[active_select]] = True
        return h, evidence, selected_mask

    def full_order_logits(self, slot_embeds: torch.Tensor) -> torch.Tensor:
        b, k, _ = slot_embeds.shape
        h, evidence = self.initial_state(slot_embeds)
        selected_mask = torch.zeros(b, k, dtype=torch.bool, device=slot_embeds.device)
        active = torch.ones(b, dtype=torch.bool, device=slot_embeds.device)
        for idx in range(k):
            action = torch.full((b,), idx, dtype=torch.long, device=slot_embeds.device)
            h, evidence, selected_mask = self.update_with_action(
                h, evidence, selected_mask, slot_embeds, action, active
            )
        return self.classify(
            h,
            evidence,
            self.selected_pool(slot_embeds, selected_mask),
            slot_embeds,
            selected_mask,
        )

    def forward_greedy(
        self,
        slots: torch.Tensor,
        slot_pos: Optional[torch.Tensor] = None,
        *,
        valid_slot_mask: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        early_exit_conf: Optional[float] = None,
    ) -> RolloutOutput:
        self.eval()
        cfg = self.cfg
        max_steps = min(max_steps or cfg.max_steps, cfg.num_slots)
        early_exit_conf = cfg.early_exit_conf if early_exit_conf is None else early_exit_conf
        slot_embeds = self.embed_slots(slots, slot_pos)
        b, k, _ = slot_embeds.shape
        h, evidence = self.initial_state(slot_embeds)
        selected_mask = torch.zeros(b, k, dtype=torch.bool, device=slots.device)
        active = torch.ones(b, dtype=torch.bool, device=slots.device)
        final_logits = self.classify(
            h,
            evidence,
            self.selected_pool(slot_embeds, selected_mask),
            slot_embeds,
            selected_mask,
        )
        stop_counts = torch.zeros(max_steps, dtype=torch.float32, device=slots.device)

        for step in range(max_steps):
            pool = self.selected_pool(slot_embeds, selected_mask)
            action_logits = self.policy_logits(
                h,
                slot_embeds,
                selected_mask,
                step=step,
                valid_slot_mask=valid_slot_mask,
                min_steps=min_steps,
            )
            action = action_logits.argmax(dim=-1)
            is_stop = action == self.stop_idx
            select = active & ~is_stop
            stop = active & is_stop
            h, evidence, selected_mask = self.update_with_action(
                h, evidence, selected_mask, slot_embeds, action, select
            )
            step_logits = self.classify(
                h,
                evidence,
                self.selected_pool(slot_embeds, selected_mask),
                slot_embeds,
                selected_mask,
            )
            final_logits = torch.where(active.unsqueeze(-1), step_logits, final_logits)
            active = active & ~is_stop
            conf = step_logits.softmax(dim=-1).max(dim=-1).values
            next_min_steps = min_steps if min_steps is not None else cfg.min_steps
            enough = selected_mask.sum(dim=1) >= next_min_steps
            tau_stop = active & (conf >= early_exit_conf) & enough
            if stop.any() or tau_stop.any():
                stop_counts[step] = (stop | tau_stop).float().mean()
            active = active & ~tau_stop
            if not active.any():
                break

        return RolloutOutput(
            logits=final_logits,
            selected_mask=selected_mask,
            selected_counts=selected_mask.sum(dim=1),
            loss=None,
            actor_loss=slots.new_tensor(0.0),
            value_loss=slots.new_tensor(0.0),
            class_loss=slots.new_tensor(0.0),
            full_order_class_loss=slots.new_tensor(0.0),
            entropy=slots.new_tensor(0.0),
            mean_reward=slots.new_tensor(0.0),
            mean_return=slots.new_tensor(0.0),
            mean_advantage=slots.new_tensor(0.0),
            positive_reward_rate=slots.new_tensor(0.0),
            stop_rate_by_step=stop_counts,
        )


def compute_returns(rewards: list[torch.Tensor], gamma: float) -> list[torch.Tensor]:
    returns: list[torch.Tensor] = []
    running = torch.zeros_like(rewards[-1])
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def rollout_actor_critic(
    model: SlotSelectorAC,
    slots: torch.Tensor,
    labels: torch.Tensor,
    *,
    slot_pos: Optional[torch.Tensor] = None,
    valid_slot_mask: Optional[torch.Tensor] = None,
) -> RolloutOutput:
    cfg = model.cfg
    slot_embeds = model.embed_slots(slots, slot_pos)
    b, k, _ = slot_embeds.shape
    h, evidence = model.initial_state(slot_embeds)
    full_pool = slot_embeds.mean(dim=1)
    selected_mask = torch.zeros(b, k, dtype=torch.bool, device=slots.device)
    active = torch.ones(b, dtype=torch.bool, device=slots.device)
    prev_score = slots.new_full((b,), -math.log(cfg.num_classes))
    final_logits = model.classify(
        h,
        evidence,
        model.selected_pool(slot_embeds, selected_mask),
        slot_embeds,
        selected_mask,
    )
    target_min_slots = model.min_steps_for_classes(labels)

    rewards: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    active_weights: list[torch.Tensor] = []
    class_losses: list[torch.Tensor] = []
    stop_counts = torch.zeros(cfg.max_steps, dtype=torch.float32, device=slots.device)

    for step in range(min(cfg.max_steps, k)):
        if not active.any():
            break
        pool = model.selected_pool(slot_embeds, selected_mask)
        value_t = model.value(h, evidence, pool, full_pool)
        action_logits = model.policy_logits(
            h,
            slot_embeds,
            selected_mask,
            step=step,
            valid_slot_mask=valid_slot_mask,
        )
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        is_stop = action == model.stop_idx
        select = active & ~is_stop
        stop = active & is_stop

        h, evidence, selected_mask = model.update_with_action(
            h, evidence, selected_mask, slot_embeds, action, select
        )
        logits_cls = model.classify(
            h,
            evidence,
            model.selected_pool(slot_embeds, selected_mask),
            slot_embeds,
            selected_mask,
        )
        final_logits = torch.where(active.unsqueeze(-1), logits_cls, final_logits)
        log_p_true = F.log_softmax(logits_cls, dim=-1).gather(1, labels[:, None]).squeeze(1)
        ce = F.cross_entropy(logits_cls, labels, reduction="none")
        selected_count = selected_mask.sum(dim=1)
        pred = logits_cls.argmax(dim=-1)
        conf = logits_cls.softmax(dim=-1).max(dim=-1).values
        enough = selected_count >= cfg.min_steps

        select_reward = (log_p_true - prev_score) - cfg.lambda_slot
        if cfg.target_max_slots > 0:
            too_many = (selected_count > cfg.target_max_slots).to(slots.dtype)
        else:
            too_many = torch.zeros_like(selected_count, dtype=slots.dtype)
        too_few = (selected_count < target_min_slots).to(slots.dtype)
        stop_reward = torch.where(
            pred == labels,
            slots.new_full((b,), cfg.r_correct),
            slots.new_full((b,), -cfg.r_wrong),
        )
        stop_reward = stop_reward - cfg.lambda_over * too_many - cfg.lambda_under * too_few
        terminal_by_horizon = active & ~is_stop & (step == min(cfg.max_steps, k) - 1)
        reward = torch.where(is_stop | terminal_by_horizon, stop_reward, select_reward)
        reward = torch.where(active, reward, torch.zeros_like(reward))
        active_f = active.to(slots.dtype)

        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value_t)
        entropies.append(entropy)
        active_weights.append(active_f)
        class_losses.append(ce)
        if stop.any():
            stop_counts[step] = stop.float().mean()

        prev_score = torch.where(active, log_p_true.detach(), prev_score)
        active = active & ~is_stop

    if not rewards:
        zero = slots.new_tensor(0.0)
        return RolloutOutput(
            logits=final_logits,
            selected_mask=selected_mask,
            selected_counts=selected_mask.sum(dim=1),
            loss=zero,
            actor_loss=zero,
            value_loss=zero,
            class_loss=zero,
            full_order_class_loss=zero,
            entropy=zero,
            mean_reward=zero,
            mean_return=zero,
            mean_advantage=zero,
            positive_reward_rate=zero,
            stop_rate_by_step=stop_counts,
        )

    returns = compute_returns(rewards, cfg.gamma)
    weight = torch.stack(active_weights)
    log_prob_t = torch.stack(log_probs)
    value_t = torch.stack(values)
    return_t = torch.stack(returns)
    entropy_t = torch.stack(entropies)
    class_loss_t = torch.stack(class_losses)
    reward_t = torch.stack(rewards)
    denom = weight.sum().clamp_min(1.0)
    advantage = return_t - value_t

    actor_loss = -(log_prob_t * advantage.detach() * weight).sum() / denom
    value_loss = (F.mse_loss(value_t, return_t.detach(), reduction="none") * weight).sum() / denom
    class_loss = (class_loss_t * weight).sum() / denom
    full_order_class_loss = F.cross_entropy(model.full_order_logits(slot_embeds), labels)
    entropy = (entropy_t * weight).sum() / denom
    loss = (
        actor_loss
        + cfg.value_coef * value_loss
        + cfg.class_coef * class_loss
        + cfg.full_order_class_coef * full_order_class_loss
        - cfg.entropy_coef * entropy
    )

    return RolloutOutput(
        logits=final_logits,
        selected_mask=selected_mask,
        selected_counts=selected_mask.sum(dim=1),
        loss=loss,
        actor_loss=actor_loss.detach(),
        value_loss=value_loss.detach(),
        class_loss=class_loss.detach(),
        full_order_class_loss=full_order_class_loss.detach(),
        entropy=entropy.detach(),
        mean_reward=(reward_t * weight).sum().detach() / denom,
        mean_return=(return_t * weight).sum().detach() / denom,
        mean_advantage=(advantage * weight).sum().detach() / denom,
        positive_reward_rate=((reward_t > 0).to(slots.dtype) * weight).sum().detach() / denom,
        stop_rate_by_step=stop_counts.detach(),
    )


@torch.no_grad()
def evaluate_greedy(
    model: SlotSelectorAC,
    loader,
    slot_fn,
    device: torch.device,
    num_classes: int,
    *,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    selected_sum = 0.0
    selected_values = []
    count_hist = torch.zeros(model.cfg.num_slots + 1, dtype=torch.float64)
    per_class_correct = torch.zeros(num_classes, dtype=torch.float64)
    per_class_total = torch.zeros(num_classes, dtype=torch.float64)
    per_class_selected = torch.zeros(num_classes, dtype=torch.float64)

    for batch_idx, (images, _, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        labels = labels.to(device)
        slot_batch = slot_fn(images)
        if isinstance(slot_batch, tuple):
            slots, slot_pos = slot_batch
        else:
            slots, slot_pos = slot_batch, None
        out = model.forward_greedy(slots, slot_pos=slot_pos)
        logits = out.logits
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        pred = logits.argmax(dim=-1)
        counts = out.selected_counts.detach().cpu()
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        selected_sum += counts.sum().item()
        selected_values.extend(counts.tolist())
        for count in counts:
            count_hist[int(count.item())] += 1
        for cls in range(num_classes):
            mask = labels == cls
            n = mask.sum().item()
            per_class_total[cls] += n
            if n:
                per_class_correct[cls] += (pred[mask] == labels[mask]).sum().item()
                per_class_selected[cls] += out.selected_counts[mask].sum().item()

    if selected_values:
        selected_tensor = torch.tensor(selected_values, dtype=torch.float32)
        median_selected = float(selected_tensor.median().item())
    else:
        median_selected = 0.0

    metrics: dict[str, float] = {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "avg_selected": selected_sum / max(total, 1),
        "median_selected": median_selected,
        "total": float(total),
    }
    for count in range(model.cfg.num_slots + 1):
        metrics[f"selected_count_{count}"] = count_hist[count].item() / max(total, 1)
    for cls in range(num_classes):
        cls_total = per_class_total[cls].item()
        metrics[f"class_{cls}_accuracy"] = per_class_correct[cls].item() / max(cls_total, 1.0)
        metrics[f"class_{cls}_avg_selected"] = per_class_selected[cls].item() / max(cls_total, 1.0)
        metrics[f"class_{cls}_count"] = cls_total
    return metrics
