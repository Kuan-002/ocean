"""
Sequential slot selector training: §4.1 expert imitation + GRPO (frozen SA & DeepSets).

Training rollouts never use confidence-based early exit (only STOP or max_steps).
Eval may use p>=tau early exit unless eval_disable_conf_early_exit is True.
When force_full_rollout is enabled, train rollouts select every slot exactly once
and reward early prefixes that reach the frozen DeepSets full-set confidence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.classification.deepsets import DeepSetsClassifier
from src.classification.training import batch_slots
from src.explanation.rnn_selector import SlotSelectionPolicyGRU


@torch.no_grad()
def run_rnn_inference_batch(
    policy: SlotSelectionPolicyGRU,
    slots: torch.Tensor,
    labels: torch.Tensor,
    tau: float,
    force_full: bool,
    global_init: bool,
) -> dict:
    """RNN-only inference for a batch. No DeepSets in the forward path."""
    bsz, k, d = slots.shape
    device = slots.device
    h = policy.init_hidden_from_slots(slots) if global_init else policy.init_hidden(bsz, device)
    selected_mask = torch.zeros(bsz, k, device=device, dtype=torch.bool)
    done = torch.zeros(bsz, device=device, dtype=torch.bool)
    slot_embeds = policy.precompute_slot_embeds(slots)  # [B, K, embed_dim]

    final_pred = torch.zeros(bsz, device=device, dtype=torch.long)
    final_pmax = torch.zeros(bsz, device=device)
    final_p_true = torch.zeros(bsz, device=device)
    steps = torch.zeros(bsz, device=device)
    stopped = torch.zeros(bsz, device=device, dtype=torch.bool)

    for step in range(k):
        active = ~done
        if not active.any():
            break
        steps = steps + active.float()

        action_logits = policy.apply_action_mask(policy.forward_logits(h), selected_mask)
        if action_logits.size(1) > k:
            fill = torch.finfo(action_logits.dtype).min / 2
            action_logits = action_logits.clone()
            action_logits[:, k:] = fill

        actions = action_logits.argmax(dim=-1)
        active_idx = active.nonzero(as_tuple=True)[0]
        selected_slots = torch.zeros(bsz, d, device=device, dtype=slots.dtype)
        selected_slots[active_idx] = slots[active_idx, actions[active_idx]]
        h_new = policy.step_hidden(selected_slots, h)
        h = torch.where(active.unsqueeze(-1), h_new, h)

        upd = torch.zeros_like(selected_mask)
        upd[active_idx, actions[active_idx]] = True
        selected_mask = selected_mask | upd

        cls_logits, _ = policy.class_head(h, slot_embeds, selected_mask)
        probs = F.softmax(cls_logits, dim=-1)
        pred = cls_logits.argmax(dim=-1)
        pmax = probs.max(dim=-1).values
        p_true = probs[torch.arange(bsz, device=device), labels]

        final_pred[active_idx] = pred[active_idx]
        final_pmax[active_idx] = pmax[active_idx]
        final_p_true[active_idx] = p_true[active_idx]

        reached = active & (pmax >= tau)
        if not force_full:
            stopped = stopped | reached
            done = done | reached

    still = ~done
    if still.any():
        cls_logits, _ = policy.class_head(h, slot_embeds, selected_mask)
        probs = F.softmax(cls_logits, dim=-1)
        final_pred[still] = cls_logits.argmax(dim=-1)[still]
        final_pmax[still] = probs.max(dim=-1).values[still]
        final_p_true[still] = probs[torch.arange(bsz, device=device), labels][still]

    return {
        "pred": final_pred,
        "pmax": final_pmax,
        "p_true": final_p_true,
        "steps": steps,
        "stopped": stopped,
        "selected_mask": selected_mask,
    }


class SelectorConfig:
    """Runtime config for expert, imitation, GRPO, and eval."""

    __slots__ = (
        "num_slots",
        "tau",
        "epsilon",
        "max_steps",
        "lambda_len",
        "force_full_rollout",
        "global_init",
        "area_weight",
        "area_decay_alpha",
        "success_reward",
        "fail_penalty",
        "grpo_group_size",
        "grpo_beta",
        "grpo_eps",
        "alpha_class",
        "max_grad_norm",
        "grpo_adv_clip",
        "imitation_alpha_class",
        "eval_require_tau_to_stop",
        "eval_disable_conf_early_exit",
        "eval_min_slots",
        "tstar_p_full_scale",
        "reward_cls_acc_weight",
        "reward_cls_ce_bonus",
        "grpo_class_teacher_kl",
        "grpo_class_teacher_temp",
        "reward_rule_weight",
        "reward_rank_weight",
        "reward_area_ptrue_weight",
        "kl_step_ramp",
        "lclass_step_ramp",
    )

    def __init__(
        self,
        *,
        num_slots: int,
        tau: float,
        epsilon: float,
        max_steps: int,
        lambda_len: float,
        success_reward: float,
        fail_penalty: float,
        grpo_group_size: int,
        grpo_beta: float,
        grpo_eps: float,
        alpha_class: float,
        force_full_rollout: bool = False,
        global_init: bool = False,
        area_weight: float = 0.25,
        area_decay_alpha: float = 0.0,
        max_grad_norm: float = 1.0,
        grpo_adv_clip: float = 10.0,
        imitation_alpha_class: float = 2.0,
        eval_require_tau_to_stop: bool = False,
        eval_disable_conf_early_exit: bool = False,
        eval_min_slots: int = 0,
        tstar_p_full_scale: float = 1.0,
        reward_cls_acc_weight: float = 0.0,
        reward_cls_ce_bonus: float = 0.0,
        grpo_class_teacher_kl: float = 0.0,
        grpo_class_teacher_temp: float = 1.0,
        reward_rule_weight: float = 1.0,
        reward_rank_weight: float = 0.0,
        reward_area_ptrue_weight: float = 0.0,
        kl_step_ramp: bool = True,
        lclass_step_ramp: bool = False,
    ) -> None:
        self.num_slots = num_slots
        self.tau = tau
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.lambda_len = lambda_len
        self.force_full_rollout = force_full_rollout
        self.global_init = global_init
        self.area_weight = area_weight
        self.area_decay_alpha = area_decay_alpha
        self.success_reward = success_reward
        self.fail_penalty = fail_penalty
        self.grpo_group_size = grpo_group_size
        self.grpo_beta = grpo_beta
        self.grpo_eps = grpo_eps
        self.alpha_class = alpha_class
        self.max_grad_norm = max_grad_norm
        self.grpo_adv_clip = grpo_adv_clip
        self.imitation_alpha_class = imitation_alpha_class
        self.eval_require_tau_to_stop = eval_require_tau_to_stop
        self.eval_disable_conf_early_exit = eval_disable_conf_early_exit
        self.eval_min_slots = eval_min_slots
        self.tstar_p_full_scale = tstar_p_full_scale
        self.reward_cls_acc_weight = reward_cls_acc_weight
        self.reward_cls_ce_bonus = reward_cls_ce_bonus
        self.grpo_class_teacher_kl = grpo_class_teacher_kl
        self.grpo_class_teacher_temp = grpo_class_teacher_temp
        self.reward_rule_weight = reward_rule_weight
        self.reward_rank_weight = reward_rank_weight
        self.reward_area_ptrue_weight = reward_area_ptrue_weight
        self.kl_step_ramp = kl_step_ramp
        self.lclass_step_ramp = lclass_step_ramp


def train_class_head_prewarm_epoch(
    sa,
    clf: DeepSetsClassifier,
    policy: SlotSelectionPolicyGRU,
    loader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    cfg: SelectorConfig,
    masks_per_image: int,
    slot_opts: dict | None = None,
) -> dict:
    """
    Pretrain class_head (and slot embedding / GRU) on random-prefix states.
    Teacher label = argmax DeepSets(slots, None). action_head gradients are disabled.
    """
    policy.train()
    clf.eval()
    sa.eval()
    slot_kw = slot_opts or {}
    k = cfg.num_slots
    meter = {"prewarm_loss": 0.0, "n": 0}

    prev_action_req: list[bool] = []
    for p in policy.action_head.parameters():
        prev_action_req.append(p.requires_grad)
        p.requires_grad_(False)
    try:
        for images, _, gt_labels in loader:
            images = images.to(device)
            gt_labels = gt_labels.to(device)
            bsz = images.size(0)
            with torch.no_grad():
                slots = batch_slots(sa, images, device, **slot_kw)

            total_loss = torch.zeros((), device=device)
            n_terms = 0
            for bi in range(bsz):
                s = slots[bi : bi + 1]
                gt_bi = gt_labels[bi : bi + 1]
                slot_embeds_s = policy.precompute_slot_embeds(s)  # [1, K, embed_dim]
                for _ in range(masks_per_image):
                    kk = int(torch.randint(1, k + 1, (1,), device=device).item())
                    perm = torch.randperm(k, device=device)[:kk]
                    h = (
                        policy.init_hidden_from_slots(s)
                        if cfg.global_init
                        else policy.init_hidden(1, device)
                    )
                    sel_mask = torch.zeros(1, k, device=device, dtype=torch.bool)
                    for j in range(kk):
                        pj = int(perm[j].item())
                        h = policy.step_hidden(s[:, pj], h)
                        sel_mask[0, pj] = True
                    logits, _ = policy.class_head(h, slot_embeds_s, sel_mask)
                    total_loss = total_loss + F.cross_entropy(logits, gt_bi)
                    n_terms += 1

            loss = total_loss / max(n_terms, 1)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimiser.step()

            meter["prewarm_loss"] += float(loss.detach().item()) * bsz
            meter["n"] += bsz
    finally:
        for p, was in zip(policy.action_head.parameters(), prev_action_req):
            p.requires_grad_(was)

    n = max(meter["n"], 1)
    return {"prewarm_loss": meter["prewarm_loss"] / n}


@torch.no_grad()
def build_expert_batch(
    clf: DeepSetsClassifier,
    slots: torch.Tensor,
    y_hat_full: torch.Tensor,
    p_full: torch.Tensor,
    stop_idx: int,
    cfg: SelectorConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy expert by marginal gain (masked DeepSets); vectorized over batch."""
    bsz, k, _ = slots.shape
    device = slots.device
    dtype = slots.dtype

    if cfg.force_full_rollout:
        selected = torch.zeros(bsz, k, device=device, dtype=dtype)
        sequences: list[list[int]] = [[] for _ in range(bsz)]
        b_range = torch.arange(bsz, device=device)

        for _ in range(k):
            logits_s = clf(slots, selected)
            p_s = F.softmax(logits_s, dim=-1)[b_range, y_hat_full]
            b_idx = b_range.unsqueeze(1).expand(bsz, k).reshape(bsz * k)
            j_idx = torch.arange(k, device=device).unsqueeze(0).expand(bsz, k).reshape(bsz * k)
            slot_stacked = slots[b_idx]
            sel2 = selected[b_idx].clone()
            row = torch.arange(bsz * k, device=device)
            sel2[row, j_idx] = torch.where(
                selected[b_idx, j_idx] < 0.5,
                torch.ones((), device=device, dtype=dtype),
                sel2[row, j_idx],
            )
            logits_c = clf(slot_stacked, sel2)
            p_cand = F.softmax(logits_c, dim=-1)[row, y_hat_full[b_idx]]
            gain = (p_cand - p_s[b_idx]).view(bsz, k)
            gain = gain.masked_fill(selected > 0.5, float("-inf"))
            best_j = gain.argmax(dim=1)

            for bi, act in enumerate(best_j.tolist()):
                sequences[bi].append(int(act))
            selected[b_range, best_j] = 1.0

        padded = torch.tensor(sequences, device=device, dtype=torch.long)
        mask = torch.ones(bsz, k, device=device, dtype=torch.bool)
        return padded, mask

    selected = torch.zeros(bsz, k, device=device, dtype=dtype)
    thresh = torch.maximum(torch.full_like(p_full, cfg.tau), p_full - cfg.epsilon)
    active = torch.ones(bsz, dtype=torch.bool, device=device)
    sequences: list[list[int]] = [[] for _ in range(bsz)]

    b_range = torch.arange(bsz, device=device)

    while active.any():
        logits_s = clf(slots, selected)
        p_s = F.softmax(logits_s, dim=-1)[b_range, y_hat_full]
        done_now = active & (p_s >= thresh)
        active = active & ~done_now
        if not active.any():
            break

        b_idx = b_range.unsqueeze(1).expand(bsz, k).reshape(bsz * k)
        j_idx = torch.arange(k, device=device).unsqueeze(0).expand(bsz, k).reshape(bsz * k)
        slot_stacked = slots[b_idx]
        sel2 = selected[b_idx].clone()
        row = torch.arange(bsz * k, device=device)
        sel2[row, j_idx] = torch.where(
            selected[b_idx, j_idx] < 0.5,
            torch.ones((), device=device, dtype=dtype),
            sel2[row, j_idx],
        )
        logits_c = clf(slot_stacked, sel2)
        p_cand = F.softmax(logits_c, dim=-1)[row, y_hat_full[b_idx]]
        gain = (p_cand - p_s[b_idx]).view(bsz, k)
        gain = gain.masked_fill(selected > 0.5, float("-inf"))
        gain = gain.masked_fill(~active.unsqueeze(1), float("-inf"))
        best_j = gain.argmax(dim=1)

        act_idx = active.nonzero(as_tuple=True)[0]
        for bi in act_idx.tolist():
            sequences[bi].append(int(best_j[bi].item()))
        selected[active, best_j[active]] = 1.0
        active = active & (selected < 0.5).any(dim=1)

    for bi in range(bsz):
        sequences[bi].append(stop_idx)

    max_len = max(len(s) for s in sequences)
    padded = torch.full((bsz, max_len), stop_idx, device=device, dtype=torch.long)
    mask = torch.zeros(bsz, max_len, device=device, dtype=torch.bool)
    for bi, s in enumerate(sequences):
        L = len(s)
        padded[bi, :L] = torch.tensor(s, device=device, dtype=torch.long)
        mask[bi, :L] = True
    return padded, mask


def train_imitation_epoch(
    sa,
    clf: DeepSetsClassifier,
    policy: SlotSelectionPolicyGRU,
    loader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    cfg: SelectorConfig,
    slot_opts: dict | None = None,
) -> dict:
    """
    Supervise action_head on expert actions; supervise class_head after each slot GRU step
    (targets = frozen clf(slots, None) argmax y_hat).
    """
    policy.train()
    clf.eval()
    sa.eval()
    meter = {"imitation_loss": 0.0, "imitation_cls_loss": 0.0, "n": 0}

    slot_kw = slot_opts or {}
    for images, _, gt_labels in loader:
        images = images.to(device)
        gt_labels = gt_labels.to(device)
        with torch.no_grad():
            slots = batch_slots(sa, images, device, **slot_kw)
            logits_full = clf(slots, None)
            probs_full = F.softmax(logits_full, dim=-1)
            y_hat = logits_full.argmax(dim=-1)
            p_full = probs_full[torch.arange(slots.size(0), device=device), y_hat]
            targets, targ_mask = build_expert_batch(clf, slots, y_hat, p_full, policy.stop_idx, cfg)

        bsz, k, d = slots.shape
        h = policy.init_hidden_from_slots(slots) if cfg.global_init else policy.init_hidden(bsz, device)
        slot_embeds = policy.precompute_slot_embeds(slots)  # [B, K, embed_dim]
        selected_mask = torch.zeros(bsz, k, device=device, dtype=torch.bool)
        max_len = targets.size(1)
        total_loss = torch.zeros((), device=device)
        cls_accum = torch.zeros((), device=device)
        cls_steps = 0
        denom = targ_mask.float().sum().clamp(min=1.0)

        for t in range(max_len):
            logits = policy.apply_action_mask(policy.forward_logits(h), selected_mask)
            step_ce = F.cross_entropy(logits, targets[:, t], reduction="none")
            total_loss = total_loss + (step_ce * targ_mask[:, t].float()).sum() / denom

            a = targets[:, t]
            is_stop = a == policy.stop_idx
            pick_slot = (~is_stop) & targ_mask[:, t]
            if pick_slot.any():
                ss = torch.zeros(bsz, d, device=device, dtype=slots.dtype)
                idx = torch.arange(bsz, device=device)[pick_slot]
                pj = a[pick_slot].clamp(max=k - 1)
                ss[idx] = slots[idx, pj]
                h = policy.step_hidden(ss, h)
                upd = torch.zeros_like(selected_mask)
                upd[idx, pj] = True
                selected_mask = selected_mask | upd

                if cfg.imitation_alpha_class > 0:
                    cls_logits, _ = policy.class_head(h, slot_embeds, selected_mask)
                    cls_loss = F.cross_entropy(cls_logits[idx], gt_labels[idx], reduction="mean")
                    total_loss = total_loss + cfg.imitation_alpha_class * cls_loss
                    cls_accum = cls_accum + cls_loss.detach()
                    cls_steps += 1

        optimiser.zero_grad(set_to_none=True)
        total_loss.backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimiser.step()

        b = images.size(0)
        meter["imitation_loss"] += float(total_loss.detach().item()) * b
        if cls_steps > 0:
            meter["imitation_cls_loss"] += float((cls_accum / cls_steps).item()) * b
        meter["n"] += b

    n = max(meter["n"], 1)
    out = {"imitation_loss": meter["imitation_loss"] / n}
    if meter["imitation_cls_loss"] > 0:
        out["imitation_cls_loss"] = meter["imitation_cls_loss"] / n
    else:
        out["imitation_cls_loss"] = 0.0
    return out


def _run_full_order_rollout(
    policy: SlotSelectionPolicyGRU,
    ref_policy: SlotSelectionPolicyGRU,
    slots: torch.Tensor,
    y_hat_full: torch.Tensor,
    cfg: SelectorConfig,
    train: bool,
    action_trace: list[torch.Tensor] | None,
    clf: DeepSetsClassifier | None,
    p_full: torch.Tensor | None,
    gt_labels: torch.Tensor | None = None,
) -> dict:
    """Select every slot during training; score by first prefix reaching p_full."""
    if train and (clf is None or p_full is None):
        raise ValueError("force_full_rollout training requires clf and p_full")

    bsz, k, d = slots.shape
    device = slots.device
    h = policy.init_hidden_from_slots(slots) if cfg.global_init else policy.init_hidden(bsz, device)
    slot_embeds = policy.precompute_slot_embeds(slots)  # [B, K, embed_dim]
    selected_mask = torch.zeros(bsz, k, device=device, dtype=torch.bool)
    done = torch.zeros(bsz, device=device, dtype=torch.bool)
    h_dim = h.size(-1)
    num_cls = policy.num_classes
    h_final = torch.zeros(bsz, h_dim, device=device)
    cls_logits_final = torch.zeros(bsz, num_cls, device=device)

    log_probs_steps: list[torch.Tensor] = []
    kl_steps: list[torch.Tensor] = []
    valid_steps: list[torch.Tensor] = []
    cls_logits_steps: list[torch.Tensor] = []
    cls_valid_steps: list[torch.Tensor] = []
    prefix_conf_steps: list[torch.Tensor] = []
    episode_len = torch.zeros(bsz, device=device)

    class_kl_steps: list[torch.Tensor] = []
    p_true_steps: list[torch.Tensor] = []

    t_star = torch.full((bsz,), k, device=device, dtype=torch.long)
    t_star_found = torch.zeros(bsz, device=device, dtype=torch.bool)
    threshold = torch.full((bsz,), cfg.tau, device=device)
    b_range = torch.arange(bsz, device=device)

    # Constant across steps — compute once
    _step_target = gt_labels if gt_labels is not None else y_hat_full
    with torch.no_grad():
        teacher_logits = clf(slots, None) if (cfg.grpo_class_teacher_kl > 0 and clf is not None) else None

    for step in range(k):
        active = ~done
        if not active.any():
            break
        episode_len = episode_len + active.float()

        logits = policy.apply_action_mask(policy.forward_logits(h), selected_mask)
        with torch.no_grad():
            lr = ref_policy.apply_action_mask(ref_policy.forward_logits(h), selected_mask)
        if logits.size(1) > k:
            fill = torch.finfo(logits.dtype).min / 2
            logits = logits.clone()
            logits[:, k:] = fill
            lr = lr.clone()
            lr[:, k:] = fill

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "Non-finite action logits in run_grpo_rollout. "
                "Lower RNN_SEL_LR / RNN_SEL_ALPHA_CLASS, tighten RNN_SEL_MAX_GRAD_NORM, "
                "or raise RNN_SEL_GRPO_ADV_CLIP; use CUDA_LAUNCH_BLOCKING=1."
            )

        log_pt = F.log_softmax(logits, dim=-1)
        log_pref = F.log_softmax(lr, dim=-1)
        pt = log_pt.exp()
        kl = (pt * (log_pt - log_pref)).sum(dim=-1)

        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample() if train else logits.argmax(dim=-1)
        if action_trace is not None:
            action_trace.append(actions.detach().clone())
        log_prob = dist.log_prob(actions)

        log_probs_steps.append(torch.where(active, log_prob, torch.zeros_like(log_prob)))
        kl_steps.append(torch.where(active, kl, torch.zeros_like(kl)))
        valid_steps.append(active.float())

        ss = torch.zeros(bsz, d, device=device, dtype=slots.dtype)
        b_idx = active.nonzero(as_tuple=True)[0]
        ss[b_idx] = slots[b_idx, actions[b_idx]]
        h_new = policy.step_hidden(ss, h)
        h = torch.where(active.unsqueeze(-1), h_new, h)
        upd = torch.zeros_like(selected_mask)
        upd[b_idx, actions[b_idx]] = True
        selected_mask = selected_mask | upd

        cls_logits, _ = policy.class_head(h, slot_embeds, selected_mask)
        cls_logits_steps.append(cls_logits)
        cls_valid_steps.append(active.float())

        with torch.no_grad():
            prefix_conf = cls_logits.detach().softmax(-1).max(-1).values
        prefix_conf_steps.append(prefix_conf)

        # p_true: probability assigned to GT class at this step (for R_area_ptrue)
        p_true_step = cls_logits.detach().softmax(-1)[b_range, _step_target]
        p_true_steps.append(torch.where(active, p_true_step, torch.zeros_like(p_true_step)))

        # Per-step KL ramped by step/K: near-zero early, full weight at last step
        # teacher_logits = DeepSets(all slots) — precomputed once outside loop
        if cfg.grpo_class_teacher_kl > 0 and teacher_logits is not None:
            ramp = (step + 1) / float(k) if cfg.kl_step_ramp else 1.0
            T = cfg.grpo_class_teacher_temp
            with torch.no_grad():
                t_lp = F.log_softmax(teacher_logits / T, dim=-1)
                t_p = t_lp.exp()
            s_lp = F.log_softmax(cls_logits / T, dim=-1)
            kl_cls = (t_p * (t_lp - s_lp)).sum(dim=-1)
            class_kl_steps.append(ramp * torch.where(active, kl_cls, torch.zeros_like(kl_cls)))

        # t*: first step where class_head is GT-correct AND pmax >= tau
        pred_step = cls_logits.detach().argmax(-1)
        just_reached = (active & ~t_star_found & (prefix_conf >= threshold)
                        & (pred_step == _step_target))
        if just_reached.any():
            t_star[just_reached] = step + 1
            t_star_found = t_star_found | just_reached

        # Eval-only early exit: require minimum slots before stopping
        should_stop = torch.zeros(bsz, device=device, dtype=torch.bool)
        if (not train) and (not cfg.eval_disable_conf_early_exit):
            if step + 1 >= cfg.eval_min_slots:
                should_stop = cls_logits.softmax(-1).max(-1).values >= cfg.tau
        if should_stop.any():
            si = (active & should_stop).nonzero(as_tuple=True)[0]
            h_final[si] = h[si]
            cls_logits_final[si] = cls_logits[si]
            done[si] = True

    still = ~done
    if still.any():
        h_final[still] = h[still]
        cls_logits_final[still], _ = policy.class_head(
            h[still], None, selected_mask[still]
        )

    log_st = torch.stack(log_probs_steps, dim=0)
    kl_st = torch.stack(kl_steps, dim=0)
    val_st = torch.stack(valid_steps, dim=0)
    per_traj_logp = (log_st * val_st).sum(dim=0)
    kl_mean = (kl_st * val_st).sum() / val_st.sum().clamp(min=1.0)

    prefix_conf_stack = torch.stack(prefix_conf_steps, dim=0)  # [T, B]
    prefix_valid = torch.stack(valid_steps, dim=0)              # [T, B]

    # pmax area (kept for logging; not used in reward)
    area = (prefix_conf_stack * prefix_valid).sum(dim=0) / prefix_valid.sum(dim=0).clamp(min=1.0)

    _reward_target = gt_labels if gt_labels is not None else y_hat_full

    # R_rule: binary GT correctness at final step — primary correctness gate
    pred_final_r = cls_logits_final.argmax(dim=-1)
    R_rule = (pred_final_r == _reward_target).float()

    # R_rank: NDCG position reward — t* = first step GT-correct AND pmax >= tau
    dcg_w = 1.0 / torch.log2(
        torch.arange(1, k + 1, device=device, dtype=torch.float) + 1.0
    )  # [K]: position 0 → 1.0, position k-1 → small
    dcg_w = dcg_w / dcg_w.sum()
    R_rank = torch.where(
        t_star_found,
        dcg_w[(t_star - 1).clamp(min=0, max=k - 1)],
        torch.zeros(bsz, device=device),
    )

    # R_area_ptrue: mean probability assigned to GT class across all steps
    p_true_stack = torch.stack(p_true_steps, dim=0)  # [T, B]
    R_area_ptrue = (p_true_stack * prefix_valid).sum(dim=0) / prefix_valid.sum(dim=0).clamp(min=1e-6)

    R = (cfg.reward_rule_weight * R_rule
         + cfg.reward_rank_weight * R_rule * R_rank
         + cfg.reward_area_ptrue_weight * R_area_ptrue)

    if class_kl_steps:
        kl_class_stack = torch.stack(class_kl_steps, dim=0)
        kl_class_mean = (kl_class_stack * prefix_valid).sum() / prefix_valid.sum().clamp(min=1.0)
    else:
        kl_class_mean = torch.zeros((), device=device)

    cls_stack = torch.stack(cls_logits_steps, dim=0)
    cls_valid = torch.stack(cls_valid_steps, dim=0)
    _cls_target = gt_labels if gt_labels is not None else y_hat_full
    ce_steps = F.cross_entropy(
        cls_stack.reshape(-1, num_cls),
        _cls_target.unsqueeze(0).expand(cls_stack.size(0), bsz).reshape(-1),
        reduction="none",
    ).view(cls_stack.size(0), bsz)
    if cfg.lclass_step_ramp and cfg.area_decay_alpha > 0.0:
        # Earlier steps get higher weight: w_t = (1-alpha)^t, t=0 is first slot
        T_s = cls_stack.size(0)
        step_w = torch.pow(
            torch.tensor(1.0 - cfg.area_decay_alpha, device=device),
            torch.arange(T_s, device=device, dtype=torch.float),
        )
        step_w = step_w / step_w.sum()
        L_class = (ce_steps * cls_valid * step_w.unsqueeze(1)).sum() / (
            cls_valid * step_w.unsqueeze(1)
        ).sum().clamp(min=1e-6)
    else:
        L_class = (ce_steps * cls_valid).sum() / cls_valid.sum().clamp(min=1.0)

    p_conf = cls_logits_final.softmax(-1).max(-1).values
    return {
        "R": R,
        "per_traj_logp": per_traj_logp,
        "kl_mean": kl_mean,
        "L_class": L_class,
        "cls_logits_final": cls_logits_final,
        "subset_norm": selected_mask.float().sum(dim=-1) / float(cfg.num_slots),
        "p_conf": p_conf,
        "stopped_with_stop": torch.zeros(bsz, device=device, dtype=torch.bool),
        "timed_out": torch.zeros(bsz, device=device, dtype=torch.bool),
        "selected_mask": selected_mask,
        "episode_len": episode_len,
        "t_star": t_star,
        "t_star_found": t_star_found,
        "prefix_conf_area": area,
        "R_rule": R_rule,
        "R_rank": R_rank,
        "R_area_ptrue": R_area_ptrue,
        "kl_class_mean": kl_class_mean,
    }


def run_grpo_rollout(
    policy: SlotSelectionPolicyGRU,
    ref_policy: SlotSelectionPolicyGRU,
    slots: torch.Tensor,
    y_hat_full: torch.Tensor,
    cfg: SelectorConfig,
    train: bool,
    action_trace: list[torch.Tensor] | None = None,
    clf: DeepSetsClassifier | None = None,
    p_full: torch.Tensor | None = None,
    gt_labels: torch.Tensor | None = None,
) -> dict:
    """
    train=True: sample actions; end only via STOP or max_steps (no p>=tau early exit).
    train=False: greedy; optional mask STOP until tau; optional p>=tau early exit after slot.
    In force_full_rollout mode, train=True selects all slots exactly once with no STOP.
    gt_labels: ground-truth class labels used for L_class; y_hat_full is kept for reward shaping only.
    """
    if cfg.force_full_rollout:
        return _run_full_order_rollout(
            policy, ref_policy, slots, y_hat_full, cfg, train, action_trace, clf, p_full,
            gt_labels=gt_labels,
        )

    bsz, k, d = slots.shape
    device = slots.device
    h = policy.init_hidden_from_slots(slots) if cfg.global_init else policy.init_hidden(bsz, device)
    slot_embeds = policy.precompute_slot_embeds(slots)  # [B, K, embed_dim]
    selected_mask = torch.zeros(bsz, k, device=device, dtype=torch.bool)
    done = torch.zeros(bsz, device=device, dtype=torch.bool)
    h_dim = h.size(-1)
    num_cls = policy.num_classes
    h_final = torch.zeros(bsz, h_dim, device=device)
    cls_logits_final = torch.zeros(bsz, num_cls, device=device)
    stopped_with_stop = torch.zeros(bsz, device=device, dtype=torch.bool)
    timed_out = torch.zeros(bsz, device=device, dtype=torch.bool)

    log_probs_steps: list[torch.Tensor] = []
    kl_steps: list[torch.Tensor] = []
    valid_steps: list[torch.Tensor] = []
    episode_len = torch.zeros(bsz, device=device)

    for _ in range(cfg.max_steps):
        active = ~done
        if not active.any():
            break
        episode_len = episode_len + active.float()

        logits = policy.forward_logits(h)
        logits = policy.apply_action_mask(logits, selected_mask)
        with torch.no_grad():
            lr = ref_policy.forward_logits(h)
            lr = ref_policy.apply_action_mask(lr, selected_mask)

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "Non-finite action logits in run_grpo_rollout. "
                "Lower RNN_SEL_LR / RNN_SEL_ALPHA_CLASS, tighten RNN_SEL_MAX_GRAD_NORM, "
                "or raise RNN_SEL_GRPO_ADV_CLIP; use CUDA_LAUNCH_BLOCKING=1."
            )

        if not train and cfg.eval_require_tau_to_stop:
            cls_chk, _ = policy.class_head(h, slot_embeds, selected_mask)
            pmax_chk = cls_chk.softmax(-1).max(-1).values
            slots_left = (~selected_mask).any(dim=1)
            mask_stop = active & slots_left & (pmax_chk < cfg.tau)
            if mask_stop.any():
                logits = logits.clone()
                fill = torch.finfo(logits.dtype).min / 2
                logits[mask_stop, policy.stop_idx] = fill

        log_pt = F.log_softmax(logits, dim=-1)
        log_pref = F.log_softmax(lr, dim=-1)
        pt = log_pt.exp()
        kl = (pt * (log_pt - log_pref)).sum(dim=-1)

        dist = torch.distributions.Categorical(logits=logits)
        if train:
            actions = dist.sample()
        else:
            actions = logits.argmax(dim=-1)
        if action_trace is not None:
            action_trace.append(actions.detach().clone())
        log_prob = dist.log_prob(actions)

        log_prob = torch.where(active, log_prob, torch.zeros_like(log_prob))
        kl = torch.where(active, kl, torch.zeros_like(kl))
        log_probs_steps.append(log_prob)
        kl_steps.append(kl)
        valid_steps.append(active.float())

        is_stop = active & (actions == policy.stop_idx)
        pick = actions.clamp(max=k - 1)

        if is_stop.any():
            idx = is_stop.nonzero(as_tuple=True)[0]
            h_final[idx] = h[idx]
            cls_logits_final[idx], _ = policy.class_head(
                h[idx], None, selected_mask[idx]
            )
            stopped_with_stop[idx] = True
            done[idx] = True

        slot_pick = active & ~is_stop
        if slot_pick.any():
            ss = torch.zeros(bsz, d, device=device, dtype=slots.dtype)
            b_idx = slot_pick.nonzero(as_tuple=True)[0]
            ss[b_idx] = slots[b_idx, pick[b_idx]]
            h_new = policy.step_hidden(ss, h)
            h = torch.where(slot_pick.unsqueeze(-1), h_new, h)
            upd = torch.zeros_like(selected_mask)
            upd[b_idx, pick[b_idx]] = True
            selected_mask = selected_mask | upd

            cls_slot, _ = policy.class_head(h, slot_embeds, selected_mask)
            if not train and not cfg.eval_disable_conf_early_exit:
                pmax = cls_slot.softmax(-1).max(-1).values
                safety = slot_pick & ~done & (pmax >= cfg.tau)
                if safety.any():
                    si = safety.nonzero(as_tuple=True)[0]
                    h_final[si] = h[si]
                    cls_logits_final[si] = cls_slot[si]
                    done[si] = True

    still = ~done
    if still.any():
        timed_out[still] = True
        h_final[still] = h[still]
        cls_logits_final[still], _ = policy.class_head(
            h[still], None, selected_mask[still]
        )

    subset_norm = selected_mask.float().sum(dim=-1) / float(cfg.num_slots)
    _reward_target = gt_labels if gt_labels is not None else y_hat_full
    ce = F.cross_entropy(cls_logits_final, _reward_target, reduction="none")
    p_conf = cls_logits_final.softmax(-1).max(-1).values

    R = -ce - cfg.lambda_len * subset_norm
    R = R + torch.where(
        stopped_with_stop & (p_conf >= cfg.tau),
        cfg.success_reward * (1.0 - subset_norm),
        torch.zeros_like(R),
    )
    R = R + torch.where(
        stopped_with_stop & (p_conf < cfg.tau),
        torch.full_like(R, -cfg.fail_penalty),
        torch.zeros_like(R),
    )
    R = R + torch.where(timed_out, torch.full_like(R, -cfg.fail_penalty), torch.zeros_like(R))

    log_st = torch.stack(log_probs_steps, dim=0)
    kl_st = torch.stack(kl_steps, dim=0)
    val_st = torch.stack(valid_steps, dim=0)
    per_traj_logp = (log_st * val_st).sum(dim=0)
    kl_mean = (kl_st * val_st).sum() / val_st.sum().clamp(min=1.0)

    _cls_target = gt_labels if gt_labels is not None else y_hat_full
    L_class = F.cross_entropy(cls_logits_final, _cls_target, reduction="mean")

    return {
        "R": R,
        "per_traj_logp": per_traj_logp,
        "kl_mean": kl_mean,
        "L_class": L_class,
        "cls_logits_final": cls_logits_final,
        "subset_norm": subset_norm,
        "p_conf": p_conf,
        "stopped_with_stop": stopped_with_stop,
        "timed_out": timed_out,
        "selected_mask": selected_mask,
        "episode_len": episode_len,
    }


def train_grpo_epoch(
    sa,
    clf: DeepSetsClassifier,
    policy: SlotSelectionPolicyGRU,
    ref_policy: SlotSelectionPolicyGRU,
    loader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    cfg: SelectorConfig,
    slot_opts: dict | None = None,
) -> dict:
    policy.train()
    ref_policy.eval()
    clf.eval()
    sa.eval()

    g = cfg.grpo_group_size
    meter = {
        "loss": 0.0,
        "L_grpo": 0.0,
        "L_class": 0.0,
        "mean_R": 0.0,
        "success_rate": 0.0,
        "gt_acc": 0.0,
        "avg_subset_size": 0.0,
        "avg_final_p": 0.0,
        "avg_t_star": 0.0,
        "n": 0,
    }

    slot_kw = slot_opts or {}
    for images, _, gt_labels in loader:
        images = images.to(device)
        gt_labels = gt_labels.to(device)
        bsz = images.size(0)
        with torch.no_grad():
            slots = batch_slots(sa, images, device, **slot_kw)
            logits_full = clf(slots, None)
            probs_full = F.softmax(logits_full, dim=-1)
            y_hat = logits_full.argmax(dim=-1)
            p_full = probs_full[torch.arange(bsz, device=device), y_hat]

        slots_e = slots.repeat_interleave(g, dim=0)
        y_hat_e = y_hat.repeat_interleave(g, dim=0)
        p_full_e = p_full.repeat_interleave(g, dim=0)
        gt_labels_e = gt_labels.repeat_interleave(g, dim=0)
        try:
            out = run_grpo_rollout(
                policy,
                ref_policy,
                slots_e,
                y_hat_e,
                cfg,
                train=True,
                clf=clf,
                p_full=p_full_e,
                gt_labels=gt_labels_e,
            )
        except (RuntimeError, ValueError) as err:
            msg = str(err)
            if "Non-finite action logits" in msg or (
                "Categorical" in msg and "invalid values" in msg
            ):
                optimiser.zero_grad(set_to_none=True)
                continue
            raise

        R = out["R"]
        R_mat = R.view(bsz, g)
        if g <= 1:
            A = torch.zeros(bsz * g, device=device)
        else:
            A = (R_mat - R_mat.mean(dim=1, keepdim=True)) / (
                R_mat.std(dim=1, keepdim=True) + cfg.grpo_eps
            )
            A = A.reshape(-1)
            if cfg.grpo_adv_clip > 0:
                A = A.clamp(-cfg.grpo_adv_clip, cfg.grpo_adv_clip)

        loss_pg = -(A * out["per_traj_logp"]).mean()
        loss_kl = cfg.grpo_beta * out["kl_mean"]
        L_grpo = loss_pg + loss_kl
        L_class = out["L_class"]
        loss = L_grpo + cfg.alpha_class * L_class
        if cfg.grpo_class_teacher_kl > 0 and "kl_class_mean" in out:
            loss = loss + cfg.grpo_class_teacher_kl * out["kl_class_mean"]

        if not torch.isfinite(loss):
            optimiser.zero_grad(set_to_none=True)
            continue

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimiser.step()

        with torch.no_grad():
            # For force_full_rollout: success = teacher threshold reached (t_star found).
            # For standard rollout: success = class_head confident at stopping point.
            if "t_star_found" in out:
                succ = out["t_star_found"]
            else:
                succ = out["p_conf"] >= cfg.tau
            # GT accuracy of the final class_head prediction
            gt_acc = (out["cls_logits_final"].argmax(-1) == gt_labels_e).float().mean()
            k_avg = out["selected_mask"].float().sum(-1).mean()

        meter["loss"] += float(loss.item()) * bsz
        meter["L_grpo"] += float(L_grpo.item()) * bsz
        meter["L_class"] += float(L_class.item()) * bsz
        meter["mean_R"] += float(R.mean().item()) * bsz
        meter["success_rate"] += float(succ.float().mean().item()) * bsz
        meter["gt_acc"] += float(gt_acc.item()) * bsz
        meter["avg_subset_size"] += float(k_avg.item()) * bsz
        meter["avg_final_p"] += float(out["p_conf"].mean().item()) * bsz
        if "t_star" in out:
            meter["avg_t_star"] += float(out["t_star"].float().mean().item()) * bsz
        meter["n"] += bsz

    n = max(meter["n"], 1)
    return {k: (v / n if k != "n" else v) for k, v in meter.items()}


@torch.no_grad()
def eval_rnn_classifier_accuracy(
    sa,
    policy: SlotSelectionPolicyGRU,
    loader,
    device: torch.device,
    cfg: SelectorConfig,
    max_samples: int | None = None,
    slot_opts: dict | None = None,
) -> float:
    """Ground-truth classification accuracy with RNN-only greedy rollout (no DeepSets)."""
    policy.eval()
    sa.eval()
    slot_kw = slot_opts or {}
    correct = 0
    total = 0
    for batch in loader:
        images = batch[0].to(device)
        labels = batch[2].to(device)
        slots = batch_slots(sa, images, device, **slot_kw)
        out = run_rnn_inference_batch(
            policy,
            slots,
            labels,
            cfg.tau,
            cfg.force_full_rollout,
            cfg.global_init,
        )
        correct += int((out["pred"] == labels).sum().item())
        total += labels.size(0)
        if max_samples is not None and total >= max_samples:
            break
    return correct / max(total, 1)


@torch.no_grad()
def eval_rnn_selector(
    sa,
    policy: SlotSelectionPolicyGRU,
    loader,
    device: torch.device,
    cfg: SelectorConfig,
    max_samples: int | None = None,
    clf: DeepSetsClassifier | None = None,
    slot_opts: dict | None = None,
) -> dict:
    policy.eval()
    sa.eval()
    total = 0
    acc = {
        "success_rate": 0.0,
        "gt_acc": 0.0,
        "avg_subset_size": 0.0,
        "avg_final_p": 0.0,
        "mean_R": 0.0,
        "mean_steps": 0.0,
        "avg_t_star": 0.0,
    }

    ref_policy = policy
    slot_kw = slot_opts or {}

    for batch in loader:
        images = batch[0].to(device)
        labels = batch[2].to(device)
        slots = batch_slots(sa, images, device, **slot_kw)
        if cfg.force_full_rollout and clf is not None:
            logits_full = clf(slots, None)
            probs_full = F.softmax(logits_full, dim=-1)
            y_hat = logits_full.argmax(dim=-1)
            p_full = probs_full[torch.arange(slots.size(0), device=device), y_hat]
            out = run_grpo_rollout(
                policy,
                ref_policy,
                slots,
                y_hat,
                cfg,
                train=False,
                clf=clf,
                p_full=p_full,
                gt_labels=labels,
            )
        else:
            out = run_grpo_rollout(policy, ref_policy, slots, labels, cfg, train=False)
        ce = F.cross_entropy(out["cls_logits_final"], labels, reduction="none")
        subset_norm = out["selected_mask"].float().sum(-1) / float(cfg.num_slots)
        R = -ce - cfg.lambda_len * subset_norm

        b = images.size(0)
        # For force_full_rollout: success = teacher threshold found at some prefix step
        if "t_star_found" in out:
            succ_rate = out["t_star_found"].float().mean().item()
        else:
            succ_rate = (out["p_conf"] >= cfg.tau).float().mean().item()
        gt_correct = (out["cls_logits_final"].argmax(-1) == labels).float().mean().item()

        acc["success_rate"] += float(succ_rate) * b
        acc["gt_acc"] += float(gt_correct) * b
        acc["avg_subset_size"] += float(out["selected_mask"].float().sum(-1).mean().item()) * b
        acc["avg_final_p"] += float(out["p_conf"].mean().item()) * b
        acc["mean_R"] += float(R.mean().item()) * b
        acc["mean_steps"] += float(out["episode_len"].mean().item()) * b
        if "t_star" in out:
            acc["avg_t_star"] += float(out["t_star"].float().mean().item()) * b
        total += b
        if max_samples is not None and total >= max_samples:
            break

    total = max(total, 1)
    return {k: v / total for k, v in acc.items()}
