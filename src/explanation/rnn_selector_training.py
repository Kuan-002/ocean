"""
Sequential slot selector training: §4.1 expert imitation + GRPO (frozen SA & DeepSets).

Training rollouts never use confidence-based early exit (only STOP or max_steps).
Eval may use p>=tau early exit unless eval_disable_conf_early_exit is True.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.classification.deepsets import DeepSetsClassifier
from src.classification.training import batch_slots
from src.explanation.rnn_selector import SlotSelectionPolicyGRU


class SelectorConfig:
    """Runtime config for expert, imitation, GRPO, and eval."""

    __slots__ = (
        "num_slots",
        "tau",
        "epsilon",
        "max_steps",
        "lambda_len",
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
        max_grad_norm: float = 1.0,
        grpo_adv_clip: float = 10.0,
        imitation_alpha_class: float = 2.0,
        eval_require_tau_to_stop: bool = False,
        eval_disable_conf_early_exit: bool = False,
    ) -> None:
        self.num_slots = num_slots
        self.tau = tau
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.lambda_len = lambda_len
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
) -> dict:
    """
    Supervise action_head on expert actions; supervise class_head after each slot GRU step
    (targets = frozen clf(slots, None) argmax y_hat).
    """
    policy.train()
    clf.eval()
    sa.eval()
    meter = {"imitation_loss": 0.0, "imitation_cls_loss": 0.0, "n": 0}

    for images, _, _ in loader:
        images = images.to(device)
        with torch.no_grad():
            slots = batch_slots(sa, images, device)
            logits_full = clf(slots, None)
            probs_full = F.softmax(logits_full, dim=-1)
            y_hat = logits_full.argmax(dim=-1)
            p_full = probs_full[torch.arange(slots.size(0), device=device), y_hat]
            targets, targ_mask = build_expert_batch(clf, slots, y_hat, p_full, policy.stop_idx, cfg)

        bsz, k, d = slots.shape
        h = policy.init_hidden(bsz, device)
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
                    cls_logits = policy.class_head(h)
                    cls_loss = F.cross_entropy(cls_logits[idx], y_hat[idx], reduction="mean")
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


def run_grpo_rollout(
    policy: SlotSelectionPolicyGRU,
    ref_policy: SlotSelectionPolicyGRU,
    slots: torch.Tensor,
    y_hat_full: torch.Tensor,
    cfg: SelectorConfig,
    train: bool,
    action_trace: list[torch.Tensor] | None = None,
) -> dict:
    """
    train=True: sample actions; end only via STOP or max_steps (no p>=tau early exit).
    train=False: greedy; optional mask STOP until tau; optional p>=tau early exit after slot.
    """
    bsz, k, d = slots.shape
    device = slots.device
    h = policy.init_hidden(bsz, device)
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
            cls_chk = policy.class_head(h)
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
            cls_logits_final[idx] = policy.class_head(h[idx])
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

            cls_slot = policy.class_head(h)
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
        cls_logits_final[still] = policy.class_head(h[still])

    subset_norm = selected_mask.float().sum(dim=-1) / float(cfg.num_slots)
    ce = F.cross_entropy(cls_logits_final, y_hat_full, reduction="none")
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

    L_class = F.cross_entropy(cls_logits_final, y_hat_full, reduction="mean")

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
        "avg_subset_size": 0.0,
        "avg_final_p": 0.0,
        "n": 0,
    }

    for images, _, _ in loader:
        images = images.to(device)
        bsz = images.size(0)
        with torch.no_grad():
            slots = batch_slots(sa, images, device)
            logits_full = clf(slots, None)
            y_hat = logits_full.argmax(dim=-1)

        slots_e = slots.repeat_interleave(g, dim=0)
        y_hat_e = y_hat.repeat_interleave(g, dim=0)
        try:
            out = run_grpo_rollout(policy, ref_policy, slots_e, y_hat_e, cfg, train=True)
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

        if not torch.isfinite(loss):
            optimiser.zero_grad(set_to_none=True)
            continue

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimiser.step()

        with torch.no_grad():
            succ = out["p_conf"] >= cfg.tau
            k_avg = out["selected_mask"].float().sum(-1).mean()

        meter["loss"] += float(loss.item()) * bsz
        meter["L_grpo"] += float(L_grpo.item()) * bsz
        meter["L_class"] += float(L_class.item()) * bsz
        meter["mean_R"] += float(R.mean().item()) * bsz
        meter["success_rate"] += float(succ.float().mean().item()) * bsz
        meter["avg_subset_size"] += float(k_avg.item()) * bsz
        meter["avg_final_p"] += float(out["p_conf"].mean().item()) * bsz
        meter["n"] += bsz

    n = max(meter["n"], 1)
    return {k: (v / n if k != "n" else v) for k, v in meter.items()}


@torch.no_grad()
def eval_rnn_selector(
    sa,
    policy: SlotSelectionPolicyGRU,
    loader,
    device: torch.device,
    cfg: SelectorConfig,
    max_samples: int | None = None,
) -> dict:
    policy.eval()
    sa.eval()
    total = 0
    acc = {
        "success_rate": 0.0,
        "avg_subset_size": 0.0,
        "avg_final_p": 0.0,
        "mean_R": 0.0,
        "mean_steps": 0.0,
    }

    ref_policy = policy

    for batch in loader:
        images = batch[0].to(device)
        labels = batch[2].to(device)
        slots = batch_slots(sa, images, device)
        out = run_grpo_rollout(policy, ref_policy, slots, labels, cfg, train=False)
        ce = F.cross_entropy(out["cls_logits_final"], labels, reduction="none")
        subset_norm = out["selected_mask"].float().sum(-1) / float(cfg.num_slots)
        R = -ce - cfg.lambda_len * subset_norm

        b = images.size(0)
        acc["success_rate"] += float((out["p_conf"] >= cfg.tau).float().mean().item()) * b
        acc["avg_subset_size"] += float(out["selected_mask"].float().sum(-1).mean().item()) * b
        acc["avg_final_p"] += float(out["p_conf"].mean().item()) * b
        acc["mean_R"] += float(R.mean().item()) * b
        acc["mean_steps"] += float(out["episode_len"].mean().item()) * b
        total += b
        if max_samples is not None and total >= max_samples:
            break

    total = max(total, 1)
    return {k: v / total for k, v in acc.items()}
