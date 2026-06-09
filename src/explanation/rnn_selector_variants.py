"""
Variant policy models for the sequential slot selector.

Provides two lightweight alternatives to the Recurrent Evidence Head:

  SlotSelectionPolicyLinear  — classification from final GRU hidden state h_T
                               (same as early experiments before Evidence Head)

  SlotSelectionPolicyCrossAttn — classification via cross-attention over all
                                  selected slot embeddings, using h as the query

Both expose the same interface as SlotSelectionPolicyGRU so they can be used
transparently by the shared training / evaluation infrastructure:

  init_hidden, init_evidence, step_hidden, step_evidence,
  forward_logits, class_head, precompute_slot_embeds, apply_action_mask

All variants are compatible with SelectorConfig, train_grpo_epoch,
eval_rnn_selector, eval_rnn_classifier_accuracy, and run_rnn_inference_batch.

Factory
───────
  build_policy(head_type, ...) → nn.Module
    "linear"   → SlotSelectionPolicyLinear
    "crossattn" → SlotSelectionPolicyCrossAttn
    "evidence"  → SlotSelectionPolicyGRU  (imported from rnn_selector)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.explanation.rnn_selector import SlotCrossAttentionClassHead


# ═════════════════════════════════════════════════════════════════════════════
# A. Linear head  —  logits = Linear(h_T)
# ═════════════════════════════════════════════════════════════════════════════

class SlotSelectionPolicyLinear(nn.Module):
    """
    Sequential slot-selection policy with a plain linear classification head.

    Architecture (per step t)
    ─────────────────────────
      x_t          = input_proj(slot_t)        SlotEncoder
      h_t          = GRU(x_t, h_{t-1})         main GRU hidden state
      policy_t     = PolicyHead(h_t)            slot-selection logits [K]
      stop_t       = StopHead(h_t)              stop logit [1]
      class_logits = Linear(h_T)               final hidden state only

    Evidence state E is a no-op pass-through (API compatibility with Evidence head).
    """

    def __init__(
        self,
        slot_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_slots: int,
        num_classes: int,
        use_stop_action: bool = True,
        **kwargs,                    # absorb unused class_head_* kwargs
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.use_stop_action = use_stop_action
        self.stop_idx = num_slots
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(slot_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gru = nn.GRUCell(embed_dim, hidden_dim)
        self._policy_head = nn.Linear(hidden_dim, num_slots)
        self._stop_head = nn.Linear(hidden_dim, 1) if use_stop_action else None
        self._class_fc = nn.Linear(hidden_dim, num_classes)

    # ── initialisation ────────────────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def init_evidence(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compatibility stub; Evidence GRU is not used in this variant."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    # ── per-step computations ─────────────────────────────────────────────────

    def step_hidden(
        self, selected_slots: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(selected_slots)
        h_new = self.gru(x, hidden)
        return h_new, x

    def init_slot_cache(self, batch_size: int, device):
        """No-op: Linear head does not use a slot cache."""
        return None

    def update_slot_cache(self, slot_cache, x: torch.Tensor, active: torch.Tensor):
        """No-op: Linear head does not use a slot cache."""
        return None

    def step_evidence(
        self, delta: torch.Tensor, x: torch.Tensor, E_prev: torch.Tensor,
        slot_cache=None,
    ) -> torch.Tensor:
        """No-op: returns E_prev unchanged (no evidence accumulation)."""
        return E_prev

    # ── heads ─────────────────────────────────────────────────────────────────

    def forward_logits(self, hidden: torch.Tensor, slot_embeds=None) -> torch.Tensor:
        """policy_t ∥ stop_t  →  [B, K] or [B, K+1]."""
        policy_logits = self._policy_head(hidden)
        if self.use_stop_action and self._stop_head is not None:
            stop_logit = self._stop_head(hidden)
            return torch.cat([policy_logits, stop_logit], dim=-1)
        return policy_logits

    def class_head(
        self,
        h: torch.Tensor,
        slot_embeds=None,       # unused; accepted for API compatibility
        selected_mask=None,     # unused; accepted for API compatibility
        E: torch.Tensor | None = None,  # unused; accepted for API compatibility
    ) -> tuple[torch.Tensor, None]:
        """class_logits = Linear(h).  h is the main GRU hidden state."""
        return self._class_fc(h), None

    def precompute_slot_embeds(self, slots: torch.Tensor) -> None:
        """No-op: Linear classifier does not use slot embeddings directly."""
        return None

    def apply_action_mask(
        self, logits: torch.Tensor, selected_mask: torch.Tensor
    ) -> torch.Tensor:
        masked = logits.clone()
        mask_fill = torch.finfo(logits.dtype).min / 2
        masked[:, : self.num_slots] = masked[:, : self.num_slots].masked_fill(
            selected_mask, mask_fill
        )
        return masked


# ═════════════════════════════════════════════════════════════════════════════
# B. Cross-attention head  —  logits from cross-attn over selected slots
# ═════════════════════════════════════════════════════════════════════════════

class SlotSelectionPolicyCrossAttn(nn.Module):
    """
    Sequential slot-selection policy with a cross-attention classification head.

    Architecture (per step t)
    ─────────────────────────
      x_t          = input_proj(slot_t)         SlotEncoder
      h_t          = GRU(x_t, h_{t-1})          main GRU hidden state
      policy_t     = PolicyHead(h_t)             slot-selection logits [K]
      stop_t       = StopHead(h_t)               stop logit [1]
      class_logits = CrossAttnHead(h_t,          cross-attention over
                                   slot_embeds,  all selected slot embeds
                                   selected_mask)

    slot_embeds = input_proj(slots) is precomputed once per rollout and reused
    across all steps, with selected_mask masking out unselected slots.

    Evidence state E is a no-op pass-through (API compatibility with Evidence head).
    """

    def __init__(
        self,
        slot_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_slots: int,
        num_classes: int,
        use_stop_action: bool = True,
        class_head_num_heads: int = 4,
        class_head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.use_stop_action = use_stop_action
        self.stop_idx = num_slots
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(slot_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gru = nn.GRUCell(embed_dim, hidden_dim)
        self._policy_head = nn.Linear(hidden_dim, num_slots)
        self._stop_head = nn.Linear(hidden_dim, 1) if use_stop_action else None
        self._class_head = SlotCrossAttentionClassHead(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_heads=class_head_num_heads,
            dropout=class_head_dropout,
        )

    # ── initialisation ────────────────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def init_evidence(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compatibility stub; Evidence GRU is not used in this variant."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    # ── per-step computations ─────────────────────────────────────────────────

    def step_hidden(
        self, selected_slots: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(selected_slots)
        h_new = self.gru(x, hidden)
        return h_new, x

    def init_slot_cache(self, batch_size: int, device):
        """No-op: CrossAttn head does not use an evidence slot cache."""
        return None

    def update_slot_cache(self, slot_cache, x: torch.Tensor, active: torch.Tensor):
        """No-op: CrossAttn head does not use an evidence slot cache."""
        return None

    def step_evidence(
        self, delta: torch.Tensor, x: torch.Tensor, E_prev: torch.Tensor,
        slot_cache=None,
    ) -> torch.Tensor:
        """No-op: returns E_prev unchanged (no evidence accumulation)."""
        return E_prev

    # ── heads ─────────────────────────────────────────────────────────────────

    def forward_logits(self, hidden: torch.Tensor, slot_embeds=None) -> torch.Tensor:
        """policy_t ∥ stop_t  →  [B, K] or [B, K+1]."""
        policy_logits = self._policy_head(hidden)
        if self.use_stop_action and self._stop_head is not None:
            stop_logit = self._stop_head(hidden)
            return torch.cat([policy_logits, stop_logit], dim=-1)
        return policy_logits

    def class_head(
        self,
        h: torch.Tensor,
        slot_embeds: torch.Tensor | None,
        selected_mask: torch.Tensor | None,
        E: torch.Tensor | None = None,   # unused; accepted for API compatibility
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        class_logits = CrossAttn(h, slot_embeds[selected_mask]).
        Falls back to h_only when slot_embeds is None.
        """
        if slot_embeds is None or selected_mask is None:
            return self._class_head.h_only(h), None
        return self._class_head(h, slot_embeds, selected_mask)

    def precompute_slot_embeds(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Returns input_proj(slots) = [B, K, embed_dim].
        Computed once per rollout and reused at every step as cross-attn K/V.
        Gradient flows through this tensor during training.
        """
        return self.input_proj(slots)

    def apply_action_mask(
        self, logits: torch.Tensor, selected_mask: torch.Tensor
    ) -> torch.Tensor:
        masked = logits.clone()
        mask_fill = torch.finfo(logits.dtype).min / 2
        masked[:, : self.num_slots] = masked[:, : self.num_slots].masked_fill(
            selected_mask, mask_fill
        )
        return masked


# ═════════════════════════════════════════════════════════════════════════════
# D. Attention-augmented Evidence head  —  E_{t-1} cross-attends over M_t
# ═════════════════════════════════════════════════════════════════════════════

class SlotSelectionPolicyGRUAttn(nn.Module):
    """
    SlotSelectionPolicyGRU + cross-attention over accumulated selected-slot
    embeddings in the Evidence GRU update step.

    At each step t, the previous evidence state E_{t-1} queries the set of
    already-selected slot embeddings M_t = {x_{a_1}, ..., x_{a_t}} (including
    the current step's x_t, written to the cache before this call) via
    cross-attention to produce a context vector ctx_t.

    Evidence update (only change vs. base class):
      query  = EvidenceQueryProj(E_{t-1})              [B, 1, embed_dim]
      M_t    = slot embedding cache (selected only)     [B, t, embed_dim]
      ctx_t  = LayerNorm(CrossAttn(query, M_t, M_t))   [B, embed_dim]

      e_t    = EvidenceMLP([ctx_t ‖ delta_t ‖ x_t])   input_dim = 1536
               (vs. [delta_t ‖ x_t] dim=1024 in the base class)
      E_t    = EvidenceGRU(e_t, E_{t-1})

    Constraint: keys/values are ONLY the already-selected slot embeddings.
    Unselected slots are never accessed.  This is identical to the access
    restriction already satisfied by delta_t and x_t.

    All other components (input_proj, gru, policy_head, stop_head, class_fc)
    are identical to SlotSelectionPolicyGRU.
    """

    def __init__(
        self,
        slot_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_slots: int,
        num_classes: int,
        use_stop_action: bool = True,
        evidence_attn_heads: int = 4,
        **kwargs,   # absorb unused class_head_* kwargs
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.use_stop_action = use_stop_action
        self.stop_idx = num_slots
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # ── shared with base class ───────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(slot_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gru = nn.GRUCell(embed_dim, hidden_dim)
        self._policy_head = nn.Linear(hidden_dim, num_slots)
        self._stop_head = nn.Linear(hidden_dim, 1) if use_stop_action else None
        self._class_fc = nn.Linear(hidden_dim, num_classes)
        self._evidence_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # ── new: cross-attention over selected-slot cache ───────────────────
        # query = Linear(E_{t-1}) → [B, 1, embed_dim]
        self._evidence_query_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        # K/V = M_t (selected slot embeddings only)
        self._evidence_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=evidence_attn_heads,
            batch_first=True,
        )
        self._evidence_attn_norm = nn.LayerNorm(embed_dim)

        # ── EvidenceMLP: [ctx_t ‖ delta_t ‖ x_t]  (dim 1536 vs 1024) ──────
        self._evidence_mlp = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim + embed_dim, hidden_dim),  # 1536 → 512
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    # ── initialisation ────────────────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def init_evidence(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def init_slot_cache(self, batch_size: int, device: torch.device):
        """Initialise an empty slot-embedding cache.

        Returns:
            (cache [B, K, embed_dim], count [B, long])
            cache is zero-filled; count tracks how many slots have been written.
        """
        cache = torch.zeros(batch_size, self.num_slots, self.embed_dim, device=device)
        count = torch.zeros(batch_size, device=device, dtype=torch.long)
        return (cache, count)

    def update_slot_cache(
        self,
        slot_cache: tuple,
        x: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple:
        """Append x_t to the cache for each active item.

        Args:
            slot_cache: (cache [B, K, embed_dim], count [B])
            x:          [B, embed_dim]  current slot embedding
            active:     [B] bool — which items to update
        Returns:
            Updated (cache, count) tuple.
        """
        cache, count = slot_cache
        new_cache = cache.clone()
        new_count = count.clone()
        b_idx = active.nonzero(as_tuple=True)[0]
        if b_idx.numel() > 0:
            pos = count[b_idx]                      # write positions [num_active]
            valid = pos < self.num_slots            # guard against overflow
            b_v = b_idx[valid]
            p_v = pos[valid]
            new_cache[b_v, p_v] = x[b_v]
            new_count[b_v] = p_v + 1
        return (new_cache, new_count)

    # ── per-step computations ─────────────────────────────────────────────────

    def step_hidden(
        self, selected_slots: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(selected_slots)
        h_new = self.gru(x, hidden)
        return h_new, x

    def step_evidence(
        self,
        delta: torch.Tensor,
        x: torch.Tensor,
        E_prev: torch.Tensor,
        slot_cache=None,
    ) -> torch.Tensor:
        """Evidence update with cross-attention over already-selected slots.

        Args:
            delta:      [B, hidden_dim]  delta_t = h_t - h_{t-1}
            x:          [B, embed_dim]   x_t = encode(current slot)
            E_prev:     [B, hidden_dim]  E_{t-1}
            slot_cache: (cache [B, K, embed_dim], count [B])
                        Cache must be updated (x_t appended) BEFORE this call.
                        If None, ctx_t is zeroed (graceful degradation).
        Returns:
            E_new: [B, hidden_dim]
        """
        bsz = x.size(0)
        if slot_cache is None:
            ctx_t = torch.zeros(bsz, self.embed_dim, device=x.device, dtype=x.dtype)
        else:
            cache, count = slot_cache          # [B, K, embed_dim], [B]
            K = self.num_slots

            # key_padding_mask[b, i] = True means "ignore position i"
            # Use count.clamp(min=1) so that items with count=0 attend to
            # position 0 (zeros) instead of producing NaN from all-masked attn.
            count_safe = count.clamp(min=1)
            positions = torch.arange(K, device=x.device).unsqueeze(0)  # [1, K]
            key_padding_mask = positions >= count_safe.unsqueeze(1)     # [B, K]

            # query from previous evidence state
            query = self._evidence_query_proj(E_prev).unsqueeze(1)  # [B, 1, embed_dim]
            attn_out, _ = self._evidence_cross_attn(
                query=query,
                key=cache,
                value=cache,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            ctx_t = self._evidence_attn_norm(attn_out.squeeze(1))   # [B, embed_dim]

        e = self._evidence_mlp(torch.cat([ctx_t, delta, x], dim=-1))  # [B, hidden_dim]
        return self._evidence_gru(e, E_prev)

    # ── heads ─────────────────────────────────────────────────────────────────

    def forward_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        policy_logits = self._policy_head(hidden)
        if self.use_stop_action and self._stop_head is not None:
            stop_logit = self._stop_head(hidden)
            return torch.cat([policy_logits, stop_logit], dim=-1)
        return policy_logits

    def class_head(
        self,
        h: torch.Tensor,
        slot_embeds=None,
        selected_mask=None,
        E: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        context = E if E is not None else h
        return self._class_fc(context), None

    def precompute_slot_embeds(self, slots: torch.Tensor) -> None:
        return None

    def apply_action_mask(
        self, logits: torch.Tensor, selected_mask: torch.Tensor
    ) -> torch.Tensor:
        masked = logits.clone()
        mask_fill = torch.finfo(logits.dtype).min / 2
        masked[:, : self.num_slots] = masked[:, : self.num_slots].masked_fill(
            selected_mask, mask_fill
        )
        return masked


# ═════════════════════════════════════════════════════════════════════════════
# E. Global-Attn head  —  class_head attends over ALL K slots (h0 = 0)
# ═════════════════════════════════════════════════════════════════════════════

class SlotSelectionPolicyGlobalAttn(nn.Module):
    """
    GRU action selection (h0 = 0) + cross-attention classification over ALL K slots.

    Architecture (per step t)
    ─────────────────────────
      x_t          = input_proj(slot_t)       SlotEncoder
      h_t          = GRU(x_t, h_{t-1})        main GRU  (h_0 = 0, never warm-started)
      policy_t     = PolicyHead(h_t)           slot-selection logits [K]
      stop_t       = StopHead(h_t)             stop logit [1]
      class_logits = CrossAttnHead(h_t,        cross-attention over ALL K slots
                                   all_embeds, (no selected_mask applied)
                                   all_true)

    Unlike SlotSelectionPolicyCrossAttn (which masks unselected slots),
    the classifier here always has a full view of every slot embedding.
    The GRU hidden state encodes the sequential selection context;
    the cross-attention head lets the classifier directly weight all K slots.

    selected_mask is used only for action masking (no repeat selections), NOT
    for the classification cross-attention.

    Evidence state E is a no-op pass-through (API compatibility).
    """

    def __init__(
        self,
        slot_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_slots: int,
        num_classes: int,
        use_stop_action: bool = True,
        class_head_num_heads: int = 4,
        class_head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.use_stop_action = use_stop_action
        self.stop_idx = num_slots
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(slot_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gru = nn.GRUCell(embed_dim, hidden_dim)
        self._policy_head = nn.Linear(hidden_dim, num_slots)
        self._stop_head = nn.Linear(hidden_dim, 1) if use_stop_action else None
        self._class_head = SlotCrossAttentionClassHead(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_heads=class_head_num_heads,
            dropout=class_head_dropout,
        )

    # ── initialisation ────────────────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def init_evidence(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def init_slot_cache(self, batch_size: int, device) -> None:
        return None

    def update_slot_cache(self, slot_cache, x: torch.Tensor, active: torch.Tensor) -> None:
        return None

    # ── per-step computations ─────────────────────────────────────────────────

    def step_hidden(
        self, selected_slots: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(selected_slots)
        h_new = self.gru(x, hidden)
        return h_new, x

    def step_evidence(
        self, delta: torch.Tensor, x: torch.Tensor, E_prev: torch.Tensor,
        slot_cache=None,
    ) -> torch.Tensor:
        return E_prev

    # ── heads ─────────────────────────────────────────────────────────────────

    def forward_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        policy_logits = self._policy_head(hidden)
        if self.use_stop_action and self._stop_head is not None:
            stop_logit = self._stop_head(hidden)
            return torch.cat([policy_logits, stop_logit], dim=-1)
        return policy_logits

    def class_head(
        self,
        h: torch.Tensor,
        slot_embeds: torch.Tensor | None,
        selected_mask: torch.Tensor | None = None,  # ignored for classification
        E: torch.Tensor | None = None,               # unused; API compat
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        class_logits = CrossAttn(h, ALL slot_embeds).
        selected_mask is intentionally ignored here: classification attends to
        every slot regardless of which slots the action GRU has visited so far.
        """
        if slot_embeds is None:
            return self._class_head.h_only(h), None
        # All-true mask → attend to every slot (key_padding_mask = all False)
        all_mask = torch.ones(
            slot_embeds.shape[:2], dtype=torch.bool, device=slot_embeds.device
        )
        return self._class_head(h, slot_embeds, all_mask)

    def precompute_slot_embeds(self, slots: torch.Tensor) -> torch.Tensor:
        """Returns input_proj(slots) [B, K, embed_dim]; computed once per rollout."""
        return self.input_proj(slots)

    def apply_action_mask(
        self, logits: torch.Tensor, selected_mask: torch.Tensor
    ) -> torch.Tensor:
        masked = logits.clone()
        mask_fill = torch.finfo(logits.dtype).min / 2
        masked[:, : self.num_slots] = masked[:, : self.num_slots].masked_fill(
            selected_mask, mask_fill
        )
        return masked


# ═════════════════════════════════════════════════════════════════════════════
# F. Attention-select head  —  action via attn(h, ALL slots); classify via SELECTED slots
# ═════════════════════════════════════════════════════════════════════════════

class SlotSelectionPolicyAttnSelect(nn.Module):
    """
    Sequential slot-selection policy separating global awareness from local evidence.

    Architecture (per step t)
    ─────────────────────────
      x_t          = input_proj(slot_t)         SlotEncoder
      h_t          = GRU(x_t, h_{t-1})          main GRU  (h_0 = 0)

      [ACTION — global view]
      q_t          = action_query_proj(h_t)      [B, embed_dim]
      scores_t     = q_t @ E_all^T / √d          [B, K]  dot-product with ALL slots
      mask → argmax → select next slot

      [CLASSIFICATION — restricted view]
      class_logits = CrossAttnHead(h_t,          cross-attention over
                                   E_selected)   ONLY already-selected slots

    Separation principle:
      The agent can survey the full scene when deciding what to look at next,
      but its classification conclusion is only based on slots it has explicitly
      examined.  This forces genuine evidence accumulation: confidence grows
      only as more relevant slots are selected.

    Evidence state E is a no-op pass-through (API compatibility).
    """

    def __init__(
        self,
        slot_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_slots: int,
        num_classes: int,
        use_stop_action: bool = True,
        class_head_num_heads: int = 4,
        class_head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.use_stop_action = use_stop_action
        self.stop_idx = num_slots
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(slot_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gru = nn.GRUCell(embed_dim, hidden_dim)

        # Action head: project h to embed_dim for dot-product with slot embeddings
        self._action_query_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self._stop_head = nn.Linear(hidden_dim, 1) if use_stop_action else None

        # Classification head: cross-attention over SELECTED slots only
        self._class_head = SlotCrossAttentionClassHead(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_heads=class_head_num_heads,
            dropout=class_head_dropout,
        )

        # Evidence accumulation: gated running sum of selected x_t embeddings.
        # gate(delta_t) controls how much x_t is folded into E.
        self._evidence_gate = nn.Linear(hidden_dim, embed_dim)
        # Project accumulated E (embed_dim) → hidden_dim to augment cross-attn query.
        self._evidence_merge = nn.Linear(embed_dim, hidden_dim, bias=False)

    # ── initialisation ────────────────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def init_evidence(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # E lives in embed_dim space (gated running sum of x_t vectors).
        return torch.zeros(batch_size, self.embed_dim, device=device)

    def init_slot_cache(self, batch_size: int, device) -> None:
        return None

    def update_slot_cache(self, slot_cache, x: torch.Tensor, active: torch.Tensor) -> None:
        return None

    # ── per-step computations ─────────────────────────────────────────────────

    def step_hidden(
        self, selected_slots: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(selected_slots)
        h_new = self.gru(x, hidden)
        return h_new, x

    def step_evidence(
        self, delta: torch.Tensor, x: torch.Tensor, E_prev: torch.Tensor,
        slot_cache=None,
    ) -> torch.Tensor:
        # Gated running sum: gate driven by GRU delta (how much h changed this step).
        gate = torch.sigmoid(self._evidence_gate(delta))  # [B, embed_dim]
        return E_prev + gate * x

    # ── heads ─────────────────────────────────────────────────────────────────

    def forward_logits(self, hidden: torch.Tensor, slot_embeds=None) -> torch.Tensor:
        """
        Action logits via dot-product attention: proj(h) @ E_all^T / √d.
        Falls back to a linear head if slot_embeds is not provided.
        Stop logit appended when use_stop_action is True.
        """
        if slot_embeds is not None:
            q = self._action_query_proj(hidden)                    # [B, embed_dim]
            scores = (q.unsqueeze(1) @ slot_embeds.transpose(-1, -2)).squeeze(1)
            scores = scores / math.sqrt(self.embed_dim)            # [B, K]
        else:
            # Fallback: linear projection (used only if slot_embeds is unavailable)
            scores = self._action_query_proj(hidden) @ self._action_query_proj.weight.T
            scores = scores[:, :self.num_slots]

        if self.use_stop_action and self._stop_head is not None:
            stop_logit = self._stop_head(hidden)                   # [B, 1]
            return torch.cat([scores, stop_logit], dim=-1)         # [B, K+1]
        return scores

    def class_head(
        self,
        h: torch.Tensor,
        slot_embeds: torch.Tensor | None,
        selected_mask: torch.Tensor | None = None,
        E: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        class_logits = CrossAttn(h_aug, E_selected).
        Query is h augmented by the accumulated evidence E via _evidence_merge.
        Falls back to h_only when no slots have been selected yet.
        """
        h_aug = h + self._evidence_merge(E) if E is not None else h
        if slot_embeds is None or selected_mask is None or not selected_mask.any():
            return self._class_head.h_only(h_aug), None
        return self._class_head(h_aug, slot_embeds, selected_mask)

    def precompute_slot_embeds(self, slots: torch.Tensor) -> torch.Tensor:
        """Returns input_proj(slots) [B, K, embed_dim]; computed once per rollout."""
        return self.input_proj(slots)

    def apply_action_mask(
        self, logits: torch.Tensor, selected_mask: torch.Tensor
    ) -> torch.Tensor:
        masked = logits.clone()
        mask_fill = torch.finfo(logits.dtype).min / 2
        masked[:, : self.num_slots] = masked[:, : self.num_slots].masked_fill(
            selected_mask, mask_fill
        )
        return masked


# ═════════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════════

def build_policy(
    head_type: str,
    slot_dim: int,
    embed_dim: int,
    hidden_dim: int,
    num_slots: int,
    num_classes: int,
    use_stop_action: bool = True,
    class_head_num_heads: int = 4,
    class_head_dropout: float = 0.1,
) -> nn.Module:
    """
    Factory to instantiate a slot-selection policy by head type.

    Args:
        head_type: one of "linear", "crossattn", "evidence"
        slot_dim:  dimension of raw slot vectors
        embed_dim: slot encoder output dimension
        hidden_dim: GRU hidden dimension
        num_slots: number of slots K
        num_classes: classification output dimension
        use_stop_action: whether to include a stop action (usually True)
        class_head_num_heads: cross-attention heads (crossattn only)
        class_head_dropout:   dropout in cross-attention head (crossattn only)

    Returns:
        Instantiated policy (not yet moved to device).
    """
    ht = head_type.strip().lower()
    if ht == "linear":
        return SlotSelectionPolicyLinear(
            slot_dim=slot_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            num_classes=num_classes,
            use_stop_action=use_stop_action,
        )
    elif ht == "crossattn":
        return SlotSelectionPolicyCrossAttn(
            slot_dim=slot_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            num_classes=num_classes,
            use_stop_action=use_stop_action,
            class_head_num_heads=class_head_num_heads,
            class_head_dropout=class_head_dropout,
        )
    elif ht == "evidence":
        from src.explanation.rnn_selector import SlotSelectionPolicyGRU
        return SlotSelectionPolicyGRU(
            slot_dim=slot_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            num_classes=num_classes,
            use_stop_action=use_stop_action,
        )
    elif ht == "attn_evidence":
        return SlotSelectionPolicyGRUAttn(
            slot_dim=slot_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            num_classes=num_classes,
            use_stop_action=use_stop_action,
            evidence_attn_heads=class_head_num_heads,
        )
    elif ht == "all_slots_attn":
        return SlotSelectionPolicyGlobalAttn(
            slot_dim=slot_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            num_classes=num_classes,
            use_stop_action=use_stop_action,
            class_head_num_heads=class_head_num_heads,
            class_head_dropout=class_head_dropout,
        )
    elif ht == "attn_select":
        return SlotSelectionPolicyAttnSelect(
            slot_dim=slot_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_slots=num_slots,
            num_classes=num_classes,
            use_stop_action=use_stop_action,
            class_head_num_heads=class_head_num_heads,
            class_head_dropout=class_head_dropout,
        )
    else:
        raise ValueError(
            f"Unknown head_type '{head_type}'. "
            "Choose from: 'linear', 'crossattn', 'evidence', 'attn_evidence', "
            "'all_slots_attn', 'attn_select'."
        )
