from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotCrossAttentionClassHead(nn.Module):
    """
    Classification head using cross-attention over selected slot embeddings.

    Replaces the bottlenecked Linear(h → classes) with direct access to all
    selected slot features:

      Query  : linear projection of GRU hidden state h
      Keys/Values : input_proj(selected slots)  [precomputed once per rollout]

    Advantages over a linear head
    ─────────────────────────────
    • No information bottleneck — each selected slot contributes directly.
    • Multi-head attention captures slot interactions (critical for medical
      datasets where combinations of findings drive diagnosis).
    • attn_weights [B, K] is a per-slot contribution score usable as CoT
      evidence: "slot 3 (region X) drove 60% of the diagnosis confidence."

    Interface
    ─────────
    forward(h, slot_embeds, selected_mask) → (logits, attn_weights | None)
      h            : [B, hidden_dim]  GRU hidden state
      slot_embeds  : [B, K, embed_dim]  policy.precompute_slot_embeds(slots)
      selected_mask: [B, K] bool, True = slot selected
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Project GRU hidden state to the attention query space
        self.query_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        # Multi-head cross-attention: Q from GRU, K/V from slot embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        # Classifier: concat(h, attn_context) → num_classes
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        # Fallback used before the first slot is selected
        self.h_only = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        h: torch.Tensor,              # [B, hidden_dim]
        slot_embeds: torch.Tensor,    # [B, K, embed_dim]
        selected_mask: torch.Tensor,  # [B, K] bool — True = selected
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (logits [B, C], attn_weights [B, K] or None)."""
        if not selected_mask.any():
            return self.h_only(h), None

        query = self.query_proj(h).unsqueeze(1)          # [B, 1, embed_dim]
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=slot_embeds,
            value=slot_embeds,
            key_padding_mask=~selected_mask,             # True = ignore (unselected)
        )
        # attn_out: [B, 1, embed_dim]; attn_weights: [B, 1, K]
        attn_out = self.attn_norm(attn_out.squeeze(1))   # [B, embed_dim]
        attn_weights = attn_weights.squeeze(1)            # [B, K]
        attn_weights = attn_weights * selected_mask.float()  # zero unselected

        combined = torch.cat([h, attn_out], dim=-1)
        return self.mlp(combined), attn_weights


class SlotSelectionPolicyGRU(nn.Module):
    """
    Inductive sequential subset policy:
    - GRU hidden state summarizes selected slots (drives action selection).
    - action_head: logits over K slots, optionally plus STOP (index K).
    - class_head: linear(hidden_dim → num_classes), slot-order-independent.

    Usage
    ─────
    slot_embeds = policy.precompute_slot_embeds(slots)  # returns None (unused)
    logits, _ = policy.class_head(h, slot_embeds, selected_mask)
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
    ):
        super().__init__()
        self.num_slots = num_slots
        self.use_stop_action = use_stop_action
        self.stop_idx = num_slots
        self.num_classes = num_classes
        self.input_proj = nn.Sequential(
            nn.Linear(slot_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gru = nn.GRUCell(embed_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_slots + int(use_stop_action))
        # Simple linear classifier on GRU hidden state.  The cross-attention
        # variant (SlotCrossAttentionClassHead) was tried but made classification
        # too easy at GRPO start, collapsing the advantage signal.
        self._class_fc = nn.Linear(hidden_dim, num_classes)
        # Learnable projection for h0 init: allows embed_dim != hidden_dim
        self.h0_proj = (
            nn.Linear(embed_dim, hidden_dim, bias=False)
            if embed_dim != hidden_dim
            else nn.Identity()
        )

    # Unified call interface: (h, slot_embeds, selected_mask) → (logits, None).
    # slot_embeds and selected_mask are accepted but ignored by the linear head,
    # keeping all callers in rnn_selector_training.py unchanged.
    def class_head(
        self,
        h: torch.Tensor,
        slot_embeds,
        selected_mask,
    ) -> tuple[torch.Tensor, None]:
        return self._class_fc(h), None

    def precompute_slot_embeds(self, slots: torch.Tensor) -> None:
        """No-op: linear class_head does not use slot embeddings."""
        return None

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def init_hidden_from_slots(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Initialize h0 from the mean encoded slot, giving the first action global
        image context. A learnable projection maps embed_dim → hidden_dim so
        the two dimensions are no longer forced to be equal.
        """
        slot_ctx = self.input_proj(slots).mean(dim=1)  # [B, embed_dim]
        return self.h0_proj(slot_ctx)  # [B, hidden_dim]

    def forward_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.action_head(hidden)

    def step_hidden(self, selected_slots: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # selected_slots: [B, D]
        x = self.input_proj(selected_slots)
        return self.gru(x, hidden)

    def apply_action_mask(self, logits: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        """
        selected_mask: [B, K], True means slot already selected.
        Mask those slot logits to a large negative finite value (not -inf) so masked
        softmax / Categorical stay numerically stable on CUDA.
        STOP action, when enabled, remains available.
        """
        masked = logits.clone()
        # dtype-aware large negative (fp16-safe); avoids -inf + softmax/Categorical NaNs on CUDA
        mask_fill = torch.finfo(logits.dtype).min / 2
        masked[:, : self.num_slots] = masked[:, : self.num_slots].masked_fill(
            selected_mask, mask_fill
        )
        return masked
