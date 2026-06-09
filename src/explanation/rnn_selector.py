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
    Inductive sequential subset policy with Recurrent Evidence GRU.

    Architecture (per step t)
    ─────────────────────────
      x_t         = encode(slot_t)                  SlotEncoder (input_proj)
      h_t         = GRU(x_t, h_{t-1})               main GRU hidden state
      delta_t     = h_t - h_{t-1}                   change in hidden state
      e_t         = EvidenceMLP([delta_t, x_t])      per-step evidence signal
      E_t         = EvidenceGRU(e_t, E_{t-1})        accumulated evidence state
      policy_t    = PolicyHead(h_t)                  slot-selection logits [K]
      stop_t      = StopHead(h_t)                    stop logit [1]
      class_logits= Classifier(E_t)                  classification from evidence

    Initialisation: h_0 = 0,  E_0 = 0  (always zero — no warm-start from slots).

    Usage
    ─────
    h = policy.init_hidden(bsz, device)
    E = policy.init_evidence(bsz, device)
    for each step:
        h_new, x = policy.step_hidden(slot, h)
        h = where(active, h_new, h)
        delta = h - h_prev
        E_new = policy.step_evidence(delta, x, E)
        E = where(active, E_new, E)
        logits, _ = policy.class_head(h, None, mask, E=E)
    """

    def __init__(
        self,
        slot_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_slots: int,
        num_classes: int,
        use_stop_action: bool = True,
        class_head_num_heads: int = 4,   # kept for API compat, unused
        class_head_dropout: float = 0.1, # kept for API compat, unused
    ):
        super().__init__()
        self.num_slots = num_slots
        self.use_stop_action = use_stop_action
        self.stop_idx = num_slots
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # x_t = encode(slot_t)
        self.input_proj = nn.Sequential(
            nn.Linear(slot_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # h_t = GRU(x_t, h_{t-1})
        self.gru = nn.GRUCell(embed_dim, hidden_dim)

        # e_t = EvidenceMLP([delta_t, x_t])
        self._evidence_mlp = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # E_t = EvidenceGRU(e_t, E_{t-1})
        self._evidence_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # policy_t = PolicyHead(h_t)
        self._policy_head = nn.Linear(hidden_dim, num_slots)

        # stop_t = StopHead(h_t)
        self._stop_head = nn.Linear(hidden_dim, 1) if use_stop_action else None

        # class_logits = Classifier(E_t)
        self._class_fc = nn.Linear(hidden_dim, num_classes)

    # ── initialisation ────────────────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialised main GRU state [B, hidden_dim]."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def init_evidence(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialised evidence GRU state [B, hidden_dim]."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    # ── per-step computations ─────────────────────────────────────────────────

    def step_hidden(
        self, selected_slots: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one step of the main GRU.

        Args:
            selected_slots: [B, slot_dim]
            hidden:         [B, hidden_dim]  h_{t-1}
        Returns:
            h_new: [B, hidden_dim]  h_t
            x:     [B, embed_dim]   x_t = encode(slot_t)  (needed for EvidenceMLP)
        """
        x = self.input_proj(selected_slots)   # x_t
        h_new = self.gru(x, hidden)            # h_t
        return h_new, x

    def init_slot_cache(self, batch_size: int, device: torch.device):
        """No-op: base Evidence GRU does not use a slot cache. Returns None."""
        return None

    def update_slot_cache(self, slot_cache, x: torch.Tensor, active: torch.Tensor):
        """No-op: base Evidence GRU does not use a slot cache. Returns None."""
        return None

    def step_evidence(
        self,
        delta: torch.Tensor,
        x: torch.Tensor,
        E_prev: torch.Tensor,
        slot_cache=None,
    ) -> torch.Tensor:
        """Run one step of the Evidence GRU.

        Args:
            delta:      [B, hidden_dim]  delta_t = h_t - h_{t-1}
            x:          [B, embed_dim]   x_t     = encode(slot_t)
            E_prev:     [B, hidden_dim]  E_{t-1}
            slot_cache: ignored in the base class (accepted for API compat with
                        SlotSelectionPolicyGRUAttn which uses a cross-attention cache)
        Returns:
            E_new:  [B, hidden_dim]  E_t = EvidenceGRU(EvidenceMLP([delta, x]), E_{t-1})
        """
        e = self._evidence_mlp(torch.cat([delta, x], dim=-1))  # e_t
        return self._evidence_gru(e, E_prev)                    # E_t

    # ── heads ─────────────────────────────────────────────────────────────────

    def forward_logits(self, hidden: torch.Tensor, slot_embeds=None) -> torch.Tensor:
        """policy_t ∥ stop_t  →  [B, K] or [B, K+1]."""
        policy_logits = self._policy_head(hidden)             # [B, K]
        if self.use_stop_action and self._stop_head is not None:
            stop_logit = self._stop_head(hidden)              # [B, 1]
            return torch.cat([policy_logits, stop_logit], dim=-1)  # [B, K+1]
        return policy_logits

    def class_head(
        self,
        h: torch.Tensor,
        slot_embeds,          # accepted but unused (API compat)
        selected_mask,        # accepted but unused (API compat)
        E: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """class_logits = Classifier(E_t).  Falls back to h when E is None."""
        context = E if E is not None else h
        return self._class_fc(context), None

    def precompute_slot_embeds(self, slots: torch.Tensor) -> None:
        """No-op: Classifier does not use slot embeddings directly."""
        return None

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
