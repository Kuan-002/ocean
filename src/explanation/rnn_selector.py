from __future__ import annotations

import torch
import torch.nn as nn


class SlotSelectionPolicyGRU(nn.Module):
    """
    Inductive sequential subset policy:
    - GRU hidden state summarizes selected slots; STOP ends without a slot GRU step.
    - action_head: logits over K slots + STOP (index K).
    - class_head: logits over C classes from current hidden state (subset prediction).
    """

    def __init__(
        self, slot_dim: int, embed_dim: int, hidden_dim: int, num_slots: int, num_classes: int
    ):
        super().__init__()
        self.num_slots = num_slots
        self.stop_idx = num_slots
        self.num_classes = num_classes
        self.input_proj = nn.Sequential(
            nn.Linear(slot_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gru = nn.GRUCell(embed_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_slots + 1)
        self.class_head = nn.Linear(hidden_dim, num_classes)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

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
        STOP action (last index) remains available.
        """
        masked = logits.clone()
        # dtype-aware large negative (fp16-safe); avoids -inf + softmax/Categorical NaNs on CUDA
        mask_fill = torch.finfo(logits.dtype).min / 2
        masked[:, : self.num_slots] = masked[:, : self.num_slots].masked_fill(
            selected_mask, mask_fill
        )
        return masked
