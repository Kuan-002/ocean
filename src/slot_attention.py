import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class SlotAttention(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        num_slots: int = 7,
        slot_dim: int = 64,
        routing_iters: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.routing_iters = routing_iters

        self.ln_inputs = nn.LayerNorm(input_dim)
        self.ln_slots = nn.LayerNorm(self.slot_dim)

        self.W_q = nn.Parameter(torch.rand(self.slot_dim, self.slot_dim))
        self.W_k = nn.Parameter(torch.rand(input_dim, self.slot_dim))
        self.W_v = nn.Parameter(torch.rand(input_dim, self.slot_dim))
        self.loc = nn.Parameter(torch.zeros(1, self.slot_dim))
        self.logscale = nn.Parameter(torch.zeros(1, self.slot_dim))

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.slot_dim),
        )

    def forward(
        self,
        x: Tensor,
        num_slots: Optional[int] = None,
        slot_init_noise: Optional[Tensor] = None,
    ):
        # b: batch_size, n: num_inputs, c: input_dim, K: num_slots, d: slot_dim
        b = x.shape[0]
        # (b, n, c)
        x = self.ln_inputs(x)
        # (b, n, d)
        k = torch.einsum("bnc,cd->bnd", x, self.W_k)
        v = torch.einsum("bnc,cd->bnd", x, self.W_v)
        # (b, k, d)
        K = num_slots if num_slots is not None else self.num_slots
        if slot_init_noise is not None:
            if slot_init_noise.shape != (b, K, self.slot_dim):
                raise ValueError(
                    f"slot_init_noise expected {(b, K, self.slot_dim)}, got {tuple(slot_init_noise.shape)}"
                )
            noise = slot_init_noise.to(device=x.device, dtype=x.dtype)
        else:
            noise = torch.randn(b, K, self.slot_dim, device=x.device, dtype=x.dtype)
        slots = self.loc + self.logscale.exp() * noise

        for _ in range(self.routing_iters):
            slots_prev = slots
            slots = self.ln_slots(slots)
            # (b, k, d)
            q = torch.einsum("bkd,dd->bkd", slots, self.W_q)
            q = q * self.slot_dim**-0.5
            # (b, k, n)
            agreement = torch.einsum("bkd,bdn->bkn", q, k.transpose(-2, -1))
            attn = agreement.softmax(dim=1) + 1e-8
            mean_attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean
            # (b, k, d)
            updates = torch.einsum("bkn,bnd->bkd", mean_attn, v)
            # (b*k, d)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            )
            # (b, k, d)
            slots = slots.reshape(b, -1, self.slot_dim)
            slots = slots + self.mlp(slots)
        return slots, attn
