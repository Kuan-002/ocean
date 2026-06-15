import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from src.slot_attention import SlotAttention
from torch.optim import AdamW

class SlotAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_shape=(3, config.dataset_image_resolution, config.dataset_image_resolution)
        enc_act = nn.ReLU()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_shape[0], config.width, 5, padding=2),
            enc_act,
            *[nn.Conv2d(config.width, config.width, 5, padding=2), enc_act] * 3,
            PositionEmbed(config.width, in_shape[1:]),
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.ReLU(),
            nn.Linear(config.width, config.width),
        )

        self.slot_attention = SlotAttention(
                input_dim=config.width,
                num_slots=config.num_slots,
                slot_dim=config.slot_dim,
                routing_iters=config.routing_iters,
                hidden_dim=config.width,
            )

        self.slot_grid = (in_shape[1] // 16, in_shape[2] // 16)
        dec_act = nn.LeakyReLU()
        self.decoder = nn.Sequential(
            PositionEmbed(config.slot_dim, self.slot_grid),
            *[
                nn.ConvTranspose2d(
                    config.width, config.width, 5, stride=2, padding=2, output_padding=1
                ),
                dec_act,
            ]
            * 4,
            nn.ConvTranspose2d(config.width, config.width, 5, stride=1, padding=2),
            dec_act,
            nn.ConvTranspose2d(
                config.width, in_shape[0] + 1, 3, stride=1, padding=1
            ),  # 4 output channels
        )

        self.optimiser = AdamW(
            self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        
        if config.use_gpu:
            self.cuda()

    def forward_slots_only(
        self, x: Tensor, slot_init_noise: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Encode images to slot vectors without running the decoder (plan: reuse for explanation)."""
        x = self.encoder(x)
        x = self.mlp(x.reshape(*x.shape[:2], -1).permute(0, 2, 1))
        slots, attn = self.slot_attention(x, slot_init_noise=slot_init_noise)
        return slots, attn

    def forward(self, x: Tensor, slot_init_noise: Optional[Tensor] = None):
        # b: batch_size, c: channels, h: height, w: width, d: out_channels
        b, c, h, w = x.shape
        # (b, d, h, w)
        enc = self.encoder(x)
        # (b, h*w, d)
        enc = self.mlp(enc.reshape(*enc.shape[:2], -1).permute(0, 2, 1))  # flatten img

        # (b, num_slots, slot_dim)
        slots, attn = self.slot_attention(enc, slot_init_noise=slot_init_noise)

        return self._decode_slots_to_recon(b, c, h, w, slots, attn)

    def _decode_slots_to_recon(
        self, b: int, c: int, h: int, w: int, slots: Tensor, attn: Tensor
    ):
        # (b*num_slots, slot_dim, init_h, init_w)
        x = slots.view(-1, slots.shape[-1])[:, :, None, None]
        x = x.repeat(1, 1, *self.slot_grid)
        # (b*num_slots, c + 1, h, w)
        x = self.decoder(x)

        # (b, num_slots, c + 1, h, w)
        x = x.view(b, -1, c + 1, h, w)
        # (b, num_slots, c, h, w), (b, num_slots, 1, h, w)
        recons, masks = torch.split(x, [c, 1], dim=2)
        masks = masks.softmax(dim=1)
        # (b, c, h, w)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, slots, attn
    
    def loss(self, x: Tensor, recon_combined: Tensor):
        loss = torch.mean((x - recon_combined) ** 2)
        return loss
    
    def training_step(self, loss: Tensor):
        loss.backward()
        self.optimiser.step()
        return loss.detach()
    
    def save(self, path: str, best_loss: float):
        save_dict = {
            "best_loss": best_loss,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimiser.state_dict(),
        }
        torch.save(save_dict, path)

    def load(self, path: str):
        device = next(self.parameters()).device
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["best_loss"]

class PositionEmbed(nn.Module):
    def __init__(self, out_channels: int, resolution: Tuple[int, int]):
        super().__init__()
        # (1, height, width, 4)
        self.register_buffer("grid", self.build_grid(resolution))
        self.mlp = nn.Linear(4, out_channels)  # 4 for (x, y, 1-x, 1-y)

    def forward(self, x: Tensor):
        # (1, height, width, out_channels)
        grid = self.mlp(self.grid)
        # (batch_size, out_channels, height, width)
        return x + grid.permute(0, 3, 1, 2)

    def build_grid(self, resolution: Tuple[int, int]) -> Tensor:
        xy = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        xx, yy = torch.meshgrid(xy, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        grid = grid.unsqueeze(0)
        return torch.cat([grid, 1.0 - grid], dim=-1)
