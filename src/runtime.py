from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from src.slot_autoencoder import SlotAutoencoder


def seed_all(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def reconstruct_autoencoder(
    checkpoint: str | Path,
    config,
) -> tuple[SlotAutoencoder, float]:
    model = SlotAutoencoder(config)
    best_loss = model.load(str(checkpoint))
    return model, float(best_loss)
