from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.classification.deepsets import DeepSetsClassifier, DeepSetsSubsetEvaluator, kl_distill_loss
from src.classification.subset_sampling import stratified_subset_masks
from src.slot_autoencoder import SlotAutoencoder


def make_per_image_slot_init_noise(
    images: torch.Tensor,
    num_slots: int,
    slot_dim: int,
    base_seed: int,
) -> torch.Tensor:
    """Reproducible SlotAttention init noise: same image tensor -> same noise (per-row seeds)."""
    b, _, _, _ = images.shape
    device = images.device
    dtype = images.dtype
    noise = torch.empty(b, num_slots, slot_dim, device=device, dtype=dtype)
    flat = images.detach().reshape(b, -1)
    for i in range(b):
        g = torch.Generator(device=device)
        s = int(flat[i].sum().item() * 1e6) % (2**31 - 2)
        g.manual_seed(int((base_seed + s + i * 100_003) % (2**31 - 1)))
        noise[i] = torch.randn(num_slots, slot_dim, generator=g, device=device, dtype=dtype)
    return noise


@torch.no_grad()
def batch_slots(
    sa: SlotAutoencoder,
    images: torch.Tensor,
    device: torch.device,
    *,
    sa_deterministic: bool = False,
    sa_noise_seed: int = 0,
    num_slots: int = 11,
    slot_dim: int = 64,
) -> torch.Tensor:
    sa.eval()
    x = images.to(device)
    slot_noise = None
    if sa_deterministic:
        slot_noise = make_per_image_slot_init_noise(x, num_slots, slot_dim, sa_noise_seed)
    slots, _ = sa.forward_slots_only(x, slot_init_noise=slot_noise)
    return slots


def train_classifier_epoch(
    sa: SlotAutoencoder,
    clf: DeepSetsClassifier,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    clf.train()
    total = 0.0
    n = 0
    for images, _, labels in loader:
        labels = labels.to(device)
        slots = batch_slots(sa, images, device)
        logits = clf(slots)
        loss = F.cross_entropy(logits, labels)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        total += loss.item() * labels.size(0)
        n += labels.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_classifier_accuracy(
    sa: SlotAutoencoder,
    clf: DeepSetsClassifier,
    loader: DataLoader,
    device: torch.device,
) -> float:
    clf.eval()
    correct = 0
    total = 0
    for images, _, labels in loader:
        labels = labels.to(device)
        slots = batch_slots(sa, images, device)
        pred = clf(slots).argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def train_subset_evaluator_epoch(
    sa: SlotAutoencoder,
    teacher: DeepSetsClassifier,
    student: DeepSetsSubsetEvaluator,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    num_slots: int,
    temperature: float,
) -> float:
    teacher.eval()
    student.train()
    loss_acc = 0.0
    n = 0
    for images, _, _ in loader:
        b = images.size(0)
        slots = batch_slots(sa, images, device)
        with torch.no_grad():
            t_logits = teacher(slots, None)
            t_probs = F.softmax(t_logits / temperature, dim=-1)
        mask = stratified_subset_masks(b, num_slots, device)
        s_logits = student(slots, mask)
        loss = kl_distill_loss(s_logits, t_probs, temperature)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        loss_acc += loss.item() * b
        n += b
    return loss_acc / max(n, 1)
