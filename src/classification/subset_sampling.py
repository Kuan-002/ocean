import torch


def stratified_subset_masks(batch_size: int, num_slots: int, device: torch.device) -> torch.Tensor:
    """
    Per sample: draw k ~ Uniform({1, ..., K}), then a uniformly random k-subset of slots.
    Plan §4.2 — stratified by subset size so large k (e.g. K-1, K-2) are not starved.

    Returns:
        mask [B, K] with values in {0, 1}, at least one 1 per row.
    """
    k = num_slots
    ks = torch.randint(1, k + 1, (batch_size,), device=device)
    masks = torch.zeros(batch_size, k, device=device)
    for i in range(batch_size):
        perm = torch.randperm(k, device=device)
        masks[i, perm[: ks[i].item()]] = 1.0
    return masks
