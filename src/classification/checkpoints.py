from __future__ import annotations

import torch


def save_deepsets_checkpoint(path: str, state_dict: dict, meta: dict | None = None):
    payload = {"state_dict": state_dict, "meta": meta or {}}
    torch.save(payload, path)


def load_deepsets_checkpoint(path: str, map_location=None) -> tuple[dict, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    return ckpt["state_dict"], ckpt.get("meta", {})
