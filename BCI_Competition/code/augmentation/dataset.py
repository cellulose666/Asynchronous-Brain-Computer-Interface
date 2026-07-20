"""AugmentedDataset — online data augmentation wrapper.

Supports two families of augmentation:
  - single-sample: applied independently per sample (all _BUILTIN methods)
  - pair-based:    mixes two same-class samples (mixup, trial_mixup, ch_mixure, dwta)

Pair-based methods match the offline reference behaviour: the reference
pre-generates all pair combinations and concatenates them to the dataset.
We approximate this online by, with probability p, fetching a second
random sample of the same class and mixing it with the current sample.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    """Wraps (features, labels) tensors and applies augmentations on-the-fly.

    Parameters
    ----------
    features : torch.Tensor  shape (N, C, T)
    labels   : torch.Tensor  shape (N,)  — used to find same-class partners for
                                         pair-based augmentations
    pipeline : callable | None
        Single-sample augmentation pipeline (AugmentationPipeline or similar).
    pair_aug : callable | None
        Pair-based augmentation (ChannelMixup / TrialMixup / etc.).  Applied
        with its own internal ``p`` probability.
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor,
                 pipeline, pair_aug=None):
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        self.features = features
        self.labels = labels
        self.pipeline = pipeline
        self.pair_aug = pair_aug

        # Pre-compute per-class indices for fast same-class partner lookup
        self._class_indices: dict[int, torch.Tensor] = {}
        unique_labels = labels.unique().long().tolist()
        for label in unique_labels:
            self._class_indices[int(label)] = torch.where(labels == label)[0]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.labels[idx]

        # Step 1: single-sample augmentations (noise, scale, neg, freq_shift, ...)
        if self.pipeline is not None:
            x = self.pipeline(x.unsqueeze(0)).squeeze(0)

        # Step 2: pair-based augmentation (requires same-class partner)
        if self.pair_aug is not None and self.pair_aug.p:
            if isinstance(self.pair_aug.p, float) and torch.rand(1).item() < self.pair_aug.p:
                partner = self._sample_partner(idx)
                if partner is not None:
                    x = self._apply_pair(x.unsqueeze(0), partner.unsqueeze(0)).squeeze(0)

        return x, y

    def _sample_partner(self, idx: int) -> torch.Tensor | None:
        """Return a random same-class sample different from idx, or None."""
        label = int(self.labels[idx].item())
        pool = self._class_indices[label]
        if len(pool) <= 1:
            return None  # no other sample of this class
        # pick a random partner that is not idx
        partner_pool = pool[pool != idx]
        if len(partner_pool) == 0:
            return None
        partner_idx = partner_pool[torch.randint(0, len(partner_pool), (1,)).item()]
        return self.features[partner_idx]

    def _apply_pair(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Apply pair-based augmentation.  If called, self.pair_aug is not None."""
        from .augmentations import TrialMixup
        if isinstance(self.pair_aug, TrialMixup):
            out1, _ = self.pair_aug(x1, x2)  # TrialMixup returns a tuple, take first
            return out1
        return self.pair_aug(x1, x2)
