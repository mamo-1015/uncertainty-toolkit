"""
aleatoric.py
------------
Aleatoric (data) uncertainty via stochastic augmentation disagreement.

Theory
~~~~~~
For each input x we sample K augmented views  {x̃_k}_{k=1}^K  and run a
single deterministic forward pass per view.  The aleatoric uncertainty is:

    Ua(x) = mean_c  Var_k [ p_c(x̃_k) ]

i.e. the mean variance of the softmax probabilities across augmented views,
averaged over classes.  High variance means the model's output is sensitive
to plausible input perturbations → the input is inherently ambiguous.

Usage
-----
>>> from uncertainty_toolkit import AleatoricEstimator
>>> estimator = AleatoricEstimator(model, n_augmentations=20)
>>> result = estimator.estimate(test_loader)
>>> result.augmentation_variance   # (N,) — main aleatoric signal
>>> result.mean_entropy            # (N,) — entropy-based alternative


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .base import UncertaintyEstimator


# ---------------------------------------------------------------------------
# Default augmentation pipeline
# ---------------------------------------------------------------------------

def default_augmentations(image_size: int = 28) -> Callable:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(
            degrees=12,              # back toward mild — 30 was too much
            translate=(0.08, 0.08), # reduced from 0.15
            scale=(0.90, 1.10),     # reduced from 0.80–1.20
            shear=8,                # reduced from 15
        ),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=3, sigma=(0.3, 1.5))],
            p=0.5,                  # reduced sigma and probability
        ),
        T.RandomApply([
            T.ColorJitter(brightness=0.3, contrast=0.3)
        ], p=0.4),
        # RandomErasing kept but much smaller and less frequent
        # Small erasing stresses ambiguous images without destroying easy ones
        T.RandomErasing(
            p=0.3,                  # was 0.5
            scale=(0.01, 0.06),     # was 0.02–0.15 — much smaller patches
            ratio=(0.3, 3.0),
            value=0.0,
        ),
    ])

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AleatoricResult:
    """
    All outputs of the aleatoric estimator — one value per sample.

    Attributes
    ----------
    labels : Tensor (N,)
        Ground-truth class indices.
    predictions : Tensor (N,)
        Argmax of the mean probability vector across augmented views.
    correct : Tensor (N,) bool
        True where ``predictions == labels``.
    augmentation_variance : Tensor (N,)
        Main aleatoric signal: mean over classes of Var_k[p_c(x̃_k)].
        High → prediction is sensitive to input perturbations → ambiguous input.
    mean_entropy : Tensor (N,)
        Shannon entropy of the mean softmax vector across augmented views.
        Complementary signal; correlated with augmentation_variance but
        information-theoretically distinct.
    per_class_variance : Tensor (N, C)
        Per-class variance Var_k[p_c(x̃_k)] before averaging over classes.
        Useful for diagnosing *which* classes contribute to ambiguity.
    mean_probs : Tensor (N, C)
        Mean softmax vector across K augmented views.
    """

    labels: torch.Tensor
    predictions: torch.Tensor
    correct: torch.Tensor
    augmentation_variance: torch.Tensor
    mean_entropy: torch.Tensor
    conditional_entropy: torch.Tensor
    per_class_variance: torch.Tensor
    mean_probs: torch.Tensor

    def to(self, device: Union[str, torch.device]) -> "AleatoricResult":
        for field_name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                object.__setattr__(self, field_name, value.to(device))
        return self


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class AleatoricEstimator(UncertaintyEstimator):
    """
    Augmentation-disagreement estimator for aleatoric uncertainty.

    Applies K stochastic augmentations to each input, runs one forward
    pass per view, and measures the variance of the resulting softmax
    distributions.  High variance indicates the model's prediction is
    sensitive to plausible input perturbations — a signal of inherent
    input ambiguity (aleatoric uncertainty).

    The model is kept in eval mode throughout; no weight changes are made.

    Parameters
    ----------
    model : nn.Module
        Any pretrained classifier.  No dropout or special layers required.
    n_augmentations : int
        Number of augmented views per input (K).  Higher K reduces
        Monte-Carlo variance in the variance estimate.  20–30 is
        sufficient for most use-cases.
    augmentation_fn : Callable | None
        A callable ``f(image_tensor) -> image_tensor`` applied K times per
        input.  Receives [0,1] denormalized tensors — _augment_batch handles
        normalization wrapping automatically.
        Defaults to a strong affine + blur + erasing pipeline.
    device : str | torch.device | None
        Inference device.  Auto-selects CUDA when available.

    Normalization constants
    -----------------------
    Default values match Fashion-MNIST.  Override for other datasets:
        estimator._norm_mean = torch.tensor([0.4914, 0.4822, 0.4465])  # CIFAR-10
        estimator._norm_std  = torch.tensor([0.2470, 0.2435, 0.2616])

    Examples
    --------
    >>> estimator = AleatoricEstimator(model, n_augmentations=20)
    >>> result = estimator.estimate(test_loader)
    >>> top_idx = result.augmentation_variance.argsort(descending=True)[:10]
    """

    def __init__(
        self,
        model: nn.Module,
        n_augmentations: int = 20,
        augmentation_fn: Optional[Callable] = None,
        device: Union[str, torch.device, None] = None,
        norm_mean: Optional[torch.Tensor] = None,
        norm_std:  Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(model, device)

        if n_augmentations < 2:
            raise ValueError(
                f"n_augmentations must be ≥ 2 to compute variance, got {n_augmentations}"
            )

        self.n_augmentations  = n_augmentations
        self.augmentation_fn  = augmentation_fn or default_augmentations()

        # Normalization constants — default to Fashion-MNIST values
        # These must match whatever transforms the DataLoader applies
        self._norm_mean = norm_mean if norm_mean is not None else torch.tensor([0.2860])
        self._norm_std  = norm_std  if norm_std  is not None else torch.tensor([0.3530])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, dataloader: DataLoader) -> AleatoricResult:
        """
        Run augmentation-disagreement estimation over all batches.

        Parameters
        ----------
        dataloader : DataLoader
            Yields ``(inputs, labels)`` pairs.  Inputs should be the
            normalized float tensors produced by the DataLoader — the
            estimator handles denormalization internally before augmenting.

        Returns
        -------
        AleatoricResult
            All uncertainty metrics, one scalar (or vector) per sample.
        """
        all_aug_probs: list[torch.Tensor] = []
        all_labels:    list[torch.Tensor] = []

        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch[0], batch[1]
                # inputs: (B, C, H, W) normalized float tensor on CPU

                view_probs: list[torch.Tensor] = []
                for _ in range(self.n_augmentations):
                    # _augment_batch: denormalize → augment → renormalize
                    augmented = self.augment_batch(inputs)
                    augmented = self.to_device(augmented)
                    logits = self.model(augmented)
                    probs  = torch.softmax(logits, dim=-1).cpu()
                    view_probs.append(probs)

                # Stack K views → (B, K, C)
                stacked_batch = torch.stack(view_probs, dim=1)
                all_aug_probs.append(stacked_batch)
                all_labels.append(labels.cpu())

        # Concatenate over batches → (N, K, C)
        all_probs = torch.cat(all_aug_probs, dim=0)
        labels    = torch.cat(all_labels,    dim=0)

        return self._compute_metrics(all_probs, labels)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def augment_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Denormalize → augment → renormalize each image in the batch.

        By denormalizing first, all augmentations behave as intended on
        real [0,1] image data.  Renormalization restores the input space
        the model was trained in.

        Parameters
        ----------
        batch : Tensor (B, C, H, W) — normalized, on CPU

        Returns
        -------
        Tensor (B, C, H, W) — augmented and renormalized, on CPU
        """
        mean = self._norm_mean.view(-1, 1, 1).to(batch.device)  # (C,1,1) for broadcasting
        std  = self._norm_std.view(-1, 1, 1).to(batch.device)

        augmented = []
        for img in batch:
            # 1. Denormalize: normalized → [0, 1]
            img_01 = (img * std) + mean
            img_01 = img_01.clamp(0.0, 1.0)

            # 2. Augment in [0, 1] space — all transforms work correctly here
            try:
                aug_img = self.augmentation_fn(img_01)
            except (RuntimeError, ValueError, TypeError):
                # Fallback: skip augmentation if pipeline fails
                # (e.g. grayscale image fed to 3-channel ColorJitter)
                aug_img = img_01

            # 3. Renormalize back to match model's expected input distribution
            aug_img = (aug_img - mean) / std
            augmented.append(aug_img)

        return torch.stack(augmented, dim=0)

    def _compute_metrics(
            self,
            stacked: torch.Tensor,  # (N, K, C)
            labels:  torch.Tensor,  # (N,)
        ) -> AleatoricResult:
            """Derive all scalar uncertainty metrics from the (N, K, C) cube."""

            # Mean softmax across K augmented views: (N, C)
            mean_probs = stacked.mean(dim=1)

            # Per-class variance across views: Var_k[p_c(x̃_k)] → (N, C)
            per_class_variance = stacked.var(dim=1, unbiased = False)

            # Variance signal: mean over classes → (N,)
            augmentation_variance = per_class_variance.mean(dim=-1)

            # Mean entropy H[mean_k p(x̃_k)]: entropy of the averaged distribution
          
            mean_entropy = self.entropy(mean_probs)                   # (N,)

            # Conditional entropy E_k[H[p(x̃_k)]]: mean of per-view entropies
            # This is the average uncertainty under each individual augmented view
            per_view_entropy = self.entropy(stacked)                  # (N, K)
            conditional_entropy = per_view_entropy.mean(dim=1)         # (N,)

            # Predictions from mean distribution
            predictions = mean_probs.argmax(dim=-1)
            correct     = predictions == labels

            return AleatoricResult(
                labels=labels,
                predictions=predictions,
                correct=correct,
                augmentation_variance=augmentation_variance,
                mean_entropy=mean_entropy,
                conditional_entropy=conditional_entropy,
                per_class_variance=per_class_variance,
                mean_probs=mean_probs,
            )