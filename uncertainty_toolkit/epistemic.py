"""
epistemic.py
------------
Epistemic (model) uncertainty via Monte-Carlo Dropout.

Theory
~~~~~~
At inference time, keeping dropout active turns the network into an
approximate Bayesian model (Gal & Ghahramani, 2016).  Running T stochastic
forward passes yields a distribution of softmax outputs from which we derive:

  Predictive Entropy  H[y|x, D]
    = -Σ_c  p̄_c · log(p̄_c)
    where p̄_c = (1/T) Σ_t p_c^(t)   (mean probability per class)

  Mutual Information  I[y, ω | x, D]
    = H[y|x,D]  −  (1/T) Σ_t H[y|x,ω^(t)]
    ≈ predictive_entropy − mean_of_per_pass_entropies
    This decomposes total uncertainty into the part attributable to model
    parameter uncertainty (epistemic) vs. data noise (aleatoric).

Both metrics increase when the model is uncertain, but mutual information
isolates the *model's* contribution and is therefore the canonical epistemic
uncertainty measure.

Usage
-----
>>> estimator = EpistemicEstimator(model, n_passes=30)
>>> result = estimator.estimate(test_loader)
>>> result.predictive_entropy   # shape (N,) — total predictive uncertainty
>>> result.mutual_information   # shape (N,) — epistemic uncertainty
>>> result.variance             # shape (N,) — mean variance of softmax probs
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import UncertaintyEstimator


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EpistemicResult:
    """
    All outputs of the epistemic estimator — one value per sample.

    Attributes
    ----------
    labels : Tensor (N,)
        Ground-truth class indices from the DataLoader.
    predictions : Tensor (N,)
        Argmax of the mean softmax distribution across T passes.
    correct : Tensor (N,) bool
        True where ``predictions == labels``.
    mean_probs : Tensor (N, C)
        Mean softmax probability vector averaged over T passes.
    predictive_entropy : Tensor (N,)
        H[y|x,D] — total uncertainty of the predictive distribution.
    mutual_information : Tensor (N,)
        I[y,ω|x,D] — epistemic component of uncertainty.
        High values indicate the model parameters are uncertain.
    variance : Tensor (N,)
        Mean over classes of Var_t[p_c^(t)] — an intuitive proxy metric
        complementary to the information-theoretic measures above.
    raw_passes : Tensor (N, T, C) or None
        Per-pass softmax outputs.  Stored only when
        ``EpistemicEstimator(keep_raw=True)``; useful for debugging.
    """

    labels: torch.Tensor
    predictions: torch.Tensor
    correct: torch.Tensor
    mean_probs: torch.Tensor
    predictive_entropy: torch.Tensor
    mutual_information: torch.Tensor
    variance: torch.Tensor
    raw_passes: torch.Tensor | None = None

    def to(self, device: Union[str, torch.device]) -> "EpistemicResult":
        """Move all tensors to ``device`` (in-place), return self."""
        for field_name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                object.__setattr__(self, field_name, value.to(device))
        return self


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class EpistemicEstimator(UncertaintyEstimator):
    """
    Monte-Carlo Dropout estimator for epistemic uncertainty.

    Works with *any* ``nn.Module`` that contains at least one ``nn.Dropout``
    or ``nn.Dropout2d`` layer.  No weight changes are made; the estimator
    only temporarily switches those layers to training mode during inference.

    Parameters
    ----------
    model : nn.Module
        Pretrained classifier.  Must contain dropout layers.
    n_passes : int
        Number of stochastic forward passes (T).  Higher values reduce
        Monte-Carlo variance at the cost of compute.  30–50 is a reasonable
        default for most use-cases.
    device : str | torch.device | None
        Inference device.  Auto-selects CUDA when available.
    keep_raw : bool
        If True, the result stores all (N, T, C) per-pass softmax tensors.
        Memory-intensive for large datasets; disabled by default.

    
    Examples
    --------
    >>> estimator = EpistemicEstimator(model, n_passes=50)
    >>> result = estimator.estimate(test_loader)
    >>> # Highest-uncertainty samples
    >>> top_idx = result.mutual_information.argsort(descending=True)[:10]
    """

    # Dropout layer types we detect and keep active during inference
    _DROPOUT_TYPES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)

    def __init__(
        self,
        model: nn.Module,
        n_passes: int = 30,
        device: Union[str, torch.device, None] = None,
        keep_raw: bool = False,
    ) -> None:
        super().__init__(model, device)

        if n_passes < 1:
            raise ValueError(f"n_passes must be ≥ 1, got {n_passes}")
        self.n_passes = n_passes
        self.keep_raw = keep_raw

        self._dropout_layers = self.find_dropout_layers()
        if not self._dropout_layers:
            raise ValueError(
                "No nn.Dropout layers found in the model. "
                "MC Dropout requires at least one dropout layer. "
                "Add dropout to your architecture or use a different estimator."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, dataloader: DataLoader) -> EpistemicResult:
        """
        Run MC Dropout over all batches in ``dataloader``.

        Parameters
        ----------
        dataloader : DataLoader
            Yields ``(inputs, labels)`` pairs.

        Returns
        -------
        EpistemicResult
            All uncertainty metrics, one scalar per sample.
        """
        all_pass_probs: list[torch.Tensor] = []  # each: (N, C)  per pass
        all_labels: list[torch.Tensor] = []

        self.model.eval()  # BN uses running stats; only dropout stays active

        with self.dropout_active(), torch.no_grad():
            for _ in range(self.n_passes):
                batch_probs: list[torch.Tensor] = []

                for batch in dataloader:
                    inputs, labels = batch[0], batch[1]
                    inputs = self.to_device(inputs)

                    logits = self.model(inputs)
                    probs = torch.softmax(logits, dim=-1).cpu()
                    batch_probs.append(probs)

                    
                    if len(all_pass_probs) == 0:
                        all_labels.append(labels.cpu())

                all_pass_probs.append(torch.cat(batch_probs, dim=0))  # (N, C)

        # Stack → (T, N, C), then transpose → (N, T, C)
        stacked = torch.stack(all_pass_probs, dim=0).permute(1, 0, 2)
        labels = torch.cat(all_labels, dim=0)

        return self.compute_metrics(stacked, labels)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def find_dropout_layers(self) -> list[nn.Module]:
        """Return all dropout sublayers in the model."""
        return [
            m for m in self.model.modules()
            if isinstance(m, self._DROPOUT_TYPES)
        ]

    @contextmanager
    def dropout_active(self):
        """
        Context manager that temporarily puts only dropout layers into
        training mode (so they sample masks), while the rest of the model
        stays in eval mode (batch norm uses running stats, etc.).

        Restores the original training flags on exit — even on exception.
        """
        original_states = {m: m.training for m in self._dropout_layers}
        try:
            for layer in self._dropout_layers:
                layer.train()
            yield
        finally:
            for layer, state in original_states.items():
                layer.training = state

    def compute_metrics(
        self,
        stacked: torch.Tensor,   # (N, T, C)
        labels: torch.Tensor,    # (N,)
    ) -> EpistemicResult:
        """Derive all scalar uncertainty metrics from the (N, T, C) cube."""

        # Mean predictive distribution: p̄(y|x) averaged over passes
        mean_probs = stacked.mean(dim=1)          # (N, C)

        # --- Predictive Entropy: H[y | x, D] ---
        # Total uncertainty of the mean predictive distribution
        predictive_entropy = self.entropy(mean_probs)  # (N,)

        # --- Per-pass entropies: H[y | x, ω^(t)] ---
        per_pass_entropy = self.entropy(stacked)       # (N, T)
        mean_per_pass_entropy = per_pass_entropy.mean(dim=1)  # (N,)

        # --- Mutual Information: epistemic uncertainty ---
        # I[y, ω | x, D] = H[y|x,D] - E_ω[H[y|x,ω]]
        # Clamp to 0 to avoid small negatives from floating-point noise
        mutual_information = (predictive_entropy - mean_per_pass_entropy).clamp(min=0.0)

        # --- Variance: mean Var_t[p_c] across classes ---
        variance = stacked.var(dim=1).mean(dim=-1)      # (N,)

        # --- Predictions ---
        predictions = mean_probs.argmax(dim=-1)         # (N,)
        correct = predictions == labels

        return EpistemicResult(
            labels=labels,
            predictions=predictions,
            correct=correct,
            mean_probs=mean_probs,
            predictive_entropy=predictive_entropy,
            mutual_information=mutual_information,
            variance=variance,
            raw_passes=stacked if self.keep_raw else None,
        )
