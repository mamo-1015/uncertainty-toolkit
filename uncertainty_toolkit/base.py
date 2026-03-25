"""
base.py
-------
Abstract base class that all uncertainty estimators must implement.

Adding a new estimator? Subclass UncertaintyEstimator, implement `estimate`,
and return a dataclass that carries whichever uncertainty fields your method
produces.  The rest of the toolkit will work without any other changes.
"""

from __future__ import annotations

import abc
from typing import Union

import torch
from torch import nn
from torch.utils.data import DataLoader


class UncertaintyEstimator(abc.ABC):
    """
    Abstract base for plug-and-play uncertainty estimators.

    All estimators accept an arbitrary ``nn.Module`` classifier at
    construction time and expose a single public method — ``estimate`` —
    that consumes a DataLoader and returns an estimator-specific result
    object whose fields are plain ``torch.Tensor`` arrays of shape ``(N,)``.

    Parameters
    ----------
    model : nn.Module
        Any pretrained PyTorch classifier.  The estimator must never
        mutate the model's weights or its ``training`` flag permanently.
    device : str | torch.device | None
        Target device.  Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

    @abc.abstractmethod
    def estimate(self, dataloader: DataLoader):
        """
        Run uncertainty estimation over a full DataLoader.

        Parameters
        ----------
        dataloader : DataLoader
            Yields ``(inputs, labels)`` batches.  Labels are used only
            to compute correctness flags in the result — they are not
            fed to the model.

        Returns
        -------
        A dataclass instance with at minimum:
            - ``labels``        : ground-truth class indices  (N,)
            - ``predictions``   : argmax class predictions    (N,)
            - ``correct``       : boolean correctness mask    (N,)
        Plus estimator-specific uncertainty tensors.
        """

    # ------------------------------------------------------------------
    # Shared helpers available to all subclasses
    # ------------------------------------------------------------------

    def to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device, non_blocking=True)

    @staticmethod
    def safe_log(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Numerically stable log — clamps near-zero values before log."""
        return torch.log(p.clamp(min=eps))

    @staticmethod
    def entropy(probs: torch.Tensor) -> torch.Tensor:
        """
        entropy of a probability distribution.

        Parameters
        ----------
        probs : Tensor of shape (..., C)
            Probability vectors that sum to 1 along the last dimension.

        Returns
        -------
        Tensor of shape (...,)
        """
        return -(probs * UncertaintyEstimator.safe_log(probs)).sum(dim=-1)
