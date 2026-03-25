"""
uncertainty_toolkit
===================
Plug-and-play uncertainty estimation for any PyTorch classifier.

Quickstart
----------
>>> from uncertainty_toolkit import EpistemicEstimator, AleatoricEstimator

>>> # Part 1 — epistemic uncertainty (MC Dropout)
>>> ep = EpistemicEstimator(model, n_passes=30)
>>> ep_result = ep.estimate(test_loader)
>>> ep_result.mutual_information      # (N,)  epistemic uncertainty

>>> # Part 2 — aleatoric uncertainty (augmentation disagreement)
>>> al = AleatoricEstimator(model, n_augmentations=20)
>>> al_result = al.estimate(test_loader)
>>> al_result.augmentation_variance   # (N,)  aleatoric uncertainty
"""

from .epistemic import EpistemicEstimator, EpistemicResult
from .aleatoric import AleatoricEstimator, AleatoricResult
from .base import UncertaintyEstimator

__all__ = [
    "EpistemicEstimator",
    "EpistemicResult",
    "AleatoricEstimator",
    "AleatoricResult",
    "UncertaintyEstimator",
]

__version__ = "0.1.0"
