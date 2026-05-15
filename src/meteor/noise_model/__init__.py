"""
METEOR noise model subpackage.

Provides the base class, registry, shared components, and concrete noise model
implementations.  All noise models are accessed through the registry:

    from meteor.noise_model import get_noise_model_class, PCAVARXNoiseModel

    cls = get_noise_model_class("pca-varx")   # or "meteor1.6" / "meteor16"
    model = cls(n_modes=40, lag_order=2)

New model types can self-register with the ``@register_noise_model`` decorator
so that ``train_noise_model_from_cmip6`` in ``noise_generator`` automatically
routes to them via the ``model_type`` keyword.
"""

from .base import NoiseModelBase, TrainingBundle
from .pca_varx import PCAVARXNoiseModel  # registers itself on import
from .registry import NOISE_MODEL_REGISTRY, get_noise_model_class, register_noise_model
from .seasonal import SeasonalModel

__all__ = [
    "NoiseModelBase",
    "TrainingBundle",
    "SeasonalModel",
    "PCAVARXNoiseModel",
    "NOISE_MODEL_REGISTRY",
    "get_noise_model_class",
    "register_noise_model",
]
