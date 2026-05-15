"""
Registry for METEOR noise model classes.

Noise models register themselves at import time using the
``@register_noise_model`` decorator so that ``train_noise_model_from_cmip6``
can look them up by keyword without hard-coding class names.

Example
-------
>>> from meteor.noise_model.registry import register_noise_model

>>> @register_noise_model("my-model", "mymodel1.0")
... class MyNoiseModel(NoiseModelBase):
...     model_type = "my-model"
...     ...
"""

from __future__ import annotations

from typing import Dict, Type

from .base import NoiseModelBase

# Global registry mapping lowercase alias → class
NOISE_MODEL_REGISTRY: Dict[str, Type[NoiseModelBase]] = {}
# maps alias → canonical key (first registered alias)
NOISE_MODEL_ALIASES: Dict[str, str] = {}

def _all_registered_models() -> str:
    """Return a human-friendly string listing all registered models and aliases."""
    lines = []
    for key, cls in sorted(NOISE_MODEL_REGISTRY.items()):
        aliases = [alias for alias, canonical in NOISE_MODEL_ALIASES.items() if canonical == key]
        lines.append(f"{key} (aliases: {', '.join(aliases)}) → {cls.__name__}")
    return_string = ",\n".join(lines[:-1])
    return_string = return_string + ", and\n" + lines[-1]
    return return_string

get_model_alias_error_msg = (
    "Unknown noise model {}: '{}'. "
    "Available registered models: {}. "
    "Register new models with the @register_noise_model(...) decorator."
)

def normalize_key(key: str) -> str:
    """Normalize a registry key to the first registered alias.
    This is important for consistent cache filenames and user-friendly
    error messages. For example if decorator is used as
    @register_noise_model("pca-varx", "meteor1.6"),
    both "pca-varx" and "meteor1.6" will be normalized to "pca-varx".
    """
    key = key.lower()
    try:
        return NOISE_MODEL_ALIASES[key]
    except KeyError:
        err_msg = get_model_alias_error_msg.format(
            "alias", key, _all_registered_models()
        )
        raise ValueError(err_msg) from None

def register_noise_model(*aliases: str):
    """
    Class decorator that registers a :class:`NoiseModelBase` subclass under
    one or more aliases (case-insensitive).

    Raises ``ValueError`` if an alias is already taken.

    Parameters
    ----------
    *aliases : str
        One or more names under which the class is accessible.

    Returns
    -------
    Callable[[type], type]
        The unmodified class (decorator passthrough).

    Example
    -------
    >>> @register_noise_model("pca-varx", "meteor1.6")
    ... class PCAVARXNoiseModel(NoiseModelBase):
    ...     ...
    """

    def decorator(cls: Type[NoiseModelBase]) -> Type[NoiseModelBase]:
        for alias in aliases:
            key = alias.lower()
            if key in NOISE_MODEL_REGISTRY:
                raise ValueError(
                    f"Noise model alias '{key}' is already registered to "
                    f"'{NOISE_MODEL_REGISTRY[key].__name__}'. "
                    "Choose a different alias or remove the existing registration."
                )
            NOISE_MODEL_REGISTRY[key] = cls
            # Set the canonical alias for normalization
            if key not in NOISE_MODEL_ALIASES:
                NOISE_MODEL_ALIASES[key] = aliases[0].lower()
        return cls

    return decorator


def get_noise_model_class(name: str) -> Type[NoiseModelBase]:
    """
    Return the noise model class registered under *name* (case-insensitive).

    Parameters
    ----------
    name : str
        Registry key or alias (e.g. ``'pca-varx'``, ``'meteor1.6'``).

    Returns
    -------
    type[NoiseModelBase]

    Raises
    ------
    ValueError
        If *name* is not found, with a helpful message listing available keys.
    """
    key = normalize_key(name)
    if key not in NOISE_MODEL_REGISTRY:
        err_msg = get_model_alias_error_msg.format(
            "type", key, _all_registered_models()
        )
        raise ValueError(err_msg) from None
    return NOISE_MODEL_REGISTRY[key]
