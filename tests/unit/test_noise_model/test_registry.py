"""Tests for meteor.noise_model.registry."""

import pytest

from meteor.noise_model.base import NoiseModelBase
from meteor.noise_model.registry import (
    NOISE_MODEL_REGISTRY,
    get_noise_model_class,
    register_noise_model,
)


def _make_stub(name="stub"):
    """Return a minimal concrete NoiseModelBase subclass."""

    class _Stub(NoiseModelBase):
        model_type = name

        def fit(self, training_data, variable_name=None, **kwargs):
            pass

        def generate_realization(self, t, n_realizations=1, **kwargs):
            pass

        def save_model(self, filepath):
            pass

        @classmethod
        def load_model_from_file(cls, filepath):
            pass

    return _Stub


def test_register_and_lookup():
    """A class decorated with register_noise_model can be retrieved by name."""
    StubA = _make_stub("stub-a")
    register_noise_model("test-stub-a", "test-stub-a-alias")(StubA)

    assert get_noise_model_class("test-stub-a") is StubA
    assert get_noise_model_class("TEST-STUB-A") is StubA  # case-insensitive
    assert get_noise_model_class("test-stub-a-alias") is StubA


def test_register_duplicate_alias_raises():
    """Registering the same alias twice raises ValueError."""
    StubB = _make_stub("stub-b")
    StubC = _make_stub("stub-c")
    register_noise_model("test-dupe-key")(StubB)

    with pytest.raises(ValueError, match="already registered"):
        register_noise_model("test-dupe-key")(StubC)


def test_get_unknown_key_raises():
    """Looking up an unknown key raises ValueError with helpful message."""
    with pytest.raises(ValueError, match="Unknown noise model alias"):
        get_noise_model_class("does-not-exist-xyz")


def test_pca_varx_registered():
    """PCAVARXNoiseModel must be registered under its expected aliases."""
    # Importing triggers the decorator
    from meteor.noise_model import PCAVARXNoiseModel

    assert get_noise_model_class("pca-varx") is PCAVARXNoiseModel
    assert get_noise_model_class("meteor1.6") is PCAVARXNoiseModel
    assert get_noise_model_class("meteor16") is PCAVARXNoiseModel
