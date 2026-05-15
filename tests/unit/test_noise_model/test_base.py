"""Tests for meteor.noise_model.base — NoiseModelBase and TrainingBundle."""

import pytest

from meteor.noise_model.base import NoiseModelBase, TrainingBundle


# ---------------------------------------------------------------------------
# TrainingBundle
# ---------------------------------------------------------------------------


def test_training_bundle_is_dict():
    """TrainingBundle is a plain dict subclass."""
    bundle = TrainingBundle(monthly_data="dummy", variable_name="tas")
    assert isinstance(bundle, dict)
    assert bundle["monthly_data"] == "dummy"
    assert bundle["variable_name"] == "tas"


def test_training_bundle_extra_keys():
    """TrainingBundle accepts arbitrary extra keys (it is an open dict)."""
    bundle = TrainingBundle(
        monthly_data="x",
        variable_name="pr",
        picontrol_baseline=1.5,
        additional_variables={"tas": "data"},
    )
    assert bundle["picontrol_baseline"] == 1.5
    assert bundle["additional_variables"]["tas"] == "data"


# ---------------------------------------------------------------------------
# NoiseModelBase — cannot be instantiated directly
# ---------------------------------------------------------------------------


def test_noise_model_base_cannot_be_instantiated():
    """NoiseModelBase is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        NoiseModelBase()


def test_noise_model_base_required_data_default():
    """The default required_data returns the two mandatory keys."""

    class _MinimalModel(NoiseModelBase):
        def fit(self, training_data, variable_name=None, **kwargs):
            pass

        def generate_realization(self, global_temp_trajectory, n_realizations=1, **kwargs):
            pass

        def save_model(self, filepath):
            pass

        @classmethod
        def load_model_from_file(cls, filepath):
            pass

    assert NoiseModelBase.required_data() == {"monthly_data", "variable_name"}
    assert _MinimalModel.required_data() == {"monthly_data", "variable_name"}


def test_noise_model_base_generate_regional_raises():
    """The default generate_regional_mean_realizations raises NotImplementedError."""

    class _MinimalModel(NoiseModelBase):
        def fit(self, training_data, variable_name=None, **kwargs):
            pass

        def generate_realization(self, global_temp_trajectory, n_realizations=1, **kwargs):
            pass

        def save_model(self, filepath):
            pass

        @classmethod
        def load_model_from_file(cls, filepath):
            pass

    model = _MinimalModel()
    with pytest.raises(NotImplementedError):
        model.generate_regional_mean_realizations([1, 2, 3])


def test_noise_model_base_load_model_delegates():
    """load_model() (in-place) copies state from load_model_from_file result."""

    class _MemModel(NoiseModelBase):
        def fit(self, training_data, variable_name=None, **kwargs):
            pass

        def generate_realization(self, global_temp_trajectory, n_realizations=1, **kwargs):
            pass

        def save_model(self, filepath):
            pass

        @classmethod
        def load_model_from_file(cls, filepath):
            obj = cls.__new__(cls)
            obj.loaded_from = filepath
            return obj

    model = _MemModel()
    model.load_model("/some/path.pkl")
    assert model.loaded_from == "/some/path.pkl"
