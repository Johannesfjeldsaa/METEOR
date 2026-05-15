"""Tests for meteor.noise_model.pca_varx — PCAVARXNoiseModel."""

import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from meteor.noise_model.base import NoiseModelBase, TrainingBundle
from meteor.noise_model.pca_varx import PCAVARXNoiseModel
from meteor.noise_model.seasonal import SeasonalModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_training_dataset(n_time=120, n_lat=5, n_lon=6, seed=0):
    """Return a minimal xr.Dataset suitable for fitting PCAVARXNoiseModel."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_time)
    lats = np.linspace(-90, 90, n_lat)
    lons = np.linspace(-180, 180, n_lon)
    ens = np.array([1])

    # Simple field with trend + seasonality to give a non-trivial fit
    temp = np.zeros((n_time, n_lat, n_lon, 1))
    for i, t in enumerate(time):
        seasonal = 10 * np.sin(2 * np.pi * t / 12.0)
        trend = 0.01 * t
        for j, lat in enumerate(lats):
            temp[i, j, :, 0] = seasonal + trend + 0.1 * lat + rng.standard_normal(n_lon)

    da = xr.DataArray(
        temp,
        coords={"month": time, "lat": lats, "lon": lons, "ens": ens},
        dims=["month", "lat", "lon", "ens"],
    )
    return xr.Dataset({"tas": da}), time, lats, lons


# ---------------------------------------------------------------------------
# Inheritance and interface
# ---------------------------------------------------------------------------


def test_pca_varx_is_noise_model_base():
    assert issubclass(PCAVARXNoiseModel, NoiseModelBase)


def test_pca_varx_model_type():
    assert PCAVARXNoiseModel.model_type == "pca-varx"


def test_pca_varx_required_data():
    req = PCAVARXNoiseModel.required_data()
    assert "monthly_data" in req
    assert "variable_name" in req
    assert "picontrol_baseline" in req


# ---------------------------------------------------------------------------
# Initialisation validation
# ---------------------------------------------------------------------------


def test_init_defaults():
    model = PCAVARXNoiseModel()
    assert model.n_modes == 10
    assert model.lag_order == 2
    assert model.use_exog == "temp_only"
    assert model.noise_pc_distribution == "normal"
    assert model.t_df is None
    assert not model.fitted


def test_init_invalid_distribution():
    with pytest.raises(ValueError, match="noise_pc_distribution must be one of"):
        PCAVARXNoiseModel(noise_pc_distribution="bad")


def test_init_invalid_t_df_string():
    with pytest.raises(ValueError, match="t_df string must be 'mle'"):
        PCAVARXNoiseModel(t_df="wrong")


def test_init_t_df_too_small():
    with pytest.raises(ValueError, match="t_df must be > 2"):
        PCAVARXNoiseModel(noise_pc_distribution="t", t_df=1.5)


# ---------------------------------------------------------------------------
# Backward-compat wrappers
# ---------------------------------------------------------------------------


def test_create_harmonic_features_wrapper():
    """_create_harmonic_features delegates to SeasonalModel static method."""
    model = PCAVARXNoiseModel()
    time = np.arange(12)
    t_glob = np.ones(12)
    X1 = model._create_harmonic_features(time, t_glob)
    X2 = SeasonalModel.create_harmonic_features(time, t_glob)
    np.testing.assert_array_equal(X1, X2)


def test_extract_exog_variables_wrapper():
    """_extract_exog_variables delegates to SeasonalModel static method."""
    model = PCAVARXNoiseModel(use_exog="temp_only")
    X = np.random.rand(10, 9)
    out = model._extract_exog_variables(X)
    assert out.shape == (10, 1)


def test_extract_exog_variables_invalid_use_exog():
    model = PCAVARXNoiseModel(use_exog="not_exog_var")
    X = np.random.rand(10, 9)
    with pytest.raises(ValueError, match="Invalid use_exog"):
        model._extract_exog_variables(X)


# ---------------------------------------------------------------------------
# fit — xr.Dataset call (backward-compat)
# ---------------------------------------------------------------------------


def test_fit_dataset_direct():
    """fit(xr.Dataset, variable_name) works (backward-compat path)."""
    ds, _, _, _ = _make_training_dataset()
    model = PCAVARXNoiseModel(n_modes=2)
    t_glob = np.ones(120)
    model.fit(ds, "tas", custom_global_temp=t_glob)
    assert model.fitted
    assert model.variable_name == "tas"
    assert isinstance(model.seasonal_model, SeasonalModel)


def test_fit_wrong_variable_raises():
    ds, _, _, _ = _make_training_dataset()
    model = PCAVARXNoiseModel(n_modes=2)
    t_glob = np.ones(120)
    with pytest.raises(ValueError, match="Variable 'pr' not found"):
        model.fit(ds, "pr", custom_global_temp=t_glob)


def test_fit_custom_temp_wrong_length():
    ds, _, _, _ = _make_training_dataset()
    model = PCAVARXNoiseModel(n_modes=2)
    with pytest.raises(ValueError, match="custom_global_temp length"):
        model.fit(ds, "tas", custom_global_temp=np.ones(50))


# ---------------------------------------------------------------------------
# fit — TrainingBundle dict call (new API)
# ---------------------------------------------------------------------------


def test_fit_training_bundle():
    """fit(TrainingBundle) correctly unpacks and fits the model."""
    ds, _, _, _ = _make_training_dataset()
    t_glob = np.ones(120)
    bundle = TrainingBundle(
        monthly_data=ds,
        variable_name="tas",
        custom_global_temp=t_glob,
        picontrol_baseline=None,
    )
    model = PCAVARXNoiseModel(n_modes=2)
    model.fit(bundle)
    assert model.fitted


def test_fit_training_bundle_overrides_positional_variable_name():
    """variable_name in bundle takes precedence but positional also works."""
    ds, _, _, _ = _make_training_dataset()
    t_glob = np.ones(120)
    bundle = TrainingBundle(
        monthly_data=ds,
        variable_name="tas",
        custom_global_temp=t_glob,
    )
    model = PCAVARXNoiseModel(n_modes=2)
    model.fit(bundle)
    assert model.variable_name == "tas"


# ---------------------------------------------------------------------------
# generate_realization
# ---------------------------------------------------------------------------


def test_generate_realization_not_fitted():
    model = PCAVARXNoiseModel()
    with pytest.raises(ValueError, match="Model must be fitted"):
        model.generate_realization(np.ones(24))


def test_generate_realization_single():
    ds, _, lats, lons = _make_training_dataset(n_lat=4, n_lon=5)
    model = PCAVARXNoiseModel(n_modes=2)
    model.fit(ds, "tas", custom_global_temp=np.ones(120))

    traj = np.ones(24)
    result = model.generate_realization(traj)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("month", "lat", "lon")
    assert result.sizes["month"] == 24


def test_generate_realization_multiple():
    ds, _, _, _ = _make_training_dataset(n_lat=4, n_lon=5)
    model = PCAVARXNoiseModel(n_modes=2)
    model.fit(ds, "tas", custom_global_temp=np.ones(120))

    results = model.generate_realization(np.ones(24), n_realizations=3)
    assert len(results) == 3


def test_generate_realization_noise_only():
    ds, _, _, _ = _make_training_dataset(n_lat=4, n_lon=5)
    model = PCAVARXNoiseModel(n_modes=2)
    model.fit(ds, "tas", custom_global_temp=np.ones(120))

    result = model.generate_realization(np.ones(24), noise_only=True)
    assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# generate_stochastic_pcs
# ---------------------------------------------------------------------------


def test_generate_stochastic_pcs_shape_single():
    ds, _, _, _ = _make_training_dataset()
    model = PCAVARXNoiseModel(n_modes=3)
    model.fit(ds, "tas", custom_global_temp=np.ones(120))
    pcs = model.generate_stochastic_pcs(np.ones(24))
    assert pcs.shape == (24, 3)


def test_generate_stochastic_pcs_shape_multiple():
    ds, _, _, _ = _make_training_dataset()
    model = PCAVARXNoiseModel(n_modes=3)
    model.fit(ds, "tas", custom_global_temp=np.ones(120))
    pcs = model.generate_stochastic_pcs(np.ones(24), n_realizations=5)
    assert pcs.shape == (5, 24, 3)


# ---------------------------------------------------------------------------
# generate_regional_mean_realizations
# ---------------------------------------------------------------------------


def test_generate_regional_mean_global():
    ds, _, _, _ = _make_training_dataset(n_lat=5, n_lon=6)
    model = PCAVARXNoiseModel(n_modes=2)
    model.fit(ds, "tas", custom_global_temp=np.ones(120))

    result = model.generate_regional_mean_realizations(np.ones(24), region="global")
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("month",)


def test_generate_regional_mean_multiple():
    ds, _, _, _ = _make_training_dataset(n_lat=5, n_lon=6)
    model = PCAVARXNoiseModel(n_modes=2)
    model.fit(ds, "tas", custom_global_temp=np.ones(120))

    result = model.generate_regional_mean_realizations(
        np.ones(24), region="global", n_realizations=4
    )
    assert result.dims == ("realization", "month")
    assert result.sizes["realization"] == 4


def test_generate_regional_mean_lat_lon_only_raises():
    ds, _, _, _ = _make_training_dataset()
    model = PCAVARXNoiseModel(n_modes=2)
    model.fit(ds, "tas", custom_global_temp=np.ones(120))
    with pytest.raises(ValueError, match="Both lat and lon must be provided"):
        model.generate_regional_mean_realizations(np.ones(24), lat=10)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip():
    ds, _, _, _ = _make_training_dataset(n_lat=4, n_lon=5)
    model = PCAVARXNoiseModel(n_modes=2)
    model.fit(ds, "tas", custom_global_temp=np.ones(120))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pkl")
        model.save_model(path)

        loaded = PCAVARXNoiseModel.load_model_from_file(path)
        assert loaded.fitted
        assert loaded.n_modes == 2
        assert loaded.variable_name == "tas"
        assert isinstance(loaded.seasonal_model, SeasonalModel)


def test_save_unfitted_raises():
    model = PCAVARXNoiseModel()
    with pytest.raises(ValueError, match="Model must be fitted before saving"):
        model.save_model("/tmp/test.pkl")


def test_load_nonexistent_raises():
    model = PCAVARXNoiseModel()
    with pytest.raises(FileNotFoundError):
        model.load_model("/non/existent/path.pkl")


def test_load_model_inplace():
    """load_model (in-place) populates an empty instance correctly."""
    ds, _, _, _ = _make_training_dataset(n_lat=4, n_lon=5)
    orig = PCAVARXNoiseModel(n_modes=2)
    orig.fit(ds, "tas", custom_global_temp=np.ones(120))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pkl")
        orig.save_model(path)

        empty = PCAVARXNoiseModel()
        assert not empty.fitted
        empty.load_model(path)
        assert empty.fitted
        assert empty.n_modes == 2


# ---------------------------------------------------------------------------
# Backward-compat: MeteorNoiseGenerator alias
# ---------------------------------------------------------------------------


def test_meteor_noise_generator_alias():
    """noise_generator.MeteorNoiseGenerator is PCAVARXNoiseModel."""
    from meteor.noise_generator import MeteorNoiseGenerator

    assert MeteorNoiseGenerator is PCAVARXNoiseModel
