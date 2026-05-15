"""Tests for meteor.noise_model.seasonal — SeasonalModel."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from meteor.noise_model.seasonal import SeasonalModel


# ---------------------------------------------------------------------------
# create_harmonic_features
# ---------------------------------------------------------------------------


def test_create_harmonic_features_shape():
    time = np.arange(120)
    t_glob = np.linspace(0, 1, 120)
    X = SeasonalModel.create_harmonic_features(time, t_glob)
    assert X.shape == (120, 9)


def test_create_harmonic_features_columns():
    """Column layout follows the documented specification."""
    time = np.array([0.0, 6.0, 12.0])
    t_glob = np.array([1.0, 2.0, 3.0])
    X = SeasonalModel.create_harmonic_features(time, t_glob)

    # Column 0: t_glob
    np.testing.assert_allclose(X[:, 0], t_glob)
    # Column 1: cos(2πt/12) at t=0 → cos(0)=1
    np.testing.assert_allclose(X[0, 1], 1.0)
    # Column 5: t_glob * annual_cos
    np.testing.assert_allclose(X[:, 5], t_glob * X[:, 1])


def test_create_harmonic_features_matches_old_inline():
    """Output must be numerically identical to the old inline implementation."""
    time = np.arange(60)
    t_glob = np.random.default_rng(42).uniform(-1, 2, 60)
    months_per_year = 12
    annual_cos = np.cos(2 * np.pi * time / months_per_year)
    annual_sin = np.sin(2 * np.pi * time / months_per_year)
    semiannual_cos = np.cos(4 * np.pi * time / months_per_year)
    semiannual_sin = np.sin(4 * np.pi * time / months_per_year)
    X_ref = np.vstack(
        [
            t_glob,
            annual_cos,
            annual_sin,
            semiannual_cos,
            semiannual_sin,
            t_glob * annual_cos,
            t_glob * annual_sin,
            t_glob * semiannual_cos,
            t_glob * semiannual_sin,
        ]
    ).T

    X_new = SeasonalModel.create_harmonic_features(time, t_glob)
    np.testing.assert_allclose(X_new, X_ref)


# ---------------------------------------------------------------------------
# extract_exog_variables
# ---------------------------------------------------------------------------


def test_extract_exog_all():
    X = np.random.rand(10, 9)
    out = SeasonalModel.extract_exog_variables(X, "all")
    assert out.shape == (10, 3)
    np.testing.assert_array_equal(out, X[:, :3])


def test_extract_exog_temp_only():
    X = np.random.rand(10, 9)
    out = SeasonalModel.extract_exog_variables(X, "temp_only")
    assert out.shape == (10, 1)
    np.testing.assert_array_equal(out, X[:, :1])


def test_extract_exog_none():
    X = np.random.rand(10, 9)
    assert SeasonalModel.extract_exog_variables(X, "none") is None


def test_extract_exog_invalid():
    X = np.random.rand(10, 9)
    with pytest.raises(ValueError, match="Invalid use_exog"):
        SeasonalModel.extract_exog_variables(X, "bad_value")


# ---------------------------------------------------------------------------
# Instance fit / predict / score
# ---------------------------------------------------------------------------


def _make_dummy_data(n_time=120, n_space=50, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(n_time)
    t_glob = rng.uniform(0, 2, n_time)
    Y = rng.standard_normal((n_time, n_space))
    return time, t_glob, Y


def test_fit_sets_fitted():
    time, t_glob, Y = _make_dummy_data()
    sm = SeasonalModel()
    assert not sm.fitted
    sm.fit(time, t_glob, Y)
    assert sm.fitted


def test_predict_shape():
    time, t_glob, Y = _make_dummy_data(n_time=120, n_space=50)
    sm = SeasonalModel().fit(time, t_glob, Y)
    pred = sm.predict(time, t_glob)
    assert pred.shape == (120, 50)


def test_score_in_range():
    time, t_glob, Y = _make_dummy_data()
    sm = SeasonalModel().fit(time, t_glob, Y)
    r2 = sm.score(time, t_glob, Y)
    assert 0.0 <= r2 <= 1.0


def test_get_anomalies_residuals():
    time, t_glob, Y = _make_dummy_data()
    sm = SeasonalModel().fit(time, t_glob, Y)
    anomalies = sm.get_anomalies(Y, time, t_glob)
    predicted = sm.predict(time, t_glob)
    np.testing.assert_allclose(anomalies, Y - predicted)


def test_get_anomalies_with_baseline():
    time, t_glob, Y = _make_dummy_data()
    sm = SeasonalModel().fit(time, t_glob, Y)
    baseline = 2.5
    anomalies = sm.get_anomalies(Y, time, t_glob, baseline=baseline)
    predicted = sm.predict(time, t_glob)
    np.testing.assert_allclose(anomalies, Y - predicted - baseline)


def test_coef_and_intercept_properties():
    time, t_glob, Y = _make_dummy_data(n_time=120, n_space=10)
    sm = SeasonalModel().fit(time, t_glob, Y)
    assert sm.coef_.shape == (10, 9)
    assert sm.intercept_.shape == (10,)


def test_unfitted_raises():
    sm = SeasonalModel()
    with pytest.raises(ValueError, match="must be fitted"):
        sm.predict(np.arange(10), np.ones(10))
    with pytest.raises(ValueError, match="must be fitted"):
        _ = sm.coef_


# ---------------------------------------------------------------------------
# _from_sklearn_lr backward-compat wrapper
# ---------------------------------------------------------------------------


def test_from_sklearn_lr():
    time, t_glob, Y = _make_dummy_data(n_time=120, n_space=5)
    X = SeasonalModel.create_harmonic_features(time, t_glob)
    lr = LinearRegression(fit_intercept=True).fit(X, Y)

    sm = SeasonalModel._from_sklearn_lr(lr)
    assert sm.fitted
    pred_from_sm = sm.predict(time, t_glob)
    pred_from_lr = lr.predict(X)
    np.testing.assert_allclose(pred_from_sm, pred_from_lr)


# ---------------------------------------------------------------------------
# Fit returns self (method chaining)
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    time, t_glob, Y = _make_dummy_data()
    sm = SeasonalModel()
    result = sm.fit(time, t_glob, Y)
    assert result is sm
