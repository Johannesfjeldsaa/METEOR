"""
PCA-VARX noise model for METEOR (METEORv1.6 standard architecture).

This module contains :class:`PCAVARXNoiseModel`, a concrete implementation of
:class:`~meteor.noise_model.base.NoiseModelBase` that replicates the standard
METEORv1.6 noise pipeline:

1. Temperature-dependent seasonal cycle extraction using modulated harmonic regression
        xr.DataArray or list of xr.DataArray
            Generated climate realizations. If noise_only=True, returns the
            stochastic component plus temperature-modulated harmonics (but without
            direct temperature trends) that can be added to other predictions.PCA-based spatial decomposition of anomalies
2. VARX modeling of principal components
3. Stochastic simulation of new climate realizations
4. Noise-only generation for combining with annual climate projections

The class is registered under the aliases ``"pca-varx"``, ``"meteor1.6"`` and
``"meteor16"`` so that ``train_noise_model_from_cmip6(..., model_type="pca-varx")``
automatically routes here.
"""

from __future__ import annotations

import pickle  # nosec - Used for trusted model serialization only
import warnings
from typing import Optional

import numpy as np
import regionmask
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR

from ..geo_data_utils import global_mean
from .base import NoiseModelBase, TrainingBundle
from .registry import register_noise_model
from .seasonal import SeasonalModel

_IMPLEMENTED_DISTRIBUTIONS = ("normal", "t")
PCA_VARX_ALIASES = ("pca-varx", "meteor1.6", "meteor16")


def _apply_student_t_posthoc_scaling(
    synthetic_pcs: np.ndarray,
    fitted_df: np.ndarray,
) -> np.ndarray:
    """
    Apply post-hoc Student-t scaling to Gaussian VAR output PCs.

    The VAR loop is always simulated with Gaussian innovations. For Student-t
    output, each PC and timestep is scaled as::

        y_t = sqrt(df / V_t) * y_t_normal,   V_t ~ chi2(df)

    This preserves the VAR temporal structure while widening the output
    marginal tails.
    """
    df_arr = np.asarray(fitted_df)
    chi2_samples = np.column_stack(
        [np.random.chisquare(df_j, size=synthetic_pcs.shape[0]) for df_j in df_arr]
    )
    scale_factors = np.sqrt(df_arr[np.newaxis, :] / chi2_samples)
    return synthetic_pcs * scale_factors

@register_noise_model(*PCA_VARX_ALIASES)
class PCAVARXNoiseModel(NoiseModelBase):
    """
    Climate noise generator using PCA and VARX modelling (METEORv1.6).

    This class generates stochastic climate realizations by separating the
    deterministic (temperature-dependent seasonal cycle) and stochastic
    (internal variability) components of monthly climate data.

    Can generate either full 3-D climate realizations or noise-only
    components that can be added to annual projections from METEOR-CORE.

    Attributes
    ----------
    model_type : str
        ``"pca-varx"``
    n_modes : int
        Number of PCA modes retained.
    lag_order : int
        Lag order for the VARX model.
    use_exog : str
        Exogenous-variable strategy for VARX (``'all'``, ``'temp_only'``,
        ``'none'``).
    noise_pc_distribution : str
        Output distribution for generated PCs (``'normal'`` or ``'t'``).
    t_df : float, str, or None
        Degrees-of-freedom specification for Student-t output.
    seasonal_model : SeasonalModel or None
        Fitted seasonal cycle component.
    pca : sklearn.decomposition.PCA or None
        Fitted PCA for anomaly decomposition.
    varx_results : statsmodels VAR results or None
        Fitted VARX model.
    coords : dict or None
        Lat/lon coordinate arrays from training data.
    fitted : bool
        Whether the model has been fitted.
    variable_name : str or None
        Variable name used during fitting.
    """

    model_type: str = "pca-varx"

    # ------------------------------------------------------------------
    # Data declaration
    # ------------------------------------------------------------------

    @classmethod
    def required_data(cls):
        return {
            "monthly_data",
            "variable_name",
            "custom_global_temp",
            "picontrol_baseline"
        }

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _check_noise_pc_distribution(self, noise_pc_distribution, t_df):
        if noise_pc_distribution not in _IMPLEMENTED_DISTRIBUTIONS:
            raise ValueError(
                f"noise_pc_distribution must be one of "
                f"{list(_IMPLEMENTED_DISTRIBUTIONS)}, got '{noise_pc_distribution}'"
            )
        if isinstance(t_df, str) and t_df != "mle":
            raise ValueError(f"t_df string must be 'mle', got '{t_df}'")
        if isinstance(t_df, (int, float)) and not isinstance(t_df, bool):
            if t_df <= 2:
                raise ValueError(
                    f"t_df must be > 2 for finite variance, got {t_df}"
                )
            if t_df <= 4:
                warnings.warn(
                    f"t_df={t_df} <= 4 implies infinite kurtosis. "
                    "Consider using t_df > 4 for well-behaved moments.",
                    stacklevel=2,
                )

    def __init__(
        self,
        n_modes: int = 10,
        lag_order: int = 2,
        use_exog: str = "temp_only",
        noise_pc_distribution: str = "normal",
        t_df=None,
    ):
        """
        Initialise the noise generator.

        Parameters
        ----------
        n_modes : int, default 10
            Number of PCA modes to retain.
        lag_order : int, default 2
            Lag order for the VARX model.
        use_exog : str, default ``'temp_only'``
            Exogenous variables for VARX:

            * ``'all'``       — temperature, annual_cos, annual_sin
            * ``'temp_only'`` — temperature only (recommended)
            * ``'none'``      — pure VAR
        noise_pc_distribution : str, default ``'normal'``
            Output distribution for generated PCs:

            * ``'normal'`` — Gaussian VAR output (no scaling)
            * ``'t'``      — post-hoc per-PC Student-t scaling via
              chi-squared mixtures (VAR innovations stay Gaussian)
        t_df : float, str, or None, default None
            Degrees of freedom for Student-t (``noise_pc_distribution='t'`` only):

            * ``None``    — use 10.0 for every PC
            * float       — use this value for every PC
            (must be > 2 for finite variance, > 4 for finite kurtosis)
            * ``'mle'``   — per-PC univariate MLE on VARX residuals
        """
        self._check_noise_pc_distribution(noise_pc_distribution, t_df)
        self.n_modes = n_modes
        self.lag_order = lag_order
        self.use_exog = use_exog
        self.noise_pc_distribution = noise_pc_distribution
        self.t_df = t_df
        self._fitted_df = None  # shape (n_modes,) after _resolve_df, or None

        self.seasonal_model: Optional[SeasonalModel] = None
        self.pca: Optional[PCA] = None
        self.varx_results = None
        self.coords: Optional[dict] = None
        self.fitted: bool = False
        self.variable_name: Optional[str] = None

        # In-memory cache for regional EOF projections (model-invariant)
        self._regional_eof_projections: dict = {}

        # Diagnostic outputs (set during fit when save_diagnostics=True)
        self.diagnostics: dict = {
            "X_features": None,
            "t_glob": None,
            "time": None,
            "seasonal_r2": None,
            "total_variance_explained": None,
            "seasonal_coef": None,
            "seasonal_intercept": None,
            "Y_data": None,
        }

    # ------------------------------------------------------------------
    # Backward-compat thin wrappers (used by existing tests and notebooks)
    # ------------------------------------------------------------------

    def _create_harmonic_features(
        self, time: np.ndarray, t_glob: np.ndarray
    ) -> np.ndarray:
        """Thin wrapper around :meth:`SeasonalModel.create_harmonic_features`."""
        return SeasonalModel.create_harmonic_features(time, t_glob)

    def _extract_exog_variables(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Thin wrapper around :meth:`SeasonalModel.extract_exog_variables`."""
        return SeasonalModel.extract_exog_variables(X, self.use_exog)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _fix_coords_to_np(self):
        """Ensure coordinate arrays are NumPy arrays for serialization."""
        if hasattr(self.coords["lat"], "values"):
            self.coords["lat"] = self.coords["lat"].values
        if hasattr(self.coords["lon"], "values"):
            self.coords["lon"] = self.coords["lon"].values
        if not isinstance(self.coords["lat"], np.ndarray):
            raise ValueError(
                "Latitude coordinates must be NumPy arrays or xarray.DataArray"
            )
        if not isinstance(self.coords["lon"], np.ndarray):
            raise ValueError(
                "Longitude coordinates must be NumPy arrays or xarray.DataArray"
            )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def fit(
        self,
        monthly_data,
        variable_name: Optional[str] = None,
        custom_global_temp=None,
        picontrol_baseline=None,
        save_diagnostics: bool = False,
        verbose: bool = False,
    ):
        """
        Fit the noise generator to monthly climate data.

        Parameters
        ----------
        monthly_data : xr.Dataset or TrainingBundle (dict)
            Monthly climate data **(month, lat, lon, ens)** or a
            :class:`~meteor.noise_model.base.TrainingBundle` dict.  When a
            dict is given all keyword arguments below are extracted from it;
            any values passed as explicit kwargs override bundle values.
        variable_name : str, optional
            Name of the variable to model (e.g. ``'tas'``, ``'pr'``).
            Required when ``monthly_data`` is an ``xr.Dataset``.
        custom_global_temp : array-like, optional
            Pre-computed smoothed global-mean temperature trajectory.
            Must have the same length as the time dimension of
            ``monthly_data``.  If ``None``, computed internally.
        picontrol_baseline : float or xr.DataArray, optional
            Pre-industrial control baseline to subtract from the global
            temperature and anomaly fields.
        save_diagnostics : bool, default False
            Store intermediate arrays in ``self.diagnostics`` for debugging.
        verbose : bool, default False
            Print variance-decomposition statistics after fitting.
        """
        # ----------------------------------------------------------
        # Dispatch: accept either a TrainingBundle (dict) or bare xr.Dataset
        # ----------------------------------------------------------
        if isinstance(monthly_data, dict):
            bundle = monthly_data
            variable_name = bundle.get("variable_name", variable_name)
            custom_global_temp = bundle.get("custom_global_temp", custom_global_temp)
            picontrol_baseline = bundle.get("picontrol_baseline", picontrol_baseline)
            monthly_data = bundle["monthly_data"]

        # Extract the variable data
        if variable_name not in monthly_data:
            raise ValueError(f"Variable '{variable_name}' not found in dataset")

        ds = monthly_data.copy()

        # Get time coordinate
        time = ds["month"].values

        # Calculate or use provided global temperature
        if custom_global_temp is not None:
            # Validate custom temperature array
            if len(custom_global_temp) != len(time):
                raise ValueError(
                    f"custom_global_temp length ({len(custom_global_temp)}) "
                    f"must match data time dimension ({len(time)})"
                )
            t_glob = np.array(custom_global_temp)

        else:
            # Calculate latitude-weighted global mean temperature
            t_globm = global_mean(ds[variable_name].mean(dim=["ens"]))

            # Check if timeseries is long enough for rolling smoothing
            rolling_window = 60  # 5 years
            if len(time) < rolling_window:
                raise ValueError(
                    f"Time series too short for noise model fitting. "
                    f"Need at least {rolling_window} months "
                    f"({rolling_window / 12:.1f} years), "
                    f"but got {len(time)} months ({len(time) / 12:.1f} years). "
                    "Consider using a longer training period."
                )

            # Apply baseline correction
            if picontrol_baseline is not None:
                # Use piControl baseline for consistency with pattern scaling
                if isinstance(picontrol_baseline, (int, float)):
                    baseline = picontrol_baseline
                else:
                    # Assume it's an array-like, take mean
                    baseline = float(np.mean(picontrol_baseline))
                t_globm = t_globm - baseline
                print(f"   Using piControl baseline: {baseline:.3f}")
            else:
                # Fall back to original method (first 42 years)
                t_globm = t_globm - t_globm[:500].mean()  # Remove baseline
                print("   Using first 42 years as baseline")

            t_glob = (
                t_globm.rolling(month=rolling_window, center=True, min_periods=1)
                .mean()
                .interpolate_na("month", method="nearest", fill_value="extrapolate")
                .values
            )

        # Create harmonic features
        X = SeasonalModel.create_harmonic_features(time, t_glob)
        # print(t_glob)
        # print(t_globm)

        # Save diagnostic outputs if requested
        if save_diagnostics:
            self.diagnostics["X_features"] = X.copy()
            self.diagnostics["t_glob"] = t_glob.copy()
            self.diagnostics["time"] = time.copy()
            print("   📊 Diagnostic outputs saved:")
            print(f"      X_features shape: {X.shape}")
            print(f"      t_glob shape: {t_glob.shape}")
            print(f"      time shape: {time.shape}")

        # Prepare data for seasonal cycle fitting
        Y_xr = (  # pylint: disable=invalid-name
            ds[variable_name].mean(dim=["ens"]).stack(space=("lat", "lon"))
        )
        Y = Y_xr.data  # pylint: disable=invalid-name

        # Fit seasonal cycle model
        self.seasonal_model = SeasonalModel()
        self.seasonal_model.fit(time, t_glob, Y)

        # Save additional diagnostic outputs if requested
        if save_diagnostics:
            self.diagnostics["seasonal_coef"] = self.seasonal_model.coef_.copy()
            self.diagnostics["seasonal_intercept"] = (
                self.seasonal_model.intercept_.copy()
            )
            self.diagnostics["Y_data"] = Y.copy()
            print("   📊 Seasonal model diagnostics saved:")
            print(
                f"      Coefficients shape: {self.seasonal_model.coef_.shape} (gridpoints × features)"
            )
            print(f"      Intercept shape: {self.seasonal_model.intercept_.shape}")
            print(f"      Y data shape: {Y.shape} (time × gridpoints)")

        # Reconstruct seasonal cycle
        seasonal_cycle_fit = self.seasonal_model.predict(time, t_glob)
        seasonal_cycle_fit_xr = xr.DataArray(
            seasonal_cycle_fit, coords=Y_xr.coords, dims=Y_xr.dims
        ).unstack("space")

        # Calculate anomalies
        if picontrol_baseline is not None:
            anomalies = (
                ds[variable_name].mean(dim=["ens"])
                - seasonal_cycle_fit_xr
                - picontrol_baseline
            )
        else:
            anomalies = ds[variable_name].mean(dim=["ens"]) - seasonal_cycle_fit_xr

        # Fit PCA to anomalies
        anomalies_flat = anomalies.stack(space=("lat", "lon")).data
        self.pca = PCA(n_components=self.n_modes)
        pcs = self.pca.fit_transform(anomalies_flat)

        # Fit VARX model to PCs
        X_exog = SeasonalModel.extract_exog_variables(X, self.use_exog)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_var = VAR(endog=pcs, exog=X_exog)
            self.varx_results = model_var.fit(self.lag_order)

        # Store coordinate information
        self.coords = {
            "lat": ds.coords["lat"],
            "lon": ds.coords["lon"],
            "month": ds.coords["month"],
        }
        self._fix_coords_to_np()
        # Store variable name for future reference
        self.variable_name = variable_name
        self.fitted = True

        # Compute seasonal model R² (variance explained by
        # temperature-dependent harmonics)
        seasonal_r2 = self.seasonal_model.score(time, t_glob, Y)
        pca_var_explained = self.pca.explained_variance_ratio_.sum()

        # Total variance explained = seasonal component +
        # (remaining fraction × PCA)
        total_var_explained = seasonal_r2 + (1 - seasonal_r2) * pca_var_explained

        # Store for access
        self.diagnostics["seasonal_r2"] = seasonal_r2
        self.diagnostics["total_variance_explained"] = total_var_explained

        if verbose:
            print("Noise generator fitted successfully.")
            print(f"   - PCA modes: {self.n_modes}")
            print(f"   - Seasonal model R²: {seasonal_r2:.2%}")
            print(f"   - Anomaly variance explained (PCA): {pca_var_explained:.2%}")
            print(f"   - Total variance explained: {total_var_explained:.2%}")
            print(f"   - VARX lag order: {self.lag_order}")
            if self.use_exog == "all":
                print("   - Exogenous vars: t_glob, annual_cos, annual_sin")
            elif self.use_exog == "temp_only":
                print("   - Exogenous vars: t_glob only")
            else:
                print("   - Exogenous vars: none (pure VAR)")

        # Resolve Student-t degrees of freedom (after VARX is fitted)
        if self.noise_pc_distribution == "t":
            self._resolve_df(verbose=verbose)

    # ------------------------------------------------------------------
    # Student-t helpers
    # ------------------------------------------------------------------

    def _resolve_df(self, verbose: bool = False):
        """
        Resolve the Student-t degrees of freedom after VARX fitting.

        The result ``_fitted_df`` is always a 1-D array of shape
        ``(n_modes,)`` so that each principal component can have its own
        tail weight.

        If t_df is None, uses default 10.0 for every PC. If a float,
        broadcasts to every PC. If 'mle', performs per-PC univariate
        MLE on VARX residuals.
        """
        if self.t_df is None:
            self._fitted_df = np.full(self.n_modes, 10.0)
            if verbose:
                print(f"   - Student-t df: 10.0 for all {self.n_modes} PCs (default)")
        elif isinstance(self.t_df, (int, float)) and not isinstance(self.t_df, bool):
            self._fitted_df = np.full(self.n_modes, float(self.t_df))
            if verbose:
                print(f"   - Student-t df: {float(self.t_df)} for all {self.n_modes} PCs (user-specified)")
        elif self.t_df == "mle":
            self._fitted_df = self._select_df_mle(verbose=verbose)
        else:
            raise ValueError(
                f"t_df must be None, a number, or 'mle', got {self.t_df!r}"
            )

    def _select_df_mle(self, verbose: bool = False) -> np.ndarray:
        """
        Select optimal Student-t df **per PC** via univariate MLE on VARX residuals.

        For each principal component, fits a univariate Student-t distribution
        to the VARX residual time series using ``scipy.stats.t.fit()`` and
        returns the MLE df for every PC.

        Parameters
        ----------
        verbose : bool
            Print search results

        Returns
        -------
        np.ndarray, shape (n_modes,)
            Optimal degrees of freedom per principal component
        """
        from scipy.stats import t as scipy_t_dist  # noqa: F811

        residuals = self.varx_results.resid  # (n_obs, n_modes)
        n_modes = residuals.shape[1]
        fitted_dfs = np.zeros(n_modes)

        for j in range(n_modes):
            r = residuals[:, j]
            # scipy.stats.t.fit returns (df, loc, scale)
            df_fit, _, _ = scipy_t_dist.fit(r)
            # Clamp to reasonable range
            df_fit = max(df_fit, 2.1)
            fitted_dfs[j] = df_fit

        if verbose:
            print(f"   - Student-t df per PC (MLE):")
            print(f"     median={np.median(fitted_dfs):.1f}, "
                  f"min={fitted_dfs.min():.1f} (PC {fitted_dfs.argmin()}), "
                  f"max={fitted_dfs.max():.1f} (PC {fitted_dfs.argmax()})")
            n_heavy = (fitted_dfs < 15).sum()
            print(f"     {n_heavy}/{n_modes} PCs with df < 15")
            # Variance inflation: Var(t_df) = df/(df-2) vs Var(normal)=1
            var_inflation = fitted_dfs / (fitted_dfs - 2)
            weights = self.pca.explained_variance_ratio_
            weighted_inflation = np.average(var_inflation, weights=weights)
            print(f"   - Variance inflation df/(df-2): "
                  f"weighted mean={weighted_inflation:.2f}, "
                  f"max={var_inflation.max():.2f} (PC {var_inflation.argmax()})")

        return fitted_dfs

    # ------------------------------------------------------------------
    # PC generation
    # ------------------------------------------------------------------

    def generate_stochastic_pcs(
        self,
        global_temp_trajectory,
        n_realizations: int = 1,
        random_seed=None,
    ) -> np.ndarray:
        """
        Generate stochastic principal component time series.

        This method generates only the stochastic PC loadings, which can be
        used to reconstruct either gridded fields or regional/global means.
        This enables self-consistent ensemble generation across different
        spatial aggregations.

        Parameters
        ----------
        global_temp_trajectory : array-like
            Global temperature trajectory (used for exogenous variables in VARX model)
        n_realizations : int, default 1
            Number of realizations to generate
        random_seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Stochastic PC time series with shape:
            - (n_time, n_modes) if n_realizations == 1
            - (n_realizations, n_time, n_modes) if n_realizations > 1

        Examples
        --------
        >>> # Generate PCs once, use for multiple outputs
        >>> pcs = model.generate_stochastic_pcs(monthly_warming, n_realizations=100)
        >>> global_means = model.generate_regional_mean_realizations(
        ...     monthly_warming, region='global', stochastic_pcs=pcs)
        >>> neu_means = model.generate_regional_mean_realizations(
        ...     monthly_warming, region='NEU', stochastic_pcs=pcs)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before generating realizations")

        if random_seed is not None:
            np.random.seed(random_seed)

        n_time = len(global_temp_trajectory)
        time = np.arange(n_time)

        # Create exogenous variables for VARX
        X = SeasonalModel.create_harmonic_features(time, global_temp_trajectory)
        X_exog = SeasonalModel.extract_exog_variables(X, self.use_exog)

        # Generate stochastic PCs for each realization
        if n_realizations == 1:
            return self._generate_stochastic_pcs(X_exog, n_time)
        all_pcs = []
        for _ in range(n_realizations):
            pcs = self._generate_stochastic_pcs(X_exog, n_time)
            all_pcs.append(pcs)
        return np.array(all_pcs)  # Shape: (n_realizations, n_time, n_modes)

    # pylint: disable=too-many-locals
    def generate_realization(
        self,
        global_temp_trajectory,
        n_realizations: int = 1,
        random_seed=None,
        noise_only: bool = False,
        add_base=None,
        **kwargs,
    ):
        """
        Generate stochastic climate realizations.

        Parameters
        ----------
        global_temp_trajectory : array-like
            Global temperature trajectory to drive the seasonal cycle.
            If noise_only=True, this can be any length array (values ignored for temperature effects).
        n_realizations : int, default 1
            Number of realizations to generate
        random_seed : int, optional
            Random seed for reproducibility
        noise_only : bool, default False
            If True, generate only the stochastic noise component without direct temperature
            effects or constant terms, but preserve temperature-modulated seasonal harmonics.
            This is useful for adding to METEOR annual predictions.
        add_base : xr.DataArray, optional
            Base climatology to add to each realization. If provided, the addition is done
            efficiently in NumPy before XArray conversion, avoiding expensive XArray operations.
            Must have compatible shape with the output.

        Returns
        -------
        xr.DataArray or list of xr.DataArray
            Generated climate realizations. If noise_only=True, returns just the
            stochastic component that can be added to other predictions.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before generating realizations")

        if random_seed is not None:
            np.random.seed(random_seed)

        # Create time coordinate (shared across all realizations)
        n_time = len(global_temp_trajectory)
        time = np.arange(n_time)

        # Precompute harmonic features (shared across all realizations)
        X = SeasonalModel.create_harmonic_features(time, global_temp_trajectory)
        X_exog = SeasonalModel.extract_exog_variables(X, self.use_exog)

        # Precompute seasonal cycle as NumPy array (shared across all realizations)
        if noise_only:
            # For noise-only: keep seasonal harmonics AND temperature-modulated harmonics
            # but remove the direct temperature effect (intercept + t_glob term)
            seasonal_cycle = self.seasonal_model.predict(time, global_temp_trajectory)

            # Calculate what to subtract (intercept + direct temperature effect)
            intercept_effect = self.seasonal_model.intercept_
            temp_effect = (
                self.seasonal_model.coef_[:, 0] * global_temp_trajectory[:, np.newaxis]
            )

            # Remove intercept and direct temperature effect from seasonal cycle
            seasonal_cycle_np = (
                seasonal_cycle - intercept_effect[np.newaxis, :] - temp_effect
            )
        else:
            # Standard operation: full seasonal cycle with temperature dependence
            seasonal_cycle_np = self.seasonal_model.predict(
                time, global_temp_trajectory
            )

        # Reshape seasonal cycle once
        n_lat = len(self.coords["lat"])
        n_lon = len(self.coords["lon"])
        seasonal_cycle_reshaped = seasonal_cycle_np.reshape(n_time, n_lat, n_lon)

        # Convert base climatology to NumPy if provided (once for all realizations)
        base_clim_np = None
        if add_base is not None:
            # Handle potential ensemble dimension and squeeze it
            base_values = add_base.values
            if base_values.ndim == 4:
                # Shape is (ens, time, lat, lon) - squeeze out ensemble dimension
                base_values = base_values.squeeze()

            # Now reshape to (n_time, n_lat, n_lon)
            # If the time dimension doesn't match, select the first n_time steps
            if base_values.shape[0] != n_time:
                base_values = base_values[:n_time, :, :]

            base_clim_np = base_values.reshape(n_time, n_lat, n_lon)

        # Generate realizations (only stochastic component varies)
        realizations = []
        for _ in range(n_realizations):
            # Generate stochastic component (this is the only unique part per realization)
            synthetic_pcs = self._generate_stochastic_pcs(X_exog, n_time)

            # Reconstruct anomalies (NumPy)
            reconstructed_anomalies = synthetic_pcs @ self.pca.components_
            reconstructed_anomalies_reshaped = reconstructed_anomalies.reshape(
                n_time, n_lat, n_lon
            )

            # Combine seasonal cycle and anomalies in NumPy (FAST!)
            realization_np = seasonal_cycle_reshaped + reconstructed_anomalies_reshaped

            # Add base climatology in NumPy if provided (FAST!)
            if base_clim_np is not None:
                realization_np = realization_np + base_clim_np

            # Convert to xarray only once at the end
            realization_xr = xr.DataArray(
                realization_np,
                coords={
                    "month": time,
                    "lat": self.coords["lat"],
                    "lon": self.coords["lon"],
                },
                dims=("month", "lat", "lon"),
            )
            realizations.append(realization_xr)

        return realizations if n_realizations > 1 else realizations[0]

    # pylint: disable=invalid-name
    def _generate_stochastic_pcs(
        self, X_exog: Optional[np.ndarray], n_time: int
    ) -> np.ndarray:
        """
        Generate stochastic principal components using fitted VARX model.

        This optimized implementation uses batched random generation
        (generating all random shocks at once) and manual VAR time loop
        instead of repeatedly calling statsmodels forecast() which has
        significant overhead from redundant SVD decompositions.

        When ``noise_pc_distribution='t'``, the VAR is still driven by
        Gaussian innovations (preserving the autoregressive dynamics),
        but the *output* PCs are scaled per time step by
        ``sqrt(df / V_t)`` where ``V_t ~ chi2(df)``.  This gives the
        output field multivariate Student-t marginals while preserving
        the temporal correlation structure.

        Parameters
        ----------
        X_exog : np.ndarray or None
            Exogenous variables for VARX model (n_time, n_exog), or None for pure VAR
        n_time : int
            Number of time steps to generate

        Returns
        -------
        np.ndarray
            Generated principal components (n_time, n_modes)
        """
        # Extract coefficient matrices from fitted VARX model
        params = self.varx_results.params
        n_exog = X_exog.shape[1] if X_exog is not None else 0

        # Intercept (n_modes,)
        intercept = params[0, :]

        # Lag coefficient matrices A₁, A₂, ... (each n_modes × n_modes)
        A_matrices = []
        for lag_i in range(self.lag_order):
            start_idx = 1 + lag_i * self.n_modes
            end_idx = start_idx + self.n_modes
            A_matrices.append(params[start_idx:end_idx, :].T)

        # Exogenous coefficient matrix B (n_modes × n_exog)
        B_matrix = params[-n_exog:, :].T if n_exog else None

        # Residual covariance matrix Σ (n_modes × n_modes)
        residual_cov = self.varx_results.sigma_u

        # 🚀 KEY OPTIMIZATION: Pre-generate ALL random shocks at once
        # This eliminates 97% of the bottleneck (4,212 separate MVN calls → 1 batched call)
        # Always use normal innovations for the VAR loop (even for Student-t output)
        mean_shock = np.zeros(self.n_modes)
        all_shocks = np.random.multivariate_normal(
            mean_shock, residual_cov, size=n_time
        )

        # Initialize synthetic PCs with zero initial conditions
        synthetic_pcs = np.zeros((n_time, self.n_modes))
        synthetic_pcs[: self.lag_order] = 0

        # Time loop (still needed for autoregressive structure)
        # VAR equation: y_t = intercept + A₁y_{t-1} + A₂y_{t-2} + ... + B·x_t + ε_t
        for t in range(self.lag_order, n_time):
            # Start with intercept
            forecast = intercept.copy()

            # Add lag contributions: A₁y_{t-1} + A₂y_{t-2} + ...
            for lag_i in range(self.lag_order):
                y_lag = synthetic_pcs[t - lag_i - 1]
                forecast += A_matrices[lag_i] @ y_lag

            # Add exogenous contribution: B·x_t (if using exogenous variables)
            if B_matrix is not None and X_exog is not None:
                forecast += B_matrix @ X_exog[t]

            # Add pre-generated random shock (no MVN call here!)
            synthetic_pcs[t] = forecast + all_shocks[t]

        # Apply per-PC, per-timestep Student-t scaling to the output PCs.
        # Each PC gets its own df (fitted via MLE), so PCs with heavier tails
        # in the training data get more tail inflation.
        # Construction: y_{t,j}^(t) = sqrt(df_j / V_{t,j}) * y_{t,j}^(normal)
        #   where V_{t,j} ~ chi2(df_j) independently for each (t, j).
        if self.noise_pc_distribution == "t" and self._fitted_df is not None:
            synthetic_pcs = _apply_student_t_posthoc_scaling(
                synthetic_pcs, self._fitted_df
            )

        return synthetic_pcs

    # TODO check if we can use the weights calculator from geo_data_utils.py
    def _compute_spatial_weights(self) -> np.ndarray:
        """Compute area-weighted spatial averaging weights (cosine of latitude)."""
        return np.cos(np.deg2rad(self.coords["lat"]))

    def _find_nearest_gridpoint(
        self, target_lat: float, target_lon: float
    ):
        """
        Find the nearest gridpoint to the target latitude and longitude.

        Parameters
        ----------
        target_lat : float
            Target latitude in degrees
        target_lon : float
            Target longitude in degrees (0-360 or -180 to 180)

        Returns
        -------
        tuple
            (lat_idx, lon_idx) indices of the nearest gridpoint
        """
        lats = self.coords["lat"]
        lons = self.coords["lon"]

        # Normalize longitude to 0-360 range
        target_lon = target_lon % 360
        lons_normalized = lons % 360

        # Find nearest latitude
        # print(lats)
        # print(target_lat)
        lat_idx = np.argmin(np.abs(lats - target_lat))

        # Find nearest longitude
        lon_idx = np.argmin(np.abs(lons_normalized - target_lon))

        return lat_idx, lon_idx

    def _get_point_eof_values(self, lat: float, lon: float) -> np.ndarray:
        """
        Get EOF values at a specific point (no averaging).

        Cached in memory since EOFs are model-invariant.

        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees

        Returns
        -------
        np.ndarray
            EOF values at the point, shape (n_modes,)
        """
        # Create cache key
        point_id = f"point_{lat:.2f}_{lon:.2f}"
        if point_id in self._regional_eof_projections:
            return self._regional_eof_projections[point_id]

        # Find nearest gridpoint
        lat_idx, lon_idx = self._find_nearest_gridpoint(lat, lon)

        # Get EOFs reshaped to spatial grid
        n_lat = len(self.coords["lat"])
        n_lon = len(self.coords["lon"])
        eof_components = self.pca.components_.reshape(self.n_modes, n_lat, n_lon)

        # Extract values at the point (no averaging needed)
        eof_point_values = eof_components[:, lat_idx, lon_idx]

        # Cache and return
        self._regional_eof_projections[point_id] = eof_point_values
        return eof_point_values

    # TODO region masking and averaging from geo_data_utils.py could be reused here?

    def _get_regional_eof_projection(
        self, region: str, region_mask=None
    ) -> np.ndarray:
        """
        Get or compute the spatial mean projection of each EOF for a region.

        Cached in memory since EOFs are model-invariant.

        Parameters
        ----------
        region : str
            Region identifier ('global' or AR6 region code like 'NEU')
        region_mask : np.ndarray, optional
            Custom 2D boolean mask (n_lat, n_lon) for the region

        Returns
        -------
        np.ndarray
            Mean projection of each EOF mode for the region, shape (n_modes,)
        """
        # Check cache first
        region_id = (
            region if region_mask is None else f"custom_{id(region_mask)}"
        )
        if region_id in self._regional_eof_projections:
            return self._regional_eof_projections[region_id]

        # Compute EOF projections
        n_lat = len(self.coords["lat"])
        n_lon = len(self.coords["lon"])

        # Get EOFs reshaped to spatial grid (n_modes, n_lat, n_lon)
        eof_components = self.pca.components_.reshape(self.n_modes, n_lat, n_lon)
        if region_mask is None and region != "global":
            region_mask = self._get_ar6_region_mask(region)

        eof_projections = self._weighted_mean_over_region(
            eof_components, None, None, region_mask, region
        )

        # Cache and return
        self._regional_eof_projections[region_id] = eof_projections
        return eof_projections

    # ------------------------------------------------------------------
    # Regional mean generation
    # ------------------------------------------------------------------

    # pylint: disable=too-many-locals,too-many-branches
    def generate_regional_mean_realizations(
        self,
        global_temp_trajectory,
        region: str = "global",
        region_mask=None,
        lat=None,
        lon=None,
        n_realizations: int = 1,
        random_seed=None,
        noise_only: bool = False,
        add_base=None,
        stochastic_pcs=None,
        return_numpy: bool = False,
    ):
        """
        Generate regional/global mean or point-scale realizations efficiently.

        This method avoids creating full 3D gridded fields by computing the
        output directly from the PC projections. This is orders of magnitude
        faster when only scalar time series are needed.

        Parameters
        ----------
        global_temp_trajectory : array-like
            Global temperature trajectory to drive the seasonal cycle
        region : str, default 'global'
            Region identifier: 'global' or AR6 region code (e.g., 'NEU', 'WNA').
            Ignored if lat/lon are provided.
        region_mask : np.ndarray, optional
            Custom 2D boolean mask (n_lat, n_lon) for region. If provided, overrides `region`.
            Ignored if lat/lon are provided.
        lat : float, optional
            Latitude for point extraction (degrees). If provided with `lon`, extracts
            time series at the nearest gridpoint instead of computing regional mean.
        lon : float, optional
            Longitude for point extraction (degrees, 0-360 or -180 to 180).
            Must be provided together with `lat`.
        n_realizations : int, default 1
            Number of realizations to generate
        random_seed : int, optional
            Random seed for reproducibility
        noise_only : bool, default False
            If True, generate only stochastic component (for adding to predictions)
        add_base : xr.DataArray or np.ndarray, optional
            Base climatology to add. Can be:
            - Scalar time series (n_time,) - will be added directly
            - Gridded field (n_time, n_lat, n_lon) - will be spatially averaged/extracted at point
        stochastic_pcs : np.ndarray, optional
            Pre-generated stochastic PCs from generate_stochastic_pcs().
            If provided, these PCs are used (enabling self-consistent multi-region generation).
            Shape: (n_time, n_modes) or (n_realizations, n_time, n_modes)
        return_numpy : bool, default False
            If True, return numpy arrays. If False, return xarray DataArrays.

        Returns
        -------
        xr.DataArray or np.ndarray
            Regional/global mean or point-scale time series.

            - If n_realizations == 1:
              Shape (n_time,) with dims ('month',)
            - If n_realizations > 1:
              Shape (n_realizations, n_time) with dims ('realization', 'month')

            When return_numpy=False (default), returns xarray DataArray with proper
            coordinates and dims. When return_numpy=True, returns numpy array.

        Examples
        --------
        >>> # Fast generation of 100 global mean realizations
        >>> global_means = model.generate_regional_mean_realizations(
        ...     monthly_warming, region='global', n_realizations=100)
        >>> # Returns shape (100, n_time) with dims ('realization', 'month')
        >>>
        >>> # Point-scale generation (e.g., New York City: 40.7°N, 74°W = 286°E)
        >>> nyc_temps = model.generate_regional_mean_realizations(
        ...     monthly_warming, lat=40.7, lon=286, n_realizations=100)
        >>>
        >>> # Self-consistent multi-location generation
        >>> pcs = model.generate_stochastic_pcs(monthly_warming, n_realizations=50)
        >>> global_m = model.generate_regional_mean_realizations(
        ...     monthly_warming, region='global', stochastic_pcs=pcs)
        >>> london = model.generate_regional_mean_realizations(
        ...     monthly_warming, lat=51.5, lon=0, stochastic_pcs=pcs)
        >>> # Global mean and London share the same stochastic variability
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before generating realizations")

        # Validate lat/lon parameters
        if (lat is None) != (lon is None):
            raise ValueError(
                "Both lat and lon must be provided together for point extraction"
            )

        if random_seed is not None and stochastic_pcs is None:
            np.random.seed(random_seed)

        n_time = len(global_temp_trajectory)
        time = np.arange(n_time)

        # Get EOF projections/values (cached)
        if lat is not None and lon is not None:
            # Point extraction mode
            eof_projections = self._get_point_eof_values(lat, lon)
            location_type = "point"
            location_id = f"{lat:.2f}N_{lon:.2f}E"
        else:
            # Regional mean mode
            eof_projections = self._get_regional_eof_projection(region, region_mask)
            location_type = "region"
            location_id = region

        # Compute seasonal cycle regional mean
        X = SeasonalModel.create_harmonic_features(time, global_temp_trajectory)

        if noise_only:
            # For noise-only: seasonal harmonics without direct temperature effect
            seasonal_cycle = self.seasonal_model.predict(time, global_temp_trajectory)
            intercept_effect = self.seasonal_model.intercept_
            temp_effect = (
                self.seasonal_model.coef_[:, 0] * global_temp_trajectory[:, np.newaxis]
            )
            seasonal_cycle_full = (
                seasonal_cycle - intercept_effect[np.newaxis, :] - temp_effect
            )
        else:
            seasonal_cycle_full = self.seasonal_model.predict(
                time, global_temp_trajectory
            )

        # Compute seasonal mean (point extraction or regional average)
        n_lat = len(self.coords["lat"])
        n_lon = len(self.coords["lon"])
        seasonal_cycle_grid = seasonal_cycle_full.reshape(n_time, n_lat, n_lon)

        if lat is None and lon is None and region != "global" and region_mask is None:
            # Get mask from AR6 regions
            region_mask = self._get_ar6_region_mask(region)

        seasonal_mean = self._weighted_mean_over_region(
            seasonal_cycle_grid, lat, lon, region_mask, region
        )
        # Compute base climatology (if provided)
        base_mean = None
        if add_base is not None:
            if isinstance(add_base, xr.DataArray):
                add_base = add_base.values

            if add_base.ndim == 1:
                # Already a time series
                base_mean = add_base
            elif add_base.ndim == 3:
                # Gridded field - extract point or compute regional mean
                base_mean = self._weighted_mean_over_region(
                    add_base, lat, lon, region_mask, region
                )

        # Generate or use provided stochastic PCs
        if stochastic_pcs is None:
            X_exog = SeasonalModel.extract_exog_variables(X, self.use_exog)
            if n_realizations == 1:
                pcs_to_use = [self._generate_stochastic_pcs(X_exog, n_time)]
            else:
                pcs_to_use = [
                    self._generate_stochastic_pcs(X_exog, n_time)
                    for _ in range(n_realizations)
                ]
        else:
            # Use provided PCs
            if stochastic_pcs.ndim == 2:
                # Single realization
                pcs_to_use = [stochastic_pcs]
            else:
                # Multiple realizations
                pcs_to_use = list(stochastic_pcs)

        # Reconstruct regional means from PCs
        realizations = []
        for pcs in pcs_to_use:
            # Anomaly contribution: PCs @ EOF_projections
            anomaly_mean = pcs @ eof_projections  # (n_time,)

            # Combine components
            realization = seasonal_mean + anomaly_mean
            if base_mean is not None:
                realization = realization + base_mean

            realizations.append(realization)

        # Return format numpy array
        if return_numpy:
            # Return as numpy array with shape (n_realizations, n_time) or (n_time,) if single
            if len(realizations) == 1:
                return realizations[0]
            return np.array(realizations)

        # Else xarray dataset or dataArray, so build attributes
        attrs = {location_type: location_id}
        if lat is not None and lon is not None:
            attrs["latitude"] = lat
            attrs["longitude"] = lon

        # Return as xarray DataArray
        if len(realizations) == 1:
            # Single realization - return 1D DataArray
            return xr.DataArray(
                realizations[0],
                coords={"month": time},
                dims=("month",),
                attrs=attrs,
            )
        # Multiple realizations - concatenate with 'realization' dimension
        return xr.DataArray(
            np.array(realizations),
            coords={
                "realization": np.arange(len(realizations)),
                "month": time,
            },
            dims=("realization", "month"),
            attrs=attrs,
        )

    # ------------------------------------------------------------------
    # Spatial averaging utilities
    # ------------------------------------------------------------------

    def _weighted_mean_over_region(
        self, data: np.ndarray, lat, lon, region_mask, region: str
    ) -> np.ndarray:
        """
        Compute weighted mean over a region or point extraction.

        Parameters
        ----------
        data : np.ndarray
            Input data with shape (n_time, n_lat, n_lon)
        lat : float, optional
            Latitude for point extraction (degrees)
        lon : float, optional
            Longitude for point extraction (degrees)
        region_mask : np.ndarray, optional
            Custom 2D boolean mask (n_lat, n_lon) for region
        Returns
        -------
        np.ndarray
            Weighted mean time series with shape (n_time,)
        """
        # Point extraction
        if lat is not None and lon is not None:
            lat_idx, lon_idx = self._find_nearest_gridpoint(lat, lon)
            return data[:, lat_idx, lon_idx]

        # Regional or global mean, start by computing area weights
        weights = self._compute_spatial_weights()
        weight_grid = weights[:, np.newaxis]
        if region == "global" and region_mask is None:
            # Global mean
            total_weight = np.sum(weights) * data.shape[2]
            return np.array(
                [
                    np.sum(data[t] * weight_grid) / total_weight
                    for t in range(data.shape[0])
                ]
            )
        # Regional mean
        mean_values = np.zeros(data.shape[0])
        for t in range(data.shape[0]):
            masked_data = np.where(region_mask, data[t], np.nan)
            masked_weights = np.where(region_mask, weight_grid, 0)
            mean_values[t] = np.nansum(masked_data * masked_weights) / np.sum(
                masked_weights
            )
        return mean_values

    def _get_ar6_region_mask(self, region: str) -> np.ndarray:
        """
        Get AR6 region mask for the model grid.

        Parameters
        ----------
        region : str
            AR6 region code (e.g., 'NEU', 'WNA')

        Returns
        -------
        np.ndarray
            2D boolean mask (n_lat, n_lon) for the region
        """
        ar6_regions = regionmask.defined_regions.ar6.all

        # Find region number
        region_number = None
        for r in ar6_regions:
            if r.abbrev == region:
                region_number = r.number
                break

        if region_number is None:
            raise ValueError(f"AR6 region '{region}' not found")

        # Create mask on this grid
        lons = self.coords["lon"]

        lats = self.coords["lat"]

        lon_2d, lat_2d = np.meshgrid(lons, lats)
        mask_3d = ar6_regions.mask(lon_2d, lat_2d)
        region_mask = mask_3d == region_number
        return region_mask

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_model(self, filepath: str):
        """
        Serialize the fitted model to *filepath* (pickle).

        Parameters
        ----------
        filepath : str
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "model_type": self.model_type,
            "n_modes": self.n_modes,
            "lag_order": self.lag_order,
            "use_exog": self.use_exog,
            "noise_pc_distribution": self.noise_pc_distribution,
            "t_df": self.t_df,
            "_fitted_df": self._fitted_df,
            "seasonal_model": self.seasonal_model,
            "pca": self.pca,
            "varx_results": self.varx_results,
            "coords": self.coords,
            "fitted": self.fitted,
            "variable_name": self.variable_name,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load_model_from_file(cls, filepath: str) -> "PCAVARXNoiseModel":
        """
        Deserialize a :class:`PCAVARXNoiseModel` from *filepath*.

        Backward-compatible: handles old pickle files where
        ``seasonal_model`` is a bare ``sklearn.LinearRegression``.

        Parameters
        ----------
        filepath : str

        Returns
        -------
        PCAVARXNoiseModel
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)  # nosec - Loading trusted model files only

        obj = cls.__new__(cls)
        obj.n_modes = model_data["n_modes"]
        obj.lag_order = model_data["lag_order"]
        obj.use_exog = model_data.get("use_exog", "all")
        obj.noise_pc_distribution = model_data.get("noise_pc_distribution", "normal")
        obj.t_df = model_data.get("t_df", None)
        obj._fitted_df = model_data.get("_fitted_df", None)
        obj.pca = model_data["pca"]
        obj.varx_results = model_data["varx_results"]
        obj.coords = model_data["coords"]
        obj.fitted = model_data["fitted"]
        obj.variable_name = model_data.get("variable_name", None)
        obj._regional_eof_projections = {}

        # Diagnostics may be absent in older pickles
        obj.diagnostics = {
            "X_features": None,
            "t_glob": None,
            "time": None,
            "seasonal_r2": None,
            "total_variance_explained": None,
            "seasonal_coef": None,
            "seasonal_intercept": None,
            "Y_data": None,
        }

        # Backward compat: old pickles stored a bare LinearRegression
        seasonal_raw = model_data["seasonal_model"]
        if isinstance(seasonal_raw, LinearRegression):
            obj.seasonal_model = SeasonalModel._from_sklearn_lr(seasonal_raw)
        else:
            obj.seasonal_model = seasonal_raw

        obj._fix_coords_to_np()
        print(f"Model loaded from {filepath}")
        return obj
