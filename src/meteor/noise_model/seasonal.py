
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

VALID_USE_EXOG = ("all", "temp_only", "none")


class SeasonalModel:
    """
    Temperature-dependent harmonic seasonal model for METEOR noise models.

    The `SeasonalModel` is a shared, composable component that all METEOR
    noise-model variants can use for the seasonal pre-processing step.
    It fits one ``sklearn.LinearRegression`` per call to :meth:`fit` using
    the 9-column design matrix produced by :meth:`create_harmonic_features`.
    All fitted state lives inside the wrapped ``LinearRegression``, so the
    model is trivially pickle-compatible. The model is of the form

    .. math::

        X(t, x, y)
        =
        \\beta_0
        +
        \\sum_j \\beta_j \\, \\varphi_j(t, T_{\\text{glob}})
        +
        \\varepsilon(t, x, y),

    where the features :math:`\\varphi_j` are defined as 9 variables:
    the global temperature :math:`T_{\\text{glob}}`, annual and semiannual
    cosine/sine harmonics, and their pairwise interactions with
    :math:`T_{\\text{glob}}`.

    Attributes
    ----------
    fitted : bool
        ``True`` after :meth:`fit` has been called successfully.

    Notes
    -----
    The seasonal representation uses 9 features:

    - :math:`T_{\\text{glob}}`
    - annual cosine harmonic
    - annual sine harmonic
    - semiannual cosine harmonic
    - semiannual sine harmonic
    - pairwise interactions of each harmonic with :math:`T_{\\text{glob}}`

    These features are intended for use both inside `SeasonalModel` and as
    exogenous variables in other models.

    Methods
    -------
    create_harmonic_features(t, T_glob, ...)
        Construct the 9 temperature-dependent harmonic features used in the
        seasonal linear regression.
    extract_exog_variables(data, ...)
        Extract and assemble the exogenous variable matrix (e.g., for VARX)
        from the harmonic features.

    See Also
    --------
    SeasonalModel.create_harmonic_features : Standalone helper for feature construction.
    SeasonalModel.extract_exog_variables : Standalone helper for exogenous matrices.
    """

    def __init__(self):
        self._lr: LinearRegression | None = None
        self.fitted: bool = False

    # ------------------------------------------------------------------
    # Stateless helpers (no instance state required)
    # ------------------------------------------------------------------

    @staticmethod
    def create_harmonic_features(
        time: np.ndarray,
        t_glob: np.ndarray,
    ) -> np.ndarray:
        """
        Build the 9-column design matrix from time indices and global temperature.

        Column layout (0-indexed):

        =====  ==========================================
        Index  Feature
        =====  ==========================================
        0      T_glob
        1      cos(2πt/12)   — annual cosine
        2      sin(2πt/12)   — annual sine
        3      cos(4πt/12)   — semiannual cosine
        4      sin(4πt/12)   — semiannual sine
        5      T_glob · cos(2πt/12)
        6      T_glob · sin(2πt/12)
        7      T_glob · cos(4πt/12)
        8      T_glob · sin(4πt/12)
        =====  ==========================================

        Parameters
        ----------
        time : np.ndarray, shape (n_time,)
            Integer month indices (0-based).
        t_glob : np.ndarray, shape (n_time,)
            Smoothed global-mean temperature trajectory.

        Returns
        -------
        np.ndarray, shape (n_time, 9)
            Design matrix X with harmonic features and interactions

        """
        months_per_year = 12
        annual_cos = np.cos(2 * np.pi * time / months_per_year)
        annual_sin = np.sin(2 * np.pi * time / months_per_year)
        semiannual_cos = np.cos(4 * np.pi * time / months_per_year)
        semiannual_sin = np.sin(4 * np.pi * time / months_per_year)

        return np.vstack(
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

    @staticmethod
    def extract_exog_variables(
        X: np.ndarray,
        use_exog: str,
    ) -> np.ndarray | None:
        """
        Slice the exogenous columns from a feature matrix for a VARX model.

        Parameters
        ----------
        X : np.ndarray, shape (n_time, 9)
            Output of :meth:`create_harmonic_features`.
        use_exog : str
            One of:

            * ``'all'``       — T_glob + annual cos/sin (columns 0-2)
            * ``'temp_only'`` — T_glob only (column 0)
            * ``'none'``      — no exogenous variables; returns ``None``

        Returns
        -------
        np.ndarray or None
        """
        if use_exog == "all":
            return X[:, :3]  # T_glob, annual_cos, annual_sin
        if use_exog == "temp_only":
            return X[:, :1]  # T_glob only
        if use_exog == "none":
            return None
        raise ValueError(
            f"Invalid use_exog value: '{use_exog}'. "
            f"Must be one of {VALID_USE_EXOG}."
        )

    # ------------------------------------------------------------------
    # Fit / predict interface
    # ------------------------------------------------------------------

    def fit(
        self,
        time: np.ndarray,
        t_glob: np.ndarray,
        Y_flat: np.ndarray,
    ) -> "SeasonalModel":
        """
        Fit the harmonic seasonal model to a flattened climate field.

        Parameters
        ----------
        time : np.ndarray, shape (n_time,)
            Integer month indices.
        t_glob : np.ndarray, shape (n_time,)
            Smoothed global-mean temperature.
        Y_flat : np.ndarray, shape (n_time, n_space)
            Climate field with lat×lon stacked into a single spatial axis.

        Returns
        -------
        SeasonalModel
            ``self``, for method chaining.
        """
        X = self.create_harmonic_features(time, t_glob)
        self._lr = LinearRegression(fit_intercept=True)
        self._lr.fit(X, Y_flat)
        self.fitted = True
        return self

    def predict(
        self,
        time: np.ndarray,
        t_glob: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the seasonal cycle for a given time / temperature trajectory.

        Parameters
        ----------
        time : np.ndarray, shape (n_time,)
        t_glob : np.ndarray, shape (n_time,)

        Returns
        -------
        np.ndarray, shape (n_time, n_space)
        """
        self._check_fitted("predict")
        X = self.create_harmonic_features(time, t_glob)
        return self._lr.predict(X)

    def score(
        self,
        time: np.ndarray,
        t_glob: np.ndarray,
        Y_flat: np.ndarray,
    ) -> float:
        """Return R² of the seasonal fit on the provided data."""
        self._check_fitted("score")
        X = self.create_harmonic_features(time, t_glob)
        return self._lr.score(X, Y_flat)

    def get_anomalies(
        self,
        Y_flat: np.ndarray,
        time: np.ndarray,
        t_glob: np.ndarray,
        baseline=None,
    ) -> np.ndarray:
        """
        Compute anomalies: ``Y_flat − predicted_seasonal [− baseline]``.

        Parameters
        ----------
        Y_flat : np.ndarray, shape (n_time, n_space)
        time : np.ndarray, shape (n_time,)
        t_glob : np.ndarray, shape (n_time,)
        baseline : float, array-like, or None
            Optional pre-industrial control baseline to subtract after the
            seasonal residual is computed.  Scalar or broadcastable array.

        Returns
        -------
        np.ndarray, shape (n_time, n_space)
        """
        residuals = Y_flat - self.predict(time, t_glob)
        if baseline is not None:
            if hasattr(baseline, "values"):
                baseline = baseline.values
            residuals = residuals - float(np.mean(baseline))
        return residuals

    # ------------------------------------------------------------------
    # Properties that expose the underlying regression coefficients
    # ------------------------------------------------------------------

    @property
    def coef_(self) -> np.ndarray:
        """Regression coefficients, shape ``(n_space, n_features)``."""
        self._check_fitted("coef_")
        return self._lr.coef_

    @property
    def intercept_(self) -> np.ndarray:
        """Regression intercepts, shape ``(n_space,)``."""
        self._check_fitted("intercept_")
        return self._lr.intercept_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self, parent_method: str):
        if not self.fitted or self._lr is None:
            raise ValueError(
                f"SeasonalModel must be fitted before calling {parent_method}."
            )

    @classmethod
    def _from_sklearn_lr(cls, lr: LinearRegression) -> "SeasonalModel":
        """
        Wrap an already-fitted ``sklearn.LinearRegression`` in a SeasonalModel.

        This is used when loading old pickle files that stored the bare
        ``LinearRegression`` directly rather than a ``SeasonalModel``.
        """
        sm = cls()
        sm._lr = lr
        sm.fitted = True
        return sm
