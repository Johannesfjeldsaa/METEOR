from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Set


class TrainingBundle(dict):
    """
    Dictionary bundle of training data and metadata for noise models.

    This is a plain ``dict`` subclass so models can inspect arbitrary keys
    without a hard schema.

    Recognized Keys
    ---------------
    monthly_data : xr.Dataset
        Monthly climate data with dimensions ``(month, lat, lon, ens)``.
    variable_name : str
        Target variable (for example, ``"tas"``, ``"pr"``).
    custom_global_temp : array-like, optional
        Pre-computed smoothed global-mean temperature trajectory. The length
        must be equal to the time dimension of ``monthly_data``.
    picontrol_baseline : float or xr.DataArray, optional
        Pre-industrial control baseline for anomaly calculation.
    additional_variables : dict, optional
        Extra variable datasets for models that require more than one
        variable (for example, a conditional VAE).
    """


class NoiseModelBase(ABC):
    """
    Abstract base class for all METEOR noise models.

    Subclasses must implement ``fit``, ``generate_realization``,
    ``save_model``, and ``load_model_from_file``.

    The default ``generate_regional_mean_realizations`` implementation raises
    ``NotImplementedError``. Efficient subclasses (for example,
    ``PCAVARXNoiseModel`` in ``meteor.noise_model.pca_varx``) can override
    it with direct EOF-projection paths.

    Class Attributes
    ----------------
    model_type : str
        Short identifier used in cache filenames and the registry.
    """

    model_type: str = ""

    # ------------------------------------------------------------------
    # Data declaration
    # ------------------------------------------------------------------

    @classmethod
    def required_data(cls) -> Set[str]:
        """
        Return the set of ``TrainingBundle`` keys this model needs.

        The training helper ``train_noise_model_from_cmip6`` calls this to
        decide which CMIP6 datasets to fetch before invoking ``fit``.
        Subclasses should override this when they need data beyond the
        defaults.

        Returns
        -------
        set of str
            Required keys in a ``TrainingBundle`` for this model.
        """
        return {"monthly_data", "variable_name"}

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, training_data, variable_name: Optional[str] = None, **kwargs):
        """
        Fit the noise model.

        Parameters
        ----------
        training_data : xr.Dataset or TrainingBundle
            If a ``dict`` or ``TrainingBundle``, it must contain at minimum
            ``"monthly_data"`` and ``"variable_name"``. If an ``xr.Dataset``,
            this is a backward-compatible call and ``variable_name`` must be
            given as the second positional argument.
        variable_name : str, optional
            Required when ``training_data`` is an ``xr.Dataset``.
        **kwargs
            Model-specific keyword arguments (for example,
            ``save_diagnostics``, ``verbose``).
        """

    @abstractmethod
    def generate_realization(
        self,
        global_temp_trajectory,
        n_realizations: int = 1,
        **kwargs,
    ):
        """
        Generate stochastic climate realizations.

        Parameters
        ----------
        global_temp_trajectory : array-like
            Monthly global temperature trajectory driving the model.
        n_realizations : int, default 1
            Number of realizations to generate.
        **kwargs
            Model-specific keyword arguments.

        Returns
        -------
        xr.DataArray or list of xr.DataArray
            Generated stochastic climate realizations.
        """

    def generate_regional_mean_realizations(
        self,
        global_temp_trajectory,
        region: str = "global",
        region_mask=None,
        lat=None,
        lon=None,
        n_realizations: int = 1,
        **kwargs,
    ):
        """
        Generate regional-mean or point-scale realizations.

        The default implementation raises ``NotImplementedError``. Subclasses
        with efficient spatial aggregation paths should override this method.

        Parameters
        ----------
        global_temp_trajectory : array-like
            Monthly global temperature trajectory driving the model.
        region : str, default "global"
            Name of the region for which to generate realizations.
        region_mask : array-like, optional
            Boolean or numeric mask identifying the region on the spatial grid.
        lat : array-like, optional
            Latitude coordinates corresponding to the spatial grid.
        lon : array-like, optional
            Longitude coordinates corresponding to the spatial grid.
        n_realizations : int, default 1
            Number of realizations to generate.
        **kwargs
            Model-specific keyword arguments.

        Raises
        ------
        NotImplementedError
            Always raised by the base class. Subclasses must override this
            method to provide an implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement "
            "generate_regional_mean_realizations(). "
            "Use generate_realization() and spatially average manually, "
            "or use a model class that provides this method."
        )

    # ------------------------------------------------------------------
    # Serialization interface
    # ------------------------------------------------------------------

    @abstractmethod
    def save_model(self, filepath: str):
        """
        Serialize the fitted model to a file.

        Parameters
        ----------
        filepath : str
            Path to the file where the model will be saved.
        """

    @classmethod
    @abstractmethod
    def load_model_from_file(cls, filepath: str) -> "NoiseModelBase":
        """
        Deserialize a model from a file and return a new instance.

        Implementations are responsible for backward-compatible loading of
        older pickle formats.

        Parameters
        ----------
        filepath : str
            Path to the file from which the model will be loaded.

        Returns
        -------
        NoiseModelBase
            A new instance of the deserialized model.
        """

    def load_model(self, filepath: str):
        """
        Load model parameters from a file into this instance (in-place).

        This is a convenience wrapper around ``load_model_from_file`` for
        callers that construct an empty instance first and then populate it
        (the pattern used by ``validate_noise_model_cache``).

        Parameters
        ----------
        filepath : str
            Path to the file from which the model will be loaded.
        """
        loaded = type(self).load_model_from_file(filepath)
        self.__dict__.update(loaded.__dict__)
