import inspect
import warnings
from typing import Optional

import matrix_utils as mu
from pydantic import BaseModel, ConfigDict

# ``derive_group_seeds`` was added in matrix_utils 0.10:
# https://github.com/brightway-lca/matrix_utils/pull/52
MU_DERIVE_GROUP_SEEDS = "derive_group_seeds" in inspect.signature(mu.MappedMatrix).parameters

LEGACY_ARGUMENTS = ("seed_override", "use_arrays", "use_distributions", "selective_use")


class MatrixConfig(BaseModel):
    """Settings for one matrix, such as ``technosphere_matrix``.

    Each field left as ``None`` uses the value of the same field in ``CalculationConfig``.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    use_arrays: Optional[bool] = None
    use_distributions: Optional[bool] = None
    seed: Optional[int] = None
    derive_group_seeds: Optional[bool] = None


class CalculationConfig(BaseModel):
    """Settings for how matrices are built from datapackages, and how they are sampled.

    Parameters
    ----------
    use_arrays : bool
        Use arrays instead of vectors from the given datapackages. Default ``False``.
    use_distributions : bool
        Use probability distributions from the given datapackages. Default ``False``.
    seed : int, optional
        RNG seed to use in place of the datapackage ``seed``, if any.
    derive_group_seeds : bool, optional
        Passed to ``matrix_utils.MappedMatrix``. If ``True``, each resource group gets its own
        RNG seed, derived deterministically from the base seed, the matrix label and the group
        label, so groups don't draw correlated samples. If ``False``, every resource group uses
        the base seed directly, as in matrix_utils < 0.10. ``None`` (the default) uses the
        matrix_utils default. Setting it requires matrix_utils >= 0.10. See
        https://github.com/brightway-lca/brightway2-calc/issues/189.
    matrices : dict[str, MatrixConfig]
        Settings for individual matrices, keyed by matrix label (``technosphere_matrix``,
        ``biosphere_matrix``, ``characterization_matrix``, ``normalization_matrix``, or
        ``weighting_matrix``). These override the global settings for that matrix. Replaces
        ``selective_use``.

    Example
    -------

    .. code-block:: python

        CalculationConfig(
            use_distributions=True,
            seed=42,
            matrices={"biosphere_matrix": {"use_distributions": False}},
        )

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    use_arrays: bool = False
    use_distributions: bool = False
    seed: Optional[int] = None
    derive_group_seeds: Optional[bool] = None
    matrices: dict[str, MatrixConfig] = {}

    @classmethod
    def from_legacy(
        cls,
        seed_override: Optional[int] = None,
        use_arrays: Optional[bool] = None,
        use_distributions: Optional[bool] = None,
        selective_use: Optional[dict] = None,
    ) -> "CalculationConfig":
        """Build from the arguments ``LCA`` and ``MultiLCA`` took before ``CalculationConfig``."""
        return cls(
            use_arrays=bool(use_arrays),
            use_distributions=bool(use_distributions),
            seed=seed_override,
            matrices=selective_use or {},
        )

    def _get(self, matrix_label: str, field: str):
        value = getattr(self.matrices.get(matrix_label, MatrixConfig()), field)
        return getattr(self, field) if value is None else value

    def flags(self, matrix_label: str) -> tuple[bool, bool]:
        """``(use_arrays, use_distributions)`` for ``matrix_label``."""
        return (
            self._get(matrix_label, "use_arrays"),
            self._get(matrix_label, "use_distributions"),
        )

    def matrix_kwargs(self, matrix_label: str) -> dict:
        """Keyword arguments for ``matrix_utils.MappedMatrix`` for ``matrix_label``."""
        use_arrays, use_distributions = self.flags(matrix_label)
        kwargs = {
            "use_arrays": use_arrays,
            "use_distributions": use_distributions,
            "seed_override": self._get(matrix_label, "seed"),
        }
        derive_group_seeds = self._get(matrix_label, "derive_group_seeds")
        if derive_group_seeds is not None:
            if not MU_DERIVE_GROUP_SEEDS:
                raise ValueError(
                    "`derive_group_seeds` requires matrix_utils >= 0.10; found " f"{mu.__version__}"
                )
            kwargs["derive_group_seeds"] = derive_group_seeds
        return kwargs

    @property
    def selective_use(self) -> dict:
        """The matrix flags in the legacy ``selective_use`` format."""
        return {
            label: {
                key: value
                for key, value in matrix.model_dump(
                    include={"use_arrays", "use_distributions"}
                ).items()
                if value is not None
            }
            for label, matrix in self.matrices.items()
            if matrix.use_arrays is not None or matrix.use_distributions is not None
        }

    def set_selective_use(self, selective_use: Optional[dict]) -> None:
        """Replace the matrix flags with ``selective_use``, keeping other matrix settings."""
        selective_use = selective_use or {}
        matrices = {}
        for label in set(self.matrices).union(selective_use):
            current = self.matrices.get(label, MatrixConfig())
            flags = selective_use.get(label, {})
            matrix = MatrixConfig(
                **{
                    **current.model_dump(),
                    "use_arrays": flags.get("use_arrays"),
                    "use_distributions": flags.get("use_distributions"),
                }
            )
            if matrix != MatrixConfig():
                matrices[label] = matrix
        self.matrices = matrices


def resolve_calculation_config(
    calculation_config: Optional[CalculationConfig], **legacy
) -> CalculationConfig:
    """Get a ``CalculationConfig`` from either a config or the legacy arguments, not both."""
    given = sorted(
        key
        for key, value in legacy.items()
        if value is not None and value is not False and value != {}
    )
    if calculation_config is not None:
        if given:
            raise ValueError(
                f"Can't pass both `calculation_config` and the legacy arguments {given}; "
                "set them in `calculation_config` instead"
            )
        if isinstance(calculation_config, dict):
            return CalculationConfig.model_validate(calculation_config)
        return calculation_config.model_copy(deep=True)
    if given:
        warnings.warn(
            f"The arguments {given} are deprecated; use "
            "`calculation_config=CalculationConfig(...)` instead",
            DeprecationWarning,
            stacklevel=3,
        )
    return CalculationConfig.from_legacy(**legacy)
