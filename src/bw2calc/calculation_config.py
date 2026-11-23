import inspect
import os
import sys
import warnings
from typing import Optional, Union

import matrix_utils as mu
from pydantic import BaseModel, ConfigDict

# ``derive_group_seeds`` was added in matrix_utils 0.10:
# https://github.com/brightway-lca/matrix_utils/pull/52
MU_DERIVE_GROUP_SEEDS = "derive_group_seeds" in inspect.signature(mu.MappedMatrix).parameters

# ``MatrixConfig`` fields for which an explicit ``None`` overrides the global value
NULLABLE = {"seed", "derive_group_seeds"}


class MatrixConfig(BaseModel):
    """Settings for one matrix, such as ``technosphere_matrix``.

    Each field which isn't given uses the value of the same field in ``CalculationConfig``.

    ``seed`` and ``derive_group_seeds`` can be explicitly set to ``None`` to override a global
    value with ``None``: ``MatrixConfig(seed=None)`` uses the datapackage ``seed`` for this
    matrix even if ``CalculationConfig.seed`` is set, and ``MatrixConfig(derive_group_seeds=None)``
    uses the matrix_utils default. For ``use_arrays`` and ``use_distributions``, ``None`` always
    means the global value.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    use_arrays: Optional[bool] = None
    use_distributions: Optional[bool] = None
    seed: Optional[int] = None
    derive_group_seeds: Optional[bool] = None


class CalculationConfig(BaseModel):
    """Settings for how matrices are built from datapackages, how they are sampled, and how
    the technosphere matrix is solved.

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
    iterative_refinement : bool
        Let UMFPACK or PARDISO refine each solution of a factorized technosphere matrix.
        Default ``True``. ``False`` skips the refinement, which makes each solve about 3x faster
        with UMFPACK (reference BLAS or OpenBLAS) and about 2x faster with PARDISO, but can lose
        accuracy on badly scaled matrices; see
        https://github.com/brightway-lca/brightway2-calc/issues/184. Applies to
        ``lci(factorize=True)`` and every later solve which reuses that factorization, and to
        ``FastSupplyArraysMixin``. Solves without ``factorize=True``,
        ``FastScoresOnlyMultiLCA.solve_adjoint``, and the reference solves of ``IterativeLCA``
        are always refined. With PARDISO the setting holds until the next Monte Carlo
        iteration resets the solver, also for solves on a changed technosphere matrix. No
        effect with SuperLU.

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
    iterative_refinement: bool = True

    def _matrix(self, matrix_label: str) -> MatrixConfig:
        matrix = self.matrices.get(matrix_label)
        if matrix is None:
            return MatrixConfig()
        if not isinstance(matrix, MatrixConfig):
            # Dicts added to ``matrices`` in place aren't validated by pydantic
            matrix = self.matrices[matrix_label] = MatrixConfig.model_validate(matrix)
        return matrix

    def _get(self, matrix_label: str, field: str):
        matrix = self._matrix(matrix_label)
        value = getattr(matrix, field)
        if value is None and not (field in NULLABLE and field in matrix.model_fields_set):
            return getattr(self, field)
        return value

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
        result = {}
        for label in self.matrices:
            flags = {
                key: value
                for key, value in self._matrix(label)
                .model_dump(include={"use_arrays", "use_distributions"})
                .items()
                if value is not None
            }
            if flags:
                result[label] = flags
        return result

    def set_selective_use(self, selective_use: Optional[dict]) -> None:
        """Replace the matrix flags with ``selective_use``, keeping other matrix settings."""
        selective_use = selective_use or {}
        matrices = {}
        for label in set(self.matrices).union(selective_use):
            # Only the fields which were set, to keep an explicit ``None`` apart from "not set"
            values = self._matrix(label).model_dump(exclude_unset=True)
            for key in ("use_arrays", "use_distributions"):
                values.pop(key, None)
                if selective_use.get(label, {}).get(key) is not None:
                    values[key] = selective_use[label][key]
            if values:
                matrices[label] = MatrixConfig(**values)
        self.matrices = matrices


def _external_stacklevel() -> int:
    """``stacklevel`` for ``warnings.warn`` of the first caller outside ``bw2calc``.

    A fixed value would be wrong for subclasses, as each ``super().__init__()`` adds a frame.
    """
    package = os.path.dirname(os.path.abspath(__file__)) + os.sep
    # Level 1 is the function calling ``warnings.warn``, which is our caller
    frame, level = sys._getframe(1), 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(package):
        frame, level = frame.f_back, level + 1
    return level


def resolve_calculation_config(
    calculation_config: Union[CalculationConfig, dict, None],
    seed_override: Optional[int] = None,
    use_arrays: Optional[bool] = None,
    use_distributions: Optional[bool] = None,
    selective_use: Optional[dict] = None,
) -> CalculationConfig:
    """Get a new ``CalculationConfig`` from a config, a dict, or the deprecated arguments.

    Any deprecated argument which isn't ``None`` counts as given. Giving one together with
    ``calculation_config`` raises a ``ValueError``, even if the values agree.
    """
    legacy = {
        "seed_override": seed_override,
        "use_arrays": use_arrays,
        "use_distributions": use_distributions,
        "selective_use": selective_use,
    }
    given = [key for key, value in legacy.items() if value is not None]
    if calculation_config is None:
        if given:
            warnings.warn(
                f"The arguments {given} are deprecated; use "
                "`calculation_config=CalculationConfig(...)` instead",
                DeprecationWarning,
                stacklevel=_external_stacklevel(),
            )
        return CalculationConfig(
            use_arrays=bool(use_arrays),
            use_distributions=bool(use_distributions),
            seed=seed_override,
            matrices=selective_use or {},
        )
    if given:
        raise ValueError(
            f"Can't pass both `calculation_config` and the deprecated arguments {given}; "
            "set them in `calculation_config` only"
        )
    if isinstance(calculation_config, dict):
        return CalculationConfig.model_validate(calculation_config)
    if not isinstance(calculation_config, CalculationConfig):
        raise TypeError(
            "`calculation_config` must be a `CalculationConfig` or a dict, not "
            f"{type(calculation_config).__name__}"
        )
    return calculation_config.model_copy(deep=True)
