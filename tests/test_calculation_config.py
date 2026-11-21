import bw_processing as bwp
import numpy as np
import pytest
from stats_arrays import LognormalUncertainty

import bw2calc.calculation_config as cc
from bw2calc import (
    LCA,
    CalculationConfig,
    IterativeLCA,
    JacobiGMRESLCA,
    MatrixConfig,
    MultiLCA,
    fast_supply_arrays,
    iterative_lca,
    lca_base,
)
from bw2calc.fast_supply_arrays import FastSupplyArraysMixin
from bw2calc.utils import lu_solver


def lognormal_datapackage():
    """One lognormal technosphere input and one lognormal biosphere flow with the same
    parameters, as in https://github.com/brightway-lca/brightway2-calc/issues/189."""
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        name="technosphere",
        indices_array=np.array([(1, 1), (2, 2), (2, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 1.0, 1.0]),
        flip_array=np.array([False, False, True]),
        distributions_array=np.array(
            [
                (0, 1.0, np.nan, np.nan, np.nan, np.nan, False),
                (0, 1.0, np.nan, np.nan, np.nan, np.nan, False),
                (LognormalUncertainty.id, 0.0, np.log(1.5), np.nan, np.nan, np.nan, False),
            ],
            dtype=bwp.UNCERTAINTY_DTYPE,
        ),
    )
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        name="biosphere",
        indices_array=np.array([(100, 1)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0]),
        distributions_array=np.array(
            [(LognormalUncertainty.id, 0.0, np.log(1.5), np.nan, np.nan, np.nan, False)],
            dtype=bwp.UNCERTAINTY_DTYPE,
        ),
    )
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        name="characterization",
        indices_array=np.array([(100, 0)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0]),
    )
    return dp


def draws(lca, iterations=200):
    """Sampled (technosphere, biosphere) values of the uncertain exchanges."""
    lca.lci()
    tech_row = lca.dicts.product[2]
    col = lca.dicts.activity[1]
    bio_row = lca.dicts.biosphere[100]
    result = []
    for _ in range(iterations):
        result.append((-lca.technosphere_matrix[tech_row, col], lca.biosphere_matrix[bio_row, col]))
        next(lca)
    return np.array(result)


# CalculationConfig


def test_defaults():
    config = CalculationConfig()
    assert config.flags("technosphere_matrix") == (False, False)
    assert config.matrix_kwargs("technosphere_matrix") == {
        "use_arrays": False,
        "use_distributions": False,
        "seed_override": None,
    }


def test_matrix_overrides():
    config = CalculationConfig(
        use_distributions=True,
        seed=1,
        matrices={
            "biosphere_matrix": {"use_distributions": False, "seed": 2},
            "characterization_matrix": MatrixConfig(use_arrays=True),
        },
    )
    assert config.matrix_kwargs("technosphere_matrix") == {
        "use_arrays": False,
        "use_distributions": True,
        "seed_override": 1,
    }
    assert config.matrix_kwargs("biosphere_matrix") == {
        "use_arrays": False,
        "use_distributions": False,
        "seed_override": 2,
    }
    assert config.flags("characterization_matrix") == (True, True)


def test_seed_zero():
    assert CalculationConfig(seed=0).matrix_kwargs("biosphere_matrix")["seed_override"] == 0


def test_extra_fields_forbidden():
    with pytest.raises(ValueError):
        CalculationConfig(seed_override=1)
    with pytest.raises(ValueError):
        CalculationConfig(matrices={"biosphere_matrix": {"use_distribution": True}})


def test_derive_group_seeds_passed(monkeypatch):
    monkeypatch.setattr(cc, "MU_DERIVE_GROUP_SEEDS", True)
    config = CalculationConfig(
        derive_group_seeds=False,
        matrices={"biosphere_matrix": {"derive_group_seeds": True}},
    )
    assert config.matrix_kwargs("technosphere_matrix")["derive_group_seeds"] is False
    assert config.matrix_kwargs("biosphere_matrix")["derive_group_seeds"] is True


def test_derive_group_seeds_needs_new_matrix_utils(monkeypatch):
    monkeypatch.setattr(cc, "MU_DERIVE_GROUP_SEEDS", False)
    with pytest.raises(ValueError, match="matrix_utils >= 0.10"):
        CalculationConfig(derive_group_seeds=True).matrix_kwargs("technosphere_matrix")
    # Not set, so not passed on
    assert "derive_group_seeds" not in CalculationConfig().matrix_kwargs("technosphere_matrix")


def test_selective_use_round_trip():
    config = CalculationConfig(matrices={"biosphere_matrix": {"seed": 5}})
    assert config.selective_use == {}
    config.set_selective_use(
        {
            "biosphere_matrix": {"use_distributions": True},
            "technosphere_matrix": {"use_arrays": False},
        }
    )
    assert config.selective_use == {
        "biosphere_matrix": {"use_distributions": True},
        "technosphere_matrix": {"use_arrays": False},
    }
    # Other matrix settings are kept
    assert config.matrices["biosphere_matrix"].seed == 5
    config.set_selective_use(None)
    assert config.selective_use == {}
    assert config.matrices == {"biosphere_matrix": MatrixConfig(seed=5)}


# LCA


def test_lca_calculation_config():
    config = CalculationConfig(
        use_distributions=True, seed=7, matrices={"biosphere_matrix": {"seed": 8}}
    )
    lca = LCA({1: 1}, data_objs=[lognormal_datapackage()], calculation_config=config)
    lca.lci()
    assert lca.technosphere_mm.seed_override == 7
    assert lca.biosphere_mm.seed_override == 8
    assert lca.use_distributions
    assert lca.seed_override == 7
    # The config is copied, so changing the original afterwards has no effect
    config.seed = 1
    assert lca.calculation_config.seed == 7


def test_lca_calculation_config_dict():
    lca = LCA(
        {1: 1},
        data_objs=[lognormal_datapackage()],
        calculation_config={"use_distributions": True, "seed": 3},
    )
    assert lca.calculation_config == CalculationConfig(use_distributions=True, seed=3)


def test_lca_reproducible():
    def scores(**kwargs):
        lca = LCA({1: 1}, data_objs=[lognormal_datapackage()], **kwargs)
        return draws(lca, 10)

    config = CalculationConfig(use_distributions=True, seed=42)
    assert np.allclose(scores(calculation_config=config), scores(calculation_config=config))


def test_lca_attributes_write_through():
    lca = LCA({1: 1}, data_objs=[lognormal_datapackage()])
    lca.use_distributions = True
    lca.use_arrays = True
    lca.seed_override = 4
    lca.selective_use = {"biosphere_matrix": {"use_distributions": False}}
    assert lca.calculation_config == CalculationConfig(
        use_arrays=True,
        use_distributions=True,
        seed=4,
        matrices={"biosphere_matrix": {"use_distributions": False}},
    )
    assert lca.check_selective_use("biosphere_matrix") == (True, False)


def test_multi_lca_calculation_config():
    mlca = MultiLCA(
        demands={"a": {1: 1}},
        method_config={"impact_categories": [("foo",)]},
        data_objs=[lognormal_datapackage()],
        calculation_config=CalculationConfig(use_distributions=True, seed=7),
    )
    assert mlca.check_selective_use("technosphere_matrix") == (False, True)
    assert mlca.seed_override == 7


# https://github.com/brightway-lca/brightway2-calc/issues/189


@pytest.mark.skipif(not cc.MU_DERIVE_GROUP_SEEDS, reason="Needs matrix_utils >= 0.10")
def test_derive_group_seeds_independent_draws():
    lca = LCA(
        {1: 1},
        data_objs=[lognormal_datapackage()],
        calculation_config=CalculationConfig(use_distributions=True, seed=1),
    )
    samples = draws(lca)
    assert not np.allclose(samples[:, 0], samples[:, 1])
    assert abs(np.corrcoef(np.log(samples.T))[0, 1]) < 0.3


@pytest.mark.skipif(not cc.MU_DERIVE_GROUP_SEEDS, reason="Needs matrix_utils >= 0.10")
def test_derive_group_seeds_false_identical_draws():
    lca = LCA(
        {1: 1},
        data_objs=[lognormal_datapackage()],
        calculation_config=CalculationConfig(
            use_distributions=True, seed=1, derive_group_seeds=False
        ),
    )
    samples = draws(lca)
    assert np.allclose(samples[:, 0], samples[:, 1])


# Review of https://github.com/brightway-lca/brightway2-calc/pull/190


def test_calculation_config_wrong_type():
    with pytest.raises(TypeError, match="must be a `CalculationConfig` or a dict, not int"):
        LCA({1: 1}, data_objs=[lognormal_datapackage()], calculation_config=42)


def test_matrices_dict_added_in_place():
    lca = LCA(
        {1: 1},
        data_objs=[lognormal_datapackage()],
        calculation_config=CalculationConfig(seed=1),
    )
    lca.calculation_config.matrices["biosphere_matrix"] = {"use_distributions": True, "seed": 2}
    assert lca.check_selective_use("biosphere_matrix") == (False, True)
    assert lca.selective_use == {"biosphere_matrix": {"use_distributions": True}}
    lca.lci()
    assert lca.biosphere_mm.seed_override == 2
    assert lca.calculation_config.matrices["biosphere_matrix"] == MatrixConfig(
        use_distributions=True, seed=2
    )


def test_matrices_invalid_dict_added_in_place():
    config = CalculationConfig()
    config.matrices["biosphere_matrix"] = {"use_distribution": True}
    with pytest.raises(ValueError):
        config.matrix_kwargs("biosphere_matrix")


def test_matrix_explicit_none_overrides_global(monkeypatch):
    monkeypatch.setattr(cc, "MU_DERIVE_GROUP_SEEDS", True)
    config = CalculationConfig(
        seed=42,
        derive_group_seeds=False,
        use_distributions=True,
        matrices={
            "biosphere_matrix": {"seed": None, "derive_group_seeds": None},
            "characterization_matrix": MatrixConfig(seed=None),
            # `None` always means the global value for the flags
            "normalization_matrix": {"use_distributions": None},
        },
    )
    assert config.matrix_kwargs("technosphere_matrix") == {
        "use_arrays": False,
        "use_distributions": True,
        "seed_override": 42,
        "derive_group_seeds": False,
    }
    assert config.matrix_kwargs("biosphere_matrix") == {
        "use_arrays": False,
        "use_distributions": True,
        "seed_override": None,
    }
    assert config.matrix_kwargs("characterization_matrix") == {
        "use_arrays": False,
        "use_distributions": True,
        "seed_override": None,
        "derive_group_seeds": False,
    }
    assert config.matrix_kwargs("normalization_matrix")["use_distributions"] is True
    assert config.matrix_kwargs("normalization_matrix")["seed_override"] == 42


def test_matrix_explicit_none_kept():
    config = CalculationConfig(seed=42, matrices={"biosphere_matrix": {"seed": None}})
    # Copied when passed to an LCA
    lca = LCA({1: 1}, data_objs=[lognormal_datapackage()], calculation_config=config)
    assert lca.calculation_config.matrix_kwargs("biosphere_matrix")["seed_override"] is None
    # Assigned after construction
    config = CalculationConfig(seed=42, matrices={"biosphere_matrix": {"use_arrays": True}})
    config.matrices["biosphere_matrix"].seed = None
    assert config.matrix_kwargs("biosphere_matrix")["seed_override"] is None
    # Changing the legacy flags
    config.set_selective_use({"technosphere_matrix": {"use_arrays": True}})
    assert config.matrix_kwargs("biosphere_matrix")["seed_override"] is None
    assert config.matrix_kwargs("technosphere_matrix")["seed_override"] == 42
    assert config.selective_use == {"technosphere_matrix": {"use_arrays": True}}


LEGACY = [
    {"seed_override": 1},
    {"seed_override": 0},
    {"use_arrays": True},
    {"use_arrays": False},
    {"use_distributions": True},
    {"use_distributions": False},
    {"selective_use": {"biosphere_matrix": {"use_distributions": True}}},
    {"selective_use": {}},
]


@pytest.mark.parametrize("kwargs", LEGACY)
def test_legacy_arguments_deprecated(kwargs):
    with pytest.warns(DeprecationWarning, match="calculation_config") as record:
        lca = LCA({1: 1}, data_objs=[lognormal_datapackage()], **kwargs)
    assert record[0].filename == __file__
    for key, value in kwargs.items():
        assert getattr(lca, key) == value
    with pytest.warns(DeprecationWarning, match="calculation_config"):
        MultiLCA(
            demands={"a": {1: 1}},
            method_config={"impact_categories": [("foo",)]},
            data_objs=[lognormal_datapackage()],
            **kwargs,
        )


@pytest.mark.parametrize("kwargs", LEGACY)
def test_legacy_arguments_and_config_conflict(kwargs):
    # An error even if the values agree
    config = CalculationConfig(
        seed=kwargs.get("seed_override"),
        use_arrays=kwargs.get("use_arrays", False),
        use_distributions=kwargs.get("use_distributions", False),
        matrices=kwargs.get("selective_use", {}),
    )
    with pytest.raises(ValueError, match="deprecated arguments"):
        LCA({1: 1}, data_objs=[lognormal_datapackage()], calculation_config=config, **kwargs)
    with pytest.raises(ValueError, match="deprecated arguments"):
        MultiLCA(
            demands={"a": {1: 1}},
            method_config={"impact_categories": [("foo",)]},
            data_objs=[lognormal_datapackage()],
            calculation_config=config,
            **kwargs,
        )


def test_legacy_none_is_not_given():
    config = CalculationConfig(use_distributions=True)
    lca = LCA(
        {1: 1},
        data_objs=[lognormal_datapackage()],
        calculation_config=config,
        seed_override=None,
        use_arrays=None,
        use_distributions=None,
        selective_use=None,
    )
    assert lca.calculation_config == config


def test_legacy_matches_config():
    def samples(**kwargs):
        return draws(LCA({1: 1}, data_objs=[lognormal_datapackage()], **kwargs), 10)

    with pytest.warns(DeprecationWarning):
        legacy = samples(use_distributions=True, seed_override=42)
    assert np.allclose(legacy, samples(calculation_config={"use_distributions": True, "seed": 42}))


def test_seed_still_positional():
    with pytest.warns(DeprecationWarning):
        lca = LCA({1: 1}, None, None, None, [lognormal_datapackage()], None, None, 42)
    assert lca.seed_override == 42


def test_calculation_config_keyword_only():
    with pytest.raises(TypeError, match="positional"):
        LCA({1: 1}, None, None, None, [lognormal_datapackage()], *[None] * 6, CalculationConfig())


def test_deprecation_warning_points_at_caller_for_subclasses():
    with pytest.warns(DeprecationWarning, match="calculation_config") as record:
        JacobiGMRESLCA({1: 1}, data_objs=[lognormal_datapackage()], use_distributions=True)
    assert record[0].filename == __file__


# Iterative refinement


@pytest.fixture
def lu_solver_calls(monkeypatch):
    """Solve with `lu_solver`, not PARDISO, and collect the keyword arguments it gets."""
    calls = []

    def recording(matrix, **kwargs):
        calls.append(kwargs)
        return lu_solver(matrix, **kwargs)

    for module in (lca_base, fast_supply_arrays, iterative_lca):
        monkeypatch.setattr(module, "PYPARDISO", False)
        monkeypatch.setattr(module, "lu_solver", recording)
    monkeypatch.setattr(fast_supply_arrays, "UMFPACK", True)
    return calls


def test_iterative_refinement_default():
    assert CalculationConfig().iterative_refinement is True
    with pytest.raises(ValueError):
        MatrixConfig(iterative_refinement=False)


@pytest.mark.parametrize("config, refine", [(None, True), ({"iterative_refinement": False}, False)])
def test_iterative_refinement_factorized_lci(lu_solver_calls, config, refine):
    lca = LCA({1: 1}, data_objs=[lognormal_datapackage()], calculation_config=config)
    lca.lci(factorize=True)
    assert lu_solver_calls == [{"refine": refine}]
    expected = np.linalg.solve(lca.technosphere_matrix.toarray(), lca.demand_array)
    assert np.allclose(lca.supply_array, expected)
    lca.lci(demand={2: 1})
    assert len(lu_solver_calls) == 1
    expected = np.linalg.solve(lca.technosphere_matrix.toarray(), lca.demand_array)
    assert np.allclose(lca.supply_array, expected)


@pytest.mark.parametrize("config, refine", [(None, True), ({"iterative_refinement": False}, False)])
def test_iterative_refinement_fast_supply_arrays(lu_solver_calls, config, refine):
    class FastLCA(LCA, FastSupplyArraysMixin):
        pass

    lca = FastLCA({1: 1}, data_objs=[lognormal_datapackage()], calculation_config=config)
    lca.load_lci_data()
    lca.build_demand_array()
    result = lca.calculate_supply_arrays([lca.demand_array])
    assert lu_solver_calls == [{"refine": refine}]
    expected = np.linalg.solve(lca.technosphere_matrix.toarray(), lca.demand_array)
    assert np.allclose(result[:, 0], expected)


def test_iterative_refinement_always_on_for_iterative_lca_reference(lu_solver_calls):
    """`IterativeLCA.estimate_error` trusts the reference solve."""
    lca = IterativeLCA(
        {1: 1},
        data_objs=[lognormal_datapackage()],
        calculation_config={"iterative_refinement": False},
    )
    lca.lci()
    assert lca.solver_stats["reference"] == 1
    assert lu_solver_calls == [{"copy": False}]
