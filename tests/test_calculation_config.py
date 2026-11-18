import bw_processing as bwp
import numpy as np
import pytest
from stats_arrays import LognormalUncertainty

import bw2calc.calculation_config as cc
from bw2calc import LCA, CalculationConfig, MatrixConfig, MultiLCA


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
    with pytest.warns(DeprecationWarning):
        legacy = scores(use_distributions=True, seed_override=42)
    assert np.allclose(scores(calculation_config=config), legacy)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed_override": 1},
        {"use_arrays": True},
        {"use_distributions": True},
        {"selective_use": {"biosphere_matrix": {"use_distributions": True}}},
    ],
)
def test_lca_legacy_arguments_deprecated(kwargs):
    with pytest.warns(DeprecationWarning, match="calculation_config"):
        LCA({1: 1}, data_objs=[lognormal_datapackage()], **kwargs)


def test_lca_legacy_and_config_conflict():
    with pytest.raises(ValueError, match="legacy"):
        LCA(
            {1: 1},
            data_objs=[lognormal_datapackage()],
            calculation_config=CalculationConfig(),
            use_distributions=True,
        )


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
