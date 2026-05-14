"""Tests for PartitionedMonteCarloLCA.

System under test
-----------------
Static database (name "static_db"):
    S1 (id=100): produces P1 (id=1), emits F1 (id=1000) amount=2.0
    S2 (id=200): produces P2 (id=2), consumes P1 (id=1) amount=0.5,
                 emits F2 (id=2000) amount=1.0

Stochastic database (name "stochastic_db"):
    A1 (id=300): produces P3 (id=3), consumes P2 (id=2) amount=1.0,
                 emits F1 (id=1000) amount=0.1

LCIA method:
    F1 (id=1000): CF=2.0,  F2 (id=2000): CF=1.0

Expected score for {P3: 1}: 1.1*2.0 + 1.0*1.0 = 3.2
"""

import bw_processing as bwp
import numpy as np
import pytest

from bw2calc import PartitionedMonteCarloLCA
from bw2calc.errors import (
    CyclicDependencyGraph,
    DemandInStaticDatabase,
    MissingDatabaseDependencies,
    StaticDependsOnStochastic,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_static_dp(database_dependencies=None):
    """Static package: S1 (100) → P1 (1), S2 (200) → P2 (2)."""
    dp = bwp.create_datapackage(name="static_db")

    tech_indices = np.array([(1, 100), (2, 200), (1, 200)], dtype=bwp.INDICES_DTYPE)
    tech_data = np.array([1.0, 1.0, 0.5])
    tech_flip = np.array([False, False, True])  # last is consumption P1 by S2

    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        indices_array=tech_indices,
        data_array=tech_data,
        flip_array=tech_flip,
    )

    bio_indices = np.array([(1000, 100), (2000, 200)], dtype=bwp.INDICES_DTYPE)
    bio_data = np.array([2.0, 1.0])

    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        indices_array=bio_indices,
        data_array=bio_data,
    )

    dp.metadata["database_dependencies"] = (
        database_dependencies if database_dependencies is not None else []
    )
    return dp


def _make_stochastic_dp(
    p2_amount=1.0, f1_amount=0.1, with_distributions=False, database_dependencies=None
):
    """Stochastic package: A1 (300) → P3 (3), consumes P2 (2)."""
    dp = bwp.create_datapackage(name="stochastic_db")

    tech_indices = np.array([(3, 300), (2, 300)], dtype=bwp.INDICES_DTYPE)
    tech_data = np.array([1.0, p2_amount])
    tech_flip = np.array([False, True])

    if with_distributions:
        tech_dists = np.array(
            [
                (0, 1.0, 0.0, np.nan, np.nan, np.nan, False),
                (4, p2_amount, 0.0, np.nan, p2_amount * 0.5, p2_amount * 1.5, False),
            ],
            dtype=bwp.UNCERTAINTY_DTYPE,
        )
        dp.add_persistent_vector(
            matrix="technosphere_matrix",
            indices_array=tech_indices,
            data_array=tech_data,
            flip_array=tech_flip,
            distributions_array=tech_dists,
        )
    else:
        dp.add_persistent_vector(
            matrix="technosphere_matrix",
            indices_array=tech_indices,
            data_array=tech_data,
            flip_array=tech_flip,
        )

    bio_indices = np.array([(1000, 300)], dtype=bwp.INDICES_DTYPE)
    bio_data = np.array([f1_amount])

    if with_distributions:
        bio_dists = np.array(
            [(4, f1_amount, 0.0, np.nan, f1_amount * 0.5, f1_amount * 2.0, False)],
            dtype=bwp.UNCERTAINTY_DTYPE,
        )
        dp.add_persistent_vector(
            matrix="biosphere_matrix",
            indices_array=bio_indices,
            data_array=bio_data,
            distributions_array=bio_dists,
        )
    else:
        dp.add_persistent_vector(
            matrix="biosphere_matrix",
            indices_array=bio_indices,
            data_array=bio_data,
        )

    dp.metadata["database_dependencies"] = (
        database_dependencies if database_dependencies is not None else ["static_db"]
    )
    return dp


def _make_method_dp():
    """Characterization factors: F1=2.0, F2=1.0."""
    dp = bwp.create_datapackage(name="method")

    char_indices = np.array([(1000, 1000), (2000, 2000)], dtype=bwp.INDICES_DTYPE)
    char_data = np.array([2.0, 1.0])

    dp.add_persistent_vector(
        matrix="characterization_matrix",
        indices_array=char_indices,
        data_array=char_data,
    )
    return dp


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


def test_first_score_matches_reference():
    """Partitioned LCA must give the same score as a direct (non-partitioned) LCA."""
    from bw2calc import LCA

    static_dp = _make_static_dp()
    stochastic_dp = _make_stochastic_dp()
    method_dp = _make_method_dp()

    # Reference: full system in one LCA
    ref_lca = LCA(
        demand={3: 1.0},
        data_objs=[static_dp, stochastic_dp, method_dp],
    )
    ref_lca.lci()
    ref_lca.lcia()
    expected = ref_lca.score

    # Partitioned LCA
    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=["static_db"],
        data_objs=[static_dp, stochastic_dp, method_dp],
    )
    p_lca.lci()
    p_lca.lcia()

    assert np.isclose(p_lca.score, expected, rtol=1e-6)
    assert np.isclose(p_lca.score, 3.2, rtol=1e-6)


def test_score_value():
    """Verify the expected analytical score of 3.2."""
    static_dp = _make_static_dp()
    stochastic_dp = _make_stochastic_dp()
    method_dp = _make_method_dp()

    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=["static_db"],
        data_objs=[static_dp, stochastic_dp, method_dp],
    )
    p_lca.lci()
    p_lca.lcia()

    assert np.isclose(p_lca.score, 3.2, rtol=1e-6)


def test_iteration_without_distributions():
    """Iteration without distributions gives same score each time."""
    static_dp = _make_static_dp()
    stochastic_dp = _make_stochastic_dp()
    method_dp = _make_method_dp()

    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=["static_db"],
        data_objs=[static_dp, stochastic_dp, method_dp],
    )
    p_lca.lci()
    p_lca.lcia()
    first_score = p_lca.score

    for _ in range(5):
        next(p_lca)
        assert np.isclose(p_lca.score, first_score, rtol=1e-6)


def test_iteration_with_distributions_varies():
    """Iteration with stochastic distributions produces varying scores."""
    static_dp = _make_static_dp()
    stochastic_dp = _make_stochastic_dp(with_distributions=True)
    method_dp = _make_method_dp()

    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=["static_db"],
        data_objs=[static_dp, stochastic_dp, method_dp],
        seed_override=42,
    )
    p_lca.lci()
    p_lca.lcia()

    scores = [p_lca.score]
    for _ in range(20):
        next(p_lca)
        scores.append(p_lca.score)

    # Scores should vary (uniform distribution on consumption amount)
    assert len(set(np.round(scores, 8))) > 1, "Scores should vary across MC iterations"


def test_keep_first_iteration():
    """keep_first_iteration uses first sample as iteration 0."""
    static_dp = _make_static_dp()
    stochastic_dp = _make_stochastic_dp(with_distributions=True)
    method_dp = _make_method_dp()

    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=["static_db"],
        data_objs=[static_dp, stochastic_dp, method_dp],
        seed_override=1,
    )
    p_lca.lci()
    p_lca.lcia()
    first_score = p_lca.score

    p_lca.keep_first_iteration()
    next(p_lca)
    kept_score = p_lca.score

    assert np.isclose(first_score, kept_score, rtol=1e-6)


def test_properties_accessible():
    """inventory, supply_array, dicts, and matrix properties are accessible."""
    static_dp = _make_static_dp()
    stochastic_dp = _make_stochastic_dp()
    method_dp = _make_method_dp()

    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=["static_db"],
        data_objs=[static_dp, stochastic_dp, method_dp],
    )
    p_lca.lci()
    p_lca.lcia()

    assert p_lca.supply_array is not None
    assert p_lca.inventory is not None
    assert p_lca.characterized_inventory is not None
    assert p_lca.dicts is not None
    assert p_lca.technosphere_matrix is not None
    assert p_lca.biosphere_matrix is not None
    assert p_lca.characterization_matrix is not None


def test_iterable():
    """PartitionedMonteCarloLCA is iterable via for loop."""
    static_dp = _make_static_dp()
    stochastic_dp = _make_stochastic_dp()
    method_dp = _make_method_dp()

    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=["static_db"],
        data_objs=[static_dp, stochastic_dp, method_dp],
    )
    p_lca.lci()
    p_lca.lcia()

    scores = []
    for _ in range(3):
        next(p_lca)
        scores.append(p_lca.score)

    assert len(scores) == 3


# ---------------------------------------------------------------------------
# Package classification
# ---------------------------------------------------------------------------


def test_package_classification():
    """Packages are correctly classified as static, stochastic, or method."""
    static_dp = _make_static_dp()
    stochastic_dp = _make_stochastic_dp()
    method_dp = _make_method_dp()

    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=["static_db"],
        data_objs=[static_dp, stochastic_dp, method_dp],
    )

    # Each package has tech + bio groups, so 2 filtered packages per LCI package
    assert len(p_lca.static_packages) == 2
    assert len(p_lca.stochastic_packages) == 2
    assert len(p_lca.method_packages) == 1
    assert all(dp.metadata["name"] == "static_db" for dp in p_lca.static_packages)
    assert all(dp.metadata["name"] == "stochastic_db" for dp in p_lca.stochastic_packages)


def test_multiple_static_databases():
    """Two static packages both end up in static_packages."""
    static_dp1 = _make_static_dp()
    static_dp2 = bwp.create_datapackage(name="extra_static")

    # Minimal valid static package
    static_dp2.add_persistent_vector(
        matrix="technosphere_matrix",
        indices_array=np.array([(99, 999)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0]),
    )
    static_dp2.metadata["database_dependencies"] = []

    stochastic_dp = _make_stochastic_dp()
    method_dp = _make_method_dp()

    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=["static_db", "extra_static"],
        data_objs=[static_dp1, static_dp2, stochastic_dp, method_dp],
    )
    # static_db: 2 groups (tech + bio); extra_static: 1 group (tech only)
    assert len(p_lca.static_packages) == 3
    assert len(p_lca.stochastic_packages) == 2


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_non_integer_demand_key_raises():
    """Raises TypeError when demand keys are not integers."""
    static_dp = _make_static_dp()
    stochastic_dp = _make_stochastic_dp()
    method_dp = _make_method_dp()

    with pytest.raises(TypeError, match="integers"):
        PartitionedMonteCarloLCA(
            demand={"P3": 1.0},
            static_databases=["static_db"],
            data_objs=[static_dp, stochastic_dp, method_dp],
        )


def test_demand_in_static_database_raises():
    """Raises DemandInStaticDatabase when the demand key is an activity in the static system."""
    static_dp = _make_static_dp()
    stochastic_dp = _make_stochastic_dp()
    method_dp = _make_method_dp()

    # Activity 100 is in the static db, not the stochastic one
    p_lca = PartitionedMonteCarloLCA(
        demand={100: 1.0},
        static_databases=["static_db"],
        data_objs=[static_dp, stochastic_dp, method_dp],
    )
    with pytest.raises(DemandInStaticDatabase):
        p_lca.lci()


def test_missing_database_dependencies_raises():
    """Raises MissingDatabaseDependencies when metadata field is absent."""
    static_dp = _make_static_dp()
    # Remove the field we just added
    del static_dp.metadata["database_dependencies"]

    stochastic_dp = _make_stochastic_dp()
    method_dp = _make_method_dp()

    with pytest.raises(MissingDatabaseDependencies):
        PartitionedMonteCarloLCA(
            demand={3: 1.0},
            static_databases=["static_db"],
            data_objs=[static_dp, stochastic_dp, method_dp],
        )


def test_static_depends_on_stochastic_raises():
    """Raises StaticDependsOnStochastic when static db lists stochastic as dependency."""
    # static_db "depends on" stochastic_db — invalid
    static_dp = _make_static_dp(database_dependencies=["stochastic_db"])
    stochastic_dp = _make_stochastic_dp()
    method_dp = _make_method_dp()

    with pytest.raises(StaticDependsOnStochastic):
        PartitionedMonteCarloLCA(
            demand={3: 1.0},
            static_databases=["static_db"],
            data_objs=[static_dp, stochastic_dp, method_dp],
        )


def test_cyclic_dependency_raises():
    """Raises CyclicDependencyGraph when two static databases depend on each other."""
    dp_a = bwp.create_datapackage(name="db_a")
    dp_a.add_persistent_vector(
        matrix="technosphere_matrix",
        indices_array=np.array([(1, 10)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0]),
    )
    dp_a.metadata["database_dependencies"] = ["db_b"]

    dp_b = bwp.create_datapackage(name="db_b")
    dp_b.add_persistent_vector(
        matrix="technosphere_matrix",
        indices_array=np.array([(2, 20)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0]),
    )
    dp_b.metadata["database_dependencies"] = ["db_a"]

    stochastic_dp = _make_stochastic_dp(database_dependencies=[])
    method_dp = _make_method_dp()

    with pytest.raises(CyclicDependencyGraph):
        PartitionedMonteCarloLCA(
            demand={3: 1.0},
            static_databases=["db_a", "db_b"],  # both static so cycle is detected
            data_objs=[dp_a, dp_b, stochastic_dp, method_dp],
        )


def test_no_interface_products_returns_empty_dp():
    """When stochastic system is self-contained, dynamic dp is empty but LCA works."""
    # A self-contained stochastic system (no interface products)
    dp = bwp.create_datapackage(name="self_contained")
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        indices_array=np.array([(3, 300)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0]),
    )
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        indices_array=np.array([(1000, 300)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([0.5]),
    )
    dp.metadata["database_dependencies"] = []

    method_dp = _make_method_dp()

    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=[],  # no static databases
        data_objs=[dp, method_dp],
    )
    p_lca.lci()
    p_lca.lcia()

    assert np.isclose(p_lca.score, 0.5 * 2.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# Mixed-matrix datapackage tests
# ---------------------------------------------------------------------------


def test_mixed_lci_and_characterization_in_one_package():
    """A single datapackage containing both static LCI groups and a characterization group."""
    # Combine static tech + bio + characterization into one datapackage
    combined_dp = bwp.create_datapackage(name="static_and_method")

    combined_dp.add_persistent_vector(
        matrix="technosphere_matrix",
        name=bwp.clean_datapackage_name("static_db technosphere_matrix"),
        indices_array=np.array([(1, 100), (2, 200), (1, 200)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 1.0, 0.5]),
        flip_array=np.array([False, False, True]),
    )
    combined_dp.add_persistent_vector(
        matrix="biosphere_matrix",
        name=bwp.clean_datapackage_name("static_db biosphere_matrix"),
        indices_array=np.array([(1000, 100), (2000, 200)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([2.0, 1.0]),
    )
    combined_dp.add_persistent_vector(
        matrix="characterization_matrix",
        name="method_group",
        indices_array=np.array([(1000, 1000), (2000, 2000)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([2.0, 1.0]),
    )
    combined_dp.metadata["database_dependencies"] = []

    stochastic_dp = _make_stochastic_dp()

    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=["static_db"],
        data_objs=[combined_dp, stochastic_dp],
    )
    assert len(p_lca.static_packages) == 2  # tech + bio groups
    assert len(p_lca.stochastic_packages) == 2  # stochastic tech + bio groups
    assert len(p_lca.method_packages) == 1

    p_lca.lci()
    p_lca.lcia()
    assert np.isclose(p_lca.score, 3.2, rtol=1e-6)


def test_static_and_stochastic_groups_in_one_package():
    """A single datapackage containing both a static and a stochastic technosphere group."""
    combined_dp = bwp.create_datapackage(name="combined")

    combined_dp.add_persistent_vector(
        matrix="technosphere_matrix",
        name=bwp.clean_datapackage_name("static_db technosphere_matrix"),
        indices_array=np.array([(1, 100), (2, 200), (1, 200)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 1.0, 0.5]),
        flip_array=np.array([False, False, True]),
    )
    combined_dp.add_persistent_vector(
        matrix="biosphere_matrix",
        name=bwp.clean_datapackage_name("static_db biosphere_matrix"),
        indices_array=np.array([(1000, 100), (2000, 200)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([2.0, 1.0]),
    )
    combined_dp.add_persistent_vector(
        matrix="technosphere_matrix",
        name=bwp.clean_datapackage_name("stochastic_db technosphere_matrix"),
        indices_array=np.array([(3, 300), (2, 300)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 1.0]),
        flip_array=np.array([False, True]),
    )
    combined_dp.add_persistent_vector(
        matrix="biosphere_matrix",
        name=bwp.clean_datapackage_name("stochastic_db biosphere_matrix"),
        indices_array=np.array([(1000, 300)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([0.1]),
    )
    combined_dp.metadata["database_dependencies"] = []

    method_dp = _make_method_dp()

    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=["static_db"],
        data_objs=[combined_dp, method_dp],
    )
    assert len(p_lca.static_packages) == 2  # static tech + bio
    assert len(p_lca.stochastic_packages) == 2  # stochastic tech + bio
    assert len(p_lca.method_packages) == 1

    p_lca.lci()
    p_lca.lcia()
    assert np.isclose(p_lca.score, 3.2, rtol=1e-6)


def test_stochastic_characterization_varies():
    """Stochastic characterization factors produce varying scores across MC iterations."""
    static_dp = _make_static_dp()
    stochastic_dp = _make_stochastic_dp()

    # Method with uniform distribution on CF for F1: range [1.0, 3.0], nominal 2.0
    method_dp = bwp.create_datapackage(name="stochastic_method")
    char_indices = np.array([(1000, 1000), (2000, 2000)], dtype=bwp.INDICES_DTYPE)
    char_data = np.array([2.0, 1.0])
    char_dists = np.array(
        [
            (4, 2.0, 0.0, np.nan, 1.0, 3.0, False),  # uniform [1.0, 3.0] for F1
            (0, 1.0, 0.0, np.nan, np.nan, np.nan, False),  # deterministic for F2
        ],
        dtype=bwp.UNCERTAINTY_DTYPE,
    )
    method_dp.add_persistent_vector(
        matrix="characterization_matrix",
        indices_array=char_indices,
        data_array=char_data,
        distributions_array=char_dists,
    )

    p_lca = PartitionedMonteCarloLCA(
        demand={3: 1.0},
        static_databases=["static_db"],
        data_objs=[static_dp, stochastic_dp, method_dp],
        seed_override=99,
    )
    p_lca.lci()
    p_lca.lcia()

    scores = [p_lca.score]
    for _ in range(20):
        next(p_lca)
        scores.append(p_lca.score)

    assert len(set(np.round(scores, 8))) > 1, "Scores must vary when CFs have distributions"
