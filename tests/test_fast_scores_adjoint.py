import warnings
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

from bw2calc.errors import InaccurateSolution
from bw2calc.fast_scores import PYPARDISO, UMFPACK, FastScoresOnlyMultiLCA
from bw2calc.multi_lca import MultiLCA
from bw2calc.utils import get_datapackage, relative_residual

pytestmark = pytest.mark.filterwarnings("ignore:Using UMFPACK")

needs_solver = pytest.mark.skipif(
    not (PYPARDISO or UMFPACK), reason="Fast sparse solvers not installed"
)

FULL_CONFIG = {
    "impact_categories": [
        ("first", "category"),
        ("second", "category"),
    ],
    "normalizations": {
        ("n", "1"): [("first", "category")],
        ("n", "2"): [("second", "category")],
    },
    "weightings": {
        ("w", "1"): [("n", "1")],
        ("w", "2"): [("n", "2")],
    },
}


@pytest.fixture
def full_test_data(basic_test_data, fixture_dir):
    return {
        "dps": basic_test_data["dps"]
        + [
            get_datapackage(fixture_dir / "multi_lca_simple_normalization.zip"),
            get_datapackage(fixture_dir / "multi_lca_simple_weighting.zip"),
        ],
        "config": FULL_CONFIG,
        "demands": basic_test_data["demands"],
    }


def fast_scores(data, demands=None, **kwargs):
    lca = FastScoresOnlyMultiLCA(
        demands=data["demands"] if demands is None else demands,
        method_config=data["config"],
        data_objs=data["dps"],
        **kwargs,
    )
    lca.calculate()
    return lca


def multi_lca_scores(data, demands=None):
    lca = MultiLCA(
        demands=data["demands"] if demands is None else demands,
        method_config=data["config"],
        data_objs=data["dps"],
    )
    lca.lci()
    lca.lcia()
    return lca.scores


def assert_scores_match_multi_lca(fsmlca, expected):
    """`MultiLCA` keys scores by `(method key, demand)`; `FastScoresOnlyMultiLCA` by strings."""
    assert len(expected) == fsmlca.scores.size
    for (method, demand), value in expected.items():
        assert np.isclose(fsmlca.scores.loc[str(method), demand], value)


@needs_solver
@pytest.mark.parametrize("data_fixture", ["basic_test_data", "full_test_data"])
def test_adjoint_matches_forward_and_multi_lca(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    adjoint = fast_scores(data, direction="adjoint")
    forward = fast_scores(data, direction="forward")

    assert adjoint.scores.dims == forward.scores.dims == ("LCIA", "processes")
    assert list(adjoint.scores.coords["LCIA"].values) == list(forward.scores.coords["LCIA"].values)
    assert np.allclose(adjoint.scores.values, forward.scores.values)


@needs_solver
def test_adjoint_matches_multi_lca(basic_test_data):
    adjoint = fast_scores(basic_test_data, direction="adjoint")
    assert_scores_match_multi_lca(adjoint, multi_lca_scores(basic_test_data))


@needs_solver
def test_adjoint_non_unit_and_multi_product_demands(basic_test_data):
    demands = {
        "mixed": {100: 2.5, 103: -1, 105: 0.5},
        "single": {104: 7},
        "pair": {101: 1, 102: 3},
    }
    adjoint = fast_scores(basic_test_data, demands=demands, direction="adjoint")
    assert_scores_match_multi_lca(adjoint, multi_lca_scores(basic_test_data, demands=demands))


@needs_solver
@pytest.mark.parametrize("data_fixture", ["basic_test_data", "full_test_data"])
def test_product_scores_match_unit_demands(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    adjoint = fast_scores(data, direction="adjoint")

    products = list(adjoint.product_scores.coords["products"].values)
    assert sorted(products) == sorted(adjoint.dicts.product)
    assert adjoint.product_scores.dims == ("LCIA", "products")

    forward = fast_scores(
        data, demands={str(p): {int(p): 1} for p in products}, direction="forward"
    )
    for method in forward.scores.coords["LCIA"].values:
        for product in products:
            assert np.isclose(
                adjoint.product_scores.loc[method, product],
                forward.scores.loc[method, str(product)],
            )


@needs_solver
def test_auto_direction(basic_test_data):
    # Three demands, two impact categories
    lca = fast_scores(basic_test_data, direction="auto")
    assert lca.direction == "auto"
    assert hasattr(lca, "product_scores")
    assert not hasattr(lca, "supply_array")

    lca = fast_scores(basic_test_data, demands={"γ": {100: 1}, "ε": {103: 2}}, direction="auto")
    assert hasattr(lca, "supply_array")
    assert not hasattr(lca, "product_scores")


@needs_solver
def test_auto_direction_uses_forward_when_stochastic(basic_test_data):
    lca = fast_scores(basic_test_data, use_distributions=True, direction="auto")
    assert hasattr(lca, "supply_array")
    assert not hasattr(lca, "product_scores")

    lca = fast_scores(
        basic_test_data,
        selective_use={"characterization_matrix": {"use_arrays": True}},
        direction="auto",
    )
    assert hasattr(lca, "supply_array")
    assert not hasattr(lca, "product_scores")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_distributions": True},
        {"use_arrays": True},
        {"selective_use": {"technosphere_matrix": {"use_distributions": True}}},
    ],
)
def test_adjoint_stochastic_not_implemented(basic_test_data, kwargs):
    with pytest.raises(NotImplementedError, match="adjoint"):
        FastScoresOnlyMultiLCA(
            demands=basic_test_data["demands"],
            method_config=basic_test_data["config"],
            data_objs=basic_test_data["dps"],
            direction="adjoint",
            **kwargs,
        )


def test_invalid_direction(basic_test_data):
    with pytest.raises(ValueError, match="Invalid direction"):
        FastScoresOnlyMultiLCA(
            demands=basic_test_data["demands"],
            method_config=basic_test_data["config"],
            data_objs=basic_test_data["dps"],
            direction="sideways",
        )


@needs_solver
def test_adjoint_without_demands(basic_test_data):
    lca = fast_scores(basic_test_data, demands={}, direction="adjoint")
    assert lca.scores.shape == (2, 0)
    assert lca.product_scores.shape == (2, 6)
    assert np.isclose(lca.product_scores.loc["('first', 'category')", 100], 11)


@needs_solver
def test_adjoint_removes_stale_supply_array(basic_test_data):
    lca = fast_scores(basic_test_data, direction="forward")
    assert hasattr(lca, "supply_array")
    lca.direction = "adjoint"
    lca.calculate()
    assert not hasattr(lca, "supply_array")


@needs_solver
def test_adjoint_next(basic_test_data):
    lca = fast_scores(basic_test_data, direction="adjoint")
    first = lca.scores.values.copy()
    next(lca)
    assert np.allclose(lca.scores.values, first)


@needs_solver
def test_solve_adjoint_single_column(basic_test_data):
    lca = fast_scores(basic_test_data, direction="adjoint")
    rhs = np.ones(lca.technosphere_matrix.shape[0])
    solution = lca.solve_adjoint(rhs)
    assert solution.shape == (lca.technosphere_matrix.shape[1], 1)
    assert np.allclose(lca.technosphere_matrix.T @ solution[:, 0], rhs)


@needs_solver
def test_residual_check_raises(basic_test_data):
    lca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
        direction="adjoint",
    )
    original = FastScoresOnlyMultiLCA.solve_adjoint

    def wrong(self, rhs):
        return original(self, rhs) * 1.01

    with patch.object(FastScoresOnlyMultiLCA, "solve_adjoint", wrong):
        with pytest.raises(InaccurateSolution, match="relative residual"):
            lca.calculate()

        lca.residual_tolerance = None
        lca.calculate()
        assert hasattr(lca, "product_scores")


def test_solve_adjoint_no_solver(basic_test_data, no_solvers_available):
    lca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
        direction="adjoint",
    )
    lca.technosphere_matrix = sparse.identity(3, format="csr")
    with pytest.raises(ValueError, match="only supported with PARDISO and UMFPACK"):
        lca.solve_adjoint(np.ones((3, 2)))


def test_relative_residual():
    matrix = sparse.csr_matrix(np.array([[2.0, 0], [0, 4]]))
    rhs = np.array([[2.0, 0], [4, 0]])
    exact = np.array([[1.0, 0], [1, 0]])
    assert np.allclose(relative_residual(matrix, exact, rhs), 0)

    off = np.array([[1.0, 1], [1, 0]])
    residuals = relative_residual(matrix, off, rhs)
    assert residuals[0] == 0
    # Zero right-hand side column uses the absolute residual
    assert np.isclose(residuals[1], 2)

    # 1-d inputs are treated as a single column
    single = relative_residual(matrix, np.array([1.0, 2]), np.array([2.0, 4]))
    assert single.shape == (1,)
    assert np.isclose(single[0], 4 / np.sqrt(20))


def adjoint_lca(data, **kwargs):
    return FastScoresOnlyMultiLCA(
        demands=data["demands"],
        method_config=data["config"],
        data_objs=data["dps"],
        direction="adjoint",
        **kwargs,
    )


@needs_solver
def test_residual_check_raises_on_nan(basic_test_data):
    lca = adjoint_lca(basic_test_data)
    original = FastScoresOnlyMultiLCA.solve_adjoint

    def with_nan(self, rhs):
        solution = original(self, rhs)
        solution[0, 0] = np.nan
        return solution

    with patch.object(FastScoresOnlyMultiLCA, "solve_adjoint", with_nan):
        with pytest.raises(InaccurateSolution, match=r"\('first', 'category'\).*nan"):
            lca.calculate()


@needs_solver
def test_residual_check_raises_on_singular_technosphere(basic_test_data):
    lca = fast_scores(basic_test_data, direction="adjoint")
    singular = lca.technosphere_matrix.tolil()
    singular[1, :] = singular[0, :]
    lca.technosphere_matrix = singular.tocsr()
    # Solvers can also raise their own errors, but they must never return scores
    with pytest.raises(Exception):
        with np.errstate(all="ignore"):
            lca.calculate()
    assert not hasattr(lca, "product_scores")
    with pytest.raises(ValueError, match="Scores not calculated yet"):
        lca.scores


def test_direction_setter_validates(basic_test_data):
    lca = adjoint_lca(basic_test_data)
    with pytest.raises(ValueError, match="Invalid direction"):
        lca.direction = "adjont"
    assert lca.direction == "adjoint"

    stochastic = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
        use_distributions=True,
    )
    with pytest.raises(NotImplementedError, match="adjoint"):
        stochastic.direction = "adjoint"
    assert stochastic.direction == "forward"


@needs_solver
def test_forward_removes_stale_product_scores(basic_test_data):
    lca = fast_scores(basic_test_data, direction="adjoint")
    assert hasattr(lca, "product_scores")
    lca.direction = "forward"
    lca.calculate()
    assert not hasattr(lca, "product_scores")
    assert hasattr(lca, "supply_array")


@needs_solver
def test_failed_calculation_removes_earlier_results(basic_test_data):
    lca = fast_scores(basic_test_data, direction="adjoint")
    original = FastScoresOnlyMultiLCA.solve_adjoint

    def wrong(self, rhs):
        return original(self, rhs) * 1.01

    with patch.object(FastScoresOnlyMultiLCA, "solve_adjoint", wrong):
        with pytest.raises(InaccurateSolution):
            lca.calculate()

    assert not hasattr(lca, "product_scores")
    with pytest.raises(ValueError, match="Scores not calculated yet"):
        lca.scores


@needs_solver
def test_product_scores_with_remapped_product_dict(basic_test_data):
    lca = fast_scores(basic_test_data, direction="adjoint")
    expected = lca.product_scores.copy()
    lca.dicts.product.remap({key: ("db", str(key)) for key in list(lca.dicts.product)})
    lca.calculate()
    assert list(lca.product_scores.coords["products"].values) == list(
        expected.coords["products"].values
    )
    assert np.allclose(lca.product_scores.values, expected.values)


@needs_solver
def test_many_demands_match_forward(basic_test_data):
    rng = np.random.default_rng(42)
    demands = {
        str(i): {int(p): float(a) for p, a in zip(range(100, 106), rng.normal(size=6)) if a > 0}
        for i in range(40)
    }
    adjoint = fast_scores(basic_test_data, demands=demands, direction="adjoint")
    forward = fast_scores(basic_test_data, demands=demands, direction="forward")
    assert adjoint.scores.shape == (2, 40)
    assert np.allclose(adjoint.scores.values, forward.scores.values)


@needs_solver
def test_solve_adjoint_input_types(basic_test_data):
    lca = fast_scores(basic_test_data, direction="adjoint")
    rhs = np.arange(12, dtype=float).reshape(6, 2)
    expected = lca.solve_adjoint(rhs)

    assert np.allclose(lca.solve_adjoint(rhs.tolist()), expected)
    assert np.allclose(lca.solve_adjoint(sparse.csr_matrix(rhs)), expected)
    assert np.allclose(lca.solve_adjoint(rhs[:, 0].tolist()), expected[:, :1])
    assert lca.solve_adjoint(np.zeros((6, 0))).shape == (6, 0)

    with pytest.raises(ValueError, match="5 rows"):
        lca.solve_adjoint(np.ones((5, 2)))


def test_solve_adjoint_pardiso_error_not_masked(basic_test_data):
    class FakePardisoError(Exception):
        pass

    class FailingSolver:
        freed = False

        def factorize(self, matrix):
            raise FakePardisoError("factorization failed")

        def free_memory(self, everything=False):
            FailingSolver.freed = True
            raise FakePardisoError("nothing to free")

    lca = adjoint_lca(basic_test_data)
    lca.technosphere_matrix = sparse.identity(3, format="csr")
    with (
        patch("bw2calc.fast_scores.PYPARDISO", True),
        patch("bw2calc.fast_scores.PyPardisoSolver", FailingSolver),
        patch("bw2calc.fast_scores.PyPardisoError", FakePardisoError),
    ):
        with pytest.raises(FakePardisoError, match="factorization failed"):
            lca.solve_adjoint(np.ones((3, 2)))
    assert FailingSolver.freed


def test_relative_residual_input_types():
    matrix = sparse.csr_matrix(np.array([[2.0, 0], [0, 4]]))
    assert np.allclose(relative_residual(matrix, [[1.0], [1.0]], [[2.0], [4.0]]), 0)
    assert np.allclose(relative_residual(matrix, [1.0, 1.0], sparse.csr_matrix([[2.0], [4.0]])), 0)
    assert np.isnan(relative_residual(matrix, [np.nan, 1.0], [2.0, 4.0])[0])


@needs_solver
def test_default_direction_is_forward_with_future_warning(basic_test_data):
    # Three demands, two impact categories: "auto" would use adjoint
    with pytest.warns(FutureWarning, match='change from "forward" to "auto"'):
        lca = fast_scores(basic_test_data)
    assert lca.direction == "forward"
    assert hasattr(lca, "supply_array")
    assert not hasattr(lca, "product_scores")

    # Only once per instance
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        lca.calculate()


@needs_solver
@pytest.mark.parametrize(
    "kwargs",
    [
        {"direction": "forward"},
        {"direction": "auto"},
        {"direction": "adjoint"},
        # "auto" would also use forward for these
        {"demands": {"γ": {100: 1}, "ε": {103: 2}}},
        {"use_distributions": True},
    ],
)
def test_no_future_warning_when_default_makes_no_difference(basic_test_data, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        fast_scores(basic_test_data, **kwargs)


@needs_solver
def test_no_future_warning_after_setting_direction(basic_test_data):
    lca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    lca.direction = "forward"
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        lca.calculate()
