import itertools
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from analysis import bh, exact_paired_p, metadata, validate_expression, ROOT


def test_bh_known():
    assert np.allclose(
        bh([0.01, 0.04, 0.03, 0.9]), [0.04, 0.053333333333, 0.053333333333, 0.9]
    )


def test_bh_matches_scipy():
    p = np.random.default_rng(8).uniform(size=100)
    assert np.allclose(bh(p), stats.false_discovery_control(p))


@pytest.mark.parametrize("p", [[], [np.nan], [-0.1], [1.1], [[0.1]]])
def test_bad_p(p):
    with pytest.raises(ValueError):
        bh(p)


def test_exact_independent_scalar():
    d = np.array([[1, 0, -2], [3, 0, -1], [2, 0, -4], [4, 0, -3]], float)
    expected = []
    for col in d.T:
        expected.append(
            sum(
                abs(np.dot(s, col) / 4) >= abs(np.mean(col)) - 1e-12
                for s in itertools.product([-1, 1], repeat=4)
            )
            / 16
        )
    assert np.allclose(exact_paired_p(d), expected)
    assert exact_paired_p(d)[1] == 1
    assert exact_paired_p(d)[0] == 0.125


@pytest.mark.parametrize(
    "d",
    [
        np.ones((1, 2)),
        np.array([[1, np.nan], [2, 3]]),
        np.ones(3),
        np.ones((17, 2)),
        np.ones((4, 0)),
    ],
)
def test_bad_pair(d):
    with pytest.raises(ValueError):
        exact_paired_p(d)


def test_negating_pair_keeps_p():
    d = np.arange(12).reshape(4, 3)
    assert np.array_equal(exact_paired_p(d), exact_paired_p(-d))


@pytest.mark.parametrize("kind", ["negative", "missing", "duplicate", "nonfinite"])
def test_expression_rejects(kind):
    m = pd.DataFrame({"sample_id": ["a", "b"]})
    x = pd.DataFrame(np.ones((2, 500)), index=["a", "b"])
    if kind == "negative":
        x.iloc[0, 0] = -1
    if kind == "missing":
        x.index = ["a", "c"]
    if kind == "duplicate":
        x.index = ["a", "a"]
    if kind == "nonfinite":
        x.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        validate_expression(x, m)


def test_alignment_is_by_id():
    m = pd.DataFrame({"sample_id": ["b", "a"]})
    x = pd.DataFrame(np.ones((2, 500)), index=["a", "b"])
    assert list(validate_expression(x, m).index) == ["b", "a"]


def test_metadata_join_unique():
    p = ROOT / "data/raw/GSE107011_series_matrix.txt.gz"
    if not p.exists():
        pytest.skip("Live metadata downloaded by analysis runner")
    m = metadata(p)
    assert len(m) == 127
    assert m.sample_id.is_unique
