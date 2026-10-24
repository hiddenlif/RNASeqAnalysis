"""Offline tests of the evaluation design and the metadata parser, on small synthetic matrices and files (no downloads).

The matrices have 4 donors x 6 classes x 2 replicates over 2,000 features. They let the tests check, with known truth, that
donor holdout never shares a donor, that real signal is recovered, that shuffled labels fall to chance when feature selection
sits inside each training fold, and that the same shuffled labels do NOT fall to chance when features are chosen on all
samples first (the leak the design avoids).
"""
import gzip
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from analysis import (bh, evaluate, exact_paired_p, metadata, permutation_control, selection_outside_folds_accuracy)

DONORS, CLASSES, REPS, FEATURES = 4, 6, 2, 2000


def synthetic(seed, signal):
    rng = np.random.default_rng(seed)
    donor, label = np.meshgrid(np.arange(DONORS), np.arange(CLASSES), indexing="ij")
    donor, label = np.repeat(donor.ravel(), REPS), np.repeat(label.ravel(), REPS)
    x = rng.normal(size=(len(label), FEATURES))
    if signal:
        x[:, :40] += 2.5 * rng.normal(size=(CLASSES, 40))[label]
    return x, np.array([f"type{k}" for k in label], dtype=object), np.array([f"D{k}" for k in donor], dtype=object)


def test_donor_holdout_never_shares_a_donor_and_scores_every_sample_once():
    x, y, g = synthetic(0, signal=True)
    pred, baseline, records, folds = evaluate(x, y, g, LeaveOneGroupOut(), True)
    assert [r["overlapping_donors"] for r in records] == [0] * DONORS
    assert sorted(f["sample_index"] for f in folds) == list(range(len(y)))
    for f in folds:
        assert f["donor"] == g[f["sample_index"]]


def test_a_splitter_that_shares_donors_is_rejected():
    x, y, g = synthetic(0, signal=True)
    with pytest.raises(ValueError, match="Donor leakage"):
        evaluate(x, y, g, StratifiedKFold(3, shuffle=True, random_state=0), True)


def test_real_signal_is_recovered_in_unseen_donors():
    x, y, g = synthetic(1, signal=True)
    pred, *_ = evaluate(x, y, g, LeaveOneGroupOut(), True)
    assert (pred == y).mean() > 0.9


def test_the_dummy_baseline_cannot_exceed_the_class_prior():
    x, y, g = synthetic(1, signal=True)
    _, baseline, *_ = evaluate(x, y, g, LeaveOneGroupOut(), True)
    assert (baseline == y).mean() <= 1 / CLASSES + 1e-9


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_shuffled_labels_fall_to_chance_in_fold_but_not_when_selection_comes_first(seed):
    x, y, g = synthetic(seed, signal=False)
    shuffled = np.random.default_rng(seed).permutation(y)
    pred, *_ = evaluate(x, shuffled, g, LeaveOneGroupOut(), True)
    in_fold = float((pred == shuffled).mean())
    leaky = selection_outside_folds_accuracy(x, shuffled, g, k=100)
    assert in_fold < 1 / CLASSES + 0.12
    assert leaky > in_fold + 0.25


def test_permutation_control_is_reproducible_and_complete():
    x, y, g = synthetic(5, signal=True)
    a = permutation_control(x, y, g, seeds=range(3))
    b = permutation_control(x, y, g, seeds=range(3))
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["seed", "in_fold_accuracy", "selection_outside_folds_accuracy"]
    assert a.seed.tolist() == [0, 1, 2] and ((a.drop(columns="seed") >= 0) & (a.drop(columns="seed") <= 1)).all().all()


@pytest.mark.parametrize("seed", range(25))
def test_bh_matches_scipy_with_ties_and_extremes(seed):
    rng = np.random.default_rng(seed)
    p = np.round(rng.uniform(size=int(rng.integers(1, 60))), 2)      # rounding creates ties
    p[rng.integers(len(p))] = 1.0
    assert np.allclose(bh(p), stats.false_discovery_control(p))
    assert (bh(p) >= p - 1e-12).all() and (bh(p) <= 1).all()


@pytest.mark.parametrize("seed", range(15))
def test_exact_paired_p_matches_brute_force_for_random_designs(seed):
    rng = np.random.default_rng(seed)
    donors, genes = int(rng.integers(2, 8)), int(rng.integers(1, 6))
    d = np.round(rng.normal(size=(donors, genes)), 1)                  # rounding creates zeros and ties
    expected = [
        sum(abs(np.dot(s, col) / donors) >= abs(col.mean()) - 1e-12 for s in itertools.product([-1, 1], repeat=donors)) / 2 ** donors
        for col in d.T
    ]
    assert np.allclose(exact_paired_p(d), expected)
    assert (exact_paired_p(d) >= 2 / 2 ** donors - 1e-12).all()


def series_matrix(path, titles, cell_types, source=None, extra=()):
    source = cell_types if source is None else source
    lines = [
        "!Series_title\tx",
        "!Sample_title\t" + "\t".join(f'"{t}"' for t in titles),
        "!Sample_geo_accession\t" + "\t".join(f'"GSM{i}"' for i in range(len(titles))),
        "!Sample_source_name_ch1\t" + "\t".join(f'"{c}"' for c in source),
        "!Sample_characteristics_ch1\t" + "\t".join(f'"cell type: {c}"' for c in cell_types),
        *extra,
    ]
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return path


def test_metadata_normalises_replicates_and_derives_donors(tmp_path):
    m = metadata(series_matrix(tmp_path / "s.gz", ["A1_NK", "A1_B_rep1", "B2_NK", "B2_B_rep2"], ["NK", "B", "NK", "B"]))
    assert m.sample_id.tolist() == ["A1_NK", "A1_B", "B2_NK", "B2_B"]
    assert m.donor.tolist() == ["A1", "A1", "B2", "B2"]
    assert m.cell_type.tolist() == ["NK", "B", "NK", "B"]


@pytest.mark.parametrize("case", ["duplicate title", "replicates collapse to one sample", "source disagrees", "two title rows", "no cell type row"])
def test_metadata_rejects_ambiguous_files(tmp_path, case):
    titles, types, source, extra = ["A1_NK", "A1_B", "B2_NK"], ["NK", "B", "NK"], None, ()
    if case == "duplicate title":
        titles = ["A1_NK", "A1_NK", "B2_NK"]
    elif case == "replicates collapse to one sample":
        titles = ["A1_NK_rep1", "A1_NK_rep2", "B2_NK"]
    elif case == "source disagrees":
        source = ["NK", "T", "NK"]
    elif case == "two title rows":
        extra = ('!Sample_title\t"a"\t"b"\t"c"',)
    path = series_matrix(tmp_path / "s.gz", titles, types, source, extra)
    if case == "no cell type row":
        text = gzip.open(path, "rt").read().replace("!Sample_characteristics_ch1", "!Sample_other")
        with gzip.open(path, "wt") as f:
            f.write(text)
    with pytest.raises(ValueError):
        metadata(path)
