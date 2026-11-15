import itertools

import numpy as np
import pytest

from emputation.patterns import (ccmv_terms, expand_tree_edges, mcar_terms, pattern_to_str, response_matrix,
                                 str_to_pattern, strictly_less, tree_terms, unique_patterns, validate_tree_edges)


def brute_force_mcar(R):
    """Literal transcription of the MCAR term loop of the R package."""
    uniq = []
    for r in map(tuple, R):
        if r not in uniq:
            uniq.append(r)
    rows, pats, masks = [], [], []
    for i, Ri in enumerate(map(np.array, R)):
        for r in uniq:
            r = np.array(r)
            if np.all(r <= Ri) and np.any(r < Ri):
                rows.append(i), pats.append(r), masks.append((Ri - r) == 1)
    return np.array(rows), np.array(pats), np.array(masks)


def test_pattern_strings_round_trip():
    r = np.array([1, 0, 1, 1], dtype=np.int8)
    assert pattern_to_str(r) == "1011"
    assert np.array_equal(str_to_pattern("1011"), r)


def test_unique_patterns_keep_first_appearance_order():
    R = np.array([[1, 0], [1, 1], [1, 0], [0, 1]])
    patterns, inv = unique_patterns(R)
    assert patterns.tolist() == [[1, 0], [1, 1], [0, 1]]
    assert np.array_equal(patterns[inv], R)


def test_strictly_less_matches_definition():
    P = np.array(list(itertools.product([0, 1], repeat=3)))
    less = strictly_less(P)
    for a, b in itertools.product(range(len(P)), repeat=2):
        assert less[a, b] == (np.all(P[b] <= P[a]) and np.any(P[b] < P[a]))


def test_mcar_terms_match_brute_force(incomplete_data):
    R = response_matrix(incomplete_data[1])
    terms = mcar_terms(R)
    rows, pats, masks = brute_force_mcar(R)
    assert np.array_equal(terms.rows, rows)
    assert np.array_equal(terms.input_patterns, pats)
    assert np.array_equal(terms.target_mask, masks)
    assert np.all(terms.n_masked >= 1)


def test_ccmv_terms(incomplete_data):
    R = response_matrix(incomplete_data[1])
    terms = ccmv_terms(R)
    complete = np.flatnonzero(R.all(1))
    incomplete = {pattern_to_str(r) for r in R if not r.all()}
    assert set(terms.rows) == set(complete)
    assert terms.n_terms == complete.size * len(incomplete)
    assert np.array_equal(terms.target_mask, terms.input_patterns == 0)


def test_ccmv_requires_complete_cases():
    with pytest.raises(ValueError, match="complete case"):
        ccmv_terms(np.array([[1, 0], [0, 1]]))


def test_tree_terms_and_expansion():
    R = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]])
    full, small, idx = expand_tree_edges({"10": "11", "00": "10"}, [1, 2], None, 3)
    assert full == {"110": "111", "100": "110"} and idx.tolist() == [1, 2]
    terms = tree_terms(R, full)
    # rows 0 and 3 (complete) train pattern 110; rows 1 and 4 (pattern 110) train 100
    assert terms.rows.tolist() == [0, 1, 3, 4]
    assert [pattern_to_str(m) for m in terms.target_mask] == ["001", "010", "001", "010"]


@pytest.mark.parametrize("edges, message", [
    ({"11": "11"}, "root"),
    ({"01": "10"}, "dominate"),
    ({"0": "1"}, "length"),
    ({"00": "01", "01": "00"}, "dominate"),
    ({"00": "01"}, "path to the root"),
])
def test_validate_tree_edges_errors(edges, message):
    with pytest.raises(ValueError, match=message):
        validate_tree_edges(edges, 2)
