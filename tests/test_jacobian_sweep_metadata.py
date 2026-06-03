import pytest

from functions.jacobian_rank_sweep import _parse_explicit_sample_indices


def test_explicit_indices_drive_the_recorded_sample_count():
    # When --sample_indices is given, the recorded sample count must come from
    # the actual list, not from the (unrelated) --num_samples default.
    indices = _parse_explicit_sample_indices("39508,23784", 60000)
    assert indices == [39508, 23784]
    assert len(indices) == 2


def test_explicit_indices_tolerate_spaces_and_trailing_commas():
    assert _parse_explicit_sample_indices(" 5, 7 ,9, ", 100) == [5, 7, 9]


def test_explicit_indices_reject_out_of_range():
    with pytest.raises(ValueError, match="out of range for dataset of size 100"):
        _parse_explicit_sample_indices("5,100", 100)
    with pytest.raises(ValueError):
        _parse_explicit_sample_indices("-1", 100)
