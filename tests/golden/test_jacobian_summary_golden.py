"""Golden test: the Jacobian-sweep mean/std summarization is numerically stable
across the Phase 8 move.

`_summarize_rank_results` converts per-sample rank lists into the mean/std series
that the sweep CLI writes to its CSV and plots. The synthetic inputs make the
output reproducible without a dataset or GPU. Imported from the OLD path here;
repointed to stable_ginv.jacobian.sweep in Task 5.
"""
from functions.jacobian_rank_sweep import _summarize_rank_results
from tests.golden.helpers import load_or_regen

# Three "samples" per row count; a separate qr pool so both branches are exercised.
_RANK_RESULTS = {
    3072: [3000, 3050, 3072],
    4000: [3500, 3600, 3700],
    5000: [4100, 4250, 4400],
}
_RANK_RESULTS_QR = {
    3072: [3010, 3040, 3070],
    4000: [3550, 3620, 3680],
    5000: [4150, 4260, 4380],
}


def _produce():
    run = _summarize_rank_results(
        list(_RANK_RESULTS.keys()), _RANK_RESULTS, _RANK_RESULTS_QR
    )
    return {
        "xs": run["xs"],
        "mean_ranks": run["mean_ranks"],
        "std_ranks": run["std_ranks"],
        "mean_ranks_qr": run["mean_ranks_qr"],
        "std_ranks_qr": run["std_ranks_qr"],
    }


def test_jacobian_summary_stable():
    golden = load_or_regen("jacobian_summary_golden.json", _produce)
    assert _produce() == golden
