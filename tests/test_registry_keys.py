from types import SimpleNamespace

from functions.io_utils import baseline_key_from_args, masked_key_from_args

# Golden hashes lock the registry-key serialization. Existing registries on disk
# are addressed by these keys, so a change here would orphan stored results.
# Do NOT update these constants to make a refactor pass -- a diff here means the
# comparable-args dict or its JSON serialization changed.
#
# MASKED_KEY was updated once on purpose: the masked comparable dict gained the
# "fc_forced": False regime marker when the last FC layer stopped being
# force-included in the reconstruction mask (it is now used only for label
# inference). That marker deliberately re-keys masked runs so post-change results
# never append to pre-change entries.
MASKED_KEY = "dd009e9b82619dabccd075aa50249c51"
BASELINE_KEY = "a6727abd177d73cbab44e811a5f2bb41"


def _args(**overrides):
    base = dict(
        dataset="cifar100", network="vgg13", pretrained=False, lr=0.1, gamma=0.5,
        grad_loss="cos", num_dummy=1, iteration=5000, tv_weight=0.0,
        optimizer="signed_adamw", num_restarts=1, max_iteration=20, history_size=100,
        mask_mode="gradsize_topfrac_entries_layer", gradsize_topk=None,
        gradsize_topfrac=0.1, gradsize_metric="abs", prefixes="conv1:0.5,fc",
        num_exp=30, run_id=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_masked_key_is_stable():
    key, comparable = masked_key_from_args(_args())
    assert key == MASKED_KEY
    # The comparable dict carries the sample range back to the caller even though
    # it is excluded from the hash.
    assert comparable["num_exp"] == 30
    assert comparable["run_id"] == 0


def test_baseline_key_is_stable():
    key, comparable = baseline_key_from_args(_args())
    assert key == BASELINE_KEY
    assert comparable["num_exp"] == 30
    assert comparable["run_id"] == 0


def test_keys_ignore_sample_range():
    # num_exp/run_id must not affect the hash so appendable runs share one key.
    assert masked_key_from_args(_args(num_exp=99, run_id=42))[0] == MASKED_KEY
    assert baseline_key_from_args(_args(num_exp=99, run_id=42))[0] == BASELINE_KEY


def test_masked_and_baseline_keys_differ():
    # Masked keys incorporate the mask configuration; baseline keys do not.
    assert masked_key_from_args(_args())[0] != baseline_key_from_args(_args())[0]
