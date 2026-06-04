"""Registry key hashing from masking/baseline hyperparameters."""
import hashlib
import json


def _registry_key_hash(comparable):
    """MD5 of the comparable-args dict with stable key ordering.

    Call before num_exp/run_id are added so appendable runs share one key.
    """
    key_json = json.dumps(comparable, sort_keys=True)
    return hashlib.md5(key_json.encode("utf-8")).hexdigest()


def masked_key_from_args(args):
    """Hash of masking hyperparameters, excluding the sample range for appendable runs."""
    comparable = {
        "dataset": args.dataset,
        "network": args.network,
        "pretrained": bool(args.pretrained),
        "lr": args.lr,
        "gamma": args.gamma,
        "grad_loss": args.grad_loss,
        "num_dummy": args.num_dummy,
        "iteration": args.iteration,
        "tv_weight": args.tv_weight,
        "optimizer": args.optimizer,
        "num_restarts": args.num_restarts,
        "max_iteration": args.max_iteration,
        "history_size": args.history_size,
        "mask_mode": args.mask_mode,
        "gradsize_topk": args.gradsize_topk,
        "gradsize_topfrac": args.gradsize_topfrac,
        "gradsize_metric": args.gradsize_metric,
        "prefixes": args.prefixes,
        # Regime marker: the last FC layer is no longer force-included in the
        # reconstruction mask (it is used only for label inference). This field
        # gives new runs a distinct key so their results never collide with or
        # append to pre-change entries that were produced with FC force-included.
        "fc_forced": False,
    }
    key_hash = _registry_key_hash(comparable)
    comparable["num_exp"] = args.num_exp
    comparable["run_id"] = args.run_id
    return key_hash, comparable


def baseline_key_from_args(args):
    """Hash of baseline hyperparameters, excluding the sample range for appendable runs."""
    comparable = {
        "dataset": args.dataset,
        "network": args.network,
        "pretrained": bool(args.pretrained),
        "lr": args.lr,
        "gamma": args.gamma,
        "grad_loss": args.grad_loss,
        "num_dummy": args.num_dummy,
        "iteration": args.iteration,
        "tv_weight": args.tv_weight,
        "optimizer": args.optimizer,
        "num_restarts": args.num_restarts,
        "max_iteration": args.max_iteration,
        "history_size": args.history_size,
    }

    key_hash = _registry_key_hash(comparable)
    comparable["num_exp"] = args.num_exp
    comparable["run_id"] = args.run_id
    return key_hash, comparable
