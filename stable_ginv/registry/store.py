"""JSON registry load/save for masked and baseline registries."""
import json
import os

from stable_ginv.io import safe_write


def _load_registry(path):
    """Load a JSON registry from path; return an empty dict if it does not exist."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_masked_registry(path):
    """Load JSON masked registry from path; returns empty dict if file does not exist."""
    return _load_registry(path)


def _save_registry(path, registry):
    """Write a JSON registry to path. Failures are logged, not raised."""
    return safe_write(path, lambda f: json.dump(registry, f, indent=2))


def save_masked_registry(path, registry):
    """Write masked registry dict to JSON at path. Failures are logged, not raised."""
    return _save_registry(path, registry)


def load_baseline_registry(path):
    """Load JSON baseline registry from path; returns empty dict if file does not exist."""
    return _load_registry(path)


def save_baseline_registry(path, registry):
    """Write baseline registry dict to JSON at path. Failures are logged, not raised."""
    return _save_registry(path, registry)
