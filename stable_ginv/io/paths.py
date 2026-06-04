"""HPC/local storage path resolution."""
import os


class StoragePaths:
    """Resolve dataset/results paths: DTU HPC tree when writable, else local."""

    HPC_ROOT = "/work3/s234843/bachelor"

    @classmethod
    def resolve(cls, root_path="."):
        """Return (data_path, save_path) for DTU HPC when available, otherwise local paths."""
        if os.access(cls.HPC_ROOT, os.R_OK | os.W_OK | os.X_OK):
            return f"{cls.HPC_ROOT}/datasets", f"{cls.HPC_ROOT}/results"
        data_path = os.path.join(root_path, "data").replace("\\", "/")
        save_path = os.path.join(root_path, "results").replace("\\", "/")
        return data_path, save_path


def resolve_storage_paths(root_path="."):
    """Return (data_path, save_path) for DTU HPC when available, otherwise local paths."""
    return StoragePaths.resolve(root_path)
