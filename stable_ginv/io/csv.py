"""Append-only CSV writers built on safe_write."""
import csv
import os

from stable_ginv.io.fs import safe_write


def append_csv_rows(path, rows, fieldnames):
    """Append rows to a CSV file, writing the header when the file is new."""
    file_exists = os.path.isfile(path)

    def _append(f):
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    return safe_write(path, _append, mode="a", newline="")


def append_csv_row(path, row, fieldnames):
    """Append one row to a CSV file, writing the header when the file is new."""
    return append_csv_rows(path, [row], fieldnames)
