import csv
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python sort_results.py <path_to_csv>")
    sys.exit(1)

input_file = Path(sys.argv[1])
if not input_file.exists():
    print(f"Error: file not found: {input_file}")
    sys.exit(1)

output_file = input_file  # overwrite in place; change if you want a separate file

with open(input_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

rows.sort(key=lambda r: r["timestamp"])

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Sorted {len(rows)} rows by timestamp → {output_file}")