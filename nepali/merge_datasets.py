#!/usr/bin/env python3
"""
Merge all 4 Nepali complaint CSVs into a single training-ready dataset.
Keeps columns: text, category, source, split
"""

import pandas as pd
import os

BASE = os.path.join(os.path.dirname(__file__), "data")

files = [
    "electricity_nepali.csv",
    "water_nepali.csv",
    "road_nepali.csv",
    "garbage_nepali.csv",
]

dfs = []
for f in files:
    path = os.path.join(BASE, f)
    df = pd.read_csv(path)
    print(f"  Loaded {f}: {len(df)} rows")
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)

# Keep only relevant columns
merged = merged[["text", "category", "source", "split"]]

# Assign new sequential IDs
merged.insert(0, "id", range(1, len(merged) + 1))

# Summary
print(f"\n{'='*50}")
print(f"Total rows: {len(merged)}")
print(f"\nCategory distribution:")
print(merged["category"].value_counts().to_string())
print(f"\nSource distribution:")
print(merged["source"].value_counts().to_string())
print(f"\nSplit distribution:")
print(merged["split"].value_counts().to_string())

# Save
out_path = os.path.join(BASE, "merged_nepali.csv")
merged.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
