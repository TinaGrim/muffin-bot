#!/usr/bin/env python3
"""Render an evaluation JSONL as a readable side-by-side report."""

import argparse
import json
from pathlib import Path


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    keys = ["exact", "char", "f1", "ngram", "len", "score"]
    return {k: round(sum(r[k] for r in rows) / n, 3) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    loaded = {}
    for path in args.files:
        rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
        loaded[path] = rows
        print(f"\n## {path}  ({len(rows)} samples)")
        print(summarize(rows))

    first = next(iter(loaded.values()))
    print("\n## Side by side\n")
    for row in first[: args.show]:
        print(f"**Bong:** {row['user']}")
        print(f"**Muffin (real):** {row['reference']}")
        for path, rows in loaded.items():
            match = next((r for r in rows if r["i"] == row["i"]), None)
            if match:
                print(f"**{Path(path).stem}:** {match['generated']}")
        print()


if __name__ == "__main__":
    main()
