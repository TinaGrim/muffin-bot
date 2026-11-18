#!/usr/bin/env python3
"""Summarize accumulated audit results and group remaining mismatches."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    all_path = ROOT / "data" / "audit_all.jsonl"
    rows = [json.loads(line) for line in all_path.open(encoding="utf-8")]
    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(row["question"], row)
    rows = list(unique.values())
    exact = [r for r in rows if r["exact"]]
    near = [r for r in rows if r["exact"] or r["overlap"] >= 0.6]
    bad = [r for r in rows if not r["exact"]]

    print(f"unique questions: {len(rows)}")
    print(f"exact:            {len(exact)} ({len(exact) / len(rows):.1%})")
    print(f"same meaning:     {len(near)} ({len(near) / len(rows):.1%})")
    print(f"mismatches:       {len(bad)}")

    buckets = Counter()
    for row in bad:
        q, real = row["question"], row["real"]
        if re.fullmatch(r"[\W\d_]+", real.strip()):
            buckets["real answer is emoji/punctuation only"] += 1
        elif "http" in real:
            buckets["real answer is a link"] += 1
        elif "ភ" in real or re.search(r"[\u1780-\u17ff]", real):
            buckets["Khmer script converted to Khmerlish"] += 1
        elif row["diagnosis"] == "deflection":
            buckets["deliberate deflection (no context)"] += 1
        else:
            buckets["ambiguous or context-dependent"] += 1
    print("\nremaining mismatch buckets:")
    for name, count in buckets.most_common():
        print(f"  {count:4d}  {name}")

    lines = [
        "# Audit summary",
        "",
        f"**{len(exact)}/{len(rows)} exact ({len(exact) / len(rows):.1%})** "
        f"· same meaning {len(near)}/{len(rows)} "
        f"({len(near) / len(rows):.1%})",
        "",
        "## Remaining mismatches",
        "",
        "| Bong said | Real Muffin | Bot |",
        "|---|---|---|",
    ]
    for row in bad:
        clip = lambda s: s.replace("\n", " / ")[:70]
        lines.append(
            f"| {clip(row['question'])} | {clip(row['real'])} | {clip(row['bot'])} |"
        )
    (ROOT / "data" / "AUDIT_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\nwrote data/AUDIT_SUMMARY.md")


if __name__ == "__main__":
    main()
