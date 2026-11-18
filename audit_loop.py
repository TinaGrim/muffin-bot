#!/usr/bin/env python3
"""Run repeated fresh-question audits and accumulate the results.

Each round excludes every question already audited, so the loop keeps finding
new strange questions instead of re-testing the same ones.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    parser.add_argument("--statements", action="store_true")
    parser.add_argument("--all", default="data/audit_all.jsonl")
    args = parser.parse_args()

    all_path = ROOT / args.all
    seen = read(all_path)
    seen_keys = {row["question"] for row in seen}
    print(f"starting with {len(seen)} already-audited questions", flush=True)

    for round_no in range(1, args.rounds + 1):
        seed = args.seed + round_no * 101
        seen_path = ROOT / "data" / "audit_seen_tmp.jsonl"
        write(seen_path, seen)
        out = f"data/audit_round_{round_no:03d}"
        started = time.time()
        print(f"\n=== round {round_no}/{args.rounds} seed={seed} ===", flush=True)
        cmd = [
            sys.executable,
            str(ROOT / "question_audit.py"),
            "--n",
            str(args.n),
            "--seed",
            str(seed),
            "--exclude",
            str(seen_path),
            "--out",
            out,
            "--model",
            args.model,
        ]
        if args.statements:
            cmd.append("--statements")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode:
            print(f"round {round_no} FAILED", flush=True)
            break
        rows = read(ROOT / f"{out}.jsonl")
        seen += rows
        write(all_path, seen)
        exact = sum(row["exact"] for row in rows)
        close = sum(row["exact"] or row["overlap"] >= 0.6 for row in rows)
        print(
            f"round {round_no}: exact {exact}/{len(rows)} "
            f"same {close}/{len(rows)} in {time.time() - started:.0f}s",
            flush=True,
        )
        new = [row for row in rows if row["question"] not in seen_keys]
        seen_keys.update(row["question"] for row in new)
        if not new:
            print("no new questions left", flush=True)
            break

    total = read(all_path)
    exact = sum(row["exact"] for row in total)
    close = sum(row["exact"] or row["overlap"] >= 0.6 for row in total)
    print(f"\nTOTAL {len(total)} questions: exact {exact} ({exact / len(total):.1%})")
    print(f"same meaning {close} ({close / len(total):.1%})")


if __name__ == "__main__":
    main()
