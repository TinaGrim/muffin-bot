#!/usr/bin/env python3
"""Build a Q/A pair set from the real Muffin conversation (data/pairs.jsonl)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LINK = re.compile(r"https?://|www\.", re.I)


def clean(row: dict) -> bool:
    user = (row.get("user") or "").strip()
    reply = (row.get("reply") or "").strip()
    if not (2 <= len(user) <= 160 and 2 <= len(reply) <= 160):
        return False
    if LINK.search(user) or LINK.search(reply):
        return False
    if user.casefold() == reply.casefold():
        return False
    if str(row.get("reply_voice", "False")).lower() == "true":
        return False
    if str(row.get("reply_sticker", "False")).lower() == "true":
        return False
    if re.fullmatch(r"[\d\W_]+", reply):
        return False
    return True


def context_messages(row: dict) -> list[dict]:
    messages = []
    for turn in row.get("context") or []:
        user = (turn.get("user") or "").strip()
        reply = (turn.get("reply") or "").strip()
        if user:
            messages.append({"role": "user", "content": user})
        if reply:
            messages.append({"role": "assistant", "content": reply})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(DATA / "pairs.jsonl"))
    parser.add_argument("--out", default=str(DATA / "jev_pairs.jsonl"))
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.pairs).open(encoding="utf-8") if line.strip()]
    clean_rows = [row for row in rows if clean(row)]
    if not clean_rows:
        raise SystemExit("no clean pairs")

    # Spread the sample across the whole timeline instead of only one period.
    step = max(1, len(clean_rows) // args.limit)
    sample = clean_rows[::step][: args.limit]

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(sample, 1):
            record = {
                "id": f"jev-{index:04d}",
                "source_id": row.get("id"),
                "date": row.get("date"),
                "question": row["user"].strip(),
                "answer": row["reply"].strip(),
                "context": context_messages(row),
                "source": "real_chat",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{len(rows)} raw pairs -> {len(clean_rows)} clean -> {len(sample)} prepared")
    print(out)


if __name__ == "__main__":
    main()
