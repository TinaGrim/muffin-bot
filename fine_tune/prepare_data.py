#!/usr/bin/env python3
"""Build a clean SFT bundle for LoRA fine-tuning.

Reads the existing chat-format splits, removes noise and duplicates, caps
ramble length, and writes train/valid JSONL ready for Unsloth, Axolotl or
MLX. Every record keeps the same `messages` shape the bot uses at inference.

Usage:
    python fine_tune/prepare_data.py
    python fine_tune/prepare_data.py --max-chars 200 --min-score 0
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = re.compile(r"https?://|www\.", re.I)
REPEAT = re.compile(r"(.)\1{3,}")
LINK_OR_CODE = re.compile(r"```|<html|function\s*\(")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_record(record: dict, max_chars: int, min_words: int) -> dict | None:
    messages = record.get("messages") or []
    if len(messages) < 3:
        return None
    if messages[0].get("role") != "system":
        return None
    if messages[-1].get("role") != "assistant":
        return None
    if messages[-2].get("role") != "user":
        return None

    cleaned = []
    for message in messages:
        content = norm(message.get("content", ""))
        if not content:
            return None
        cleaned.append({"role": message["role"], "content": content})

    user = cleaned[-2]["content"]
    reply = cleaned[-1]["content"]
    if len(user) > 400 or len(reply) > max_chars:
        return None
    if len(reply.split()) < min_words:
        return None
    if URL.search(user) or URL.search(reply):
        return None
    if LINK_OR_CODE.search(user) or LINK_OR_CODE.search(reply):
        return None
    if REPEAT.search(reply) or not re.search(r"[A-Za-z\u1780-\u17ff]", reply):
        return None
    if reply.casefold() == user.casefold():
        return None
    return {"messages": cleaned}


def dedupe(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[tuple[str, str]] = set()
    kept: list[dict] = []
    dropped = 0
    for record in records:
        messages = record["messages"]
        key = (messages[-2]["content"].casefold(), messages[-1]["content"].casefold())
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(record)
    return kept, dropped


def load(path: Path, max_chars: int, min_words: int) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        print(f"skip missing {path}")
        return records
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        cleaned = clean_record(json.loads(line), max_chars, min_words)
        if cleaned:
            records.append(cleaned)
    return records


def write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def stats(records: list[dict], label: str) -> None:
    lengths = [len(record["messages"][-1]["content"].split()) for record in records]
    if not lengths:
        print(f"{label}: 0")
        return
    lengths.sort()
    print(
        f"{label}: {len(records)} | words p50 {lengths[len(lengths)//2]} "
        f"p95 {lengths[int(len(lengths)*0.95)]} max {lengths[-1]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "data" / "sft_train.jsonl"))
    parser.add_argument("--valid", default=str(ROOT / "data" / "sft_val.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "fine_tune" / "data"))
    parser.add_argument("--max-chars", type=int, default=200)
    parser.add_argument("--min-words", type=int, default=1)
    args = parser.parse_args()

    train = load(Path(args.train), args.max_chars, args.min_words)
    valid = load(Path(args.valid), args.max_chars, args.min_words)
    train, dropped_dupes = dedupe(train)

    # A held-out valid answer must not appear in train.
    train_keys = {
        (r["messages"][-2]["content"].casefold(), r["messages"][-1]["content"].casefold())
        for r in train
    }
    valid = [
        r
        for r in valid
        if (r["messages"][-2]["content"].casefold(), r["messages"][-1]["content"].casefold())
        not in train_keys
    ]
    valid, _ = dedupe(valid)

    out = Path(args.out)
    train_path = out / "train.jsonl"
    valid_path = out / "valid.jsonl"
    write(train_path, train)
    write(valid_path, valid)

    stats(train, "train")
    stats(valid, "valid")
    print(f"duplicates removed: {dropped_dupes}")
    print(f"wrote {train_path} and {valid_path}")


if __name__ == "__main__":
    main()
