#!/usr/bin/env python3
"""Re-check previously failed audit questions with the current bot."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(ROOT / "src"))

from muffin import Chat, Muffin, load_config  # noqa: E402
from mood import Mood, MoodStore  # noqa: E402
from life import Activity, LifeStore  # noqa: E402
from history import HistoryStore  # noqa: E402
from reactions import ReactionState, ReactionStore  # noqa: E402
from retrieval import CachedRetriever, normalize  # noqa: E402
from threads import ThreadState, ThreadStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    rows: list[dict] = []
    for name in args.files:
        rows += [
            json.loads(line)
            for line in (ROOT / name).open(encoding="utf-8")
            if line.strip()
        ]
    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(row["question"], row)
    rows = [row for row in unique.values() if not row["exact"]]

    config = load_config(
        {
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_API_KEY": "ollama",
            "LLM_MODEL": args.model,
            "DIRECT_REPLY_THRESHOLD": "0.9",
            "USE_RULES": "true",
            "EXAMPLES_MODE": "never",
        },
        ROOT / "persona.md",
    )
    retriever = CachedRetriever(
        ROOT / "data" / "pairs.jsonl", ROOT / "data" / "retrieval.pkl"
    ).get()
    tmp = Path(tempfile.mkdtemp())
    mood = MoodStore(tmp / "mood.json")
    life = LifeStore(tmp / "life.json")
    history = HistoryStore(tmp / "history.json")
    reactions = ReactionStore(tmp / "reactions.json")
    threads = ThreadStore(tmp / "threads.json")
    muffin = Muffin(config, retriever, mood, life, history, reactions, threads)

    fixed = 0
    today = date.today().isoformat()
    for i, row in enumerate(rows, 1):
        chat_id = 10000 + i
        muffin.chats[chat_id] = Chat(history=[])
        mood.chats[str(chat_id)] = Mood(energy=60, affection=60, patience=70)
        now = time.time()
        life.chats[str(chat_id)] = Activity(
            key=row["life"], started=now, ends=now + 3600
        )
        reactions.chats[str(chat_id)] = ReactionState(
            last_user_date=today, morning_greeted=today, last_user_ts=now
        )
        threads.chats[str(chat_id)] = ThreadState()
        answer = " | ".join(muffin.reply(chat_id, row["question"]))
        ok = normalize(answer) == normalize(row["real"])
        fixed += ok
        if not ok:
            print(
                f"BAD {row['question'][:44]!r}\n    real={row['real'][:56]!r}"
                f"\n    bot ={answer[:56]!r}"
            )
    print(f"\nrecheck: {fixed}/{len(rows)} previously-failed now exact")


if __name__ == "__main__":
    main()
