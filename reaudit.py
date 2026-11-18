#!/usr/bin/env python3
"""Re-run a stored audit set against the current bot and rewrite its results."""

from __future__ import annotations

import argparse
import json
import logging
import re
import tempfile
import time
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
    parser.add_argument("--in", dest="src", default="data/audit_all.jsonl")
    parser.add_argument("--out", dest="out", default="data/audit_all.jsonl")
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    rows = [
        json.loads(line)
        for line in (ROOT / args.src).open(encoding="utf-8")
        if line.strip()
    ]
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

    today = date.today().isoformat()
    exact = 0
    for i, row in enumerate(rows, 1):
        chat_id = 100000 + i
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
        row["bot"] = answer
        row["exact"] = normalize(answer) == normalize(row["real"])
        tokens = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
        a, b = tokens(answer), tokens(row["real"])
        row["overlap"] = round(len(a & b) / len(a | b), 3) if a and b else 0.0
        exact += row["exact"]

    out = ROOT / args.out
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{exact}/{len(rows)} exact ({exact / len(rows):.1%}) -> {out}")


if __name__ == "__main__":
    main()
