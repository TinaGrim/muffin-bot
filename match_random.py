#!/usr/bin/env python3
"""Ask the bot random real questions from the archive and diff the replies."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from life import Activity, LifeStore  # noqa: E402
from history import HistoryStore  # noqa: E402
from reactions import ReactionState, ReactionStore  # noqa: E402
from threads import ThreadStore  # noqa: E402
from mood import Mood, MoodStore  # noqa: E402
from muffin import Chat, Muffin, load_config  # noqa: E402
from retrieval import CachedRetriever, normalize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--out", default="data/match_random.jsonl")
    args = parser.parse_args()

    pairs = [
        json.loads(line)
        for line in (ROOT / "data" / "pairs.jsonl").open(encoding="utf-8")
    ]
    pairs = [
        p
        for p in pairs
        if 3 <= len(p["user"]) <= 100
        and 1 <= len(p["reply"]) <= 120
        and p.get("context")
    ]
    random.Random(args.seed).shuffle(pairs)
    sample = pairs[: args.n]

    config = load_config(
        {
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_API_KEY": "ollama",
            "LLM_MODEL": args.model,
            "DIRECT_REPLY_THRESHOLD": "0.9",
            "USE_RULES": "true",
            "EXAMPLES_MODE": "never",
            "ADAPTER_MODE": "false",
        },
        ROOT / "persona.md",
    )
    retriever = CachedRetriever(
        ROOT / "data" / "pairs.jsonl", ROOT / "data" / "retrieval.pkl"
    ).get()
    store = MoodStore(Path(tempfile.mkdtemp()) / "mood.json")
    life = LifeStore(Path(tempfile.mkdtemp()) / "life.json")
    history = HistoryStore(Path(tempfile.mkdtemp()) / "history.json")
    reactions = ReactionStore(Path(tempfile.mkdtemp()) / "reactions.json")
    threads = ThreadStore(Path(tempfile.mkdtemp()) / "threads.json")
    from datetime import date

    muffin = Muffin(config, retriever, store, life, history, reactions, threads)

    results = []
    exact = 0
    for i, pair in enumerate(sample, 1):
        history = []
        if not args.no_context:
            for turn in pair["context"]:
                history.append({"role": "user", "content": turn["user"]})
                history.append({"role": "assistant", "content": turn["reply"]})
        muffin.chats[i] = Chat(history=history)
        store.chats[str(i)] = Mood(energy=60, affection=60, patience=70)
        reactions.chats[str(i)] = ReactionState(
            last_user_date=date.today().isoformat(),
            morning_greeted=date.today().isoformat(),
            last_user_ts=time.time(),
        )
        life.chats[str(i)] = Activity(
            key=random.choice(["chill", "dinner", "sleep", "out_food", "study", "work"]),
            started=time.time(),
            ends=time.time() + 3600,
        )

        generated = "\n".join(muffin.reply(i, pair["user"]))
        ok = normalize(generated) == normalize(pair["reply"])
        exact += ok
        results.append(
            {
                "i": i,
                "user": pair["user"],
                "reference": pair["reply"],
                "generated": generated,
                "exact": ok,
            }
        )
        print(f"{i:3d}/{len(sample)} {'OK ' if ok else 'BAD'} {pair['user'][:55]!r}")
        if not ok:
            print(f"      real: {pair['reply'][:80]!r}")
            print(f"      bot : {generated[:80]!r}")

    out = ROOT / args.out
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nexact {exact}/{len(sample)} = {exact / len(sample):.1%}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
