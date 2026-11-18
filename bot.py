#!/usr/bin/env python3
"""Run Muffin as a Telegram bot, or chat with the same pipeline in the terminal."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from feedback import Corrections  # noqa: E402
from history import HistoryStore  # noqa: E402
from life import LifeStore  # noqa: E402
from mood import MoodStore  # noqa: E402
from muffin import Muffin, load_config  # noqa: E402
from reactions import ReactionStore  # noqa: E402
from retrieval import CachedRetriever  # noqa: E402
from telegram_app import allowed, run_telegram  # noqa: E402
from telegram_client import Telegram  # noqa: E402
from threads import ThreadStore  # noqa: E402
from worker import (  # noqa: E402
    ChatWorker,
    feedback_keyboard,
    get_worker,
    handle_callback,
)

ROOT = Path(__file__).parent
log = logging.getLogger("muffin")


def make_muffin(env: dict[str, str]) -> Muffin:
    config = load_config(env, ROOT / "persona.md")
    retriever = CachedRetriever(
        ROOT / "data" / "pairs.jsonl",
        ROOT / "data" / "retrieval.pkl",
        embeddings=env.get("USE_EMBEDDINGS", "true").lower()
        not in {"0", "false", "no"},
        embed_model=env.get("EMBED_MODEL", "bge-m3"),
        embed_base=env.get("EMBED_BASE_URL", "http://localhost:11434"),
        embed_path=ROOT / "data" / env.get("EMBED_INDEX", "embeddings_bge_m3.npz"),
    ).get()
    mood = MoodStore(ROOT / "data" / "mood.json")
    life = LifeStore(ROOT / "data" / "life.json")
    history = HistoryStore(ROOT / "data" / "history.json")
    reactions = ReactionStore(ROOT / "data" / "reactions.json")
    threads = ThreadStore(ROOT / "data" / "threads.json")
    corrections = Corrections(ROOT / "data" / "corrections.json")
    log.info(
        "loaded %d exchanges, model=%s base_url=%s",
        len(retriever.pairs),
        config.model,
        config.base_url,
    )
    return Muffin(
        config, retriever, mood, life, history, reactions, threads, corrections
    )


def run_cli(muffin: Muffin) -> None:
    print("Muffin clone. Ctrl+D to quit.\n")
    chat_id = 1
    while True:
        try:
            text = input("Bong> ").strip()
        except EOFError:
            print()
            return
        if not text:
            continue
        if text in {"/reset", "/quit", "/exit"}:
            muffin.reset(chat_id)
            print("Muffin> (memory cleared)")
            continue
        try:
            parts = muffin.reply(chat_id, text)
        except Exception as exc:  # noqa: BLE001
            print(f"! {exc}")
            continue
        for part in parts:
            print(f"Muffin> {part}")


def run_retrieve(muffin: Muffin, query: str, k: int = 8) -> None:
    for match in muffin.retriever.search(query, k=k):
        print(f"{match['score']:.3f}  {match['user']!r}\n       -> {match['reply']!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true", help="terminal chat")
    parser.add_argument("--retrieve", metavar="TEXT", help="show nearest real exchanges")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # httpx logs the full request URL at INFO, which contains the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    load_dotenv(ROOT / ".env", override=True)
    env = dict(os.environ)
    muffin = make_muffin(env)

    if args.retrieve:
        run_retrieve(muffin, args.retrieve)
    elif args.chat:
        run_cli(muffin)
    else:
        run_telegram(muffin, env)


if __name__ == "__main__":
    main()
