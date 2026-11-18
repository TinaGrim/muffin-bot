#!/usr/bin/env python3
"""Snapshot the bot's live state into data/backups and keep the last 14."""

from __future__ import annotations

import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BACKUPS = DATA / "backups"
KEEP = 14

FILES = [
    "history.json",
    "mood.json",
    "life.json",
    "reactions.json",
    "threads.json",
    "chats.json",
    "sent_replies.json",
    "feedback.jsonl",
    "corrections.json",
    "train.json",
    "offset.txt",
    "stickers.json",
    "stats.json",
    "vocab.json",
]


def main() -> None:
    BACKUPS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = BACKUPS / f"muffin-{stamp}.tar.gz"
    with tarfile.open(out, "w:gz") as archive:
        for name in FILES:
            path = DATA / name
            if path.exists():
                archive.add(path, arcname=name)
    old = sorted(BACKUPS.glob("muffin-*.tar.gz"))
    for path in old[:-KEEP]:
        path.unlink(missing_ok=True)
    print(out)


if __name__ == "__main__":
    main()
