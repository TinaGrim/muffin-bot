#!/usr/bin/env python3
"""Run a fixed set of prompts through the real backend to tune the persona."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from bot import make_muffin  # noqa: E402

PROMPTS = [
    "hello",
    "good morning bong",
    "what are you doing",
    "I miss you",
    "do you love me",
    "ort nk morning ban ort?",
    "I'm sad today",
    "tv na?",
    "can you help me with my homework",
    "haha you are so cute",
]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="override LLM_MODEL for this run")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    env = dict(os.environ)
    if args.model:
        env["LLM_MODEL"] = args.model
    muffin = make_muffin(env)
    for prompt in PROMPTS:
        muffin.reset(4096)
        try:
            parts = muffin.reply(4096, prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"Bong:   {prompt}\n!       {exc}\n")
            continue
        print(f"Bong:   {prompt}")
        for part in parts:
            print(f"Muffin: {part}")
        print()


if __name__ == "__main__":
    main()
