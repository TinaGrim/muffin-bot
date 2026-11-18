#!/usr/bin/env python3
"""Audit the bot against 100 random real questions from the archive.

Each question is asked with no prior context. Her scheduled activity for the
real timestamp is used, so location-dependent answers are judged fairly.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from history import HistoryStore  # noqa: E402
from life import Activity, LifeStore  # noqa: E402
from mood import Mood, MoodStore  # noqa: E402
from muffin import Chat, Muffin, load_config  # noqa: E402
from reactions import ReactionState, ReactionStore  # noqa: E402
from retrieval import CachedRetriever, normalize  # noqa: E402
from threads import ThreadState, ThreadStore  # noqa: E402

BAD = re.compile(
    r"\b(sex|fuck|fucking|bitch|slut|porn|nude|naked|kill|suicide|rape|"
    r"dick|cock|pussy|boobs|horny)\b",
    re.I,
)

# Rough moods so the audit spans the range the bot actually lives in.
MOOD_ENERGY = {
    "neutral": 60,
    "playful": 85,
    "warm": 75,
    "tired": 30,
    "soft": 70,
    "annoyed": 40,
}
MOOD_AFFECTION = {
    "neutral": 60,
    "playful": 70,
    "warm": 90,
    "tired": 55,
    "soft": 85,
    "annoyed": 35,
}
MOOD_PATIENCE = {
    "neutral": 70,
    "playful": 65,
    "warm": 80,
    "tired": 35,
    "soft": 75,
    "annoyed": 20,
}


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def overlap(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def clip(text: str, limit: int) -> str:
    text = text.replace("\n", " / ").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    parser.add_argument(
        "--statements",
        action="store_true",
        help="also sample lines without a question mark (much larger pool)",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="path to a previous audit jsonl; its questions are not sampled again",
    )
    parser.add_argument("--out", default="data/question_audit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    raw = [
        json.loads(line)
        for line in (ROOT / "data" / "pairs.jsonl").open(encoding="utf-8")
    ]
    # One pair per unique question, no explicit/NSFW lines.
    by_question: dict[str, dict] = {}
    for pair in raw:
        user = pair["user"].strip()
        if not args.statements and "?" not in user:
            continue
        if not (3 <= len(user) <= 300 and 1 <= len(pair["reply"]) <= 500):
            continue
        if BAD.search(user) or BAD.search(pair["reply"]):
            continue
        key = normalize(user)
        if key and key not in by_question:
            by_question[key] = pair

    pool = list(by_question.values())
    if args.exclude and Path(args.exclude).exists():
        seen = {
            normalize(row["question"])
            for row in (
                json.loads(line)
                for line in Path(args.exclude).open(encoding="utf-8")
            )
        }
        pool = [pair for pair in pool if normalize(pair["user"]) not in seen]
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))
    if not sample:
        print("no unseen questions left")
        return

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
    tmp = Path(tempfile.mkdtemp())
    mood = MoodStore(tmp / "mood.json")
    life = LifeStore(tmp / "life.json")
    history = HistoryStore(tmp / "history.json")
    reactions = ReactionStore(tmp / "reactions.json")
    threads = ThreadStore(tmp / "threads.json")
    muffin = Muffin(config, retriever, mood, life, history, reactions, threads)

    today = date.today().isoformat()
    now = time.time()
    results = []
    exact = 0
    close = 0

    for i, pair in enumerate(sample, 1):
        muffin.chats[i] = Chat(history=[])
        # Sample several moods the way a real day swings, so the audit does not
        # only measure the neutral persona.
        mood_cycle = ("neutral", "playful", "warm", "tired", "soft", "annoyed")
        mood_name = mood_cycle[(i - 1) % len(mood_cycle)]
        mood.chats[str(i)] = Mood(
            energy=MOOD_ENERGY[mood_name],
            affection=MOOD_AFFECTION[mood_name],
            patience=MOOD_PATIENCE[mood_name],
        )
        real_dt = datetime.fromisoformat(pair["date"])
        key = life._scheduled(real_dt).key
        life.chats[str(i)] = Activity(key=key, started=now, ends=now + 3600)
        reactions.chats[str(i)] = ReactionState(
            last_user_date=today,
            morning_greeted=today,
            last_user_ts=now,
        )
        threads.chats[str(i)] = ThreadState()

        generated = " | ".join(muffin.reply(i, pair["user"]))
        ok = normalize(generated) == normalize(pair["reply"])
        near = overlap(generated, pair["reply"]) >= 0.6
        exact += ok
        close += near

        diagnosis = ""
        if not ok:
            if generated in {"Huh", "Mic", "Ort", "Nham bay nv"}:
                diagnosis = "deflection"
            elif muffin.life.conflicts(life.chats[str(i)], pair["reply"]):
                diagnosis = "real answer blocked by activity"
            elif near:
                diagnosis = "same meaning, different words"
            else:
                diagnosis = "different reply"

        results.append(
            {
                "i": i,
                "date": pair["date"],
                "life": key,
                "question": pair["user"],
                "real": pair["reply"],
                "bot": generated,
                "exact": ok,
                "overlap": round(overlap(generated, pair["reply"]), 3),
                "diagnosis": diagnosis,
            }
        )
        print(
            f"{i:3d}/{len(sample)} {'OK ' if ok else 'BAD'} "
            f"{clip(pair['user'], 48)!r}"
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# 100 real questions vs the bot",
        "",
        f"seed `{args.seed}` · no context · scheduled activity from the real"
        " timestamp · moods rotate through six states",
        "",
        "| # | Bong asked | Real Muffin | Bot | ✓ |",
        "|---:|---|---|---|:---:|",
    ]
    for row in results:
        mark = "✓" if row["exact"] else ("~" if row["overlap"] >= 0.6 else "✗")
        lines.append(
            f"| {row['i']} | {clip(row['question'], 55)} | "
            f"{clip(row['real'], 55)} | {clip(row['bot'], 55)} | {mark} |"
        )
    lines += [
        "",
        f"**Exact {exact}/{len(sample)} = {exact / len(sample):.0%}** · "
        f"same-meaning {close}/{len(sample)} = {close / len(sample):.0%}",
        "",
        "## Mismatches",
        "",
    ]
    for row in results:
        if row["exact"]:
            continue
        lines += [
            f"### {row['i']}. {clip(row['question'], 90)}",
            "",
            f"- real: {clip(row['real'], 120)}",
            f"- bot: {clip(row['bot'], 120)}",
            f"- {row['diagnosis']} · overlap {row['overlap']:.2f} · life `{row['life']}`",
            "",
        ]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nexact {exact}/{len(sample)} = {exact / len(sample):.1%}")
    print(f"same-meaning {close}/{len(sample)} = {close / len(sample):.1%}")
    print(f"wrote {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
