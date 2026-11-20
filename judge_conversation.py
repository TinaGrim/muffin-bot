#!/usr/bin/env python3
"""LLM judge for a real conversation stored in data/history.json."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from history import HistoryStore  # noqa: E402
from life import LifeStore  # noqa: E402
from mood import MoodStore  # noqa: E402
from muffin import Muffin, load_config  # noqa: E402
from reactions import ReactionStore  # noqa: E402
from retrieval import CachedRetriever  # noqa: E402
from threads import ThreadStore  # noqa: E402

PROMPT = """You are an expert reviewer of a Telegram chatbot that imitates
Muffin, a young Cambodian woman texting in Khmerlish (Romanized Khmer mixed
with English).

Judge the real conversation below. Be strict and concrete.

Score 1-5 for:
- persona: sounds like blunt, playful Muffin, not an assistant
- context: continues the conversation, remembers what was just said
- style: Khmerlish, short, no Khmer script, no commas, no English paragraphs
- relevance: answers what Bong actually asked; no non-sequiturs

Also give overall 1-10.

Return only JSON:
{"overall": 6, "persona": 4, "context": 3, "style": 4, "relevance": 2,
 "summary": "one or two sentences",
 "turns": [{"n": 1, "score": 1, "issue": "..."}],
 "top_fixes": ["...", "..."]}"""


def build_muffin(env: dict[str, str]) -> Muffin:
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
    tmp = Path("/private/var/folders/h_/twq_l0tj0413yy24tz0pxz840000gn/T")
    return Muffin(
        config,
        retriever,
        MoodStore(tmp / "judge-mood.json"),
        LifeStore(tmp / "judge-life.json"),
        HistoryStore(tmp / "judge-history.json"),
        ReactionStore(tmp / "judge-reactions.json"),
        ThreadStore(tmp / "judge-threads.json"),
    )


def transcript(messages: list[dict]) -> str:
    lines = []
    number = 0
    for message in messages:
        if message["role"] == "user":
            number += 1
            lines.append(f"{number}. Bong: {message['content']}")
        else:
            lines.append(f"   Muffin: {message['content']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", type=int, default=1725557743)
    parser.add_argument("--messages", type=int, default=20)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    env = dict(os.environ)
    history = HistoryStore(ROOT / "data" / "history.json")
    messages = history.get(args.chat)[-args.messages :]
    if not messages:
        raise SystemExit("no messages for that chat")

    muffin = build_muffin(env)
    judge_messages = [
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": "Conversation:\n\n" + transcript(messages),
        },
    ]
    raw = ""
    try:
        raw = muffin.llm.chat(
            judge_messages,
            temperature=0.0,
            max_tokens=900,
            session_id="muffin-judge",
            timeout=45,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"primary judge failed: {exc}", file=sys.stderr)
    if not raw:
        raw = muffin._fallback_chat(
            judge_messages,
            temperature=0.0,
            max_tokens=900,
            session_id="muffin-judge",
        )

    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise SystemExit(f"judge did not return JSON:\n{raw[:1000]}")
    try:
        data = json.loads(match.group(0))
    except ValueError as exc:
        raise SystemExit(f"judge JSON invalid: {exc}\n{raw[:1000]}")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Conversation judge",
        "",
        f"Chat `{args.chat}`, last {len(messages)} messages, {stamp}",
        "",
        f"- overall: **{data.get('overall')}/10**",
        f"- persona: **{data.get('persona')}/5**",
        f"- context: **{data.get('context')}/5**",
        f"- style: **{data.get('style')}/5**",
        f"- relevance: **{data.get('relevance')}/5**",
        "",
        f"{data.get('summary', '')}",
        "",
    ]
    issues = data.get("turns") or []
    if issues:
        lines += ["## Turn issues", "", "| turn | score | issue |", "|---:|---:|---|"]
        for item in issues:
            lines.append(
                f"| {item.get('n')} | {item.get('score')} | {item.get('issue', '')} |"
            )
        lines.append("")
    fixes = data.get("top_fixes") or []
    if fixes:
        lines += ["## Top fixes", ""] + [f"- {fix}" for fix in fixes] + [""]

    report = ROOT / "data" / "CONVERSATION_JUDGE.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "data" / "CONVERSATION_JUDGE.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("\n".join(lines[:12]))
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
