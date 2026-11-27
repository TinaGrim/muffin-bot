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
from retrieval import CachedRetriever, normalize  # noqa: E402
from threads import ThreadStore  # noqa: E402

PROMPT = """You are an expert reviewer of a Telegram chatbot that imitates
Muffin, a young Cambodian woman texting in Khmerlish (Romanized Khmer mixed
with English).

The bot's rules, which you must use when scoring:
- Short Khmerlish, Latin letters only, no Khmer script, no commas, no English
  paragraphs.
- She is blunt and bratty. Anger is intentional. If Bong ignored her texts,
  she stays cold until he apologises or comforts her. Do not mark a cold reply
  as a persona failure just because it is cold.
- A line marked [note: ...] was rewritten because Bong asked for it. Judge it
  against that note, not as a bot invention.
- [proactive], [ritual] and [follow-up] messages were sent by Muffin first.
  They are not replies to the next message. Do not score them as answers.
- Food is only allowed at breakfast 6-10, lunch 11-14 and dinner 17-20.
- She must not contradict what she just said, and must not repeat the same
  question she asked an hour ago.

Judge the real conversation below. Be strict and concrete.

Messages may have a timestamp and `--- Xm gap ---` markers. Messages separated
by a gap are different sessions: do not score a message sent hours earlier as
if it were the reply to the next one.

Score 1-5 for:
- persona: sounds like blunt, playful Muffin, not an assistant
- context: continues the conversation, remembers what was just said
- style: Khmerlish, short, no Khmer script, no commas, no English paragraphs
- relevance: answers what Bong actually asked; no non-sequiturs
- reach_out: the [proactive]/[ritual] messages are worth answering; not generic
  filler and not repeated

Also give overall 1-10.

Return only JSON:
{"overall": 6, "persona": 4, "context": 3, "style": 4, "relevance": 2,
 "reach_out": 3,
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
    tmp = Path(
        "/private/var/folders/h_/twq_l0tj0413yy24tz0pxz840000gn/T"
    )
    return Muffin(
        config,
        retriever,
        MoodStore(tmp / "judge-mood.json"),
        LifeStore(tmp / "judge-life.json"),
        HistoryStore(tmp / "judge-history.json"),
        ReactionStore(tmp / "judge-reactions.json"),
        ThreadStore(tmp / "judge-threads.json"),
    )


def load_notes() -> dict[str, str]:
    """Notes Bong attached to a corrected reply, keyed by the sent text."""
    path = ROOT / "data" / "sent_replies.json"
    if not path.exists():
        return {}
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    notes: dict[str, str] = {}
    for record in records.values():
        if not isinstance(record, dict):
            continue
        note = (record.get("note") or "").strip()
        corrected = (record.get("corrected") or "").strip()
        if note and corrected:
            notes[normalize(corrected)] = note
    return notes


def transcript(messages: list[dict], notes: dict[str, str] | None = None) -> str:
    notes = notes or {}
    lines = []
    number = 0
    previous = None
    previous_role = None
    for message in messages:
        stamp = message.get("ts")
        if stamp and previous and stamp - previous >= 45 * 60:
            minutes = (stamp - previous) / 60
            gap = f"{minutes:.0f}m" if minutes < 120 else f"{minutes / 60:.1f}h"
            lines.append(f"   --- {gap} gap ---")
        when = ""
        if stamp:
            when = datetime.fromtimestamp(stamp).strftime("%m-%d %H:%M ")
        if message["role"] == "user":
            number += 1
            lines.append(f"{number}. {when}Bong: {message['content']}")
            previous_role = "user"
        else:
            tags: list[str] = []
            kind = message.get("kind") or ""
            if kind and kind != "reply":
                tags.append(kind)
            elif not kind and previous_role == "assistant":
                tags.append("follow-up")
            note = message.get("note") or notes.get(normalize(message["content"]))
            if note:
                tags.append(f"note: {note}")
            tag = f" [{' / '.join(tags)}]" if tags else ""
            lines.append(f"   {when}Muffin{tag}: {message['content']}")
            previous_role = "assistant"
        previous = stamp or previous
    return "\n".join(lines)


def judge_once(
    muffin: Muffin, messages: list[dict], temperature: float
) -> dict:
    judge_messages = [
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": "Conversation:\n\n" + transcript(messages, load_notes()),
        },
    ]
    raw = ""
    try:
        raw = muffin.llm.chat(
            judge_messages,
            temperature=temperature,
            max_tokens=900,
            session_id="muffin-judge",
            timeout=45,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"primary judge failed: {exc}", file=sys.stderr)
    if not raw:
        raw = muffin._fallback_chat(
            judge_messages,
            temperature=temperature,
            max_tokens=900,
            session_id="muffin-judge",
        )

    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"judge did not return JSON:\n{raw[:1000]}")
    try:
        return json.loads(match.group(0))
    except ValueError as exc:
        raise ValueError(f"judge JSON invalid: {exc}\n{raw[:1000]}") from exc


def average_runs(runs: list[dict]) -> dict:
    """Average the scores, keep the median run for turns and summary."""
    keys = ("overall", "persona", "context", "style", "relevance", "reach_out")
    averaged = {
        key: round(sum(float(run.get(key, 0)) for run in runs) / len(runs), 1)
        for key in keys
    }
    ordered = sorted(runs, key=lambda run: float(run.get("overall", 0)))
    base = dict(ordered[len(ordered) // 2])
    base.update(averaged)
    fixes: list[str] = []
    for run in runs:
        for fix in run.get("top_fixes") or []:
            if fix not in fixes:
                fixes.append(fix)
    base["top_fixes"] = fixes[:8]
    base["runs"] = len(runs)
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", type=int, default=1725557743)
    parser.add_argument("--messages", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    env = dict(os.environ)
    history = HistoryStore(ROOT / "data" / "history.json")
    messages = history.get(args.chat)[-args.messages :]
    if not messages:
        raise SystemExit("no messages for that chat")

    muffin = build_muffin(env)
    runs: list[dict] = []
    for index in range(max(1, args.runs)):
        try:
            runs.append(judge_once(muffin, messages, args.temperature))
        except ValueError as exc:
            print(f"run {index + 1} failed: {exc}", file=sys.stderr)
    if not runs:
        raise SystemExit("judge failed every run")

    data = average_runs(runs)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Conversation judge",
        "",
        f"Chat `{args.chat}`, last {len(messages)} messages, {stamp}",
        f"Runs: **{len(runs)}**",
        "",
        f"- overall: **{data.get('overall')}/10**",
        f"- persona: **{data.get('persona')}/5**",
        f"- context: **{data.get('context')}/5**",
        f"- style: **{data.get('style')}/5**",
        f"- relevance: **{data.get('relevance')}/5**",
        f"- reach_out: **{data.get('reach_out')}/5**",
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
    payload = dict(data)
    payload["individual"] = runs
    (ROOT / "data" / "CONVERSATION_JUDGE.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("\n".join(lines[:14]))
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
