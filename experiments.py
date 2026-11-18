#!/usr/bin/env python3
"""Prompt experiments for Khmerlish understanding (not part of the bot)."""

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import CachedRetriever  # noqa: E402

PERSONA = (ROOT / "persona.md").read_text(encoding="utf-8").strip()
R = CachedRetriever(ROOT / "data/pairs.jsonl", ROOT / "data" / "retrieval.pkl").get()
HISTORY = [
    {"role": "user", "content": "hello oun"},
    {"role": "assistant", "content": "Bart"},
]


def examples(text, k=3, minscore=0.55):
    out = []
    for ex in R.search(text, k=12):
        if ex["score"] < minscore:
            continue
        if "http" in ex["reply"].lower() or len(ex["reply"]) > 160:
            continue
        out.append(ex)
        if len(out) == k:
            break
    return out


def ask(messages):
    r = httpx.post(
        "http://localhost:11434/v1/chat/completions",
        json={
            "model": "qwen2.5:7b-instruct",
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 120,
        },
        timeout=120,
    )
    return r.json()["choices"][0]["message"]["content"].strip()


def run(label, text, system):
    messages = [{"role": "system", "content": system}] + HISTORY + [
        {"role": "user", "content": text}
    ]
    print(f"\n[{label}] Bong: {text}\nMuffin: {ask(messages)}")


def main():
    text = "tnaii ng ban ey nham bay nv"
    ex = examples(text)
    exblock = "\n".join(f"Bong: {e['user']}\nMuffin: {e['reply']}" for e in ex)
    base = f"{PERSONA}\n\nRight now it is Monday, 17 November 2026, 16:50.\nThis is an ongoing conversation."

    run("persona only", text, base)
    run(
        "examples",
        text,
        base
        + "\n\nSimilar real exchanges. They are references for slang and tone, not replies to copy:\n"
        + exblock,
    )
    run(
        "translation step",
        text,
        base
        + "\n\nFirst silently translate Bong's Khmerlish into English, work out what he is asking, then answer as Muffin. Output only Muffin's reply.",
    )
    run(
        "gloss + examples",
        text,
        base
        + "\n\nKhmerlish glossary: tnaii/tngaii=today, nham bay=eat, ban=past tense, ey=what, ng=what/which, ort=no, tv=go, jg=want, vinh=right now, nv=here/now, pong tvr ey=what are you doing.\n"
        + "Bong's message means: What did you eat today?\n"
        + "Similar exchanges:\n"
        + exblock,
    )

    text2 = "pong tvr ey vinh"
    ex2 = examples(text2)
    exblock2 = "\n".join(f"Bong: {e['user']}\nMuffin: {e['reply']}" for e in ex2)
    run(
        "second: examples",
        text2,
        base
        + "\n\nSimilar real exchanges (references only):\n"
        + exblock2
        + "\nAnswer the latest message in context.",
    )


if __name__ == "__main__":
    main()
