#!/usr/bin/env python3
"""Rate real-conversation Q/A pairs 1-5.

Scorer order:
1. TypeSafe Jev (`TYPESAFE_API_KEY`) via the same /v1/systemone choice API
   used by JEV Ultrafast. This is the actual Jev rating path.
2. OpenAI-compatible text judge fallback (OpenCode Go by default), clearly
   marked in the output so the two are never confused.

Outputs:
- data/jev_ratings.jsonl  (all scored pairs)
- data/jev_train.jsonl    (score >= 4, SFT messages with real context)
- data/JEV_RATINGS.md     (summary + examples)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

RUBRIC = """You are a strict judge of a real Khmerlish chat between Bong and Muffin.
Decide how well the ANSWER replies to the QUESTION, using the CONTEXT.

Score 1-5:
1 = unrelated, ignores the question, or nonsense
2 = related topic but does not answer
3 = partial, evasive, or needs the question repeated
4 = answers the question correctly, in Muffin's voice
5 = direct, correct, natural, and in character

Do not reward Khmerlish alone. Judge meaning and relevance.
Return only JSON: {"score": 1-5, "reason": "short reason"}"""

TYPE_CRITERIA = {
    "1": "The answer is unrelated, ignores the question, or is nonsense.",
    "2": "The answer is on a related topic but does not answer the question.",
    "3": "The answer is partial or evasive and needs the question repeated.",
    "4": "The answer correctly replies to the question in Muffin's voice.",
    "5": "The answer directly and naturally replies to the question, in character.",
}


def context_text(pair: dict, limit: int = 4) -> str:
    context = pair.get("context") or []
    parts = []
    for message in context[-limit:]:
        role = "Bong" if message.get("role") == "user" else "Muffin"
        parts.append(f"{role}: {message.get('content', '')}")
    return "\n".join(parts)


def rate_typesafe(pair: dict, key: str, model: str) -> dict:
    body = {
        "model": model,
        "state": {
            "page": {"url": "local://qna", "title": "Q/A judge", "text": ""},
            "elements": [],
            "recent_actions": [],
        },
        "questions": {
            "rating": {
                "type": "choice",
                "criteria": TYPE_CRITERIA,
                "instructions": {
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "context": context_text(pair),
                    "rules": RUBRIC,
                },
            }
        },
    }
    response = httpx.post(
        "https://api.typesafe.ai/v1/systemone",
        headers={"Authorization": f"Bearer {key}"},
        json=body,
        timeout=45,
    )
    response.raise_for_status()
    answer = response.json()["answers"]["rating"]
    return {
        "score": int(answer["choice"]),
        "reason": "TypeSafe Jev choice",
        "scorer": "typesafe-jev",
        "confidence": answer.get("confidence"),
        "probabilities": answer.get("probabilities"),
    }


def rate_text(pair: dict, base: str, key: str, model: str) -> dict:
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": pair["question"],
                        "answer": pair["answer"],
                        "context": context_text(pair),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    last = None
    for attempt in range(3):
        try:
            response = httpx.post(
                base.rstrip("/") + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "x-opencode-session": "muffin-jev-rate",
                },
                json=payload,
                timeout=45,
            )
            if response.status_code in {429, 503, 529} and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            if response.is_error:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            content = response.json()["choices"][0]["message"]["content"] or ""
            try:
                parsed = json.loads(content)
            except ValueError:
                match = re.search(r'"score"\s*:\s*([1-5])', content)
                parsed = {"score": int(match.group(1))} if match else {}
            score = int(parsed.get("score", 0))
            if 1 <= score <= 5:
                return {
                    "score": score,
                    "reason": str(parsed.get("reason", ""))[:200],
                    "scorer": f"text-fallback:{model}",
                }
            last = f"invalid score: {content[:120]}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0 * (attempt + 1))
    return {"score": 0, "reason": last or "failed", "scorer": f"text-fallback:{model}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(DATA / "jev_pairs.jsonl"))
    parser.add_argument("--out", default=str(DATA / "jev_ratings.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--min-score", type=int, default=4)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    env = dict(os.environ)
    pairs = [
        json.loads(line)
        for line in Path(args.pairs).open(encoding="utf-8")
        if line.strip()
    ]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("no pairs")

    typesafe_key = env.get("TYPESAFE_API_KEY", "").strip()
    if typesafe_key:
        model = env.get("TYPESAFE_MODEL", "jev-latest")
        scorer = lambda pair: rate_typesafe(pair, typesafe_key, model)  # noqa: E731
        print(f"scorer: TypeSafe Jev ({model})")
    else:
        base = env.get("JUDGE_BASE_URL", env.get("LLM_BASE_URL", "https://opencode.ai/inference/go/v1"))
        key = env.get("JUDGE_API_KEY", env.get("LLM_API_KEY", "")).strip()
        model = env.get("JUDGE_MODEL", "deepseek-v4.1-flash")
        if not key:
            raise SystemExit("no TYPESAFE_API_KEY and no LLM_API_KEY for the text fallback")
        scorer = lambda pair: rate_text(pair, base, key, model)  # noqa: E731
        print(f"scorer: text fallback ({model} at {base}); set TYPESAFE_API_KEY for Jev")

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(scorer, pair): pair for pair in pairs}
        for index, (future, pair) in enumerate(futures.items(), 1):
            outcome = future.result()
            results.append({**pair, **outcome})
            if index % 10 == 0 or index == len(pairs):
                print(f"  {index}/{len(pairs)}")

    with Path(args.out).open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    persona = (ROOT / "persona.md").read_text(encoding="utf-8").strip()
    train = []
    for row in results:
        if row["score"] < args.min_score:
            continue
        messages = [{"role": "system", "content": persona}]
        messages += row.get("context") or []
        messages += [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        train.append({"messages": messages, "score": row["score"], "source_id": row.get("source_id")})
    (DATA / "jev_train.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in train) + ("\n" if train else ""),
        encoding="utf-8",
    )

    counts = Counter(row["score"] for row in results)
    mean = sum(row["score"] for row in results) / len(results)
    scorer_name = results[0]["scorer"] if results else "?"
    lines = [
        "# JEV pair ratings",
        "",
        f"- pairs: **{len(results)}**",
        f"- scorer: **{scorer_name}**",
        f"- mean score: **{mean:.2f}/5**",
        f"- score >= {args.min_score}: **{len(train)}**",
        "",
        "| score | count |",
        "|---:|---:|",
    ]
    for score in range(5, 0, -1):
        lines.append(f"| {score} | {counts.get(score, 0)} |")
    lines.append("")
    lines += [
        "## Examples",
        "",
        "| id | Q | A | score | reason |",
        "|---|---|---|---:|---|",
    ]
    for row in results[:30]:
        q = row["question"].replace("|", "\\|")[:60]
        a = row["answer"].replace("|", "\\|")[:60]
        reason = str(row.get("reason", "")).replace("|", "\\|")[:50]
        lines.append(f"| {row['id']} | {q} | {a} | {row['score']} | {reason} |")
    (DATA / "JEV_RATINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {args.out}, data/jev_train.jsonl and data/JEV_RATINGS.md")
    print(f"mean {mean:.2f}, >= {args.min_score}: {len(train)}/{len(results)}")


if __name__ == "__main__":
    main()
