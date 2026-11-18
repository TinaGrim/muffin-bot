#!/usr/bin/env python3
"""Held-out evaluation: compare the clone's replies with real Muffin replies.

Usage:
  python evaluate.py --build-dev --n 80
  python evaluate.py --temp 0.8 --examples always --rules
  python evaluate.py --temp 0.6 --examples first --no-rules

Reports exact match, character similarity, token F1, 3-gram cosine and length
similarity, and writes per-sample results for inspection.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from life import Activity, LifeStore  # noqa: E402
from history import HistoryStore  # noqa: E402
from reactions import ReactionStore  # noqa: E402
from threads import ThreadStore  # noqa: E402
from mood import Mood, MoodStore  # noqa: E402
from muffin import Chat, Muffin, load_config  # noqa: E402
from retrieval import Retriever, load_pairs, ngrams  # noqa: E402
from rules import rule_reply  # noqa: E402

DEV = ROOT / "data" / "dev.jsonl"
VAL = ROOT / "data" / "sft_val.jsonl"


def build_dev(n: int = 80, seed: int = 42) -> None:
    rows = [json.loads(line) for line in VAL.open(encoding="utf-8")]
    rows = [
        r
        for r in rows
        if 1 <= len(r["messages"][-1]["content"]) <= 160
        and 1 <= len(r["messages"][-2]["content"]) <= 160
    ]
    random.Random(seed).shuffle(rows)
    chosen = rows[:n]
    chosen.sort(key=lambda r: r.get("date", ""))
    with DEV.open("w", encoding="utf-8") as f:
        for row in chosen:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(chosen)} dev samples to {DEV}")


def normalize(text: str) -> str:
    text = text.lower().replace("\u200b", "")
    text = re.sub(r"[^\w\u1780-\u17ff]+", " ", text)
    return " ".join(text.split())


def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def token_f1(a: str, b: str) -> float:
    ta = re.findall(r"[\w\u1780-\u17ff]+", a.lower())
    tb = re.findall(r"[\w\u1780-\u17ff]+", b.lower())
    if not ta or not tb:
        return 0.0
    common = Counter(ta) & Counter(tb)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def ngram_cosine(a: str, b: str) -> float:
    ca, cb = Counter(ngrams(a)), Counter(ngrams(b))
    if not ca or not cb:
        return 0.0
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def length_score(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    return min(la, lb) / max(la, lb) if max(la, lb) else 1.0


KHMERLISH = {
    "bong", "oun", "pov", "ort", "ot", "tv", "jg", "hz", "nham", "bay",
    "slnh", "teh", "nv", "vinh", "ban", "ng", "ey", "tvr", "nas", "men",
    "kit", "jam", "oy", "kor", "dek", "min", "mean", "ke", "mk", "dg",
    "teah", "hov", "jol", "srok", "kmean", "chngay", "merl", "kit", "tos",
}
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def style_metrics(text: str) -> dict:
    words = re.findall(r"[\w\u1780-\u17ff]+", text.lower())
    letters = [c for c in text if c.isalpha()]
    return {
        "khmer": (sum(1 for c in letters if "\u1780" <= c <= "\u17ff") / len(letters))
        if letters
        else 0.0,
        "khmerlish": (sum(1 for w in words if w in KHMERLISH) / len(words))
        if words
        else 0.0,
        "emoji": len(EMOJI.findall(text)),
        "question": 1 if "?" in text else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dev", action="store_true")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N dev rows")
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--examples", choices=["always", "first", "never"], default="always")
    parser.add_argument("--rules", action="store_true", help="keep intent rules on")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--adapter-mode", action="store_true", help="minimal prompt for the LoRA adapter")
    parser.add_argument(
        "--retrieval",
        choices=["train", "all"],
        default="train",
        help="train excludes the val pairs so exact matches are not leaked",
    )
    parser.add_argument("--out", default="data/eval_last.jsonl")
    args = parser.parse_args()

    if args.build_dev or not DEV.exists():
        build_dev(args.n)

    env = {
        "LLM_BASE_URL": args.base_url,
        "LLM_API_KEY": args.api_key,
        "LLM_MODEL": args.model or "qwen2.5:7b-instruct",
        "LLM_TEMPERATURE_OVERRIDE": str(args.temp),
        "EXAMPLES_MODE": args.examples,
        "USE_RULES": "true" if args.rules else "false",
        "RETRIEVE_K": str(args.k),
        "DIRECT_REPLY_THRESHOLD": "0",
        "ADAPTER_MODE": "true" if args.adapter_mode else "false",
    }
    config = load_config(env, ROOT / "persona.md")
    val_ids = {json.loads(line)["id"] for line in VAL.open(encoding="utf-8")}
    pairs = load_pairs(ROOT / "data" / "pairs.jsonl")
    if args.retrieval == "train":
        pairs = [p for p in pairs if p["id"] not in val_ids]
    print(f"retrieval: {len(pairs):,} pairs ({args.retrieval})")
    retriever = Retriever(pairs)
    store = MoodStore(Path(tempfile.mkdtemp()) / "mood.json")
    life = LifeStore(Path(tempfile.mkdtemp()) / "life.json")
    history = HistoryStore(Path(tempfile.mkdtemp()) / "history.json")
    reactions = ReactionStore(Path(tempfile.mkdtemp()) / "reactions.json")
    threads = ThreadStore(Path(tempfile.mkdtemp()) / "threads.json")
    muffin = Muffin(config, retriever, store, life, history, reactions, threads)

    rows = [json.loads(line) for line in DEV.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    started = time.time()

    def run_one(item: tuple[int, dict]) -> dict:
        i, row = item
        messages = row["messages"]
        context = messages[1:-2]
        user = messages[-2]["content"]
        reference = messages[-1]["content"]
        rule_hit = bool(rule_reply(user)) if args.rules else False

        chat_id = i
        muffin.chats[chat_id] = Chat(history=context[-20:])
        store.chats[str(chat_id)] = Mood(energy=60, affection=55, patience=60)

        generated = ""
        try:
            parts = muffin.reply(chat_id, user)
            generated = "\n".join(parts)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! eval {i} failed: {exc}")

        result = {
            "i": i,
            "date": row.get("date"),
            "user": user,
            "reference": reference,
            "generated": generated,
            "exact": normalize(generated) == normalize(reference),
            "char": round(char_similarity(generated, reference), 4),
            "f1": round(token_f1(generated, reference), 4),
            "ngram": round(ngram_cosine(generated, reference), 4),
            "len": round(length_score(generated, reference), 4),
            "rule": rule_hit,
        }
        result["score"] = round(
            0.45 * result["ngram"] + 0.25 * result["char"] + 0.2 * result["f1"] + 0.1 * result["len"],
            4,
        )
        gs, rs = style_metrics(generated), style_metrics(reference)
        result.update(
            {
                "gen_khmer": round(gs["khmer"], 3),
                "ref_khmer": round(rs["khmer"], 3),
                "gen_khmerlish": round(gs["khmerlish"], 3),
                "ref_khmerlish": round(rs["khmerlish"], 3),
                "gen_emoji": gs["emoji"],
                "ref_emoji": rs["emoji"],
                "gen_question": gs["question"],
                "ref_question": rs["question"],
            }
        )
        return result

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, (i, row)): i for i, row in enumerate(rows, 1)}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['i']:3d}/{len(rows)} exact={int(result['exact'])} "
                f"char={result['char']:.2f} score={result['score']:.2f} "
                f"{result['generated'][:50]!r} != {result['reference'][:50]!r}"
            )
    results.sort(key=lambda r: r["i"])

    out = ROOT / args.out
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(results)
    summary = {
        "model": config.model,
        "temp": args.temp,
        "examples": args.examples,
        "rules": args.rules,
        "k": args.k,
        "retrieval": args.retrieval,
        "workers": args.workers,
        "n": n,
        "exact": round(sum(r["exact"] for r in results) / n, 3),
        "char": round(sum(r["char"] for r in results) / n, 3),
        "f1": round(sum(r["f1"] for r in results) / n, 3),
        "ngram": round(sum(r["ngram"] for r in results) / n, 3),
        "len": round(sum(r["len"] for r in results) / n, 3),
        "score": round(sum(r["score"] for r in results) / n, 3),
        "rule_hits": sum(1 for r in results if r["rule"]),
        "gen_len_median": statistics.median(len(r["generated"]) for r in results),
        "ref_len_median": statistics.median(len(r["reference"]) for r in results),
        "gen_khmer": round(sum(r["gen_khmer"] for r in results) / n, 3),
        "ref_khmer": round(sum(r["ref_khmer"] for r in results) / n, 3),
        "gen_khmerlish": round(sum(r["gen_khmerlish"] for r in results) / n, 3),
        "ref_khmerlish": round(sum(r["ref_khmerlish"] for r in results) / n, 3),
        "gen_emoji_rate": round(sum(r["gen_emoji"] for r in results) / n, 3),
        "ref_emoji_rate": round(sum(r["ref_emoji"] for r in results) / n, 3),
        "gen_question_rate": round(sum(r["gen_question"] for r in results) / n, 3),
        "ref_question_rate": round(sum(r["ref_question"] for r in results) / n, 3),
        "empty": sum(1 for r in results if not r["generated"]),
        "seconds": round(time.time() - started, 1),
        "out": str(out),
    }
    print("\n" + json.dumps(summary, indent=2))

    print("\nWorst:")
    for r in sorted(results, key=lambda x: x["score"])[:5]:
        print(f"  B: {r['user'][:70]}")
        print(f"  R: {r['reference'][:70]}")
        print(f"  G: {r['generated'][:70]}  (score {r['score']})")
    print("\nBest:")
    for r in sorted(results, key=lambda x: x["score"], reverse=True)[:5]:
        print(f"  B: {r['user'][:70]}")
        print(f"  R: {r['reference'][:70]}")
        print(f"  G: {r['generated'][:70]}  (score {r['score']})")


if __name__ == "__main__":
    main()
