#!/usr/bin/env python3
"""Build short samples for 3B QLoRA on an 8 GB Mac.

The 3B model needs sequence length 192 to avoid swapping on 8 GB, so the long
persona prompt and long conversations are replaced by the short prompt in
`persona_3b.md`. Records longer than `--max-tokens` are dropped, not truncated,
so the masked assistant target is never empty (an empty target produced NaN
loss in testing).

Usage:
    python fine_tune/make_3b_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "data" / "sft_train.jsonl"))
    parser.add_argument("--valid", default=str(ROOT / "data" / "sft_val.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "data" / "mlx3_3b"))
    parser.add_argument("--system", default=str(ROOT / "persona_3b.md"))
    parser.add_argument("--tokenizer", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=180)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    system = Path(args.system).read_text(encoding="utf-8").strip()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def build(src: str, dst: Path) -> None:
        kept = 0
        dropped = 0
        with Path(src).open(encoding="utf-8") as handle, dst.open(
            "w", encoding="utf-8"
        ) as sink:
            for line in handle:
                if not line.strip():
                    continue
                messages = json.loads(line)["messages"]
                user = (messages[-2].get("content") or "").strip()
                reply = (messages[-1].get("content") or "").strip()
                if not user or not reply:
                    dropped += 1
                    continue
                record = {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": reply},
                    ]
                }
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False
                )
                tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                if tokens > args.max_tokens:
                    dropped += 1
                    continue
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
        print(f"{dst}: kept {kept}, dropped {dropped}")

    build(args.train, out / "train.jsonl")
    build(args.valid, out / "valid.jsonl")
    print(f"system tokens: {len(tokenizer(system)['input_ids'])}")


if __name__ == "__main__":
    main()
