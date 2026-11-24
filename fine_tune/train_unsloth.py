#!/usr/bin/env python3
"""QLoRA fine-tune of Muffin on a single consumer GPU (RTX 4060, 8-16 GB).

Runs on Linux / WSL2 with Unsloth. The data comes from
`fine_tune/prepare_data.py` and is already in chat `messages` format.

    python fine_tune/train_unsloth.py --model Qwen/Qwen2.5-7B-Instruct

This script only trains and saves the adapter. Use `--merge` to also write a
merged 16-bit model for GGUF export, and `--gguf` to quantize it.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train", default="fine_tune/data/train.jsonl")
    parser.add_argument("--valid", default="fine_tune/data/valid.jsonl")
    parser.add_argument("--out", default="fine_tune/out/muffin-lora")
    parser.add_argument(
        "--max-seq",
        type=int,
        default=1024,
        help="1024 is safe on 8 GB; use 2048 on a 16 GB card.",
    )
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--gguf", action="store_true")
    return parser.parse_args()


def build_trainer_config(kwargs: dict) -> SFTConfig:
    """trl renamed several SFTConfig fields; accept both names."""
    params = set(inspect.signature(SFTConfig.__init__).parameters)
    if "max_length" in params and "max_seq_length" not in params:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    if "eval_strategy" in params and "evaluation_strategy" not in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    allowed = {key: value for key, value in kwargs.items() if key in params}
    return SFTConfig(**allowed)


def main() -> None:
    args = parse_args()
    for path in (args.train, args.valid):
        if not Path(path).exists():
            raise SystemExit(f"missing {path}; run fine_tune/prepare_data.py first")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    dataset = load_dataset(
        "json", data_files={"train": args.train, "valid": args.valid}
    )

    def to_text(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for messages in batch["messages"]
            ]
        }

    dataset = dataset.map(to_text, batched=True, remove_columns=["messages"])

    config = build_trainer_config(
        {
            "output_dir": args.out,
            "dataset_text_field": "text",
            "max_seq_length": args.max_seq,
            "per_device_train_batch_size": args.batch,
            "gradient_accumulation_steps": args.grad_accum,
            "per_device_eval_batch_size": 1,
            "warmup_steps": 20,
            "num_train_epochs": args.epochs,
            "learning_rate": args.lr,
            "fp16": not is_bfloat16_supported(),
            "bf16": is_bfloat16_supported(),
            "logging_steps": 10,
            "optim": "paged_adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "seed": 3407,
            "report_to": "none",
            "save_strategy": "steps",
            "save_steps": args.save_steps,
            "save_total_limit": 2,
            "evaluation_strategy": "steps",
            "eval_steps": args.save_steps,
            "packing": False,
        }
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        args=config,
    )
    # Loss only on Muffin's reply, not on Bong's message or the persona.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"adapter saved to {args.out}")

    if args.merge:
        merged = f"{args.out}-merged"
        model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")
        print(f"merged 16-bit model saved to {merged}")

    if args.gguf:
        gguf = f"{args.out}-gguf"
        model.save_pretrained_gguf(gguf, tokenizer, quantization_method="q4_k_m")
        print(f"GGUF saved to {gguf}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
