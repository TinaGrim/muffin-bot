# Fine-tuning Muffin

Prompt-only tuning fixes the shape of Muffin's replies but not the language: a
general 7B model invents Khmerlish words. This is the reproducible LoRA run on
the real 26k exchanges.

## Why 0.5B

This machine has 8 GB of unified memory. The attempts and outcomes:

| Model | Result |
|---|---|
| Qwen2.5-1.5B, bf16, full-length samples | ~4 s/iter, > 1.5 h |
| Qwen2.5-1.5B, bf16 + `--grad-checkpoint` | NaN loss |
| Qwen2.5-1.5B, 4-bit QLoRA | stalled under swap |
| **Qwen2.5-0.5B, bf16, compact samples** | **~2 s/iter, fits** |

A 0.5B model is not as smart, but it can learn Muffin's short Khmerlish voice,
which is what the prompt baseline cannot do.

## Data

`data/mlx3/` is built from the SFT splits with a compact system prompt
(`persona_compact.md`) and the most recent exchange:

- `train.jsonl` — 26,336 samples.
- `valid.jsonl` — 200 held-out samples.

The full `data/sft_train.jsonl` keeps up to four context turns; that version is
used by the prompt baseline and by any future cloud run.

## Run

```bash
.venv/bin/mlx_lm.lora \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train \
  --data data/mlx3 \
  --batch-size 4 \
  --iters 2000 \
  --val-batches 5 \
  --learning-rate 2e-5 \
  --max-seq-length 320 \
  --mask-prompt \
  --adapter-path adapters/muffin-0.5b \
  --save-every 250 \
  --steps-per-eval 250 \
  --steps-per-report 50
```

- `--mask-prompt` trains only on Muffin's reply.
- Checkpoints are written every 250 iterations; resume with
  `--resume-adapter-file adapters/muffin-0.5b/adapters.safetensors`.

## Serve

```bash
./serve_mlx.sh
```

Then point the bot at it:

```dotenv
LLM_BASE_URL=http://127.0.0.1:8080/v1
LLM_API_KEY=none
LLM_MODEL=qwen2.5-0.5b-muffin
ADAPTER_MODE=true
```

Restart `./run.sh`. `ADAPTER_MODE=true` uses the compact system prompt the
adapter was trained with and skips retrieved examples.

## Evaluate

```bash
.venv/bin/python evaluate.py --adapter-mode --rules --limit 80 \
  --base-url http://127.0.0.1:8080/v1 --api-key none \
  --model qwen2.5-0.5b-muffin --out data/eval_adapter.jsonl
```

The held-out exact-match rate is not a useful target: even returning the
nearest real reply from the training set scores 0% exact on these conversations,
because Muffin's wording is highly context-specific. The useful signals are
length/style match and whether the reply answers the message.
