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

## 3B QLoRA on the Mac (M1, 8 GB)

This works, but only with short samples. Long sequences (512+) swap and stall
on 8 GB. The tested recipe:

```bash
.venv/bin/python fine_tune/make_3b_data.py   # data/mlx3_3b, seq-safe samples
./fine_tune/train_3b_mac.sh                  # 3000 iters, seq 192, batch 1
```

Settings: 4-bit base, LoRA rank 8 on 8 layers, batch 1, seq 192, lr 2e-5,
gradient checkpointing, prompt masked. 26,012 train / 1,334 valid samples.

Measured on this Mac:

| Run | Result |
|---|---|
| 10 iters, seq 512, batch 1 | 2.2 s, loss 12.4 -> 5.9 |
| full run, seq 512 | swaps after ~100 iters, ~3 s/iter |
| full run, seq 192 | **~0.8-1.0 s/iter, stable** |
| 2,000 iters, seq 192 | 33 min, final val loss **1.25** |
| +1,000 more iters at lr 1e-4 | loss plateau ~3.5, no real gain |

Sample replies after 3,000 iterations (short prompt, temp 0.8):

```
Pong tvr ey       -> Oun xa, bong.
Morning oun       -> Good morning bong
Nham bay nv?      -> Bay chii, nem tii.
Bong slnh oun?    -> Oun slnh bong.
```

The adapter becomes more coherent than the base model, which produced
`Oun ka oun bong` and `Ooun ñi, bong`, but it has **not** learned Muffin's
slang. 3,000 single-sample steps are about 11% of one epoch; the train loss
never fell below ~3.2, so this is underfitting, not a hyperparameter problem.

**Verdict:** 8 GB is enough to *run* the 3B QLoRA job, not enough to train it to
convergence. A full epoch is ~26,000 steps at 0.8 s/iter, about 6 hours. Use the
4060 PC for that run (batch 8-16 and CUDA QLoRA, 1.5-3 hours), or accept the
0.5B adapter on the Mac.

If you still want to continue on the Mac, the adapter resumes automatically.
Pause the bot and Ollama first so macOS has memory to spare, and keep an eye on
swap:

```bash
launchctl unload ~/Library/LaunchAgents/com.muffin.bot.plist
brew services stop ollama
ITERS=6000 ./fine_tune/train_3b_mac.sh
# after training
launchctl load ~/Library/LaunchAgents/com.muffin.bot.plist
brew services start ollama
```

Serve the adapter with the prompt it was trained on:

```bash
MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit ADAPTER=adapters/muffin-3b ./serve_mlx.sh
```

```dotenv
LLM_BASE_URL=http://127.0.0.1:8080/v1
LLM_MODEL=Qwen2.5-3B-Instruct-4bit
ADAPTER_MODE=true
COMPACT_PERSONA_FILE=persona_3b.md
```

## Decision: production stays on Argon (2026-11-25)

The Mac 3B adapter is a research result, not the production brain.

- `.env` keeps `LLM_MODEL=gemini-4-argon` and `ADAPTER_MODE=false`.
- The adapter stays at `adapters/muffin-3b/` (gitignored) and is never loaded
  by the running bot.
- Revisit only after a full-epoch run (2 epochs, batch 8+) on the 4060 PC, with
  a blind 1-5 comparison against Argon and the 26-turn eval before switching.
