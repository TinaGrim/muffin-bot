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

The 0.5B recipe above still works. A 3B 4-bit model fits *only* when the Mac
has free memory; see the warning below.

```bash
# one-time model download
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; print(d('mlx-community/Qwen2.5-3B-Instruct-4bit'))"

# train (resumes automatically if an adapter exists)
./fine_tune/train_3b_mac.sh

# longer run
ITERS=4000 ./fine_tune/train_3b_mac.sh
```

Settings: 4-bit base, LoRA rank 8 on 16 layers, batch 1, seq 512,
gradient checkpointing, lr 5e-5, prompt masked.

Measured on this Mac:

| Run | Result |
|---|---|
| 10 iters, seq 512, batch 1 | 2.2 s, loss 12.4 → 5.9 |
| 3B full run, 8 GB under pressure | ~3 s/iter, swaps hard |
| 3B after a fresh reboot (expected) | ~0.2–0.5 s/iter |

If the log stalls after the first `val` line and CPU stays near 10%, macOS is
swapping. Check:

```bash
sysctl vm.swapusage
vm_stat | grep -E "Pages free|Pages wired"
```

With more than ~4 GB of swap in use, reboot first. The Mac had 7 days of uptime
and 10 GB of swap when 3B thrashing was observed. Pause the bot and Ollama
during training:

```bash
launchctl unload ~/Library/LaunchAgents/com.muffin.bot.plist
brew services stop ollama
# after training
launchctl load ~/Library/LaunchAgents/com.muffin.bot.plist
brew services start ollama
```

Serve the adapter:

```bash
MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit ADAPTER=adapters/muffin-3b ./serve_mlx.sh
```

Then set `LLM_MODEL=Qwen2.5-3B-Instruct-4bit` (the server reports the base name)
and `ADAPTER_MODE=true` in `.env`.

The 4060 PC is still much faster for the same 3B/7B job: CUDA QLoRA uses 8 GB
of VRAM instead of fighting macOS for unified memory.
