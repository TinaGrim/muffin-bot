# Training Muffin on a 32 GB RAM + RTX 4060 PC

This is the real fine-tune path. It runs on the PC, not on the 8 GB Mac.

## What you need

| Component | Requirement |
|---|---|
| GPU | RTX 4060 8 GB or 16 GB (8 GB works with the settings below) |
| RAM | 16 GB minimum, 32 GB comfortable |
| OS | Ubuntu or Windows + WSL2 (Unsloth does not run natively on macOS) |
| Disk | ~20 GB free (7B base + adapter + merged model) |
| Time | 1–3 hours for 2 epochs on 25k samples |

## 1. Copy the data

Run this on the Mac first:

```bash
.venv/bin/python fine_tune/prepare_data.py
```

Output:

```
train: 25463 | words p50 5 p95 23 max 52
valid: 1334  | words p50 4 p95 17 max 45
```

Copy `fine_tune/data/` and `fine_tune/train_unsloth.py` into the same folder
on the PC.

## 2. Install on the PC

```bash
# WSL2 Ubuntu 22.04+, then:
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes datasets
```

Check the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"
```

## 3. Choose the model for your VRAM

| VRAM | Model | `--max-seq` | `--batch` | `--grad-accum` |
|---|---|---|---|---|
| 8 GB | `Qwen/Qwen2.5-7B-Instruct` | 1024 | 1 | 16 |
| 8 GB, if it OOMs | `Qwen/Qwen2.5-3B-Instruct` | 2048 | 2 | 8 |
| 16 GB | `Qwen/Qwen2.5-7B-Instruct` | 2048 | 2 | 8 |

7B in 4-bit is about 4.5 GB of weights. The rest is activations, so 8 GB only
works with sequence length 1024, batch 1 and gradient checkpointing. Do not
raise `--max-seq` first; raise it only after a successful run.

## 4. Train

```bash
python fine_tune/train_unsloth.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --train fine_tune/data/train.jsonl \
  --valid fine_tune/data/valid.jsonl \
  --out fine_tune/out/muffin-7b \
  --max-seq 1024 --batch 1 --grad-accum 16 \
  --epochs 2 --lr 2e-4 \
  --save-steps 200
```

What the script does:

- loads the base model in 4-bit (QLoRA)
- LoRA `r=16`, `alpha=32` on all attention and MLP projections
- trains only on Muffin's reply (`train_on_responses_only`), not on the
  persona or Bong's message
- evaluates on the held-out 1,334 samples every 200 steps
- saves checkpoints to `fine_tune/out/muffin-7b`

Expected loss: starts around 1.8–2.2, settles near 0.7–1.0. If it stays above
1.5 after 1 epoch, lower the learning rate to `1e-4`.

## 5. Export for the Mac

Merge the adapter and quantize to GGUF Q4_K_M:

```bash
python fine_tune/train_unsloth.py ... --merge --gguf
```

This writes:

- `fine_tune/out/muffin-7b-merged/` (16-bit HF model, ~15 GB)
- `fine_tune/out/muffin-7b-gguf/` (Q4_K_M, ~4.7 GB)

Copy the GGUF file to the Mac and import it into Ollama:

```bash
# on the Mac
ollama create muffin-7b -f Modelfile
```

`Modelfile`:

```
FROM ./muffin-7b.Q4_K_M.gguf
PARAMETER temperature 0.8
PARAMETER num_predict 60
PARAMETER stop "<|im_end|>"
```

Then point the bot at it:

```dotenv
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=muffin-7b
LLM_TIMEOUT=20
MAX_REPLY_SECONDS=25
ADAPTER_MODE=false
```

Keep the retrieval index and `persona.md`; the fine-tune changes the voice,
not the facts. `ADAPTER_MODE=false` because the 7B model was trained with the
full system prompt, not the compact one.

## 6. Measure before switching

Do not trust the training loss. Compare on the same 26-turn eval:

```bash
.venv/bin/python eval_conversations.py --judge
```

Current production baseline: **style 0 issues, 3 fallbacks, p95 13.8 s**.
A fine-tune that improves relevance but writes long paragraphs is a regression.
Also run 100 real messages A/B: fine-tuned model vs Argon, blind, scored 1–5
with `jev_rate.py`'s rubric.

## 7. Recommended production shape

A 7B Q4 on an 8 GB M1 is slower than the hosted Argon model. Best of both:

- **Primary:** fine-tuned 7B for normal chat (voice, slang, short replies)
- **Fallback:** Argon for messages that need reasoning or when the local model
  exceeds the deadline
- **Keep:** retrieval, style guard, burst handling, feedback buttons

That gives Muffin's real voice without giving up the safety net.

## Files

- `fine_tune/prepare_data.py` — cleaning, dedupe, held-out split
- `fine_tune/train_unsloth.py` — QLoRA training + merge/GGUF export
- `fine_tune/data/train.jsonl` — 25,463 clean samples
- `fine_tune/data/valid.jsonl` — 1,334 held-out samples
