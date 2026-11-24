# JEV pair rating

## Installed

`vendor/jev-ultrafast` (Browser Use × TypeSafe Jev):

```bash
cd vendor/jev-ultrafast
uv sync
uv run pytest -q   # 31 passed
```

## Pairs from the real conversation

`data/pairs.jsonl` is the real Bong ↔ Muffin export.

```bash
.venv/bin/python prepare_jev_pairs.py
# 27,819 raw -> 22,032 clean -> 120 prepared
```

Output: `data/jev_pairs.jsonl` (question, answer, real context, date, source id).

## Rating

`jev_rate.py` has two scorers:

1. **True Jev** — TypeSafe `/v1/systemone` choice API, the same model JEV
   Ultrafast uses. Requires:
   ```bash
   TYPESAFE_API_KEY=...
   TYPESAFE_MODEL=jev-latest
   ```
   The five rating choices are offered to Jev: 1 = unrelated, 5 = direct and
   in character.

2. **Text fallback** — an OpenAI-compatible judge, used only when no TypeSafe
   key is configured. The output is labelled `text-fallback:<model>` so it is
   never confused with a real Jev score.

```bash
# uses TypeSafe Jev when TYPESAFE_API_KEY is set
.venv/bin/python jev_rate.py --limit 120 --concurrency 3

# explicit fallback
JUDGE_MODEL=gemini-4-argon .venv/bin/python jev_rate.py
```

Outputs:

- `data/jev_ratings.jsonl` — every pair, score, reason, scorer
- `data/jev_train.jsonl` — score ≥ 4 only, SFT messages with real context
- `data/JEV_RATINGS.md` — distribution and examples

## Current run

- 120 pairs rated
- mean **3.73 / 5**
- score ≥ 4: **77** rows in `data/jev_train.jsonl`
- scorer: `text-fallback:gemini-4-argon`

No `TYPESAFE_API_KEY` is configured on this machine, so the true Jev choice
scorer could not be called. Add the key and rerun the first command to produce
real Jev ratings with the same script.
