# Quality evaluation

There are two different measurements. They answer different questions.

| Tool | Question | Use |
|---|---|---|
| `question_audit.py` / `reaudit.py` | Can the bot reproduce a real archive line for a known question? | Regression check on retrieval and rules |
| `eval_conversations.py` | Does a multi-turn conversation stay in character and follow context? | Quality gate before changing prompts or models |

The exact-match score (98.2%) is **not** a conversation score. It cannot see a
27-second reply, a Khmer-script leak, or an answer copied from an unrelated
chat. The multi-turn eval is the quality gate for those.

## Run

```bash
.venv/bin/python eval_conversations.py           # human review page
.venv/bin/python eval_conversations.py --judge   # + LLM judge scores
.venv/bin/python eval_conversations.py --limit 1 # one turn per dialog
```

Outputs:

- `data/EVAL_REPORT.md` — every dialog, latency, style flags, fallback use
- `data/EVAL_REVIEW.html` — OK / Fix + note per turn, with **Copy review**

Each dialog runs in a throwaway state directory, so live history, mood and
feedback are never touched.

## Metrics

Every reply records latency, the provider that answered, fallback use and style
fixes in `data/metrics.jsonl`. In Telegram:

```
/metrics
```

shows count, p50/p95 latency, fallback rate, style fixes and the top providers.
