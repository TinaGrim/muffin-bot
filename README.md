# Muffin clone

A Telegram bot that talks like **Muffin**, built from the 27,819 real exchanges
in the exported chat. Two layers:

1. **Retrieval** — every reply is grounded in the closest real messages Muffin
   sent, used as few-shot examples (and reused directly on a near-exact match).
2. **LLM** — an OpenAI-compatible chat model writes the reply in Muffin's voice.

No fine-tuning is required for this baseline. The cleaned dataset is ready for
LoRA training next (`data/sft_train.jsonl`).

## Dataset

Built from `../ChatExport_Muffin ( pitite bot )_3`:

| | |
|---|---|
| Raw messages | 137,269 |
| Merged turns | 75,739 |
| Clean exchanges | **27,819** |
| Train / val | 26,428 / 1,391 |
| Date range | 2024-12-21 → 2026-11-17 |
| Muffin reply length | median 23 chars, p90 78 |
| Voice-note replies skipped | 169 |
| Sticker-only replies skipped | 4,535 |

Rebuild with:

```bash
.venv/bin/python build_dataset.py
```

Outputs: `data/pairs.jsonl`, `data/sft_train.jsonl`, `data/sft_val.jsonl`,
`data/stats.json`, `data/style_report.md`.

## Setup

```bash
cd muffin-bot
cp .env.example .env
```

### 1. Telegram token

1. Open **@BotFather** in Telegram.
2. Send `/newbot`, pick a name and username.
3. Put the token in `.env` as `TELEGRAM_BOT_TOKEN=...`.

### 2. A model backend

**Local (recommended on this Mac):**

```bash
brew install ollama
ollama pull qwen2.5:7b-instruct
```

The defaults in `.env.example` already point at Ollama. This M1/8 GB machine
runs the 7B model at roughly 3 s per short reply using Ollama's Metal backend.
If memory pressure is a problem, switch to `qwen2.5:3b-instruct` — faster, but
noticeably less coherent.

**Hosted (smarter, but the chat text leaves the machine):**

```dotenv
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=google/gemini-2.0-flash-001
```

Any OpenAI-compatible endpoint works.

### 3. Lock it down

Set `ALLOWED_USER_IDS` to your Telegram numeric ID (message **@userinfobot** to
get it) so nobody else can use your bot and burn your API credits.

## Run

```bash
./run.sh                # Telegram long polling
./run.sh --chat         # terminal chat, useful for testing the voice
./run.sh --retrieve "good morning"   # inspect nearest real exchanges
```

Stop with Ctrl+C. The bot remembers the update offset in `data/offset.txt`, so a
restart does not re-answer old messages; messages older than
`MAX_MESSAGE_AGE_SECONDS` are skipped.

## How a reply is produced

1. Normalize the incoming text and, on the first message of a conversation,
   search 27k real user messages for a few short style examples.
2. Hard-code the greetings and courtesies Muffin always gets right (`hello` →
   `Bart`, `good morning` → `Morning`, `thanks` → `Okay`).
3. Wait 1.2 s for a burst of messages, merge the burst into one prompt, and
   send it with the last 20 messages of real conversation history. Retrieved
   examples are only used on the first turn, so they cannot override the flow.
4. Ask the model for a short in-character reply, cap it at 220 characters, strip
   markdown and speaker prefixes, and split short line bursts into separate
   Telegram messages.

`DIRECT_REPLY_THRESHOLD` can re-enable reusing an exact past reply, but it is
`0` by default because it ignores context.

## Feelings

Each chat has a small persistent state in `data/mood.json`:

- **energy** — drops with every reply and late at night, recovers after silence.
- **affection** — rises when you compliment it or seem sad, falls when ignored.
- **patience** — drops when you ping it repeatedly or insult it, recovers when
  you apologise or after a break.
- **mood** — derived from those numbers: `playful`, `warm`, `tired`, `sleepy`,
  `annoyed`, `soft`, or `neutral`.

The mood is injected into the prompt as a private feeling, never announced:
annoyed means one-word answers, tired means low effort, soft means no teasing.
It also changes reply length, temperature, and how long the typing indicator
runs. Send `/mood` to see the current numbers out of character.

Recent topics (food, work, affection, sadness) are remembered for up to 8 hours
so the bot can bring them up naturally, e.g. asking about the meal it discussed
earlier.

## Files

```
muffin-bot/
├── bot.py              Telegram polling + terminal chat
├── build_dataset.py    Telegram HTML export -> clean training data
├── persona.md          system prompt (voice, not biography)
├── src/muffin.py       retrieval prompt assembly + LLM client
├── src/retrieval.py    char n-gram index with pickled cache
├── data/               generated dataset (gitignored index cache)
└── .env                token and model config (never commit)
```

## Quality eval

`question_audit.py` is a regression check on exact archive lines. Real quality is
measured with multi-turn dialogs:

```bash
.venv/bin/python eval_conversations.py          # review page
.venv/bin/python eval_conversations.py --judge  # + LLM scores
```

See `EVAL.md`. In Telegram, `/metrics` shows latency, fallback rate and style fixes.

## Privacy

The export is a private two-person chat. `data/pairs.jsonl`, `data/sft_*.jsonl`
and the retrieval cache contain real message text — keep them local and do not
publish them without the other person's consent. `.env` holds your bot token;
revoke it with BotFather if it leaks.

## Next step: fine-tune

`data/sft_train.jsonl` is already in chat format (system + up to 4 context
turns + target). On this M1/8 GB the realistic options are:

- **LoRA on Qwen2.5-1.5B/3B with MLX** locally (a few hours, slow but free).
- **Cloud LoRA** on a 7B–14B model (a few dollars) for noticeably better Khmerlish.

Keep the retrieval layer after fine-tuning: it supplies facts and recent slang,
while the adapter supplies the voice.

## Mood and reaching out

Mood is persistent, not per-message. An insult, being ignored for hours, or
hearing about another girl starts a grudge (`upset` in `data/mood.json`).

- Time alone never clears it; normal chat does not clear it either.
- An apology chips at it; real pampering (`slnh`, `cute`, `sorry`) clears it
  and leaves her soft for a while.
- While the grudge is active, replies stay short and cold, and proactive texts
  are cold nudges instead of cheerful ones.

Reach-outs run every 1.5-3 hours during 08:00-23:00 (see `PROACTIVE_*` in
`.env`), with a dedicated nudge after one hour of silence while she is upset.
Morning and good-night rituals still apply.

## Real chat context

The bot now keeps three layers of context, not just the last message:

1. **Recent turns** - up to 40 messages or 2,000 characters (`HISTORY_MESSAGES`,
   `HISTORY_CHARS`), stored with timestamps and persisted across restarts.
   Profane or junk bot lines are removed before the prompt, so a bad fallback
   does not poison the next reply.
2. **Durable facts** - `data/facts.json` stores what Bong actually said about
   exams, school, work, plans, family, health and preferences. Only explicit
   statements are stored; questions are never turned into facts. The prompt
   gets the five most relevant lines, not the whole list.
3. **Thread state** - current topic, the question she asked last, and the last
   question Bong asked, plus how long the chat has been quiet.

`/reset` clears history, facts and thread state for the chat.

## Mood repair and reach-out limits

Being ignored is still a grudge, but it is no longer a trap:

- A grudge starts only when a **direct question or greeting** was left unanswered
  for **3 hours**, outside school (07-11) and teaching (17-21) hours. A random
  `Bong?` does not start one.
- Asking why she is cold (`Hoy ey oun`, `mix ban hoy`) makes her say what is
  wrong in one line. An apology or real pampering clears the grudge; coming back,
  explaining, or asking about her chips at it.
- She may be cold, but at most two cold one-word replies in a row. The third
  explains the grievance and asks for repair.
- Proactive cadence is 3-5 hours plus morning/goodnight, about 5-6 reach-outs a
  day. At most two can go unanswered, and only one empty `Bong?` / `Hoy` is
  allowed per day.

`judge_conversation.py` now runs three reviews by default, labels proactive and
note-corrected turns, and includes the persona rules in the prompt. A note you
asked for is no longer scored as a bot failure.
