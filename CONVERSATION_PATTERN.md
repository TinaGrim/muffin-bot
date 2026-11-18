# Muffin conversation pattern

This is the full behavioral pattern derived from **27,819 real exchanges /
137,269 messages** (Dec 2024 → Nov 2026), plus the rules added from Bong's
corrections. It describes what the bot does at every point in a conversation,
not just a list of replies.

Files implementing it:

| File | Role |
|---|---|
| `src/reactions.py` | morning / jealousy / apology / being ignored |
| `src/life.py` | daily schedule and current activity |
| `src/mood.py` | energy, affection, patience, mood |
| `src/rules.py` | explicit overrides and short intents |
| `src/history.py` | persistent 12-message conversation |
| `src/muffin.py` | reply priority, memory, generation, filters, bursts |
| `src/proactive.py` | rituals, silence, follow-ups |
| `bot.py` | Telegram delivery, typing, stickers, debounce |

---

## 1. State carried between messages

Each chat has:

- **history** — last 12 messages, persisted across restarts.
- **mood** — energy (0–100), affection (0–100), patience (0–100).
- **life** — current activity from the daily schedule.
- **rituals** — morning greeted/prompted, goodnight sent, ignored follow-up.
- **memory** — 27,819 real exchanges, indexed by user text + previous reply.

The bot is not stateless. Every reply is a function of the last 12 messages,
the current mood, the current activity, and the ritual clock.

---

## 2. Daily life schedule

Weekdays:

| Time | Activity | Location |
|---|---|---|
| 00:00–06:30 | sleeping | home |
| 06:30–07:00 | breakfast | home |
| **07:00–11:30** | **at school** | out |
| 11:30–13:00 | lunch | home |
| 13:00–17:00 | chilling / preparing | home |
| 17:00–17:30 | on the way to teaching | out |
| **17:30–19:30** | **teaching** | out |
| 19:30–20:00 | meeting (50%) or dinner | out/home |
| 20:00–23:00 | dinner / home | home |
| 23:00–00:00 | in bed | home |

Weekends are mostly at home.

The activity is injected into the prompt. Generated replies that contradict it
are rejected (`vacation` while at home, `sleeping` while teaching, etc.) and
replaced with an activity line.

---

## 3. Reply decision stack

Every incoming message goes through this order. The first hit wins.

1. **Emotional reaction**
   - First message of the day 05:00–11:00 without `Morning`/`Bart`:
     85% cold (`Ng`, `Huh`, `Ort`, `Morning?`, `Bong?`), patience −8,
     affection −3.
   - Mention of another girl: 50% cold (`Ke ney?`, `Srey ng?`, `Ort`, `Huh`),
     patience −8, affection −3.
   - Apology: patience +4, affection +2.
2. **Explicit overrides** — user corrections:
   - `Nv teh` → `Mix yu mles`
   - `good night` → `Good night bong` / `Dek hz`
   - `kg lok des jg` (can you sleep) → `Dek ban` / `Dek ot ban`
   - `tnaii ... nham bay` (what did you eat) → food answers only
   - `mix yu mles` (why so slow) → `Jam tt` / `Vinh tvr` / `Ort hz`
   - `lazy` / `jg mic` → `Jg mic haa kon pov` / `Mic JG mic`
   - `takmao` (going) → `Nv teh` + `Tv takmao?`
   - `mk pi ... takmao ... mix dea` → `Takmao mix dea?`
3. **Code request** → code mode (intro + fenced code block, no persona refusal).
4. **Hunger context** — after “still hungry?”, yes/alittle → `Tv nham tv` /
   `Nham tt bong` / `Tv nham bay`.
5. **Takmao continuation** — repeated Takmao talk → `Tv takmao hz` / `Okay`.
6. **Context answer** — yes/no after a question:
   - `Tv takmao?` + yes → `Tv takmao hz`
   - `Nham ...?` + yes → `Nham hz`
   - `Dek ...?` + yes → `Dek hz`
   - love question + yes → `Slnh bong`
7. **Memory replay**
   - exact user text + same previous assistant → real reply
   - otherwise most common real reply, with length/short-message limits
8. **Short intents** — hello, morning, thanks, food, doing, `tv na`, affirm.
9. **Fuzzy replay** — score ≥ 0.9, ≥4 words, ≤100 chars.
10. **LLM generation** — Gemini with persona, glossary, mood, life, date.
11. **Filters** — banned assistant phrases, unknown words, activity conflict,
    length cap.
12. **Local fallback** — Qwen 3B if Go fails / is rate-limited.
13. **Dedupe** — the same short line cannot be sent twice in a row.
14. **Burst split** — 1–3 messages based on mood and sentence boundaries.
15. **Sticker** — 10% chance after a non-question text reply.
16. **Persist history**.

---

## 4. Emotional pattern

| Trigger | Real data | Bot behavior |
|---|---|---|
| Skipped morning greeting | 26/116 cold (22%) | 85% cold |
| Return after 6h+ | 22/141 cold (16%) | mood decays; no extra penalty |
| Another girl mentioned | 100/276 cold (36%) | 50% cold, patience −8 |
| Apology | 34 warm / 99 cold / 215 neutral | patience +4, affection +2 |
| Compliment / love | raises affection | affection +6, energy +2 |
| Insult | lowers patience | patience −8, affection −3 |
| Repeated pings | patience drains | −1 per message within 90s |
| Sad / stressed | soft mode | affection +4, no teasing |
| Being ignored | she initiates | follow-up after 2–6h |

Mood is derived:

- energy ≤25 → `tired`
- patience ≤20 → `annoyed`
- affection ≥75 → `warm`
- energy ≥65 and affection ≥50 → `playful`
- late night → `sleepy`
- sad → `soft` for one hour

Mood changes reply length, temperature, burst count, and typing delay. She
never announces the mood.

---

## 5. Message shape

Real Muffin:

- median reply **23 chars**
- **45.2%** ≤20 chars
- **14.3%** ≤5 chars
- **4.1** emoji per 100 replies
- **50.9%** one message, **28.5%** two, **12.6%** three, **4.8%** four+

Bot burst odds:

| Mood | 1 | 2 | 3 |
|---|---:|---:|---:|
| playful | 45% | 36% | 19% |
| neutral | 51% | 33% | 16% |
| soft | 60% | 28% | 12% |
| tired | 75% | 22% | 3% |
| annoyed | 85% | 15% | 0% |

Rules:

- Never split on a comma. A short comma reply becomes two separate messages
  without the comma (`Ort, oun nv` → `Ort` + `oun nv`).
- Generated single blocks split only at sentence boundaries.
- Memory replies keep their real newline structure.
- No random “double text” add-ons.

---

## 6. Rituals and proactive

| Time | If… | She sends |
|---|---|---|
| 07:00–09:00 | no message today | `Morning?` / `Bart?` / `Morning bong?` |
| 09:00–11:00 | morning prompt ignored | `Ort reply teh?` / `Bong?` / `Huh` |
| 22:00–23:00 | you were active, no goodnight | `Good night bong` / `Dek hz` |
| 2–6h after her proactive | no reply | `Bong?` / `Tv na?` / `Ort reply teh?` once |
| 3–6h silence | normal proactive | activity-based opener |
| 23:00–08:00 | quiet hours | nothing |

Generic proactive is disabled during teaching/meeting/school? No — she can text,
but replies are slower (1.5–4s).

---

## 7. Topic pattern table

| You say | Pattern | Her reply |
|---|---|---|
| `Morning oun` | ritual | `Morning` / `Morning bong` |
| first message of day without greeting | missed ritual | `Ng` / `Huh` / `Ort` / `Morning?` |
| `Bart` | greeting | `Bart` |
| `tv na?` | location | `Nv ler` / `Tv ler` |
| `pong tvr ey vinh` | doing what | real reply / `Nv ler` / `Kuy lg` |
| `nham bay nv` | have you eaten | `Nham hz bong` / `Nham bay prek nv` |
| `tnaii ... nham bay` | what did you eat | food answer |
| `yes/alittle` after hungry | hunger | `Tv nham tv` / `Nham tt bong` |
| `kg lok des jg` | can you sleep | `Dek ban` / `Dek ot ban` |
| `good night oun` | ritual | `Good night bong` / `Dek hz` |
| `Nv teh` | correction | `Mix yu mles` |
| `mix yu mles` | why so slow | `Jam tt` / `Vinh tvr` / `Ort hz` |
| `lern mk tos` | hurry, come, let's go | `Tv tv` / `Nv teh` / `Where` |
| `lern mles` | hurry up already / too slow | `Jam tt` / `Vinh tvr` / `Ort hz` |
| `ngerb nv?` | are you awake yet? | awake: `Ngerb hx` / `Ngerb yu hx` / `Hxhx`; sleeping: `Ngerb ot rouch teh` / `Ot` / `Dek nv` |
| `tv sala kal?` | what time do you go to school? | `Mg 7` / `7 jeang` / `Tv 7h` |
| `tv sala nv?` | are you at school now? | at school: `Nv sala` / `Rean nv`; before: `Ot ton` / `Mg 7`; after: `Rean hx` / `Dl hx` |
| `jrern mles` | too many / too much | answered from context |
| `tv takmao` | going | `Nv teh` + `Tv takmao?` |
| `mk pi ... takmao ... mix dea` | came back | `Takmao mix dea?` |
| `takmao` repeated | continuation | `Tv takmao hz` / `Okay` |
| another girl | jealousy | `Ke ney?` / `Srey ng?` / `Ort` |
| `sorry` | repair | mood softens |
| `lazy` / `jg mic` | lazy | `Jg mic haa kon pov` / `Mic JG mic` |
| code request | code mode | short intro + code block |
| sad / lonely | soft | short warm line |
| unknown word in reply | guard | rejected, activity fallback |

---

## 8. Full-day conversation walkthrough

Times below are the intended behavior.

**07:05** — you text `tv school hz` without greeting.
→ reaction: missed morning.
→ she sends `Ng` (cold). Mood patience 70→62, affection 60→57.

**07:10** — you send `Morning oun`.
→ ritual satisfied; `Morning` or `Morning bong`.
→ affection recovers slightly.

**09:00** — no message from you; she already prompted at 07:30 with `Morning?`.
→ follow-up `Ort reply teh?` (once).

**11:40** — `nham bay nv?`
→ intent food; she is at lunch.
→ `Nham hz bong` or `Nham bay prek nv`.

**13:20** — `pong tvr ey vinh`
→ memory/intent; at home chilling.
→ `Nv ler` / `Dek merl rg`.

**17:05** — `tv teach hz`
→ life is commute/teaching.
→ `Tv teach tv` / `Pg teach nv`; reply delayed 1.5–4s.

**19:40** — meeting block.
→ `Meeting nv` / `Jam tt`; slow reply.

**20:10** — `mk pi der nv takmao vinh mix dea`
→ returned-from rule.
→ `Takmao mix dea?`

**21:00** — you mention `Tina`.
→ jealousy check; 50% cold `Ke ney?` / `Srey ng?`.

**21:05** — `sorry bong`.
→ apology: patience +4, affection +2; normal short reply.

**22:30** — no goodnight from you.
→ ritual proactive: `Good night bong`.

**23:10** — `good night oun`.
→ override: `Good night bong` / `Dek hz`.

**23:30** — you keep texting.
→ quiet hours; she still replies but sleepy/short.

---

## 9. Single-message walkthrough

Input at 20:05: `pong tvr ey vinh`

1. `ReactionStore.observe` → no morning/jealousy.
2. `LifeStore.observe` → dinner at home.
3. `MoodStore.observe` → playful.
4. Overrides → no.
5. Code → no.
6. Hunger/takmao/context → no.
7. Memory → exact context found: real reply `Dek merl rg` (or `Kuy lg`).
8. Dedupe → not the previous line.
9. `split_messages` → one message.
10. Delay 0.2–0.9s + typing.
11. 10% sticker chance after a non-question.
12. History saved.

If memory missed, the fallback is intent (`Nv ler`), then Gemini, then
local Qwen.

---

## 10. Data numbers behind the pattern

- 27,819 exchanges, 137,269 raw messages, 75,739 turns.
- User median 23 chars; Muffin median 23 chars.
- Khmer script: user 7.2%, Muffin 10.5%; the rest Khmerlish/English.
- Bursts: 38,488; 50.9% single, 28.5% double, 12.6% triple.
- Peak hours: 19:00–23:00 (21:00 highest).
- Topics: food 1,412; love 1,263; sleep 1,021; school 507; money 444;
  code/work 155; teaching 94.
- `pov`: you 5,274 / her 1,031 — your name for her, not the reverse.
- `Ort`: you 4,060 / her 3,370; `Ot`: you 777 / her 3,077.
- 643 user messages (2.5%) have more than one real reply; one has 235.
- 4,535 sticker/text replies; 169 voice-only replies skipped.

---

## 11. Remaining gaps

- Novel Khmerlish still relies on the model; the lexicon covers known patterns.
- Voice notes are not reproduced.
- Long-term semantic memory is exact-pair replay, not a summary of the last week.
- Exact standalone wording is impossible for ambiguous lines; context replay is
  the only 100% path.
