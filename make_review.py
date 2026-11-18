#!/usr/bin/env python3
"""Generate a human review sheet of real question -> bot answer pairs.

The pipeline is run with no model available, so every row shows what the
deterministic layer answers. Rows the model would have handled fall back to an
activity line, which makes the gaps obvious.

Output: data/QA_REVIEW.html (clickable checkboxes) and data/QA_REVIEW.md.
"""

from __future__ import annotations

import html
import logging
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from history import HistoryStore  # noqa: E402
from life import Activity, LifeStore  # noqa: E402
from muffin import Chat, Muffin, load_config  # noqa: E402
from mood import Mood, MoodStore  # noqa: E402
from reactions import ReactionState, ReactionStore  # noqa: E402
from retrieval import CachedRetriever  # noqa: E402
from threads import ThreadState, ThreadStore  # noqa: E402

# (section title, scheduled activity, Bong's messages)
SECTIONS: list[tuple[str, str, list[str]]] = [
    (
        "The bad conversation (at school)",
        "school",
        [
            "Ort rean heh",
            "Jenh nv",
            "Pong tvr ey",
            "Hak min men nv rean teh?",
            "Jos minh tha nv ptes",
        ],
    ),
    (
        "Asking what she is doing - at home chilling",
        "chill",
        ["Pong tvr ey", "I mean what are you doing", "Pong tvr ey", "Bong tvr ey?"],
    ),
    (
        "Asking what she is doing - at school",
        "school",
        ["Pong tvr ey", "What are you doing", "Oun tvr ey?"],
    ),
    (
        "Asking what she is doing - teaching",
        "teach",
        ["Pong tvr ey", "What are you doing"],
    ),
    (
        "Asking what she is doing - eating dinner",
        "dinner",
        ["Pong tvr ey", "Nham bay nv", "Nham ey?"],
    ),
    (
        "Asking what she is doing - sleeping",
        "sleep",
        ["Pong tvr ey", "Kg lok nv?", "Ngerb nv?"],
    ),
    (
        "Studying / not studying",
        "school",
        [
            "Ort rean heh",
            "Hak min men nv rean teh?",
            "Kru ot rean teh?",
            "Jenh nv",
            "Jos minh tha nv ptes",
        ],
    ),
    (
        "Studying / not studying - at home",
        "chill",
        ["Ort rean heh", "Hak min men nv rean teh?", "Jos minh tha nv ptes"],
    ),
    (
        "Everyday greetings",
        "chill",
        ["Bart", "Bong", "Bong tv na?", "Tv na?", "Bong kit ey?", "Bong?", "Hoy"],
    ),
    (
        "Morning and night",
        "night",
        [
            "Good night",
            "Good night pov can we call?",
            "Dek nv?",
            "Ngerb nv?",
            "Dek hz?",
            "Morning bong",
        ],
    ),
    (
        "Affection, teasing, jealousy",
        "chill",
        [
            "I love you",
            "Miss you",
            "Hug bong",
            "Kiss kiss?",
            "Bong slnh oun?",
            "Oun slnh bong",
            "Bong mean srey?",
            "Bong nguy?",
            "Bong chob ey?",
        ],
    ),
    (
        "Food and money",
        "lunch",
        [
            "Nham bay nv",
            "Nham bay hz",
            "Nham ey?",
            "Ot nham teh",
            "Nguy",
            "Luy?",
            "Ot mean luy",
            "Bong mean luy?",
        ],
    ),
    (
        "Schedule questions",
        "chill",
        [
            "Tv sala kal?",
            "Tv sala mong marn?",
            "Kal?",
            "Mong marn?",
            "Ban marn?",
            "Therb?",
        ],
    ),
    (
        "Short replies with no context",
        "chill",
        ["Hx", "Ort", "Na", "Huh", "Kuy lg", "Mix dea?", "Ng haii?", "Sry bong"],
    ),
    (
        "English messages",
        "chill",
        [
            "What are you doing",
            "I am eating",
            "I am at school",
            "I love you",
            "Good night",
            "I am sad",
        ],
    ),
    (
        "Work and code requests",
        "work",
        ["Pong tvr ey", "Write python code for me", "Tvr code nv?"],
    ),
]

FALLBACK_MARKERS = (
    "safe fallback",
    "activity conflict",
    "duplicate reply",
    "stock glossary echo",
    "memory replay",
    "time-of-day replay",
    "cold reply",
)


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def take(self) -> list[str]:
        out, self.messages = self.messages, []
        return out


def source_of(messages: list[str]) -> str:
    found = [m for m in messages if any(k in m for k in FALLBACK_MARKERS)]
    if not found:
        return "rule"
    if any("memory replay" in m or "time-of-day" in m for m in found):
        return "real chat replay"
    if any("safe fallback" in m for m in found):
        return "fallback"
    if any("activity conflict" in m for m in found):
        return "conflict"
    if any("duplicate" in m for m in found):
        return "dedupe"
    return "other"


def build_rows(muffin: Muffin, mood: MoodStore, life: LifeStore,
               reactions: ReactionStore, threads: ThreadStore, capture: Capture):
    rows = []
    for index, (title, activity, questions) in enumerate(SECTIONS):
        chat_id = 9000 + index
        muffin.chats[chat_id] = Chat(history=[])
        mood.chats[str(chat_id)] = Mood(energy=60, affection=60, patience=70)
        now = time.time()
        life.chats[str(chat_id)] = Activity(
            key=activity, started=now, ends=now + 3600
        )
        reactions.chats[str(chat_id)] = ReactionState(
            last_user_date=date.today().isoformat(),
            morning_greeted=date.today().isoformat(),
            last_user_ts=now,
        )
        threads.chats[str(chat_id)] = ThreadState()
        capture.take()
        for question in questions:
            before = len(capture.messages)
            answer = " | ".join(muffin.reply(chat_id, question))
            messages = capture.messages[before:]
            rows.append(
                {
                    "section": title,
                    "activity": activity,
                    "question": question,
                    "answer": answer,
                    "source": source_of(messages),
                }
            )
    return rows


def render_html(rows: list[dict]) -> str:
    sections: dict[str, list[dict]] = {}
    for row in rows:
        sections.setdefault(row["section"], []).append(row)
    total = len(rows)
    fallback = sum(1 for r in rows if r["source"] == "fallback")
    body = []
    for title, items in sections.items():
        body.append(f"<section><h2>{html.escape(title)}</h2>")
        body.append('<table><thead><tr>'
                    '<th class="tick">OK</th>'
                    '<th class="tick">Fix</th>'
                    '<th>Bong says</th><th>Muffin answers</th>'
                    '<th>Source</th><th>Your note</th>'
                    '</tr></thead><tbody>')
        for row in items:
            src = html.escape(row["source"])
            flag = ' class="warn"' if row["source"] == "fallback" else ""
            body.append(
                f'<tr{flag}>'
                f'<td class="tick"><input type="checkbox" aria-label="ok"></td>'
                f'<td class="tick"><input type="checkbox" aria-label="fix"></td>'
                f'<td class="q">{html.escape(row["question"])}</td>'
                f'<td class="a">{html.escape(row["answer"])}</td>'
                f'<td class="src">{src}</td>'
                f'<td><input class="note" type="text" placeholder="..."></td>'
                f'</tr>'
            )
        body.append('</tbody></table></section>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width initial-scale=1">
<title>Muffin QA review</title>
<style>
 :root {{ --ink:#1c1c1e; --muted:#6b6b70; --line:#e3e3e8; --warn:#fff8e6; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; padding:28px 20px 80px; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#fafafc; }}
 main {{ max-width:1040px; margin:0 auto; }}
 h1 {{ font-size:26px; margin:0 0 6px; }}
 .sub {{ color:var(--muted); margin:0 0 18px; }}
 .bar {{ position:sticky; top:0; z-index:5; background:#fafafc; padding:10px 0; border-bottom:1px solid var(--line); margin-bottom:18px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }}
 .bar label {{ color:var(--muted); font-size:13px; display:flex; gap:6px; align-items:center; }}
 .bar button {{ font:inherit; font-size:13px; padding:5px 10px; border:1px solid var(--line); border-radius:7px; background:#fff; cursor:pointer; }}
 section {{ margin:0 0 26px; }}
 h2 {{ font-size:17px; margin:0 0 8px; }}
 table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
 th {{ text-align:left; font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); padding:9px 10px; border-bottom:1px solid var(--line); background:#f4f4f7; }}
 td {{ padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
 tr:last-child td {{ border-bottom:0; }}
 tr.warn td {{ background:var(--warn); }}
 .tick {{ width:44px; text-align:center; }}
 input[type=checkbox] {{ width:18px; height:18px; cursor:pointer; }}
 .q {{ width:30%; font-weight:500; }}
 .a {{ width:32%; }}
 .src {{ width:13%; color:var(--muted); font-size:12px; }}
 .note {{ width:100%; border:0; border-bottom:1px dashed var(--line); padding:4px 2px; font:inherit; font-size:13px; background:transparent; }}
 .note:focus {{ outline:none; border-bottom-color:#888; }}
 .summary {{ display:flex; gap:22px; margin:0 0 18px; font-size:13px; color:var(--muted); }}
 .summary b {{ color:var(--ink); font-size:16px; display:block; }}
 @media print {{ body {{ background:#fff; padding:0; }} .bar {{ display:none; }} tr.warn td {{ background:#f4f4f4; }} }}
</style></head><body><main>
<h1>Muffin - question &amp; answer review</h1>
<p class="sub">Tick <b>OK</b> for a good answer, tick <b>Fix</b> for a wrong one, and write the correct line in the note.
Highlighted rows fell back to an activity line because no rule or real-chat match exists.</p>
<div class="summary">
 <div><b>{total}</b>lines</div>
 <div><b>{total - fallback}</b>from rules or real chat</div>
 <div><b>{fallback}</b>fallback lines</div>
</div>
<div class="bar">
 <label><input type="checkbox" id="hideOk"> hide lines I marked OK</label>
 <label><input type="checkbox" id="onlyWarn"> only highlighted lines</label>
 <button onclick="window.print()">Print / save PDF</button>
 <button id="copy" onclick="copyResults()">Copy my answers</button>
 <button onclick="resetAll()">Clear all ticks</button>
</div>
{''.join(body)}
</main>
<script>
const rows=[...document.querySelectorAll('tbody tr')];
rows.forEach((tr,i)=>{{
  const ok=tr.querySelector('[aria-label=ok]');
  const fix=tr.querySelector('[aria-label=fix]');
  const note=tr.querySelector('.note');
  const key='muffin-qa-'+i;
  const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
  ok.checked=!!saved.ok; fix.checked=!!saved.fix; note.value=saved.note||'';
  const save=()=>localStorage.setItem(key,JSON.stringify({{ok:ok.checked,fix:fix.checked,note:note.value}}));
  ok.onchange=()=>{{ if(ok.checked) fix.checked=false; save(); }};
  fix.onchange=()=>{{ if(fix.checked) ok.checked=false; save(); }};
  note.oninput=save;
}});
document.getElementById('hideOk').onchange=e=>{{
  rows.forEach(tr=>{{
    const ok=tr.querySelector('[aria-label=ok]');
    if(e.target.checked&&ok.checked) tr.style.display='none';
    else if(!document.getElementById('onlyWarn').checked) tr.style.display='';
  }});
}};
document.getElementById('onlyWarn').onchange=e=>{{
  rows.forEach(tr=>{{
    tr.style.display = e.target.checked && !tr.classList.contains('warn') ? 'none' : '';
  }});
}};
function resetAll(){{ rows.forEach(tr=>{{ tr.querySelectorAll('input[type=checkbox]').forEach(b=>b.checked=false); tr.querySelector('.note').value=''; }}); rows.forEach((tr,i)=>localStorage.removeItem('muffin-qa-'+i)); }}
function copyResults(){{
  const out=rows.map(tr=>({{
    bong:tr.querySelector('.q').textContent,
    muffin:tr.querySelector('.a').textContent,
    ok:tr.querySelector('[aria-label=ok]').checked,
    fix:tr.querySelector('[aria-label=fix]').checked,
    note:tr.querySelector('.note').value
  }}));
  navigator.clipboard.writeText(JSON.stringify(out,null,1)).then(()=>{{
    const b=document.getElementById('copy'); b.textContent='Copied'; setTimeout(()=>b.textContent='Copy my answers',1500);
  }});
}}
</script>
</body></html>
"""


def render_md(rows: list[dict]) -> str:
    lines = ["# Muffin - question & answer review", ""]
    current = None
    for row in rows:
        if row["section"] != current:
            current = row["section"]
            lines += ["", f"## {current}", ""]
        answer = " / ".join(row["answer"].splitlines())
        lines.append(f"- [ ] **{row['question']}**")
        lines.append(f"  - Muffin: `{answer}`  ({row['source']})")
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("muffin")
    logger.setLevel(logging.INFO)
    capture = Capture()
    logger.addHandler(capture)

    config = load_config(
        {
            "LLM_BASE_URL": "http://127.0.0.1:1/v1",
            "LLM_API_KEY": "none",
            "LLM_MODEL": "none",
            "FALLBACK_BASE_URL": "http://127.0.0.1:1/v1",
            "FALLBACK_MODEL": "none",
            "USE_RULES": "true",
            "EXAMPLES_MODE": "never",
        },
        ROOT / "persona.md",
    )
    retriever = CachedRetriever(
        ROOT / "data" / "pairs.jsonl", ROOT / "data" / "retrieval.pkl"
    ).get()
    tmp = Path(tempfile.mkdtemp())
    mood = MoodStore(tmp / "mood.json")
    life = LifeStore(tmp / "life.json")
    history = HistoryStore(tmp / "history.json")
    reactions = ReactionStore(tmp / "reactions.json")
    threads = ThreadStore(tmp / "threads.json")
    muffin = Muffin(config, retriever, mood, life, history, reactions, threads)

    rows = build_rows(muffin, mood, life, reactions, threads, capture)

    html_path = ROOT / "data" / "QA_REVIEW.html"
    md_path = ROOT / "data" / "QA_REVIEW.md"
    html_path.write_text(render_html(rows), encoding="utf-8")
    md_path.write_text(render_md(rows), encoding="utf-8")
    fallback = sum(1 for r in rows if r["source"] == "fallback")
    print(f"{len(rows)} lines, {fallback} fallback")
    print(f"wrote {html_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
