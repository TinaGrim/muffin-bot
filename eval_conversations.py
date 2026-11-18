#!/usr/bin/env python3
"""Multi-turn conversation eval: run scripted dialogs and review the replies.

The old exact-match audit answers "can the bot reproduce an archive line?".
This answers the real question: "does a conversation hold together?".

    .venv/bin/python eval_conversations.py            # run + HTML review
    .venv/bin/python eval_conversations.py --judge    # + LLM judge scores

Each dialog runs in a throwaway store, so live history/mood/feedback are never
touched. Uses the configured primary model, exactly like the bot.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from history import HistoryStore  # noqa: E402
from life import LifeStore  # noqa: E402
from mood import MoodStore  # noqa: E402
from muffin import Muffin, load_config  # noqa: E402
from reactions import ReactionStore  # noqa: E402
from retrieval import CachedRetriever  # noqa: E402
from threads import ThreadStore  # noqa: E402

DIALOGS = [
    ("morning", ["Morning bong", "Ngerb nv?", "Pong tvr ey", "Tv sala kal?", "Jenh nv"]),
    ("food", ["Nham bay nv", "Nham ey?", "Ot nham teh", "Nguy", "Luy?"]),
    (
        "teaching",
        [
            "Mk pi prean vinh hz men",
            "Kjel tv prean nv",
            "Heh tv prean nas ort tv mic kert",
        ],
    ),
    (
        "unclear",
        ["Der kg jit nis ter", "Mi ey vinh jos", "Mean pen 1 ort kjei sen"],
    ),
    ("affection", ["Bong slnh oun?", "Oun slnh bong", "Kiss kiss?", "Good night"]),
    ("english", ["What are you doing", "I am eating", "Good night pov can we call?"]),
    ("correction", ["Pong tvr ey", "Nham bay nv", "Tmr jg tv na ort"]),
]

JUDGE_PROMPT = """You are reviewing a chatbot that imitates Muffin, a young
Cambodian woman texting in Khmerlish (Romanized Khmer mixed with English).

For every assistant reply in the transcript, score 1-5:
- persona: does it sound like blunt, playful Muffin, not an assistant?
- context: does it continue the conversation and remember what was just said?
- style: Khmerlish, short, no Khmer script, no commas, no English paragraphs.

Return only JSON:
{"turns":[{"turn":1,"persona":4,"context":4,"style":4,"note":"..."}],
 "summary":"one sentence"}"""


@dataclass
class Turn:
    index: int
    user: str
    reply: str
    latency: float
    provider: str
    fallbacks: list[str]
    style_issues: list[str]


def build_retriever(env: dict[str, str]):
    return CachedRetriever(
        ROOT / "data" / "pairs.jsonl",
        ROOT / "data" / "retrieval.pkl",
        embeddings=env.get("USE_EMBEDDINGS", "true").lower()
        not in {"0", "false", "no"},
        embed_model=env.get("EMBED_MODEL", "bge-m3"),
        embed_base=env.get("EMBED_BASE_URL", "http://localhost:11434"),
        embed_path=ROOT / "data" / env.get("EMBED_INDEX", "embeddings_bge_m3.npz"),
    ).get()


def make_muffin(directory: Path, env: dict[str, str], retriever=None) -> Muffin:
    config = load_config(env, ROOT / "persona.md")
    retriever = retriever or build_retriever(env)
    return Muffin(
        config,
        retriever,
        MoodStore(directory / "mood.json"),
        LifeStore(directory / "life.json"),
        HistoryStore(directory / "history.json"),
        ReactionStore(directory / "reactions.json"),
        ThreadStore(directory / "threads.json"),
    )


def run(env: dict[str, str], limit: int = 0) -> dict[str, dict]:
    retriever = build_retriever(env)
    results: dict[str, dict] = {}
    for name, texts in DIALOGS:
        turns = texts[:limit] if limit else texts
        directory = Path(tempfile.mkdtemp(prefix=f"muffin-eval-{name}-"))
        muffin = make_muffin(directory, env, retriever)
        rows: list[Turn] = []
        for index, text in enumerate(turns, 1):
            started = time.perf_counter()
            try:
                parts = muffin.reply(1, text)
            except Exception as exc:  # noqa: BLE001
                parts = [f"ERROR: {exc}"]
            rows.append(
                Turn(
                    index=index,
                    user=text,
                    reply="\n".join(parts),
                    latency=time.perf_counter() - started,
                    provider=muffin.last_provider,
                    fallbacks=list(muffin.last_fallbacks),
                    style_issues=list(muffin.last_style_issues),
                )
            )
        results[name] = {"turns": rows}
    return results


def transcript(rows: list[Turn]) -> str:
    lines = []
    for row in rows:
        lines.append(f"Bong: {row.user}")
        lines.append(f"Muffin: {row.reply}")
    return "\n".join(lines)


def judge(muffin: Muffin, rows: list[Turn]) -> dict:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": transcript(rows)},
    ]
    try:
        raw = muffin.llm.chat(
            messages, temperature=0.0, max_tokens=900, session_id="muffin-eval"
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {"error": "judge did not return JSON", "raw": raw[:300]}
    try:
        return json.loads(match.group(0))
    except ValueError as exc:
        return {"error": f"judge JSON invalid: {exc}", "raw": raw[:300]}


def summarise(results: dict[str, dict]) -> dict:
    turns = [turn for data in results.values() for turn in data["turns"]]
    latencies = [turn.latency for turn in turns]
    return {
        "turns": len(turns),
        "p50": statistics.median(latencies) if latencies else 0.0,
        "p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
        "style_issues": sum(1 for turn in turns if turn.style_issues),
        "fallbacks": sum(1 for turn in turns if turn.fallbacks),
        "errors": sum(1 for turn in turns if turn.reply.startswith("ERROR:")),
    }


def report_markdown(results: dict[str, dict], stats: dict) -> str:
    lines = [
        "# Muffin multi-turn eval",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"- turns: **{stats['turns']}**",
        f"- latency p50: **{stats['p50']:.1f}s**, p95: **{stats['p95']:.1f}s**",
        f"- style issues: **{stats['style_issues']}**",
        f"- fallback replies: **{stats['fallbacks']}**",
        f"- errors: **{stats['errors']}**",
        "",
    ]
    for name, data in results.items():
        lines += [f"## {name}", "", "| # | Bong | Muffin | s | flags |", "|---|---|---|---:|---|"]
        for turn in data["turns"]:
            flags = ", ".join(turn.style_issues)
            if turn.fallbacks:
                flags = (flags + ", " if flags else "") + "fallback:" + "/".join(turn.fallbacks)
            reply = turn.reply.replace("\n", " / ").replace("|", "\\|")
            lines.append(
                f"| {turn.index} | {turn.user} | {reply} | {turn.latency:.1f} | {flags} |"
            )
        judge_data = data.get("judge", {})
        if judge_data.get("turns"):
            scores = [
                f"t{item.get('turn')}: p{item.get('persona')} c{item.get('context')} s{item.get('style')}"
                for item in judge_data["turns"]
            ]
            lines += ["", "Judge: " + "; ".join(scores)]
        if judge_data.get("summary"):
            lines += ["", f"Judge summary: {judge_data['summary']}"]
        lines.append("")
    return "\n".join(lines)


def report_html(results: dict[str, dict], stats: dict) -> str:
    cards = "".join(
        f'<div class="card"><b>{value}</b><span>{label}</span></div>'
        for value, label in [
            (stats["turns"], "turns"),
            (f"{stats['p50']:.1f}s", "p50 latency"),
            (f"{stats['p95']:.1f}s", "p95 latency"),
            (stats["style_issues"], "style issues"),
            (stats["fallbacks"], "fallback replies"),
            (stats["errors"], "errors"),
        ]
    )
    blocks = []
    for name, data in results.items():
        rows = []
        for turn in data["turns"]:
            flags = ", ".join(turn.style_issues)
            if turn.fallbacks:
                flags = (flags + ", " if flags else "") + "fallback:" + "/".join(turn.fallbacks)
            rows.append(
                f'<tr data-i="{name}-{turn.index}">'
                f'<td class="n">{turn.index}</td>'
                f'<td class="q">{html.escape(turn.user)}</td>'
                f'<td class="a">{html.escape(turn.reply)}</td>'
                f'<td class="s">{turn.latency:.1f}s</td>'
                f'<td class="f">{html.escape(flags) or "—"}</td>'
                f'<td class="v"><label><input type="radio" name="v{name}-{turn.index}" value="ok"> OK</label>'
                f'<label><input type="radio" name="v{name}-{turn.index}" value="fix"> Fix</label>'
                f'<input class="note" type="text" placeholder="note"></td>'
                "</tr>"
            )
        blocks.append(
            f'<section><h2>{html.escape(name)}</h2>'
            "<table><thead><tr><th>#</th><th>Bong</th><th>Muffin</th><th>s</th>"
            "<th>flags</th><th>review</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width initial-scale=1">
<title>Muffin multi-turn eval</title>
<style>
 :root {{ --ink:#1c1c1e; --muted:#6b6b70; --line:#e3e3e8; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; padding:26px 18px 90px; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#fafafc; }}
 main {{ max-width:1180px; margin:0 auto; }}
 h1 {{ font-size:25px; margin:0 0 4px; }}
 .sub {{ color:var(--muted); margin:0 0 16px; }}
 .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin:0 0 20px; }}
 .card {{ background:#fff; border:1px solid var(--line); border-radius:11px; padding:12px 14px; }}
 .card b {{ display:block; font-size:21px; }}
 .card span {{ color:var(--muted); font-size:12px; }}
 .bar {{ position:sticky; top:0; z-index:5; background:#fafafc; padding:10px 0; border-bottom:1px solid var(--line); margin:0 0 16px; }}
 .bar button {{ font:inherit; font-size:13px; padding:6px 11px; border:1px solid var(--line); border-radius:7px; background:#fff; cursor:pointer; }}
 section {{ margin:0 0 26px; }}
 h2 {{ font-size:16px; margin:0 0 8px; }}
 table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
 th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); padding:8px 10px; border-bottom:1px solid var(--line); background:#f4f4f7; }}
 td {{ padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; font-size:14px; }}
 tr:last-child td {{ border-bottom:0; }}
 .n {{ width:32px; color:var(--muted); font-size:12px; }}
 .q {{ width:20%; font-weight:500; }}
 .a {{ width:34%; white-space:pre-wrap; }}
 .s {{ width:56px; color:var(--muted); font-size:12px; }}
 .f {{ width:15%; color:#a33; font-size:12px; }}
 .v {{ width:22%; }}
 .v label {{ font-size:12px; margin-right:8px; white-space:nowrap; }}
 .note {{ width:100%; margin-top:4px; font:inherit; font-size:12px; padding:5px 7px; border:1px solid var(--line); border-radius:6px; }}
 .v input[type=radio] {{ vertical-align:middle; }}
 @media print {{ body {{ background:#fff; padding:0; }} .bar, .v {{ display:none; }} }}
</style></head><body><main>
<h1>Muffin multi-turn eval</h1>
<p class="sub">Scripted dialogs against the live model, with a throwaway store.
Mark OK / Fix and add a note, then press Copy review.</p>
<div class="cards">{cards}</div>
<div class="bar"><button onclick="copyReview()">Copy review</button>
<span id="count" style="color:var(--muted);font-size:13px"></span></div>
{''.join(blocks)}
</main>
<script>
const rows=[...document.querySelectorAll('tbody tr')];
function refresh(){{
  let n=0;
  rows.forEach(tr=>{{
    const picked=tr.querySelector('input[type=radio]:checked');
    if(picked) n++;
  }});
  document.getElementById('count').textContent=n+' of '+rows.length+' reviewed';
}}
rows.forEach(tr=>tr.querySelectorAll('input').forEach(i=>i.oninput=refresh));
refresh();
function copyReview(){{
  const out=rows.map(tr=>{{
    const picked=tr.querySelector('input[type=radio]:checked');
    if(!picked) return null;
    return {{
      dialog: tr.closest('section').querySelector('h2').textContent,
      turn: tr.querySelector('.n').textContent,
      bong: tr.querySelector('.q').textContent,
      muffin: tr.querySelector('.a').textContent,
      verdict: picked.value,
      note: tr.querySelector('.note').value
    }};
  }}).filter(Boolean);
  navigator.clipboard.writeText(JSON.stringify(out,null,1)).then(()=>{{
    const b=document.querySelector('.bar button'); b.textContent='Copied '+out.length;
    setTimeout(()=>b.textContent='Copy review',1500);
  }});
}}
</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="also score with the LLM")
    parser.add_argument("--limit", type=int, default=0, help="turns per dialog")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    env = dict(os.environ)
    results = run(env, limit=args.limit)

    if args.judge:
        directory = Path(tempfile.mkdtemp(prefix="muffin-eval-judge-"))
        judge_model = make_muffin(directory, env, build_retriever(env))
        for name, data in results.items():
            data["judge"] = judge(judge_model, data["turns"])

    stats = summarise(results)
    report = ROOT / "data" / "EVAL_REPORT.md"
    review = ROOT / "data" / "EVAL_REVIEW.html"
    report.write_text(report_markdown(results, stats), encoding="utf-8")
    review.write_text(report_html(results, stats), encoding="utf-8")
    print(f"{stats['turns']} turns -> {report} and {review}")
    print(
        f"p50 {stats['p50']:.1f}s p95 {stats['p95']:.1f}s "
        f"style {stats['style_issues']} fallback {stats['fallbacks']} errors {stats['errors']}"
    )


if __name__ == "__main__":
    main()
