#!/usr/bin/env python3
"""Turn the collected review feedback into a page a human can scan."""

from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

BADGE = {
    "ok": ("OK", "ok"),
    "fixed": ("Fixed by note", "fix"),
    "regenerated": ("Regenerated", "regen"),
    "pending": ("Not reviewed", "pending"),
}


def khmer_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z\u1780-\u17ff]", text)
    if not letters:
        return 0.0
    khmer = [c for c in letters if "\u1780" <= c <= "\u17ff"]
    return len(khmer) / len(letters)


def reply_of(raw: dict) -> str:
    parts = raw.get("parts")
    if parts:
        return "\n".join(parts)
    return str(raw.get("reply") or "")


def build() -> str:
    path = DATA / "sent_replies.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rows = sorted(records.values(), key=lambda r: r.get("ts", 0))

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["user"]].append(row)

    counts = Counter(row.get("status", "pending") for row in rows)
    unique = len(groups)

    duplicates = []
    khmer = []
    long_lines = []
    for user, attempts in groups.items():
        seen = Counter(reply_of(a).strip() for a in attempts if reply_of(a).strip())
        for text, count in seen.items():
            if count > 1:
                duplicates.append((user, text, count))
        for attempt in attempts:
            text = reply_of(attempt)
            if khmer_ratio(text) >= 0.8 and len(text) > 3:
                khmer.append((user, text, attempt.get("status")))
            if len(text) > 100:
                long_lines.append((user, text, attempt.get("status")))

    notes = [row for row in rows if row.get("note")]
    first_ok = sum(
        1
        for attempts in groups.values()
        if attempts[0].get("status") == "ok"
    )

    findings = [
        (
            "Regenerate repeated itself",
            f"{len(duplicates)} case(s) where the same line was sent more than "
            "once. The archive match wins before the model, so pressing "
            "Regenerate kept replaying it."
            if duplicates
            else "No duplicate answers in this batch.",
        ),
        (
            "Bubble spam",
            "A single message collected up to 8 bubbles because every "
            "Regenerate sent a new message instead of replacing the old one."
            if max((len(v) for v in groups.values()), default=0) >= 4
            else "No excessive regeneration chains.",
        ),
        (
            "Khmer script leaked in",
            f"{len(khmer)} answer(s) were almost pure Khmer, copied from real "
            "archive lines. The style should stay Khmerlish unless Bong asks "
            "for Khmer."
            if khmer
            else "All answers stayed in Khmerlish.",
        ),
        (
            "Long rambling lines",
            f"{len(long_lines)} answer(s) were over 100 characters, which reads "
            "like a pasted archive reply rather than a chat line."
            if long_lines
            else "Answers stayed short.",
        ),
        (
            "Notes worked",
            f"{len(notes)} note(s) were applied. Every corrected answer followed "
            "the note and replaced the old bubble."
            if notes
            else "No notes were sent.",
        ),
    ]

    cards = "".join(
        f'<div class="card"><b>{value}</b><span>{label}</span></div>'
        for value, label in [
            (len(rows), "replies reviewed"),
            (unique, "unique messages"),
            (counts["ok"], "accepted"),
            (counts["fixed"], "fixed by note"),
            (counts["regenerated"], "regenerated"),
            (f"{first_ok}/{unique}", "OK on first try"),
        ]
    )

    finding_html = "".join(
        f'<li><b>{html.escape(title)}</b><span>{html.escape(body)}</span></li>'
        for title, body in findings
    )

    items = []
    for user, attempts in sorted(groups.items(), key=lambda kv: kv[1][0].get("ts", 0)):
        final = next(
            (a for a in reversed(attempts) if a.get("status") in {"ok", "fixed"}),
            attempts[-1],
        )
        chips = "".join(
            f'<span class="badge {BADGE.get(a.get("status"), BADGE["pending"])[1]}">'
            f'{BADGE.get(a.get("status"), BADGE["pending"])[0]}</span>'
            for a in attempts
        )
        tries = []
        for index, attempt in enumerate(attempts, 1):
            text = reply_of(attempt)
            note = attempt.get("note") or ""
            corrected = attempt.get("corrected") or ""
            extra = ""
            if note:
                extra = f'<div class="note"><b>Note:</b> {html.escape(note)}</div>'
            elif (
                attempt.get("status") == "fixed"
                and corrected
                and corrected.strip() != text.strip()
            ):
                extra = (
                    '<div class="note"><b>Replaced with:</b> '
                    f"{html.escape(corrected)}</div>"
                )
            tries.append(
                f'<div class="try {BADGE.get(attempt.get("status"), BADGE["pending"])[1]}">'
                f'<span class="n">#{index}</span>'
                f'{BADGE.get(attempt.get("status"), BADGE["pending"])[0]}'
                f'<div class="text">{html.escape(text) or "<i>empty</i>"}</div>'
                f"{extra}</div>"
            )
        final_text = reply_of(final)
        items.append(
            '<article class="item">'
            f'<h2>Bong: “{html.escape(user)}”</h2>'
            f'<div class="chips">{chips}</div>'
            f'<div class="tries">{"".join(tries)}</div>'
            f'<div class="final"><span>Final</span>{html.escape(final_text)}</div>'
            "</article>"
        )

    duplicate_html = "".join(
        f'<li><code>{html.escape(text[:70])}</code> — {count}× '
        f"for “{html.escape(user)}”</li>"
        for user, text, count in duplicates
    )
    khmer_html = "".join(
        f'<li>“{html.escape(text[:90])}” — {html.escape(str(status))}</li>'
        for _user, text, status in khmer
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width initial-scale=1">
<title>Muffin feedback review</title>
<style>
 :root {{ --ink:#1c1c1e; --muted:#6b6b70; --line:#e3e3e8; --ok:#1f7a3a; --fix:#8a5a00; --regen:#a33; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; padding:28px 18px 80px; font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#fafafc; }}
 main {{ max-width:1000px; margin:0 auto; }}
 h1 {{ font-size:26px; margin:0 0 4px; }}
 .sub {{ color:var(--muted); margin:0 0 18px; }}
 .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin:0 0 22px; }}
 .card {{ background:#fff; border:1px solid var(--line); border-radius:11px; padding:12px 14px; }}
 .card b {{ display:block; font-size:22px; line-height:1.1; }}
 .card span {{ color:var(--muted); font-size:12px; }}
 h2.section {{ font-size:16px; margin:26px 0 10px; }}
 .findings {{ margin:0; padding:0; list-style:none; }}
 .findings li {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:11px 14px; margin:0 0 8px; }}
 .findings li b {{ display:block; }}
 .findings li span {{ color:var(--muted); font-size:13px; }}
 .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
 @media (max-width:760px) {{ .cols {{ grid-template-columns:1fr; }} }}
 .box {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px 15px; }}
 .box h3 {{ margin:0 0 6px; font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
 .box ul {{ margin:0; padding-left:18px; font-size:13px; }}
 .bar {{ position:sticky; top:0; z-index:5; background:#fafafc; padding:10px 0; border-bottom:1px solid var(--line); margin:24px 0 16px; display:flex; gap:8px; flex-wrap:wrap; }}
 .bar button {{ font:inherit; font-size:12px; padding:5px 10px; border:1px solid var(--line); border-radius:7px; background:#fff; cursor:pointer; }}
 .bar button.on {{ background:var(--ink); color:#fff; border-color:var(--ink); }}
 .item {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin:0 0 14px; }}
 .item h2 {{ font-size:16px; margin:0 0 8px; }}
 .chips {{ display:flex; gap:6px; flex-wrap:wrap; margin:0 0 8px; }}
 .badge {{ font-size:11px; padding:3px 8px; border-radius:999px; background:#f1f1f4; color:var(--muted); }}
 .badge.ok {{ background:#e9f7ee; color:var(--ok); }}
 .badge.fix {{ background:#fdf3df; color:var(--fix); }}
 .badge.regen {{ background:#fdecec; color:var(--regen); }}
 .try {{ border-left:3px solid var(--line); padding:6px 0 6px 10px; margin:0 0 8px; font-size:14px; }}
 .try.ok {{ border-color:#7fc297; }}
 .try.fix {{ border-color:#e0b566; }}
 .try.regen {{ border-color:#e3a1a1; }}
 .try .n {{ color:var(--muted); font-size:11px; margin-right:8px; }}
 .try .text {{ white-space:pre-wrap; margin-top:2px; }}
 .note {{ font-size:12px; color:var(--fix); background:#fdf8ec; border-radius:6px; padding:5px 8px; margin-top:5px; }}
 .final {{ margin-top:10px; padding:8px 10px; background:#f5f7fa; border-radius:8px; white-space:pre-wrap; font-size:14px; }}
 .final span {{ display:block; font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-bottom:2px; }}
 code {{ font-size:12px; }}
</style></head><body><main>
<h1>Muffin feedback review</h1>
<p class="sub">What Bong pressed on {len(rows)} bot replies. Raw data:
<code>data/sent_replies.json</code> and <code>data/feedback.jsonl</code>.</p>
<div class="cards">{cards}</div>

<h2 class="section">What the feedback says</h2>
<ul class="findings">{finding_html}</ul>

<div class="cols">
 <div class="box"><h3>Repeated answers</h3><ul>{duplicate_html or "<li>none</li>"}</ul></div>
 <div class="box"><h3>Khmer-script answers</h3><ul>{khmer_html or "<li>none</li>"}</ul></div>
</div>

<div class="bar">
 <button class="on" onclick="filt('all', this)">All</button>
 <button onclick="filt('ok', this)">Accepted</button>
 <button onclick="filt('fix', this)">Fixed</button>
 <button onclick="filt('regen', this)">Regenerated</button>
</div>
{''.join(items)}
</main>
<script>
function filt(kind, btn) {{
  document.querySelectorAll('.bar button').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  document.querySelectorAll('.item').forEach(item => {{
    const has = kind === 'all'
      || item.querySelector('.badge.' + kind)
      || item.querySelector('.try.' + kind);
    item.style.display = has ? '' : 'none';
  }});
}}
</script>
</body></html>
"""


def main() -> None:
    out = DATA / "FEEDBACK_REVIEW.html"
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
