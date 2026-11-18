#!/usr/bin/env python3
"""Build a corrections sheet: only the lines Bong flagged, one box each.

Reads data/review_flags.json (the export from the review page), pairs each flag
with the section it came from, and writes data/QA_CORRECTIONS.html. Type the
correct Muffin reply in each box, then press "Copy corrections" and paste the
result back to me.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from life import DEFS  # noqa: E402
from make_review import SECTIONS  # noqa: E402


def build() -> list[dict]:
    flags = json.loads((ROOT / "data" / "review_flags.json").read_text())
    meta = []
    for title, activity, questions in SECTIONS:
        for _ in questions:
            meta.append((title, activity))
    rows = []
    for index, flag in enumerate(flags):
        if not flag.get("fix"):
            continue
        title, activity = meta[index]
        rows.append(
            {
                "index": index,
                "section": title,
                "where": DEFS[activity].label,
                "question": flag["bong"].strip(),
                "current": flag["muffin"].strip(),
                "correct": flag.get("note", "").strip(),
            }
        )
    return rows


def render(rows: list[dict]) -> str:
    sections: dict[str, list[dict]] = {}
    for row in rows:
        sections.setdefault(row["section"], []).append(row)
    body = []
    for title, items in sections.items():
        body.append(f"<section><h2>{html.escape(title)}</h2>")
        body.append(
            "<table><thead><tr>"
            '<th class="num">#</th><th class="q">Bong says</th>'
            '<th class="cur">Bot says now</th>'
            '<th class="fixcol">Correct answer (type here)</th>'
            "</tr></thead><tbody>"
        )
        for row in items:
            body.append(
                f'<tr data-i="{row["index"]}">'
                f'<td class="num">{row["index"] + 1}</td>'
                f'<td class="q">{html.escape(row["question"])}'
                f'<span class="where">{html.escape(row["where"])}</span></td>'
                f'<td class="cur">{html.escape(row["current"])}</td>'
                f'<td class="fixcol">'
                f'<input class="fixin" type="text" '
                f'placeholder="write the correct reply">'
                f'<button class="same" title="copy this answer to every row with the same message">same ↓</button>'
                f"</td></tr>"
            )
        body.append("</tbody></table></section>")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width initial-scale=1">
<title>Muffin - corrections</title>
<style>
 :root {{ --ink:#1c1c1e; --muted:#6b6b70; --line:#e3e3e8; --done:#eefbf0; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; padding:26px 18px 90px; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#fafafc; }}
 main {{ max-width:1080px; margin:0 auto; }}
 h1 {{ font-size:25px; margin:0 0 4px; }}
 .sub {{ color:var(--muted); margin:0 0 16px; max-width:70ch; }}
 .bar {{ position:sticky; top:0; z-index:5; background:#fafafc; padding:10px 0; border-bottom:1px solid var(--line); margin-bottom:16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
 .bar button {{ font:inherit; font-size:13px; padding:6px 11px; border:1px solid var(--line); border-radius:7px; background:#fff; cursor:pointer; }}
 .bar .count {{ color:var(--muted); font-size:13px; }}
 section {{ margin:0 0 26px; }}
 h2 {{ font-size:16px; margin:0 0 8px; }}
 table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
 th {{ text-align:left; font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); padding:8px 10px; border-bottom:1px solid var(--line); background:#f4f4f7; }}
 td {{ padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
 tr:last-child td {{ border-bottom:0; }}
 tr.filled td {{ background:var(--done); }}
 .num {{ width:36px; color:var(--muted); font-size:12px; }}
 .q {{ width:26%; font-weight:500; }}
 .q .where {{ display:block; font-weight:400; font-size:12px; color:var(--muted); margin-top:2px; }}
 .cur {{ width:24%; color:#a33; font-size:13px; }}
 .fixcol {{ width:44%; }}
 .fixin {{ width:calc(100% - 62px); font:inherit; font-size:14px; padding:7px 9px; border:1px solid var(--line); border-radius:7px; background:#fff; }}
 .fixin:focus {{ outline:2px solid #cfe3ff; border-color:#8fb8f0; }}
 .same {{ font:inherit; font-size:11px; padding:6px 7px; margin-left:6px; border:1px solid var(--line); border-radius:6px; background:#fff; cursor:pointer; color:var(--muted); vertical-align:middle; }}
 .same:hover {{ color:var(--ink); }}
 @media print {{ body {{ background:#fff; padding:0; }} .bar, .same {{ display:none; }} }}
</style></head><body><main>
<h1>Muffin - corrections</h1>
<p class="sub">Only the {len(rows)} lines you marked <b>Fix</b>. Type the reply Muffin should give.
"same ↓" copies your answer to every row with the same message. Everything you type is saved in this browser,
and <b>Copy corrections</b> gives me the whole list.</p>
<div class="bar">
 <span class="count" id="count">0 of {len(rows)} filled</span>
 <button id="copy" onclick="copyAll()">Copy corrections</button>
 <button onclick="hideFilled()">Hide filled rows</button>
 <button onclick="showAll()">Show all</button>
 <button onclick="clearAll()">Clear all</button>
</div>
{''.join(body)}
</main>
<script>
const rows=[...document.querySelectorAll('tbody tr')];
const KEY='muffin-corr-';
function refresh(){{
  let n=0;
  rows.forEach(tr=>{{
    const i=tr.dataset.i, inp=tr.querySelector('.fixin');
    inp.value=localStorage.getItem(KEY+i)||'';
    tr.classList.toggle('filled', !!inp.value.trim());
    if(inp.value.trim()) n++;
  }});
  document.getElementById('count').textContent=n+' of '+rows.length+' filled';
}}
rows.forEach(tr=>{{
  const i=tr.dataset.i, inp=tr.querySelector('.fixin');
  inp.oninput=()=>{{ localStorage.setItem(KEY+i, inp.value); tr.classList.toggle('filled', !!inp.value.trim()); refresh(); }};
  tr.querySelector('.same').onclick=()=>{{
    const msg=tr.querySelector('.q').firstChild.textContent.trim();
    const val=inp.value;
    if(!val) return;
    rows.forEach(other=>{{
      const m=other.querySelector('.q').firstChild.textContent.trim();
      if(other===tr||m!==msg) return;
      other.querySelector('.fixin').value=val;
      localStorage.setItem(KEY+other.dataset.i,val);
    }});
    refresh();
  }};
}});
refresh();
function hideFilled(){{ rows.forEach(tr=>{{ if(tr.classList.contains('filled')) tr.style.display='none'; }}); }}
function showAll(){{ rows.forEach(tr=>tr.style.display=''); }}
function clearAll(){{ rows.forEach(tr=>{{ localStorage.removeItem(KEY+tr.dataset.i); }}); refresh(); }}
function copyAll(){{
  const out=rows.filter(tr=>tr.querySelector('.fixin').value.trim()).map(tr=>({{
    section:tr.closest('section').querySelector('h2').textContent,
    where:tr.querySelector('.where').textContent,
    bong:tr.querySelector('.q').firstChild.textContent.trim(),
    current:tr.querySelector('.cur').textContent.trim(),
    correct:tr.querySelector('.fixin').value.trim()
  }}));
  navigator.clipboard.writeText(JSON.stringify(out,null,1)).then(()=>{{
    const b=document.getElementById('copy'); b.textContent='Copied '+out.length; setTimeout(()=>b.textContent='Copy corrections',1500);
  }});
}}
</script>
</body></html>
"""


def main() -> None:
    rows = build()
    out = ROOT / "data" / "QA_CORRECTIONS.html"
    out.write_text(render(rows), encoding="utf-8")
    print(f"{len(rows)} flagged lines -> {out}")


if __name__ == "__main__":
    main()
