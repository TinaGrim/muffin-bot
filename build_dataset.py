#!/usr/bin/env python3
"""Turn a Telegram HTML export of the Muffin chat into clean training data.

Outputs (in ./data):
  pairs.jsonl        one user -> Muffin exchange per line
  sft_train.jsonl    95% split, chat-format samples with context
  sft_val.jsonl      5% split
  stats.json         dataset statistics
  style_report.md    style summary used to write the persona prompt
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}

URL_ONLY = re.compile(r"^https?://\S+$", re.I)
WHITESPACE = re.compile(r"[ \t\u00a0]+")
NEWLINES = re.compile(r"\n{2,}")

SYSTEM_PROMPT = (Path(__file__).parent / "persona.md").read_text(encoding="utf-8").strip()


@dataclass
class Message:
    msg_id: int
    ts: datetime
    sender: str
    text: str
    has_voice: bool = False
    has_sticker: bool = False
    has_media: bool = False
    is_reply: bool = False


@dataclass
class Turn:
    sender: str
    ts: datetime
    text: str
    ids: list[int] = field(default_factory=list)
    has_voice: bool = False
    has_sticker: bool = False
    has_media: bool = False

    @property
    def is_command(self) -> bool:
        return self.text.lstrip().startswith("/")


def page_number(path: Path) -> int:
    m = re.fullmatch(r"messages(\d*)\.html", path.name)
    return int(m.group(1)) if m and m.group(1) else 0


def clean_text(node) -> str:
    for br in node.find_all("br"):
        br.replace_with("\n")
    for img in node.find_all("img"):
        img.replace_with(img.get("alt") or "")
    text = node.get_text()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE.sub(" ", text)
    text = NEWLINES.sub("\n", text)
    return text.strip()


def parse_date(raw: str) -> datetime | None:
    # "21 December 2024, 17:48:05"
    m = re.match(r"(\d{1,2}) (\w+) (\d{4}), (\d{2}):(\d{2}):(\d{2})", raw or "")
    if not m:
        return None
    day, month_name, year, hh, mm, ss = m.groups()
    month = MONTHS.get(month_name)
    if not month:
        return None
    return datetime(int(year), month, int(day), int(hh), int(mm), int(ss))


def parse_page(path: Path, previous_sender: str | None):
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    messages: list[Message] = []

    for div in soup.select("div.message.default.clearfix"):
        classes = div.get("class") or []
        if "service" in classes:
            continue
        body = div.find("div", class_="body", recursive=False)
        if body is None:
            continue
        msg_id = int(re.sub(r"\D", "", div.get("id", "message0")) or 0)

        name_div = body.find("div", class_="from_name", recursive=False)
        if name_div is not None:
            sender = name_div.get_text(" ", strip=True)
            previous_sender = sender
        else:
            sender = previous_sender or ""

        date_div = body.find("div", class_="date", recursive=False)
        ts = parse_date(date_div.get("title", "") if date_div else "")
        if ts is None:
            continue

        text_div = body.find("div", class_="text", recursive=False)
        text = clean_text(text_div) if text_div else ""

        messages.append(
            Message(
                msg_id=msg_id,
                ts=ts,
                sender=sender,
                text=text,
                has_voice=body.find("a", class_="media_voice_message") is not None,
                has_sticker=body.find("a", class_="sticker_wrap") is not None,
                has_media=(
                    body.find("div", class_="media_wrap") is not None
                    or body.find("a", class_="media_photo") is not None
                    or body.find("a", class_="media_video") is not None
                ),
                is_reply=body.find("div", class_="reply_to") is not None,
            )
        )

    return messages, previous_sender


def merge_turns(messages: list[Message], gap_minutes: int = 15) -> list[Turn]:
    days = 60 * 60 * 24
    turns: list[Turn] = []
    for msg in messages:
        if not msg.sender:
            continue
        if turns:
            last = turns[-1]
            gap = (msg.ts - last.ts).total_seconds()
            if last.sender == msg.sender and 0 <= gap <= gap_minutes * 60:
                last.text = (last.text + "\n" + msg.text).strip()
                last.ids.append(msg.msg_id)
                last.has_voice |= msg.has_voice
                last.has_sticker |= msg.has_sticker
                last.has_media |= msg.has_media
                continue
        turns.append(
            Turn(
                sender=msg.sender,
                ts=msg.ts,
                text=msg.text,
                ids=[msg.msg_id],
                has_voice=msg.has_voice,
                has_sticker=msg.has_sticker,
                has_media=msg.has_media,
            )
        )
    return turns


def detect_bot_name(turns: list[Turn], export: Path) -> str:
    """The chat is named after the bot; use the export header, not message counts."""
    senders = {t.sender for t in turns}
    header = export / "messages.html"
    if header.exists():
        soup = BeautifulSoup(header.read_text(encoding="utf-8", errors="ignore"), "lxml")
        title = soup.find("title")
        chat_name = title.get_text(strip=True) if title else ""
        for sender in senders:
            if chat_name and (sender == chat_name or chat_name in sender):
                return sender
    for sender in senders:
        if "bot" in sender.lower():
            return sender
    return collections.Counter(t.sender for t in turns).most_common(1)[0][0]


def normalize(text: str) -> str:
    text = text.replace("\u200b", "").strip()
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def is_usable_user(text: str) -> bool:
    if not text or text.startswith("/"):
        return False
    if URL_ONLY.match(text):
        return False
    if len(text) > 500:
        return False
    return True


def is_usable_reply(text: str) -> bool:
    if not text or text.startswith("/"):
        return False
    if URL_ONLY.match(text):
        return False
    if len(text) > 600:
        return False
    return True


def emoji_count(text: str) -> int:
    return len(re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", text))


def khmer_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    khmer = sum(1 for c in letters if "\u1780" <= c <= "\u17ff")
    return khmer / len(letters)


def build(export: Path, out: Path, bot_name: str | None, seed: int = 7) -> dict:
    pages = sorted(export.glob("messages*.html"), key=page_number)
    if not pages:
        raise SystemExit(f"No messages*.html found in {export}")

    all_messages: list[Message] = []
    previous_sender = None
    for page in pages:
        messages, previous_sender = parse_page(page, previous_sender)
        all_messages.extend(messages)

    turns = merge_turns(all_messages)
    if not turns:
        raise SystemExit("No messages parsed.")

    bot = bot_name or detect_bot_name(turns, export)
    owner = collections.Counter(
        t.sender for t in turns if t.sender != bot
    ).most_common(1)[0][0]

    turns = [t for t in turns if t.sender in (bot, owner)]

    pairs = []
    dropped = collections.Counter()
    for i, turn in enumerate(turns):
        if turn.sender != owner:
            continue
        if i + 1 >= len(turns) or turns[i + 1].sender != bot:
            dropped["no_reply_follows"] += 1
            continue
        reply_turn = turns[i + 1]
        user = normalize(turn.text)
        reply = normalize(reply_turn.text)
        if not is_usable_user(user):
            dropped["bad_user"] += 1
            continue
        if not is_usable_reply(reply):
            dropped["bad_reply"] += 1
            continue
        if user.casefold() == reply.casefold():
            dropped["echo"] += 1
            continue
        pairs.append(
            {
                "id": turn.ids[0],
                "date": turn.ts.isoformat(timespec="minutes"),
                "user": user,
                "reply": reply,
                "reply_voice": reply_turn.has_voice,
                "reply_sticker": reply_turn.has_sticker,
                "context": [],
            }
        )

    # Attach up to 4 earlier exchanges as context.
    by_pair = {p["id"]: p for p in pairs}
    pair_index = {p["id"]: i for i, p in enumerate(pairs)}
    for p in pairs:
        idx = pair_index[p["id"]]
        ctx = []
        j = idx - 1
        while j >= 0 and len(ctx) < 4:
            prev = pairs[j]
            # stop at a large time gap (new session)
            if (
                datetime.fromisoformat(p["date"]) - datetime.fromisoformat(prev["date"])
            ).total_seconds() > 6 * 3600:
                break
            ctx.append({"user": prev["user"], "reply": prev["reply"]})
            j -= 1
        ctx.reverse()
        p["context"] = ctx

    # Deduplicate exact exchanges, keep the first occurrence.
    seen = set()
    unique = []
    for p in pairs:
        key = (p["user"].casefold(), p["reply"].casefold())
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        unique.append(p)
    pairs = unique

    random.Random(seed).shuffle(pairs)
    pairs.sort(key=lambda p: p["date"])

    def to_sample(p: dict) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for c in p["context"]:
            messages.append({"role": "user", "content": c["user"]})
            messages.append({"role": "assistant", "content": c["reply"]})
        messages.append({"role": "user", "content": p["user"]})
        messages.append({"role": "assistant", "content": p["reply"]})
        return {"messages": messages, "id": p["id"], "date": p["date"]}

    split = max(1, int(len(pairs) * 0.95))
    train, val = pairs[:split], pairs[split:]

    out.mkdir(parents=True, exist_ok=True)
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with (out / "sft_train.jsonl").open("w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(to_sample(p), ensure_ascii=False) + "\n")
    with (out / "sft_val.jsonl").open("w", encoding="utf-8") as f:
        for p in val:
            f.write(json.dumps(to_sample(p), ensure_ascii=False) + "\n")

    replies = [p["reply"] for p in pairs]
    lengths = sorted(len(r) for r in replies)
    user_lengths = sorted(len(p["user"]) for p in pairs)
    top_messages = collections.Counter(replies).most_common(20)
    words = collections.Counter(
        w.lower()
        for r in replies
        for w in re.findall(r"[A-Za-z']{2,}", r)
    )

    stats = {
        "export": str(export),
        "pages": len(pages),
        "raw_messages": len(all_messages),
        "turns": len(turns),
        "bot": bot,
        "owner": owner,
        "pairs": len(pairs),
        "train": len(train),
        "val": len(val),
        "dropped": dict(dropped),
        "voice_replies": sum(1 for p in pairs if p["reply_voice"]),
        "sticker_replies": sum(1 for p in pairs if p["reply_sticker"]),
        "reply_chars": {
            "avg": round(sum(lengths) / len(lengths), 1),
            "median": lengths[len(lengths) // 2],
            "p90": lengths[int(len(lengths) * 0.9)],
            "max": lengths[-1],
        },
        "user_chars": {
            "avg": round(sum(user_lengths) / len(user_lengths), 1),
            "median": user_lengths[len(user_lengths) // 2],
            "p90": user_lengths[int(len(user_lengths) * 0.9)],
        },
        "emoji_per_reply": round(
            sum(emoji_count(r) for r in replies) / len(replies), 2
        ),
        "khmer_share_of_letters": round(
            sum(khmer_ratio(r) for r in replies) / len(replies), 3
        ),
        "top_replies": top_messages,
        "top_words": words.most_common(40),
        "first_date": pairs[0]["date"] if pairs else None,
        "last_date": pairs[-1]["date"] if pairs else None,
    }
    (out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        f"# {bot} — style report",
        "",
        f"- Owner: {owner}",
        f"- Exchanges: {len(pairs):,} (train {len(train):,} / val {len(val):,})",
        f"- Date range: {stats['first_date']} -> {stats['last_date']}",
        f"- Reply length: avg {stats['reply_chars']['avg']}, median {stats['reply_chars']['median']}, p90 {stats['reply_chars']['p90']}",
        f"- User length: avg {stats['user_chars']['avg']}, median {stats['user_chars']['median']}",
        f"- Emoji per reply: {stats['emoji_per_reply']}",
        f"- Khmer share of letters: {stats['khmer_share_of_letters']:.1%}",
        f"- Voice-note replies skipped: {stats['voice_replies']:,}",
        "",
        "## Most frequent replies",
        "",
    ]
    for text, count in top_messages:
        report.append(f"- {count:5d}  {text}")
    report += ["", "## Most frequent English words", ""]
    report.append(", ".join(f"{w} ({c})" for w, c in words.most_common(30)))
    report += ["", "## Sample exchanges", ""]
    for p in pairs[:: max(1, len(pairs) // 20)][:20]:
        report.append(f"- **Bong:** {p['user']}")
        report.append(f"  **Muffin:** {p['reply']}")
    (out / "style_report.md").write_text("\n".join(report), encoding="utf-8")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "export",
        nargs="?",
        default="/Users/grim/Downloads/ChatExport_Muffin ( pitite bot )_3",
    )
    parser.add_argument("--out", default="data")
    parser.add_argument("--bot-name", default=None)
    args = parser.parse_args()

    stats = build(Path(args.export), Path(args.out), args.bot_name)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
