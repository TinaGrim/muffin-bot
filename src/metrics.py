"""Lightweight reply metrics: latency, provider use and style fixes."""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path

from storage import atomic_write_text

KEEP_LINES = 5000


class Metrics:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def record(
        self,
        chat_id: int,
        latency: float,
        provider: str,
        fallbacks: list[str] | None = None,
        style_issues: list[str] | None = None,
        reply: str = "",
    ) -> None:
        entry = {
            "ts": round(time.time(), 3),
            "chat_id": chat_id,
            "latency": round(max(0.0, latency), 2),
            "provider": provider,
            "fallbacks": list(fallbacks or []),
            "style_issues": list(style_issues or []),
            "reply_chars": len(reply),
        }
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            self._rotate_locked()

    def _rotate_locked(self) -> None:
        try:
            with self.path.open("rb") as handle:
                if sum(1 for _ in handle) <= KEEP_LINES:
                    return
            lines = self.path.read_text(encoding="utf-8").splitlines()
            atomic_write_text(self.path, "\n".join(lines[-KEEP_LINES // 2 :]) + "\n")
        except OSError:
            pass

    def summary(self, limit: int = 200) -> dict:
        rows: list[dict] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines[-limit:]:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        if not rows:
            return {"count": 0}
        latencies = sorted(float(row.get("latency", 0)) for row in rows)
        count = len(latencies)
        p50 = latencies[count // 2]
        p95 = latencies[min(count - 1, int(count * 0.95))]
        fallback = sum(1 for row in rows if row.get("fallbacks"))
        providers = Counter(row.get("provider", "?") for row in rows)
        styles = sum(1 for row in rows if row.get("style_issues"))
        return {
            "count": count,
            "p50": p50,
            "p95": p95,
            "fallback_rate": fallback / count,
            "style_fixes": styles,
            "providers": providers,
        }
