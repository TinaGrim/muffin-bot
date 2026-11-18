"""Semantic retrieval over clean real exchanges, using a multilingual embedder.

The corpus is Khmerlish/Khmer, so an English-only encoder loses most of the
signal. bge-m3 (or multilingual-e5) is used through Ollama's local API.

The index is built once into an .npz file and then memory-mapped. Retrieval is
a single matrix multiply, so per-message cost is negligible next to the LLM
call.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import httpx
import numpy as np

log = logging.getLogger("muffin")

DEFAULT_MODEL = "bge-m3"
DEFAULT_BASE = "http://localhost:11434"
DIM_CACHE: dict[str, int] = {}

LINK = re.compile(r"https?://|www\.", re.I)
ONLY_PUNCT = re.compile(r"^[\W_]+$")


def is_clean(pair: dict) -> bool:
    """Keep examples that teach style, not links or one-off noise."""
    user = (pair.get("user") or "").strip()
    reply = (pair.get("reply") or "").strip()
    if not (2 <= len(user) <= 160):
        return False
    if not (2 <= len(reply) <= 200):
        return False
    if LINK.search(reply) or LINK.search(user):
        return False
    if reply.casefold() == user.casefold():
        return False
    if ONLY_PUNCT.match(reply):
        return False
    return True


class OllamaEmbedder:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE,
        batch: int = 64,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch = batch
        self.client = httpx.Client(timeout=timeout)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        if response.status_code == 404:
            # Older Ollama: one prompt per call.
            return [self._embed_one(t) for t in texts]
        response.raise_for_status()
        return response.json()["embeddings"]

    def _embed_one(self, text: str) -> list[float]:
        response = self.client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            chunk = texts[start : start + self.batch]
            vectors.extend(self._embed_batch(chunk))
            done = min(start + self.batch, len(texts))
            if done % 400 == 0 or done == len(texts):
                log.info("embedded %d/%d", done, len(texts))
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms


class SemanticIndex:
    """Top-k cosine search over the clean real exchanges."""

    def __init__(
        self,
        vectors: np.ndarray,
        users: list[str],
        replies: list[str],
        dates: list[str],
        embedder: OllamaEmbedder,
    ) -> None:
        self.vectors = vectors
        self.users = users
        self.replies = replies
        self.dates = dates
        self.embedder = embedder
        self.pairs = [
            {"user": u, "reply": r, "date": d}
            for u, r, d in zip(users, replies, dates)
        ]

    @classmethod
    def load(
        cls, path: Path, embedder: OllamaEmbedder
    ) -> "SemanticIndex | None":
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=False)
        return cls(
            data["vectors"],
            [str(x) for x in data["users"]],
            [str(x) for x in data["replies"]],
            [str(x) for x in data["dates"]],
            embedder,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            users=np.asarray(self.users, dtype=np.str_),
            replies=np.asarray(self.replies, dtype=np.str_),
            dates=np.asarray(self.dates, dtype=np.str_),
        )

    def search(
        self, query: str, k: int = 8, min_score: float = 0.0
    ) -> list[dict]:
        if not query.strip():
            return []
        vector = self.embedder.embed([query])[0]
        scores = self.vectors @ vector
        take = min(k, len(scores))
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        results = []
        for idx in top:
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append(
                {
                    "score": round(score, 4),
                    "user": self.users[idx],
                    "reply": self.replies[idx],
                    "date": self.dates[idx] or None,
                }
            )
        return results


def build_index(
    pairs_path: Path,
    out_path: Path,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE,
    limit: int | None = None,
) -> SemanticIndex:
    pairs = [
        json.loads(line)
        for line in pairs_path.open(encoding="utf-8")
        if line.strip()
    ]
    seen: set[tuple[str, str]] = set()
    clean: list[dict] = []
    for pair in pairs:
        if not is_clean(pair):
            continue
        key = (pair["user"].strip().casefold(), pair["reply"].strip().casefold())
        if key in seen:
            continue
        seen.add(key)
        clean.append(pair)
        if limit and len(clean) >= limit:
            break

    log.info("clean examples: %d of %d", len(clean), len(pairs))
    embedder = OllamaEmbedder(model=model, base_url=base_url)
    vectors = embedder.embed([p["user"] for p in clean])
    index = SemanticIndex(
        vectors,
        [p["user"] for p in clean],
        [p["reply"] for p in clean],
        [p.get("date") or "" for p in clean],
        embedder,
    )
    index.save(out_path)
    log.info("wrote %s (%s, %d x %d)", out_path, model, *vectors.shape)
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/pairs.jsonl")
    parser.add_argument("--out", default="data/embeddings_bge_m3.npz")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query", default=None, help="test a query after build")
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    pairs = Path(args.pairs)
    if not pairs.is_absolute():
        pairs = root / pairs
    if out.exists():
        index = SemanticIndex.load(out, OllamaEmbedder(args.model, args.base_url))
        if index is None:
            sys.exit("index missing")
        log.info("loaded %s", out)
    else:
        index = build_index(
            pairs,
            out,
            model=args.model,
            base_url=args.base_url,
            limit=args.limit,
        )

    if args.query:
        for hit in index.search(args.query, k=args.k):
            print(f"{hit['score']:.3f}  {hit['user'][:60]!r} -> {hit['reply'][:70]!r}")


if __name__ == "__main__":
    main()
