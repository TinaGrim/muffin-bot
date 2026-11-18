"""Character n-gram retrieval over real Muffin exchanges.

Used both for few-shot examples in the prompt and, when no LLM backend is
configured, as a nearest-neighbour "reply like last time" fallback.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
import time
from collections import Counter
from pathlib import Path

log = logging.getLogger("muffin")

NGRAM = 3
CLEAN = re.compile(r"[^0-9a-z\u1780-\u17ff\s']+")
SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower().replace("\u200b", "").replace("\u00a0", " ")
    text = CLEAN.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def ngrams(text: str) -> list[str]:
    s = f"  {normalize(text)}  "
    return [s[i : i + NGRAM] for i in range(len(s) - NGRAM + 1)]


class Retriever:
    def __init__(self, pairs: list[dict]):
        self.pairs = pairs
        self.doc_norms: list[float] = []
        self.postings: dict[str, list[tuple[int, float]]] = {}
        self._build()

    def _build(self) -> None:
        df: Counter[str] = Counter()
        doc_counts: list[Counter[str]] = []
        for pair in self.pairs:
            counts = Counter(set(ngrams(pair["user"])))
            doc_counts.append(counts)
            df.update(counts.keys())

        n = len(self.pairs)
        idf = {g: math.log((n + 1) / (c + 1)) + 1.0 for g, c in df.items()}

        for idx, counts in enumerate(doc_counts):
            weights = {
                g: (1.0 + math.log(c)) * idf[g] for g, c in counts.items()
            }
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            self.doc_norms.append(norm)
            for g, w in weights.items():
                self.postings.setdefault(g, []).append((idx, w))

    def search(self, query: str, k: int = 6) -> list[dict]:
        counts = Counter(ngrams(query))
        if not counts:
            return []
        qweights = {g: 1.0 + math.log(c) for g, c in counts.items()}
        qnorm = math.sqrt(sum(w * w for w in qweights.values())) or 1.0

        scores: dict[int, float] = {}
        for g, qw in qweights.items():
            for idx, dw in self.postings.get(g, ()):
                scores[idx] = scores.get(idx, 0.0) + qw * dw

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        results = []
        for idx, score in ranked[:k]:
            pair = self.pairs[idx]
            results.append(
                {
                    "score": round(score / (qnorm * self.doc_norms[idx]), 4),
                    "user": pair["user"],
                    "reply": pair["reply"],
                    "date": pair.get("date"),
                }
            )
        results.sort(key=lambda r: r["score"], reverse=True)
        return results


def load_pairs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ResilientRetriever:
    """Semantic search with an automatic n-gram fallback.

    Embeddings need Ollama. When it is stopped or a request fails, replies must
    still work: the character n-gram index is used instead of raising into the
    chat pipeline (which used to answer "Ort").
    """

    def __init__(self, semantic, fallback: Retriever):
        self.semantic = semantic
        self.fallback = fallback
        self.pairs = semantic.pairs
        self._last_warn = 0.0

    def search(self, query: str, k: int = 8, min_score: float = 0.0) -> list[dict]:
        try:
            return self.semantic.search(query, k=k, min_score=min_score)
        except Exception as exc:  # noqa: BLE001 - keep chatting without Ollama
            now = time.time()
            if now - self._last_warn > 300:
                self._last_warn = now
                log.warning(
                    "embedding search unavailable (%s); using n-grams", exc
                )
            return self.fallback.search(query, k=k)


class CachedRetriever:
    """Build once, reuse until pairs.jsonl changes.

    With ``embeddings=True`` the semantic index is used when it exists (bge-m3
    or multilingual-e5 through Ollama); otherwise the character n-gram retriever
    is used, so the bot still works with no embedding server.
    """

    def __init__(
        self,
        pairs_path: Path,
        cache_path: Path,
        embeddings: bool = False,
        embed_model: str = "bge-m3",
        embed_base: str = "http://localhost:11434",
        embed_path: Path | None = None,
    ):
        self.pairs_path = pairs_path
        self.cache_path = cache_path
        self.embeddings = embeddings
        self.embed_model = embed_model
        self.embed_base = embed_base
        self.embed_path = embed_path or cache_path.with_name("embeddings_bge_m3.npz")
        self.retriever = None

    def _semantic(self):
        from embeddings import OllamaEmbedder, SemanticIndex

        try:
            index = SemanticIndex.load(
                self.embed_path, OllamaEmbedder(self.embed_model, self.embed_base)
            )
        except Exception as exc:  # noqa: BLE001 - fall back to n-grams
            log.warning("embedding index unusable (%s); using n-grams", exc)
            return None
        if index is None:
            log.warning("no embedding index at %s; using n-grams", self.embed_path)
        return index

    def _ngram(self) -> Retriever:
        stamp = (
            self.pairs_path.stat().st_mtime,
            self.pairs_path.stat().st_size,
        )
        if self.cache_path.exists():
            try:
                with self.cache_path.open("rb") as f:
                    cached_stamp, retriever = pickle.load(f)
                if cached_stamp == stamp:
                    return retriever
            except Exception:
                pass

        retriever = Retriever(load_pairs(self.pairs_path))
        with self.cache_path.open("wb") as f:
            pickle.dump((stamp, retriever), f)
        return retriever

    def get(self):
        if self.retriever is not None:
            return self.retriever

        if self.embeddings:
            semantic = self._semantic()
            if semantic is not None:
                log.info("retrieval: %s over %d examples", self.embed_model, len(semantic.users))
                self.retriever = ResilientRetriever(semantic, self._ngram())
                return self.retriever

        self.retriever = self._ngram()
        return self.retriever
