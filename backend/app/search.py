"""Autosuggest over HPO term names and synonyms (pure Python, no framework).

Matching is on normalised text (lower case, punctuation -> space), so "tonic clonic" finds
"Bilateral tonic-clonic seizure". Ranking tiers (lower is better; synonym matches rank one tier
below the same match on the official name):
    0 exact | 2 prefix | 4 every query word starts a word | 6 substring
Ties: shorter text first, then name. One hit per term (its best match).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
MIN_QUERY_CHARS = 2


def normalise(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


@dataclass(frozen=True)
class Hit:
    id: str
    name: str
    matched_synonym: str | None = None


class SearchIndex:
    def __init__(self, terms: Iterable[tuple[str, str, list[str]]]) -> None:
        """terms: (id, official name, synonyms)."""
        self._entries: list[tuple[str, str, str, bool, str]] = []   # (normalised text, id, name, is_synonym, original)
        for tid, name, synonyms in terms:
            self._entries.append((normalise(name), tid, name, False, name))
            for syn in synonyms:
                n = normalise(syn)
                if n:
                    self._entries.append((n, tid, name, True, syn))

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _tier(text: str, q: str, tokens: list[str]) -> int | None:
        if text == q:
            return 0
        if text.startswith(q):
            return 2
        if all(t in text for t in tokens):                      # cheap pre-check before the word test
            words = text.split()
            if all(any(w.startswith(t) for w in words) for t in tokens):
                return 4
        if q in text:
            return 6
        return None

    def search(self, query: str, limit: int = 10) -> list[Hit]:
        q = normalise(query)
        if len(q) < MIN_QUERY_CHARS or limit < 1:
            return []
        tokens = q.split()
        best: dict[str, tuple[tuple[int, int, str], Hit]] = {}
        for text, tid, name, is_syn, original in self._entries:
            tier = self._tier(text, q, tokens)
            if tier is None:
                continue
            key = (tier + (1 if is_syn else 0), len(text), name)
            cur = best.get(tid)
            if cur is None or key < cur[0]:
                best[tid] = (key, Hit(tid, name, original if is_syn else None))
        return [h for _, h in sorted(best.values(), key=lambda kv: (kv[0], kv[1].id))[:limit]]
