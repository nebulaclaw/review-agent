"""Multi-pattern string matching (Aho–Corasick) for fast wordlist recall."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ACMatch:
    start: int
    end: int
    pattern: str
    category: str = ""


def _transition(goto: list[dict[str, int]], fail: list[int], s: int, ch: str) -> int:
    while s != 0 and ch not in goto[s]:
        s = fail[s]
    return goto[s].get(ch, 0)


class AhoCorasickAutomaton:
    """Unicode character–grained Aho–Corasick automaton (patterns and text are str)."""

    __slots__ = ("_goto", "_fail", "_out", "_built")

    def __init__(self) -> None:
        self._goto: list[dict[str, int]] = [{}]
        self._fail: list[int] = [0]
        self._out: list[list[tuple[str, str]]] = [[]]
        self._built = False

    def _new_state(self) -> int:
        self._goto.append({})
        self._fail.append(0)
        self._out.append([])
        return len(self._goto) - 1

    def add(self, pattern: str, category: str = "") -> None:
        if self._built:
            raise RuntimeError("automaton already built")
        if not pattern:
            return
        s = 0
        for ch in pattern:
            if ch not in self._goto[s]:
                self._goto[s][ch] = self._new_state()
            s = self._goto[s][ch]
        self._out[s].append((pattern, category))

    def build(self) -> None:
        if self._built:
            return
        q: deque[int] = deque()
        for _, u in self._goto[0].items():
            self._fail[u] = 0
            q.append(u)
        while q:
            s = q.popleft()
            for ch, u in self._goto[s].items():
                q.append(u)
                f = self._fail[s]
                while f != 0 and ch not in self._goto[f]:
                    f = self._fail[f]
                self._fail[u] = self._goto[f][ch] if ch in self._goto[f] else 0
        self._built = True

    def find_all(self, text: str) -> list[ACMatch]:
        if not self._built:
            self.build()
        goto, fail, outs = self._goto, self._fail, self._out
        res: list[ACMatch] = []
        s = 0
        for i, ch in enumerate(text):
            s = _transition(goto, fail, s, ch)
            t = s
            while True:
                for pat, cat in outs[t]:
                    end = i + 1
                    start = end - len(pat)
                    if start >= 0:
                        res.append(ACMatch(start=start, end=end, pattern=pat, category=cat))
                if t == 0:
                    break
                t = fail[t]
        return res


__all__ = ["ACMatch", "AhoCorasickAutomaton"]
