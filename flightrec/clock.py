"""Lamport logical clock (concurrency-ready bookkeeping)."""
from __future__ import annotations


class LamportClock:
    def __init__(self) -> None:
        self.value = 0

    def tick(self) -> int:
        self.value += 1
        return self.value

    def update(self, other: int) -> int:
        self.value = max(self.value, other) + 1
        return self.value


class VectorClock:
    def __init__(self, agent_id: str, initial: dict | None = None):
        self.agent_id = agent_id
        self.v: dict[str, int] = dict(initial or {})

    def tick(self) -> dict:
        self.v[self.agent_id] = self.v.get(self.agent_id, 0) + 1
        return dict(self.v)

    def merge(self, other: dict) -> None:
        for k, val in other.items():
            self.v[k] = max(self.v.get(k, 0), val)

    def snapshot(self) -> dict:
        return dict(self.v)


def vc_rank(v: dict) -> int:
    return sum(v.values())


def happens_before(a: dict, b: dict) -> bool:
    keys = set(a) | set(b)
    le = all(a.get(k, 0) <= b.get(k, 0) for k in keys)
    return le and a != b


def concurrent(a: dict, b: dict) -> bool:
    return a != b and not happens_before(a, b) and not happens_before(b, a)
