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
