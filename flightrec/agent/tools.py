"""Mock tools. Pure functions; all randomness is supplied by the caller via a recorded seed."""
from __future__ import annotations

import random
from typing import Any


def search(query: str, seed: float) -> dict:
    """Deterministic given (query, seed). Different seeds yield different results, so a
    live run (random seed) never reproduces without record/replay."""
    rng = random.Random(f"{query}|{seed}")
    pool = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    picks = rng.sample(pool, 3)
    results = [f"{query}: {p} ({rng.randint(1000, 9999)})" for p in picks]
    return {"query": query, "results": results}


def run_tool(name: str, args: dict) -> Any:
    if name == "search":
        return search(args["query"], args["seed"])
    raise ValueError(f"unknown tool: {name}")
