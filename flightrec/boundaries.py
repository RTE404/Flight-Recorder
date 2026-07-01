"""The only sanctioned non-deterministic primitives. Agent code MUST use these."""
from __future__ import annotations

import os
import random
import time
import uuid
from typing import Any

from .interceptor import current

DEFAULT_MODEL = "groq/llama-3.1-8b-instant"


def _model() -> str:
    return os.environ.get("FLIGHTREC_MODEL", DEFAULT_MODEL)


def llm(messages: list, *, agent_id: str, **kwargs) -> dict:
    model = _model()
    request = {"model": model, "messages": messages, **kwargs}

    def live():
        current().guard_real_call()
        import litellm
        resp = litellm.completion(model=model, messages=messages, **kwargs)
        content = resp.choices[0].message.content or ""
        return {"role": "assistant", "content": content}

    return current().cross(agent_id, "llm_call", request, live)


def tool_call(name: str, args: dict, *, agent_id: str) -> Any:
    request = {"name": name, "args": args}

    def live():
        current().guard_real_call()
        from .agent import tools
        return tools.run_tool(name, args)

    return current().cross(agent_id, "tool_call", request, live)


def now(*, agent_id: str) -> float:
    def live():
        current().guard_real_call()
        return time.time()

    return current().cross(agent_id, "clock", {"op": "now"}, live)


def new_uuid(*, agent_id: str) -> str:
    def live():
        current().guard_real_call()
        return str(uuid.uuid4())

    return current().cross(agent_id, "random", {"op": "uuid"}, live)


def rand(*, agent_id: str) -> float:
    def live():
        current().guard_real_call()
        return random.random()

    return current().cross(agent_id, "random", {"op": "rand"}, live)


def agent_msg(from_agent: str, to_agent: str, payload: Any) -> Any:
    request = {"from": from_agent, "to": to_agent, "payload": payload}

    def live():
        current().guard_real_call()
        return payload

    return current().cross(from_agent, "agent_msg", request, live)
