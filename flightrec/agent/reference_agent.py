"""Sequential reference pipeline: planner -> worker_a / worker_b -> synthesizer."""
from __future__ import annotations

import json

from .. import boundaries as b

PLANNER = "planner"
SYNTH = "synthesizer"


def _plan(task: str) -> dict:
    prompt = (
        "You are a planner. Break the task into exactly two sub-questions. "
        'Reply ONLY with JSON: {"sub_questions": ["...", "..."]}.\n\nTask: ' + task
    )
    resp = b.llm([{"role": "user", "content": prompt}], agent_id=PLANNER)
    try:
        data = json.loads(resp["content"])
        subs = list(data["sub_questions"])[:2]
        if len(subs) != 2:
            raise ValueError
    except Exception:
        subs = [f"What is essential about: {task}?", f"What are the risks of: {task}?"]
    return {"sub_questions": subs}


def _work(agent_id: str, sub_question: str) -> str:
    req_id = b.new_uuid(agent_id=agent_id)
    ts = b.now(agent_id=agent_id)
    seed = b.rand(agent_id=agent_id)
    results = b.tool_call("search", {"query": sub_question, "seed": seed}, agent_id=agent_id)
    prompt = (
        f"request_id={req_id} ts={ts}\n"
        f"Using these search results, answer the question in one sentence.\n"
        f"Question: {sub_question}\nResults: {json.dumps(results['results'])}"
    )
    resp = b.llm([{"role": "user", "content": prompt}], agent_id=agent_id)
    return resp["content"]


def run_agent(task: str) -> dict:
    plan = _plan(task)
    sub_a, sub_b = plan["sub_questions"]

    b.agent_msg(PLANNER, "worker_a", {"sub_question": sub_a})
    ans_a = _work("worker_a", sub_a)
    b.agent_msg("worker_a", SYNTH, {"answer": ans_a})

    b.agent_msg(PLANNER, "worker_b", {"sub_question": sub_b})
    ans_b = _work("worker_b", sub_b)
    b.agent_msg("worker_b", SYNTH, {"answer": ans_b})

    prompt = (
        "Combine these two answers into a final response.\n"
        f"A: {ans_a}\nB: {ans_b}"
    )
    final = b.llm([{"role": "user", "content": prompt}], agent_id=SYNTH)["content"]
    return {"task": task, "plan": plan, "answers": {"worker_a": ans_a, "worker_b": ans_b},
            "final": final}
