"""Concurrent reference pipeline: planner -> worker_a / worker_b (parallel) -> synthesizer."""
from __future__ import annotations

import json
import threading

from .. import boundaries as b

PLANNER = "planner"
SYNTH = "synthesizer"
WORKER_IDS = ["worker_a", "worker_b"]


def _plan(task: str) -> list[str]:
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
    return subs


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


def _synthesize(task: str, answers: dict) -> str:
    prompt = (
        "Combine these two answers into a final response.\n"
        f"A: {answers['worker_a']}\nB: {answers['worker_b']}"
    )
    return b.llm([{"role": "user", "content": prompt}], agent_id=SYNTH)["content"]


def run_agent(task: str) -> dict:
    sub_questions = _plan(task)

    # Planner -> worker handoffs happen in the main thread, in fixed order, so
    # planner's own events are totally ordered and never race with each other.
    assignments = {wid: b.agent_msg(PLANNER, wid, sq)
                   for wid, sq in zip(WORKER_IDS, sub_questions)}

    answers: dict[str, str] = {}
    errors: dict[str, BaseException] = {}
    lock = threading.Lock()

    def worker_entry(wid: str) -> None:
        try:
            ans = _work(wid, assignments[wid])
            b.agent_msg(wid, SYNTH, ans)  # real join edge -> taints synth on fork
            with lock:
                answers[wid] = ans
        except BaseException as e:  # threads don't propagate exceptions through join()
            with lock:
                errors[wid] = e

    threads = [threading.Thread(target=worker_entry, args=(wid,)) for wid in WORKER_IDS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:  # re-raise deterministically, in WORKER_IDS order
        for wid in WORKER_IDS:
            if wid in errors:
                raise errors[wid]

    final = _synthesize(task, answers)
    return {"final": final, "answers": answers}
