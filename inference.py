#!/usr/bin/env python3

"""Baseline inference script for Email Research Env."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR.parent))

from openai import OpenAI

from my_env import EmailAction, MyEnv


TASKS = json.loads((ROOT_DIR / "data" / "tasks.json").read_text())
BENCHMARK = "email-research-env"
IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME", "").strip()
ENV_URL = os.environ.get("ENV_URL", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "z-ai/glm-5").strip()
INFERENCE_LOG_PATH = os.environ.get("INFERENCE_LOG_PATH", "")
API_KEY = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
SEED = int(os.environ.get("SEED", "42"))
MAX_STEPS_BUFFER = 2
MODEL_MAX_RETRIES = 3
TEMPERATURE = 0.0
MAX_TOKENS = 300
LOG_FILE = None
DEBUG_INFERENCE = os.environ.get("DEBUG_INFERENCE", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


def emit_log_line(line: str) -> None:
    print(line, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.write(f"{line}\n")
        LOG_FILE.flush()


def emit_debug_line(line: str) -> None:
    if DEBUG_INFERENCE:
        print(line, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    emit_log_line(f"[START] task={task} env={env} model={model}")


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    error_value = (error or "null").replace("\n", " ").replace("\r", "")[:200]
    emit_log_line(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}"
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    emit_log_line(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} "
        f"rewards={rewards_str}"
    )


async def create_env() -> MyEnv:
    if ENV_URL:
        env = MyEnv(base_url=ENV_URL)
        await env.connect()
        return env
    if not IMAGE_NAME:
        raise RuntimeError("missing_local_image_name")
    return await MyEnv.from_docker_image(IMAGE_NAME)


SYSTEM_PROMPT = """You are an email research agent operating in an OpenEnv benchmark.

You must answer with exactly one valid JSON object on each turn.

Output requirements:
- Output raw JSON only.
- Do not use markdown.
- Do not wrap the JSON in code fences.
- Do not add explanations before or after the JSON.
- Do not return an empty response.
- If your output is not valid JSON, the action will fail.
- Use double-quoted JSON strings.
- Do not use comments, trailing commas, or Python values like None/True/False.
- Omit fields that are not relevant to the chosen action.

Available actions:
1. search_emails
   Use this to retrieve candidate emails by keyword and optional metadata filters.
   {"action_type":"search_emails","query":"...","top_k":5,"sent_after":"YYYY-MM-DD","sent_before":"YYYY-MM-DD","sender":"name@company.com"}

2. read_email
   Use this to inspect one specific email returned by search.
   {"action_type":"read_email","email_id":"..."}

3. return_final_answer
   Use this only when you have enough evidence to answer the question.
   {"action_type":"return_final_answer","answer":"...","cited_email_ids":["id1","id2"]}

Output schema:
- `action_type` is required and must be one of:
  - `"search_emails"`
  - `"read_email"`
  - `"return_final_answer"`
- If `action_type` is `"search_emails"`:
  - required: `query`
  - optional: `top_k`, `sent_after`, `sent_before`, `sender`
- If `action_type` is `"read_email"`:
  - required: `email_id`
- If `action_type` is `"return_final_answer"`:
  - required: `answer`
  - optional but recommended: `cited_email_ids`

Action-specific requirements:
- `search_emails` must include a non-empty `query`.
- `read_email` must include a specific `email_id` returned by search.
- `return_final_answer` must include a non-empty `answer`.
- `return_final_answer` should include the supporting `cited_email_ids`.
- If you do not know the answer yet, do not use `return_final_answer`; keep searching or read another email.

How to use the tools well:
- `search_emails` is for finding candidate email IDs.
- `read_email` is for gathering evidence from a candidate email.
- `return_final_answer` is for submitting a grounded answer with citations.
- Search first when you do not yet know which email is relevant.
- Read an email when search gives you plausible candidates.
- Search again only when you need a better query or a concrete refinement clue from evidence you already found.
- Use sender/date filters only when they help narrow a search based on a clue.
- Avoid repeating the same search with tiny wording changes unless you learned something new.
- Cite the email IDs that support your answer.
- Follow the requested answer format exactly.

Examples:

Example 1: direct factual lookup
Question: "Who owns the launch plan?"
Good sequence:
{"action_type":"search_emails","query":"launch owner","top_k":5}
{"action_type":"read_email","email_id":"email_17"}
{"action_type":"return_final_answer","answer":"Person A","cited_email_ids":["email_17"]}

Example 2: refinement after a clue
Question: "Who approved the pricing exception?"
Good sequence:
{"action_type":"search_emails","query":"pricing exception","top_k":5}
{"action_type":"read_email","email_id":"email_08"}
If that email reveals the clue "Person B", refine:
{"action_type":"search_emails","query":"pricing exception Person B","top_k":5}
{"action_type":"read_email","email_id":"email_12"}
{"action_type":"return_final_answer","answer":"Person B","cited_email_ids":["email_08","email_12"]}

Example 3: what not to do
Bad sequence:
{"action_type":"search_emails","query":"pricing exception approval","top_k":5}
{"action_type":"search_emails","query":"pricing exception approval update","top_k":5}
{"action_type":"search_emails","query":"pricing exception approval latest","top_k":5}

When search results already contain plausible candidates, open one with `read_email` instead of endlessly reformulating the search.

Valid examples:
{"action_type":"search_emails","query":"launch owner","top_k":5}
{"action_type":"read_email","email_id":"email_17"}
{"action_type":"return_final_answer","answer":"Person A","cited_email_ids":["email_17"]}

Invalid examples:
{"action":"search_emails","query":"launch owner"}
{"action_type":"return_final_answer","answer":"","cited_email_ids":[]}
```json {"action_type":"read_email","email_id":"email_17"} ```
"""


def build_user_prompt(observation: Any, history: list[str]) -> str:
    lines = [
        f"Question: {observation.question}",
        f"Steps used: {observation.steps_used}",
        f"Steps remaining: {observation.steps_remaining}",
    ]
    if observation.tool_message:
        lines.append(f"Tool message: {observation.tool_message}")
    if observation.search_results:
        lines.append("Search results:")
        for result in observation.search_results:
            lines.append(
                f"- {result.email_id} | {result.sent_at} | {result.sender} | "
                f"{result.subject} | {result.snippet}"
            )
    if observation.email:
        email = observation.email
        lines.extend(
            [
                "Opened email:",
                f"- ID: {email.email_id}",
                f"- Subject: {email.subject}",
                f"- Sender: {email.sender}",
                f"- Recipients: {', '.join(email.recipients)}",
                f"- Date: {email.sent_at}",
                f"- Body: {email.body}",
            ]
        )
    if history:
        lines.append("Recent actions:")
        lines.extend(f"- {item}" for item in history[-5:])
    return "\n".join(lines)


def parse_action(text: str) -> dict[str, Any] | None:
    block_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    candidates = [block_match.group(1)] if block_match else []
    candidates.extend([text])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def format_action(action: dict[str, Any]) -> str:
    action_type = action.get("action_type", "unknown")
    if action_type == "search_emails":
        return (
            f"search_emails(query={action.get('query','')}, sender={action.get('sender','')}, "
            f"sent_after={action.get('sent_after','')}, sent_before={action.get('sent_before','')})"
        )
    if action_type == "read_email":
        return f"read_email(email_id={action.get('email_id','')})"
    if action_type == "return_final_answer":
        return (
            f"return_final_answer(answer={action.get('answer','')}, "
            f"cited_email_ids={action.get('cited_email_ids',[])})"
        )
    return json.dumps(action)


def get_model_action(
    llm_client: OpenAI, observation: Any, history: list[str]
) -> tuple[dict[str, Any], str | None]:
    prompt = build_user_prompt(observation, history)
    text = ""
    last_error: str | None = None
    for attempt in range(1, MODEL_MAX_RETRIES + 1):
        try:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                seed=SEED,
            )
            text = response.choices[0].message.content or ""
            last_error = None
            break
        except Exception as exc:
            last_error = f"model_request_failed:{type(exc).__name__}"
            if attempt == MODEL_MAX_RETRIES:
                return (
                    {"action_type": "return_final_answer", "answer": "", "cited_email_ids": []},
                    last_error,
                )
            time.sleep(2 ** (attempt - 1))

    action_dict = parse_action(text)
    if action_dict is None:
        raw_preview = text.strip().replace("\n", "\\n")[:500]
        emit_debug_line(f"[DEBUG] parse_model_action_failed raw={raw_preview}")
        return (
            {"action_type": "return_final_answer", "answer": "", "cited_email_ids": []},
            "parse_model_action_failed",
        )
    try:
        EmailAction.model_validate(action_dict)
        return action_dict, last_error
    except Exception as exc:
        raw_preview = text.strip().replace("\n", "\\n")[:500]
        emit_debug_line(f"[DEBUG] invalid_model_action raw={raw_preview}")
        return (
            {"action_type": "return_final_answer", "answer": "", "cited_email_ids": []},
            f"invalid_model_action:{exc}",
        )


async def run_task(llm_client: OpenAI, task: dict[str, Any]) -> dict[str, Any]:
    rewards: list[float] = []
    history: list[str] = []
    final_score = 0.0
    success = False
    env: MyEnv | None = None
    log_start(task=task["task_id"], env=BENCHMARK, model=MODEL_NAME)
    try:
        env = await create_env()
        result = await env.reset(task_id=task["task_id"])
        observation = result.observation
        max_steps = task["max_steps"] + MAX_STEPS_BUFFER

        for step_number in range(1, max_steps + 1):
            action_dict, error = await asyncio.to_thread(
                get_model_action,
                llm_client,
                observation,
                history,
            )
            action = EmailAction.model_validate(action_dict)

            step_result = await env.step(action)
            observation = step_result.observation
            reward = float(step_result.reward or 0.0)
            rewards.append(reward)
            history.append(format_action(action_dict))
            env_error = None
            if getattr(observation, "tool_message", "").startswith("Email '") and "not found" in getattr(
                observation, "tool_message", ""
            ):
                env_error = observation.tool_message
            log_step(
                step=step_number,
                action=format_action(action_dict),
                reward=reward,
                done=step_result.done,
                error=env_error or error,
            )
            if step_result.done:
                final_score = reward
                success = final_score > 0.5
                break
        else:
            final_score = rewards[-1] if rewards else 0.0
    except Exception as exc:
        emit_debug_line(f"[DEBUG] task runtime failed ({task['task_id']}): {exc}")
        final_score = 0.0
        success = False
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as exc:
                emit_debug_line(f"[DEBUG] env.close() error: {exc}")
        log_end(success=success, steps=len(rewards), score=final_score, rewards=rewards)

    return {
        "task_id": task["task_id"],
        "success": success,
        "score": final_score,
        "steps": len(rewards),
    }


async def main() -> int:
    global LOG_FILE
    if not MODEL_NAME:
        print("MODEL_NAME must be set", file=sys.stderr)
        return 1
    if not API_BASE_URL:
        print("API_BASE_URL must be set", file=sys.stderr)
        return 1
    if not API_KEY:
        print("HF_TOKEN must be set", file=sys.stderr)
        return 1
    if not ENV_URL and not IMAGE_NAME:
        print("LOCAL_IMAGE_NAME must be set when ENV_URL is not provided", file=sys.stderr)
        return 1
    if INFERENCE_LOG_PATH:
        log_path = Path(INFERENCE_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE = log_path.open("w", encoding="utf-8")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    try:
        for task in TASKS:
            await run_task(llm_client, task)
    finally:
        if LOG_FILE is not None:
            LOG_FILE.close()
            LOG_FILE = None
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
