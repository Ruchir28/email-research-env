# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Email research environment implementation."""

from __future__ import annotations

import math
import re
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

try:
    from ..models import (
        EmailAction,
        EmailObservation,
        ReadEmailPayload,
        RewardBreakdown,
        ReturnFinalAnswerPayload,
        SearchEmailsPayload,
    )
    from .inbox_repository import ensure_inbox_db, load_tasks, read_email, search_emails
except ImportError:
    from models import (
        EmailAction,
        EmailObservation,
        ReadEmailPayload,
        RewardBreakdown,
        ReturnFinalAnswerPayload,
        SearchEmailsPayload,
    )
    from server.inbox_repository import (
        ensure_inbox_db,
        load_tasks,
        read_email,
        search_emails,
    )


def _normalize_text(value: str) -> str:
    collapsed = re.sub(r"[^a-z0-9| -]+", " ", value.lower())
    return re.sub(r"\s+", " ", collapsed).strip()


class MyEnvironment(Environment):
    """A tool-use environment for answering questions over an email inbox."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _task_cursor: int = 0
    _cached_tasks: list[dict[str, Any]] | None = None

    def __init__(self):
        ensure_inbox_db()
        self._tasks = self._load_tasks()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: dict[str, Any] | None = None
        self._visited_email_ids: set[str] = set()
        self._repeated_reads = 0
        self._repeated_searches = 0
        self._search_signatures: set[tuple[Any, ...]] = set()

    @classmethod
    def _load_tasks(cls) -> list[dict[str, Any]]:
        if cls._cached_tasks is None:
            cls._cached_tasks = load_tasks()
        return cls._cached_tasks

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> EmailObservation:
        del seed, kwargs
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._visited_email_ids = set()
        self._repeated_reads = 0
        self._repeated_searches = 0
        self._search_signatures = set()

        if task_id:
            task = next((item for item in self._tasks if item["task_id"] == task_id), None)
            if task is None:
                raise ValueError(f"Unknown task_id: {task_id}")
        else:
            task = self._tasks[self.__class__._task_cursor % len(self._tasks)]
            self.__class__._task_cursor += 1

        self._current_task = task
        return EmailObservation(
            question=task["question"],
            tool_message=(
                "Search the inbox, read supporting emails, and submit a grounded final "
                "answer with cited email IDs."
            ),
            steps_used=0,
            steps_remaining=task["max_steps"],
            reward=0.0,
            done=False,
            metadata={
                "max_steps": task["max_steps"],
                "available_tools": [
                    "search_emails",
                    "read_email",
                    "return_final_answer",
                ],
            },
        )

    def step(
        self, action: EmailAction, timeout_s: float | None = None, **kwargs: Any
    ) -> EmailObservation:
        del timeout_s, kwargs
        if self._current_task is None:
            self.reset()

        assert self._current_task is not None
        self._state.step_count += 1
        payload = action.payload

        if isinstance(payload, SearchEmailsPayload):
            observation = self._handle_search(payload)
        elif isinstance(payload, ReadEmailPayload):
            observation = self._handle_read(payload)
        else:
            observation = self._handle_final_answer(payload)

        max_steps = self._current_task["max_steps"]
        if not observation.done and self._state.step_count >= max_steps:
            observation.done = True
            observation.tool_message = (
                f"{observation.tool_message} Step budget exhausted."
            ).strip()
            observation.reward = min(observation.reward or 0.0, 0.05)

        observation.steps_used = self._state.step_count
        observation.steps_remaining = max(0, max_steps - self._state.step_count)
        observation.question = self._current_task["question"]
        return observation

    def _handle_search(self, action: SearchEmailsPayload) -> EmailObservation:
        assert self._current_task is not None
        signature = (
            action.query or "",
            action.top_k,
            action.sent_after or "",
            action.sent_before or "",
            (action.sender or "").lower(),
        )
        repeated = signature in self._search_signatures
        self._search_signatures.add(signature)
        if repeated:
            self._repeated_searches += 1

        results = search_emails(
            query=action.query,
            top_k=action.top_k,
            sent_after=action.sent_after,
            sent_before=action.sent_before,
            sender=action.sender,
        )
        reward = 0.0 if repeated else 0.02
        message = f"Returned {len(results)} search result(s)."
        if repeated:
            message += " This search repeated a previous query."
        return EmailObservation(
            last_action="search_emails",
            search_results=results,
            tool_message=message,
            reward=reward,
            done=False,
        )

    def _handle_read(self, action: ReadEmailPayload) -> EmailObservation:
        assert self._current_task is not None
        record = read_email(action.email_id)
        if record is None:
            return EmailObservation(
                last_action="read_email",
                tool_message=f"Email '{action.email_id}' was not found.",
                reward=0.0,
                done=False,
            )

        repeated = record["email_id"] in self._visited_email_ids
        if repeated:
            self._repeated_reads += 1
        else:
            self._visited_email_ids.add(record["email_id"])

        reward = 0.0
        if record["email_id"] in self._current_task["gold_email_ids"] and not repeated:
            reward = 0.12
        elif not repeated:
            reward = 0.03

        message = "Opened email."
        if repeated:
            message = "Opened email again."

        return EmailObservation(
            last_action="read_email",
            email=record,
            tool_message=message,
            reward=reward,
            done=False,
        )

    def _handle_final_answer(
        self, action: ReturnFinalAnswerPayload
    ) -> EmailObservation:
        assert self._current_task is not None
        answer_score = self._score_answer(action.answer, self._current_task)
        citation_score = self._score_citations(action.cited_email_ids, self._current_task)
        efficiency_score = self._score_efficiency(self._current_task)
        final_score = round(
            (0.65 * answer_score)
            + (0.2 * citation_score)
            + (0.15 * efficiency_score),
            4,
        )

        citations = action.cited_email_ids or []
        message = "Final answer graded."
        if not citations:
            message += " No citations were provided."

        return EmailObservation(
            last_action="return_final_answer",
            tool_message=message,
            reward_breakdown=RewardBreakdown(
                answer_score=answer_score,
                citation_score=citation_score,
                efficiency_score=efficiency_score,
                final_score=final_score,
            ),
            reward=final_score,
            done=True,
            metadata={
                "answer_score": answer_score,
                "citation_score": citation_score,
                "efficiency_score": efficiency_score,
                "submitted_answer": action.answer,
                "submitted_citations": citations,
                "gold_email_ids": self._current_task["gold_email_ids"],
            },
        )

    def _score_answer(self, answer: str, task: dict[str, Any]) -> float:
        normalized_answer = _normalize_text(answer)
        accepted = [_normalize_text(task["canonical_answer"])] + [
            _normalize_text(alias) for alias in task.get("answer_aliases", [])
        ]
        if normalized_answer in accepted:
            return 1.0
        if any(candidate in normalized_answer for candidate in accepted):
            return 0.85
        return 0.0

    def _score_citations(self, cited_email_ids: list[str], task: dict[str, Any]) -> float:
        if not cited_email_ids:
            return 0.0
        gold = set(task["gold_email_ids"])
        provided = set(cited_email_ids)
        overlap = len(gold & provided)
        if overlap == 0:
            return 0.0
        precision = overlap / len(provided)
        recall = overlap / len(gold)
        return round((precision + recall) / 2, 4)

    def _score_efficiency(self, task: dict[str, Any]) -> float:
        max_steps = task["max_steps"]
        waste = self._repeated_reads + self._repeated_searches
        if max_steps <= 1:
            step_component = 1.0
        else:
            step_component = 1 - ((self._state.step_count - 1) / (max_steps - 1))
        raw = step_component - (0.08 * waste)
        return round(max(0.0, min(1.0, raw)), 4)

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Email Research Env",
            description=(
                "Answer questions over a synthetic enterprise inbox using search, "
                "read, and grounded final-answer actions."
            ),
            version="0.1.0",
        )
