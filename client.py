# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Async client for the email research environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import EmailAction, EmailObservation
except ImportError:
    from models import EmailAction, EmailObservation


class MyEnv(EnvClient[EmailAction, EmailObservation, State]):
    """Client for the email research environment."""

    def _step_payload(self, action: EmailAction) -> Dict[str, Any]:
        payload = action.model_dump(exclude_none=True)
        nested_payload = payload.pop("payload", {})
        return {**nested_payload, **payload}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[EmailObservation]:
        obs_data = payload.get("observation", {})
        observation = EmailObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", False),
                "reward": payload.get("reward"),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
