"""Typed models for the email research environment."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, model_validator


class SearchResult(BaseModel):
    """Compact search result returned by ``search_emails``."""

    email_id: str = Field(..., description="Unique identifier for the email")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Sender email address")
    sent_at: str = Field(..., description="Email timestamp in ISO date format")
    snippet: str = Field(..., description="Short body snippet for quick triage")


class EmailRecord(BaseModel):
    """Full email record returned by ``read_email``."""

    email_id: str = Field(..., description="Unique identifier for the email")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Sender email address")
    recipients: list[str] = Field(
        default_factory=list, description="Recipient email addresses"
    )
    sent_at: str = Field(..., description="Email timestamp in ISO date format")
    body: str = Field(..., description="Full email body")


class RewardBreakdown(BaseModel):
    """Structured reward details for graded final-answer turns."""

    answer_score: float = Field(..., description="Answer correctness score in [0, 1]")
    citation_score: float = Field(
        ..., description="Citation grounding score in [0, 1]"
    )
    efficiency_score: float = Field(
        ..., description="Efficiency score based on step use in [0, 1]"
    )
    final_score: float = Field(..., description="Final blended score in [0, 1]")


class SearchEmailsPayload(BaseModel):
    """Payload for ``search_emails`` actions."""

    action_type: Literal["search_emails"] = Field(
        default="search_emails", description="Discriminator for search actions"
    )
    query: Optional[str] = Field(
        default=None, description="Full-text query for search actions"
    )
    top_k: int = Field(default=5, ge=1, le=10, description="Maximum search hits")
    sent_after: Optional[str] = Field(
        default=None, description="Inclusive lower date bound in YYYY-MM-DD format"
    )
    sent_before: Optional[str] = Field(
        default=None, description="Inclusive upper date bound in YYYY-MM-DD format"
    )
    sender: Optional[str] = Field(
        default=None, description="Optional sender email filter"
    )

    @model_validator(mode="after")
    def _validate_search_shape(self) -> "SearchEmailsPayload":
        if not self.query and not any([self.sent_after, self.sent_before, self.sender]):
            raise ValueError(
                "search_emails requires a query or at least one metadata filter"
            )
        return self


class ReadEmailPayload(BaseModel):
    """Payload for ``read_email`` actions."""

    action_type: Literal["read_email"] = Field(
        default="read_email", description="Discriminator for read actions"
    )
    email_id: str = Field(..., description="Email identifier for read actions")


class ReturnFinalAnswerPayload(BaseModel):
    """Payload for ``return_final_answer`` actions."""

    action_type: Literal["return_final_answer"] = Field(
        default="return_final_answer",
        description="Discriminator for answer submission actions",
    )
    answer: str = Field(
        ..., description="Final answer for answer submission"
    )
    cited_email_ids: list[str] = Field(
        default_factory=list,
        description="Email IDs cited as supporting evidence in the final answer",
    )


EmailActionPayload = Annotated[
    SearchEmailsPayload | ReadEmailPayload | ReturnFinalAnswerPayload,
    Field(discriminator="action_type"),
]


class EmailAction(Action):
    """Top-level OpenEnv action wrapping typed per-tool payload models."""

    payload: EmailActionPayload = Field(..., description="Tool-specific action payload")

    @model_validator(mode="before")
    @classmethod
    def _coerce_flat_payload(cls, value: object) -> object:
        if isinstance(value, dict) and "payload" not in value and "action_type" in value:
            metadata = value.get("metadata", {})
            payload = {key: item for key, item in value.items() if key != "metadata"}
            return {"payload": payload, "metadata": metadata}
        return value

    @property
    def action_type(self) -> str:
        return self.payload.action_type


class EmailObservation(Observation):
    """Observation model for the email research environment."""

    question: str = Field(default="", description="Question the agent must answer")
    last_action: str = Field(default="", description="Name of the most recent action")
    search_results: list[SearchResult] = Field(
        default_factory=list, description="Search hits for search actions"
    )
    email: Optional[EmailRecord] = Field(
        default=None, description="Full email record for read actions"
    )
    tool_message: str = Field(
        default="", description="Human-readable status or tool feedback"
    )
    reward_breakdown: Optional[RewardBreakdown] = Field(
        default=None,
        description="Structured reward details populated on final graded turns",
    )
    steps_used: int = Field(default=0, description="Steps used in the current episode")
    steps_remaining: int = Field(
        default=0, description="Remaining budget for the current episode"
    )


# Backwards-compatible aliases for the original scaffold imports.
MyAction = EmailAction
MyObservation = EmailObservation
