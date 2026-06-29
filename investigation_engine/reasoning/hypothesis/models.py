"""Hypothesis-layer models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    """Evidence-backed analytical hypothesis."""

    hypothesis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this hypothesis.",
    )
    title: str = Field(..., min_length=1, max_length=200, description="Short hypothesis title.")
    statement: str = Field(
        ...,
        min_length=1,
        description="Traceable analytical statement derived from evidence.",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="EvidenceUnit IDs supporting this hypothesis.",
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="EvidenceUnit IDs that weaken or complicate this hypothesis.",
    )
    unknowns: list[str] = Field(
        default_factory=list,
        description="Explicit unknowns that remain unresolved.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Hypothesis confidence.")
    plausible_causes: list[str] = Field(
        default_factory=list,
        description="Possible causes suggested by the evidence.",
    )
    validation_steps: list[str] = Field(
        default_factory=list,
        description="Concrete analyst validation steps.",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Lineage back to knowledge objects, evidence units, and findings.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible hypothesis metadata.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC creation timestamp.",
    )
