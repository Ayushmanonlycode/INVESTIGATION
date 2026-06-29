"""Investigation model — the fused evidence reasoning container.

Each Investigation combines multiple related Findings into a higher-level,
cohesive hypothesis.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Investigation(BaseModel):
    """A higher-level investigation representing a group of fused Findings.

    Attributes:
        investigation_id: Unique UUID.
        title: Short synthesized title of the investigation.
        summary: Contextual summary of the issue.
        hypothesis: Root cause or analytical theory.
        supporting_findings: List of Finding IDs that support this hypothesis.
        evidence_score: Combined strength of the evidence (0.0-100.0).
        confidence: Combined confidence in the investigation (0.0-1.0).
        priority: Calculated investigation priority (0.0-100.0) for ranking.
        possible_causes: List of bulleted possible causes.
        recommended_next_steps: List of recommended remediation steps.
        affected_columns: List of columns affected.
        metadata: Extensible metadata for downstream retrieval/RAG.
        created_at: Generation timestamp.
    """

    investigation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this investigation.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Synthesized description of the investigation topic.",
    )
    summary: str = Field(
        ...,
        description="Contextual explanation of what was observed.",
    )
    hypothesis: str = Field(
        ...,
        description="Core explanation or theory explaining the observations.",
    )
    supporting_findings: list[str] = Field(
        default_factory=list,
        description="List of IDs of Findings backing this investigation.",
    )
    evidence_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Unified rating of how severe the supporting evidence is.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Consolidated confidence in this finding.",
    )
    priority: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Weighted priority rank of this investigation.",
    )
    possible_causes: list[str] = Field(
        default_factory=list,
        description="Potential root causes of the observed patterns.",
    )
    recommended_next_steps: list[str] = Field(
        default_factory=list,
        description="Actionable steps the analyst should take next.",
    )
    affected_columns: list[str] = Field(
        default_factory=list,
        description="List of columns involved in this investigation.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata for GraphRAG or IPS integration.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC creation time.",
    )

    model_config = {
        "frozen": False,
        "json_schema_extra": {
            "title": "DatasetInvestigation",
            "description": "A synthesized investigation representing the output of Evidence Fusion.",
        },
    }
