"""Evidence-layer models for compressed analytical facts."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class EvidenceUnit(BaseModel):
    """Compressed analytical fact derived from one or more findings."""

    evidence_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this evidence unit.",
    )
    category: str = Field(..., min_length=1, description="Evidence category label.")
    title: str = Field(..., min_length=1, max_length=200, description="Short evidence title.")
    summary: str = Field(..., min_length=1, description="Analytical summary of the evidence.")
    supporting_findings: list[str] = Field(
        default_factory=list,
        description="IDs of findings compressed into this evidence unit.",
    )
    community_id: str | None = Field(
        default=None,
        description="Identifier of the evidence community that produced this unit.",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Traceable lineage back to findings and investigators.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the evidence.")
    strength: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Strength of the evidence independent of raw finding count.",
    )
    community_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized internal strength of the evidence community.",
    )
    structural_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregate structural similarity reused from evidence-graph edges.",
    )
    statistical_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregate statistical similarity reused from evidence-graph edges.",
    )
    semantic_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregate semantic similarity reused from evidence-graph edges.",
    )
    merge_explanation: list[str] = Field(
        default_factory=list,
        description="Human-readable explanation of why the findings were merged.",
    )
    affected_columns: list[str] = Field(
        default_factory=list,
        description="Columns implicated by this evidence unit.",
    )
    representative_columns: list[str] = Field(
        default_factory=list,
        description="Representative columns summarizing the evidence community.",
    )
    affected_rows: list[int] | None = Field(
        default=None,
        description="Optional row indices implicated by this evidence unit.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible evidence metadata.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC creation timestamp.",
    )
