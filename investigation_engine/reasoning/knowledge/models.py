"""Knowledge-layer models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeObject(BaseModel):
    """Reusable semantic concept built from evidence units."""

    knowledge_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this knowledge object.",
    )
    concept: str = Field(..., min_length=1, description="Semantic concept name.")
    description: str = Field(..., min_length=1, description="Description of the concept.")
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="EvidenceUnit IDs supporting this concept.",
    )
    related_concepts: list[str] = Field(
        default_factory=list,
        description="Concept names linked semantically to this concept.",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Lineage back to evidence units, findings, and investigators.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the concept.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible knowledge metadata.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC creation timestamp.",
    )
