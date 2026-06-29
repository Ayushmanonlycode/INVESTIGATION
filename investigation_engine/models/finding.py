"""Finding model — the universal evidence schema.

Every investigation module produces `Finding` instances. This is the single
output contract between modules and all downstream consumers:
    - Investigation Priority Score (IPS) engine
    - AI Reasoning Engine
    - LLM integrations
    - Interactive Investigation UI

Design Principles:
    - Machine-readable, never formatted text
    - Self-contained: a Finding carries all evidence needed to evaluate it
    - IPS-ready: metadata dict exposes scoring dimensions
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


from investigation_engine.models.severity import Severity


class Finding(BaseModel):
    """A single investigative finding produced by an investigation module.

    Attributes:
        id: Unique identifier (UUID4).
        module: Name of the module that generated this finding.
        title: Short, descriptive title summarizing the finding.
        description: Detailed explanation of what was discovered and why it matters.
        evidence: Raw statistical evidence supporting the finding.
            Should contain reproducible metrics (e.g., z-scores, p-values, counts).
        severity: Classification of investigative urgency.
        confidence: Confidence in the finding's validity (0.0 to 1.0).
        recommendation: Actionable next step for the analyst.
        affected_columns: List of column names involved in this finding.
        affected_rows: Optional list of row indices affected.
            None if the finding is column-level or dataset-level.
        metadata: Extensible metadata for IPS scoring.
            Expected keys (not enforced in Phase 1):
                - anomaly_strength: float — magnitude of deviation
                - statistical_significance: float — p-value or equivalent
                - rarity: float — how unusual this pattern is (0.0-1.0)
                - correlation_strength: float — strength of discovered relationship
                - impact: float — estimated impact on analysis (0.0-1.0)
                - confidence_interval: tuple[float, float] — CI bounds
        created_at: Timestamp when the finding was generated.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this finding.",
    )
    module: str = Field(
        ...,
        min_length=1,
        description="Name of the investigation module that produced this finding.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Short descriptive title summarizing the finding.",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Detailed explanation of the finding and its significance.",
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw statistical evidence (metrics, scores, distributions).",
    )
    severity: Severity = Field(
        ...,
        description="Investigative urgency classification.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in finding validity (0.0 to 1.0).",
    )
    recommendation: str = Field(
        ...,
        min_length=1,
        description="Actionable next step for the analyst.",
    )
    affected_columns: list[str] = Field(
        default_factory=list,
        description="Column names involved in this finding.",
    )
    affected_rows: list[int] | None = Field(
        default=None,
        description="Row indices affected. None for column-level or dataset-level findings.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extensible metadata for IPS scoring. Expected keys: "
            "anomaly_strength, statistical_significance, rarity, "
            "correlation_strength, impact, confidence_interval."
        ),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this finding was generated.",
    )

    model_config = {
        "frozen": False,
        "json_schema_extra": {
            "title": "InvestigationFinding",
            "description": (
                "Universal evidence schema for the AI Investigation Intelligence Engine. "
                "Produced by investigation modules, consumed by IPS, LLM, and UI layers."
            ),
        },
    }

    @field_validator("affected_columns")
    @classmethod
    def validate_affected_columns(cls, v: list[str]) -> list[str]:
        """Ensure column names are non-empty strings."""
        for col in v:
            if not col.strip():
                raise ValueError("Column names must be non-empty strings.")
        return v

    @field_validator("affected_rows")
    @classmethod
    def validate_affected_rows(cls, v: list[int] | None) -> list[int] | None:
        """Ensure row indices are non-negative."""
        if v is not None:
            for idx in v:
                if idx < 0:
                    raise ValueError(f"Row index must be non-negative, got {idx}.")
        return v

    @property
    def is_actionable(self) -> bool:
        """Whether this finding requires analyst action."""
        return self.severity in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM)

    @property
    def column_count(self) -> int:
        """Number of columns affected by this finding."""
        return len(self.affected_columns)

    @property
    def row_count(self) -> int | None:
        """Number of rows affected, or None if not row-level."""
        return len(self.affected_rows) if self.affected_rows is not None else None

    def to_evidence_dict(self) -> dict[str, Any]:
        """Export finding as a flat evidence dictionary for downstream consumption.

        Returns a dictionary optimized for IPS scoring and serialization,
        combining core fields with metadata.
        """
        return {
            "finding_id": self.id,
            "module": self.module,
            "severity": self.severity.value,
            "severity_weight": self.severity.numeric_weight,
            "confidence": self.confidence,
            "column_count": self.column_count,
            "row_count": self.row_count,
            **self.metadata,
        }
