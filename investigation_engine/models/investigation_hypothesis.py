"""Investigation model — the fused evidence reasoning container.

Each Investigation combines multiple related Findings into a higher-level,
cohesive hypothesis.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class LikelyCause(BaseModel):
    """Structured, provenance-backed explanation for a likely cause."""

    cause: str = Field(..., min_length=1, description="Analyst-facing likely cause statement.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this likely cause.")
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="EvidenceUnit IDs that explicitly support this cause.",
    )
    supporting_findings: list[str] = Field(
        default_factory=list,
        description="Finding IDs transitively supporting this cause.",
    )
    knowledge_objects: list[str] = Field(
        default_factory=list,
        description="KnowledgeObject IDs that anchor this cause semantically.",
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="EvidenceUnit IDs that weaken or complicate this cause.",
    )
    unknown_evidence: list[str] = Field(
        default_factory=list,
        description="Explicit unresolved evidence gaps relevant to this cause.",
    )


class EvidenceStrengthAssessment(BaseModel):
    """Structured explanation of empirical support strength, distinct from confidence."""

    rating: str = Field(..., min_length=1, description="Qualitative strength rating.")
    supporting_evidence_units: list[str] = Field(
        default_factory=list,
        description="Supporting evidence unit IDs used for the strength assessment.",
    )
    supporting_findings: list[str] = Field(
        default_factory=list,
        description="Supporting finding IDs used for the strength assessment.",
    )
    cross_investigator_agreement: str = Field(
        ...,
        min_length=1,
        description="Qualitative agreement rating across contributing investigators.",
    )
    contradictions: str = Field(
        ...,
        min_length=1,
        description="Qualitative contradiction level for this investigation.",
    )
    unknown_evidence: str = Field(
        ...,
        min_length=1,
        description="Qualitative unresolved-evidence level for this investigation.",
    )
    graph_cohesion: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic cohesion score of the merged hypothesis group.",
    )
    community_strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average internal strength of the supporting evidence communities.",
    )


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
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="EvidenceUnit IDs supporting this investigation.",
    )
    supporting_hypotheses: list[str] = Field(
        default_factory=list,
        description="Hypothesis IDs that produced this investigation.",
    )
    supporting_knowledge_objects: list[str] = Field(
        default_factory=list,
        description="KnowledgeObject IDs supporting this investigation.",
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="EvidenceUnit IDs that weaken or complicate this investigation.",
    )
    unknown_evidence: list[str] = Field(
        default_factory=list,
        description="Outstanding unknown evidence gaps relevant to this investigation.",
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
    confidence_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Explainable weighted factors contributing to investigation confidence.",
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
    likely_causes: list[LikelyCause] = Field(
        default_factory=list,
        description="Structured, provenance-backed likely causes for this investigation.",
    )
    recommended_next_steps: list[str] = Field(
        default_factory=list,
        description="Actionable steps the analyst should take next.",
    )
    evidence_strength_details: EvidenceStrengthAssessment | None = Field(
        default=None,
        description="Structured empirical support assessment, separate from confidence.",
    )
    priority_explanation: list[str] = Field(
        default_factory=list,
        description="Human-readable explanation of the computed priority.",
    )
    affected_columns: list[str] = Field(
        default_factory=list,
        description="List of columns involved in this investigation.",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Lineage linking the investigation back to hypotheses and findings.",
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
