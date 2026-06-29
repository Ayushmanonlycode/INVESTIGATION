"""Comprehensive unit tests for the Evidence Fusion Engine (Phase 1.5).

Tests verify graph construction, rule matching, synthesized title and
description accuracy, priority scoring, confidence aggregation, and Pydantic
serialization of final Investigation objects.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
import pytest

from investigation_engine.config.settings import Settings
from investigation_engine.fusion.engine import EvidenceFusionEngine
from investigation_engine.fusion.graph import EvidenceGraph
from investigation_engine.models.finding import Finding
from investigation_engine.models.investigation_hypothesis import Investigation
from investigation_engine.models.severity import Severity


# ═══════════════════════════════════════════════════════════════════════
#  Helper: build mock findings
# ═══════════════════════════════════════════════════════════════════════

def _make_finding(
    fid: str,
    title: str,
    category: str,
    columns: list[str],
    severity: Severity = Severity.MEDIUM,
    confidence: float = 0.8,
) -> Finding:
    """Helper to create a valid Finding object."""
    return Finding(
        id=fid,
        module="integrity_investigator",
        title=title,
        description=f"Detailed description of {title}.",
        evidence={"score": 42.0},
        severity=severity,
        confidence=confidence,
        recommendation="Recommended remediation step.",
        affected_columns=columns,
        metadata={"category": category},
        created_at=datetime.now(timezone.utc),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Test: Evidence Graph Construction
# ═══════════════════════════════════════════════════════════════════════

class TestEvidenceGraph:
    def test_adds_nodes_and_edges(self):
        f1 = _make_finding("f1", "Missing value in col_a", "missing_values", ["col_a"])
        f2 = _make_finding("f2", "Missing value in col_b", "missing_values", ["col_b"])
        # f3 has a missingness correlation involving col_a and col_b
        f3 = _make_finding(
            "f3",
            "Missingness correlation col_a & col_b",
            "missingness_relationships",
            ["col_a", "col_b"],
        )

        eg = EvidenceGraph()
        eg.build_from_findings([f1, f2, f3])

        # Verify all nodes added
        assert "f1" in eg.graph.nodes
        assert "f2" in eg.graph.nodes
        assert "f3" in eg.graph.nodes

        # Verify f3 is connected to f1 and f2 due to column overlap / missingness correlation
        assert eg.graph.has_edge("f1", "f3")
        assert eg.graph.has_edge("f2", "f3")

    def test_partitions_connected_components(self):
        # Component 1 (col_a, col_b missingness correlation)
        f1 = _make_finding("f1", "Missing values col_a", "missing_values", ["col_a"])
        f2 = _make_finding("f2", "Correlation col_a & col_b", "missingness_relationships", ["col_a", "col_b"])

        # Component 2 (col_c constant)
        f3 = _make_finding("f3", "Constant col_c", "constant_features", ["col_c"])

        eg = EvidenceGraph()
        eg.build_from_findings([f1, f2, f3])

        components = eg.get_connected_subgraphs()
        assert len(components) == 2

        # Verify findings grouped properly
        comp_ids = [{f.id for f in comp} for comp in components]
        assert {"f1", "f2"} in comp_ids
        assert {"f3"} in comp_ids


# ═══════════════════════════════════════════════════════════════════════
#  Test: Evidence Fusion Engine
# ═══════════════════════════════════════════════════════════════════════

class TestEvidenceFusionEngine:
    def test_missing_values_rule_applied(self):
        f1 = _make_finding("f1", "Column col_a is 30% missing", "missing_values", ["col_a"], Severity.HIGH)
        f2 = _make_finding("f2", "Column col_b is 30% missing", "missing_values", ["col_b"], Severity.HIGH)
        f3 = _make_finding("f3", "Correlation col_a & col_b", "missingness_relationships", ["col_a", "col_b"], Severity.MEDIUM)

        engine = EvidenceFusionEngine()
        investigations = engine.fuse_findings([f1, f2, f3])

        # Should fuse into a single missingness investigation
        assert len(investigations) == 1
        inv = investigations[0]
        assert inv.metadata.get("rule_applied") == "missing_values_fusion"
        assert "col_a" in inv.affected_columns
        assert "col_b" in inv.affected_columns
        assert len(inv.supporting_findings) == 3

    def test_constant_rule_applied(self):
        f1 = _make_finding("f1", "Constant column col_c", "constant_features", ["col_c"])
        f2 = _make_finding("f2", "Low cardinality col_c", "cardinality", ["col_c"])

        engine = EvidenceFusionEngine()
        investigations = engine.fuse_findings([f1, f2])

        assert len(investigations) == 1
        inv = investigations[0]
        assert inv.metadata.get("rule_applied") == "constant_feature_fusion"
        assert inv.affected_columns == ["col_c"]

    def test_duplicate_feature_rule_applied(self):
        f1 = _make_finding("f1", "Duplicate rows", "duplicates", ["col_d1", "col_d2"])
        f2 = _make_finding("f2", "Identical columns", "duplicate_features", ["col_d1", "col_d2"])

        engine = EvidenceFusionEngine()
        investigations = engine.fuse_findings([f1, f2])

        assert len(investigations) == 1
        inv = investigations[0]
        assert inv.metadata.get("rule_applied") == "duplicate_feature_fusion"

    def test_identifier_rule_applied(self):
        f1 = _make_finding("f1", "Non-unique ID col_e", "identifiers", ["col_e"])
        f2 = _make_finding("f2", "Null ID col_e", "identifiers", ["col_e"])

        engine = EvidenceFusionEngine()
        investigations = engine.fuse_findings([f1, f2])

        assert len(investigations) == 1
        inv = investigations[0]
        assert inv.metadata.get("rule_applied") == "identifier_fusion"

    def test_unfused_findings_wrapped_cleanly(self):
        f1 = _make_finding("f1", "Missing values col_a", "missing_values", ["col_a"])
        # f2 is totally unrelated to f1 (different column, different category)
        f2 = _make_finding("f2", "Constant col_c", "constant_features", ["col_c"])

        engine = EvidenceFusionEngine()
        investigations = engine.fuse_findings([f1, f2])

        # Should produce two separate wrapped investigations
        assert len(investigations) == 2
        rules = {inv.metadata.get("rule_applied") for inv in investigations}
        assert rules == {"single_evidence_wrapper"}

    def test_corroboration_confidence_bonus(self):
        # Average confidence = 0.8
        f1 = _make_finding("f1", "Null ID col_e", "identifiers", ["col_e"], confidence=0.8)
        f2 = _make_finding("f2", "Duplicate ID col_e", "identifiers", ["col_e"], confidence=0.8)

        engine = EvidenceFusionEngine()
        investigations = engine.fuse_findings([f1, f2])
        inv = investigations[0]

        # Expect confidence to be greater than 0.8 due to corroboration bonus (+5%)
        assert inv.confidence > 0.8
        assert inv.confidence == 0.85


# ═══════════════════════════════════════════════════════════════════════
#  Test: Investigation Validity & Serialization
# ═══════════════════════════════════════════════════════════════════════

class TestInvestigationValidity:
    def test_investigation_model_fields(self):
        f1 = _make_finding("f1", "Null ID col_e", "identifiers", ["col_e"], severity=Severity.HIGH)
        f2 = _make_finding("f2", "Duplicate ID col_e", "identifiers", ["col_e"], severity=Severity.MEDIUM)

        engine = EvidenceFusionEngine()
        investigations = engine.fuse_findings([f1, f2])
        assert len(investigations) == 1
        inv = investigations[0]

        # Check required fields exist
        assert inv.investigation_id
        assert inv.title
        assert inv.summary
        assert inv.hypothesis
        assert len(inv.supporting_findings) == 2
        assert 0.0 <= inv.evidence_score <= 100.0
        assert 0.0 <= inv.confidence <= 1.0
        assert 0.0 <= inv.priority <= 100.0
        assert len(inv.possible_causes) > 0
        assert len(inv.recommended_next_steps) > 0
        assert inv.affected_columns == ["col_e"]

    def test_investigation_is_serializable(self):
        f1 = _make_finding("f1", "Null ID col_e", "identifiers", ["col_e"])
        engine = EvidenceFusionEngine()
        investigations = engine.fuse_findings([f1])
        inv = investigations[0]

        json_str = inv.model_dump_json()
        assert len(json_str) > 0
        reconstructed = Investigation.model_validate_json(json_str)
        assert reconstructed.investigation_id == inv.investigation_id
