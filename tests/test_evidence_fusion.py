"""Tests for the evidentiary reasoning pipeline."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from investigation_engine.config.settings import Settings
from investigation_engine.core.engine import InvestigationEngine
from investigation_engine.fusion.engine import EvidenceFusionEngine
from investigation_engine.models.finding import Finding
from investigation_engine.models.severity import Severity


def _make_finding(
    fid: str,
    title: str,
    category: str,
    columns: list[str],
    severity: Severity = Severity.MEDIUM,
    confidence: float = 0.8,
    module: str = "integrity_investigator",
) -> Finding:
    return Finding(
        id=fid,
        module=module,
        title=title,
        description=f"Detailed description of {title}.",
        evidence={
            "correlation_coefficient": 0.9 if "correlation" in category else None,
            "p_value": 0.001 if "correlation" in category else None,
            "mutual_information_score": 0.3 if category == "mutual_information" else None,
            "vif": 12.0 if category == "multicollinearity" else None,
            "thresholds": {"test": 1},
        },
        severity=severity,
        confidence=confidence,
        recommendation="Recommended remediation step.",
        affected_columns=columns,
        metadata={"category": category},
        created_at=datetime.now(UTC),
    )


class TestEvidenceCompression:
    def test_compresses_many_findings_into_fewer_evidence_units(self):
        findings = [
            _make_finding(
                f"f{i}",
                f"Correlation finding {i}",
                "pearson_correlation",
                ["neighbor_rsrp", "neighbor_rsrq", "neighbor_sinr"],
                severity=Severity.MEDIUM,
                confidence=0.82,
                module="relationship_investigator",
            )
            for i in range(8)
        ]

        engine = EvidenceFusionEngine()
        artifacts = engine.build_reasoning_artifacts(findings)

        assert len(artifacts.evidence_units) < len(findings)
        evidence = artifacts.evidence_units[0]
        assert evidence.category == "Correlation Community"
        assert set(evidence.supporting_findings) == {finding.id for finding in findings}

    def test_preserves_provenance_chain(self):
        findings = [
            _make_finding("f1", "Missing col_a", "missing_values", ["neighbor_a"]),
            _make_finding(
                "f2",
                "Missingness relation",
                "missingness_relationships",
                ["neighbor_a", "neighbor_b"],
            ),
        ]

        engine = EvidenceFusionEngine()
        artifacts = engine.build_reasoning_artifacts(findings)
        investigation = artifacts.investigation_queue.investigations[0]

        assert investigation.provenance["hypothesis_id"]
        assert investigation.provenance["evidence_ids"]
        assert investigation.provenance["finding_ids"] == ["f1", "f2"]
        assert set(investigation.supporting_findings) == {"f1", "f2"}


class TestPrioritizationBiasControl:
    def test_priority_does_not_reward_raw_finding_count(self):
        verbose_findings = [
            _make_finding(
                f"r{i}",
                f"Correlation {i}",
                "pearson_correlation",
                ["corr_a", "corr_b", "corr_c"],
                severity=Severity.MEDIUM,
                confidence=0.78,
                module="relationship_investigator",
            )
            for i in range(12)
        ]
        critical_findings = [
            _make_finding(
                "i1",
                "Duplicate identifiers",
                "identifiers",
                ["user_id"],
                severity=Severity.CRITICAL,
                confidence=0.92,
                module="integrity_investigator",
            )
        ]

        engine = EvidenceFusionEngine()
        investigations = engine.fuse_findings(verbose_findings + critical_findings)

        by_title = {investigation.title: investigation for investigation in investigations}
        correlation = next(
            investigation
            for title, investigation in by_title.items()
            if "related signal family" in title.lower()
        )
        identifier = next(
            investigation
            for title, investigation in by_title.items()
            if "record linkage" in title.lower() or "identifier" in title.lower()
        )

        assert identifier.priority >= correlation.priority
        assert identifier.evidence_score >= correlation.evidence_score


class TestReasoningArtifacts:
    def test_builds_evidence_knowledge_hypothesis_and_investigations(self):
        findings = [
            _make_finding("f1", "Constant a", "constant_features", ["a"], Severity.HIGH),
            _make_finding("f2", "Low-cardinality a", "cardinality", ["a"], Severity.MEDIUM),
            _make_finding("f3", "Duplicate id", "identifiers", ["record_id"], Severity.HIGH),
        ]

        engine = EvidenceFusionEngine()
        artifacts = engine.build_reasoning_artifacts(findings)

        assert artifacts.evidence_units
        assert artifacts.knowledge_objects
        assert artifacts.hypotheses
        assert artifacts.investigation_queue.investigations
        assert all(hypothesis.supporting_evidence for hypothesis in artifacts.hypotheses)
        assert all(
            investigation.priority_explanation
            for investigation in artifacts.investigation_queue.investigations
        )

    def test_backward_compatible_fuse_findings_api(self):
        findings = [
            _make_finding("f1", "Null ID", "identifiers", ["entity_id"], Severity.HIGH),
        ]
        engine = EvidenceFusionEngine()
        investigations = engine.fuse_findings(findings)

        assert len(investigations) == 1
        assert investigations[0].supporting_findings == ["f1"]


class TestEngineIntegration:
    def test_engine_result_exposes_new_reasoning_layers(self):
        df = pd.DataFrame({
            "user_id": [*list(range(50)), 49],
            "metric_a": list(range(51)),
            "metric_b": [value * 2 for value in range(51)],
        })

        result = InvestigationEngine(Settings()).investigate_dataframe(df, name="reasoning_test")

        assert result.evidence_units
        assert result.knowledge_objects
        assert result.hypotheses
        assert result.investigations
        assert "reasoning_summary" in result.metadata
