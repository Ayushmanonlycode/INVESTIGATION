"""Evidence Fusion Engine — the reasoning/synthesis orchestrator.

Coordinates evidence graph construction, connected component partitioning,
and rule evaluation to produce high-level Investigations from raw Findings.
"""

from __future__ import annotations

from loguru import logger

from investigation_engine.config.settings import Settings
from investigation_engine.fusion.graph import EvidenceGraph
from investigation_engine.fusion.rules import (
    BaseFusionRule,
    ConstantFeatureRule,
    DuplicateFeatureRule,
    GenericGroupRule,
    IdentifierRule,
    MissingValuesRule,
)
from investigation_engine.models.finding import Finding
from investigation_engine.models.investigation_hypothesis import Investigation
from investigation_engine.models.severity import Severity


class EvidenceFusionEngine:
    """Combines isolated Findings into high-level Investigations.

    Uses an internal EvidenceGraph to group related findings and applies
    concrete BaseFusionRules to synthesize the unified hypotheses.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or Settings()
        self.graph_builder = EvidenceGraph()

        # Rules ordered by priority of matching (specific to general)
        self.rules: list[BaseFusionRule] = [
            MissingValuesRule(),
            IdentifierRule(),
            DuplicateFeatureRule(),
            ConstantFeatureRule(),
            GenericGroupRule(),
        ]

    def fuse_findings(self, findings: list[Finding]) -> list[Investigation]:
        """Orchestrate the evidence fusion workflow.

        Args:
            findings: List of raw findings.

        Returns:
            List of synthesized Investigations.
        """
        if not findings:
            return []

        # 1. Filter out metadata findings like the Structural Health Score
        #    which shouldn't be clustered as regular evidence.
        evidence_findings = [
            f for f in findings
            if f.metadata.get("category") != "structural_health_score"
        ]
        health_score_findings = [
            f for f in findings
            if f.metadata.get("category") == "structural_health_score"
        ]

        # 2. Build the relationship graph
        self.graph_builder.build_from_findings(evidence_findings)

        # 3. Partition the graph into connected subgraphs (clusters)
        clusters = self.graph_builder.get_connected_subgraphs()
        logger.info("EvidenceFusionEngine: partitioned findings into {} clusters", len(clusters))

        investigations: list[Investigation] = []

        # Track processed finding IDs to ensure no duplicates or leaks
        fused_finding_ids: set[str] = set()

        # 4. Process multi-finding clusters
        for cluster in clusters:
            if len(cluster) < 2:
                # Handled later as single-evidence investigations
                continue

            fused_inv = self._apply_rules_to_cluster(cluster)
            if fused_inv:
                investigations.append(fused_inv)
                fused_finding_ids.update(fused_inv.supporting_findings)

        # 5. Process leftover single findings so that no evidence is lost.
        #    Each isolated finding gets wrapped in a clean, direct Investigation.
        for f in evidence_findings:
            if f.id in fused_finding_ids:
                continue

            isolated_inv = self._wrap_single_finding(f)
            investigations.append(isolated_inv)

        # 6. Pass through Structural Health Score finding as a summary investigation
        for f in health_score_findings:
            investigations.append(self._wrap_health_score_finding(f))

        # Sort investigations by priority (highest first)
        investigations.sort(key=lambda inv: -inv.priority)

        logger.info("EvidenceFusionEngine: fused findings into {} investigations", len(investigations))
        return investigations

    def _apply_rules_to_cluster(self, cluster: list[Finding]) -> Investigation | None:
        """Find the first matching rule and use it to fuse the cluster."""
        for rule in self.rules:
            if rule.matches(cluster):
                try:
                    logger.debug("EvidenceFusionEngine: applying rule '{}'", rule.rule_name)
                    return rule.fuse(cluster)
                except Exception as e:
                    logger.error(
                        "EvidenceFusionEngine: failed to apply rule '{}': {}",
                        rule.rule_name,
                        e,
                        exc_info=True,
                    )
        return None

    @staticmethod
    def _wrap_single_finding(finding: Finding) -> Investigation:
        """Wrap an isolated, single finding in an Investigation container."""
        # Score calculation for single evidence
        ev_score = finding.severity.numeric_weight * 100.0
        confidence = finding.confidence
        priority = round(ev_score * 0.7 + confidence * 20.0, 1)

        category = finding.metadata.get("category", "quality_issue")

        # Map categories to root causes & recommendations
        causes = [
            f"Localized anomalies detected by the {finding.module} module.",
            "Isolated recording issue or outlier observation."
        ]

        return Investigation(
            title=finding.title,
            summary=finding.description,
            hypothesis=f"Isolated data quality symptom of type '{category}' detected in column(s) {finding.affected_columns}.",
            supporting_findings=[finding.id],
            evidence_score=ev_score,
            confidence=confidence,
            priority=priority,
            possible_causes=causes,
            recommended_next_steps=[finding.recommendation],
            affected_columns=finding.affected_columns,
            metadata={
                "rule_applied": "single_evidence_wrapper",
                "original_category": category,
            },
        )

    @staticmethod
    def _wrap_health_score_finding(finding: Finding) -> Investigation:
        """Wrap the overall structural health score summary finding."""
        ev_score = finding.severity.numeric_weight * 100.0
        confidence = finding.confidence

        return Investigation(
            title=finding.title,
            summary=finding.description,
            hypothesis="Unified structural integrity assessment score of the dataset.",
            supporting_findings=[finding.id],
            evidence_score=ev_score,
            confidence=confidence,
            priority=0.0,  # Summaries sit at the bottom of the priority ranking
            possible_causes=["Composite health score based on all structural checks."],
            recommended_next_steps=[finding.recommendation],
            affected_columns=[],
            metadata={
                "rule_applied": "structural_health_summary",
                "original_category": "structural_health_score",
            },
        )
