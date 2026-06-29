"""Rule-based evidence compression."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter

from investigation_engine.models.finding import Finding
from investigation_engine.models.severity import Severity
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.provenance.tracker import ProvenanceTracker


def _title_case_token(token: str) -> str:
    return token.replace("_", " ").strip().title()


class BaseEvidenceCompressionRule(ABC):
    """Compresses related findings into a single evidence unit."""

    category: str

    @property
    @abstractmethod
    def rule_name(self) -> str:
        ...

    @abstractmethod
    def matches(self, findings: list[Finding]) -> bool:
        ...

    def compress(self, findings: list[Finding]) -> EvidenceUnit:
        columns = self._union_columns(findings)
        rows = self._union_rows(findings)
        modules = sorted({finding.module for finding in findings})
        confidences = [finding.confidence for finding in findings]
        severity_weight = max(finding.severity.numeric_weight for finding in findings)
        module_bonus = min(0.1, 0.05 * max(len(modules) - 1, 0))
        confidence = round(min(1.0, sum(confidences) / len(confidences) + module_bonus), 2)
        strength = round(min(100.0, severity_weight * 60.0 + confidence * 40.0), 1)
        title, summary = self._compose_text(findings, columns)

        return EvidenceUnit(
            category=self.category,
            title=title,
            summary=summary,
            supporting_findings=[finding.id for finding in findings],
            provenance={
                **ProvenanceTracker.for_findings(findings),
                "compression_rule": self.rule_name,
            },
            confidence=confidence,
            strength=strength,
            affected_columns=columns,
            affected_rows=rows,
            metadata={
                "compression_rule": self.rule_name,
                "source_categories": sorted({
                    str(finding.metadata.get("category", "unknown")) for finding in findings
                }),
                "semantic_label": self._semantic_label(findings, columns),
                "concept_key": self._concept_key(findings, columns),
                "max_severity": self._max_severity(findings).value,
                "source_modules": modules,
            },
        )

    @abstractmethod
    def _compose_text(self, findings: list[Finding], columns: list[str]) -> tuple[str, str]:
        ...

    def _semantic_label(self, findings: list[Finding], columns: list[str]) -> str:
        family = self._feature_family(columns)
        if family is not None:
            return f"{_title_case_token(family)} Measurements"
        if columns:
            return f"{_title_case_token(columns[0])} Feature Set"
        return self.category

    def _concept_key(self, findings: list[Finding], columns: list[str]) -> str:
        family = self._feature_family(columns)
        if family is not None:
            return family.lower()
        if columns:
            return columns[0].lower()
        return self.category.lower().replace(" ", "_")

    @staticmethod
    def _union_columns(findings: list[Finding]) -> list[str]:
        return sorted({column for finding in findings for column in finding.affected_columns})

    @staticmethod
    def _union_rows(findings: list[Finding]) -> list[int] | None:
        rows = sorted({
            row
            for finding in findings
            for row in (finding.affected_rows or [])
        })
        return rows or None

    @staticmethod
    def _feature_family(columns: list[str]) -> str | None:
        if not columns:
            return None
        prefixes = []
        for column in columns:
            match = re.match(r"^(neighbor\d+|neighbor|serving|target)", column, re.IGNORECASE)
            if match:
                prefixes.append(match.group(1).lower())
                continue
            tokens = column.split("_")
            if len(tokens) > 1 and len(tokens[0]) > 2:
                prefixes.append(tokens[0].lower())
        if not prefixes:
            return None
        most_common, count = Counter(prefixes).most_common(1)[0]
        return most_common if count >= max(1, len(columns) // 2) else None

    @staticmethod
    def _max_severity(findings: list[Finding]) -> Severity:
        return max(findings, key=lambda finding: finding.severity.numeric_weight).severity


class CorrelationCommunityRule(BaseEvidenceCompressionRule):
    category = "Correlation Community"

    @property
    def rule_name(self) -> str:
        return "correlation_community"

    def matches(self, findings: list[Finding]) -> bool:
        relationship_categories = {
            "pearson_correlation",
            "spearman_correlation",
            "mutual_information",
            "relationship_communities",
            "strong_feature_groups",
        }
        return any(
            finding.metadata.get("category") in relationship_categories
            for finding in findings
        )

    def _compose_text(self, findings: list[Finding], columns: list[str]) -> tuple[str, str]:
        label = self._semantic_label(findings, columns)
        title = f"{label} form a correlation community"
        summary = (
            f"{len(columns)} columns exhibit related statistical behavior across "
            f"{len(findings)} relationship finding(s). The pattern is treated as one analytical "
            "fact rather than many pairwise repetitions."
        )
        return title, summary


class RedundantFeatureGroupRule(BaseEvidenceCompressionRule):
    category = "Redundant Feature Group"

    @property
    def rule_name(self) -> str:
        return "redundant_feature_group"

    def matches(self, findings: list[Finding]) -> bool:
        categories = {"redundant_features", "duplicate_features"}
        return any(finding.metadata.get("category") in categories for finding in findings)

    def _compose_text(self, findings: list[Finding], columns: list[str]) -> tuple[str, str]:
        label = self._semantic_label(findings, columns)
        return (
            f"{label} contain redundant features",
            (
                f"Columns {columns} repeat substantially the same signal. The compressed evidence "
                "captures redundancy as one fact so downstream prioritization does not "
                "reward verbosity."
            ),
        )


class SystematicMissingnessRule(BaseEvidenceCompressionRule):
    category = "Systematic Missingness"

    @property
    def rule_name(self) -> str:
        return "systematic_missingness"

    def matches(self, findings: list[Finding]) -> bool:
        categories = {"missing_values", "missingness_relationships"}
        return any(finding.metadata.get("category") in categories for finding in findings)

    def _compose_text(self, findings: list[Finding], columns: list[str]) -> tuple[str, str]:
        label = self._semantic_label(findings, columns)
        return (
            f"{label} appear systematically unavailable",
            (
                f"Missingness signals across {columns} are compressed into one pattern because the "
                "evidence points to structured unavailability rather than isolated null counts."
            ),
        )


class IdentifierIntegrityFailureRule(BaseEvidenceCompressionRule):
    category = "Identifier Integrity Failure"

    @property
    def rule_name(self) -> str:
        return "identifier_integrity_failure"

    def matches(self, findings: list[Finding]) -> bool:
        categories = {"identifiers"}
        return any(finding.metadata.get("category") in categories for finding in findings)

    def _compose_text(self, findings: list[Finding], columns: list[str]) -> tuple[str, str]:
        return (
            "Identifier integrity failure detected",
            (
                f"Identifier-related findings affecting {columns} are compressed into a single "
                "integrity fact because uniqueness and completeness failures jointly "
                "threaten linkage."
            ),
        )


class ConstantFeatureGroupRule(BaseEvidenceCompressionRule):
    category = "Constant Feature Group"

    @property
    def rule_name(self) -> str:
        return "constant_feature_group"

    def matches(self, findings: list[Finding]) -> bool:
        return any(finding.metadata.get("category") == "constant_features" for finding in findings)

    def _compose_text(self, findings: list[Finding], columns: list[str]) -> tuple[str, str]:
        return (
            "Constant features provide no variation",
            f"Columns {columns} behave as a single zero-variance artifact group.",
        )


class ConfigurationArtifactRule(BaseEvidenceCompressionRule):
    category = "Configuration Artifact"

    @property
    def rule_name(self) -> str:
        return "configuration_artifact"

    def matches(self, findings: list[Finding]) -> bool:
        return any(finding.metadata.get("category") == "datatype_integrity" for finding in findings)

    def _compose_text(self, findings: list[Finding], columns: list[str]) -> tuple[str, str]:
        return (
            "Configuration or typing artifact detected",
            f"Columns {columns} show evidence consistent with schema, configuration, "
            "or ingestion artifacts.",
        )


class LowInformationFeatureGroupRule(BaseEvidenceCompressionRule):
    category = "Low Information Feature Group"

    @property
    def rule_name(self) -> str:
        return "low_information_feature_group"

    def matches(self, findings: list[Finding]) -> bool:
        categories = {"near_constant_features", "cardinality"}
        return any(finding.metadata.get("category") in categories for finding in findings)

    def _compose_text(self, findings: list[Finding], columns: list[str]) -> tuple[str, str]:
        return (
            "Low-information features detected",
            (
                f"Columns {columns} contribute limited independent information because variance or "
                "cardinality is too low."
            ),
        )


class OutlierCommunityRule(BaseEvidenceCompressionRule):
    category = "Outlier Community"

    @property
    def rule_name(self) -> str:
        return "outlier_community"

    def matches(self, findings: list[Finding]) -> bool:
        return any("outlier" in str(finding.metadata.get("category", "")) for finding in findings)

    def _compose_text(self, findings: list[Finding], columns: list[str]) -> tuple[str, str]:
        return (
            "Outlier community detected",
            f"Findings across {columns} indicate related extreme-value behavior.",
        )


class ClusterCommunityRule(BaseEvidenceCompressionRule):
    category = "Cluster Community"

    @property
    def rule_name(self) -> str:
        return "cluster_community"

    def matches(self, findings: list[Finding]) -> bool:
        return any("cluster" in str(finding.metadata.get("category", "")) for finding in findings)

    def _compose_text(self, findings: list[Finding], columns: list[str]) -> tuple[str, str]:
        return (
            "Clustered behavior detected",
            f"Findings across {columns} point to a shared cluster structure that deserves review.",
        )


class GenericEvidenceRule(BaseEvidenceCompressionRule):
    category = "Configuration Artifact"

    @property
    def rule_name(self) -> str:
        return "generic_evidence_wrapper"

    def matches(self, findings: list[Finding]) -> bool:
        return True

    def _compose_text(self, findings: list[Finding], columns: list[str]) -> tuple[str, str]:
        categories = sorted({
            str(finding.metadata.get("category", "unknown")) for finding in findings
        })
        return (
            "Compressed analytical evidence",
            (
                f"Findings touching {columns or ['dataset-wide scope']} were compressed "
                "without loss "
                f"under categories {categories}."
            ),
        )
