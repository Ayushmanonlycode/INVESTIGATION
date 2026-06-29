"""Fusion Rules — extensible rule-based reasoning engine.

Defines the BaseFusionRule interface and concrete fusion rules that match
patterns in groups of related findings and synthesize them into Investigations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from investigation_engine.models.finding import Finding
from investigation_engine.models.investigation_hypothesis import Investigation
from investigation_engine.models.severity import Severity


class BaseFusionRule(ABC):
    """Abstract base class for all Evidence Fusion Rules.

    Rules evaluate a group of related findings and determine if they should
    be fused into a single unified Investigation.
    """

    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Unique identifier for this rule."""
        ...

    @abstractmethod
    def matches(self, findings: list[Finding]) -> bool:
        """Determine if this rule is applicable to the group of findings."""
        ...

    @abstractmethod
    def fuse(self, findings: list[Finding]) -> Investigation:
        """Synthesize findings into an Investigation.

        Must only reference existing evidence from findings (no hallucinated facts).
        """
        ...

    # ── Shared Heuristics ─────────────────────────────────────────────

    @staticmethod
    def _union_columns(findings: list[Finding]) -> list[str]:
        """Get union of all affected columns in findings."""
        cols: set[str] = set()
        for f in findings:
            cols.update(f.affected_columns)
        return sorted(cols)

    @staticmethod
    def _compute_metrics(findings: list[Finding]) -> tuple[float, float, float]:
        """Compute average evidence score, confidence, and priority.

        Returns:
            Tuple of (evidence_score, confidence, priority).
        """
        if not findings:
            return 0.0, 0.0, 0.0

        # Evidence score is the average of severity weights
        total_sev_weight = sum(f.severity.numeric_weight * 100 for f in findings)
        evidence_score = round(total_sev_weight / len(findings), 1)

        # Confidence increases slightly when multiple independent findings support the same hypothesis
        avg_confidence = sum(f.confidence for f in findings) / len(findings)
        # Apply corroboration bonus: +5% per additional finding, capped at 1.0
        bonus = min(0.15, (len(findings) - 1) * 0.05)
        confidence = round(min(1.0, avg_confidence + bonus), 2)

        # Temporary Priority heuristic: weighted average of severity, confidence, and count
        max_severity_weight = max(f.severity.numeric_weight for f in findings)
        # Base priority from max severity (up to 70 pts)
        base_priority = max_severity_weight * 70.0
        # Confidence contribution (up to 20 pts)
        conf_priority = confidence * 20.0
        # Support corroboration count contribution (up to 10 pts)
        count_bonus = min(10.0, (len(findings) - 1) * 3.0)

        priority = round(min(100.0, base_priority + conf_priority + count_bonus), 1)

        return evidence_score, confidence, priority


# ═══════════════════════════════════════════════════════════════════════
#  1. MISSING VALUES FUSION RULE
# ═══════════════════════════════════════════════════════════════════════

class MissingValuesRule(BaseFusionRule):
    """Fuses high missingness and missingness correlations for related columns."""

    @property
    def rule_name(self) -> str:
        return "missing_values_fusion"

    def matches(self, findings: list[Finding]) -> bool:
        categories = {f.metadata.get("category") for f in findings}
        # Matches if we have missing values and possibly missingness relationships
        return bool(
            "missing_values" in categories
            or "missingness_relationships" in categories
        )

    def fuse(self, findings: list[Finding]) -> Investigation:
        cols = self._union_columns(findings)
        ev_score, conf, priority = self._compute_metrics(findings)

        # Separate individual missingness vs relationship findings
        missing_findings = [f for f in findings if f.metadata.get("category") == "missing_values"]
        rel_findings = [f for f in findings if f.metadata.get("category") == "missingness_relationships"]

        # Synthesize title and details
        if len(cols) == 1:
            title = f"Systematic missingness in column '{cols[0]}'"
            summary = (
                f"Column '{cols[0]}' has significant missingness "
                f"({missing_findings[0].evidence.get('missing_ratio', 0.0):.1%} missing). "
                f"No surrounding missingness correlation was found, indicating isolated dropout."
            )
        else:
            title = f"Correlated missingness across related feature family ({', '.join(cols[:3])})"
            summary = (
                f"Significant and highly correlated missingness detected across "
                f"{len(cols)} columns: {', '.join(cols)}. "
            )
            if rel_findings:
                summary += f"Perfect/high missingness correlation ({rel_findings[0].evidence.get('missingness_correlation', 0.0)}) was observed."

        # Hypothesis & Causes based strictly on evidence
        hypothesis = (
            f"Values in columns {cols} are missing systematically rather than randomly (MNAR/MAR). "
            f"This is supported by {len(findings)} independent indicators."
        )

        possible_causes = [
            "Upstream telemetry or collection pipeline dropout.",
            "Conditional data recording (e.g. values only logged under specific states).",
            "Hardware sensor limitation or failure in specific operating conditions."
        ]
        next_steps = [
            f"Check ingestion logs for the columns: {cols}.",
            "Perform a profile split to see if missingness is correlated with a specific state (e.g. root_cause).",
            "Verify if these columns should be imputed or dropped during model training."
        ]

        return Investigation(
            title=title,
            summary=summary,
            hypothesis=hypothesis,
            supporting_findings=[f.id for f in findings],
            evidence_score=ev_score,
            confidence=conf,
            priority=priority,
            possible_causes=possible_causes,
            recommended_next_steps=next_steps,
            affected_columns=cols,
            metadata={"rule_applied": self.rule_name},
        )


# ═══════════════════════════════════════════════════════════════════════
#  2. CONSTANT & LOW-VARIANCE FUSION RULE
# ═══════════════════════════════════════════════════════════════════════

class ConstantFeatureRule(BaseFusionRule):
    """Fuses constant, near-constant, and low-cardinality findings."""

    @property
    def rule_name(self) -> str:
        return "constant_feature_fusion"

    def matches(self, findings: list[Finding]) -> bool:
        categories = {f.metadata.get("category") for f in findings}
        return bool(
            "constant_features" in categories
            or "near_constant_features" in categories
            or "cardinality" in categories
        )

    def fuse(self, findings: list[Finding]) -> Investigation:
        cols = self._union_columns(findings)
        ev_score, conf, priority = self._compute_metrics(findings)

        # Detect the type of feature issues
        const_findings = [f for f in findings if f.metadata.get("category") == "constant_features"]
        near_const = [f for f in findings if f.metadata.get("category") == "near_constant_features"]

        if const_findings:
            title = f"Redundant constant columns detected: {', '.join(cols)}"
            summary = (
                f"Columns {cols} have zero variance (only a single unique value). "
                f"These columns provide no analytical or predictive power."
            )
            hypothesis = "These features represent fixed parameters or system constants that do not vary in this dataset sample."
            possible_causes = [
                "The dataset represents a single operational state where this parameter is fixed.",
                "Data collection error failed to capture parameter variance.",
                "Leftover helper or constant indicator columns from preprocessing."
            ]
            next_steps = [
                f"Safely remove constant columns {cols} from downstream modelling pipelines.",
                "Verify if these variables vary in other dataset partitions or time ranges."
            ]
        else:
            title = f"Near-constant / extremely low-variance features: {', '.join(cols)}"
            summary = (
                f"Columns {cols} show extremely low variation or cardinality, dominated almost "
                f"entirely by a single value (e.g. >95% dominance)."
            )
            hypothesis = "These features have highly skewed distributions with insufficient variance to contribute signal to ML models."
            possible_causes = [
                "Rare event characteristics (imbalanced categorical indicator).",
                "Ingestion pipeline defaults setting most values to a default fallback value."
            ]
            next_steps = [
                f"Check the distribution of minority values in columns {cols}.",
                "Consider converting these to binary indicators or merging rare classes."
            ]

        return Investigation(
            title=title,
            summary=summary,
            hypothesis=hypothesis,
            supporting_findings=[f.id for f in findings],
            evidence_score=ev_score,
            confidence=conf,
            priority=priority,
            possible_causes=possible_causes,
            recommended_next_steps=next_steps,
            affected_columns=cols,
            metadata={"rule_applied": self.rule_name},
        )


# ═══════════════════════════════════════════════════════════════════════
#  3. DUPLICATE FEATURE FUSION RULE
# ═══════════════════════════════════════════════════════════════════════

class DuplicateFeatureRule(BaseFusionRule):
    """Fuses duplicate rows, duplicate features, or high-cardinality features."""

    @property
    def rule_name(self) -> str:
        return "duplicate_feature_fusion"

    def matches(self, findings: list[Finding]) -> bool:
        categories = {f.metadata.get("category") for f in findings}
        return bool(
            "duplicates" in categories
            or "duplicate_features" in categories
        )

    def fuse(self, findings: list[Finding]) -> Investigation:
        cols = self._union_columns(findings)
        ev_score, conf, priority = self._compute_metrics(findings)

        dup_rows = [f for f in findings if f.metadata.get("category") == "duplicates"]
        dup_feats = [f for f in findings if f.metadata.get("category") == "duplicate_features"]

        if dup_feats and dup_rows:
            title = "Severe record duplication and redundant columns detected"
            summary = "The dataset has both exact duplicate rows and identical columns, indicating general data quality issues."
            hypothesis = "Data pipeline joins or concatenation failed, multiplying records and repeating features."
        elif dup_feats:
            title = f"Redundant identical columns: {', '.join(cols)}"
            summary = f"Columns {cols} contain identical values across all rows, representing duplicate information."
            hypothesis = "Features were duplicated during extraction, or represent synonyms (e.g. original vs scaled values)."
        else:
            title = f"Duplicate rows detected ({len(dup_rows)} indicators)"
            summary = "The dataset contains exact duplicate rows."
            hypothesis = "System logs or network packets were recorded multiple times, or database join fan-out occurred."

        possible_causes = [
            "Flawed ETL or pipeline join logic creating record duplication.",
            "Synonymous or copy-pasted columns created during database export.",
            "Multiple ingestion workers writing duplicate transaction logs."
        ]
        next_steps = [
            "Perform deduplication using df.drop_duplicates().",
            f"Remove redundant identical columns: {cols[1:]} (keeping only {cols[0]})."
        ]

        return Investigation(
            title=title,
            summary=summary,
            hypothesis=hypothesis,
            supporting_findings=[f.id for f in findings],
            evidence_score=ev_score,
            confidence=conf,
            priority=priority,
            possible_causes=possible_causes,
            recommended_next_steps=next_steps,
            affected_columns=cols,
            metadata={"rule_applied": self.rule_name},
        )


# ═══════════════════════════════════════════════════════════════════════
#  4. IDENTIFIER INTEGRITY FUSION RULE
# ═══════════════════════════════════════════════════════════════════════

class IdentifierRule(BaseFusionRule):
    """Fuses identifier-related findings."""

    @property
    def rule_name(self) -> str:
        return "identifier_fusion"

    def matches(self, findings: list[Finding]) -> bool:
        categories = {f.metadata.get("category") for f in findings}
        return "identifiers" in categories

    def fuse(self, findings: list[Finding]) -> Investigation:
        cols = self._union_columns(findings)
        ev_score, conf, priority = self._compute_metrics(findings)

        title = f"Identifier integrity failure on column(s): {', '.join(cols)}"
        summary = (
            f"Identifier column(s) {cols} contain nulls or duplicates, "
            f"violating entity integrity constraints."
        )
        hypothesis = "Primary keys or record keys have corrupted values, risking incorrect linkage and data aggregation errors."
        possible_causes = [
            "Key generation algorithm allowed duplicate values.",
            "Records were merged from multiple databases without primary key re-indexing.",
            "Null keys injected during partial data recording or left joins."
        ]
        next_steps = [
            f"Find row indices with non-unique or null identifiers in {cols}.",
            "Resolve key conflicts before performing joins or database uploads."
        ]

        return Investigation(
            title=title,
            summary=summary,
            hypothesis=hypothesis,
            supporting_findings=[f.id for f in findings],
            evidence_score=ev_score,
            confidence=conf,
            priority=priority,
            possible_causes=possible_causes,
            recommended_next_steps=next_steps,
            affected_columns=cols,
            metadata={"rule_applied": self.rule_name},
        )


# ═══════════════════════════════════════════════════════════════════════
#  5. GENERIC FALLBACK FUSION RULE
# ═══════════════════════════════════════════════════════════════════════

class GenericGroupRule(BaseFusionRule):
    """Fallback rule to combine any connected component findings that don't match specific rules."""

    @property
    def rule_name(self) -> str:
        return "generic_fallback_fusion"

    def matches(self, findings: list[Finding]) -> bool:
        return len(findings) > 1

    def fuse(self, findings: list[Finding]) -> Investigation:
        cols = self._union_columns(findings)
        ev_score, conf, priority = self._compute_metrics(findings)

        categories = {f.metadata.get("category", "unknown") for f in findings}
        title = f"Co-occurring data issues affecting columns: {', '.join(cols[:3])}"
        summary = (
            f"Multiple data quality issues ({', '.join(categories)}) are co-occurring "
            f"on columns: {cols}. This suggests a cluster of related data anomalies."
        )
        hypothesis = "Multiple symptoms point to localized data corruption, collection dropouts, or configuration issues in these columns."
        possible_causes = [
            "Co-dependent database table ingestion issues.",
            "Localization of faults inside specific sub-components of the telemetry collection."
        ]
        next_steps = [
            f"Investigate columns {cols} for general data quality issues.",
            "Run subset checks to see if these anomalies are concentrated in specific partitions."
        ]

        return Investigation(
            title=title,
            summary=summary,
            hypothesis=hypothesis,
            supporting_findings=[f.id for f in findings],
            evidence_score=ev_score,
            confidence=conf,
            priority=priority,
            possible_causes=possible_causes,
            recommended_next_steps=next_steps,
            affected_columns=cols,
            metadata={"rule_applied": self.rule_name},
        )
