"""Explainable similarity scoring for graph-based evidence compression."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from investigation_engine.config.settings import EvidenceCompressionSettings
from investigation_engine.models.finding import Finding

_STOPWORDS = {
    "column",
    "columns",
    "feature",
    "features",
    "value",
    "values",
    "detected",
    "relationship",
    "community",
    "group",
    "related",
    "systematic",
    "finding",
    "findings",
    "synthetic",
    "integrity",
}

_NUMERIC_EVIDENCE_KEYS = (
    "correlation_coefficient",
    "p_value",
    "mutual_information_score",
    "vif",
    "missing_ratio",
    "missingness_correlation",
    "duplicate_ratio",
    "dominance_ratio",
    "health_score",
)


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """Full explainable similarity score between two findings."""

    total_weight: float
    structural_similarity: float
    statistical_similarity: float
    semantic_similarity: float
    signals: dict[str, float]
    reasons: list[str]


class FindingSimilarityScorer:
    """Computes deterministic weighted similarity between findings."""

    def __init__(self, config: EvidenceCompressionSettings) -> None:
        self._config = config

    def score(self, left: Finding, right: Finding) -> SimilarityResult:
        left_family = self.feature_family(left)
        right_family = self.feature_family(right)
        structural_signals = self._structural_signals(left, right)
        statistical_signals = self._statistical_signals(left, right)
        semantic_signals = self._semantic_signals(left, right)

        structural_similarity = self._weighted_average(
            structural_signals,
            {
                "shared_columns": self._config.shared_columns_weight,
                "shared_rows": self._config.shared_rows_weight,
                "shared_metadata": self._config.shared_metadata_weight,
                "same_investigator": self._config.same_investigator_weight,
                "feature_family": self._config.feature_family_weight,
            },
        )
        statistical_similarity = self._weighted_average(
            statistical_signals,
            {
                "numeric_evidence": self._config.numeric_evidence_similarity_weight,
                "severity": self._config.severity_similarity_weight,
                "confidence": self._config.confidence_similarity_weight,
            },
        )
        semantic_similarity = self._weighted_average(
            semantic_signals,
            {
                "prefix": self._config.semantic_prefix_weight,
                "token_overlap": self._config.semantic_token_weight,
                "domain_terms": self._config.semantic_domain_term_weight,
            },
        )

        total_weight = self._weighted_average(
            {
                "structural": structural_similarity,
                "statistical": statistical_similarity,
                "semantic": semantic_similarity,
            },
            {
                "structural": self._config.structural_weight,
                "statistical": self._config.statistical_weight,
                "semantic": self._config.semantic_weight,
            },
        )
        if (
            left_family is not None
            and right_family is not None
            and left_family != right_family
            and structural_signals["shared_columns"] == 0.0
        ):
            total_weight *= 0.15

        reasons = [
            f"{name.replace('_', ' ')}={value:.2f}"
            for name, value in (
                list(structural_signals.items())
                + list(statistical_signals.items())
                + list(semantic_signals.items())
            )
            if value >= 0.20
        ]

        return SimilarityResult(
            total_weight=round(total_weight, 4),
            structural_similarity=round(structural_similarity, 4),
            statistical_similarity=round(statistical_similarity, 4),
            semantic_similarity=round(semantic_similarity, 4),
            signals={
                **structural_signals,
                **statistical_signals,
                **semantic_signals,
            },
            reasons=sorted(reasons),
        )

    def feature_family(self, finding: Finding) -> str | None:
        columns = tuple(sorted(finding.affected_columns))
        return self._feature_family(columns)

    def semantic_tokens(self, finding: Finding) -> set[str]:
        return set(self._semantic_tokens(finding.title, tuple(sorted(finding.affected_columns))))

    def _structural_signals(self, left: Finding, right: Finding) -> dict[str, float]:
        left_columns = set(left.affected_columns)
        right_columns = set(right.affected_columns)
        left_rows = set(left.affected_rows or [])
        right_rows = set(right.affected_rows or [])

        shared_columns = self._jaccard(left_columns, right_columns)
        shared_rows = self._jaccard(left_rows, right_rows)
        shared_metadata = self._metadata_similarity(left.metadata, right.metadata)
        same_investigator = 1.0 if left.module == right.module else 0.0
        left_family = self.feature_family(left)
        right_family = self.feature_family(right)
        feature_family = (
            1.0
            if left_family is not None and left_family == right_family
            else 0.0
        )

        return {
            "shared_columns": shared_columns,
            "shared_rows": shared_rows,
            "shared_metadata": shared_metadata,
            "same_investigator": same_investigator,
            "feature_family": feature_family,
        }

    def _statistical_signals(self, left: Finding, right: Finding) -> dict[str, float]:
        numeric_evidence = self._numeric_evidence_similarity(left.evidence, right.evidence)
        left_severity = left.severity.numeric_weight
        right_severity = right.severity.numeric_weight
        severity_similarity = 1.0 - abs(left_severity - right_severity)
        confidence_similarity = 1.0 - abs(left.confidence - right.confidence)

        return {
            "numeric_evidence": max(0.0, min(1.0, numeric_evidence)),
            "severity": max(0.0, min(1.0, severity_similarity)),
            "confidence": max(0.0, min(1.0, confidence_similarity)),
        }

    def _semantic_signals(self, left: Finding, right: Finding) -> dict[str, float]:
        left_family = self.feature_family(left)
        right_family = self.feature_family(right)
        prefix_similarity = 1.0 if left_family is not None and left_family == right_family else 0.0

        left_tokens = self.semantic_tokens(left)
        right_tokens = self.semantic_tokens(right)
        token_overlap = self._jaccard(left_tokens, right_tokens)

        left_domain_terms = self._domain_terms(left)
        right_domain_terms = self._domain_terms(right)
        domain_terms = self._jaccard(left_domain_terms, right_domain_terms)

        return {
            "prefix": prefix_similarity,
            "token_overlap": token_overlap,
            "domain_terms": domain_terms,
        }

    @staticmethod
    def _weighted_average(values: dict[str, float], weights: dict[str, float]) -> float:
        numerator = 0.0
        denominator = 0.0
        for key, value in values.items():
            weight = weights.get(key, 0.0)
            numerator += value * weight
            denominator += weight
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _jaccard(left: set[Any], right: set[Any]) -> float:
        if not left and not right:
            return 0.0
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _metadata_similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        left_keys = {
            key for key, value in left.items() if isinstance(value, (str, int, float, bool))
        }
        right_keys = {
            key for key, value in right.items() if isinstance(value, (str, int, float, bool))
        }
        if not left_keys and not right_keys:
            return 0.0

        shared_keys = left_keys & right_keys
        if not shared_keys:
            return 0.0

        matches = 0.0
        for key in shared_keys:
            left_value = left.get(key)
            right_value = right.get(key)
            if left_value == right_value:
                matches += 1.0
            elif isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                matches += self._relative_similarity(float(left_value), float(right_value))

        return matches / len(shared_keys)

    def _numeric_evidence_similarity(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> float:
        similarities: list[float] = []
        for key in _NUMERIC_EVIDENCE_KEYS:
            left_value = self._as_float(left.get(key))
            right_value = self._as_float(right.get(key))
            if left_value is None or right_value is None:
                continue
            if key == "correlation_coefficient":
                similarities.append(self._correlation_similarity(left_value, right_value))
            else:
                similarities.append(self._relative_similarity(left_value, right_value))

        if not similarities:
            return 0.0
        return sum(similarities) / len(similarities)

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            number = float(value)
            return number if math.isfinite(number) else None
        return None

    def _correlation_similarity(self, left: float, right: float) -> float:
        magnitude_similarity = 1.0 - min(abs(abs(left) - abs(right)), 1.0)
        if left == 0.0 or right == 0.0:
            return magnitude_similarity
        same_direction_bonus = 1.0 if math.copysign(1.0, left) == math.copysign(1.0, right) else 0.2
        return magnitude_similarity * same_direction_bonus

    @staticmethod
    def _relative_similarity(left: float, right: float) -> float:
        scale = max(abs(left), abs(right), 1.0)
        return max(0.0, 1.0 - abs(left - right) / scale)

    def _domain_terms(self, finding: Finding) -> set[str]:
        tokens = self.semantic_tokens(finding)
        return {
            token for token in tokens
            if token.isalpha() or any(char.isdigit() for char in token)
        }

    @staticmethod
    @lru_cache(maxsize=4096)
    def _feature_family(columns: tuple[str, ...]) -> str | None:
        if not columns:
            return None
        prefixes: list[str] = []
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
        family, count = Counter(prefixes).most_common(1)[0]
        return family if count >= max(1, len(columns) // 2) else None

    @staticmethod
    @lru_cache(maxsize=8192)
    def _semantic_tokens(title: str, columns: tuple[str, ...]) -> tuple[str, ...]:
        tokens: set[str] = set()
        text_parts = [title, *columns]
        for part in text_parts:
            for token in re.split(r"[^a-zA-Z0-9]+", part.lower()):
                if len(token) < 3 or token in _STOPWORDS:
                    continue
                tokens.add(token)
        return tuple(sorted(tokens))
