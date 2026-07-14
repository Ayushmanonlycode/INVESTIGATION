"""Relationship Investigator — autonomous relationship discovery.

Discovers statistically meaningful relationships between structured features
and turns them into analyst-facing findings rather than raw statistics.
"""

from __future__ import annotations

import math
import time
from itertools import combinations
from typing import Any, ClassVar, Literal, cast

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as scipy_stats  # type: ignore[import-untyped]
from sklearn.feature_selection import (  # type: ignore[import-untyped]
    mutual_info_classif,
    mutual_info_regression,
)

from investigation_engine.config.settings import RelationshipSettings, Settings
from investigation_engine.core.plugin import register_module
from investigation_engine.models.dataset import DatasetInfo
from investigation_engine.models.finding import Finding
from investigation_engine.models.severity import Severity
from investigation_engine.modules.base import BaseInvestigationModule


def _severity_from_strength(strength: float, high: float, critical: float) -> Severity:
    """Map relationship strength to a severity tier."""
    if strength >= critical:
        return Severity.HIGH
    if strength >= high:
        return Severity.MEDIUM
    return Severity.LOW


def _safe_float(value: Any) -> float | None:
    """Convert numeric-looking values to finite floats."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


@register_module
class RelationshipInvestigator(BaseInvestigationModule):
    """Discovers pairwise and grouped feature relationships worth investigation."""

    name: ClassVar[str] = "relationship_investigator"
    description: ClassVar[str] = (
        "Investigates statistically significant relationships between structured "
        "features, including correlation, redundancy, multicollinearity, mutual "
        "information, and related feature communities."
    )
    version: ClassVar[str] = "0.1.0"
    tags: ClassVar[list[str]] = ["relationships", "statistics", "features"]

    def can_run(self, df: pd.DataFrame, dataset_info: DatasetInfo) -> bool:
        return dataset_info.row_count >= 2 and len(dataset_info.numeric_columns) >= 2

    def get_skip_reason(self, df: pd.DataFrame, dataset_info: DatasetInfo) -> str | None:
        if dataset_info.row_count < 2:
            return "Dataset must contain at least two rows."
        if len(dataset_info.numeric_columns) < 2:
            return "Relationship Investigator requires at least two numeric columns."
        return None

    def investigate(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        config: Settings,
    ) -> list[Finding]:
        cfg = config.relationship
        findings: list[Finding] = []
        profiles = {profile.name: profile for profile in dataset_info.columns}

        numeric_columns = [
            col
            for col in dataset_info.numeric_columns
            if (
                col in df.columns
                and pd.api.types.is_numeric_dtype(df[col])
                and profiles[col].unique_count > 1
            )
        ]
        if len(numeric_columns) < 2:
            return findings

        # All selected columns are already known to be numeric, so coercion is unnecessary.
        numeric_df = df.loc[:, numeric_columns]
        pair_metrics: dict[tuple[str, str], dict[str, Any]] = {}
        full_analysis = config.engine.analysis_profile == "full"

        investigations = [
            ("correlations", self._investigate_correlations),
            ("mutual_information", self._investigate_mutual_information),
            ("multicollinearity", self._investigate_multicollinearity),
            ("groups", self._investigate_feature_groups),
        ]

        for domain_name, method in investigations:
            domain_start = time.perf_counter()
            try:
                findings.extend(
                    method(
                        numeric_df,
                        df,
                        dataset_info,
                        cfg,
                        pair_metrics,
                        full_analysis,
                    )
                )
            except Exception as exc:
                logger.error(
                    "Relationship/{} failed: {}",
                    domain_name,
                    exc,
                    exc_info=True,
                )
            finally:
                logger.info(
                    "Relationship/{} completed in {:.3f}s",
                    domain_name,
                    time.perf_counter() - domain_start,
                )

        findings = self._cap_findings(findings, config.engine.max_findings_per_module)
        return self.validate_findings(findings)

    def _investigate_correlations(
        self,
        numeric_df: pd.DataFrame,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: RelationshipSettings,
        pair_metrics: dict[tuple[str, str], dict[str, Any]],
        full_analysis: bool,
    ) -> list[Finding]:
        findings: list[Finding] = []

        for method, threshold, balanced_max_rows in (
            ("pearson", cfg.pearson_threshold, cfg.pearson_max_rows),
            ("spearman", cfg.spearman_threshold, cfg.spearman_max_rows),
        ):
            analysis_df = self._sample_rows(
                numeric_df,
                None if full_analysis else balanced_max_rows,
            )
            correlation_method = cast(Literal["pearson", "spearman"], method)
            corr_matrix = analysis_df.corr(
                method=correlation_method,
                min_periods=cfg.min_pair_observations,
            )
            flagged_pairs = self._iter_flagged_pairs(corr_matrix, threshold)

            # Retain every strong edge for group discovery without performing
            # another row scan or constructing an individual finding for each.
            for left, right, coefficient in flagged_pairs:
                key = self._pair_key(left, right)
                metrics = pair_metrics.setdefault(key, self._base_pair_metrics(cfg))
                metrics[f"{method}_correlation"] = coefficient
                if metrics["correlation_coefficient"] is None or abs(coefficient) > abs(
                    metrics["correlation_coefficient"]
                ):
                    metrics["correlation_coefficient"] = coefficient

            candidates = (
                flagged_pairs
                if full_analysis
                else flagged_pairs[: cfg.max_correlation_candidates_per_method]
            )
            for left, right, coefficient in candidates:
                key = self._pair_key(left, right)
                metrics = pair_metrics[key]
                pair_df = analysis_df.loc[:, [left, right]].dropna()
                if len(pair_df) < cfg.min_pair_observations:
                    continue

                test_result = (
                    scipy_stats.pearsonr(pair_df[left], pair_df[right])
                    if method == "pearson"
                    else scipy_stats.spearmanr(pair_df[left], pair_df[right])
                )
                p_value = _safe_float(test_result.pvalue)
                metrics[f"{method}_p_value"] = p_value
                if (
                    metrics["p_value"] is None
                    or (p_value is not None and p_value < metrics["p_value"])
                ):
                    metrics["p_value"] = p_value

                severity = _severity_from_strength(
                    abs(coefficient),
                    threshold,
                    min(0.98, threshold + 0.15),
                )
                direction = "positive" if coefficient > 0 else "negative"
                findings.append(Finding(
                    module=self.name,
                    title=f"{method.title()} {direction} relationship: '{left}' vs '{right}'",
                    description=(
                        f"Columns '{left}' and '{right}' show a strong {direction} {method} "
                        f"relationship ({coefficient:.3f}), which is large enough to influence "
                        "model behavior, attribution, and analyst interpretation."
                    ),
                    evidence={
                        **metrics,
                        "pair": [left, right],
                        "method": method,
                        "observations": len(pair_df),
                        "analysis_rows": len(analysis_df),
                        "total_rows": len(numeric_df),
                        "sampled": len(analysis_df) < len(numeric_df),
                    },
                    severity=severity,
                    confidence=1.0 if p_value is not None and p_value <= 0.001 else 0.9,
                    recommendation=(
                        "Review whether both features should remain in the same model or report, "
                        "and determine whether the relationship is causal, engineered, "
                        "or redundant."
                    ),
                    affected_columns=[left, right],
                    metadata={
                        "category": f"{method}_correlation",
                        "correlation_strength": abs(coefficient),
                        "statistical_significance": p_value,
                        "impact": min(1.0, abs(coefficient)),
                    },
                ))

            logger.info(
                "Relationship/{} scan: {} rows, {} strong pairs, {} tested candidates",
                method,
                len(analysis_df),
                len(flagged_pairs),
                len(candidates),
            )

        return findings

    def _investigate_mutual_information(
        self,
        numeric_df: pd.DataFrame,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: RelationshipSettings,
        pair_metrics: dict[tuple[str, str], dict[str, Any]],
        full_analysis: bool,
    ) -> list[Finding]:
        findings: list[Finding] = []

        target_column = self._find_target_column(df, dataset_info, cfg)
        if target_column is not None:
            findings.extend(
                self._investigate_target_mutual_information(
                    numeric_df,
                    df[target_column],
                    target_column,
                    cfg,
                )
            )
            if not full_analysis:
                return findings

        sample_df = self._sample_rows(numeric_df, cfg.mi_max_rows)
        coverage = sample_df.notna().sum()
        variability = sample_df.nunique(dropna=True)
        ranked_columns = sorted(
            sample_df.columns,
            key=lambda column: (
                int(coverage[column]),
                int(variability[column]),
                str(column),
            ),
            reverse=True,
        )[: cfg.mi_max_columns]

        if not full_analysis:
            max_columns = self._max_columns_for_pair_budget(cfg.mi_max_pairs)
            ranked_columns = ranked_columns[:max_columns]

        candidate_pairs = list(combinations(ranked_columns, 2))
        if not candidate_pairs:
            return findings

        for left, right in candidate_pairs:
            pair_df = sample_df.loc[:, [left, right]].dropna()
            if len(pair_df) < cfg.min_pair_observations:
                continue

            mi_score = self._estimate_pairwise_mutual_information(
                pair_df[left].to_numpy(dtype=np.float64),
                pair_df[right].to_numpy(dtype=np.float64),
                cfg,
                symmetric=full_analysis,
            )
            if mi_score < cfg.mutual_information_threshold:
                continue

            key = self._pair_key(left, right)
            metrics = pair_metrics.setdefault(key, self._base_pair_metrics(cfg))
            metrics["mutual_information_score"] = mi_score

            pearson_value = metrics.get("pearson_correlation")
            spearman_value = metrics.get("spearman_correlation")
            strongest_linear = max(
                abs(value) for value in (pearson_value, spearman_value) if value is not None
            ) if any(value is not None for value in (pearson_value, spearman_value)) else 0.0

            if strongest_linear >= max(cfg.pearson_threshold, cfg.spearman_threshold):
                continue

            severity = _severity_from_strength(
                mi_score,
                cfg.mutual_information_threshold,
                cfg.mutual_information_threshold * 2.0,
            )
            findings.append(Finding(
                module=self.name,
                title=f"Nonlinear dependence detected: '{left}' vs '{right}'",
                description=(
                    f"Columns '{left}' and '{right}' share measurable dependence "
                    f"(mutual information {mi_score:.3f}) despite not standing out as a strong "
                    "linear relationship. This deserves review for hidden nonlinear structure."
                ),
                evidence={
                    **metrics,
                    "pair": [left, right],
                    "method": "mutual_information",
                    "observations": len(pair_df),
                    "analysis_rows": len(sample_df),
                    "total_rows": len(numeric_df),
                    "sampled": len(sample_df) < len(numeric_df),
                    "symmetric_estimate": full_analysis,
                },
                severity=severity,
                confidence=0.8,
                recommendation=(
                    "Test nonlinear transformations or nonlinear models and inspect whether this "
                    "relationship reflects a meaningful mechanism or a derived feature."
                ),
                affected_columns=[left, right],
                metadata={
                    "category": "mutual_information",
                    "correlation_strength": strongest_linear,
                    "anomaly_strength": mi_score,
                    "impact": min(1.0, mi_score),
                },
            ))

        logger.info(
            "Relationship/mutual_information scan: {} rows, {} columns, {} pairs",
            len(sample_df),
            len(ranked_columns),
            len(candidate_pairs),
        )
        return findings

    def _investigate_target_mutual_information(
        self,
        numeric_df: pd.DataFrame,
        target: pd.Series,
        target_column: str,
        cfg: RelationshipSettings,
    ) -> list[Finding]:
        target_non_null = target.dropna()
        if len(target_non_null) < cfg.min_pair_observations:
            return []

        is_classification_target = (
            pd.api.types.is_bool_dtype(target_non_null)
            or not pd.api.types.is_numeric_dtype(target_non_null)
            or target_non_null.nunique(dropna=True) <= 10
        )

        scored_features: list[tuple[str, float, int]] = []
        feature_columns = [
            column
            for column in numeric_df.columns
            if column != target_column
        ][: cfg.mi_max_columns]

        for column in feature_columns:
            pair_df = pd.concat(
                [
                    numeric_df[column].rename("__feature"),
                    target.rename("__target"),
                ],
                axis=1,
            ).dropna()
            if len(pair_df) < cfg.min_pair_observations:
                continue
            pair_df = self._sample_rows(pair_df, cfg.mi_max_rows)

            feature_values = pair_df["__feature"].to_numpy(dtype=np.float64).reshape(-1, 1)
            if is_classification_target:
                target_values = pd.Categorical(pair_df["__target"]).codes
                score = mutual_info_classif(
                    feature_values,
                    target_values,
                    discrete_features=False,
                    n_neighbors=cfg.mi_n_neighbors,
                    random_state=42,
                )[0]
            else:
                score = mutual_info_regression(
                    feature_values,
                    pair_df["__target"].to_numpy(dtype=np.float64),
                    discrete_features=False,
                    n_neighbors=cfg.mi_n_neighbors,
                    random_state=42,
                )[0]

            if float(score) >= cfg.mutual_information_threshold:
                scored_features.append((str(column), float(score), len(pair_df)))

        if not scored_features:
            return []

        scored_features.sort(key=lambda item: item[1], reverse=True)
        top_features = scored_features[: min(10, len(scored_features))]
        best_feature, best_score, best_observations = top_features[0]

        return [Finding(
            module=self.name,
            title=f"Features strongly depend on target '{target_column}'",
            description=(
                f"{len(top_features)} feature(s) show meaningful dependence on target "
                f"'{target_column}'. The strongest relationship is '{best_feature}' with mutual "
                f"information {best_score:.3f}, which suggests these columns deserve "
                "focused review."
            ),
            evidence={
                "correlation_coefficient": None,
                "p_value": None,
                "mutual_information_score": best_score,
                "vif": None,
                "thresholds": self._thresholds_dict(cfg),
                "target_column": target_column,
                "top_features": [
                    (name, score) for name, score, _observations in top_features
                ],
                "observations": best_observations,
                "total_rows": len(target),
                "sampled": best_observations < len(target_non_null),
            },
            severity=_severity_from_strength(
                best_score,
                cfg.mutual_information_threshold,
                cfg.mutual_information_threshold * 2.0,
            ),
            confidence=0.8,
            recommendation=(
                "Prioritize these features for target leakage review, interpretation, and model "
                "selection work before relying on them in downstream analysis."
            ),
            affected_columns=[
                target_column,
                *[name for name, _score, _observations in top_features],
            ],
            metadata={
                "category": "target_mutual_information",
                "anomaly_strength": best_score,
                "impact": min(1.0, best_score),
            },
        )]

    def _investigate_multicollinearity(
        self,
        numeric_df: pd.DataFrame,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: RelationshipSettings,
        pair_metrics: dict[tuple[str, str], dict[str, Any]],
        full_analysis: bool,
    ) -> list[Finding]:
        vif_df = self._sample_rows(
            numeric_df,
            None if full_analysis else cfg.vif_max_rows,
        )
        vif_scores = self._compute_vif_scores(vif_df)
        findings: list[Finding] = []

        for column, vif in vif_scores.items():
            if vif < cfg.vif_high_threshold:
                continue

            severity = (
                Severity.CRITICAL if vif >= cfg.vif_very_high_threshold else Severity.HIGH
            )
            findings.append(Finding(
                module=self.name,
                title=f"Problematic multicollinearity in '{column}'",
                description=(
                    f"Column '{column}' has VIF {vif:.2f}, indicating that its information is "
                    "substantially explained by other features. Coefficient stability and "
                    "interpretability are likely compromised."
                ),
                evidence={
                    "correlation_coefficient": None,
                    "p_value": None,
                    "mutual_information_score": None,
                    "vif": vif,
                    "thresholds": self._thresholds_dict(cfg),
                    "top_vif_scores": vif_scores,
                    "analysis_rows": len(vif_df),
                    "total_rows": len(numeric_df),
                    "sampled": len(vif_df) < len(numeric_df),
                },
                severity=severity,
                confidence=0.95,
                recommendation=(
                    "Remove, combine, or regularize this feature set before fitting models that "
                    "assume stable independent signals."
                ),
                affected_columns=[column],
                metadata={
                    "category": "multicollinearity",
                    "anomaly_strength": vif,
                    "impact": min(1.0, vif / max(cfg.vif_very_high_threshold, 1.0)),
                },
            ))

        return findings

    def _investigate_feature_groups(
        self,
        numeric_df: pd.DataFrame,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: RelationshipSettings,
        pair_metrics: dict[tuple[str, str], dict[str, Any]],
        full_analysis: bool,
    ) -> list[Finding]:
        del df, dataset_info, full_analysis
        findings: list[Finding] = []
        redundancy_graph = nx.Graph()
        strong_group_graph = nx.Graph()
        community_graph = nx.Graph()

        redundancy_graph.add_nodes_from(numeric_df.columns)
        strong_group_graph.add_nodes_from(numeric_df.columns)
        community_graph.add_nodes_from(numeric_df.columns)

        for (left, right), metrics in pair_metrics.items():
            strengths = [
                abs(value)
                for value in (
                    metrics.get("pearson_correlation"),
                    metrics.get("spearman_correlation"),
                )
                if value is not None
            ]
            if metrics.get("mutual_information_score") is not None:
                strengths.append(float(metrics["mutual_information_score"]))
            if not strengths:
                continue

            edge_strength = max(strengths)
            if edge_strength >= cfg.redundancy_threshold:
                redundancy_graph.add_edge(left, right, strength=edge_strength)
            if edge_strength >= cfg.strong_group_threshold:
                strong_group_graph.add_edge(left, right, strength=edge_strength)
            if edge_strength >= cfg.community_threshold:
                community_graph.add_edge(left, right, strength=edge_strength)

        findings.extend(
            self._build_group_findings(
                redundancy_graph,
                "redundant_features",
                "Highly redundant feature group",
                cfg,
                pair_metrics,
                minimum_size=2,
                severity=Severity.HIGH,
                recommendation=(
                    "Keep one representative feature from each redundant cluster unless domain "
                    "requirements justify preserving multiple near-duplicate signals."
                ),
            )
        )
        findings.extend(
            self._build_group_findings(
                strong_group_graph,
                "strong_feature_groups",
                "Strongly related feature group",
                cfg,
                pair_metrics,
                minimum_size=3,
                severity=Severity.MEDIUM,
                recommendation=(
                    "Review the group together rather than as isolated pairs; the shared signal "
                    "may indicate a latent factor or a repeated transformation pipeline."
                ),
            )
        )
        findings.extend(
            self._build_group_findings(
                community_graph,
                "relationship_communities",
                "Relationship community",
                cfg,
                pair_metrics,
                minimum_size=3,
                severity=Severity.MEDIUM,
                recommendation=(
                    "Investigate this community as a connected subsystem and check whether it "
                    "reflects operational linkage, duplication, or a hidden generating process."
                ),
            )
        )

        return findings

    def _build_group_findings(
        self,
        graph: nx.Graph,
        category: str,
        title_prefix: str,
        cfg: RelationshipSettings,
        pair_metrics: dict[tuple[str, str], dict[str, Any]],
        minimum_size: int,
        severity: Severity,
        recommendation: str,
    ) -> list[Finding]:
        findings: list[Finding] = []

        for component in nx.connected_components(graph):
            columns = sorted(component)
            if len(columns) < minimum_size:
                continue

            pair_details = []
            strongest_corr: float | None = None
            smallest_p: float | None = None
            best_mi: float | None = None

            for left, right in combinations(columns, 2):
                metrics = pair_metrics.get(self._pair_key(left, right))
                if metrics is None:
                    continue
                pair_details.append({
                    "pair": [left, right],
                    "pearson": metrics.get("pearson_correlation"),
                    "spearman": metrics.get("spearman_correlation"),
                    "mutual_information": metrics.get("mutual_information_score"),
                })
                for corr_key in ("pearson_correlation", "spearman_correlation"):
                    corr_value = metrics.get(corr_key)
                    if corr_value is not None and (
                        strongest_corr is None or abs(corr_value) > abs(strongest_corr)
                    ):
                        strongest_corr = float(corr_value)
                p_value = metrics.get("p_value")
                if p_value is not None and (smallest_p is None or p_value < smallest_p):
                    smallest_p = float(p_value)
                mi_score = metrics.get("mutual_information_score")
                if mi_score is not None and (best_mi is None or mi_score > best_mi):
                    best_mi = float(mi_score)

            title = (
                f"{title_prefix}: {', '.join(columns[:4])}"
                + ("..." if len(columns) > 4 else "")
            )
            findings.append(Finding(
                module=self.name,
                title=title,
                description=(
                    f"{len(columns)} features form a connected relationship group: "
                    f"{', '.join(columns)}. These variables should be investigated "
                    "together because "
                    "their shared behavior can distort attribution, inflate confidence, or hide a "
                    "single underlying factor."
                ),
                evidence={
                    "correlation_coefficient": strongest_corr,
                    "p_value": smallest_p,
                    "mutual_information_score": best_mi,
                    "vif": None,
                    "thresholds": self._thresholds_dict(cfg),
                    "pair_details": pair_details,
                    "group_size": len(columns),
                },
                severity=severity,
                confidence=0.85,
                recommendation=recommendation,
                affected_columns=columns,
                metadata={
                    "category": category,
                    "correlation_strength": (
                        abs(strongest_corr) if strongest_corr is not None else 0.0
                    ),
                    "impact": min(1.0, len(columns) / max(len(graph.nodes), 1)),
                },
            ))

        return findings

    def _estimate_pairwise_mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cfg: RelationshipSettings,
        *,
        symmetric: bool,
    ) -> float:
        mi_xy = mutual_info_regression(
            x.reshape(-1, 1),
            y,
            discrete_features=False,
            n_neighbors=cfg.mi_n_neighbors,
            random_state=42,
        )[0]
        if not symmetric:
            return float(mi_xy)

        mi_yx = mutual_info_regression(
            y.reshape(-1, 1),
            x,
            discrete_features=False,
            n_neighbors=cfg.mi_n_neighbors,
            random_state=42,
        )[0]
        return float((float(mi_xy) + float(mi_yx)) / 2.0)

    @staticmethod
    def _pair_key(left: str, right: str) -> tuple[str, str]:
        return (left, right) if left <= right else (right, left)

    @staticmethod
    def _sample_rows(df: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
        if maximum is None or len(df) <= maximum:
            return df
        return df.sample(n=maximum, random_state=42)

    @staticmethod
    def _max_columns_for_pair_budget(maximum_pairs: int) -> int:
        columns = max(2, (1 + math.isqrt(1 + 8 * maximum_pairs)) // 2)
        while columns * (columns - 1) // 2 > maximum_pairs:
            columns -= 1
        return columns

    def _compute_vif_scores(self, numeric_df: pd.DataFrame) -> dict[str, float]:
        vif_df = numeric_df.copy()
        vif_df = vif_df.fillna(vif_df.median())
        vif_df = vif_df.loc[:, vif_df.nunique(dropna=False) > 1]
        if vif_df.shape[1] < 2:
            return {}

        standardized = (vif_df - vif_df.mean()) / vif_df.std(ddof=0)
        standardized = standardized.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        corr = standardized.corr().to_numpy(dtype=np.float64)
        inverse = np.linalg.pinv(corr, hermitian=True)
        diag = np.diag(inverse)

        scores: dict[str, float] = {}
        for column, value in zip(vif_df.columns, diag, strict=False):
            finite_value = float(value) if np.isfinite(value) else float("inf")
            scores[column] = max(1.0, finite_value)
        return scores

    def _find_target_column(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: RelationshipSettings,
    ) -> str | None:
        metadata_target = dataset_info.metadata.get("target_column")
        if isinstance(metadata_target, str) and metadata_target in df.columns:
            return metadata_target

        candidates = {candidate.lower() for candidate in cfg.target_column_candidates}
        for column in df.columns:
            if str(column).lower() in candidates:
                return str(column)
        return None

    def _iter_flagged_pairs(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float,
    ) -> list[tuple[str, str, float]]:
        cols = list(corr_matrix.columns)
        matrix = corr_matrix.to_numpy(dtype=np.float64)
        upper_rows, upper_cols = np.triu_indices_from(matrix, k=1)
        pairs: list[tuple[str, str, float]] = []

        for row_idx, col_idx in zip(upper_rows, upper_cols, strict=False):
            coefficient = matrix[row_idx, col_idx]
            if not np.isfinite(coefficient) or abs(coefficient) < threshold:
                continue
            pairs.append((cols[row_idx], cols[col_idx], float(coefficient)))

        pairs.sort(key=lambda item: abs(item[2]), reverse=True)
        return pairs

    def _base_pair_metrics(self, cfg: RelationshipSettings) -> dict[str, Any]:
        return {
            "correlation_coefficient": None,
            "p_value": None,
            "mutual_information_score": None,
            "vif": None,
            "thresholds": self._thresholds_dict(cfg),
            "pearson_correlation": None,
            "spearman_correlation": None,
            "pearson_p_value": None,
            "spearman_p_value": None,
        }

    def _thresholds_dict(self, cfg: RelationshipSettings) -> dict[str, Any]:
        return {
            "pearson_threshold": cfg.pearson_threshold,
            "spearman_threshold": cfg.spearman_threshold,
            "mutual_information_threshold": cfg.mutual_information_threshold,
            "redundancy_threshold": cfg.redundancy_threshold,
            "strong_group_threshold": cfg.strong_group_threshold,
            "community_threshold": cfg.community_threshold,
            "vif_high_threshold": cfg.vif_high_threshold,
            "vif_very_high_threshold": cfg.vif_very_high_threshold,
        }

    def _cap_findings(self, findings: list[Finding], maximum: int) -> list[Finding]:
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }

        def score(finding: Finding) -> tuple[int, float, float]:
            impact = float(finding.metadata.get("impact", 0.0) or 0.0)
            strength = float(finding.metadata.get("correlation_strength", 0.0) or 0.0)
            anomaly = float(finding.metadata.get("anomaly_strength", 0.0) or 0.0)
            return (
                severity_order[finding.severity],
                -(impact + strength + anomaly),
                -finding.confidence,
            )

        return sorted(findings, key=score)[:maximum]
