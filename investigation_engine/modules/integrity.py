"""Integrity Investigator — Autonomous structural integrity analysis.

The first investigator in the Investigation Intelligence Engine. It operates
like an experienced data quality engineer, autonomously discovering structural
problems that deserve analyst attention.

Question answered:
    "Should an analyst investigate something related to the structural
     integrity of this dataset?"

Investigation Domains:
    1. Missing Value Investigation
    2. Duplicate Row Investigation
    3. Duplicate Feature Investigation
    4. Identifier Investigation
    5. Constant Feature Investigation
    6. Near-Constant Feature Investigation
    7. Datatype Integrity Investigation
    8. Cardinality Investigation
    9. Missingness Relationship Investigation
   10. Structural Health Score

Design:
    - Each domain is a private method returning list[Finding]
    - The main investigate() orchestrates all domains
    - All thresholds come from IntegritySettings (no magic numbers)
    - Vectorized pandas operations for performance
    - Graceful exception handling per domain (one failure ≠ total failure)
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from loguru import logger

from investigation_engine.config.settings import IntegritySettings, Settings
from investigation_engine.core.plugin import register_module
from investigation_engine.models.dataset import DatasetInfo
from investigation_engine.models.finding import Finding
from investigation_engine.models.severity import Severity
from investigation_engine.modules.base import BaseInvestigationModule


def _is_string_dtype(dtype: Any) -> bool:
    """Check if dtype is a string type (handles both pandas 2.x 'object' and 3.x 'str')."""
    return (
        dtype == "object"
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    )


# ═══════════════════════════════════════════════════════════════════════
#  Helper: severity from ratio against tiered thresholds
# ═══════════════════════════════════════════════════════════════════════

def _severity_from_ratio(
    ratio: float,
    critical: float,
    high: float,
    medium: float,
    low: float,
) -> Severity:
    """Map a ratio to a Severity level using configurable thresholds."""
    if ratio >= critical:
        return Severity.CRITICAL
    if ratio >= high:
        return Severity.HIGH
    if ratio >= medium:
        return Severity.MEDIUM
    if ratio >= low:
        return Severity.LOW
    return Severity.INFO


def _severity_numeric(severity: Severity) -> int:
    """Convert severity to a 0-100 integer for the evidence payload."""
    return {
        Severity.CRITICAL: 90,
        Severity.HIGH: 70,
        Severity.MEDIUM: 50,
        Severity.LOW: 30,
        Severity.INFO: 10,
    }[severity]


# ═══════════════════════════════════════════════════════════════════════
#  Integrity Investigator
# ═══════════════════════════════════════════════════════════════════════

@register_module
class IntegrityInvestigator(BaseInvestigationModule):
    """Autonomous investigator for dataset structural integrity.

    Discovers missing values, duplicates, constant features, datatype
    inconsistencies, identifier problems, cardinality anomalies,
    missingness correlations, and computes a structural health score.
    """

    name: ClassVar[str] = "integrity_investigator"
    description: ClassVar[str] = (
        "Investigates structural integrity of the dataset: missing values, "
        "duplicates, constants, datatype issues, identifiers, cardinality, "
        "and missingness relationships."
    )
    version: ClassVar[str] = "0.2.0"
    tags: ClassVar[list[str]] = ["integrity", "quality", "structural"]

    # ── Plugin Interface ──────────────────────────────────────────────

    def can_run(self, df: pd.DataFrame, dataset_info: DatasetInfo) -> bool:
        """Can run on any dataset with at least 1 row and 1 column."""
        return dataset_info.row_count > 0 and dataset_info.column_count > 0

    def get_skip_reason(self, df: pd.DataFrame, dataset_info: DatasetInfo) -> str | None:
        if dataset_info.row_count == 0:
            return "Dataset has no rows."
        if dataset_info.column_count == 0:
            return "Dataset has no columns."
        return None

    def investigate(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        config: Settings,
    ) -> list[Finding]:
        """Execute all integrity investigations and return findings."""
        cfg = config.integrity
        findings: list[Finding] = []

        # Each investigation domain is isolated — one crash does not
        # affect others.
        investigations = [
            ("missing_values", self._investigate_missing_values),
            ("duplicate_rows", self._investigate_duplicate_rows),
            ("duplicate_features", self._investigate_duplicate_features),
            ("identifiers", self._investigate_identifiers),
            ("constant_features", self._investigate_constant_features),
            ("near_constant_features", self._investigate_near_constant_features),
            ("datatype_integrity", self._investigate_datatype_integrity),
            ("cardinality", self._investigate_cardinality),
            ("missingness_relationships", self._investigate_missingness_relationships),
        ]

        domain_scores: dict[str, float] = {}

        for domain_name, method in investigations:
            try:
                domain_findings = method(df, dataset_info, cfg)
                findings.extend(domain_findings)
                # Track domain health for structural score
                domain_scores[domain_name] = self._compute_domain_score(domain_findings)
                logger.debug(
                    "Integrity/{}: {} findings",
                    domain_name,
                    len(domain_findings),
                )
            except Exception as e:
                logger.error(
                    "Integrity/{} failed: {}",
                    domain_name,
                    e,
                    exc_info=True,
                )
                domain_scores[domain_name] = 50.0  # neutral on failure

        # Structural Health Score (always produced)
        health_finding = self._compute_structural_health_score(
            domain_scores, dataset_info, cfg,
        )
        findings.append(health_finding)

        return findings

    # ══════════════════════════════════════════════════════════════════
    #  1. MISSING VALUE INVESTIGATION
    # ══════════════════════════════════════════════════════════════════

    def _investigate_missing_values(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: IntegritySettings,
    ) -> list[Finding]:
        """Detect columns with actionable missing value ratios."""
        findings: list[Finding] = []
        n_rows = len(df)
        if n_rows == 0:
            return findings

        # Vectorized null computation across all columns
        null_counts: pd.Series = df.isnull().sum()
        null_ratios: pd.Series = null_counts / n_rows

        # ── Completely empty columns ─────────────────────────────────
        empty_cols = null_ratios[null_ratios >= cfg.missing_empty_column_threshold].index.tolist()
        if empty_cols:
            findings.append(Finding(
                module=self.name,
                title=f"{len(empty_cols)} completely empty column(s) detected",
                description=(
                    f"The following columns contain ≥{cfg.missing_empty_column_threshold:.0%} "
                    f"missing values and carry no analytical value: {', '.join(empty_cols)}. "
                    f"These columns will introduce noise into any downstream modelling or "
                    f"aggregation and should be removed or investigated for data pipeline issues."
                ),
                evidence={
                    "empty_columns": empty_cols,
                    "empty_column_count": len(empty_cols),
                    "threshold": cfg.missing_empty_column_threshold,
                    "missing_ratios": {c: round(float(null_ratios[c]), 4) for c in empty_cols},
                },
                severity=Severity.CRITICAL,
                confidence=1.0,
                recommendation=(
                    "Remove these columns or investigate the data ingestion pipeline. "
                    "They contain no usable information."
                ),
                affected_columns=empty_cols,
                metadata={
                    "anomaly_strength": 1.0,
                    "impact": len(empty_cols) / dataset_info.column_count,
                    "category": "missing_values",
                },
            ))

        # ── Per-column missing value findings ────────────────────────
        for col in df.columns:
            ratio = float(null_ratios[col])
            count = int(null_counts[col])

            # Skip if below minimum reporting threshold or already reported as empty
            if ratio < cfg.missing_low_threshold or col in empty_cols:
                continue

            severity = _severity_from_ratio(
                ratio,
                cfg.missing_critical_threshold,
                cfg.missing_high_threshold,
                cfg.missing_medium_threshold,
                cfg.missing_low_threshold,
            )

            findings.append(Finding(
                module=self.name,
                title=f"Column '{col}' has {ratio:.1%} missing values",
                description=(
                    f"Column '{col}' contains {count:,} missing values out of {n_rows:,} rows "
                    f"({ratio:.2%}). This exceeds the configured threshold of "
                    f"{cfg.missing_low_threshold:.0%} and may introduce bias during downstream "
                    f"modelling, distort aggregations, or indicate data collection issues."
                ),
                evidence={
                    "column": col,
                    "missing_count": count,
                    "missing_ratio": round(ratio, 4),
                    "total_rows": n_rows,
                    "threshold": cfg.missing_low_threshold,
                    "severity_score": _severity_numeric(severity),
                },
                severity=severity,
                confidence=1.0,
                recommendation=(
                    f"Investigate why '{col}' has {ratio:.1%} missing values. Consider "
                    f"imputation strategies, or determine if the missingness is informative "
                    f"(MCAR/MAR/MNAR analysis recommended)."
                ),
                affected_columns=[col],
                metadata={
                    "anomaly_strength": ratio,
                    "impact": ratio,
                    "category": "missing_values",
                },
            ))

        return findings

    # ══════════════════════════════════════════════════════════════════
    #  2. DUPLICATE ROW INVESTIGATION
    # ══════════════════════════════════════════════════════════════════

    def _investigate_duplicate_rows(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: IntegritySettings,
    ) -> list[Finding]:
        """Detect exact duplicate rows."""
        findings: list[Finding] = []
        n_rows = len(df)
        if n_rows < 2:
            return findings

        dup_mask = df.duplicated(keep="first")
        dup_count = int(dup_mask.sum())
        dup_ratio = dup_count / n_rows

        if dup_ratio < cfg.duplicate_low_threshold:
            return findings

        severity = _severity_from_ratio(
            dup_ratio,
            cfg.duplicate_critical_threshold,
            cfg.duplicate_high_threshold,
            cfg.duplicate_medium_threshold,
            cfg.duplicate_low_threshold,
        )

        # Identify a sample of duplicate row indices (max 100)
        dup_indices = dup_mask[dup_mask].index.tolist()
        sample_indices = dup_indices[:100]

        findings.append(Finding(
            module=self.name,
            title=f"{dup_count:,} exact duplicate rows detected ({dup_ratio:.1%})",
            description=(
                f"The dataset contains {dup_count:,} exact duplicate rows out of "
                f"{n_rows:,} total rows ({dup_ratio:.2%}). Duplicate rows can inflate "
                f"statistical estimates, bias model training, and distort distribution "
                f"analyses. This may indicate data pipeline issues such as repeated "
                f"ingestion or join fan-out."
            ),
            evidence={
                "duplicate_count": dup_count,
                "duplicate_ratio": round(dup_ratio, 4),
                "total_rows": n_rows,
                "threshold": cfg.duplicate_low_threshold,
                "sample_duplicate_indices": sample_indices,
                "severity_score": _severity_numeric(severity),
            },
            severity=severity,
            confidence=1.0,
            recommendation=(
                f"Investigate the source of {dup_count:,} duplicate rows. Determine "
                f"whether duplicates are expected (e.g., time-series resampling) or "
                f"indicate a data ingestion issue. Consider deduplication."
            ),
            affected_columns=list(df.columns),
            affected_rows=sample_indices,
            metadata={
                "anomaly_strength": dup_ratio,
                "impact": dup_ratio,
                "category": "duplicates",
            },
        ))

        return findings

    # ══════════════════════════════════════════════════════════════════
    #  3. DUPLICATE FEATURE INVESTIGATION
    # ══════════════════════════════════════════════════════════════════

    def _investigate_duplicate_features(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: IntegritySettings,
    ) -> list[Finding]:
        """Detect columns with identical values (redundant features)."""
        findings: list[Finding] = []
        columns = list(df.columns)
        n_cols = len(columns)

        if n_cols < 2:
            return findings

        # Build hash fingerprints for fast comparison
        col_hashes: dict[str, int] = {}
        for col in columns:
            try:
                col_hashes[col] = hash(df[col].astype(str).str.cat())
            except Exception:
                col_hashes[col] = hash(str(df[col].tolist()))

        # Group columns by hash
        hash_groups: dict[int, list[str]] = {}
        for col, h in col_hashes.items():
            hash_groups.setdefault(h, []).append(col)

        # For hash collisions, verify with actual equality
        duplicate_groups: list[list[str]] = []
        for _h, group in hash_groups.items():
            if len(group) < 2:
                continue
            # Verify actual equality for each pair
            verified: list[list[str]] = []
            used: set[str] = set()
            for i in range(len(group)):
                if group[i] in used:
                    continue
                equiv = [group[i]]
                for j in range(i + 1, len(group)):
                    if group[j] in used:
                        continue
                    if df[group[i]].equals(df[group[j]]):
                        equiv.append(group[j])
                        used.add(group[j])
                if len(equiv) > 1:
                    verified.append(equiv)
                    used.add(group[i])
            duplicate_groups.extend(verified)

        for group in duplicate_groups:
            findings.append(Finding(
                module=self.name,
                title=f"Identical columns detected: {', '.join(group)}",
                description=(
                    f"Columns {group} contain identical values across all {len(df):,} rows. "
                    f"Redundant features waste memory, increase computation time, and can "
                    f"cause multicollinearity in models. Only one column from this group "
                    f"should be retained."
                ),
                evidence={
                    "identical_columns": group,
                    "group_size": len(group),
                    "total_rows": len(df),
                },
                severity=Severity.MEDIUM,
                confidence=1.0,
                recommendation=(
                    f"Remove {len(group) - 1} redundant column(s) from the group "
                    f"{group}. Keep the column with the most descriptive name."
                ),
                affected_columns=group,
                metadata={
                    "anomaly_strength": 1.0,
                    "impact": (len(group) - 1) / n_cols,
                    "category": "duplicate_features",
                },
            ))

        return findings

    # ══════════════════════════════════════════════════════════════════
    #  4. IDENTIFIER INVESTIGATION
    # ══════════════════════════════════════════════════════════════════

    def _investigate_identifiers(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: IntegritySettings,
    ) -> list[Finding]:
        """Detect potential identifier columns and investigate their integrity."""
        findings: list[Finding] = []
        n_rows = len(df)
        if n_rows == 0:
            return findings

        id_candidates = self._detect_id_columns(df, cfg)

        for col in id_candidates:
            series = df[col]
            null_count = int(series.isnull().sum())
            unique_count = int(series.nunique())
            total_non_null = n_rows - null_count
            duplicate_count = total_non_null - unique_count if total_non_null > unique_count else 0

            issues: list[str] = []

            # Null IDs
            if null_count > 0:
                issues.append(f"{null_count:,} null values")

            # Non-unique IDs
            if duplicate_count > 0:
                issues.append(f"{duplicate_count:,} duplicate values")

            if not issues:
                continue

            null_ratio = null_count / n_rows if n_rows > 0 else 0
            dup_ratio = duplicate_count / n_rows if n_rows > 0 else 0
            worst_ratio = max(null_ratio, dup_ratio)

            severity = _severity_from_ratio(
                worst_ratio,
                critical=0.10,
                high=0.05,
                medium=0.01,
                low=0.001,
            )

            findings.append(Finding(
                module=self.name,
                title=f"Identifier column '{col}' has integrity issues",
                description=(
                    f"Column '{col}' appears to be an identifier but has the following "
                    f"issues: {'; '.join(issues)}. Identifiers with nulls or duplicates "
                    f"can cause join failures, data loss during merges, and incorrect "
                    f"record linkage in downstream analysis."
                ),
                evidence={
                    "column": col,
                    "null_count": null_count,
                    "null_ratio": round(null_ratio, 4),
                    "duplicate_id_count": duplicate_count,
                    "duplicate_id_ratio": round(dup_ratio, 4),
                    "unique_count": unique_count,
                    "total_rows": n_rows,
                },
                severity=severity,
                confidence=0.85,
                recommendation=(
                    f"Verify that '{col}' is intended as an identifier. If so, investigate "
                    f"the source of {'; '.join(issues)}. Non-unique identifiers can cause "
                    f"silent data corruption in joins and aggregations."
                ),
                affected_columns=[col],
                metadata={
                    "anomaly_strength": worst_ratio,
                    "impact": worst_ratio,
                    "category": "identifiers",
                },
            ))

        return findings

    def _detect_id_columns(
        self,
        df: pd.DataFrame,
        cfg: IntegritySettings,
    ) -> list[str]:
        """Heuristically detect columns that are likely identifiers."""
        candidates: list[str] = []
        n_rows = len(df)

        for col in df.columns:
            col_lower = str(col).lower().replace("_", "").replace("-", "").replace(" ", "")

            # Name-based detection
            name_match = any(
                pattern in col_lower
                for pattern in cfg.id_column_patterns
            )

            # Uniqueness-based detection
            unique_ratio = df[col].nunique() / n_rows if n_rows > 0 else 0
            uniqueness_match = unique_ratio >= cfg.id_uniqueness_threshold

            # Both name and uniqueness, or strong name match with decent uniqueness
            if name_match and uniqueness_match:
                candidates.append(str(col))
            elif name_match and unique_ratio > 0.5:
                # Name strongly suggests ID — include even if not perfectly unique
                candidates.append(str(col))

        return candidates

    # ══════════════════════════════════════════════════════════════════
    #  5. CONSTANT FEATURE INVESTIGATION
    # ══════════════════════════════════════════════════════════════════

    def _investigate_constant_features(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: IntegritySettings,
    ) -> list[Finding]:
        """Detect columns with zero variance (single unique value)."""
        findings: list[Finding] = []
        n_rows = len(df)
        if n_rows == 0:
            return findings

        nunique = df.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()

        for col in constant_cols:
            non_null = df[col].dropna()
            constant_value = non_null.iloc[0] if len(non_null) > 0 else None

            findings.append(Finding(
                module=self.name,
                title=f"Constant column '{col}' has zero variance",
                description=(
                    f"Column '{col}' contains a single unique value "
                    f"({repr(constant_value)}) across all {n_rows:,} rows. "
                    f"Zero-variance features carry no information for modelling, "
                    f"increase dimensionality without benefit, and should be removed."
                ),
                evidence={
                    "column": col,
                    "unique_count": 1,
                    "constant_value": str(constant_value),
                    "total_rows": n_rows,
                    "variance": 0.0,
                },
                severity=Severity.MEDIUM,
                confidence=1.0,
                recommendation=(
                    f"Remove column '{col}' from the analysis. It contains only the "
                    f"value {repr(constant_value)} and provides no discriminative power."
                ),
                affected_columns=[col],
                metadata={
                    "anomaly_strength": 1.0,
                    "impact": 1 / dataset_info.column_count,
                    "category": "constant_features",
                },
            ))

        return findings

    # ══════════════════════════════════════════════════════════════════
    #  6. NEAR-CONSTANT FEATURE INVESTIGATION
    # ══════════════════════════════════════════════════════════════════

    def _investigate_near_constant_features(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: IntegritySettings,
    ) -> list[Finding]:
        """Detect columns where one value dominates almost entirely."""
        findings: list[Finding] = []
        n_rows = len(df)
        if n_rows == 0:
            return findings

        # Skip columns already flagged as constant (unique count <= 1)
        nunique = df.nunique()
        constant_cols = set(nunique[nunique <= 1].index)

        for col in df.columns:
            if col in constant_cols:
                continue

            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            # Compute dominance of the most frequent value
            top_count = int(non_null.value_counts().iloc[0])
            dominance = top_count / len(non_null)

            if dominance < cfg.near_constant_dominance_threshold:
                continue

            # Already at constant level — skip (handled by constant investigation)
            if dominance >= cfg.constant_dominance_threshold:
                continue

            top_value = non_null.value_counts().index[0]
            minority_count = len(non_null) - top_count

            findings.append(Finding(
                module=self.name,
                title=f"Near-constant column '{col}' — {dominance:.1%} single value",
                description=(
                    f"Column '{col}' is dominated by value {repr(top_value)} which appears "
                    f"in {dominance:.2%} of non-null rows ({top_count:,} of {len(non_null):,}). "
                    f"Only {minority_count:,} rows have different values. Near-constant "
                    f"features provide minimal discriminative power and can cause numerical "
                    f"instability in some algorithms."
                ),
                evidence={
                    "column": col,
                    "dominant_value": str(top_value),
                    "dominance_ratio": round(dominance, 4),
                    "dominant_count": top_count,
                    "minority_count": minority_count,
                    "unique_count": int(nunique[col]),
                    "total_non_null": len(non_null),
                    "threshold": cfg.near_constant_dominance_threshold,
                },
                severity=Severity.LOW,
                confidence=0.9,
                recommendation=(
                    f"Evaluate whether '{col}' provides useful signal. With {dominance:.1%} "
                    f"dominated by a single value, consider removal or encoding the minority "
                    f"values as a binary flag."
                ),
                affected_columns=[col],
                metadata={
                    "anomaly_strength": dominance,
                    "impact": dominance,
                    "category": "near_constant_features",
                },
            ))

        return findings

    # ══════════════════════════════════════════════════════════════════
    #  7. DATATYPE INTEGRITY INVESTIGATION
    # ══════════════════════════════════════════════════════════════════

    def _investigate_datatype_integrity(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: IntegritySettings,
    ) -> list[Finding]:
        """Detect datatype inconsistencies: numeric-as-string, mixed types."""
        findings: list[Finding] = []
        n_rows = len(df)
        if n_rows == 0:
            return findings

        for col in df.columns:
            dtype = df[col].dtype

            # ── Numeric stored as string ─────────────────────────────
            if _is_string_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                finding = self._check_numeric_as_string(df, col, n_rows, cfg)
                if finding is not None:
                    findings.append(finding)

                # ── Mixed types within string columns ─────────────────
                finding = self._check_mixed_types(df, col, n_rows, cfg)
                if finding is not None:
                    findings.append(finding)

        return findings

    def _check_numeric_as_string(
        self,
        df: pd.DataFrame,
        col: str,
        n_rows: int,
        cfg: IntegritySettings,
    ) -> Finding | None:
        """Check if a string column actually contains numeric data."""
        non_null = df[col].dropna()
        if len(non_null) == 0:
            return None

        # Sample for performance
        sample = non_null.sample(
            n=min(cfg.numeric_string_sample_size, len(non_null)),
            random_state=42,
        )

        # Attempt numeric conversion
        numeric_parsed = pd.to_numeric(sample, errors="coerce")
        parseable_count = int(numeric_parsed.notna().sum())
        parseable_ratio = parseable_count / len(sample)

        if parseable_ratio < cfg.numeric_string_threshold:
            return None

        return Finding(
            module=self.name,
            title=f"Column '{col}' contains numeric data stored as strings",
            description=(
                f"Column '{col}' has dtype 'object' but {parseable_ratio:.1%} of sampled "
                f"values ({parseable_count}/{len(sample)}) are parseable as numbers. "
                f"Numeric data stored as strings prevents mathematical operations, "
                f"breaks sorting, and wastes memory. This typically indicates upstream "
                f"data pipeline issues."
            ),
            evidence={
                "column": col,
                "dtype": str(df[col].dtype),
                "parseable_numeric_ratio": round(parseable_ratio, 4),
                "parseable_count": parseable_count,
                "sample_size": len(sample),
                "threshold": cfg.numeric_string_threshold,
                "sample_values": non_null.head(5).tolist(),
            },
            severity=Severity.HIGH,
            confidence=parseable_ratio,
            recommendation=(
                f"Convert column '{col}' to a numeric dtype using "
                f"pd.to_numeric(df['{col}'], errors='coerce'). Investigate non-parseable "
                f"values before conversion."
            ),
            affected_columns=[col],
            metadata={
                "anomaly_strength": parseable_ratio,
                "impact": 0.5,
                "category": "datatype_integrity",
            },
        )

    def _check_mixed_types(
        self,
        df: pd.DataFrame,
        col: str,
        n_rows: int,
        cfg: IntegritySettings,
    ) -> Finding | None:
        """Check if a string column has mixed inferred types."""
        non_null = df[col].dropna()
        if len(non_null) < 10:
            return None

        sample = non_null.sample(
            n=min(cfg.numeric_string_sample_size, len(non_null)),
            random_state=42,
        )

        # Infer types per value
        type_counts: dict[str, int] = {}
        for val in sample:
            inferred = self._infer_value_type(val)
            type_counts[inferred] = type_counts.get(inferred, 0) + 1

        if len(type_counts) < 2:
            return None

        # Compute the fraction of minority types
        total = sum(type_counts.values())
        dominant_type = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]
        dominant_count = type_counts[dominant_type]
        minority_fraction = 1 - (dominant_count / total)

        if minority_fraction < cfg.mixed_type_threshold:
            return None

        return Finding(
            module=self.name,
            title=f"Column '{col}' contains mixed data types",
            description=(
                f"Column '{col}' contains multiple inferred data types: "
                f"{type_counts}. Mixed types can cause unexpected coercion errors, "
                f"break type-dependent operations, and indicate data quality issues "
                f"at the source."
            ),
            evidence={
                "column": col,
                "type_distribution": type_counts,
                "dominant_type": dominant_type,
                "minority_fraction": round(minority_fraction, 4),
                "sample_size": len(sample),
                "threshold": cfg.mixed_type_threshold,
            },
            severity=Severity.MEDIUM if minority_fraction > 0.1 else Severity.LOW,
            confidence=0.8,
            recommendation=(
                f"Investigate column '{col}' for data quality issues. Separate or "
                f"clean mixed types before analysis. The dominant type is '{dominant_type}'."
            ),
            affected_columns=[col],
            metadata={
                "anomaly_strength": minority_fraction,
                "impact": minority_fraction,
                "category": "datatype_integrity",
            },
        )

    @staticmethod
    def _infer_value_type(value: Any) -> str:
        """Infer the semantic type of a single value."""
        if isinstance(value, (int, float, np.integer, np.floating)):
            return "numeric"
        s = str(value).strip()
        if not s:
            return "empty"
        # Try numeric
        try:
            float(s)
            return "numeric"
        except (ValueError, TypeError):
            pass
        # Try boolean
        if s.lower() in ("true", "false", "yes", "no", "1", "0"):
            return "boolean"
        # Try date-like patterns
        if re.match(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", s):
            return "datetime"
        return "string"

    # ══════════════════════════════════════════════════════════════════
    #  8. CARDINALITY INVESTIGATION
    # ══════════════════════════════════════════════════════════════════

    def _investigate_cardinality(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: IntegritySettings,
    ) -> list[Finding]:
        """Detect columns with suspicious cardinality levels."""
        findings: list[Finding] = []
        n_rows = len(df)
        if n_rows == 0:
            return findings

        id_candidates = set(self._detect_id_columns(df, cfg))

        for col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            unique_count = int(non_null.nunique())
            unique_ratio = unique_count / len(non_null)
            dtype = df[col].dtype

            # ── High cardinality (non-ID columns) ────────────────────
            if (
                col not in id_candidates
                and _is_string_dtype(dtype)
                and unique_ratio >= cfg.high_cardinality_ratio
                and unique_count > 100
            ):
                findings.append(Finding(
                    module=self.name,
                    title=f"High cardinality in column '{col}' — {unique_count:,} unique values",
                    description=(
                        f"Categorical column '{col}' has {unique_count:,} unique values "
                        f"({unique_ratio:.1%} of rows). This is unusually high for a "
                        f"categorical feature and may indicate that it is actually an "
                        f"identifier, free-text, or improperly encoded feature. High "
                        f"cardinality causes issues with one-hot encoding, increases "
                        f"model complexity, and may lead to overfitting."
                    ),
                    evidence={
                        "column": col,
                        "unique_count": unique_count,
                        "unique_ratio": round(unique_ratio, 4),
                        "total_non_null": len(non_null),
                        "threshold": cfg.high_cardinality_ratio,
                        "sample_values": non_null.head(5).tolist(),
                    },
                    severity=Severity.MEDIUM,
                    confidence=0.85,
                    recommendation=(
                        f"Determine whether '{col}' should be treated as an identifier, "
                        f"free-text, or categorical. If categorical, consider grouping rare "
                        f"categories or using target encoding instead of one-hot."
                    ),
                    affected_columns=[col],
                    metadata={
                        "anomaly_strength": unique_ratio,
                        "impact": 0.3,
                        "category": "cardinality",
                    },
                ))

            # ── Low cardinality (non-boolean columns) ─────────────────
            if (
                unique_count <= cfg.low_cardinality_max_unique
                and dtype != "bool"
                and not pd.api.types.is_bool_dtype(df[col])
                and n_rows > 100
                # Don't flag binary-looking numeric columns that are clearly flags
                and not (pd.api.types.is_numeric_dtype(dtype) and set(non_null.unique()) <= {0, 1})
            ):
                values = non_null.unique().tolist()
                findings.append(Finding(
                    module=self.name,
                    title=f"Low cardinality in column '{col}' — only {unique_count} unique value(s)",
                    description=(
                        f"Column '{col}' contains only {unique_count} unique value(s): "
                        f"{values}. This may indicate a column that should be boolean, "
                        f"an improperly encoded feature, or a feature with insufficient "
                        f"variation for meaningful analysis."
                    ),
                    evidence={
                        "column": col,
                        "unique_count": unique_count,
                        "unique_values": [str(v) for v in values],
                        "total_non_null": len(non_null),
                        "threshold": cfg.low_cardinality_max_unique,
                    },
                    severity=Severity.INFO,
                    confidence=0.7,
                    recommendation=(
                        f"Verify whether '{col}' should be encoded as boolean or if "
                        f"additional values are expected."
                    ),
                    affected_columns=[col],
                    metadata={
                        "anomaly_strength": 1 - (unique_count / max(n_rows, 1)),
                        "impact": 0.1,
                        "category": "cardinality",
                    },
                ))

        return findings

    # ══════════════════════════════════════════════════════════════════
    #  9. MISSINGNESS RELATIONSHIP INVESTIGATION
    # ══════════════════════════════════════════════════════════════════

    def _investigate_missingness_relationships(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        cfg: IntegritySettings,
    ) -> list[Finding]:
        """Detect columns that are frequently missing together."""
        findings: list[Finding] = []
        n_rows = len(df)
        if n_rows == 0:
            return findings

        # Only consider columns with meaningful missingness
        null_ratios = df.isnull().sum() / n_rows
        missing_cols = null_ratios[
            null_ratios >= cfg.missingness_min_missing_ratio
        ].index.tolist()

        if len(missing_cols) < 2:
            return findings

        # Build missingness indicator matrix
        miss_matrix = df[missing_cols].isnull().astype(int)

        # Compute pairwise correlations of missingness indicators
        try:
            miss_corr = miss_matrix.corr()
        except Exception:
            return findings

        # Find highly correlated pairs
        reported: set[frozenset[str]] = set()
        for i, col_a in enumerate(missing_cols):
            for col_b in missing_cols[i + 1:]:
                corr_val = float(miss_corr.loc[col_a, col_b])

                if abs(corr_val) < cfg.missingness_correlation_threshold:
                    continue

                pair = frozenset([col_a, col_b])
                if pair in reported:
                    continue
                reported.add(pair)

                # Count how often both are missing simultaneously
                both_missing = int((df[col_a].isnull() & df[col_b].isnull()).sum())
                both_missing_ratio = both_missing / n_rows

                findings.append(Finding(
                    module=self.name,
                    title=(
                        f"Correlated missingness between '{col_a}' and '{col_b}' "
                        f"(r={corr_val:.2f})"
                    ),
                    description=(
                        f"Columns '{col_a}' and '{col_b}' tend to be missing together "
                        f"(missingness correlation = {corr_val:.3f}). They are both missing "
                        f"in {both_missing:,} rows ({both_missing_ratio:.1%}). This pattern "
                        f"suggests a shared data source, systematic collection failure, or "
                        f"conditional dependency. This violates MCAR assumptions and can "
                        f"bias imputation strategies."
                    ),
                    evidence={
                        "column_a": col_a,
                        "column_b": col_b,
                        "missingness_correlation": round(corr_val, 4),
                        "both_missing_count": both_missing,
                        "both_missing_ratio": round(both_missing_ratio, 4),
                        "missing_ratio_a": round(float(null_ratios[col_a]), 4),
                        "missing_ratio_b": round(float(null_ratios[col_b]), 4),
                        "threshold": cfg.missingness_correlation_threshold,
                    },
                    severity=Severity.MEDIUM if abs(corr_val) > 0.85 else Severity.LOW,
                    confidence=min(abs(corr_val), 1.0),
                    recommendation=(
                        f"Investigate whether '{col_a}' and '{col_b}' share a data source "
                        f"or have a causal dependency. Consider joint imputation or "
                        f"indicator-based strategies that preserve the missingness pattern."
                    ),
                    affected_columns=[col_a, col_b],
                    metadata={
                        "anomaly_strength": abs(corr_val),
                        "correlation_strength": abs(corr_val),
                        "impact": both_missing_ratio,
                        "category": "missingness_relationships",
                    },
                ))

        return findings

    # ══════════════════════════════════════════════════════════════════
    #  10. STRUCTURAL HEALTH SCORE
    # ══════════════════════════════════════════════════════════════════

    def _compute_domain_score(self, findings: list[Finding]) -> float:
        """Compute a 0-100 health score for a single investigation domain.

        100 = no issues found. Score decreases with finding count and severity.
        """
        if not findings:
            return 100.0

        # Weighted penalty per finding
        penalty = 0.0
        for f in findings:
            weight = {
                Severity.CRITICAL: 25.0,
                Severity.HIGH: 15.0,
                Severity.MEDIUM: 8.0,
                Severity.LOW: 3.0,
                Severity.INFO: 1.0,
            }.get(f.severity, 5.0)
            penalty += weight

        return max(0.0, 100.0 - penalty)

    def _compute_structural_health_score(
        self,
        domain_scores: dict[str, float],
        dataset_info: DatasetInfo,
        cfg: IntegritySettings,
    ) -> Finding:
        """Compute the overall structural health score (0-100).

        This is NOT the future IPS. It represents dataset structural health only.
        """
        # Map domains to their health weights
        weight_map = {
            "missing_values": cfg.health_weight_missing,
            "duplicate_rows": cfg.health_weight_duplicates,
            "constant_features": cfg.health_weight_constants,
            "near_constant_features": cfg.health_weight_constants,
            "datatype_integrity": cfg.health_weight_types,
            "identifiers": cfg.health_weight_identifiers,
            "cardinality": cfg.health_weight_cardinality,
            "duplicate_features": cfg.health_weight_duplicates,
            "missingness_relationships": cfg.health_weight_missing,
        }

        weighted_sum = 0.0
        total_weight = 0.0
        domain_details: dict[str, float] = {}

        for domain, score in domain_scores.items():
            weight = weight_map.get(domain, 0.1)
            weighted_sum += score * weight
            total_weight += weight
            domain_details[domain] = round(score, 1)

        health_score = round(weighted_sum / total_weight, 1) if total_weight > 0 else 100.0

        # Determine severity based on health score
        if health_score >= 90:
            severity = Severity.INFO
            assessment = "excellent structural integrity"
        elif health_score >= 70:
            severity = Severity.LOW
            assessment = "good structural integrity with minor issues"
        elif health_score >= 50:
            severity = Severity.MEDIUM
            assessment = "moderate structural issues requiring attention"
        elif health_score >= 30:
            severity = Severity.HIGH
            assessment = "significant structural problems"
        else:
            severity = Severity.CRITICAL
            assessment = "severe structural integrity failure"

        return Finding(
            module=self.name,
            title=f"Structural Health Score: {health_score}/100",
            description=(
                f"The dataset '{dataset_info.name}' has a structural health score of "
                f"{health_score}/100, indicating {assessment}. This score is a weighted "
                f"composite of {len(domain_scores)} investigation domains covering "
                f"missing values, duplicates, constants, datatypes, identifiers, "
                f"cardinality, and missingness patterns."
            ),
            evidence={
                "health_score": health_score,
                "domain_scores": domain_details,
                "weights": {k: round(v, 2) for k, v in weight_map.items()},
                "assessment": assessment,
                "dataset_name": dataset_info.name,
                "row_count": dataset_info.row_count,
                "column_count": dataset_info.column_count,
            },
            severity=severity,
            confidence=0.95,
            recommendation=(
                f"The dataset scores {health_score}/100 on structural integrity. "
                + (
                    "No immediate structural concerns."
                    if health_score >= 80
                    else "Review the individual domain findings for specific actionable issues."
                )
            ),
            affected_columns=[],
            metadata={
                "anomaly_strength": max(0, (100 - health_score) / 100),
                "impact": max(0, (100 - health_score) / 100),
                "category": "structural_health_score",
            },
        )
