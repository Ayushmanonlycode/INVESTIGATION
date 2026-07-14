"""Investigation Engine configuration system.

Uses Pydantic Settings for hierarchical configuration with the following
priority (highest to lowest):
    1. Programmatic overrides (constructor kwargs)
    2. Environment variables (prefixed with IE_)
    3. .env file
    4. Code defaults

All thresholds are intentionally conservative defaults. Investigation modules
should respect these values but may override them for domain-specific needs
via the module-level config mechanism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EngineSettings(BaseSettings):
    """Core engine execution settings."""

    analysis_profile: Literal["balanced", "full"] = Field(
        default="balanced",
        description=(
            "Analysis strategy. 'balanced' uses deterministic budgets for expensive "
            "statistical scans; 'full' removes balanced row and candidate budgets."
        ),
    )
    max_rows_sample: int = Field(
        default=1_000_000,
        ge=100,
        description="Maximum rows to load before sampling. Datasets larger than this are sampled.",
    )
    max_findings_per_module: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum findings a single module can produce.",
    )
    parallel_modules: bool = Field(
        default=False,
        description="Execute investigation modules in parallel (experimental).",
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop investigation on first module failure.",
    )
    enabled_modules: list[str] | None = Field(
        default=None,
        description="Explicit list of module names to run. None means all registered modules.",
    )
    disabled_modules: list[str] = Field(
        default_factory=list,
        description="Module names to exclude from investigation.",
    )
    investigation_semantic_cohesion_threshold: float = Field(
        default=0.46,
        ge=0.0,
        le=1.0,
        description="Minimum deterministic semantic cohesion required before an investigation is split.",
    )
    max_investigation_evidence_share: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Maximum share of all evidence units one investigation should absorb unless strongly justified.",
    )


class IntegritySettings(BaseSettings):
    """Configurable thresholds for the Integrity Investigator.

    All severity boundaries and detection thresholds are configurable
    to allow domain-specific tuning without code changes.
    """

    # ── Missing Value Thresholds ──────────────────────────────────────
    missing_critical_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Missing ratio above this is CRITICAL severity.",
    )
    missing_high_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Missing ratio above this is HIGH severity.",
    )
    missing_medium_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Missing ratio above this is MEDIUM severity.",
    )
    missing_low_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Missing ratio above this is LOW severity. Below this is not reported.",
    )
    missing_empty_column_threshold: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Missing ratio above this flags column as completely empty.",
    )

    # ── Duplicate Thresholds ──────────────────────────────────────────
    duplicate_critical_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Duplicate ratio above this is CRITICAL severity.",
    )
    duplicate_high_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Duplicate ratio above this is HIGH severity.",
    )
    duplicate_medium_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Duplicate ratio above this is MEDIUM severity.",
    )
    duplicate_low_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Duplicate ratio above this is LOW severity.",
    )

    # ── Constant / Near-Constant Thresholds ───────────────────────────
    constant_dominance_threshold: float = Field(
        default=0.9999,
        ge=0.0,
        le=1.0,
        description="Value dominance ratio above this flags column as constant.",
    )
    near_constant_dominance_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Value dominance ratio above this flags column as near-constant.",
    )

    # ── Cardinality Thresholds ────────────────────────────────────────
    high_cardinality_ratio: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Unique ratio above this flags high cardinality for non-ID columns.",
    )
    low_cardinality_max_unique: int = Field(
        default=2,
        ge=1,
        description="Unique count at or below this flags low cardinality (non-boolean).",
    )

    # ── Identifier Detection Thresholds ───────────────────────────────
    id_uniqueness_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Unique ratio above this to consider a column as a potential identifier.",
    )
    id_column_patterns: list[str] = Field(
        default_factory=lambda: [
            "id", "uuid", "uid", "key", "code", "identifier",
            "index", "pk", "serial", "number", "num", "no",
        ],
        description="Substrings in column names that suggest identifier columns.",
    )

    # ── Datatype Integrity ────────────────────────────────────────────
    numeric_string_sample_size: int = Field(
        default=1000,
        ge=10,
        description="Sample size for detecting numeric values stored as strings.",
    )
    numeric_string_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Fraction of parseable numeric values to flag string column as mistyped.",
    )
    mixed_type_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Fraction of values with different inferred type to flag mixed types.",
    )

    # ── Missingness Correlation ───────────────────────────────────────
    missingness_correlation_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Correlation between missing indicators above this is reported.",
    )
    missingness_min_missing_ratio: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum missing ratio for a column to be included in missingness correlation.",
    )
    missingness_max_rows: int = Field(
        default=50_000,
        ge=100,
        description=(
            "Maximum deterministic row sample used to discover missingness relationships "
            "in the balanced analysis profile. Exact counts are computed on the full data."
        ),
    )

    # ── Structural Health Score Weights ────────────────────────────────
    health_weight_missing: float = Field(default=0.25, ge=0.0, le=1.0)
    health_weight_duplicates: float = Field(default=0.15, ge=0.0, le=1.0)
    health_weight_constants: float = Field(default=0.10, ge=0.0, le=1.0)
    health_weight_types: float = Field(default=0.20, ge=0.0, le=1.0)
    health_weight_identifiers: float = Field(default=0.15, ge=0.0, le=1.0)
    health_weight_cardinality: float = Field(default=0.15, ge=0.0, le=1.0)


class RelationshipSettings(BaseSettings):
    """Configurable thresholds for the Relationship Investigator."""

    pearson_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum absolute Pearson correlation to report.",
    )
    spearman_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum absolute Spearman correlation to report.",
    )
    redundancy_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum absolute correlation to treat features as redundant.",
    )
    strong_group_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum edge strength for strong feature groups.",
    )
    community_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum edge strength for relationship communities.",
    )
    mutual_information_threshold: float = Field(
        default=0.20,
        ge=0.0,
        description="Minimum mutual information score to report.",
    )
    vif_high_threshold: float = Field(
        default=5.0,
        ge=1.0,
        description="VIF above this is reported as problematic multicollinearity.",
    )
    vif_very_high_threshold: float = Field(
        default=10.0,
        ge=1.0,
        description="VIF above this is severe multicollinearity.",
    )
    min_pair_observations: int = Field(
        default=30,
        ge=5,
        description="Minimum complete observations required for a pairwise test.",
    )
    pearson_max_rows: int = Field(
        default=50_000,
        ge=100,
        description="Maximum rows used for Pearson scans in the balanced profile.",
    )
    spearman_max_rows: int = Field(
        default=25_000,
        ge=100,
        description="Maximum rows used for Spearman scans in the balanced profile.",
    )
    vif_max_rows: int = Field(
        default=50_000,
        ge=100,
        description="Maximum rows used for VIF estimation in the balanced profile.",
    )
    max_correlation_candidates_per_method: int = Field(
        default=200,
        ge=1,
        description=(
            "Maximum strongest Pearson or Spearman pairs receiving significance tests "
            "and individual findings in the balanced profile. All threshold-crossing "
            "pairs remain available for group discovery."
        ),
    )
    mi_max_rows: int = Field(
        default=5000,
        ge=100,
        description="Maximum sampled rows used for pairwise mutual information.",
    )
    mi_max_columns: int = Field(
        default=150,
        ge=2,
        description="Maximum numeric columns included in pairwise mutual information scans.",
    )
    mi_max_pairs: int = Field(
        default=500,
        ge=1,
        description=(
            "Maximum shortlisted feature pairs scanned for mutual information in the "
            "balanced profile. The full profile scans all eligible pairs."
        ),
    )
    mi_n_neighbors: int = Field(
        default=3,
        ge=1,
        description="Neighbors parameter for sklearn mutual information estimation.",
    )
    target_column_candidates: list[str] = Field(
        default_factory=lambda: [
            "target",
            "label",
            "class",
            "response",
            "outcome",
            "y",
        ],
        description="Common column names treated as target candidates when metadata is absent.",
    )


class EvidenceCompressionSettings(BaseSettings):
    """Configuration for Graph-Based Evidence Compression."""

    algorithm: str = Field(
        default="louvain",
        description="Community detection algorithm: connected_components, louvain, greedy_modularity.",
    )
    edge_weight_threshold: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="Minimum edge weight required to connect two findings.",
    )
    structural_weight: float = Field(
        default=0.40,
        ge=0.0,
        le=10.0,
        description="Top-level weight for structural similarity.",
    )
    statistical_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=10.0,
        description="Top-level weight for statistical similarity.",
    )
    semantic_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=10.0,
        description="Top-level weight for semantic similarity.",
    )
    shared_columns_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    shared_rows_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    shared_metadata_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    same_investigator_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    feature_family_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    severity_similarity_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    confidence_similarity_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    numeric_evidence_similarity_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    semantic_prefix_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    semantic_token_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    semantic_domain_term_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    candidate_shared_columns: bool = Field(
        default=True,
        description="Generate similarity candidates from shared columns.",
    )
    candidate_shared_feature_family: bool = Field(
        default=True,
        description="Generate similarity candidates from shared feature families.",
    )
    candidate_shared_terms: bool = Field(
        default=True,
        description="Generate similarity candidates from shared semantic tokens.",
    )
    candidate_shared_category: bool = Field(
        default=True,
        description="Generate similarity candidates from shared finding categories.",
    )
    max_index_bucket_size: int = Field(
        default=250,
        ge=2,
        description="Maximum bucket size to expand into pairwise candidate comparisons.",
    )
    benchmark_repeat_runs: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of runs used for benchmark timing summaries.",
    )
    louvain_resolution: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Resolution parameter passed to Louvain community detection.",
    )
    greedy_resolution: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Resolution parameter passed to greedy modularity detection.",
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Deterministic random seed for graph algorithms that support it.",
    )

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, value: str) -> str:
        allowed = {"connected_components", "louvain", "greedy_modularity"}
        normalized = value.lower()
        if normalized not in allowed:
            raise ValueError(f"Invalid evidence compression algorithm: {value}.")
        return normalized


class ThresholdSettings(BaseSettings):
    """Statistical thresholds for investigation modules.

    These are conservative defaults designed to minimize false positives
    while catching genuinely interesting patterns.
    """

    # Outlier Detection
    outlier_z_threshold: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Z-score threshold for outlier detection.",
    )
    outlier_iqr_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        le=5.0,
        description="IQR multiplier for Tukey's fence outlier detection.",
    )

    # Correlation
    correlation_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum absolute correlation to report as a finding.",
    )

    # Missing Values
    missing_value_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum missing ratio to flag as a finding.",
    )

    # Low Variance
    low_variance_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Coefficient of variation below this is flagged as low variance.",
    )

    # Class Imbalance
    class_imbalance_ratio: float = Field(
        default=10.0,
        ge=1.0,
        description="Majority/minority class ratio threshold for imbalance detection.",
    )

    # Duplicate Detection
    duplicate_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum duplicate ratio to flag as a finding.",
    )

    # Clustering
    min_cluster_samples: int = Field(
        default=50,
        ge=2,
        description="Minimum samples required for cluster discovery.",
    )
    max_clusters: int = Field(
        default=20,
        ge=2,
        le=100,
        description="Maximum clusters to discover.",
    )

    # Feature Importance
    min_feature_importance: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum feature importance score to report.",
    )

    # Statistical Significance
    significance_level: float = Field(
        default=0.05,
        ge=0.001,
        le=0.1,
        description="P-value threshold for statistical significance.",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    log_file: str | None = Field(
        default=None,
        description="Optional file path for log output.",
    )
    log_format: str = Field(
        default=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        description="Loguru format string.",
    )
    log_rotation: str = Field(
        default="10 MB",
        description="Log file rotation size.",
    )
    log_retention: str = Field(
        default="7 days",
        description="Log file retention period.",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}.")
        return upper


class Settings(BaseSettings):
    """Root configuration for the Investigation Engine.

    Composes engine, threshold, and logging settings into a single
    configuration object. All settings can be overridden via environment
    variables with the IE_ prefix.

    Examples:
        IE_MAX_ROWS_SAMPLE=500000
        IE_OUTLIER_Z_THRESHOLD=2.5
        IE_LOG_LEVEL=DEBUG
    """

    model_config = SettingsConfigDict(
        env_prefix="IE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Composed settings groups
    engine: EngineSettings = Field(default_factory=EngineSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    integrity: IntegritySettings = Field(default_factory=IntegritySettings)
    relationship: RelationshipSettings = Field(default_factory=RelationshipSettings)
    evidence_compression: EvidenceCompressionSettings = Field(
        default_factory=EvidenceCompressionSettings
    )

    # Output
    output_dir: Path | None = Field(
        default=None,
        description="Directory to write investigation output. None for no file output.",
    )
    output_format: str = Field(
        default="json",
        description="Output format: 'json' or 'pickle'.",
    )

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Ensure output format is supported."""
        valid_formats = {"json", "pickle"}
        lower = v.lower()
        if lower not in valid_formats:
            raise ValueError(f"Invalid output format: {v}. Must be one of {valid_formats}.")
        return lower

    def get_module_config(self, module_name: str) -> dict[str, Any]:
        """Retrieve threshold-level configuration relevant to a specific module.

        This provides a convention-based way for modules to access their
        thresholds without coupling to the full Settings object.

        Args:
            module_name: Name of the module requesting config.

        Returns:
            Dictionary of relevant configuration values.
        """
        threshold_dict = self.thresholds.model_dump()
        engine_dict = self.engine.model_dump()
        return {
            "module_name": module_name,
            **threshold_dict,
            **engine_dict,
        }
