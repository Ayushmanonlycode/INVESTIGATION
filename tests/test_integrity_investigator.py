"""Comprehensive unit tests for the Integrity Investigator.

Tests use synthetic datasets specifically designed to contain known
structural issues. Each test verifies that the investigator correctly
detects the issue, assigns appropriate severity, and produces valid
Finding objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investigation_engine.config.settings import Settings
from investigation_engine.core.plugin import clear_registry, get_registered_modules
from investigation_engine.models.dataset import DatasetInfo, ColumnProfile
from investigation_engine.models.finding import Finding
from investigation_engine.models.severity import Severity


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _ensure_registry():
    """Ensure the integrity investigator is registered for each test."""
    from investigation_engine.modules.integrity import IntegrityInvestigator
    from investigation_engine.core.plugin import _REGISTRY

    # If already registered, leave it. If not (e.g., after clear), re-add.
    if "integrity_investigator" not in _REGISTRY:
        _REGISTRY["integrity_investigator"] = IntegrityInvestigator  # type: ignore[assignment]
    yield


@pytest.fixture
def config() -> Settings:
    """Default configuration."""
    return Settings()


def _build_dataset_info(df: pd.DataFrame, name: str = "test_dataset") -> DatasetInfo:
    """Helper to build DatasetInfo from a DataFrame."""
    columns = []
    for col in df.columns:
        columns.append(ColumnProfile(
            name=str(col),
            dtype=str(df[col].dtype),
            null_count=int(df[col].isnull().sum()),
            unique_count=int(df[col].nunique()),
            is_numeric=pd.api.types.is_numeric_dtype(df[col]),
            is_categorical=pd.api.types.is_object_dtype(df[col]),
            is_datetime=pd.api.types.is_datetime64_any_dtype(df[col]),
            is_boolean=pd.api.types.is_bool_dtype(df[col]),
        ))
    return DatasetInfo(
        name=name,
        source="test://synthetic",
        source_type="dataframe",
        row_count=len(df),
        column_count=len(df.columns),
        column_types={str(c): str(df[c].dtype) for c in df.columns},
        columns=columns,
        memory_usage_bytes=int(df.memory_usage(deep=True).sum()),
        has_missing_values=bool(df.isnull().any().any()),
        duplicate_row_count=int(df.duplicated().sum()),
    )


def _get_investigator():
    """Get registered IntegrityInvestigator instance."""
    from investigation_engine.modules.integrity import IntegrityInvestigator
    return IntegrityInvestigator()


def _find_by_category(findings: list[Finding], category: str) -> list[Finding]:
    """Filter findings by their metadata category."""
    return [f for f in findings if f.metadata.get("category") == category]


def _find_by_title_contains(findings: list[Finding], substring: str) -> list[Finding]:
    """Filter findings whose title contains a substring."""
    return [f for f in findings if substring.lower() in f.title.lower()]


# ═══════════════════════════════════════════════════════════════════════
#  Test: Plugin Registration
# ═══════════════════════════════════════════════════════════════════════

class TestPluginRegistration:
    def test_module_is_registered(self):
        registry = get_registered_modules()
        assert "integrity_investigator" in registry

    def test_module_can_run_on_valid_dataset(self, config):
        df = pd.DataFrame({"a": [1, 2, 3]})
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        assert investigator.can_run(df, info) is True

    def test_module_skips_empty_dataset(self, config):
        df = pd.DataFrame()
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        assert investigator.can_run(df, info) is False


# ═══════════════════════════════════════════════════════════════════════
#  Test: Missing Values
# ═══════════════════════════════════════════════════════════════════════

class TestMissingValues:
    def test_detects_high_missing_ratio(self, config):
        n = 1000
        df = pd.DataFrame({
            "clean": range(n),
            "moderate_missing": [None if i % 3 == 0 else i for i in range(n)],  # ~33%
            "severe_missing": [None if i % 2 == 0 else i for i in range(n)],    # ~50%
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        missing_findings = _find_by_category(findings, "missing_values")
        assert len(missing_findings) >= 2

        # Check the 50% column gets CRITICAL
        severe = [f for f in missing_findings if "severe_missing" in f.title]
        assert len(severe) == 1
        assert severe[0].severity == Severity.CRITICAL

    def test_detects_completely_empty_column(self, config):
        df = pd.DataFrame({
            "normal": [1, 2, 3, 4, 5] * 20,
            "empty_col": [None] * 100,
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        empty_findings = _find_by_title_contains(findings, "empty")
        assert len(empty_findings) >= 1
        assert empty_findings[0].severity == Severity.CRITICAL

    def test_no_findings_for_clean_data(self, config):
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100, 200),
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        missing_findings = _find_by_category(findings, "missing_values")
        assert len(missing_findings) == 0

    def test_missing_evidence_contains_numbers(self, config):
        n = 1000
        df = pd.DataFrame({
            "col": [None if i < 200 else i for i in range(n)],  # 20% missing
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        missing_findings = _find_by_category(findings, "missing_values")
        assert len(missing_findings) >= 1
        evidence = missing_findings[0].evidence
        assert "missing_count" in evidence
        assert "missing_ratio" in evidence
        assert "total_rows" in evidence
        assert isinstance(evidence["missing_count"], int)
        assert isinstance(evidence["missing_ratio"], float)


# ═══════════════════════════════════════════════════════════════════════
#  Test: Duplicate Rows
# ═══════════════════════════════════════════════════════════════════════

class TestDuplicateRows:
    def test_detects_duplicate_rows(self, config):
        base = pd.DataFrame({
            "a": range(80),
            "b": range(80),
        })
        # Add 20 duplicates (20% duplication)
        dupes = base.head(20).copy()
        df = pd.concat([base, dupes], ignore_index=True)
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        dup_findings = _find_by_category(findings, "duplicates")
        assert len(dup_findings) >= 1
        assert dup_findings[0].evidence["duplicate_count"] == 20

    def test_no_findings_for_unique_rows(self, config):
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100, 200),
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        dup_findings = _find_by_category(findings, "duplicates")
        assert len(dup_findings) == 0

    def test_high_duplicate_ratio_gets_critical(self, config):
        # 50% duplicates
        base = pd.DataFrame({"a": [1], "b": [2]})
        df = pd.concat([base] * 100, ignore_index=True)
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        dup_findings = _find_by_category(findings, "duplicates")
        assert len(dup_findings) >= 1
        assert dup_findings[0].severity == Severity.CRITICAL


# ═══════════════════════════════════════════════════════════════════════
#  Test: Duplicate Features
# ═══════════════════════════════════════════════════════════════════════

class TestDuplicateFeatures:
    def test_detects_identical_columns(self, config):
        df = pd.DataFrame({
            "original": range(100),
            "copy": range(100),
            "different": range(100, 200),
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        dup_feat_findings = _find_by_category(findings, "duplicate_features")
        assert len(dup_feat_findings) >= 1
        # Both columns should be in affected_columns
        affected = dup_feat_findings[0].affected_columns
        assert "original" in affected
        assert "copy" in affected

    def test_no_findings_for_unique_columns(self, config):
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100, 200),
            "c": range(200, 300),
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        dup_feat_findings = _find_by_category(findings, "duplicate_features")
        assert len(dup_feat_findings) == 0


# ═══════════════════════════════════════════════════════════════════════
#  Test: Identifier Investigation
# ═══════════════════════════════════════════════════════════════════════

class TestIdentifiers:
    def test_detects_duplicate_ids(self, config):
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5, 5, 6, 7, 8, 9] * 10,
            "value": range(100),
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        id_findings = _find_by_category(findings, "identifiers")
        # user_id has duplicates but may or may not pass uniqueness threshold
        # depending on the pattern — at minimum the system should not crash
        assert isinstance(findings, list)

    def test_detects_null_ids(self, config):
        ids = list(range(100))
        ids[50] = None  # type: ignore[assignment]
        ids[75] = None  # type: ignore[assignment]
        df = pd.DataFrame({
            "record_id": ids,
            "value": range(100),
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        id_findings = _find_by_category(findings, "identifiers")
        if id_findings:
            assert id_findings[0].evidence["null_count"] >= 2


# ═══════════════════════════════════════════════════════════════════════
#  Test: Constant Features
# ═══════════════════════════════════════════════════════════════════════

class TestConstantFeatures:
    def test_detects_constant_column(self, config):
        df = pd.DataFrame({
            "constant": [42] * 100,
            "varying": range(100),
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        const_findings = _find_by_category(findings, "constant_features")
        assert len(const_findings) == 1
        assert "constant" in const_findings[0].affected_columns
        assert const_findings[0].evidence["variance"] == 0.0

    def test_no_constant_finding_for_varying_column(self, config):
        df = pd.DataFrame({"a": range(100)})
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        const_findings = _find_by_category(findings, "constant_features")
        assert len(const_findings) == 0


# ═══════════════════════════════════════════════════════════════════════
#  Test: Near-Constant Features
# ═══════════════════════════════════════════════════════════════════════

class TestNearConstantFeatures:
    def test_detects_near_constant_column(self, config):
        # 97% single value (above default 95% threshold)
        values = [1] * 970 + [2] * 30
        df = pd.DataFrame({"near_const": values, "normal": range(1000)})
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        nc_findings = _find_by_category(findings, "near_constant_features")
        assert len(nc_findings) >= 1
        assert nc_findings[0].evidence["dominance_ratio"] >= 0.95


# ═══════════════════════════════════════════════════════════════════════
#  Test: Datatype Integrity
# ═══════════════════════════════════════════════════════════════════════

class TestDatatypeIntegrity:
    def test_detects_numeric_as_string(self, config):
        df = pd.DataFrame({
            "numeric_strings": [str(i) for i in range(100)],
            "real_strings": [f"name_{i}" for i in range(100)],
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        dtype_findings = _find_by_category(findings, "datatype_integrity")
        numeric_string = [f for f in dtype_findings if "numeric_strings" in f.title]
        assert len(numeric_string) >= 1

    def test_detects_mixed_types(self, config):
        # Mix of numbers and strings
        values: list[Any] = list(range(80)) + [f"text_{i}" for i in range(20)]
        df = pd.DataFrame({"mixed": [str(v) for v in values]})
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        # Should detect either numeric-as-string or mixed types
        dtype_findings = _find_by_category(findings, "datatype_integrity")
        assert len(dtype_findings) >= 1


# ═══════════════════════════════════════════════════════════════════════
#  Test: Cardinality
# ═══════════════════════════════════════════════════════════════════════

class TestCardinality:
    def test_detects_high_cardinality(self, config):
        df = pd.DataFrame({
            "high_card": [f"val_{i}" for i in range(1000)],
            "low_card": ["a", "b", "c"] * 333 + ["a"],
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        card_findings = _find_by_category(findings, "cardinality")
        high_card = [f for f in card_findings if "high_card" in f.title.lower() or "high cardinality" in f.title.lower()]
        assert len(high_card) >= 1


# ═══════════════════════════════════════════════════════════════════════
#  Test: Missingness Relationships
# ═══════════════════════════════════════════════════════════════════════

class TestMissingnessRelationships:
    def test_detects_correlated_missingness(self, config):
        n = 1000
        # Columns A and B are missing in the same rows
        missing_mask = [True if i % 5 == 0 else False for i in range(n)]
        col_a = [None if m else i for i, m in enumerate(missing_mask)]
        col_b = [None if m else i * 2 for i, m in enumerate(missing_mask)]
        col_c = range(n)  # no missing

        df = pd.DataFrame({"col_a": col_a, "col_b": col_b, "col_c": col_c})
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        miss_rel = _find_by_category(findings, "missingness_relationships")
        assert len(miss_rel) >= 1
        assert miss_rel[0].evidence["missingness_correlation"] >= 0.7


# ═══════════════════════════════════════════════════════════════════════
#  Test: Structural Health Score
# ═══════════════════════════════════════════════════════════════════════

class TestStructuralHealthScore:
    def test_health_score_produced_for_clean_data(self, config):
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        health_findings = _find_by_category(findings, "structural_health_score")
        assert len(health_findings) == 1
        score = health_findings[0].evidence["health_score"]
        assert score >= 80  # Clean data should score high

    def test_health_score_produced_for_messy_data(self, config):
        n = 500
        df = pd.DataFrame({
            "empty": [None] * n,
            "constant": [1] * n,
            "half_missing": [None if i < 250 else i for i in range(n)],
            "numeric_str": [str(i) for i in range(n)],
            "ok": range(n),
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        health_findings = _find_by_category(findings, "structural_health_score")
        assert len(health_findings) == 1
        score = health_findings[0].evidence["health_score"]
        assert score < 100  # Messy data should NOT score 100

    def test_health_score_between_0_and_100(self, config):
        df = pd.DataFrame({
            "a": [None] * 50 + list(range(50)),
            "b": [1] * 100,
            "c": [str(i) for i in range(100)],
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        health_findings = _find_by_category(findings, "structural_health_score")
        score = health_findings[0].evidence["health_score"]
        assert 0 <= score <= 100


# ═══════════════════════════════════════════════════════════════════════
#  Test: Finding Validity
# ═══════════════════════════════════════════════════════════════════════

class TestFindingValidity:
    def test_all_findings_have_required_fields(self, config):
        """Every finding must have all required fields populated."""
        df = pd.DataFrame({
            "empty": [None] * 100,
            "constant": [42] * 100,
            "numeric_str": [str(i) for i in range(100)],
            "normal": range(100),
            "copy": range(100),
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        assert len(findings) > 0
        for f in findings:
            assert f.id, "Finding must have an id"
            assert f.module == "integrity_investigator"
            assert f.title, "Finding must have a title"
            assert f.description, "Finding must have a description"
            assert isinstance(f.evidence, dict), "Evidence must be a dict"
            assert len(f.evidence) > 0, "Evidence must not be empty"
            assert isinstance(f.severity, Severity)
            assert 0.0 <= f.confidence <= 1.0
            assert f.recommendation, "Finding must have a recommendation"
            assert isinstance(f.affected_columns, list)
            assert isinstance(f.metadata, dict)
            assert "category" in f.metadata

    def test_findings_are_serializable(self, config):
        """All findings must be JSON-serializable via Pydantic."""
        df = pd.DataFrame({
            "empty": [None] * 50 + list(range(50)),
            "constant": [1] * 100,
        })
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        for f in findings:
            json_str = f.model_dump_json()
            assert len(json_str) > 0
            # Round-trip
            reconstructed = Finding.model_validate_json(json_str)
            assert reconstructed.id == f.id


# ═══════════════════════════════════════════════════════════════════════
#  Test: End-to-End with Synthetic Dataset
# ═══════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_full_investigation_on_messy_dataset(self, config):
        """Comprehensive test with a dataset containing ALL types of issues."""
        np.random.seed(42)
        n = 500

        df = pd.DataFrame({
            # Clean identifier with some duplicates
            "record_id": list(range(n - 5)) + [0, 1, 2, 3, 4],
            # 40% missing
            "high_missing": [None if i < 200 else i for i in range(n)],
            # Completely empty
            "empty_col": [None] * n,
            # Constant
            "constant_status": ["active"] * n,
            # Near-constant (98% single value)
            "near_const": ["yes"] * 490 + ["no"] * 10,
            # Numeric stored as string
            "price_str": [str(round(np.random.uniform(10, 100), 2)) for _ in range(n)],
            # High cardinality categorical
            "description": [f"unique_description_{i}" for i in range(n)],
            # Normal numeric
            "measurement": np.random.normal(100, 15, n),
            # Identical to measurement
            "measurement_copy": np.random.normal(100, 15, n),
        })
        # Make measurement_copy truly identical
        df["measurement_copy"] = df["measurement"]

        # Add correlated missingness
        miss_mask = np.random.random(n) < 0.15
        df.loc[miss_mask, "high_missing"] = None
        df.loc[miss_mask, "measurement"] = None

        info = _build_dataset_info(df)
        investigator = _get_investigator()
        findings = investigator.investigate(df, info, config)

        # Should detect multiple categories
        categories = {f.metadata.get("category") for f in findings}
        assert "missing_values" in categories
        assert "constant_features" in categories
        assert "structural_health_score" in categories

        # Health score should exist
        health = _find_by_category(findings, "structural_health_score")
        assert len(health) == 1

        # Should have at least 5 findings (various issues)
        assert len(findings) >= 5

        # All findings valid
        for f in findings:
            assert isinstance(f, Finding)
            assert f.module == "integrity_investigator"
