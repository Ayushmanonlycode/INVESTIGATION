"""Unit tests for the Relationship Investigator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investigation_engine.config.settings import Settings
from investigation_engine.core.plugin import get_registered_modules
from investigation_engine.models.dataset import ColumnProfile, DatasetInfo
from investigation_engine.models.finding import Finding
from investigation_engine.models.severity import Severity


@pytest.fixture(autouse=True)
def _ensure_registry():
    """Ensure the relationship investigator is registered for each test."""
    from investigation_engine.core.plugin import _REGISTRY
    from investigation_engine.modules.relationship import RelationshipInvestigator

    if "relationship_investigator" not in _REGISTRY:
        _REGISTRY["relationship_investigator"] = RelationshipInvestigator  # type: ignore[assignment]
    yield


@pytest.fixture
def config() -> Settings:
    return Settings()


def _build_dataset_info(df: pd.DataFrame, name: str = "relationship_test") -> DatasetInfo:
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
    from investigation_engine.modules.relationship import RelationshipInvestigator

    return RelationshipInvestigator()


def _find_by_category(findings: list[Finding], category: str) -> list[Finding]:
    return [f for f in findings if f.metadata.get("category") == category]


def _find_by_title_contains(findings: list[Finding], substring: str) -> list[Finding]:
    return [f for f in findings if substring.lower() in f.title.lower()]


class TestPluginRegistration:
    def test_module_is_registered(self):
        registry = get_registered_modules()
        assert "relationship_investigator" in registry

    def test_module_can_run_with_numeric_dataset(self, config):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        assert investigator.can_run(df, info) is True

    def test_module_skips_without_enough_numeric_columns(self, config):
        df = pd.DataFrame({"label": ["a", "b", "c"]})
        info = _build_dataset_info(df)
        investigator = _get_investigator()
        assert investigator.can_run(df, info) is False


class TestCorrelationDetection:
    def test_detects_perfect_positive_pearson_correlation(self, config):
        x = np.arange(200)
        df = pd.DataFrame({
            "x": x,
            "x_clone": x * 3,
            "noise": np.random.default_rng(42).normal(size=200),
        })
        info = _build_dataset_info(df)

        findings = _get_investigator().investigate(df, info, config)

        pearson_findings = _find_by_category(findings, "pearson_correlation")
        matched = [f for f in pearson_findings if set(f.affected_columns) == {"x", "x_clone"}]
        assert len(matched) == 1
        assert matched[0].evidence["correlation_coefficient"] == pytest.approx(1.0)
        assert matched[0].evidence["p_value"] is not None

    def test_detects_strong_negative_relationship(self, config):
        x = np.arange(200)
        df = pd.DataFrame({
            "up": x,
            "down": -x,
            "other": np.random.default_rng(7).normal(size=200),
        })
        info = _build_dataset_info(df)

        findings = _get_investigator().investigate(df, info, config)

        matched = [
            f for f in _find_by_category(findings, "pearson_correlation")
            if set(f.affected_columns) == {"up", "down"}
        ]
        assert len(matched) == 1
        assert matched[0].evidence["correlation_coefficient"] == pytest.approx(-1.0)

    def test_detects_monotonic_spearman_relationship(self, config):
        x = np.linspace(0, 20, 250)
        df = pd.DataFrame({
            "x": x,
            "exp_x": np.exp(x / 5.0),
            "noise": np.random.default_rng(9).normal(size=250),
        })
        info = _build_dataset_info(df)

        findings = _get_investigator().investigate(df, info, config)

        spearman_findings = _find_by_category(findings, "spearman_correlation")
        matched = [f for f in spearman_findings if set(f.affected_columns) == {"x", "exp_x"}]
        assert len(matched) == 1
        assert (
            abs(matched[0].evidence["correlation_coefficient"])
            >= config.relationship.spearman_threshold
        )


class TestMutualInformation:
    def test_detects_nonlinear_relationship(self, config):
        x = np.linspace(-3, 3, 600)
        rng = np.random.default_rng(123)
        df = pd.DataFrame({
            "x": x,
            "x_squared": x**2 + rng.normal(scale=0.05, size=len(x)),
            "noise": rng.normal(size=len(x)),
        })
        info = _build_dataset_info(df)

        findings = _get_investigator().investigate(df, info, config)

        mi_findings = _find_by_category(findings, "mutual_information")
        matched = [f for f in mi_findings if set(f.affected_columns) == {"x", "x_squared"}]
        assert len(matched) == 1
        assert (
            matched[0].evidence["mutual_information_score"]
            >= config.relationship.mutual_information_threshold
        )


class TestRedundancyAndGroups:
    def test_detects_redundant_columns(self, config):
        x = np.arange(300)
        df = pd.DataFrame({
            "base": x,
            "copy_a": x + 1,
            "copy_b": x * 2,
            "independent": np.random.default_rng(11).normal(size=300),
        })
        info = _build_dataset_info(df)

        findings = _get_investigator().investigate(df, info, config)

        redundant = _find_by_category(findings, "redundant_features")
        assert len(redundant) >= 1
        assert {"base", "copy_a", "copy_b"}.issubset(set(redundant[0].affected_columns))

    def test_detects_relationship_community(self, config):
        x = np.arange(250)
        rng = np.random.default_rng(88)
        df = pd.DataFrame({
            "g1": x,
            "g2": x * 1.5,
            "g3": x + rng.normal(scale=2.0, size=250),
            "outside": rng.normal(size=250),
        })
        info = _build_dataset_info(df)

        findings = _get_investigator().investigate(df, info, config)

        communities = _find_by_category(findings, "relationship_communities")
        assert len(communities) >= 1
        assert {"g1", "g2", "g3"}.issubset(set(communities[0].affected_columns))

    def test_detects_strong_feature_group(self, config):
        x = np.arange(180)
        df = pd.DataFrame({
            "a": x,
            "b": x * 2,
            "c": x * 3,
            "noise": np.random.default_rng(101).normal(size=180),
        })
        info = _build_dataset_info(df)

        findings = _get_investigator().investigate(df, info, config)

        groups = _find_by_category(findings, "strong_feature_groups")
        assert len(groups) >= 1
        assert {"a", "b", "c"}.issubset(set(groups[0].affected_columns))


class TestMulticollinearity:
    def test_detects_problematic_vif(self, config):
        rng = np.random.default_rng(5)
        x1 = rng.normal(size=500)
        x2 = rng.normal(size=500)
        x3 = 2.0 * x1 - 0.5 * x2 + rng.normal(scale=0.001, size=500)

        df = pd.DataFrame({
            "x1": x1,
            "x2": x2,
            "x3": x3,
        })
        info = _build_dataset_info(df)

        findings = _get_investigator().investigate(df, info, config)

        vif_findings = _find_by_category(findings, "multicollinearity")
        assert len(vif_findings) >= 1
        assert any(
            f.evidence["vif"] >= config.relationship.vif_high_threshold
            for f in vif_findings
        )
        assert any(f.severity in {Severity.HIGH, Severity.CRITICAL} for f in vif_findings)


class TestFindingSchema:
    def test_all_findings_include_required_evidence_fields(self, config):
        x = np.arange(150)
        rng = np.random.default_rng(17)
        df = pd.DataFrame({
            "a": x,
            "b": -x,
            "c": x**2 + rng.normal(scale=0.1, size=150),
            "d": x * 1.1,
        })
        info = _build_dataset_info(df)

        findings = _get_investigator().investigate(df, info, config)

        assert findings
        for finding in findings:
            evidence = finding.evidence
            assert "correlation_coefficient" in evidence
            assert "p_value" in evidence
            assert "mutual_information_score" in evidence
            assert "vif" in evidence
            assert "thresholds" in evidence
            assert finding.title
            assert finding.description
            assert finding.recommendation
            assert isinstance(finding.affected_columns, list)
