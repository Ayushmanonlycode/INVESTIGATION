"""Tests for balanced and exhaustive analysis profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from investigation_engine.cli.main import AnalysisProfile, _build_config, app
from investigation_engine.config.settings import (
    EngineSettings,
    IntegritySettings,
    RelationshipSettings,
    Settings,
)
from investigation_engine.models.dataset import ColumnProfile, DatasetInfo
from investigation_engine.modules.integrity import IntegrityInvestigator
from investigation_engine.modules.relationship import RelationshipInvestigator


def _dataset_info(df: pd.DataFrame) -> DatasetInfo:
    return DatasetInfo(
        name="profile_test",
        source="test://profile",
        source_type="dataframe",
        row_count=len(df),
        column_count=len(df.columns),
        column_types={str(column): str(df[column].dtype) for column in df.columns},
        columns=[
            ColumnProfile(
                name=str(column),
                dtype=str(df[column].dtype),
                null_count=int(df[column].isnull().sum()),
                unique_count=int(df[column].nunique()),
                is_numeric=pd.api.types.is_numeric_dtype(df[column]),
            )
            for column in df.columns
        ],
        duplicate_row_count=int(df.duplicated().sum()),
    )


def _relationship_config(profile: str) -> Settings:
    return Settings(
        engine=EngineSettings(
            analysis_profile=profile,
            max_findings_per_module=50,
        ),
        relationship=RelationshipSettings(
            pearson_max_rows=100,
            spearman_max_rows=100,
            vif_max_rows=100,
            mi_max_rows=100,
            mi_max_pairs=3,
        ),
    )


def test_balanced_profile_samples_relationship_rows() -> None:
    values = np.arange(500)
    df = pd.DataFrame({
        "left": values,
        "right": values * 2,
        "noise": np.random.default_rng(42).normal(size=len(values)),
    })

    findings = RelationshipInvestigator().investigate(
        df,
        _dataset_info(df),
        _relationship_config("balanced"),
    )
    pearson = next(
        finding
        for finding in findings
        if finding.metadata.get("category") == "pearson_correlation"
        and set(finding.affected_columns) == {"left", "right"}
    )

    assert pearson.evidence["sampled"] is True
    assert pearson.evidence["analysis_rows"] == 100
    assert pearson.evidence["total_rows"] == 500


def test_full_profile_uses_all_relationship_rows() -> None:
    values = np.arange(500)
    df = pd.DataFrame({
        "left": values,
        "right": values * 2,
        "noise": np.random.default_rng(7).normal(size=len(values)),
    })

    findings = RelationshipInvestigator().investigate(
        df,
        _dataset_info(df),
        _relationship_config("full"),
    )
    pearson = next(
        finding
        for finding in findings
        if finding.metadata.get("category") == "pearson_correlation"
        and set(finding.affected_columns) == {"left", "right"}
    )

    assert pearson.evidence["sampled"] is False
    assert pearson.evidence["analysis_rows"] == 500


def test_missingness_pairs_are_compressed_into_one_group() -> None:
    rows = 500
    mask = np.arange(rows) % 4 == 0
    df = pd.DataFrame({
        "first": np.where(mask, np.nan, np.arange(rows)),
        "second": np.where(mask, np.nan, np.arange(rows) * 2),
        "third": np.where(mask, np.nan, np.arange(rows) * 3),
        "complete": np.arange(rows),
    })
    config = Settings(
        integrity=IntegritySettings(missingness_max_rows=100),
    )

    findings = IntegrityInvestigator().investigate(
        df,
        _dataset_info(df),
        config,
    )
    groups = [
        finding
        for finding in findings
        if finding.metadata.get("category") == "missingness_relationships"
    ]

    assert len(groups) == 1
    assert set(groups[0].affected_columns) == {"first", "second", "third"}
    assert groups[0].evidence["qualifying_pair_count"] == 3
    assert groups[0].evidence["analysis_rows"] == 100
    assert groups[0].evidence["sampled"] is True


def test_cli_exposes_analysis_profile() -> None:
    result = CliRunner().invoke(app, ["investigate", "--help"])

    assert result.exit_code == 0
    assert "--profile" in result.stdout
    assert "balanced" in result.stdout
    assert "full" in result.stdout


def test_cli_profile_overrides_configuration() -> None:
    config = _build_config("INFO", AnalysisProfile.FULL)

    assert config.engine.analysis_profile == "full"
